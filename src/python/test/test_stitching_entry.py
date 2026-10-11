"""PL14 to PL17 of docs/stitching-assembly-validation.md: persistence, entry and registry, determinism, Dataset.stitch.

Fixtures: ``stub_grid`` and the results of the other checks (PL4, PL10 with an approval,
PL18); PL16 runs the ``stub_variants`` in three single-thread processes; PL17 writes a
synthetic dataset folder of four FOVs of 4×16×24 in the test. The import part of PL14
(``view_map.csv``) belongs to the BigStitcher issue.
"""
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

import starfinder.io
import starfinder.stitching
from starfinder._registry import Dependency, config_type_for, names, spec_for
from starfinder.dataset import FOV, Dataset
from starfinder.image import IncompatibleGeometryError
from starfinder.io import save_volume
from starfinder.segmentation import reference_grid_from_file
from starfinder.stitching import (STITCHING_METHODS, StitchingBackendUnavailableError, StitchingSpec,
                                  approve_placement, load_placement, placement_from_table, positions_from_grid,
                                  positions_from_table, save_placement, stitch)
from starfinder.stitching import _persist
from starfinder.stitching._stitch import stitch as real_stitch

from .stitching_fixtures import (FRACTIONAL, STUB_SPEC, StubConfig, determinism_hashes, grid, grids_for, inputs_for,
                                 check_components, maf_text, metadata, stub_run, stub_table, zero)

pytestmark = [pytest.mark.stitching, pytest.mark.validation]

SRC = Path(__file__).resolve().parents[1]


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setitem(STITCHING_METHODS, StubConfig, STUB_SPEC)


def moved_approved():
    table = stub_table()
    frame = table.positions.drop(columns=["grid_column", "grid_row"])
    frame.loc[5, "nominal_x"] += 12.0
    moved = positions_from_table(frame)
    result = stitch(inputs_for(moved, grids_for(moved)), moved, config=StubConfig(FRACTIONAL), sample="s1",
                    fallback="nominal")
    return approve_placement(result, reviewer="jiahao", note="FOV5 has no overlap")


def supplied_approved():
    frame = pd.DataFrame({"fov_id": ["FOV1", "FOV2"], "nominal_z": [0.0, 0.0], "nominal_y": [0.0, 0.0],
                          "nominal_x": [0.0, 12.0]})
    table = positions_from_table(frame)
    grids = {fov: grid((4, 16, 16), (1.0, 1.0, 1.0)) for fov in table.fov_ids}
    result = placement_from_table(table, grids, sample="s1",
                                  origins={"FOV1": (0.3, 0.25, 1.5), "FOV2": (-0.7, -0.125, 15.0)})
    return approve_placement(result, reviewer="jiahao", note="supplied")


# --- PL14 ---------------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["fractional", "moved_approved", "supplied_approved"])
def test_save_and_load_round_trip(stub, tmp_path, case):
    table = stub_table()
    result = {"fractional": lambda: stitch(inputs_for(table, grids_for(table)), table, config=StubConfig(FRACTIONAL),
                                           sample="s1"),
              "moved_approved": moved_approved, "supplied_approved": supplied_approved}[case]()
    folder = save_placement(result, tmp_path / "stitching" / "s1")
    assert sorted(p.name for p in folder.iterdir()) == ["fovs.csv", "links.csv", "placement.json"]
    loaded = load_placement(folder)
    check_components(result)
    check_components(loaded)
    pd.testing.assert_frame_equal(loaded.fovs, result.fovs, check_exact=True)
    pd.testing.assert_frame_equal(loaded.links, result.links, check_exact=True)
    origins = ["origin_z", "origin_y", "origin_x"]
    assert loaded.fovs[origins].to_numpy().tobytes() == result.fovs[origins].to_numpy().tobytes()
    assert loaded.record == result.record and loaded.sample == result.sample
    saved = json.loads((folder / "placement.json").read_text())
    assert saved["format_version"] == 1 and {"frame", "gauge", "table", "inputs", "review"} <= set(saved)
    for fov, entry in saved["inputs"].items():
        assert {"source", "file_sha256"} <= set(entry)
        assert {"shape_zyx", "metadata", "sha256"} <= set(entry["grid"])
    for name in ("fovs", "links"):
        data = (folder / f"{name}.csv").read_bytes()
        assert saved["tables"][name]["sha256"] == hashlib.sha256(data).hexdigest()
    if case != "fractional":
        (approval,) = saved["review"]["approvals"]
        assert approval["fovs_sha256"] == saved["tables"]["fovs"]["sha256"]
        assert approval["links_sha256"] == saved["tables"]["links"]["sha256"]


def test_a_changed_table_is_refused_naming_it(tmp_path):
    folder = save_placement(supplied_approved(), tmp_path / "s1")
    path = folder / "fovs.csv"
    original = path.read_bytes()
    path.write_bytes(original.replace(b"0.29999999999999999", b"0.29999999999999998"))
    with pytest.raises(ValueError, match="placement table fovs .*fovs.csv has SHA-256"):
        load_placement(folder)
    path.write_bytes(original)
    load_placement(folder)
    (folder / "links.csv").unlink()
    with pytest.raises(ValueError, match="placement table links .*links.csv is missing"):
        load_placement(folder)


def test_an_interrupted_save_leaves_no_partial_folder(monkeypatch, tmp_path):
    result = supplied_approved()
    written = []

    def fail_after_first(path, data):
        if written:
            raise RuntimeError("interrupted")
        written.append(path)
        Path(path).write_bytes(data)

    monkeypatch.setattr(_persist, "_write_file", fail_after_first)
    with pytest.raises(RuntimeError, match="interrupted"):
        save_placement(result, tmp_path / "stitching" / "s1")
    assert written and not (tmp_path / "stitching" / "s1").exists()
    assert list((tmp_path / "stitching").iterdir()) == []
    monkeypatch.undo()
    folder = save_placement(result, tmp_path / "stitching" / "s1")
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    written.clear()
    monkeypatch.setattr(_persist, "_write_file", fail_after_first)
    with pytest.raises(RuntimeError, match="interrupted"):
        save_placement(placement_from_table(stub_table(), grids_for(stub_table()), sample="s1"), folder)
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
    assert sorted(p.name for p in (tmp_path / "stitching").iterdir()) == ["s1"]
    (folder / "notes.txt").write_text("mine")
    monkeypatch.undo()
    with pytest.raises(FileExistsError, match="notes.txt"):
        save_placement(result, folder)


# --- PL15 ---------------------------------------------------------------------------------

class Counting:
    calls = 0


def counting_run(tiles, grids, nominal_zyx, pairs, *, config):
    Counting.calls += 1
    return stub_run(tiles, grids, nominal_zyx, pairs, config=config)


def test_stitch_refuses_bad_input_before_calling_the_method(monkeypatch):
    monkeypatch.setitem(STITCHING_METHODS, StubConfig, StitchingSpec("stub", counting_run))
    Counting.calls = 0
    table = stub_table()
    grids = grids_for(table)
    inputs = inputs_for(table, grids)
    config = StubConfig(zero(6))

    class Subclass(StubConfig):
        pass

    with pytest.raises(TypeError, match="no stitching method is registered for object"):
        stitch(inputs, table, config=object())
    with pytest.raises(TypeError, match="exact config type"):
        stitch(inputs, table, config=Subclass(zero(6)))
    with pytest.raises(ValueError, match="missing FOV5"):
        stitch(inputs[:5], table, config=config)
    with pytest.raises(ValueError, match="in table order; got the order FOV1, FOV0"):
        stitch([inputs[1], inputs[0], *inputs[2:]], table, config=config)
    with pytest.raises(TypeError, match="StitchingInput"):
        stitch([*inputs[:5], "FOV5"], table, config=config)
    odd = inputs_for(table, grids_for(table, FOV3=((4, 16, 24), (0.5, 0.25, 0.25))))
    with pytest.raises(IncompatibleGeometryError, match="one spacing"):
        stitch(odd, table, config=config)
    with pytest.raises(ValueError, match="fallback"):
        stitch(inputs, table, config=config, fallback="stage")
    assert Counting.calls == 0
    stitch(inputs, table, config=config)
    assert Counting.calls == 1


def test_stitch_and_the_supplied_placement_touch_no_file(stub, monkeypatch, tmp_path):
    table = stub_table()
    grids = grids_for(table)
    inputs = inputs_for(table, grids)
    work = tmp_path / "readonly"
    work.mkdir()
    work.chmod(stat.S_IRUSR | stat.S_IXUSR)
    monkeypatch.chdir(work)
    try:
        result = stitch(inputs, table, config=StubConfig(FRACTIONAL))
        supplied = placement_from_table(table, grids, sample="s1")
    finally:
        work.chmod(stat.S_IRWXU)
    assert list(work.iterdir()) == []
    assert set(result.fovs.status) == {"image_supported"} and len(supplied.fovs) == 6


def test_registry_uses_the_shared_mechanism(stub):
    assert names(STITCHING_METHODS) == ("multiview_stitcher", "stub")
    assert config_type_for(STITCHING_METHODS, "stub", "stitching method") is StubConfig
    assert spec_for(STITCHING_METHODS, StubConfig(zero(6)), "stitching method") is STUB_SPEC
    assert "supplied" not in names(STITCHING_METHODS) and "bigstitcher" not in names(STITCHING_METHODS)
    table = stub_table()
    result = stitch(inputs_for(table, grids_for(table)), table, config=StubConfig(zero(6)))
    assert result.record["method"] == {
        "stage": "stitching", "method": "stub", "config_type": "test.stitching_fixtures.StubConfig",
        "implementation": "test.stitching_fixtures.stub_run",
        "config": {"translations": [[0.0, 0.0, 0.0]] * 6, "overrides": [], "method": "stub"},
        "requires": {}, "artifacts": []}


def test_the_package_registers_only_multiview_stitcher():
    assert names(STITCHING_METHODS) == ("multiview_stitcher",)
    assert starfinder.stitching.STITCHING_METHODS is STITCHING_METHODS


def test_spec_rules_and_a_missing_dependency(monkeypatch):
    spec = StitchingSpec("plain", stub_run)
    assert spec.devices == ("cpu",) and spec.transform_kind == "translation"
    assert spec.dimensions == frozenset({2, 3}) and spec.min_shape_zyx == (1, 1, 1) and spec.requires == ()
    for change, match in ((dict(devices=("cuda",)), "CPU only"), (dict(devices=("cpu", "cuda")), "CPU only"),
                          (dict(transform_kind="affine"), "translation"), (dict(dimensions=frozenset({1})), "2 and 3"),
                          (dict(name="Bad Name"), "snake_case")):
        with pytest.raises(ValueError, match=match):
            StitchingSpec(**(dict(name="plain", run=stub_run) | change))
    missing = StitchingSpec("needs_missing", stub_run, requires=(
        Dependency("starfinder_no_such_module", "no-such-dist", extra="multiview-stitcher"),))
    monkeypatch.setitem(STITCHING_METHODS, StubConfig, missing)
    table = stub_table()
    with pytest.raises(StitchingBackendUnavailableError,
                       match=r"requires starfinder_no_such_module; install the 'multiview-stitcher' extra"):
        stitch(inputs_for(table, grids_for(table)), table, config=StubConfig(zero(6)))


def _environment():
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", NUMBA_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), *filter(None, [env.get("PYTHONPATH")])])
    return env


def test_importing_the_module_imports_no_backend():
    code = ("import sys, starfinder.stitching; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('multiview_stitcher', 'spatialdata')))")
    output = subprocess.run([sys.executable, "-c", code], cwd=SRC, env=_environment(), capture_output=True,
                            text=True, check=True)
    assert output.stdout.strip() == "[]"


# --- PL16 ---------------------------------------------------------------------------------

@pytest.mark.slow
def test_stub_path_is_deterministic_across_processes(tmp_path):
    code = ("import json, sys; from test.stitching_fixtures import determinism_hashes; "
            "print(json.dumps(determinism_hashes(sys.argv[1]), sort_keys=True))")
    runs = []
    for k in range(3):
        output = subprocess.run([sys.executable, "-c", code, str(tmp_path / f"run{k}")], cwd=SRC,
                                env=_environment(), capture_output=True, text=True, check=True)
        runs.append(json.loads(output.stdout))
    assert runs[0] == runs[1] == runs[2]
    assert len(runs[0]) == 8
    assert runs[0] == determinism_hashes(tmp_path / "here")


# --- PL17 ---------------------------------------------------------------------------------

IDS = [f"Position{k:03d}" for k in range(1, 5)]
REFERENCE = metadata(frame="reference")


def dataset_folder(tmp_path, shapes=None):
    dataset = Dataset(tmp_path / "input", tmp_path / "output", "dataset", "sample", "output")
    images = {}
    for k, fov in enumerate(IDS):
        shape = (shapes or {}).get(fov, (4, 16, 24))
        image = (np.arange(np.prod(shape)).reshape(shape) % 200 + k).astype(np.uint8)
        save_volume(image, tmp_path / "output" / "images" / "ref_merged" / f"{fov}.tif", metadata=REFERENCE)
        images[fov] = image
    return dataset, images


POSITIONS = {"source": "regular_grid",
             "regular_grid": {"columns": 2, "rows": 2, "order": "column_major_down_right", "overlap": 0.25}}


@pytest.fixture
def no_fov_stage(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("Dataset.stitch must not call a per-FOV stage")
    for name in ("run", "segment", "assign"):
        monkeypatch.setattr(FOV, name, refuse)


@pytest.fixture
def spy(monkeypatch):
    calls = []

    def recording(inputs, table, **kwargs):
        calls.append((list(inputs), table, kwargs))
        result = real_stitch(inputs, table, **kwargs)
        check_components(result)
        return result

    monkeypatch.setattr(starfinder.stitching, "stitch", recording)
    return calls


@pytest.fixture
def reads(monkeypatch):
    counts = {"load_volume": 0, "load_volume_zyxc": 0}
    for name in counts:
        original = getattr(starfinder.io, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(starfinder.io, name, counted)
    return counts


def test_dataset_stitch_builds_the_table_from_the_grids(stub, no_fov_stage, spy, reads, tmp_path):
    dataset, images = dataset_folder(tmp_path)
    out = tmp_path / "output" / "stitching" / "sample1"
    result = dataset.stitch("sample1", IDS, table=POSITIONS, config=StubConfig(FRACTIONAL[:4]), output_dir=out)
    ((inputs, table, kwargs),) = spy
    assert reads == {"load_volume": 4, "load_volume_zyxc": 0}
    expected = positions_from_grid(columns=2, rows=2, order="column_major_down_right", tile_shape_yx=(16, 24),
                                   spacing_yx=(0.25, 0.25), overlap=0.25, fov_ids=IDS)
    pd.testing.assert_frame_equal(table.positions, expected.positions, check_exact=True)
    assert table.record["tile_shape_yx"] == [16, 24] and table.record["spacing_yx"] == [0.25, 0.25]
    assert table.record["geometry_source"] == {"tile_shape_yx": "reference_grids", "spacing_yx": "reference_grids"}
    assert [item.fov_id for item in inputs] == IDS and kwargs["sample"] == "sample1"
    for item in inputs:
        path = tmp_path / "output" / "images" / "ref_merged" / f"{item.fov_id}.tif"
        assert np.array_equal(item.image, images[item.fov_id]) and item.source == "reference_merged"
        assert item.file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        reference = reference_grid_from_file(path)
        assert item.grid == reference
    assert sorted(p.name for p in out.iterdir()) == ["fovs.csv", "links.csv", "placement.json"]
    loaded = load_placement(out)
    pd.testing.assert_frame_equal(loaded.fovs, result.fovs, check_exact=True)
    assert result.sample == "sample1" and set(result.fovs.status) == {"image_supported"}


def test_dataset_stitch_builds_inputs_in_table_order(stub, no_fov_stage, spy, tmp_path):
    dataset, _ = dataset_folder(tmp_path)
    grid_table = positions_from_grid(columns=2, rows=2, order="column_major_down_right", tile_shape_yx=(16, 24),
                                     spacing_yx=(0.25, 0.25), overlap=0.25, fov_ids=IDS)
    reordered = positions_from_table(grid_table.positions.iloc[[2, 0, 3, 1]].drop(columns=["grid_column",
                                                                                             "grid_row"]))
    result = dataset.stitch("sample1", IDS, table=reordered, config=StubConfig(zero(4)))
    ((inputs, table, _),) = spy
    assert [item.fov_id for item in inputs] == [IDS[2], IDS[0], IDS[3], IDS[1]] == list(result.fovs.fov_id)
    with pytest.raises(ValueError, match="exactly fov_ids"):
        dataset.stitch("sample1", IDS[:3], table=reordered, config=StubConfig(zero(3)))


def test_dataset_stitch_refuses_another_tile_size(stub, no_fov_stage, tmp_path):
    dataset, _ = dataset_folder(tmp_path, shapes={"Position003": (4, 16, 20)})
    with pytest.raises(IncompatibleGeometryError,
                       match=r"tile_shape_yx \(FOVs Position001, Position002, Position004: \(16, 24\); "
                             r"FOVs Position003: \(16, 20\)\)"):
        dataset.stitch("sample1", IDS, table=POSITIONS, config=StubConfig(zero(4)))
    given = dict(POSITIONS, tile_shape_yx=[16, 24], spacing_yx=[0.25, 0.25])
    result = dataset.stitch("sample1", IDS, table=given, config=StubConfig(zero(4)))
    check_components(result)
    assert result.record["table"]["geometry_source"] == {"tile_shape_yx": "configuration",
                                                         "spacing_yx": "configuration"}


def test_dataset_stitch_reads_the_saved_reference_stain(stub, no_fov_stage, spy, reads, tmp_path):
    dataset, images = dataset_folder(tmp_path)
    for fov, image in images.items():
        stain = (image // 2)[..., None]
        save_volume(stain, tmp_path / "output" / "checkpoints" / fov / "other_rounds" / "reference_stain" /
                    "image.ome.tif", metadata=REFERENCE)
    result = dataset.stitch("sample1", IDS, table=POSITIONS, config=StubConfig(zero(4)), input="reference_stain")
    ((inputs, _, _),) = spy
    for item in inputs:
        assert item.source == "reference_stain" and np.array_equal(item.image, images[item.fov_id] // 2)
    assert reads == {"load_volume": 4, "load_volume_zyxc": 4}
    assert result.record["inputs"][IDS[0]]["source"] == "reference_stain"


def test_dataset_stitch_reads_a_file_pattern(stub, no_fov_stage, spy, tmp_path):
    dataset, images = dataset_folder(tmp_path)
    for fov, image in images.items():
        path = tmp_path / "output" / "images" / "DAPI" / f"{fov}.tif"
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(path, image[::-1].copy(), photometric="minisblack")
    (tmp_path / "output" / "images" / "DAPI" / f"{IDS[3]}.tif").unlink()
    result = dataset.stitch("sample1", IDS, table=POSITIONS, config=StubConfig(zero(4)),
                            input="file:images/DAPI/{fovID}.tif")
    ((inputs, _, _),) = spy
    for item in inputs[:3]:
        expected = (tmp_path / "output" / "images" / "DAPI" / f"{item.fov_id}.tif").resolve()
        assert item.source == f"file:{expected}" and np.array_equal(item.image, images[item.fov_id][::-1])
    assert inputs[3].image is None and result.fovs.status[3] == "missing"
    tifffile.imwrite(tmp_path / "output" / "images" / "DAPI" / f"{IDS[3]}.tif", np.zeros((4, 16, 20), np.uint8),
                     photometric="minisblack")
    with pytest.raises(IncompatibleGeometryError, match=r"FOV 'Position004' .* shape \(4, 16, 20\); its reference "
                                                        r"grid \(4, 16, 24\)"):
        dataset.stitch("sample1", IDS, table=POSITIONS, config=StubConfig(zero(4)),
                       input="file:images/DAPI/{fovID}.tif")


def test_dataset_stitch_errors(stub, tmp_path):
    dataset, _ = dataset_folder(tmp_path)
    with pytest.raises(ValueError, match="input must be"):
        dataset.stitch("sample1", IDS, table=POSITIONS, config=StubConfig(zero(4)), input="dapi")
    with pytest.raises(ValueError, match="unknown positions keys"):
        dataset.stitch("sample1", IDS, table=dict(POSITIONS, guess=1), config=StubConfig(zero(4)))
    with pytest.raises(ValueError, match="stage_axes"):
        dataset.stitch("sample1", IDS, table={"source": "stage_file"}, config=StubConfig(zero(4)))
    (tmp_path / "output" / "images" / "ref_merged" / f"{IDS[0]}.tif").unlink()
    with pytest.raises(FileNotFoundError, match="Position001"):
        dataset.stitch("sample1", IDS, table=POSITIONS, config=StubConfig(zero(4)))


def test_dataset_stitch_reads_the_sample_stage_file(stub, no_fov_stage, tmp_path):
    dataset, _ = dataset_folder(tmp_path)
    points = ((1, "0.05", "0.02"), (2, "0.05", "0.020003"), (3, "0.050003", "0.02"), (4, "0.050003", "0.020003"))
    path = tmp_path / "output" / "documents" / "maf" / "sample1.maf"
    path.parent.mkdir(parents=True)
    path.write_text(maf_text(points))
    positions = {"source": "stage_file", "stage_axes": {"x": "+x", "y": "+y"}, "snap_to_grid": {"overlap": 0.25}}
    result = dataset.stitch("sample1", IDS, table=positions, config=StubConfig(zero(4)))
    check_components(result)
    record = result.record["table"]
    assert record["converter"] == "positions_from_leica" and record["path"] == str(path.resolve())
    assert record["fov_ids"] == {str(k): f"Position{k:03d}" for k in range(1, 5)}
    assert record["snap_to_grid"]["tile_shape_yx"] == [16, 24] and record["snap_to_grid"]["cell_step_um"] == 3.0
    assert record["geometry_source"] == {"tile_shape_yx": "reference_grids", "spacing_yx": "reference_grids"}
    assert list(result.fovs.nominal_y) == [0.0, 3.0, 0.0, 3.0] and list(result.fovs.nominal_x) == [0.0, 0.0, 4.5, 4.5]
