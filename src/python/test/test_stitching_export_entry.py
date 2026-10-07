"""EX1 (its default and fuse parts), EX2 (checks 1 to 5), EX3 (a) and (b), EX4, AS15 and FI11 (through the entry)
of docs/stitching-assembly-validation.md: ``Dataset.export``, its record and its folder.

Fixture: ``export_folder``, ``assembly_golden`` (3D, and with one-plane FOVs for the 2D
case of ``export_cases``) written as a dataset by ``fusion_fixtures.export_folder``: the saved
placement, the §2.9 assignment folders, the reference images and a ``reference_stain`` per
FOV (uint8, ``(z + y + x) mod 251``, 8×48×64). The label and SpatialData parts are not
implemented yet, so they are off. The missing extras are made by making the import fail
inside the test. dask runs with the synchronous scheduler.
"""
import builtins
import importlib.util
import json
import re
import sys
import weakref

import numpy as np
import pandas as pd
import pytest
import tifffile
from pandas.testing import assert_frame_equal

import starfinder.assembly
import starfinder.assignment._persist as assignment_persist
import starfinder.sample_output._export as export_module
import starfinder.sample_output._tiff as tiff_module
from starfinder.assembly import assemble, load_assembly, save_assembly
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.io import save_volume
from starfinder.sample_output import (
    ExportConfig,
    ExportIncompleteWarning,
    FusionConfig,
    FusionInput,
    check_export,
    fuse,
)
from starfinder.sample_output._export import assembly_inputs, fusion_inputs
from starfinder.sample_output._record import section_sha256
from starfinder.sample_output._tiff import read_ome_layout
from starfinder.stitching import PlacementReviewRequiredError, load_placement, save_placement

from .assembly_fixtures import SAMPLE, golden_origins, golden_placement
from .fusion_fixtures import canonical, export_folder, listing, make_case, quad

EXTRA = "the multiview-stitcher extra (starfinder[multiview-stitcher]) is not installed"
pytestmark = [pytest.mark.stitching, pytest.mark.validation,
              pytest.mark.skipif(importlib.util.find_spec("anndata") is None,
                                 reason="the anndata extra (starfinder[anndata]) is not installed")]
needs_library = pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None, reason=EXTRA)
DEFAULT_FILES = ["export.json", "reference_stain_projection.ome.tif", "sample.h5ad", "tables/cells.csv",
                 "tables/counts.csv", "tables/fovs.csv", "tables/molecules.csv"]
TABLE_FILES = ["export.json", "sample.h5ad", "tables/cells.csv", "tables/counts.csv", "tables/fovs.csv",
               "tables/molecules.csv"]


@pytest.fixture(autouse=True)
def synchronous():
    import dask
    with dask.config.set(scheduler="synchronous"):
        yield


@pytest.fixture
def folder(tmp_path):
    return export_folder(tmp_path)


def export(f, config=None, **kwargs):
    return f.dataset.export(SAMPLE, placement=kwargs.pop("placement", f.placement_dir), config=config,
                            output_dir=kwargs.pop("output_dir", f.export_dir), **kwargs)


def record_of(f):
    return json.loads((f.export_dir / "export.json").read_text())


def temporaries(folder):
    return [p.name for p in folder.rglob("*") if p.name.startswith(".") or ".tmp" in p.name]


# --- EX1: the parts, in order, and the default folder ---------------------------------------------------------

@needs_library
def test_default_export(folder, monkeypatch):
    calls = []
    real_assemble, real_fuse = starfinder.assembly.assemble, export_module.fuse
    monkeypatch.setattr(starfinder.assembly, "assemble", lambda *a, **k: calls.append("assemble") or
                        real_assemble(*a, **k))
    monkeypatch.setattr(export_module, "fuse", lambda *a, **k: calls.append("fuse") or real_fuse(*a, **k))
    before = {p for p in listing(folder.root) if not p.startswith("export/")}
    record = export(folder)
    assert calls == ["assemble", "fuse"]
    assert sorted(listing(folder.export_dir)) == DEFAULT_FILES
    assert {p for p in listing(folder.root) if not p.startswith("export/")} == before
    assert not (folder.root / "images" / "fused").exists()
    exported = record_of(folder)
    assert exported == dict(record.record) and record.complete and exported["complete"] is True
    assert exported["parts"] == {"assemble": "written", "fuse": "written", "fuse_labels": "off",
                                 "export_spatialdata": "off"}
    assert exported["labels"] == {"status": "off"} and exported["spatialdata"] == {"status": "off"}
    entry = exported["fusion"]["outputs"][0]
    assert entry["path"] == "reference_stain_projection.ome.tif" and entry["dimensionality"] == "projection"
    assert entry["projection"] == "max" and entry["factor_yx"] == 2 and entry["dtype"] == "uint8"
    assert tifffile.imread(folder.export_dir / entry["path"]).dtype == np.uint8
    assert exported["fusion"]["assembly_sha256"] == section_sha256(exported["assembly"])
    assert exported["fusion"]["viewer_qualification"] == "deferred"
    assert all(e["images"]["reference_stain"]["path"] == str(folder.stain_path(e["fov_id"]).resolve())
               for e in exported["fusion"]["fovs"])
    check_export(folder.export_dir)


@needs_library
def test_two_image_names(folder):
    record = export(folder, ExportConfig(fusion=FusionConfig(images=("reference_stain", "reference_merged"))))
    outputs = record.record["fusion"]["outputs"]
    assert [o["path"] for o in outputs] == ["reference_stain_projection.ome.tif",
                                            "reference_merged_projection.ome.tif"]
    for key in ("shape_zyx", "spacing_zyx", "origin_zyx"):
        assert outputs[0][key] == outputs[1][key]
    assert not np.array_equal(*(tifffile.imread(folder.export_dir / o["path"]) for o in outputs))
    check_export(folder.export_dir)


@needs_library
def test_parts_called_alone(folder):
    """Each part alone, on the saved files of the parts before it, writes its section and leaves the others."""
    export(folder)
    before = record_of(folder)
    placement = load_placement(folder.placement_dir)
    saved = load_assembly(folder.export_dir)
    fuse(placement, fusion_inputs(folder.dataset, placement, ExportConfig()), assembly=saved,
         config=FusionConfig(dimensionality="stack"), output_dir=folder.export_dir)
    after = record_of(folder)
    assert canonical(after["assembly"]) == canonical(before["assembly"])
    assert after["fusion"]["assembly_sha256"] == section_sha256(after["assembly"])
    assert [o["path"] for o in after["fusion"]["outputs"]] == ["reference_stain_stack.ome.tif"]
    assert not (folder.export_dir / "reference_stain_projection.ome.tif").exists()
    check_export(folder.export_dir)
    result = assemble(placement, assembly_inputs(folder.dataset, placement, "default"))
    save_assembly(result, folder.export_dir)
    again = record_of(folder)
    assert canonical(again["fusion"]) == canonical(after["fusion"])


# --- AS15: the assemble part -----------------------------------------------------------------------------------

def test_assemble_part(folder, monkeypatch):
    live, peak, calls = set(), [0], [0]
    original = assignment_persist.read_labels

    def tracked(path):
        labels, metadata = original(path)
        key = id(labels)
        live.add(key)
        calls[0] += 1
        peak[0] = max(peak[0], len(live))
        weakref.finalize(labels, live.discard, key)
        return labels, metadata

    monkeypatch.setattr(assignment_persist, "read_labels", tracked)
    record = export(folder, ExportConfig(fuse=False))
    assert calls[0] >= 2 * 3 and peak[0] <= 2                       # each territory read, at most two at a time
    monkeypatch.setattr(assignment_persist, "read_labels", original)
    placement = load_placement(folder.placement_dir)
    direct = assemble(placement, assembly_inputs(folder.dataset, placement, "default"))
    saved = load_assembly(folder.export_dir)
    for name in ("fovs", "cells", "molecules", "counts"):
        assert_frame_equal(getattr(saved, name), getattr(direct, name), check_exact=True)
    section = record.record["assembly"]
    unsaved = ("files", "table_format")                       # what save_assembly adds; no timing field differs
    assert {k: v for k, v in section.items() if k not in unsaved} == \
        {k: v for k, v in json.loads(json.dumps(direct.record)).items() if k not in unsaved}
    assert sorted(listing(folder.export_dir)) == TABLE_FILES
    assert record.parts["assemble"] == "written" and record.record["fusion"] == {"status": "off"}
    check_export(folder.export_dir)
    # A rerun with assemble off reuses the saved assembly, checking every SHA-256.
    record = export(folder, ExportConfig(assemble=False, fuse=False))
    assert record.parts["assemble"] == "reused" and sorted(listing(folder.export_dir)) == TABLE_FILES
    check_export(folder.export_dir)
    table = folder.export_dir / "tables" / "cells.csv"
    table.write_text(table.read_text() + "\n")
    with pytest.raises(ValueError, match="cells.csv"):
        export(folder, ExportConfig(assemble=False, fuse=False))


def test_assemble_off_without_a_saved_assembly(folder):
    record = export(folder, ExportConfig(assemble=False, fuse=False))
    assert record.parts["assemble"] == "off" and record.record["assembly"] == {"status": "off"}
    assert sorted(listing(folder.export_dir)) == ["export.json"]
    check_export(folder.export_dir)


def test_assemble_part_input_files(folder):
    """A changed assignment file raises naming the path; a missing or changed territory file names both hashes."""
    target = folder.assignment_dir("Position002") / "cell_labels.tif"
    original = target.read_bytes()
    target.unlink()
    with pytest.raises(ValueError, match=f"{re.escape(str(target))}.*missing.*recorded SHA-256"):
        export(folder, ExportConfig(fuse=False))
    labels = tifffile.imread(folder.assignment_dir("Position001") / "cell_labels.tif")
    save_volume(labels.astype(np.uint32), target)
    with pytest.raises(ValueError, match=f"{re.escape(str(target))} has SHA-256 [0-9a-f]{{64}}, the record "
                                         "[0-9a-f]{64}"):
        export(folder, ExportConfig(fuse=False))
    target.write_bytes(original)
    table = folder.assignment_dir("Position003") / "cells.csv"
    table.write_text(table.read_text().replace("\n", "\n", 1) + "\n")
    with pytest.raises(ValueError, match=re.escape(str(table))):
        export(folder, ExportConfig(fuse=False))
    assert not folder.export_dir.exists()


def test_assemble_part_from_molecule_tables(folder):
    """assignment=None: the molecule tables of signal/<fov>_goodSpots.csv, a molecules_only sample."""
    record = export(folder, ExportConfig(fuse=False), assignment=None)
    placement = load_placement(folder.placement_dir)
    direct = assemble(placement, assembly_inputs(folder.dataset, placement, None))
    saved = load_assembly(folder.export_dir)
    assert record.record["assembly"]["mode"] == "molecules_only" and len(saved.cells) == 0
    for name in ("fovs", "cells", "molecules", "counts"):
        assert_frame_equal(getattr(saved, name), getattr(direct, name), check_exact=True)
    assert len(saved.molecules) == sum(len(r.molecules) for r in folder.results.values())
    check_export(folder.export_dir)


def test_molecule_tables_are_the_goodspots_rows(folder):
    """assignment=None: the assembled molecules are exactly the rows of each FOV's goodSpots CSV.

    The CSV ``signal/<fov_id>_goodSpots.csv`` is read here with pandas, not with
    ``molecule_table_from_csv`` or anything built on it. The rule is that of
    docs/assignment-contract.md ("The assign entry", "Inputs", ``molecule_table_from_csv``): a
    legacy one-based ``x, y, z, gene`` CSV, 1 subtracted from each coordinate, row ``i`` (zero-based)
    with the identity ``spot_id = "csv:<i>"``; the namespace is the FOV's
    ``[dataset_id, sample_id, fov_id, null]``. Assignments made by the §2.9 workflow route from the
    same files (``molecule_table_from_csv``, then ``assign_molecules``, saved as ``from_csv``)
    give a sample with the same molecule keys.
    """
    from starfinder.assignment import assign_molecules, molecule_table_from_csv
    from starfinder.assignment._persist import write_assignment

    from .assembly_fixtures import GOLDEN_GENES, GOLDEN_IDS, label_run, unit_grid, with_record
    from .fusion_fixtures import TILE_ZYX
    export(folder, ExportConfig(fuse=False), assignment=None)
    molecules = load_assembly(folder.export_dir).molecules
    assert set(molecules.fov_id) == set(GOLDEN_IDS)
    for fov in GOLDEN_IDS:
        rows = pd.read_csv(folder.root / "signal" / f"{fov}_goodSpots.csv", dtype={"gene": str})
        got = molecules[molecules.fov_id == fov].set_index("spot_id")
        assert len(got) == len(rows) > 0
        assert set(got.index) == {f"csv:{i}" for i in range(len(rows))}
        namespace = json.dumps(["data", "sample", fov, None], separators=(",", ":"))
        assert set(got.spot_namespace) == {namespace}
        for i, row in rows.iterrows():
            molecule = got.loc[f"csv:{i}"]
            assert molecule.gene_id == row.gene
            assert (molecule.z, molecule.y, molecule.x) == (row.z - 1, row.y - 1, row.x - 1)
    grid = unit_grid(TILE_ZYX)
    for fov in GOLDEN_IDS:
        namespace = json.dumps(["data", "sample", fov, None], separators=(",", ":"))
        table = molecule_table_from_csv(folder.root / "signal" / f"{fov}_goodSpots.csv", spot_namespace=namespace,
                                        genes=GOLDEN_GENES)
        result = assign_molecules(table, label_run(folder.results[fov].cell_labels, fov, grid), grid=grid)
        write_assignment(folder.assignment_dir(fov, "from_csv"), with_record(result, name="from_csv"),
                         metadata=grid.metadata)
    other = folder.root / "export" / "from_csv"
    export(folder, ExportConfig(fuse=False), assignment="from_csv", output_dir=other)
    assigned = load_assembly(other).molecules
    keys = set(zip(molecules.spot_namespace, molecules.spot_id))
    assert keys == set(zip(assigned.spot_namespace, assigned.spot_id)) and len(keys) == len(molecules)


# --- EX2: checks before anything is written ----------------------------------------------------------------------

def unchanged(f, call, error, match=None):
    before = listing(f.root)
    with pytest.raises(error, match=match):
        call()
    assert listing(f.root) == before


def test_checks_before_writing(folder, monkeypatch, tmp_path):
    # 1. Config.
    with pytest.raises(ValueError, match="dimensionality"):
        ExportConfig(fusion=FusionConfig(dimensionality="cube"))
    # 2. Review: before 3, so a missing extra is not reported first.
    required = golden_placement(review="required")
    monkeypatch.setitem(sys.modules, "anndata", None)
    unchanged(folder, lambda: export(folder, placement=required), PlacementReviewRequiredError)
    # 3. Required extras.
    unchanged(folder, lambda: export(folder), ModuleNotFoundError, "'anndata' extra")
    monkeypatch.delitem(sys.modules, "anndata")
    # pandas 3 itself imports pyarrow, so pyarrow is made unimportable for the export entry's own check only.
    real_import = builtins.__import__

    def without_pyarrow(name, globals=None, *args, **kwargs):
        if name == "pyarrow" and (globals or {}).get("__name__") == export_module.__name__:
            raise ModuleNotFoundError("No module named 'pyarrow'")
        return real_import(name, globals, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pyarrow)
    unchanged(folder, lambda: export(folder, ExportConfig(table_format="parquet")), ModuleNotFoundError,
              "'checkpoint' extra")
    monkeypatch.setattr(builtins, "__import__", real_import)
    # 4. Assembly: assemble off with a later part that needs it and no saved assembly.
    unchanged(folder, lambda: export(folder, ExportConfig(assemble=False, fuse_labels=True)), ValueError,
              "holds no saved assembly")


def test_check_saved_assembly_of_another_placement(folder):
    export(folder, ExportConfig(fuse=False))
    moved = dict(golden_origins(), Position003=(-0.6, -0.3, 55.5))
    other = golden_placement(origins=moved)
    save_placement(other, folder.root / "other")
    unchanged(folder, lambda: export(folder, ExportConfig(assemble=False, fuse_labels=True),
                                     placement=folder.root / "other"), ValueError, "another placement")
    case = load_placement(folder.root / "other")
    saved = load_assembly(folder.export_dir)
    unchanged(folder, lambda: fuse(case, fusion_inputs(folder.dataset, case, ExportConfig()), assembly=saved,
                                   output_dir=folder.export_dir), ValueError, "another placement")


def test_check_fusion_inputs(folder):
    placement = load_placement(folder.placement_dir)
    inputs = fusion_inputs(folder.dataset, placement, ExportConfig())
    unchanged(folder, lambda: fuse(placement, inputs[1:], output_dir=folder.export_dir), ValueError,
              "FOVs without one: Position001")
    foreign = FusionInput("Position009", inputs[0].grid, inputs[0].images)
    unchanged(folder, lambda: fuse(placement, inputs + [foreign], output_dir=folder.export_dir), ValueError,
              "not included FOVs.*Position009")
    # A grid other than the placement's: through the entry (assemble meets it first), and with assemble off.
    grid = inputs[1].grid
    other = ImageMetadata("fov", (1.0, 0.5, 0.5), (0.0, 0.0, 0.0), grid.metadata.direction_zyx, "µm")
    save_volume(np.zeros(grid.shape_zyx, np.uint8), folder.root / "images" / "ref_merged" / "Position002.tif",
                metadata=other)
    unchanged(folder, lambda: export(folder), IncompatibleGeometryError, "Position002")
    unchanged(folder, lambda: export(folder, ExportConfig(assemble=False)), IncompatibleGeometryError,
              "Position002.*not the grid the placement recorded")


def test_check_file_source_shape(folder):
    for fov in ("Position001", "Position002", "Position003"):
        save_volume(np.ones((8, 48, 64), np.uint8), folder.root / "extra" / f"{fov}.tif")
    save_volume(np.ones((8, 48, 60), np.uint8), folder.root / "extra" / "Position003.tif")
    config = ExportConfig(fusion=FusionConfig(images=("dapi=file:extra/{fov_id}.tif",)))
    unchanged(folder, lambda: export(folder, config), IncompatibleGeometryError,
              "source image 'dapi' of FOV Position003.*\\(8, 48, 60\\)")


def test_check_spacing_and_planes(tmp_path):
    case = quad()
    grids = dict(case.grids)
    from .fusion_fixtures import grid_of
    two = make_case("two_spacings", {"Q0": (0.0, 0.0, 0.0), "Q1": (0.0, 26.3, 0.4)},
                    {"Q0": case.arrays["Q0"], "Q1": case.arrays["Q1"]},
                    grids={"Q0": grids["Q0"], "Q1": grid_of((4, 32, 32), (1.0, 0.5, 0.5))})
    with pytest.raises(ValueError, match="share one spacing"):
        fuse(two.placement, two.inputs(), output_dir=tmp_path)
    mixed = make_case("mixed_planes", {"Q0": (0.0, 0.0, 0.0), "Q1": (0.0, 26.3, 0.4)},
                      {"Q0": case.arrays["Q0"], "Q1": case.arrays["Q1"][:1]})
    with pytest.raises(ValueError, match="one-plane FOVs cannot be fused beside multi-plane"):
        fuse(mixed.placement, mixed.inputs(), output_dir=tmp_path)
    assert not any(tmp_path.iterdir())


# --- EX3: a part that cannot run --------------------------------------------------------------------------------

def test_fuse_without_the_extra(folder, monkeypatch):
    """(a) multiview_stitcher not importable: assemble runs, fuse is not_run (extra_missing)."""
    monkeypatch.setitem(sys.modules, "multiview_stitcher", None)
    with pytest.warns(ExportIncompleteWarning, match="fuse could not run: extra_missing") as caught:
        record = export(folder)
    assert len([w for w in caught if w.category is ExportIncompleteWarning]) == 1
    exported = record.record
    assert exported["parts"] == {"assemble": "written", "fuse": "not_run", "fuse_labels": "off",
                                 "export_spatialdata": "off"}
    assert exported["complete"] is False and not record.complete
    fusion = exported["fusion"]
    assert fusion["reason"] == "extra_missing" and fusion["extra"] == "multiview-stitcher" and fusion["outputs"] == []
    assert sorted(listing(folder.export_dir)) == TABLE_FILES
    check_export(folder.export_dir)


@needs_library
def test_fuse_with_a_source_image_missing(folder):
    """(b) one FOV without its reference_stain: fuse is not_run (source_image_missing), no image for any name."""
    folder.stain_path("Position002").unlink()
    config = ExportConfig(fusion=FusionConfig(images=("reference_stain", "reference_merged")))
    with pytest.warns(ExportIncompleteWarning, match="source_image_missing \\(Position002 reference_stain\\)"):
        record = export(folder, config)
    fusion = record.record["fusion"]
    assert record.parts["fuse"] == "not_run" and fusion["reason"] == "source_image_missing"
    assert fusion["missing"] == [{"fov_id": "Position002", "image": "reference_stain",
                                  "path": str(folder.stain_path("Position002"))}]
    assert record.record["complete"] is False and sorted(listing(folder.export_dir)) == TABLE_FILES
    check_export(folder.export_dir)


# --- EX4: the folder equals its record ----------------------------------------------------------------------------

@needs_library
def test_folder_equals_its_record(folder, monkeypatch):
    export(folder, ExportConfig(fusion=FusionConfig(dimensionality="both")))
    assert (folder.export_dir / "reference_stain_stack.ome.tif").exists()
    check_export(folder.export_dir)
    unrelated = folder.export_dir / "notes.txt"
    unrelated.write_text("kept")
    export(folder, ExportConfig(fusion=FusionConfig(dimensionality="projection")))
    assert not (folder.export_dir / "reference_stain_stack.ome.tif").exists() and unrelated.read_text() == "kept"
    with pytest.raises(ValueError, match="R2: the file notes.txt is not listed"):
        check_export(folder.export_dir)
    unrelated.unlink()
    check_export(folder.export_dir)
    # A listed file deleted, then the assembly section edited.
    projection = folder.export_dir / "reference_stain_projection.ome.tif"
    kept = projection.read_bytes()
    projection.unlink()
    with pytest.raises(ValueError, match="R2: the listed file reference_stain_projection.ome.tif is missing"):
        check_export(folder.export_dir)
    projection.write_bytes(kept)
    record = record_of(folder)
    record["assembly"]["mode"] = "edited"
    (folder.export_dir / "export.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="R1: the fusion section records the assembly section's SHA-256"):
        check_export(folder.export_dir)


@needs_library
def test_interrupted_export(folder, monkeypatch):
    """A run interrupted between two files of the fuse part: the record lists what finished, nothing temporary stays."""
    real, calls = tiff_module.write_ome_tiff, []

    def interrupted(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 2:
            raise RuntimeError("interrupted by the test")
        return real(*args, **kwargs)

    monkeypatch.setattr(tiff_module, "write_ome_tiff", interrupted)
    config = ExportConfig(fusion=FusionConfig(images=("reference_stain", "reference_merged")))
    with pytest.raises(RuntimeError, match="interrupted by the test"):
        export(folder, config)
    exported = record_of(folder)
    assert exported["parts"]["assemble"] == "written" and exported["parts"]["fuse"] == "failed"
    assert exported["fusion"]["error"] == {"type": "RuntimeError", "message": "interrupted by the test"}
    assert [o["path"] for o in exported["fusion"]["outputs"]] == ["reference_stain_projection.ome.tif"]
    assert exported["complete"] is False
    assert sorted(listing(folder.export_dir)) == sorted(TABLE_FILES + ["reference_stain_projection.ome.tif"])
    assert temporaries(folder.export_dir) == []
    check_export(folder.export_dir)


@needs_library
def test_assemble_off_and_fuse_on(folder):
    """With the assemble part off and no saved assembly: no table, no sample.h5ad, R3 holds; fuse runs alone."""
    record = export(folder, ExportConfig(assemble=False))
    assert record.parts["assemble"] == "off" and record.parts["fuse"] == "written"
    assert record.record["fusion"]["assembly_sha256"] is None
    assert sorted(listing(folder.export_dir)) == ["export.json", "reference_stain_projection.ome.tif"]
    check_export(folder.export_dir)


# --- FI11: the stack alone through the entry ---------------------------------------------------------------------

@needs_library
def test_stack_alone_through_the_entry(folder):
    stacks = {}
    for projection in ("max", "sum"):
        record = export(folder, ExportConfig(fusion=FusionConfig(dimensionality="stack", projection=projection)))
        files = sorted(listing(folder.export_dir))
        assert files == sorted(TABLE_FILES + ["reference_stain_stack.ome.tif"])
        section = record.record["fusion"]
        assert section["config"]["dimensionality"] == "stack"
        entry = section["outputs"][0]
        grid = section["grid"]
        assert [o["path"] for o in section["outputs"]] == ["reference_stain_stack.ome.tif"]
        assert entry["dimensionality"] == "stack" and entry["factor_yx"] == 2 and entry["dtype"] == "uint8"
        assert entry["shape_zyx"] == [grid["shape_zyx"][0], -(-grid["shape_zyx"][1] // 2),
                                      -(-grid["shape_zyx"][2] // 2)]
        assert abs(entry["spacing_zyx"][0] - grid["spacing_zyx"][0]) <= 1e-9
        assert abs(entry["origin_zyx"][0] - grid["origin_zyx"][0]) <= 1e-9
        path = folder.export_dir / entry["path"]
        whole = tifffile.imread(path)
        i = entry["shape_zyx"][0] // 2
        assert np.array_equal(tifffile.imread(path, key=i), whole[i])
        layout = read_ome_layout(path)
        assert layout["axes"] == "ZYX"
        plane = layout["planes"][i]
        assert abs(float(plane["PositionZ"]) - (entry["origin_zyx"][0] + i * entry["spacing_zyx"][0])) <= 1e-9
        assert abs(float(plane["PositionY"]) - entry["origin_zyx"][1]) <= 1e-9
        assert abs(float(plane["PositionX"]) - entry["origin_zyx"][2]) <= 1e-9
        check_export(folder.export_dir)
        stacks[projection] = whole
    assert np.array_equal(stacks["max"], stacks["sum"])
    record = export(folder, ExportConfig(fusion=FusionConfig(dimensionality="stack",
                                                             images=("reference_stain", "reference_merged"))))
    assert [o["path"] for o in record.record["fusion"]["outputs"]] == ["reference_stain_stack.ome.tif",
                                                                      "reference_merged_stack.ome.tif"]
    assert not list(folder.export_dir.glob("*_projection.ome.tif"))
    check_export(folder.export_dir)


@needs_library
def test_stack_of_one_plane_fovs(tmp_path):
    """export_cases' 2D assembly_golden (one-plane FOVs) with dimensionality stack: a 2D file, its only plane."""
    f = export_folder(tmp_path, dimensions="plane")
    record = export(f, ExportConfig(fusion=FusionConfig(dimensionality="stack")))
    section = record.record["fusion"]
    entry = section["outputs"][0]
    assert section["library"]["inputs_2d"] is True and entry["path"] == "reference_stain_stack.ome.tif"
    assert section["grid"]["origins_z"] == {"Position001": 0.0, "Position002": 1.3, "Position003": -0.6}
    z_lo, z_hi = section["grid"]["slab_z"]
    assert abs(entry["spacing_zyx"][0] - (z_hi - z_lo)) <= 1e-9
    assert abs(entry["origin_zyx"][0] - (z_lo + z_hi) / 2) <= 1e-9
    path = f.export_dir / entry["path"]
    whole = tifffile.imread(path)
    assert whole.ndim == 2 and read_ome_layout(path)["axes"] == "YX" and entry["shape_zyx"][0] == 1
    assert np.array_equal(tifffile.imread(path, key=0), whole)
    assert sorted(listing(f.export_dir)) == sorted(TABLE_FILES + ["reference_stain_stack.ome.tif"])
    check_export(f.export_dir)
