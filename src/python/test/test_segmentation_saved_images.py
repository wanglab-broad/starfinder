"""Segmenting from saved reference-frame images in a separate process (W-338).

docs/segmentation-contract.md ("Coordination per FOV"), docs/assignment-contract.md ("The grid
rule") and docs/checkpoints.md. The dataset is the W-337 dataset of test_morphology_rounds,
written in session (8×32×32 uint16): the reference round1 (ch00–ch03 and the stain file ch04,
DAPI), the sequencing round2 and the other round morph (Flamingo, RBD, DAPI), here with a
calibrated ImageMetadata (spacing 1.0, 0.5, 0.5 µm), which the seeded watershed needs.
FOV.run and FOV.prepare_morphology rotate by 90 degrees.

The three-process test runs each call of a separate process with this module's ``main``,
in a child Python with the parent's environment and PYTHONPATH set to this tree's
``src/python``. Process 3 takes its molecules from the ``candidates`` and ``pre_qc``
checkpoints of process 1 (no image is restored) and repeats the final filtering.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml
from skimage.measure import label

from starfinder.assignment import AssignmentConfig, assign_molecules, molecule_table
from starfinder.assignment import _assign
from starfinder.barcode import ReadFilterConfig
from starfinder.dataset import CheckpointConfig, MorphologyConfig, PipelineConfig
from starfinder.dataset.workflow import _nuclei_checkpoints, _nuclei_registration
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.io import save_volume
from starfinder.preprocessing import ProjectionConfig
from starfinder.segmentation import (SEGMENTATION_METHODS, CompositeConfig, FlamingoEnhancementConfig, InputChannel,
                                     ReferenceGrid, SeededWatershedConfig, SegmentationPlan, SegmentationResult,
                                     SegmentationRun, reference_grid_from_file)
from starfinder.segmentation._labels import array_sha256, same_grid
from starfinder.segmentation._plan import assemble_input

from .test_morphology_rounds import (LABELS, MORPH, SEQUENCING_SHIFT, STAIN, THRESHOLD, ThresholdConfig, files,
                                     make_dataset, masked, pipeline)
from .test_registration_other_rounds import SHIFT, grid, texture

pytestmark = [pytest.mark.segmentation, pytest.mark.dataset]

SRC = Path(__file__).resolve().parents[1]           # src/python of this tree
ROOT = SRC.parents[1]
ANGLE = 90
SPACING = dict(spacing_zyx=(1.0, 0.5, 0.5), spatial_unit="micrometer")
SCHEMA = yaml.safe_load((ROOT / "workflow" / "schemas" / "config.schema.yaml").read_text())


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    """SimpleITK at one thread, as the project contract requires."""
    sitk = pytest.importorskip("SimpleITK")
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


def write_inputs(root):
    """The W-337 files under root/in, with calibrated metadata; returns {"root": root}, as make_dataset takes it."""
    p = grid()
    round1 = np.stack([texture(s, p) for s in (1, 2, 3, 6)], axis=-1)
    round2 = np.ascontiguousarray(np.roll(round1, SEQUENCING_SHIFT, axis=(0, 1, 2)))
    morph = np.stack([texture(s, p - SHIFT.reshape(3, 1, 1, 1)) for s in (4, 5, 0)], axis=-1)
    written = {"round1": [*zip(LABELS, np.moveaxis(round1, -1, 0)), ("ch04", texture(0, p))],
               "round2": list(zip(LABELS, np.moveaxis(round2, -1, 0))),
               "morph": list(zip((c.channel for c in MORPH), np.moveaxis(morph, -1, 0)))}
    for name, channels in written.items():
        for channel, image in channels:
            save_volume(np.ascontiguousarray(image), Path(root) / "in" / name / "FOV" / f"FOV_{channel}.tif",
                        metadata=ImageMetadata(f"FOV/{name}", **SPACING))
    return {"root": Path(root)}


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    return write_inputs(tmp_path_factory.mktemp("w338"))


def dataset(root, output):
    return make_dataset({"root": Path(root)}, output=output)


# --- The calls of the three processes, also made in one process -------------------------------------------

def plan(fov):
    """A nucleus run on the reference stain, a seeded-watershed cell run on the composite of that stain and the
    reference merged image, and a nucleus run on the Flamingo-enhanced DAPI of morph.

    The thresholds are the 90th percentiles of the stain and of the enhanced DAPI, so every process computes the
    same levels from the same images.
    """
    stain = fov.images["reference_stain"][..., 0]
    enhanced = assemble_input(fov, morph_run(0.0), fov.reference_grid())[0].image[..., 0]
    return SegmentationPlan((
        SegmentationRun("nucleus", "nucleus", (InputChannel("nuclear", round="reference_stain", channel="DAPI"),),
                        ThresholdConfig(float(np.percentile(stain, 90)))),
        SegmentationRun("cell", "cell", (InputChannel("composite", round="reference_stain", channel="DAPI",
                                                      prepare=CompositeConfig()),),
                        SeededWatershedConfig(), seeds="nucleus"),
        morph_run(float(np.percentile(enhanced, 90))),
    ))


def morph_run(level):
    return SegmentationRun("morph_nucleus", "nucleus", (InputChannel(
        "nuclear", round="morph", channel="DAPI", prepare=FlamingoEnhancementConfig(), prepare_channel="Flamingo"),),
        ThresholdConfig(level))


def process_run(root, output):
    """Process 1: FOV.run with checkpoints and save_reference_image.

    overwrite=True lets it run after process 2, which made the FOV directory: run's overwrite removes checkpoint
    stage files only, so other_rounds/ stays (docs/checkpoints.md).
    """
    fov = dataset(root, output).fov("FOV").run(pipeline(rotation=ANGLE), checkpoints=CheckpointConfig(overwrite=True))
    fov.save_reference_image()
    return fov


def process_prepare(root, output):
    """Process 2: FOV.prepare_morphology with checkpoints (the raw files only)."""
    return dataset(root, output).fov("FOV").prepare_morphology(MorphologyConfig(rotation_degrees=ANGLE),
                                                                checkpoints=CheckpointConfig())


def process_segment(root, output):
    """Process 3: load_reference_image, load_registered_round, segment and assign, all with checkpoints."""
    fov = dataset(root, output).fov("FOV").load_reference_image()
    for name in ("reference_stain", "morph"):
        fov.load_registered_round(name)
    resident = {"images": sorted(fov.images), "metadata": sorted(fov.metadata)}
    fov.segment(plan(fov), checkpoints=CheckpointConfig())
    # The molecules of process 1: two saved stages that restore no image, then the final filtering again.
    fov.load_checkpoint("candidates").load_checkpoint("pre_qc")
    fov.run(PipelineConfig(filtering=ReadFilterConfig()))
    fov.assign(AssignmentConfig(), nuclei="nucleus", checkpoints=CheckpointConfig())
    return fov, dict(resident, after={"images": sorted(fov.images), "metadata": sorted(fov.metadata)})


def main(argv):
    """The child process: main(<call> <input root> <output name>); prints a JSON line for process 3."""
    call, root, output = argv
    if call == "run":
        process_run(root, output)
    elif call == "prepare":
        process_prepare(root, output)
    else:
        # Test-only: the threshold method of test_morphology_rounds, registered here because monkeypatch
        # does not reach a child process. It is not a method of the package.
        SEGMENTATION_METHODS[ThresholdConfig] = THRESHOLD
        _, resident = process_segment(root, output)
        print(json.dumps(resident))


def child(call, root, output):
    """Run one call in a separate Python process: the parent's environment, PYTHONPATH this tree's src/python."""
    env = dict(os.environ, PYTHONPATH=str(SRC))
    code = "import sys; from test.test_segmentation_saved_images import main; main(sys.argv[1:])"
    done = subprocess.run([sys.executable, "-c", code, call, str(root), output], env=env, cwd=SRC,
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    return done.stdout


@pytest.fixture
def threshold(monkeypatch):
    monkeypatch.setitem(SEGMENTATION_METHODS, ThresholdConfig, THRESHOLD)


def one_process(root, output):
    """The same calls in one process and on one FOV object, without checkpoints."""
    fov = dataset(root, output).fov("FOV").run(pipeline(rotation=ANGLE))
    fov.save_reference_image()
    fov.prepare_morphology(MorphologyConfig(rotation_degrees=ANGLE))
    fov.segment(plan(fov))
    fov.assign(AssignmentConfig(), nuclei="nucleus")
    return fov


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def comparable(path, output_root):
    """A written file's content for comparing two runs of the same calls into two output roots.

    OME UUIDs are masked; run.json is compared without its times (started_at, ended_at and the seconds
    of each step) and with the output root written as <out>. tifffile writes a random UUID into every
    OME-TIFF, so the file SHA-256 that a saved form's registration.json records differs between any two
    writes; it is checked against its own file instead.
    """
    if path.name == "run.json":
        data = json.loads(path.read_text().replace(str(output_root), "<out>"))
        data["steps"] = [{k: v for k, v in step.items() if k != "seconds"} for step in data["steps"]]
        return json.dumps({k: v for k, v in data.items() if k not in ("started_at", "ended_at")}, sort_keys=True)
    if path.name == "registration.json":
        data = json.loads(path.read_text())
        assert data["image"].pop("file_sha256") == file_sha256(path.parent / data["image"]["path"])
        return json.dumps(data, sort_keys=True)
    return masked(path)


def without_links(channel):
    """An input channel entry without the links that only a reload adds (the saved files)."""
    entry = {k: v for k, v in channel.items() if k != "reference_image"}
    if entry.get("registration"):
        entry["registration"] = {k: v for k, v in entry["registration"].items() if k != "saved"}
    return entry


# --- The reference image ---------------------------------------------------------------------------------

@pytest.mark.parametrize("projected", [False, True])
def test_load_reference_image_restores_the_grid_and_the_merged_image(raw, threshold, tmp_path, projected):
    ds = dataset(raw["root"], f"ref{int(projected)}")
    ran = ds.fov("FOV").run(pipeline(rotation=ANGLE))
    path = ran.save_reference_image(projection=ProjectionConfig() if projected else None)
    expected = ran.reference_grid().projected() if projected else ran.reference_grid()

    fresh = ds.fov("FOV").load_reference_image()
    assert fresh.images == {} and fresh.metadata == {}
    restored = fresh.reference_grid()
    assert restored.shape_zyx == expected.shape_zyx == ((1, 32, 32) if projected else (8, 32, 32))
    assert restored.metadata == expected.metadata and same_grid(restored, expected)
    from_file = reference_grid_from_file(path)
    assert restored == from_file and restored.source == f"file:{path.resolve()}"
    saved = tifffile.imread(path)
    assert restored.sha256 == hashlib.sha256(np.ascontiguousarray(saved).tobytes()).hexdigest()

    # InputChannel(reference_merged=True) reads the file, and the record links it.
    run = SegmentationRun("merged", "nucleus", (InputChannel("nuclear", reference_merged=True),),
                          ThresholdConfig(float(np.mean(saved))))
    segmentation_input, _ = assemble_input(fresh, run, restored)
    assert np.array_equal(segmentation_input.image[..., 0], saved.reshape(restored.shape_zyx))
    assert segmentation_input.image.dtype == saved.dtype == np.uint16
    fresh.segment(SegmentationPlan((run,)))
    result = fresh.segmentation_results["merged"]
    (entry,) = result.record["input"]["channels"]
    assert entry["sha256"] == array_sha256(saved.reshape(restored.shape_zyx))
    assert entry["reference_image"] == {"path": "images/ref_merged/FOV.tif", "sha256": file_sha256(path)}
    assert result.grid == restored and result.record["grid"]["source"] == restored.source
    if not projected:
        # In the process that ran FOV.run the same run reads the channel maximum and gives the same labels.
        ran.segment(SegmentationPlan((run,)))
        assert np.array_equal(ran.segmentation_results["merged"].labels, result.labels)
        assert "reference_image" not in ran.segmentation_results["merged"].record["input"]["channels"][0]


def test_a_missing_or_disagreeing_reference_image_raises(raw, tmp_path):
    ds = dataset(raw["root"], "disagree")
    with pytest.raises(FileNotFoundError):
        ds.fov("FOV").load_reference_image()
    ran = ds.fov("FOV").run(pipeline(rotation=ANGLE))
    path = ran.save_reference_image()
    # A file that agrees with the resident reference round is accepted, and the round is read.
    ran.load_reference_image()
    assert ran.reference_grid().source == "fov:round1"
    merged = ran.images["round1"].max(axis=3)
    for changed, metadata, message in ((merged + 1, ran.metadata["round1"], "with other values"),
                                       (merged, ImageMetadata("FOV/other", **SPACING), "FOV/other"),
                                       (merged[:4], ran.metadata["round1"], r"\(4, 32, 32\)")):
        save_volume(changed, path, metadata=metadata)
        with pytest.raises(ValueError, match="is not the channel maximum of the resident reference round") as error:
            ds.fov("FOV").run(pipeline(rotation=ANGLE)).load_reference_image()
        assert re.search(message, str(error.value))
        # A reference round resident after the load is checked by reference_grid.
        restored = ds.fov("FOV").load_reference_image()
        restored.images["round1"], restored.metadata["round1"] = ran.images["round1"], ran.metadata["round1"]
        with pytest.raises(ValueError, match="is not the channel maximum"):
            restored.reference_grid()
    # A projected file agrees with the round's projection.
    ran.save_reference_image(projection=ProjectionConfig())
    assert ran.load_reference_image().reference_grid().source == "fov:round1"
    # Neither the round nor the file: the error names both routes.
    with pytest.raises(ValueError, match="load_reference_image"):
        ds.fov("FOV").reference_grid()


def test_a_channel_of_the_reference_round_still_needs_the_round(raw, threshold):
    ds = dataset(raw["root"], "channel")
    ds.fov("FOV").run(pipeline(rotation=ANGLE)).save_reference_image()
    fresh = ds.fov("FOV").load_reference_image()
    for channel in (InputChannel("nuclear", round="round1", channel="ch00"),
                    InputChannel("nuclear", round="round1", channel="ch00", prepare=CompositeConfig())):
        run = SegmentationRun("nucleus", "nucleus", (channel,), ThresholdConfig(0.0))
        with pytest.raises(ValueError, match="needs the reference round resident"):
            fresh.segment(SegmentationPlan((run,)))
    assert fresh.segmentation_results == {}
    # Without the file either, the merged image asks for one of the routes.
    run = SegmentationRun("nucleus", "nucleus", (InputChannel("nuclear", reference_merged=True),), ThresholdConfig(0.0))
    with pytest.raises(ValueError, match="call run\\(\\), load_checkpoint"):
        ds.fov("FOV").segment(SegmentationPlan((run,)))


def test_assign_refuses_the_grid_of_a_projected_reference_image(raw, threshold):
    ds = dataset(raw["root"], "projected_assign")
    ran = ds.fov("FOV").run(pipeline(rotation=ANGLE))
    ran.save_reference_image(projection=ProjectionConfig())
    fresh = ds.fov("FOV").load_reference_image()
    fresh.segment(SegmentationPlan((SegmentationRun("cell", "nucleus", (InputChannel("nuclear", reference_merged=True),),
                                                    ThresholdConfig(1000.0)),)))
    fresh.spot_result, fresh.filtering_result = ran.spot_result, ran.filtering_result
    with pytest.raises(ValueError, match="is a Z projection"):
        fresh.assign()


# --- Three processes -------------------------------------------------------------------------------------

@pytest.mark.slow
def test_three_processes_segment_and_assign_as_one_process(raw, threshold, tmp_path):
    root = raw["root"]
    child("run", root, "three")
    child("prepare", root, "three")
    resident = json.loads(child("segment", root, "three").strip().splitlines()[-1])
    # No sequencing round, nor any image of the reference round, is resident in process 3.
    assert resident == {"images": ["morph", "reference_stain"], "metadata": ["morph", "reference_stain"],
                        "after": {"images": ["morph", "reference_stain"], "metadata": ["morph", "reference_stain"]}}

    one = one_process(root, "one")
    ds = dataset(root, "three")
    checkpoint = ds.output_root / "checkpoints" / "FOV"
    loaded = ds.fov("FOV")
    for name in ("nucleus", "cell", "morph_nucleus"):
        result, expected = loaded.load_segmentation(name), one.segmentation_results[name]
        assert result.labels.max() > 0 and np.array_equal(result.labels, expected.labels), name
        assert same_grid(result.grid, expected.grid) and result.grid.source.startswith("file:")
        record, other = result.record, expected.record
        # The records agree except where they say which grid and which saved files were read.
        differs = {"grid", "upstream", "input", "labels", "software"}
        assert {k: v for k, v in record.items() if k not in differs} == \
            json.loads(json.dumps({k: v for k, v in other.items() if k not in differs}))
        assert record["input"]["sha256"] == other["input"]["sha256"]
        assert [without_links(c) for c in record["input"]["channels"]] == \
            json.loads(json.dumps([without_links(c) for c in other["input"]["channels"]]))
        assert record["labels"]["sha256"] == other["labels"]["sha256"]
        # Every link resolves with its SHA-256.
        folder = checkpoint / "segmentation" / name
        assert file_sha256(folder / record["input"]["path"]) == record["input"]["file_sha256"]
        assert file_sha256(folder / record["labels"]["path"]) == record["labels"]["file_sha256"]
        for channel in record["input"]["channels"]:
            saved = channel["registration"]["saved"]
            assert file_sha256(checkpoint / saved["path"]) == saved["sha256"]
            if "reference_image" in channel:
                link = channel["reference_image"]
                assert file_sha256(ds.output_root / link["path"]) == link["sha256"]
    assert loaded.segmentation_results["cell"].record["input"]["channels"][0]["reference_image"]["path"] == \
        "images/ref_merged/FOV.tif"
    composite = loaded.segmentation_results["cell"].record["input"]["channels"][0]["prepare"]
    assert composite == json.loads(json.dumps(one.segmentation_results["cell"].record["input"]["channels"][0]["prepare"]))

    assignment = loaded.load_assignment("default")
    expected = one.assignment_results["default"]
    for table in ("molecules", "cells", "counts", "nuclei"):
        pd.testing.assert_frame_equal(getattr(assignment, table), getattr(expected, table), check_exact=True)
    assert (assignment.molecules.assignment_status == "assigned").sum() > 0
    assert assignment.record["counts"] == json.loads(json.dumps(expected.record["counts"]))
    assert assignment.record["grid"]["check"] == "checked" and assignment.record["grid"]["source"].startswith("file:")
    folder = checkpoint / "assignment" / "default"
    for entry in [*assignment.record["files"].values(), assignment.record["inputs"]["cells"]["file"],
                  assignment.record["inputs"]["nuclei"]["file"]]:
        assert file_sha256(os.path.normpath(folder / entry["path"])) == entry["sha256"]

    # Processes 1 and 2 give the same files in either order.
    child("prepare", root, "reversed")
    child("run", root, "reversed")
    first, second = ds.output_root, dataset(root, "reversed").output_root
    assert [p for p in files(first) if not re.search("/(segmentation|assignment)/", p)] == files(second)
    for name in files(second):
        assert comparable(first / name, first) == comparable(second / name, second), name


# --- The grid rule ---------------------------------------------------------------------------------------

@pytest.fixture
def no_sampling(monkeypatch):
    calls = []
    original = _assign.sample_labels

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(_assign, "sample_labels", counting)
    return calls


def test_labels_on_the_file_grid_and_molecules_of_run_are_one_grid(raw, threshold, no_sampling):
    one = one_process(raw["root"], "grid_rule")
    fresh = dataset(raw["root"], "grid_rule").fov("FOV").load_reference_image()
    fresh.prepare_morphology(MorphologyConfig(rotation_degrees=ANGLE))
    fresh.segment(plan(fresh))
    cells, nuclei = fresh.segmentation_results["cell"], fresh.segmentation_results["nucleus"]
    molecules = molecule_table(one.spot_result, one.filtering_result, genes=one.codebook.genes)
    run_grid = one.reference_grid()
    assert cells.grid.source != run_grid.source and cells.grid.sha256 != run_grid.sha256
    result = assign_molecules(molecules, cells, grid=run_grid, nuclei=nuclei)
    expected = one.assignment_results["default"]
    assert result.molecules.assignment_status.tolist() == expected.molecules.assignment_status.tolist()
    pd.testing.assert_frame_equal(result.counts, expected.counts, check_exact=True)
    pd.testing.assert_frame_equal(result.cells, expected.cells, check_exact=True)
    assert result.record["grid"]["check"] == "checked" and result.record["grid"]["source"] == "fov:round1"

    # Another shape or other metadata raises before sampling.
    no_sampling.clear()
    shape = (8, 32, 31)
    other_shape = SegmentationResult(np.ascontiguousarray(cells.labels[..., :31]),
                                     ReferenceGrid(shape, cells.grid.metadata, cells.grid.source),
                                     "cell", "volume", cells.label_namespace, cells.record)
    other_metadata = SegmentationResult(cells.labels, ReferenceGrid(cells.grid.shape_zyx, ImageMetadata(
        "FOV/other", **SPACING), cells.grid.source), "cell", "volume", cells.label_namespace, cells.record)
    for labels in (other_shape, other_metadata):
        with pytest.raises(IncompatibleGeometryError, match="are not on the molecule grid"):
            assign_molecules(molecules, labels, grid=run_grid)
    assert no_sampling == []


# --- The nuclei_registration rule ------------------------------------------------------------------------

def rule_config(tmp_path, angle, projection, checkpoints=None):
    """The shared keys of a real dataset: sequencing ch00–ch03, the stain ch04 of dapi_round = ref_round."""
    config = dict(starfinder_path=str(ROOT), root_input_path=str(tmp_path / "in_rule"), dataset_id="dataset",
                  sample_id="sample", root_output_path=str(tmp_path / "out_rule"), output_id="output",
                  fov_id_pattern="FOV%d", n_rounds=2, ref_round="round1", dapi_round="round1",
                  seq_channel_order=list(LABELS), ref_channel="DAPI", maximum_projection=projection, backend="python",
                  additional_round=[dict(round_name="morph", channel_order=[c.record() for c in MORPH])],
                  rules=dict(nuclei_registration=dict(run=True)))
    if angle is not None:
        config["rotate_angle"] = angle
    if checkpoints is not None:
        config["rules"]["nuclei_registration"]["parameters"] = dict(checkpoints=checkpoints)
    return config


def run_rule(config, raw):
    source = Path(config["root_input_path"]) / "dataset" / "sample"
    if not source.exists():
        source.parent.mkdir(parents=True)
        os.symlink(raw["root"] / "in", source)
    snakemake = SimpleNamespace(config=config, wildcards=SimpleNamespace(fovID="FOV"), input=[], output=[])
    runpy.run_path(str(ROOT / "workflow" / "scripts" / "nuclei_registration.py"), init_globals={"snakemake": snakemake})
    return Path(config["root_output_path"]) / "dataset" / "output"


DECLARED = ["images/morph/DAPI/FOV.tif", "images/morph/Flamingo/FOV.tif", "images/morph/RBD/FOV.tif",
            "log/FOV_nr.txt", "log/gr_shifts/FOV_nr.txt"]


@pytest.mark.parametrize("projection", [False, True])
@pytest.mark.parametrize("angle", [None, 90])
def test_the_rule_writes_the_saved_form_only_with_checkpoints(raw, tmp_path, angle, projection):
    plain = run_rule(rule_config(tmp_path / "plain", angle, projection), raw)
    saved = run_rule(rule_config(tmp_path / "saved", angle, projection, checkpoints={}), raw)
    assert files(plain) == DECLARED
    assert files(saved) == sorted(DECLARED + [f"checkpoints/FOV/other_rounds/{name}/{file}"
                                              for name in ("morph", "reference_stain")
                                              for file in ("image.ome.tif", "registration.json")])
    for name in DECLARED:
        assert (plain / name).read_bytes() == (saved / name).read_bytes(), name
    log = json.loads((plain / "log" / "FOV_nr.txt").read_text())
    assert list(log["registration_attempts"]) == ["morph"]
    assert log["registration_attempts"]["morph"][0]["reference"] == "round1:ch04"
    shifts = pd.read_csv(plain / "log" / "gr_shifts" / "FOV_nr.txt")
    rotated = {None: (3, -2), 90: (2, 3)}[angle]       # the displacement (0, 3, -2) rotated by 90 degrees CCW
    assert shifts.to_dict("records") == [dict(fov_id="FOV", round="morph", row=rotated[0], col=rotated[1], z=0)]

    # The saved form reloads, and its image is the one the rule wrote; without checkpoints there is nothing.
    rule_dataset = _nuclei_registration(rule_config(tmp_path / "saved", angle, projection))[0]
    fov = rule_dataset.fov("FOV")
    for name in ("reference_stain", "morph"):
        fov.load_registered_round(name)
    assert fov.registration_chains["morph"].transforms[0].displacement_zyx == (0.0, *map(float, rotated))
    for c, channel in enumerate(MORPH):
        written = tifffile.imread(saved / "images" / "morph" / channel.name / "FOV.tif")
        image = fov.images["morph"][..., c]
        assert np.array_equal(written, image.max(axis=0) if projection else image)
    with pytest.raises(FileNotFoundError):
        _nuclei_registration(rule_config(tmp_path / "plain", angle, projection))[0].fov("FOV").load_registered_round("morph")


def test_the_rule_checkpoints_key(tmp_path):
    for value in ({}, {"directory": "/somewhere", "overwrite": True}, None):
        config = yaml.safe_load((ROOT / "docs/examples/workflow-full.yaml").read_text())
        config["rules"]["nuclei_registration"]["parameters"] = dict(checkpoints=value)
        jsonschema.validate(config, SCHEMA)
    config["rules"]["nuclei_registration"]["parameters"] = dict(checkpoints=[1])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(config, SCHEMA)
    assert _nuclei_checkpoints(rule_config(tmp_path, None, False)) is None
    assert _nuclei_checkpoints(rule_config(tmp_path, None, False, checkpoints=None)) is None
    assert _nuclei_checkpoints(rule_config(tmp_path, None, False, checkpoints={})) == CheckpointConfig()
    with pytest.raises(ValueError, match="Python-only and need backend: python"):
        _nuclei_checkpoints(dict(rule_config(tmp_path, None, False, checkpoints={}), backend="matlab"))
    with pytest.raises(ValueError, match="unknown rules.nuclei_registration.parameters.checkpoints keys"):
        _nuclei_checkpoints(rule_config(tmp_path, None, False, checkpoints={"stage": 1}))
