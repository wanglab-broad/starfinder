"""The morphology, segment and assign commands (moved by W-750 from the tests of the Python nuclei_registration rule's
adapter, the Python branch of stardist_segmentation and the reads_assignment rule's route of an assembly block:
test_registration_other_rounds.py, test_segmentation_saved_images.py, test_dataset_fov_layout.py,
test_segmentation_workflow.py and test_stitching_workflow.py).

Each test writes the raw images the old test wrote, plans a configuration in the Python shape on them and runs the
tasks by their commands. What the contract changed is named where a test asserts it: the morphology task always
writes the saved form of each prepared image and writes the channel images with ``morphology.channel_images``
(D6; design/specs/workflows.md, "Nuclei registration"), the segment task's saved run replaces the legacy label image
(D6), and the tasks' records and logs are under ``workflow/`` (D8).
"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile

from starfinder import _layout
from starfinder.image import ImageMetadata
from starfinder.io import save_volume

from . import workflow_examples as X
from . import workflow_tasks as T

pytestmark = [pytest.mark.workflow, pytest.mark.segmentation, pytest.mark.contract]


def files(directory, *, leave=("workflow/",)):
    return sorted(n for n in (p.relative_to(directory).as_posix() for p in Path(directory).rglob("*") if p.is_file())
                  if not n.startswith(leave))


def saved_forms(fov, names):
    return [f"fovs/{fov}/other_rounds/{name}/{file}" for name in names for file in ("image.ome.tif",
                                                                                     "registration.json")]


def morphology_config(work, *, sequencing, stain, morph, angle=0, projection=False, output_id="out"):
    """The morphology block (with its channel images, ZYX or the Z maximum) on two sequencing rounds of the labels
    ``sequencing``, the reference stain ``stain`` (named DAPI) and the other round ``morph``."""
    config = T.dataset_keys(work, n_rounds=2, output_id=output_id, seq_channel_order=list(sequencing),
                            reference_stains=[{"channel": stain, "name": "DAPI"}], ref_channel="DAPI",
                            rotate_angle=angle, additional_round=[{"round_name": "morph", "channel_order": morph}],
                            morphology={"channel_images": {"projection": "max" if projection else None}})
    return config


def run_morphology(work, config, fov="FOV"):
    root = T.input_root(config)
    _layout.fovs_file(root).write_text(f"sample,fov_id\nsample,{fov}\n")
    manifest, _ = T.plan(work, "morphology", config)
    T.run_kinds(manifest, "morphology")
    return manifest, T.output_root(config)


# --- the morphology command (from test_registration_other_rounds.py) ---------------------------------------------

@pytest.mark.parametrize("maximum_projection", [False, True])
def test_the_morphology_command_writes_the_matlab_rules_images_under_their_labels(tmp_path, maximum_projection):
    """From test_the_python_workflow_rule_mirrors_the_matlab_outputs: the inputs written unrotated, the task rotates
    every image by rotate_angle 90; it writes the registration record, each channel's image in the folder of its
    unique channel label (MATLAB names it by the descriptive name), the Z maximum with projection max, and the
    saved form of each prepared image (always, D6)."""
    from .test_registration_other_rounds import OTHER, OVERLAP, SEQUENCING, SHAPE, rounds
    reference, sequencing, morph, truth, _ = rounds()
    config = morphology_config(tmp_path, sequencing=SEQUENCING[:3], stain=SEQUENCING[3], angle=90,
                               projection=maximum_projection, morph=[
                                   dict(wavelength=405, channel="ch00", name="DAPI"),
                                   dict(wavelength=488, channel="ch01", name="Nissl"),
                                   dict(wavelength=647, channel="ch02", name="GFAP")])
    source = T.input_root(config)
    for name, image, labels in (("round1", reference, SEQUENCING), ("round2", sequencing, SEQUENCING),
                                ("morph", morph, OTHER)):
        for c, label in enumerate(labels):
            save_volume(np.rot90(image[..., c], k=-1, axes=(1, 2)), source / name / "FOV" / f"FOV_{label}.tif")
    _, output = run_morphology(tmp_path, config)
    assert files(output) == sorted(["fovs/FOV/nuclei_registration.json", "images/morph/ch00/FOV.tif",
                                    "images/morph/ch01/FOV.tif", "images/morph/ch02/FOV.tif"]
                                   + saved_forms("FOV", ("morph", "reference_stain")))
    record = json.loads((output / "fovs" / "FOV" / "nuclei_registration.json").read_text())
    assert list(record["registration"]) == ["morph"]
    assert record["registration"]["morph"][0]["qc"]["details"]["transform"]["displacement_zyx"] == [0, 3, -2]
    for c, folder in enumerate(("ch00", "ch01", "ch02")):
        saved = tifffile.imread(output / "images" / "morph" / folder / "FOV.tif")
        assert saved.dtype == np.uint16
        if maximum_projection:
            assert saved.shape == SHAPE[1:]
        else:
            assert np.array_equal(saved[OVERLAP], truth[OVERLAP + (c,)])


def test_the_declared_reference_stain_drives_the_morphology_command(tmp_path):
    """From test_the_declared_reference_stain_drives_the_python_nuclei_rule: the reference round's stain is the file
    reference_stains declares, here ch09: the morphology task registers the other round against it (reference
    round1:ch09); no ch04 is assumed. (rotate_nuclei, whose input was that file, is retired for Python, D6.)"""
    from .test_registration_other_rounds import OTHER, SEQUENCING, rounds
    reference, sequencing, morph, _, _ = rounds()
    labels = (*SEQUENCING[:3], "ch09")
    config = morphology_config(tmp_path, sequencing=labels[:3], stain="ch09", morph=[
        dict(channel="ch00", name="DAPI"), dict(channel="ch01", name="Nissl"), dict(channel="ch02", name="GFAP")])
    source = T.input_root(config)
    for name, image, names in (("round1", reference, labels), ("round2", sequencing, SEQUENCING),
                               ("morph", morph, OTHER)):
        for c, label in enumerate(names):
            save_volume(image[..., c], source / name / "FOV" / f"FOV_{label}.tif")
    manifest, output = run_morphology(tmp_path, config)
    record = json.loads((output / "fovs" / "FOV" / "nuclei_registration.json").read_text())
    attempt = record["registration"]["morph"][0]
    assert attempt["reference"] == "round1:ch09"
    assert attempt["qc"]["details"]["transform"]["displacement_zyx"] == [0, 3, -2]
    task = T.task(manifest, "morphology", fov="FOV")
    assert "input:round1/FOV/FOV_ch09.tif" in task["inputs"]
    assert not [i for i in task["inputs"] if i.endswith("ch04.tif")]


# --- the saved form and the channel images (from test_segmentation_saved_images.py) -------------------------------

@pytest.mark.parametrize("projection", [False, True])
@pytest.mark.parametrize("angle", [None, 90])
def test_the_morphology_command_writes_the_saved_form_and_the_channel_images(tmp_path, angle, projection):
    """From test_the_rule_writes_the_saved_form_only_with_checkpoints: the morphology task writes the registration
    record, the channel images and the saved form of each prepared image (always, D6, where the rule wrote it only
    with its checkpoints key); the record registers morph on the reference stain round1:ch04 by the displacement
    rotated with the images; the saved form reloads, and each channel image is the prepared image (its Z maximum
    with projection max)."""
    from .test_morphology_rounds import LABELS, MORPH
    from .test_segmentation_saved_images import write_inputs
    from starfinder.dataset.workflow import _nuclei_registration
    write_inputs(tmp_path / "raw")
    config = morphology_config(tmp_path, sequencing=LABELS, stain="ch04", angle=angle or 0, projection=projection,
                               morph=[c.record() for c in MORPH])
    source = T.input_root(config)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.symlink_to(tmp_path / "raw" / "in")
    _, output = run_morphology(tmp_path, config)
    declared = ["fovs/FOV/nuclei_registration.json"] + [f"images/morph/{c.channel}/FOV.tif" for c in MORPH]
    assert files(output) == sorted(declared + saved_forms("FOV", ("morph", "reference_stain")))
    record = json.loads((output / "fovs" / "FOV" / "nuclei_registration.json").read_text())
    assert list(record["registration"]) == list(record["registered_rounds"]) == ["morph"]
    assert record["registration"]["morph"][0]["reference"] == "round1:ch04"
    rotated = {None: (3, -2), 90: (2, 3)}[angle]       # the displacement (0, 3, -2) rotated by 90 degrees CCW
    assert record["registration"]["morph"][0]["qc"]["details"]["transform"]["displacement_zyx"] == [0, *rotated]
    # The saved form reloads (the translator's dataset of the same configuration), and its images are the ones
    # the task wrote.
    old = dict(config, root_input_path=str(tmp_path / "input"), root_output_path=str(tmp_path / "output"),
               output_id="out", backend="python", rules={"nuclei_registration": {"run": True}},
               maximum_projection=projection)
    old.pop("morphology")
    if angle is None:
        old.pop("rotate_angle")
    fov = _nuclei_registration(old)[0].fov("FOV")
    for name in ("reference_stain", "morph"):
        fov.load_registered_round(name)
    assert fov.registration_chains["morph"].transforms[0].displacement_zyx == (0.0, *map(float, rotated))
    for c, channel in enumerate(MORPH):
        written = tifffile.imread(output / "images" / "morph" / channel.channel / "FOV.tif")
        image = fov.images["morph"][..., c]
        assert np.array_equal(written, image.max(axis=0) if projection else image)


# --- the channel images by label (from test_dataset_fov_layout.py) ------------------------------------------------

def test_channels_that_share_a_descriptive_name_write_one_file_each_under_their_labels(tmp_path):
    """From the test of the same name: two channels named Nissl in one round write images/morph/ch01/ and
    images/morph/ch02/, each its own channel; MATLAB's name folders would have written one file for both."""
    from .test_registration_other_rounds import OTHER, SEQUENCING, rounds
    reference, sequencing, morph, _, _ = rounds()
    config = morphology_config(tmp_path, sequencing=SEQUENCING[:3], stain=SEQUENCING[3], morph=[
        dict(wavelength=405, channel="ch00", name="DAPI"), dict(wavelength=488, channel="ch01", name="Nissl"),
        dict(wavelength=647, channel="ch02", name="Nissl")])
    source = T.input_root(config)
    for name, image, labels in (("round1", reference, SEQUENCING), ("round2", sequencing, SEQUENCING),
                                ("morph", morph, OTHER)):
        for c, label in enumerate(labels):
            save_volume(image[..., c], source / name / "FOV" / f"FOV_{label}.tif")
    _, output = run_morphology(tmp_path, config)
    assert [f for f in files(output) if f.startswith("images/")] == [
        "images/morph/ch00/FOV.tif", "images/morph/ch01/FOV.tif", "images/morph/ch02/FOV.tif"]
    first, second = (tifffile.imread(output / "images" / "morph" / label / "FOV.tif") for label in ("ch01", "ch02"))
    assert not np.array_equal(first, second)


# --- the segment command's StarDist run (from test_segmentation_workflow.py) -------------------------------------

def stardist_route(work, image, *, operations=True):
    """The legacy StarDist run of test_segmentation_workflow.py (the stand-in model folder, rescale false, the
    expansion by 4 pixels, planar) as a segment run on the reference stain's saved form of a FOV whose reference
    stain is ``image`` (16 × 64 × 64, spacing 0.35 × 0.1 × 0.1 µm); the morphology and segment tasks run."""
    from .test_segmentation_workflow import model_folder
    from .test_segmentation_golden import DISTANCE
    model_folder(work / "models" / "stand_in")
    metadata = ImageMetadata("FOV", spacing_zyx=(0.35, 0.1, 0.1), spatial_unit="micrometer")
    run = {"name": "nucleus", "target": "nucleus", "method": "stardist", "model_path": str(work / "models" /
                                                                                         "stand_in"),
           "scale": 1.0, "inputs": [{"role": "nuclear", "round": "reference_stain", "channel": "ch04"}]}
    if operations:
        run["operations"] = [{"operation": "expand_labels", "distance": DISTANCE, "unit": "pixel", "mode": "planar"}]
    config = T.dataset_keys(work, n_rounds=1, seq_channel_order=["ch00"], ref_channel="DAPI",
                            reference_stains=[{"channel": "ch04", "name": "DAPI"}],
                            additional_round=[{"round_name": "morph", "channel_order": [
                                {"channel": "ch00", "name": "DAPI"}]}],
                            morphology={}, segmentation={"runs": [run]})
    source = T.input_root(config)
    for name, channels in (("round1", ("ch00", "ch04")), ("morph", ("ch00",))):
        for channel in channels:
            save_volume(image, source / name / "FOV" / f"FOV_{channel}.tif", metadata=metadata)
    _layout.fovs_file(source).write_text("sample,fov_id\nsample,FOV\n")
    manifest, _ = T.plan(work, "stardist", config)
    T.run_kinds(manifest, "morphology", "segment")
    folder = _layout.segmentation_dir(_layout.fov_dir(T.output_root(config), "FOV"), "nucleus")
    return tifffile.imread(folder / "labels.tif"), json.loads((folder / "segmentation.json").read_text())


@pytest.fixture
def stand_in_stardist(monkeypatch):
    """The stardist method with its run replaced by the stand-in, and no backend import (no TensorFlow)."""
    from starfinder.segmentation import SEGMENTATION_METHODS, StarDistConfig
    from .test_segmentation_workflow import stand_in_run
    spec = SEGMENTATION_METHODS[StarDistConfig]
    monkeypatch.setitem(SEGMENTATION_METHODS, StarDistConfig, replace(spec, run=stand_in_run, requires=()))


def test_the_segment_command_writes_uint32_labels_equal_to_the_legacy_pins(tmp_path, stand_in_stardist):
    """From test_the_rule_writes_uint32_labels_equal_to_the_legacy_pins[python]: the segment task's StarDist run of
    the legacy translation (rescale false, expand_labels true, distance 4) on the DAPI fixture gives
    LABELS_3D[(False, True)], uint32, with the nucleus target, the expansion and the stand-in's effective settings
    in its record."""
    from .coordination_helpers import digest
    from .test_segmentation_golden import LABELS_3D, fixture
    labels, record = stardist_route(tmp_path, fixture()["dapi"])
    assert labels.dtype == np.uint32 and labels.shape == (16, 64, 64)
    assert digest(labels.astype("uint16")) == LABELS_3D[(False, True)]
    assert record["labels"]["dtype"] == "uint32" and record["target"] == "nucleus"
    assert [op["operation"] for op in record["operations"]] == ["expand_labels"]
    assert record["grid"]["metadata"]["spacing_zyx"] == [0.35, 0.1, 0.1]
    assert record["methods"][0]["effective"] == {"stand_in": True}


def test_an_image_without_foreground_gives_an_empty_label_image(tmp_path, monkeypatch):
    """From the test of the same name (its Python half): no foreground gate: an all-zero image is outcome empty."""
    from starfinder.segmentation import SEGMENTATION_METHODS, StarDistConfig

    def empty_run(image, config, context):
        return np.zeros(image.shape[:3], np.int32), {}

    spec = SEGMENTATION_METHODS[StarDistConfig]
    monkeypatch.setitem(SEGMENTATION_METHODS, StarDistConfig, replace(spec, run=empty_run, requires=()))
    labels, record = stardist_route(tmp_path, np.zeros((16, 64, 64), np.uint8), operations=False)
    assert labels.dtype == np.uint32 and not labels.any() and record["outcome"] == "empty"


# --- the assign command saves the FOV's folder alone (from test_stitching_workflow.py) ---------------------------

@pytest.mark.parametrize("previews", [None, True])
def test_the_assign_command_writes_the_saved_folder_alone(tmp_path, previews):
    """From test_wf3_with_an_assembly_block_reads_assignment_writes_the_saved_folder_alone: no tile configuration is
    read, the FOV's assignment folder is saved on the grid of the FOV's reference image, and nothing else is written
    (no raw.h5ad, reads_assignment.csv or log.txt: the record holds the counts; the segment task's saved run, D6,
    and the tasks' records, D8, aside); with assignment.previews on, the two figures of the ZYX label image go to
    fovs/{fov}/previews/, which the task declares."""
    from starfinder.assignment._persist import read_assignment
    from starfinder.dataset._declared import declared_metadata
    from starfinder.segmentation import ReferenceGrid, reference_grid_from_file
    from starfinder.segmentation._labels import same_grid
    from .coordination_helpers import write_molecules_csv
    from .test_assignment_golden import SHAPE_ZYX, label_fixture
    from .test_assignment_workflow import CODEBOOK, KEEP, NAMESPACE
    work = tmp_path / "work"
    config = T.dataset_keys(work, voxel_size_xy=0.1, voxel_size_z=0.35, processing=dict(X.PROCESSING),
                            segmentation={"runs": [{"name": "cell", "target": "cell", "method": "import",
                                                    "path": str(work / "labels" / "{fov_id}.tif")}]},
                            assignment={"cells": "cell", **({} if previews is None else {"previews": previews})})
    T.write_inputs(config, "sample,fov_id\nsample1,Position001\n",
                   codebook="".join(f"{g},{b}\n" for g, b in CODEBOOK))
    (work / "labels").mkdir()
    tifffile.imwrite(work / "labels" / "Position001.tif", label_fixture())
    manifest, _ = T.plan(work, "assign", config)
    out = T.output_root(config)
    write_molecules_csv(_layout.spot_table(out, "Position001"), NAMESPACE, [p for p, _ in KEEP],
                        [g for _, g in KEEP], None)
    reference = _layout.reference_image_path(out, "Position001")
    save_volume(np.zeros(SHAPE_ZYX, np.uint8), reference,
                metadata=declared_metadata("data", "out", "Position001", voxel_size_xy=0.1, voxel_size_z=0.35))
    T.succeed(manifest, "fov")
    before = set(files(out))
    T.run_kinds(manifest, "segment", "assign")
    folder = out / "fovs" / "Position001" / "assignment" / "default"
    figures = ([] if previews is None else
               ["fovs/Position001/previews/assignment.png", "fovs/Position001/previews/assignment_single_layer.png"])
    written = [f for f in files(out, leave=("workflow/", "fovs/Position001/segmentation/")) if f not in before]
    assert written == sorted([f"fovs/Position001/assignment/default/{p.name}" for p in folder.iterdir()] + figures)
    assert not (out / "expr").exists()
    saved = read_assignment(folder, "default")
    assert saved.record["name"] == "default"
    counts = saved.record["counts"]
    assert counts["molecules"] == len(saved.molecules) == len(KEEP)
    assert saved.molecules.spot_id.tolist() == [f"m{i}" for i in range(len(KEEP))]     # input order, unfiltered
    assert {"assigned", "unassigned", "excluded_cell", "outside_grid", "cells_kept"} <= set(counts)
    recorded = ReferenceGrid(tuple(saved.record["grid"]["shape_zyx"]), ImageMetadata(**saved.record["grid"]["metadata"]),
                             "declared")
    assert same_grid(recorded, reference_grid_from_file(reference))   # the grid check of assemble passes
