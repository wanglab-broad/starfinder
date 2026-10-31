"""The morphology, segment and assign commands of the workflow (D6, W-736) on ``tiny_sample``.

C1 and C3 of design/validation/orchestration.md for these kinds beyond the expected manifests: a segment task of a
``cuda`` run asks for a GPU (``execution.gpu`` from the run's ``device``), and the morphology-only variant plans its 8
tasks with no processing block. The commands: a nuclear-stain run without a processing block lies on the reference
stain's saved form (OP4); a seeded run reads its seeds from the earlier segment task and gives the labels of the
whole plan; a given ``morphology.recipe`` and ``morphology.channel_images``; the previews of the assign and export
commands, written only when ``assignment.previews`` or ``export.previews`` is on (P5). And the Python route of the
assignment: the assign command reads ``fovs/<fov>/molecules.csv`` by key (moved from test_assignment_workflow.py,
which tested the reads_assignment rule's Python route).
"""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from starfinder import _layout
from starfinder.workflow import plan

from . import workflow_examples as X
from .coordination_helpers import write_molecules_csv
from .test_assignment_golden import SHAPE_ZYX, label_fixture
from .test_assignment_workflow import ASSIGNMENT_READS, CODEBOOK, KEEP, NAMESPACE, run_rule

pytestmark = [pytest.mark.workflow, pytest.mark.segmentation, pytest.mark.contract]

STAIN_RUN = {"name": "nucleus", "target": "nucleus", "method": "import"}


@pytest.fixture(scope="module")
def fixture_work(tmp_path_factory):
    work = tmp_path_factory.mktemp("tiny") / "work"
    X.golden().write_fixture(work)
    X.write_supplied(work)
    return work


def _plan(work, name, config):
    manifest = plan(X.write_config(work, name, config))
    return Path(manifest["roots"]["output"]) / "workflow" / "manifest.json", manifest


def _run(manifest, *kinds):
    """Run every task of the kinds, in the manifest's order."""
    for task in json.loads(Path(manifest).read_text())["tasks"]:
        if task["kind"] in kinds:
            assert X.run_task(manifest, task["id"]) == 0, task["id"]


def _single(work, output_id, **blocks):
    """FOV_001 of the fixture with the dataset keys and the given blocks."""
    return dict(X.dataset(work, output_id), selected_fovs=["FOV_001"], **blocks)


# --- C1 and C3: the GPU of a cuda run and the morphology-only variant -------------------------------------------

@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_a_segment_task_of_a_cuda_run_asks_for_a_gpu(fixture_work, device):
    config = X.variants(fixture_work)["morphology-only"]
    config["segmentation"]["device"] = device
    path, manifest = _plan(fixture_work, f"morphology-only-{device}", config)
    resolved = json.loads((path.parent / "resolved.json").read_text())["configuration"]
    assert "processing" not in resolved and "processing" not in resolved["api"]
    counts = {}
    for task in manifest["tasks"]:
        counts[task["kind"]] = counts.get(task["kind"], 0) + 1
    assert counts == {"morphology": 4, "segment": 4} and len(manifest["tasks"]) == 8
    for task in manifest["tasks"]:
        assert task["execution"]["gpu"] is (task["kind"] == "segment" and device == "cuda"), task["id"]
        if task["kind"] == "segment":
            fov = task["identity"]["fov"]
            assert task["needs"] == [f"morphology/sample/{fov}"]
            assert task["inputs"] == [f"fovs/{fov}/other_rounds/reference_stain/image.ome.tif",
                                      f"fovs/{fov}/other_rounds/reference_stain/registration.json"]


def test_a_nuclear_stain_run_lies_on_the_reference_stains_saved_form(fixture_work):
    """OP4: without a processing block the reference stain's saved form gives the run's grid (FOV.reference_grid)."""
    work = fixture_work
    run = dict(STAIN_RUN, path=str(work / "labels" / "{fov_id}.tif"))
    manifest, _ = _plan(work, "stain-grid", _single(work, "stain-grid", morphology={}, segmentation={"runs": [run]}))
    _run(manifest, "morphology", "segment")
    out = manifest.parents[1]
    record = json.loads((out / "fovs" / "FOV_001" / "segmentation" / "nucleus" / "segmentation.json").read_text())
    saved = json.loads((out / "fovs" / "FOV_001" / "other_rounds" / "reference_stain" / "registration.json")
                       .read_text())
    assert record["grid"]["source"] == "fov:reference_stain"
    assert record["grid"]["shape_zyx"] == [5, 36, 36] and record["grid"]["metadata"] == saved["image"]["metadata"]
    labels = tifffile.imread(out / "fovs" / "FOV_001" / "segmentation" / "nucleus" / "labels.tif")
    assert np.array_equal(labels, tifffile.imread(work / "labels" / "FOV_001.tif"))


def test_a_seeded_run_reads_its_seeds_from_the_earlier_segment_task(fixture_work):
    """One segment task per (FOV, run), in the order of their seeds (D6): the seeded run's task needs the seeds' task,
    reads its saved labels and gives the labels of the whole plan segmented in one call (FOV.segment)."""
    from starfinder.dataset.workflow import _workflow_dataset
    from starfinder.workflow._translate import _base, segmentation
    work = fixture_work
    processing = {k: v for k, v in copy.deepcopy(X.PROCESSING).items()
                  if k in ("load_codebook", "preprocessing", "registration")}
    runs = [dict(STAIN_RUN, path=str(work / "labels" / "{fov_id}.tif")),
            {"name": "cell", "target": "cell", "method": "seeded_watershed", "seeds": "nucleus",
             "inputs": [{"role": "amplicon", "reference_merged": True}]}]
    config = _single(work, "seeded", processing=dict(processing, checkpoints=[]), segmentation={"runs": runs})
    manifest, planned = _plan(work, "seeded", config)
    cell = next(t for t in planned["tasks"] if t["id"] == "segment/sample/FOV_001/cell")
    assert cell["needs"] == ["fov/sample/FOV_001", "segment/sample/FOV_001/nucleus"]
    _run(manifest, "fov", "segment")
    out = manifest.parents[1]
    resolved = json.loads((manifest.parent / "resolved.json").read_text())["configuration"]
    dataset = _workflow_dataset(_base(resolved))
    dataset.input_root, dataset.output_root = X.input_root(work), out
    whole = dataset.fov("FOV_001").load_reference_image()
    plan_, _ = segmentation(resolved)
    nucleus = plan_.runs[0]
    from dataclasses import replace
    from starfinder.segmentation import SegmentationPlan
    plan_ = SegmentationPlan((replace(nucleus, method=replace(nucleus.method, path=work / "labels" / "FOV_001.tif")),
                              plan_.runs[1]))
    whole.segment(plan_)
    for name in ("nucleus", "cell"):
        saved = tifffile.imread(out / "fovs" / "FOV_001" / "segmentation" / name / "labels.tif")
        assert np.array_equal(saved, whole.segmentation_results[name].labels), name
    record = json.loads((out / "fovs" / "FOV_001" / "segmentation" / "cell" / "segmentation.json").read_text())
    assert record["seeds"]["run"] == "nucleus"
    assert record["seeds"]["sha256"] == whole.segmentation_results["nucleus"].record["labels"]["sha256"]
    # FOV.segment(run=...) computes that run only, its seeds from segmentation_results.
    alone = dataset.fov("FOV_001").load_reference_image()
    with pytest.raises(ValueError, match="its seeds 'nucleus' are not in segmentation_results"):
        alone.segment(plan_, run="cell")
    with pytest.raises(ValueError, match="run 'other' is not a run of the plan"):
        alone.segment(plan_, run="other")
    alone.load_segmentation("nucleus")
    alone.segment(plan_, run="cell")
    assert set(alone.segmentation_results) == {"nucleus", "cell"}
    assert np.array_equal(alone.segmentation_results["cell"].labels, whole.segmentation_results["cell"].labels)


def test_the_reference_stain_gives_the_grid_without_the_reference_round(fixture_work, tmp_path):
    """OP4 in the API: FOV.reference_grid of a FOV whose only resident image is the prepared reference stain."""
    from starfinder.dataset import CheckpointConfig
    from starfinder.dataset.workflow import _workflow_dataset
    from starfinder.workflow._translate import _base
    dataset = _workflow_dataset(_base(dict(X.dataset(fixture_work, "api"), root_output_path=str(tmp_path))))
    with pytest.raises(ValueError, match=r'load_registered_round\("reference_stain"\) first'):
        dataset.fov("FOV_001").reference_grid()
    prepared = dataset.fov("FOV_001").prepare_morphology(checkpoints=CheckpointConfig())
    restored = dataset.fov("FOV_001").load_registered_round("reference_stain")
    for fov in (prepared, restored):
        grid = fov.reference_grid()
        assert grid.shape_zyx == (5, 36, 36) and grid.source == "fov:reference_stain"
        assert grid.metadata == fov.metadata["reference_stain"]
    assert prepared.reference_grid() == restored.reference_grid()


# --- the morphology block ----------------------------------------------------------------------------------------

def test_a_given_recipe_and_the_channel_images_of_the_morphology_block(fixture_work):
    """morphology.recipe, translated as a registration recipe, is the recipe of every round; here the default's one
    translation step on the shared stain, so the record equals the default's. The channel images are written only
    with morphology.channel_images, ZYX or (projection: max) their Z maximum."""
    work = fixture_work
    recipe = {"steps": [{"method": "translation"}],
              "signal": {"mode": "channel", "reference_channel": "ch04", "moving_channel": "ch04"}}
    records, outs = {}, {}
    for name, block in (("default", {}), ("recipe", {"recipe": recipe, "channel_images": {"projection": "max"}}),
                        ("volumes", {"channel_images": {"projection": None}})):
        manifest, _ = _plan(work, f"morphology-{name}", _single(work, f"morphology-{name}", morphology=block))
        _run(manifest, "morphology")
        outs[name] = manifest.parents[1]
        records[name] = json.loads((outs[name] / "fovs" / "FOV_001" / "nuclei_registration.json").read_text())
    resolved = json.loads((outs["recipe"] / "workflow" / "resolved.json").read_text())["configuration"]
    assert resolved["api"]["morphology"]["recipe"]["type"] == "RegistrationRecipe"
    assert records["recipe"] == records["default"]
    assert not (outs["default"] / "images").exists()
    for label in ("ch04", "ch05"):
        volume = tifffile.imread(outs["volumes"] / "images" / "flamingo" / label / "FOV_001.tif")
        plane = tifffile.imread(outs["recipe"] / "images" / "flamingo" / label / "FOV_001.tif")
        assert volume.shape == (5, 36, 36) and plane.shape == (36, 36) and plane.dtype == volume.dtype
        assert np.array_equal(plane, volume.max(axis=0))


# --- the previews (P5) -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("previews", [False, True])
def test_the_assign_previews_are_written_only_when_their_option_is_on(fixture_work, previews):
    work = fixture_work
    run = {"name": "cell", "target": "cell", "method": "import", "path": str(work / "labels" / "{fov_id}.tif")}
    name = f"assign-previews-{str(previews).lower()}"
    config = _single(work, name, processing=copy.deepcopy(X.PROCESSING), segmentation={"runs": [run]},
                     assignment={"cells": "cell", "previews": previews})
    manifest, planned = _plan(work, name, config)
    task = next(t for t in planned["tasks"] if t["kind"] == "assign")
    assert ("fovs/FOV_001/previews/" in task["outputs"]) is previews
    _run(manifest, "fov", "segment", "assign")
    out = manifest.parents[1]
    folder = _layout.fov_previews_dir(out, "FOV_001")
    assert folder.is_dir() is previews
    if previews:
        assert sorted(p.name for p in folder.iterdir()) == ["assignment.png", "assignment_single_layer.png"]
        record = json.loads(_layout.task_record(out, task["id"]).read_text())
        (entry,) = [o for o in record["outputs"] if o["path"] == "fovs/FOV_001/previews/"]
        assert [f["path"] for f in entry["files"]] == ["assignment.png", "assignment_single_layer.png"]


@pytest.mark.slow
def test_the_export_previews_are_written_only_when_their_option_is_on(fixture_work):
    """The per-fov variant run to its export, with both previews on and with both off."""
    if importlib.util.find_spec("multiview_stitcher") is None or importlib.util.find_spec("anndata") is None:
        pytest.skip("the stitching and the export of this variant need the multiview-stitcher and anndata extras")
    work = fixture_work
    for previews in (True, False):
        config = X.variants(work)["per-fov"]
        name = f"export-previews-{str(previews).lower()}"
        config["output_id"] = name
        config["assignment"]["previews"] = config["export"]["previews"] = previews
        manifest, planned = _plan(work, name, config)
        export = next(t for t in planned["tasks"] if t["kind"] == "export")
        assert ("export/sample/previews/" in export["outputs"]) is previews
        _run(manifest, "summarize", "merge", "fov", "morphology", "segment", "assign", "stitch", "export")
        out = manifest.parents[1]
        folder = _layout.export_previews_dir(out, "sample")
        assert folder.is_dir() is previews
        for fov in X.FOVS:
            assert _layout.fov_previews_dir(out, fov).is_dir() is previews
            if previews:
                assert sorted(p.name for p in (folder / fov).iterdir()) == ["assignment.png",
                                                                            "assignment_single_layer.png"]


# --- the Python route of the assignment (moved from test_assignment_workflow.py) -------------------------------------

def _assign_by_command(tmp_path, labels, spot_ids):
    """The assign command on one FOV (Position001) of run_rule's molecules, labels and codebook.

    The plan needs the processing block (the assignment's molecules) and its raw images, which are empty files here:
    the fov task is not run. Its molecule table is written as FOV.export_spots writes it, its reference image (the
    grid) with the configured spacing, and its task record says succeeded; the segment task imports the labels
    (``projection: true`` for a YX label image, a plane on the Z projection of the grid)."""
    from starfinder.dataset._declared import declared_metadata
    from starfinder.io import save_volume
    work = tmp_path / "work"
    root = work / "input" / "data" / "sample"
    root.mkdir(parents=True)
    (root / "genes.csv").write_text("".join(f"{g},{b}\n" for g, b in CODEBOOK))
    (root / "fovs.csv").write_text("sample,fov_id\nsample1,Position001\n")
    for r in (1, 2, 3):
        for c in range(4):
            (root / f"round{r}" / "Position001").mkdir(parents=True, exist_ok=True)
            (root / f"round{r}" / "Position001" / f"ch0{c}.tif").write_bytes(b"")
    (work / "labels").mkdir()
    tifffile.imwrite(work / "labels" / "Position001.tif", labels)
    run = {"name": "cell", "target": "cell", "method": "import", "path": str(work / "labels" / "{fov_id}.tif")}
    if labels.ndim == 2:
        run["projection"] = True
    config = {"root_input_path": str(work / "input"), "root_output_path": str(work / "output"), "dataset_id": "data",
              "sample_id": "sample", "output_id": "out", "n_rounds": 3, "ref_round": "round1", "rotate_angle": 0,
              "voxel_size_xy": 0.1, "voxel_size_z": 0.35, "seq_channel_order": ["ch00", "ch01", "ch02", "ch03"],
              "processing": copy.deepcopy(X.PROCESSING), "segmentation": {"runs": [run]},
              "assignment": {"cells": "cell"}}
    manifest, _ = _plan(work, "python-route", config)
    out = manifest.parents[1]
    molecules = write_molecules_csv(_layout.spot_table(out, "Position001"), NAMESPACE, [p for p, _ in KEEP],
                                    [g for _, g in KEEP], spot_ids)
    save_volume(np.zeros(SHAPE_ZYX, np.uint16), _layout.reference_image_path(out, "Position001"),
                metadata=declared_metadata("data", "out", "Position001", voxel_size_xy=0.1, voxel_size_z=0.35))
    record = _layout.task_record(out, "fov/sample1/Position001")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"id": "fov/sample1/Position001", "status": "succeeded"}) + "\n")
    _run(manifest, "segment", "assign")
    return _layout.assignment_dir(_layout.fov_dir(out, "Position001"), "default"), molecules


@pytest.mark.parametrize("dimensions", ["3d", "2d"])
def test_the_python_route_reads_molecules_csv_by_key(tmp_path, dimensions):
    """The Python route: the assign command reads fovs/<fov>/molecules.csv with load_molecules, so the saved
    assignment keeps the molecules' spot_ids (no csv:<i>) and zero-based coordinates; the statuses, the cells and
    the counts equal those of the same molecules on the legacy route (the reads_assignment rule of the MATLAB
    backend)."""
    labels = label_fixture() if dimensions == "3d" else label_fixture().max(axis=0)
    ids = [f"spot-{9 - i}" for i in range(len(KEEP))]  # identities that are neither row numbers nor in row order
    folder, path = _assign_by_command(tmp_path, labels, ids)
    legacy_adata, legacy_reads, _ = run_rule(tmp_path / "legacy", labels, legacy=True)
    reads = pd.read_csv(folder / "molecules.csv", keep_default_na=False, na_values=["", "<NA>"])
    assert path.name == "molecules.csv" and path.parent.name == "Position001"
    assert reads.spot_id.tolist() == ids and not reads.spot_id.str.startswith("csv:").any()
    assert set(reads.spot_namespace) == {NAMESPACE}
    assert reads[["z", "y", "x"]].to_numpy().tolist() == [list(p) for p, _ in KEEP]
    # The legacy rows, in the order of the molecules (csv:<row>), against the saved assignment's.
    legacy = legacy_reads.assign(row=legacy_reads.spot_id.str[len("csv:"):].astype(int)).sort_values("row")
    assert reads.gene_id.tolist() == legacy.gene.tolist()
    for column in ASSIGNMENT_READS:
        assert [None if pd.isna(v) else v for v in reads[column]] == \
            [None if pd.isna(v) else v for v in legacy[column]], column
    counts = pd.read_csv(folder / "counts.csv")
    matrix = counts.pivot_table(index="cell_id", columns="gene_id", values="count", aggfunc="sum", fill_value=0)
    matrix = matrix.reindex(index=legacy_adata.obs.seg_label.astype(int).tolist(),
                            columns=list(legacy_adata.var_names), fill_value=0)
    assert np.array_equal(matrix.to_numpy(), np.asarray(legacy_adata.X))
    cells = pd.read_csv(folder / "cells.csv")
    assert cells.cell_id.tolist() == legacy_adata.obs.seg_label.astype(int).tolist()
    assert cells.n_molecules.tolist() == legacy_adata.obs.n_molecules.tolist()
    record = json.loads((folder / "assignment.json").read_text())
    assert record["inputs"]["molecules"]["source"]["path"] == str(path)
    assert record["inputs"]["cells"]["saved_under_run"] is True
