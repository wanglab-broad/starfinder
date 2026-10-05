"""The §2.9 engineering validation of assignment: the A checks the task groups did not add (W-318).

Rows of the engineering validation design in docs/assignment-algorithms.md ("Checks"). Rows A1
to A20 are in the modules of their task groups; the section "Implemented checks" of that page
maps every row to its tests, and ``test_every_a_row_names_existing_tests`` keeps that map true.
This module adds:

* A2 on ``assign_golden`` with every identity of docs/assignment-contract.md ("Count
  accounting", 1 to 5) recomputed from the tables, in 3D and 2D, without and with the legacy
  expansion: the fixture without the gene-``Z`` molecule, its statuses from the matching
  subset of the pinned ``SEG_LABELS`` (the unmodified fixture, with gene ``Z``, serves the
  unknown-gene rejection and the legacy pins in ``test_assignment_golden.py``);
* A15's last clause on the ``boxes`` FOV: the files ``FOV.run`` writes (``run.json`` and the
  ``registered`` checkpoint of a load-only run of the ``seeded`` stain, and the ``candidates``
  checkpoint of the ``boxes`` molecules) and the saved segmentation runs of the ``boxes``
  masks are byte-identical before and after ``FOV.assign``.

``assign_golden`` and ``boxes`` use no random number; the ``seeded`` stain uses seed 101.
"""
import pytest

from starfinder.assignment import AssignmentConfig, assign_molecules
from starfinder.dataset import CheckpointConfig, Dataset, PipelineConfig, RoundState
from starfinder.io import ImageLoadConfig, save_volume
from starfinder.segmentation import ExpandLabelsConfig, SegmentationPlan

from .segmentation_fixtures import BOXES_METADATA, BOXES_SHAPE, boxes, seeded_stain
from .test_assignment import BOXES_MOLECULES, GOLDEN_GRID, GOLDEN_KEEP, check_accounting, golden_inputs
from .test_assignment_golden import MOLECULES, SEG_LABELS
from .test_assignment_persistence import EXPANSION, TABLES, boxes_codebook, hashes, import_run, set_boxes_reads
from .test_segmentation_validation import check_rows

pytestmark = [pytest.mark.segmentation, pytest.mark.validation]


def test_every_a_row_names_existing_tests():
    check_rows("A", 20)


@pytest.mark.parametrize("expand", [False, True], ids=["no_expansion", "planar_pixel_4"])
@pytest.mark.parametrize("dimensions", ["3d", "2d"])
def test_a2_assign_golden_statuses_and_identities_1_to_5(dimensions, expand):
    mols, cells = golden_inputs(dimensions)
    assert len(mols) == len(MOLECULES) - 1 == 18 and "Z" not in set(mols.table.gene_id)
    config = AssignmentConfig(expansion=ExpandLabelsConfig(4, "pixel", "planar") if expand else None)
    result = assign_molecules(mols, cells, grid=GOLDEN_GRID, config=config)
    pinned = [SEG_LABELS[(dimensions, expand)][i] for i in GOLDEN_KEEP]
    assert result.molecules.assignment_status.tolist() == ["assigned" if c else "unassigned" for c in pinned]
    counts = result.record["counts"]
    n_assigned = sum(1 for c in pinned if c)
    assert (counts["assigned"], counts["unassigned"], counts["excluded_cell"], counts["outside_grid"]) == \
        (n_assigned, len(pinned) - n_assigned, 0, 0)
    check_accounting(result, mols)


@pytest.mark.parametrize("table_format", ["csv", "parquet"])
def test_a15_fov_run_and_segmentation_files_of_boxes_are_untouched_by_assign(tmp_path, table_format):
    if table_format == "parquet":
        pytest.importorskip("pyarrow")
    data, root = tmp_path / "data", tmp_path / "checkpoints"
    save_volume(seeded_stain(), data / "round1" / "FOV_001" / "ch00.tif", metadata=BOXES_METADATA)
    masks = {"cell": tmp_path / "masks" / "cells.tif", "nucleus": tmp_path / "masks" / "nuclei.tif"}
    for (name, path), labels in zip(masks.items(), boxes()):
        save_volume(labels, path, metadata=BOXES_METADATA)
    dataset = Dataset(data, data / "out", "data", "sample", "out", RoundState(["round1"], reference_round="round1"),
                      ["ch00"])
    dataset.codebook = boxes_codebook()
    checkpoints = CheckpointConfig(directory=root, table_format=table_format, stages=("registered",))
    fov = dataset.fov("FOV_001").run(PipelineConfig(load=ImageLoadConfig(channel_labels=("ch00",))),
                                     checkpoints=checkpoints)
    assert fov.reference_grid().shape_zyx == BOXES_SHAPE and fov.reference_grid().metadata == BOXES_METADATA
    set_boxes_reads(fov)
    fov.save_checkpoint("candidates", checkpoints=checkpoints)
    folder = root / "FOV_001"
    run_files = hashes(root)
    assert {p.relative_to(folder).as_posix() for p in run_files} == {
        "run.json", "registered/round1.ome.tif", "registered/transforms.json", f"candidates.{table_format}",
        "candidates.json"}

    fov.segment(SegmentationPlan((import_run("nucleus", masks), import_run("cell", masks))), checkpoints=checkpoints)
    before = hashes(root)
    assert {p: h for p, h in before.items() if p in run_files} == run_files
    assert {p.relative_to(folder).as_posix() for p in before if p not in run_files} == {
        f"segmentation/{run}/{name}" for run in ("cell", "nucleus") for name in ("labels.tif", "segmentation.json")}

    fov.assign(AssignmentConfig(expansion=EXPANSION), nuclei="nucleus", checkpoints=checkpoints)
    after = hashes(root)
    assert {p: h for p, h in after.items() if p in before} == before       # FOV.run and segmentation untouched
    assignment = folder / "assignment" / "default"
    assert {p for p in after if p not in before} == {assignment / name for name in (
        "assignment.json", "territories.tif", *(f"{t}.{table_format}" for t in TABLES))}
    assert len(fov.assignment_results["default"].molecules) == len(BOXES_MOLECULES)
