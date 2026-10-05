"""The saved layout of an assignment: ``FOV.assign(checkpoints=…)`` and ``FOV.load_assignment`` (W-315).

Row A15 of the engineering validation design in docs/assignment-algorithms.md over every
case of the table "Label images of a checkpointed assignment" of
docs/assignment-contract.md, in CSV and Parquet; another checkpoint root counts as
unsaved; the excluded cells are preserved; and the files of ``FOV.run`` and of
segmentation are untouched by ``FOV.assign``. The ``boxes`` FOV is synthetic: its
reference round, spot result, filtering result and codebook are set in memory.
"""
import hashlib
import json
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from starfinder.assignment import AssignmentConfig
from starfinder.barcode import ReadFilterConfig, ReadFilteringResult
from starfinder.dataset import CheckpointConfig, Dataset, RoundState
from starfinder.io import save_volume
from starfinder.io._checkpoint import FORMAT_VERSION, STAGES
from starfinder.segmentation import (ExpandLabelsConfig, InputChannel, LabelImportConfig, SeededWatershedConfig,
                                     SegmentationPlan, SegmentationRun, import_labels)
from starfinder.segmentation._persist import read_labels
from starfinder.spot_finding import LocalMaximaConfig, SpotFindingResult
from starfinder.synthetic import development_scene_preset

from .segmentation_fixtures import BOXES_METADATA, boxes, paint, seeded_stain
from .test_assignment import BOXES_MOLECULES

pytestmark = [pytest.mark.segmentation, pytest.mark.validation]

SPOT_NAMESPACE = json.dumps(["data", "sample", "FOV_001", None], separators=(",", ":"))
EXPANSION = ExpandLabelsConfig(0.1, "um", "planar")
TABLES = ("molecules", "cells", "counts", "nuclei")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(root):
    return {p: file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def genes():
    """The development codebook with four entries, one per boxes gene A to D."""
    book, _ = development_scene_preset("clean", size="small")
    sequences = ["123", "214", "341", "432"]
    return replace(book, table=pd.DataFrame({"entry_id": sequences, "gene_id": list("ABCD"),
                                             "color_sequence": sequences}))


def boxes_fov(root, book):
    """FOV_001 on the boxes grid with the boxes molecules as its accepted reads."""
    dataset = Dataset(root, root / "out", "data", "sample", "out", RoundState(["round1"], reference_round="round1"),
                      ["ch00"])
    dataset.codebook = book
    fov = dataset.fov("FOV_001")
    fov.images["round1"] = seeded_stain()[..., None]
    fov.metadata["round1"] = BOXES_METADATA
    ids = pd.array([f"m{i}" for i in range(len(BOXES_MOLECULES))], dtype="string")
    positions = np.array([p for p, _, _, _ in BOXES_MOLECULES], np.float64)
    spots = pd.DataFrame({"spot_id": ids, "z": positions[:, 0], "y": positions[:, 1], "x": positions[:, 2]})
    fov.spot_result = SpotFindingResult(spots, BOXES_METADATA, SPOT_NAMESPACE, LocalMaximaConfig(), {})
    reads = pd.DataFrame({"spot_namespace": pd.array([SPOT_NAMESPACE] * len(ids), dtype="string"), "spot_id": ids,
                          "gene_id": pd.array([g for _, g, _, _ in BOXES_MOLECULES], dtype="string"),
                          "accepted": np.ones(len(ids), bool)})
    n = len(ids)
    fov.filtering_result = ReadFilteringResult(reads, SPOT_NAMESPACE, ReadFilterConfig(),
                                               {"total": n, "accepted": n, "rejected": 0}, {"accepted": 1.0}, {})
    return fov


@pytest.fixture
def masks(tmp_path):
    """The boxes cells and nuclei written as TIFFs with the boxes metadata, outside every checkpoint root."""
    cells, nuclei = boxes()
    paths = {"cell": tmp_path / "masks" / "cells.tif", "nucleus": tmp_path / "masks" / "nuclei.tif"}
    save_volume(cells, paths["cell"], metadata=BOXES_METADATA)
    save_volume(nuclei, paths["nucleus"], metadata=BOXES_METADATA)
    return paths


def import_run(name, paths):
    return SegmentationRun(name, name, (), LabelImportConfig(str(paths[name]), name))


@pytest.mark.parametrize("expansion", [False, True], ids=["no_expansion", "planar_0.1um"])
@pytest.mark.parametrize("nuclei", ["absent", "saved", "unsaved"])
@pytest.mark.parametrize("cells_saved", [True, False], ids=["cells_saved", "cells_unsaved"])
@pytest.mark.parametrize("table_format", ["csv", "parquet"])
def test_a15_round_trip_in_each_case_of_the_label_table(tmp_path, genes, masks, table_format, cells_saved, nuclei,
                                                        expansion):
    if table_format == "parquet":
        pytest.importorskip("pyarrow")
    root = tmp_path / "checkpoints"
    checkpoints = CheckpointConfig(directory=root, table_format=table_format)
    fov = boxes_fov(tmp_path, genes)
    with_nuclei = nuclei != "absent"
    segmentation = root / "FOV_001" / "segmentation"
    # Saved runs by FOV.segment(checkpoints=…); an unsaved cell run in memory only; unsaved nuclei made by
    # import_labels (their source file lies outside the layout).
    saved = [run for run, on in (("cell", cells_saved), ("nucleus", nuclei == "saved")) if on]
    if saved:
        fov.segment(SegmentationPlan(tuple(import_run(r, masks) for r in saved)), checkpoints=checkpoints)
    if not cells_saved:
        fov.segment(SegmentationPlan((import_run("cell", masks),)))
    argument = {"absent": None, "saved": "nucleus"}.get(nuclei)
    if nuclei == "unsaved":
        argument = import_labels(masks["nucleus"], grid=fov.reference_grid(), target="nucleus",
                                 label_namespace=json.dumps(["data", "sample", "FOV_001", None, "nucleus"],
                                                            separators=(",", ":")))
    before = hashes(root) if root.exists() else {}
    fov.assign(AssignmentConfig(expansion=EXPANSION if expansion else None), cells="cell", nuclei=argument,
               checkpoints=checkpoints)
    # The files of segmentation are byte-identical after FOV.assign (linked, never written).
    assert {p: h for p, h in hashes(root).items() if p in before} == before
    assert all(segmentation in p.parents for p in before)

    # Exactly the files of the table "Label images of a checkpointed assignment".
    folder = root / "FOV_001" / "assignment" / "default"
    expected = {"assignment.json"} | {f"{t}.{table_format}" for t in TABLES if t != "nuclei" or with_nuclei}
    expected |= {"territories.tif"} if expansion else set()
    expected |= set() if cells_saved else {"cell_labels.tif"}
    expected |= {"nucleus_labels.tif"} if nuclei == "unsaved" else set()
    assert {p.name for p in folder.iterdir()} == expected
    assert {p for p in hashes(root) if p not in before} == {folder / name for name in expected}

    # Every file link resolves to an existing file with the recorded SHA-256.
    record = json.loads((folder / "assignment.json").read_text())
    inputs = record["inputs"]
    cell_path = "../../segmentation/cell/labels.tif" if cells_saved else "cell_labels.tif"
    assert inputs["cells"]["saved_under_run"] is cells_saved and inputs["cells"]["file"]["path"] == cell_path
    links = [inputs["cells"]["file"]] + list(record["files"].values())
    if with_nuclei:
        nucleus_path = {"saved": "../../segmentation/nucleus/labels.tif", "unsaved": "nucleus_labels.tif"}[nuclei]
        assert inputs["nuclei"]["saved_under_run"] is (nuclei == "saved")
        assert inputs["nuclei"]["file"]["path"] == nucleus_path
        links.append(inputs["nuclei"]["file"])
    else:
        assert inputs["nuclei"] is None
    for link in links:
        path = (folder / link["path"]).resolve()
        assert path.is_file() and file_hash(path) == link["sha256"]
    assert set(record["files"]) == ({t for t in TABLES if t != "nuclei" or with_nuclei}
                                    | ({"territories"} if expansion else set())
                                    | (set() if cells_saved else {"cell_labels"})
                                    | ({"nucleus_labels"} if nuclei == "unsaved" else set()))
    if expansion:
        assert record["expansion"]["original"]["path"] == cell_path
        assert record["expansion"]["expanded"]["path"] == "territories.tif"
        assert record["expansion"]["expanded"]["file_sha256"] == record["files"]["territories"]["sha256"]
    else:
        assert record["expansion"] is None

    # The reload equals the stored result: tables, label arrays and record.
    result = fov.assignment_results["default"]
    assert result.record == record
    fresh = boxes_fov(tmp_path, genes)
    loaded = fresh.load_assignment("default", checkpoints=CheckpointConfig(directory=root))
    assert fresh.assignment_results["default"] is loaded
    for name in TABLES:
        if getattr(result, name) is None:
            assert getattr(loaded, name) is None and not with_nuclei
        else:
            assert_frame_equal(getattr(loaded, name), getattr(result, name), check_exact=True)
    for name in ("cell_labels", "territories", "nucleus_labels"):
        if getattr(result, name) is None:
            assert getattr(loaded, name) is None
        else:
            assert getattr(loaded, name).dtype == np.uint32
            assert np.array_equal(getattr(loaded, name), getattr(result, name))
    assert (loaded.territories is not None) is expansion and (loaded.nucleus_labels is not None) is with_nuclei
    assert loaded.genes == result.genes and loaded.cell_namespace == result.cell_namespace
    assert loaded.record == result.record
    np.testing.assert_array_equal(loaded.matrix()[0], result.matrix()[0])


@pytest.mark.parametrize("fault", ["written_table", "written_labels", "linked_changed", "linked_removed"])
def test_a15_a_changed_or_removed_file_raises_naming_the_path(tmp_path, genes, masks, fault):
    root = tmp_path / "checkpoints"
    checkpoints = CheckpointConfig(directory=root)
    fov = boxes_fov(tmp_path, genes)
    fov.segment(SegmentationPlan((import_run("cell", masks),)), checkpoints=checkpoints)
    nuclei = import_labels(masks["nucleus"], grid=fov.reference_grid(), target="nucleus",
                           label_namespace=json.dumps(["data", "sample", "FOV_001", None, "nucleus"],
                                                      separators=(",", ":")))
    fov.assign(AssignmentConfig(expansion=EXPANSION), nuclei=nuclei, checkpoints=checkpoints)
    folder = root / "FOV_001" / "assignment" / "default"
    linked = root / "FOV_001" / "segmentation" / "cell" / "labels.tif"
    target = {"written_table": folder / "counts.csv", "written_labels": folder / "nucleus_labels.tif",
              "linked_changed": linked, "linked_removed": linked}[fault]
    if fault == "linked_removed":
        target.unlink()
    elif fault == "written_table":
        target.write_text(target.read_text().replace(",1\n", ",2\n", 1))
    else:
        labels, _ = read_labels(target)
        labels[0, 0, 0] += 1
        save_volume(labels, target, compress=True, metadata=BOXES_METADATA)
    with pytest.raises(ValueError, match="missing" if fault == "linked_removed" else "SHA-256") as error:
        boxes_fov(tmp_path, genes).load_assignment("default", checkpoints=checkpoints)
    assert str(target) in str(error.value)


def test_a15_errors_and_overwrite(tmp_path, genes, masks):
    root = tmp_path / "checkpoints"
    checkpoints = CheckpointConfig(directory=root)
    fov = boxes_fov(tmp_path, genes)
    fov.segment(SegmentationPlan((import_run("cell", masks),)))
    with pytest.raises(FileNotFoundError, match="no saved assignment 'default'"):
        fov.load_assignment("default", checkpoints=checkpoints)
    fov.assign(checkpoints=checkpoints)
    before = hashes(root)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        fov.assign(AssignmentConfig(expansion=EXPANSION), checkpoints=checkpoints)
    assert hashes(root) == before and fov.assignment_results["default"].territories is None
    # Overwriting removes the files of the earlier layout that the new one does not write.
    fov.assign(AssignmentConfig(expansion=EXPANSION), checkpoints=replace(checkpoints, overwrite=True))
    fov.assign(checkpoints=replace(checkpoints, overwrite=True))
    folder = root / "FOV_001" / "assignment" / "default"
    assert {p.name for p in folder.iterdir()} == {"assignment.json", "molecules.csv", "cells.csv", "counts.csv",
                                                  "cell_labels.tif"}
    with pytest.raises(TypeError, match="CheckpointConfig"):
        fov.assign(checkpoints=str(root))


def test_another_checkpoint_root_counts_as_unsaved(tmp_path, genes, masks):
    first, second = tmp_path / "first", tmp_path / "second"
    fov = boxes_fov(tmp_path, genes)
    fov.segment(SegmentationPlan((import_run("cell", masks),)), checkpoints=CheckpointConfig(directory=first))
    before = hashes(first)
    fov.assign(checkpoints=CheckpointConfig(directory=second))
    folder = second / "FOV_001" / "assignment" / "default"
    assert (folder / "cell_labels.tif").is_file()
    assert not (second / "FOV_001" / "segmentation").exists()
    record = json.loads((folder / "assignment.json").read_text())
    cells = record["inputs"]["cells"]
    assert cells["saved_under_run"] is False
    assert cells["file"] == {"path": "cell_labels.tif", "sha256": file_hash(folder / "cell_labels.tif")}
    labels, _ = read_labels(folder / "cell_labels.tif")
    assert np.array_equal(labels, fov.segmentation_results["cell"].labels)
    assert hashes(first) == before    # nothing in the first root changes


def other_run(run):
    return "nucleus" if run == "cell" else "cell"


@pytest.mark.parametrize("source", ["segmented_on_another_fov_object", "loaded_on_another_fov_object",
                                    "first_of_two_loads"])
@pytest.mark.parametrize("run", ["cell", "nucleus"])
@pytest.mark.parametrize("table_format", ["csv", "parquet"])
def test_a15_a_saved_run_stays_saved_after_a_reload_or_on_another_fov_object(tmp_path, genes, masks, table_format,
                                                                             run, source):
    """A run written by FOV.segment(checkpoints=…) or loaded by FOV.load_segmentation in the assignment's root is
    linked, also when another FOV object of the same FOV assigns it, or when the run was loaded again since."""
    if table_format == "parquet":
        pytest.importorskip("pyarrow")
    root = tmp_path / "checkpoints"
    checkpoints = CheckpointConfig(directory=root, table_format=table_format)
    saving = boxes_fov(tmp_path, genes)
    saving.segment(SegmentationPlan((import_run(run, masks),)), checkpoints=checkpoints)
    assigning = saving if source == "first_of_two_loads" else boxes_fov(tmp_path, genes)
    if source == "segmented_on_another_fov_object":
        result = saving.segmentation_results[run]
    elif source == "loaded_on_another_fov_object":
        result = saving.load_segmentation(run, checkpoints=checkpoints)
    else:
        result = saving.load_segmentation(run, checkpoints=checkpoints)
        second = saving.load_segmentation(run, checkpoints=checkpoints)
        assert second is not result and saving.segmentation_results[run] is second
    # The other run is kept in memory on the assigning FOV object: unsaved, so written.
    assigning.segment(SegmentationPlan((import_run(other_run(run), masks),)))
    runs = {run: result, other_run(run): assigning.segmentation_results[other_run(run)]}
    before = hashes(root)
    assigning.assign(cells=runs["cell"], nuclei=runs["nucleus"], checkpoints=checkpoints)
    assert {p: h for p, h in hashes(root).items() if p in before} == before

    folder = root / "FOV_001" / "assignment" / "default"
    names = {"cell": "cell_labels.tif", "nucleus": "nucleus_labels.tif"}
    present = {p.name for p in folder.iterdir()}
    assert names[run] not in present and names[other_run(run)] in present
    record = json.loads((folder / "assignment.json").read_text())
    keys = {"cell": "cells", "nucleus": "nuclei"}
    linked, written = record["inputs"][keys[run]], record["inputs"][keys[other_run(run)]]
    labels_tif = root / "FOV_001" / "segmentation" / run / "labels.tif"
    assert linked["saved_under_run"] is True
    assert linked["file"] == {"path": f"../../segmentation/{run}/labels.tif", "sha256": file_hash(labels_tif)}
    assert written["saved_under_run"] is False
    assert written["file"] == {"path": names[other_run(run)], "sha256": file_hash(folder / names[other_run(run)])}
    assert set(record["files"]) == {"molecules", "cells", "counts", "nuclei", names[other_run(run)][:-4]}

    stored = assigning.assignment_results["default"]
    loaded = boxes_fov(tmp_path, genes).load_assignment("default", checkpoints=checkpoints)
    for name in TABLES:
        assert_frame_equal(getattr(loaded, name), getattr(stored, name), check_exact=True)
    assert np.array_equal(loaded.cell_labels, stored.cell_labels)
    assert np.array_equal(loaded.nucleus_labels, stored.nucleus_labels)
    assert loaded.record == stored.record == record


@pytest.mark.parametrize("case", ["label_file_changed", "caller_made_copy", "segment_without_checkpoints",
                                  "import_labels"])
def test_a15_inputs_the_rule_calls_unsaved_are_written(tmp_path, genes, masks, case):
    root = tmp_path / "checkpoints"
    checkpoints = CheckpointConfig(directory=root)
    fov = boxes_fov(tmp_path, genes)
    fov.segment(SegmentationPlan((import_run("cell", masks),)), checkpoints=checkpoints)
    labels_tif = root / "FOV_001" / "segmentation" / "cell" / "labels.tif"
    cells = fov.load_segmentation("cell", checkpoints=checkpoints)
    if case == "label_file_changed":
        changed = cells.labels.copy()
        changed[0, 0, 0] += 1
        save_volume(changed, labels_tif, compress=True, metadata=BOXES_METADATA)
    elif case == "caller_made_copy":
        cells = replace(cells)
    elif case == "segment_without_checkpoints":
        cells = boxes_fov(tmp_path, genes).segment(SegmentationPlan((import_run("cell", masks),))) \
            .segmentation_results["cell"]
    else:
        cells = import_labels(labels_tif, grid=fov.reference_grid(), target="cell",
                              label_namespace=json.dumps(["data", "sample", "FOV_001", None, "cell"],
                                                         separators=(",", ":")))
    fov.assign(cells=cells, checkpoints=checkpoints)
    folder = root / "FOV_001" / "assignment" / "default"
    entry = json.loads((folder / "assignment.json").read_text())["inputs"]["cells"]
    assert entry["saved_under_run"] is False
    assert entry["file"] == {"path": "cell_labels.tif", "sha256": file_hash(folder / "cell_labels.tif")}
    labels, _ = read_labels(folder / "cell_labels.tif")
    assert np.array_equal(labels, cells.labels)


@pytest.mark.parametrize("table_format", ["csv", "parquet"])
def test_excluded_cells_are_preserved(tmp_path, genes, masks, table_format):
    if table_format == "parquet":
        pytest.importorskip("pyarrow")
    checkpoints = CheckpointConfig(directory=tmp_path / "checkpoints", table_format=table_format)
    fov = boxes_fov(tmp_path, genes)
    fov.segment(SegmentationPlan((import_run("nucleus", masks), import_run("cell", masks))), checkpoints=checkpoints)
    fov.assign(nuclei="nucleus", checkpoints=checkpoints)
    loaded = boxes_fov(tmp_path, genes).load_assignment("default", checkpoints=checkpoints)
    assert loaded.record["config"]["exclusion_source"] == "default"
    cells = loaded.cells.set_index(loaded.cells.cell_id.astype(int))
    for cell in (7, 8):
        assert cells.loc[cell, "status"] == "excluded_no_nucleus"
        assert cells.loc[cell, "exclusion_reason"] == "no_matched_nucleus"
        expected = [f"m{i}" for i, (_, _, c, _) in enumerate(BOXES_MOLECULES) if c == cell]
        mine = loaded.molecules[loaded.molecules.cell_id.eq(cell).fillna(False).to_numpy(bool)]
        assert mine.spot_id.tolist() == expected and len(expected) == cells.loc[cell, "n_molecules"] > 0
        assert mine.assignment_status.eq("excluded_cell").all()
        assert not loaded.counts.cell_id.eq(cell).any()
    assert set(cells.index[cells.status.eq("kept")]) == {1, 2, 3, 4, 5, 6}
    # Their values are in the cell-label file the record links, and in the reloaded array.
    path = (tmp_path / "checkpoints" / "FOV_001" / "assignment" / "default"
            / loaded.record["inputs"]["cells"]["file"]["path"]).resolve()
    labels, _ = read_labels(path)
    assert {7, 8} <= set(np.unique(labels)) and np.array_equal(labels, loaded.cell_labels)
    assert np.array_equal(labels, boxes()[0])


# --- Untouched files: FOV.run, then FOV.segment and FOV.assign in one root ---------------------------

@pytest.mark.dataset
def test_fov_run_and_segmentation_files_are_untouched_by_assign(tmp_path):
    from starfinder.barcode import NeighborhoodSumConfig, WtaDecoderConfig
    from starfinder.dataset import PipelineConfig, RegistrationRecipe, RegistrationStep
    from starfinder.io import ImageLoadConfig
    from starfinder.registration import TranslationConfig
    from starfinder.synthetic import generate_formed_scene

    book, config = development_scene_preset("clean", size="small")
    scene = generate_formed_scene(book, config=config)
    data = tmp_path / "data"
    for name, image in scene.rounds.items():
        metadata = replace(scene.round_metadata[name], spacing_zyx=(0.35, 0.1, 0.1), spatial_unit="micrometer")
        for c, channel in enumerate(scene.channel_labels):
            save_volume(image[..., c], data / name / "FOV_001" / f"{channel}.tif", metadata=metadata)
    rounds = RoundState(list(scene.round_labels), reference_round=scene.round_labels[0])
    dataset = Dataset(data, data / "out", "dev", "sample", "out", rounds, list(scene.channel_labels))
    dataset.codebook = book
    pipeline = PipelineConfig(load=ImageLoadConfig(channel_labels=tuple(scene.channel_labels)),
                              registration=RegistrationRecipe((RegistrationStep(TranslationConfig()),)),
                              spot_finding=LocalMaximaConfig(), extraction=NeighborhoodSumConfig(),
                              decoding=WtaDecoderConfig(), filtering=ReadFilterConfig())
    root = tmp_path / "checkpoints"
    checkpoints = CheckpointConfig(directory=root)
    fov = dataset.fov("FOV_001")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fov.run(pipeline, checkpoints=checkpoints)
    after_run = hashes(root)
    grid = fov.reference_grid()
    shape = grid.shape_zyx
    nuclei = paint({5: ((0, shape[0]), (2, 6), (2, 6)), 9: ((0, shape[0]), (20, 24), (20, 24))}, shape)
    save_volume(nuclei, tmp_path / "nuclei.tif", metadata=grid.metadata)
    fov.segment(SegmentationPlan((
        SegmentationRun("nucleus", "nucleus", (), LabelImportConfig(str(tmp_path / "nuclei.tif"), "nucleus")),
        SegmentationRun("cell", "cell", (InputChannel("amplicon", reference_merged=True),),
                        SeededWatershedConfig(sigma_um=0.1), seeds="nucleus"))), checkpoints=checkpoints)
    before = hashes(root)
    assert {p: h for p, h in before.items() if p in after_run} == after_run    # segment leaves FOV.run's files
    fov.assign(AssignmentConfig(expansion=EXPANSION), nuclei="nucleus", checkpoints=checkpoints)
    after = hashes(root)
    assert {p: h for p, h in after.items() if p in before} == before
    folder = root / "FOV_001" / "assignment" / "default"
    assert {p.parent for p in after if p not in before} == {folder}
    assert {p.name for p in folder.iterdir()} == {"assignment.json", "molecules.csv", "cells.csv", "counts.csv",
                                                  "nuclei.csv", "territories.tif"}
    run = json.loads((root / "FOV_001" / "run.json").read_text())
    assert run["format_version"] == 1 and run["config"]["checkpoints"]["stages"] == list(STAGES)
    assert STAGES == CheckpointConfig().stages == ("registered", "candidates", "pre_qc") and FORMAT_VERSION == 2
    result = fov.assignment_results["default"]
    assert len(result.molecules) == fov.filtering_result.counts["accepted"]
    loaded = dataset.fov("FOV_001").load_assignment("default", checkpoints=checkpoints)
    for name in TABLES:
        assert_frame_equal(getattr(loaded, name), getattr(result, name), check_exact=True)
    assert loaded.record == result.record
