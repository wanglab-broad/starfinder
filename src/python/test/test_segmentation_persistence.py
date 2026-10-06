"""The saved format of a segmentation run: ``FOV.segment(checkpoints=…)`` and ``FOV.load_segmentation`` (W-315).

Row L13 of the engineering validation design in docs/assignment-algorithms.md, on an
imported run of the ``seg_golden`` stand-in labels and a ``seeded_watershed`` run on
``seeded`` (option F1 of docs/segmentation-contract.md, "Saved format"). Each FOV is
synthetic: its reference round is set in memory, without ``FOV.run``.
"""
import hashlib
import json

import numpy as np
import pytest
import tifffile

from starfinder.dataset import CheckpointConfig, Dataset, RoundState
from starfinder.image import ImageMetadata
from starfinder.io import load_volume, load_volume_zyxc, save_volume
from starfinder.segmentation import (InputChannel, LabelImportConfig, SeededWatershedConfig, SegmentationPlan,
                                     SegmentationRun)

from .segmentation_fixtures import BOXES_METADATA, NUCLEUS_BOXES, boxes, seeded_stain
from .test_segmentation_golden import digest, fixture, stand_in_model

pytestmark = [pytest.mark.segmentation, pytest.mark.validation]


def synthetic_fov(root, image, metadata, fov_id="FOV_001"):
    """A FOV whose reference round ``round1`` holds ``image`` (ZYXC) with ``metadata``."""
    rounds = RoundState(["round1"], reference_round="round1")
    dataset = Dataset(root, root / "out", "data", "sample", "out", rounds, ["ch00"])
    fov = dataset.fov(fov_id)
    fov.images["round1"] = np.ascontiguousarray(image)
    fov.metadata["round1"] = metadata
    return fov


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_run(name, path, target="nucleus"):
    return SegmentationRun(name, target, (), LabelImportConfig(str(path), target))


@pytest.fixture
def golden(tmp_path):
    """The seg_golden FOV and its stand-in labels written as a TIFF without metadata."""
    dapi = fixture()["dapi"]
    labels = stand_in_model(dapi)
    path = tmp_path / "stand_in.tif"
    tifffile.imwrite(path, labels)
    return synthetic_fov(tmp_path, dapi[..., None], ImageMetadata("seg_golden")), path, labels


@pytest.fixture
def seeded(tmp_path):
    """The seeded FOV (stain of the boxes cells) and the boxes nuclei written with their metadata."""
    _, nuclei = boxes()
    path = tmp_path / "nuclei.tif"
    save_volume(nuclei, path, metadata=BOXES_METADATA)
    plan = SegmentationPlan((import_run("nucleus", path),
                             SegmentationRun("cell", "cell", (InputChannel("amplicon", reference_merged=True),),
                                             SeededWatershedConfig(sigma_um=0.1), seeds="nucleus")))
    return synthetic_fov(tmp_path, seeded_stain()[..., None], BOXES_METADATA), plan, nuclei


def check_reload(fov, name, checkpoints, folder):
    """The run reloads into a fresh FOV equal to the stored result: arrays, record, grid and identity."""
    saved = fov.segmentation_results[name]
    record = json.loads((folder / "segmentation.json").read_text())
    fresh = synthetic_fov(fov.dataset.input_root, fov.images["round1"], fov.metadata["round1"])
    loaded = fresh.load_segmentation(name, checkpoints=checkpoints)
    assert fresh.segmentation_results[name] is loaded
    assert loaded.labels.dtype == np.uint32 and np.array_equal(loaded.labels, saved.labels)
    assert loaded.record == saved.record == record
    assert (loaded.grid, loaded.target, loaded.geometry, loaded.label_namespace) == (
        saved.grid, saved.target, saved.geometry, saved.label_namespace)
    labels = record["labels"]
    assert labels["path"] == "labels.tif" and labels["dtype"] == "uint32"
    assert labels["sha256"] == digest(saved.labels) and labels["file_sha256"] == file_hash(folder / "labels.tif")
    stored = load_volume(folder / "labels.tif")
    assert stored.image.dtype == np.uint32 and stored.metadata == saved.grid.metadata
    return loaded


def test_l13_an_imported_run_writes_its_labels_and_record(golden, tmp_path):
    fov, path, labels = golden
    checkpoints = CheckpointConfig(directory=tmp_path / "checkpoints")
    assert fov.segment(SegmentationPlan((import_run("nucleus", path),)), checkpoints=checkpoints) is fov
    folder = tmp_path / "checkpoints" / "FOV_001" / "segmentation" / "nucleus"
    # An import has no segmentation input, so no input.ome.tif.
    assert sorted(p.name for p in folder.iterdir()) == ["labels.tif", "segmentation.json"]
    assert sorted(p.name for p in (tmp_path / "checkpoints" / "FOV_001").iterdir()) == ["segmentation"]
    loaded = check_reload(fov, "nucleus", checkpoints, folder)
    assert np.array_equal(loaded.labels, labels.astype(np.uint32))
    assert loaded.record["input"] is None and loaded.record["import"]["path"] == str(path)
    assert loaded.record["methods"] == [] and loaded.record["import"]["metadata_source"] == "declared"


def test_l13_a_seeded_watershed_run_writes_its_input(seeded, tmp_path):
    fov, plan, nuclei = seeded
    checkpoints = CheckpointConfig(directory=tmp_path / "checkpoints")
    fov.segment(plan, checkpoints=checkpoints)
    base = tmp_path / "checkpoints" / "FOV_001" / "segmentation"
    assert sorted(p.name for p in (base / "nucleus").iterdir()) == ["labels.tif", "segmentation.json"]
    assert sorted(p.name for p in (base / "cell").iterdir()) == ["input.ome.tif", "labels.tif", "segmentation.json"]
    check_reload(fov, "nucleus", checkpoints, base / "nucleus")
    cell = check_reload(fov, "cell", checkpoints, base / "cell")
    assert set(np.unique(cell.labels)) - {0} == set(NUCLEUS_BOXES)
    assert cell.record["seeds"]["sha256"] == digest(nuclei)
    # input.ome.tif holds the segmentation input the labels were computed from.
    entry = cell.record["input"]
    assert entry["path"] == "input.ome.tif" and entry["file_sha256"] == file_hash(base / "cell" / "input.ome.tif")
    stored = load_volume_zyxc(base / "cell" / "input.ome.tif", channel_labels=("amplicon",))
    assert np.array_equal(stored.image, seeded_stain()[..., None]) and stored.metadata == BOXES_METADATA
    assert digest(stored.image) == entry["sha256"]


def test_l13_a_changed_label_file_raises_naming_the_hash(seeded, tmp_path):
    fov, plan, _ = seeded
    checkpoints = CheckpointConfig(directory=tmp_path / "checkpoints")
    fov.segment(plan, checkpoints=checkpoints)
    folder = tmp_path / "checkpoints" / "FOV_001" / "segmentation" / "cell"
    recorded = fov.segmentation_results["cell"].record["labels"]["file_sha256"]
    changed = fov.segmentation_results["cell"].labels.copy()
    changed[0, 0, 0] += 1
    save_volume(changed, folder / "labels.tif", compress=True, metadata=BOXES_METADATA)
    fresh = synthetic_fov(tmp_path, seeded_stain()[..., None], BOXES_METADATA)
    with pytest.raises(ValueError, match=recorded) as error:
        fresh.load_segmentation("cell", checkpoints=checkpoints)
    assert str(folder / "labels.tif") in str(error.value) and file_hash(folder / "labels.tif") in str(error.value)
    # A changed input file and a removed label file raise too; the nucleus run is unaffected.
    (folder / "labels.tif").unlink()
    with pytest.raises(ValueError, match="missing"):
        fresh.load_segmentation("cell", checkpoints=checkpoints)
    fresh.load_segmentation("nucleus", checkpoints=checkpoints)


def test_saved_runs_are_not_overwritten_by_default(seeded, tmp_path):
    fov, plan, _ = seeded
    checkpoints = CheckpointConfig(directory=tmp_path / "checkpoints")
    fov.segment(plan, checkpoints=checkpoints)
    before = {p: file_hash(p) for p in (tmp_path / "checkpoints").rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError, match="overwrite=True"):
        fov.segment(plan, checkpoints=checkpoints)
    assert {p: file_hash(p) for p in (tmp_path / "checkpoints").rglob("*") if p.is_file()} == before
    fov.segment(plan, checkpoints=CheckpointConfig(directory=tmp_path / "checkpoints", overwrite=True))
    assert sorted(p.name for p in (tmp_path / "checkpoints" / "FOV_001" / "segmentation" / "cell").iterdir()) == [
        "input.ome.tif", "labels.tif", "segmentation.json"]


def test_load_segmentation_errors(seeded, tmp_path):
    fov, plan, _ = seeded
    checkpoints = CheckpointConfig(directory=tmp_path / "checkpoints")
    with pytest.raises(FileNotFoundError, match="no saved segmentation run 'cell'"):
        fov.load_segmentation("cell", checkpoints=checkpoints)
    fov.segment(plan, checkpoints=checkpoints)
    other = synthetic_fov(tmp_path, seeded_stain()[..., None], BOXES_METADATA, fov_id="FOV_002")
    base = tmp_path / "checkpoints"
    (base / "FOV_002").mkdir()
    (base / "FOV_001" / "segmentation").rename(base / "FOV_002" / "segmentation")
    with pytest.raises(ValueError, match="fov_id 'FOV_001'"):
        other.load_segmentation("cell", checkpoints=checkpoints)
    folder = base / "FOV_002" / "segmentation" / "cell"
    (folder / "input.ome.tif").write_bytes(b"changed")
    with pytest.raises(ValueError, match="segmentation input"):
        other.load_segmentation("cell", checkpoints=checkpoints)
