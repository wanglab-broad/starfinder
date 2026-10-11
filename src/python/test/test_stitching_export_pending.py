"""W-397 criterion 5: an export that stops during a part leaves a record that is not complete.

Before its first part, ``Dataset.export`` records each part the configuration will run as
``not_run`` with the reason ``export_not_finished`` (and each part that is off as ``off``); each
part replaces that entry with its own section when it ends. Fixture: the synthetic sample scene
``sample-tiny-v1`` written as a dataset with stored metadata
(``test_stitching_declared_grids.scene_dataset``), exported with assemble and fuse on. The
interruption is a ``KeyboardInterrupt`` raised where the export does not catch it: by the fusion
part as the export calls it, before the part's own handler runs, which is what a process stopped
by a signal leaves (W-391, finding D). dask runs with the synchronous scheduler.
"""
import importlib.util
import json

import pytest

from starfinder.sample_output import ExportConfig, FusionConfig, _export, _tiff, check_export
from starfinder.sample_output._record import PARTS, read_export

from .test_stitching_declared_grids import SAMPLE, scene_dataset

pytestmark = [pytest.mark.stitching, pytest.mark.validation]

needs_parts = pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None
                                 or importlib.util.find_spec("anndata") is None,
                                 reason="the multiview-stitcher extra (starfinder[multiview-stitcher]) or the anndata "
                                        "extra (starfinder[anndata]) is not installed")
EXPORT = ExportConfig(route="per_fov", fusion=FusionConfig(images=("reference_merged",)))
TIMES = ("wall_s", "cpu_s", "peak_rss_kib")
PENDING = {"status": "not_run", "reason": "export_not_finished"}


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    from starfinder.synthetic import generate_sample_scene, sample_scene_preset, save_sample_scene
    book, config = sample_scene_preset("sample-tiny-v1")
    generated = generate_sample_scene(book, config=config)
    root = tmp_path_factory.mktemp("w397_pending")
    saved = save_sample_scene(generated, root / "scene")
    return scene_dataset(root / "data", generated, book, saved, stored=True)


@pytest.fixture
def synchronous():
    import dask
    with dask.config.set(scheduler="synchronous"):
        yield


def without_times(value):
    """The record without the recorded times (wall and CPU seconds, peak RSS), keys in their order."""
    if isinstance(value, dict):
        return {k: without_times(v) for k, v in value.items() if k not in TIMES}
    if isinstance(value, list):
        return [without_times(v) for v in value]
    return value


def export(dataset, folder, config=EXPORT):
    return dataset.export(SAMPLE, placement=dataset.output_root / "stitching" / SAMPLE, config=config,
                          output_dir=folder)


def interrupt(*args, **kwargs):
    raise KeyboardInterrupt


def listing(folder):
    """Every file of the folder with its SHA-256, export.json aside (it holds the times; compared without them)."""
    from starfinder.sample_output._record import EXPORT_FILE, directory_files
    return {name: digest for name, digest in directory_files(folder).items() if name != EXPORT_FILE}


@needs_parts
def test_an_interrupted_export_is_not_complete_and_the_next_run_completes_it(scene, tmp_path, monkeypatch,
                                                                            synchronous):
    reference, folder = tmp_path / "reference", tmp_path / "export"
    export(scene, reference)
    with monkeypatch.context() as patch:
        patch.setattr(_export, "fuse", interrupt)
        with pytest.raises(KeyboardInterrupt):
            export(scene, folder)
    record = read_export(folder)
    assert record["complete"] is False
    assert record["parts"] == {"assemble": "written", "fuse": "not_run", "segment": "off", "fuse_labels": "off",
                               "export_spatialdata": "off"}
    assert record["fusion"] == PENDING
    assert without_times(record["assembly"]) == without_times(read_export(reference)["assembly"])
    with pytest.raises(ValueError, match=r"^check_export R1: part fuse was not finished: the export stopped before "
                                         r"the part wrote its section \(status 'not_run', reason "
                                         r"'export_not_finished'\)$"):
        check_export(folder)
    # The same export again, without the interruption: the record of an export that was never interrupted.
    finished = export(scene, folder).record
    assert finished["complete"] is True
    expected = read_export(reference)
    assert json.dumps(without_times(read_export(folder))) == json.dumps(without_times(expected))
    assert json.dumps(without_times(finished)) == json.dumps(without_times(expected))
    assert listing(folder) == listing(reference)
    check_export(folder)


@needs_parts
def test_an_interruption_inside_the_fusion_records_it_failed_as_before(scene, tmp_path, monkeypatch, synchronous):
    """Stopped inside the library's computation, the fusion's own handler records the part failed (unchanged): the
    record is not complete either."""
    folder = tmp_path / "export"
    with monkeypatch.context() as patch:
        patch.setattr(_tiff, "write_ome_tiff", interrupt)
        with pytest.raises(KeyboardInterrupt):
            export(scene, folder)
    record = read_export(folder)
    assert record["complete"] is False and record["parts"]["fuse"] == "failed"
    assert record["fusion"]["error"]["type"] == "KeyboardInterrupt"


@needs_parts
def test_only_the_parts_that_will_run_are_pending(scene, tmp_path, monkeypatch, synchronous):
    """Stopped in its first part, the record holds the entry for each part that will run and off for the others;
    a part that is off is recorded off at the end, as before."""
    import starfinder.assembly as assembly
    config = ExportConfig(route="per_fov", fuse_labels=True, fusion=FusionConfig(images=("reference_merged",)))
    folder = tmp_path / "labels"
    with monkeypatch.context() as patch:
        patch.setattr(assembly, "save_assembly", interrupt)
        with pytest.raises(KeyboardInterrupt):
            export(scene, folder, config)
    record = read_export(folder)
    assert record["complete"] is False
    assert record["parts"] == {"assemble": "not_run", "fuse": "not_run", "segment": "off", "fuse_labels": "not_run",
                               "export_spatialdata": "off"}
    assert [record[name] for name in ("assembly", "fusion", "labels")] == [PENDING] * 3
    assert record["spatialdata"] == {"status": "off"}
    with pytest.raises(ValueError, match="part assemble was not finished"):
        check_export(folder)
    off = export(scene, tmp_path / "off", ExportConfig(route="per_fov", fuse=False)).record
    assert off["complete"] is True and off["parts"] == {"assemble": "written", "fuse": "off", "segment": "off",
                                                        "fuse_labels": "off", "export_spatialdata": "off"}
    assert [off[name] for name in ("fusion", "labels", "spatialdata")] == [{"status": "off"}] * 3
    assert [k for k in off if k in ("assembly", "fusion", "labels", "spatialdata")] == [
        "assembly", "fusion", "labels", "spatialdata"]
    assert set(off["parts"]) == set(PARTS)
    check_export(tmp_path / "off")
