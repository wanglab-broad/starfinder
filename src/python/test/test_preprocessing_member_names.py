"""The member renames of W-558 (W-526, NM-02, NM-03, NM-06, NM-07, NM-10, NM-19, NM-20, NM-23, NM-26, NC-07, SM-03
and SM-23): the old names are gone, with no alias, and the new ones are defined and exported. The look-alikes that
keep their names (the config fields ``normalize_percentiles``, ``SampleScene.sample_labels`` and the method
``FOV.load_checkpoint``) stay."""
import dataclasses

import pytest

import starfinder
from starfinder import assignment, dataset, preprocessing, report, segmentation
from starfinder.dataset import FOV

pytestmark = pytest.mark.contract

PREPROCESSING = {"normalize_intensity": "normalize_min_max", "reconstruct_background": "correct_background",
                 "summary_stage": "summary_recipe", "supplied_section": "fit_supplied_section",
                 "read_histograms": "load_histograms", "write_histograms": "save_histograms",
                 "read_supplied_statistics": "load_supplied_statistics",
                 "write_supplied_statistics": "save_supplied_statistics"}
SEGMENTATION = {"labels_to_grid": "resample_labels", "normalize_percentiles": "normalize_percentile_unclipped"}
ASSIGNMENT = {"molecule_table_from_csv": "import_legacy_molecules_csv", "sample_labels": "labels_at_positions"}
CASES = ([pytest.param(preprocessing, old, new, marks=pytest.mark.preprocessing) for old, new in PREPROCESSING.items()]
         + [pytest.param(starfinder, old, new, marks=pytest.mark.preprocessing)
            for old, new in (("normalize_intensity", "normalize_min_max"),
                             ("reconstruct_background", "correct_background"))]
         + [pytest.param(segmentation, old, new, marks=pytest.mark.segmentation) for old, new in SEGMENTATION.items()]
         + [pytest.param(assignment, old, new, marks=pytest.mark.segmentation) for old, new in ASSIGNMENT.items()]
         + [pytest.param(dataset, "read_checkpoint", "load_checkpoint", marks=pytest.mark.dataset),
            pytest.param(report, "read_report", "load_report", marks=pytest.mark.report)])


@pytest.mark.parametrize(("module", "old", "new"), CASES, ids=lambda v: getattr(v, "__name__", v))
def test_old_member_names_are_gone_and_the_new_ones_exported(module, old, new):
    assert not hasattr(module, old) and old not in module.__all__
    assert hasattr(module, new) and new in module.__all__
    assert getattr(module, new).__name__ == new


@pytest.mark.preprocessing
def test_root_exports_are_the_preprocessing_functions():
    assert starfinder.normalize_min_max is preprocessing.normalize_min_max
    assert starfinder.correct_background is preprocessing.correct_background


@pytest.mark.dataset
@pytest.mark.parametrize(("old", "new"), [("normalize_intensity", "normalize_min_max"),
                                          ("reconstruct_background", "correct_background")])
def test_fov_methods_are_renamed(old, new):
    assert not hasattr(FOV, old)
    assert callable(getattr(FOV, new)) and getattr(FOV, new).__name__ == new


@pytest.mark.dataset
def test_fov_load_checkpoint_stays_a_method_beside_the_module_function():
    # dataset.load_checkpoint returns a dict; the method restores state onto a FOV and keeps its name.
    assert callable(FOV.load_checkpoint) and FOV.load_checkpoint is not dataset.load_checkpoint
    assert not hasattr(FOV, "read_checkpoint")


@pytest.mark.segmentation
@pytest.mark.parametrize("config", [segmentation.StarDistConfig, segmentation.CellposeConfig])
def test_the_normalize_percentiles_config_fields_stay(config):
    assert "normalize_percentiles" in {field.name for field in dataclasses.fields(config)}
    assert not hasattr(segmentation, "normalize_percentiles")


@pytest.mark.synthetic
def test_the_sample_scene_field_sample_labels_stays():
    from starfinder.synthetic import SampleScene
    assert "sample_labels" in {field.name for field in dataclasses.fields(SampleScene)}
    assert not hasattr(assignment, "sample_labels")
