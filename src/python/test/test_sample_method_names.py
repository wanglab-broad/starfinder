"""The sample-level and FOV/Dataset renames of W-560 (W-526, SM-17, NM-04, NM-05, NC-03, NM-08, NM-09, NM-21, NC-04
and NC-05): the old names are gone, with no alias, and the new ones are defined and exported. The look-alikes that keep
their names (the export part ``fuse``, ``Dataset.channel_info`` and ``Dataset.channel_labels``) stay, and
``FOV.register_rounds`` is merged into ``FOV.register``."""
import dataclasses
import inspect

import pytest

import starfinder
from starfinder import assembly, export, stitching
from starfinder.dataset import FOV, Dataset

pytestmark = pytest.mark.contract

FUNCTIONS = [pytest.param(assembly, "sample_anndata", "write_sample_h5ad", marks=pytest.mark.stitching),
             pytest.param(stitching, "bigstitcher_macro", "export_bigstitcher", marks=pytest.mark.stitching),
             pytest.param(export, "fuse", "fuse_images", marks=pytest.mark.stitching)]
METHODS = {"assign": "assign_molecules", "save_spots": "export_spots", "create_subtiles": "save_subtiles"}


@pytest.mark.parametrize(("module", "old", "new"), FUNCTIONS, ids=lambda v: getattr(v, "__name__", v))
def test_old_function_names_are_gone_and_the_new_ones_exported(module, old, new):
    assert not hasattr(module, old) and old not in module.__all__
    assert hasattr(module, new) and new in module.__all__
    assert getattr(module, new).__name__ == new
    assert not hasattr(starfinder, old) and old not in starfinder.__all__


@pytest.mark.dataset
@pytest.mark.parametrize(("old", "new"), METHODS.items())
def test_fov_methods_are_renamed(old, new):
    assert not hasattr(FOV, old)
    assert callable(getattr(FOV, new)) and getattr(FOV, new).__name__ == new


@pytest.mark.dataset
def test_register_rounds_is_merged_into_register():
    assert not hasattr(FOV, "register_rounds")
    # The decorated step keeps the name register, from which run.json and the log take the step name.
    assert FOV.register.__name__ == "register"
    parameters = inspect.signature(FOV.register).parameters
    assert list(parameters) == ["self", "recipe", "rounds", "source", "reference", "checkpoints"]
    assert all(parameters[name].kind is inspect.Parameter.KEYWORD_ONLY and parameters[name].default is None
               for name in ("rounds", "source", "reference", "checkpoints"))


@pytest.mark.dataset
def test_channel_record_is_private_and_channel_info_and_labels_stay():
    assert not hasattr(Dataset, "channel_record") and callable(Dataset._channel_record)
    assert callable(Dataset.channel_info) and callable(Dataset.channel_labels)


@pytest.mark.stitching
def test_the_export_part_fuse_keeps_its_name():
    assert "fuse" in {field.name for field in dataclasses.fields(export.ExportConfig)}
