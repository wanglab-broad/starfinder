"""The synthetic renames of W-559 (W-526, NC-06, NM-01, NM-15, NM-22 and SM-20): formed → amplicon and context →
cellular. The old names are gone from ``starfinder.synthetic``, with no alias, and the new ones are defined and
exported. ``CellularSceneConfig`` changed meaning: it names the former ``ContextSceneConfig``, and the sample scene's
cell configuration is ``CellsConfig``."""
import dataclasses

import pytest

import starfinder
from starfinder import synthetic

pytestmark = [pytest.mark.synthetic, pytest.mark.contract]

RENAMES = {"FormedSceneConfig": "AmpliconSceneConfig", "FormedScene": "AmpliconScene",
           "generate_formed_scene": "generate_amplicon_scene", "formed_scene_preset": "amplicon_scene_preset",
           "save_formed_scene": "save_amplicon_scene", "ContextScene": "CellularScene",
           "generate_context_scene": "generate_cellular_scene", "save_context_scene": "save_cellular_scene",
           "save_context_dataset": "save_cellular_dataset", "context_scene_preset": "cellular_scene_preset",
           "CONTEXT_PRESETS": "CELLULAR_PRESETS", "CONTEXT_PRESET_VERSION": "CELLULAR_PRESET_VERSION",
           "CONTEXT_TRUTH_VERSION": "CELLULAR_TRUTH_VERSION", "generate_dataset": "generate_amplicon_dataset",
           "closed_form_bank": "generate_closed_form_bank"}
CLASSES = ("AmpliconSceneConfig", "AmpliconScene", "CellularScene", "CellularSceneConfig", "CellsConfig")


@pytest.mark.parametrize(("old", "new"), list(RENAMES.items()))
def test_old_synthetic_names_are_gone_and_the_new_ones_exported(old, new):
    assert not hasattr(synthetic, old) and old not in synthetic.__all__
    assert not hasattr(starfinder, old)
    assert hasattr(synthetic, new) and new in synthetic.__all__


def test_the_context_scene_config_name_is_gone():
    # ContextSceneConfig is renamed to CellularSceneConfig, whose former class is now CellsConfig.
    assert not hasattr(synthetic, "ContextSceneConfig") and "ContextSceneConfig" not in synthetic.__all__
    assert "CellsConfig" in synthetic.__all__ and "CellularSceneConfig" in synthetic.__all__


@pytest.mark.parametrize("name", CLASSES + ("save_amplicon_dataset",) + tuple(
    new for new in RENAMES.values() if not new.isupper() and new not in CLASSES))
def test_renamed_classes_and_functions_carry_their_new_names(name):
    assert getattr(synthetic, name).__name__ == name


def test_cellular_scene_config_names_the_former_context_scene_config():
    # The cellular scene's configuration (it holds the sample scene and the cellular settings) ...
    _, config = synthetic.cellular_scene_preset("context-dense3d-v1")
    assert type(config) is synthetic.CellularSceneConfig
    assert {"sample", "nuclei", "packing", "territories"} <= {f.name for f in dataclasses.fields(config)}
    # ... and the sample scene's cell configuration is CellsConfig.
    assert type(synthetic.SampleSceneConfig().cells) is synthetic.CellsConfig
    assert type(config.sample.cells) is synthetic.CellsConfig
    assert synthetic.CellsConfig is not synthetic.CellularSceneConfig


def test_amplicon_names_are_the_formed_scene_classes():
    _, config = synthetic.benchmark_scene_preset("tiny")
    assert type(config) is synthetic.AmpliconSceneConfig
    assert type(synthetic.amplicon_scene_preset()[1]) is synthetic.AmpliconSceneConfig
    assert synthetic.SampleSceneConfig.__dataclass_fields__["fov"].type in ("AmpliconSceneConfig",
                                                                             synthetic.AmpliconSceneConfig)


def test_the_kept_fields_and_values_stay():
    # The formed-amplicon table and the appearance scene's context keep their names (written as formed.csv and the
    # context key of appearance.json); the preset version and IDs are format identifiers.
    assert "formed" in {f.name for f in dataclasses.fields(synthetic.AmpliconScene)}
    assert "formed" in {f.name for f in dataclasses.fields(synthetic.SyntheticDataset)}
    assert "context" in {f.name for f in dataclasses.fields(synthetic.AppearanceSceneConfig)}
    assert "context" in {f.name for f in dataclasses.fields(synthetic.AppearanceScene)}
    assert synthetic.CELLULAR_PRESET_VERSION == "context-presets-v1" and synthetic.CELLULAR_TRUTH_VERSION == "1"
    assert "context-dense3d-v1" in synthetic.CELLULAR_PRESETS
