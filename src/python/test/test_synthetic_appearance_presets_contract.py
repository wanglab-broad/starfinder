"""Contract checks of the appearance presets (docs/synthetic-real-appearance-validation.md, GA2's presets).

PR1 of docs/synthetic-real-appearance.md ("Presets"): six appearance presets, version ``appearance-presets-v1``, beside
the six accepted context presets, which stay unchanged; each takes the grid, spacing, jitter, overlap, seed, cell
types and molecules of the accepted preset of its context and tier; the routine presets name the closed-form bank of
their context (CF1, DN3) and the ``slow`` presets a real bank by name and SHA-256 (ST1, VA7, DN1). The names are
exported and on the API pages. A test that needs a real bank skips with a reason naming STARFINDER_APPEARANCE_BANKS or
the missing file, checked here on directories made in the test. Nothing is read from a run directory or a real image.
"""
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import re

import pytest

import starfinder.synthetic as synthetic
from starfinder.synthetic import (APPEARANCE_PRESET_VERSION, APPEARANCE_PRESETS, APPEARANCE_TRUTH_VERSION,
                                  CELLULAR_PRESETS, AppearanceSceneConfig, AugmentationConfig, CompositionConfig,
                                  LayoutConfig, appearance_scene_preset, cellular_scene_preset)
from starfinder.synthetic._appearance_bank import BANK_DIRECTORY_VARIABLE
from starfinder.synthetic._formed import _plain

from .cellular_appearance_scenes import ROUTINE_PRESETS, SLOW_BANKS, SLOW_PRESETS, real_bank_missing

pytestmark = [pytest.mark.synthetic, pytest.mark.contract]

DOCS = Path(__file__).resolve().parents[3] / "docs"
NAMES = ("appearance_scene_preset", "APPEARANCE_PRESETS", "APPEARANCE_PRESET_VERSION")
# The accepted preset of each appearance preset's context and tier (PR1).
ACCEPTED = {"appearance-dense3d-v1": "context-dense3d-v1", "appearance-tissue2d-v1": "context-tissue2d-v1",
            "appearance-culture-v1": "context-culture-v1", "appearance-dense3d-slow-v1": "context-dense3d-slow-v1",
            "appearance-tissue2d-slow-v1": "context-tissue2d-slow-v1",
            "appearance-culture-slow-v1": "context-culture-slow-v1"}
# SHA-256 of CELLULAR_PRESETS (json.dumps, sorted keys) at the batch's base revision 8bf9a5b, before any change.
CONTEXT_PRESETS_SHA256 = "0921de9aa6a45b8cc9454d643eb30fdbf605614a014118c9ea61209dea3a8edc"
GRID = ("columns", "rows", "fov_shape_zyx", "spacing_zyx", "overlap", "jitter_max_zyx")


def close(a, b):
    """Equal values, with floats (also inside tuples) within 1e-9."""
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return a == pytest.approx(b, abs=1e-9)
    return a == b


def test_six_presets_with_their_version():
    assert APPEARANCE_PRESET_VERSION == "appearance-presets-v1" and APPEARANCE_TRUTH_VERSION == "1"
    assert list(APPEARANCE_PRESETS) == [*ROUTINE_PRESETS, *SLOW_PRESETS]
    for name, entry in APPEARANCE_PRESETS.items():
        assert entry["version"] == APPEARANCE_PRESET_VERSION and entry["scene_key"] == name
        assert entry["accepted_preset"] == ACCEPTED[name]
        assert entry["tier"] == ("slow" if name in SLOW_PRESETS else "routine")
        _, config = appearance_scene_preset(name)
        assert isinstance(config, AppearanceSceneConfig)
        assert config.context.sample.fov.scene_key == name and config.context.sample.fov.dataset_version == name
    with pytest.raises(ValueError, match="unknown appearance preset"):
        appearance_scene_preset("appearance-dense3d-v2")


@pytest.mark.parametrize("name", [*ROUTINE_PRESETS, *SLOW_PRESETS])
def test_each_preset_takes_the_accepted_preset_of_its_context_and_tier(name):
    book, config = appearance_scene_preset(name)
    accepted_book, accepted = cellular_scene_preset(ACCEPTED[name])
    assert book.genes == accepted_book.genes and book.round_labels == accepted_book.round_labels
    context = config.context
    a, b = context.sample.acquisition, accepted.sample.acquisition
    for field in fields(a):
        assert close(getattr(a, field.name), getattr(b, field.name)), field.name
    entry, registry = APPEARANCE_PRESETS[name], CELLULAR_PRESETS[ACCEPTED[name]]
    for key in GRID:
        assert close(entry[key], registry[key]), key
        assert close(getattr(a, key), registry[key]), key
    assert context.sample.fov.seed == accepted.sample.fov.seed == entry["seed"] == 42
    assert context.sample.fov.split == accepted.sample.fov.split
    assert context.types == accepted.types and context.molecules == accepted.molecules
    assert context.sample.cells == accepted.sample.cells
    assert context.territories == accepted.territories and context.morphology == accepted.morphology
    assert context.stains == accepted.stains
    # Only the dataset version and the scene key are the preset's own.
    mine, theirs = _plain(asdict(context.sample.fov)), _plain(asdict(accepted.sample.fov))
    assert {k for k in mine if mine[k] != theirs[k]} == {"dataset_version", "scene_key"}
    assert config.augmentation == AugmentationConfig() and config.composition == CompositionConfig()


@pytest.mark.parametrize("name", ROUTINE_PRESETS)
def test_routine_presets_place_a_count_on_their_closed_form_bank(name):
    _, config = appearance_scene_preset(name)
    entry = APPEARANCE_PRESETS[name]
    assert config.bank.preset == ACCEPTED[name] == entry["bank_preset"] and config.bank.n == entry["bank_n"] == 4
    assert config.bank.name is None and config.bank.sha256 is None and entry["bank_sha256"] is None
    assert config.layout == LayoutConfig(density="count", count=entry["count"]) and entry["max_refusals"] == 4000
    assert entry["density"] == "count" and entry["count"] >= 1


@pytest.mark.parametrize("name", SLOW_PRESETS)
def test_slow_presets_name_their_real_bank_by_name_and_sha256(name):
    _, config = appearance_scene_preset(name)
    entry = APPEARANCE_PRESETS[name]
    assert config.bank.name == SLOW_BANKS[name] == entry["bank"] and config.bank.preset is None
    assert re.fullmatch(r"[0-9a-f]{64}", config.bank.sha256) and config.bank.sha256 == entry["bank_sha256"]
    assert config.bank.directory is None
    assert config.layout.density == "visible" == entry["density"]
    assert 0 < config.layout.target < 1 and config.layout.threshold > 0
    assert (config.layout.target, config.layout.threshold, config.layout.check_every) == (
        entry["target"], entry["threshold"], entry["check_every"])
    for key in ("contact", "max_proposals", "floor"):
        assert getattr(config.layout, key) == getattr(LayoutConfig(), key), key
    # The culture stops after 500 refusals in a row (the amendment of 2026-10-17 of "Presets"), the others after
    # the prototype's 4,000.
    assert config.layout.max_refusals == entry["max_refusals"] == (500 if name == "appearance-culture-slow-v1"
                                                                    else 4000)
    assert {SLOW_BANKS[p] for p in SLOW_PRESETS} == {"ln-sample1-v1", "tissue2d-v1", "culture3d-v1"}


def test_the_accepted_presets_are_unchanged():
    assert len(CELLULAR_PRESETS) == 6 and set(CELLULAR_PRESETS) == set(ACCEPTED.values())
    digest = hashlib.sha256(json.dumps(CELLULAR_PRESETS, sort_keys=True).encode()).hexdigest()
    assert digest == CONTEXT_PRESETS_SHA256


def test_preset_names_are_exported_and_on_the_api_pages():
    pages = {page: (DOCS / "api" / page).read_text(encoding="utf-8")
             for page in ("synthetic.rst",)}
    for name in NAMES:
        assert name in synthetic.__all__ and hasattr(synthetic, name), name
        assert re.search(rf"^   {name}$", pages["synthetic.rst"], re.M), name


def test_a_real_bank_test_skips_naming_the_variable_or_the_file(tmp_path, monkeypatch):
    monkeypatch.delenv(BANK_DIRECTORY_VARIABLE, raising=False)
    for bank in SLOW_BANKS.values():
        reason = real_bank_missing(bank)
        assert reason is not None and BANK_DIRECTORY_VARIABLE in reason and bank in reason
    monkeypatch.setenv(BANK_DIRECTORY_VARIABLE, str(tmp_path))
    reason = real_bank_missing("ln-sample1-v1")
    assert reason is not None and str(tmp_path / "ln-sample1-v1.npz") in reason
    for suffix in (".npz", ".record.json"):
        (tmp_path / f"ln-sample1-v1{suffix}").write_text("{}")
    reason = real_bank_missing("ln-sample1-v1")
    assert reason is not None and str(tmp_path / "ln-sample1-v1.config.json") in reason
    (tmp_path / "ln-sample1-v1.config.json").write_text("{}")
    assert real_bank_missing("ln-sample1-v1") is None
