"""Scaled candidate controls, without generating a full-size calibration scene."""
from dataclasses import fields, replace

import numpy as np
import pytest

from starfinder.synthetic import (
    UNCALIBRATED_PREPROCESSING_CONDITIONS, UNCALIBRATED_PREPROCESSING_VERSION,
    calibrated_preprocessing_scene_config, generate_amplicon_scene,
    uncalibrated_preprocessing_scene_config,
)

pytestmark = [pytest.mark.synthetic, pytest.mark.contract]


@pytest.mark.parametrize("condition", UNCALIBRATED_PREPROCESSING_CONDITIONS)
def test_scaled_candidate_preserves_appearance_outside_authorized_fields(condition):
    base = "clean" if condition == "dense" else condition
    book, small = calibrated_preprocessing_scene_config(base, dtype="uint8", seed=7, k=2.0)
    scaled_book, large = uncalibrated_preprocessing_scene_config(condition, seed=7, k=2.0)
    assert scaled_book.round_labels == book.round_labels
    assert scaled_book.channel_labels == book.channel_labels
    assert large.shape_zyx == (30, 512, 512)
    assert large.count == (12000 if condition == "dense" else 5000)
    assert large.max_count >= large.count
    assert large.dataset_version == f"{UNCALIBRATED_PREPROCESSING_VERSION}-{condition}"
    assert large.scene_key == "scaled-calibrated-preprocessing-candidate"
    changed = {"shape_zyx", "count", "max_count", "background", "dataset_version", "scene_key"}
    for field in fields(small):
        if field.name not in changed:
            assert getattr(large, field.name) == getattr(small, field.name), field.name
    bg_changes = {"region_centers", "region_sigma_zyx", "texture"}
    for field in fields(small.background):
        if field.name not in bg_changes:
            assert getattr(large.background, field.name) == getattr(small.background, field.name), field.name
    texture = large.background.texture
    assert texture.count == (960 if condition == "bright_outliers" else 1920)
    assert texture.max_count >= texture.count
    for field in fields(texture):
        if field.name not in {"count", "max_count"}:
            assert getattr(texture, field.name) == getattr(small.background.texture, field.name), field.name
    np.testing.assert_allclose(large.background.region_sigma_zyx, [[15, 80, 80]] * 3, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(large.background.region_centers) / [29, 511, 511],
                               np.asarray(small.background.region_centers) / [7, 63, 63], rtol=0, atol=1e-12)
    # Reconfiguration has not mutated the historical small preset or its registry.
    assert small.shape_zyx == (8, 64, 64) and small.count == 80
    assert UNCALIBRATED_PREPROCESSING_CONDITIONS == (
        "clean", "combined", "baseline", "gradient", "regions", "texture", "gain", "trend",
        "gain_strong", "bright_outliers", "saturation", "clean_unbalanced", "dense",
    )


def test_scaled_candidate_rejects_unknown_condition_and_requires_saturation_multiplier():
    with pytest.raises(ValueError, match="unknown uncalibrated preprocessing condition"):
        uncalibrated_preprocessing_scene_config("unknown")
    with pytest.raises(ValueError, match="saturation needs"):
        uncalibrated_preprocessing_scene_config("saturation")


def test_scaled_candidate_generates_on_a_bounded_test_grid():
    book, config = uncalibrated_preprocessing_scene_config("combined", seed=0)
    bg = config.background
    config = replace(config, shape_zyx=(8, 32, 32), count=8,
                     background=replace(bg, texture=replace(bg.texture, count=2)))
    result = generate_amplicon_scene(book, config=config)
    assert len(result.formed) == 8
    assert len(result.rounds) == 4
    for image in result.rounds.values():
        assert image.shape == (8, 32, 32, 4)
        assert image.dtype == np.uint8
        assert np.isfinite(image).all()
