"""The calibrated preprocessing conditions of starfinder.synthetic (W-239's item 5, C1, C4, C6 and C8).

``calibrated_preprocessing_scene_config`` returns the calibrated preset with only the named fields changed. These
checks moved here from the retired benchmarks/preprocessing_synthetic.py tests (W-781; design/baselines/benchmark.md,
"Test disposition rules") and keep their fixture numbers; they call the public configurations directly.
"""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from starfinder.synthetic import (CALIBRATED_CONDITIONS, calibrated_preprocessing_scene_config,
                                  calibrated_scene_preset, development_codebook, generate_amplicon_scene)

pytestmark = [pytest.mark.synthetic, pytest.mark.preprocessing]


def clipped_fraction(counts, rounds):
    """Item 5: the generator's ``above`` clipping counts summed over rounds, over rounds x channels x voxels."""
    return sum(int(c["above"]) for c in counts.values()) / sum(int(rounds[r].size) for r in counts)


def test_calibrated_conditions_are_the_preset_as_returned_with_the_balanced_codebook():
    balanced = development_codebook("balanced").table
    for condition in CALIBRATED_CONDITIONS:
        for dtype in ("uint8", "uint16"):
            book, config = calibrated_preprocessing_scene_config(condition, dtype=dtype, seed=101)
            _, preset = calibrated_scene_preset(condition, dtype, seed=101, codebook="balanced")
            assert config == preset
            pd.testing.assert_frame_equal(book.table, balanced)
    book, config = calibrated_preprocessing_scene_config("clean_unbalanced", dtype="uint8", seed=3)
    _, preset = calibrated_scene_preset("clean", "uint8", seed=3, codebook="unbalanced")
    pd.testing.assert_frame_equal(book.table, development_codebook("unbalanced").table)
    assert replace(config, dataset_version=preset.dataset_version) == preset
    assert config.dataset_version == "calibrated-development-v1-clean_unbalanced-unbalanced"


@pytest.mark.parametrize("dtype, median", [("uint8", 352.0), ("uint16", 352.0 * 16)])
def test_bright_outliers_are_four_blobs_at_four_times_the_brightness_median(dtype, median):
    book, config = calibrated_preprocessing_scene_config("bright_outliers", dtype=dtype, seed=0)
    _, clean = calibrated_scene_preset("clean", dtype, seed=0)
    texture = config.background.texture
    assert config.background.texture_enabled and texture.count == 4
    assert (texture.axial_width.mode, texture.axial_width.parameters) == ("constant", (1.5,))
    assert (texture.lateral_width.mode, texture.lateral_width.parameters) == ("constant", (2.0,))
    assert texture.brightness.mode == "lognormal" and texture.brightness.parameters[1] == 0.1
    assert np.exp(texture.brightness.parameters[0]) == pytest.approx(median, rel=1e-12)
    assert np.exp(clean.brightness.parameters[0]) * 4 == pytest.approx(median, rel=1e-12)
    assert np.array_equal(config.background.tissue_weights, np.ones((4, 4)))
    unchanged = replace(config, dataset_version=clean.dataset_version, background=clean.background)
    assert unchanged == clean


def test_saturation_scales_the_uint16_intensity_parameters_by_k():
    _, clean = calibrated_scene_preset("clean", "uint8", seed=1)
    _, one = calibrated_preprocessing_scene_config("saturation", dtype="uint8", seed=1, k=1.0)
    assert replace(one, dataset_version=clean.dataset_version) == clean
    k = 2.0 ** (5 / 4)
    _, config = calibrated_preprocessing_scene_config("saturation", dtype="uint8", seed=1, k=k)
    assert np.exp(config.brightness.parameters[0]) == pytest.approx(88.0 * k, rel=1e-12)
    assert config.brightness.parameters[1] == clean.brightness.parameters[1]
    np.testing.assert_allclose(config.background.baseline, np.asarray(clean.background.baseline) * k)
    for field in ("alpha", "sigma", "correlated_sigma"):
        assert getattr(config.noise, field) == pytest.approx(getattr(clean.noise, field) * k)
    rest = replace(config, dataset_version=clean.dataset_version, brightness=clean.brightness,
                   background=clean.background, noise=clean.noise)
    assert rest == clean
    with pytest.raises(ValueError, match="searched k"):
        calibrated_preprocessing_scene_config("saturation", dtype="uint8", seed=1)


def test_clipped_fraction_counts_voxels_above_the_dtype_range():
    # The retired test's fixture: 3 + 5 voxels above the range over 2 rounds of 2 x 4 x 4 x 4.
    rounds = {"round1": np.zeros((2, 4, 4, 4)), "round2": np.zeros((2, 4, 4, 4))}
    counts = {"round1": dict(below=7, above=3), "round2": dict(below=0, above=5)}
    assert clipped_fraction(counts, rounds) == 8 / 256
    # A bounded saturation scene: each round's `above` count is the number of voxels of the same scene generated in
    # float64 whose rounded value exceeds the uint8 maximum.
    book, config = calibrated_preprocessing_scene_config("saturation", dtype="uint8", seed=1, k=2.0 ** (5 / 4),
                                                         shape=(8, 32, 32), count=12)
    with pytest.warns(RuntimeWarning, match="clipped to the uint8 range"):
        scene = generate_amplicon_scene(book, config=config)
    unclipped = generate_amplicon_scene(book, config=replace(config, dtype="float64"))
    counts = scene.provenance["clipping_counts"]
    assert set(counts) == set(book.round_labels)
    above = {r: int(np.count_nonzero(np.rint(unclipped.rounds[r]) > np.iinfo(np.uint8).max)) for r in counts}
    assert {r: int(c["above"]) for r, c in counts.items()} == above
    assert sum(above.values()) > 0
    assert clipped_fraction(counts, scene.rounds) == sum(above.values()) / (4 * 8 * 32 * 32 * 4)


def test_gain_strong_has_the_ln_channel_spread_and_keeps_the_geometric_mean():
    _, clean = calibrated_scene_preset("clean", "uint8", seed=0)
    _, config = calibrated_preprocessing_scene_config("gain_strong", dtype="uint8", seed=0)
    gains, base = np.asarray(config.readout.gains), np.asarray(clean.readout.gains)
    for row, clean_row in zip(gains, base):
        assert row.max() / row.min() == pytest.approx(3.15, rel=1e-12)
        assert np.all(np.diff(row) < 0)
        assert np.exp(np.mean(np.log(row))) == pytest.approx(np.exp(np.mean(np.log(clean_row))), rel=1e-12)
    assert replace(config, dataset_version=clean.dataset_version, readout=clean.readout) == clean


def test_multi_fov_fovs_replace_count_gains_and_scene_key():
    _, clean = calibrated_scene_preset("combined", "uint8", seed=100)
    _, config = calibrated_preprocessing_scene_config("combined", dtype="uint8", seed=100, fov_id="Position003",
                                                      gain=0.5, count=2, scene_key="k/Position003")
    assert (config.count, config.FOV_id, config.scene_key) == (2, "Position003", "k/Position003")
    assert config.coordinates is None and config.amplicon_ids is None and config.gene_ids is None
    np.testing.assert_allclose(config.readout.gains, np.asarray(clean.readout.gains) * 0.5)
