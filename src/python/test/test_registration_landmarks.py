"""The noise landmarks of TPS and CPD and their record in RegistrationDiagnostics (W-836).

Landmarks take local maxima's noise threshold: the median and MAD over the voxels above the minimum when more than half
sit at it (W-789), the one-grey-level MAD floor on an integer image (W-848), and the refusal of a floating-point channel
with signal and a MAD of 0. The fixtures are 16x32x32 (16384 voxels) with 27 isolated peaks on a 3x3x3 grid, which spans
3D for the TPS fit. Each registration below estimates an image against itself, so every landmark matches itself.
"""
import warnings

import numpy as np
import pytest
from scipy.spatial import cKDTree

from starfinder._records import json_record
from starfinder.image import ImageMetadata
from starfinder.registration import (CpdConfig, RegistrationEstimationError, TpsConfig, TranslationConfig,
                                     estimate_transform)
from starfinder.registration._cpd import _subsample_with_neighbors
from starfinder.spot_finding import LocalMaximaConfig, SpotFindingWarning, find_spots

from .expected_values import FLOAT64

pytestmark = pytest.mark.registration

SHAPE = (16, 32, 32)
VOXELS = 16384
META = ImageMetadata("landmarks")
GRID = [(z, y, x) for z in (4, 8, 12) for y in (6, 14, 22) for x in (6, 14, 22)]
# 64 copies of each grey level 33..48 (median 40.5, MAD 4 over them alone).
NOISE = np.repeat(np.arange(33, 49), 64)
TPS = TpsConfig(min_matches=4, max_control_points=20)
CPD = CpdConfig(max_control_points=20)
NOISE_KEYS = ["threshold", "median", "mad", "mad_floored", "estimation", "n_estimation"]
# The keys of a saved RegistrationDiagnostics before W-836 (support_mask is left out while None).
BASE_KEYS = ["method", "backend", "effective_config", "converged", "iterations_completed", "reference_landmark_count",
             "moving_landmark_count", "matched_landmark_count", "warnings", "final_metric_value", "stop_condition",
             "elapsed_iterations", "final_rms_change", "backend_versions", "backend_parameters", "spacing_source"]


def truncated(values=NOISE, peak=200, dtype=np.uint8):
    """values at seeded random voxels off the grid, 0 elsewhere, and a peak on each of the 27 grid voxels."""
    flat = np.zeros(VOXELS, dtype=np.float64)
    grid = np.ravel_multi_index(np.array(GRID).T, SHAPE)
    free = np.setdiff1d(np.arange(VOXELS), grid)
    flat[np.random.default_rng(836).permutation(free)[:len(values)]] = values
    flat[grid] = peak
    return flat.reshape(SHAPE).astype(dtype)


def estimate(reference, moving, config):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = estimate_transform(reference, moving, config=config, reference_metadata=META, moving_metadata=META)
    return result, [str(w.message) for w in caught if issubclass(w.category, SpotFindingWarning)]


def local_maxima_record(image, sigma):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SpotFindingWarning)
        result = find_spots(image, config=LocalMaximaConfig(threshold_value=sigma), metadata=META,
                            spot_namespace="landmarks")
    return {key: result.diagnostics["noise"]["0"][key] for key in NOISE_KEYS}


def cpd_counts(points, config):
    """CPD's matched count and control points, from its subsampling on the landmarks (moving = reference here)."""
    reference = _subsample_with_neighbors(points, max_anchors=max(1, config.max_control_points // 4), k_neighbors=3)
    near = cKDTree(points).query_ball_point(reference, r=config.candidate_radius_voxels)
    return len(set().union(*map(set, near))), len(reference)


@pytest.mark.parametrize("config", [TPS, CPD], ids=lambda c: c.method)
def test_a_truncated_landmark_image_takes_the_local_maxima_threshold_and_is_recorded(config):
    # Above 0: 1024 noise voxels and 27 peaks; the 526th of those 1051 values is 41, and so the median; MAD 4.
    image = truncated()
    result, emitted = estimate(image, image, config)
    sigma = config.detection_noise_sigma
    diagnostics = result.diagnostics
    landmarks = diagnostics.landmarks
    expected = local_maxima_record(image, sigma)
    assert (expected["median"], expected["mad"], expected["n_estimation"]) == (41.0, 4.0, 1051)
    np.testing.assert_allclose(expected["threshold"], 41.0 + sigma * 1.4826 * 4.0, **FLOAT64)
    assert list(landmarks) == ["reference", "moving", "control_points"]
    for which in ("reference", "moving"):
        assert landmarks[which] == {"noise": {"0": expected}, "candidates": 27}
    assert expected["estimation"] == "above_minimum" and expected["mad_floored"] is False
    assert (diagnostics.reference_landmark_count, diagnostics.moving_landmark_count) == (27, 27)
    points = np.array(GRID, dtype=float)
    matched, control_points = (27, 20) if config.method == "tps" else cpd_counts(points, config)
    assert (diagnostics.matched_landmark_count, landmarks["control_points"]) == (matched, control_points)
    assert len(diagnostics.warnings) == 2 and diagnostics.warnings == tuple(emitted)
    assert diagnostics.warnings[0].startswith("reference image landmarks: spot finding: channel '0': more than half")
    assert diagnostics.warnings[1].startswith("moving image landmarks: spot finding: channel '0': more than half")


def test_a_floored_landmark_mad_is_recorded_and_noted():
    # Above 0: 900 voxels of 2 and 124 of 3..6 besides the peaks: median 2, MAD 0, floored at one grey level; the
    # threshold 2 + 3 x 1.4826 = 6.448 lies above every voxel but the peaks.
    image = truncated(np.concatenate([np.full(900, 2), np.resize(np.arange(3, 7), 124)]))
    result, emitted = estimate(image, image, TPS)
    record = result.diagnostics.landmarks["reference"]["noise"]["0"]
    assert record == local_maxima_record(image, TPS.detection_noise_sigma)
    assert (record["median"], record["mad"], record["mad_floored"]) == (2.0, 0.0, True)
    np.testing.assert_allclose(record["threshold"], 2.0 + 3.0 * 1.4826, **FLOAT64)
    assert result.diagnostics.reference_landmark_count == 27
    assert sum("MAD floor of 1" in note for note in result.diagnostics.warnings) == 2
    assert result.diagnostics.warnings == tuple(emitted)


@pytest.mark.parametrize("config", [TPS, CPD], ids=lambda c: c.method)
@pytest.mark.parametrize("which", ["reference", "moving"])
def test_a_float_landmark_channel_with_signal_and_a_mad_of_zero_raises_naming_the_image(config, which):
    # Impulses of 100 on 0: the estimation set is the 27 impulses alone, whose MAD is 0.
    impulses, usable = truncated(values=(), peak=100, dtype=np.float32), truncated()
    images = (impulses, usable) if which == "reference" else (usable, impulses)
    with pytest.raises(RegistrationEstimationError,
                       match=rf"^{which} image landmarks: spot finding: channel '0': the noise MAD is 0 over its "
                             r"estimation set 'above_minimum' \(27 voxels\)"):
        estimate(*images, config)


@pytest.mark.parametrize("config", [TPS, CPD], ids=lambda c: c.method)
def test_an_untruncated_landmark_image_keeps_the_all_voxel_threshold(config):
    # 1024 voxels of each grey level 133..148 (none at 0) and the 27 peaks of 250: 6.25 % at the minimum.
    image = np.resize(NOISE + 100, VOXELS).reshape(SHAPE).astype(np.uint8)
    for peak in GRID:
        image[peak] = 250
    result, emitted = estimate(image, image, config)
    values = image.astype(np.float64)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    record = result.diagnostics.landmarks["moving"]["noise"]["0"]
    assert (record["estimation"], record["n_estimation"], record["mad_floored"]) == ("all", VOXELS, False)
    assert record["threshold"] == median + config.detection_noise_sigma * mad * 1.4826  # the base's expression
    assert result.diagnostics.reference_landmark_count == result.diagnostics.moving_landmark_count == 27
    assert result.diagnostics.warnings == () and emitted == []


def test_the_landmarks_record_is_saved_only_for_tps_and_cpd():
    image = truncated()
    tps, _ = estimate(image, image, TPS)
    translation, _ = estimate(image, image, TranslationConfig())
    saved = json_record(tps.diagnostics)
    assert list(saved) == BASE_KEYS + ["landmarks"]
    assert saved["landmarks"] == json_record(tps.diagnostics.landmarks)
    assert [saved[key] for key in BASE_KEYS[5:8]] == [27, 27, 27]
    # The other methods leave the counts and landmarks None, and their records keep the keys they had before W-836.
    other = json_record(translation.diagnostics)
    assert list(other) == BASE_KEYS
    assert translation.diagnostics.landmarks is None and [other[key] for key in BASE_KEYS[5:8]] == [None] * 3
