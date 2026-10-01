"""The detection plan with per-channel overrides, the effective settings, and the §2.5 noise-mode record
and warning (W-270, docs/spot-finding-contract.md).

The bounds here are provisional: new diagnostics and bookkeeping without a W-266 reference
(parts of checks S4 and S6 of docs/spot-finding-algorithms.md).
"""
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from starfinder.dataset import PipelineConfig
from starfinder.image import ImageMetadata
from starfinder.io._checkpoint import _jsonable, _tuples
from starfinder.spot_finding import (ChannelOverride, LocalMaximaConfig, NoiseLandmarkConfig, PercentileCentroidConfig,
    SpotFindingPlan, SpotFindingWarning, find_spots)

from .test_spot_finding_golden import CHANNELS, fixture_image, fov_with_fixture, golden_dataset

META = ImageMetadata("golden/round1")
NAMESPACE = "golden/sample/FOV_001"
BASE = LocalMaximaConfig(channel_labels=CHANNELS)
OVERRIDE = LocalMaximaConfig(threshold_mode="adaptive", threshold_value=0.2)


def detect(image, config, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SpotFindingWarning)
        return find_spots(image, config=config, metadata=META, spot_namespace=NAMESPACE, **options)


def settings(config):
    return _tuples(_jsonable(config))


# --- Noise record and warning -------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["noise", "adaptive", "adaptive_round", "global"])
@pytest.mark.parametrize("dims", ["3d", "z1"])
def test_noise_record_equals_a_direct_numpy_computation(dims, mode):
    image = fixture_image(dims)
    value = {"noise": 5.0, "global": 0.01}.get(mode, 0.2)
    result = detect(image, replace(BASE, threshold_mode=mode, threshold_value=value))
    noise = result.diagnostics["noise"]
    assert list(noise) == list(CHANNELS)
    for c, label in enumerate(CHANNELS):
        values = image[..., c].astype(np.float64)
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        expected = {"zero_fraction": np.mean(values == 0), "median": median, "mad": mad,
                    "threshold": result.diagnostics["thresholds"][c]}
        assert set(noise[label]) == set(expected)
        for key, value_ in expected.items():
            assert abs(noise[label][key] - value_) <= 1e-12, (label, key)
        if mode == "noise":
            assert abs(noise[label]["threshold"] - (median + 5.0 * 1.4826 * mad)) <= 1e-12
    assert noise["ch02"]["mad"] == 0 and noise["ch02"]["zero_fraction"] > 0.5
    assert all(noise[label]["mad"] > 0 and noise[label]["zero_fraction"] < 0.5 for label in ("ch00", "ch01", "ch03"))


@pytest.mark.parametrize("dims", ["3d", "z1"])
def test_the_mad_zero_channel_warns_once_and_names_it(dims):
    with pytest.warns(SpotFindingWarning) as caught:
        result = find_spots(fixture_image(dims), config=BASE, metadata=META, spot_namespace=NAMESPACE)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "channel 'ch02'" in message and "MAD is 0" in message and "unchanged" in message
    assert result.diagnostics["warnings"] == (message,)
    assert result.diagnostics["thresholds"][2] == 0.0


def zero_channels(seed=270):
    """Two channels with 10 spots each: channel 0 is exactly 60 % zeros (MAD 0), channel 1 exactly 40 % (MAD > 0)."""
    rng = np.random.default_rng(seed)
    shape = (10, 32, 32)  # 10240 voxels: 60 % and 40 % are whole numbers of voxels
    channels = []
    for zeros in (6144, 4096):
        values = rng.integers(20, 60, size=int(np.prod(shape))).astype(np.uint16)
        order = rng.permutation(values.size)
        values[order[:zeros]] = 0
        values[order[zeros:zeros + 10]] = 1000
        channels.append(values.reshape(shape))
    return np.stack(channels, axis=-1)


def test_sixty_percent_zeros_warn_and_forty_percent_do_not():
    image = zero_channels()
    assert (np.mean(image[..., 0] == 0), np.mean(image[..., 1] == 0)) == (0.6, 0.4)
    config = LocalMaximaConfig(channel_labels=("sixty", "forty"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = find_spots(image, config=config, metadata=META, spot_namespace=NAMESPACE)
    noise = result.diagnostics["noise"]
    assert (noise["sixty"]["zero_fraction"], noise["sixty"]["mad"]) == (0.6, 0.0)
    assert noise["forty"]["zero_fraction"] == 0.4 and noise["forty"]["mad"] > 0
    assert [w.category for w in caught] == [SpotFindingWarning]
    assert "channel 'sixty'" in str(caught[0].message)
    assert len(result.diagnostics["warnings"]) == 1


def test_fov_detection_names_the_round_in_the_warning(tmp_path):
    fov = fov_with_fixture(golden_dataset(tmp_path), "3d")
    with pytest.warns(SpotFindingWarning, match="round 'round1', channel 'ch02'"):
        fov.find_spots(config=LocalMaximaConfig())


def test_registration_landmarks_and_centroids_record_no_noise_and_do_not_warn():
    image = fixture_image("3d")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for config in (NoiseLandmarkConfig(), PercentileCentroidConfig()):
            result = find_spots(image, config=config, metadata=META, spot_namespace=NAMESPACE)
            assert "noise" not in result.diagnostics and result.diagnostics["warnings"] == ()
    assert caught == []


# --- Per-channel overrides and effective settings -----------------------------------------------------

@pytest.mark.parametrize("dims", ["3d", "z1"])
def test_an_override_changes_only_its_channel(dims):
    image = fixture_image(dims)
    base = detect(image, BASE)
    plan = SpotFindingPlan(BASE, (ChannelOverride("ch02", OVERRIDE),))
    result = detect(image, plan)
    columns = ["z", "y", "x", "channel", "peak_intensity"]
    spots = result.spots
    assert spots.spot_id.tolist() == [str(i) for i in range(len(spots))]
    assert spots.channel.is_monotonic_increasing
    others = spots[spots.channel != 2][columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(others, base.spots[base.spots.channel != 2][columns].reset_index(drop=True),
                                  check_exact=True)
    single = detect(image[..., 2:3], replace(OVERRIDE, channel_labels=("ch02",)))
    expected = single.spots[columns].assign(channel=np.int64(2))
    pd.testing.assert_frame_equal(spots[spots.channel == 2][columns].reset_index(drop=True), expected,
                                  check_exact=True)
    assert len(expected) > 0
    effective = result.diagnostics["effective_settings"]
    assert list(effective) == list(CHANNELS)
    assert effective["ch02"] == settings(replace(OVERRIDE, channel_labels=CHANNELS))
    assert all(effective[label] == settings(BASE) for label in ("ch00", "ch01", "ch03"))
    thresholds = result.diagnostics["thresholds"]
    assert thresholds[2] == single.diagnostics["thresholds"][0]
    assert [thresholds[c] for c in (0, 1, 3)] == [base.diagnostics["thresholds"][c] for c in (0, 1, 3)]
    assert result.config == BASE


@pytest.mark.parametrize("override_measures", [True, False])
@pytest.mark.parametrize("plan_measures", [True, False])
@pytest.mark.parametrize("dims", ["3d", "z1"])
def test_an_override_keeps_the_output_columns(dims, plan_measures, override_measures):
    # An override may change any setting except one that changes the output columns (W-270 review).
    image = fixture_image(dims)
    base_config = replace(BASE, measure_peak_intensity=plan_measures)
    override = replace(OVERRIDE, measure_peak_intensity=override_measures)
    if plan_measures != override_measures:
        with pytest.raises(ValueError, match="'ch02'.*measure_peak_intensity"):
            SpotFindingPlan(base_config, (ChannelOverride("ch02", override),))
        return
    result = detect(image, SpotFindingPlan(base_config, (ChannelOverride("ch02", override),)))
    base = detect(image, base_config)
    columns = ["z", "y", "x", "channel"] + (["peak_intensity"] if plan_measures else [])
    assert list(result.spots.columns) == ["spot_id", *columns]
    pd.testing.assert_frame_equal(result.spots[result.spots.channel != 2][columns].reset_index(drop=True),
                                  base.spots[base.spots.channel != 2][columns].reset_index(drop=True),
                                  check_exact=True)
    single = detect(image[..., 2:3], replace(override, channel_labels=("ch02",)))
    pd.testing.assert_frame_equal(result.spots[result.spots.channel == 2][columns].reset_index(drop=True),
                                  single.spots[columns].assign(channel=np.int64(2)), check_exact=True)


def test_without_overrides_every_channel_has_the_config():
    result = detect(fixture_image("3d"), BASE)
    assert result.diagnostics["effective_settings"] == {label: settings(BASE) for label in CHANNELS}
    unlabeled = detect(fixture_image("3d"), LocalMaximaConfig())
    assert list(unlabeled.diagnostics["effective_settings"]) == ["0", "1", "2", "3"]
    pd.testing.assert_frame_equal(detect(fixture_image("3d"), SpotFindingPlan(BASE)).spots, result.spots,
                                  check_exact=True)


def test_a_plan_runs_through_the_pipeline_and_the_fov(tmp_path):
    plan = SpotFindingPlan(LocalMaximaConfig(), (ChannelOverride("ch02", OVERRIDE),))
    assert PipelineConfig(spot_finding=plan).spot_finding == plan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SpotFindingWarning)
        fov = fov_with_fixture(golden_dataset(tmp_path), "3d").run(PipelineConfig(spot_finding=plan))
    direct = detect(fixture_image("3d"), SpotFindingPlan(BASE, plan.channel_overrides))
    pd.testing.assert_frame_equal(fov.spot_result.spots, direct.spots, check_exact=True)
    assert fov.spot_result.config == BASE
    assert fov.spot_result.diagnostics["effective_settings"] == direct.diagnostics["effective_settings"]


@pytest.mark.parametrize("make, error", [
    (lambda: SpotFindingPlan(BASE, (ChannelOverride("ch02", NoiseLandmarkConfig()),)), TypeError),
    (lambda: SpotFindingPlan(BASE, (ChannelOverride("ch09", OVERRIDE),)), ValueError),
    (lambda: SpotFindingPlan(BASE, (ChannelOverride("ch02", OVERRIDE), ChannelOverride("ch02", OVERRIDE))),
     ValueError),
    (lambda: SpotFindingPlan(BASE, (ChannelOverride("ch02", replace(OVERRIDE, channel_labels=("a", "b", "c", "d"))),)),
     ValueError),
    (lambda: SpotFindingPlan(BASE, ({"channel": "ch02", "config": OVERRIDE},)), TypeError),
    (lambda: ChannelOverride("", OVERRIDE), ValueError),
    (lambda: ChannelOverride("ch02", "adaptive"), TypeError),
])
def test_invalid_overrides_are_rejected(make, error):
    with pytest.raises(error):
        make()


def test_invalid_overrides_are_rejected_at_detection(tmp_path):
    image = fixture_image("3d")
    with pytest.raises(ValueError, match="channel_labels"):
        detect(image, SpotFindingPlan(LocalMaximaConfig(), (ChannelOverride("ch02", OVERRIDE),)))
    with pytest.raises(ValueError, match="combines channels"):
        detect(image, SpotFindingPlan(PercentileCentroidConfig(channel_labels=CHANNELS),
                                      (ChannelOverride("ch02", PercentileCentroidConfig()),)))
    fov = fov_with_fixture(golden_dataset(tmp_path), "3d")
    with pytest.raises(ValueError, match="ch09"):
        fov.find_spots(config=SpotFindingPlan(LocalMaximaConfig(), (ChannelOverride("ch09", OVERRIDE),)))
