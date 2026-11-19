"""W-880: adaptive 0.2 is the default threshold of local maxima, and the two weak points of a threshold that follows
each channel's maximum are noted.

The default of ``LocalMaximaConfig`` is ``threshold_mode`` ``adaptive`` at ``threshold_value`` 0.2 (before W-880 it was
``noise`` at 5.0). A bright artifact (the maximum far above the signal, so 0.2 times it lies above most of the signal) and an empty channel (the
maximum does not exceed the channel's noise threshold, so 0.2 times it calls noise) each record a flag in the channel's
noise record and a ``SpotFindingWarning``, also listed in ``diagnostics['warnings']``. Noise mode, and the noise
landmarks of registration, are unchanged. The fixture is uint8 1x64x64x4 (ZYXC): noise of the integers 0..6 (median 3, MAD
2, so the channel's noise threshold is 3 + 5 x 1.4826 x 2 = 17.826), and 400 isolated spots of 50 on a stride-3 grid.
"""
import warnings
from dataclasses import replace

import numpy as np
import pytest

from starfinder import spot_finding
from starfinder.image import ImageMetadata
from starfinder.spot_finding import LocalMaximaConfig, NoiseLandmarkConfig, SpotFindingPlan, find_spots

from .expected_values import FLOAT64
from .test_spot_finding_golden import CHANNELS, golden_dataset

pytestmark = pytest.mark.spot_finding

CLOSE = {"rel": FLOAT64["rtol"], "abs": FLOAT64["atol"]}  # pytest.approx names the float64 tolerance of docs/contributing.md
approx = pytest.approx

META = ImageMetadata("default/round1")
NAMESPACE = "default/sample/FOV_001"
SHAPE = (1, 64, 64)
GRID = [(0, y, x) for y in range(3, 63, 3) for x in range(3, 63, 3)]  # 400 spots, none on a border or next to another
SPOT = 50
ARTIFACT = (0, 61, 62)
NOISE_CONFIG = LocalMaximaConfig(threshold_mode="noise", threshold_value=5.0, channel_labels=CHANNELS)
DEFAULT = LocalMaximaConfig(channel_labels=CHANNELS)
ADAPTIVE_KEYS = ["maximum", "noise_threshold", "empty_channel", "bright_artifact"]
RECORD_KEYS = ["zero_fraction", "median", "mad", "threshold", "estimation", "n_estimation", "mad_floored"]


def noise_plane(seed):
    return np.random.default_rng(seed).integers(0, 7, size=SHAPE).astype(np.uint8)


def with_spots(image, artifact):
    image = image.copy()
    for point in GRID:
        image[point] = SPOT
    if artifact:
        image[ARTIFACT] = 255
    return image


def image_of_four():
    """ch00 the spots and a bright artifact, ch01 the spots, ch02 noise only (empty), ch03 constant."""
    return np.stack([with_spots(noise_plane(1), True), with_spots(noise_plane(2), False), noise_plane(3),
                     np.full(SHAPE, 7, dtype=np.uint8)], axis=-1)


def detect(image, config=DEFAULT, **options):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = find_spots(image, config=config, metadata=META, spot_namespace=NAMESPACE, **options)
    return result, [str(w.message) for w in caught if issubclass(w.category, spot_finding.SpotFindingWarning)]


def noise_threshold(plane):
    """The channel's noise threshold at 5 sigma of an integer image: median + 5 x 1.4826 x max(MAD, 1), over every voxel."""
    values = plane.astype(np.float64)
    median = np.median(values)
    return median + 5.0 * 1.4826 * max(np.median(np.abs(values - median)), 1.0)


# --- the default ---------------------------------------------------------------------------------------

def test_the_default_is_adaptive_at_a_fifth_of_the_channel_maximum():
    config = LocalMaximaConfig()
    assert (config.threshold_mode, config.threshold_value) == ("adaptive", 0.2)
    image = image_of_four()
    result, _ = detect(image)
    for c, label in enumerate(CHANNELS):
        assert result.diagnostics["thresholds"][c] == approx(0.2 * float(image[..., c].max()), **CLOSE)
        assert result.diagnostics["noise"][label]["threshold"] == result.diagnostics["thresholds"][c]
    assert result.config == DEFAULT and result.diagnostics["effective_settings"]["ch00"]["threshold_mode"] == "adaptive"
    assert result.diagnostics["thresholds"] == approx((51.0, 10.0, 1.2, 1.4), **CLOSE)


def test_the_default_reaches_the_plan_the_fov_and_the_workflow_block():
    from starfinder.dataset import PipelineConfig
    from starfinder.dataset.workflow import from_workflow_config
    assert SpotFindingPlan(LocalMaximaConfig()).config == LocalMaximaConfig()
    assert PipelineConfig(spot_finding=LocalMaximaConfig()).spot_finding.threshold_mode == "adaptive"
    block = {"n_rounds": 1, "ref_round": "round1", "dataset_id": "d", "sample_id": "s", "output_id": "o",
             "root_input_path": "in", "root_output_path": "out", "seq_channel_order": list(CHANNELS),
             "rules": {"rsf_single_fov": {"parameters": {"load_raw_images": {"run": False},
                                                         "spot_finding": {"run": True}}}}}
    assert from_workflow_config(block).pipeline.spot_finding == LocalMaximaConfig()


def test_a_noise_value_needs_the_noise_mode_and_the_noise_mode_is_still_accepted():
    with pytest.raises(ValueError, match=r"^threshold_value must be finite and in \[0, 1\]$"):
        LocalMaximaConfig(threshold_value=5.0)
    assert LocalMaximaConfig(threshold_mode="noise", threshold_value=5.0).threshold_value == 5.0
    assert LocalMaximaConfig(threshold_mode="absolute", threshold_value=50.0).threshold_value == 50.0


def test_the_registration_landmarks_keep_the_noise_rule():
    # W-880 leaves NoiseLandmarkConfig as it is: noise at 5.0 sigma, whatever local maxima's default is.
    assert NoiseLandmarkConfig().noise_sigma == 5.0
    plane = with_spots(noise_plane(4), False)
    result, notes = detect(plane, NoiseLandmarkConfig())
    assert result.diagnostics["thresholds"] == (approx(noise_threshold(plane), **CLOSE),)
    assert list(result.diagnostics["noise"]["0"]) == RECORD_KEYS and notes == []
    assert len(result.spots) == 400


# --- the two weak points of adaptive mode ------------------------------------------------------------------

def test_a_bright_artifact_raises_the_threshold_above_the_spots_and_is_noted():
    image = image_of_four()[..., :2]
    result, notes = detect(image, replace(DEFAULT, channel_labels=CHANNELS[:2]))
    plane = image[..., 0]
    record = result.diagnostics["noise"]["ch00"]
    reference = noise_threshold(plane)
    signal = plane[plane > reference].astype(np.float64)
    median = np.median(plane.astype(np.float64))
    # The 400 spots of 50 and the artifact are the only voxels above the noise threshold 17.826; the 99th percentile
    # of those 401 values is 50, and the artifact (255) lies 252 above the median against 3 x 47 for the spots.
    assert reference == approx(17.826, **CLOSE) and signal.size == 401
    assert np.percentile(signal, 99) == 50.0 and median == 3.0
    assert record["maximum"] == 255.0 and record["noise_threshold"] == approx(reference, **CLOSE)
    assert record["bright_artifact"] is True and record["empty_channel"] is False
    assert (record["maximum"] - median) > 3.0 * (np.percentile(signal, 99) - median)
    # The threshold 51 is above every spot of 50: only the artifact is called. Without it the threshold is 10.
    assert record["threshold"] == approx(51.0, **CLOSE)
    assert result.diagnostics["counts"] == {"ch00": 1, "ch01": 400}
    clean = result.diagnostics["noise"]["ch01"]
    assert (clean["bright_artifact"], clean["empty_channel"], clean["maximum"]) == (False, False, 50.0)
    assert len(notes) == 1 and notes[0] == result.diagnostics["warnings"][0]
    assert "channel 'ch00'" in notes[0] and "bright artifact" in notes[0] and "(255.0)" in notes[0]
    assert "51.0" in notes[0] and "round" not in notes[0]


def test_an_empty_channel_is_noted_and_its_threshold_calls_noise():
    image = image_of_four()[..., :3]
    result, notes = detect(image, replace(DEFAULT, channel_labels=CHANNELS[:3]))
    plane = image[..., 2]
    record = result.diagnostics["noise"]["ch02"]
    reference = noise_threshold(plane)
    # The channel holds the integers 0..6 only: its maximum 6 does not exceed its noise threshold 17.826.
    assert plane.max() == 6 and reference == approx(17.826, **CLOSE)
    assert record["maximum"] == 6.0 and record["noise_threshold"] == approx(reference, **CLOSE)
    assert record["empty_channel"] is True and record["bright_artifact"] is False
    # 0.2 x 6 = 1.2 lies inside the noise, so isolated noise voxels are called.
    assert record["threshold"] == approx(1.2, **CLOSE)
    spots = result.spots[result.spots.channel == 2]
    assert len(spots) > 0 and (spots.peak_intensity > 1.2).all() and result.diagnostics["outcomes"]["ch02"] == "ok"
    assert len(notes) == 2 and any("channel 'ch02'" in n and "holds no signal above noise" in n for n in notes)
    assert result.diagnostics["warnings"] == tuple(notes)


def test_a_channel_with_signal_and_a_constant_channel_are_not_noted():
    image = image_of_four()[..., [1, 3]]
    result, notes = detect(image, replace(DEFAULT, channel_labels=("signal", "constant")))
    assert notes == [] and result.diagnostics["warnings"] == ()
    for label in ("signal", "constant"):
        record = result.diagnostics["noise"][label]
        assert (record["empty_channel"], record["bright_artifact"]) == (False, False)
        assert list(record) == RECORD_KEYS + ADAPTIVE_KEYS
    assert result.diagnostics["outcomes"] == {"signal": "ok", "constant": "constant"}
    assert result.diagnostics["noise"]["constant"]["threshold"] == approx(1.4, **CLOSE)


def test_the_notes_are_emitted_once_each_and_name_the_round_in_a_fov_detection(tmp_path):
    dataset = golden_dataset(tmp_path)
    fov = dataset.fov("FOV_001")
    fov.images["round1"] = image_of_four()
    fov.metadata["round1"] = META
    with pytest.warns(spot_finding.SpotFindingWarning) as caught:
        fov.find_spots(config=LocalMaximaConfig())
    messages = [str(w.message) for w in caught]
    assert len(messages) == 2
    assert any("round 'round1', channel 'ch00'" in m and "bright artifact" in m for m in messages)
    assert any("round 'round1', channel 'ch02'" in m and "holds no signal above noise" in m for m in messages)
    assert fov.spot_result.diagnostics["warnings"] == tuple(messages)


def test_the_tiled_route_lists_the_notes_naming_the_reference_round(tmp_path):
    # The global task takes the FOV-level statistics of the reference round: the notes say so and name the round.
    from starfinder.dataset import PipelineConfig
    from starfinder.dataset.tiled import _detection_statistics, _merged_detection
    fov = golden_dataset(tmp_path).fov("FOV_001")
    image = image_of_four()
    pipeline = PipelineConfig(spot_finding=LocalMaximaConfig())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        statistics = {"detection": _detection_statistics(fov, pipeline.spot_finding, image),
                      "reference_metadata": {"frame_id": "default"}}
    emitted = [str(w.message) for w in caught if issubclass(w.category, spot_finding.SpotFindingWarning)]
    flags = [(r["channel"], r["bright_artifact"], r["empty_channel"]) for r in statistics["detection"]["channels"]]
    assert flags == [("ch00", True, False), ("ch01", False, False), ("ch02", False, True), ("ch03", False, False)]
    assert len(emitted) == 2 and all("(the FOV-level statistics of round 'round1', applied to every round)" in m
                                     for m in emitted)
    direct = detect(image)[0]
    merged = _merged_detection(fov, pipeline, statistics, direct.diagnostics, direct.spots, records=[])
    assert merged.diagnostics["noise"] == direct.diagnostics["noise"]
    assert merged.diagnostics["warnings"] == tuple(emitted)


# --- noise mode is unchanged -----------------------------------------------------------------------------------

def test_noise_mode_keeps_its_record_threshold_and_spots_and_has_no_adaptive_notes():
    image = image_of_four()
    result, notes = detect(image, NOISE_CONFIG)
    for c, label in enumerate(CHANNELS[:3]):
        record = result.diagnostics["noise"][label]
        assert list(record) == RECORD_KEYS
        assert record["threshold"] == approx(noise_threshold(image[..., c]), **CLOSE)
        assert (record["estimation"], record["n_estimation"], record["mad_floored"]) == ("all", 4096, False)
    # ch00 and ch01: the 400 spots (and, in ch00, the artifact) lie above 17.826; the noise voxels are below it.
    assert result.diagnostics["counts"] == {"ch00": 401, "ch01": 400, "ch02": 0, "ch03": 0}
    assert result.diagnostics["thresholds"][:3] == approx((17.826, 17.826, 17.826), **CLOSE)
    assert notes == [] and result.diagnostics["warnings"] == ()


@pytest.mark.parametrize("mode, value", [("noise", 5.0), ("adaptive_round", 0.2), ("global", 0.2), ("absolute", 20.0)])
def test_only_adaptive_mode_records_the_checks(mode, value):
    config = LocalMaximaConfig(threshold_mode=mode, threshold_value=value, channel_labels=CHANNELS)
    result, notes = detect(image_of_four(), config)
    assert all(list(record) == RECORD_KEYS for record in result.diagnostics["noise"].values())
    assert notes == []
