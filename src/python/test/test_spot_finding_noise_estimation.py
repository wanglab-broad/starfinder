"""Known answers of the local-maxima noise threshold on truncated channels (W-789).

In ``threshold_mode`` ``noise`` a channel's median and MAD are taken over all its voxels when at most half of them
equal its minimum, and otherwise over the voxels above the minimum, which a truncation did not clip. A channel with
signal whose MAD over that set is 0 raises ``SpotFindingError``. The fixtures are uint8 10×32×32 (10240 voxels),
built from repeated values so that every median, MAD and threshold is known in closed form.
"""
import warnings

import numpy as np
import pytest

from starfinder import spot_finding
from starfinder.image import ImageMetadata
from starfinder.spot_finding import LocalMaximaConfig, NoiseLandmarkConfig, SpotFindingPlan, find_spots

from .expected_values import FLOAT64
from .test_spot_finding_golden import CHANNELS, golden_dataset

pytestmark = pytest.mark.spot_finding

SHAPE = (10, 32, 32)
VOXELS = 10240
META = ImageMetadata("noise-estimation")
NAMESPACE = "noise-estimation/test"
# 64 copies of each grey level 33..48: median 40.5; absolute deviations 0.5..7.5, 128 voxels each, so MAD 4.0.
NOISE = np.repeat(np.arange(33, 49), 64)
NOISE_MEDIAN, NOISE_MAD = 40.5, 4.0
RECORD_KEYS = ["zero_fraction", "median", "mad", "threshold", "estimation", "n_estimation"]
# A channel without zeros: 640 copies of each level 133..148, so 6.25 % of its voxels are at its minimum.
FULL = np.resize(NOISE + 100, VOXELS).astype(np.uint8).reshape(SHAPE)


def channel(values, seed=789):
    """A uint8 ZYX channel holding values at seeded random voxels and 0 elsewhere."""
    flat = np.zeros(VOXELS, dtype=np.uint8)
    flat[np.random.default_rng(seed).permutation(VOXELS)[:len(values)]] = values
    return flat.reshape(SHAPE)


def detect(image, config):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = find_spots(image, config=config, metadata=META, spot_namespace=NAMESPACE)
    return result, [str(w.message) for w in caught if issubclass(w.category, spot_finding.SpotFindingWarning)]


def test_a_ninety_percent_zero_channel_takes_its_noise_from_the_nonzero_voxels():
    image = channel(NOISE)
    assert np.mean(image == 0) == 0.9
    result, notes = detect(image, LocalMaximaConfig())
    record = result.diagnostics["noise"]["0"]
    threshold = NOISE_MEDIAN + 5.0 * 1.4826 * NOISE_MAD  # 70.152, above every noise voxel
    np.testing.assert_allclose(result.diagnostics["thresholds"], [threshold], **FLOAT64)
    np.testing.assert_allclose(record["threshold"], threshold, **FLOAT64)
    assert record["median"] == NOISE_MEDIAN and record["mad"] == NOISE_MAD
    assert record["zero_fraction"] == 0.9
    assert (record["estimation"], record["n_estimation"]) == ("above_minimum", 1024)
    assert len(result.spots) == 0
    assert len(notes) == 1 and "'above_minimum'" in notes[0] and "1024 voxels" in notes[0]
    assert result.diagnostics["warnings"] == tuple(notes)


@pytest.mark.parametrize("zeros", [0, VOXELS // 2])
def test_at_most_half_at_the_minimum_keeps_the_all_voxel_threshold(zeros):
    # 320 copies of each level 33..48 fill the voxels that are not zero. With half the voxels zero the median is
    # (0 + 33) / 2 = 16.5 and so is the MAD; without zeros the minimum 33 holds 640 voxels and the median is 40.5.
    levels = np.repeat(np.arange(33, 49), (VOXELS - zeros) // 16)
    image = channel(levels)
    result, notes = detect(image, LocalMaximaConfig())
    record = result.diagnostics["noise"]["0"]
    values = image.astype(np.float64)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    assert (record["estimation"], record["n_estimation"]) == ("all", VOXELS)
    assert record["threshold"] == median + 5.0 * mad * 1.4826  # today's expression, bit for bit
    expected = (16.5, 16.5) if zeros else (40.5, 4.0)
    assert (record["median"], record["mad"]) == expected
    np.testing.assert_allclose(record["threshold"], expected[0] + 5.0 * 1.4826 * expected[1], **FLOAT64)
    assert notes == [] and result.diagnostics["warnings"] == ()


def flat_signal(kind):
    """'above_minimum': 1024 voxels of 7 on zeros (90 % at the minimum); 'all': 10 % zeros, 7168 voxels of 5 and
    2048 noise voxels, so at most half are at the minimum and the MAD over every voxel is 0."""
    if kind == "above_minimum":
        return channel(np.full(1024, 7, dtype=np.uint8))
    return channel(np.concatenate([np.full(7168, 5), NOISE, NOISE]).astype(np.uint8))


@pytest.mark.parametrize("kind", ["above_minimum", "all"])
def test_a_channel_with_signal_and_mad_zero_is_refused_naming_round_and_channel(tmp_path, kind):
    image = np.stack([FULL if c != 1 else flat_signal(kind) for c in range(4)], axis=-1)
    fov = golden_dataset(tmp_path).fov("FOV_001")
    fov.images["round1"] = image
    fov.metadata["round1"] = ImageMetadata("FOV_001/round1")
    with pytest.raises(ValueError, match=r"^spot finding: round 'round1', channel 'ch01': ") as caught:
        fov.find_spots(config=LocalMaximaConfig())
    assert isinstance(caught.value, spot_finding.SpotFindingError)
    message = str(caught.value)
    at_minimum = "9216 of its 10240 voxels (90.0%)" if kind == "above_minimum" else "1024 of its 10240 voxels (10.0%)"
    assert f"estimation set '{kind}'" in message and f"{at_minimum} are at its minimum 0.0" in message
    assert "threshold_mode 'adaptive' or 'absolute'" in message
    with pytest.raises(spot_finding.SpotFindingError, match=r"^spot finding: channel '1': "):
        find_spots(image, config=LocalMaximaConfig(), metadata=META, spot_namespace=NAMESPACE)
    for mode, value in (("adaptive", 0.2), ("absolute", 50.0)):
        assert detect(image, LocalMaximaConfig(threshold_mode=mode, threshold_value=value))[0].diagnostics


def test_an_all_zero_channel_gives_no_spots_and_no_error():
    image = np.stack([np.zeros(SHAPE, dtype=np.uint8), channel(NOISE)], axis=-1)
    result, notes = detect(image, LocalMaximaConfig(channel_labels=("empty", "noise")))
    assert not any("channel 'empty'" in note for note in notes)
    assert result.diagnostics["counts"]["empty"] == 0
    assert result.diagnostics["outcomes"]["empty"] == "constant"
    record = result.diagnostics["noise"]["empty"]
    assert record == {"zero_fraction": 1.0, "median": 0.0, "mad": 0.0, "threshold": 0.0, "estimation": "all",
                      "n_estimation": VOXELS}
    assert len(notes) == 1 and "channel 'noise'" in notes[0]


@pytest.mark.parametrize("mode, value", [("noise", 5.0), ("adaptive", 0.2), ("adaptive_round", 0.2),
                                         ("global", 0.2), ("absolute", 50.0)])
def test_the_noise_record_carries_the_estimation_set(tmp_path, mode, value):
    image = np.stack([channel(NOISE), FULL], axis=-1)
    config = LocalMaximaConfig(threshold_mode=mode, threshold_value=value, channel_labels=("truncated", "full"))
    noise = detect(image, config)[0].diagnostics["noise"]
    assert [list(noise[label]) for label in noise] == [RECORD_KEYS, RECORD_KEYS]
    # Only noise mode estimates above the minimum; the other modes record the all-voxel median and MAD.
    truncated = ("above_minimum", 1024) if mode == "noise" else ("all", VOXELS)
    assert (noise["truncated"]["estimation"], noise["truncated"]["n_estimation"]) == truncated
    if mode != "noise":
        assert (noise["truncated"]["median"], noise["truncated"]["mad"]) == (0.0, 0.0)
    assert (noise["full"]["estimation"], noise["full"]["n_estimation"]) == ("all", VOXELS)
    dataset = golden_dataset(tmp_path)
    fov = dataset.fov("FOV_001")
    fov.images["round1"] = np.concatenate([image, image], axis=-1)
    fov.metadata["round1"] = ImageMetadata("FOV_001/round1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", spot_finding.SpotFindingWarning)
        fov.find_spots(config=SpotFindingPlan(LocalMaximaConfig(threshold_mode=mode, threshold_value=value),
                                              rounds=("round1",)))
    rounds = fov.spot_result.diagnostics["rounds"]["round1"]["noise"]
    assert list(rounds) == list(CHANNELS) and all(list(rounds[label]) == RECORD_KEYS for label in CHANNELS)
    assert rounds["ch00"]["estimation"] == ("above_minimum" if mode == "noise" else "all")


def test_noise_landmarks_keep_the_all_voxel_threshold():
    # Registration landmarks are not pipeline detection: their noise threshold stays the all-voxel one.
    result, notes = detect(channel(NOISE), NoiseLandmarkConfig())
    assert result.diagnostics["thresholds"] == (0.0,) and len(result.spots) > 0 and notes == []


def test_tiled_statistics_carry_the_estimation_set_and_refuse_alike(tmp_path):
    # Tiled execution computes each channel's noise record once, on the FOV's reference round (the global task).
    from starfinder.dataset.tiled import _detection_statistics
    fov = golden_dataset(tmp_path).fov("FOV_001")
    image = np.stack([channel(NOISE), FULL, channel(NOISE), FULL], axis=-1)
    records = _detection_statistics(fov, LocalMaximaConfig(), image)["channels"]
    assert [(r["channel"], r["estimation"], r["n_estimation"]) for r in records] == [
        ("ch00", "above_minimum", 1024), ("ch01", "all", VOXELS), ("ch02", "above_minimum", 1024),
        ("ch03", "all", VOXELS)]
    np.testing.assert_allclose(records[0]["threshold"], NOISE_MEDIAN + 5.0 * 1.4826 * NOISE_MAD, **FLOAT64)
    image[..., 1] = flat_signal("above_minimum")
    with pytest.raises(spot_finding.SpotFindingError, match=r"^spot finding: round 'round1', channel 'ch01': "):
        _detection_statistics(fov, LocalMaximaConfig(), image)
