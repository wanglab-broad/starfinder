"""Known answers of the local-maxima noise threshold on truncated channels (W-789) and its MAD floor (W-848), which the
registration noise landmarks share (W-836).

In ``threshold_mode`` ``noise`` a channel's median and MAD are taken over all its voxels when at most half of them
equal its minimum, and otherwise over the voxels above the minimum, which a truncation did not clip. On an integer
image the MAD has a floor of one grey level; a floating-point channel with signal whose MAD over that set is 0 raises
``SpotFindingError``. The fixtures are uint8 10×32×32 (10240 voxels), built from repeated values so that every
median, MAD and threshold is known in closed form.
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
RECORD_KEYS = ["zero_fraction", "median", "mad", "threshold", "estimation", "n_estimation", "mad_floored"]
SIGMA = 5.0 * 1.4826  # the default threshold_value times the MAD scale
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


@pytest.mark.parametrize("zeros", [0, VOXELS // 2])
def test_a_mad_of_at_least_one_grey_level_is_not_floored(zeros):
    # The fixtures of the test above: MAD 4.0 and 16.5; the threshold is unchanged and there is no floor warning (W-848).
    levels = np.repeat(np.arange(33, 49), (VOXELS - zeros) // 16)
    result, notes = detect(channel(levels), LocalMaximaConfig())
    record = result.diagnostics["noise"]["0"]
    assert record["mad"] in (4.0, 16.5) and record["mad_floored"] is False
    assert record["threshold"] == record["median"] + 5.0 * record["mad"] * 1.4826
    assert not [n for n in notes if "MAD floor" in n]


def flat_signal(kind):
    """'above_minimum': 1024 voxels of 7 on zeros (90 % at the minimum); 'all': 10 % zeros, 7168 voxels of 5 and
    2048 noise voxels, so at most half are at the minimum and the MAD over every voxel is 0."""
    if kind == "above_minimum":
        return channel(np.full(1024, 7, dtype=np.uint8))
    return channel(np.concatenate([np.full(7168, 5), NOISE, NOISE]).astype(np.uint8))


def mad_zero(dtype=np.uint8):
    """90 % zeros; of the 1024 voxels above 0, 700 are 2 (their median and MAD over them: 2 and 0), 324 are 3..8, and
    two isolated spots of 50 stand above every threshold the tests expect (2 + 5 * 1.4826 = 9.413)."""
    values = np.concatenate([np.full(700, 2), np.resize(np.arange(3, 9), 322), [50, 50]])
    image = channel(values.astype(np.uint8))
    return image.astype(dtype)


def test_a_mad_of_zero_on_an_integer_channel_is_floored_at_one_grey_level():
    image = mad_zero()
    result, notes = detect(image, LocalMaximaConfig())
    record = result.diagnostics["noise"]["0"]
    threshold = 2.0 + SIGMA * 1.0
    assert (record["median"], record["mad"], record["mad_floored"]) == (2.0, 0.0, True)
    assert (record["estimation"], record["n_estimation"]) == ("above_minimum", 1024)
    np.testing.assert_allclose(record["threshold"], threshold, **FLOAT64)
    np.testing.assert_allclose(result.diagnostics["thresholds"], [threshold], **FLOAT64)
    assert len(result.spots) == 2 and set(result.spots["peak_intensity"]) == {50.0}
    floor = [n for n in notes if "MAD floor" in n]
    assert len(floor) == 1 and len(notes) == 2  # the above-minimum note and the floor note
    assert "channel '0'" in floor[0] and "measured noise MAD (0.0)" in floor[0] and "'above_minimum'" in floor[0]
    assert f"the threshold is {record['threshold']!r}" in floor[0]
    assert result.diagnostics["warnings"] == tuple(notes)


def test_a_mad_of_one_half_is_floored_to_one():
    # 5120 voxels of 4, 5117 of 5 and three spots of 40: the minimum holds exactly half, so every voxel is used
    # (median 4.5, every deviation but the spots' 0.5, MAD 0.5).
    flat = np.concatenate([np.full(5120, 4), np.full(5117, 5), np.full(3, 40)]).astype(np.uint8)
    image = flat[np.random.default_rng(848).permutation(VOXELS)].reshape(SHAPE)
    result, notes = detect(image, LocalMaximaConfig())
    record = result.diagnostics["noise"]["0"]
    assert (record["estimation"], record["median"], record["mad"], record["mad_floored"]) == ("all", 4.5, 0.5, True)
    np.testing.assert_allclose(record["threshold"], 4.5 + SIGMA * 1.0, **FLOAT64)  # not 4.5 + SIGMA * 0.5
    assert len(notes) == 1 and "MAD floor" in notes[0] and "measured noise MAD (0.5)" in notes[0]


def test_a_mad_of_exactly_one_grey_level_is_not_floored():
    # 5120 voxels of 3 and 5120 of 5: median 4, MAD 1, and the minimum holds exactly half of the voxels.
    flat = np.concatenate([np.full(5120, 3), np.full(5120, 5)]).astype(np.uint8)
    image = flat[np.random.default_rng(848).permutation(VOXELS)].reshape(SHAPE)
    result, notes = detect(image, LocalMaximaConfig())
    record = result.diagnostics["noise"]["0"]
    assert (record["median"], record["mad"], record["mad_floored"]) == (4.0, 1.0, False)
    assert record["threshold"] == 4.0 + 5.0 * 1.0 * 1.4826  # the unfloored expression, bit for bit
    assert notes == [] and result.diagnostics["warnings"] == ()


def test_the_floor_names_the_round_and_channel_in_a_fov_detection(tmp_path):
    image = np.stack([FULL if c != 1 else mad_zero() for c in range(4)], axis=-1)
    fov = golden_dataset(tmp_path).fov("FOV_001")
    fov.images["round1"] = image
    fov.metadata["round1"] = ImageMetadata("FOV_001/round1")
    with pytest.warns(spot_finding.SpotFindingWarning, match=r"round 'round1', channel 'ch01': ") as caught:
        fov.find_spots(config=LocalMaximaConfig())
    floor = [str(w.message) for w in caught if "MAD floor" in str(w.message)]
    assert len(floor) == 1 and "round 'round1', channel 'ch01': the measured noise MAD (0.0)" in floor[0]
    noise = fov.spot_result.diagnostics["noise"]
    assert [noise[label]["mad_floored"] for label in CHANNELS] == [False, True, False, False]
    assert floor[0] in fov.spot_result.diagnostics["warnings"]


@pytest.mark.parametrize("kind", ["above_minimum", "all"])
def test_a_float_channel_with_signal_and_mad_zero_is_refused_naming_round_and_channel(tmp_path, kind):
    flat = flat_signal(kind).astype(np.float32)
    image = np.stack([FULL.astype(np.float32) if c != 1 else flat for c in range(4)], axis=-1)
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


def test_a_float_channel_with_a_mad_of_zero_over_signal_raises_where_the_integer_one_is_floored():
    for dtype, raises in ((np.uint8, False), (np.uint16, False), (np.float32, True), (np.float64, True)):
        image = mad_zero(dtype)
        if raises:
            with pytest.raises(spot_finding.SpotFindingError):
                find_spots(image, config=LocalMaximaConfig(), metadata=META, spot_namespace=NAMESPACE)
        else:
            assert detect(image, LocalMaximaConfig())[0].diagnostics["noise"]["0"]["mad_floored"] is True


def test_an_all_zero_channel_gives_no_spots_and_no_error():
    image = np.stack([np.zeros(SHAPE, dtype=np.uint8), channel(NOISE)], axis=-1)
    result, notes = detect(image, LocalMaximaConfig(channel_labels=("empty", "noise")))
    assert not any("channel 'empty'" in note for note in notes)
    assert result.diagnostics["counts"]["empty"] == 0
    assert result.diagnostics["outcomes"]["empty"] == "constant"
    record = result.diagnostics["noise"]["empty"]
    assert record == {"zero_fraction": 1.0, "median": 0.0, "mad": 0.0, "threshold": 0.0, "estimation": "all",
                      "n_estimation": VOXELS, "mad_floored": False}  # no signal: the floor does not apply (W-848)
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


# W-836: eight isolated landmark voxels inside the border, on voxels that channel() leaves at 0 for both fixtures below.
PEAKS = [(z, y, x) for z in (3, 6) for y in (8, 20) for x in (8, 20)]


def with_peaks(image, value=200):
    image = image.copy()
    for peak in PEAKS:
        assert image[peak] == 0
        image[peak] = value
    return image


def regional_maxima_above(image, threshold):
    """The number of full-connectivity regional maxima above threshold, counted with scikit-image and SciPy alone."""
    from scipy.ndimage import label
    from skimage.morphology import local_maxima
    mask = local_maxima(image, connectivity=image.ndim, allow_borders=True) & (image > threshold)
    return label(mask, structure=np.ones((3,) * image.ndim, dtype=bool))[1]


@pytest.mark.parametrize("sigma", [3.0, 5.0])
@pytest.mark.parametrize("dims", ["3d", "z1"])
def test_noise_landmarks_take_the_local_maxima_noise_threshold_on_a_truncated_channel(sigma, dims):
    # 90 % zeros; above 0 the 1024 noise voxels (33..48) and the eight peaks: median 41 and MAD 4 over those 1032.
    image = with_peaks(channel(NOISE))
    if dims == "z1":  # one plane of 32x32: z=6 holds four of the peaks; its own noise record follows from the plane.
        image = image[6:7].copy()
    result, notes = detect(image, NoiseLandmarkConfig(noise_sigma=sigma))
    pipeline, _ = detect(image, LocalMaximaConfig(threshold_value=sigma))
    record = result.diagnostics["noise"]["0"]
    assert record == pipeline.diagnostics["noise"]["0"]
    assert result.diagnostics["thresholds"] == pipeline.diagnostics["thresholds"]
    assert record["estimation"] == "above_minimum" and record["mad_floored"] is False
    if dims == "3d":
        assert (record["median"], record["mad"], record["n_estimation"]) == (41.0, 4.0, 1032)
        np.testing.assert_allclose(record["threshold"], 41.0 + sigma * 1.4826 * 4.0, **FLOAT64)
    assert record["threshold"] > 48  # above every noise voxel, not 0
    assert len(result.spots) == regional_maxima_above(image, record["threshold"]) == (8 if dims == "3d" else 4)
    assert sorted(map(tuple, result.spots[["z", "y", "x"]].to_numpy().astype(int))) == sorted(
        p if dims == "3d" else (0, p[1], p[2]) for p in PEAKS if dims == "3d" or p[0] == 6)
    assert len(notes) == 1 and "'above_minimum'" in notes[0] and result.diagnostics["warnings"] == tuple(notes)


def test_noise_landmarks_floor_the_mad_of_an_integer_channel_as_local_maxima_does():
    # mad_zero: above 0, 700 voxels of 2 hold the median and MAD (2 and 0); the MAD is floored at one grey level.
    image = with_peaks(mad_zero())
    result, notes = detect(image, NoiseLandmarkConfig())
    pipeline, _ = detect(image, LocalMaximaConfig())
    record = result.diagnostics["noise"]["0"]
    assert record == pipeline.diagnostics["noise"]["0"]
    assert (record["median"], record["mad"], record["mad_floored"]) == (2.0, 0.0, True)
    np.testing.assert_allclose(record["threshold"], 2.0 + SIGMA * 1.0, **FLOAT64)
    # The eight peaks and mad_zero's two isolated spots of 50, all inside the border.
    assert len(result.spots) == regional_maxima_above(image, record["threshold"]) == 10
    assert len(notes) == 2 and "MAD floor of 1" in notes[1] and result.diagnostics["warnings"] == tuple(notes)


def test_noise_landmarks_refuse_a_float_channel_with_signal_and_a_mad_of_zero():
    with pytest.raises(spot_finding.SpotFindingError, match=r"^spot finding: channel '0': the noise MAD is 0 "):
        detect(with_peaks(mad_zero(np.float32)), NoiseLandmarkConfig())


def test_noise_landmarks_keep_the_all_voxel_threshold_when_at_most_half_sit_at_the_minimum():
    # FULL: 6.25 % of the voxels at its minimum 133. The threshold is the base's all-voxel expression, bit for bit.
    image = FULL.copy()
    for peak in PEAKS:
        image[peak] = 250
    result, notes = detect(image, NoiseLandmarkConfig())
    values = image.astype(np.float64)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    record = result.diagnostics["noise"]["0"]
    assert (record["estimation"], record["n_estimation"], record["mad_floored"]) == ("all", VOXELS, False)
    assert result.diagnostics["thresholds"] == (median + 5.0 * mad * 1.4826,)
    assert len(result.spots) == regional_maxima_above(image, record["threshold"]) == 8
    assert notes == [] and result.diagnostics["warnings"] == ()


def test_tiled_statistics_carry_the_estimation_set_and_the_floor(tmp_path):
    # Tiled execution computes each channel's noise record once, on the FOV's reference round (the global task).
    from starfinder.dataset.tiled import _detection_statistics
    fov = golden_dataset(tmp_path).fov("FOV_001")
    image = np.stack([channel(NOISE), FULL, channel(NOISE), FULL], axis=-1)
    records = _detection_statistics(fov, LocalMaximaConfig(), image)["channels"]
    assert [(r["channel"], r["estimation"], r["n_estimation"], r["mad_floored"]) for r in records] == [
        ("ch00", "above_minimum", 1024, False), ("ch01", "all", VOXELS, False), ("ch02", "above_minimum", 1024, False),
        ("ch03", "all", VOXELS, False)]
    np.testing.assert_allclose(records[0]["threshold"], NOISE_MEDIAN + 5.0 * 1.4826 * NOISE_MAD, **FLOAT64)
    # W-848: an integer channel with signal and a MAD of 0 is floored, as direct detection does, and not refused.
    image[..., 1] = mad_zero()
    records = _detection_statistics(fov, LocalMaximaConfig(), image)["channels"]
    assert [r["mad_floored"] for r in records] == [False, True, False, False]
    assert (records[1]["median"], records[1]["mad"], records[1]["constant"]) == (2.0, 0.0, False)
    np.testing.assert_allclose(records[1]["threshold"], 2.0 + SIGMA, **FLOAT64)
    direct = detect(image, LocalMaximaConfig(channel_labels=CHANNELS))[0].diagnostics["noise"]
    assert [{k: v for k, v in r.items() if k not in ("channel", "constant")} for r in records] == [
        direct[label] for label in CHANNELS]
    # A float channel keeps the refusal.
    with pytest.raises(spot_finding.SpotFindingError, match=r"^spot finding: round 'round1', channel 'ch01': "):
        _detection_statistics(fov, LocalMaximaConfig(), image.astype(np.float32))


def floor_warnings(caught):
    return [str(w.message) for w in caught if issubclass(w.category, spot_finding.SpotFindingWarning)
            and "MAD floor" in str(w.message)]


def test_tiled_statistics_warn_for_a_floored_channel_naming_the_reference_round(tmp_path):
    # W-848: the global task's FOV-level statistics emit the floor warning, naming the reference round and the channel.
    from starfinder.dataset.tiled import _detection_statistics
    fov = golden_dataset(tmp_path).fov("FOV_001")
    image = np.stack([channel(NOISE), mad_zero(), channel(NOISE), FULL], axis=-1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        records = _detection_statistics(fov, LocalMaximaConfig(), image)["channels"]
    emitted = floor_warnings(caught)
    assert len(emitted) == 1 and [r["mad_floored"] for r in records] == [False, True, False, False]
    message = emitted[0]
    assert message.startswith("spot finding: round 'round1', channel 'ch01': the measured noise MAD (0.0) over its "
                              "estimation set 'above_minimum' (1024 voxels) is below one grey level")
    assert f"the threshold is {records[1]['threshold']!r}" in message and "FOV-level statistics of round 'round1'" in message
    # No channel is floored: no floor warning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _detection_statistics(fov, LocalMaximaConfig(), np.stack([channel(NOISE), FULL, FULL, FULL], axis=-1))
    assert floor_warnings(caught) == []


def test_the_tiled_merge_records_the_floor_and_lists_its_note(tmp_path):
    from starfinder.dataset import PipelineConfig
    from starfinder.dataset.tiled import _detection_statistics, _merged_detection
    fov = golden_dataset(tmp_path).fov("FOV_001")
    image = np.stack([channel(NOISE), mad_zero(), channel(NOISE), FULL], axis=-1)
    pipeline = PipelineConfig(spot_finding=LocalMaximaConfig())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        statistics = {"detection": _detection_statistics(fov, pipeline.spot_finding, image),
                      "reference_metadata": {"frame_id": "noise-estimation"}}
    emitted = floor_warnings(caught)
    direct = detect(image, LocalMaximaConfig(channel_labels=CHANNELS))[0]
    merged = _merged_detection(fov, pipeline, statistics, direct.diagnostics, direct.spots, records=[])
    assert merged.diagnostics["noise"] == direct.diagnostics["noise"]
    assert merged.diagnostics["noise"]["ch01"]["mad_floored"] is True
    # The listed floor note is the string the global task emitted (it names the reference round); W-789's above-minimum
    # notes are as at the base: listed without a round and not emitted.
    assert len(emitted) == 1 and merged.diagnostics["warnings"].count(emitted[0]) == 1
    expected = tuple(emitted[0] if "MAD floor" in n else n.replace("round 'round1', ", "")
                     for n in direct.diagnostics["warnings"])
    assert merged.diagnostics["warnings"] == expected
