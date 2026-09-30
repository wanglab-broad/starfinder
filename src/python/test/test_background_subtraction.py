"""Scalar and 3D background subtraction and the scalar-background histogram shortcut (W-231).

Expectations are hand-computed inverted-CDF percentiles, histograms of the
subtracted volumes, and scipy.ndimage.grey_opening with an ellipsoid written
out explicitly in the test.
"""
from dataclasses import asdict
import json

import numpy as np
import pytest
from scipy import ndimage

from starfinder.dataset import Dataset, PipelineConfig, RoundState
from starfinder.image import ImageMetadata
from starfinder.preprocessing import (PREPROCESSING_METHODS, Background3DConfig, PercentileNormalizationConfig, PreprocessingRecipe,
    PreprocessingStep, ScalarBackgroundConfig, StepContext, merge_histograms, read_supplied_statistics, run_step,
    scalar_background_histograms, subtract_background_3d, subtract_scalar_background, summarize_histograms,
    summary_stage, supplied_section, supplied_statistics, write_supplied_statistics)

from .test_preprocessing_golden import fixture_rounds

CHANNELS = ("ch00", "ch01")
ROUNDS = ("round1", "round2")
DTYPES = (np.uint8, np.uint16)
CONTEXT = StepContext("round1", "round1", ImageMetadata("frame"))


def ellipsoid(rz, ry, rx):
    """(dz/rz)^2 + (dy/ry)^2 + (dx/rx)^2 <= 1, written out; an axis with radius 0 has length 1."""
    dz, dy, dx = np.meshgrid(np.arange(-rz, rz + 1), np.arange(-ry, ry + 1), np.arange(-rx, rx + 1), indexing="ij")
    term = lambda d, r: (d / r) ** 2 if r else np.zeros(d.shape)
    return term(dz, rz) + term(dy, ry) + term(dx, rx) <= 1


def top_hat(x, footprint):
    """max(x - grey_opening(x, footprint), 0) per channel, in int64."""
    return np.stack([np.maximum(x[..., c].astype(np.int64) - ndimage.grey_opening(x[..., c], footprint=footprint), 0)
                     for c in range(x.shape[-1])], axis=-1)


def random_fovs(dtype, seed=0):
    """Three FOVs of two rounds, with channel-specific pedestals so backgrounds differ."""
    rng = np.random.default_rng(seed)
    top = np.iinfo(dtype).max
    fovs = {}
    for i, (depth, spread) in enumerate(((3, top // 4), (2, top // 2), (4, top // 40))):
        fovs[f"FOV_{i + 1:03d}"] = {name: (rng.integers(0, spread, (depth, 9, 7, 2), endpoint=True)
                                           + np.array([top // 20, top // 8])).astype(dtype) for name in ROUNDS}
    return fovs


def dataset(tmp_path):
    return Dataset(tmp_path, tmp_path / "out", "test", "sample", "out",
                   rounds=RoundState(sequencing_rounds=list(ROUNDS), reference_round="round1"), channel_order=CHANNELS)


def resident(ds, fov_id, rounds, spacing=None):
    fov = ds.fov(fov_id)
    for name, volume in rounds.items():
        fov.images[name] = volume.copy()
        fov.metadata[name] = ImageMetadata(f"{fov_id}/{name}", spacing_zyx=spacing)
    return fov


# --- Registration and config ------------------------------------------------------

def test_both_steps_are_registered_per_channel_and_preserving():
    for config_type, name in ((ScalarBackgroundConfig, "scalar_background"), (Background3DConfig, "background_3d")):
        spec = PREPROCESSING_METHODS[config_type]
        assert (spec.name, spec.category, spec.scope, spec.dtype_policy) == (name, "background", "per_channel", "preserve")
    assert asdict(ScalarBackgroundConfig()) == {"percentile": 10.0, "fit": "fov"}
    for bad in (dict(percentile=-1), dict(percentile=100), dict(percentile=float("nan")), dict(percentile=True),
                dict(percentile="10"), dict(fit="median"), dict(fit="mode")):
        with pytest.raises(ValueError):
            ScalarBackgroundConfig(**bad)
    assert Background3DConfig(radius_voxels_zyx=[1, 2, 3]).radius_voxels_zyx == (1, 2, 3)


def test_3d_config_requires_exactly_one_nonnegative_radius():
    for bad in (dict(), dict(radius_um_zyx=(1, 1, 1), radius_voxels_zyx=(1, 1, 1)),
                dict(radius_voxels_zyx=(1, -1, 2)), dict(radius_um_zyx=(0.5, -0.2, 1.0)),
                dict(radius_voxels_zyx=(1.0, 2, 3)), dict(radius_voxels_zyx=(True, 2, 3)), dict(radius_voxels_zyx=(1, 2)),
                dict(radius_um_zyx=(1, float("inf"), 1)), dict(radius_um_zyx="123")):
        with pytest.raises(ValueError):
            Background3DConfig(**bad)


# --- Scalar estimator ----------------------------------------------------------------

def test_scalar_estimator_equals_hand_computed_inverted_cdf_percentiles_with_uint8_ties():
    # 10 voxels; value 3 appears 4 times and value 5 twice. Inverted CDF: the smallest
    # value whose cumulative count reaches p/100 * 10.
    values = np.array([9, 3, 200, 5, 3, 7, 3, 9, 5, 3], dtype=np.uint8)
    x = values.reshape(1, 2, 5)
    hand = {0: 3, 10: 3, 40: 3, 41: 5, 50: 5, 60: 5, 61: 7, 70: 7, 71: 9, 90: 9, 91: 200, 99.9: 200}
    for p, b in hand.items():
        result = run_step(x, ScalarBackgroundConfig(percentile=p), CONTEXT)
        assert result.fitted == {"background": [b]}, p
        assert np.percentile(values, p, method="inverted_cdf") == b  # the documented definition
        np.testing.assert_array_equal(result.image, np.maximum(x.astype(int) - b, 0))
        assert result.image.dtype == np.uint8
    # Ties at a grey level zero more voxels than percentile / 100: at p = 10 four of ten.
    assert run_step(x, ScalarBackgroundConfig(), CONTEXT).diagnostics["zero_fraction"] == [0.4]


def test_scalar_estimator_on_uint16_and_float_arrays():
    x16 = np.array([1000, 2, 65535, 2, 40000, 7, 7, 7], dtype=np.uint16).reshape(2, 2, 2)
    # Sorted: 2 2 7 7 7 1000 40000 65535.
    for p, b in {0: 2, 25: 2, 26: 7, 62.5: 7, 63: 1000, 75: 1000, 76: 40000, 88: 65535}.items():
        result = run_step(x16, ScalarBackgroundConfig(percentile=p), CONTEXT)
        assert result.fitted == {"background": [b]}, p
        np.testing.assert_array_equal(result.image, np.maximum(x16.astype(int) - b, 0))
        assert result.image.dtype == np.uint16
    xf = np.array([0.5, -1.25, 3.0, 3.0, 2.0]).reshape(1, 1, 5)
    # Sorted: -1.25 0.5 2.0 3.0 3.0.
    for dtype in (np.float32, np.float64):
        for p, b in {0: -1.25, 20: -1.25, 21: 0.5, 50: 2.0, 60: 2.0, 61: 3.0}.items():
            result = run_step(xf.astype(dtype), ScalarBackgroundConfig(percentile=p), CONTEXT)
            assert result.fitted == {"background": [b]}, p
            assert result.image.dtype == dtype
            np.testing.assert_array_equal(result.image, np.maximum(xf - b, 0).astype(dtype))


@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.float32, np.float64))
def test_scalar_constant_channel_gives_zeros(dtype):
    x = np.zeros((2, 3, 4, 2), dtype=dtype)
    x[..., 0] = 7
    x[..., 1] = np.arange(24).reshape(2, 3, 4)
    result = run_step(x, ScalarBackgroundConfig(), CONTEXT)
    assert result.fitted == {"background": [7, 2]}
    np.testing.assert_array_equal(result.image[..., 0], 0)
    np.testing.assert_array_equal(result.image[..., 1], np.maximum(x[..., 1].astype(np.float64) - 2, 0))
    assert result.diagnostics["constant_channel"] == [True, False]


def test_scalar_supplied_values_and_rounding_policy():
    x = np.array([3, 4, 5, 6, 250], dtype=np.uint8).reshape(1, 1, 5)
    config = ScalarBackgroundConfig(fit="supplied")
    # A non-integer supplied level: integer output rounds half to even (2.5 -> 2, 3.5 -> 4, 4.5 -> 4).
    supplied = {"fitted": {"round1": {"background": [0.5]}}}
    result = run_step(x, config, StepContext("round1", "round1", ImageMetadata("f"), supplied=supplied))
    np.testing.assert_array_equal(result.image.ravel(), [2, 4, 4, 6, 250])
    assert result.fitted == {"background": [0.5]}
    # Float output is not rounded; a level above the data gives zeros, never a wrapped value.
    np.testing.assert_array_equal(subtract_scalar_background(x.astype(np.float32), config=config,
                                                             fitted={"background": [0.5]}).ravel(),
                                  np.array([2.5, 3.5, 4.5, 5.5, 249.5], dtype=np.float32))
    np.testing.assert_array_equal(subtract_scalar_background(x, config=config, fitted={"background": [300]}), 0)
    with pytest.raises(ValueError, match="no values for round"):
        run_step(x, config, StepContext("round2", "round1", ImageMetadata("f"), supplied=supplied))
    for fitted in ({"background": [1, 2]}, {"background": [float("nan")]}, {"low": [1]}):
        with pytest.raises(ValueError):
            subtract_scalar_background(x, config=config, fitted=fitted)
    with pytest.raises(ValueError, match="requires"):
        subtract_scalar_background(x, config=config)
    with pytest.raises(ValueError, match="only with"):
        subtract_scalar_background(x, fitted={"background": [1]})


def test_input_is_not_modified_and_invalid_input_raises():
    x = np.arange(4 * 6 * 6 * 2, dtype=np.uint16).reshape(4, 6, 6, 2)
    before = x.copy()
    subtract_scalar_background(x)
    subtract_background_3d(x, config=Background3DConfig(radius_voxels_zyx=(1, 1, 1)))
    np.testing.assert_array_equal(x, before)
    for bad in (np.zeros((0, 2, 2), np.uint8), np.array([[[np.nan, 1.0]]])):
        with pytest.raises(ValueError):
            subtract_scalar_background(bad)
        with pytest.raises(ValueError):
            subtract_background_3d(bad, config=Background3DConfig(radius_voxels_zyx=(0, 0, 0)))


# --- Scalar-background histogram shortcut ----------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
def test_fov_shortcut_equals_histograms_of_the_subtracted_volumes(dtype):
    fovs = random_fovs(dtype, seed=1)
    config = ScalarBackgroundConfig(percentile=30)
    after = ({"step": "scalar_background", "config": {"percentile": 30, "fit": "fov"}},)
    derived, direct = [], []
    for fov_id, rounds in fovs.items():
        raw = summarize_histograms(rounds, channel_labels=CHANNELS, fov_id=fov_id)
        derived.append(scalar_background_histograms(raw, config))
        subtracted = {name: subtract_scalar_background(volume, config=config) for name, volume in rounds.items()}
        assert any(np.count_nonzero(v == 0) > 0 for v in subtracted.values())
        direct.append(summarize_histograms(subtracted, channel_labels=CHANNELS, fov_id=fov_id, summarized_after=after))
        np.testing.assert_array_equal(derived[-1].counts, direct[-1].counts)  # per FOV
        assert derived[-1].summarized_after == direct[-1].summarized_after
    merged = merge_histograms(derived)
    np.testing.assert_array_equal(merged.counts, merge_histograms(direct).counts)  # merged counts
    # fit="fov" backgrounds differ per FOV, so a merged summary is rejected.
    with pytest.raises(ValueError, match="before merging"):
        scalar_background_histograms(merge_histograms([summarize_histograms(r, channel_labels=CHANNELS, fov_id=f)
                                                       for f, r in fovs.items()]), config)


@pytest.mark.parametrize("dtype", DTYPES)
def test_supplied_shortcut_on_merged_counts_equals_the_subtracted_volumes(dtype, tmp_path):
    fovs = random_fovs(dtype, seed=2)
    path = tmp_path / "supplied.json"
    scalar = ScalarBackgroundConfig(fit="supplied")
    recipe = PreprocessingRecipe((PreprocessingStep(scalar), PreprocessingStep(PercentileNormalizationConfig(fit="supplied"))),
                                 supplied_statistics=path)
    # Pass 1: histograms of the raw volumes give the scalar background section.
    raw = merge_histograms([summarize_histograms(r, channel_labels=CHANNELS, fov_id=f) for f, r in fovs.items()])
    section = supplied_section(scalar, raw)
    for r, name in enumerate(ROUNDS):
        for c in range(2):
            concatenated = np.concatenate([rounds[name][..., c].ravel() for rounds in fovs.values()])
            assert section["fitted"][name]["background"][c] == np.percentile(concatenated, 10, method="inverted_cdf")
    # The next summary follows from the merged counts without another pass.
    derived = scalar_background_histograms(raw, scalar, supplied=section)
    direct = merge_histograms([summarize_histograms(
        {name: subtract_scalar_background(v, config=scalar, fitted=section["fitted"][name]) for name, v in r.items()},
        channel_labels=CHANNELS, fov_id=f, summarized_after=summary_stage(recipe, "percentile_normalization")[1])
        for f, r in fovs.items()])
    np.testing.assert_array_equal(derived.counts, direct.counts)
    assert derived.summarized_after == direct.summarized_after
    assert (derived.fovs_used, derived.fovs_excluded) == (raw.fovs_used, raw.fovs_excluded)
    # Two passes suffice: the file with both sections drives the application pass.
    normalization = supplied_section(recipe.steps[1].config, derived)
    write_supplied_statistics(supplied_statistics(raw, {"scalar_background": section,
                                                        "percentile_normalization": normalization}), path)
    fov = resident(dataset(tmp_path), "FOV_002", fovs["FOV_002"]).run(PipelineConfig(preprocessing=recipe))
    top = np.iinfo(dtype).max
    for name in ROUNDS:
        for c in range(2):
            b = section["fitted"][name]["background"][c]
            low, high = (normalization["fitted"][name][k][c] for k in ("low", "high"))
            x = np.maximum(fovs["FOV_002"][name][..., c].astype(np.float64) - b, 0)
            np.testing.assert_array_equal(fov.images[name][..., c],
                                          np.rint(np.clip((x - low) / (high - low), 0, 1) * top))
        assert [r["fitted"] for r in fov.preprocessing_record["rounds"][name]] == \
            [section["fitted"][name], normalization["fitted"][name]]


def test_shortcut_faults_raise():
    rounds = random_fovs(np.uint8, seed=3)["FOV_001"]
    raw = summarize_histograms(rounds, channel_labels=CHANNELS, fov_id="FOV_001")
    scalar = ScalarBackgroundConfig(fit="supplied")
    section = supplied_section(scalar, raw)
    for bad in (None, {"fitted": {"round1": section["fitted"]["round1"]}, "params": section["params"]},
                {**section, "params": {"percentile": 20.0}},
                {**section, "summarized_after": [{"step": "white_tophat", "config": {"radius_yx": 3}}]},
                {**section, "fitted": {n: {"background": [1.5, 2]} for n in ROUNDS}},
                {**section, "fitted": {n: {"background": [-1, 2]} for n in ROUNDS}},
                {**section, "fitted": {n: {"background": [1]} for n in ROUNDS}}):
        with pytest.raises(ValueError):
            scalar_background_histograms(raw, scalar, supplied=bad)
    with pytest.raises(ValueError, match="only with"):
        scalar_background_histograms(raw, ScalarBackgroundConfig(), supplied=section)
    # A level at or above the top bin puts every voxel in bin 0.
    top = {**section, "fitted": {n: {"background": [255, 1000]} for n in ROUNDS}}
    counts = scalar_background_histograms(raw, scalar, supplied=top).counts
    assert (counts[..., 0] == raw.counts.sum(axis=-1)).all() and not counts[..., 1:].any()


def test_scalar_supplied_section_is_validated_against_the_recipe(tmp_path):
    rounds = random_fovs(np.uint8, seed=4)["FOV_001"]
    raw = summarize_histograms(rounds, channel_labels=CHANNELS, fov_id="FOV_001")
    section = supplied_section(ScalarBackgroundConfig(percentile=5, fit="supplied"), raw)
    assert section["params"] == {"percentile": 5.0} and section["summarized_after"] == []
    path = tmp_path / "s.json"
    write_supplied_statistics(supplied_statistics(raw, {"scalar_background": section}), path)
    recipe = lambda p: PreprocessingRecipe((PreprocessingStep(ScalarBackgroundConfig(percentile=p, fit="supplied")),),
                                           supplied_statistics=path)
    assert read_supplied_statistics(path, recipe(5), rounds=ROUNDS)["steps"]["scalar_background"] == section
    with pytest.raises(ValueError, match="percentile"):
        read_supplied_statistics(path, recipe(10))
    with pytest.raises(ValueError, match="rounds"):
        read_supplied_statistics(path, recipe(5), rounds=ROUNDS + ("round3",))
    good = supplied_statistics(raw, {"scalar_background": section})
    for change in (lambda s: s["params"].update(percentile=100), lambda s: s["params"].update(extra=1),
                   lambda s: s["fitted"]["round1"].update(background=[1]),
                   lambda s: s["fitted"]["round1"].update(background=[1, "2"]), lambda s: s["fitted"].clear()):
        candidate = json.loads(json.dumps(good))
        change(candidate["steps"]["scalar_background"])
        with pytest.raises(ValueError):
            write_supplied_statistics(candidate, tmp_path / "bad.json")
    with pytest.raises(ValueError, match="reference_round"):
        supplied_section(ScalarBackgroundConfig(fit="supplied"), raw, reference_round="round1")


# --- 3D background -----------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("radii", ((1, 2, 3), (2, 1, 1), (0, 2, 1), (1, 0, 0)))
def test_3d_output_equals_independent_grey_opening_with_explicit_ellipsoid(dtype, radii):
    rng = np.random.default_rng(sum(radii))
    x = rng.integers(0, np.iinfo(dtype).max, (7, 11, 13, 2), endpoint=True, dtype=dtype)
    footprint = ellipsoid(*radii)
    assert footprint.shape == tuple(2 * r + 1 for r in radii)
    result = run_step(x, Background3DConfig(radius_voxels_zyx=radii), CONTEXT)
    assert result.image.dtype == dtype
    np.testing.assert_array_equal(result.image, top_hat(x, footprint))
    assert result.fitted == {"radius_voxels_zyx": list(radii)}
    assert result.diagnostics["footprint_voxels"] == int(footprint.sum())
    # A single-channel ZYX volume gives the same result as that channel.
    np.testing.assert_array_equal(subtract_background_3d(x[..., 1], config=Background3DConfig(radius_voxels_zyx=radii)),
                                  result.image[..., 1])


def test_explicit_ellipsoid_for_radii_1_2_3():
    # Written out slice by slice: the z = 0 plane is the (2, 3) ellipse, z = +-1 the centre voxel only.
    plane = np.array([[0, 0, 0, 1, 0, 0, 0],
                      [0, 1, 1, 1, 1, 1, 0],
                      [1, 1, 1, 1, 1, 1, 1],
                      [0, 1, 1, 1, 1, 1, 0],
                      [0, 0, 0, 1, 0, 0, 0]], dtype=bool)
    single = np.zeros((5, 7), dtype=bool)
    single[2, 3] = True
    np.testing.assert_array_equal(ellipsoid(1, 2, 3), np.stack([single, plane, single]))


@pytest.mark.parametrize("dtype", DTYPES)
def test_3d_background_of_constant_plus_small_punctum_is_the_constant(dtype):
    c = 40
    x = np.full((9, 21, 21), c, dtype=dtype)
    # A punctum of 7 voxels (centre and its six face neighbours), far smaller than the footprint.
    x[4, 10, 10] += 150
    for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        x[4 + dz, 10 + dy, 10 + dx] += 60
    radii = (2, 3, 3)
    # Opening removes the punctum, so the background is c everywhere and the output is x - c.
    np.testing.assert_array_equal(ndimage.grey_opening(x, footprint=ellipsoid(*radii)), c)
    y = subtract_background_3d(x, config=Background3DConfig(radius_voxels_zyx=radii))
    np.testing.assert_array_equal(y, x.astype(int) - c)
    punctum = x > c
    assert punctum.sum() == 7 and (y[~punctum] == 0).all() and y[4, 10, 10] == 150
    # A constant channel gives zeros and the constant_channel diagnostic.
    result = run_step(np.stack([x, np.full_like(x, c)], axis=-1), Background3DConfig(radius_voxels_zyx=radii), CONTEXT)
    np.testing.assert_array_equal(result.image[..., 1], 0)
    assert result.diagnostics["constant_channel"] == [False, True]


def test_um_radii_convert_with_spacing_by_rounding(tmp_path):
    x = np.random.default_rng(5).integers(0, 255, (9, 16, 16, 2), dtype=np.uint8)
    spacing = (0.5, 0.25, 0.25)
    # 1.0 / 0.5 = 2; 0.625 / 0.25 = 2.5 -> 2 and 0.875 / 0.25 = 3.5 -> 4 (half to even).
    config = Background3DConfig(radius_um_zyx=(1.0, 0.625, 0.875))
    result = run_step(x, config, StepContext("round1", "round1", ImageMetadata("f", spacing_zyx=spacing)))
    assert result.fitted == {"radius_voxels_zyx": [2, 2, 4]}
    np.testing.assert_array_equal(result.image, top_hat(x, ellipsoid(2, 2, 4)))
    config = Background3DConfig(radius_um_zyx=(0.6, 0.3, 0.8))  # 1.2 -> 1, 1.2 -> 1, 3.2 -> 3
    result = run_step(x, config, StepContext("round1", "round1", ImageMetadata("f", spacing_zyx=spacing)))
    assert result.fitted == {"radius_voxels_zyx": [1, 1, 3]}
    # Through FOV.run the spacing comes from the round's ImageMetadata.
    fov = resident(dataset(tmp_path), "FOV_001", {n: x for n in ROUNDS}, spacing=spacing)
    fov.run(PipelineConfig(preprocessing=PreprocessingRecipe((PreprocessingStep(config),))))
    np.testing.assert_array_equal(fov.images["round2"], result.image)
    assert fov.preprocessing_record["rounds"]["round2"][0]["step"] == "background_3d"
    assert fov.preprocessing_record["rounds"]["round2"][0]["fitted"] == {"radius_voxels_zyx": [1, 1, 3]}


def test_3d_failures_raise():
    x = np.zeros((5, 8, 10), dtype=np.uint16)
    with pytest.raises(ValueError, match="spacing"):
        subtract_background_3d(x, config=Background3DConfig(radius_um_zyx=(1, 1, 1)))
    with pytest.raises(ValueError, match="spacing"):
        run_step(x, Background3DConfig(radius_um_zyx=(1, 1, 1)), CONTEXT)
    # 2r + 1 must not exceed the volume along any axis: 5 fits Z = 5, 7 does not.
    subtract_background_3d(x, config=Background3DConfig(radius_voxels_zyx=(2, 3, 4)))
    for radii in ((3, 1, 1), (1, 4, 1), (1, 1, 5)):
        with pytest.raises(ValueError, match="larger than the volume"):
            subtract_background_3d(x, config=Background3DConfig(radius_voxels_zyx=radii))
    with pytest.raises(ValueError, match="larger than the volume"):
        subtract_background_3d(x, config=Background3DConfig(radius_um_zyx=(3.0, 1, 1)),
                               metadata=ImageMetadata("f", spacing_zyx=(1, 1, 1)))
    # All-zero radii leave the image unfiltered: the opening is the identity.
    np.testing.assert_array_equal(subtract_background_3d(x + 9, config=Background3DConfig(radius_voxels_zyx=(0, 0, 0))), 0)


# --- Diagnostics on the golden fixture ------------------------------------------------

@pytest.mark.parametrize("config", (ScalarBackgroundConfig(), Background3DConfig(radius_voxels_zyx=(1, 4, 4))),
                         ids=("scalar_background", "background_3d"))
def test_diagnostics_on_the_golden_fixture_round2(config):
    x = fixture_rounds()["round2"]
    result = run_step(x, config, StepContext("round2", "round1", ImageMetadata("golden/round2")))
    diagnostics = result.diagnostics
    assert {"zero_fraction", "median", "mad", "noise_threshold", "mad_zero", "constant_channel"} <= set(diagnostics)
    for c in range(x.shape[-1]):
        values = result.image[..., c].astype(np.float64)
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        assert diagnostics["zero_fraction"][c] == np.mean(values == 0)
        assert diagnostics["median"][c] == median and diagnostics["mad"][c] == mad
        assert diagnostics["noise_threshold"][c] == median + 5 * mad * 1.4826
        assert diagnostics["mad_zero"][c] == (mad == 0)
    assert diagnostics["constant_channel"] == [False] * 4
    assert json.dumps(diagnostics) and json.dumps(result.fitted)
    if type(config) is ScalarBackgroundConfig:
        # 10th percentile of uint16 data with few ties: about 10 % of voxels become zero.
        assert all(0.1 <= z < 0.11 for z in diagnostics["zero_fraction"])
