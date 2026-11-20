"""The steps of a segmentation input recipe: rows ST1 to ST8 (W-891).

design/validation/segmentation-input-recipe.md, "Steps (ST)": the two new registered steps
``gaussian`` and ``clahe``, the radii in µm of ``white_tophat`` and ``reconstruction``, the fit
region of ``percentile_normalization``, the steps inside ``FOV.run``, the refusals of the
recipe and the registry. Every expected value comes from a formula or a library call written
in the test; tolerances follow docs/contributing.md ("Expected values and tolerances").
Every fixture is at most 1 × 64 × 64 voxels.
"""
import math

import numpy as np
import pytest

from starfinder.image import ImageMetadata
from starfinder.preprocessing import (PREPROCESSING_METHODS, Background3DConfig, ClaheConfig, GaussianConfig,
                                      HistogramMatchingConfig, MinMaxNormalizationConfig,
                                      PercentileNormalizationConfig, PreprocessingRecipe, PreprocessingStep,
                                      ReconstructionConfig, ScalarBackgroundConfig, StepContext, TophatConfig,
                                      correct_background, filter_tophat, run_step)
from starfinder.segmentation import (FlamingoEnhancementConfig, InputChannel, InputRecipe, ReferenceGrid, RecipeStep,
                                     SeededWatershedConfig, SegmentationRun, apply_input_recipe)

from . import docs_pages
from .test_segmentation_golden import fixture

pytestmark = [pytest.mark.segmentation, pytest.mark.preprocessing, pytest.mark.validation, pytest.mark.contract]

F32 = {"rtol": 1e-6, "atol": 1e-6}
F64 = {"rtol": 1e-9, "atol": 1e-12}
SPACING = (1.0, 0.1, 0.1)


def metadata(spacing=SPACING, name="recipe"):
    return ImageMetadata(name, spacing_zyx=spacing, spatial_unit="micrometer")


def step(image, config, spacing=SPACING, mask=None):
    return run_step(image, config, StepContext("nuclear", "cell", metadata(spacing), fit_mask=mask))


def truncated_kernel_variance(sigma):
    """``Σ k² w_k / Σ w_k`` of skimage's sampled Gaussian: ``w_k = exp(−k²/(2σ²))``, |k| ≤ int(4σ + 0.5)."""
    radius = int(4 * sigma + 0.5)
    k = np.arange(-radius, radius + 1, dtype=np.float64)
    w = np.exp(-k ** 2 / (2 * sigma ** 2))
    return float((k ** 2 * w).sum() / w.sum())


def spatial_variance(image, spacing, axis):
    """The variance in µm² of an image's intensity along Y (axis 1) or X (axis 2); positions (i + ½)·s."""
    values = np.asarray(image, np.float64)[0]
    profile = values.sum(axis=1 if axis == 1 else 0)
    position = (np.arange(profile.size) + 0.5) * spacing
    mean = (position * profile).sum() / profile.sum()
    return float(((position - mean) ** 2 * profile).sum() / profile.sum())


# --- ST1: gaussian on a delta ---------------------------------------------------------------------------

def test_st1_gaussian_on_a_delta():
    from skimage.filters import gaussian
    delta = np.zeros((1, 63, 63), np.float32)
    delta[0, 31, 31] = 1.0
    config = GaussianConfig(0.3)
    from starfinder.preprocessing.contrast import gaussian_used
    sigma = gaussian_used(config, metadata(), delta.shape)["sigma_px_zyx"]
    np.testing.assert_allclose(sigma, [0.0, 0.3 / 0.1, 0.3 / 0.1], **F64)
    assert sigma[0] == 0.0                                       # an axis of length 1 is not smoothed
    result = step(delta, config)
    assert result.image.dtype == np.float32 and result.fitted == {} and result.diagnostics == {}
    np.testing.assert_allclose(result.image, gaussian(delta, sigma=(0, 3, 3)), **F32)
    assert result.image.sum() == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    expected = truncated_kernel_variance(3.0)
    assert expected == pytest.approx(8.995245, rel=1e-6, abs=1e-6)   # the validation page's scratch value
    for axis in (1, 2):
        moment = spatial_variance(result.image, 1.0, axis)       # in pixels²
        assert moment == pytest.approx(expected, rel=1e-6, abs=1e-6)


# --- ST2: the same µm on two grids --------------------------------------------------------------------

def blob(spacing, n, sigma0=0.5):
    """A continuous Gaussian blob of σ₀ µm at the centre of a square of n·s µm, sampled at (i + ½)·s, float32."""
    centre = n * spacing / 2
    position = (np.arange(n) + 0.5) * spacing
    y, x = np.meshgrid(position, position, indexing="ij")
    return np.exp(-((y - centre) ** 2 + (x - centre) ** 2) / (2 * sigma0 ** 2)).astype(np.float32)[np.newaxis]


def test_st2_gaussian_means_the_same_in_um_on_two_grids():
    variances = {}
    for spacing, n in ((0.1, 64), (0.2, 32)):
        image = blob(spacing, n)
        result = step(image, GaussianConfig(0.3), spacing=(1.0, spacing, spacing))
        sigma_px = 0.3 / spacing
        for axis in (1, 2):
            # The variances of a convolution add: the sampled blob's, and the truncated kernel's times s².
            expected = spatial_variance(image, spacing, axis) + truncated_kernel_variance(sigma_px) * spacing ** 2
            got = spatial_variance(result.image, spacing, axis)
            assert got == pytest.approx(expected, **{"rel": 1e-6, "abs": 1e-6}), (spacing, axis)
        variances[spacing] = got
        assert got == pytest.approx(0.5 ** 2 + 0.3 ** 2, abs=1e-3)   # near σ₀² + 0.3² = 0.34 µm²
    # The two grids differ only by the kernel's truncation: at most 0.12 % of 0.09 µm².
    assert abs(variances[0.1] - variances[0.2]) <= 0.0012 * 0.09


# --- ST3: white_tophat and reconstruction in µm -----------------------------------------------------------

@pytest.mark.parametrize("config_type, function", [(TophatConfig, filter_tophat),
                                                    (ReconstructionConfig, correct_background)])
def test_st3_radii_in_um(config_type, function):
    plane = np.ascontiguousarray(fixture()["dapi"][7:8])
    assert plane.dtype == np.uint8
    for radius_um, radius_yx in ((0.25, 3), (0.24, 2)):        # floor(2.5 + ½) = 3 (round(2.5) would give 2)
        result = step(plane, config_type(radius_um=radius_um))
        assert result.image.dtype == np.uint8
        np.testing.assert_array_equal(result.image, function(plane, config=config_type(radius_yx=radius_yx)))
    assert round(2.5) == 2
    with pytest.raises(ValueError, match="needs equal Y and X spacing"):
        step(plane, config_type(radius_um=0.25), spacing=(1.0, 0.1, 0.11))
    with pytest.raises(ValueError, match="exactly one of radius_yx"):
        config_type(radius_yx=2, radius_um=0.2)
    assert config_type() == config_type(radius_yx=3)              # the default in pixels is unchanged
    with pytest.raises(ValueError, match="radius_um needs the image's spacing"):
        function(plane, config=config_type(radius_um=0.25))


def test_st3_a_record_keeps_its_form_without_radius_um():
    """FOV.run's records do not change (spec, "The steps"): a pixel radius records radius_yx alone."""
    from starfinder.preprocessing._lengths import config_record
    assert config_record(TophatConfig()) == {"radius_yx": 3}
    assert config_record(TophatConfig(radius_um=0.25)) == {"radius_yx": None, "radius_um": 0.25}
    assert config_record(FlamingoEnhancementConfig()) == {"flamingo_quantile": 0.005, "nuclear_quantile": 0.001,
                                                          "median_radius_px": 1}


# --- ST4: percentile_normalization with a fit region ------------------------------------------------------

def test_st4_percentile_normalization_with_a_fit_region():
    rng = np.random.default_rng(891)
    image = np.zeros((1, 64, 64), np.float32)
    mask = np.zeros((1, 64, 64), bool)
    mask[0, 8:56, 8:56] = True
    image[mask] = rng.uniform(10.0, 200.0, mask.sum()).astype(np.float32)
    config = PercentileNormalizationConfig(p_low=2.0, p_high=98.0)
    for region in (mask, None):
        values = image[region] if region is not None else image
        low = np.percentile(values, 2.0, method="inverted_cdf")
        high = np.percentile(values, 98.0, method="inverted_cdf")
        result = step(image, config, mask=region)
        np.testing.assert_allclose(result.fitted["low"], [low], **F32)
        np.testing.assert_allclose(result.fitted["high"], [high], **F32)
        expected = np.clip((image.astype(np.float64) - low) / (high - low), 0, 1)
        np.testing.assert_allclose(result.image, expected, **F32)
    # The masked fit is not the canvas's: the zero border pulls the unmasked low percentile to 0.
    assert step(image, config).fitted["low"] == [0.0] and step(image, config, mask=mask).fitted["low"][0] > 10.0
    with pytest.raises(ValueError, match="fit_mask must be a boolean array"):
        step(image, config, mask=mask[:, :32])


# --- ST5: clahe ------------------------------------------------------------------------------------------

def test_st5_clahe():
    from skimage.exposure import equalize_adapthist
    y, x = np.indices((64, 64))
    ramp = (3.0 + 0.5 * y + 0.25 * x).astype(np.float32)[np.newaxis]
    config = ClaheConfig(1.6, clip_limit=0.01)
    from starfinder.preprocessing.contrast import clahe_used
    assert clahe_used(config, metadata(), ramp.shape) == {"kernel_px_zyx": [1, 16, 16]}
    result = step(ramp, config)
    low, high = float(ramp.min()), float(ramp.max())
    expected = equalize_adapthist(np.clip((ramp.astype(np.float64) - low) / (high - low), 0, 1),
                                  kernel_size=(1, 16, 16), clip_limit=0.01, nbins=256).astype(np.float32)
    assert result.image.dtype == np.float32
    np.testing.assert_allclose(result.image, expected, **F32)
    np.testing.assert_allclose(result.fitted["in_range"], [low, high], **F32)
    assert result.diagnostics == {"degenerate_range": False}
    constant = step(np.full((1, 64, 64), 7.0, np.float32), config)
    assert not constant.image.any() and constant.diagnostics == {"degenerate_range": True}
    with pytest.raises(TypeError):
        ClaheConfig()                                               # kernel_um has no default
    with pytest.raises(ValueError, match="kernel_um is required"):
        ClaheConfig(None)


def test_st5_clahe_fits_its_range_on_the_fit_region():
    image = np.zeros((1, 64, 64), np.float32)
    mask = np.zeros((1, 64, 64), bool)
    mask[0, 16:48, 16:48] = True
    image[mask] = np.linspace(5.0, 9.0, mask.sum(), dtype=np.float32)
    image[0, 0, 0] = 100.0                                          # outside the fit region
    result = step(image, ClaheConfig(1.6), mask=mask)
    np.testing.assert_allclose(result.fitted["in_range"], [5.0, 9.0], **F32)


# --- ST6: gaussian and clahe inside FOV.run --------------------------------------------------------------------

def test_st6_the_steps_keep_an_integer_dtype_inside_a_preprocessing_recipe():
    from skimage.exposure import equalize_adapthist
    from skimage.filters import gaussian
    y, x = np.indices((64, 64))
    plane = ((y * 997 + x * 211) % 60000 + 500).astype(np.uint16)[np.newaxis]
    recipe = PreprocessingRecipe((PreprocessingStep(GaussianConfig(0.3)), PreprocessingStep(ClaheConfig(1.6))))
    smoothed = step(plane, recipe.steps[0].config)
    assert smoothed.image.dtype == np.uint16
    float_result = gaussian(plane, sigma=(0, 3, 3), preserve_range=True)
    difference = np.abs(smoothed.image.astype(np.int64) - np.rint(float_result).astype(np.int64))
    # Exact, but for a float within f64 of a half-integer, where np.rint may round either way.
    near_half = np.abs(float_result - np.floor(float_result) - 0.5) <= 1e-9 * np.abs(float_result)
    assert difference.max() <= 1 and not (difference[~near_half]).any()
    equalized = step(smoothed.image, recipe.steps[1].config)
    assert equalized.image.dtype == np.uint16
    values = smoothed.image.astype(np.float64)
    low, high = values.min(), values.max()
    float_result = equalize_adapthist((values - low) / (high - low), kernel_size=(1, 16, 16),
                                      clip_limit=0.01, nbins=256) * 65535
    difference = np.abs(equalized.image.astype(np.int64) - np.rint(float_result).astype(np.int64))
    near_half = np.abs(float_result - np.floor(float_result) - 0.5) <= 1e-9 * np.abs(float_result)
    assert difference.max() <= 1 and not (difference[~near_half]).any()


@pytest.mark.dataset
def test_st6_gaussian_and_clahe_run_in_fov_run(tmp_path):
    """FOV.run applies both registered steps per channel with the round's spacing; its record keeps the config."""
    from dataclasses import replace

    from starfinder.dataset import Dataset, PipelineConfig, RoundState
    from starfinder.io import ImageLoadConfig, save_volume
    from starfinder.synthetic import development_scene_preset, generate_amplicon_scene
    book, config = development_scene_preset("clean", size="small")
    scene = generate_amplicon_scene(book, config=config)
    for name, image in scene.rounds.items():
        meta = replace(scene.round_metadata[name], spacing_zyx=(0.35, 0.1, 0.1), spatial_unit="micrometer")
        for c, channel in enumerate(scene.channel_labels):
            save_volume(image[..., c], tmp_path / name / "FOV_001" / f"{channel}.tif", metadata=meta)
    rounds = RoundState(list(scene.round_labels), reference_round=scene.round_labels[0])
    dataset = Dataset(tmp_path, tmp_path / "out", "dev", "sample", "out", rounds, list(scene.channel_labels))
    recipe = PreprocessingRecipe((PreprocessingStep(GaussianConfig(0.3)), PreprocessingStep(ClaheConfig(1.6))))
    fov = dataset.fov("FOV_001").run(PipelineConfig(load=ImageLoadConfig(channel_labels=tuple(scene.channel_labels)),
                                                    preprocessing=recipe))
    records = fov.preprocessing_record["rounds"][scene.round_labels[0]]
    assert [(r["step"], r["config"]) for r in records] == [
        ("gaussian", {"sigma_um": 0.3}), ("clahe", {"kernel_um": 1.6, "clip_limit": 0.01, "nbins": 256})]
    assert records[0]["input_dtype"] == records[1]["output_dtype"]


# --- ST7: refusals ----------------------------------------------------------------------------------------------

REFUSED = [
    (lambda: RecipeStep(MinMaxNormalizationConfig("uint8", (0, 255))), "step 'min_max_normalization' is refused"),
    (lambda: RecipeStep(HistogramMatchingConfig()), "step 'histogram_matching' is refused"),
    (lambda: RecipeStep(ScalarBackgroundConfig()), "step 'scalar_background' is refused"),
    (lambda: RecipeStep(Background3DConfig(radius_um_zyx=(1.0, 1.0, 1.0))), "step 'background_3d' is refused"),
    (lambda: RecipeStep(PercentileNormalizationConfig(fit="supplied")), "fit: supplied is refused"),
    (lambda: RecipeStep(TophatConfig(radius_yx=3)), "gives radius_yx \\(pixels\\)"),
    (lambda: RecipeStep(ReconstructionConfig()), "gives radius_yx \\(pixels\\)"),
    (lambda: InputRecipe(combine=FlamingoEnhancementConfig(median_radius_px=1)), "gives median_radius_px"),
    (lambda: TophatConfig(radius_yx=2, radius_um=0.2), "exactly one of radius_yx"),
]


@pytest.mark.parametrize("build, message", REFUSED)
def test_st7_refusals(build, message):
    with pytest.raises(ValueError, match=message) as raised:
        build()
    if "is refused" in message or "gives radius_yx" in message:
        # Each step refusal names the step and the rule of the admitted steps.
        assert "an input recipe admits a registered step of scope per_channel" in str(raised.value)


def test_st7_a_role_the_channels_do_not_have_and_a_grid_without_spacing():
    nuclear = (InputChannel("cytoplasm", "round1", "ch00"),)
    recipe = InputRecipe(channels=(RecipeStep(GaussianConfig(0.3), roles=("nuclear",)),))
    with pytest.raises(ValueError, match="step 'gaussian' of stage channels names the role 'nuclear'"):
        SegmentationRun("cell", "cell", nuclear, SeededWatershedConfig(), seeds="nucleus", input_recipe=recipe)
    grid = ReferenceGrid((1, 16, 16), ImageMetadata("no_spacing"), "declared")
    with pytest.raises(ValueError, match="voxel_size_xy and voxel_size_z") as raised:
        apply_input_recipe([("nuclear", np.zeros((1, 16, 16), np.float32), {})], grid,
                           recipe=InputRecipe(channels=(GaussianConfig(0.3),)))
    assert "GaussianConfig.sigma_um" in str(raised.value)


# --- ST8: the registry ---------------------------------------------------------------------------------------------

def test_st8_the_registry_holds_gaussian_and_clahe():
    specs = {spec.name: spec for spec in PREPROCESSING_METHODS.values()}
    for name, config_type in (("gaussian", GaussianConfig), ("clahe", ClaheConfig)):
        spec = PREPROCESSING_METHODS[config_type]
        assert (spec.name, spec.category, spec.scope, spec.dtype_policy) == (name, "contrast", "per_channel",
                                                                            "preserve")
        assert specs[name] is spec
    # The preprocessing reference renders its method table from the registry.
    for page in ("preprocessing-contract.md", "preprocessing-algorithms.md"):
        assert "```{method-table} starfinder.preprocessing.PREPROCESSING_METHODS" in docs_pages.read(page)
    assert math.isfinite(GaussianConfig((0.5, 0.3, 0.3)).sigma_um[0])
