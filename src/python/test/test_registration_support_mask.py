"""The B-spline support mask after earlier steps (W-824).

A B-spline step after earlier steps is estimated on the moving signal resampled through them, which is zero-filled
where their pull leaves the moving grid. elastix then gets a fixed-image mask: the reference voxels whose pull lands
inside the moving grid, eroded by 2 voxels from its edges inside the grid. The fixtures are one smooth random texture (8x48x48), cropped twice from a
larger field so that both rounds have content everywhere; the moving crop is offset by whole voxels in Y and X, so the
true transform is a pure translation and the B-spline's true displacement is zero.
"""
import importlib.util

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from starfinder._records import json_record
from starfinder.dataset import Dataset, RoundState
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.registration import (BSplineConfig, InvalidRegistrationConfigError, RegistrationEstimationError,
                                     RegistrationRecipe, RegistrationStep, TransformChain, TranslationConfig,
                                     TranslationTransform, WarpConfig, apply_transform, estimate_transform)

pytestmark = [
    pytest.mark.registration,
    pytest.mark.skipif(importlib.util.find_spec("itk") is None or importlib.util.find_spec("SimpleITK") is None,
                       reason="requires the registration-elastix and local-registration extras"),
    # SWIG wrappers of itk warn when pytest inspects their builtin types.
    pytest.mark.filterwarnings("ignore:builtin type .* has no __module__ attribute:DeprecationWarning"),
]

SHAPE = (8, 48, 48)
OFFSET_YX = (5, 6)
# 50 iterations per level keeps each estimate near 2 s; the default 200 gives the same picture more slowly.
BSPLINE = BSplineConfig(iterations=50)
META = ImageMetadata("support-mask")


def texture_pair(offset_yx=OFFSET_YX, shape=SHAPE):
    """Reference and moving crops of one smooth texture; moving[q] = reference[q + offset] where both exist."""
    rng = np.random.default_rng(824)
    field = gaussian_filter(rng.random((shape[0], shape[1] + 16, shape[2] + 16)), 2.0)
    (dy, dx), (_, ny, nx) = offset_yx, shape
    return field[:, 8:8 + ny, 8:8 + nx].copy(), field[:, 8 + dy:8 + dy + ny, 8 + dx:8 + dx + nx].copy()


def translated(moving, displacement_zyx):
    """The moving signal resampled through a translation, and that translation's one-step chain."""
    transform = TranslationTransform(displacement_zyx, SHAPE, SHAPE, META, META)
    return apply_transform(moving, transform, config=WarpConfig()), TransformChain((transform,))


def texture_fov(tmp_path):
    reference, moving = (np.rint(v * 60000)[..., None].astype(np.uint16) for v in texture_pair())
    dataset = Dataset(tmp_path, tmp_path / "out", "support", "sample", "out",
                      rounds=RoundState(sequencing_rounds=["round1", "round2"], reference_round="round1"),
                      channel_order=("ch00",))
    fov = dataset.fov("FOV")
    fov.images = {"round1": reference, "round2": moving}
    fov.metadata = {name: META for name in fov.images}
    return fov


def test_translation_then_bspline_recovers_zero_deformation_inside_the_support(tmp_path):
    """FOV.register with (translation, bspline) on a pure whole-voxel translation.

    The B-spline's true displacement is 0. Inside the support of the translated round (y >= 5 and x >= 6), its
    p95 displacement norm is within 0.5 voxel with the mask; at the base revision (aa3bcae, no mask) it was 5.247
    voxels, because elastix fitted the zero-filled border.
    """
    fov = texture_fov(tmp_path)
    fov.register(RegistrationRecipe((RegistrationStep(TranslationConfig()), RegistrationStep(BSPLINE))))
    translation, bspline = fov.registration_chains["round2"].transforms
    assert translation.displacement_zyx == (0.0, -5.0, -6.0)
    u = np.linalg.norm(bspline.dense().displacement_zyx, axis=-1)
    assert np.percentile(u[:, 5:, 6:], 95) <= 0.5
    # The support y >= 5, x >= 6 eroded by 2 voxels at its edges inside the grid (y = 5 and x = 6): y >= 7, x >= 8.
    voxels = 8 * 41 * 40
    expected = dict(used=True, reason=None, voxels=voxels, fraction=pytest.approx(voxels / (8 * 48 * 48),
                                                                                  rel=1e-9, abs=1e-12))
    diagnostics = fov.registration_results["round2"][1].diagnostics
    assert diagnostics.support_mask == expected
    assert diagnostics.backend_parameters["ErodeMask"] == "false"
    attempt = [a for a in fov.registration_attempts["round2"] if a["record"] == "estimation"][1]
    assert attempt["actual_method"] == "bspline" and attempt["support_mask"] == diagnostics.support_mask
    assert "support_mask" not in fov.registration_attempts["round2"][0]
    assert fov.registration_results["round2"][0].diagnostics.support_mask is None


def test_bspline_as_the_first_step_uses_no_mask(tmp_path):
    fov = texture_fov(tmp_path)
    fov.register(RegistrationRecipe((RegistrationStep(BSPLINE),)))
    diagnostics = fov.registration_results["round2"][0].diagnostics
    assert diagnostics.support_mask == dict(used=False, reason="no earlier step", voxels=None, fraction=None)
    assert "ErodeMask" not in diagnostics.backend_parameters
    assert fov.registration_attempts["round2"][0]["support_mask"] == diagnostics.support_mask


def test_mask_support_false_estimates_as_without_earlier_steps():
    reference, moving = texture_pair()
    signal, chain = translated(moving, (0.0, -5.0, -6.0))
    config = BSplineConfig(iterations=10, mask_support=False)
    off = estimate_transform(reference, signal, config=config, reference_metadata=META, moving_metadata=META,
                             earlier_steps=chain)
    alone = estimate_transform(reference, signal, config=config, reference_metadata=META, moving_metadata=META)
    assert off.diagnostics.support_mask == dict(used=False, reason="mask_support=False", voxels=None, fraction=None)
    assert "ErodeMask" not in off.diagnostics.backend_parameters
    # Same elastix call, one thread: the same coefficients.
    np.testing.assert_array_equal(off.transform.coefficients, alone.transform.coefficients)


def test_an_empty_support_mask_raises_with_its_voxel_count():
    """A Y shift of 46 leaves 2 rows of support, which the erosion by 2 voxels removes."""
    reference, moving = texture_pair()
    signal, chain = translated(moving, (0.0, 46.0, 0.0))
    with pytest.raises(RegistrationEstimationError, match=r"support mask is empty: 0 of 18432 voxels"):
        estimate_transform(reference, signal, config=BSPLINE, reference_metadata=META, moving_metadata=META,
                           earlier_steps=chain)


def test_a_round_shifted_wholly_out_of_the_grid_raises_the_empty_mask_error():
    """A Y shift of 48 leaves no support at all, and the resampled moving signal is all zeros.

    The empty mask is refused before the constant-signal check, so the error names the mask and its voxel count;
    at the base (aa3bcae) and at f7eb3a9 this raised "constant registration signal" instead.
    """
    reference, moving = texture_pair()
    signal, chain = translated(moving, (0.0, 48.0, 0.0))
    assert np.ptp(signal) == 0
    with pytest.raises(RegistrationEstimationError, match=r"support mask is empty: 0 of 18432 voxels"):
        estimate_transform(reference, signal, config=BSPLINE, reference_metadata=META, moving_metadata=META,
                           earlier_steps=chain)


def test_a_mask_too_small_for_elastix_raises_with_its_voxel_count():
    """Shifts of 45 in Y and X leave a 3x3 support per plane, its corner voxel on each of 8 planes after erosion.

    elastix's grid sampler for the optimizer's parameter estimation finds none of the 8 voxels; the error names
    the mask and keeps elastix's reason, and nothing falls back to an unmasked estimate.
    """
    reference, moving = texture_pair()
    signal, chain = translated(moving, (0.0, 45.0, 45.0))
    with pytest.raises(RegistrationEstimationError,
                       match=r"support mask of 8 voxels \(0\.000434 of the grid\) failed: .*No valid voxels"):
        estimate_transform(reference, signal, config=BSPLINE, reference_metadata=META, moving_metadata=META,
                           earlier_steps=chain)


def test_the_support_mask_is_kept_in_records_only_for_bspline():
    reference, moving = texture_pair()
    shift = estimate_transform(reference, moving, config=TranslationConfig(), reference_metadata=META,
                               moving_metadata=META)
    signal, chain = translated(moving, shift.transform.displacement_zyx)
    result = estimate_transform(reference, signal, config=BSPLINE, reference_metadata=META, moving_metadata=META,
                                earlier_steps=chain)
    assert json_record(result.diagnostics)["support_mask"] == dict(used=True, reason=None, voxels=8 * 41 * 40,
                                                                  fraction=result.diagnostics.support_mask["fraction"])
    # The other methods leave it None, and their records keep the keys they had before W-824.
    assert "support_mask" not in json_record(shift.diagnostics)


def test_earlier_steps_are_checked():
    reference, moving = texture_pair()
    with pytest.raises(InvalidRegistrationConfigError, match="earlier_steps must be a TransformChain"):
        estimate_transform(reference, moving, config=BSPLINE, reference_metadata=META, moving_metadata=META,
                           earlier_steps=(TranslationTransform((0, 1, 0), SHAPE, SHAPE, META, META),))
    other = (4, 48, 48)
    chain = TransformChain((TranslationTransform((0, 1, 0), other, other, META, META),))
    with pytest.raises(IncompatibleGeometryError, match="earlier_steps do not end on the moving image's grid"):
        estimate_transform(reference, moving, config=BSPLINE, reference_metadata=META, moving_metadata=META,
                           earlier_steps=chain)
    with pytest.raises(InvalidRegistrationConfigError, match="mask_support must be bool"):
        BSplineConfig(mask_support=1)
