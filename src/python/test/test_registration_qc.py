"""Routine registration QC and truth-field metrics (registration contract, "Routine QC")."""
import numpy as np
import pytest
from skimage.metrics import structural_similarity as skimage_ssim

from starfinder.evaluation.registration import (
    evaluate_displacement_field,
    normalized_cross_correlation,
    registration_qc,
    structural_similarity,
)
from starfinder.image import ImageMetadata
from starfinder.registration import (
    DenseDisplacementTransform,
    InvalidRegistrationConfigError,
    RegistrationDiagnostics,
    RegistrationEstimationError,
    RegistrationQcConfig,
    RegistrationRejectedError,
    TranslationConfig,
    TranslationTransform,
    WarpConfig,
    apply_transform,
)

REF = ImageMetadata("reference")
MOV = ImageMetadata("moving")


def translation(correction, shape):
    return TranslationTransform(correction, shape, shape, REF, MOV)


def ramp_pair(shape, dy):
    """A float64 linear ramp in X and its copy displaced by dy voxels in Y (zero-filled)."""
    reference = np.broadcast_to(np.arange(shape[2], dtype=np.float64), shape).copy()
    moving = np.zeros(shape)
    moving[:, dy:, :] = reference[:, :shape[1] - dy, :]
    return reference, moving


def test_translation_qc_on_matched_domains():
    reference, moving = ramp_pair((1, 32, 32), 5)
    transform = translation((0, -5, 0), (1, 32, 32))
    after = apply_transform(moving, transform, config=WarpConfig())
    qc = registration_qc(reference, moving, after, transform)
    assert qc.values["coverage"] == 27 / 32
    assert qc.counts["valid"] == 27 * 32 and qc.counts["valid_columns"] == 27 * 32
    assert qc.counts["ssim_columns"] == (27 - 6) * (32 - 6)
    assert abs(qc.values["ncc_after"] - 1) <= 1e-12
    assert abs(qc.values["ssim_after"] - 1) <= 1e-12
    # "before" is measured on the same valid overlap and eroded columns.
    valid = np.zeros((1, 32, 32), dtype=bool)
    valid[:, :27] = True
    assert qc.values["ncc_before"] == normalized_cross_correlation(reference, moving, mask=valid).values["ncc"]
    assert qc.values["ncc_before"] < 0.99
    assert qc.values["ncc_gain"] == qc.values["ncc_after"] - qc.values["ncc_before"]
    assert qc.values["ssim_before"] < qc.values["ssim_after"]
    assert qc.config["data_range"] == 25.0  # X from 3 to 28 on the eroded columns
    assert qc.status == "ok" and qc.reasons == {}
    assert qc.details["transform"] == {"kind": "translation", "correction_zyx": [0.0, -5.0, 0.0]}
    assert qc.details["optimizer"] is None and "projections" not in qc.details


def test_undefined_qc_values_are_none_with_a_reason():
    # A constant moving signal: NCC is undefined on both sides.
    reference, _ = ramp_pair((1, 32, 32), 5)
    moving = np.full((1, 32, 32), 7.0)
    transform = translation((0, -5, 0), (1, 32, 32))
    qc = registration_qc(reference, moving, apply_transform(moving, transform, config=WarpConfig()), transform)
    assert qc.values["ncc_before"] is None and qc.values["ncc_after"] is None and qc.values["ncc_gain"] is None
    assert qc.reasons["ncc_after"] == "constant signal over the valid overlap"
    assert qc.values["coverage"] == 27 / 32 and qc.status == "undefined"
    # A signal smaller than the 7x7 window: SSIM is undefined.
    reference, moving = ramp_pair((1, 6, 6), 1)
    transform = translation((0, -1, 0), (1, 6, 6))
    qc = registration_qc(reference, moving, apply_transform(moving, transform, config=WarpConfig()), transform)
    assert qc.values["ssim_before"] is None and qc.values["ssim_after"] is None
    assert qc.reasons["ssim_after"] == "Y or X is smaller than the 7x7 window"
    assert qc.values["coverage"] == 5 / 6 and abs(qc.values["ncc_after"] - 1) <= 1e-12
    # A correction larger than the grid: no valid overlap.
    reference, moving = ramp_pair((1, 32, 32), 5)
    qc = registration_qc(reference, moving, np.zeros_like(moving), translation((0, -40, 0), (1, 32, 32)))
    assert qc.values["coverage"] == 0
    assert qc.values["ncc_before"] is None and qc.values["ncc_after"] is None
    assert qc.values["ssim_before"] is None and qc.values["ssim_after"] is None
    assert qc.reasons["ncc_after"] == "no valid overlap"
    assert qc.reasons["ssim_after"] == "the eroded valid columns are empty"


def test_dense_qc_summary_folds_projections_and_diagnostics():
    shape = (4, 16, 16)
    rng = np.random.default_rng(0)
    reference = rng.random(shape)
    field = np.zeros((*shape, 3))
    field[..., 2] = 0.5
    field[2, 8, 8, 2] = -3.0  # d u_x / d x is -1.75 at (2, 8, 7): one folded voxel
    transform = DenseDisplacementTransform(field, shape, shape, ImageMetadata("r", spacing_zyx=(2, 1, 1)),
                                           ImageMetadata("m", spacing_zyx=(2, 1, 1)))
    diagnostics = RegistrationDiagnostics("demons", "simpleitk", TranslationConfig(), converged=True)
    qc = registration_qc(reference, reference, reference, transform, diagnostics=diagnostics,
                         config=RegistrationQcConfig(projections=True))
    summary = qc.details["transform"]
    # The last column pulls from x = 15.5, outside the grid; (2, 8, 8) pulls from x = 5, inside it.
    assert qc.values["coverage"] == (4 * 16 * 15) / (4 * 16 * 16)
    assert summary["kind"] == "dense"
    assert summary["displacement_voxels"] == {"median": 0.5, "p95": 0.5, "max": 3.0}
    assert summary["displacement_physical"] == {"median": 0.5, "p95": 0.5, "max": 3.0}
    assert summary["fold_fraction"] == 1 / (4 * 16 * 16)
    assert qc.details["optimizer"]["converged"] is True and qc.details["optimizer"]["final_metric_value"] is None
    assert set(qc.details["projections"]) == {"reference", "before", "after"}
    np.testing.assert_array_equal(qc.details["projections"]["after"], reference.max(axis=0))
    with pytest.raises(ValueError, match="reference grid"):
        registration_qc(reference[:2], reference[:2], reference[:2], transform)


def test_displacement_field_error_statistics():
    rng = np.random.default_rng(1)
    truth = rng.normal(scale=2.0, size=(8, 32, 32, 3))
    estimated = truth + np.array([0.3, 0, 0])
    result = evaluate_displacement_field(estimated, truth, mask=None, spacing_zyx=(2, 1, 1))
    for key in ("median", "p95", "max"):
        assert abs(result.values[f"{key}_error"] - 0.3) <= 1e-12
        assert abs(result.values[f"{key}_error_physical"] - 0.6) <= 1e-12
    assert result.units["max_error"] == "voxel" and result.units["max_error_physical"] == "physical"
    mask = np.zeros((8, 32, 32), dtype=bool)
    mask[:4] = True
    estimated[4:] += 10
    masked = evaluate_displacement_field(estimated, truth, mask=mask)
    assert abs(masked.values["max_error"] - 0.3) <= 1e-12 and "max_error_physical" not in masked.values
    empty = evaluate_displacement_field(estimated, truth, mask=np.zeros_like(mask))
    assert empty.values["median_error"] is None and empty.reasons["median_error"] == "empty mask"
    with pytest.raises(ValueError):
        evaluate_displacement_field(estimated[:4], truth, mask=None)


def test_mask_keywords():
    rng = np.random.default_rng(2)
    a, b = rng.random((3, 16, 16)), rng.random((3, 16, 16))
    everything = np.ones(a.shape, dtype=bool)
    unmasked = normalized_cross_correlation(a, b)
    masked = normalized_cross_correlation(a, b, mask=everything)
    assert abs(masked.values["ncc"] - unmasked.values["ncc"]) <= 1e-12 and masked.counts["masked"] == a.size
    assert normalized_cross_correlation(a, b, mask=everything & (np.arange(16) < 1)).values["ncc"] is not None
    single = np.zeros(a.shape, dtype=bool)
    single[0, 0, 0] = True
    assert normalized_cross_correlation(a, b, mask=single).reasons["ncc"] == "fewer than two voxels in the mask"
    # SSIM averaged over the uncropped interior equals scikit-image's cropped mean.
    interior = np.zeros((16, 16), dtype=bool)
    interior[3:13, 3:13] = True
    plane = structural_similarity(a[0], b[0], data_range=1.0, policy="plane", win_size=7, mask=interior)
    assert abs(plane.values["ssim"] - skimage_ssim(a[0], b[0], data_range=1.0, win_size=7)) <= 1e-12
    empty = structural_similarity(a, b, data_range=1.0, policy="mip", mask=np.zeros((16, 16), dtype=bool))
    assert empty.values["ssim"] is None and empty.reasons["ssim"] == "empty mask"
    for bad in (np.ones((16, 16)), np.ones((15, 16), dtype=bool)):
        with pytest.raises(ValueError, match="mask"):
            structural_similarity(a, b, data_range=1.0, policy="mip", mask=bad)
        with pytest.raises(ValueError, match="mask"):
            normalized_cross_correlation(a[0], b[0], mask=bad)


def test_qc_config_and_rejection_error():
    config = RegistrationQcConfig()
    assert (config.min_coverage, config.min_ncc_gain, config.max_fold_fraction, config.max_translation_voxels,
            config.projections) == (None, None, None, None, False)
    RegistrationQcConfig(min_coverage=0.5, min_ncc_gain=-0.1, max_fold_fraction=0, max_translation_voxels=4)
    for bad in (dict(min_coverage=1.5), dict(min_coverage=-0.1), dict(max_fold_fraction=2), dict(min_ncc_gain=np.inf),
                dict(max_translation_voxels=-1), dict(projections=1)):
        with pytest.raises(InvalidRegistrationConfigError):
            RegistrationQcConfig(**bad)
    assert issubclass(RegistrationRejectedError, RegistrationEstimationError)
    with pytest.raises(TypeError):
        registration_qc(np.zeros((1, 8, 8)), np.zeros((1, 8, 8)), np.zeros((1, 8, 8)),
                        translation((0, 0, 0), (1, 8, 8)), config={"projections": True})
