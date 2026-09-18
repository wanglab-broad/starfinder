"""Bounded shared registration contracts; no scientific benchmark execution."""

import importlib.util
import sys
from dataclasses import FrozenInstanceError, asdict

import numpy as np
import pytest

from starfinder.image import ImageMetadata, IncompatibleGeometryError, InvalidImageError
from starfinder.registration import (
    CpdConfig,
    DemonsConfig,
    DenseDisplacementTransform,
    InsufficientLandmarksError,
    InvalidRegistrationConfigError,
    RegistrationBackendUnavailableError,
    TpsConfig,
    TranslationConfig,
    TranslationTransform,
    UnsupportedTransformOperationError,
    WarpConfig,
    apply_transform,
    estimate_transform,
)

REF = ImageMetadata("reference")
MOV = ImageMetadata("moving")


def estimate(reference, moving, config):
    return estimate_transform(
        reference, moving, config=config, reference_metadata=REF, moving_metadata=MOV
    )


@pytest.mark.parametrize("backend", ["scipy_fft", "skimage"])
def test_translation_contract(backend):
    source = np.zeros((1, 15, 17), dtype=np.int16)
    source[0, 7, 8] = 100
    moving = np.roll(source, (0, -2, 3), axis=(0, 1, 2))
    original = moving.copy()
    config = TranslationConfig(backend=backend)
    result = estimate(source, moving, config)
    assert result.transform.correction_zyx == (0, 2, -3)
    assert result.transform.direction == "moving_to_reference"
    assert result.transform.units == "voxel_index"
    assert result.transform.reference_metadata == REF
    assert result.diagnostics.method == "translation"
    assert result.diagnostics.backend == backend
    assert result.diagnostics.effective_config is config
    assert result.diagnostics.converged is None
    assert not hasattr(result.transform, "displacement_zyx")
    images = np.stack([moving, -moving], axis=-1)
    actual = apply_transform(images, result.transform, config=result.application_config)
    np.testing.assert_array_equal(actual, np.stack([source, -source], axis=-1))
    np.testing.assert_array_equal(moving, original)


@pytest.mark.parametrize("backend", ["scipy", "simpleitk"])
def test_dense_pull_direction_and_shared_channels(backend):
    source = np.zeros((8, 12, 14), dtype=np.float32)
    source[4, 6, 7] = -17
    field = np.zeros((*source.shape, 3), dtype=np.float32)
    field[..., 2] = 1
    transform = DenseDisplacementTransform(field, source.shape, source.shape, REF, MOV)
    field[...] = 99  # transform owns its immutable field
    images = np.stack([source, source * 2], axis=-1)
    result = apply_transform(images, transform, config=WarpConfig(backend=backend))
    assert result[4, 6, 6, 0] == -17 and result[4, 6, 6, 1] == -34
    assert transform.direction == "reference_to_moving"
    assert not transform.displacement_zyx.flags.writeable
    assert source[4, 6, 7] == -17


@pytest.mark.parametrize("variant", ["demons", "diffeomorphic", "symmetric", "fast_symmetric"])
def test_demons_identity_and_identity_metadata(variant):
    source = np.random.default_rng(140).random((8, 12, 12)).astype(np.float32)
    before = source.copy()
    config = DemonsConfig(variant=variant, iterations=(1,), pyramid_mode="sitk")
    result = estimate(source, source, config)
    assert result.diagnostics.method == "demons"
    assert result.diagnostics.effective_config.variant == variant
    assert result.diagnostics.backend == result.application_config.backend == "simpleitk"
    assert result.diagnostics.iterations_completed is None
    np.testing.assert_allclose(result.transform.displacement_zyx, 0, atol=1e-6)
    np.testing.assert_allclose(
        apply_transform(source, result.transform, config=result.application_config),
        source,
        atol=1e-6,
    )
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize(
    "config", [TpsConfig(min_matches=4, max_control_points=20), CpdConfig(max_control_points=20)]
)
def test_landmark_identity_and_insufficiency(config):
    # Separated impulses span 3D, avoiding a rank-deficient TPS fit.
    source = np.zeros((16, 32, 32), dtype=np.float32)
    source[np.ix_([4, 8, 12], [6, 14, 22], [6, 14, 22])] = 100
    result = estimate(source, source, config)
    assert result.diagnostics.method == config.method
    assert result.diagnostics.backend == result.application_config.backend == "scipy"
    assert result.transform.displacement_zyx.dtype == np.float32
    assert result.transform.reference_metadata == REF
    assert np.isfinite(result.transform.displacement_zyx).all()
    if isinstance(config, TpsConfig):
        np.testing.assert_allclose(
            apply_transform(source, result.transform, config=result.application_config),
            source,
            atol=1e-3,
        )
    else:
        # EM termination is approximate. This fixture's pre-migration maximum
        # residual is 0.000445 voxel; see external equivalence evidence.
        assert np.max(np.abs(result.transform.displacement_zyx)) < 0.001
    with pytest.raises(InsufficientLandmarksError):
        estimate(np.zeros_like(source), np.zeros_like(source), config)


@pytest.mark.parametrize("config", [DemonsConfig(), TpsConfig(), CpdConfig()])
def test_declared_local_2d_rejection(config):
    with pytest.raises(IncompatibleGeometryError, match="3D"):
        estimate(np.ones((1, 8, 8)), np.ones((1, 8, 8)), config)


@pytest.mark.parametrize(
    "constructor,kwargs",
    [
        (TranslationConfig, {"backend": "invalid"}),
        (TranslationConfig, {"backend": "skimage", "fft_workers": 2}),
        (DemonsConfig, {"iterations": ()}),
        (DemonsConfig, {"variant": "invalid"}),
        (TpsConfig, {"interpolation_order": 6}),
        (CpdConfig, {"outlier_fraction": 1}),
        (CpdConfig, {"kernel_width_voxels": 0}),
        (WarpConfig, {"boundary_mode": "nearest"}),
        (WarpConfig, {"clip_to_dtype": False}),
        (WarpConfig, {"fill_value": float("nan")}),
    ],
)
def test_invalid_config(constructor, kwargs):
    with pytest.raises(InvalidRegistrationConfigError):
        constructor(**kwargs)


def test_frozen_discriminator_and_defaults():
    assert asdict(TranslationConfig())["method"] == "translation"
    with pytest.raises(FrozenInstanceError):
        TranslationConfig().backend = "skimage"
    with pytest.raises(TypeError):
        TpsConfig(method="cpd")
    with pytest.raises(TypeError):
        TranslationConfig(unknown=True)
    assert CpdConfig().detection_noise_sigma == 5
    assert CpdConfig().grid_spacing_voxels == 16


def test_invalid_images_geometry_and_transform():
    source = np.zeros((8, 8, 8))
    with pytest.raises(InvalidImageError):
        estimate(source + np.nan, source, TranslationConfig())
    with pytest.raises(InvalidImageError):
        estimate(source[..., None], source, TranslationConfig())
    with pytest.raises(IncompatibleGeometryError):
        estimate(source, source[:4], TranslationConfig())
    with pytest.raises(IncompatibleGeometryError):
        estimate_transform(
            source,
            source,
            config=TranslationConfig(),
            reference_metadata=REF,
            moving_metadata=ImageMetadata("moving", spacing_zyx=(1, 1, 1)),
        )
    with pytest.raises(UnsupportedTransformOperationError):
        TranslationTransform(
            (0, 0, 0), source.shape, source.shape, REF, MOV, direction="reference_to_moving"
        )
    with pytest.raises(UnsupportedTransformOperationError):
        TranslationTransform((0, 0, 0), source.shape, source.shape, REF, MOV, units="um")
    with pytest.raises(IncompatibleGeometryError):
        DenseDisplacementTransform(np.zeros((8, 8, 8, 2)), source.shape, source.shape, REF, MOV)
    result = estimate(source, source, TranslationConfig())
    with pytest.raises(IncompatibleGeometryError):
        apply_transform(source[:4], result.transform, config=result.application_config)
    with pytest.raises(UnsupportedTransformOperationError):
        apply_transform(source, result.transform, config=WarpConfig(backend="scipy"))


def test_missing_backend_is_explicit(monkeypatch):
    monkeypatch.setitem(sys.modules, "SimpleITK", None)
    with pytest.raises(RegistrationBackendUnavailableError, match="SimpleITK"):
        estimate(np.zeros((8, 8, 8)), np.zeros((8, 8, 8)), DemonsConfig(iterations=(1,)))


def test_removed_public_paths_and_exports():
    import starfinder.registration as registration

    for name in (
        "phase_correlate",
        "apply_shift",
        "register_volume",
        "demons_register",
        "tps_register",
        "cpd_register",
        "matlab_compatible_config",
    ):
        assert not hasattr(registration, name)
    for name in ("phase_correlation", "demons", "pointset", "pyramid", "metrics", "benchmark"):
        assert importlib.util.find_spec("starfinder.registration." + name) is None


def test_fov_cpd_defaults_and_no_fallback(tmp_path, monkeypatch):
    import starfinder.registration as registration
    from starfinder.dataset import LayerState, STARMapDataset

    dataset = STARMapDataset(
        input_root=tmp_path,
        output_root=tmp_path,
        dataset_id="test",
        sample_id="sample",
        output_id="test",
        layers=LayerState(seq=["r1", "r2"], ref="r1"),
    )
    fov = dataset.fov("FOV")
    fov.images = {r: np.zeros((8, 8, 8, 2), dtype=np.uint8) for r in ["r1", "r2"]}
    calls = []

    def fail(reference, moving, *, config, **kwargs):
        calls.append(config)
        raise InsufficientLandmarksError("test insufficient landmarks")

    monkeypatch.setattr(registration, "estimate_transform", fail)
    with pytest.raises(InsufficientLandmarksError):
        fov.local_registration(method="cpd")
    assert len(calls) == 1
    assert isinstance(calls[0], CpdConfig)
    assert calls[0].detection_noise_sigma == 3
    assert calls[0].grid_spacing_voxels == 32
    assert not fov.local_registered


def test_simpleitk_prepares_transform_once_for_channels(monkeypatch):
    import SimpleITK as sitk

    original = sitk.DisplacementFieldTransform
    calls = []

    def prepare(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(sitk, "DisplacementFieldTransform", prepare)
    image = np.ones((4, 6, 8, 4), dtype=np.uint16)
    transform = DenseDisplacementTransform(
        np.zeros((4, 6, 8, 3), dtype=np.float32), image.shape[:3], image.shape[:3], REF, MOV
    )
    np.testing.assert_array_equal(
        apply_transform(image, transform, config=WarpConfig(backend="simpleitk")), image
    )
    assert len(calls) == 1
