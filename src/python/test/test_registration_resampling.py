"""Bounded numerical regressions, separate from the registration API migration."""

import numpy as np
import pytest

from starfinder.registration import apply_shift, phase_correlate, phase_correlate_skimage
from starfinder.registration.pointset import apply_tps_deformation


@pytest.mark.parametrize("shape,displacement", [
    ((1, 5, 7), (0, -2, 3)),  # odd correlation peak exactly n//2
    ((5, 1, 7), (2, 0, -3)),
    ((5, 7, 1), (-2, 3, 0)),
    ((1, 1, 1), (0, 0, 0)),
    ((4, 6, 8), (2, 3, 4)),  # even half-period is ambiguous
    ((4, 6, 8), (-2, -3, -4)),
    ((5, 7, 9), (1, -2, 3)),
])
@pytest.mark.parametrize("backend", [phase_correlate, phase_correlate_skimage])
def test_tiny_translation_alignment(shape, displacement, backend):
    reference = np.random.default_rng(139).uniform(0, 10, shape).astype(np.float32)
    moving = np.roll(reference, displacement, axis=(0, 1, 2))
    expected = list(displacement)
    for axis, n in enumerate(shape):
        if n % 2 == 0 and abs(expected[axis]) == n // 2:
            # Preserve each backend's native half-period convention.
            expected[axis] = n // 2 if backend is phase_correlate else -n // 2
    detected = backend(reference, moving)
    np.testing.assert_array_equal(detected, expected)
    aligned = apply_shift(moving, tuple(-s for s in detected))
    interior = tuple(slice(max(0, -int(s)), min(n, n - int(s)))
                     for n, s in zip(shape, detected))
    np.testing.assert_array_equal(aligned[interior], reference[interior])


@pytest.mark.parametrize("dtype,tolerance", [(np.float32, 2e-6), (np.float64, 1e-12)])
@pytest.mark.parametrize("shift", [-0.5, 0.5])
def test_signed_fourier_translation(dtype, tolerance, shift):
    # Odd length has no Nyquist ambiguity; a sinusoid has an analytic shift.
    n = 9
    x = np.arange(n)
    source = (-3 + np.cos(2 * np.pi * x / n))[None, None, :].astype(dtype)
    expected = (-3 + np.cos(2 * np.pi * (x - shift) / n))[None, None, :]
    expected[..., 0 if shift > 0 else -1] = 0
    result = apply_shift(source, (0, 0, shift))
    assert result.dtype == dtype
    np.testing.assert_allclose(result, expected, atol=tolerance, rtol=0)
    assert result.min() < 0  # legacy abs(ifft) made these values positive


def test_signed_constant_and_impulse():
    source = np.full((1, 3, 9), -7, dtype=np.float32)
    np.testing.assert_allclose(apply_shift(source, (0, 0, .5))[..., 1:], -7, atol=2e-6)
    impulse = np.zeros((1, 3, 9), dtype=np.float32)
    impulse[0, 1, 4] = 20
    positive = apply_shift(impulse, (0, 0, .5))
    negative = apply_shift(-impulse, (0, 0, .5))
    np.testing.assert_allclose(negative, -positive, atol=2e-6)
    assert positive.min() < 0  # signed Fourier ringing is retained


def test_fourier_integer_output_saturates_ringing():
    source = np.zeros((1, 1, 9), dtype=np.uint8)
    source[..., 3:6] = 255
    floating = apply_shift(source, (0, 0, .5), output_dtype="float64")
    assert floating.min() < 0 and floating.max() > 255
    integer = apply_shift(source, (0, 0, .5))
    np.testing.assert_array_equal(integer, np.clip(np.rint(floating), 0, 255).astype(np.uint8))
    assert integer.dtype == source.dtype
    assert floating.dtype == np.float64


@pytest.mark.parametrize("shift", [-20, -9, 9, 20, -20.5, 20.5])
def test_translation_outside_grid_is_zero(shift):
    assert not apply_shift(np.ones((1, 3, 9)), (0, 0, shift)).any()


def _native_warp(backend):
    if backend == "scipy":
        return apply_tps_deformation
    pytest.importorskip("SimpleITK")
    from starfinder.registration.demons import apply_deformation
    return apply_deformation


@pytest.mark.parametrize("backend", ["scipy", "simpleitk"])
@pytest.mark.parametrize("dtype,values", [
    (np.uint8, [0, 1, 2, 3, 4]),
    (np.int16, [-4, -3, -2, -1, 0]),
    (np.uint16, [65531, 65532, 65533, 65534, 65535]),
    (np.float32, [-4, -3, -2, -1, 0]),
])
def test_native_fractional_rounding_and_float_output(backend, dtype, values):
    source = np.broadcast_to(np.array(values, dtype=dtype), (3, 3, 5)).copy()
    original = source.copy()
    field = np.zeros((*source.shape, 3), dtype=np.float64)
    field[..., 2] = .5
    warp = _native_warp(backend)
    expected = (np.array(values[:-1], dtype=float) + values[1:]) / 2
    result = warp(source, field)
    floating = warp(source, field, output_dtype="float64")
    expected_integer = np.rint(expected) if np.dtype(dtype).kind in "iu" else expected
    np.testing.assert_array_equal(result[1, 1, :-1], expected_integer)
    np.testing.assert_array_equal(floating[1, 1, :-1], expected)
    assert result.dtype == dtype and floating.dtype == np.float64
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(warp(source, field * 0), source)


@pytest.mark.parametrize("method", ["tps", "cpd", "local"])
def test_channels_use_same_final_conversion(method, monkeypatch):
    if method == "local":
        pytest.importorskip("SimpleITK")
        from starfinder.registration import demons as module
        estimator = "demons_register"
    else:
        from starfinder.registration import pointset as module
        estimator = method + "_register"
    source = np.broadcast_to(np.arange(5, dtype=np.int16), (3, 3, 5)).copy()
    images = np.stack([source, -source], axis=-1)
    field = np.zeros((*source.shape, 3), dtype=np.float64)
    field[..., 2] = .5
    # Isolate application from estimation; exercise the actual native resampler.
    monkeypatch.setattr(module, estimator, lambda *args, **kwargs: field)
    register = getattr(module, "register_volume_" + method)
    actual, returned = register(images, source, source)
    floating, _ = register(images, source, source, output_dtype="float32")
    np.testing.assert_array_equal(actual[1, 1, :-1, 0], [0, 2, 2, 4])
    np.testing.assert_array_equal(actual[1, 1, :-1, 1], [0, -2, -2, -4])
    np.testing.assert_array_equal(floating[1, 1, :-1, 0], [.5, 1.5, 2.5, 3.5])
    assert floating.dtype == np.float32 and actual.dtype == images.dtype
    assert returned is field


def test_multichannel_translation_float_output():
    from starfinder.registration import register_volume
    source = np.zeros((1, 5, 7), dtype=np.int16)
    source[0, 2, 3] = 13
    moving = np.roll(source, 1, axis=2)
    images = np.stack([moving, -moving], axis=-1)
    result, shift = register_volume(images, source, moving, output_dtype="float64")
    assert shift == (0, 0, 1) and result.dtype == np.float64
    np.testing.assert_array_equal(result, np.stack([source, -source], axis=-1))


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16, np.int64, np.uint64])
def test_final_cast_clips_without_overflow(dtype):
    from starfinder.registration._resampling import _cast_warp_output
    limits = np.iinfo(dtype)
    values = np.array([float(limits.min) - 1024, .5, 1.5, float(limits.max) + 1024])
    result = _cast_warp_output(values, dtype)
    np.testing.assert_array_equal(result, np.array([limits.min, 0, 2, limits.max], dtype=dtype))
