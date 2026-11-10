"""The memory of global registration (W-742): the same results with fewer volume-sized arrays.

W-742 lowered the peak of the global task, set by the translation of whole rounds, without changing a result:
``phase_correlate`` releases its float32 copies after their transforms and forms the cross-power spectrum in place,
an integer translation is written straight into its output, ``_validate_image`` checks finiteness plane by plane, and
``FOV._register_round`` computes the application QC before it resamples the round and releases the moving signals
then. These tests check the values against the formulas they replaced (exact: integer shifts, copies and the same
ufuncs on the same operands), bound the traced peaks, and pin the exactness fixture's shifts and molecule table at the
base revision (``test/data/expected/test_registration_global_memory/``).
"""
from __future__ import annotations

import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from starfinder.image import ImageMetadata, InvalidImageError, _validate_image
from starfinder.registration import TransformChain, TranslationTransform, WarpConfig, apply_transform
from starfinder.registration._chain import resample
from starfinder.registration._translation import _shift_into, _zero_wrapped_edges, apply_shift, phase_correlate

pytestmark = [pytest.mark.registration, pytest.mark.core]

EXPECTED = Path(__file__).parent / "data" / "expected" / "test_registration_global_memory"


def _traced_peak(function, *args, **kwargs):
    """(result, peak bytes traced by tracemalloc during the call above what was traced before it)."""
    tracing = tracemalloc.is_tracing()
    if not tracing:
        tracemalloc.start()
    try:
        start = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        result = function(*args, **kwargs)
        peak = tracemalloc.get_traced_memory()[1] - start
    finally:
        if not tracing:
            tracemalloc.stop()
    return result, peak


def _roll_and_zero(volume, shift, dtype):
    """The integer path of apply_shift before W-742: np.roll, the wrapped regions zeroed, then astype."""
    if all(s == 0 for s in shift):
        result = volume.copy()
    else:
        result = np.roll(volume, shift=shift, axis=(0, 1, 2))
        _zero_wrapped_edges(result, shift)
    return result.astype(dtype, copy=False)


def _volumes(shape, seed):
    rng = np.random.default_rng(seed)
    reference = rng.uniform(0, 100, shape)
    reference[tuple(n // 3 for n in shape)] += 5000.0
    return reference


# --- phase_correlate ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("shape,displacement", [((12, 40, 48), (2, -5, 7)), ((9, 33, 31), (-3, 4, -6))])
def test_phase_correlate_gives_the_out_of_place_formula_s_shift(shape, displacement):
    from scipy.fft import irfftn, rfftn
    reference = _volumes(shape, 742)
    moving = np.roll(reference, displacement, axis=(0, 1, 2)) + np.random.default_rng(1).uniform(0, 1, shape)
    f, m = rfftn(reference.astype(np.float32)), rfftn(moving.astype(np.float32))
    product = f * np.conj(m)
    in_place = f.copy()
    conjugate = m.copy()
    np.conjugate(conjugate, out=conjugate)
    np.multiply(in_place, conjugate, out=in_place)
    # The in-place cross-power spectrum is the out-of-place one, bit for bit.
    np.testing.assert_array_equal(in_place, product)
    cc = irfftn(product, s=shape)
    index = np.unravel_index(np.argmax(np.abs(cc)), shape)
    expected = tuple(float(-(i if i < (n + 1) // 2 else i - n)) for i, n in zip(index, shape))
    assert phase_correlate(reference, moving) == expected == tuple(float(d) for d in displacement)


def test_phase_correlate_holds_at_most_about_three_float32_volumes():
    shape = (16, 64, 64)
    reference = _volumes(shape, 7)
    moving = np.roll(reference, (1, -2, 3), axis=(0, 1, 2))
    shift, peak = _traced_peak(phase_correlate, reference, moving)
    assert shift == (1.0, -2.0, 3.0)
    volume = reference.size * 4
    # The transform of fixed, the float32 copy of moving and its transform (complex64 half spectra: 33/32 of a
    # float32 volume each); before W-742 about six float32 volumes were traced at once.
    assert peak <= 3.5 * volume, f"peak {peak / volume:.2f} float32 volumes"


# --- integer translation --------------------------------------------------------------------------------------------

SHIFTS = [(0, 0, 0), (1, -2, 3), (-3, 5, -1), (0, 7, 0), (6, 0, 0), (-6, 1, 1), (2, -40, 0)]


@pytest.mark.parametrize("dtype,output_dtype", [(np.uint16, "input"), (np.uint8, "float32"), (np.float32, "input"),
                                                (np.float64, "float64"), (np.int16, "float64")])
@pytest.mark.parametrize("shift", SHIFTS)
def test_integer_shift_equals_roll_and_zero(dtype, output_dtype, shift):
    rng = np.random.default_rng(3)
    volume = (rng.uniform(-50, 250, (6, 20, 24))).astype(dtype)
    target = {"input": volume.dtype, "float32": np.float32, "float64": np.float64}[output_dtype]
    expected = _roll_and_zero(volume, shift, target)
    actual = apply_shift(volume, tuple(float(s) for s in shift), output_dtype=output_dtype)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    # Into a strided channel view of a ZYXC array, as resample and apply_transform write it.
    channels = np.full((*volume.shape, 3), 99, dtype=target)
    _shift_into(volume, tuple(float(s) + 1e-7 for s in shift), channels[..., 1])
    np.testing.assert_array_equal(channels[..., 1], expected)
    assert (channels[..., 0] == 99).all() and (channels[..., 2] == 99).all()


def test_fractional_shift_is_unchanged_by_the_output_buffer():
    volume = _volumes((6, 20, 24), 11)
    shift = (0.5, -1.25, 2.0)
    expected = apply_shift(volume, shift)
    output = np.empty_like(volume)
    _shift_into(volume, shift, output)
    np.testing.assert_array_equal(output, expected)


def _translation(shape, displacement):
    metadata = ImageMetadata("w742", spacing_zyx=(1.0, 1.0, 1.0))
    return TranslationTransform(displacement, reference_shape_zyx=shape, moving_shape_zyx=shape,
                                reference_metadata=metadata, moving_metadata=metadata)


def test_resample_and_apply_transform_write_the_translation_without_a_copy():
    shape = (16, 64, 64)
    signal = _volumes(shape, 5)
    rounds = np.random.default_rng(6).integers(0, 4000, (*shape, 4), dtype=np.uint16)
    transform = _translation(shape, (-2.0, 3.0, -4.0))
    chain = TransformChain((transform,))
    shift = (2, -3, 4)
    after, peak = _traced_peak(resample, [signal], chain, WarpConfig(output_dtype="float64"))
    np.testing.assert_array_equal(after[0], _roll_and_zero(signal, shift, np.float64))
    # The float64 output and one plane of the finiteness check; before W-742 also a rolled copy of the volume.
    assert peak <= 1.2 * signal.nbytes, f"peak {peak / signal.nbytes:.2f} float64 volumes"
    images = resample([rounds], chain, WarpConfig())[0]
    applied = apply_transform(rounds, transform, config=WarpConfig())
    for c in range(rounds.shape[-1]):
        expected = _roll_and_zero(rounds[..., c], shift, np.uint16)
        np.testing.assert_array_equal(images[..., c], expected)
        np.testing.assert_array_equal(applied[..., c], expected)


# --- _validate_image ------------------------------------------------------------------------------------------------

def test_validate_image_checks_finiteness_plane_by_plane():
    image = np.ones((8, 64, 64, 4), dtype=np.float32)
    _, peak = _traced_peak(_validate_image, image)
    # One Boolean plane (64 × 64 × 4 bytes) instead of a Boolean image (8 such planes).
    assert peak <= image.size // 4
    integers = np.ones((8, 64, 64, 4), dtype=np.uint16)
    _, peak = _traced_peak(_validate_image, integers)
    assert peak < 4096
    for bad in (np.nan, np.inf, -np.inf):
        image[-1, -1, -1, -1] = bad
        with pytest.raises(InvalidImageError):
            _validate_image(image)
    with pytest.raises(InvalidImageError):
        _validate_image(np.ones((2, 3, 3), dtype=complex))


# --- the exactness fixture against the base revision ----------------------------------------------------------------

#: The shifts of exactness_fov's moving rounds at the base revision 0231258 (integer-valued: exact).
EXACTNESS_SHIFTS = {"round2": (1.0, 3.0, 0.0), "round3": (0.0, -2.0, 1.0), "round4": (-2.0, 2.0, -1.0)}


@pytest.mark.dataset
def test_exactness_fov_keeps_the_base_revision_s_shifts_and_molecules(tmp_path):
    from starfinder import _layout
    from starfinder.dataset.config import CheckpointConfig

    from . import tiled_fixtures as F
    from .test_workflow_tiled import _adapted
    name = F.write_exactness_fov(tmp_path / "in")
    adapted = _adapted(tmp_path / "in", tmp_path / "out", name)
    fov = adapted.dataset.fov(F.FOV)
    fov.run(adapted.pipeline, execution=adapted.execution,
            checkpoints=CheckpointConfig(stages=("candidates", "pre_qc"), overwrite=True))
    shifts = {name: tuple(float(v) for v in results[0].transform.displacement_zyx)
              for name, results in fov.registration_results.items()}
    assert shifts == EXACTNESS_SHIFTS
    # The application QC, now computed before the round is resampled, reads the same signals as the one translation
    # step's QC: the same values.
    for name in EXACTNESS_SHIFTS:
        estimation, application = fov.registration_attempts[name][-2:]
        assert application["record"] == "application" and application["outcome"] == "succeeded"
        assert application["qc"]["values"] == estimation["qc"]["values"]
    fov.export_spots()
    actual = pd.read_csv(_layout.fov_dir(adapted.dataset.output_root, F.FOV) / "molecules.csv",
                         keep_default_na=False)
    expected = pd.read_csv(EXPECTED / "exactness_molecules.csv", keep_default_na=False)
    assert list(actual.columns) == list(expected.columns)
    assert len(actual) == len(expected) == 62
    for column in expected.columns:
        if column == "qc_score":
            np.testing.assert_allclose(actual[column].to_numpy(np.float64), expected[column].to_numpy(np.float64),
                                       rtol=1e-9, atol=1e-12)
        else:
            assert actual[column].astype(str).tolist() == expected[column].astype(str).tolist(), column
