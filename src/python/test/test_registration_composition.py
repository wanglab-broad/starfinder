"""Composition of registration steps into one pull map and the one final resampling (W-256).

The four analytic examples of docs/registration-contract.md ("Analytic
composition examples"): TransformChain.pull_field() at the named points, with
tolerance 1e-12 voxels, and the value the wrong order would give. Then the
final resampling: an integer translation does not quantize a later affine step.
"""
import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from starfinder.dataset import Dataset, PipelineConfig, RegistrationRecipe, RegistrationStep, RoundState
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.registration import (AffineConfig, AffineTransform, BSplineTransform, DemonsConfig,
    DenseDisplacementTransform, TransformChain, TranslationConfig, TranslationTransform,
    UnsupportedTransformOperationError, WarpConfig, apply_transform)
from starfinder.registration._elastix import _index_matrix

TOLERANCE = 1e-12
REFERENCE, MOVING = ImageMetadata("test/reference"), ImageMetadata("test/moving")


def geometry(shape):
    return dict(reference_shape_zyx=shape, moving_shape_zyx=shape, reference_metadata=REFERENCE,
                moving_metadata=MOVING)


def affine(a, b, shape):
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = a, b
    return AffineTransform(matrix, **geometry(shape))


def translation(displacement, shape):
    return TranslationTransform(displacement, **geometry(shape))


def assert_field_at(chain, table):
    """pull_field() at each named point equals the expected u(p); returns the field."""
    field = chain.pull_field()
    assert field.displacement_zyx.dtype == np.float64
    assert field.displacement_zyx.shape == (*chain.reference_shape_zyx, 3)
    for point, expected in table.items():
        np.testing.assert_allclose(field.displacement_zyx[point], expected, rtol=0, atol=TOLERANCE)
    return field


def test_translation_then_affine_pull_field():
    shape = (8, 32, 32)
    first = translation((1, 4, -2), shape)
    second = affine([[1, 0, 0], [0, 1, 0.1], [0, 0, 1]], (0, 0.5, 0), shape)
    field = assert_field_at(TransformChain((first, second)), {(0, 0, 0): (1, 4.5, -2), (2, 10, 20): (1, 6.5, -2)})
    z, y, x = np.indices(shape)
    np.testing.assert_allclose(field.displacement_zyx[..., 1], 4.5 + 0.1 * x, rtol=0, atol=TOLERANCE)
    # The wrong order T2(T1(p)) is the chain with its steps reversed.
    assert_field_at(TransformChain((second, first)), {(0, 0, 0): (1, 4.3, -2), (2, 10, 20): (1, 6.3, -2)})


def test_affine_then_dense_pull_field():
    shape = (8, 32, 32)
    first = affine(np.diag([1, 1.1, 0.9]), (0, -2, 3), shape)
    y = np.indices(shape, dtype=np.float64)[1]
    u2 = np.zeros((*shape, 3))
    u2[..., 2] = 0.05 * y
    second = DenseDisplacementTransform(u2, **geometry(shape))
    expected = {(0, 0, 0): (0, -2, 3), (4, 20, 10): (0, 0, 2.9), (2, 30, 31): (0, 1, 1.25)}
    assert_field_at(TransformChain((first, second)), expected)
    # The wrong order evaluates u2 at T1(p): (0, 0.1 y - 2, -0.1 x + 0.05 (1.1 y - 2) + 3).
    wrong = {(0, 0, 0): (0, -2, 2.9), (4, 20, 10): (0, 0, 3), (2, 30, 31): (0, 1, 1.45)}
    for point, value in wrong.items():
        z, yy, xx = point
        analytic = (0, 0.1 * yy - 2, -0.1 * xx + 0.05 * (1.1 * yy - 2) + 3)
        np.testing.assert_allclose(analytic, value, rtol=0, atol=TOLERANCE)
        assert max(abs(a - b) for a, b in zip(value, expected[point])) > 0.05
    # A dense field can only be the last (local) step.
    with pytest.raises(UnsupportedTransformOperationError, match="only the last transform"):
        TransformChain((second, first))


def test_z1_translation_then_affine_pull_field():
    shape = (1, 32, 32)
    first = translation((0, 3, -1), shape)
    second = affine([[1, 0, 0], [0, 0.98, 0.05], [0, -0.05, 0.98]], (0, 0.2, -0.4), shape)
    field = assert_field_at(TransformChain((first, second)),
                            {(0, 0, 0): (0, 3.2, -1.4), (0, 10, 20): (0, 4.0, -2.3), (0, 31, 0): (0, 2.58, -2.95)})
    assert field.displacement_zyx.shape == (1, 32, 32, 3)
    assert np.all(field.displacement_zyx[..., 0] == 0)
    assert_field_at(TransformChain((second, first)),
                    {(0, 0, 0): (0, 3.09, -1.53), (0, 10, 20): (0, 3.89, -2.43), (0, 31, 0): (0, 2.47, -3.08)})


def test_physical_rigid_to_index_matrix():
    theta = np.deg2rad(2.0)
    c, s = np.cos(theta), np.sin(theta)
    rotation_zyx = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    spacing = np.array([2.0, 0.5, 0.5])
    # The backend's physical matrix is in XYZ order: P R P.
    matrix = _index_matrix(rotation_zyx[::-1, ::-1], np.zeros(3), np.zeros(3), spacing)
    expected = np.diag(1 / spacing) @ rotation_zyx @ np.diag(spacing)
    np.testing.assert_allclose(matrix[:3, :3], expected, rtol=0, atol=TOLERANCE)
    np.testing.assert_allclose(matrix[:3, 3], 0, rtol=0, atol=TOLERANCE)
    np.testing.assert_array_equal(np.round(matrix[:3, :3], 6),
                                  [[0.999391, 0, -0.008725], [0, 1, 0], [0.139598, 0, 0.999391]])
    # Not orthogonal with anisotropic spacing: why the index matrix is stored beside the physical parameters.
    assert not np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-3)
    chain = TransformChain((AffineTransform(matrix, **geometry((8, 32, 32))),))
    assert_field_at(chain, {(4, 10, 20): expected @ (4, 10, 20) - (4, 10, 20)})


def test_every_transform_pulls_from_reference_to_moving():
    """Every transform type reports reference_to_moving; a translation with displacement d pulls from p + d."""
    shape, d = (4, 8, 8), (1.0, -2.0, 3.0)
    t = translation(d, shape)
    bspline = BSplineTransform((4, 4, 4), (0.0, 0.0, 0.0), (3.0, 3.0, 2.0), tuple(np.eye(3).ravel()),
                               np.zeros((3, 4, 4, 4)), (1.0, 1.0, 1.0), **geometry(shape))
    for transform in (t, affine(np.eye(3), (0, 0, 0), shape), bspline,
                      DenseDisplacementTransform(np.zeros((*shape, 3)), **geometry(shape)), TransformChain((t,))):
        assert transform.direction == "reference_to_moving"
    field = TransformChain((t,)).pull_field().displacement_zyx
    np.testing.assert_array_equal(field, np.broadcast_to(d, (*shape, 3)))
    # A ramp that is unique per voxel: the resampled ramp is the exact shift, zero where p + d leaves the grid.
    ramp = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    expected = np.zeros(shape)
    expected[:3, 2:, :5] = ramp[1:, :6, 3:]
    np.testing.assert_array_equal(apply_transform(ramp, t, config=WarpConfig()), expected)
    np.testing.assert_array_equal(apply_transform(ramp, TransformChain((t,)), config=WarpConfig(backend="scipy")),
                                  expected)


def test_a_chain_of_translations_reduces_to_one_translation():
    shape = (4, 8, 8)
    chain = TransformChain((translation((-1, 2, -0.5), shape), translation((1, -3, -1), shape)))
    assert chain.translation().displacement_zyx == (0.0, -1.0, -1.5)
    assert TransformChain((translation((-1, 0, 0), shape), affine(np.eye(3), (0, 0, 0), shape))).translation() is None
    with pytest.raises(IncompatibleGeometryError, match="one grid shape"):
        TransformChain((translation((0, 0, 0), shape), translation((0, 0, 0), (4, 8, 9))))
    with pytest.raises(UnsupportedTransformOperationError, match="nonempty"):
        TransformChain(())


def test_the_translation_backend_applies_only_chains_of_translations():
    shape = (4, 8, 8)
    image = np.arange(4 * 8 * 8, dtype=np.uint16).reshape(shape)
    shift = TransformChain((translation((-1, 2, 0), shape), translation((0, -1, -1), shape)))
    np.testing.assert_array_equal(apply_transform(image, shift, config=WarpConfig()),
                                  apply_transform(image, translation((-1, 1, -1), shape), config=WarpConfig()))
    # A dense backend samples a chain of translations with its own boundary policy.
    nearest = apply_transform(image, shift, config=WarpConfig(backend="scipy", boundary_mode="nearest"))
    z, y, x = np.indices(shape)
    np.testing.assert_array_equal(nearest, image[np.clip(z - 1, 0, 3), np.clip(y + 1, 0, 7), np.clip(x - 1, 0, 7)])
    mixed = TransformChain((translation((-1, 0, 0), shape), affine(np.eye(3), (0, 0, 0), shape)))
    with pytest.raises(UnsupportedTransformOperationError, match="translation backend"):
        apply_transform(image, mixed, config=WarpConfig())


# --- One final resampling -------------------------------------------------------------------

SHAPE = (8, 32, 32)
DISPLACEMENT = (1.0, 4.0, -2.0)
A = np.array([[1.0, 0.0, 0.0], [0.013, 0.97, 0.061], [0.0, -0.047, 1.029]])
B = np.array([0.0, 0.37, -0.23])


def textured_volume():
    rng = np.random.default_rng(256)
    z, y, x = np.indices(SHAPE, dtype=np.float64)
    smooth = 20000 + 15000 * np.sin(x / 3.1) * np.cos(y / 4.3) + 3000 * np.cos(z / 2.0)
    channels = [smooth + rng.normal(0, 800, SHAPE), 30000 - 0.5 * smooth + rng.normal(0, 800, SHAPE)]
    return np.clip(np.rint(np.stack(channels, axis=-1)), 0, 65535).astype(np.uint16)


def test_integer_translation_does_not_quantize_a_later_affine_step(tmp_path, monkeypatch):
    """FOV.run with a (translation, affine) recipe equals one map_coordinates resampling at Phi(p), cast once."""
    import starfinder.registration as registration

    def fixed(reference, moving, *, config, reference_metadata, moving_metadata):
        shape = dict(reference_shape_zyx=reference.shape, moving_shape_zyx=moving.shape,
                     reference_metadata=reference_metadata, moving_metadata=moving_metadata)
        if isinstance(config, TranslationConfig):
            transform = TranslationTransform(DISPLACEMENT, **shape)
        else:
            matrix = np.eye(4)
            matrix[:3, :3], matrix[:3, 3] = A, B
            transform = AffineTransform(matrix, **shape)
        return registration.RegistrationResult(transform, registration.RegistrationDiagnostics(
            config.method, "fixture", config), WarpConfig())

    monkeypatch.setattr(registration, "estimate_transform", fixed)
    dataset = Dataset(tmp_path, tmp_path / "out", "composition", "sample", "out",
                      rounds=RoundState(sequencing_rounds=["round1", "round2"], reference_round="round1"),
                      channel_order=("ch00", "ch01"))
    fov = dataset.fov("FOV_001")
    original = textured_volume()
    fov.images = {"round1": original.copy(), "round2": original.copy()}
    fov.metadata = {name: ImageMetadata(f"FOV_001/{name}") for name in fov.images}
    fov.run(PipelineConfig(registration=RegistrationRecipe(
        (RegistrationStep(TranslationConfig()), RegistrationStep(AffineConfig())))))
    registered = fov.images["round2"]
    assert fov.registration_record["application"]["round2"] == WarpConfig(backend="scipy")
    # The analytic composite pull points Phi(p) = A p + b + d, sampled once and cast once.
    z, y, x = np.indices(SHAPE, dtype=np.float64)
    points = np.stack([A[i, 0] * z + A[i, 1] * y + A[i, 2] * x + B[i] + DISPLACEMENT[i] for i in range(3)])
    expected = np.stack([map_coordinates(original[..., c], points, order=1, mode="constant", cval=0,
                                         output=np.float64) for c in range(2)], axis=-1)
    expected = np.clip(np.rint(expected), 0, 65535).astype(np.uint16)
    np.testing.assert_array_equal(registered, expected)
    # The per-step (two-pass) result rounds after the translation and differs.
    first, second = (r.transform for r in fov.registration_results["round2"])
    two_pass = apply_transform(apply_transform(original, first, config=WarpConfig()), second,
                               config=WarpConfig(backend="scipy"))
    assert not np.array_equal(registered, two_pass)


def test_recipe_steps_follow_the_allowed_sequences():
    for steps in ((TranslationConfig(),), (TranslationConfig(), AffineConfig()), (DemonsConfig(),),
                  (TranslationConfig(), DemonsConfig())):
        RegistrationRecipe(tuple(map(RegistrationStep, steps)))
    for steps in ((DemonsConfig(), TranslationConfig()), (DemonsConfig(), DemonsConfig())):
        with pytest.raises(ValueError, match="global steps followed by at most one local step"):
            RegistrationRecipe(tuple(map(RegistrationStep, steps)))
    with pytest.raises(ValueError, match="at least one step"):
        RegistrationRecipe(())
    with pytest.raises(TypeError, match="RegistrationStep entries"):
        RegistrationRecipe((TranslationConfig(),))
