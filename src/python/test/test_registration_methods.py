"""Rigid, affine and B-spline (elastix) and Z=1 demons registration (registration task group 3).

Known-answer scenes use the benchmark appearance of registration_scene_preset
at the development sizes, as in the W-244 comparison, with seeds 100 to 102.
Errors are evaluate_displacement_field of the estimated pull field against
forward_displacement over the valid overlap of the estimated transform.
"""
import hashlib
import importlib.util
import math
import subprocess
import sys
import types
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pytest

from starfinder.dataset import Dataset, RegistrationStep, RoundState

from starfinder.evaluation.registration import _valid_overlap, evaluate_displacement_field, registration_qc
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.registration import (
    REGISTRATION_METHODS,
    AffineConfig,
    AffineTransform,
    BSplineConfig,
    BSplineTransform,
    CpdConfig,
    DemonsConfig,
    DenseDisplacementTransform,
    RegistrationBackendUnavailableError,
    RegistrationEstimationError,
    RigidConfig,
    TpsConfig,
    TranslationConfig,
    WarpConfig,
    apply_transform,
    estimate_transform,
)
from starfinder.registration import _elastix
from starfinder.registration._methods import _check_shape
from starfinder.synthetic import (BENCHMARK_PRESETS, DEVELOPMENT_SIZES, GeometryConfig, forward_displacement,
                                  generate_formed_scene, registration_scene_preset)

pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("itk") is None or importlib.util.find_spec("SimpleITK") is None,
                       reason="requires the registration-elastix and local-registration extras"),
    # SWIG wrappers of itk warn when pytest inspects their builtin types.
    pytest.mark.filterwarnings("ignore:builtin type .* has no __module__ attribute:DeprecationWarning"),
]

SEEDS = (100, 101, 102)
# Amplicons per scene: the W-244 development presets (20 on 9x32x32, 8 on 1x32x32) for V2 and V3, and
# 128 per 64x64 plane (512 on 16x64x64) for the V4 and V5 scenes (operator decision on W-255, 2026-09-30,
# pending Jiahao's confirmation). At the W-244 development density (32 per plane, 128 on 16x64x64) the
# provisional V4 and V5 bounds are exceeded; see docs/registration-algorithms.md.
COUNTS = {(9, 32, 32): 20, (1, 32, 32): 8, (16, 64, 64): 512, (1, 64, 64): 128}
MISSING_EXTRA = ("registration method 'affine' requires itk; install the 'registration-elastix' extra "
                 "(starfinder[registration-elastix])")


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    """ITK and SimpleITK at one thread, as the project contract requires."""
    import itk
    import SimpleITK as sitk
    itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(1)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


@contextmanager
def preset(shape):
    """A temporary benchmark preset of the given shape (benchmark appearance, W-244 style)."""
    name = "w255-" + "x".join(str(n) for n in shape)
    BENCHMARK_PRESETS[name] = dict(shape_zyx=shape, count=COUNTS[shape], seed=0, fovs=1, e2e_shift=(0, 0),
                                   registration_shift=(0, 0), genes=12)
    try:
        yield name
    finally:
        del BENCHMARK_PRESETS[name]


def scene(shape, seed, *, deformation="shift", geometry=None):
    """(reference ch00, moving ch00, truth pull field) of one registration pair.

    deformation names a DEFORMATION_PRESETS entry or "shift"; a supplied
    geometry replaces the geometry of the moving round.
    """
    with preset(shape) as name:
        codebook, config = registration_scene_preset(name, deformation)
    config = replace(config, seed=seed, **({} if geometry is None else dict(geometry=geometry)))
    formed = generate_formed_scene(codebook, config=config)
    record = next(t for t in formed.provenance["transforms"].values() if t["round_label"] == deformation)
    return (formed.rounds["reference"][..., 0].astype(np.float64), formed.rounds[deformation][..., 0].astype(np.float64),
            forward_displacement(record, shape))


def rigid_geometry(shape, seed):
    """The W-244 rigid case: a rotation of at most 3 degrees about the grid centre in YX and a shift of at most
    3 voxels (2 in Z)."""
    rng = np.random.default_rng([244, seed])
    theta = np.deg2rad(rng.uniform(-3.0, 3.0))
    shift = rng.uniform(-1, 1, 3) * np.array([2.0 if shape[0] > 1 else 0.0, 3.0, 3.0])
    rotation = np.eye(3)
    rotation[1:, 1:] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    return GeometryConfig(translation_enabled=True, translations_zyx=[[0.0] * 3, shift.tolist()],
                          affine_enabled=True, affine_zyx=[np.zeros((3, 3)), rotation - np.eye(3)],
                          reference_round="reference")


def polynomial_geometry():
    """A supplied quadratic term of peak 3 voxels per component: dy = 3 u_x^2, dx = 3 u_y^2."""
    coefficients = np.zeros((2, 3, 6))
    coefficients[1, 1, 2] = 3.0  # Y row, xx
    coefficients[1, 2, 1] = 3.0  # X row, yy
    return GeometryConfig(polynomial_enabled=True, polynomial_zyx=coefficients, reference_round="reference")


def gaussian_geometry(shape):
    """One local Gaussian control of magnitude 2 voxels and scale 8 (random direction from the seed)."""
    center = tuple(float(f * (n - 1)) for f, n in zip((.5, .4, .6), shape))
    return GeometryConfig(local_enabled=True, centers_zyx=(center,), scales=(8.0,), local_magnitude=2.0,
                          reference_round="reference")


def metadata(spacing=None):
    return ImageMetadata("reference", spacing_zyx=spacing), ImageMetadata("moving", spacing_zyx=spacing)


def estimate(reference, moving, config, spacing=None):
    reference_metadata, moving_metadata = metadata(spacing)
    return estimate_transform(reference, moving, config=config, reference_metadata=reference_metadata,
                              moving_metadata=moving_metadata)


def errors(transform, truth):
    """evaluate_displacement_field values over the valid overlap of the estimated pull field."""
    u = transform.dense().displacement_zyx if hasattr(transform, "dense") else transform.displacement_zyx
    return evaluate_displacement_field(u, truth, mask=_valid_overlap(u, u.shape[:3])).values


def digest(transform):
    """SHA-256 of the stored transform parameters."""
    h = hashlib.sha256()
    if isinstance(transform, AffineTransform):
        h.update(transform.matrix_zyx.tobytes())
        h.update(np.asarray(transform.physical["parameters"], dtype=np.float64).tobytes())
    else:
        for values in (transform.grid_size_xyz, transform.grid_origin_xyz, transform.grid_spacing_xyz,
                       transform.grid_direction_xyz):
            h.update(np.asarray(values, dtype=np.float64).tobytes())
        h.update(transform.coefficients.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------- registry and lazy import
def test_registry_entries_of_the_new_methods():
    table = {spec.name: (config_type, spec.step_kind, spec.dimensions, spec.min_shape_zyx, spec.transform_kind,
                         spec.space, tuple((d.module, d.distribution, d.extra) for d in spec.requires))
             for config_type, spec in REGISTRATION_METHODS.items()}
    elastix = ("itk", "itk-elastix", "registration-elastix")
    simpleitk = ("SimpleITK", "SimpleITK", "local-registration")
    assert table["rigid"] == (RigidConfig, "global", {2, 3}, (4, 16, 16), "affine", "physical", (elastix,))
    assert table["affine"] == (AffineConfig, "global", {2, 3}, (4, 16, 16), "affine", "physical", (elastix,))
    assert table["bspline"] == (BSplineConfig, "local", {2, 3}, (4, 16, 16), "bspline", "physical",
                                (elastix, simpleitk))
    assert table["demons"] == (DemonsConfig, "local", {2, 3}, (4, 4, 4), "dense", "index", (simpleitk,))


def test_config_defaults_follow_the_algorithm_page():
    assert (RigidConfig().metric, RigidConfig().histogram_bins, RigidConfig().iterations, RigidConfig().samples,
            RigidConfig().levels, RigidConfig().random_seed) == ("mattes", 32, 200, 4096, None, 1)
    assert (AffineConfig().metric, AffineConfig().iterations, AffineConfig().samples, AffineConfig().levels,
            AffineConfig().random_seed) == ("ncc", 200, 4096, None, 1)
    assert (BSplineConfig().metric, BSplineConfig().grid_spacing_physical, BSplineConfig().iterations,
            BSplineConfig().samples, BSplineConfig().levels, BSplineConfig().random_seed) == ("ncc", None, 200, 4096,
                                                                                             None, 1)


def test_importing_registration_does_not_import_itk():
    code = ("import sys, starfinder, starfinder.registration, starfinder.dataset, starfinder.evaluation; "
            "print('itk' in sys.modules)")
    output = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    assert output.strip() == "False"


def test_affine_without_the_extra_names_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "itk", None)
    reference = np.random.default_rng(0).random((1, 32, 32))
    with pytest.raises(RegistrationBackendUnavailableError) as info:
        estimate(reference, reference, AffineConfig())
    assert str(info.value) == MISSING_EXTRA
    # Through a registration step: the error propagates, is recorded and nothing is substituted.
    fov = fov_with(reference, reference)
    with pytest.raises(RegistrationBackendUnavailableError) as info:
        fov.register(RegistrationStep(AffineConfig()))
    assert str(info.value) == MISSING_EXTRA
    assert_single_failed_attempt(fov, "affine", "RegistrationBackendUnavailableError", MISSING_EXTRA)


def test_itk_without_elastix_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "itk", types.ModuleType("itk"))
    reference = np.random.default_rng(0).random((1, 32, 32))
    with pytest.raises(RegistrationBackendUnavailableError, match=r"'registration-elastix' extra"):
        estimate(reference, reference, RigidConfig())


# --------------------------------------------------------------------- known answers
@pytest.mark.parametrize("size,spacing", [("small", None), ("z1", None), ("small", (1.0, 2.0, 2.0))])
@pytest.mark.parametrize("seed", SEEDS)
def test_rigid_recovers_a_known_rigid_map(size, spacing, seed):
    shape = DEVELOPMENT_SIZES[size]
    reference, moving, truth = scene(shape, seed, geometry=rigid_geometry(shape, seed))
    result = estimate(reference, moving, RigidConfig(), spacing)
    values = errors(result.transform, truth)
    assert values["median_error"] <= 0.25 and values["p95_error"] <= 0.5, values
    physical = result.transform.physical
    assert physical["transform"] == "EulerTransform"
    assert len(physical["parameters"]) == len(physical["parameter_names"]) == (3 if shape[0] == 1 else 6)
    assert result.diagnostics.spacing_source == ("unknown_unit" if spacing is None else "metadata")
    if shape[0] == 1:
        assert np.array_equal(result.transform.matrix_zyx[0], [1, 0, 0, 0])
        assert np.array_equal(result.transform.matrix_zyx[:, 0], [1, 0, 0, 0])


@pytest.mark.parametrize("size", ["small", "z1"])
@pytest.mark.parametrize("seed", SEEDS)
def test_affine_recovers_a_known_affine_map(size, seed):
    reference, moving, truth = scene(DEVELOPMENT_SIZES[size], seed, deformation="linear_small")
    result = estimate(reference, moving, AffineConfig())
    values = errors(result.transform, truth)
    assert values["median_error"] <= 0.25 and values["p95_error"] <= 0.5, values
    assert result.transform.physical["transform"] == "AffineTransform"


def capture_registration(monkeypatch):
    """Wrap the elastix call so a test can reuse the elastix result and the moving image."""
    captured = {}
    original = _elastix._register

    def wrapper(itk, fixed, moving, parameter_object, directory):
        captured.update(itk=itk, moving=moving,
                        registration=original(itk, fixed, moving, parameter_object, directory))
        return captured["registration"]

    monkeypatch.setattr(_elastix, "_register", wrapper)
    return captured


@pytest.fixture(scope="module")
def bspline_runs():
    """One elastix B-spline estimate per V4 scene, with the elastix result it came from."""
    runs = {}
    for shape in ((16, 64, 64), (1, 64, 64)):
        reference, moving, truth = scene(shape, 100, geometry=polynomial_geometry())
        with pytest.MonkeyPatch.context() as monkeypatch:
            captured = capture_registration(monkeypatch)
            result = estimate(reference, moving, BSplineConfig())
        runs[shape] = dict(result=result, truth=truth, **captured)
    return runs


@pytest.mark.parametrize("shape", [(16, 64, 64), (1, 64, 64)])
def test_bspline_recovers_a_known_deformation(shape, bspline_runs):
    run = bspline_runs[shape]
    transform, truth = run["result"].transform, run["truth"]
    assert isinstance(transform, BSplineTransform) and len(transform.grid_size_xyz) == (2 if shape[0] == 1 else 3)
    u = transform.dense().displacement_zyx
    values = errors(transform, truth)
    identity = evaluate_displacement_field(np.zeros_like(truth), truth, mask=_valid_overlap(u, shape)).values
    assert identity["median_error"] > 1  # by construction
    assert values["median_error"] <= 0.5 and values["p95_error"] <= 1.0, (values, identity)
    assert values["p95_error"] < identity["p95_error"], (values, identity)


@pytest.mark.parametrize("shape", [(16, 64, 64), (1, 64, 64)])
def test_bspline_grid_evaluation_equals_the_transformix_field(shape, bspline_runs, tmp_path):
    """BSplineTransform.dense() (SimpleITK on the stored coefficients) against transformix on the same result."""
    run = bspline_runs[shape]
    transform, itk = run["result"].transform, run["itk"]
    u = transform.dense().displacement_zyx
    field = np.asarray(itk.array_from_image(itk.transformix_deformation_field(
        run["moving"], run["registration"].GetTransformParameterObject(), output_directory=str(tmp_path),
        log_to_console=False)), dtype=np.float64)
    dim = field.shape[-1]
    transformix = np.zeros_like(u)
    transformix[..., 3 - dim:] = (field[..., ::-1] / np.asarray(transform.spacing_zyx)[3 - dim:]).reshape(
        (*shape, dim))
    assert np.max(np.abs(u - transformix)) <= 1e-6
    if shape[0] == 1:
        assert np.all(u[..., 0] == 0)


def test_demons_runs_on_z1_as_2d():
    shape = (1, 64, 64)
    reference, moving, truth = scene(shape, 100, geometry=gaussian_geometry(shape))
    result = estimate(reference, moving, DemonsConfig())
    field = result.transform.displacement_zyx
    assert field.shape == (1, 64, 64, 3)
    assert np.all(field[..., 0] == 0)
    values = errors(result.transform, truth)
    assert values["median_error"] <= 0.5 and values["p95_error"] <= 1.0, values
    assert len(result.diagnostics.elapsed_iterations) == len(result.diagnostics.final_rms_change) == 3
    registered = apply_transform(moving, result.transform, config=result.application_config)
    assert registered.shape == shape


def test_demons_z1_uses_a_yx_only_pyramid(monkeypatch):
    """The SimpleITK demons filter receives 2D images at every level."""
    import SimpleITK as sitk
    sizes = []
    original = sitk.DemonsRegistrationFilter.Execute

    def execute(self, fixed, *args):
        sizes.append(fixed.GetSize())
        return original(self, fixed, *args)

    monkeypatch.setattr(sitk.DemonsRegistrationFilter, "Execute", execute)
    source = np.random.default_rng(3).random((1, 32, 32))
    for mode in ("antialias", "sitk"):
        sizes.clear()
        estimate(source, source, DemonsConfig(iterations=(2, 2, 2), pyramid_mode=mode))
        assert sizes == [(8, 8), (16, 16), (32, 32)]


# --------------------------------------------------------------------- geometry rules
# TPS and CPD cases where the unchanged landmark estimators cannot estimate on any landmark fixture: their
# noise-landmark detection excludes a one-voxel border, so Z=2 has no detectable landmarks and Z=3 has them
# only in plane 1, where the TPS affine term is singular (fixtures tried: W-255 worker notes).
LANDMARK_LIMITED = {("tps", 2), ("tps", 3), ("cpd", 2)}


def landmark_fixture(z):
    """Random background plus 60 bright spots, and the same image shifted by one voxel in X."""
    rng = np.random.default_rng(7)
    reference = rng.random((z, 32, 32))
    reference[rng.integers(0, z, 60), rng.integers(3, 29, 60), rng.integers(3, 29, 60)] += 100
    return reference, np.roll(reference, 1, axis=2)


@pytest.mark.parametrize("z", [1, 2, 3])
@pytest.mark.parametrize("config", [TranslationConfig(), RigidConfig(iterations=5), AffineConfig(iterations=5),
                                    BSplineConfig(iterations=5), DemonsConfig(iterations=(2,)),
                                    TpsConfig(min_matches=4, max_control_points=20),
                                    CpdConfig(max_control_points=20, affine_first=False)], ids=lambda c: c.method)
def test_small_z_and_dimension_rules(config, z, monkeypatch):
    rng = np.random.default_rng(z)
    reference = rng.random((z, 32, 32))
    moving = np.roll(reference, 1, axis=2)
    if (z in (2, 3) and config.method in ("rigid", "affine", "bspline", "demons")
            or z == 1 and config.method in ("tps", "cpd")):
        with pytest.raises(IncompatibleGeometryError):
            estimate(reference, moving, config)
        return
    if config.method in ("tps", "cpd"):
        reference, moving = landmark_fixture(z)
    if (config.method, z) in LANDMARK_LIMITED:
        # The dimension check accepts the shape: estimate_transform reaches the registered estimator.
        calls = []

        def reached(reference, moving, config, geometry):
            calls.append(reference.shape)
            field = np.zeros((*reference.shape, 3), dtype=np.float32)
            return DenseDisplacementTransform(field, **geometry), "fixture", WarpConfig(backend="scipy")

        spec = REGISTRATION_METHODS[type(config)]
        monkeypatch.setitem(REGISTRATION_METHODS, type(config), replace(spec, run=reached))
        estimate(reference, moving, config)
        assert calls == [(z, 32, 32)]
        return
    result = estimate(reference, moving, config)
    transform = result.transform
    u = (transform.dense().displacement_zyx if hasattr(transform, "dense") else
         np.broadcast_to(-np.asarray(transform.correction_zyx), (z, 32, 32, 3))
         if hasattr(transform, "correction_zyx") else transform.displacement_zyx)
    assert u.shape == (z, 32, 32, 3) and np.isfinite(u).all()
    if z == 1:
        assert np.all(u[..., 0] == 0)
    if config.method == "cpd":
        assert abs(np.median(u[..., 2]) - 1) <= 0.1  # the one-voxel X shift of the landmark fixture


def test_physical_rigid_to_index_matrix():
    """Example 4 of the registration contract: spacing (2, 0.5, 0.5), a 2 degree rotation in the Z-X plane."""
    theta = math.radians(2)
    c, s = math.cos(theta), math.sin(theta)
    rotation_zyx = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    spacing = (2.0, 0.5, 0.5)
    center_xyz = np.array([7.75, 7.75, 7.0])
    rotation_xyz = rotation_zyx[::-1, ::-1]
    matrix = _elastix._index_matrix(rotation_xyz, center_xyz, np.zeros(3), spacing)
    reference_metadata, moving_metadata = metadata(spacing)
    transform = AffineTransform(matrix, (8, 32, 32), (8, 32, 32), reference_metadata, moving_metadata)
    expected = np.array([[c, 0, -s * 0.25], [0, 1, 0], [s * 4, 0, c]])  # S^-1 R S
    assert np.max(np.abs(transform.matrix_zyx[:3, :3] - expected)) <= 1e-12
    np.testing.assert_allclose(transform.matrix_zyx[:3, :3], [[0.999391, 0, -0.008725], [0, 1, 0],
                                                              [0.139598, 0, 0.999391]], atol=5e-7, rtol=0)
    # The rotation centre is fixed: b = S^-1 P (c - M c), so the centre index maps to itself.
    center_zyx = center_xyz[::-1] / np.asarray(spacing)
    assert np.max(np.abs(transform.matrix_zyx[:3, :3] @ center_zyx + transform.matrix_zyx[:3, 3]
                         - center_zyx)) <= 1e-12
    assert not transform.matrix_zyx.flags.writeable


def test_two_dimensional_physical_map_is_embedded_with_no_z_motion():
    matrix = _elastix._index_matrix([[1.0, 0.1], [0.0, 1.0]], [3.0, 5.0], [0.5, -1.0], (4.0, 2.0, 1.0))
    # M = [[1, 0.1], [0, 1]] (XY); A_yx = S^-1 P M P S with S = diag(2, 1).
    np.testing.assert_array_equal(matrix[0], [1, 0, 0, 0])
    np.testing.assert_array_equal(matrix[:, 0], [1, 0, 0, 0])
    assert np.max(np.abs(matrix[1:3, 1:3] - [[1, 0], [0.2, 1]])) <= 1e-12
    # b = S^-1 P (t + c - M c) = S^-1 P ((0.5, -1) + (-0.5, 0)) = (-1 / 2, 0 / 1).
    assert np.max(np.abs(matrix[1:3, 3] - [-0.5, 0.0])) <= 1e-12


# --------------------------------------------------------------------- determinism
DETERMINISM_CASES = {
    "rigid-small": ("rigid", DEVELOPMENT_SIZES["small"], None),
    "rigid-z1": ("rigid", DEVELOPMENT_SIZES["z1"], None),
    "rigid-small-spacing122": ("rigid", DEVELOPMENT_SIZES["small"], (1.0, 2.0, 2.0)),
    "affine-small": ("affine", DEVELOPMENT_SIZES["small"], None),
    "affine-z1": ("affine", DEVELOPMENT_SIZES["z1"], None),
    "bspline-16x64x64": ("bspline", (16, 64, 64), None),
    "bspline-1x64x64": ("bspline", (1, 64, 64), None),
}


@pytest.mark.parametrize("case", DETERMINISM_CASES)
def test_estimation_is_deterministic_at_one_thread(case):
    """The rigid, affine and B-spline known-answer fixtures at seed 100, each estimated twice at one thread."""
    import itk
    assert itk.MultiThreaderBase.GetGlobalDefaultNumberOfThreads() == 1
    method, shape, spacing = DETERMINISM_CASES[case]
    if method == "rigid":
        (reference, moving, _), config = scene(shape, 100, geometry=rigid_geometry(shape, 100)), RigidConfig()
    elif method == "affine":
        (reference, moving, _), config = scene(shape, 100, deformation="linear_small"), AffineConfig()
    else:
        (reference, moving, _), config = scene(shape, 100, geometry=polynomial_geometry()), BSplineConfig()
    first, second = (estimate(reference, moving, config, spacing) for _ in range(2))
    assert digest(first.transform) == digest(second.transform)
    assert first.diagnostics == second.diagnostics


# --------------------------------------------------------------------- failures and records
def fov_with(reference, moving):
    dataset = Dataset(input_root="unused", output_root="unused", dataset_id="test", sample_id="sample",
                      output_id="test", rounds=RoundState(sequencing_rounds=["r1", "r2"], reference_round="r1"))
    fov = dataset.fov("FOV")
    fov.images = {"r1": reference[..., None], "r2": moving[..., None]}
    fov.metadata = {name: ImageMetadata(name) for name in ("r1", "r2")}
    return fov


def assert_single_failed_attempt(fov, method, error_type, message):
    (attempt,) = fov.registration_attempts["r2"]
    assert (attempt["requested_method"], attempt["actual_method"], attempt["outcome"]) == (method, method, "failed")
    assert attempt["failure"]["type"] == error_type
    assert message in attempt["failure"]["message"]
    assert not fov.registration_results


@pytest.mark.parametrize("config", [RigidConfig(), AffineConfig(), BSplineConfig()], ids=lambda c: c.method)
def test_constant_signal_fails_before_elastix(config, monkeypatch):
    calls = []
    monkeypatch.setattr(_elastix, "_register", lambda *args: calls.append(args))
    reference = np.random.default_rng(0).random((9, 32, 32))
    constant = np.full_like(reference, 7.0)
    with pytest.raises(RegistrationEstimationError) as info:
        estimate(reference, constant, config)
    assert type(info.value) is RegistrationEstimationError and str(info.value) == "constant registration signal"
    fov = fov_with(reference, constant)
    with pytest.raises(RegistrationEstimationError, match="^constant registration signal$"):
        fov.register(RegistrationStep(config))
    assert_single_failed_attempt(fov, config.method, "RegistrationEstimationError", "constant registration signal")
    assert calls == []


def test_elastix_errors_become_estimation_errors(monkeypatch):
    def fail(*args):
        raise RuntimeError("ITK ERROR: test")

    monkeypatch.setattr(_elastix, "_register", fail)
    shape = DEVELOPMENT_SIZES["small"]
    reference, moving, _ = scene(shape, 100, geometry=rigid_geometry(shape, 100))
    with pytest.raises(RegistrationEstimationError, match="ITK ERROR: test"):
        estimate(reference, moving, RigidConfig())
    fov = fov_with(reference, moving)
    with pytest.raises(RegistrationEstimationError, match="ITK ERROR: test"):
        fov.register(RegistrationStep(RigidConfig()))
    assert_single_failed_attempt(fov, "rigid", "RegistrationEstimationError", "ITK ERROR: test")


# --------------------------------------------------------------------- diagnostics and application
def test_elastix_diagnostics_and_application():
    shape = DEVELOPMENT_SIZES["z1"]
    reference, moving, _ = scene(shape, 100, geometry=rigid_geometry(shape, 100))
    result = estimate(reference, moving, RigidConfig())
    diagnostics = result.diagnostics
    assert (diagnostics.method, diagnostics.backend, diagnostics.converged) == ("rigid", "elastix", None)
    assert diagnostics.iterations_completed == (200, 200)  # two levels: min(Y, X) = 32
    assert len(diagnostics.final_metric_value) == len(diagnostics.stop_condition) == 2
    assert all(math.isfinite(v) for v in diagnostics.final_metric_value)
    assert "Maximum number of iterations" in diagnostics.stop_condition[0]
    assert diagnostics.backend_versions == {"itk-elastix": "0.25.4", "itk": "5.4.7"}
    assert diagnostics.spacing_source == "unknown_unit" and "unit spacing" in diagnostics.warnings[0]
    parameters = diagnostics.backend_parameters
    assert parameters["Interpolator"] == "LinearInterpolator"
    assert (parameters["Metric"], parameters["NumberOfHistogramBins"]) == ("AdvancedMattesMutualInformation", "32")
    assert (parameters["NumberOfSpatialSamples"], parameters["RandomSeed"]) == ("4096", "1")
    assert parameters["WriteResultImage"] == "false"
    assert result.application_config == WarpConfig(backend="scipy")
    registered = apply_transform(moving.astype(np.uint16), result.transform, config=result.application_config)
    assert registered.shape == shape and registered.dtype == np.uint16
    qc = registration_qc(reference, moving, registered.astype(np.float64), result.transform, diagnostics=diagnostics)
    summary = qc.details["transform"]
    assert summary["kind"] == "affine" and not summary["reflection_or_collapse"]
    assert abs(summary["rotation_angle"]) <= math.radians(3.5)
    assert qc.details["optimizer"]["iterations_completed"] == (200, 200)
    assert qc.values["ncc_after"] > qc.values["ncc_before"]


def test_pyramid_rule_and_z_cap():
    assert [_elastix._levels((1, n, n)) for n in (16, 31, 32, 64, 128, 512)] == [1, 1, 2, 3, 4, 4]
    assert _elastix._schedule(3, (16, 64, 64), 3) == [4, 4, 4, 2, 2, 2, 1, 1, 1]
    assert _elastix._schedule(4, (4, 512, 512), 3) == [8, 8, 1, 4, 4, 1, 2, 2, 1, 1, 1, 1]
    assert _elastix._schedule(2, (9, 32, 32), 3) == [2, 2, 2, 1, 1, 1]
    assert _elastix._schedule(2, (1, 32, 32), 2) == [2, 2, 1, 1]


def test_bspline_default_grid_spacing_is_the_x_extent_over_eight():
    reference = np.random.default_rng(5).random((1, 32, 32))
    result = estimate(reference, np.roll(reference, 1, axis=2), BSplineConfig(iterations=5), (1.0, 2.0, 0.5))
    assert result.diagnostics.backend_parameters["FinalGridSpacingInPhysicalUnits"] == "2.0"
    assert result.transform.grid_spacing_xyz == (2.0, 2.0)
