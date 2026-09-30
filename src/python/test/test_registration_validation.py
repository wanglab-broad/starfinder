"""Engineering validation of the §2.6 registration methods (registration task group 6, W-258).

Checks V1 to V11 of docs/registration-algorithms.md ("Engineering validation
design"): known-answer synthetic scenes with pass/fail tolerances fixed before
the run. Where the design names a recipe the check runs FOV.run with that
RegistrationRecipe and compares the round's composite pull field
(TransformChain.pull_field) with forward_displacement over its valid overlap;
otherwise it calls estimate_transform or the transform objects directly. The
V4 and V5 checks, and their determinism runs, are extended-tier. This is
engineering validation only: no method comparison (E01, W-94).
"""
import hashlib
import importlib.util
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from starfinder.benchmark import BenchmarkCase, evaluate_benchmark, run_benchmark
from starfinder.dataset import (CheckpointConfig, Dataset, PipelineConfig, RegistrationRecipe, RegistrationStep,
                                RoundState)
from starfinder.evaluation.registration import (_valid_overlap, evaluate_displacement_field, evaluate_translation,
                                                registration_qc)
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.io import read_checkpoint
from starfinder.io._checkpoint import write_registered_round
from starfinder.registration import (REGISTRATION_METHODS, AffineConfig, AffineTransform, BSplineConfig,
                                     BSplineTransform, CpdConfig, DemonsConfig, DenseDisplacementTransform,
                                     InvalidRegistrationConfigError, RegistrationEstimationError,
                                     RegistrationSignalConfig, RigidConfig, TpsConfig, TransformChain,
                                     TranslationConfig, TranslationTransform, WarpConfig, apply_transform,
                                     estimate_transform)
from starfinder.registration._chain import transform_kind
from starfinder.registration._elastix import _index_matrix
from starfinder.synthetic import (DEVELOPMENT_SIZES, GeometryConfig, forward_displacement, generate_formed_scene,
                                  registration_scene_preset)

from .test_registration_golden import fixture_rounds
from .test_registration_methods import gaussian_geometry, landmark_fixture, polynomial_geometry, preset, rigid_geometry
from .test_registration_recipe import write_version_1

HAS_ELASTIX = importlib.util.find_spec("itk") is not None and importlib.util.find_spec("SimpleITK") is not None
requires_elastix = pytest.mark.skipif(not HAS_ELASTIX,
                                      reason="requires the registration-elastix and local-registration extras")

pytestmark = [
    # SWIG wrappers of itk warn when pytest inspects their builtin types.
    pytest.mark.filterwarnings("ignore:builtin type .* has no __module__ attribute:DeprecationWarning"),
]

SEEDS = (100, 101, 102)
# Fixed before the run (docs/registration-algorithms.md); never re-tuned in a run. A V4 or V5 failure goes to
# Jiahao (W-246 approval, 2026-09-30).
TRANSLATION_TOLERANCE = 0.5          # V1, per axis, strict
GLOBAL_BOUNDS = (0.25, 0.5)          # V2, V3: median, p95 (voxels)
LOCAL_BOUNDS = (0.5, 1.0)            # V4, V5: median, p95 (voxels)
EXACT = 1e-12                        # V6
# The known-answer scenes carry their signal in ch00 (registration_scene_preset).
SIGNAL = RegistrationSignalConfig("channel", 0)


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    """ITK and SimpleITK at one thread, as the project contract requires."""
    if importlib.util.find_spec("SimpleITK") is not None:
        import SimpleITK as sitk
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    if importlib.util.find_spec("itk") is not None:
        import itk
        itk.MultiThreaderBase.SetGlobalDefaultNumberOfThreads(1)


def pair(shape, seed, *, deformation="shift", geometry=None):
    """(reference round, moving round, truth pull field) of one benchmark-appearance pair, rounds ZYXC.

    deformation names a DEFORMATION_PRESETS entry or "shift"; a supplied
    geometry replaces the geometry of the moving round.
    """
    with preset(shape) as name:
        codebook, config = registration_scene_preset(name, deformation)
    config = replace(config, seed=seed, **({} if geometry is None else dict(geometry=geometry)))
    formed = generate_formed_scene(codebook, config=config)
    record = next(t for t in formed.provenance["transforms"].values() if t["round_label"] == deformation)
    return formed.rounds["reference"], formed.rounds[deformation], forward_displacement(record, shape)


def translation_geometry(translation):
    return GeometryConfig(translation_enabled=True, translations_zyx=[[0.0] * 3, list(translation)],
                          reference_round="reference")


def recipe_fov(reference, moving, spacing=None):
    """A two-round FOV (reference, moving) holding the rounds in memory, with the given spacing."""
    dataset = Dataset(input_root="unused", output_root="unused", dataset_id="validation", sample_id="sample",
                      output_id="validation",
                      rounds=RoundState(sequencing_rounds=["reference", "moving"], reference_round="reference"))
    fov = dataset.fov("FOV")
    fov.images = {"reference": reference.copy(), "moving": moving.copy()}
    fov.metadata = {name: ImageMetadata(f"FOV/{name}", spacing_zyx=spacing) for name in fov.images}
    return fov


def run_recipe(configs, reference, moving, spacing=None):
    """FOV.run with the recipe of the given step configs on the ch00 signal; returns the FOV."""
    recipe = RegistrationRecipe(tuple(RegistrationStep(c) for c in configs), signal=SIGNAL)
    return recipe_fov(reference, moving, spacing).run(PipelineConfig(registration=recipe))


def field_errors(u, truth):
    """evaluate_displacement_field of pull field u against truth over the valid overlap of u."""
    mask = _valid_overlap(u, u.shape[:3])
    return (evaluate_displacement_field(u, truth, mask=mask).values,
            evaluate_displacement_field(np.zeros_like(u), truth, mask=mask).values)


def digests(fov):
    """SHA-256 of every stored step transform, the composite pull field and the registered round."""
    out = {}
    for index, result in enumerate(fov.registration_results["moving"]):
        transform, h = result.transform, hashlib.sha256()
        if isinstance(transform, TranslationTransform):
            h.update(np.asarray(transform.displacement_zyx, dtype=np.float64).tobytes())
        elif isinstance(transform, AffineTransform):
            h.update(transform.matrix_zyx.tobytes())
        elif isinstance(transform, BSplineTransform):
            for values in (transform.grid_size_xyz, transform.grid_origin_xyz, transform.grid_spacing_xyz,
                           transform.grid_direction_xyz):
                h.update(np.asarray(values, dtype=np.float64).tobytes())
            h.update(transform.coefficients.tobytes())
        else:
            h.update(transform.displacement_zyx.tobytes())
        out[f"step{index}:{transform_kind(transform)}"] = h.hexdigest()
    out["pull_field"] = hashlib.sha256(
        fov.registration_chains["moving"].pull_field().displacement_zyx.tobytes()).hexdigest()
    out["image"] = hashlib.sha256(np.ascontiguousarray(fov.images["moving"]).tobytes()).hexdigest()
    return out


# ============================================================================ V1: known translation
TRANSLATION_CASES = [("small", (2.0, -3.0, 4.0)), ("small", (0.0, -3.0, 4.0)), ("z1", (0.0, -3.0, 4.0))]
# The two V1 cases the unchanged translation estimator misses by one voxel (operator decision on W-258,
# 2026-09-30, confirmed by Jiahao on 2026-09-30). The bound is not loosened: the gate is still asserted, and a case
# that starts passing fails the run (strict).
KNOWN_MISSES = {
    ("small", (2.0, -3.0, 4.0), 101): "measured displacement (1, -3, 4), expected (2, -3, 4): Z misses by one voxel",
    ("z1", (0.0, -3.0, 4.0), 100): "measured displacement (0, -2, 4), expected (0, -3, 4): Y misses by one voxel",
}


def translation_params():
    """(size, translation, seed) of every V1 case; the known misses are strict expected failures."""
    return [pytest.param(size, translation, seed, id=f"{size}-{translation}-{seed}",
                         marks=[pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
                             f"{KNOWN_MISSES[size, translation, seed]}; W-259 investigates one-voxel misses of "
                             "the translation estimator on benchmark-appearance fixtures"))]
                         if (size, translation, seed) in KNOWN_MISSES else [])
            for size, translation in TRANSLATION_CASES for seed in SEEDS]


def translation_pair(size, translation, seed):
    """(reference ch00, moving ch00, displacement) of a V1 pair; the displacement t pulls from F(p) = p + t."""
    shape = DEVELOPMENT_SIZES[size]
    reference, moving, truth = pair(shape, seed, geometry=translation_geometry(translation))
    np.testing.assert_array_equal(truth, np.broadcast_to(np.asarray(translation, np.float32), truth.shape))
    return (reference[..., 0].astype(np.float64), moving[..., 0].astype(np.float64),
            list(translation))


@pytest.mark.parametrize("size,translation,seed", translation_params())
def test_v1_known_translation(size, translation, seed):
    """estimate_transform and evaluate_translation with the strict gate."""
    reference, moving, displacement = translation_pair(size, translation, seed)
    metadata = ImageMetadata("reference")
    result = estimate_transform(reference, moving, config=TranslationConfig(), reference_metadata=metadata,
                                moving_metadata=ImageMetadata("moving"))
    direct = evaluate_translation({"moving": result.transform.displacement_zyx}, {"moving": displacement},
                                  reference_metadata=metadata, observed_metadata=metadata, units="voxel",
                                  tolerance=TRANSLATION_TOLERANCE)
    assert direct.values["passed"] is True, direct
    assert max(direct.details["per_round"]["moving"]["error"]) < TRANSLATION_TOLERANCE


@pytest.mark.parametrize("size,translation,seed", translation_params())
def test_v1_known_translation_benchmark_task(size, translation, seed, tmp_path):
    """The registration benchmark task on the same pair, with evaluation {translation: {tolerance: 0.5}}."""
    reference, moving, displacement = translation_pair(size, translation, seed)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    np.save(inputs / "reference.npy", reference)
    np.save(inputs / "moving.npy", moving)
    (inputs / "displacement.json").write_text(json.dumps(displacement))
    case = BenchmarkCase(f"v1-{size}-{seed}", "registration", {"reference": "reference.npy", "moving": "moving.npy"},
                         {"registration": {"method": "translation"}, "reference_metadata": {"frame_id": "reference"},
                          "moving_metadata": {"frame_id": "moving"},
                          "evaluation": {"translation": {"tolerance": TRANSLATION_TOLERANCE}}},
                         truth={"displacement": "displacement.json"})
    run = run_benchmark([case], input_root=inputs, output_root=tmp_path / "runs", owner="pytest")
    (record,) = json.loads((evaluate_benchmark(run) / "results.json").read_text())
    assert record["status"]["processing"] == "success"
    metric = record["metrics"]["translation"]
    assert metric["config"]["tolerance"] == TRANSLATION_TOLERANCE and metric["config"]["boundary"] == "exclusive"
    assert metric["values"]["passed"] is True and metric["values"]["max_error"] < TRANSLATION_TOLERANCE, metric


# ============================================================================ V2, V3: known rigid and affine maps
GLOBAL_CASES = {
    "rigid-small": ("rigid", "small", None),
    "rigid-z1": ("rigid", "z1", None),
    "rigid-small-spacing122": ("rigid", "small", (1.0, 2.0, 2.0)),
    "affine-small": ("affine", "small", None),
    "affine-z1": ("affine", "z1", None),
}


def global_run(case, seed):
    """FOV.run of (translation, rigid) or (translation, affine) on the V2 or V3 scene; (FOV, truth)."""
    method, size, spacing = GLOBAL_CASES[case]
    shape = DEVELOPMENT_SIZES[size]
    if method == "rigid":
        reference, moving, truth = pair(shape, seed, geometry=rigid_geometry(shape, seed))
        configs = (TranslationConfig(), RigidConfig())
    else:
        reference, moving, truth = pair(shape, seed, deformation="linear_small")
        configs = (TranslationConfig(), AffineConfig())
    return run_recipe(configs, reference, moving, spacing), truth


@requires_elastix
@pytest.mark.parametrize("case", GLOBAL_CASES)
@pytest.mark.parametrize("seed", SEEDS)
def test_v2_v3_known_rigid_and_affine_maps_through_recipes(case, seed):
    fov, truth = global_run(case, seed)
    chain = fov.registration_chains["moving"]
    assert [transform_kind(t) for t in chain.transforms] == ["translation", "affine"]
    assert fov.registration_record["recipe"]["steps"] == ["translation", GLOBAL_CASES[case][0]]
    u = chain.pull_field().displacement_zyx
    values, identity = field_errors(u, truth)
    median, p95 = GLOBAL_BOUNDS
    assert values["median_error"] <= median and values["p95_error"] <= p95, (values, identity)
    if u.shape[0] == 1:
        assert np.all(u[..., 0] == 0)


# ============================================================================ V4, V5: known deformations
LOCAL_CASES = {
    "bspline-16x64x64": ((16, 64, 64), BSplineConfig),
    "bspline-1x64x64": ((1, 64, 64), BSplineConfig),
    "demons-1x64x64": ((1, 64, 64), DemonsConfig),
}


def local_run(case):
    """FOV.run of (translation, bspline) on the V4 scenes or (translation, demons) on the V5 scene, seed 100."""
    shape, config = LOCAL_CASES[case]
    geometry = polynomial_geometry() if config is BSplineConfig else gaussian_geometry(shape)
    reference, moving, truth = pair(shape, 100, geometry=geometry)
    return run_recipe((TranslationConfig(), config()), reference, moving), truth


@pytest.fixture(scope="module")
def local_runs():
    """Each V4 and V5 recipe run twice at one thread (V11 compares the two)."""
    return {case: (local_run(case), local_run(case)) for case in LOCAL_CASES}


@pytest.mark.extended
@requires_elastix
@pytest.mark.parametrize("case", LOCAL_CASES)
def test_v4_v5_known_deformations_through_recipes(case, local_runs):
    (fov, truth), _ = local_runs[case]
    chain = fov.registration_chains["moving"]
    local_kind = "bspline" if LOCAL_CASES[case][1] is BSplineConfig else "dense"
    assert [transform_kind(t) for t in chain.transforms] == ["translation", local_kind]
    u = chain.pull_field().displacement_zyx
    values, identity = field_errors(u, truth)
    if local_kind == "bspline":
        assert identity["median_error"] > 1  # by construction of the polynomial term
    median, p95 = LOCAL_BOUNDS
    assert values["median_error"] <= median and values["p95_error"] <= p95, (values, identity)
    assert values["p95_error"] < identity["p95_error"], (values, identity)
    if u.shape[0] == 1:
        assert np.all(u[..., 0] == 0)
        assert np.all(chain.transforms[-1].dense().displacement_zyx[..., 0] == 0 if local_kind == "bspline"
                      else chain.transforms[-1].displacement_zyx[..., 0] == 0)


# ============================================================================ V6: composition
def geometry(shape):
    return dict(reference_shape_zyx=shape, moving_shape_zyx=shape, reference_metadata=ImageMetadata("v6/reference"),
                moving_metadata=ImageMetadata("v6/moving"))


def affine(a, b, shape):
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = a, b
    return AffineTransform(matrix, **geometry(shape))


def dense(u, shape):
    return DenseDisplacementTransform(u, **geometry(shape))


def translation(displacement, shape):
    return TranslationTransform(displacement, **geometry(shape))


def u2_example_2(shape):
    u = np.zeros((*shape, 3))
    u[..., 2] = 0.05 * np.indices(shape, dtype=np.float64)[1]
    return u


# (steps, shape, {point: expected u(p)}, {point: wrong-order value}) of examples 1 to 3 of the contract.
COMPOSITION_EXAMPLES = {
    "example-1-translation-then-affine": (
        lambda s: (translation((1, 4, -2), s), affine([[1, 0, 0], [0, 1, 0.1], [0, 0, 1]], (0, 0.5, 0), s)),
        (8, 32, 32), {(0, 0, 0): (1, 4.5, -2), (2, 10, 20): (1, 6.5, -2)},
        {(0, 0, 0): (1, 4.3, -2), (2, 10, 20): (1, 6.3, -2)}),
    "example-2-affine-then-dense": (
        lambda s: (affine(np.diag([1, 1.1, 0.9]), (0, -2, 3), s), dense(u2_example_2(s), s)),
        (8, 32, 32), {(0, 0, 0): (0, -2, 3), (4, 20, 10): (0, 0, 2.9), (2, 30, 31): (0, 1, 1.25)},
        {(0, 0, 0): (0, -2, 2.9), (4, 20, 10): (0, 0, 3), (2, 30, 31): (0, 1, 1.45)}),
    "example-3-z1-translation-then-affine": (
        lambda s: (translation((0, 3, -1), s),
                   affine([[1, 0, 0], [0, 0.98, 0.05], [0, -0.05, 0.98]], (0, 0.2, -0.4), s)),
        (1, 32, 32), {(0, 0, 0): (0, 3.2, -1.4), (0, 10, 20): (0, 4.0, -2.3), (0, 31, 0): (0, 2.58, -2.95)},
        {(0, 0, 0): (0, 3.09, -1.53), (0, 10, 20): (0, 3.89, -2.43), (0, 31, 0): (0, 2.47, -3.08)}),
}


def sequential(steps, points):
    """Phi(p) = T1(T2(...Tn(p))) at integer grid points, evaluating each step's own map from its parameters."""
    q = np.asarray(points, dtype=np.float64)
    for index, step in enumerate(reversed(steps)):
        if isinstance(step, DenseDisplacementTransform):
            assert index == 0 and np.array_equal(q, np.rint(q))  # only the last step, at grid points
            q = q + step.displacement_zyx[tuple(q.astype(int).T)]
        elif isinstance(step, AffineTransform):
            q = q @ step.matrix_zyx[:3, :3].T + step.matrix_zyx[:3, 3]
        else:
            q = q + np.asarray(step.displacement_zyx)
    return q


@pytest.mark.parametrize("example", COMPOSITION_EXAMPLES)
def test_v6_analytic_composition_examples(example):
    build, shape, expected, wrong = COMPOSITION_EXAMPLES[example]
    steps = build(shape)
    field = TransformChain(steps).pull_field().displacement_zyx
    assert field.dtype == np.float64 and field.shape == (*shape, 3)
    points = np.array(list(expected))
    for point, value in expected.items():
        np.testing.assert_allclose(field[point], value, rtol=0, atol=EXACT)
        # The wrong order T_n(...T_1(p)) differs from the expected value (so an order error fails).
        assert max(abs(a - b) for a, b in zip(wrong[point], value)) > 0.05
    np.testing.assert_allclose(sequential(steps, points) - points, [expected[tuple(p)] for p in points],
                               rtol=0, atol=EXACT)
    if shape[0] == 1:
        assert np.all(field[..., 0] == 0)
    if not any(isinstance(s, DenseDisplacementTransform) for s in steps):
        # Without a dense step the reversed chain is valid and gives the wrong-order values.
        reversed_field = TransformChain(steps[::-1]).pull_field().displacement_zyx
        for point, value in wrong.items():
            np.testing.assert_allclose(reversed_field[point], value, rtol=0, atol=EXACT)


def test_v6_example_4_physical_to_index_matrix():
    """spacing (2, 0.5, 0.5), a 2 degree physical rotation in the Z-X plane: the index matrix S^-1 R S."""
    theta = math.radians(2)
    c, s = math.cos(theta), math.sin(theta)
    rotation_zyx = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    spacing = np.array([2.0, 0.5, 0.5])
    matrix = _index_matrix(rotation_zyx[::-1, ::-1], np.zeros(3), np.zeros(3), spacing)
    expected = np.diag(1 / spacing) @ rotation_zyx @ np.diag(spacing)
    np.testing.assert_allclose(matrix[:3, :3], expected, rtol=0, atol=EXACT)
    np.testing.assert_array_equal(np.round(matrix[:3, :3], 6),
                                  [[0.999391, 0, -0.008725], [0, 1, 0], [0.139598, 0, 0.999391]])
    field = TransformChain((AffineTransform(matrix, **geometry((8, 32, 32))),)).pull_field().displacement_zyx
    for point in ((0, 0, 0), (4, 10, 20), (7, 31, 31)):
        np.testing.assert_allclose(field[point], expected @ point - point, rtol=0, atol=EXACT)


def test_v6_translation_affine_dense_chain_equals_sequential_evaluation():
    shape = (8, 32, 32)
    rng = np.random.default_rng(258)
    a = np.eye(3) + rng.uniform(-0.05, 0.05, (3, 3))
    u3 = rng.uniform(-1.5, 1.5, (*shape, 3))
    steps = (translation((-1.0, 2.5, -3.25), shape), affine(a, rng.uniform(-2, 2, 3), shape), dense(u3, shape))
    field = TransformChain(steps).pull_field().displacement_zyx
    points = np.moveaxis(np.indices(shape, dtype=np.float64), 0, -1).reshape(-1, 3)
    expected = (sequential(steps, points) - points).reshape(*shape, 3)
    assert np.max(np.abs(field - expected)) <= EXACT


# ============================================================================ V7: persistence round trip
TRANSFORM_KINDS = {
    "translation": ((TranslationConfig(),), ["translation"]),
    "rigid": ((RigidConfig(),), ["affine"]),
    "affine": ((AffineConfig(),), ["affine"]),
    "bspline": ((BSplineConfig(),), ["bspline"]),
    "dense": ((DemonsConfig(iterations=(20, 10)),), ["dense"]),
    "chain-translation-affine-bspline": ((TranslationConfig(), AffineConfig(), BSplineConfig()),
                                         ["translation", "affine", "bspline"]),
    "chain-translation-rigid-dense": ((TranslationConfig(), RigidConfig(), DemonsConfig(iterations=(20, 10))),
                                      ["translation", "affine", "dense"]),
}


def golden_fov(tmp_path, z1):
    """The seeded golden fixture (8x32x32, four channels; its plane z=4 for Z=1) in an FOV writing to tmp_path."""
    dataset = Dataset(tmp_path, tmp_path / "out", "validation", "sample", "out",
                      rounds=RoundState(sequencing_rounds=["round1", "round2"], reference_round="round1"),
                      channel_order=("ch00", "ch01", "ch02", "ch03"))
    fov = dataset.fov("FOV_001")
    for name, volume in fixture_rounds().items():
        fov.images[name] = volume[4:5].copy() if z1 else volume
        fov.metadata[name] = ImageMetadata(f"FOV_001/{name}")
    return fov


@requires_elastix
@pytest.mark.parametrize("z1", [False, True], ids=["8x32x32", "1x32x32"])
@pytest.mark.parametrize("kind", TRANSFORM_KINDS)
def test_v7_version_2_registered_checkpoint_round_trip(tmp_path, kind, z1):
    configs, kinds = TRANSFORM_KINDS[kind]
    fov = golden_fov(tmp_path, z1)
    before = {name: image.copy() for name, image in fov.images.items()}
    fov.run(PipelineConfig(registration=RegistrationRecipe(tuple(map(RegistrationStep, configs)))),
            checkpoints=CheckpointConfig(stages=("registered",)))
    header = json.loads((fov.paths.checkpoint_dir / "registered" / "transforms.json").read_text())
    assert header["format_version"] == 2
    reloaded = fov.dataset.fov("FOV_001").load_checkpoint("registered")
    chain, original = reloaded.registration_chains["round2"], fov.registration_chains["round2"]
    assert [transform_kind(t) for t in chain.transforms] == kinds
    assert np.array_equal(chain.pull_field().displacement_zyx, original.pull_field().displacement_zyx)
    warp = reloaded.registration_record["application"]["round2"]
    assert warp == fov.registration_record["application"]["round2"]
    assert np.array_equal(apply_transform(before["round2"], chain, config=warp), fov.images["round2"])
    assert np.array_equal(reloaded.images["round2"], fov.images["round2"])
    assert np.array_equal(reloaded.images["round1"], before["round1"])


def test_v7_version_1_checkpoint_loads_as_sequential(tmp_path):
    """A version-1 registered checkpoint (the start revision's writer) loads as sequential, unconverted."""
    fov = golden_fov(tmp_path, False)
    reference, moving = (fov.images[name].sum(axis=-1, dtype=np.float64) for name in ("round1", "round2"))
    result = estimate_transform(reference, moving, config=TranslationConfig(), reference_metadata=fov.metadata["round1"],
                                moving_metadata=fov.metadata["round2"])
    directory = fov.paths.checkpoint_dir
    for name, image in fov.images.items():
        write_registered_round(directory, name, image, fov.metadata[name])
    write_version_1(directory, dict(fov._checkpoint_header(), image_rounds=["round1", "round2"], snapshots=[],
                                    registration_attempts={}, preprocessing=None), {"round2": [result]})
    loaded = read_checkpoint(directory, "registered")
    assert loaded["registration_record"]["semantics"] == "sequential" and loaded["registration_chains"] == {}
    assert loaded["registration_results"]["round2"] == [result]


# ============================================================================ V8: Z=1 and small-Z rules
# Expected outcome per method and Z from the design: methods with 2 in dimensions succeed on Z=1 with no Z
# motion; TPS and CPD reject Z=1; the elastix methods and demons reject Z=2 and 3 (min_shape_zyx Z=4); translation
# accepts every Z; TPS and CPD accept Z=2 and 3 by their declaration (their landmark estimators may still fail).
SMALL_Z = {
    "translation": ("ok", "ok", "ok"),
    "rigid": ("ok", "geometry", "geometry"),
    "affine": ("ok", "geometry", "geometry"),
    "bspline": ("ok", "geometry", "geometry"),
    "demons": ("ok", "geometry", "geometry"),
    "tps": ("geometry", "accepted", "accepted"),
    "cpd": ("geometry", "accepted", "accepted"),
}
SMALL_Z_CONFIGS = {"translation": TranslationConfig(), "rigid": RigidConfig(), "affine": AffineConfig(),
                   "bspline": BSplineConfig(), "demons": DemonsConfig(),
                   "tps": TpsConfig(min_matches=4, max_control_points=20),
                   "cpd": CpdConfig(max_control_points=20, affine_first=False)}


def test_v8_covers_every_registered_method():
    assert set(SMALL_Z) == set(SMALL_Z_CONFIGS) == {spec.name for spec in REGISTRATION_METHODS.values()}
    for spec in REGISTRATION_METHODS.values():
        # The expectations agree with the declared capabilities.
        assert (SMALL_Z[spec.name][0] == "ok") == (2 in spec.dimensions)
        assert (SMALL_Z[spec.name][1] == "geometry") == (spec.min_shape_zyx[0] > 2)


@requires_elastix
@pytest.mark.parametrize("z", [1, 2, 3])
@pytest.mark.parametrize("method", SMALL_Z)
def test_v8_z1_and_small_z_rules(method, z):
    config = SMALL_Z_CONFIGS[method]
    rng = np.random.default_rng(z)
    reference = rng.random((z, 32, 32))
    moving = np.roll(reference, 1, axis=2)
    if method in ("tps", "cpd"):
        reference, moving = landmark_fixture(z)
    kwargs = dict(config=config, reference_metadata=ImageMetadata("reference"), moving_metadata=ImageMetadata("moving"))
    expected = SMALL_Z[method][z - 1]
    if expected == "geometry":
        with pytest.raises(IncompatibleGeometryError):
            estimate_transform(reference, moving, **kwargs)
        return
    try:
        transform = estimate_transform(reference, moving, **kwargs).transform
    except RegistrationEstimationError:
        # Only a declared-3D landmark method may fail here, and not on geometry (IncompatibleGeometryError is a
        # ValueError, never a RegistrationEstimationError).
        assert expected == "accepted"
        return
    u = (TransformChain((transform,)).pull_field().displacement_zyx)
    assert u.shape == (z, 32, 32, 3) and np.isfinite(u).all()
    if z == 1:
        assert np.all(u[..., 0] == 0)


# ============================================================================ V9: boundary and coverage
def ramp(shape=(1, 32, 32)):
    """A float64 ramp increasing in Y and X, positive everywhere, so fill values and edge rows are distinct."""
    z, y, x = np.indices(shape, dtype=np.float64)
    return 1 + 32 * y + x


def test_v9_constant_boundary_of_a_translation_only_chain():
    shape = (1, 32, 32)
    image = ramp(shape)
    chain = TransformChain((translation((0, 5, 0), shape),))
    with pytest.raises(InvalidRegistrationConfigError, match="constant zero fill"):
        WarpConfig(boundary_mode="nearest")
    after = apply_transform(image, chain, config=WarpConfig(output_dtype="float64"))
    np.testing.assert_array_equal(after[:, :27], image[:, 5:])
    np.testing.assert_array_equal(after[:, 27:], 0)
    qc = registration_qc(image, image, after, chain)
    assert qc.values["coverage"] == 27 / 32


def test_v9_nearest_boundary_of_an_affine_chain():
    shape = (1, 32, 32)
    image = ramp(shape)
    chain = TransformChain((affine(np.eye(3), (0, 5, 0), shape),))
    assert chain.translation() is None
    after = apply_transform(image, chain, config=WarpConfig(backend="scipy", boundary_mode="nearest",
                                                            output_dtype="float64"))
    np.testing.assert_array_equal(after[:, :27], image[:, 5:])
    np.testing.assert_array_equal(after[:, 27:], np.broadcast_to(image[:, 31:32], (1, 5, 32)))
    qc = registration_qc(image, image, after, chain)
    assert qc.values["coverage"] == 27 / 32


# ============================================================================ V10: empty and constant input
@requires_elastix
@pytest.mark.parametrize("config", [RigidConfig(), AffineConfig(), BSplineConfig()], ids=lambda c: c.method)
def test_v10_constant_moving_signal_is_rejected(config):
    reference = np.random.default_rng(0).random((9, 32, 32))
    constant = np.full_like(reference, 7.0)
    with pytest.raises(RegistrationEstimationError) as info:
        estimate_transform(reference, constant, config=config, reference_metadata=ImageMetadata("reference"),
                           moving_metadata=ImageMetadata("moving"))
    assert type(info.value) is RegistrationEstimationError and str(info.value) == "constant registration signal"
    # Through a recipe the failure propagates and is recorded; nothing is substituted.
    fov = recipe_fov(reference[..., None], constant[..., None])
    with pytest.raises(RegistrationEstimationError, match="^constant registration signal$"):
        fov.run(PipelineConfig(registration=RegistrationRecipe((RegistrationStep(config),))))
    (attempt,) = fov.registration_attempts["moving"]
    assert attempt["outcome"] == "failed" and attempt["failure"] == {"type": "RegistrationEstimationError",
                                                                     "message": "constant registration signal"}


def test_v10_constant_signal_qc_is_undefined_with_a_reason():
    shape = (1, 32, 32)
    constant = np.full(shape, 7.0)
    chain = TransformChain((translation((0, 0, 0), shape),))
    qc = registration_qc(ramp(shape), constant, constant, chain)
    assert qc.values["coverage"] == 1
    for key in ("ncc_before", "ncc_after", "ncc_gain"):
        assert qc.values[key] is None
    assert qc.reasons["ncc_after"] == "constant signal over the valid overlap"


def test_v10_moving_round_beyond_the_grid():
    shape = (1, 32, 32)
    reference = ramp(shape)
    # The reference content moved 40 rows down: nothing of it is left on the moving grid.
    moving = np.zeros(shape)
    chain = TransformChain((translation((0, 40, 0), shape),))
    after = apply_transform(moving, chain, config=WarpConfig(output_dtype="float64"))
    qc = registration_qc(reference, moving, after, chain)
    assert qc.values["coverage"] == 0
    for key in ("ncc_before", "ncc_after", "ssim_before", "ssim_after"):
        assert qc.values[key] is None and qc.reasons[key], key
    assert qc.reasons["ncc_after"] == "no valid overlap"
    assert qc.reasons["ssim_after"] == "the eroded valid columns are empty"


# ============================================================================ V11: determinism
@requires_elastix
@pytest.mark.parametrize("case", GLOBAL_CASES)
def test_v11_global_recipes_are_deterministic_at_one_thread(case):
    import itk
    assert itk.MultiThreaderBase.GetGlobalDefaultNumberOfThreads() == 1
    first, second = (digests(global_run(case, 100)[0]) for _ in range(2))
    assert first == second


@pytest.mark.extended
@requires_elastix
@pytest.mark.parametrize("case", LOCAL_CASES)
def test_v11_local_recipes_are_deterministic_at_one_thread(case, local_runs):
    import SimpleITK as sitk
    assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == 1
    (first, _), (second, _) = local_runs[case]
    assert digests(first) == digests(second)
