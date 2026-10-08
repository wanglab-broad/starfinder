"""MV1 to MV6 of docs/stitching-assembly-validation.md: the ``multiview_stitcher`` placement method.

Fixtures: ``cut_synthetic`` and its derived cases, rebuilt in ``stitching_cut_fixtures``
from W-324's recipe (seeds 20261006, 324, 1000 + tile, 1050); ``scene_small``
(``sample-small-v1``, seed 42) for MV3. The checks that run the library are ``slow`` and
skip, naming the extra, where ``multiview_stitcher`` is not importable; every one runs at
one thread with dask's synchronous scheduler. The configuration rules, the registry entry,
the shape bound and the missing-extra behavior need no library and run in the default
tier; the missing extra is made by making the import fail inside the test.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import starfinder.stitching
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.segmentation import ReferenceGrid
from starfinder.stitching import (STITCHING_METHODS, MultiviewStitcherConfig, StitchingBackendUnavailableError,
                                  StitchingInput, load_placement, neighbor_pairs, positions_from_grid,
                                  save_placement, stitch)

from .stitching_cut_fixtures import IDENTITY, case, errors_voxels

pytestmark = [pytest.mark.stitching, pytest.mark.validation]

SRC = Path(__file__).resolve().parents[1]
EXTRA = "the multiview-stitcher extra (starfinder[multiview-stitcher]) is not installed"
needs_library = pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None, reason=EXTRA)


def run(case_, fallback=None, **fields):
    import dask
    with dask.config.set(scheduler="synchronous"):
        return stitch(case_.inputs(), case_.table(), config=MultiviewStitcherConfig(**fields), sample="cut",
                      fallback=fallback)


def report(name, result, errors=None):
    """Print the statuses, review and errors of a case (a measurement, read with ``-s``)."""
    fovs = result.fovs
    line = {"case": name, "statuses": dict(zip(fovs.fov_id, fovs.status)),
            "review": [result.record["review"]["state"], result.record["review"]["reasons"]]}
    if errors is not None and len(errors):
        line["max_abs_error_voxels_zyx"] = [float(v) for v in np.abs(errors).max(axis=0)]
    print(json.dumps(line))


def supported(result):
    return list(result.fovs.fov_id[result.fovs.status == "image_supported"])


# --- Configuration, registry and the missing extra (default tier) ---------------------------

def test_config_defaults_are_the_starting_configuration():
    config = MultiviewStitcherConfig()
    assert config.method == "multiview_stitcher" == STITCHING_METHODS[MultiviewStitcherConfig].name
    assert (config.binning_zyx, config.pairwise, config.upsample_factor, config.quality_threshold,
            config.drop_nan_quality, config.pruning, config.overlap_tolerance_um, config.groupwise,
            config.abs_tol_um, config.rel_tol, config.max_iter) == (
        (4, 4, 4), "phase_correlation", None, 0.7, True, "alternating_pattern", 0.0, "global_optimization",
        None, None, None)
    assert MultiviewStitcherConfig(binning_zyx=[2, 2, 2], pruning=None).binning_zyx == (2, 2, 2)


@pytest.mark.parametrize("fields", [
    dict(binning_zyx=(0, 4, 4)), dict(binning_zyx=(4, 4, 0)), dict(quality_threshold=1.5),
    dict(upsample_factor=0), dict(overlap_tolerance_um=-0.1), dict(pairwise="ants"),
    dict(pruning="keep_axis_aligned"), dict(groupwise="shortest_paths")])
def test_config_refuses_each_invalid_field_at_construction(fields):
    with pytest.raises(ValueError, match=next(iter(fields))):
        MultiviewStitcherConfig(**fields)


def test_spec_declares_the_contract_capabilities():
    spec = STITCHING_METHODS[MultiviewStitcherConfig]
    assert [(d.module, d.distribution, d.extra) for d in spec.requires] == [
        ("multiview_stitcher", "multiview-stitcher", "multiview-stitcher")]
    assert spec.dimensions == frozenset({2, 3}) and spec.devices == ("cpu",) and spec.min_shape_zyx == (1, 16, 16)
    assert spec.transform_kind == "translation"
    assert starfinder.stitching.STITCHING_METHODS is STITCHING_METHODS


def small_inputs(shape, n=2):
    """Two FOVs of ``shape`` on a 2 × 1 grid at unit spacing, with zero images (no registration runs)."""
    table = positions_from_grid(columns=n, rows=1, order="column_major_down_right", tile_shape_yx=shape[1:],
                                spacing_yx=(1.0, 1.0), overlap=0.25, fov_ids=[f"FOV{k}" for k in range(n)])
    grid = ReferenceGrid(shape, ImageMetadata("fov", (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), IDENTITY, "µm"), "declared")
    return table, [StitchingInput(fov, np.zeros(shape, np.uint16), grid, "reference_merged") for fov in table.fov_ids]


@pytest.mark.parametrize("shape", [(4, 15, 32), (4, 32, 15), (1, 15, 16)])
def test_a_tile_below_16_px_in_y_or_x_is_refused(shape):
    table, inputs = small_inputs(shape)
    with pytest.raises(IncompatibleGeometryError, match=r"needs tiles of at least"):
        stitch(inputs, table, config=MultiviewStitcherConfig())


def test_a_missing_extra_raises_naming_the_module_and_the_extra(monkeypatch):
    for name in [m for m in sys.modules if m == "multiview_stitcher" or m.startswith("multiview_stitcher.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "multiview_stitcher", None)   # the import now raises ImportError
    table, inputs = small_inputs((4, 16, 16))
    with pytest.raises(StitchingBackendUnavailableError,
                       match=r"'multiview_stitcher' requires multiview_stitcher; install the 'multiview-stitcher' "
                             r"extra \(starfinder\[multiview-stitcher\]\)"):
        stitch(inputs, table, config=MultiviewStitcherConfig())


def _environment():
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", NUMBA_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), *filter(None, [env.get("PYTHONPATH")])])
    return env


def test_importing_the_module_and_building_a_config_imports_no_library():
    code = ("import sys, starfinder.stitching as s; c = s.MultiviewStitcherConfig(); "
            "spec = s.STITCHING_METHODS[type(c)]; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] == 'multiview_stitcher'))")
    output = subprocess.run([sys.executable, "-c", code], cwd=SRC, env=_environment(), capture_output=True,
                            text=True, check=True)
    assert output.stdout.strip() == "[]"


# --- MV1, MV2: the known-answer cut fixture --------------------------------------------------

@pytest.mark.slow
@needs_library
@pytest.mark.parametrize("binning, bound", [((4, 4, 4), 2.0), ((2, 2, 2), 1.0)], ids=["MV1", "MV2"])
def test_known_answer_cut_fixture(binning, bound):
    fixture = case("cut_synthetic")
    result = run(fixture, binning_zyx=binning)
    errors = errors_voxels(result, fixture)
    report(f"cut_synthetic binning {binning}", result, errors)
    assert list(result.fovs.status) == ["image_supported"] * 9
    assert set(result.fovs.component) == {0}
    assert np.abs(errors).max() <= bound


# --- MV3: fractional jitter on the synthetic scene -------------------------------------------

@pytest.mark.slow
@needs_library
def test_fractional_jitter_on_the_synthetic_scene():
    import dask

    from starfinder.stitching import positions_from_table
    from starfinder.synthetic import generate_sample_scene, sample_scene_preset
    book, config = sample_scene_preset("sample-small-v1")
    scene = generate_sample_scene(book, config=config)
    reference = config.fov.geometry.reference_round or book.round_labels[0]
    inputs = []
    for fov, formed in scene.fovs.items():
        image = formed.rounds[reference].max(axis=-1)   # the reference_merged analogue
        inputs.append(StitchingInput(fov, image, ReferenceGrid(image.shape, formed.metadata, "declared"),
                                     "reference_merged"))
    # The MV3 row runs with the link filter off (NaN-quality links still dropped): it tests precision at
    # fractional offsets, and the link rule is tested by MV4 and MV5.
    config_mv3 = MultiviewStitcherConfig(binning_zyx=(1, 1, 1), quality_threshold=-1.0)
    assert config_mv3.drop_nan_quality
    with dask.config.set(scheduler="synchronous"):
        result = stitch(inputs, positions_from_table(scene.positions), sample=scene.sample, config=config_mv3)
    truth = scene.placement.set_index("fov_id").loc[list(result.fovs.fov_id)]
    origin = result.fovs[["origin_z", "origin_y", "origin_x"]].to_numpy()
    errors = (origin - truth[["true_origin_g1_z", "true_origin_g1_y", "true_origin_g1_x"]].to_numpy()) \
        / np.asarray(config.acquisition.spacing_zyx)
    report("scene_small binning (1, 1, 1), link filter off", result, errors)
    assert list(result.fovs.status) == ["image_supported"] * 9
    assert np.abs(errors).max() <= 0.5


# --- MV4: the hand-built cases ---------------------------------------------------------------

FAILING_TILE = {"no_signal_zero": "tile4", "no_signal_noise": "tile4", "disconnected": "tile8"}
CASES = ("two_tiles", "z1_3d", "plane_2d", "spacing_1_2_2", "non_square", "no_signal_zero", "no_signal_noise",
         "disconnected")


@pytest.mark.slow
@needs_library
@pytest.mark.parametrize("name", CASES)
def test_hand_built_cases(name):
    fixture = case(name)
    for fallback in (None, "nominal") if name in FAILING_TILE else (None,):
        result = run(fixture, fallback=fallback)
        placed = supported(result)
        errors = errors_voxels(result, fixture, placed)
        report(f"{name} fallback={fallback}", result, errors)
        fovs = result.fovs.set_index("fov_id")
        reference = set(fovs.loc[placed, "component"])
        assert len(reference) == 1
        # Every FOV of the reference component, each axis in its own voxels (spacing_1_2_2 included).
        assert np.abs(errors).max() <= 2.0
        if name in ("z1_3d", "plane_2d"):   # registered in 2D
            placed_rows = fovs[np.isfinite(fovs.origin_z)]
            assert (placed_rows.correction_z == 0.0).all() and (placed_rows.method_translation_z == 0.0).all()
        if name in FAILING_TILE:
            tile = fovs.loc[FAILING_TILE[name]]
            if fallback is None:
                assert tile.status == "failed" and np.isnan(tile.origin_x)
            else:
                assert tile.status == "metadata_only" and "fallback" in tile["flags"].split(";")
                assert [tile[f"origin_{a}"] for a in "zyx"] == [tile[f"nominal_{a}"] for a in "zyx"]
            assert result.record["review"]["state"] == "required"


# --- MV5: the wrapper rules ------------------------------------------------------------------

def _pair_set(frame):
    return set(zip(frame.fov_a, frame.fov_b))


@pytest.mark.slow
@needs_library
def test_wrapper_rules_on_the_cut_fixture(tmp_path):
    fixture = case("cut_synthetic")
    result = run(fixture)
    links = result.links
    inputs = fixture.inputs()
    pairs = neighbor_pairs(fixture.table(), {item.fov_id: item.grid for item in inputs})
    expected = {(p.fov_a, p.fov_b) for p in pairs}
    registered = links[links.reason != "pruned"]
    pruned = links[links.reason == "pruned"]
    diagnostics = result.record["method_diagnostics"]
    # One row per neighbour pair; the registered pairs are a subset, every other pair is pruned.
    assert _pair_set(links) == expected and len(links) == len(expected)
    assert _pair_set(registered) <= expected and len(registered) and len(pruned)
    assert pruned.quality.isna().all() and not pruned.kept.any()
    assert _pair_set(pruned) == {tuple(p) for p in diagnostics["pruned_pairs"]}
    # The library's overlap graph gives the contract's pairs on this fixture.
    assert {tuple(p) for p in diagnostics["overlap_pairs"]} == expected
    assert diagnostics["overlap_rows_outside_neighbor_pairs"] == []
    # L1: below 0.7 is not kept with quality_below_threshold; no NaN-quality link is kept.
    below = registered.quality < 0.7
    assert below.any()
    assert list(registered.reason[below]) == ["quality_below_threshold"] * int(below.sum())
    assert not registered.kept[below].any() and registered.kept[~below & registered.quality.notna()].all()
    assert not (links.kept & links.quality.isna()).any()
    # Translations in µm, view to common frame: nominal + translation gives the true offsets up to one shift.
    fovs = result.fovs
    s = np.asarray(fixture.spacing)
    placed = (fovs[["nominal_z", "nominal_y", "nominal_x"]].to_numpy()
              + fovs[["method_translation_z", "method_translation_y", "method_translation_x"]].to_numpy()) / s
    true = fixture.true_zyx()
    assert np.abs((placed - placed.mean(0)) - (true - true.mean(0))).max() <= 2.0
    # A kept link's shift is the difference of its two FOVs' translations, so its residual is small.
    kept = links[links.kept]
    assert (kept.residual < 0.5).all()
    folder = save_placement(result, tmp_path / "cut")
    loaded = load_placement(folder)
    pd.testing.assert_frame_equal(loaded.links, result.links, check_exact=True)


@pytest.mark.slow
@needs_library
def test_no_nan_quality_link_is_kept_and_nan_is_recorded():
    result = run(case("no_signal_zero"))
    links = result.links
    nan = links[(links.reason != "pruned") & links.quality.isna()]
    assert len(nan) and set(nan.reason) == {"nan_quality"} and not nan.kept.any()
    assert "nan_quality_link" in result.fovs.set_index("fov_id").loc["tile4", "flags"].split(";")


@pytest.mark.slow
@needs_library
@pytest.mark.parametrize("name, fields", [("no_overlap", {}), ("two_tiles", {"quality_threshold": 1.0})],
                         ids=["no_overlap", "every_link_dropped"])
def test_with_no_link_left_every_fov_is_failed_and_the_result_is_saved(tmp_path, name, fields):
    fixture = case(name)
    for fallback in (None, "nominal"):
        result = run(fixture, fallback=fallback, **fields)
        report(f"{name} fallback={fallback}", result)
        assert list(result.fovs.status) == ["failed"] * len(fixture.tiles)
        assert result.fovs.origin_x.isna().all() and not result.links.kept.any()
        diagnostics = result.record["method_diagnostics"]
        assert diagnostics["outcome"] == "not_enough_overlap" and diagnostics["error"]
        assert result.record["review"]["state"] == "required"
        folder = save_placement(result, tmp_path / f"{name}-{fallback}")
        loaded = load_placement(folder)
        pd.testing.assert_frame_equal(loaded.fovs, result.fovs, check_exact=True)
    if name == "two_tiles":
        assert list(result.links.reason) == ["quality_below_threshold"]


# --- The linear-part tolerance of the adapter (W-388) -----------------------------------------
# multiview-stitcher returned, on tissue-2D (W-376), a pair transform whose linear part is one unit
# in the last place off the identity. The adapter accepts a linear part within the import's
# LINEAR_TOLERANCE per entry and takes the translation column unchanged.

TISSUE_2D_LINEAR = ((0.9999999999999999, 0.0), (0.0, 1.0))


def _offset_identity(ndim, d, entry):
    """A homogeneous affine whose linear part is the identity with ``d`` added to ``entry``."""
    matrix = np.eye(ndim + 1)
    matrix[:ndim, ndim] = [-1.4006, -0.4921, 2.25][:ndim]
    matrix[entry] += d
    return matrix


def test_the_tolerance_is_one_object_shared_with_the_import():
    import ast

    from starfinder.stitching import _bigstitcher, _methods, _multiview
    assert _methods.LINEAR_TOLERANCE == 1e-9
    assert _multiview.LINEAR_TOLERANCE is _methods.LINEAR_TOLERANCE is _bigstitcher.LINEAR_TOLERANCE
    package = Path(starfinder.stitching.__file__).parent
    defined = [path.name for path in sorted(package.glob("*.py"))
               for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
               if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "LINEAR_TOLERANCE"
                                                       for t in node.targets)]
    assert defined == ["_methods.py"]


@pytest.mark.parametrize("ndim", [2, 3])
def test_translation_accepts_an_entry_within_the_tolerance(ndim):
    from starfinder.stitching._multiview import _translation
    entries = [(i, j) for i in range(ndim) for j in range(ndim)]
    for entry in entries:
        # Off the diagonal the difference is 1e-9 exactly; on it, 1 − 1e-9 is 9.99999972e-10 off in float64.
        for d in ((1e-9, -1e-9) if entry[0] != entry[1] else (-1e-9,)):
            matrix = _offset_identity(ndim, d, entry)
            assert abs(matrix[entry] - float(entry[0] == entry[1])) <= 1e-9
            out = _translation(matrix[None], ndim)   # the library's (t, x_in, x_out) shape
            assert out.dtype == np.float64 and np.array_equal(out, matrix[:ndim, ndim])
    tissue = _offset_identity(ndim, 0.0, (0, 0))
    tissue[:2, :2] = TISSUE_2D_LINEAR
    assert np.array_equal(_translation(tissue, ndim), tissue[:ndim, ndim])


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("d", [2e-9, 1e-6, -2e-9, -1e-6, np.nan])
def test_translation_refuses_an_entry_beyond_the_tolerance(ndim, d):
    from starfinder.stitching._multiview import _translation
    for entry in [(i, j) for i in range(ndim) for j in range(ndim)]:
        with pytest.raises(ValueError, match=r"^multiview-stitcher returned a transform that is not a translation$"):
            _translation(_offset_identity(ndim, d, entry), ndim)


def _spy_translation(monkeypatch):
    """Record each ``_translation`` call of the adapter: (the affine as float64, ndim, the returned translation)."""
    from starfinder.stitching import _multiview
    calls, original = [], _multiview._translation

    def spy(affine, ndim):
        out = original(affine, ndim)
        calls.append((np.asarray(affine, dtype=np.float64).squeeze().copy(), ndim, out.copy()))
        return out

    monkeypatch.setattr(_multiview, "_translation", spy)
    return calls


def _replace_pairwise(monkeypatch, edge, change, optimizer_sees_original=False):
    """Let the library's pairwise registration run, then apply ``change`` to the matrix of one edge.

    With ``optimizer_sees_original`` the library's global optimization gets the edge's
    original matrix back, so that only the adapter reads the changed one.
    """
    from multiview_stitcher import param_resolution, registration
    original, optimize = registration.compute_pairwise_registrations, param_resolution.groupwise_resolution
    returned = {}

    def replaced(*args, **kwargs):
        graph = original(*args, **kwargs)
        returned["transform"] = graph.edges[edge]["transform"]
        transform = returned["transform"].copy(deep=True)
        change(transform.data[0])
        graph.edges[edge]["transform"] = transform
        return graph

    def restored(graph, *args, **kwargs):
        if graph.has_edge(*edge):
            graph.edges[edge]["transform"] = returned["transform"]
        return optimize(graph, *args, **kwargs)

    monkeypatch.setattr(registration, "compute_pairwise_registrations", replaced)
    if optimizer_sees_original:
        monkeypatch.setattr(param_resolution, "groupwise_resolution", restored)


def _replace_global(monkeypatch, view, change):
    """Let the library's global optimization run, then apply ``change`` to the matrix of one view."""
    from multiview_stitcher import param_resolution
    original = param_resolution.groupwise_resolution

    def replaced(*args, **kwargs):
        params, info = original(*args, **kwargs)
        params = dict(params)
        params[view] = params[view].copy(deep=True)
        change(params[view].data[0])
        return params, info

    monkeypatch.setattr(param_resolution, "groupwise_resolution", replaced)


@pytest.mark.slow
@needs_library
def test_the_tissue_2d_transform_is_accepted_and_changes_nothing(monkeypatch):
    fixture = case("plane_2d")   # 2 × 2 tiles of 120 × 120, registered in 2D
    with monkeypatch.context() as patch:
        calls = _spy_translation(patch)
        before = run(fixture)
        pair_calls = calls[:len(before.links[before.links.reason != "pruned"])]
    edge = (0, 1)
    own = pair_calls[0][0]
    tissue = own.copy()
    tissue[:2, :2] = TISSUE_2D_LINEAR
    assert not np.array_equal(tissue[:2, :2], np.eye(2)) and np.array_equal(tissue[:2, 2], own[:2, 2])

    def to_tissue(matrix):
        matrix[...] = tissue

    pair = tuple(fixture.fov_ids[i] for i in edge)
    results = {}
    for optimizer_sees_original in (True, False):
        with monkeypatch.context() as patch:
            calls = _spy_translation(patch)
            _replace_pairwise(patch, edge, to_tissue, optimizer_sees_original)
            after = results[optimizer_sees_original] = run(fixture)   # no exception
        matrix, ndim, taken = calls[0]
        assert ndim == 2 and np.array_equal(matrix, tissue)
        assert taken.dtype == np.float64 and np.array_equal(taken, tissue[:2, 2])
        assert np.array_equal(taken, own[:2, 2])
        link = after.links.set_index(["fov_a", "fov_b"]).loc[pair]
        assert [link.shift_y, link.shift_x] == list(-tissue[:2, 2])
        assert list(after.fovs.status) == list(before.fovs.status) == ["image_supported"] * 4
    # Only the adapter reads the replaced matrix: every status and every other value equals the run
    # without the replacement, bit for bit.
    isolated = results[True]
    pd.testing.assert_frame_equal(isolated.fovs, before.fovs, check_exact=True)
    pd.testing.assert_frame_equal(isolated.links, before.links, check_exact=True)
    assert isolated.record == before.record
    # The library's global optimization reads the replaced matrix too, as on tissue-2D: every status,
    # flag, component, link decision and the review are the same; the origins it solves are measured.
    literal = results[False]
    exact = ["fov_id", "status", "flags", "component", "n_links_kept", "nominal_z", "nominal_y", "nominal_x"]
    pd.testing.assert_frame_equal(literal.fovs[exact], before.fovs[exact], check_exact=True)
    columns = [c for c in before.links.columns if c != "residual"]
    pd.testing.assert_frame_equal(literal.links[columns], before.links[columns], check_exact=True)
    assert literal.record["review"] == before.record["review"]
    numeric = [c for c in before.fovs.columns if c not in exact and before.fovs[c].dtype.kind == "f"]
    differences = {c: float(np.abs(literal.fovs[c] - before.fovs[c]).max()) for c in numeric}
    differences["links.residual"] = float(np.nanmax(np.abs(literal.links.residual - before.links.residual)))
    print(json.dumps({"case": "plane_2d, the optimizer reads the tissue-2D matrix",
                      "max_abs_difference_from_the_run_without_replacement": differences}))


@pytest.mark.slow
@needs_library
@pytest.mark.parametrize("site", ["pairwise", "global"])
@pytest.mark.parametrize("name, ndim", [("plane_2d", 2), ("two_tiles", 3)], ids=["2d", "3d"])
@pytest.mark.parametrize("d, accepted", [(1e-9, True), (2e-9, False), (1e-6, False)],
                         ids=["1e-9", "2e-9", "1e-6"])
def test_the_bound_at_both_calls_of_the_adapter(monkeypatch, site, name, ndim, d, accepted):
    fixture = case(name)
    entry = (0, 1)   # off the diagonal, so the entry is d off the identity exactly

    def offset(matrix):
        matrix[entry] += d

    if site == "pairwise":
        _replace_pairwise(monkeypatch, (0, 1), offset)
    else:
        _replace_global(monkeypatch, 1, offset)
    calls = _spy_translation(monkeypatch)
    if not accepted:
        with pytest.raises(ValueError, match=r"^multiview-stitcher returned a transform that is not a translation$"):
            run(fixture)
        return
    result = run(fixture)
    n_pairs = len(result.links[result.links.reason != "pruned"])
    matrix, called_ndim, taken = calls[0] if site == "pairwise" else calls[n_pairs + 1]
    assert called_ndim == ndim and matrix[entry] == d
    assert np.array_equal(taken, matrix[:ndim, ndim])
    assert list(result.fovs.status) == ["image_supported"] * len(fixture.tiles)
    if site == "global":
        axes = ["y", "x"] if ndim == 2 else ["z", "y", "x"]
        row = result.fovs.iloc[1]
        assert [row[f"method_translation_{a}"] for a in axes] == list(taken)


# --- MV6: determinism ------------------------------------------------------------------------

DETERMINISM = """
import hashlib, json, sys
import dask
from test.stitching_cut_fixtures import case
from starfinder.stitching import MultiviewStitcherConfig, stitch
fixture = case("cut_synthetic")
with dask.config.set(scheduler="synchronous"):
    result = stitch(fixture.inputs(), fixture.table(), config=MultiviewStitcherConfig())
columns = [f"method_translation_{a}" for a in "zyx"]
data = result.fovs[columns].to_numpy(dtype="float64").tobytes()
print(json.dumps({"sha256": hashlib.sha256(data).hexdigest(), "hex": data.hex()}))
"""


@pytest.mark.slow
@needs_library
def test_translations_are_bitwise_identical_in_two_processes():
    outputs = []
    for _ in range(2):
        done = subprocess.run([sys.executable, "-c", DETERMINISM], cwd=SRC, env=_environment(), capture_output=True,
                              text=True, check=True)
        outputs.append(json.loads(done.stdout.strip().splitlines()[-1]))
    print(json.dumps({"translations_sha256": [o["sha256"] for o in outputs]}))
    assert outputs[0] == outputs[1]
    assert len(bytes.fromhex(outputs[0]["hex"])) == 9 * 3 * 8
    assert hashlib.sha256(bytes.fromhex(outputs[0]["hex"])).hexdigest() == outputs[0]["sha256"]
