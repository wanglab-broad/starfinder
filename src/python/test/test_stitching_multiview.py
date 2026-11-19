"""MV1 to MV6 of design/validation/stitching-assembly.md: the ``multiview_stitcher`` placement method.

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
import math
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
        (4, 4, 4), "phase_correlation", None, None, True, "alternating_pattern", 0.0, "global_optimization",
        None, None, None)
    assert MultiviewStitcherConfig(quality_threshold=0.3).quality_threshold == 0.3
    assert MultiviewStitcherConfig(quality_threshold="auto").quality_threshold == "auto"
    assert MultiviewStitcherConfig(binning_zyx=[2, 2, 2], pruning=None).binning_zyx == (2, 2, 2)


@pytest.mark.parametrize("fields", [
    dict(binning_zyx=(0, 4, 4)), dict(binning_zyx=(4, 4, 0)), dict(quality_threshold=1.5),
    dict(quality_threshold="Auto"), dict(quality_threshold=math.nan),
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


# --- The link-quality threshold estimated from the data (W-874) ------------------------------
# A synthetic pair with a known offset: the fixed overlap is a window of an i.i.d. texture, and the moving
# overlap the window that the shift ``t`` maps onto it (``moving[o + t] = fixed[o]``), as in the library's pixel
# space. Rank correlation is unchanged by the [0, 1] rescaling, so the expected qualities come from the slices.

KNOWN_SHIFT = (3.0, -2.0)


def _known_pair(shape=(48, 16), shift=KNOWN_SHIFT):
    texture = np.random.default_rng(874).random((64, 64))
    (h, w), (ty, tx), (y0, x0) = shape, map(int, shift), (8, 24)
    return texture[y0:y0 + h, x0:x0 + w], texture[y0 - ty:y0 - ty + h, x0 - tx:x0 - tx + w]


def _sliced_quality(fixed, moving, offset):
    """Spearman correlation of ``fixed[o]`` and ``moving[o + offset]`` over the o where both exist (integers)."""
    from scipy import stats
    region = tuple(slice(max(0, -int(u)), min(n, n - int(u))) for n, u in zip(fixed.shape, offset))
    moved = tuple(slice(r.start + int(u), r.stop + int(u)) for r, u in zip(region, offset))
    return float(stats.spearmanr(fixed[region].ravel(), moving[moved].ravel()).correlation), fixed[region].size


@pytest.mark.contract
def test_the_default_threshold_resolves_per_tile_set():
    # Decision B (W-874, Jiahao, 2026-11-19): the default is "auto" for single-plane tiles and 0.7 for Z > 1; an
    # explicit number or an explicit "auto" applies to either kind.
    from starfinder.stitching._multiview import FIXED_DEFAULT, FIXED_DEFAULT_RULE, _resolved_threshold
    assert (FIXED_DEFAULT, FIXED_DEFAULT_RULE) == (0.7, "fixed_default_z_above_1")
    assert _resolved_threshold(None, True) == ("auto", None)
    assert _resolved_threshold(None, False) == ("fixed_default_z_above_1", 0.7)
    assert _resolved_threshold("auto", True) == ("auto", None)
    assert _resolved_threshold("auto", False) == ("auto", None)
    assert _resolved_threshold(0.3, True) == ("given", 0.3)
    assert _resolved_threshold(-1.0, False) == ("given", -1.0)


@pytest.mark.contract
def test_null_offsets_move_the_shift_half_the_overlap_width_along_the_narrow_axis():
    from starfinder.stitching._multiview import _null_offsets
    assert [list(o) for o in _null_offsets((48, 16), KNOWN_SHIFT)] == [[3.0, -10.0], [3.0, 6.0]]
    assert [list(o) for o in _null_offsets((15, 40), (0.5, 0.0))] == [[-6.5, 0.0], [7.5, 0.0]]
    # 3D: the narrower of Y and X, never Z.
    assert [list(o) for o in _null_offsets((4, 30, 20), (0.0, 0.0, 0.0))] == [[0.0, 0.0, -10.0], [0.0, 0.0, 10.0]]
    assert [list(o) for o in _null_offsets((4, 1, 20), (0.0, 0.0, 0.0))] == [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]


@pytest.mark.contract
def test_null_qualities_of_a_pair_with_a_known_offset():
    from starfinder.stitching._multiview import _null_offsets, _null_qualities, _quality_at
    fixed, moving = _known_pair()
    # At the known offset the moved overlap is the fixed one: quality 1.
    assert _quality_at(fixed, moving, KNOWN_SHIFT) == pytest.approx(1.0, abs=1e-12)
    null = _null_qualities(fixed, moving, KNOWN_SHIFT)
    for value, offset in zip(null, _null_offsets(fixed.shape, KNOWN_SHIFT)):
        expected, n = _sliced_quality(fixed, moving, offset)
        assert value == pytest.approx(expected, abs=1e-12)
        # No texture lines up: within 5 standard deviations (1/sqrt(n - 1)) of 0.
        assert abs(value) <= 5 / math.sqrt(n - 1)
    # A shift that leaves less than a tenth of the moving overlap has no value.
    assert math.isnan(_quality_at(fixed, moving, (0.0, 15.0)))


@pytest.mark.contract
def test_auto_threshold_is_the_larger_of_the_floor_and_the_null_quantile():
    from starfinder.stitching._multiview import NULL_QUANTILE, QUALITY_FLOOR, _auto_threshold

    def pairs(edges):
        return [[f"FOV{a}", f"FOV{b}"] for a, b in edges]

    assert (QUALITY_FLOOR, NULL_QUANTILE) == (0.2, 0.99)
    # 101 values 0, 0.01, ..., 1: the 99th percentile (linear) is 0.99.
    values = np.linspace(0.0, 1.0, 101)
    nulls = {(0, 1): list(values[:50]), (1, 2): list(values[50:]) + [math.nan]}
    threshold, record = _auto_threshold(nulls, [(0, 1), (1, 2)], pairs)
    assert record["rule"] == "auto"
    assert threshold == pytest.approx(0.99, abs=1e-12) and record["threshold"] == threshold
    assert record["null"]["values"] == 101 and record["null"]["pairs"] == 2
    assert record["null"]["quantile"] == pytest.approx(0.99, abs=1e-12)
    assert record["null"]["min"] == 0.0 and record["null"]["max"] == 1.0
    assert record["null_pairs"][1][:2] == ["FOV1", "FOV2"] and record["null_pairs"][1][-1] is None
    # A null below the floor gives the floor; no null value at all gives the floor too.
    low, record = _auto_threshold({(0, 1): [0.05, -0.1]}, [(0, 1)], pairs)
    assert low == 0.2 and record["null"]["quantile"] < 0.2
    empty, record = _auto_threshold({(0, 1): [math.nan, math.nan]}, [(0, 1), (1, 2)], pairs)
    assert empty == 0.2 and record["null"] == {"pairs": 0, "values": 0}


@pytest.mark.slow
@needs_library
def test_the_recording_registration_is_the_library_s_and_measures_its_quality():
    import xarray as xr
    from multiview_stitcher import registration

    from starfinder.stitching._multiview import _null_qualities, _quality_at, _recording
    fixed, moving = _known_pair()
    null = []
    data = {"fixed_data": xr.DataArray(fixed, dims=("y", "x")), "moving_data": xr.DataArray(moving, dims=("y", "x"))}
    result = _recording(null)(**data)
    library = registration.phase_correlation_registration(**data)
    assert np.array_equal(result["affine_matrix"], library["affine_matrix"]) and result["quality"] == library["quality"]
    shift = np.asarray(result["affine_matrix"])[:2, 2]
    assert list(shift) == list(KNOWN_SHIFT)
    # The null's measure is the library's: at the library's shift it gives the library's quality.
    assert _quality_at(fixed, moving, shift) == pytest.approx(library["quality"], abs=1e-12)
    assert null == _null_qualities(fixed, moving, shift)


@pytest.mark.slow
@needs_library
def test_auto_threshold_against_an_explicit_number_on_the_cut_fixture():
    fixture = case("cut_synthetic")   # 3D tiles: an explicit "auto" applies to them too
    auto = run(fixture, quality_threshold="auto")
    quality = auto.record["method_diagnostics"]["link_quality"]
    assert quality["rule"] == "auto"
    values = [q for pair in quality["null_pairs"] for q in pair[2:] if q is not None]
    registered = auto.links[auto.links.reason != "pruned"]
    assert len(quality["null_pairs"]) == len(registered) and quality["null"]["values"] == len(values)
    assert quality["threshold"] == max(0.2, float(np.quantile(values, 0.99)))
    below = registered.quality < quality["threshold"]
    assert (registered.reason[below] == "quality_below_threshold").all()
    assert registered.reason[~below].isin(["", "optimizer_removed"]).all()
    # The same number given explicitly: the same placement, bit for bit, and no null measured.
    explicit = run(fixture, quality_threshold=quality["threshold"])
    pd.testing.assert_frame_equal(explicit.fovs, auto.fovs, check_exact=True)
    pd.testing.assert_frame_equal(explicit.links, auto.links, check_exact=True)
    assert "link_quality" not in explicit.record["method_diagnostics"]
    assert explicit.record["method"]["config"]["quality_threshold"] == quality["threshold"]
    assert auto.record["method"]["config"]["quality_threshold"] == "auto"
    # The default for these 3D tiles is 0.7, higher: the link tile4–tile5 (0.46) is then dropped by rule L1.
    default = run(fixture)
    assert default.record["method"]["config"]["quality_threshold"] is None
    assert default.record["method_diagnostics"]["link_quality"] == {"rule": "fixed_default_z_above_1",
                                                                     "threshold": 0.7}
    strict = default.links.set_index(["fov_a", "fov_b"])
    loose = auto.links.set_index(["fov_a", "fov_b"])
    assert strict.loc[("tile4", "tile5"), "reason"] == "quality_below_threshold"
    assert loose.loc[("tile4", "tile5"), "reason"] != "quality_below_threshold"


@pytest.mark.slow
@needs_library
@pytest.mark.parametrize("name, given, rule", [
    ("plane_2d", None, "auto"), ("two_tiles", None, "fixed_default_z_above_1"), ("plane_2d", 0.5, "given"),
    ("two_tiles", 0.5, "given"), ("two_tiles", "auto", "auto")],
    ids=["single_plane_default", "z_above_1_default", "single_plane_given", "z_above_1_given", "z_above_1_auto"])
def test_the_record_names_the_rule_that_set_the_threshold(name, given, rule):
    fixture = case(name)
    assert (fixture.inputs()[0].grid.shape_zyx[0] == 1) == (name == "plane_2d")
    result = run(fixture, **({} if given is None else {"quality_threshold": given}))
    quality = result.record["method_diagnostics"].get("link_quality")
    assert result.record["method"]["config"]["quality_threshold"] == given
    if rule == "given":
        assert quality is None   # the method config names the given value
        threshold = given
    elif rule == "auto":
        assert quality["rule"] == "auto" and quality["null"]["values"] > 0
        threshold = quality["threshold"]
    else:
        assert quality == {"rule": "fixed_default_z_above_1", "threshold": 0.7}
        threshold = 0.7
    registered = result.links[(result.links.reason != "pruned") & result.links.quality.notna()]
    assert (registered.reason[registered.quality < threshold] == "quality_below_threshold").all()
    assert (registered.reason[registered.quality >= threshold] != "quality_below_threshold").all()


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


TISSUE_EDGE = (0, 1)
# The operator's corrected criterion (2026-10-08): the optimization is not exact under a change of
# one unit in the last place, so the solved values equal the run without the replacement within this.
SOLVED_TOLERANCE_UM = 1e-9


def _tissue_runs(monkeypatch, optimizer_sees_original):
    """``plane_2d`` without and with the tissue-2D matrix for the pair (0, 1), and what the adapter read.

    Checks, for the run with the replacement: no exception; the adapter read the replaced matrix and
    took its last column as float64, bit for bit; the link's shift is that column negated.
    """
    fixture = case("plane_2d")   # 2 × 2 tiles of 120 × 120, registered in 2D
    with monkeypatch.context() as patch:
        calls = _spy_translation(patch)
        before = run(fixture)
    own = calls[0][0]   # the first registered pair is (0, 1)
    tissue = own.copy()
    tissue[:2, :2] = TISSUE_2D_LINEAR
    assert not np.array_equal(tissue[:2, :2], np.eye(2)) and np.array_equal(tissue[:2, 2], own[:2, 2])

    def to_tissue(matrix):
        matrix[...] = tissue

    with monkeypatch.context() as patch:
        calls = _spy_translation(patch)
        _replace_pairwise(patch, TISSUE_EDGE, to_tissue, optimizer_sees_original)
        after = run(fixture)   # no exception
    matrix, ndim, taken = calls[0]
    assert ndim == 2 and np.array_equal(matrix, tissue)
    assert taken.dtype == np.float64 and np.array_equal(taken, tissue[:2, 2]) and np.array_equal(taken, own[:2, 2])
    link = after.links.set_index(["fov_a", "fov_b"]).loc[tuple(fixture.fov_ids[i] for i in TISSUE_EDGE)]
    assert [link.shift_y, link.shift_x] == list(-tissue[:2, 2])
    return before, after


def _record_differences(after, before, path="record"):
    """Compare two records: every float within the tolerance (NaN with NaN), everything else exactly.

    Returns the largest absolute float difference, by path.
    """
    if isinstance(before, dict):
        assert isinstance(after, dict) and after.keys() == before.keys(), path
        found = {}
        for key in before:
            found.update(_record_differences(after[key], before[key], f"{path}.{key}"))
        return found
    if isinstance(before, (list, tuple)):
        assert type(after) is type(before) and len(after) == len(before), path
        found = {}
        for i, (a, b) in enumerate(zip(after, before)):
            found.update(_record_differences(a, b, f"{path}[{i}]"))
        return found
    if isinstance(before, float) and not isinstance(before, bool):
        assert isinstance(after, float), path
        if math.isnan(before):
            assert math.isnan(after), path
            return {}
        difference = abs(after - before)
        assert difference <= SOLVED_TOLERANCE_UM, (path, after, before)
        return {path: difference} if difference else {}
    assert after == before, (path, after, before)
    return {}


@pytest.mark.slow
@needs_library
def test_the_tissue_2d_transform_is_accepted_and_changes_nothing(monkeypatch):
    # The replaced matrix reaches the library's global optimization, as it does in production.
    before, after = _tissue_runs(monkeypatch, optimizer_sees_original=False)
    # Every status, flag, component, link decision and the review state equal the run without it.
    exact_fovs = ["fov_id", "status", "flags", "component", "n_links_kept"]
    assert list(after.fovs.status) == ["image_supported"] * 4
    pd.testing.assert_frame_equal(after.fovs[exact_fovs], before.fovs[exact_fovs], check_exact=True)
    exact_links = ["fov_a", "fov_b", "kept", "reason"]
    pd.testing.assert_frame_equal(after.links[exact_links], before.links[exact_links], check_exact=True)
    assert after.record["review"] == before.record["review"]
    # Every other value of the placement (origins, corrections, method translations, link shifts,
    # quality and residuals, the record's numbers) within 1e-9 µm, NaN where the run without it has NaN.
    differences = {}
    for name, columns in (("fovs", [c for c in before.fovs.columns if c not in exact_fovs]),
                          ("links", [c for c in before.links.columns if c not in exact_links])):
        frame_after, frame_before = getattr(after, name)[columns], getattr(before, name)[columns]
        pd.testing.assert_frame_equal(frame_after, frame_before, check_exact=False, rtol=0.0,
                                      atol=SOLVED_TOLERANCE_UM)
        delta = (frame_after - frame_before).abs().max()
        differences.update({f"{name}.{c}": float(v) for c, v in delta.items() if v > 0})
    differences.update(_record_differences(after.record, before.record))
    print(json.dumps({"case": "plane_2d, the optimization reads the tissue-2D matrix",
                      "max_abs_difference_from_the_run_without_replacement_um": differences}))
    assert max(differences.values(), default=0.0) <= SOLVED_TOLERANCE_UM, differences


@pytest.mark.slow
@needs_library
def test_the_adapter_alone_changes_nothing_when_the_optimization_reads_the_library_matrix(monkeypatch):
    """Isolates the adapter: only it reads the tissue-2D matrix; the optimization gets the library's own.

    Everything then equals the run without the replacement bit for bit. This is a check of the adapter
    alone; the criterion of the tissue-2D case is the test above.
    """
    before, after = _tissue_runs(monkeypatch, optimizer_sees_original=True)
    pd.testing.assert_frame_equal(after.fovs, before.fovs, check_exact=True)
    pd.testing.assert_frame_equal(after.links, before.links, check_exact=True)
    assert after.record == before.record


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
