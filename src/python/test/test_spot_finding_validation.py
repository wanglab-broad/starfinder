"""Engineering validation of the §2.7 spot-finding methods (spot-finding task group 6, W-274).

Checks S1 to S16 of docs/spot-finding-algorithms.md ("Engineering validation
design"): known-answer synthetic fixtures (spot_finding_fixtures, the W-266
ports in spot_finding_scenes and learned_detectors, and the multiround fixture
of test_spot_finding_rounds) with the pass/fail tolerances of the design table,
fixed before the run. One test function per check, parametrized by method,
model, case and seed (100 to 102). Local maxima and the Starfish LoG run in
the default tier; Spotiflow and Piscis in the extended tier (-m extended, CPU,
CUDA_VISIBLE_DEVICES="", one thread), loading their weights from
STARFINDER_WEIGHTS_DIR. Metrics come from evaluate_spots, localization_errors
and classify_detections with W-266's policy (greedy, 3.0 voxels, inclusive)
unless a check says otherwise. Each test records the values it gates with
record_property (the worker notes read them from the JUnit XML).

This is engineering validation only: every method is compared with the
check's tolerance, never with another method (E02).
"""
from dataclasses import replace
from functools import cache
import hashlib
import json
from pathlib import Path
import socket
import sys
import types
import urllib.request
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from starfinder.dataset import CheckpointConfig
from starfinder.evaluation.spot_finding import evaluate_spots
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.io._checkpoint import _jsonable, _tuples, candidates_frame
from starfinder.preprocessing import normalize_intensity
from starfinder.spot_finding import (KNOWN_WEIGHTS, SPOT_FINDING_METHODS, ChannelOverride, LocalMaximaConfig,
    MissingWeightsError, PiscisConfig, SpotFindingBackendUnavailableError, SpotFindingPlan, SpotFindingWarning,
    SpotiflowConfig, StarfishLogConfig, WeightsFile, WeightsHashMismatchError, find_spots, resolve_weights)
from starfinder.spot_finding import _learned
from starfinder.spot_finding._starfish_log import starfish_view

from . import spot_finding_fixtures as fixtures
from .learned_detectors import THREAD_VARIABLES, W266_MATCH, evaluate, run_python, seam_scene, table_digest
from .spot_finding_scenes import FORMED16_SHAPE, SEEDS, formed16, isolated_scene
from .test_spot_finding_diagnostics import assert_typed_empty
from .test_spot_finding_golden import PINNED_FIND_SPOTS, golden_dataset
from .test_spot_finding_local_maxima import RADIUS, RECIPE1, s16
from .test_spot_finding_rounds import (CHANNELS as ROUND_CHANNELS, DATA as VERSION_2_CHECKPOINT, PLAN_KEYS, ROUNDS,
    SHAPE_ZYX as MULTIROUND_SHAPE, _round_trip, assert_reloaded, multiround, multiround_fov)
from .test_starfish_log import FIXTURES as PARITY, RECORD as PARITY_RECORD, TABLES as PARITY_TABLES
from .test_starfish_log import parity_image, starfish_table

META = ImageMetadata("validation")
NAMESPACE = "validation/test"

# --- Tolerances (docs/spot-finding-algorithms.md, "Engineering validation design") -------------------------
# Fixed before the run and never re-tuned in it: a correct implementation that misses one is recorded as a
# strict expected failure with the bound unchanged and reported to Jiahao (W-268; as W-258 did for §2.6).

# S1, W-266 detectors.csv: recall 1.0 for every method; precision 0.98 to 1.0 in 3D and 1.0 on Z=1.
S1_RECALL = 1.0
S1_PRECISION = 0.98
# S1, LoG on iso_z1, W-266 detectors.csv: recall 0.45, 0.51, 0.48 and precision 1.0 (threshold 0.01 is near
# the median-spot response of a brightness-1500 plane).
S1_LOG_Z1_RECALL = (0.40, 0.60)
# S1, LM on iso_z1_sparse: provisional, W-266 has no row (LM is empty on iso_z1 at its density, which is
# flagged, not gated).
S1_LM_SPARSE_RECALL = 1.0
S1_LM_SPARSE_PRECISION = 0.98
# S2, W-266 localization-per-axis.csv maxima: Spotiflow 3D models, 3D distance (max 0.432 voxels).
S2_SF3_DISTANCE = 0.5
# S2, Spotiflow 2D models, lateral distance (max 0.404 px).
S2_SF2_LATERAL = 0.5
# S2, Piscis plane mode, lateral distance (max 0.098 px).
S2_PI_PLANE_LATERAL = 0.15
# S2, Piscis stack mode, lateral distance (max 0.119 px).
S2_PI_STACK_LATERAL = 0.15
# S2, Piscis stack mode, absolute Z error: provisional, W-266 max 1.763 voxels on one scene (W-266 withdrew 1.0).
S2_PI_STACK_ABS_Z = 2.0
# S2, LM and LoG on iso3d, 3D distance (max 0.818 voxels).
S2_LM_LOG_3D_DISTANCE = 0.9
# S2, LoG on iso_z1, lateral distance (max 0.546 px).
S2_LOG_Z1_LATERAL = 0.9
# S2, LM on iso_z1_sparse, lateral distance: provisional, W-266 has no LM row on a plane.
S2_LM_Z1_LATERAL = 0.9
# S3, pair recall: the Z=1 lateral pairs of SF2 and PI derive from W-266 iso_z1 (all 100 spots at 6-px spacing
# resolved) and the axial pairs of SF3 from the W-266 aligned probe (smfish_3d found all 147 spots stacked 8
# planes apart); the other 3D cases are provisional (no W-266 pair rows for LM, LoG or PI in 3D, nor for
# lateral pairs in 3D). PI axial pairs are not gated: stack mode merges them by specification.
S3_PAIR_RECALL = 1.0
# S4, provisional (W-266 did not measure background offsets): LM noise thresholds exceed channel 0's by the
# offset within this bound (the offset cancels in the median, the MAD and the maximum filter).
S4_LM_THRESHOLD_OFFSET = 1e-9
# S4, provisional: LoG, SF and PI coordinates of channels 1 and 2 against channel 0 (voxels), about twice
# W-266's one- against four-thread differences (at most 4.2e-6), a different perturbation.
S4_COORDINATES = 1e-5
S4_OFFSETS = {1: 300, 2: 1500}
S4_OVERRIDES = {"local_maxima": dict(threshold_value=6.0), "starfish_log": dict(threshold=0.02),
                "spotiflow": dict(prob_thresh=0.5), "piscis": dict(threshold=0.6)}
# S7, provisional (W-266 placed no truth spot near a face): LoG, SF and PI detect every spot at least this
# many voxels from every face; spots closer are reported, not gated.
S7_GATED_DISTANCE = 2
# S8, Piscis with input_size=32, W-266 piscis-seams.csv: every off-boundary and control spot single, maximum
# lateral shift against the untiled run 0.147 px.
S8_PI_CANDIDATES = 1
S8_PI_SHIFT = 0.2
PISCIS_SEAM_INPUT_SIZE = 32
# S8, Spotiflow with forced n_tiles, W-266 medium and 1x512x512: same detections, matched shift at most
# 0.018 px. On 64-pixel images this only shows that n_tiles is passed and recorded.
S8_SF_SHIFT = 0.05
S8_SF_TILES = {"iso_z1": (2, 2), "iso3d": (1, 2, 2)}
# S10, provisional (option A of the contract, a new interface): rows of one shared position per round lie
# within the matching threshold of its centre.
S10_RADIUS = W266_MATCH["threshold"]
# S16, the W-218 notebook policy (W-267 re-measurement): greedy, 5.0 voxels, exclusive, eligible truth
# center_in_bounds; with the merge radius (2, 2, 2) same-channel duplicates are 0 and no match is lost; without
# border exclusion no eligible amplicon within this distance of a face is missed. Derived from the W-218
# re-measurement on `small` (2 same-channel ties removed, border misses 5 -> 0) and provisional, because
# formed16 is smaller and denser than `small`.
S16_SAME_CHANNEL_DUPLICATES = 0
S16_FACE_DISTANCE = 1

# --- Methods ---------------------------------------------------------------------------------------------

# LoG uses the starfish ISS tutorial values W-266 used (an example, not a default); every other method its
# native defaults.
LOG = StarfishLogConfig(min_sigma=1, max_sigma=10, num_sigma=30, threshold=0.01)
METHODS = {"local_maxima": LocalMaximaConfig(), "starfish_log": LOG,
           **{f"spotiflow-{m}": SpotiflowConfig(m) for m in ("synth_3d", "smfish_3d", "general", "hybiss")},
           **{f"piscis-{m}": PiscisConfig(m) for m in ("20230905", "20251212")}}
LM, LOG_ = ["local_maxima"], ["starfish_log"]
SF3, SF2 = ["spotiflow-synth_3d", "spotiflow-smfish_3d"], ["spotiflow-general", "spotiflow-hybiss"]
PI = ["piscis-20230905", "piscis-20251212"]
LEARNED = ("spotiflow", "piscis")


def family(method):
    return method.split("-")[0]


def strict_xfail(reason):
    """A correct implementation that misses its bound: the bound is unchanged and the case goes to Jiahao."""
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)


# The cases a correct implementation misses (W-274 worker notes, with the measured values).
S4_PISCIS_OFFSETS = strict_xfail(
    "Piscis standardizes each zero-padded 256-pixel tile, so a constant offset changes its input: channels 1 and "
    "2 are 1.001 to 1.009 voxels from channel 0 (integer stack-mode z changes by 1; lateral up to 0.135 px), "
    "above the provisional 1e-5 bound; counts are equal (W-274, reported to Jiahao)")
S5_SMFISH_102 = strict_xfail(
    "smfish_3d on coincident seed 102: one channel-0 spot (z 11.69) has probability 0.396, below the stored "
    "prob_thresh 0.4, so channel 0 recall is 0.95; the channel's table equals its single-channel run (W-274, "
    "reported to Jiahao)")
S7_SYNTH_3D = strict_xfail(
    "synth_3d on borders (16x64x64) has no candidate within 16 voxels of the spots 1 and 2 planes from the "
    "low Z face, while it finds the spots at 0 and 3 planes; the spot at 2 planes is gated (W-274, reported to "
    "Jiahao)")
S10_PISCIS_COLUMNS = strict_xfail(
    "Piscis stack mode merges the multiround fixture's two spots of one column (z 4 and 11, 7 planes apart) into "
    "one component near z 8 (specified behavior, W-266 choice 4), so a shared position in such a column has no "
    "row within 3 voxels in that round (W-274, reported to Jiahao)")


def params(*rows, marks=None):
    """pytest.param for each row (a tuple whose first item is the method), extended-tier when it runs a
    Spotiflow or Piscis model of METHODS; marks maps a row to extra marks (strict expected failures)."""
    out = []
    for row in rows:
        extra = list((marks or {}).get(row, ()))
        if row[0] in METHODS and family(row[0]) in LEARNED:
            extra.append(pytest.mark.extended)
        out.append(pytest.param(*row, id="-".join(str(v) for v in row), marks=extra))
    return out


@pytest.fixture(autouse=True)
def one_thread(monkeypatch):
    """One numerical thread and no GPU (the checks also set these before the process starts)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")


def backend(config):
    """Skip without the method's extra; hold torch to one thread for the learned methods."""
    if config.method in LEARNED:
        pytest.importorskip(config.method)
        pytest.importorskip("torch").set_num_threads(1)


def detect(image, config):
    backend(config.config if isinstance(config, SpotFindingPlan) else config)
    return find_spots(image, config=config, metadata=META, spot_namespace=NAMESPACE)


def matched_truth(match):
    return {int(i) for i, _, _ in match.details["matched_pairs"]}


def paired_shift(spots, reference):
    """Largest distance of a one-to-one nearest-neighbour pairing of two tables' ZYX (inf if none exists)."""
    a, b = (t[["z", "y", "x"]].to_numpy(float) for t in (spots, reference))
    if len(a) != len(b):
        return float("inf")
    if not len(a):
        return 0.0
    distances, index = cKDTree(b).query(a)
    return float(distances.max()) if len(set(index.tolist())) == len(a) else float("inf")


def report(record_property, **values):
    """Record the gated values of one case for the worker notes."""
    record_property("values", json.dumps(values, default=float, sort_keys=True))


def settings(config):
    return _tuples(_jsonable(config))


def assert_settings(effective, config):
    """Every setting the config fixes (not None) is in effect; None settings are the method's resolution."""
    for key, value in settings(config).items():
        if key != "channel_labels" and value is not None:
            assert effective[key] == value, key


def without_labels(effective):
    return {k: v for k, v in effective.items() if k != "channel_labels"}


# --- Scenes, cached so that each detection runs once per process ----------------------------------------

def scene(name, seed):
    """(image, truth) of the isolated-spot scenes iso3d, iso_z1 (W-266) and iso_z1_sparse."""
    return fixtures.iso_z1_sparse(seed) if name == "iso_z1_sparse" else isolated_scene(name, seed)


@cache
def isolated(method, name, seed):
    """(image, truth, result) of one method on an isolated-spot scene (S1, S2, S8, S9, S11, S13)."""
    image, truth = scene(name, seed)
    return image, truth, detect(image, METHODS[method])


@cache
def multiround_run(method, seed):
    """(shared centres, FOV detected with the plan rounds=ROUNDS, default result without rounds) (S10, S11)."""
    images, shared = multiround(seed)
    config = METHODS[method]
    backend(config)
    fov = multiround_fov(Path("unused"), images)
    fov.find_spots(config=SpotFindingPlan(config, rounds=ROUNDS))
    default = multiround_fov(Path("unused"), images)
    default.find_spots(config=config)
    return shared, fov, default.spot_result


def test_fixtures_stay_within_the_resource_bounds_and_their_stated_densities():
    """32x64x64 voxels, four channels and four rounds at most; densities and separations as the design states."""
    seed = SEEDS[0]
    sparse, sparse_truth = fixtures.iso_z1_sparse(seed)
    pair_image, pair_truth, kinds = fixtures.pairs(seed)
    channel_image, channel_truth = fixtures.channels(seed)
    coincident_image, coincident_truth = fixtures.coincident(seed)
    border_image, border_truth, distances = fixtures.borders(seed)
    plane_image, plane_truth, plane_distances = fixtures.borders(seed, z1=True)
    images, _ = multiround(seed)
    arrays = [sparse, pair_image, fixtures.pairs_z1(seed)[0], channel_image, coincident_image, border_image,
              plane_image, fixtures.zero_channels(seed), *images.values(), formed16(seed)[0]]
    for array in arrays:
        assert all(n <= m for n, m in zip(array.shape[:3], (32, 64, 64)))
        assert array.ndim == 3 or array.shape[3] <= 4
    assert len(images) == 3 <= 4 and MULTIROUND_SHAPE == (16, 64, 64)
    assert len(sparse_truth) == 25 and round(25 / sparse.size, 4) == 0.0061
    assert (len(pair_truth), (kinds == "lateral").sum()) == (40, 24) and round(40 / pair_image.size, 5) == 3.1e-4
    assert len(channel_truth) == 25 and round(25 / channel_image[..., 0].size, 5) == 3.8e-4
    assert len(coincident_truth) == 20 and round(20 / coincident_image[..., 0].size, 5) == 3.1e-4
    assert (len(border_truth), len(plane_truth)) == (24, 16)
    assert sorted(np.bincount(distances)) == [6, 6, 6, 6] and sorted(np.bincount(plane_distances)) == [4, 4, 4, 4]
    shape = np.array(border_image.shape)
    assert np.array_equal(np.minimum(border_truth, shape - 1 - border_truth).min(axis=1), distances)
    assert fixtures.min_separation(border_truth) >= 8 and fixtures.min_separation(plane_truth) >= 8
    # Members of one pair are 6 px (lateral) or 8 planes (axial) apart; different pairs at least 12 voxels.
    members = pair_truth.reshape(20, 2, 3)
    separation = np.linalg.norm(members[:, 0] - members[:, 1], axis=1)
    assert np.allclose(separation[:12], 6.0) and np.allclose(separation[12:], 8.0)
    between = np.linalg.norm(members[:, None, :, None] - members[None, :, None, :], axis=-1)
    assert between[~np.eye(20, dtype=bool)].min() >= 12
    assert np.array_equal(channel_image[..., 1], channel_image[..., 0] + 300)
    assert np.array_equal(channel_image[..., 2], channel_image[..., 0] + 1500)
    assert not np.array_equal(channel_image[..., 3], channel_image[..., 0])
    zeros = fixtures.zero_channels(seed)
    assert (np.mean(zeros[..., 0] == 0), np.mean(zeros[..., 1] == 0)) == (0.6, 0.4)


# --- S1: isolated-spot recall and precision --------------------------------------------------------------

S1_CASES = ([(m, "iso3d") for m in LM + LOG_ + SF3 + PI] + [(m, "iso_z1") for m in LOG_ + SF2 + PI]
            + [("local_maxima", "iso_z1_sparse")])


@pytest.mark.parametrize("method, name, seed", params(*[(m, n, s) for m, n in S1_CASES for s in SEEDS]))
def test_s1_isolated_spot_recall_and_precision(method, name, seed, record_property):
    _, truth, result = isolated(method, name, seed)
    match, _ = evaluate(result.spots, truth)
    recall, precision = match.values["recall"], match.values["precision"]
    report(record_property, recall=recall, precision=precision, detections=len(result.spots))
    if name != "iso3d":
        assert (result.spots.z == 0).all()
    if (method, name) == ("starfish_log", "iso_z1"):
        assert S1_LOG_Z1_RECALL[0] <= recall <= S1_LOG_Z1_RECALL[1]
        assert precision >= S1_PRECISION
    elif name == "iso_z1_sparse":
        assert recall == S1_LM_SPARSE_RECALL and precision >= S1_LM_SPARSE_PRECISION
    else:
        assert recall == S1_RECALL and precision >= S1_PRECISION


# --- S2: localization -------------------------------------------------------------------------------------

def s2_bounds(method, name):
    """The S2 bounds of a method on a scene: {localization_errors value: upper bound}."""
    kind = family(method)
    if kind == "spotiflow":
        return {"dist_max": S2_SF3_DISTANCE} if name == "iso3d" else {"lateral_max": S2_SF2_LATERAL}
    if kind == "piscis":
        return ({"lateral_max": S2_PI_STACK_LATERAL, "abs_z_max": S2_PI_STACK_ABS_Z} if name == "iso3d"
                else {"lateral_max": S2_PI_PLANE_LATERAL})
    if name == "iso3d":
        return {"dist_max": S2_LM_LOG_3D_DISTANCE}
    return {"lateral_max": S2_LOG_Z1_LATERAL if kind == "starfish_log" else S2_LM_Z1_LATERAL}


@pytest.mark.parametrize("method, name, seed", params(*[(m, n, s) for m, n in S1_CASES for s in SEEDS]))
def test_s2_localization_of_the_s1_matches(method, name, seed, record_property):
    _, truth, result = isolated(method, name, seed)
    match, errors = evaluate(result.spots, truth)
    bounds = s2_bounds(method, name)
    report(record_property, **{k: errors.values[k] for k in ("dist_max", "lateral_max", "abs_z_max")},
           n_abs_z_gt_1=errors.counts["n_abs_z_gt_1"], matched=errors.counts["matched"])
    assert errors.status == "ok" and errors.counts["matched"] == match.counts["matched"] > 0
    for key, bound in bounds.items():
        assert errors.values[key] <= bound, key


# --- S3: resolvable pairs ---------------------------------------------------------------------------------

S3_CASES = [(m, "pairs") for m in LM + LOG_ + SF3 + PI] + [(m, "pairs_z1") for m in SF2 + PI]


@cache
def pair_run(method, case, seed):
    if case == "pairs":
        image, truth, kinds = fixtures.pairs(seed)
    else:
        image, truth = fixtures.pairs_z1(seed)
        kinds = np.array(["lateral"] * len(truth))
    return truth, kinds, detect(image, METHODS[method])


@pytest.mark.parametrize("method, case, seed", params(*[(m, c, s) for m, c in S3_CASES for s in SEEDS]))
def test_s3_every_member_of_a_resolvable_pair_is_matched(method, case, seed, record_property):
    truth, kinds, result = pair_run(method, case, seed)
    match, _ = evaluate(result.spots, truth)
    matched = matched_truth(match)
    recall = {kind: np.mean([i in matched for i in np.flatnonzero(kinds == kind)]) for kind in set(kinds)}
    report(record_property, **{f"{kind}_pair_recall": value for kind, value in recall.items()},
           detections=len(result.spots))
    gated = np.flatnonzero(kinds == "lateral") if family(method) == "piscis" else np.arange(len(truth))
    assert np.mean([i in matched for i in gated]) == S3_PAIR_RECALL


# --- S4: per-channel backgrounds and overrides -----------------------------------------------------------

CHANNEL_LABELS = ("ch00", "ch01", "ch02", "ch03")


@pytest.mark.parametrize("method, seed", params(*[(m, s) for m in LM + LOG_ + SF3 + PI for s in SEEDS],
                                                marks={(m, s): [S4_PISCIS_OFFSETS] for m in PI for s in SEEDS}))
def test_s4_offset_channels_and_a_channel_override(method, seed, record_property):
    image, _ = fixtures.channels(seed)
    config = METHODS[method]
    base = replace(config, channel_labels=CHANNEL_LABELS)
    override = replace(config, **S4_OVERRIDES[family(method)])
    result = detect(image, SpotFindingPlan(base, (ChannelOverride("ch03", override),)))
    single = detect(image[..., 3], override)
    spots, thresholds = result.spots, result.diagnostics["thresholds"]
    rows = {c: spots[spots.channel == c].reset_index(drop=True) for c in range(4)}
    shifts = {c: paired_shift(rows[c], rows[0]) for c in S4_OFFSETS}
    report(record_property, counts=[len(rows[c]) for c in range(4)], shifts=shifts,
           threshold_offsets={c: thresholds[c] - thresholds[0] for c in S4_OFFSETS})
    assert len(rows[0]) > 0
    for c, offset in S4_OFFSETS.items():
        assert len(rows[c]) == len(rows[0])
        if method == "local_maxima":
            pd.testing.assert_frame_equal(rows[c][["z", "y", "x"]], rows[0][["z", "y", "x"]], check_exact=True)
            assert np.array_equal(rows[c].peak_intensity, rows[0].peak_intensity + offset)
            assert abs(thresholds[c] - thresholds[0] - offset) <= S4_LM_THRESHOLD_OFFSET
        else:
            assert shifts[c] <= S4_COORDINATES
    effective = result.diagnostics["effective_settings"]
    assert list(effective) == list(CHANNEL_LABELS)
    assert effective["ch00"] == effective["ch01"] == effective["ch02"]
    assert_settings(effective["ch00"], base)
    assert_settings(effective["ch03"], override)
    assert without_labels(effective["ch03"]) == without_labels(single.diagnostics["effective_settings"]["0"])
    pd.testing.assert_frame_equal(rows[3].drop(columns=["spot_id", "channel"]),
                                  single.spots.drop(columns=["spot_id", "channel"]), check_exact=True)
    assert result.diagnostics["thresholds"][3] == single.diagnostics["thresholds"][0]


# --- S5: coincident cross-channel candidates --------------------------------------------------------------

@pytest.mark.parametrize("method, seed", params(*[(m, s) for m in LM + LOG_ + SF3 + PI for s in SEEDS],
                                                marks={("spotiflow-smfish_3d", 102): [S5_SMFISH_102]}))
def test_s5_coincident_spots_keep_one_row_per_channel(method, seed, record_property):
    image, truth = fixtures.coincident(seed)
    config = METHODS[method]
    result = detect(image, config)
    counts, recalls = [], []
    for c in range(2):
        single = detect(image[..., c], config)
        rows = result.spots[result.spots.channel == c].drop(columns=["spot_id", "channel"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(rows, single.spots.drop(columns=["spot_id", "channel"]), check_exact=True)
        match, _ = evaluate(rows, truth)
        counts.append(len(rows))
        recalls.append(len(matched_truth(match)) / len(truth))
    report(record_property, counts=counts, recalls=recalls)
    assert recalls == [1.0, 1.0]


# --- S6: empty input and zero channels -------------------------------------------------------------------

@pytest.fixture
def spy(monkeypatch):
    """Record the channels each call of a method's private function receives."""
    calls = []

    def install(config_type):
        spec = SPOT_FINDING_METHODS[config_type]

        def run(image, config, context):
            calls.append(context.channels)
            return spec.run(image, config, context)

        monkeypatch.setitem(SPOT_FINDING_METHODS, config_type, replace(spec, run=run))
        return calls

    return install


@pytest.mark.parametrize("method, seed", params(*[(m, s) for m in LM + LOG_ + SF3 + PI for s in SEEDS]))
def test_s6_empty_input_zero_channels_and_the_mad_diagnostics(method, seed, spy, record_property):
    config = METHODS[method]
    calls = spy(type(config))
    for image in (np.zeros((8, 32, 32, 2), dtype=np.uint16), np.full((8, 32, 32), 100, dtype=np.uint16)):
        # Local maxima still records the noise of a constant channel, whose MAD 0 warns (W-270).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SpotFindingWarning)
            result = detect(image, config)
        assert calls == []
        assert set(result.diagnostics["outcomes"].values()) == {"constant"}
        assert_typed_empty(result, config)
    image = fixtures.zero_channels(seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = detect(image, config)
    found = [str(w.message) for w in caught if issubclass(w.category, SpotFindingWarning)]
    noise = result.diagnostics.get("noise")
    report(record_property, warnings=len(found), outcomes=result.diagnostics["outcomes"], noise=noise)
    assert calls == [(0, 1)]
    assert list(result.spots.columns) == ["spot_id", *[c.rstrip("?") for c in
                                                       SPOT_FINDING_METHODS[type(config)].output_columns]]
    if method == "local_maxima":
        assert (noise["0"]["zero_fraction"], noise["0"]["mad"]) == (0.6, 0.0)
        assert noise["1"]["zero_fraction"] == 0.4 and noise["1"]["mad"] > 0
        assert len(found) == 1 and "channel '0'" in found[0]
        assert result.diagnostics["warnings"] == tuple(found)
    else:
        # The noise record and its warning are the local-maxima §2.5 record (docs/spot-finding-contract.md).
        assert noise is None and found == [] and result.diagnostics["warnings"] == ()


# --- S7: borders ------------------------------------------------------------------------------------------

S7_CASES = ([("local_maxima", True, "3d"), ("local_maxima", False, "3d")] + [(m, None, "3d") for m in LOG_ + SF3 + PI]
            + [("local_maxima", True, "z1"), ("local_maxima", False, "z1")] + [(m, None, "z1") for m in LOG_ + SF2 + PI])


@pytest.mark.parametrize("method, exclude_border, dims, seed",
                         params(*[(m, b, d, s) for m, b, d in S7_CASES for s in SEEDS],
                                marks={("spotiflow-synth_3d", None, "3d", s): [S7_SYNTH_3D] for s in SEEDS}))
def test_s7_spots_near_the_faces(method, exclude_border, dims, seed, record_property):
    image, truth, distances = fixtures.borders(seed, z1=dims == "z1")
    config = METHODS[method] if exclude_border is None else replace(METHODS[method], exclude_border=exclude_border)
    result = detect(image, config)
    match, _ = evaluate(result.spots, truth)
    matched = matched_truth(match)
    present = {int(d): int(sum(i in matched for i in np.flatnonzero(distances == d))) for d in set(distances)}
    report(record_property, present_by_distance=present, planted_by_distance=np.bincount(distances).tolist())
    if exclude_border is True:
        assert matched == set(np.flatnonzero(distances != 0).tolist())
    elif exclude_border is False:
        assert matched == set(range(len(truth)))
    else:
        assert set(np.flatnonzero(distances >= S7_GATED_DISTANCE).tolist()) <= matched


# --- S8: tiling seams -------------------------------------------------------------------------------------

def near(spots, point, radius=W266_MATCH["threshold"]):
    coords = spots[["z", "y", "x"]].to_numpy()
    return coords[np.linalg.norm(coords - point, axis=1) <= radius]


S8_CASES = ([(m, c) for m in PI for c in ("seam_z1", "seam3d")] + [(m, "iso_z1") for m in SF2]
            + [(m, "iso3d") for m in SF3])


@pytest.mark.parametrize("method, case, seed", params(*[(m, c, s) for m, c in S8_CASES for s in SEEDS]))
def test_s8_tiling_seams(method, case, seed, record_property):
    config = METHODS[method]
    if family(method) == "piscis":
        image, truth, layout = seam_scene(case, seed)
        tiled = detect(image, replace(config, input_size=PISCIS_SEAM_INPUT_SIZE))
        untiled = detect(image, config)
        assert len(truth) == 18
        assert tiled.diagnostics["geometry"]["keep_boundaries"] == {"y": [30.5, 59.5], "x": [30.5, 59.5]}
        candidates, shifts = [], []
        for point in truth:
            found, reference = near(tiled.spots, point), near(untiled.spots, point)
            candidates.append(len(found))
            if len(found) == len(reference) == 1:
                shifts.append(float(np.linalg.norm(found[0, 1:] - reference[0, 1:])))
        report(record_property, candidates=candidates, max_shift=max(shifts, default=None),
               layout=[f"{kind} {offset:+.1f}" for kind, offset in layout])
        assert candidates == [S8_PI_CANDIDATES] * len(truth)
        assert [len(near(untiled.spots, p)) for p in truth] == [1] * len(truth)
        assert max(shifts) <= S8_PI_SHIFT
    else:
        image, _, untiled = isolated(method, case, seed)
        tiled = detect(image, replace(config, n_tiles=S8_SF_TILES[case]))
        shift = paired_shift(tiled.spots, untiled.spots)
        report(record_property, detections=[len(tiled.spots), len(untiled.spots)], max_shift=shift)
        assert tiled.diagnostics["geometry"]["n_tiles"] == S8_SF_TILES[case]
        assert tiled.diagnostics["effective_settings"]["0"]["n_tiles"] == S8_SF_TILES[case]
        assert shift <= S8_SF_SHIFT


# --- S9: explicit scaling ---------------------------------------------------------------------------------

S9_CASES = [("spotiflow", "construction"), ("piscis", "construction"), ("starfish_log", "anisotropic")] + [
    (m, "effective") for m in SF3 + SF2 + PI]
SCENE_OF = {**{m: "iso3d" for m in SF3}, **{m: "iso_z1" for m in SF2}}


@pytest.mark.parametrize("method, case", params(*S9_CASES))
def test_s9_explicit_scaling(method, case):
    if case == "construction":
        config = SpotiflowConfig("smfish_3d") if method == "spotiflow" else PiscisConfig("20251212")
        for scale in (2, 0.5, 2.0):
            with pytest.raises(ValueError, match="scale must be 1"):
                replace(config, scale=scale)
        assert replace(config, scale=1).scale == 1
    elif case == "effective":
        for name in ([SCENE_OF[method]] if method in SCENE_OF else ["iso3d", "iso_z1"]):
            (effective,) = isolated(method, name, SEEDS[0])[2].diagnostics["effective_settings"].values()
            assert effective["scale"] == 1.0
    else:
        # LoG per-axis sigma: the anisotropic W-266 parity case, both channels, exactly.
        settings_ = PARITY.CASES["volume-2ch-anisotropic"][1]
        assert (tuple(settings_["min_sigma"]), tuple(settings_["max_sigma"])) == ((2.0, 1.0, 1.0), (6.0, 3.0, 3.0))
        for key in PARITY_RECORD["cases"]["volume-2ch-anisotropic"]["tables"]:
            assert_parity("volume-2ch-anisotropic", key)


# --- S10: multi-round identities -------------------------------------------------------------------------

@pytest.mark.parametrize("method, seed", params(*[(m, s) for m in LM + LOG_ + SF3 + PI for s in SEEDS],
                                                marks={(m, s): [S10_PISCIS_COLUMNS] for m in PI for s in (100, 101)}))
def test_s10_multi_round_identities(method, seed, record_property):
    shared, fov, default = multiround_run(method, seed)
    result = fov.spot_result
    spots = result.spots
    report(record_property, rows_per_round={r: int((spots["round"] == r).sum()) for r in ROUNDS})
    assert isinstance(spots["round"].dtype, pd.StringDtype) and not spots["round"].isna().any()
    assert list(dict.fromkeys(spots["round"])) == list(ROUNDS)
    assert spots.spot_id.tolist() == [str(i) for i in range(len(spots))]
    frame = candidates_frame(result)
    joined = spots.assign(spot_namespace=result.spot_namespace).merge(
        frame, on=["spot_namespace", "spot_id"], validate="one_to_one", suffixes=("", "_frame"))
    assert len(joined) == len(spots) == len(frame)
    for c in range(len(ROUND_CHANNELS)):
        rows = spots[spots.channel == c]
        zyx = rows[["z", "y", "x"]].to_numpy()
        for centre in shared[c]:
            found = rows[np.linalg.norm(zyx - centre, axis=1) <= S10_RADIUS]
            assert sorted(found["round"]) == sorted(ROUNDS), (c, centre)
    reference = spots[spots["round"] == ROUNDS[0]].drop(columns="round").reset_index(drop=True)
    pd.testing.assert_frame_equal(reference, default.spots, check_exact=True)
    with pytest.raises(ValueError, match="§2.8"):
        fov.decode_barcodes()


# --- S11: checkpoint round trip ---------------------------------------------------------------------------

S11_CASES = ([("multiround", m, s, f) for m in LM + LOG_ + SF3 + PI for s in SEEDS for f in ("csv", "parquet")]
             + [("iso3d", m, s, f) for m in LOG_ + SF3 + PI for s in SEEDS for f in ("csv", "parquet")])


@pytest.mark.parametrize("method, source, seed, table_format", params(*[(m, src, s, f) for src, m, s, f in S11_CASES]))
def test_s11_candidates_checkpoint_round_trip(method, source, seed, table_format, tmp_path):
    if source == "multiround":
        result = multiround_run(method, seed)[1].spot_result
    else:
        result = isolated(method, "iso3d", seed)[2]
    assert len(result.spots)
    header, reloaded = _round_trip(result, tmp_path, table_format)
    assert header["format_version"] == 2 and all(key in header for key in PLAN_KEYS)
    assert header["detection_rounds"] == (list(ROUNDS) if source == "multiround" else None)
    if family(method) in LEARNED:
        assert header["weights"] == json.loads(json.dumps(result.diagnostics["model"]["artifacts"]))
    else:
        assert header["weights"] == []
    assert_reloaded(result, reloaded)


def test_s11_a_version_2_checkpoint_from_before_the_plan_keys_loads_unchanged(tmp_path):
    """The golden test helper's checkpoint (global, exclude_border true, Z=1); its table and CSV digests equal
    the 42f652d pins of test_spot_finding_golden.py (data/spot_finding_candidates_7a178da/README.md)."""
    header = json.loads((VERSION_2_CHECKPOINT / "FOV_001" / "candidates.json").read_text())
    assert header["format_version"] == 2 and not any(key in header for key in PLAN_KEYS)
    fov = golden_dataset(tmp_path).fov("FOV_001").load_checkpoint(
        "candidates", checkpoints=CheckpointConfig(directory=VERSION_2_CHECKPOINT))
    result = fov.spot_result
    assert table_digest(result.spots) == PINNED_FIND_SPOTS[("global", True, "z1")][1]
    assert result.plan == SpotFindingPlan(result.config) and result.plan.rounds is None
    assert result.config == LocalMaximaConfig(threshold_mode="global", threshold_value=0.01,
                                              channel_labels=("ch00", "ch01", "ch02", "ch03"))


# --- S12: Starfish parity ---------------------------------------------------------------------------------

def assert_parity(case, key):
    name, settings_ = PARITY.CASES[case]
    image = parity_image(name)
    result = detect(image, StarfishLogConfig(**settings_))
    channel = int(key.split("_c")[1])
    rows = result.spots[result.spots.channel == channel].drop(columns="spot_id")
    pd.testing.assert_frame_equal(starfish_view(rows, image[..., channel]),
                                  starfish_table(PARITY_RECORD["cases"][case]["tables"][key]), check_exact=True)


@pytest.mark.parametrize("case, key", [*PARITY_TABLES, ("plane-anisotropic-3tuple", None)])
def test_s12_starfish_parity(case, key):
    if key is None:
        name, settings_ = PARITY.ERROR_CASES[case]
        with pytest.raises(IncompatibleGeometryError, match="Z=1"):
            detect(parity_image(name), StarfishLogConfig(**settings_))
    else:
        assert_parity(case, key)


# --- S13: determinism -------------------------------------------------------------------------------------

S13_CASES = ([(m, n) for m in LM + LOG_ + PI for n in ("iso3d", "iso_z1")] + [(m, "iso3d") for m in SF3]
             + [(m, "iso_z1") for m in SF2])


def process_group(method):
    """The second-process group of an S13 case: LM and LoG, Spotiflow, or one Piscis model (the extended checks
    run each Piscis model separately, so a group never runs another group's detections)."""
    return method if family(method) == "piscis" else family(method) if family(method) in LEARNED else "default"


@cache
def second_process_digests(group):
    """Table digests of every S13 case of one process_group, seed 100, from a second Python process at one
    thread."""
    cases = [(m, n) for m, n in S13_CASES if process_group(m) == group]
    code = ("import json\n"
            "from test.test_spot_finding_validation import METHODS, detect, scene\n"
            "from test.learned_detectors import table_digest\n"
            f"cases = {cases!r}\n"
            "print(json.dumps({f'{m}/{n}': table_digest(detect(scene(n, 100)[0], METHODS[m]).spots)"
            " for m, n in cases}))\n")
    return json.loads(run_python(code))


@pytest.mark.parametrize("method, name", params(*S13_CASES))
def test_s13_tables_are_identical_twice_in_one_process_and_in_a_second(method, name, record_property):
    image, _, first = isolated(method, name, SEEDS[0])
    second = detect(image, METHODS[method])
    other = second_process_digests(process_group(method))[f"{method}/{name}"]
    report(record_property, digests=[table_digest(first.spots), table_digest(second.spots), other],
           rows=len(first.spots))
    assert table_digest(first.spots) == table_digest(second.spots) == other


# --- S14: dependency and weights errors -------------------------------------------------------------------

S14_CASES = [("spotiflow", "missing-library"), ("spotiflow", "missing-torch"), ("piscis", "missing-library"),
             ("piscis", "missing-torch"), ("weights", "missing"), ("weights", "hash-mismatch"),
             ("weights", "unknown-model"), ("spotiflow-general", "no-network"), ("piscis-20251212", "no-network")]
S14_IMAGES = {"spotiflow": (SpotiflowConfig("smfish_3d"), (8, 16, 16)), "piscis": (PiscisConfig("20251212"), (2, 16, 16))}
PAYLOAD = b"starfinder fixture weights\n"


@pytest.mark.parametrize("method, case", params(*S14_CASES))
def test_s14_dependency_and_weights_errors(method, case, tmp_path, monkeypatch):
    if case != "no-network":
        # An empty cache: nothing the operator fetched is read or changed.
        monkeypatch.setenv("STARFINDER_WEIGHTS_DIR", str(tmp_path / "weights"))
    if case.startswith("missing-"):
        config, shape = S14_IMAGES[method]
        module = method if case == "missing-library" else "torch"
        if module == "torch":
            # Present library, missing torch: an empty stand-in for the library, which require() finds.
            monkeypatch.setitem(sys.modules, method, types.ModuleType(method))
        monkeypatch.setitem(sys.modules, module, None)
        with pytest.raises(SpotFindingBackendUnavailableError) as raised:
            find_spots(np.ones(shape, np.uint16), config=config, metadata=META, spot_namespace=NAMESPACE)
        assert str(raised.value) == (f"spot-finding method '{method}' requires {module}; install the '{method}' "
                                     f"extra (starfinder[{method}])")
    elif case == "missing":
        for (name, model), entry in KNOWN_WEIGHTS.items():
            with pytest.raises(MissingWeightsError) as raised:
                resolve_weights(name, model)
            message = str(raised.value)
            assert str(tmp_path / "weights" / name / model / entry.files[0].path) in message
            assert f"'starfinder weights fetch {name} {model}'" in message
    elif case == "hash-mismatch":
        digest = hashlib.sha256(PAYLOAD).hexdigest()
        entry = replace(KNOWN_WEIGHTS[("piscis", "20251212")], model="fixture", url="file:///unused",
                        revision="fixture", sha256=digest, bytes=len(PAYLOAD),
                        files=(WeightsFile("fixture.pt", digest, len(PAYLOAD)),))
        monkeypatch.setitem(KNOWN_WEIGHTS, ("piscis", "fixture"), entry)
        path = tmp_path / "weights" / "piscis" / "fixture" / "fixture.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(PAYLOAD)
        assert resolve_weights("piscis", "fixture") == path.parent
        changed = bytes([PAYLOAD[0] ^ 1]) + PAYLOAD[1:]
        path.write_bytes(changed)
        actual = hashlib.sha256(changed).hexdigest()
        with pytest.raises(WeightsHashMismatchError) as raised:
            resolve_weights("piscis", "fixture")
        assert f"{path} has SHA-256 {actual}, expected {digest}" in str(raised.value)
    elif case == "unknown-model":
        with pytest.raises(ValueError, match=r"known models: \['20230905', '20251212'\]"):
            resolve_weights("piscis", "latest")
        with pytest.raises(ValueError, match=r"known models: \['general', 'hybiss', 'smfish_3d', 'synth_3d'\]"):
            SpotiflowConfig("latest")
    else:
        # The detection builds its model from the operator's cache with the network patched to raise.
        monkeypatch.setattr(_learned, "_MODELS", {})

        def refuse(*args, **kwargs):
            raise AssertionError("a detection tried to use the network")
        monkeypatch.setattr(socket, "socket", refuse)
        monkeypatch.setattr(urllib.request, "urlopen", refuse)
        image, truth = isolated_scene("iso_z1", SEEDS[0])
        result = detect(image, METHODS[method])
        assert len(result.spots) > 0 and result.diagnostics["model"]["model"] == METHODS[method].model



# --- S15: dimensionality rules ----------------------------------------------------------------------------

S15_SHAPES = [(1, 32, 32), (2, 32, 32), (6, 32, 32), (7, 8, 8), (1, 5, 5), (1, 8, 8)]
LOG_ANISOTROPIC = "starfish_log-anisotropic"


def s15_expected(method, shape):
    """'ok' or 'raise' (IncompatibleGeometryError) for one method and shape, from W-266's minimum-shape and
    dimensionality probes and its parity error case."""
    z, y, x = shape
    if method == LOG_ANISOTROPIC:
        return "raise" if z == 1 else "ok"
    if method in SF3:
        return "ok" if z >= 7 and y >= 8 and x >= 8 else "raise"
    if method in SF2:
        return "ok" if z == 1 and y >= 6 and x >= 6 else "raise"
    return "ok"


@pytest.mark.parametrize("method, shape", params(*[(m, s) for m in LM + LOG_ + SF3 + SF2 + PI + [LOG_ANISOTROPIC]
                                                   for s in S15_SHAPES]))
def test_s15_dimensionality_rules(method, shape, record_property):
    config = (StarfishLogConfig(**PARITY.LOG_SETTINGS_ANISOTROPIC) if method == LOG_ANISOTROPIC
              else METHODS[method])
    image = fixtures.probe(shape)
    if s15_expected(method, shape) == "raise":
        with pytest.raises(IncompatibleGeometryError):
            detect(image, config)
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SpotFindingWarning)
        result = detect(image, config)
    report(record_property, rows=len(result.spots), z=sorted(set(result.spots.z.tolist())))
    assert list(result.spots.columns)[:4] == ["spot_id", "z", "y", "x"]
    if shape[0] == 1:
        assert (result.spots.z == 0).all()


# --- S16: the W-218 resolution on the §2.12 formed-amplicon scene -----------------------------------------

@pytest.mark.parametrize("part, variant, seed", [pytest.param(p, v, s, id=f"{p}-{v}-{s}") for p, v in (
    ("merge", "uint8"), ("border", "uint16"), ("border", "uint8")) for s in SEEDS])
def test_s16_the_w218_option_and_exclude_border_on_formed16(part, variant, seed, record_property):
    image, metadata, truth, eligible = formed16(seed)
    if variant == "uint8":
        image = normalize_intensity(image, config=RECIPE1)
    if part == "merge":
        _, legacy, _ = s16(image, metadata, truth, eligible, LocalMaximaConfig())
        result, merged, _ = s16(image, metadata, truth, eligible, LocalMaximaConfig(merge_radius_zyx=RADIUS))
        report(record_property, legacy=legacy, merged=merged, removed=result.diagnostics["merged"])
        assert merged["duplicate_same_group"] == S16_SAME_CHANNEL_DUPLICATES
        assert merged["matched"] >= legacy["matched"]
    else:
        shape = np.asarray(FORMED16_SHAPE)
        near_face = set(np.flatnonzero(eligible & (np.minimum(truth, shape - 1 - truth).min(axis=1)
                                                   <= S16_FACE_DISTANCE)).tolist())
        _, excluded, matched_excluded = s16(image, metadata, truth, eligible, LocalMaximaConfig())
        _, kept, matched = s16(image, metadata, truth, eligible, LocalMaximaConfig(exclude_border=False))
        report(record_property, near_face=len(near_face), missed_with_exclusion=len(near_face - matched_excluded),
               missed_without_exclusion=len(near_face - matched), eligible=int(eligible.sum()),
               matched=[excluded["matched"], kept["matched"]],
               cross_channel_duplicates=[excluded.get("duplicate_other_group"), kept.get("duplicate_other_group")])
        assert near_face and near_face <= matched
