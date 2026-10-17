"""Generator checks of the appearance presets and the real banks (docs/synthetic-real-appearance-validation.md).

GA8 (labels against the placements), GA9 (ownership and anchors), GA10 (placement and its stop), GA11 (territories) and
GA18 (projection truth) on ``app_routine``, the three routine appearance presets on their closed-form banks (DN3), as
generated; recomputed by the test oracle (``cellular_appearance_oracle.py``) with the helpers of the layout's checks
(``test_synthetic_appearance_scene.py``). GA24 (the cost of each preset and of each bank build, each in its own process
under /usr/bin/time -v), GA26 (the real banks against their records, W-392's inventory and W-513's counts) and GA27 (a
rebuild gives the recorded digest) on ``real_banks``, ``app_slow`` and ``app_slow_acquired``. The routine checks read no
real data. A real-bank check is ``slow``, reads the banks through STARFINDER_APPEARANCE_BANKS and skips with a reason
naming the variable or the missing file. Reported values are printed.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from starfinder.synthetic import APPEARANCE_PRESETS, BankReference, load_appearance_bank

from . import cellular_appearance_oracle as oracle
from .cellular_appearance_scenes import (ACQUIRED_SPACING, RG1_RSS_KIB, RG1_SECONDS, ROUTINE_PRESETS, ROUTINE_RSS_KIB,
                                         SLOW_BANKS, SLOW_PRESETS, app_routine, generate_measured, real_bank,
                                         rebuild_measured)
from .test_synthetic_appearance_scene import grids, lattice, pieces, redrawn_growth, replay_centre

pytestmark = [pytest.mark.synthetic, pytest.mark.validation]

slow = pytest.mark.slow
BANKS = tuple(SLOW_BANKS.values())
# W-392's inventory of the real inputs, with the SHA-256 of each file (GA26).
W392_INVENTORY = Path("/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-392/20261009T0323Z-inputs/"
                      "inventory")
# SRC2's held-out image of each bank, never a source.
HELD_OUT = {"ln-sample1-v1": "Position048", "tissue2d-v1": "crop_y18432_x0", "culture3d-v1": "Position397"}
# SEL1's counts per source image, W-513 tables/bank_selection_<context>.csv (docs/synthetic-real-appearance.md,
# "Selection of entries"): labels, left out by rule 1 (by the Z end alone, in LN where that rule applies), by rule 2,
# by rule 3, by rule 4, kept; and the plateau.
W513 = {
    "ln-sample1-v1": {"Position049": (2475, 1018, 993, 146, 0, 110, 1201, 45.0),
                      "Position057": (2405, 856, 838, 156, 0, 102, 1291, 39.0),
                      "Position058": (1207, 565, 558, 66, 0, 35, 541, 50.0)},
    "tissue2d-v1": {"crop_y20480_x2048": (291, 23, 0, 28, 0, 69, 171, 80.0),
                    "crop_y20480_x4096": (242, 34, 0, 22, 0, 53, 133, 92.5)},
    "culture3d-v1": {"Position398": (54, 23, 0, 4, 0, 0, 27, 48.0), "Position407": (56, 17, 0, 4, 0, 0, 35, 46.0),
                     "Position408": (57, 22, 0, 4, 2, 0, 29, 45.0)},
}
COUNTS = ("labels", "left_out_border", "left_out_border_z_alone", "left_out_size_after_border",
          "left_out_nucleus_after_size", "left_out_squashed_after_rest", "kept")
ENTRIES = {"ln-sample1-v1": 3033, "tissue2d-v1": 304, "culture3d-v1": 91}
# O8's g, the median plateau of each bank's source images.
PLATEAU = {"ln-sample1-v1": 45.0, "tissue2d-v1": 86.25, "culture3d-v1": 46.0}
# The objects of each slow preset at the prototype's density (W-513 tables/resource_estimates.csv; estimates), against
# which the bank size is reported (BS2).
ESTIMATED_OBJECTS = {"ln-sample1-v1": 738, "tissue2d-v1": 337, "culture3d-v1": 32}
GUARD = 1.5


def routine(name):
    _, config, scene, bank, placed = app_routine(name)
    # The generation draws the placement's streams: its placements and labels are the placement's.
    pd.testing.assert_frame_equal(scene.placements, placed.placements)
    sample = grids(placed)[0][3]
    for target in ("nucleus", "cell"):
        assert np.array_equal(scene.context.sample.sample_labels[target], sample[target])
    fov = {g[0]: g[3] for g in grids(placed)[1:]}
    for f, labels in scene.context.sample.fov_labels.items():
        for target in ("nucleus", "cell"):
            assert np.array_equal(labels[target], fov[f][target]), (f, target)
    return config, scene, bank, placed


# ------------------------------------------------------------------------------------------------ GA8 to GA11, GA18
# Each of these checks, run alone, generates its routine preset (5 to 11 s with its closed-form bank on one CPU),
# so each is `slow` (docs/contributing.md, "Test markers").


@slow
@pytest.mark.parametrize("name", ROUTINE_PRESETS)
def test_ga8_routine_labels_equal_the_oracle_on_every_grid(name):
    config, _, bank, placed = routine(name)
    objects = oracle.frames(bank, placed.placements)
    growth = redrawn_growth(placed, config.context)
    boundary_total = 0
    for grid_name, origin, shape, generated in grids(placed):
        points = oracle.grid_points(origin, placed.layout.s, shape)
        N, C, boundary, _ = oracle.labels(points.reshape(-1, 3), objects, oracle.kind_of(bank), growth)
        N, C, boundary = (a.reshape(tuple(shape)) for a in (N, C, boundary))
        differ = (generated["nucleus"] != N) | (generated["cell"] != C)
        assert not (differ & ~boundary).any(), f"{name} {grid_name}: {int((differ & ~boundary).sum())} voxels differ"
        point_C, point_N = placed.labels.labels(points)
        assert np.array_equal(point_N, generated["nucleus"]) and np.array_equal(point_C, generated["cell"])
        boundary_total += int(boundary.sum())
    sample = grids(placed)[0][3]["nucleus"]
    counts = [pieces(sample, k) for k in placed.placements.object_id]
    print(f"GA8 {name}: {len(grids(placed))} grids, boundary voxels {boundary_total}; pieces per label on the sample "
          f"grid {counts}")
    assert boundary_total == 0


@slow
@pytest.mark.parametrize("name", ROUTINE_PRESETS)
def test_ga9_routine_ownership_overlap_and_anchors(name):
    config, _, bank, placed = routine(name)
    table = placed.placements
    objects = oracle.frames(bank, table)
    origin, spacing, shape, plane, voxel = lattice(placed)
    canvas = np.zeros(shape, dtype=bool)
    for o, row in zip(objects, table.itertuples()):
        lo, mask, _, _ = oracle.lattice_mask(o, origin, spacing, plane)
        voxels = int(mask.sum())
        window = oracle.on_grid(lo, mask, shape)
        overlap = 0 if window is None else int((canvas[window[0]] & mask[window[1]]).sum())
        assert 100 * overlap <= voxels, f"object {o.k}: {overlap} of {voxels} voxels on earlier masks"
        assert row.volume == pytest.approx(voxels * voxel, abs=1e-9)
        assert row.overlap_share == pytest.approx(overlap / voxels, abs=1e-9)
        if window is not None:
            canvas[window[0]] |= mask[window[1]]
        anchor = np.array([row.anchor_z, row.anchor_y, row.anchor_x])
        assert np.abs(anchor - o.deepest()).max() <= 1e-9
    anchors = table[["anchor_z", "anchor_y", "anchor_x"]].to_numpy()
    N, C, _, _ = oracle.labels(anchors, objects, oracle.kind_of(bank), redrawn_growth(placed, config.context))
    assert np.array_equal(C if placed.kind == "cell" else N, table.object_id.to_numpy())
    for _, _, _, generated in grids(placed):
        nucleus, cell = generated["nucleus"], generated["cell"]
        assert np.all(cell[nucleus > 0] == nucleus[nucleus > 0])


@slow
@pytest.mark.parametrize("name", ROUTINE_PRESETS)
def test_ga10_routine_placement_and_its_stop(name):
    config, scene, bank, placed = routine(name)
    layout = config.layout
    table = placed.placements
    objects = oracle.frames(bank, table)
    origin, spacing, shape, plane, _ = lattice(placed)
    canvas = np.zeros(shape, dtype=bool)
    nuclei = np.zeros(shape, dtype=bool)
    rows = list(table.itertuples())
    for i, (o, row) in enumerate(zip(objects, rows)):
        centre, _ = replay_centre(placed, bank, row, o, rows[:i], canvas, plane, layout.contact)
        recorded = np.array([row.centre_z, row.centre_y, row.centre_x])
        assert np.abs(recorded - centre).max() <= 1e-9, f"object {o.k}: {recorded} against {centre}"
        lo, mask, nucleus, points = oracle.lattice_mask(o, origin, spacing, plane)
        assert 100 * int(oracle.inside(points[mask], placed.layout.omega).sum()) >= 35 * int(mask.sum())
        window = oracle.on_grid(lo, mask, shape)
        if nucleus is not None and window is not None:
            assert not (nuclei[window[0]] & nucleus[window[1]]).any(), f"cell {o.k}: its nucleus touches an earlier one"
            nuclei[window[0]] |= nucleus[window[1]]
        if window is not None:
            canvas[window[0]] |= mask[window[1]]
    # DN3 placed its count; the record equals a recount, in the placement and in appearance.json.
    density = placed.record["density"]
    proposals = placed.proposals
    assert density["stopped_by"] == "count" and density["placed"] == density["reached"] == len(table) == layout.count
    assert layout.count == APPEARANCE_PRESETS[name]["count"]
    assert density["proposals"] == len(proposals) == int(table.proposal.max())
    assert density["refused"] == int((proposals.outcome == "refused").sum())
    assert scene.record["placed"] == len(table) and scene.record["proposals"] == len(proposals)
    assert scene.record["density"]["stopped_by"] == "count" and scene.record["density"]["reached"] == layout.count
    print(f"GA10 {name}: {len(table)} objects in {density['proposals']} proposals ({density['refusals']} refused); "
          f"bank of {len(bank)} entries, cycles {placed.record['bank']['cycles']}")


@slow
@pytest.mark.parametrize("name", ROUTINE_PRESETS)
def test_ga11_routine_territories(name):
    config, _, bank, placed = routine(name)
    table, cells = placed.placements, placed.cells
    assert list(cells.cell_id) == list(table.object_id) == list(cells.nucleus_id)
    assert set(cells.kind) == {"ordinary"} and set(cells.n_nuclei) == {1}
    growth = redrawn_growth(placed, config.context)
    objects = oracle.frames(bank, table)
    ids = table.object_id.to_numpy()
    ties = 0
    for _, origin, shape, generated in grids(placed):
        flat = oracle.grid_points(origin, placed.layout.s, shape).reshape(-1, 3)
        N, C = generated["nucleus"].ravel(), generated["cell"].ravel()
        if placed.kind == "cell":
            # CU1: the first placed cell whose mask holds the voxel.
            first = np.zeros(len(flat), dtype=np.int64)
            for k, o in zip(ids, objects):
                first[(first == 0) & (o.d(flat) <= 0)] = k
            assert np.array_equal(C, first)
            continue
        assert np.abs(cells.growth.to_numpy() - growth).max() <= 1e-9
        # CT1 by the oracle's distances δ_k = m_k · D_e(T_k⁻¹ σ), one object at a time.
        free = N == 0
        best = np.full(int(free.sum()), np.inf)
        second = np.full(int(free.sum()), np.inf)
        nearest = np.zeros(int(free.sum()), dtype=np.int64)
        for i, o in enumerate(objects):
            d = o.m * o.d(flat[free])
            better = d < best
            second = np.where(better, best, np.minimum(second, d))
            best = np.where(better, d, best)
            nearest[better] = i
        cell = C[free]
        territory = cell > 0
        assert np.array_equal(cell[territory], ids[nearest[territory]])
        assert np.all(best[territory] <= growth[nearest[territory]]) and np.all(best[territory] > 0)
        assert np.all(~np.isfinite(best[~territory]) | (best[~territory] > growth[nearest[~territory]]))
        finite = np.isfinite(best) & np.isfinite(second)
        ties += int((second[finite] - best[finite] <= oracle.BOUNDARY).sum())
    print(f"GA11 {name}: " + (f"CT1 ties within 1e-9 µm {ties}" if placed.kind != "cell" else "CU1"))
    assert ties == 0


@slow
@pytest.mark.parametrize("name", ["appearance-tissue2d-v1", "appearance-culture-v1"])
def test_ga18_routine_projection_truth(name):
    _, scene, _, placed = routine(name)
    grid = placed.layout
    checked = list(grids(placed))
    offset = grid.origin + np.array([0.37, 0.21, 0.43]) * grid.s
    C, N = placed.labels.labels(oracle.grid_points(offset, grid.s, grid.n))
    checked.append(("offset", offset, grid.n, dict(nucleus=N, cell=C)))
    counts = {}
    for grid_name, _, _, generated in checked:
        N = generated["nucleus"]
        ids = np.sort(N, axis=0)
        counts[grid_name] = int(((ids[1:] != ids[:-1]) & (ids[:-1] > 0)).any(axis=0).sum())
        if name == "appearance-tissue2d-v1":
            # TZ1: each entry a column, so no column holds two IDs and every plane equals the Z maximum P.
            assert counts[grid_name] == 0
            assert np.array_equal(N, np.broadcast_to(N.max(axis=0, keepdims=True), N.shape))
    if name == "appearance-tissue2d-v1":
        # The composed stain is a column too: B (a baseline), κH and S are the same in every plane.
        for fov, stains in scene.context.stains.items():
            image = stains["PI"]
            assert np.array_equal(image, np.broadcast_to(image[image.shape[0] // 2], image.shape)), fov
    print(f"GA18 {name}: columns holding two nucleus IDs per grid {counts}")


# ------------------------------------------------------------------------------------------------ GA24


@pytest.fixture(scope="session")
def app_slow(tmp_path_factory):
    """``app_slow`` (and with ``acquired`` ``app_slow_acquired``): a slow preset generated once per session in its own
    process under /usr/bin/time -v, its scene saved under the session's temporary directory."""
    measured = {}

    def get(name, acquired=False):
        if (name, acquired) not in measured:
            real_bank(SLOW_BANKS[name])
            directory = tmp_path_factory.mktemp(f"{name}-acquired" if acquired else name)
            measured[name, acquired] = generate_measured(name, directory, ACQUIRED_SPACING[name] if acquired else None)
        return measured[name, acquired]
    return get


@pytest.fixture(scope="session")
def rebuilt(tmp_path_factory):
    """GA27's rebuild of each real bank, once per session, in its own process under /usr/bin/time -v."""
    measured = {}

    def get(name):
        if name not in measured:
            measured[name] = (real_bank(name), rebuild_measured(real_bank(name), tmp_path_factory.mktemp(name)))
        return measured[name]
    return get


def within_rg1(result, what):
    assert result.exit == 0, f"{what}: exit {result.exit}\n{result.output}"
    assert result.wall_s <= RG1_SECONDS and result.peak_rss_kib <= RG1_RSS_KIB, (
        f"{what}: {result.wall_s} s and {result.peak_rss_kib} KiB against RG1")


@slow
@pytest.mark.parametrize("name", ROUTINE_PRESETS)
def test_ga24_routine_preset_generation_cost(name, tmp_path):
    result = generate_measured(name, tmp_path)
    assert result.exit == 0, result.output
    assert result.peak_rss_kib <= ROUTINE_RSS_KIB
    assert result.summary["placed"] == APPEARANCE_PRESETS[name]["count"]
    print(f"GA24 {name}: process {result.wall_s:.2f} s (generation {result.summary['generation_s']:.2f} s), peak RSS "
          f"{result.peak_rss_kib} KiB")


def report_slow(name, result, label):
    summary = result.summary
    density = summary["density"]
    assert summary["bank"]["sha256"] == APPEARANCE_PRESETS[name]["bank_sha256"]
    assert density["mode"] == "visible" and density["target"] == APPEARANCE_PRESETS[name]["target"]
    assert density["stopped_by"] == "target" or density["reached"] >= density["floor"] * density["target"]
    print(f"GA24 {label}: process {result.wall_s:.2f} s (generation {summary['generation_s']:.2f} s), peak RSS "
          f"{result.peak_rss_kib} KiB; {summary['placed']} objects in {summary['proposals']} proposals, visible share "
          f"{density['reached']:.4f} against {density['target']:.4f} ({density['stopped_by']}); bank cycles "
          f"{summary['bank']['cycles']}")


@slow
@pytest.mark.parametrize("name", SLOW_PRESETS)
def test_ga24_slow_preset_generation_within_rg1(name, app_slow):
    result = app_slow(name)
    within_rg1(result, name)
    report_slow(name, result, name)


@slow
@pytest.mark.parametrize("name", list(ACQUIRED_SPACING))
def test_ga24_acquired_pixel_variant_within_rg1(name, app_slow):
    result = app_slow(name, acquired=True)
    within_rg1(result, f"{name} at the acquired pixel")
    assert result.summary["spacing_zyx"] == pytest.approx(list(ACQUIRED_SPACING[name]), abs=1e-12)
    report_slow(name, result, f"{name} at {ACQUIRED_SPACING[name][1]} µm")


@slow
@pytest.mark.parametrize("name", BANKS)
def test_ga24_bank_build_within_rg1(name, rebuilt):
    _, result = rebuilt(name)
    within_rg1(result, f"build of {name}")
    print(f"GA24 build {name}: process {result.wall_s:.2f} s (build {result.summary['build_s']:.2f} s), peak RSS "
          f"{result.peak_rss_kib} KiB, file {result.summary['file']['size']} bytes")


# ------------------------------------------------------------------------------------------------ GA26 and GA27


def inventory():
    """W-392's inventory: path -> SHA-256; a skip naming the file where it is missing."""
    out = {}
    for context in ("ln", "culture", "tissue2d"):
        path = W392_INVENTORY / f"{context}.json"
        if not path.is_file():
            pytest.skip(f"no file {path}: GA26 compares the build records' inputs with W-392's inventory")
        out.update({f["path"]: f["sha256"] for f in json.loads(path.read_text(encoding="utf-8"))["files"]})
    return out


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def crops_meet(a, b):
    if a is None or b is None:
        return True
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


@slow
@pytest.mark.parametrize("name", BANKS)
def test_ga26_real_banks(name):
    bank = real_bank(name)
    record = bank.record
    # The file is the one its preset names, and its arrays give the recorded digest.
    assert sha256(bank.path) == bank.sha256 == record["file"]["sha256"]
    loaded = load_appearance_bank(BankReference(name=name, sha256=bank.sha256))
    assert loaded.digest == record["digest"] and len(loaded) == ENTRIES[name] == record["counts"]["entries"]
    # Every input is inventoried by W-392 with an equal SHA-256; the configuration is the recorded one.
    known = inventory()
    files = [*record["inputs"], *(i for h in record["held_out"] for i in h["inputs"])]
    assert files and all(f["path"] in known and known[f["path"]] == f["sha256"] for f in files), [
        f["path"] for f in files if known.get(f["path"]) != f["sha256"]]
    assert [s["image"] for s in bank.config["sources"]] == [s["image"] for s in record["sources"]]
    # The held-out image is never a source: not among the images, and no source reads its pixels.
    assert [h["image"] for h in record["held_out"]] == [HELD_OUT[name]]
    images = [i["image"] for i in record["images"]]
    assert HELD_OUT[name] not in images and images == list(W513[name])
    for held in record["held_out"]:
        for f in held["inputs"]:
            for source in record["inputs"]:
                if source["sha256"] == f["sha256"]:
                    assert not crops_meet(held.get("crop_yx"), source["crop_yx"]), (held["image"], source["image"])
    # SEL1's counts per image equal W-513's, and the plateaus; BR4's guard on every source.
    for image in record["images"]:
        *counts, plateau = W513[name][image["image"]]
        assert tuple(image[k] for k in COUNTS) == tuple(counts), image["image"]
        assert image["plateau"] == pytest.approx(plateau, abs=1e-9)
        assert 1 / GUARD <= image["plateau_over_median"] <= GUARD
        assert 1 / GUARD <= image["visible_over_plateau"] <= GUARD
    assert record["plateau_median"] == pytest.approx(PLATEAU[name], abs=1e-9)
    assert record["counts"]["kept"] == sum(v[6] for v in W513[name].values()) == ENTRIES[name]
    estimate = ESTIMATED_OBJECTS[name]
    print(f"GA26 {name}: {len(loaded)} entries from {images} (held out {HELD_OUT[name]}), {record['counts']['patches']} "
          f"patches, g {record['plateau_median']}; plateau over the median "
          f"{[round(i['plateau_over_median'], 4) for i in record['images']]}, visible median over the plateau "
          f"{[round(i['visible_over_plateau'], 4) for i in record['images']]}; bank size against about {estimate} "
          f"placements (estimate): {len(loaded) / estimate:.2f} times, reuse without replacement "
          f"{max(0, estimate - len(loaded)) / estimate:.3f}")


@slow
@pytest.mark.parametrize("name", BANKS)
def test_ga27_a_rebuild_gives_the_recorded_digest(name, rebuilt):
    bank, result = rebuilt(name)
    assert result.exit == 0, result.output
    assert result.summary["digest"] == bank.record["digest"]
    assert result.summary["counts"] == bank.record["counts"]
    print(f"GA27 {name}: rebuilt digest {result.summary['digest']} equals the record's; the rebuilt file's SHA-256 "
          f"{result.summary['file']['sha256']} (the stored {bank.sha256}; a zip archive's bytes may carry a time)")
