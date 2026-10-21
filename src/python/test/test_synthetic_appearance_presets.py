"""Generator checks of the routine appearance presets (docs/synthetic-real-appearance-validation.md).

GA8 (labels against the placements), GA9 (ownership and anchors), GA10 (placement and its stop), GA11 (territories) and
GA18 (projection truth) on ``app_routine``, the three routine appearance presets on their closed-form banks (DN3), as
generated; recomputed by the test oracle (``cellular_appearance_oracle.py``) with the helpers of the layout's checks
(``test_synthetic_appearance_scene.py``). GA24's routine part: each routine preset generated in its own process under
/usr/bin/time -v within the project's 4 GiB. These checks read no real data. The real-bank checks GA23, GA26, GA27
and GA24's ``slow`` presets, acquired variants and bank builds were moved by W-536 to
``benchmarks/appearance_evidence.py``, run by the Chapter II evidence project. Reported values are printed.
"""
import numpy as np
import pandas as pd
import pytest

from starfinder.synthetic import APPEARANCE_PRESETS

from . import cellular_appearance_oracle as oracle
from .cellular_appearance_scenes import ROUTINE_PRESETS, ROUTINE_RSS_KIB, app_routine, generate_measured
from .test_synthetic_appearance_scene import grids, lattice, pieces, redrawn_growth, replay_centre

pytestmark = [pytest.mark.synthetic, pytest.mark.validation]

slow = pytest.mark.slow


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


# ------------------------------------------------------------------------------------------------ GA24 (routine)
# GA24's slow presets, acquired variants and bank builds, and GA23, GA26 and GA27, read the real banks and allow RG1
# (3,600 s and 16 GiB per step); W-536 moved them to benchmarks/appearance_evidence.py, which the Chapter II evidence
# project runs.


@slow
@pytest.mark.parametrize("name", ROUTINE_PRESETS)
def test_ga24_routine_preset_generation_cost(name, tmp_path):
    result = generate_measured(name, tmp_path)
    assert result.exit == 0, result.output
    assert result.peak_rss_kib <= ROUTINE_RSS_KIB
    assert result.summary["placed"] == APPEARANCE_PRESETS[name]["count"]
    print(f"GA24 {name}: process {result.wall_s:.2f} s (generation {result.summary['generation_s']:.2f} s), peak RSS "
          f"{result.peak_rss_kib} KiB")
