"""Generator checks of the appearance scene's augmentation, layout and labels
(docs/synthetic-real-appearance-validation.md, "Generator checks").

GA7 (transforms recorded and in range), GA8 (labels against the placements), GA9 (ownership and anchors), GA10
(placement and its stop; DN1's stop is A3's, on ``app_dn1``), GA11 (territories), GA18 (projection truth: the
columns), the label parts of GA19 (``Z=1``) and GA20 (the layout's streams), and the placement refusals of GA3, on the
fixtures ``app_tiny``, ``app_plane``, ``app_morph``, ``app_cases``, ``app_crowded``, ``app_streams`` and
``app_cu_offset`` (``cellular_appearance_scenes.py``), recomputed by the test oracle's label part
(``cellular_appearance_oracle.py``). Every bank is a closed-form bank or is built in the session from hand-made arrays
under the test's temporary directory; nothing is read from a run directory or a real image. Reported values (boundary
voxels, pieces per label, the culture's centroid median and columns with two nucleus IDs, anchors off the plane) are
printed.
"""
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

from starfinder.synthetic import BankReference, build_appearance_bank, load_appearance_bank
from starfinder.synthetic._appearance_layout import (_LAYOUT_COMPONENTS, AugmentationConfig, LayoutConfig,
                                                     _fov_labels, _labels_from_table, _morphology_labels,
                                                     _place_appearance, _sample_labels)
from starfinder.synthetic._context import _CONTEXT_COMPONENTS
from starfinder.synthetic._formed import _COMPONENTS as _FORMED_COMPONENTS
from starfinder.synthetic._sample import _SAMPLE_COMPONENTS

from . import cellular_appearance_oracle as oracle
from .cellular_appearance_scenes import (APP_COUNTS, GAIN, MORPH_WHOLE, REFUSAL_RUN, app_context, app_crowded,
                                         app_layout, app_tiny, case_placements, cu_offset_case, spheres_case,
                                         write_case)

pytestmark = [pytest.mark.synthetic]

COPIES = ("dense3d", "tissue2d", "culture")
PLANES = ("dense3d-plane", "tissue2d-plane", "culture-plane")
MORPH = ("culture-morph-drawn", "culture-morph-supplied")
# The accepted components an appearance layout may reuse: the FOV jitter and the morphology round's translation.
REUSED = {"sample.jitter", "context.morphology.translation"}


def built_bank(case, directory):
    """Build a hand-made case into ``directory`` and load it by name and SHA-256."""
    config = write_case(case, Path(directory) / "source")
    path = Path(directory) / "bank" / f"{config['bank']}.npz"
    build_appearance_bank(config, path)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return load_appearance_bank(BankReference(name=config["bank"], sha256=sha256, directory=str(path.parent)))


@pytest.fixture(scope="session")
def cu_bank(tmp_path_factory):
    """The culture bank built from ``cu_offset_case`` (four cells through 7 planes)."""
    return built_bank(cu_offset_case(), tmp_path_factory.mktemp("cu-offset"))


@pytest.fixture(scope="session")
def cu_offset(cu_bank):
    """app_cu_offset: the culture app_tiny configuration with 7 planes per FOV and DN3, on ``cu_bank``."""
    bank = cu_bank
    context = app_context("culture", planes=7)
    layout = app_layout(APP_COUNTS["cu-offset"])
    return context, bank, AugmentationConfig(), layout, _place_appearance(context, bank, AugmentationConfig(), layout)


@pytest.fixture(scope="session")
def spheres(tmp_path_factory):
    """app_cases' two-entry dense bank (two balls of radius 1 µm)."""
    return built_bank(spheres_case(), tmp_path_factory.mktemp("spheres"))


def placed_of(name, request):
    """A placed copy by name; app_cu_offset's session fixture is built only when it is asked for."""
    return request.getfixturevalue("cu_offset") if name == "cu-offset" else app_tiny(name)


def redrawn_growth(placed, context):
    """CT1's growth of every placed object redrawn from ``context.appearance.growth`` (entity ``cell-<k>``) with the
    context's lognormal growth law; None in the culture."""
    if placed.kind == "cell":
        return None
    law = context.territories.growth
    assert law.mode == "lognormal"
    scheme = placed.record["stream"]
    return np.array([math.exp(law.parameters[0] + law.parameters[1]
                              * oracle.stream(scheme, "context.appearance.growth", f"cell-{k}").standard_normal())
                     for k in placed.placements.object_id])


def grids(placed):
    """(name, origin, shape) of every grid of a placed layout: the sample grid, each FOV at its true origin and, with a
    morphology round, each FOV's morphology grid true_f + τ_f + q ⊙ s; and the generator's labels on each."""
    grid = placed.layout
    out = [("sample", grid.origin, grid.shape, _sample_labels(placed))]
    fov = _fov_labels(placed)
    out += [(f, placed.true[k], grid.n, fov[f]) for k, f in enumerate(placed.fov_ids)]
    morph = _morphology_labels(placed)
    out += [(f"{f}/morphology", placed.true[k] + placed.shifts[f], grid.n, morph[f])
            for k, f in enumerate(placed.fov_ids) if f in morph]
    return out


def pieces(N, k):
    """Connected pieces of label k: 26-connected in 3D, 8-connected in a plane."""
    mask = N == k
    if N.shape[0] == 1:
        return ndi.label(mask[0], structure=np.ones((3, 3)))[1]
    return ndi.label(mask, structure=np.ones((3, 3, 3)))[1]


# ------------------------------------------------------------------------------------------------ GA3


@pytest.mark.contract
def test_ga3_augmentation_and_layout_refusals():
    for field, value in (("scale", (0.0, 1.1)), ("scale", (-0.5, 1.1)), ("gain", (0.0, 1.0)), ("gain", (1.1, 0.9)),
                         ("scale", (0.9,))):
        with pytest.raises(ValueError, match=field):
            AugmentationConfig(**{field: value})
    with pytest.raises(ValueError, match="flip_y"):
        AugmentationConfig(flip_y=1.5)
    for floor in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="floor"):
            LayoutConfig(density="count", count=5, floor=floor)
    assert LayoutConfig(density="count", count=5, floor=1.0).floor == 1.0
    with pytest.raises(ValueError, match="count"):
        LayoutConfig(density="count")
    with pytest.raises(ValueError, match="target"):
        LayoutConfig(target=1.5)
    with pytest.raises(ValueError, match="density"):
        LayoutConfig(density="packed")
    context, bank, augmentation, _, _ = app_tiny("tissue2d")
    with pytest.raises(ValueError, match="DN1"):
        _place_appearance(context, bank, augmentation, LayoutConfig(target=0.5, threshold=0.4))


@pytest.mark.contract
def test_ga3_a_reached_count_below_floor_raises_naming_both_values():
    _, _, layout, error = app_crowded("raise")
    assert isinstance(error, ValueError)
    message = str(error)
    count = APP_COUNTS["dense3d"]
    assert f"a count of {count} " in message
    assert f"floor · count = 0.9 · 100 = {0.9 * 100}" in message
    assert f"max_refusals={REFUSAL_RUN}" in message


# ------------------------------------------------------------------------------------------------ GA7


@pytest.mark.validation
@pytest.mark.parametrize("name", [*COPIES, "dense3d-gain", "cu-offset"])
def test_ga7_transforms_are_in_range_redrawn_and_entries_follow_bs2(name, request):
    context, bank, augmentation, _, placed = placed_of(name, request)
    table = placed.placements
    kind = oracle.kind_of(bank)
    scheme = placed.record["stream"]
    ranges = dict(scale=augmentation.scale, gain=augmentation.gain, flip_y=augmentation.flip_y,
                  flip_z=augmentation.flip_z)
    ids = set(bank.entries.entry_id)
    for row in table.itertuples():
        assert row.entry_id in ids
        assert 0 <= row.theta < 2 * math.pi
        assert row.flip_y in (-1, 1)
        assert row.flip_z in ((-1, 1) if kind == "nucleus" else (1,))
        assert augmentation.scale[0] <= row.scale_yx <= augmentation.scale[1]
        assert row.scale_z == (row.scale_yx if kind == "nucleus" else 1.0)
        assert augmentation.gain[0] <= row.gain <= augmentation.gain[1]
        draws = oracle.augment_draws(scheme, int(row.object_id), int(row.proposal), ranges, kind)
        assert {key: getattr(row, key) for key in draws} == draws
    # BS2: the queue replayed from context.appearance.entry gives every placed entry; no entry twice before the bank
    # is exhausted.
    placed_at = {int(row.proposal): row.entry_id for row in table.itertuples()}
    drawn = oracle.replay_queue(scheme, len(bank), placed_at, int(placed.record["density"]["proposals"]))
    entry_ids = list(bank.entries.entry_id)
    assert {j: entry_ids[drawn[j]] for j in placed_at} == placed_at
    first = list(table.entry_id[:len(bank)])
    assert len(set(first)) == len(first)
    if kind != "cell":
        return
    # The culture's Z: each placed nucleus centroid is its source centroid plus one shift Δ, which moves the median
    # source nucleus centroid of the bank onto z_mid.
    rows = {e: bank.entry(e) for e in ids}
    source, offset = {}, {}
    for entry_id, i in rows.items():
        _, _, nucleus = oracle.stored(bank, entry_id)
        info = bank.entries.iloc[i]
        z = np.nonzero(nucleus <= 0)[0].mean() * info.spacing_z
        source[entry_id] = info.box_origin_z * info.spacing_z + z
        offset[entry_id] = z - (info.shape_z - 1) / 2 * info.spacing_z
    z_mid = (context.sample.acquisition.fov_shape_zyx[0] - 1) / 2 * context.sample.acquisition.spacing_zyx[0]
    shift = z_mid - float(np.median(list(source.values())))
    assert placed.record["culture_shift_um"] == pytest.approx(shift, abs=1e-9)
    placed_z = np.array([row.centre_z + offset[row.entry_id] for row in table.itertuples()])
    expected = np.array([source[row.entry_id] + shift for row in table.itertuples()])
    assert np.abs(placed_z - expected).max() <= 1e-9
    median = float(np.median(placed_z))
    print(f"GA7 {name}: z_mid {z_mid} µm, shift {shift:.6f} µm, median placed nucleus centroid {median:.6f} µm "
          f"({len(table)} objects)")
    if name == "cu-offset":
        assert median == pytest.approx(z_mid, abs=1e-9)
        # The box centre is one source plane below the nucleus centroid, so a check of centre_z would fail here.
        assert np.abs(table.centre_z.to_numpy() - expected).min() > 0.3


# ------------------------------------------------------------------------------------------------ GA8


@pytest.mark.validation
@pytest.mark.parametrize("name", [*COPIES, *PLANES, *MORPH])
def test_ga8_labels_equal_the_oracle_on_every_grid(name):
    context, bank, _, _, placed = app_tiny(name)
    objects = oracle.frames(bank, placed.placements)
    growth = redrawn_growth(placed, context)
    boundary_total = edge_total = 0
    for grid_name, origin, shape, generated in grids(placed):
        points = oracle.grid_points(origin, placed.layout.s, shape)
        N, C, boundary, edge = oracle.labels(points.reshape(-1, 3), objects, oracle.kind_of(bank), growth)
        N, C, boundary, edge = (a.reshape(tuple(shape)) for a in (N, C, boundary, edge))
        differ = (generated["nucleus"] != N) | (generated["cell"] != C)
        assert not (differ & ~boundary).any(), f"{name} {grid_name}: {int((differ & ~boundary).sum())} voxels differ"
        point_C, point_N = placed.labels.labels(points)
        assert np.array_equal(point_N, generated["nucleus"]) and np.array_equal(point_C, generated["cell"])
        boundary_total += int(boundary.sum())
        edge_total += int(edge.sum())
    sample = grids(placed)[0][3]["nucleus"]
    counts = [pieces(sample, k) for k in placed.placements.object_id]
    print(f"GA8 {name}: boundary voxels {boundary_total}, at a growth edge {edge_total}; pieces per label on the "
          f"sample grid {counts}")
    assert boundary_total == 0


@pytest.mark.validation
def test_ga8_ga9_hand_made_placements(spheres, cu_bank):
    """app_cases: the review's two unit spheres 1.9 µm apart, two spheres tying under CT1, and a culture cell whose
    nucleus lies in an earlier cell's overlap; each on a grid at a fractional offset in Z, Y and X and at Z=1."""
    one, two = sorted(spheres.entries.entry_id)
    first, second = sorted(cu_bank.entries.entry_id)[:2]
    # case: (bank, placements, growth, spacing, grids as (origin, shape)); the tie case's X step of 0.125 µm puts voxel
    # centres exactly on the mid-plane x = 0 between its two spheres.
    cases = dict(
        review=(spheres, case_placements([(1, one, (0.0, 0.0, 0.0)), (2, two, (0.0, 0.0, 1.9))]), [1.0, 1.0],
                (0.1, 0.1, 0.1), [((-1.2137, -1.1871, -1.3213), (25, 25, 46)),
                                  ((0.0371, -1.1871, -1.3213), (1, 25, 46))]),
        tie=(spheres, case_placements([(1, one, (0.0, 0.0, -1.5)), (2, two, (0.0, 0.0, 1.5))]), [1.0, 1.0],
             (0.1, 0.1, 0.125), [((-1.2137, -1.1871, -3.0), (25, 25, 49)), ((0.0371, -1.1871, -3.0), (1, 25, 49))]),
        culture=(cu_bank, case_placements([(1, first, (1.0, 0.0, 0.0)), (2, second, (1.0, 0.0, 3.6))]), None,
                 (0.1731275, 0.2, 0.2), [((0.0137, -3.4171, -3.4213), (8, 18, 37)),
                                         ((1.0371, -3.4171, -3.4213), (1, 18, 37))]))
    report = {}
    for case, (bank, table, growth, spacing, case_grids) in cases.items():
        evaluator = _labels_from_table(bank, table, growth)
        objects = oracle.frames(bank, table)
        kind = oracle.kind_of(bank)
        boundary_total = 0
        for origin, shape in case_grids:
            points = oracle.grid_points(origin, spacing, shape)
            flat = points.reshape(-1, 3)
            N, C, boundary, _ = (a.reshape(shape) for a in oracle.labels(flat, objects, kind, growth))
            gen_C, gen_N = evaluator.grid(origin, spacing, shape)
            point_C, point_N = evaluator.labels(points)
            assert np.array_equal(gen_C, point_C) and np.array_equal(gen_N, point_N)
            differ = (gen_N != N) | (gen_C != C)
            assert not (differ & ~boundary).any()
            # One ownership rule: N = n > 0 implies C = cell(n) (= n) on every voxel.
            assert np.all(gen_C[gen_N > 0] == gen_N[gen_N > 0])
            boundary_total += int(boundary.sum())
            if case == "tie":
                delta = evaluator.delta(points)
                tied = (gen_N == 0) & (delta[..., 0] == delta[..., 1]) & (delta[..., 0] <= 1.0)
                assert tied.any() and np.all(gen_C[tied] == 1)
            if case == "culture":
                inside_first = objects[0].d(flat).reshape(shape) <= 0
                second_nucleus = (objects[1].nd(flat) <= 0) & (objects[1].d(flat) <= 0)
                overlap = inside_first & second_nucleus.reshape(shape)
                assert overlap.any()
                assert np.all(gen_C[overlap] == 1) and np.all(gen_N[overlap] != 2)
        report[case] = boundary_total
    assert report["review"] == 0 and report["culture"] == 0 and report["tie"] > 0
    # The review's point 0.98 µm from the first centre on the line of centres: both spheres hold it; N = C = 1.
    review = cases["review"][1]
    evaluator = _labels_from_table(spheres, review, [1.0, 1.0])
    point = np.array([[0.0, 0.0, 0.98]])
    C, N = evaluator.labels(point)
    assert (int(N[0]), int(C[0])) == (1, 1)
    delta = evaluator.delta(point)[0]
    assert (delta <= 0).all()
    N, C, _, _ = oracle.labels(point, oracle.frames(spheres, review), "nucleus", [1.0, 1.0])
    assert (int(N[0]), int(C[0])) == (1, 1)
    print(f"GA8 app_cases: boundary voxels per case {report} (the tie case is built to have them); δ at the review's "
          f"point {delta.tolist()} µm (the page's continuous spheres: −0.02, −0.08)")


# ------------------------------------------------------------------------------------------------ GA9 and GA10


def lattice(placed):
    """The placement lattice of a placed layout (the sample grid, one plane in a planar placement) and its voxel."""
    grid = placed.layout
    plane = placed.record["placement_plane_z"]
    shape = grid.shape.copy()
    if plane is not None:
        shape[0] = 1
    voxel = float(grid.s[1] * grid.s[2]) if plane is not None else float(np.prod(grid.s))
    return grid.origin, grid.s, tuple(int(v) for v in shape), plane, voxel


@pytest.mark.validation
@pytest.mark.parametrize("name", [*COPIES, *PLANES])
def test_ga9_ownership_overlap_and_anchors(name):
    context, bank, _, _, placed = app_tiny(name)
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
    N, C, _, _ = oracle.labels(anchors, objects, oracle.kind_of(bank), redrawn_growth(placed, context))
    own = (C if placed.kind == "cell" else N)
    assert np.array_equal(own, table.object_id.to_numpy())
    point_C, point_N = placed.labels.labels(anchors)
    assert np.array_equal(point_C if placed.kind == "cell" else point_N, table.object_id.to_numpy())
    for _, _, _, generated in grids(placed):
        nucleus, cell = generated["nucleus"], generated["cell"]
        assert np.all(cell[nucleus > 0] == nucleus[nucleus > 0])


def replay_centre(placed, bank, row, frame, earlier, canvas, plane, contact):
    """The centre PL2's rule gives for a placed row when its proposal is replayed (the contact ladder against the
    earlier masks, or the uniform point in Ω widened by 0.3 of the box), with the steps of the ladder below the one
    taken; the culture's and a planar placement's Z is the recorded one."""
    scheme = placed.record["stream"]
    kind = oracle.kind_of(bank)
    k, j = int(row.object_id), int(row.proposal)
    origin, spacing, shape, _, _ = lattice(placed)
    draws = oracle.placement_draws(scheme, k, j, contact, k - 1, plane is not None or kind == "cell")
    r_k = oracle.equivalent_radius(bank, row.entry_id, row.scale_z, row.scale_yx)
    omega = placed.layout.omega
    if not draws["contact"]:
        w = oracle.widening(bank, row.entry_id, row.scale_z, row.scale_yx)
        centre = omega[0] - w + draws["uniform"] * (omega[1] - omega[0] + 2 * w)
        centre[0] = row.centre_z if (kind == "cell" or plane is not None) else centre[0]
        return centre, None
    partner = earlier[draws["partner"]]
    r_i = oracle.equivalent_radius(bank, partner.entry_id, partner.scale_z, partner.scale_yx)
    c_i = np.array([partner.centre_z, partner.centre_y, partner.centre_x])
    for f in oracle.LADDER:
        centre = c_i + f * (r_i + r_k) * draws["direction"]
        if kind == "cell" or plane is not None:
            centre[0] = row.centre_z
        o = frame.at(centre)
        lo, mask, _, points = oracle.lattice_mask(o, origin, spacing, plane)
        voxels = int(mask.sum())
        inside = int(oracle.inside(points[mask], omega).sum())
        assert voxels > 0 and 100 * inside >= 35 * voxels, f"object {k}: the ladder left Ω at f = {f}"
        window = oracle.on_grid(lo, mask, shape)
        overlap = 0 if window is None else int((canvas[window[0]] & mask[window[1]]).sum())
        if 100 * overlap <= voxels:
            return centre, f
    raise AssertionError(f"object {k}: no step of the ladder has an overlap of at most 1 %")


@pytest.mark.validation
@pytest.mark.parametrize("name", [pytest.param("dense3d", marks=pytest.mark.slow), "tissue2d", "culture", *PLANES,
                                  "cu-offset"])
def test_ga10_each_centre_is_the_one_pl2_gives(name, request):
    _, bank, _, layout, placed = placed_of(name, request)
    table = placed.placements
    objects = oracle.frames(bank, table)
    origin, spacing, shape, plane, _ = lattice(placed)
    canvas = np.zeros(shape, dtype=bool)
    nuclei = np.zeros(shape, dtype=bool)
    rows = list(table.itertuples())
    steps = []
    for i, (o, row) in enumerate(zip(objects, rows)):
        centre, step = replay_centre(placed, bank, row, o, rows[:i], canvas, plane, layout.contact)
        recorded = np.array([row.centre_z, row.centre_y, row.centre_x])
        assert np.abs(recorded - centre).max() <= 1e-9, f"object {o.k}: {recorded} against {centre}"
        steps.append(step)
        lo, mask, nucleus, points = oracle.lattice_mask(o, origin, spacing, plane)
        assert 100 * int(oracle.inside(points[mask], placed.layout.omega).sum()) >= 35 * int(mask.sum())
        window = oracle.on_grid(lo, mask, shape)
        if nucleus is not None and window is not None:
            assert not (nuclei[window[0]] & nucleus[window[1]]).any(), f"cell {o.k}: its nucleus touches an earlier one"
            nuclei[window[0]] |= nucleus[window[1]]
        if window is not None:
            canvas[window[0]] |= mask[window[1]]
    # The record against a recount: DN3 placed its count, every proposal is in the proposals table.
    density = placed.record["density"]
    proposals = placed.proposals
    assert density["stopped_by"] == "count" and density["placed"] == density["reached"] == len(table) == layout.count
    assert density["proposals"] == len(proposals) == int(table.proposal.max())
    assert list(proposals.proposal) == list(range(1, len(proposals) + 1))
    assert density["refused"] == int((proposals.outcome == "refused").sum())
    assert {r: int((proposals.reason == r).sum()) for r in density["refusals"]} == density["refusals"]
    placed_rows = proposals[proposals.outcome == "placed"]
    assert list(placed_rows.proposal) == list(table.proposal) and list(placed_rows.entry_id) == list(table.entry_id)
    print(f"GA10 {name}: {len(table)} objects in {density['proposals']} proposals; ladder steps taken "
          f"{[s for s in steps if s is not None]}")


@pytest.mark.validation
@pytest.mark.parametrize("variant", ["record", "proposals"])
def test_ga10_crowded_placement_stops_and_records(variant):
    _, bank, layout, placed = app_crowded(variant)
    density = placed.record["density"]
    table, proposals = placed.placements, placed.proposals
    assert density["placed"] == density["reached"] == len(table) < layout.count
    assert density["proposals"] == len(proposals)
    if variant == "record":
        # Stopped by the run of refusals; the reached count is app_tiny's (the same draws up to its last placement).
        assert density["stopped_by"] == "max_refusals"
        assert density["refused_in_a_row"] == REFUSAL_RUN
        assert list(proposals.outcome[-REFUSAL_RUN:]) == ["refused"] * REFUSAL_RUN
        assert len(table) == APP_COUNTS["dense3d"]
    else:
        assert density["stopped_by"] == "max_proposals" and density["proposals"] == layout.max_proposals == 30
        assert density["refused_in_a_row"] < layout.max_refusals


# ------------------------------------------------------------------------------------------------ GA11 and GA18


@pytest.mark.validation
@pytest.mark.parametrize("name", COPIES)
def test_ga11_territories(name):
    context, bank, _, _, placed = app_tiny(name)
    table, cells = placed.placements, placed.cells
    # One ordinary cell with one nucleus per placed object.
    assert list(cells.cell_id) == list(table.object_id) == list(cells.nucleus_id)
    assert set(cells.kind) == {"ordinary"} and set(cells.n_nuclei) == {1}
    growth = redrawn_growth(placed, context)
    objects = oracle.frames(bank, table)
    ties = 0
    for _, origin, shape, generated in grids(placed):
        points = oracle.grid_points(origin, placed.layout.s, shape)
        N, C = generated["nucleus"], generated["cell"]
        if placed.kind == "cell":
            # CU1: the first placed cell whose mask holds the voxel.
            D = np.stack([o.d(points.reshape(-1, 3)) for o in objects], axis=1)
            held = D <= 0
            first = np.where(held.any(axis=1), table.object_id.to_numpy()[np.argmax(held, axis=1)], 0)
            assert np.array_equal(C.ravel(), first)
            continue
        assert np.abs(cells.growth.to_numpy() - growth).max() <= 1e-9
        delta = placed.labels.delta(points)
        free = N == 0
        d = delta[free]
        nearest = np.argmin(d, axis=1)
        best = d[np.arange(len(d)), nearest]
        cell = C[free]
        ids = table.object_id.to_numpy()
        territory = cell > 0
        assert np.array_equal(cell[territory], ids[nearest[territory]])
        assert np.all(best[territory] <= growth[nearest[territory]]) and np.all(best[territory] > 0)
        assert np.all(~np.isfinite(best[~territory]) | (best[~territory] > growth[nearest[~territory]]))
        if d.shape[1] > 1:
            two = np.sort(d, axis=1)[:, :2]
            finite = np.isfinite(two).all(axis=1)
            ties += int((two[finite, 1] - two[finite, 0] <= oracle.BOUNDARY).sum())
    print(f"GA11 {name}: CT1 ties within 1e-9 µm {ties}" if placed.kind != "cell" else f"GA11 {name}: CU1")
    assert ties == 0


@pytest.mark.validation
@pytest.mark.parametrize("name", ["tissue2d", "culture"])
def test_ga18_columns_hold_one_nucleus_in_2d_tissue(name):
    _, _, _, _, placed = app_tiny(name)
    grid = placed.layout
    checked = list(grids(placed))
    offset = grid.origin + np.array([0.37, 0.21, 0.43]) * grid.s
    points = oracle.grid_points(offset, grid.s, grid.n)
    C, N = placed.labels.labels(points)
    checked.append(("offset", offset, grid.n, dict(nucleus=N, cell=C)))
    counts = {}
    for grid_name, _, _, generated in checked:
        N = generated["nucleus"]
        ids = np.sort(N, axis=0)
        two = (ids[1:] != ids[:-1]) & (ids[:-1] > 0)
        columns = int(two.any(axis=0).sum())
        counts[grid_name] = columns
        if name == "tissue2d":
            assert columns == 0
            assert np.array_equal(N, np.broadcast_to(N[:1], N.shape))
    print(f"GA18 {name}: columns holding two nucleus IDs per grid {counts}")


# ------------------------------------------------------------------------------------------------ GA19 and GA20


@pytest.mark.validation
@pytest.mark.parametrize("name", PLANES)
def test_ga19_one_plane_labels(name):
    _, _, _, _, placed = app_tiny(name)
    table = placed.placements
    assert all(generated["nucleus"].shape == (1, 36, 36) for g, _, _, generated in grids(placed) if g != "sample")
    assert _sample_labels(placed)["nucleus"].shape == (1, *placed.layout.shape[1:])
    assert placed.layout.shape[0] == 1 and placed.layout.origin[0] == 0
    assert (table.centre_z == 0.0).all()
    assert (placed.true[:, 0] == 0.0).all()
    if placed.shifts is not None:
        assert all(shift[0] == 0.0 for shift in placed.shifts.values())
    off = table.anchor_z[table.anchor_z != 0.0]
    if placed.kind == "column":
        assert off.empty
    print(f"GA19 {name}: anchors off the plane z = 0 (3D entries keep their deepest point): {off.round(3).tolist()}")


@pytest.mark.validation
@pytest.mark.parametrize("name", ["dense3d", "tissue2d", "culture-morph-drawn"])
def test_ga20_layout_components_start_with_context_appearance(name):
    _, _, _, _, placed = app_tiny(name)
    accepted = set(_SAMPLE_COMPONENTS) | set(_CONTEXT_COMPONENTS) | set(_FORMED_COMPONENTS)
    assert not any(c.startswith("context.appearance.") for c in accepted)
    drawn = {descriptor[4] for descriptor in placed.log.values()}
    assert drawn == set(placed.record["stream"]["drawn"])
    new = drawn - REUSED
    assert new and all(c.startswith("context.appearance.") for c in new) and new <= set(_LAYOUT_COMPONENTS)
    assert not new & accepted
    for prefix in ("context.nucleus.", "context.cell.kind", "context.cell.contact", "context.stain.",
                   "context.texture."):
        assert not any(c.startswith(prefix) for c in drawn)
    assert ("context.morphology.translation" in drawn) == (placed.shifts is not None)


# The page's app_streams copies are the dense 3D ones (each check places two scenes on the dense bank, 5 s or more
# alone); the 2D tissue copies keep the same checks in the routine tier.
STREAMS = ["tissue2d", pytest.param("dense3d", marks=pytest.mark.slow)]


@pytest.mark.validation
@pytest.mark.parametrize("copy", STREAMS)
def test_ga20_a_changed_gain_range_changes_no_other_draw(copy):
    """The same descriptors and proposals, every column but the gain equal, and the gain the same uniform on the new
    range."""
    _, _, _, _, base = app_tiny(copy)
    _, _, _, _, gain = app_tiny(f"{copy}-gain")
    assert set(gain.log) == set(base.log)
    assert gain.proposals.equals(base.proposals)
    assert gain.placements.drop(columns="gain").equals(base.placements.drop(columns="gain"))
    u = (base.placements.gain - 0.9) / 0.2
    assert np.abs(gain.placements.gain - (GAIN[0] + (GAIN[1] - GAIN[0]) * u)).max() <= 1e-12


@pytest.mark.validation
@pytest.mark.parametrize("copy", STREAMS)
def test_ga20_a_type_size_scale_changes_no_placement_or_label(copy):
    """Type 2's size scale 1.5 has no effect on the layout: sizes come from the entries and AU1's scale."""
    _, _, _, _, base = app_tiny(copy)
    _, _, _, _, scaled = app_tiny(f"{copy}-size-scale")
    assert scaled.placements.equals(base.placements) and set(scaled.log) == set(base.log)
    sample, other = _sample_labels(base), _sample_labels(scaled)
    assert all(np.array_equal(sample[t], other[t]) for t in sample)


@pytest.mark.validation
def test_ga8_morphology_grid_is_the_reference_grid_moved_by_whole_voxels():
    """With the supplied whole-voxel translation (0, 2, −3) voxels the morphology grid is the FOV grid shifted by
    whole voxels, so its labels equal the reference grid's labels evaluated on the shifted grid."""
    _, _, _, _, placed = app_tiny("culture-morph-supplied")
    grid = placed.layout
    morph = _morphology_labels(placed)
    for k, fov in enumerate(placed.fov_ids):
        assert np.array_equal(placed.shifts[fov], np.array(MORPH_WHOLE) * grid.s)
        C, N = placed.labels.grid(placed.true[k] + np.array(MORPH_WHOLE) * grid.s, grid.s, grid.n)
        assert np.array_equal(morph[fov]["nucleus"], N) and np.array_equal(morph[fov]["cell"], C)
