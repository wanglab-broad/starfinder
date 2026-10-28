"""The FOV ownership region on the §2.9 assignment figure (W-330): ``fov_ownership`` and the ``ownership``
argument of ``plot_assignment`` (design/specs/assignment.md, "Diagnostics"; design/specs/assembly.md,
"Ownership regions").

The region of each FOV is checked against a mask computed here from the placement origins, the grid
shapes and the spacing by the page's rules O1 and O2 with T1, without ``ownership_map``. The excluded
observations are checked against the assembled tables of ``own_cases`` (FOV2 has a cell excluded
``not_owner``, molecules excluded ``cell_not_owner`` and an unassigned one excluded ``not_owner``, and a
retained molecule flagged ``outside_owner_region``). The drawing is checked by its artists, against the
figure drawn without ``ownership``.
"""
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

from starfinder.assembly import AssemblyConfig, assemble, fov_ownership  # noqa: E402
from starfinder.assignment import AssignmentConfig, FOVOwnership, assign_molecules, plot_assignment  # noqa: E402
from starfinder.segmentation import ExpandLabelsConfig  # noqa: E402

from .assembly_fixtures import (CUT, GOLDEN_IDS, OWN_CELLS, OWN_GENES, OWN_IDS, OWN_MOLECULES, OWN_SHAPE,  # noqa: E402
                                approved, golden_origins, golden_placement, golden_results, inputs_of, label_run,
                                molecules, own_inputs, own_placement, own_results, paint, unit_grid)
from .test_workflow_assembly_golden import TILE_ZYX  # noqa: E402

pytestmark = [pytest.mark.stitching, pytest.mark.segmentation, pytest.mark.validation]

GREY, YELLOW = to_rgba("grey"), to_rgba("yellow")
REGION, EXCLUDED = "ownership region", "excluded from sample"
# The golden FOVs' grid cells (row, column) on the 2 × 2 grid of positions_from_grid: FOV1 has a neighbour
# below (FOV2) and one to the right (FOV3); the fourth cell is blank.
GOLDEN_CELLS = {"Position001": (0, 0), "Position002": (1, 0), "Position003": (0, 1)}


# --- The independent ownership rule of design/specs/assembly.md ("Ownership regions") ---------------------

def footprint(origin, shape, spacing):
    """Half-open footprint ((y0, y1), (x0, x1)) and centre (y, x): per axis [o − s/2, o + (n − ½)·s) and
    o + (n − 1)/2 · s."""
    return ([(o - s / 2, o + (n - 0.5) * s) for o, n, s in zip(origin, shape, spacing)],
            [o + (n - 1) / 2 * s for o, n, s in zip(origin, shape, spacing)])


def expected_owner(points, fovs, ownership):
    """The owner of each (y, x) sample point among fovs (table order: fov_id -> (origin_yx, shape_yx,
    spacing_yx)), or None. O1: the FOV whose footprint holds the point with the nearest centre, an exact tie
    to the first FOV of the table. O2: per FOV the footprint with each edge towards a grid neighbour moved to
    the middle of their overlap; a point on such an edge lies on two boxes (a tie: T1, the first FOV)."""
    owners = []
    boxes = {}
    if ownership == "midpoint_box":
        at = {cell: fov for fov, cell in GOLDEN_CELLS.items() if fov in fovs}
        for fov, (origin, shape, spacing) in fovs.items():
            ((y0, y1), (x0, x1)), _ = footprint(origin, shape, spacing)
            r, c = GOLDEN_CELLS[fov]
            closed = [False, False]
            for axis, (lower, upper) in enumerate((((r - 1, c), (r + 1, c)), ((r, c - 1), (r, c + 1)))):
                if lower in at:
                    other = fovs[at[lower]]
                    edge = (footprint(*other)[0][axis][1] + footprint(origin, shape, spacing)[0][axis][0]) / 2
                    (y0, x0) = (edge, x0) if axis == 0 else (y0, edge)
                if upper in at:
                    other = fovs[at[upper]]
                    edge = (footprint(origin, shape, spacing)[0][axis][1] + footprint(*other)[0][axis][0]) / 2
                    (y1, x1) = (edge, x1) if axis == 0 else (y1, edge)
                    closed[axis] = True
            boxes[fov] = (y0, y1, x0, x1, closed)
    for q_y, q_x in points:
        best, distance = None, np.inf
        for fov, (origin, shape, spacing) in fovs.items():
            ((y0, y1), (x0, x1)), (c_y, c_x) = footprint(origin, shape, spacing)
            if ownership == "nearest_center":
                if y0 <= q_y < y1 and x0 <= q_x < x1 and (q_y - c_y) ** 2 + (q_x - c_x) ** 2 < distance:
                    best, distance = fov, (q_y - c_y) ** 2 + (q_x - c_x) ** 2
            else:
                b_y0, b_y1, b_x0, b_x1, (closed_y, closed_x) = boxes[fov]
                in_y = b_y0 <= q_y and (q_y <= b_y1 if closed_y else q_y < b_y1)
                in_x = b_x0 <= q_x and (q_x <= b_x1 if closed_x else q_x < b_x1)
                if in_y and in_x:
                    best = fov
                    break
        owners.append(best)
    return owners


def golden_fovs(placement):
    """fov_id -> ((origin_y, origin_x), (Y, X), (spacing_y, spacing_x)) in table order (unit spacing)."""
    return {r.fov_id: ((r.origin_y, r.origin_x), TILE_ZYX[1:], (1.0, 1.0)) for r in placement.fovs.itertuples()}


def expected_mask(fov, fovs, ownership):
    (oy, ox), (ny, nx), (sy, sx) = fovs[fov]
    points = [(oy + i * sy, ox + j * sx) for i in range(ny) for j in range(nx)]
    return np.array([o == fov for o in expected_owner(points, fovs, ownership)]).reshape(ny, nx)


INTEGER_ORIGINS = {f"Position00{k}": tuple(float(v) for v in CUT[k]) for k in CUT}


@pytest.fixture(scope="module")
def golden_3d():
    return golden_results("3d")


@pytest.mark.parametrize("ownership", ["nearest_center", "midpoint_box"])
@pytest.mark.parametrize("origins", ["fractional", "integer"])
def test_the_region_is_the_fovs_ownership_region(golden_3d, ownership, origins):
    placement = golden_placement(origins=golden_origins() if origins == "fractional" else INTEGER_ORIGINS)
    result = assemble(placement, inputs_of(golden_3d), config=AssemblyConfig(ownership=ownership))
    fovs = golden_fovs(placement)
    regions = {}
    for fov in GOLDEN_IDS:
        record = fov_ownership(result, fov, placement=placement)
        assert record.fov_id == fov and record.region.dtype == bool and record.region.shape == TILE_ZYX[1:]
        assert np.array_equal(record.region, expected_mask(fov, fovs, ownership)), (fov, ownership)
        regions[fov] = record.region
    assert all(regions[f].any() and not regions[f].all() for f in GOLDEN_IDS)
    if origins == "fractional":
        return
    # On grids at integer origins (unit spacing) the pixel supports of the FOVs coincide, so the masks placed
    # in the sample frame cover each point exactly once that a footprint covers: on a raster 0.25 µm apart,
    # offset from every pixel edge.
    values = np.arange(-1.0, 120.0, 0.25) + 0.125
    q_y, q_x = (a.ravel() for a in np.meshgrid(values, values, indexing="ij"))
    covered = np.zeros(q_y.shape, bool)
    count = np.zeros(q_y.shape, np.int64)
    for fov, ((oy, ox), (ny, nx), _) in fovs.items():
        ((y0, y1), (x0, x1)), _ = footprint((oy, ox), (ny, nx), (1.0, 1.0))
        inside = (q_y >= y0) & (q_y < y1) & (q_x >= x0) & (q_x < x1)
        covered |= inside
        i = np.floor(q_y - oy + 0.5).astype(int)
        j = np.floor(q_x - ox + 0.5).astype(int)
        count[inside] += regions[fov][i[inside], j[inside]]
    assert covered.sum() > 0 and np.array_equal(count[covered], np.ones(covered.sum(), np.int64))
    assert not count[~covered].any()


# --- The excluded observations ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def own():
    placement = own_placement()
    return placement, assemble(placement, own_inputs()), own_results()


def test_the_excluded_observations_are_the_assemblys(own):
    placement, result, _ = own
    for fov in OWN_IDS:
        record = fov_ownership(result, fov, placement=placement)
        cells = result.cells[result.cells.fov_id == fov]
        mols = result.molecules[result.molecules.fov_id == fov]
        assert record.excluded_cells == {int(c) for c in cells.cell_id[cells.assembly_status == "excluded"]}
        assert record.excluded_molecules == set(mols.spot_id[mols.assembly_status == "excluded"])
    record = fov_ownership(result, "FOV2", placement=placement)
    rows = result.molecules[result.molecules.fov_id == "FOV2"].set_index("spot_id")
    reasons = rows.assembly_reason.astype(object).where(rows.assembly_reason.notna(), None).to_dict()
    # The fixture's cases: cells 2 and 3 excluded not_owner; m2, m3 cell_not_owner; m6 unassigned not_owner;
    # m7 outside_grid; m1 retained and flagged outside_owner_region, so not excluded.
    assert record.excluded_cells == {2, 3}
    assert record.excluded_molecules == {"m2", "m3", "m6", "m7"}
    assert (reasons["m2"], reasons["m3"], reasons["m6"]) == ("cell_not_owner", "cell_not_owner", "not_owner")
    assert rows.loc["m6", "assignment_status"] == "unassigned"
    assert rows.loc["m1", "assembly_flags"] == "outside_owner_region" and "m1" not in record.excluded_molecules
    # FOV2's region: the pixel centres at global x 14 and beyond (x 13.5 is the boundary; local x 0 and 1 are
    # owned by FOV1).
    expected = np.zeros(OWN_SHAPE[1:], bool)
    expected[:, 2:] = True
    assert np.array_equal(record.region, expected)


def test_a_fov_that_is_not_placed_and_one_the_assembly_does_not_hold(golden_3d):
    failed = approved(golden_placement(statuses={"Position003": "failed"}, review="required"))
    result = assemble(failed, inputs_of(golden_3d))
    record = fov_ownership(result, "Position003", placement=failed)
    assert record.region.shape == TILE_ZYX[1:] and not record.region.any()
    rows = result.cells[result.cells.fov_id == "Position003"]
    mols = result.molecules[result.molecules.fov_id == "Position003"]
    assert record.excluded_cells == {int(c) for c in rows.cell_id} and len(rows)
    assert record.excluded_molecules == set(mols.spot_id) and len(mols)
    missing = approved(golden_placement(statuses={"Position003": "missing"}, review="required"))
    result = assemble(missing, inputs_of({f: r for f, r in golden_3d.items() if f != "Position003"}))
    with pytest.raises(ValueError, match="holds no FOV 'Position003'"):
        fov_ownership(result, "Position003", placement=missing)
    with pytest.raises(ValueError, match="holds no FOV 'Position009'"):
        fov_ownership(result, "Position009", placement=missing)


def test_fov_ownership_refuses_another_placement_and_wrong_types(own, golden_3d):
    placement, result, _ = own
    with pytest.raises(ValueError, match="made from another placement"):
        fov_ownership(result, "FOV1", placement=own_placement(origins={"FOV1": (0.0, 0.0, 0.0),
                                                                       "FOV2": (0.0, 0.0, 12.5)}))
    with pytest.raises(TypeError, match="AssemblyResult"):
        fov_ownership(result.cells, "FOV1", placement=placement)
    with pytest.raises(TypeError, match="PlacementResult"):
        fov_ownership(result, "FOV1", placement=placement.fovs)


# --- The drawing -----------------------------------------------------------------------------------------

def region_edges(region):
    """The pixel edges between a region pixel and another pixel or the outside: ('h', r, c) is the top edge of
    pixel (r, c) (row r = Y is the bottom one), ('v', r, c) the left edge of pixel (r, c) (c = X the right)."""
    padded = np.pad(region, 1)
    edges = set()
    for r in range(region.shape[0] + 1):
        for c in range(region.shape[1]):
            if padded[r, c + 1] != padded[r + 1, c + 1]:
                edges.add(("h", r, c))
    for r in range(region.shape[0]):
        for c in range(region.shape[1] + 1):
            if padded[r + 1, c] != padded[r + 1, c + 1]:
                edges.add(("v", r, c))
    return edges


def drawn_edges(segments):
    """The unit pixel edges along the polylines (corners at half-integer x, y), each once, and every line
    closed and axis-aligned."""
    edges = []
    for line in segments:
        assert len(line) >= 5 and np.array_equal(line[0], line[-1])
        for (x0, y0), (x1, y1) in zip(line[:-1], line[1:]):
            assert x0 == x1 or y0 == y1
            assert (x0 + 0.5) % 1 == 0 and (y0 + 0.5) % 1 == 0
            if y0 == y1:
                edges += [("h", int(y0 + 0.5), int(c)) for c in np.arange(min(x0, x1) + 0.5, max(x0, x1) + 0.5)]
            else:
                edges += [("v", int(r), int(x0 + 0.5)) for r in np.arange(min(y0, y1) + 0.5, max(y0, y1) + 0.5)]
    assert len(edges) == len(set(edges))
    return set(edges)


def check_region(ax, region):
    lines = [c for c in ax.collections if c.get_gid() == "ownership_region"]
    assert len(lines) == 1
    line = lines[0]
    assert line.get_label() == REGION and [tuple(c) for c in line.get_colors()] == [YELLOW]
    assert drawn_edges(line.get_segments()) == region_edges(region)


def scatters(ax):
    return {c.get_label(): c for c in ax.collections if not c.get_label().startswith("_")
            and c.get_gid() != "ownership_region"}


def points(collection):
    return sorted(map(tuple, np.asarray(collection.get_offsets(), np.float64).tolist()))


def legend_texts(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


def stated(ax, what):
    """(retained, drawn) of the x-axis label."""
    match = re.fullmatch(rf"(\d+) of (\d+) {what} retained in the sample", ax.get_xlabel())
    assert match, ax.get_xlabel()
    return int(match[1]), int(match[2])


def check_drawing(result, record, *, view=None, z=None, image=None):
    """Panels 1 to 4 with ownership against the figure without it; returns the drawn cell ids and spot ids."""
    with_ = plot_assignment(result, image=image, view=view, z=z, ownership=record)
    without = plot_assignment(result, image=image, view=view, z=z)
    try:
        cells, mols = result.cells, result.molecules
        territories = result.territories if result.territories is not None else result.cell_labels
        plane = territories.max(axis=0) if z is None and view != "single_layer" else \
            territories[territories.shape[0] // 2 if z is None else z]
        layer = None if z is None and view != "single_layer" else (territories.shape[0] // 2 if z is None else z)
        prefix = "expanded_" if result.territories is not None else ""
        present = set(np.unique(plane).tolist()) - {0}
        drawn_cells = {int(c) for c in cells.cell_id if layer is None or int(c) in present}
        on = mols.assignment_status != "outside_grid"
        if layer is not None:
            on &= mols.voxel_z.eq(layer).fillna(False)
        drawn_mols = set(mols.spot_id[on.to_numpy(bool)])
        centre = {int(c): (float(x), float(y)) for c, x, y in
                  zip(cells.cell_id, cells[f"{prefix}centroid_x"], cells[f"{prefix}centroid_y"])}
        position = {s: (float(x), float(y)) for s, x, y in zip(mols.spot_id, mols.x, mols.y)}
        for k, (drawn, gone, where, what) in enumerate(
                ((drawn_cells, record.excluded_cells, centre, "cells"),
                 (drawn_mols, record.excluded_molecules, position, "molecules"))):
            ax, base = with_.axes[k], without.axes[k]
            check_region(ax, record.region)
            grey = drawn & gone
            got, plain = scatters(ax), scatters(base)
            # The status scatters are those drawn without ownership, less the excluded points, in their colours.
            assert [label for label in got if label != EXCLUDED] == list(plain)
            for label, collection in plain.items():
                kept = [p for p in points(collection) if p not in {where[i] for i in grey}]
                assert points(got[label]) == kept, (label, what)
                assert np.array_equal(got[label].get_facecolors(), collection.get_facecolors()) or not len(kept)
            if grey:
                assert points(got[EXCLUDED]) == sorted(where[i] for i in grey)
                assert all(tuple(c) == GREY for c in got[EXCLUDED].get_facecolors())
            else:
                assert EXCLUDED not in got
            assert legend_texts(ax) == [REGION] + legend_texts(base) + ([EXCLUDED] if grey else [])
            retained, total = stated(ax, what)
            assert ax.get_title() == base.get_title() and base.get_xlabel() == ""
            assert total == sum(len(c.get_offsets()) for c in got.values()) == len(drawn)
            assert retained == total - len(grey)
        for ax, base in zip(with_.axes[2:], without.axes[2:]):
            assert ax.get_title() == base.get_title()
            assert [(p.get_x(), p.get_width(), p.get_height()) for p in ax.patches] == \
                [(p.get_x(), p.get_width(), p.get_height()) for p in base.patches]
            assert not ax.collections and not base.collections
        return drawn_cells, drawn_mols
    finally:
        plt.close(with_)
        plt.close(without)


def test_the_drawing_in_the_z_maximum_view(own):
    placement, result, results = own
    image = np.arange(np.prod(OWN_SHAPE), dtype=np.float64).reshape(OWN_SHAPE)
    # FOV2: excluded cells and molecules in both panels; FOV1: its cell 2 (A), excluded by the default duplicate rule
    # as a second observation of FOV2's B (share 1.0; W-400), with its molecule m1, one more excluded molecule (m5)
    # drawn and one (m6, outside_grid) not drawn.
    for fov, grey_cells, grey_molecules in (("FOV2", {2, 3}, {"m2", "m3", "m6"}), ("FOV1", {2}, {"m1", "m5"})):
        record = fov_ownership(result, fov, placement=placement)
        cells, mols = check_drawing(results[fov], record, image=image)
        assert cells & record.excluded_cells == grey_cells
        assert mols & record.excluded_molecules == grey_molecules


def layer_results():
    """own_cases with FOV2's cell C (label 3, excluded not_owner) cut to the planes 0 and 1: of the excluded
    cells, E2 (label 2) occurs in the middle plane Z // 2 = 2 and C does not."""
    grid = unit_grid(OWN_SHAPE)
    cells = {**OWN_CELLS["FOV2"], 3: ((0, 1), (11, 13), (0, 0))}
    results = own_results()
    results["FOV2"] = assign_molecules(molecules("FOV2", OWN_MOLECULES["FOV2"], OWN_GENES),
                                       label_run(paint(OWN_SHAPE, cells), "FOV2", grid), grid=grid,
                                       config=AssignmentConfig(expansion=ExpandLabelsConfig(1.0, "um", "planar")))
    return results


def test_the_drawing_in_the_single_layer_view():
    results = layer_results()
    placement = own_placement()
    result = assemble(placement, inputs_of(results, {f: unit_grid(OWN_SHAPE) for f in OWN_IDS}))
    record = fov_ownership(result, "FOV2", placement=placement)
    assert record.excluded_cells == {2, 3}
    fov2 = results["FOV2"]
    assert OWN_SHAPE[0] // 2 == 2 and 2 in np.unique(fov2.territories[2]) and 3 not in np.unique(fov2.territories[2])
    # The middle plane by default, and the planes 1 and 0 given with z.
    for z, cells, mols in ((None, {1, 2, 4}, {"m5", "m6"}), (1, {1, 2, 3, 4}, {"m0", "m1", "m2", "m3", "m4"}),
                           (0, {3}, set())):
        drawn_cells, drawn_mols = check_drawing(fov2, record, view="single_layer", z=z)
        assert (drawn_cells, drawn_mols) == (cells, mols), z


def test_refused_inputs(own):
    placement, result, results = own
    fov2 = results["FOV2"]
    record = fov_ownership(result, "FOV2", placement=placement)
    figures = plt.get_fignums()
    with pytest.raises(ValueError, match=r"ownership region Y, X \(16, 15\) differ from the territories' \(16, 16\)"):
        plot_assignment(fov2, ownership=FOVOwnership(record.region[:, :15]))
    with pytest.raises(TypeError, match="ownership must be a FOVOwnership"):
        plot_assignment(fov2, ownership=record.region)
    with pytest.raises(TypeError, match="ownership must be a FOVOwnership"):
        plot_assignment(fov2, ownership=result)
    with pytest.raises(ValueError, match=r"excluded cells the result does not have: \[9\]"):
        plot_assignment(fov2, ownership=FOVOwnership(record.region, {2, 9}))
    with pytest.raises(ValueError, match=r"excluded molecules the result does not have: \['m9'\]"):
        plot_assignment(fov2, ownership=FOVOwnership(record.region, set(), {"m2", "m9"}))
    assert plt.get_fignums() == figures
    for region in (record.region.astype(np.uint8), record.region[0]):
        with pytest.raises(ValueError, match="region must be a Y, X boolean array"):
            FOVOwnership(region)
    with pytest.raises(ValueError, match="excluded_cells must hold cell_id"):
        FOVOwnership(record.region, {True})
    with pytest.raises(ValueError, match="excluded_molecules must hold spot_id"):
        FOVOwnership(record.region, set(), {1})
    with pytest.raises(ValueError, match="excluded_cells must be a collection"):
        FOVOwnership(record.region, "12")
    assert not record.region.flags.writeable
