"""AS1, AS2, AS16 and AS17 of docs/stitching-assembly-validation.md: sample coordinates and ownership.

Fixtures: ``assembly_golden`` (3D and 2D, with an expanded FOV), ``stub_variants``' LN
calibration, ``edge_points``, a 0.25-voxel raster over the golden footprints and
``placeholder_grid`` with its raster (``assembly_fixtures.py``). Every expected owner is
recomputed here by the nearest-centre rule over the owning FOVs, independently of
``ownership_map``.
"""
import numpy as np
import pandas as pd
import pytest

from starfinder.assembly import AssemblyConfig, AssemblyInput, assemble, ownership_map
from starfinder.segmentation import ReferenceGrid
from starfinder.stitching import approve_placement, placed_metadata, placement_from_table, positions_from_table

from .assembly_fixtures import (GOLDEN_IDS, GRID3_CELLS, GRID3_IDS, GRID3_PLACEHOLDERS, GRID3_SHAPE, TILE_ZYX,
                                golden_origins, golden_placement, golden_results, grid3_cell, grid3_placement,
                                grid3_raster, grid3_results, inputs_of, metadata, unit_grid)
from .stitching_fixtures import LN_SPACING

pytestmark = [pytest.mark.stitching, pytest.mark.validation]

NC = AssemblyConfig()
AXES = ("z", "y", "x")


def footprint(origin, shape, spacing):
    """(y0, y1, x0, x1), (cy, cx): the half-open footprint and centre, written out from the contract."""
    oy, ox = origin[1], origin[2]
    return ((oy - spacing[1] / 2, oy + (shape[1] - 0.5) * spacing[1], ox - spacing[2] / 2,
             ox + (shape[2] - 0.5) * spacing[2]),
            (oy + (shape[1] - 1) / 2 * spacing[1], ox + (shape[2] - 1) / 2 * spacing[2]))


def oracle_owner(points, fovs):
    """Nearest-centre owner over fovs (table order: (fov_id, footprint, centre)); first wins a tie."""
    owners = []
    for _, y, x in points:
        best, best_d = None, None
        for fov, (y0, y1, x0, x1), (cy, cx) in fovs:
            if y0 <= y < y1 and x0 <= x < x1:
                d = (y - cy) ** 2 + (x - cx) ** 2
                if best_d is None or d < best_d:
                    best, best_d = fov, d
        owners.append(best)
    return owners


def to_world(world, p):
    """index_to_world per row; NaN rows (no expanded centroid) stay NaN."""
    out = np.full(p.shape, np.nan)
    for i, q in enumerate(p):
        if np.isfinite(q).all():
            out[i] = world.index_to_world(q)
    return out


def golden_fovs():
    origins = golden_origins()
    return [(fov, *footprint(origins[fov], TILE_ZYX, (1.0, 1.0, 1.0))) for fov in GOLDEN_IDS]


def golden_raster():
    fovs = golden_fovs()
    y = np.arange(min(f[1][0] for f in fovs), max(f[1][1] for f in fovs), 0.25)
    x = np.arange(min(f[1][2] for f in fovs), max(f[1][3] for f in fovs), 0.25)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return np.stack([np.zeros(yy.size), yy.ravel(), xx.ravel()], axis=1), y, x


# --- AS1: sample coordinates -----------------------------------------------------------------------------

@pytest.mark.parametrize("dimensions", ["3d", "2d"])
def test_as1_sample_coordinates_equal_index_to_world(dimensions):
    placement = golden_placement()
    results = golden_results(dimensions, expand=("Position003",))
    for centroid in ("territory", "original"):
        result = assemble(placement, inputs_of(results), config=AssemblyConfig(centroid=centroid))
        assert not [c for c in result.cells.columns if c.startswith("global_")]
        assert not [c for c in result.molecules.columns if c.startswith("global_")]
        for fov in GOLDEN_IDS:
            world = placed_metadata(placement, fov)
            molecules = result.molecules[result.molecules.fov_id == fov]
            p = molecules[["z", "y", "x"]].to_numpy()
            expected = to_world(world, p)
            assert np.abs(molecules[["sample_z", "sample_y", "sample_x"]].to_numpy() - expected).max(
                initial=0.0) <= 1e-9
            cells = result.cells[result.cells.fov_id == fov]
            for prefix in ("centroid", "expanded_centroid"):
                p = cells[[f"{prefix}_{a}" for a in AXES]].to_numpy()
                expected = to_world(world, p)
                got = cells[[f"sample_{prefix}_{a}" for a in AXES]].to_numpy()
                if dimensions == "2d":
                    assert np.isnan(got[:, 0]).all()
                    got, expected = got[:, 1:], expected[:, 1:]
                both_nan = np.isnan(got) & np.isnan(expected)
                assert (both_nan | (np.abs(got - expected) <= 1e-9)).all()
                if prefix == "centroid":     # never truncated: the fractional centroids survive
                    assert (got != np.trunc(got)).any() or not len(got)
            expanded = fov == "Position003" and centroid == "territory"
            assert set(cells.ownership_centroid) == {"expanded" if expanded else "original"}
            pair = "sample_expanded_centroid" if expanded else "sample_centroid"
            own = cells[[f"sample_ownership_centroid_{a}" for a in AXES]].to_numpy()
            assert own.tobytes() == cells[[f"{pair}_{a}" for a in AXES]].to_numpy().tobytes()   # bitwise
            if fov != "Position003":
                assert cells[[f"sample_expanded_centroid_{a}" for a in AXES]].isna().all().all()


def test_as1_the_contract_example():
    placement = golden_placement()
    assert golden_origins()["Position002"] == (1.3, 37.4, 0.7)
    point = np.array([1.3, 37.4, 0.7]) + np.array([3.0, 8.0, 20.0]) * np.array([1.0, 1.0, 1.0])
    assert np.abs(point - np.array([4.3, 45.4, 20.7])).max() <= 1e-9
    assert np.abs(np.array(placed_metadata(placement, "Position002").index_to_world((3.0, 8.0, 20.0)))
                  - np.array([4.3, 45.4, 20.7])).max() <= 1e-9


def test_as1_the_ln_calibration_keeps_fractional_positions():
    """stub_variants' LN calibration: a molecules_only sample on (0.346255, 0.194398, 0.194398) µm grids."""
    from .assembly_fixtures import molecules
    grid = ReferenceGrid((4, 16, 24), metadata(LN_SPACING), "declared")
    table = positions_from_table(pd.DataFrame({"fov_id": ["a", "b"], "nominal_z": [0.0, 0.0],
                                               "nominal_y": [0.0, 0.0], "nominal_x": [0.0, 3.5]}))
    origins = {"a": (0.0, 250.5, 100.25), "b": (0.0, 250.5 + 1 / 64, 103.75 - 3 / 64)}
    placement = approve_placement(placement_from_table(table, {"a": grid, "b": grid}, sample="ln", origins=origins),
                                  reviewer="test", note="ln")
    rows = [((2, 10, 0), "A"), ((1.25, 3.5, 7.75), "A"), ((0.0, 15.4, 23.4), "A")]
    inputs = [AssemblyInput(f, grid, molecules=molecules(f, rows, ("A",))) for f in ("a", "b")]
    result = assemble(placement, inputs)
    first = result.molecules.iloc[0]
    assert np.abs(np.array([first.sample_z, first.sample_y, first.sample_x])
                  - np.array([0.69251, 252.44398, 100.25])).max() <= 1e-9
    for fov in ("a", "b"):
        rows_of = result.molecules[result.molecules.fov_id == fov]
        world = placed_metadata(placement, fov)
        expected = np.array([world.index_to_world(q) for q in rows_of[["z", "y", "x"]].to_numpy()])
        assert np.abs(rows_of[["sample_z", "sample_y", "sample_x"]].to_numpy() - expected).max() <= 1e-9


# --- AS2: ownership at edges and ties --------------------------------------------------------------------

def edge_placement(origins, ids=("e1", "e2"), shape=(4, 16, 24), spacing=(0.25, 0.25, 0.25)):
    grid = ReferenceGrid(shape, metadata(spacing), "declared")
    table = positions_from_table(pd.DataFrame({"fov_id": list(ids), "nominal_z": [0.0] * len(ids),
                                               "nominal_y": [origins[f][1] for f in ids],
                                               "nominal_x": [origins[f][2] for f in ids]}))
    return placement_from_table(table, {f: grid for f in ids}, sample="edge", origins=origins), \
        {f: grid for f in ids}


def test_as2_footprint_bounds_are_half_open():
    """edge_points: one FOV of 4×16×24 at a fractional Y and X origin."""
    origin = (0.0, 1.3, -2.7)
    placement, grids = edge_placement({"e1": origin}, ids=("e1",))
    (y0, y1, x0, x1), (cy, cx) = footprint(origin, (4, 16, 24), (0.25, 0.25, 0.25))
    owner = ownership_map(placement, grids, config=NC).owner
    points = [(0, y0, cx), (0, y0 - 1e-9, cx), (0, y1, cx), (0, y1 - 1e-9, cx),
              (0, cy, x0), (0, cy, x0 - 1e-9), (0, cy, x1), (0, cy, x1 - 1e-9)]
    assert list(owner(np.array(points))) == ["e1", None, None, "e1", "e1", None, None, "e1"]


def test_as2_an_exact_tie_goes_to_the_first_fov_in_table_order():
    """edge_points: two FOVs of 4×16×24 at unit spacing and X origins 0 and 10 µm; a point on the bisector."""
    origins = {"e1": (0.0, 0.0, 0.0), "e2": (0.0, 0.0, 10.0)}
    unit = (1.0, 1.0, 1.0)
    _, (_, c1) = footprint(origins["e1"], (4, 16, 24), unit)
    _, (_, c2) = footprint(origins["e2"], (4, 16, 24), unit)
    bisector = (c1 + c2) / 2
    assert (c1, c2, bisector) == (11.5, 21.5, 16.5) and (bisector - c1) ** 2 == (bisector - c2) ** 2
    point = np.array([[0.0, 1.0, bisector]])
    placement, grids = edge_placement(origins, spacing=unit)
    assert ownership_map(placement, grids, config=NC).owner(point).tolist() == ["e1"]
    reversed_placement, grids = edge_placement(origins, ids=("e2", "e1"), spacing=unit)
    assert ownership_map(reversed_placement, grids, config=NC).owner(point).tolist() == ["e2"]
    beside = np.array([[0.0, 1.0, np.nextafter(bisector, 0.0)], [0.0, 1.0, np.nextafter(bisector, 99.0)]])
    assert ownership_map(reversed_placement, grids, config=NC).owner(beside).tolist() == ["e1", "e2"]


def test_as2_the_golden_raster_has_one_owner_per_covered_point():
    placement = golden_placement()
    grids = {fov: unit_grid(TILE_ZYX) for fov in GOLDEN_IDS}
    raster, ys, xs = golden_raster()
    got = ownership_map(placement, grids, config=NC).owner(raster)
    fovs = golden_fovs()
    assert got.tolist() == oracle_owner(raster, fovs)
    covered = np.zeros(len(raster), bool)
    for _, (y0, y1, x0, x1), _ in fovs:
        covered |= (raster[:, 1] >= y0) & (raster[:, 1] < y1) & (raster[:, 2] >= x0) & (raster[:, 2] < x1)
    assert (pd.notna(got) == covered).all()
    owners = got.reshape(len(ys), len(xs))
    # FOV1/FOV3: the analytic bisector lies between x 59.1 and 59.3 on the rows of the fixture's cells
    # (y 8 to 34, docs/assembly-algorithms.md "O1"), and the raster's owners switch at it on every row
    # both footprints cover above FOV2.
    c1, c3 = fovs[0][2], fovs[2][2]
    for i, y in enumerate(ys):
        if not (max(fovs[0][1][0], fovs[2][1][0]) <= y < min(fovs[0][1][1], fovs[2][1][1], fovs[1][1][0])):
            continue
        boundary = (((c3[0] ** 2 + c3[1] ** 2) - (c1[0] ** 2 + c1[1] ** 2)) / 2 - (c3[0] - c1[0]) * y) / (
            c3[1] - c1[1])
        if 8 <= y <= 34:
            assert 59.1 <= boundary <= 59.3
        row = owners[i]
        last_fov1 = xs[[k for k, o in enumerate(row) if o == "Position001"][-1]]
        first_fov3 = xs[[k for k, o in enumerate(row) if o == "Position003"][0]]
        assert last_fov1 < boundary <= first_fov3
    # FOV1/FOV2: the boundary near y 42.2 at x 31.5 (both centres' columns).
    (y1c, x1c), (y2c, x2c) = fovs[0][2], fovs[1][2]
    k = int(np.argmin(np.abs(xs - 31.5)))
    column = owners[:, k]
    switch = ys[[i for i, o in enumerate(column) if o == "Position002"][0]]
    assert 42.0 <= switch <= 42.5
    exact = ((y2c ** 2 - y1c ** 2) + (x2c - 31.5) ** 2 - (x1c - 31.5) ** 2) / (2 * (y2c - y1c))
    assert abs(exact - 42.2) < 0.1


def test_as2_z_never_decides_an_owner():
    placement = golden_placement()
    grids = {fov: unit_grid(TILE_ZYX) for fov in GOLDEN_IDS}
    raster, _, _ = golden_raster()
    owner = ownership_map(placement, grids, config=NC).owner
    moved = raster.copy()
    moved[:, 0] = 1e6
    assert owner(raster).tolist() == owner(moved).tolist()


def test_as2_midpoint_boxes_on_the_golden_fixture():
    placement = golden_placement()
    grids = {fov: unit_grid(TILE_ZYX) for fov in GOLDEN_IDS}
    config = AssemblyConfig(ownership="midpoint_box")
    geometry = ownership_map(placement, grids, config=config).geometry.set_index("fov_id")
    o = golden_origins()
    edge_x = (o["Position001"][2] + 64 * 1.0 + o["Position003"][2]) / 2 - 0.5
    edge_y = (o["Position001"][1] + 48 * 1.0 + o["Position002"][1]) / 2 - 0.5
    assert abs(edge_x - 59.2) < 1e-12 and abs(edge_y - 42.2) < 1e-12
    assert geometry.loc["Position001", "box_x1"] == geometry.loc["Position003", "box_x0"] == edge_x
    assert geometry.loc["Position001", "box_y1"] == geometry.loc["Position002", "box_y0"] == edge_y
    # Towards no neighbour the edge stays at the footprint edge.
    assert geometry.loc["Position002", "box_x1"] == geometry.loc["Position002", "footprint_x1"]
    assert geometry.loc["Position003", "box_y1"] == geometry.loc["Position003", "footprint_y1"]
    for dimensions in ("3d", "2d"):
        results = golden_results(dimensions)
        o1 = assemble(placement, inputs_of(results))
        o2 = assemble(placement, inputs_of(results), config=config)
        assert o1.cells.assembly_status.tolist() == o2.cells.assembly_status.tolist()


def test_as2_midpoint_box_ties_and_its_grid_requirement():
    placement = golden_placement()
    grids = {fov: unit_grid(TILE_ZYX) for fov in GOLDEN_IDS}
    edge_x = ownership_map(placement, grids, config=AssemblyConfig(ownership="midpoint_box")).geometry.box_x1[0]
    point = np.array([[0.0, 10.0, edge_x]])
    half_open = AssemblyConfig(ownership="midpoint_box", tie="half_open")
    assert ownership_map(placement, grids, config=half_open).owner(point).tolist() == ["Position003"]
    table_order = AssemblyConfig(ownership="midpoint_box", tie="table_order")
    assert ownership_map(placement, grids, config=table_order).owner(point).tolist() == ["Position001"]
    supplied, own = edge_placement({"e1": (0.0, 0.0, 0.0), "e2": (0.0, 0.0, 4.0)})
    with pytest.raises(ValueError, match="midpoint_box.*grid cells"):
        ownership_map(supplied, own, config=AssemblyConfig(ownership="midpoint_box"))


# --- AS16: placeholder FOVs own no points ----------------------------------------------------------------

def grid3_fovs(statuses):
    placement = grid3_placement()
    return [(row.fov_id, *footprint((row.origin_z, row.origin_y, row.origin_x), GRID3_SHAPE, (1.0, 1.0, 1.0)))
            for row in placement.fovs.itertuples() if statuses[row.fov_id] == "image_supported"]


def test_as16_placeholders_own_no_point():
    placement = grid3_placement()
    statuses = dict(zip(placement.fovs.fov_id, placement.fovs.status))
    assert {f for f, s in statuses.items() if s == "placeholder"} == set(GRID3_PLACEHOLDERS)
    results = grid3_results()
    grids = {fov: unit_grid(GRID3_SHAPE) for fov in results}
    raster = grid3_raster()
    owner = ownership_map(placement, grids, config=NC).owner(raster)
    assert not set(owner) & set(GRID3_PLACEHOLDERS)
    owning = grid3_fovs(statuses)
    assert owner.tolist() == oracle_owner(raster, owning)
    covered = np.zeros(len(raster), bool)
    for _, (y0, y1, x0, x1), _ in owning:
        covered |= (raster[:, 1] >= y0) & (raster[:, 1] < y1) & (raster[:, 2] >= x0) & (raster[:, 2] < x1)
    assert (pd.notna(owner) == covered).all()
    # The middle of each placeholder's footprint, and its strip along the grid's border, have no owner.
    for fov in GRID3_PLACEHOLDERS:
        row = placement.fovs.set_index("fov_id").loc[fov]
        (y0, y1, x0, x1), _ = footprint((row.origin_z, row.origin_y, row.origin_x), GRID3_SHAPE, (1.0,) * 3)
        inside = (raster[:, 1] >= y0) & (raster[:, 1] < y1) & (raster[:, 2] >= x0) & (raster[:, 2] < x1)
        assert (pd.isna(owner[inside]) == ~covered[inside]).all()
        middle = inside & (np.abs(raster[:, 1] - (y0 + y1) / 2) < 2) & (np.abs(raster[:, 2] - (x0 + x1) / 2) < 2)
        assert middle.any() and pd.isna(owner[middle]).all()
    corner = (raster[:, 1] < 11.5) & (raster[:, 2] < 11.5)        # G0: the strip along both outer edges
    edge = (raster[:, 1] >= 27.5) & (raster[:, 2] >= 15.5) & (raster[:, 2] < 23.5)   # G5: along the bottom
    assert corner.any() and pd.isna(owner[corner]).all() and edge.any() and pd.isna(owner[edge]).all()


def test_as16_the_neighbours_cells_in_a_placeholders_footprint_are_retained():
    placement = grid3_placement()
    results = grid3_results()
    grids = {fov: unit_grid(GRID3_SHAPE) for fov in results}
    result = assemble(placement, inputs_of(results, grids))
    cells = result.cells.set_index("fov_id")
    assert sorted(grid3_cell(f) for f in cells.index) == sorted(GRID3_CELLS)
    fovs = result.fovs.set_index("fov_id")
    for fov in cells.index:
        row = cells.loc[fov]
        assert (row.assembly_status, row.owner_fov_id) == ("retained", fov)
        assert not {"conflict", "unclaimed"} & set(row.assembly_flags.split(";"))
        point = (row.sample_ownership_centroid_y, row.sample_ownership_centroid_x)
        assert any(fovs.loc[p, "footprint_y0"] <= point[0] < fovs.loc[p, "footprint_y1"]
                   and fovs.loc[p, "footprint_x0"] <= point[1] < fovs.loc[p, "footprint_x1"]
                   for p in GRID3_PLACEHOLDERS)
    assert fovs.loc[list(GRID3_PLACEHOLDERS), "owns_points"].tolist() == [False, False]
    assert fovs.drop(index=list(GRID3_PLACEHOLDERS)).owns_points.all()
    assert fovs.loc[list(GRID3_PLACEHOLDERS), ["footprint_y0", "footprint_x0", "centre_y", "centre_x"]].notna(
        ).all().all()
    assert result.fovs.kind.isna().tolist() == [f in GRID3_PLACEHOLDERS for f in GRID3_IDS]
    record = {entry["fov_id"]: entry for entry in result.record["fovs"]}
    assert [record[f]["owns_points"] for f in GRID3_PLACEHOLDERS] == [False, False]
    assert all(record[f]["footprint_yx"] is not None for f in GRID3_PLACEHOLDERS)


def test_as16_midpoint_box_edges_towards_a_placeholder_stay_at_the_footprint():
    placement = grid3_placement()
    grids = {fov: unit_grid(GRID3_SHAPE) for fov in GRID3_IDS if fov not in GRID3_PLACEHOLDERS}
    geometry = ownership_map(placement, grids, config=AssemblyConfig(ownership="midpoint_box")).geometry
    geometry = geometry.set_index("fov_id")
    at = {grid3_cell(f): f for f in GRID3_IDS}
    # G3 (row 0, column 1): its left neighbour is the placeholder G0, its lower one G4.
    assert geometry.loc[at[(0, 1)], "box_x0"] == geometry.loc[at[(0, 1)], "footprint_x0"]
    assert geometry.loc[at[(0, 1)], "box_y1"] < geometry.loc[at[(0, 1)], "footprint_y1"]
    # G4 (row 1, column 1): its lower neighbour is the placeholder G5.
    assert geometry.loc[at[(1, 1)], "box_y1"] == geometry.loc[at[(1, 1)], "footprint_y1"]
    assert geometry.loc[at[(1, 1)], "box_x0"] > geometry.loc[at[(1, 1)], "footprint_x0"]
    assert geometry.loc[list(GRID3_PLACEHOLDERS), "box_x0"].isna().all()


def test_as16_a_placement_of_placeholders_only_raises():
    placement = grid3_placement(all_placeholder=True)
    grids = {fov: unit_grid(GRID3_SHAPE) for fov in GRID3_IDS}
    with pytest.raises(ValueError, match="no FOV of the placement.*owns a point"):
        ownership_map(placement, grids, config=NC)
    with pytest.raises(ValueError, match="no FOV of the placement.*owns a point"):
        assemble(placement, [])


# --- AS17: candidates ------------------------------------------------------------------------------------

def test_as17_candidates_on_the_golden_raster():
    placement = golden_placement()
    grids = {fov: unit_grid(TILE_ZYX) for fov in GOLDEN_IDS}
    raster, _, _ = golden_raster()
    fovs = golden_fovs()
    subsets = [[f] for f in GOLDEN_IDS] + [[a, b] for i, a in enumerate(GOLDEN_IDS) for b in GOLDEN_IDS[i + 1:]]
    for subset in subsets + [GOLDEN_IDS]:
        got = ownership_map(placement, grids, config=NC, candidates=subset).owner(raster)
        assert got.tolist() == oracle_owner(raster, [f for f in fovs if f[0] in subset]), subset
    default = ownership_map(placement, grids, config=NC).owner(raster)
    every = ownership_map(placement, grids, config=NC, candidates=set(GOLDEN_IDS)).owner(raster)
    assert default.tolist() == every.tolist()


def test_as17_candidates_on_the_placeholder_grid_and_errors():
    placement = grid3_placement()
    grids = {fov: unit_grid(GRID3_SHAPE) for fov in GRID3_IDS if fov not in GRID3_PLACEHOLDERS}
    raster = grid3_raster()
    statuses = dict(zip(placement.fovs.fov_id, placement.fovs.status))
    owning = grid3_fovs(statuses)
    for subset in (["G1"], ["G4", "G8"], [f for f, *_ in owning]):
        got = ownership_map(placement, grids, config=NC, candidates=subset).owner(raster)
        assert got.tolist() == oracle_owner(raster, [f for f in owning if f[0] in subset])
    with pytest.raises(ValueError, match="empty"):
        ownership_map(placement, grids, config=NC, candidates=[])
    with pytest.raises(ValueError, match=r"G0 \(a placeholder\)"):
        ownership_map(placement, grids, config=NC, candidates=["G1", "G0"])
    with pytest.raises(ValueError, match=r"G99 \(not in the placement\)"):
        ownership_map(placement, grids, config=NC, candidates=["G99"])
    without = {f: g for f, g in grids.items() if f != "G4"}
    with pytest.raises(ValueError, match=r"G4 \(no grid in grids\)"):
        ownership_map(placement, without, config=NC, candidates=["G4"])
    missing = golden_placement(statuses={"Position003": "missing"})
    golden_grids = {fov: unit_grid(TILE_ZYX) for fov in GOLDEN_IDS}
    with pytest.raises(ValueError, match=r"Position003 \(not placed: missing\)"):
        ownership_map(missing, golden_grids, config=NC, candidates=["Position003"])


def test_ownership_map_records_the_geometry():
    placement = golden_placement()
    grids = {fov: unit_grid(TILE_ZYX) for fov in GOLDEN_IDS}
    geometry = ownership_map(placement, grids, config=NC).geometry
    for (fov, box, centre), row in zip(golden_fovs(), geometry.itertuples()):
        assert row.fov_id == fov and (row.footprint_y0, row.footprint_y1, row.footprint_x0,
                                      row.footprint_x1) == box
        assert (row.centre_y, row.centre_x) == centre and row.owns_points and row.candidate
