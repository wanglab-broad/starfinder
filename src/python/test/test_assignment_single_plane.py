"""W-390, criterion 4 (§2.9 side): ``assign_molecules`` on a FOV whose reference grid has one plane.

A sample whose tiles are single-plane projections is assigned by y and x only: on a
one-plane molecule grid the z of a molecule is not considered, so no molecule is
``outside_grid`` because of its z, while one outside in y or x still is; the recorded z
stays in the column ``z`` and ``voxel_z`` is the plane, 0. A grid of several planes keeps
its rule. ``assemble`` on such FOVs is ``test_stitching_declared_grids.py``.

Fixture, built here: FOVs of 1×16×16 (unit spacing) with plane cell and nucleus labels
and molecules whose z spans several planes (−3.2 to 29), some outside in y or x; the same
labels on a grid of four planes (their projection) and as a four-plane volume.
"""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from starfinder.assignment import AssignmentConfig, assign_molecules
from starfinder.segmentation import ExpandLabelsConfig

from .assembly_fixtures import label_run, molecules, paint, unit_grid

pytestmark = [pytest.mark.segmentation, pytest.mark.validation]

GENES = ("A", "B", "C")
CELLS = {1: ((0, 0), (2, 6), (2, 6)), 2: ((0, 0), (9, 13), (10, 15)), 3: ((0, 0), (11, 14), (1, 4))}
NUCLEI = {1: ((0, 0), (3, 5), (3, 5)), 2: ((0, 0), (10, 12), (11, 13))}
Z_VALUES = (-3.2, -0.5, 0.0, 0.4, 0.6, 1.0, 2.7, 7.0, 15.5, 29.0)
YX_POINTS = ((4.0, 4.0), (3.4, 2.6), (11.0, 12.0), (12.2, 14.4), (0.0, 0.0), (8.0, 8.0), (12.0, 2.0), (5.0, 10.0),
             (7.0, 6.6), (2.0, 7.4), (15.4, 7.0), (-0.6, 3.0), (16.0, 5.0), (7.0, -0.51), (7.0, 15.5), (-3.0, 20.0))
COLUMNS = ["assignment_status", "cell_id", "compartment", "nucleus_id", "voxel_y", "voxel_x", "in_expansion",
           "original_cell_id"]


def rows(z_of=None):
    """Every (y, x) at each z of Z_VALUES (or at z_of only)."""
    zs = Z_VALUES if z_of is None else (z_of,)
    return [((z, y, x), GENES[k % len(GENES)]) for z in zs for k, (y, x) in enumerate(YX_POINTS)]


def outside_yx(given, shape):
    return np.array([not (0 <= np.floor(y + 0.5) < shape[1] and 0 <= np.floor(x + 0.5) < shape[2])
                     for (_, y, x), _ in given])


def run(shape, given, *, expand=False, volume=False):
    grid = unit_grid(shape)
    planes = shape[0] if volume else 1
    cells = label_run(np.repeat(paint((1, 16, 16), CELLS), planes, axis=0), "FOV1", grid)
    nuclei = label_run(np.repeat(paint((1, 16, 16), NUCLEI), planes, axis=0), "FOV1", grid, target="nucleus",
                       run="nucleus")
    config = AssignmentConfig(expansion=ExpandLabelsConfig(1.0, "um", "planar") if expand else None)
    return assign_molecules(molecules("FOV1", given, GENES), cells, grid=grid, nuclei=nuclei, config=config)


@pytest.mark.parametrize("expand", [False, True])
def test_one_plane_grid_samples_y_and_x_only(expand):
    """Each molecule has the status, cell and compartment it has with its z on the plane, exactly."""
    given = rows()
    result = run((1, 16, 16), given, expand=expand)
    table = result.molecules
    outside = outside_yx(given, (1, 16, 16))
    status = table["assignment_status"].to_numpy(object)
    assert (status[outside] == "outside_grid").all() and not (status[~outside] == "outside_grid").any()
    assert {"assigned", "unassigned"} <= set(status[~outside]) and outside.sum() == 5 * len(Z_VALUES)
    np.testing.assert_array_equal(table["z"].to_numpy(float), [p[0] for p, _ in given])
    assert table["voxel_z"].notna().all() and (table["voxel_z"] == 0).all()
    assert result.record["sampling"]["sampled_axes"] == "yx"
    # Per z: the molecules at that z have exactly the outcome of the same molecules at z 0.
    flat = run((1, 16, 16), rows(z_of=0.0), expand=expand).molecules
    n = len(YX_POINTS)
    for k, z in enumerate(Z_VALUES):
        block = table.iloc[k * n:(k + 1) * n][COLUMNS].reset_index(drop=True)
        assert_frame_equal(block, flat[COLUMNS], check_exact=True, obj=f"z {z}")
    assert result.record["counts"]["outside_grid"] == outside.sum()
    assert result.record["counts"]["assigned"] == len(Z_VALUES) * int(flat["assignment_status"].eq("assigned").sum())


@pytest.mark.parametrize("volume", [False, True])
def test_several_planes_keep_their_rule(volume):
    """On a grid of four planes (plane labels on its projection, or volume labels) z outside [−0.5, 3.5) is
    outside_grid with voxel_z null, as before; inside, plane labels are read at y and x."""
    given = rows()
    result = run((4, 16, 16), given, volume=volume)
    table = result.molecules
    z = np.array([p[0] for p, _ in given])
    off_z = ~((np.floor(z + 0.5) >= 0) & (np.floor(z + 0.5) < 4))
    outside = off_z | outside_yx(given, (4, 16, 16))
    status = table["assignment_status"].to_numpy(object)
    assert (off_z & ~outside_yx(given, (4, 16, 16))).sum() > 0
    assert (status[outside] == "outside_grid").all() and not (status[~outside] == "outside_grid").any()
    assert table["voxel_z"].isna().to_numpy()[off_z].all()
    assert result.record["sampling"]["sampled_axes"] == ("zyx" if volume else "yx")
    plane = run((1, 16, 16), given).molecules
    inside = ~outside
    assert_frame_equal(table[inside][COLUMNS].reset_index(drop=True), plane[inside][COLUMNS].reset_index(drop=True),
                       check_exact=True)
