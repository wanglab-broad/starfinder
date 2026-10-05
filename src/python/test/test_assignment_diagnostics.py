"""The §2.9 assignment diagnostics: the four-panel figure of plot_assignment, its two views, and the
exclusion totals of summarize_assignment (W-332; docs/assignment-contract.md, "Diagnostics").

The figure is checked by its artists (axes, titles, colours, point counts), not by pixels. The
fixture is ``boxes`` with nuclei (exclusion on: cells 7 and 8 excluded; and off), its molecules
spread over the planes 1 to 6 of the cells so that the single-layer view selects a subset; every
expected count is taken from the stated geometry of test_assignment.py.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import rgb_to_hsv  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

from starfinder.assignment import (AssignmentConfig, assign_molecules, plot_assignment,  # noqa: E402
                                   summarize_assignment)
from starfinder.segmentation import ExpandLabelsConfig, ReferenceGrid  # noqa: E402

from .segmentation_fixtures import BOXES_METADATA, BOXES_SHAPE, boxes, seeded_stain  # noqa: E402
from .test_assignment import (BOXES_MOLECULES, GRID, boxes_inputs, expected_status, label_run,  # noqa: E402
                              molecules)

pytestmark = [pytest.mark.segmentation, pytest.mark.validation]

EXCLUDED = (7, 8)  # the cells without a nucleus, excluded by default
# The molecules of boxes, the ones on the grid moved to z = 1 + i % 6: inside every cell box (z 1 to 6),
# so each keeps its cell and its status; the three off the grid stay off it.
SPREAD = tuple(((1 + i % 6, y, x) if cell is not None else (z, y, x), gene, cell)
               for i, ((z, y, x), gene, cell, _) in enumerate(BOXES_MOLECULES))
STAIN = seeded_stain()
# Hue ranges (matplotlib HSV, 0 to 1) of the colours the figure promises.
HUES = {"green": (0.25, 0.42), "red": (-0.03, 0.03), "blue": (0.55, 0.70), "orange": (0.05, 0.13)}
STATUS_HUES = {"assigned": "blue", "unassigned": "red", "excluded_cell": "orange"}


def spread_result(**config):
    _, cells, nuclei, grid = boxes_inputs()
    given = molecules([(p, g) for p, g, _ in SPREAD])
    return assign_molecules(given, cells, grid=grid, nuclei=nuclei, config=AssignmentConfig(**config))


@pytest.fixture(scope="module", params=["on", "off"])
def result(request):
    return spread_result() if request.param == "on" else spread_result(exclude_cells_without_nucleus=False)


def excluded_of(result):
    return EXCLUDED if result.record["config"]["exclude_cells_without_nucleus"] else ()


def hue_is(colour, name):
    """Whether an RGBA colour is a saturated colour of the named hue."""
    h, s, v = rgb_to_hsv(np.asarray(colour[:3], np.float64))
    h = h - 1.0 if h > 0.5 and name == "red" else h
    low, high = HUES[name]
    return low <= h <= high and s > 0.6 and v > 0.6


def by_label(ax):
    return {c.get_label(): c for c in ax.collections}


def images_by_label(ax):
    return {i.get_label(): i for i in ax.get_images()}


def expected_points(result, plane=None):
    """status -> sorted (x, y) of the molecules on the grid (in plane ``plane`` when given), from SPREAD."""
    points = {}
    for (z, y, x), _, cell in SPREAD:
        if cell is None or (plane is not None and z != plane):
            continue
        points.setdefault(expected_status(cell, excluded_of(result)), []).append((float(x), float(y)))
    return {s: sorted(p) for s, p in points.items()}


def drawn_points(collection):
    return sorted(map(tuple, np.asarray(collection.get_offsets(), np.float64).tolist()))


@pytest.mark.parametrize("view", ["z_max", "single_layer"])
def test_four_panels_their_titles_and_colours(result, view):
    figure = plot_assignment(result, image=STAIN, view=view)
    where = "Z maximum" if view == "z_max" else "z = 4"
    assert [ax.get_title() for ax in figure.axes] == [f"cell centres, {where}", f"molecules, {where}",
                                                      "voxels per cell", "molecules per cell"]
    plane = result.cell_labels.max(axis=0) if view == "z_max" else result.cell_labels[4]
    for ax in figure.axes[:2]:
        images = images_by_label(ax)
        assert list(images) == ["image", "outlines"]
        assert images["image"].get_cmap().name == "gray"
        assert np.array_equal(images["image"].get_array(), STAIN.max(axis=0) if view == "z_max" else STAIN[4])
        outline = images["outlines"]
        assert np.array_equal(~np.ma.getmaskarray(outline.get_array()), find_boundaries(plane, mode="inner"))
        assert hue_is(outline.get_cmap()(1.0), "green") and hue_is(outline.get_cmap()(0.0), "green")
    centres = by_label(figure.axes[0])
    assert list(centres) == ["cell centres"]
    assert all(hue_is(c, "red") for c in centres["cell centres"].get_facecolors())
    statuses = ["assigned", "unassigned"] + (["excluded_cell"] if excluded_of(result) else [])
    scatters = by_label(figure.axes[1])
    assert list(scatters) == statuses
    for status in statuses:
        assert all(hue_is(c, STATUS_HUES[status]) for c in scatters[status].get_facecolors())
    assert [t.get_text() for t in figure.axes[1].get_legend().get_texts()] == statuses
    assert figure.axes[0].get_legend() is None
    plt.close(figure)


def test_counts_drawn_in_each_view(result):
    cells = result.cells
    # Every territory has a centre: the kept cells plus the excluded ones.
    assert int(cells.status.eq("kept").sum()) + len(excluded_of(result)) == len(cells) == 8
    for view, plane in (("z_max", None), ("single_layer", 4), ("single_layer", 1), ("single_layer", 6)):
        figure = plot_assignment(result, image=STAIN, view=view, z=plane)
        centres = by_label(figure.axes[0])["cell centres"]
        assert len(centres.get_offsets()) == 8
        assert drawn_points(centres) == sorted(zip(cells.centroid_x, cells.centroid_y))
        expected = expected_points(result, plane)
        scatters = by_label(figure.axes[1])
        for status, collection in scatters.items():
            assert drawn_points(collection) == expected.get(status, []), (view, plane, status)
        assert "outside_grid" not in scatters
        assert sum(len(c.get_offsets()) for c in scatters.values()) == sum(map(len, expected.values()))
        plt.close(figure)
    # In the Z-maximum view: every molecule of each status with a position on the grid.
    statuses = result.molecules.assignment_status
    figure = plot_assignment(result)
    for status, collection in by_label(figure.axes[1]).items():
        assert len(collection.get_offsets()) == int(statuses.eq(status).sum())
    plt.close(figure)


def test_a_plane_outside_the_territories_draws_no_centre_and_no_molecule():
    figure = plot_assignment(spread_result(), view="single_layer", z=0)
    assert len(by_label(figure.axes[0])["cell centres"].get_offsets()) == 0
    assert all(len(c.get_offsets()) == 0 for c in by_label(figure.axes[1]).values())
    plt.close(figure)


def test_the_histograms_cover_every_cell():
    result = spread_result()
    figure = plot_assignment(result)
    for ax, column in ((figure.axes[2], "size_voxels"), (figure.axes[3], "n_molecules")):
        heights = [p.get_height() for p in ax.patches]
        assert sum(heights) == len(result.cells) == 8
        assert ax.patches[0].get_x() == pytest.approx(result.cells[column].min())
    plt.close(figure)


def test_the_territories_and_centres_are_the_expanded_ones_after_an_expansion():
    result = spread_result(expansion=ExpandLabelsConfig(1, "pixel", "planar"), legacy_pixel_expansion=True)
    assert result.territories is not None and not np.array_equal(result.territories, result.cell_labels)
    figure = plot_assignment(result, view="single_layer")
    outline = images_by_label(figure.axes[0])["outlines"].get_array()
    assert np.array_equal(~np.ma.getmaskarray(outline), find_boundaries(result.territories[4], mode="inner"))
    assert drawn_points(by_label(figure.axes[0])["cell centres"]) == \
        sorted(zip(result.cells.expanded_centroid_x, result.cells.expanded_centroid_y))
    sizes = result.cells.expanded_size_voxels.to_numpy(np.float64)
    assert figure.axes[2].patches[0].get_x() == pytest.approx(sizes.min())
    plt.close(figure)


Z1_GRID = ReferenceGrid((1, *BOXES_SHAPE[1:]), BOXES_METADATA, "fov:round1")


def plane_result(kind):
    """boxes' plane z = 4 as plane labels: on the projection of the 8-plane grid (molecules at z = i % 8) or on
    a Z=1 grid (molecules at z = 0); plus one molecule off the grid."""
    cells, _ = boxes()
    if kind == "projected":
        run = label_run(cells[4:5], "cell", "cell", grid=GRID.projected(),
                        input={"projection": {"axis": "z", "method": "max"}})
        grid, n_z = GRID, BOXES_SHAPE[0]
    else:
        run, grid, n_z = label_run(cells[4:5], "cell", "cell", grid=Z1_GRID), Z1_GRID, 1
    rows = [((i % n_z, y, x), g) for i, ((_, y, x), g, c, _) in enumerate(BOXES_MOLECULES) if c is not None]
    return assign_molecules(molecules(rows + [((0, -1, 4), "A")]), run, grid=grid)


@pytest.mark.parametrize("grid", ["projected", "z1"])
def test_a_plane_or_z1_result_has_the_z_maximum_view_only(grid):
    result = plane_result(grid)
    assert result.cell_labels.shape[0] == 1
    figure = plot_assignment(result, image=STAIN[4])
    statuses = result.molecules.assignment_status
    assert int(statuses.eq("outside_grid").sum()) == 1
    for status, collection in by_label(figure.axes[1]).items():
        assert len(collection.get_offsets()) == int(statuses.eq(status).sum())
    assert len(by_label(figure.axes[0])["cell centres"].get_offsets()) == len(result.cells) == 8
    plt.close(figure)
    for options in ({"view": "single_layer"}, {"z": 0}, {"view": "single_layer", "z": 0}):
        with pytest.raises(ValueError, match="single_layer view needs territories with Z > 1"):
            plot_assignment(result, **options)


def test_a_volume_has_both_views():
    result = spread_result()
    for view in ("z_max", "single_layer"):
        plt.close(plot_assignment(result, view=view))
    figure = plot_assignment(result, z=2)  # z alone selects the single-layer view
    assert figure.axes[0].get_title() == "cell centres, z = 2"
    plt.close(figure)


def test_plot_errors():
    result = spread_result()
    with pytest.raises(ValueError, match="view must be one of"):
        plot_assignment(result, view="middle")
    with pytest.raises(ValueError, match="the z_max view takes none"):
        plot_assignment(result, view="z_max", z=4)
    for z in (8, -1, True, 2.0):
        with pytest.raises(ValueError, match="z must be a plane"):
            plot_assignment(result, z=z)
    with pytest.raises(ValueError, match="image Z 3 differs"):
        plot_assignment(result, image=STAIN[:3], view="single_layer")
    with pytest.raises(ValueError, match="image Y, X"):
        plot_assignment(result, image=STAIN[:, :16])
    with pytest.raises(TypeError, match="AssignmentResult"):
        plot_assignment(result.cells)
    # A YX image is drawn as is in both views; a Z=1 image as its one plane.
    for image in (STAIN[2], STAIN[2:3]):
        for view in ("z_max", "single_layer"):
            figure = plot_assignment(result, image=image, view=view)
            assert np.array_equal(images_by_label(figure.axes[0])["image"].get_array(), STAIN[2])
            plt.close(figure)


# --- summarize_assignment: the exclusion totals ----------------------------------------------------------

def test_exclusion_totals_before_and_after_have_the_same_keys():
    """The three results of row A11: exclusion on, off, and no nuclei."""
    mols, cells, nuclei, grid = boxes_inputs()
    default = assign_molecules(mols, cells, grid=grid, nuclei=nuclei)
    off = assign_molecules(mols, cells, grid=grid, nuclei=nuclei,
                           config=AssignmentConfig(exclude_cells_without_nucleus=False))
    alone = assign_molecules(mols, cells, grid=grid)
    keys = {"cells", "molecules", "whole", "nucleus", "cytoplasm"}
    for result in (default, off, alone):
        exclusion = summarize_assignment(result)["exclusion"]
        before, after = exclusion["before"], exclusion["after"]
        assert set(before) == set(after) == keys
        assert (before["nucleus"], before["cytoplasm"]) == (after["nucleus"], after["cytoplasm"])
        if result is not default:
            assert before == after
    before, after = (summarize_assignment(default)["exclusion"][k] for k in ("before", "after"))
    # Cells 7 and 8 and their six molecules (4 + 2) are the difference.
    assert (before["cells"] - after["cells"], before["molecules"] - after["molecules"],
            before["whole"] - after["whole"]) == (2, 6, 6)
    assert after["nucleus"] + after["cytoplasm"] == 10 and summarize_assignment(alone)["exclusion"]["before"][
        "nucleus"] == 0
    doc = " ".join(summarize_assignment.__doc__.split())
    assert "each with the same five keys" in doc
    assert "``nucleus`` and ``cytoplasm`` are equal before and after" in doc
    assert "without an excluded cell, ``before`` equals ``after``" in doc
