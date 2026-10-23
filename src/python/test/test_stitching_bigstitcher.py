"""The BigStitcher macro and the import of ``dataset.xml`` (W-366): the import parts of PL3, PL5, PL7, PL11, PL12, PL14.

Rows of docs/stitching-assembly-validation.md. Fixtures, all written in the test (no Fiji
run, no image read, nothing read from a run directory):

* ``hand_records``: the two ``dataset.xml`` files of W-324 ``scripts/parity_fixtures.py``
  (``xml_2x2_blank``, ``xml_3x2_positive``) with their ``grid.csv``, written from its
  constant values, stated here as literals; ``hand_records_setups`` is each text with one
  ``ViewSetup`` per setup inserted after the ``SpimData`` element.
* ``golden_stitching``: the stage file, ``dataset_xml()``, ``GRID_CSV`` and ``TILE_CONFIG``
  of ``test_workflow_stitching_golden.py``, imported from that module (not edited);
  ``golden_stitching_setups`` adds the ``ViewSetups`` as above.
* ``xml_links``: a 2 × 2 grid of 4×16×24 tiles with ``ViewSetups``, fractional transforms,
  a calibration with ``c_z`` 1.7894736842105263 and one ``PairwiseResult`` per neighbour
  pair, and its variants.

The expected legacy CSVs are computed by the frozen helper ``legacy_tile_config`` from the
files without ``ViewSetups``.
"""
import hashlib
import inspect
import json
import math
import re

import numpy as np
import pandas as pd
import pytest
import tifffile

from starfinder.stitching import (BigStitcherConfig, GridSnap, PlacementResult, StageAxisMapping, StitchingInput,
                                  export_bigstitcher, import_bigstitcher, legacy_tile_config_view, load_placement,
                                  positions_from_grid, positions_from_leica, positions_from_table, save_placement)
from starfinder.stitching._bigstitcher import view_map_of

from .stitching_fixtures import LN_SPACING, grid
from .test_workflow_stitching_golden import (GRID_CSV, STITCHING, SUMS, TILE_COL, TILE_CONFIG, TILE_ROW,
                                             VOXEL_SIZE_XY, VOXEL_SIZE_Z, Z_CALIBRATION, legacy_tile_config,
                                             render_macro, stage_file)
from .test_workflow_stitching_golden import dataset_xml as golden_dataset_xml

pytestmark = [pytest.mark.stitching, pytest.mark.contract]

LEGACY_DIR = "/data/root/dataset/output/images/fused/sample1/DAPI"   # render_macro's folder
TILE_CONFIG_COLUMNS = ["x", "y", "z", "start_x", "start_y", "end_x", "end_y",
                       "start_x_norm", "start_y_norm", "end_x_norm", "end_y_norm"]


# ---------------------------------------------------------------------------------------
# hand_records: W-324 parity_fixtures.py, its functions and constants as literals.
def parity_affine(tx, ty, tz, zscale=1.0):
    return f"1.0 0.0 0.0 {tx!r} 0.0 1.0 0.0 {ty!r} 0.0 0.0 {zscale!r} {tz!r}"


def parity_dataset_xml(views):
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<SpimData version="0.2">', "  <ViewRegistrations>"]
    for setup, (st, regular) in enumerate(views):
        lines.append(f'    <ViewRegistration timepoint="0" setup="{setup}">')
        for name, t, zs in (("Stitching Transform", st, 1.0), ("Translation to Regular Grid", regular, 1.0),
                            ("calibration", (0.0, 0.0, 0.0), 1.7894736842105263)):
            lines += ['      <ViewTransform type="affine">', f"        <Name>{name}</Name>",
                      f"        <affine>{parity_affine(*t, zscale=zs)}</affine>", "      </ViewTransform>"]
        lines.append("    </ViewRegistration>")
    lines += ["  </ViewRegistrations>", "</SpimData>", ""]
    return "\n".join(lines)


def parity_grid_csv(cols, rows, ids):
    out = [",col,row,id,grid"]
    k = 0
    for c in range(cols):
        for r in range(rows):
            out.append(f"{k},{c},{r},{ids[k]},tile_{r}_{c}")
            k += 1
    return "\n".join(out) + "\n"


PARITY_STEP = 0.9 * 100   # tiles of 100 × 100 px with 10 % overlap


def xml_2x2_blank_views():
    st = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (2.7, -1.6, -0.6), (-0.4, 3.2, 1.9)]
    return [(st[k], ((k // 2) * PARITY_STEP - PARITY_STEP, (k % 2) * PARITY_STEP - PARITY_STEP, -8.947368421052632))
            for k in range(4)]


def xml_3x2_positive_views():
    return [((0.5 * k, 1.25 * k, 0.75 * (k % 3)), ((k // 2) * PARITY_STEP + 10.0, (k % 2) * PARITY_STEP + 20.0, 3.0))
            for k in range(6)]


HAND_RECORDS = {
    "xml_2x2_blank": (xml_2x2_blank_views, (2, 2, [1, 0, 2, 3])),
    "xml_3x2_positive": (xml_3x2_positive_views, (3, 2, [1, 2, 3, 4, 5, 6])),
}
HAND_SHAPE, HAND_SPACING = (10, 100, 100), (0.34, 0.19, 0.19)


def with_setups(text, n_setups, size_xyz, voxel_xyz):
    """``text`` with one ViewSetup per setup j (tile attribute j) inserted after the SpimData element."""
    size = " ".join(str(v) for v in size_xyz)
    voxel = " ".join(repr(v) for v in voxel_xyz)
    parts = ["  <SequenceDescription>", "    <ViewSetups>"]
    for j in range(n_setups):
        parts += ["      <ViewSetup>", f"        <id>{j}</id>", f"        <name>{j}</name>",
                  f"        <size>{size}</size>",
                  f"        <voxelSize><unit>µm</unit><size>{voxel}</size></voxelSize>",
                  "        <attributes><illumination>0</illumination><channel>0</channel>"
                  f"<tile>{j}</tile><angle>0</angle></attributes>",
                  "      </ViewSetup>"]
    parts.append('      <Attributes name="tile">')
    parts += [f"        <Tile><id>{j}</id><name>{j}</name></Tile>" for j in range(n_setups)]
    parts += ["      </Attributes>", "    </ViewSetups>", "  </SequenceDescription>"]
    head, rest = text.split('<SpimData version="0.2">\n', 1)
    return head + '<SpimData version="0.2">\n' + "\n".join(parts) + "\n" + rest


def view_map_from_grid(grid_text, name=lambda pid: f"Position{pid:03d}"):
    """The view map of a grid.csv: view j, its cell, fov_id for a nonzero id, kind fov or padding."""
    frame = pd.read_csv(pd.io.common.StringIO(grid_text), index_col=0)
    return pd.DataFrame({"view": list(frame.index), "grid_column": list(frame["col"]), "grid_row": list(frame["row"]),
                         "fov_id": [name(pid) if pid else "" for pid in frame["id"]],
                         "kind": ["fov" if pid else "padding" for pid in frame["id"]]})


def table_for_grid(grid_text, spacing_yx, tile_yx, overlap=0.1, name=lambda pid: f"Position{pid:03d}"):
    """A supplied table of the grid's acquired cells, in PositionID order, at the regular-grid positions."""
    frame = pd.read_csv(pd.io.common.StringIO(grid_text), index_col=0)
    frame = frame[frame["id"] != 0].sort_values("id")
    step = [(1 - overlap) * n * s for n, s in zip(tile_yx, spacing_yx)]
    return positions_from_table(pd.DataFrame({"fov_id": [name(p) for p in frame["id"]], "nominal_z": 0.0,
                                              "nominal_y": frame["row"] * step[0],
                                              "nominal_x": frame["col"] * step[1]}))


@pytest.fixture
def hand_records_setups(tmp_path):
    """name -> (dataset.xml with ViewSetups, the file without, grid.csv, table, view map, grids)."""
    out = {}
    for name, (views, (cols, rows, ids)) in HAND_RECORDS.items():
        text, grid_text = parity_dataset_xml(views()), parity_grid_csv(cols, rows, ids)
        plain, setups, grid_csv = (tmp_path / f"{name}.xml", tmp_path / f"{name}_setups.xml",
                                   tmp_path / f"{name}_grid.csv")
        plain.write_text(text)
        setups.write_text(with_setups(text, cols * rows, (100, 100, HAND_SHAPE[0]), (0.19, 0.19, 0.34)))
        grid_csv.write_text(grid_text)
        table = table_for_grid(grid_text, HAND_SPACING[1:], HAND_SHAPE[1:])
        grids = {fov: grid(HAND_SHAPE, HAND_SPACING) for fov in table.fov_ids}
        out[name] = (setups, plain, grid_csv, table, view_map_from_grid(grid_text), grids)
    return out


GOLDEN_SHAPE, GOLDEN_SPACING = (1, TILE_ROW, TILE_COL), (VOXEL_SIZE_Z, VOXEL_SIZE_XY, VOXEL_SIZE_XY)


def golden_table(tmp_path):
    path = tmp_path / "golden.maf"
    path.write_text(stage_file())
    return positions_from_leica(path, stage_axes=StageAxisMapping(x="+x", y="+y"),
                                fov_ids={pid: f"Position{pid}" for pid in range(1, 9)},
                                snap_to_grid=GridSnap(overlap=0.1, tile_shape_yx=(TILE_ROW, TILE_COL),
                                                      spacing_yx=(VOXEL_SIZE_XY, VOXEL_SIZE_XY)))


@pytest.fixture
def golden_stitching_setups(tmp_path):
    """(dataset.xml with ViewSetups, the file without, grid.csv, table, view map, grids) of golden_stitching."""
    plain, setups, grid_csv = tmp_path / "golden.xml", tmp_path / "golden_setups.xml", tmp_path / "grid.csv"
    plain.write_text(golden_dataset_xml())
    setups.write_text(with_setups(golden_dataset_xml(), 9, (TILE_COL, TILE_ROW, 1), (VOXEL_SIZE_XY, VOXEL_SIZE_XY,
                                                                                      VOXEL_SIZE_Z)))
    grid_csv.write_text(GRID_CSV)
    table = golden_table(tmp_path)
    grids = {fov: grid(GOLDEN_SHAPE, GOLDEN_SPACING) for fov in table.fov_ids}
    return setups, plain, grid_csv, table, view_map_from_grid(GRID_CSV, lambda pid: f"Position{pid}"), grids


# ---------------------------------------------------------------------------------------
# xml_links: 2 × 2 grid of 4×16×24 tiles, overlap 0.25 (12 px in y, 18 px in x).
LINK_SHAPE = (4, 16, 24)
LINK_IDS = ("L0", "L1", "L2", "L3")
LINK_ST = {0: (0.0, 0.0, 0.0), 1: (0.375, -0.625, 0.25), 2: (-0.5, 0.125, -0.375), 3: (0.25, 0.75, 0.5)}
LINK_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))   # 4 sides, 2 diagonals
LINK_CORRELATION = {pair: 0.80 + 0.03 * k for k, pair in enumerate(LINK_PAIRS)}


def link_regular(view):
    col, row = divmod(view, 2)
    return (col * 18.0 - 0.0625, row * 12.0 + 0.1875, -0.4473684210526316)


def link_affine(t, linear=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))):
    return " ".join(" ".join(repr(float(v)) for v in (*linear[i], t[i])) for i in range(3))


def links_xml(*, order=(0, 1, 2, 3), rotation=None, calibration_xy=(1.0, 1.0), drop_pairs_of=None,
              linear_off=None, extra=None):
    """The xml_links text; ``order`` permutes the ViewRegistration elements."""
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', '<SpimData version="0.2">']
    body = with_setups('<SpimData version="0.2">\n', 4, (24, 16, 4), (0.19, 0.19, 0.34)).split("\n")[1:-1]
    parts += body + ["  <ViewRegistrations>"]
    for view in order:
        st = LINK_ST[view]
        linear = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        if rotation is not None and view == rotation[0]:
            c, s = math.cos(rotation[1]), math.sin(rotation[1])
            linear = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
        if linear_off is not None and view == linear_off[0]:
            linear = ((1.0 + linear_off[1], linear_off[1], 0.0), (0.0, 1.0, -linear_off[1]), (0.0, 0.0, 1.0))
        transforms = [("Stitching Transform", link_affine(st, linear))]
        if extra is not None and view == extra[0]:
            transforms.insert(0, ("Second Round", link_affine(extra[1])))
        calibration = ((calibration_xy[0], 0.0, 0.0), (0.0, calibration_xy[1], 0.0), (0.0, 0.0, Z_CALIBRATION))
        transforms += [("Translation to Regular Grid", link_affine(link_regular(view))),
                       ("calibration", link_affine((0.0, 0.0, 0.0), calibration))]
        parts.append(f'    <ViewRegistration timepoint="0" setup="{view}">')
        for name, affine in transforms:
            parts += ['      <ViewTransform type="affine">', f"        <Name>{name}</Name>",
                      f"        <affine>{affine}</affine>", "      </ViewTransform>"]
        parts.append("    </ViewRegistration>")
    parts += ["  </ViewRegistrations>", "  <StitchingResults>"]
    for a, b in LINK_PAIRS:
        if drop_pairs_of in (a, b):
            continue
        shift = tuple(vb - va for va, vb in zip(LINK_ST[a], LINK_ST[b]))
        parts += [f'    <PairwiseResult view_setup_a="{a}" timepoint_a="0" view_setup_b="{b}" timepoint_b="0">',
                  f"      <shift>{link_affine(shift)}</shift>",
                  f"      <correlation>{LINK_CORRELATION[(a, b)]!r}</correlation>",
                  f"      <hash>{a}{b}</hash>",
                  "      <overlap_boundingbox>0 0 0 24 16 4</overlap_boundingbox>",
                  "    </PairwiseResult>"]
    parts += ["  </StitchingResults>", "</SpimData>", ""]
    return "\n".join(parts)


def links_table():
    return positions_from_grid(columns=2, rows=2, order="column_major_down_right", tile_shape_yx=LINK_SHAPE[1:],
                               spacing_yx=LN_SPACING[1:], overlap=0.25, fov_ids=list(LINK_IDS))


def links_view_map():
    return pd.DataFrame({"view": [0, 1, 2, 3], "grid_column": [0, 0, 1, 1], "grid_row": [0, 1, 0, 1],
                         "fov_id": list(LINK_IDS), "kind": ["fov"] * 4})


def import_links(tmp_path, name="links.xml", fallback=None, **variant):
    path = tmp_path / name
    path.write_text(links_xml(**variant))
    table = links_table()
    return import_bigstitcher(path, table=table, view_map=links_view_map(),
                              grids={fov: grid(LINK_SHAPE, LN_SPACING) for fov in LINK_IDS},
                              config=BigStitcherConfig(), fallback=fallback, sample="s1")


def link_world(view):
    """The float64 sum of the view's Stitching Transform and Translation to Regular Grid (x, y, z)."""
    return tuple(s + r for s, r in zip(LINK_ST[view], link_regular(view)))


def expected_um(world_xyz, calibration_xyz=(1.0, 1.0, Z_CALIBRATION), spacing=LN_SPACING):
    """(z, y, x) µm: w ⊘ c, then times the reference spacing."""
    return np.array([world_xyz[2] / calibration_xyz[2] * spacing[0], world_xyz[1] / calibration_xyz[1] * spacing[1],
                     world_xyz[0] / calibration_xyz[0] * spacing[2]])


def raw_by_view(result):
    return {v["view"]: tuple(v["raw_translation_xyz"]) for v in result.record["method"]["views"]}


def method_translations(result):
    return result.fovs[[f"method_translation_{a}" for a in "zyx"]].to_numpy()


# ---------------------------------------------------------------------------------------
# PL12: import of the hand-built records.
@pytest.mark.parametrize("name", sorted(HAND_RECORDS))
def test_pl12_hand_records_raw_translations_padding_and_legacy_view(hand_records_setups, name):
    setups, plain, grid_csv, table, view_map, grids = hand_records_setups[name]
    result = import_bigstitcher(setups, table=table, view_map=view_map, grids=grids, config=BigStitcherConfig())
    views, (cols, rows, ids) = HAND_RECORDS[name]
    expected = {j: tuple(s + r for s, r in zip(st, regular)) for j, (st, regular) in enumerate(views())}
    assert raw_by_view(result) == expected
    # One row per acquired FOV, in table order; the raw translation of each in method_translation_* (z, y, x).
    assert list(result.fovs["fov_id"]) == table.fov_ids
    acquired = [j for j, pid in enumerate(ids) if pid]
    assert method_translations(result).tolist() == [list(expected[j][::-1]) for j in
                                                    sorted(acquired, key=lambda j: ids[j])]
    kinds = {v["view"]: (v["kind"], v["fov_id"]) for v in result.record["method"]["views"]}
    blanks = [j for j, pid in enumerate(ids) if not pid]
    assert all(kinds[j] == ("padding", "") for j in blanks)
    assert len(result.fovs) == len(ids) - len(blanks)
    if name == "xml_2x2_blank":
        assert blanks == [1]
    frame, _ = legacy_tile_config(plain, grid_csv, 100, 100)
    view = legacy_tile_config_view(result, img_col=100, img_row=100)
    assert view.to_csv() == frame.to_csv()


def test_pl12_golden_raw_translations_and_legacy_view(golden_stitching_setups):
    setups, plain, grid_csv, table, view_map, grids = golden_stitching_setups
    result = import_bigstitcher(setups, table=table, view_map=view_map, grids=grids, config=BigStitcherConfig())
    assert raw_by_view(result) == SUMS
    assert {j: tuple(s + r for s, r in zip(STITCHING[j], (90.0 * (j // 3) - 90.0, 72.0 * (j % 3) - 72.0,
                                                           -2.6842105263157894))) for j in range(9)} == SUMS
    padding = [v for v in result.record["method"]["views"] if v["kind"] == "padding"]
    assert [(v["view"], v["grid_column"], v["grid_row"], v["fov_id"]) for v in padding] == [(4, 1, 1, "")]
    assert list(result.fovs["fov_id"]) == [f"Position{pid}" for pid in range(1, 9)]
    frame, _ = legacy_tile_config(plain, grid_csv, TILE_COL, TILE_ROW)
    view = legacy_tile_config_view(result, img_col=TILE_COL, img_row=TILE_ROW)
    assert view.to_csv() == frame.to_csv()
    assert [tuple(int(v) for v in row) for row in view[TILE_CONFIG_COLUMNS].itertuples(index=False)] == TILE_CONFIG


def test_pl12_missing_file_and_unmapped_view_raise(hand_records_setups, tmp_path):
    setups, _, _, table, view_map, grids = hand_records_setups["xml_2x2_blank"]
    missing = tmp_path / "absent" / "dataset.xml"
    with pytest.raises(ValueError, match=str(missing)):
        import_bigstitcher(missing, table=table, view_map=view_map, grids=grids, config=BigStitcherConfig())
    with pytest.raises(ValueError, match=r"views 3 have no entry in the view map"):
        import_bigstitcher(setups, table=table, view_map=view_map[view_map["view"] != 3], grids=grids,
                           config=BigStitcherConfig())
    # The W-324 file itself holds no ViewSetup, which the import needs to map a setup to its view.
    plain = hand_records_setups["xml_2x2_blank"][1]
    with pytest.raises(ValueError, match="setup 0 has no ViewSetup"):
        import_bigstitcher(plain, table=table, view_map=view_map, grids=grids, config=BigStitcherConfig())


def test_pl12_legacy_view_needs_an_import(hand_records_setups):
    from starfinder.stitching import placement_from_table
    _, _, _, table, _, grids = hand_records_setups["xml_2x2_blank"]
    with pytest.raises(ValueError, match="view of a BigStitcher import"):
        legacy_tile_config_view(placement_from_table(table, grids, sample="s"), img_col=100, img_row=100)


# ---------------------------------------------------------------------------------------
# PL7: the legacy view of golden_stitching_setups reproduces TILE_CONFIG; the record never uses img_col.
def test_pl7_golden_legacy_view_keeps_img_col_for_the_y_extent(golden_stitching_setups):
    setups, _, _, table, view_map, grids = golden_stitching_setups
    result = import_bigstitcher(setups, table=table, view_map=view_map, grids=grids, config=BigStitcherConfig())
    view = legacy_tile_config_view(result, img_col=TILE_COL, img_row=TILE_ROW)
    assert [tuple(int(v) for v in row) for row in view[TILE_CONFIG_COLUMNS].itertuples(index=False)] == TILE_CONFIG
    # end_y = y + img_col (100), not y + img_row (80), where no lower neighbour exists (grid row 2).
    assert int(view.loc[2, "end_y"]) == int(view.loc[2, "y"]) + TILE_COL
    # img_row only moves the y start: with img_row = img_col the starts change and the ends do not.
    square = legacy_tile_config_view(result, img_col=TILE_COL, img_row=TILE_COL)
    assert list(square["end_y"]) == list(view["end_y"]) and list(square["start_y"]) != list(view["start_y"])
    # The record has no img_col: the import takes none, and every extent comes from the grids (80 × 100 px).
    assert "img_col" not in inspect.signature(import_bigstitcher).parameters
    assert "img_col" not in json.dumps(result.record)
    shapes = {entry["grid"]["shape_zyx"][1:] == [TILE_ROW, TILE_COL] for entry in result.record["inputs"].values()}
    assert shapes == {True}


# ---------------------------------------------------------------------------------------
# PL3 and PL5: composition, views and units.
def test_pl3_composition_links_and_origins(tmp_path):
    result = import_links(tmp_path)
    assert raw_by_view(result) == {v: link_world(v) for v in range(4)}
    assert method_translations(result).tolist() == [list(link_world(v)[::-1]) for v in range(4)]
    assert list(result.fovs["status"]) == ["image_supported"] * 4
    nominal = links_table().nominal_zyx()
    raw = np.array([expected_um(link_world(v)) for v in range(4)])
    correction = raw - nominal
    origin = nominal + correction - correction.mean(axis=0)
    got = result.fovs[[f"origin_{a}" for a in "zyx"]].to_numpy()
    assert np.abs(got - origin).max() <= 1e-9
    assert np.abs(result.fovs[[f"correction_{a}" for a in "zyx"]].to_numpy().mean(axis=0)).max() <= 1e-9
    # Six links, each PairwiseResult's shift in µm as shift_* and its correlation as quality.
    links = result.links
    assert list(zip(links["fov_a"], links["fov_b"])) == [(LINK_IDS[a], LINK_IDS[b]) for a, b in LINK_PAIRS]
    assert list(links["quality"]) == [LINK_CORRELATION[p] for p in LINK_PAIRS]
    assert links["kept"].all() and (links["reason"] == "").all()
    for row, (a, b) in zip(links.itertuples(index=False), LINK_PAIRS):
        shift = expected_um(tuple(vb - va for va, vb in zip(LINK_ST[a], LINK_ST[b])))
        assert np.abs(np.array([row.shift_z, row.shift_y, row.shift_x]) - shift).max() <= 1e-9
        assert row.residual <= 1e-9
    assert list(result.fovs["n_links_kept"]) == [3, 3, 3, 3]
    assert result.record["review"]["state"] == "not_needed"
    assert result.record["method"]["method"] == "bigstitcher"
    assert result.record["method"]["config_type"] == "starfinder.stitching.BigStitcherConfig"


def test_pl3_every_transform_but_calibration_is_composed_newest_first(tmp_path):
    extra = (0.125, -0.25, 0.0625)
    result = import_links(tmp_path, extra=(1, extra))
    assert raw_by_view(result)[1] == tuple(e + s + r for e, s, r in zip(extra, LINK_ST[1], link_regular(1)))
    assert result.record["method"]["views"][1]["transforms"] == ["Second Round", "Stitching Transform",
                                                                  "Translation to Regular Grid"]


def test_pl3_views_map_through_the_tile_attribute(tmp_path):
    reference = import_links(tmp_path)
    permuted = import_links(tmp_path, name="permuted.xml", order=(2, 0, 3, 1))
    pd.testing.assert_frame_equal(permuted.fovs, reference.fovs, check_exact=True)
    pd.testing.assert_frame_equal(permuted.links, reference.links, check_exact=True)
    assert permuted.record["method"]["views"] == reference.record["method"]["views"]
    # Setup ids that differ from the tiles: setup s holds tile (s + 1) % 4; the view is the tile.
    text = links_xml()
    for s in range(4):
        text = text.replace(f"<tile>{s}</tile>", f"<tile>T{(s + 1) % 4}</tile>")
    text = text.replace("<tile>T", "<tile>")
    path = tmp_path / "renumbered.xml"
    path.write_text(text)
    renumbered = import_bigstitcher(path, table=links_table(), view_map=links_view_map(),
                                    grids={fov: grid(LINK_SHAPE, LN_SPACING) for fov in LINK_IDS},
                                    config=BigStitcherConfig(), sample="s1")
    assert {v["view"]: v["setup"] for v in renumbered.record["method"]["views"]} == {1: 0, 2: 1, 3: 2, 0: 3}
    assert raw_by_view(renumbered) == {(s + 1) % 4: link_world(s) for s in range(4)}


def test_pl3_linear_part_tolerance(tmp_path):
    accepted = import_links(tmp_path, name="off.xml", linear_off=(1, 5e-10))
    assert list(accepted.fovs["status"]) == ["image_supported"] * 4
    with pytest.raises(ValueError, match=r"view 2 composes .*Stitching Transform.* off the identity"):
        import_links(tmp_path, name="rotated.xml", rotation=(2, 1e-6))


def test_pl5_units_use_the_calibration_and_the_reference_spacing(tmp_path, golden_stitching_setups):
    # Origins use each FOV's reference spacing, not the macro's 0.19 / 0.34 µm.
    result = import_links(tmp_path)
    raw = np.array([expected_um(link_world(v)) for v in range(4)])
    nominal = links_table().nominal_zyx()
    gauged = result.fovs[[f"origin_{a}" for a in "zyx"]].to_numpy() - nominal
    assert np.abs((gauged - gauged.mean(axis=0)) - ((raw - nominal) - (raw - nominal).mean(axis=0))).max() <= 1e-9
    macro_units = np.array([expected_um(link_world(v), spacing=(0.34, 0.19, 0.19)) for v in range(4)])
    assert np.abs((macro_units - nominal) - (raw - nominal)).max() > 1e-3
    # z: world z (xy-pixel units) / 1.7894736842105263 slices × the z spacing.
    assert result.fovs.loc[3, "method_translation_z"] == link_world(3)[2]
    # The golden file: method translations stay in world units, z in xy-pixel units (SUMS).
    setups, _, _, table, view_map, grids = golden_stitching_setups
    golden = import_bigstitcher(setups, table=table, view_map=view_map, grids=grids, config=BigStitcherConfig(),
                                fallback="nominal")
    acquired = [j for j in range(9) if j != 4]
    assert method_translations(golden).tolist() == [list(SUMS[j][::-1]) for j in acquired]
    with pytest.raises(ValueError, match=r"view 0 has a calibration with c_x 1.0 ≠ c_y 1.25"):
        import_links(tmp_path, name="cxy.xml", calibration_xy=(1.0, 1.25))


# ---------------------------------------------------------------------------------------
# PL11: a view without links (placed from metadata by BigStitcher's second round).
def test_pl11_fallback_in_the_import(tmp_path):
    failed = import_links(tmp_path, drop_pairs_of=3)
    nominal = import_links(tmp_path, name="nominal.xml", drop_pairs_of=3, fallback="nominal")
    for result in (failed, nominal):
        assert list(result.fovs["status"][:3]) == ["image_supported"] * 3
        assert result.fovs.loc[3, "n_links_kept"] == 0
        assert method_translations(result)[3].tolist() == list(link_world(3)[::-1])   # unrounded, world units
        assert result.record["review"]["state"] == "required"
        assert result.record["review"]["reasons"][0] == "unplaced_fovs"
    row = failed.fovs.loc[3]
    assert row["status"] == "failed" and row["flags"] == ""
    assert np.isnan([row[f"origin_{a}"] for a in "zyx"]).all()
    row = nominal.fovs.loc[3]
    assert row["status"] == "metadata_only" and row["flags"] == "fallback"
    assert [row[f"origin_{a}"] for a in "zyx"] == [row[f"nominal_{a}"] for a in "zyx"]
    assert [row[f"correction_{a}"] for a in "zyx"] == [0.0, 0.0, 0.0]
    # The other origins do not depend on the fallback: the gauge reads image_supported FOVs only.
    columns = [f"origin_{a}" for a in "zyx"]
    pd.testing.assert_frame_equal(failed.fovs.loc[:2, columns], nominal.fovs.loc[:2, columns], check_exact=True)
    assert legacy_tile_config_view(failed, img_col=24, img_row=16).to_csv() == \
        legacy_tile_config_view(nominal, img_col=24, img_row=16).to_csv()


# ---------------------------------------------------------------------------------------
# A non-finite number in a view's affine is refused when the file is parsed (W-380).
def with_value(text, setup, transform, position, value):
    """``text`` with entry ``position`` (row-major 3 × 4) of setup ``setup``'s ``transform`` affine replaced."""
    lines = text.split("\n")
    start = lines.index(f'    <ViewRegistration timepoint="0" setup="{setup}">')
    name = lines.index(f"        <Name>{transform}</Name>", start)
    line = lines[name + 1]
    numbers = line.split("<affine>")[1].split("</affine>")[0].split()
    numbers[position] = value
    lines[name + 1] = f"        <affine>{' '.join(numbers)}</affine>"
    return "\n".join(lines)


@pytest.mark.parametrize("fallback", [None, "nominal"])
@pytest.mark.parametrize("value", ["NaN", "inf"])
@pytest.mark.parametrize("position", [3, 10])   # the x translation and the z diagonal entry
@pytest.mark.parametrize("transform", ["Stitching Transform", "Translation to Regular Grid", "calibration"])
def test_a_non_finite_affine_is_refused(tmp_path, monkeypatch, transform, position, value, fallback):
    import starfinder.stitching._bigstitcher as bigstitcher
    path = tmp_path / "dataset.xml"
    path.write_text(with_value(links_xml(), 2, transform, position, value))
    calls = []
    for name in ("_pairwise", "_to_um", "review_placement"):
        real = getattr(bigstitcher, name)
        monkeypatch.setattr(bigstitcher, name, lambda *a, _name=name, _real=real, **k: calls.append(_name) or
                            _real(*a, **k))
    message = rf"{re.escape(str(path))}: setup 2 \(view 2\), transform '{transform}': the affine holds a non-finite"
    with pytest.raises(ValueError, match=message):
        import_bigstitcher(path, table=links_table(), view_map=links_view_map(),
                           grids={fov: grid(LINK_SHAPE, LN_SPACING) for fov in LINK_IDS},
                           config=BigStitcherConfig(), fallback=fallback, sample="s1")
    assert calls == []   # refused while parsing the views, before the links and any placement are built


def test_the_unedited_links_file_imports_after_the_finite_check(tmp_path):
    for fallback in (None, "nominal"):
        result = import_links(tmp_path, name=f"links_{fallback}.xml", fallback=fallback)
        assert list(result.fovs["status"]) == ["image_supported"] * 4
        assert raw_by_view(result) == {v: link_world(v) for v in range(4)}


# ---------------------------------------------------------------------------------------
# The macro, on golden_stitching's table and grids with one acquired FOV without an image.
PINNED = (  # test_bigstitcher_macro of test_workflow_stitching_golden.py, the folder as {folder}
    "path={folder}/*.tif exclude=10 pattern_0=Tiles",
    "voxel_size_x=0.19 voxel_size_y=0.19 voxel_size_z=0.34 voxel_size_unit=µm",
    "grid_type=[Down & Right             ] tiles_x=3 tiles_y=3 tiles_z=1",
    "overlap_x_(%)=10 overlap_y_(%)=10 overlap_z_(%)=10",
    "how_to_load_images=[Re-save as multiresolution HDF5]",
    "subsampling_factors=[{{2,2,2}}, {{4,4,4}}] hdf5_chunk_sizes=[{{16,16,16}}, {{16,16,16}}]",
    "method=[Phase Correlation] downsample_in_x=4 downsample_in_y=4 downsample_in_z=4",
    "filter_by_link_quality min_r=0.7 max_r=1 max_shift_in_x=0",
    "relative=2.5 absolute=3.5",
    "global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles "
    "and iterative dropping of bad links]",
    "fix_group_0-0")
WITHOUT_IMAGE = "Position5"   # grid row 5: column 1, row 2


def golden_inputs(table, spacing=GOLDEN_SPACING):
    return [StitchingInput(fov, None if fov == WITHOUT_IMAGE else np.full(GOLDEN_SHAPE, k + 1, np.uint8),
                           grid(GOLDEN_SHAPE, spacing), "reference_merged") for k, fov in enumerate(table.fov_ids)]


@pytest.mark.parametrize("pad_missing", [False, True])
def test_macro_text_view_map_and_tiles(tmp_path, pad_missing):
    table = golden_table(tmp_path)
    folder = tmp_path / "DAPI"
    text, view_map = export_bigstitcher(table, golden_inputs(table), config=BigStitcherConfig(pad_missing=pad_missing),
                                        tile_dir=folder, save_dir=folder)
    for parameter in PINNED:
        assert parameter.format(folder=folder) in text
    # With the legacy folder and voxel size the text is the legacy script's, byte for byte.
    legacy = render_macro(tmp_path)
    assert text == legacy.replace(LEGACY_DIR, str(folder))
    expected = view_map_from_grid(GRID_CSV, lambda pid: f"Position{pid}")
    expected.loc[5, "kind"] = "placeholder" if pad_missing else "padding"
    expected.loc[5, "fov_id"] = WITHOUT_IMAGE if pad_missing else ""
    pd.testing.assert_frame_equal(view_map, expected.astype({"view": "int64", "grid_column": "int64",
                                                              "grid_row": "int64", "fov_id": "str", "kind": "str"}))
    assert list(view_map.loc[view_map["kind"] == "padding", "fov_id"]) == [""] * (1 if pad_missing else 2)
    # Tiles in column-major tile_{j}.tif order: each FOV's image, a blank of the tile shape elsewhere.
    images = {item.fov_id: item.image for item in golden_inputs(table)}
    assert sorted(p.name for p in folder.iterdir()) == sorted(f"tile_{j}.tif" for j in range(9))
    for row in view_map.itertuples(index=False):
        tile = tifffile.imread(folder / f"tile_{row.view}.tif").reshape(GOLDEN_SHAPE)
        if row.kind == "fov":
            assert np.array_equal(tile, images[row.fov_id])
        else:
            assert tile.dtype == np.uint8 and not tile.any()


def test_macro_voxel_size_is_the_grids_spacing_and_paths_are_arguments(tmp_path):
    """The only differences from the legacy text: the two folders and the voxel size (LN calibration)."""
    table = golden_table(tmp_path)
    tiles, save = tmp_path / "tiles", tmp_path / "local save"
    text, _ = export_bigstitcher(table, golden_inputs(table, LN_SPACING), config=BigStitcherConfig(),
                                 tile_dir=tiles, save_dir=save)
    legacy = render_macro(tmp_path)
    differing = [(a, b) for a, b in zip(legacy.splitlines(), text.splitlines()) if a != b]
    assert len(legacy.splitlines()) == len(text.splitlines())
    assert [b for _, b in differing] == [
        f'"path={tiles}/*.tif exclude=10 pattern_0=Tiles " +',
        '"modify_voxel_size? voxel_size_x=0.194398 voxel_size_y=0.194398 voxel_size_z=0.346255 '
        'voxel_size_unit=µm " +',
        f'"dataset_save_path=[{save}] manual_mipmap_setup " +',
        f'"select=[{save}/dataset.xml] " + ',
        f'"select=[{save}/dataset.xml] " +',
        f'"select=[{save}/dataset.xml] " +']


def test_macro_refuses_a_table_without_cells_and_foreign_tiles(tmp_path):
    table = golden_table(tmp_path)
    supplied = positions_from_table(table.positions[["fov_id", "nominal_z", "nominal_y", "nominal_x"]])
    with pytest.raises(ValueError, match="needs the table's grid_column"):
        export_bigstitcher(supplied, golden_inputs(supplied), config=BigStitcherConfig(), tile_dir=tmp_path / "a",
                           save_dir=tmp_path / "a")
    folder = tmp_path / "b"
    folder.mkdir()
    (folder / "other.tif").write_bytes(b"")
    with pytest.raises(FileExistsError, match="other.tif"):
        export_bigstitcher(table, golden_inputs(table), config=BigStitcherConfig(), tile_dir=folder, save_dir=folder)
    with pytest.raises(ValueError, match="overlap_percent_xyz"):
        BigStitcherConfig(overlap_percent_xyz=(10, 10, 100))


@pytest.mark.parametrize("pad_missing", [False, True])
def test_macro_view_map_imports_the_padded_case(tmp_path, golden_stitching_setups, pad_missing):
    setups, _, _, table, _, grids = golden_stitching_setups
    _, view_map = export_bigstitcher(table, golden_inputs(table), config=BigStitcherConfig(pad_missing=pad_missing),
                                     tile_dir=tmp_path / "DAPI", save_dir=tmp_path / "DAPI")
    result = import_bigstitcher(setups, table=table, view_map=view_map, grids=grids,
                                config=BigStitcherConfig(pad_missing=pad_missing))
    row = result.fovs.set_index("fov_id").loc[WITHOUT_IMAGE]
    assert result.record["review"]["state"] == "required"
    if pad_missing:
        assert row["status"] == "placeholder"
        assert WITHOUT_IMAGE in result.record["review"]["flagged"]["unplaced_fovs"]
        # Its origin is BigStitcher's position of the blank (its regular-grid cell), in µm; no FOV is
        # image_supported here (the file holds no PairwiseResult), so no gauge shift applies.
        origin = expected_um(SUMS[5], spacing=GOLDEN_SPACING)
        assert np.abs(np.array([row[f"origin_{a}"] for a in "zyx"]) - origin).max() <= 1e-9
        assert [row[f"method_translation_{a}"] for a in "zyx"] == list(SUMS[5][::-1])
    else:
        assert row["status"] == "missing" and row["component"] == -1
        assert np.isnan([row[f"origin_{a}"] for a in "zyx"]).all()
    kinds = {v["view"]: v["kind"] for v in result.record["method"]["views"]}
    assert kinds[4] == "padding" and kinds[5] == ("placeholder" if pad_missing else "padding")


# ---------------------------------------------------------------------------------------
# PL14: persistence of an import.
def test_pl14_import_persistence(tmp_path, hand_records_setups):
    setups, _, _, table, view_map, grids = hand_records_setups["xml_2x2_blank"]
    macro = 'run("Define dataset ...", "define_dataset=[Automatic Loader (Bioformats based)]");\n'
    result = import_bigstitcher(setups, table=table, view_map=view_map, grids=grids, config=BigStitcherConfig(),
                                sample="s1", macro=macro)
    folder = save_placement(result, tmp_path / "stitching" / "s1")
    assert sorted(p.name for p in folder.iterdir()) == ["fovs.csv", "links.csv", "placement.json", "view_map.csv"]
    record = json.loads((folder / "placement.json").read_text())
    assert record["tables"]["view_map"]["sha256"] == hashlib.sha256((folder / "view_map.csv").read_bytes()).hexdigest()
    assert record["method"]["macro_sha256"] == hashlib.sha256(macro.encode()).hexdigest()
    assert record["method"]["dataset_xml_sha256"] == hashlib.sha256(setups.read_bytes()).hexdigest()
    assert record["format_version"] == 1 and record["gauge"] == "G1" and record["frame"]["frame_id"] == "sample:s1"
    assert set(record["inputs"]) == set(table.fov_ids)
    assert {"source", "file_sha256", "grid"} <= set(record["inputs"]["Position001"])
    saved = pd.read_csv(folder / "view_map.csv", keep_default_na=False)
    assert list(saved.columns) == ["view", "grid_column", "grid_row", "fov_id", "kind"]
    loaded = load_placement(folder)
    assert isinstance(loaded, PlacementResult)
    pd.testing.assert_frame_equal(loaded.fovs, result.fovs, check_exact=True)
    pd.testing.assert_frame_equal(loaded.links, result.links, check_exact=True)
    assert loaded.record == result.record
    pd.testing.assert_frame_equal(view_map_of(loaded), view_map.astype({"fov_id": "str", "kind": "str"}))
    assert legacy_tile_config_view(loaded, img_col=100, img_row=100).to_csv() == \
        legacy_tile_config_view(result, img_col=100, img_row=100).to_csv()
    (folder / "view_map.csv").write_text((folder / "view_map.csv").read_text().replace("padding", "fov"))
    with pytest.raises(ValueError, match="view_map"):
        load_placement(folder)
