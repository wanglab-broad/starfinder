"""What ``Dataset.stitch`` reads: the saved stitching images and the workflow's ``positions`` mapping.

design/specs/stitching.md ("Coordination on the dataset", "Input image", "Workflow
configuration").
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from starfinder.image import IncompatibleGeometryError
from starfinder.stitching._positions import (GridSnap, StageAxisMapping, TilePositionTable, _names, _stage_points,
                                             positions_from_grid, positions_from_leica, positions_from_table)
from starfinder.stitching._stitch import StitchingInput

INPUTS = ("reference_merged", "reference_stain")
POSITION_KEYS = ("source", "stage_axes", "snap_to_grid", "regular_grid", "tile_shape_yx", "spacing_yx", "table",
                 "unit", "stage_file")
POSITION_SOURCES = ("stage_file", "regular_grid", "table")


def _check_input(name):
    if not isinstance(name, str) or not (name in INPUTS or (name.startswith("file:") and len(name) > 5)):
        raise ValueError(f"input must be 'reference_merged', 'reference_stain' or 'file:<pattern>'; got {name!r}")


def read_inputs(output_root, fov_ids, input_name, *, reference) -> dict[str, StitchingInput]:
    """One StitchingInput per FOV, read one FOV at a time, keyed in fov_ids order.

    The grid is that of the FOV's reference image, as
    :func:`starfinder.dataset._declared.reference_grid` decides it with the
    keyword arguments ``reference`` (``reference_settings`` of the Dataset or
    the workflow configuration): the file ``reference_image_import`` or
    ``reference_image`` names (default ``fovs/{fovID}/reference.tif``), its
    stored metadata, or the grid the
    voxel sizes declare for a file that stores none. With ``reference_merged``
    that file is read once and is also the image; with another input only its
    grid is read (:func:`~starfinder.dataset._declared.reference_grid`: the
    header of a file that stores none).
    """
    from starfinder.dataset._declared import reference_grid, reference_volume
    from starfinder.dataset._export import reference_stain_path
    from starfinder.io import load_volume, load_volume_zyxc
    from starfinder._records import file_sha256
    _check_input(input_name)
    output_root = Path(output_root)
    loaded = {}
    for fov in fov_ids:
        if input_name == "reference_merged":
            grid, volume, reference_path = reference_volume(output_root, fov, **reference)
            loaded[fov] = StitchingInput(fov, volume.image, grid, input_name, file_sha256(reference_path))
            continue
        grid = reference_grid(output_root, fov, **reference)
        if input_name == "reference_stain":
            path, source = reference_stain_path(output_root, fov), input_name
        else:
            path = Path(input_name[len("file:"):].replace("{fovID}", fov))
            path = path if path.is_absolute() else output_root / path
            source = f"file:{path.resolve()}"
        if not path.is_file():
            loaded[fov] = StitchingInput(fov, None, grid, source)
            continue
        if input_name == "reference_stain":
            stain = load_volume_zyxc(path)
            if stain.image.shape[3] != 1:
                raise ValueError(f"the saved reference_stain of FOV {fov!r} has {stain.image.shape[3]} channels; "
                                 "stitching reads a one-channel stain")
            if stain.image.shape[:3] != grid.shape_zyx or stain.metadata != grid.metadata:
                raise IncompatibleGeometryError(
                    f"the reference_stain of FOV {fov!r} ({path}) is not on its reference grid: shape "
                    f"{stain.image.shape[:3]} and metadata {stain.metadata!r}, the grid {grid.shape_zyx} and "
                    f"{grid.metadata!r}")
            image = stain.image[..., 0]
        else:
            image = load_volume(path).image
            if image.shape != grid.shape_zyx:
                raise IncompatibleGeometryError(f"the stitching file of FOV {fov!r} ({path}) has shape {image.shape}; "
                                                f"its reference grid {grid.shape_zyx}")
        loaded[fov] = StitchingInput(fov, image, grid, source, file_sha256(path))
    return loaded


def _shared(grids, what, value):
    groups = {}
    for fov, grid in grids.items():
        groups.setdefault(value(grid), []).append(fov)
    if None in groups:
        raise IncompatibleGeometryError(f"{what} cannot be taken from the reference grids: FOVs "
                                        f"{_names(groups[None])} have unknown spacing; give {what} explicitly")
    if len(groups) > 1:
        described = "; ".join(f"FOVs {_names(fovs)}: {key!r}" for key, fovs in groups.items())
        raise IncompatibleGeometryError(f"the FOVs do not share one {what} ({described}); give {what} explicitly")
    return next(iter(groups))


def _geometry(positions, grids):
    """tile_shape_yx and spacing_yx from the mapping, or else from the grids, with where each came from."""
    tile, spacing, origin = positions.get("tile_shape_yx"), positions.get("spacing_yx"), {}
    origin["tile_shape_yx"] = "configuration" if tile is not None else "reference_grids"
    origin["spacing_yx"] = "configuration" if spacing is not None else "reference_grids"
    if tile is None:
        tile = _shared(grids, "tile_shape_yx", lambda g: tuple(g.shape_zyx[1:]))
    if spacing is None:
        spacing = _shared(grids, "spacing_yx",
                          lambda g: None if g.metadata.spacing_zyx is None else tuple(g.metadata.spacing_zyx[1:]))
    return tuple(tile), tuple(spacing), origin


def _section(positions, key, required):
    section = positions.get(key)
    if not isinstance(section, Mapping):
        raise ValueError(f"positions.{key} must be a mapping with {', '.join(required)}")
    missing = [k for k in required if k not in section]
    unknown = [k for k in section if k not in required]
    if missing or unknown:
        raise ValueError(f"positions.{key} needs exactly the keys {list(required)}; missing {missing}, "
                         f"unknown {unknown}")
    return section


def _selected_positions(path, fov_ids, fov_pattern, position_ids):
    """``PositionID`` → FOV ID of the FOVs of fov_ids: by position_ids, or else the stage file's PositionIDs whose
    ``fov_pattern.format(i=p)`` is one of them (in file order; none when the file is absent, which the converter
    reports)."""
    wanted = list(fov_ids)
    if position_ids is None:
        if not path.is_file():
            return {}
        named = {pid: fov_pattern.format(i=pid) for pid, *_ in _stage_points(path)}
        selected = {pid: fov for pid, fov in named.items() if fov in wanted}
        if not selected:
            raise ValueError(f"no PositionID p of {path} names one of the FOVs {_names(wanted)} as "
                             f"{fov_pattern!r}.format(i=p); give each FOV's PositionID (position_ids)")
        return selected
    if not isinstance(position_ids, Mapping):
        raise TypeError("position_ids must map each FOV ID to its PositionID")
    absent = [fov for fov in wanted if fov not in position_ids]
    if absent:
        raise ValueError(f"position_ids gives no PositionID for the FOVs {_names(absent)}")
    shared = sorted(str(p) for p, n in Counter(position_ids[fov] for fov in wanted).items() if n > 1)
    if shared:
        raise ValueError(f"position_ids gives two FOVs one PositionID: {_names(shared)}")
    return {position_ids[fov]: fov for fov in wanted}


def _rows_of(table, fov_ids):
    """The table's rows of fov_ids, in table order."""
    return TilePositionTable(table.positions[table.positions["fov_id"].isin(list(fov_ids))], table.source, table.record)


def table_from_positions(positions, fov_ids, grids, *, fov_pattern, position_ids=None,
                         grid_fov_ids=None) -> TilePositionTable:
    """The tile position table of the workflow's ``positions`` mapping, with the geometry the converter used.

    A stage file (``stage_file``, required with source ``stage_file``) may hold
    other FOVs' positions: the positions of fov_ids are selected before
    :func:`~starfinder.stitching.positions_from_leica` maps, shifts and snaps
    them. ``position_ids`` gives each FOV's ``PositionID`` (the workflow's
    ``fovs.csv``); without it ``PositionID`` ``p`` is the FOV
    ``fov_pattern.format(i=p)`` and the positions that name a FOV of fov_ids
    are selected. A supplied table (``table``) may also hold other FOVs: its rows
    of fov_ids are kept, in table order (a FOV it does not list is then missing
    from the table, which ``Dataset.stitch`` reports). A regular grid places
    ``grid_fov_ids`` (default fov_ids) in its cells and keeps the rows of
    fov_ids: the workflow passes the sample's FOVs of ``fovs.csv``, so that a
    selection keeps each FOV's cell.
    """
    if not isinstance(positions, Mapping):
        raise TypeError("table must be a TilePositionTable or the workflow's positions mapping")
    unknown = [k for k in positions if k not in POSITION_KEYS]
    if unknown:
        raise ValueError(f"unknown positions keys {unknown}; expected among {list(POSITION_KEYS)}")
    source = positions.get("source")
    geometry = None
    if source == "regular_grid":
        grid = _section(positions, "regular_grid", ("columns", "rows", "order", "overlap"))
        tile, spacing, geometry = _geometry(positions, grids)
        table = _rows_of(positions_from_grid(columns=grid["columns"], rows=grid["rows"], order=grid["order"],
                                             overlap=grid["overlap"], tile_shape_yx=tile, spacing_yx=spacing,
                                             fov_ids=fov_ids if grid_fov_ids is None else grid_fov_ids), fov_ids)
    elif source == "stage_file":
        axes = positions.get("stage_axes")
        if not isinstance(axes, Mapping) or set(axes) != {"x", "y"}:
            raise ValueError("positions.stage_axes ({x, y}) is required with source stage_file; it has no default")
        if not positions.get("stage_file"):
            raise ValueError("positions.stage_file (the stage file's path) is required with source stage_file; it "
                             "has no default")
        path = Path(positions["stage_file"])
        snap = None
        if positions.get("snap_to_grid") is not None:
            section = _section(positions, "snap_to_grid", ("overlap",))
            tile, spacing, geometry = _geometry(positions, grids)
            snap = GridSnap(overlap=section["overlap"], tile_shape_yx=tile, spacing_yx=spacing)
        ids = _selected_positions(path, fov_ids, fov_pattern, position_ids)
        table = positions_from_leica(path, stage_axes=StageAxisMapping(x=axes["x"], y=axes["y"]), fov_ids=ids,
                                     position_ids=list(ids), snap_to_grid=snap)
    elif source == "table":
        if not positions.get("table"):
            raise ValueError("positions.table (a CSV path) is required with source table")
        table = positions_from_table(positions["table"], unit=positions.get("unit", "µm"))
        # The table may hold other FOVs (other samples, unselected FOVs): only fov_ids' rows, in table order.
        table = _rows_of(table, fov_ids)
    else:
        raise ValueError(f"positions.source must be one of {list(POSITION_SOURCES)}; got {source!r}")
    if geometry is None:
        return table
    return TilePositionTable(table.positions, table.source, dict(table.record, geometry_source=geometry))
