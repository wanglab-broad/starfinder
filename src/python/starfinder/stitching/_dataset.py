"""What ``Dataset.stitch`` reads: the saved stitching images and the workflow's ``positions`` mapping.

docs/stitching-contract.md ("Coordination on the dataset", "Input image", "Workflow
configuration").
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from starfinder.image import IncompatibleGeometryError

from ._positions import (GridSnap, StageAxisMapping, TilePositionTable, _names, _stage_points, positions_from_grid,
                         positions_from_leica, positions_from_table)
from ._stitch import StitchingInput

INPUTS = ("reference_merged", "reference_stain")
POSITION_KEYS = ("source", "stage_axes", "snap_to_grid", "regular_grid", "tile_shape_yx", "spacing_yx", "table",
                 "unit", "stage_file")
POSITION_SOURCES = ("stage_file", "regular_grid", "table")
REFERENCE_STAIN_FILE = Path("other_rounds") / "reference_stain" / "image.ome.tif"


def _check_input(name):
    if not isinstance(name, str) or not (name in INPUTS or (name.startswith("file:") and len(name) > 5)):
        raise ValueError(f"input must be 'reference_merged', 'reference_stain' or 'file:<pattern>'; got {name!r}")


def read_inputs(output_root, fov_ids, input_name, *, reference) -> dict[str, StitchingInput]:
    """One StitchingInput per FOV, read one FOV at a time, keyed in fov_ids order.

    The grid is that of the FOV's reference image, as
    :func:`starfinder.dataset._declared.reference_grid` decides it with the
    keyword arguments ``reference`` (``reference_settings`` of the Dataset or
    the workflow configuration): the file ``reference_image`` names (default
    ``images/ref_merged/{fovID}.tif``), its stored metadata, or the grid the
    voxel sizes declare for a file that stores none. With ``reference_merged``
    that file is read once and is also the image.
    """
    from starfinder.dataset._declared import reference_volume
    from starfinder.dataset._paths import _FovPaths
    from starfinder.io import load_volume, load_volume_zyxc
    from starfinder.segmentation._persist import file_sha256
    _check_input(input_name)
    output_root = Path(output_root)
    loaded = {}
    for fov in fov_ids:
        paths = _FovPaths(output_root, fov)
        grid, volume, reference_path = reference_volume(output_root, fov, **reference)
        if input_name == "reference_merged":
            loaded[fov] = StitchingInput(fov, volume.image, grid, input_name, file_sha256(reference_path))
            continue
        del volume
        if input_name == "reference_stain":
            path, source = paths.checkpoint_dir / REFERENCE_STAIN_FILE, input_name
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


def table_from_positions(positions, fov_ids, grids, *, sample, output_root, fov_pattern,
                         fov_name=None) -> TilePositionTable:
    """The tile position table of the workflow's ``positions`` mapping, with the geometry the converter used.

    A stage file's ``PositionID`` ``p`` is the FOV ``fov_name(p)``; without
    ``fov_name`` (``Dataset.stitch``) it is ``fov_pattern % p``. The workflow
    adapter passes its own rule, ``fov_id_pattern.format(i=p)``.
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
        table = positions_from_grid(columns=grid["columns"], rows=grid["rows"], order=grid["order"],
                                    overlap=grid["overlap"], tile_shape_yx=tile, spacing_yx=spacing, fov_ids=fov_ids)
    elif source == "stage_file":
        axes = positions.get("stage_axes")
        if not isinstance(axes, Mapping) or set(axes) != {"x", "y"}:
            raise ValueError("positions.stage_axes ({x, y}) is required with source stage_file; it has no default")
        path = positions.get("stage_file") or Path(output_root) / "documents" / "maf" / f"{sample}.maf"
        snap = None
        if positions.get("snap_to_grid") is not None:
            section = _section(positions, "snap_to_grid", ("overlap",))
            tile, spacing, geometry = _geometry(positions, grids)
            snap = GridSnap(overlap=section["overlap"], tile_shape_yx=tile, spacing_yx=spacing)
        name = fov_name if fov_name is not None else (lambda pid: fov_pattern % pid)
        ids = {pid: name(pid) for pid, *_ in _stage_points(Path(path))} if Path(path).is_file() else {}
        table = positions_from_leica(path, stage_axes=StageAxisMapping(x=axes["x"], y=axes["y"]), fov_ids=ids,
                                     snap_to_grid=snap)
    elif source == "table":
        if not positions.get("table"):
            raise ValueError("positions.table (a CSV path) is required with source table")
        table = positions_from_table(positions["table"], unit=positions.get("unit", "µm"))
    else:
        raise ValueError(f"positions.source must be one of {list(POSITION_SOURCES)}; got {source!r}")
    if geometry is None:
        return table
    return TilePositionTable(table.positions, table.source, dict(table.record, geometry_source=geometry))
