"""The SpatialData export of an assembled sample, with each image written once.

docs/sample-output-contract.md ("SpatialData export"). An export of the sample tables and
the fused and label images of ``export/<sample>/``, not a second source: the points
``molecules``, the shapes ``cell_centroids`` and the tables ``cells``, ``expression``,
``cells_unplaced`` and ``molecules_unplaced`` take every value from the sample tables and
``sample.h5ad``; each image element and the labels element ``cell_labels`` is the output
level of its OME-TIFF, read lazily through tifffile's ``aszarr`` and written once, with
one level. Nothing is filtered, reassigned or recounted. An element whose population has
no row, or whose image does not exist, is left out and recorded ``omitted`` with its
reason. The store is written by spatialdata into a sibling staging folder and renamed into
place; the ``spatialdata`` section of ``export.json`` records every element, the formats,
the counts and the SHA-256 of every store file. spatialdata is imported only when the
export writes.
"""
from __future__ import annotations

import json
import math
import shutil
import warnings
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

from ._config import SpatialDataExportConfig
from ._record import (
    ExportIncompleteWarning,
    Timer,
    as_json,
    atomic_path,
    directory_files,
    listed_files,
    part_status,
    publish,
    read_export,
    remove_listed,
    section_sha256,
    update_section,
    verify_section_hashes,
)

EXTRA = "spatialdata"
STORE = "sample.zarr"
#: The fixed element names, by kind, in the order the section lists them.
TABLE_ELEMENTS = ("cells", "expression", "cells_unplaced", "molecules_unplaced")
POINTS, SHAPES, LABELS = "molecules", "cell_centroids", "cell_labels"
REGION, REGION_KEY, INSTANCE_KEY = SHAPES, "region", "instance_id"
CELL_LINK = "cell_instance_id"
SPATIALDATA_ATTRS = {"region": REGION, "region_key": REGION_KEY, "instance_key": INSTANCE_KEY}
#: The §2.9 index columns of a molecule and their names in the points element.
INDEX_RENAMES = {"z": "fov_index_z", "y": "fov_index_y", "x": "fov_index_x"}
#: The sample coordinates that PointsModel turns into the coordinate columns.
COORDINATE_RENAMES = {"sample_x": "x", "sample_y": "y", "sample_z": "z"}
ASSIGNMENT_STATUSES = ("assigned", "unassigned", "excluded_cell", "outside_grid")
#: The Zarr format of each store format, with spatialdata's container format and raster version string.
FORMATS = {3: {"container_format": "0.2", "raster_version": "0.5-dev-spatialdata"},
           2: {"container_format": "0.1", "raster_version": "0.4-dev-spatialdata"}}
KINDS = {"image": "images", "labels": "labels", "points": "points", "shapes": "shapes", "table": "tables"}


def check_name(name, what) -> None:
    """spatialdata's element naming rule (letters, digits, ``_``, ``-`` and ``.``; spatialdata 0.6.1
    ``check_valid_name``), applied without importing spatialdata. Raises ValueError naming ``what``."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"{what} {name!r} must be a nonempty string to name a SpatialData element")
    if name in (".", "..") or name.startswith("__") or not all(c.isalnum() or c in "_-." for c in name):
        raise ValueError(f"{what} {name!r} breaks spatialdata's element naming rule (letters, digits, '_', '-' "
                         "and '.' only, not '.' or '..', not starting with '__'), so it cannot name the store's "
                         "coordinate system")


def _check_keys(table, attribute, keys) -> None:
    """The keys of one table attribute: the naming rule and unique ignoring case (ValueError naming the key)."""
    seen = {}
    for key in keys:
        if not isinstance(key, str) or not key or key in (".", "..") or key.startswith("__") \
                or not all(c.isalnum() or c in "_-." for c in key) or (attribute in ("obs", "var") and
                                                                       key == "_index"):
            raise ValueError(f"table {table!r}: the {attribute} key {key!r} breaks spatialdata's naming rule")
        other = seen.get(key.lower())
        if other is not None:
            raise ValueError(f"table {table!r}: the {attribute} keys {other!r} and {key!r} are equal when case is "
                             "ignored, which spatialdata refuses")
        seen[key.lower()] = key


def check_table_keys(table, adata) -> None:
    """The obs and var columns and the uns, obsm and layers keys of a table (docs/sample-output-contract.md,
    "Element names"): each satisfies the naming rule and no two are equal ignoring case."""
    for attribute in ("obs", "var", "uns", "obsm", "layers"):
        _check_keys(table, attribute, [str(k) for k in getattr(adata, attribute).keys()])


# --- The source rows ---------------------------------------------------------------------------------------

def _finite(frame, columns) -> np.ndarray:
    if not len(frame):
        return np.zeros(0, bool)
    return np.isfinite(frame[list(columns)].to_numpy(np.float64)).all(axis=1)


def placed_cells(cells) -> np.ndarray:
    """The cell rows with a finite ownership centroid in Y and X (the circles of ``cell_centroids``)."""
    return _finite(cells, ("sample_ownership_centroid_y", "sample_ownership_centroid_x"))


def placed_molecules(molecules) -> np.ndarray:
    """The molecule rows whose ``sample_z/y/x`` are finite (the points of ``molecules``)."""
    return _finite(molecules, ("sample_z", "sample_y", "sample_x"))


def instance_ids(cells) -> np.ndarray:
    """``instance_id``: ``1..N`` over the rows of the sample cell table in its order (uint32)."""
    if len(cells) > 2 ** 32 - 1:
        raise ValueError(f"the sample cell table has {len(cells)} rows; instance_id is uint32, so at most "
                         f"{2 ** 32 - 1} rows can be exported")
    return np.arange(1, len(cells) + 1, dtype=np.uint32)


def cell_links(assembly) -> pd.array:
    """``cell_instance_id`` of each molecule: the ``instance_id`` of its ``(cell_namespace, cell_id)``.

    A molecule's cell key is its own ``(cell_namespace, cell_id)`` columns, the
    §2.9 key of a row of the sample cell table; null without a cell (and for
    every molecule of a ``molecules_only`` sample).
    """
    molecules, cells = assembly.molecules, assembly.cells
    if "cell_id" not in molecules.columns or not len(molecules):
        return pd.array([None] * len(molecules), dtype="UInt32")
    ids = instance_ids(cells)
    of = {(ns, int(c)): int(i) for ns, c, i in zip(cells["cell_namespace"].astype(object),
                                                   cells["cell_id"].astype("int64"), ids)}
    linked = molecules["cell_id"].notna().to_numpy(dtype=bool)
    keys = zip(molecules["cell_namespace"].astype(object)[linked], molecules["cell_id"][linked].astype("int64"))
    values = np.zeros(len(molecules), np.uint32)
    found = []
    for namespace, cell in keys:
        found.append(of.get((namespace, int(cell))))
    if any(v is None for v in found):
        first = next(i for i, v in zip(np.flatnonzero(linked), found) if v is None)
        raise ValueError(f"molecule {molecules['spot_namespace'].iloc[first]} {molecules['spot_id'].iloc[first]} "
                         f"links to cell {molecules['cell_id'].iloc[first]} of namespace "
                         f"{molecules['cell_namespace'].iloc[first]}, which the sample cell table does not hold")
    values[linked] = np.array(found, np.uint32)
    return pd.arrays.IntegerArray(values, ~linked)


def _fov_spacing(assembly) -> dict:
    """``fov_id`` → the spacing of the grid a cell of that ``fov_id`` was counted on.

    The FOV's grid on the route per FOV; on the route after fusion (``route``
    ``after_fusion``, ``mode`` ``cells``) the segmentation grid's spacing for
    every FOV, the grid of ``size_voxels`` there, which the sample assignment
    records (docs/sample-segmentation-contract.md, "Readers that change").
    """
    after_fusion = assembly.record.get("route") == "after_fusion" and assembly.record.get("mode") == "cells"
    segmentation = ((assembly.record.get("assignment") or {}).get("grid") or {}) if after_fusion else None
    out = {}
    for entry in assembly.record.get("fovs") or []:
        grid = segmentation if after_fusion else (entry.get("grid") or {})
        spacing = (grid.get("metadata") or {}).get("spacing_zyx")
        out[entry["fov_id"]] = None if spacing is None else tuple(float(v) for v in spacing)
    return out


def equivalent_radii(cells, spacings) -> np.ndarray:
    """The equivalent radius of each cell's original territory, from ``size_voxels`` and its grid's spacing.

    ``spacings`` maps ``fov_id`` to the spacing of the grid the cells of that
    ``fov_id`` were counted on: the FOV's grid on the route per FOV, the
    segmentation grid on the route after fusion (``_fov_spacing``). A
    ``volume`` territory as a sphere, ``(3·size·s_z·s_y·s_x / (4π))^(1/3)``; a
    ``plane`` territory, which the sample cell table marks with a NaN
    ``sample_centroid_z`` (docs/assembly-contract.md, "Sample tables"), as a disc,
    ``(size·s_y·s_x / π)^(1/2)``. Raises ValueError naming a cell of zero voxels,
    whose circle the shapes model refuses, or of a FOV without a known spacing.
    """
    size = cells["size_voxels"].to_numpy(np.float64)
    zero = np.flatnonzero(size <= 0)
    if len(zero):
        rows = cells.iloc[zero]
        named = ", ".join(f"{o} ({n}, cell_id {c})" for o, n, c in
                          zip(rows["obs_name"], rows["cell_namespace"], rows["cell_id"]))
        raise ValueError(f"cells of size_voxels 0 have no positive equivalent radius, which a circle of "
                         f"cell_centroids needs: {named}")
    plane = ~np.isfinite(cells["sample_centroid_z"].to_numpy(np.float64))
    radii = np.empty(len(cells), np.float64)
    for i, (fov, voxels) in enumerate(zip(cells["fov_id"].astype(object), size)):
        spacing = spacings.get(fov)
        if spacing is None:
            raise ValueError(f"cell {cells['obs_name'].iloc[i]}: its FOV {fov} has no known grid spacing, so its "
                             "equivalent radius is undefined")
        sz, sy, sx = spacing
        radii[i] = math.sqrt(voxels * sy * sx / math.pi) if plane[i] else \
            np.cbrt(3.0 * voxels * sz * sy * sx / (4.0 * math.pi))
    return radii


def _nullable_strings(frame):
    """String columns as the python-storage StringArray anndata writes (a copy; values and nulls unchanged)."""
    out = {}
    for column in frame.columns:
        values = frame[column]
        if isinstance(values.dtype, pd.StringDtype):
            values = pd.Series(pd.array(values.astype(object).where(values.notna(), None).tolist(),
                                        dtype=pd.StringDtype("python")), index=frame.index)
        out[column] = values
    return pd.DataFrame(out, index=frame.index, columns=list(frame.columns))


def _obs(frame, index):
    obs = _nullable_strings(frame.reset_index(drop=True))
    obs.index = pd.Index([str(v) for v in index], dtype=object)
    return obs


def _added(frame, names, table):
    taken = [n for n in names if n in frame.columns]
    if taken:
        raise ValueError(f"table {table!r}: the source already has the column(s) {taken} that the export adds")


# --- The elements ------------------------------------------------------------------------------------------

def _transform(output, kind):
    """The element's transform: ``Scale(S)`` then ``Translation(O − S/2)`` from its file's output level.

    ``store_axes`` are the element's own axes (Y and X for a 2D element); the
    record keeps Z too, from the file's record, which a 2D element does not carry.
    """
    spacing, origin = output["spacing_zyx"], output["origin_zyx"]
    axes = ["y", "x"] if output["axes"] == "YX" else ["z", "y", "x"]
    scale = {a: float(spacing[i]) for i, a in enumerate("zyx")}
    translation = {a: float(origin[i]) - float(spacing[i]) / 2 for i, a in enumerate("zyx")}
    return {"type": "sequence", "store_axes": axes, "scale": scale, "translation": translation}


def _identity():
    return {"type": "identity", "store_axes": None, "scale": None, "translation": None}


def _raster(path, output, kind, cs, stack):
    """One image or labels element read lazily from its OME-TIFF's output level (level 0, one level)."""
    import dask.array as da
    import tifffile
    import zarr
    from spatialdata.models import Image2DModel, Image3DModel, Labels2DModel, Labels3DModel
    from spatialdata.transformations import Scale, Sequence, Translation
    store = stack.enter_context(tifffile.imread(path, aszarr=True, level=0))
    array = da.from_zarr(zarr.open_array(store, mode="r"))
    record = _transform(output, kind)
    axes = record["store_axes"]
    transform = Sequence([Scale([record["scale"][a] for a in axes], axes=tuple(axes)),
                          Translation([record["translation"][a] for a in axes], axes=tuple(axes))])
    if kind == "image":
        model = Image2DModel if len(axes) == 2 else Image3DModel
        element = model.parse(array[None], dims=("c", *axes), transformations={cs: transform})
    else:
        model = Labels2DModel if len(axes) == 2 else Labels3DModel
        element = model.parse(array, dims=tuple(axes), transformations={cs: transform})
    return element, model.__name__, record


def _entry(name, kind, *, status="written", reason=None, **extra):
    entry = {"name": name, "kind": kind, "status": status}
    if status == "omitted":
        entry["reason"] = reason
    entry.update(extra)
    return entry


def _image_names(fusion, requested):
    """The image elements a fusion section asks for: ``<image>_<dimensionality>`` per image name and file."""
    config = (fusion or {}).get("config")
    if isinstance(config, Mapping):
        dims = ("projection", "stack") if config.get("dimensionality") == "both" else (config.get("dimensionality"),)
        return [f"{image.split('=', 1)[0]}_{d}" for image in config.get("images", ()) for d in dims]
    return list(requested or ())


def _rasters(folder, record, requested_images):
    """The image and labels elements: (written outputs with their files, omitted entries)."""
    from starfinder.segmentation._persist import check_file
    written, omitted = [], []
    fusion, status = record.get("fusion"), part_status(record, "fuse")
    if status == "failed" or part_status(record, "fuse_labels") == "failed":
        raise ValueError("a fuse or fuse_labels section of export.json records a failed part; run that part again "
                         "before the SpatialData export")
    if status == "written":
        outputs = fusion.get("outputs") or []
        zarrs = [o["path"] for o in outputs if o.get("format") == "ome_zarr"]
        if zarrs:
            raise ValueError(f"the fusion section lists standalone OME-Zarr files {zarrs}; with the SpatialData export "
                             "the store holds the one OME-Zarr copy of each fused image (run fuse with ome_zarr off)")
        for output in outputs:
            check_file(folder / output["path"], output["sha256"], "fused image")
            written.append(("image", f"{output['image']}_{output['dimensionality']}", output))
    else:
        reason = {"empty": "no_included_fov", "not_run": "part_not_run"}.get(status, "not_requested")
        for name in _image_names(fusion if status in ("empty", "not_run") else None, requested_images):
            omitted.append(_entry(name, "image", status="omitted", reason=reason))
    labels, status = record.get("labels"), part_status(record, "fuse_labels")
    if status == "written" and int(labels.get("n_ids_output") or 0) > 0:
        output = labels["outputs"][0]
        check_file(folder / output["path"], output["sha256"], "label image")
        written.append(("labels", LABELS, output))
    elif status == "written":
        omitted.append(_entry(LABELS, "labels", status="omitted", reason="empty"))
    elif status == "empty":
        omitted.append(_entry(LABELS, "labels", status="omitted", reason=labels.get("reason")))
    else:
        omitted.append(_entry(LABELS, "labels", status="omitted",
                              reason="part_not_run" if status == "not_run" else "not_requested"))
    return written, omitted


def _expression(folder, assembly, ids):
    """``sample.h5ad`` read whole, with the added obs columns ``region`` and ``instance_id``."""
    import anndata

    from starfinder.segmentation._persist import check_file
    entry = (assembly.record.get("files") or {}).get("anndata")
    if not entry:
        raise ValueError("the assembly record lists no sample.h5ad; save the assembly with save_assembly first")
    path = folder / entry["path"]
    check_file(path, entry["sha256"], "sample.h5ad")
    adata = anndata.read_h5ad(path)
    _added(adata.obs, (REGION_KEY, INSTANCE_KEY), "expression")
    of = dict(zip(assembly.cells["obs_name"].astype(object), ids))
    names = [str(v) for v in adata.obs_names]
    missing = [n for n in names if n not in of]
    if missing:
        raise ValueError(f"sample.h5ad rows {missing[:5]} are not rows of the sample cell table")
    adata.obs[REGION_KEY] = pd.Categorical([REGION] * adata.n_obs, categories=[REGION])
    adata.obs[INSTANCE_KEY] = np.array([of[n] for n in names], np.uint32)
    return adata


def build_elements(folder, assembly, record, cs, *, requested_images, stack):
    """Every element of the store, parsed, and the section's entries (in the order the section lists them).

    Applies every check of the export before anything is written: the tables'
    keys, the zero-size cells, and the models' own refusals (a table that
    already has ``uns["spatialdata_attrs"]`` is refused by ``TableModel.parse``).
    """
    import anndata
    import geopandas as gpd
    from shapely import points as shapely_points
    from spatialdata.models import PointsModel, ShapesModel, TableModel
    from spatialdata.transformations import Identity
    cells, molecules = assembly.cells, assembly.molecules
    ids = instance_ids(cells)
    links = cell_links(assembly)
    placed_c, placed_m = placed_cells(cells), placed_molecules(molecules)
    elements = {"points": {}, "shapes": {}, "tables": {}, "images": {}, "labels": {}}
    entries = []

    # points molecules
    _added(molecules, (CELL_LINK, *INDEX_RENAMES.values()), POINTS)
    rows = molecules[placed_m]
    if len(rows):
        frame = rows.rename(columns=INDEX_RENAMES).reset_index(drop=True)
        frame[CELL_LINK] = links[placed_m]
        elements["points"][POINTS] = PointsModel.parse(frame, coordinates={"x": "sample_x", "y": "sample_y",
                                                                           "z": "sample_z"},
                                                       transformations={cs: Identity()})
        entries.append(_entry(POINTS, "points", rows=int(len(frame)), transform=_identity()))
    else:
        entries.append(_entry(POINTS, "points", status="omitted", reason="empty"))

    # shapes cell_centroids and table cells, written or omitted together
    _added(cells, (REGION_KEY, INSTANCE_KEY), "cells")
    rows = cells[placed_c]
    if len(rows):
        radii = equivalent_radii(rows, _fov_spacing(assembly))
        circles = gpd.GeoDataFrame({"geometry": shapely_points(rows["sample_ownership_centroid_x"].to_numpy(np.float64),
                                                               rows["sample_ownership_centroid_y"].to_numpy(np.float64)),
                                    "radius": radii}, index=pd.Index(ids[placed_c], dtype=np.uint32))
        elements["shapes"][SHAPES] = ShapesModel.parse(circles, transformations={cs: Identity()})
        entries.append(_entry(SHAPES, "shapes", rows=int(len(rows)), transform=_identity()))
        obs = _obs(rows, rows["obs_name"])
        obs[REGION_KEY] = pd.Categorical([REGION] * len(obs), categories=[REGION])
        obs[INSTANCE_KEY] = ids[placed_c]
        table = anndata.AnnData(obs=obs)
        check_table_keys("cells", table)
        elements["tables"]["cells"] = TableModel.parse(table, region=REGION, region_key=REGION_KEY,
                                                       instance_key=INSTANCE_KEY)
        entries.append(_entry("cells", "table", rows=int(len(obs))))
    else:
        entries.append(_entry(SHAPES, "shapes", status="omitted", reason="empty"))
        entries.append(_entry("cells", "table", status="omitted", reason="empty"))

    # table expression: sample.h5ad with every part kept
    adata = _expression(folder, assembly, ids)
    if adata.n_obs:
        check_table_keys("expression", adata)
        elements["tables"]["expression"] = TableModel.parse(adata, region=REGION, region_key=REGION_KEY,
                                                            instance_key=INSTANCE_KEY)
        entries.append(_entry("expression", "table", rows=int(adata.n_obs)))
    else:
        entries.append(_entry("expression", "table", status="omitted", reason="empty"))

    # tables cells_unplaced and molecules_unplaced: no region, region key or instance key
    # (cells by obs_name, molecules by their row of the sample molecule table)
    for name, frame, keep, column, values in (
            ("cells_unplaced", cells, ~placed_c, INSTANCE_KEY, ids),
            ("molecules_unplaced", molecules, ~placed_m, CELL_LINK, links)):
        rows = frame[keep]
        if not len(rows):
            entries.append(_entry(name, "table", status="omitted", reason="empty"))
            continue
        _added(frame, (column,), name)
        obs = _obs(rows, rows["obs_name"] if name == "cells_unplaced" else np.flatnonzero(keep))
        obs[column] = values[keep]
        table = anndata.AnnData(obs=obs)
        check_table_keys(name, table)
        elements["tables"][name] = TableModel.parse(table)
        entries.append(_entry(name, "table", rows=int(len(obs))))
    order = {n: i for i, n in enumerate(TABLE_ELEMENTS)}
    tables = sorted((e for e in entries if e["kind"] == "table"), key=lambda e: order[e["name"]])
    entries = [e for e in entries if e["kind"] != "table"] + tables

    # images and labels, read lazily from the OME-TIFF files
    written, omitted = _rasters(folder, record, requested_images)
    for kind, name, output in written:
        check_name(name, f"{kind} element")
        element, model, transform = _raster(folder / output["path"], output, kind, cs, stack)
        elements[KINDS[kind]][name] = element
        entries.append(_entry(name, kind, source=output["path"], source_sha256=output["sha256"], model=model,
                              shape=[int(n) for n in element.shape], dtype=str(element.dtype), levels=1,
                              transform=transform))
    entries.extend(omitted)
    names = [e["name"] for e in entries]
    folded = {}
    for name in names:
        folded.setdefault(name.lower(), []).append(name)
    equal = [group for group in folded.values() if len(group) > 1]
    if equal:
        raise ValueError(f"element names equal ignoring case: {equal}")
    return elements, entries


# --- Counts and the section --------------------------------------------------------------------------------

def _population(frame, reasons):
    status = frame["assembly_status"].astype(object)
    excluded = frame[status.eq("excluded").to_numpy(dtype=bool)]["assembly_reason"].astype(object)
    return {"retained": int(status.eq("retained").sum()),
            "excluded": {r: int(excluded.eq(r).sum()) for r in reasons}}


def export_counts(assembly) -> dict:
    """The counts of "Read-back checks" (S5): molecules by ``assignment_status`` and by ``assembly_status`` and
    reason, cells by ``assembly_status`` and reason, and the counted cells.

    The reasons are those the assembly record counts: without the duplicate
    rule (``duplicate_threshold`` null, as on the route after fusion) its
    reasons are not listed, as in the record.
    """
    from starfinder.assembly._config import CELL_ASSEMBLY_REASONS, DUPLICATE_REASONS, MOLECULE_ASSEMBLY_REASONS
    config = assembly.record.get("config") or {}
    rule_off = "duplicate_threshold" in config and config["duplicate_threshold"] is None
    cell_reasons = tuple(r for r in CELL_ASSEMBLY_REASONS if not (rule_off and r in DUPLICATE_REASONS))
    molecule_reasons = tuple(r for r in MOLECULE_ASSEMBLY_REASONS if not (rule_off and r in DUPLICATE_REASONS))
    molecules, cells = assembly.molecules, assembly.cells
    statuses = molecules["assignment_status"].astype(object).value_counts() if len(molecules) else pd.Series()
    by_assignment = {s: int(statuses.get(s, 0)) for s in ASSIGNMENT_STATUSES}
    by_assignment.update({str(s): int(n) for s, n in statuses.items() if s not in by_assignment})
    counted = int((cells["assembly_status"].eq("retained") & cells["status"].eq("kept")).fillna(False).sum()) \
        if len(cells) else 0
    return {"molecules": {"n": int(len(molecules)), "assignment_status": by_assignment,
                          "assembly": _population(molecules, molecule_reasons)},
            "cells": {"n": int(len(cells)), "assembly": _population(cells, cell_reasons),
                      "counted": counted}}


def _dtypes(frame) -> dict:
    return {str(c): str(d) for c, d in frame.dtypes.items()}


def _round_trip(path, entries) -> None:
    """The column dtypes after the round trip and the ``spatialdata_attrs`` written, read back from the store.

    Each part is read with the reader spatialdata 0.6.1 uses for it (anndata's
    ``read_elem`` of the table's ``obs``, ``var`` and ``uns``, dask's
    ``read_parquet`` of the points, geopandas' of the shapes), reading only
    what the record needs; ``check_export`` compares them with
    ``spatialdata.read_zarr``.
    """
    import dask.dataframe as dd
    import geopandas
    import zarr
    from anndata.io import read_elem

    from starfinder.segmentation._persist import file_sha256
    for entry in entries:
        if entry["status"] != "written":
            continue
        kind = entry["kind"]
        entry["path"] = f"{STORE}/{KINDS[kind]}/{entry['name']}"
        folder = Path(path) / KINDS[kind] / entry["name"]
        if kind == "table":
            group = zarr.open_group(str(folder), mode="r")
            obs, var, uns = (read_elem(group[key]) for key in ("obs", "var", "uns"))
            entry["spatialdata_attrs"] = as_json(uns.get("spatialdata_attrs"))
            entry["dtypes"] = {"obs": _dtypes(obs), "var": _dtypes(var)}
            entry["index_dtype"] = {"obs": str(obs.index.dtype), "var": str(var.index.dtype)}
        elif kind == "points":
            entry["dtypes"] = _dtypes(dd.read_parquet(folder / "points.parquet"))
        elif kind == "shapes":
            shapes = geopandas.read_parquet(folder / "shapes.parquet")
            entry["dtypes"] = _dtypes(shapes)
            entry["index_dtype"] = str(shapes.index.dtype)
        else:
            metadata = folder / "0"
            name = "zarr.json" if (metadata / "zarr.json").is_file() else ".zarray"
            entry["array_metadata"] = f"{entry['path']}/0/{name}"
            entry["array_metadata_sha256"] = file_sha256(metadata / name)


def _versions():
    from importlib.metadata import version
    return {name: version(name) for name in ("spatialdata", "anndata", "zarr", "dask", "geopandas", "tifffile")}


def _formats(zarr_format):
    from spatialdata._io.format import (
        CurrentPointsFormat,
        CurrentRasterFormat,
        CurrentShapesFormat,
        CurrentTablesFormat,
        SpatialDataContainerFormatV01,
    )
    if zarr_format == 3:
        return None, {"zarr_format": 3, **FORMATS[3], "raster_format": CurrentRasterFormat().spatialdata_format_version,
                      "shapes_format": CurrentShapesFormat().spatialdata_format_version,
                      "points_format": CurrentPointsFormat().spatialdata_format_version,
                      "tables_format": CurrentTablesFormat().spatialdata_format_version}
    from spatialdata._io.format import ContainerV01DefaultTypes
    return SpatialDataContainerFormatV01(), {
        "zarr_format": 2, **FORMATS[2],
        **{f"{k}_format": ContainerV01DefaultTypes[k].spatialdata_format_version
           for k in ("raster", "shapes", "points", "tables")}}


def _check(assembly, config, folder):
    from starfinder.assembly import AssemblyResult
    if not isinstance(assembly, AssemblyResult):
        raise TypeError("assembly must be an AssemblyResult (typically load_assembly(output_dir))")
    if not isinstance(config, SpatialDataExportConfig):
        raise TypeError("config must be a SpatialDataExportConfig")
    config.__post_init__()
    check_name(assembly.sample, "sample name")
    record = read_export(folder)
    section = record.get("assembly")
    if not isinstance(section, Mapping) or section.get("stage") != "assembly":
        raise ValueError(f"{folder} holds no saved assembly; the SpatialData export reads its tables/ and sample.h5ad "
                         "(save the assembly with save_assembly, or run Dataset.export with assemble on)")
    if section_sha256(as_json(assembly.record)) != section_sha256(section):
        raise ValueError(f"the assembly given is not the one saved in {folder}: its record's SHA-256 differs from the "
                         "assembly section's (pass load_assembly(output_dir))")
    current = section_sha256(section)
    for name in ("fusion", "labels"):
        recorded = (record.get(name) or {}).get("assembly_sha256")
        if recorded is not None and recorded != current:
            raise ValueError(f"the {name} section of export.json was made from another assembly (its recorded "
                             f"assembly SHA-256 {recorded} is not the assembly section's {current}); run it again")
    return record


def _given(record, name, given):
    if given is None:
        return
    section = getattr(given, "section", given)
    if section_sha256(as_json(section)) != section_sha256(record.get(name)):
        raise ValueError(f"the {name} record given is not the {name} section of export.json in the folder")


def export_spatialdata(assembly, *, fused=None, labels=None,
                       config: SpatialDataExportConfig = SpatialDataExportConfig(), output_dir) -> Path:
    """Write the SpatialData store ``sample.zarr`` of an assembled sample and its ``spatialdata`` section.

    docs/sample-output-contract.md ("SpatialData export"). One coordinate
    system, named after the sample, in the sample frame's axes and µm (identity
    for points and shapes). Elements, each written only when its population has
    a row or its image exists: points ``molecules`` (every molecule with finite
    ``sample_z/y/x``; the §2.9 ``z, y, x`` renamed ``fov_index_z/y/x``,
    ``sample_x/y/z`` turned into ``x, y, z`` by ``PointsModel``,
    ``cell_instance_id`` added); shapes ``cell_centroids`` (a circle per cell
    with a finite ownership centroid, at ``(sample_ownership_centroid_x,
    sample_ownership_centroid_y)``, radius the equivalent radius of the
    original territory, index ``instance_id``); table ``cells`` (those rows,
    every column, ``region`` and ``instance_id`` added, annotating
    ``cell_centroids``); table ``expression`` (``sample.h5ad`` with every part
    kept, annotating likewise); tables ``cells_unplaced`` and
    ``molecules_unplaced`` (the rows without finite sample coordinates, with
    ``instance_id`` or ``cell_instance_id`` as plain columns); one image element
    ``<image>_<dimensionality>`` per fused OME-TIFF and the labels element
    ``cell_labels``, each its file's output level read lazily through tifffile's
    ``aszarr``, one level, transform ``Scale(S)`` then ``Translation(O − S/2)``.
    ``instance_id`` is ``1..N`` over the sample cell table's rows. The store is
    written in spatialdata's default format (Zarr format 3) or, with
    ``zarr_format`` 2, ``SpatialDataContainerFormatV01``, into a sibling staging
    folder renamed into place. The ``spatialdata`` section of
    ``output_dir/export.json`` then lists every element with ``written`` or
    ``omitted`` and its reason (``empty``, ``no_cells``, ``no_included_fov``,
    ``not_requested``, ``part_not_run``), the transforms, the
    ``spatialdata_attrs`` and column dtypes after the round trip, the counts,
    the column renames and the SHA-256 of every store file; when every element
    is omitted no store is written and the section records ``store: null``.

    When the ``spatialdata`` extra is not importable it writes no store, records
    ``not_run`` (``extra_missing``) and issues :class:`ExportIncompleteWarning`.
    A refusal (the checks below) raises before anything is written and leaves
    the folder unchanged; an error while writing removes the staging folder,
    records ``failed`` with the exception's type and message, and is raised
    again.

    Parameters
    ----------
    assembly : AssemblyResult
        The saved assembly of ``output_dir``, ``load_assembly(output_dir)``.
    fused, labels : FusionRecord, LabelRecord or None
        The records of the fuse and fuse_labels parts; when given they must be
        the folder's sections, which are read in any case.
    config : SpatialDataExportConfig
        ``zarr_format`` 3 (default) or 2.
    output_dir : str or Path
        The sample's export folder ``<OUTPUT>/export/<sample>/``.

    Returns
    -------
    Path
        The store, or ``output_dir`` when no store is written.

    Raises
    ------
    TypeError
        An argument of the wrong type.
    ValueError
        A sample name outside spatialdata's naming rule; an assembly that is not
        the folder's; a fusion or labels section made from another assembly or
        recording a failed part; a standalone OME-Zarr in the fusion section; a
        file whose SHA-256 differs from its record; a table key outside the
        naming rule or two equal ignoring case; a cell of zero voxels; and the
        library's refusals, among them a table that already has
        ``uns["spatialdata_attrs"]``.
    """
    timer = Timer()
    output_dir = Path(output_dir)
    record = _check(assembly, config, output_dir)
    _given(record, "fusion", fused)
    _given(record, "labels", labels)
    return _export(assembly, record, config, output_dir, timer, requested_images=None)


def _export(assembly, record, config, output_dir, timer, *, requested_images):
    sample = assembly.sample
    common = {"coordinate_system": sample, "frame_id": f"sample:{sample}",
              "assembly_sha256": section_sha256(record.get("assembly")),
              "fusion_sha256": section_sha256(record.get("fusion")) if "fusion" in record else None,
              "labels_sha256": section_sha256(record.get("labels")) if "labels" in record else None,
              "config": {"zarr_format": config.zarr_format}}

    def finish(status, *, warn=None, **extra):
        from ._fuse import _software
        section = as_json({"status": status, **extra, **common, "software": _software(), **timer.record(),
                           "viewer_qualification": "deferred"})
        section.setdefault("outputs", [])
        update_section(output_dir, sample, "export_spatialdata", section)
        if warn:
            warnings.warn(f"export part export_spatialdata could not run: {warn}", ExportIncompleteWarning,
                          stacklevel=3)
        return section

    try:
        import spatialdata  # noqa: F401
    except ImportError:
        finish("not_run", reason="extra_missing", extra=EXTRA, store=None,
               warn=f"extra_missing: the {EXTRA!r} extra is not installed")
        return output_dir
    import anndata
    sdata_format, formats = _formats(config.zarr_format)
    counts = export_counts(assembly)
    renames = {"molecules": {**INDEX_RENAMES, **COORDINATE_RENAMES}}
    path = output_dir / STORE
    with ExitStack() as stack:
        stack.enter_context(pd.option_context("future.infer_string", False))
        elements, entries = build_elements(output_dir, assembly, record, sample,
                                           requested_images=requested_images, stack=stack)
        if not any(entry["status"] == "written" for entry in entries):
            remove_listed(output_dir, listed_files(read_export(output_dir), "export_spatialdata"))
            finish("empty", reason="every_element_omitted", store=None, format=formats, library=_versions(),
                   elements=entries, counts=counts, renames=renames, outputs=[])
            return output_dir
        remove_listed(output_dir, listed_files(read_export(output_dir), "export_spatialdata"))
        tmp = atomic_path(path)
        try:
            sdata = spatialdata.SpatialData(**{k: v for k, v in elements.items() if v},
                                            attrs={"frame_id": f"sample:{sample}", "coordinate_system": sample})
            with anndata.settings.override(allow_write_nullable_strings=True):
                sdata.write(tmp, sdata_formats=sdata_format)
            _round_trip(tmp, entries)
            files = directory_files(tmp)
            publish(tmp, path)
        except BaseException as error:
            shutil.rmtree(tmp, ignore_errors=True)
            finish("failed", store=None, format=formats, library=_versions(), elements=entries, counts=counts,
                   renames=renames, error={"type": type(error).__name__, "message": str(error)})
            raise
    finish("written", store=STORE, format=formats, library=_versions(), elements=entries, counts=counts,
           renames=renames, outputs=[{"path": STORE, "format": "spatialdata_zarr", "zarr_format": config.zarr_format,
                                      "files": files, "sha256": section_sha256(files)}])
    return path


# --- Source lookup -----------------------------------------------------------------------------------------

def lookup_source(folder, *, spot=None, cell=None) -> dict:
    """Find where a molecule or a cell of an export folder came from.

    docs/sample-output-contract.md ("Identities and source lookup"). Reads
    ``export.json``, verifies its hashes, finds the row by ``spot =
    (spot_namespace, spot_id)`` or ``cell = (cell_namespace, cell_id)`` and
    returns, from the ``assembly`` section: ``fov_id``; ``assignment_folder``
    and ``assignment_record_sha256`` (the FOV's ``assignment.json``; None for a
    molecule table or an assignment in memory); ``row``, the row's zero-based
    position in the FOV's §2.9 table (molecules in input order, cells in
    increasing ``cell_id``); and ``kind``.

    On the route after fusion (an assembly with ``route`` ``after_fusion`` and
    ``mode`` ``cells``) a cell has no §2.9 table of a FOV:
    ``assignment_folder`` and ``assignment_record_sha256`` are None, ``row``
    is the cell's position in the sample cell table, and two keys are added,
    ``label_file`` (the cell run's label file, relative to the folder, as the
    ``assembly`` section's ``segmentation_files`` lists it) and
    ``label_value`` (its ``cell_id``) (docs/sample-segmentation-contract.md,
    "Readers that change"). A molecule is found as in a ``molecules_only``
    sample.

    The hashes verified are those ``export.json`` records about its own
    sections and the assembly's files: each SHA-256 of a section that another
    section recorded when it read it (``fusion.assembly_sha256``,
    ``labels.assembly_sha256``, and ``spatialdata``'s ``assembly_sha256``,
    ``fusion_sha256`` and ``labels_sha256``), and every file of the assembly
    as ``load_assembly`` checks it. A section absent because its part was off
    or did not run records nothing; only the ``assembly`` section is needed.
    The files of the store are not hashed here: that is condition S8, which
    :func:`check_export` evaluates.

    Raises
    ------
    ValueError
        Neither or both keys given; no ``assembly`` section; a section whose
        SHA-256 differs from the one another section recorded (an edited
        folder; the message names the section and the recording field); a file
        of the assembly whose SHA-256 differs; a key the sample does not hold.
    """
    from starfinder.assembly import load_assembly
    if (spot is None) == (cell is None):
        raise ValueError("give exactly one of spot=(spot_namespace, spot_id) or cell=(cell_namespace, cell_id)")
    folder = Path(folder)
    record = read_export(folder)
    if not isinstance(record.get("assembly"), Mapping):
        raise ValueError(f"{folder} holds no export record with an assembly section")
    verify_section_hashes(record, where=f"export.json of {folder}")
    assembly = load_assembly(folder)
    if spot is not None:
        frame, keys, kind = assembly.molecules, ("spot_namespace", "spot_id"), "spot"
        namespace, ident = str(spot[0]), str(spot[1])
        hit = (frame[keys[0]].astype(object) == namespace) & (frame[keys[1]].astype(object) == ident)
    else:
        frame, keys, kind = assembly.cells, ("cell_namespace", "cell_id"), "cell"
        namespace, ident = str(cell[0]), int(cell[1])
        hit = (frame[keys[0]].astype(object) == namespace) & (frame[keys[1]].astype("Int64") == ident).fillna(False)
    rows = np.flatnonzero(hit.to_numpy(dtype=bool))
    if len(rows) != 1:
        raise ValueError(f"the sample holds {len(rows)} rows of {kind} key ({namespace!r}, {ident!r})")
    i = int(rows[0])
    fov = frame["fov_id"].iloc[i]
    if kind == "cell" and assembly.record.get("route") == "after_fusion" and assembly.record.get("mode") == "cells":
        # The route after fusion: the cell of the sample segmentation, in its run's label file.
        run = json.loads(namespace)[-1]
        files = [e for e in assembly.record.get("segmentation_files") or [] if e.get("run") == run]
        if len(files) != 1:
            raise ValueError(f"the assembly section lists {len(files)} label files of the cell run {run!r} in its "
                             "segmentation_files")
        return {"kind": kind, "key": [namespace, ident], "fov_id": str(fov), "assignment_folder": None,
                "assignment_record_sha256": None, "row": i, "label_file": files[0]["path"], "label_value": ident}
    position = int(np.count_nonzero(frame["fov_id"].astype(object).to_numpy()[:i] == fov))
    entry = next(e for e in assembly.record["fovs"] if e["fov_id"] == fov)
    assignment = entry.get("assignment") or {}
    return {"kind": kind, "key": [namespace, ident], "fov_id": str(fov), "assignment_folder": assignment.get("folder"),
            "assignment_record_sha256": assignment.get("record_sha256") if assignment.get("folder") else None,
            "row": position}
