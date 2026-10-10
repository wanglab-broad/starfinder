"""``check_export``: the read-back checks of an export folder (docs/sample-output-contract.md, "Read-back checks").

R1 to R3 (the record and the folder) and, for each fused image of the ``fusion`` section,
F1 (the OME-TIFF and OME-Zarr layout checks and conditions 1 to 3 of "Sample position to
image position"), F2 (dtype), F3 (uncovered voxels), F7 (projection and stack grids), F8
(no included FOV) and F9 (OME-Zarr equal to OME-TIFF). F4 to F6 are fixture checks of the
tests. For the label image of the ``labels`` section: L1 (layout, uint32), L2 (no new
value), L3 (drawn by retained cells), L4 (absent IDs) and L5 (grid); L6 is a check of the
entry before anything is written. For the store of the ``spatialdata`` section, read back
with ``spatialdata.read_zarr``: S1 (elements, omissions, format) to S8 (store hashes). On the
route after fusion (docs/sample-segmentation-contract.md, "The record"): R1 with the part
``segment``, T1 to T5 for the sample segmentation, and L3 and L5 by the rule ``segmentation``.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from ._record import (
    EXPORT_FILE,
    FORMAT_VERSION,
    NO_FILES,
    PARTS,
    PENDING_REASON,
    SECTIONS,
    STAGE,
    STATUSES,
    is_complete,
    is_pending,
    listed_file_hashes,
    listed_files,
    part_status,
    section_sha256,
)

TOLERANCE_TIFF_UM = 1e-9
TOLERANCE_ZARR_UM = 1e-6
OME_TYPES = {"uint8": "uint8", "uint16": "uint16", "uint32": "uint32", "float32": "float"}


class _Failed(ValueError):
    pass


def _fail(condition, message):
    raise _Failed(f"check_export {condition}: {message}")


def _close(a, b, tolerance):
    return all(abs(float(x) - float(y)) <= tolerance for x, y in zip(a, b)) and len(a) == len(b)


# --- R1 to R3 ----------------------------------------------------------------------------------------------

def _r1(folder):
    path = folder / EXPORT_FILE
    if not path.is_file():
        _fail("R1", f"{path} does not exist")
    try:
        record = json.loads(path.read_text())
    except ValueError as error:
        _fail("R1", f"{path} does not parse: {error}")
    if not isinstance(record, dict) or record.get("format_version") != FORMAT_VERSION or record.get("stage") != STAGE:
        _fail("R1", f"{path} is not a version {FORMAT_VERSION} export record")
    for part in PARTS:
        status = part_status(record, part)
        if status not in STATUSES:
            _fail("R1", f"part {part} has status {status!r}, not one of {list(STATUSES)}")
        section = record.get(SECTIONS[part])
        if isinstance(section, dict) and "status" in section and section["status"] != status:
            _fail("R1", f"part {part} has status {status!r} in parts and {section['status']!r} in its section")
        if is_pending(record, part):
            _fail("R1", f"part {part} was not finished: the export stopped before the part wrote its section "
                        f"(status 'not_run', reason {PENDING_REASON!r})")
    if record.get("complete") is not is_complete(record):
        _fail("R1", f"complete is {record.get('complete')!r}, but the part statuses give {is_complete(record)}")
    for name in ("fusion", "labels", "spatialdata"):
        recorded = (record.get(name) or {}).get("assembly_sha256")
        if recorded is not None:
            current = section_sha256(record.get("assembly"))
            if recorded != current:
                _fail("R1", f"the {name} section records the assembly section's SHA-256 {recorded}, but the "
                            f"assembly section's is {current}")
    spatialdata = record.get("spatialdata") or {}
    for name in ("fusion", "labels"):
        key = f"{name}_sha256"
        if spatialdata.get(key) is not None and spatialdata[key] != section_sha256(record.get(name)):
            _fail("R1", f"the spatialdata section records the {name} section's SHA-256 {spatialdata[key]}, but the "
                        f"{name} section's is {section_sha256(record.get(name))}")
    return record


def _r2(folder, record):
    from starfinder.segmentation._persist import file_sha256
    for part in PARTS:
        if part_status(record, part) in NO_FILES and listed_files(record, part):
            _fail("R2", f"part {part} has status {part_status(record, part)!r} but lists files "
                        f"{listed_files(record, part)}")
    listed = listed_file_hashes(record)
    for relative, digest in sorted(listed.items()):
        path = folder / relative
        if not path.is_file():
            _fail("R2", f"the listed file {relative} is missing")
        actual = file_sha256(path)
        if actual != digest:
            _fail("R2", f"the listed file {relative} has SHA-256 {actual}, the record {digest}")
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder).as_posix()
        if path.is_file() and relative != EXPORT_FILE and relative not in listed:
            _fail("R2", f"the file {relative} is not listed in {EXPORT_FILE}")


def _r3(folder, record):
    from starfinder.assembly import load_assembly
    from starfinder.assembly._persist import ANNDATA_FILE, TABLE_DIR
    status = part_status(record, "assemble")
    if status in ("written", "reused"):
        try:
            load_assembly(folder)
        except (ValueError, FileNotFoundError) as error:
            _fail("R3", f"the assembly does not load: {error}")
    elif status == "off":
        present = [p.relative_to(folder).as_posix() for p in (folder / TABLE_DIR).glob("*")] \
            if (folder / TABLE_DIR).is_dir() else []
        if (folder / ANNDATA_FILE).exists():
            present.append(ANNDATA_FILE)
        if present:
            _fail("R3", f"the assembly part is off but {present} exist")


# --- Fused images ------------------------------------------------------------------------------------------

def _expected_level(grid, dimensionality, k):
    """(S, O, N) of the output level from the section's fusion grid (conditions 1 and 3)."""
    (sz, sy, sx), (oz, oy, ox), (nz, ny, nx) = grid["spacing_zyx"], grid["origin_zyx"], grid["shape_zyx"]
    if grid["inputs_2d"]:
        z_lo, z_hi = grid["slab_z"]
        z = (z_hi - z_lo, (z_lo + z_hi) / 2, 1)
    elif dimensionality == "stack":
        z = (sz, oz, nz)
    else:
        z = (nz * sz, oz + sz * (nz - 1) / 2, 1)
    yx = [(k * s, o + s * (k - 1) / 2, math.ceil(n / k)) for s, o, n in ((sy, oy, ny), (sx, ox, nx))]
    return [z[0], yx[0][0], yx[1][0]], [z[1], yx[0][1], yx[1][1]], [z[2], yx[0][2], yx[1][2]]


def _check_geometry(name, output, grid):
    spacing, origin, shape = _expected_level(grid, output["dimensionality"], output["factor_yx"])
    if list(output["shape_zyx"]) != shape:
        _fail("F1", f"{name}: the recorded shape {output['shape_zyx']} is not the output level {shape}")
    if not (_close(output["spacing_zyx"], spacing, TOLERANCE_TIFF_UM)
            and _close(output["origin_zyx"], origin, TOLERANCE_TIFF_UM)):
        _fail("F1", f"{name}: the recorded spacing {output['spacing_zyx']} and origin {output['origin_zyx']} are not "
                    f"the output level's {spacing} and {origin}")


def _input_dtype(section, output):
    dtypes = {entry["images"][output["image"]].get("dtype") for entry in section["fovs"]
              if entry["included"] and output["image"] in entry["images"]}
    dtypes.discard(None)
    if output["dimensionality"] == "projection" and output["projection"] == "sum":
        return "float32"
    return next(iter(dtypes)) if len(dtypes) == 1 else None


def _check_tiff(folder, name, output, section, condition="F1"):
    from starfinder.io.tiff import METADATA_NAMESPACE

    from ._tiff import ANNOTATION_NAMESPACE, read_ome_layout
    layout = read_ome_layout(folder / output["path"])
    one_plane = output["shape_zyx"][0] == 1
    axes = "YX" if one_plane else "ZYX"
    # Layout check 1.
    if not (layout["is_bigtiff"] and layout["is_ome"]):
        _fail(condition, f"{name} is not a BigTIFF OME-TIFF")
    if layout["n_series"] != 1 or layout["axes"] != axes or layout["channels"] != 1:
        _fail(condition, f"{name} has {layout['n_series']} series of axes {layout['axes']} and {layout['channels']} "
                    f"channels; expected one series of axes {axes} and one channel")
    # Layout check 2.
    shape = output["shape_zyx"][1:] if one_plane else output["shape_zyx"]
    if layout["shape"] != list(shape) or layout["dtype"] != output["dtype"]:
        _fail(condition, f"{name} has shape {layout['shape']} and dtype {layout['dtype']}; the record {shape} and "
                    f"{output['dtype']}")
    if layout["pixels"].get("Type") != OME_TYPES.get(output["dtype"]):
        _fail(condition, f"{name}: the OME pixel type {layout['pixels'].get('Type')} is not that of {output['dtype']}")
    if layout["tile_yx"] != output["tile_yx"] or layout["tile_yx"] != [512, 512] \
            or layout["compression"] not in ("adobe_deflate", "deflate", "zlib"):
        _fail(condition, f"{name} has tiles {layout['tile_yx']} and compression {layout['compression']}; expected "
                    "512 × 512 tiles and zlib")
    recorded_levels = [level["shape_zyx"][1:] if one_plane else level["shape_zyx"] for level in output["levels"]]
    if layout["level_shapes"] != [list(s) for s in recorded_levels]:
        _fail(condition, f"{name} has SubIFD levels {layout['level_shapes']}; the record {recorded_levels}")
    # Layout check 3 and condition 1.
    spacing, origin = output["spacing_zyx"], output["origin_zyx"]
    pixels = layout["pixels"]
    physical = [float(pixels.get(f"PhysicalSize{a}", "nan")) for a in "ZYX"]
    units = [pixels.get(f"PhysicalSize{a}Unit") for a in "ZYX"]
    if not _close(physical, spacing, TOLERANCE_TIFF_UM) or set(units) != {"µm"}:
        _fail(condition, f"{name}: PhysicalSize {physical} {units} is not the recorded spacing {spacing} µm")
    n_z = output["shape_zyx"][0]
    if len(layout["planes"]) != n_z:
        _fail(condition, f"{name} has {len(layout['planes'])} Plane elements for {n_z} planes")
    for i, plane in enumerate(layout["planes"]):
        position = [float(plane.get(f"Position{a}", "nan")) for a in "ZYX"]
        expected = [origin[0] + i * spacing[0], origin[1], origin[2]]
        if not _close(position, expected, TOLERANCE_TIFF_UM):
            _fail(condition, f"{name}: plane {i} is at {position}, not at {expected}")
    stored = layout["annotations"].get(METADATA_NAMESPACE)
    if not stored or not _close(stored["spacing_zyx"], spacing, TOLERANCE_TIFF_UM) \
            or not _close(stored["origin_zyx"], origin, TOLERANCE_TIFF_UM):
        _fail(condition, f"{name}: the stored ImageMetadata {stored} does not have the recorded spacing and origin")
    if layout["annotations"].get(ANNOTATION_NAMESPACE) != output.get("annotation"):
        _fail(condition, f"{name}: the {ANNOTATION_NAMESPACE} annotation differs from the record")
    for level in output["levels"]:
        f = level["factors_zyx"]
        if not (_close(level["spacing_zyx"], [a * b for a, b in zip(f, spacing)], TOLERANCE_TIFF_UM)
                and _close(level["origin_zyx"], [o + s * (k - 1) / 2 for k, s, o in zip(f, spacing, origin)],
                           TOLERANCE_TIFF_UM)):
            _fail(condition, f"{name}: level {level['level']} geometry is not S·f and O + S·(f − 1)/2")


def _covered(section, output):
    """Per output axis Y and X (and Z for the stack of multi-plane FOVs), the voxels some included FOV covers."""
    grid = section["grid"]
    spacing, origin, shape = output["spacing_zyx"], output["origin_zyx"], output["shape_zyx"]
    fused_axes = (1, 2) if output["inputs_2d"] else (0, 1, 2)
    margin = [1e-6 * s for s in grid["spacing_zyx"]]
    centres = [origin[a] + np.arange(shape[a]) * spacing[a] for a in range(3)]
    covered = np.zeros(tuple(shape), bool)
    for entry in section["fovs"]:
        if not entry["included"]:
            continue
        n, o = entry["grid"]["shape_zyx"], entry["origin_zyx"]
        s = grid["spacing_zyx"]
        inside = []
        for a in range(3):
            if a not in fused_axes:
                inside.append(np.ones(shape[a], bool))
                continue
            lo, hi = o[a], o[a] + (n[a] - 1) * s[a]
            inside.append((centres[a] >= lo - margin[a]) & (centres[a] <= hi + margin[a]))
        covered |= inside[0][:, None, None] & inside[1][None, :, None] & inside[2][None, None, :]
    return covered


def _check_values(folder, name, output, section):
    import tifffile
    path = folder / output["path"]
    covered = _covered(section, output)
    with tifffile.TiffFile(path) as tif:
        for i in range(output["shape_zyx"][0]):
            plane = tif.series[0].asarray(key=i) if output["shape_zyx"][0] > 1 else tif.series[0].asarray()
            if plane.dtype != np.dtype(output["dtype"]):
                _fail("F2", f"{name} holds {plane.dtype}, the record {output['dtype']}")
            values = plane[~covered[i]]
            if values.size and (np.isnan(values).any() if values.dtype.kind == "f" else False):
                _fail("F3", f"{name}: an uncovered voxel of plane {i} holds NaN")
            if values.size and np.any(values != 0):
                _fail("F3", f"{name}: {int(np.count_nonzero(values))} uncovered voxels of plane {i} are not 0")


_METADATA_KEYS = ("zarr.json", ".zarray", ".zattrs", ".zgroup")


def _chunk_keys_read(path, level_path, region) -> set[str]:
    """The chunk keys a read of ``region`` of one level of a Zarr group asks its store for."""
    import zarr
    store = zarr.storage.LocalStore(path, read_only=True)
    keys, get = [], store.get

    async def counted(key, *args, **kwargs):
        keys.append(key)
        return await get(key, *args, **kwargs)

    store.get = counted
    array = zarr.open_array(store=store, path=level_path, mode="r")
    keys.clear()
    array[region]
    return {k for k in keys if not k.endswith(_METADATA_KEYS)}


def _check_chunk_reads(folder, name, output, multiscale):
    """OME-Zarr layout check 3: when a level has more than one chunk, a region inside one chunk touches it only.

    The chunks are the requested ones of docs/sample-output-contract.md ("OME-Zarr layout"),
    (1, 1, 1024, 1024) for a projection and (1, 1, 16, 512, 512) for a stack, clipped to the
    level's size on each axis. The region is the first such chunk of each level that has more
    than one: one chunk is read, never a whole array.
    """
    from ._fuse import ZARR_CHUNKS
    requested = ZARR_CHUNKS["stack" if "z" in output["axes"] else "projection"]
    requested = [requested[a] for a in output["axes"]]
    for dataset, level in zip(multiscale["datasets"], output["levels"]):
        shape = level["shape"]
        chunks = [min(c, n) for c, n in zip(requested, shape)]
        if math.prod(math.ceil(n / c) for n, c in zip(shape, chunks)) <= 1:
            continue
        region = tuple(slice(0, c) for c in chunks)
        touched = _chunk_keys_read(folder / output["path"], dataset["path"], region)
        if len(touched) != 1:
            _fail("F1", f"{name}: OME-Zarr layout check 3 fails at level {dataset['path']}: a region inside one "
                        f"chunk ({[(r.start, r.stop) for r in region]}) touches {len(touched)} chunks "
                        f"{sorted(touched)}")


def _check_zarr(folder, name, output, tiff_outputs):
    import tifffile
    import zarr
    group = zarr.open_group(str(folder / output["path"]), mode="r")
    attrs = dict(group.attrs)
    ngff = attrs.get("ome", attrs)
    version = str(ngff.get("version") or ngff["multiscales"][0].get("version"))
    if group.metadata.zarr_format != output["zarr_format"] or version != output["ngff_version"]:
        _fail("F1", f"{name} is Zarr format {group.metadata.zarr_format}, NGFF {version}; the record "
                    f"{output['zarr_format']} and {output['ngff_version']}")
    multiscale = ngff["multiscales"][0]
    axes = [a["name"] for a in multiscale["axes"]]
    if axes != output["axes"] or axes[:2] != ["t", "c"]:
        _fail("F1", f"{name} has axes {axes}; the record {output['axes']}")
    units = {a["name"]: a.get("unit") for a in multiscale["axes"]}
    if any(units[a] != "micrometer" for a in axes[2:]):
        _fail("F1", f"{name}: the space axes have units {units}, not micrometer")
    if len(multiscale["datasets"]) != len(output["levels"]):
        _fail("F1", f"{name} has {len(multiscale['datasets'])} datasets; the record {len(output['levels'])} levels")
    for dataset, level in zip(multiscale["datasets"], output["levels"]):
        transforms = {t["type"]: t[t["type"]] for t in dataset["coordinateTransformations"]}
        spatial = axes[2:]
        scale = transforms["scale"][2:]
        translation = transforms.get("translation", [0.0] * len(axes))[2:]
        if not (_close(scale, [level["spacing"][a] for a in spatial], TOLERANCE_ZARR_UM)
                and _close(translation, [level["origin"][a] for a in spatial], TOLERANCE_ZARR_UM)):
            _fail("F1", f"{name}: dataset {dataset['path']} has scale {scale} and translation {translation}; the "
                        f"record {level['spacing']} and {level['origin']}")
        array = group[dataset["path"]]
        if list(array.shape) != level["shape"] or str(array.dtype) != level["dtype"] \
                or list(array.chunks) != level["chunks"]:
            _fail("F1", f"{name}: dataset {dataset['path']} has shape {array.shape}, dtype {array.dtype} and chunks "
                        f"{array.chunks}; the record {level['shape']}, {level['dtype']} and {level['chunks']}")
        from ._fuse import _codec
        if _codec(array) != level["codec"]:
            _fail("F1", f"{name}: dataset {dataset['path']} has codec {_codec(array)}; the record {level['codec']}")
    _check_chunk_reads(folder, name, output, multiscale)
    twin = next((t for t in tiff_outputs if t["image"] == output["image"]
                 and t["dimensionality"] == output["dimensionality"]), None)
    if twin is not None:
        level0 = np.asarray(group[multiscale["datasets"][0]["path"]][...])
        level0 = level0.reshape(level0.shape[2:])
        if level0.ndim == 3 and level0.shape[0] == 1 and twin["shape_zyx"][0] == 1:
            level0 = level0[0]
        image = tifffile.imread(folder / twin["path"])
        if level0.shape != image.shape or level0.dtype != image.dtype or not np.array_equal(level0, image):
            _fail("F9", f"{name}: the output level differs from {twin['path']}")


def _check_fusion(folder, record):
    section = record.get("fusion")
    status = part_status(record, "fuse")
    if status in ("off", None) or not isinstance(section, dict):
        return
    included = [entry for entry in section.get("fovs", []) if entry.get("included")]
    if (status == "empty") != (not included):
        _fail("F8", f"the fusion section has status {status!r} with {len(included)} included FOVs")
    if status == "empty" and (section.get("outputs") != [] or section.get("reason") != "no_included_fov"):
        _fail("F8", "an empty fusion section must have outputs [] and reason 'no_included_fov'")
    outputs = section.get("outputs") or []
    tiffs = [o for o in outputs if o["format"] == "ome_tiff"]
    for output in outputs:
        name = output["path"]
        if output["format"] == "ome_tiff":
            _check_geometry(name, output, section["grid"])
            _check_tiff(folder, name, output, section)
            expected = _input_dtype(section, output)
            if output["dtype"] != expected:
                _fail("F2", f"{name} has dtype {output['dtype']}; the input dtype rule gives {expected}")
            _check_values(folder, name, output, section)
        elif output["format"] == "ome_zarr":
            _check_geometry(name, output, section["grid"])
            _check_zarr(folder, name, output, tiffs)
        else:
            _fail("F1", f"{name} has an unknown format {output['format']!r}")
    for image in {o["image"] for o in tiffs}:
        files = {o["dimensionality"]: o for o in tiffs if o["image"] == image}
        if {"projection", "stack"} <= set(files):
            p, s = files["projection"], files["stack"]
            if p["shape_zyx"][1:] != s["shape_zyx"][1:] or p["spacing_zyx"][1:] != s["spacing_zyx"][1:] \
                    or p["origin_zyx"][1:] != s["origin_zyx"][1:]:
                _fail("F7", f"the projection and the stack of {image} have different Y and X grids")


# --- The sample segmentation of the route after fusion: T1 to T5 ---------------------------------------------

def _after_fusion(section) -> bool:
    """Whether an ``assembly`` section holds the cells of a sample segmentation."""
    return isinstance(section, Mapping) and section.get("route") == "after_fusion" and section.get("mode") == "cells"


def _same_grid(output, grid) -> bool:
    return list(output["shape_zyx"]) == list(grid["shape_zyx"]) \
        and _close(output["spacing_zyx"], grid["spacing_zyx"], TOLERANCE_TIFF_UM) \
        and _close(output["origin_zyx"], grid["origin_zyx"], TOLERANCE_TIFF_UM)


def _t1(folder, record, section):
    """T1: the label files with their SHA-256, uint32, on the recorded grid; with the fusion written, the fused
    projection of each input image with the recorded SHA-256 and grid."""
    import tifffile

    from starfinder.segmentation._persist import file_sha256
    grid = section["inputs"]["grid"]
    outputs = {o.get("path"): o for o in section.get("outputs") or []}
    for run in section.get("runs") or []:
        path = folder / run["path"]
        if not path.is_file():
            _fail("T1", f"the label file {run['path']} of run {run['name']!r} is missing")
        actual = file_sha256(path)
        if actual != run["sha256"]:
            _fail("T1", f"the label file {run['path']} of run {run['name']!r} has SHA-256 {actual}, the segmentation "
                        f"section {run['sha256']}")
        output = outputs.get(run["path"])
        if output is None or output.get("sha256") != run["sha256"] or output.get("dtype") != "uint32" \
                or not _same_grid(output, grid):
            _fail("T1", f"the segmentation section's output entry of {run['path']} is not a uint32 file of its SHA-256 "
                        f"on the recorded grid {grid['shape_zyx']}")
        _check_tiff(folder, run["path"], output, section, condition="T1")
        labels = tifffile.imread(path)
        if labels.dtype != np.uint32 or [1, *labels.shape[-2:]] != list(grid["shape_zyx"]) or labels.ndim != 2:
            _fail("T1", f"the label file {run['path']} holds {labels.dtype} of shape {labels.shape}, not uint32 of the "
                        f"grid's shape {grid['shape_zyx']}")
    if part_status(record, "fuse") != "written":
        return   # the fused files are absent; the recorded input SHA-256 values are provenance only
    fused = {o["image"]: o for o in (record.get("fusion") or {}).get("outputs") or []
             if o.get("format") == "ome_tiff" and o.get("dimensionality") == "projection"}
    for entry in section["inputs"].get("images") or []:
        output = fused.get(entry["image"])
        if output is None:
            _fail("T1", f"the fusion section lists no fused projection of image {entry['image']!r}, which the "
                        "segmentation was made from")
        path = folder / output["path"]
        actual = file_sha256(path) if path.is_file() else None
        if output["sha256"] != entry["sha256"] or actual != entry["sha256"]:
            _fail("T1", f"the fused projection of image {entry['image']!r} has SHA-256 {actual}, but the segmentation "
                        f"was made from {entry['sha256']}: a segmentation made from another fusion")
        if not _same_grid(output, grid):
            _fail("T1", f"the fused projection of image {entry['image']!r} is on the grid {output['shape_zyx']}, "
                        f"spacing {output['spacing_zyx']}, origin {output['origin_zyx']}; the segmentation grid "
                        f"{grid['shape_zyx']}, {grid['spacing_zyx']}, {grid['origin_zyx']}")


def _t5(record, section):
    """T5: the assembly and labels sections' segmentation_files are the segmentation section's label files (and
    territories.ome.tif)."""
    expected = [{"run": r["name"], "path": r["path"], "sha256": r["sha256"]} for r in section.get("runs") or []]
    assembly = record.get("assembly") or {}
    territories = (assembly.get("files") or {}).get("territories")
    if territories:
        expected.append({"run": None, "path": territories["path"], "sha256": territories["sha256"]})
    if assembly.get("segmentation_files") != expected:
        _fail("T5", f"the assembly section's segmentation_files {assembly.get('segmentation_files')} are not the "
                    f"label files of the segmentation section {expected}: tables made from another segmentation")
    labels = record.get("labels") or {}
    if part_status(record, "fuse_labels") == "written" and labels.get("segmentation_files") != expected:
        _fail("T5", f"the labels section's segmentation_files {labels.get('segmentation_files')} are not the label "
                    f"files of the segmentation section {expected}: a label image made from another segmentation")


def _label_image(path):
    import tifffile
    labels = tifffile.imread(path)
    return labels[np.newaxis] if labels.ndim == 2 else labels


def _t2(folder, section, cells):
    """T2: the positive values of the cell run's labels are exactly the cell_ids of the sample cell table, one row
    each, all with the run's namespace."""
    run = next((r for r in section.get("runs") or [] if r["name"] == section.get("cells")), None)
    if run is None:
        _fail("T2", f"the segmentation section has no run named by cells {section.get('cells')!r}")
    values = np.unique(_label_image(folder / run["path"]))
    values = values[values > 0].astype(np.int64).tolist()
    ids = cells["cell_id"].astype("int64").to_numpy().tolist() if len(cells) else []
    if len(set(ids)) != len(ids) or sorted(ids) != values:
        _fail("T2", f"the cell run's labels hold {len(values)} cells; the sample cell table has {len(ids)} rows of "
                    f"{len(set(ids))} cell_ids, not one row per label")
    namespaces = set(cells["cell_namespace"].astype(object)) if len(cells) else set()
    if namespaces - {run["label_namespace"]}:
        _fail("T2", f"cell rows have the namespaces {sorted(map(str, namespaces))}, not the run's "
                    f"{run['label_namespace']}")


def _t3(folder, record, section, molecules):
    """T3: every retained molecule's cell_id is the territory value at the pixel floor(j + ½) of j = (sample_yx −
    O_yx) ⊘ S_yx (0 for no cell); a molecule outside the grid is not retained, and excluded outside_grid when its
    position's owner is its FOV."""
    grid = section["inputs"]["grid"]
    files = record["assembly"].get("files") or {}
    if "territories" in files:
        territories = _label_image(folder / files["territories"]["path"])
    else:
        run = next(r for r in section["runs"] if r["name"] == section.get("cells"))
        territories = _label_image(folder / run["path"])
    if not len(molecules):
        return
    yx = molecules[["sample_y", "sample_x"]].to_numpy(np.float64)
    finite = np.isfinite(yx).all(axis=1)
    j = (yx - np.asarray(grid["origin_zyx"][1:], np.float64)) / np.asarray(grid["spacing_zyx"][1:], np.float64)
    with np.errstate(invalid="ignore"):
        pixel = np.floor(j + 0.5)
    pixel = np.where(np.isfinite(pixel), pixel, -1).astype(np.int64)
    inside = finite & (pixel >= 0).all(axis=1) & (pixel < np.asarray(territories.shape[1:])).all(axis=1)
    retained = molecules["assembly_status"].astype(object).eq("retained").to_numpy(dtype=bool)
    owner, fov = molecules["owner_fov_id"].astype(object), molecules["fov_id"].astype(object)
    own = (owner.notna() & owner.eq(fov)).fillna(False).to_numpy(dtype=bool)
    outside_grid = molecules["assembly_reason"].eq("outside_grid").fillna(False).to_numpy(dtype=bool)
    outside = finite & ~inside & (retained | (own & ~outside_grid))
    if outside.any():
        i = int(np.flatnonzero(outside)[0])
        _fail("T3", f"{int(outside.sum())} molecules outside the segmentation grid are not excluded outside_grid, "
                    f"the first {molecules['spot_namespace'].iloc[i]} {molecules['spot_id'].iloc[i]}")
    check = retained & inside
    values = territories[0, pixel[check, 0], pixel[check, 1]].astype(np.int64)
    cells = molecules["cell_id"].astype("Int64").fillna(0).to_numpy(np.int64)[check]
    wrong = values != cells
    if wrong.any():
        i = int(np.flatnonzero(check)[np.flatnonzero(wrong)[0]])
        _fail("T3", f"{int(wrong.sum())} retained molecules have a cell_id that their pixel of the territories does "
                    f"not hold, the first {molecules['spot_namespace'].iloc[i]} {molecules['spot_id'].iloc[i]} "
                    f"(cell_id {cells[np.flatnonzero(wrong)[0]]}, the pixel {values[np.flatnonzero(wrong)[0]]})")


def _recorded_ownership(section):
    """The ownership map of the placement as the assembly section records it (footprints, centres, owning FOVs and
    boxes, with ``ownership`` and ``tie``); the same point test as ``ownership_map`` gives."""
    from starfinder.assembly import AssemblyConfig
    from starfinder.assembly._ownership import GEOMETRY_COLUMNS, OwnershipMap
    config = AssemblyConfig(ownership=section["ownership"]["rule"], tie=section["ownership"]["tie"])
    nan = float("nan")
    rows = []
    for entry in section["fovs"]:
        (y0, y1), (x0, x1) = entry.get("footprint_yx") or ((nan, nan), (nan, nan))
        cy, cx = entry.get("centre_yx") or (nan, nan)
        (by0, by1), (bx0, bx1) = entry.get("box_yx") or ((nan, nan), (nan, nan))
        closed = config.tie == "table_order"
        rows.append({"fov_id": entry["fov_id"], "placement_status": entry["placement_status"], "footprint_y0": y0,
                     "footprint_y1": y1, "footprint_x0": x0, "footprint_x1": x1, "centre_y": cy, "centre_x": cx,
                     "owns_points": bool(entry["owns_points"]), "candidate": bool(entry["owns_points"]),
                     "box_y0": by0, "box_y1": by1, "box_x0": bx0, "box_x1": bx1,
                     # an upper box edge set at a neighbour's midpoint is closed under table_order
                     "box_y1_closed": closed and by1 != y1, "box_x1_closed": closed and bx1 != x1})
    geometry = pd.DataFrame(rows, columns=list(GEOMETRY_COLUMNS)).astype(
        {"fov_id": "string", "placement_status": "string", "owns_points": bool, "candidate": bool,
         "box_y1_closed": bool, "box_x1_closed": bool})
    return OwnershipMap(section["sample"], geometry, config)


def _t4(section, cells):
    """T4: every cell's owner_fov_id is the owner of its ownership centroid and its fov_id that owner, or where none
    owns it the owning FOV with the nearest footprint centre (F1)."""
    from starfinder.assembly._fused import _nearest_owner
    if not len(cells):
        return
    ownership = _recorded_ownership(section)
    points = cells[["sample_ownership_centroid_z", "sample_ownership_centroid_y",
                    "sample_ownership_centroid_x"]].to_numpy(np.float64)
    owner = ownership.owner(points)
    expected = owner.copy()
    none = np.array([v is None for v in owner], dtype=bool)
    if none.any():
        expected[none] = _nearest_owner(points[none][:, 1:], ownership.geometry)
    recorded_owner = [None if v is None or v is pd.NA else str(v) for v in cells["owner_fov_id"].astype(object)]
    fov = [str(v) for v in cells["fov_id"].astype(object)]
    for i, (a, b, c, d) in enumerate(zip(recorded_owner, owner, fov, expected)):
        if a != b or c != d:
            _fail("T4", f"cell {cells['obs_name'].iloc[i]}: fov_id {c!r} and owner_fov_id {a!r}; the rule of the "
                        f"cells gives {d!r} and {b!r}")


def _check_segmentation(folder, record):
    """T1 to T5 for the sample segmentation of the ``segmentation`` section (docs/sample-segmentation-contract.md,
    "The record"), when the part ``segment`` is ``written`` or ``reused``."""
    from starfinder.assembly import load_assembly
    section = record.get("segmentation")
    if part_status(record, "segment") not in ("written", "reused") or not isinstance(section, dict):
        return
    _t1(folder, record, section)
    if part_status(record, "assemble") not in ("written", "reused"):
        return
    _t5(record, section)
    if not _after_fusion(record.get("assembly")):
        return
    assembly = load_assembly(folder)
    _t2(folder, section, assembly.cells)
    _t3(folder, record, section, assembly.molecules)
    _t4(record["assembly"], assembly.cells)


# --- Labels ------------------------------------------------------------------------------------------------

def _formula_grid(assembly_section, k, two_d):
    """The label grid's (S, O, N) per axis Z, Y, X from the FOVs the assembly section records (L5)."""
    from ._grid import INCLUDED
    from ._labels import label_grid
    fovs = [e for e in assembly_section["fovs"] if e["placement_status"] in INCLUDED and e.get("grid")]
    origins = [e["origin_zyx"] for e in fovs]
    shapes = [e["grid"]["shape_zyx"] for e in fovs]
    spacing = fovs[0]["grid"]["metadata"]["spacing_zyx"]
    yx = label_grid(spacing, origins, shapes, k)
    sz = float(spacing[0])
    one_plane = all(shape[0] == 1 for shape in shapes)
    if one_plane:
        z_lo, z_hi = min(o[0] for o in origins) - sz / 2, max(o[0] for o in origins) + sz / 2
        z = (z_hi - z_lo, (z_lo + z_hi) / 2, 1)
    else:
        oz = min(float(o[0]) for o in origins)
        nz = int(math.floor((max(float(o[0]) + (n[0] - 1) * sz for o, n in zip(origins, shapes)) - oz) / sz
                            + 1e-9)) + 1
        z = (nz * sz, oz + sz * (nz - 1) / 2, 1) if two_d else (sz, oz, nz)
    return ([z[0], yx["spacing"]["y"], yx["spacing"]["x"]], [z[1], yx["origin"]["y"], yx["origin"]["x"]],
            [z[2], yx["shape"]["y"], yx["shape"]["x"]])


def _check_labels(folder, record):
    """L1 to L5 for the label image of the ``labels`` section."""
    import tifffile

    from starfinder.assembly import load_assembly

    from ._labels import LABEL_FILE, _read_territory, retained_by_fov, retained_image
    section = record.get("labels")
    status = part_status(record, "fuse_labels")
    if status != "written" or not isinstance(section, dict):
        return
    outputs = section.get("outputs") or []
    if [o.get("path") for o in outputs] != [LABEL_FILE]:
        _fail("L1", f"the labels section lists {[o.get('path') for o in outputs]}, not [{LABEL_FILE!r}]")
    output = outputs[0]
    if output["dtype"] != "uint32":
        _fail("L1", f"{LABEL_FILE} is recorded as {output['dtype']}, not uint32")
    _check_tiff(folder, LABEL_FILE, output, section)
    from ._labels import SEGMENTATION_RULE
    if (section.get("rule") == SEGMENTATION_RULE) != _after_fusion(record.get("assembly")):
        _fail("L3", f"the labels section has the rule {section.get('rule')!r}, but the assembly is of the route "
                    f"{(record.get('assembly') or {}).get('route')!r} with mode "
                    f"{(record.get('assembly') or {}).get('mode')!r}")
    if section.get("rule") == SEGMENTATION_RULE:
        _check_sample_labels(folder, record, section, output)
        return
    # L5: the grid.
    two_d = bool(output["inputs_2d"])
    spacing, origin, shape = _formula_grid(record["assembly"], int(output["factor_yx"]), two_d)
    if list(output["shape_zyx"]) != shape or not _close(output["spacing_zyx"], spacing, TOLERANCE_TIFF_UM) \
            or not _close(output["origin_zyx"], origin, TOLERANCE_TIFF_UM):
        _fail("L5", f"{LABEL_FILE} has spacing {output['spacing_zyx']}, origin {output['origin_zyx']} and shape "
                    f"{output['shape_zyx']}; the formulas give {spacing}, {origin} and {shape}")
    # L2 to L4: the values, against the territory files, the instance_id map and assembly_status.
    assembly = load_assembly(folder)
    retained = retained_by_fov(assembly.cells)
    retained_ids = {int(v) for _, ids in retained.values() for v in ids}
    entries = {e["fov_id"]: e for e in section.get("fovs") or []}
    if set(entries) != set(retained):
        _fail("L4", f"the labels section reads the territories of {sorted(entries)}, but the FOVs with a retained "
                    f"cell are {sorted(retained)}")
    origins = {e["fov_id"]: e["origin_zyx"] for e in record["assembly"]["fovs"]}
    spacings = {e["fov_id"]: e["grid"]["metadata"]["spacing_zyx"] for e in record["assembly"]["fovs"] if e["grid"]}
    with tifffile.TiffFile(folder / LABEL_FILE) as tif:
        n_z = output["shape_zyx"][0]
        planes = [tif.series[0].asarray(key=i) if n_z > 1 else tif.series[0].asarray() for i in range(n_z)]
    image = np.stack(planes)
    ids_output = {int(v) for v in np.unique(image) if v > 0}
    ids_input = set()
    centres = [output["origin_zyx"][a] + np.arange(output["shape_zyx"][a]) * output["spacing_zyx"][a]
               for a in range(3)]
    for fov, entry in entries.items():
        territory = _read_territory(Path(entry["path"]), entry["sha256"])
        cell_ids, ids = retained[fov]
        mapped = retained_image(territory, cell_ids, ids)
        ids_input |= {int(v) for v in np.unique(mapped) if v > 0}
        # L3: at every voxel showing a cell of this FOV, its territory sampled by floor(c + 1/2) holds that cell.
        own = np.isin(image, ids)
        if not own.any():
            continue
        index = [np.floor((centres[a] - origins[fov][a]) / spacings[fov][a] + 0.5).astype(np.int64)
                 for a in range(3)]
        if mapped.shape[0] == 1:
            index[0] = np.zeros(len(index[0]), np.int64)
        inside = [(i >= 0) & (i < n) for i, n in zip(index, mapped.shape)]
        z, y, x = np.nonzero(own)
        ok = inside[0][z] & inside[1][y] & inside[2][x]
        sampled = np.zeros(len(z), np.uint32)
        sampled[ok] = mapped[index[0][z[ok]], index[1][y[ok]], index[2][x[ok]]]
        if np.any(sampled != image[z, y, x]):
            bad = int(np.count_nonzero(sampled != image[z, y, x]))
            _fail("L3", f"{bad} voxels of {LABEL_FILE} show a cell of FOV {fov} that its territory, sampled at the "
                        "voxel centre by floor(c + 1/2), does not hold there")
    new = sorted(ids_output - retained_ids) + sorted((ids_output & retained_ids) - ids_input)
    if new:
        _fail("L2", f"{LABEL_FILE} holds values that are not the instance_id of a retained cell in ids_input: "
                    f"{new[:10]}")
    absent = sorted(ids_input - ids_output)
    if section.get("n_ids_input") != len(ids_input) or section.get("n_ids_output") != len(ids_output) \
            or section.get("absent_ids") != absent or section.get("n_absent_ids") != len(absent):
        _fail("L4", f"the labels section records {section.get('n_ids_input')} IDs in, {section.get('n_ids_output')} "
                    f"out and absent {section.get('absent_ids')}; recomputed {len(ids_input)}, {len(ids_output)} "
                    f"and {absent}")


def _check_sample_labels(folder, record, section, output):
    """L2 to L5 by the rule ``segmentation`` (docs/sample-segmentation-contract.md, "The label images"): L3 reads
    the sample's territories, L5 checks the grid of the label scale's spacing and the territories' origin."""
    from starfinder.assembly import load_assembly

    from ._labels import (LABEL_FILE, read_sample_territory, retained_ids, retained_image, sample_label_grid,
                          sample_territory_source)
    assembly = load_assembly(folder)
    territory_grid = assembly.record["assignment"]["grid"]
    shape = territory_grid["shape_zyx"]
    spacing, origin = territory_grid["metadata"]["spacing_zyx"], territory_grid["metadata"]["origin_zyx"]
    # L5: spacing k·s (s the FOVs' acquired spacing), the territories' origin, the size of the formula; Z the slab.
    from ._grid import INCLUDED
    acquired = {tuple(float(v) for v in e["grid"]["metadata"]["spacing_zyx"][1:]) for e in record["assembly"]["fovs"]
                if e["placement_status"] in INCLUDED and e.get("grid")}
    if len(acquired) != 1:
        _fail("L5", f"the included FOVs of the assembly section have the Y and X spacings {sorted(acquired)}")
    s_y, s_x = next(iter(acquired))
    k = int(output["factor_yx"])
    formula = sample_label_grid(shape, spacing, origin, (k * s_y, k * s_x))
    expected = ([float(spacing[0]), formula["spacing"]["y"], formula["spacing"]["x"]],
                [float(origin[0]), formula["origin"]["y"], formula["origin"]["x"]],
                [1, formula["shape"]["y"], formula["shape"]["x"]])
    if list(output["shape_zyx"]) != expected[2] or not _close(output["spacing_zyx"], expected[0], TOLERANCE_TIFF_UM) \
            or not _close(output["origin_zyx"], expected[1], TOLERANCE_TIFF_UM):
        _fail("L5", f"{LABEL_FILE} has spacing {output['spacing_zyx']}, origin {output['origin_zyx']} and shape "
                    f"{output['shape_zyx']}; the label scale's spacing and the territories' origin give "
                    f"{expected[0]}, {expected[1]} and {expected[2]}")
    # L2 to L4: the values, against the sample's territories, the instance_id map and assembly_status.
    relative, sha256, _ = sample_territory_source(assembly)
    territory = read_sample_territory(folder / relative, sha256)
    cell_ids, ids = retained_ids(assembly.cells)
    mapped = retained_image(territory[0], cell_ids, ids)
    image = _label_image(folder / LABEL_FILE)[0]
    ids_input = {int(v) for v in np.unique(mapped) if v > 0}
    ids_output = {int(v) for v in np.unique(image) if v > 0}
    # L3: at every pixel showing a cell, the territories sampled at its centre by floor(c + 1/2) hold that cell.
    index = [np.floor((output["origin_zyx"][a] + np.arange(output["shape_zyx"][a]) * output["spacing_zyx"][a]
                       - float(origin[a])) / float(spacing[a]) + 0.5).astype(np.int64) for a in (1, 2)]
    y, x = np.nonzero(image)
    iy, ix = index[0][y], index[1][x]
    ok = (iy >= 0) & (iy < mapped.shape[0]) & (ix >= 0) & (ix < mapped.shape[1])
    sampled = np.zeros(len(y), np.uint32)
    sampled[ok] = mapped[iy[ok], ix[ok]]
    if np.any(sampled != image[y, x]):
        bad = int(np.count_nonzero(sampled != image[y, x]))
        _fail("L3", f"{bad} pixels of {LABEL_FILE} show a cell that the sample's territories {relative}, sampled at "
                    "the pixel centre by floor(c + 1/2), do not hold there")
    retained = {int(v) for v in ids}
    new = sorted(ids_output - retained) + sorted((ids_output & retained) - ids_input)
    if new:
        _fail("L2", f"{LABEL_FILE} holds values that are not the instance_id of a retained cell in ids_input: "
                    f"{new[:10]}")
    absent = sorted(ids_input - ids_output)
    if section.get("fovs") != [] or section.get("n_ids_input") != len(ids_input) \
            or section.get("n_ids_output") != len(ids_output) or section.get("absent_ids") != absent \
            or section.get("n_absent_ids") != len(absent):
        _fail("L4", f"the labels section records {section.get('n_ids_input')} IDs in, {section.get('n_ids_output')} "
                    f"out and absent {section.get('absent_ids')}; recomputed from the sample's territories "
                    f"{len(ids_input)}, {len(ids_output)} and {absent}")


# --- SpatialData ---------------------------------------------------------------------------------------------

def _values(series):
    """A column as Python values, NaN and every null as None, for exact comparison (NaN equals NaN, null null)."""
    out = []
    for value in series.astype(object).tolist():
        if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
            out.append(None)
        elif isinstance(value, np.generic):
            out.append(value.item())
        else:
            out.append(value)
    return out


def _same_columns(condition, what, got, source, columns):
    for column in columns:
        if _values(got[column]) != _values(source[column]):
            _fail(condition, f"{what}: column {column!r} differs from its source")


def _keyed(frame, keys):
    return [tuple(v) for v in zip(*(_values(frame[k]) for k in keys))]


def _reorder(condition, what, got, source, keys):
    """``got`` reordered to the rows of ``source``, joined on ``keys``; each source row exactly once."""
    index = {}
    for i, key in enumerate(_keyed(got, keys)):
        if key in index:
            _fail(condition, f"{what}: key {key} occurs twice")
        index[key] = i
    wanted = _keyed(source, keys)
    if sorted(index, key=str) != sorted(wanted, key=str):
        _fail(condition, f"{what}: the rows joined on {list(keys)} are not the source rows exactly once")
    return got.iloc[[index[k] for k in wanted]].reset_index(drop=True)


def _sd_reasons(entry, record, assembly, counts):
    """Recompute an omitted element's reason (S1)."""
    from ._spatialdata import LABELS, POINTS, SHAPES
    reason, name, kind = entry.get("reason"), entry["name"], entry["kind"]
    if kind == "image":
        status = part_status(record, "fuse")
        expected = {"off": "not_requested", "not_run": "part_not_run", "empty": "no_included_fov"}.get(status)
        if status == "empty" and (record.get("fusion") or {}).get("reason") != "no_included_fov":
            expected = None
        return reason == expected
    if name == LABELS:
        status, labels = part_status(record, "fuse_labels"), record.get("labels") or {}
        if status == "written":
            return reason == "empty" and int(labels.get("n_ids_output") or 0) == 0
        if status == "empty":
            return reason == labels.get("reason") and (reason != "no_cells" or assembly.mode == "molecules_only")
        return reason == {"off": "not_requested", "not_run": "part_not_run"}.get(status)
    population = {POINTS: counts["placed_molecules"], SHAPES: counts["placed_cells"], "cells": counts["placed_cells"],
                  "expression": counts["counted"], "cells_unplaced": len(assembly.cells) - counts["placed_cells"],
                  "molecules_unplaced": len(assembly.molecules) - counts["placed_molecules"]}[name]
    return reason == "empty" and population == 0


def _check_spatialdata(folder, record):
    """S1 to S8 for the store of the ``spatialdata`` section, read back with ``spatialdata.read_zarr``."""
    import pandas as pd

    from starfinder.assembly import load_assembly

    from ._spatialdata import (
        CELL_LINK,
        FORMATS,
        INDEX_RENAMES,
        INSTANCE_KEY,
        KINDS,
        LABELS,
        POINTS,
        REGION,
        REGION_KEY,
        SHAPES,
        SPATIALDATA_ATTRS,
        STORE,
        TABLE_ELEMENTS,
        cell_links,
        equivalent_radii,
        export_counts,
        instance_ids,
        placed_cells,
        placed_molecules,
    )
    section, status = record.get("spatialdata"), part_status(record, "export_spatialdata")
    store_path = folder / STORE
    if status in ("off", None, "not_run", "failed") or not isinstance(section, dict):
        if status in ("off", "not_run") and store_path.exists():
            _fail("S1", f"the spatialdata part is {status} but {STORE} exists")
        return
    entries = section.get("elements") or []
    assembly = load_assembly(folder)
    cells, molecules = assembly.cells, assembly.molecules
    # The source columns of the molecule elements are those of the saved table: cell_namespace among them, unless the
    # table was saved before the column existed and load_assembly filled it (then the store was written without it).
    saved = ((assembly.record.get("files") or {}).get("molecules") or {}).get("dtypes") or {}
    if "cell_namespace" in molecules and "cell_namespace" not in saved:
        molecules = molecules.drop(columns="cell_namespace")
    placed_c, placed_m = placed_cells(cells), placed_molecules(molecules)
    counted = int((cells["assembly_status"].eq("retained") & cells["status"].eq("kept")).fillna(False).sum()) \
        if len(cells) else 0
    counts = {"placed_cells": int(placed_c.sum()), "placed_molecules": int(placed_m.sum()), "counted": counted}
    # S1: elements, omissions and format.
    names = [e["name"] for e in entries]
    fixed = [POINTS, SHAPES, *TABLE_ELEMENTS, LABELS]
    if sorted(n for n in names if n in fixed) != sorted(fixed) or len(set(names)) != len(names):
        _fail("S1", f"the section lists the elements {names}; every fixed element must be listed once: {fixed}")
    fusion_outputs = [o for o in (record.get("fusion") or {}).get("outputs") or []]
    if any(o.get("format") == "ome_zarr" for o in fusion_outputs) or list(folder.glob("*.ome.zarr")):
        _fail("S1", "a standalone OME-Zarr exists beside the SpatialData store")
    if part_status(record, "fuse") == "written":
        expected = sorted(f"{o['image']}_{o['dimensionality']}" for o in fusion_outputs)
        images = sorted(e["name"] for e in entries if e["kind"] == "image" and e["status"] == "written")
        if images != expected or any(e["kind"] == "image" and e["status"] != "written" for e in entries):
            _fail("S1", f"the image elements {images} are not one per fused image {expected}")
    for entry in entries:
        if entry["status"] == "omitted" and not _sd_reasons(entry, record, assembly, counts):
            _fail("S1", f"the element {entry['name']} is omitted with reason {entry.get('reason')!r}, which does not "
                        "hold when recomputed")
        if entry["status"] not in ("written", "omitted"):
            _fail("S1", f"the element {entry['name']} has status {entry['status']!r}")
    written = {e["name"]: e for e in entries if e["status"] == "written"}
    if not written:
        if section.get("store") is not None or store_path.exists() or status != "empty":
            _fail("S1", "every element is omitted, so the section must record store null and no sample.zarr exist")
        return
    if section.get("store") != STORE or not store_path.is_dir():
        _fail("S1", f"the section records store {section.get('store')!r}; {STORE} must exist")
    import spatialdata
    import zarr
    from spatialdata.transformations import get_transformation
    fmt = section["format"]
    with pd.option_context("future.infer_string", False):
        sdata = spatialdata.read_zarr(store_path)
        found = {name: kind for kind, plural in KINDS.items() for name in getattr(sdata, plural).keys()}
        if found != {n: e["kind"] for n, e in written.items()}:
            _fail("S1", f"the store holds {found}; the section records as written "
                        f"{ {n: e['kind'] for n, e in written.items()} }")
        if len({n.lower() for n in found}) != len(found):
            _fail("S1", f"two element names of the store are equal ignoring case: {sorted(found)}")
        group = zarr.open_group(str(store_path), mode="r")
        if group.metadata.zarr_format != fmt["zarr_format"] or fmt["raster_version"] != \
                FORMATS[fmt["zarr_format"]]["raster_version"]:
            _fail("S1", f"the store is Zarr format {group.metadata.zarr_format}; the section {fmt}")
        for name, entry in written.items():
            if entry["kind"] not in ("image", "labels"):
                continue
            attrs = dict(group[f"{KINDS[entry['kind']]}/{name}"].attrs)
            version = (attrs.get("ome") or {}).get("version") or attrs.get("multiscales", [{}])[0].get("version")
            if version != fmt["raster_version"]:
                _fail("S1", f"the raster group {name} has version {version!r}, not {fmt['raster_version']!r}")
        ids = instance_ids(cells)
        links = cell_links(assembly)
        # S2: points.
        if POINTS in written:
            points = sdata.points[POINTS].compute().reset_index(drop=True)
            if any(c in points.columns for c in ("sample_x", "sample_y", "sample_z")):
                _fail("S2", "a sample_x/y/z column remains in molecules")
            source = molecules[placed_m].reset_index(drop=True)
            points = _reorder("S2", POINTS, points, source, ("spot_namespace", "spot_id"))
            for axis in "xyz":
                if not np.array_equal(points[axis].to_numpy(np.float64), source[f"sample_{axis}"].to_numpy(np.float64)):
                    _fail("S2", f"molecules {axis} differs from sample_{axis}")
            for column in source.columns:
                if column in ("sample_x", "sample_y", "sample_z"):
                    continue
                if INDEX_RENAMES.get(column, column) not in points.columns:
                    _fail("S2", f"molecules has no column {INDEX_RENAMES.get(column, column)!r} for its source "
                                f"{column!r}")
                if _values(points[INDEX_RENAMES.get(column, column)]) != _values(source[column]):
                    _fail("S2", f"molecules: column {INDEX_RENAMES.get(column, column)!r} differs from its source "
                                f"{column!r}")
            if _values(points[CELL_LINK]) != _values(pd.Series(links[placed_m])):
                _fail("S5", "a cell_instance_id of molecules is not its cell's instance_id")
            if {str(c): str(d) for c, d in points.dtypes.items()} != written[POINTS]["dtypes"]:
                _fail("S2", "the column dtypes of molecules differ from those the section records")
        # S3 and S4: tables.
        tables = {}
        for name in TABLE_ELEMENTS:
            if name not in written:
                continue
            table = sdata.tables[name]
            tables[name] = table
            dtypes = {"obs": {str(c): str(d) for c, d in table.obs.dtypes.items()},
                      "var": {str(c): str(d) for c, d in table.var.dtypes.items()}}
            if dtypes != written[name]["dtypes"]:
                _fail("S3", f"the column dtypes of {name} differ from those the section records")
        for name, rows, column, added, keys in (
                ("cells", placed_c, INSTANCE_KEY, [REGION_KEY, INSTANCE_KEY], ("cell_namespace", "cell_id")),
                ("cells_unplaced", ~placed_c, INSTANCE_KEY, [INSTANCE_KEY], ("cell_namespace", "cell_id")),
                ("molecules_unplaced", ~placed_m, CELL_LINK, [CELL_LINK], ("spot_namespace", "spot_id"))):
            if name not in tables:
                if rows.any():
                    _fail("S3", f"{name} is omitted but its population has rows")
                continue
            source = (cells if name != "molecules_unplaced" else molecules)[rows].reset_index(drop=True)
            obs = _reorder("S3", name, tables[name].obs.reset_index(drop=True), source, keys)
            if list(obs.columns) != list(source.columns) + added:
                _fail("S4", f"{name} has the obs columns {list(obs.columns)}; its source's and {added}")
            _same_columns("S3", name, obs, source, source.columns)
            expected = ids[rows] if column == INSTANCE_KEY else pd.Series(links[rows])
            if _values(obs[column]) != _values(pd.Series(expected)):
                _fail("S5", f"{name}: {column} is not the instance_id of its cell")
            if column == INSTANCE_KEY and obs[INSTANCE_KEY].dtype != np.uint32:
                _fail("S4", f"{name}: instance_id is {obs[INSTANCE_KEY].dtype}, not uint32")
            uns = dict(tables[name].uns)
            if name == "cells":
                if uns != {"spatialdata_attrs": SPATIALDATA_ATTRS}:
                    _fail("S4", f"cells has uns {uns}")
                region = obs[REGION_KEY]
                if not isinstance(region.dtype, pd.CategoricalDtype) or set(region.astype(object)) != {REGION}:
                    _fail("S4", "cells: region is not categorical with every value cell_centroids")
            elif uns or tables[name].obsm or tables[name].layers:
                _fail("S4", f"{name} carries uns, obsm or layers keys its source lacks")
        # expression against sample.h5ad
        import anndata
        source = anndata.read_h5ad(folder / assembly.record["files"]["anndata"]["path"])
        if "expression" not in tables:
            if source.n_obs:
                _fail("S3", "expression is omitted but sample.h5ad has rows")
        else:
            table = tables["expression"]
            if list(table.obs_names) != list(source.obs_names) or list(table.var_names) != list(source.var_names):
                _fail("S3", "expression rows or variables differ from sample.h5ad")
            if not np.array_equal(np.asarray(table.X), np.asarray(source.X), equal_nan=True) \
                    or sorted(table.layers) != sorted(source.layers) \
                    or any(not np.array_equal(np.asarray(table.layers[k]), np.asarray(source.layers[k]), equal_nan=True)
                           for k in source.layers) \
                    or sorted(table.obsm) != sorted(source.obsm) \
                    or any(not np.array_equal(np.asarray(table.obsm[k]), np.asarray(source.obsm[k]), equal_nan=True)
                           for k in source.obsm):
                _fail("S3", "expression differs from sample.h5ad in X, layers or obsm")
            if list(table.var.columns) != list(source.var.columns):
                _fail("S3", "expression var columns differ from sample.h5ad")
            _same_columns("S3", "expression var", table.var, source.var, source.var.columns)
            if list(table.obs.columns) != list(source.obs.columns) + [REGION_KEY, INSTANCE_KEY]:
                _fail("S4", f"expression has the obs columns {list(table.obs.columns)}")
            _same_columns("S3", "expression", table.obs, source.obs, source.obs.columns)
            uns = dict(table.uns)
            if set(uns) != set(source.uns) | {"spatialdata_attrs"} or uns["spatialdata_attrs"] != SPATIALDATA_ATTRS \
                    or any(not _same_value(uns[k], source.uns[k]) for k in source.uns):
                _fail("S3", "expression uns differs from sample.h5ad's or lacks spatialdata_attrs")
            by_name = dict(zip(cells["obs_name"].astype(object), range(len(cells))))
            rows = [by_name[str(n)] for n in table.obs_names]
            kept = cells.iloc[rows]
            if not (kept["assembly_status"].eq("retained") & kept["status"].eq("kept")).all() or \
                    not placed_c[rows].all():
                _fail("S5", "an expression row is not a cells row with assembly_status retained and status kept")
            if _values(table.obs[INSTANCE_KEY]) != [int(v) for v in ids[rows]]:
                _fail("S5", "expression instance_id is not its cell's")
        # S5: cell_centroids, links and accounting.
        if (SHAPES in written) != ("cells" in written):
            _fail("S5", "cells and cell_centroids are not written or omitted together")
        if SHAPES in written:
            shapes = sdata.shapes[SHAPES]
            rows = cells[placed_c]
            if [int(v) for v in shapes.index] != [int(v) for v in ids[placed_c]] or \
                    [int(v) for v in shapes.index] != _values(tables["cells"].obs[INSTANCE_KEY]):
                _fail("S5", "the index of cell_centroids is not the instance_id of cells")
            x, y = shapes.geometry.x.to_numpy(), shapes.geometry.y.to_numpy()
            if not (np.array_equal(x, rows["sample_ownership_centroid_x"].to_numpy(np.float64))
                    and np.array_equal(y, rows["sample_ownership_centroid_y"].to_numpy(np.float64))):
                _fail("S5", "a circle's centre is not its cell's ownership centroid")
            from ._spatialdata import _fov_spacing
            radii = equivalent_radii(rows, _fov_spacing(assembly))
            if not np.allclose(shapes["radius"].to_numpy(np.float64), radii, rtol=1e-12, atol=0):
                _fail("S5", "a circle's radius is not its cell's equivalent radius")
        recount = export_counts(assembly)
        if recount != section["counts"]:
            _fail("S5", "the counts recounted from the sample tables differ from the section's")
        reference = assembly.record.get("counts") or {}
        for population in ("cells", "molecules"):
            mine = section["counts"][population]["assembly"]
            theirs = reference.get(population) or {}
            if mine != {"retained": theirs.get("retained"), "excluded": theirs.get("excluded")}:
                _fail("S5", f"the {population} counts differ from the assembly section's")
        # S4 (attrs), S6 and S7: points, shapes and rasters.
        cs = section["coordinate_system"]
        for name, entry in written.items():
            kind = entry["kind"]
            element = {"points": sdata.points, "shapes": sdata.shapes, "image": sdata.images,
                       "labels": sdata.labels}.get(kind, {}).get(name)
            if element is None:
                continue
            if set(element.attrs) != {"transform"}:
                _fail("S4", f"{name} carries the attributes {sorted(element.attrs)}, not the transform only")
            transform = get_transformation(element, cs)
            if kind in ("points", "shapes"):
                if type(transform).__name__ != "Identity":
                    _fail("S6", f"{name} has the transform {transform}, not the identity")
                continue
            _check_raster(folder, name, entry, element, transform, record, cs, spatialdata)
    # S8: hashes.
    from ._record import directory_files
    output = (section.get("outputs") or [{}])[0]
    if output.get("path") != STORE or directory_files(store_path) != output.get("files"):
        _fail("S8", f"the files of {STORE} or their SHA-256 differ from the section's")


def _same_value(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return set(a) == set(b) and all(_same_value(a[k], b[k]) for k in a)
    return a == b


def _check_raster(folder, name, entry, element, transform, record, cs, spatialdata):
    """S6 (transform, extent) and S7 (values, dtype, one level) for one image or labels element."""
    import tifffile
    section = record["fusion"] if entry["kind"] == "image" else record["labels"]
    output = next(o for o in section["outputs"] if o["path"] == entry["source"])
    spacing, origin, shape = output["spacing_zyx"], output["origin_zyx"], output["shape_zyx"]
    axes = ["y", "x"] if output["axes"] == "YX" else ["z", "y", "x"]
    matrix = transform.to_affine_matrix(input_axes=tuple(axes), output_axes=tuple(axes))
    for i, a in enumerate(axes):
        k = "zyx".index(a)
        if abs(matrix[i, i] - spacing[k]) > TOLERANCE_TIFF_UM or \
                abs(matrix[i, -1] - (origin[k] - spacing[k] / 2)) > TOLERANCE_TIFF_UM:
            _fail("S6", f"{name}: the transform on {a} is not Scale({spacing[k]}) then Translation(O − S/2)")
    recorded = entry["transform"]
    for k, a in enumerate("zyx"):
        if abs(recorded["scale"][a] - spacing[k]) > TOLERANCE_TIFF_UM or \
                abs(recorded["translation"][a] - (origin[k] - spacing[k] / 2)) > TOLERANCE_TIFF_UM:
            _fail("S6", f"{name}: the recorded transform on {a} is not the file's")
    extent = spatialdata.get_extent(element, coordinate_system=cs)
    for a in axes:
        k = "zyx".index(a)
        lo, hi = origin[k] - spacing[k] / 2, origin[k] + (shape[k] - 0.5) * spacing[k]
        if abs(extent[a][0] - lo) > TOLERANCE_TIFF_UM or abs(extent[a][1] - hi) > TOLERANCE_TIFF_UM:
            _fail("S6", f"{name}: the extent on {a} is {extent[a]}, not [{lo}, {hi})")
    if type(element).__name__ != "DataArray":
        _fail("S7", f"{name} has more than one level")
    data = element.data[0] if entry["kind"] == "image" else element.data
    with tifffile.TiffFile(folder / output["path"]) as tif:
        n_z = shape[0]
        for i in range(n_z if len(axes) == 3 else 1):
            plane = tif.series[0].asarray(key=i) if n_z > 1 else tif.series[0].asarray()
            got = np.asarray(data[i] if len(axes) == 3 else data)
            if got.dtype != plane.dtype or not np.array_equal(got, plane):
                _fail("S7", f"{name}: plane {i} differs from {output['path']} in value or dtype")


def check_export(folder) -> None:
    """Check an export folder against its record ``export.json``; ``ValueError`` names the first failed condition.

    R1: the record parses, every part has a known status, no part is still
    the entry ``not_run`` with the reason ``export_not_finished`` (an export
    stopped before the part ended), ``complete`` holds exactly when no part
    is ``not_run`` or ``failed``, and the ``fusion``
    section's recorded SHA-256 of the ``assembly`` section is that section's
    canonical SHA-256. R2: every listed file exists with its SHA-256, every
    file of the folder is listed, and a part ``off``, ``not_run`` or ``empty``
    lists none. R3: a written or reused assembly passes ``load_assembly``; with
    the part off no table or ``sample.h5ad`` exists. For each fused file: the
    OME-TIFF layout checks 1 to 3 or the OME-Zarr layout checks 1 to 3 (a
    region inside one chunk touches that chunk only, chunk keys counted), and
    the recorded geometry against the fusion grid (F1); the dtype (F2); 0 at
    every voxel no included FOV covers (F3); equal Y and X grids of a
    projection and a stack (F7); ``empty`` exactly without an included FOV
    (F8); the OME-Zarr's output level equal to the OME-TIFF's (F9). For the
    label image: the OME-TIFF layout checks and uint32 (L1); every nonzero
    value the ``instance_id`` of a retained cell in ``ids_input`` (L2); at
    every voxel showing a cell, that cell's FOV's territory, sampled at the
    voxel centre by ``floor(c + ½)``, holding it (L3); the recorded counts of
    ``ids_input`` and ``ids_output`` and ``absent_ids`` equal to their
    recomputation from the territory files, the cell table and the file
    (L4); the grid equal to the formulas within 1e-9 µm (L5). R1 also checks
    the ``labels`` and ``spatialdata`` sections' recorded SHA-256 of the
    ``assembly`` section, and the ``spatialdata`` section's of the ``fusion``
    and ``labels`` sections. For the SpatialData store, read back with
    ``spatialdata.read_zarr`` (the ``spatialdata`` extra): S1 elements,
    omissions with their reasons recomputed, ``store: null`` exactly when
    every element is omitted, no standalone OME-Zarr, the Zarr format and the
    raster version strings; S2 points against the sample molecule table,
    ``cell_namespace`` among its columns; S3
    tables against the sample cell table and ``sample.h5ad``, with the
    recorded dtypes; S4 the export's metadata; S5 links, circles and counts;
    S6 transforms and extents; S7 values against the OME-TIFF files; S8 the
    SHA-256 of every store file.

    On the route after fusion (docs/sample-segmentation-contract.md, "The
    record"), R1 includes the part ``segment``, and when that part is
    ``written`` or ``reused``: T1 the ``segmentation`` section's label files
    with their SHA-256, uint32, the OME-TIFF layout checks and the recorded
    grid, and, when the ``fusion`` section is ``written``, each input image's
    fused projection with the recorded SHA-256 and grid within 1e-9 µm (with
    the fusion ``off``, ``not_run`` or ``empty`` the label files alone); with
    the assembly ``written`` or ``reused``, T5 the ``assembly`` and
    ``labels`` sections' ``segmentation_files`` equal to those label files
    (and ``territories.ome.tif``); for an assembly of that route, T2 the cell
    run's label values exactly the ``cell_id``s of the cell table, one row
    each, with the run's namespace, T3 every retained molecule's ``cell_id``
    the territory value at its pixel ``floor(j + ½)`` and a molecule outside
    the grid excluded ``outside_grid``, and T4 every cell's ``fov_id`` and
    ``owner_fov_id`` by the rule of the cells, recomputed from the ownership
    the ``assembly`` section records. A label image of the rule
    ``segmentation`` is checked against the sample's territories (L3) and
    the grid of the label scale's spacing and the territories' origin (L5).

    Parameters
    ----------
    folder : str or Path
        The export folder ``<OUTPUT>/export/<sample>/``.

    Raises
    ------
    ValueError
        The first condition that fails, named (``R1`` … ``F9``, ``T1`` …
        ``T5``, ``L1`` … ``L5``, ``S1`` … ``S8``).
    """
    folder = Path(folder)
    record = _r1(folder)
    _r2(folder, record)
    _r3(folder, record)
    _check_fusion(folder, record)
    _check_segmentation(folder, record)
    _check_labels(folder, record)
    _check_spatialdata(folder, record)
