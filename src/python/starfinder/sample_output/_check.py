"""``check_export``: the read-back checks of an export folder (docs/sample-output-contract.md, "Read-back checks").

R1 to R3 (the record and the folder) and, for each fused image of the ``fusion`` section,
F1 (the OME-TIFF and OME-Zarr layout checks and conditions 1 to 3 of "Sample position to
image position"), F2 (dtype), F3 (uncovered voxels), F7 (projection and stack grids), F8
(no included FOV) and F9 (OME-Zarr equal to OME-TIFF). F4 to F6 are fixture checks of the
tests. The label and SpatialData checks belong to their parts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ._record import (
    EXPORT_FILE,
    FORMAT_VERSION,
    NO_FILES,
    PARTS,
    SECTIONS,
    STAGE,
    STATUSES,
    is_complete,
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
    if record.get("complete") is not is_complete(record):
        _fail("R1", f"complete is {record.get('complete')!r}, but the part statuses give {is_complete(record)}")
    fusion = record.get("fusion") or {}
    recorded = fusion.get("assembly_sha256")
    if recorded is not None:
        current = section_sha256(record.get("assembly"))
        if recorded != current:
            _fail("R1", f"the fusion section records the assembly section's SHA-256 {recorded}, but the assembly "
                        f"section's is {current}")
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


def _check_tiff(folder, name, output, section):
    from starfinder.io.tiff import METADATA_NAMESPACE

    from ._tiff import ANNOTATION_NAMESPACE, read_ome_layout
    layout = read_ome_layout(folder / output["path"])
    one_plane = output["shape_zyx"][0] == 1
    axes = "YX" if one_plane else "ZYX"
    # Layout check 1.
    if not (layout["is_bigtiff"] and layout["is_ome"]):
        _fail("F1", f"{name} is not a BigTIFF OME-TIFF")
    if layout["n_series"] != 1 or layout["axes"] != axes or layout["channels"] != 1:
        _fail("F1", f"{name} has {layout['n_series']} series of axes {layout['axes']} and {layout['channels']} "
                    f"channels; expected one series of axes {axes} and one channel")
    # Layout check 2.
    shape = output["shape_zyx"][1:] if one_plane else output["shape_zyx"]
    if layout["shape"] != list(shape) or layout["dtype"] != output["dtype"]:
        _fail("F1", f"{name} has shape {layout['shape']} and dtype {layout['dtype']}; the record {shape} and "
                    f"{output['dtype']}")
    if layout["pixels"].get("Type") != OME_TYPES.get(output["dtype"]):
        _fail("F1", f"{name}: the OME pixel type {layout['pixels'].get('Type')} is not that of {output['dtype']}")
    if layout["tile_yx"] != output["tile_yx"] or layout["tile_yx"] != [512, 512] \
            or layout["compression"] not in ("adobe_deflate", "deflate", "zlib"):
        _fail("F1", f"{name} has tiles {layout['tile_yx']} and compression {layout['compression']}; expected "
                    "512 × 512 tiles and zlib")
    recorded_levels = [level["shape_zyx"][1:] if one_plane else level["shape_zyx"] for level in output["levels"]]
    if layout["level_shapes"] != [list(s) for s in recorded_levels]:
        _fail("F1", f"{name} has SubIFD levels {layout['level_shapes']}; the record {recorded_levels}")
    # Layout check 3 and condition 1.
    spacing, origin = output["spacing_zyx"], output["origin_zyx"]
    pixels = layout["pixels"]
    physical = [float(pixels.get(f"PhysicalSize{a}", "nan")) for a in "ZYX"]
    units = [pixels.get(f"PhysicalSize{a}Unit") for a in "ZYX"]
    if not _close(physical, spacing, TOLERANCE_TIFF_UM) or set(units) != {"µm"}:
        _fail("F1", f"{name}: PhysicalSize {physical} {units} is not the recorded spacing {spacing} µm")
    n_z = output["shape_zyx"][0]
    if len(layout["planes"]) != n_z:
        _fail("F1", f"{name} has {len(layout['planes'])} Plane elements for {n_z} planes")
    for i, plane in enumerate(layout["planes"]):
        position = [float(plane.get(f"Position{a}", "nan")) for a in "ZYX"]
        expected = [origin[0] + i * spacing[0], origin[1], origin[2]]
        if not _close(position, expected, TOLERANCE_TIFF_UM):
            _fail("F1", f"{name}: plane {i} is at {position}, not at {expected}")
    stored = layout["annotations"].get(METADATA_NAMESPACE)
    if not stored or not _close(stored["spacing_zyx"], spacing, TOLERANCE_TIFF_UM) \
            or not _close(stored["origin_zyx"], origin, TOLERANCE_TIFF_UM):
        _fail("F1", f"{name}: the stored ImageMetadata {stored} does not have the recorded spacing and origin")
    if layout["annotations"].get(ANNOTATION_NAMESPACE) != output.get("annotation"):
        _fail("F1", f"{name}: the {ANNOTATION_NAMESPACE} annotation differs from the record")
    for level in output["levels"]:
        f = level["factors_zyx"]
        if not (_close(level["spacing_zyx"], [a * b for a, b in zip(f, spacing)], TOLERANCE_TIFF_UM)
                and _close(level["origin_zyx"], [o + s * (k - 1) / 2 for k, s, o in zip(f, spacing, origin)],
                           TOLERANCE_TIFF_UM)):
            _fail("F1", f"{name}: level {level['level']} geometry is not S·f and O + S·(f − 1)/2")


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


def check_export(folder) -> None:
    """Check an export folder against its record ``export.json``; ``ValueError`` names the first failed condition.

    R1: the record parses, every part has a known status, ``complete`` holds
    exactly when no part is ``not_run`` or ``failed``, and the ``fusion``
    section's recorded SHA-256 of the ``assembly`` section is that section's
    canonical SHA-256. R2: every listed file exists with its SHA-256, every
    file of the folder is listed, and a part ``off``, ``not_run`` or ``empty``
    lists none. R3: a written or reused assembly passes ``load_assembly``; with
    the part off no table or ``sample.h5ad`` exists. For each fused file: the
    OME-TIFF layout checks 1 to 3 or the OME-Zarr layout checks 1 and 2, and
    the recorded geometry against the fusion grid (F1); the dtype (F2); 0 at
    every voxel no included FOV covers (F3); equal Y and X grids of a
    projection and a stack (F7); ``empty`` exactly without an included FOV
    (F8); the OME-Zarr's output level equal to the OME-TIFF's (F9).

    Parameters
    ----------
    folder : str or Path
        The export folder ``<OUTPUT>/export/<sample>/``.

    Raises
    ------
    ValueError
        The first condition that fails, named (``R1`` … ``F9``).
    """
    folder = Path(folder)
    record = _r1(folder)
    _r2(folder, record)
    _r3(folder, record)
    _check_fusion(folder, record)
