"""Starfinder's OME-TIFF writer of a fused array (design/specs/sample-output.md, "OME-TIFF layout", option T1).

A file writer, not a fusion: it writes the array it is given, value for value, and
computes no voxel. BigTIFF, one series of axes ``YX`` or ``ZYX`` (each Z plane one page),
512 × 512 tiles compressed with zlib, written band by band (one row of tiles at a time)
from a lazy array, SubIFD levels given by the caller, ``PhysicalSize*`` and ``Plane``
positions in µm, and two ``CommentAnnotation``: the ``ImageMetadata`` of the output level
(namespace ``starfinder.image.ImageMetadata``, which ``load_volume`` restores) and the
sample export annotation (namespace ``ANNOTATION_NAMESPACE``). The OME-XML UUID is a
name-based UUID of the tiles and the metadata written, so the same content gives the same
bytes.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import tifffile

from starfinder._records import atomic_path, publish
from starfinder.image import ImageMetadata
from starfinder.io.tiff import METADATA_NAMESPACE, _ome_digest, _ome_uuid, _stamp_uuid

TILE = 512
COMPRESSION = "zlib"
# A format identifier that OME files carry and check_export reads back: it keeps the old
# module name after W-556 renamed starfinder.sample_output to starfinder.export.
ANNOTATION_NAMESPACE = "starfinder.sample_output"
# Written first and replaced in place by the content UUID once every tile is written.
PLACEHOLDER_UUID = "urn:uuid:00000000-0000-0000-0000-000000000000"


def _band(array, z, y0, y1):
    """Rows ``y0:y1`` of plane ``z`` (or of a 2D array), computed when lazy."""
    part = array[y0:y1] if array.ndim == 2 else array[z, y0:y1]
    return np.asarray(part.compute() if hasattr(part, "compute") else part)


def bands(shape):
    """``(z, y0, y1)`` of each band the writer computes, in writing order: plane by plane, rows of 512 (z 0 in 2D)."""
    planes = 1 if len(shape) == 2 else shape[0]
    n_y = shape[-2]
    for z in range(planes):
        for y0 in range(0, n_y, TILE):
            yield z, y0, min(y0 + TILE, n_y)


def _tiles(array, dtype):
    """The 512 × 512 tiles of an array, plane by plane and band by band (row-major), from a lazy array."""
    n_x = array.shape[-1]
    for z, y0, y1 in bands(array.shape):
        band = _band(array, z, y0, y1)
        if band.dtype != dtype:
            raise ValueError(f"a band of the fused array has dtype {band.dtype}, not {dtype}")
        for x0 in range(0, n_x, TILE):
            yield band[:, x0:x0 + TILE]


def _hashed(tiles, digest):
    """The tiles, each added to ``digest`` as it is written."""
    for tile in tiles:
        digest.update(np.ascontiguousarray(tile))
        yield tile


def plane_positions(spacing_zyx, origin_zyx, n_z) -> list[tuple[float, float, float]]:
    """``(O_z + i·S_z, O_y, O_x)`` of each plane ``i``."""
    return [(origin_zyx[0] + i * spacing_zyx[0], origin_zyx[1], origin_zyx[2]) for i in range(n_z)]


def write_ome_tiff(path, array, *, metadata: ImageMetadata, annotation: dict, levels=()) -> dict:
    """Write a fused array as an OME-TIFF; return its layout (tile shape, compression, level shapes).

    ``array`` is 2D (``YX``) or 3D (``ZYX``), a numpy or lazy (dask) array of
    uint8, uint16, uint32 or float32; ``metadata`` is the output level's
    ``ImageMetadata`` (spacing and origin in µm); ``levels`` are the SubIFD
    levels, each an array of the same dimensionality, already reduced. The
    file is written to a temporary sibling and renamed into place. Its UUID
    is a name-based UUID of the shapes, dtype, layout, metadata and tiles
    written, set in place after the last tile.
    """
    path = Path(path)
    dtype = np.dtype(array.dtype)
    axes = "YX" if array.ndim == 2 else "ZYX"
    n_z = 1 if array.ndim == 2 else array.shape[0]
    spacing, origin = metadata.spacing_zyx, metadata.origin_zyx
    positions = plane_positions(spacing, origin, n_z)
    plane = {"PositionZ": [p[0] for p in positions], "PositionY": [p[1] for p in positions],
             "PositionX": [p[2] for p in positions], "PositionZUnit": ["µm"] * n_z,
             "PositionYUnit": ["µm"] * n_z, "PositionXUnit": ["µm"] * n_z}
    if n_z == 1:
        plane = {key: value[0] for key, value in plane.items()}
    ome = {"axes": axes, "PhysicalSizeX": float(spacing[2]), "PhysicalSizeXUnit": "µm",
           "PhysicalSizeY": float(spacing[1]), "PhysicalSizeYUnit": "µm",
           "PhysicalSizeZ": float(spacing[0]), "PhysicalSizeZUnit": "µm", "Plane": plane,
           "CommentAnnotation": [{"Namespace": METADATA_NAMESPACE, "Value": json.dumps(asdict(metadata))},
                                 {"Namespace": ANNOTATION_NAMESPACE, "Value": json.dumps(annotation, sort_keys=True)}]}
    options = {"tile": (TILE, TILE), "compression": COMPRESSION, "photometric": "minisblack"}
    digest = _ome_digest({"dtype": dtype.str, "shape": list(array.shape), "options": options, "metadata": ome,
                          "level_shapes": [list(level.shape) for level in levels]})
    tmp = atomic_path(path)
    try:
        with tifffile.TiffWriter(tmp, bigtiff=True, ome=True) as tif:
            tif.write(_hashed(_tiles(array, dtype), digest), shape=tuple(array.shape), dtype=dtype,
                      metadata=dict(ome, UUID=PLACEHOLDER_UUID), subifds=len(levels) or None, **options)
            for level in levels:
                tif.write(_hashed(_tiles(level, dtype), digest), shape=tuple(level.shape), dtype=dtype,
                          subfiletype=1, metadata=None, **options)
        _stamp_uuid(tmp, PLACEHOLDER_UUID, _ome_uuid(digest))
        publish(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {"tile_yx": [TILE, TILE], "compression": COMPRESSION, "axes": axes,
            "level_shapes": [list(level.shape) for level in levels]}


def read_ome_layout(path) -> dict:
    """What a fused OME-TIFF holds: BigTIFF and OME flags, axes, shape, dtype, tiles, compression, levels,
    calibration, plane positions and the two annotations."""
    from xml.etree import ElementTree
    with tifffile.TiffFile(path) as tif:
        series = tif.series
        first = series[0]
        page = tif.pages[0]
        info = {"is_bigtiff": tif.is_bigtiff, "is_ome": tif.is_ome, "n_series": len(series), "axes": first.axes,
                "shape": list(first.shape), "dtype": str(first.dtype),
                "tile_yx": [page.tilelength, page.tilewidth], "compression": page.compression.name.lower(),
                "level_shapes": [list(level.shape) for level in first.levels[1:]],
                "level_tiles": [[level.keyframe.tilelength, level.keyframe.tilewidth] for level in first.levels[1:]],
                "level_compression": [level.keyframe.compression.name.lower() for level in first.levels[1:]]}
        xml = tif.ome_metadata
    root = ElementTree.fromstring(xml)

    def local(tag):
        return tag.rsplit("}", 1)[-1]

    pixels = next(e for e in root.iter() if local(e.tag) == "Pixels")
    info["pixels"] = dict(pixels.attrib)
    info["planes"] = [dict(e.attrib) for e in pixels if local(e.tag) == "Plane"]
    info["channels"] = sum(1 for e in pixels if local(e.tag) == "Channel")
    annotations = {}
    for element in root.iter():
        if local(element.tag) == "CommentAnnotation":
            value = next((child.text for child in element if local(child.tag) == "Value"), None)
            annotations[element.get("Namespace")] = json.loads(value) if value else None
    info["annotations"] = annotations
    return info
