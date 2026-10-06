"""External-mask import into the label contract (D3; docs/segmentation-contract.md)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from starfinder.image import IncompatibleGeometryError

from ._labels import (FORMAT_VERSION, GEOMETRIES, ReferenceGrid, SegmentationResult, _check_geometry,
    _check_namespace, _check_target, _grid_record, _namespace_ids, array_sha256, to_label_dtype)


@dataclass(frozen=True)
class LabelImportConfig:
    """An imported mask as a run of a segmentation plan; not a segmentation method.

    ``path`` is a YX or ZYX integer TIFF readable by :func:`starfinder.io.load_volume`;
    ``target`` is ``nucleus`` or ``cell``; ``geometry`` ``None`` derives ``plane``
    (Z=1) or ``volume`` (Z>1), or names ``extended``; ``relabel`` ``False`` keeps the
    file's values. See :func:`import_labels`.
    """

    path: str
    target: str
    geometry: str | None = None
    relabel: bool = False

    def __post_init__(self):
        if not isinstance(self.path, (str, os.PathLike)) or not str(self.path):
            raise TypeError("path must be a nonempty path")
        object.__setattr__(self, "path", str(self.path))
        _check_target(self.target)
        if self.geometry is not None and self.geometry not in GEOMETRIES:
            raise ValueError(f"geometry must be None, volume, plane or extended; got {self.geometry!r}")
        if not isinstance(self.relabel, bool):
            raise TypeError("relabel must be a bool")


def _software():
    from starfinder.dataset._run_record import _code, _environment
    return {"code": _code(), "environment": _environment()}


def import_labels(path: Path | str, *, grid: ReferenceGrid, target: str, geometry: str | None = None,
                  relabel: bool = False, label_namespace: str) -> SegmentationResult:
    """Import a label mask made elsewhere (CellProfiler, an earlier workflow run) onto a grid.

    Validation, in order: the file exists and is read with its byte order converted
    to native; a YX file is read as 1×Y×X and needs a grid with Z=1 (for a volume
    FOV, ``grid.projected()``); the dtype is integer; no value is negative; the
    shape equals ``grid.shape_zyx``; stored ``starfinder_metadata`` must equal
    ``grid.metadata``, and without it ``grid.metadata`` is taken as a declaration
    (``metadata_source`` ``"declared"``); values above 2**32 - 1 raise; the geometry,
    given or derived, must agree with the shape. With ``relabel=True`` the positive
    values are mapped to 1…n in increasing order and the map is recorded.

    The result's record has ``methods: []`` and an ``import`` entry: the path, the
    file's SHA-256, the source array's SHA-256, dtype and shape,
    ``metadata_source``, ``relabel`` and its map, the target and the geometry. An
    all-zero mask is a valid result with outcome ``empty``.

    Raises
    ------
    FileNotFoundError
        The file does not exist.
    TypeError
        The mask is a float or boolean image, or grid is not a ReferenceGrid.
    IncompatibleGeometryError
        The array's shape differs from grid.shape_zyx.
    ValueError
        A negative value, a value above 2**32 - 1, metadata different from the
        grid's, an unknown target or geometry, or a geometry that disagrees
        with the shape.
    """
    config = LabelImportConfig(path, target, geometry, relabel)
    if not isinstance(grid, ReferenceGrid):
        raise TypeError("grid must be a ReferenceGrid")
    _check_namespace(label_namespace)
    from starfinder.io import load_volume

    path = Path(config.path)
    if not path.is_file():
        raise FileNotFoundError(f"label file not found: {path}")
    with tifffile.TiffFile(path) as tif:
        dtype, file_shape = tif.series[0].dtype, tuple(tif.series[0].shape)
    if dtype.kind not in "iu":
        raise TypeError(f"{path} holds {dtype} values; a label mask must have an integer dtype")
    loaded = load_volume(path)
    source = loaded.image
    if not source.dtype.isnative:
        source = source.astype(source.dtype.newbyteorder("="))
    if source.dtype.kind == "i" and source.size and source.min() < 0:
        raise ValueError(f"{path} holds negative values; minimum {int(source.min())}")
    if source.shape != grid.shape_zyx:
        plane = " (a YX file is read as 1×Y×X and needs a Z=1 grid)" if len(file_shape) == 2 else ""
        raise IncompatibleGeometryError(f"{path} has shape {source.shape}, the grid {grid.shape_zyx}{plane}")
    if loaded.diagnostics["metadata_source"] == "stored":
        if loaded.metadata != grid.metadata:
            raise ValueError(f"{path} stores metadata {loaded.metadata!r}, which differs from the grid's "
                             f"{grid.metadata!r}")
        metadata_source = "stored"
    else:
        metadata_source = "declared"
    labels = to_label_dtype(source)
    geometry = config.geometry or ("plane" if labels.shape[0] == 1 else "volume")
    _check_geometry(geometry, labels.shape[0])
    relabel_map = None
    if config.relabel:
        values, inverse = np.unique(labels, return_inverse=True)
        new = inverse.reshape(labels.shape) + (0 if values[0] == 0 else 1)
        positive = values[values > 0]
        relabel_map = [[int(old), i] for i, old in enumerate(positive, start=1)]
        labels = to_label_dtype(new)
    n_labels, max_label = int(np.count_nonzero(np.unique(labels))), int(labels.max())
    record = {
        "format_version": FORMAT_VERSION, "stage": "segmentation", **_namespace_ids(label_namespace),
        "target": config.target, "geometry": geometry, "label_namespace": label_namespace,
        "grid": _grid_record(grid), "input": None, "seeds": None, "methods": [],
        "import": {"path": str(path), "file_sha256": _file_sha256(path), "sha256": array_sha256(source),
                   "source_dtype": str(source.dtype), "source_shape": list(file_shape),
                   "metadata_source": metadata_source, "relabel": config.relabel, "relabel_map": relabel_map,
                   "target": config.target, "geometry": geometry},
        "operations": [], "outcome": "ok" if max_label else "empty",
        "labels": {"dtype": str(labels.dtype), "sha256": array_sha256(labels), "n_labels": n_labels,
                   "max_label": max_label},
        "software": _software(),
    }
    return SegmentationResult(labels, grid, config.target, geometry, label_namespace, record)


def _file_sha256(path, chunk=1 << 20):
    from starfinder.dataset._run_record import _sha256
    return _sha256(path, chunk)
