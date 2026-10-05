"""The label image contract: hashes, the uint32 dtype rule, the reference grid and the result."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from starfinder.image import ImageMetadata, IncompatibleGeometryError

LABEL_DTYPE = np.dtype(np.uint32)
TARGETS = ("nucleus", "cell")
GEOMETRIES = ("volume", "plane", "extended")
FORMAT_VERSION = 1
_MAX_LABEL = int(np.iinfo(np.uint32).max)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_NAMESPACE_FIELDS = ("dataset_id", "sample_id", "fov_id", "subtile_id", "run")


def array_sha256(array) -> str:
    """SHA-256 of an array's dtype, shape and C-order bytes in native byte order.

    The convention of the segmentation golden test: ``"<dtype.str>|<shape>|"`` then
    the bytes, so arrays with equal values but different dtypes or shapes differ.
    """
    array = np.ascontiguousarray(array)
    if not array.dtype.isnative:
        array = array.astype(array.dtype.newbyteorder("="))
    digest = hashlib.sha256(f"{array.dtype.str}|{array.shape}|".encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def grid_sha256(image) -> str:
    """SHA-256 of an image's C-order bytes alone, the ``ReferenceGrid.sha256`` convention.

    No dtype or shape prefix, as ``reference_sha256`` of ``FOV.register_rounds``
    (docs/segmentation-contract.md, "The reference grid").
    """
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _json(value):
    """A JSON-native copy (tuples become lists), as the record reads back from segmentation.json."""
    return json.loads(json.dumps(value))


def to_label_dtype(labels) -> np.ndarray:
    """Convert an integer label array to C-contiguous ``uint32`` without wrapping.

    The label dtype rule of docs/segmentation-contract.md: every label image is
    ``uint32``. Values are kept as they are; the shape is not checked.

    Raises
    ------
    TypeError
        The array is not of an integer dtype (a float or boolean mask).
    ValueError
        A value is negative or above 2**32 - 1.
    """
    array = np.asarray(labels)
    if array.dtype.kind not in "iu":
        raise TypeError(f"labels must have an integer dtype; got {array.dtype}")
    if array.size:
        if array.dtype.kind == "i" and array.min() < 0:
            raise ValueError(f"labels must not be negative; minimum {int(array.min())}")
        if int(array.max()) > _MAX_LABEL:
            raise ValueError(f"labels above 2**32 - 1 do not fit uint32; maximum {int(array.max())}")
    return np.ascontiguousarray(array, dtype=LABEL_DTYPE)


def _check_target(target):
    if target not in TARGETS:
        raise ValueError(f"target must be nucleus or cell; got {target!r}")


def _check_geometry(geometry, z):
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry must be volume, plane or extended; got {geometry!r}")
    if geometry == "plane" and z != 1:
        raise ValueError(f"geometry plane needs Z=1; the labels have Z={z}")
    if geometry in ("volume", "extended") and z == 1:
        raise ValueError(f"geometry {geometry} needs Z>1; the labels have Z=1")


def _check_namespace(label_namespace):
    if not isinstance(label_namespace, str) or not label_namespace.strip():
        raise ValueError("label_namespace must be a nonempty string")


def _namespace_ids(label_namespace):
    """dataset_id, sample_id, fov_id, subtile_id and run from a JSON-list namespace; None otherwise."""
    try:
        value = json.loads(label_namespace)
    except ValueError:
        value = None
    if isinstance(value, list) and len(value) == len(_NAMESPACE_FIELDS):
        return dict(zip(_NAMESPACE_FIELDS, value))
    return dict.fromkeys(_NAMESPACE_FIELDS)


@dataclass(frozen=True)
class ReferenceGrid:
    """The grid every label image of a FOV is on: shape, geometry, source and hash.

    ``ImageMetadata`` carries no shape, so the grid holds both. ``source`` is
    ``"fov:<reference round>"`` (:meth:`starfinder.dataset.FOV.reference_grid`),
    ``"file:<path>"`` (:func:`reference_grid_from_file`) or ``"declared"`` (built by
    the caller when no molecule run exists; assign checks it later). ``sha256`` is
    the SHA-256 of the C-order bytes of the image the grid was read from, without
    a dtype or shape prefix (``None`` for a declared grid).

    Raises
    ------
    IncompatibleGeometryError
        shape_zyx is not three positive integers.
    TypeError
        metadata is not an ImageMetadata.
    ValueError
        source or sha256 has another form.
    """

    shape_zyx: tuple[int, int, int]
    metadata: ImageMetadata
    source: str
    sha256: str | None = None

    def __post_init__(self):
        shape = self.shape_zyx
        if (not isinstance(shape, (tuple, list)) or len(shape) != 3
                or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1
                       for n in shape)):
            raise IncompatibleGeometryError(f"shape_zyx must be three positive integers; got {shape!r}")
        object.__setattr__(self, "shape_zyx", tuple(int(n) for n in shape))
        if not isinstance(self.metadata, ImageMetadata):
            raise TypeError("metadata must be an ImageMetadata")
        if not isinstance(self.source, str) or not (
                self.source == "declared" or re.fullmatch(r"(fov|file):.+", self.source)):
            raise ValueError(f"source must be 'fov:<round>', 'file:<path>' or 'declared'; got {self.source!r}")
        if self.sha256 is not None and (not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256)):
            raise ValueError("sha256 must be None or 64 lowercase hexadecimal digits")

    def projected(self, *, method: str = "max") -> ReferenceGrid:
        """The Z=1 grid of a projection: shape (1, Y, X) and ``metadata.projected(method=method)``.

        The source and hash stay those of the image the grid was read from.
        """
        return ReferenceGrid((1, *self.shape_zyx[1:]), self.metadata.projected(method=method),
                             self.source, self.sha256)


def _grid_record(grid):
    """The run record's ``grid`` entry: shape, metadata, source and hash."""
    return {"shape_zyx": list(grid.shape_zyx), "metadata": _json(asdict(grid.metadata)),
            "source": grid.source, "sha256": grid.sha256}


def reference_grid_from_file(path: Path | str, *, metadata: ImageMetadata | None = None) -> ReferenceGrid:
    """The reference grid of a ZYX or YX TIFF, read with :func:`starfinder.io.load_volume`.

    The shape is the array's (a YX file gives a Z=1 grid); the metadata is the
    file's ``starfinder_metadata`` description, or ``metadata`` when the file has
    none. The natural file is ``images/ref_merged/{fovID}.tif``, which
    ``FOV.save_reference_image`` writes with the reference metadata. The source is
    ``"file:<resolved path>"`` and the hash the SHA-256 of the array's C-order
    bytes.

    Raises
    ------
    FileNotFoundError
        The file does not exist.
    TypeError
        metadata is neither None nor an ImageMetadata.
    ValueError
        The file has no stored metadata and none is given, or both are present
        and differ.
    """
    from starfinder.io import load_volume

    if metadata is not None and not isinstance(metadata, ImageMetadata):
        raise TypeError("metadata must be an ImageMetadata or None")
    loaded = load_volume(path)
    if loaded.diagnostics["metadata_source"] == "stored":
        if metadata is not None and metadata != loaded.metadata:
            raise ValueError(f"{path} stores metadata {loaded.metadata!r}, which differs from the given {metadata!r}")
        metadata = loaded.metadata
    elif metadata is None:
        raise ValueError(f"{path} has no starfinder_metadata description; pass metadata")
    return ReferenceGrid(loaded.image.shape, metadata, f"file:{Path(path).resolve()}",
                         grid_sha256(loaded.image))


@dataclass(frozen=True, eq=False)
class SegmentationResult:
    """One label image on its grid, with its target, geometry, identity scope and record.

    ``labels`` is a C-contiguous ``uint32`` ZYX array (a plane is 1×Y×X), 0 is
    background and each positive value one object; values are never relabelled.
    ``grid`` is the :class:`ReferenceGrid` the labels are on (``labels.shape ==
    grid.shape_zyx``). ``target`` is ``nucleus`` or ``cell``; ``geometry`` is
    ``volume`` (Z>1), ``plane`` (Z=1) or ``extended`` (Z>1, from
    :func:`extend_labels_through_z`). A label's identity is ``(label_namespace,
    value)``. ``record`` is the run record of docs/segmentation-contract.md and
    ``diagnostics`` holds method-specific counts and timings.

    Raises
    ------
    TypeError
        labels is not uint32, grid is not a ReferenceGrid, or record or
        diagnostics is not a mapping.
    ValueError
        labels is not a three-dimensional C-contiguous array, the target or
        geometry is unknown, the geometry disagrees with Z, or the namespace is
        empty.
    IncompatibleGeometryError
        labels.shape differs from grid.shape_zyx.
    """

    labels: np.ndarray
    grid: ReferenceGrid
    target: str
    geometry: str
    label_namespace: str
    record: Mapping[str, Any]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        labels = self.labels
        if not isinstance(labels, np.ndarray) or labels.ndim != 3:
            raise ValueError("labels must be a three-dimensional ZYX array (a plane is 1×Y×X)")
        if labels.dtype != LABEL_DTYPE:
            raise TypeError(f"labels must be uint32 (see to_label_dtype); got {labels.dtype}")
        if not labels.flags.c_contiguous:
            raise ValueError("labels must be C-contiguous")
        if not isinstance(self.grid, ReferenceGrid):
            raise TypeError("grid must be a ReferenceGrid")
        if labels.shape != self.grid.shape_zyx:
            raise IncompatibleGeometryError(f"labels shape {labels.shape} differs from the grid {self.grid.shape_zyx}")
        _check_target(self.target)
        _check_geometry(self.geometry, labels.shape[0])
        _check_namespace(self.label_namespace)
        if not isinstance(self.record, Mapping) or not isinstance(self.diagnostics, Mapping):
            raise TypeError("record and diagnostics must be mappings")

    @property
    def n_labels(self) -> int:
        """Number of distinct positive values."""
        return int(np.count_nonzero(np.unique(self.labels)))

    @property
    def max_label(self) -> int:
        """Largest value (0 for an empty image)."""
        return int(self.labels.max())
