"""The configuration of the export entry and its parts (docs/sample-output-contract.md, "Configuration")."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np

from starfinder.assembly import AssemblyConfig

DIMENSIONALITIES = ("projection", "stack", "both")
PROJECTIONS = ("max", "sum")
OME_ZARR_VERSIONS = ("0.5", "0.4")
ZARR_FORMATS = (3, 2)
TABLE_FORMATS = ("csv", "parquet")
NAMED_SOURCES = ("reference_stain", "reference_merged")
# spatialdata's element naming rule (letters, digits, _, - and .), applied to image names.
_NAME = re.compile(r"[A-Za-z0-9_.\-]+")


def scale_factor(value, name) -> int:
    """The whole factor ``k`` of a scale ``1/k`` (k ≥ 1); ValueError when ``1/value`` is not whole within 1e-9."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be 1/k with k a whole number ≥ 1; got {value!r}")
    reciprocal = 1.0 / float(value)
    k = round(reciprocal)
    if k < 1 or abs(reciprocal - k) > 1e-9:
        raise ValueError(f"{name} must be 1/k with k a whole number ≥ 1 (1/{name} whole within 1e-9); "
                         f"got {value!r}, whose reciprocal is {reciprocal!r}")
    return int(k)


def image_name(entry: str) -> str:
    """The name of a source-image entry of ``images``: the entry itself, or ``<name>`` of ``<name>=file:<pattern>``."""
    return entry.split("=", 1)[0] if "=" in entry else entry


def _check_images(images):
    if isinstance(images, str) or not isinstance(images, (tuple, list)):
        raise ValueError(f"FusionConfig.images must be a sequence of source-image names; got {images!r}")
    images = tuple(images)
    if not images:
        raise ValueError("FusionConfig.images must name at least one source image")
    for entry in images:
        if not isinstance(entry, str):
            raise ValueError(f"FusionConfig.images entries must be strings; got {entry!r}")
        if "=" in entry:
            name, source = entry.split("=", 1)
            if not source.startswith("file:") or "{fov_id}" not in source:
                raise ValueError(f"image {entry!r}: a file source is '<name>=file:<pattern>' with {{fov_id}} in "
                                 "the pattern")
        elif entry not in NAMED_SOURCES:
            raise ValueError(f"image {entry!r} is not one of {list(NAMED_SOURCES)} or '<name>=file:<pattern>'")
        else:
            name = entry
        if not _NAME.fullmatch(name):
            raise ValueError(f"image name {name!r} may use letters, digits, '_', '-' and '.' only")
    names = [image_name(entry) for entry in images]
    folded = {}
    for name in names:
        folded.setdefault(name.casefold(), []).append(name)
    equal = [group for group in folded.values() if len(group) > 1]
    if equal:
        raise ValueError(f"FusionConfig.images names must differ when case is ignored; equal: {equal}")
    return images


def _bool(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be True or False; got {value!r}")


@dataclass(frozen=True)
class FusionConfig:
    """What ``fuse`` writes: source images, dimensionality, projection, scale and file options.

    ``images`` names the source images (``reference_stain`` by default,
    ``reference_merged``, or ``<name>=file:<pattern>`` with ``{fov_id}``);
    ``dimensionality`` is ``projection`` (the 2D projection, the default),
    ``stack`` (the 3D stack alone, for dense 3D tissue) or ``both``;
    ``projection`` is ``max`` or ``sum`` (float32); ``scale_yx`` is ``1/k`` in
    Y and X, ``k`` whole (Z is never reduced); ``ome_zarr`` adds the library's
    standalone OME-Zarr beside each OME-TIFF, in ``ome_zarr_version`` ``"0.5"``
    (Z2) or ``"0.4"`` (Z1); ``pyramid_levels`` OME-TIFF SubIFD levels. The
    fusion function, interpolation order and reduction route are fixed by
    docs/sample-output-contract.md and are not keys.

    Raises
    ------
    ValueError
        An unknown value, a scale whose reciprocal is not whole within 1e-9, a
        negative ``pyramid_levels``, an empty ``images``, an invalid image name
        or names that are equal when case is ignored.
    """

    images: tuple[str, ...] = ("reference_stain",)
    dimensionality: str = "projection"
    projection: str = "max"
    scale_yx: float = 0.5
    ome_zarr: bool = False
    ome_zarr_version: str = "0.5"
    pyramid_levels: int = 0

    def __post_init__(self):
        object.__setattr__(self, "images", _check_images(self.images))
        if self.dimensionality not in DIMENSIONALITIES:
            raise ValueError(f"FusionConfig.dimensionality must be one of {list(DIMENSIONALITIES)}; "
                             f"got {self.dimensionality!r}")
        if self.projection not in PROJECTIONS:
            raise ValueError(f"FusionConfig.projection must be one of {list(PROJECTIONS)}; got {self.projection!r}")
        scale_factor(self.scale_yx, "FusionConfig.scale_yx")
        _bool(self.ome_zarr, "FusionConfig.ome_zarr")
        if self.ome_zarr_version not in OME_ZARR_VERSIONS:
            raise ValueError(f"FusionConfig.ome_zarr_version must be one of {list(OME_ZARR_VERSIONS)}; "
                             f"got {self.ome_zarr_version!r}")
        if isinstance(self.pyramid_levels, (bool, np.bool_)) or not isinstance(self.pyramid_levels, Integral) \
                or self.pyramid_levels < 0:
            raise ValueError(f"FusionConfig.pyramid_levels must be an integer ≥ 0; got {self.pyramid_levels!r}")

    @property
    def factor_yx(self) -> int:
        """The whole factor ``k`` of ``scale_yx = 1/k``."""
        return scale_factor(self.scale_yx, "FusionConfig.scale_yx")

    @property
    def image_names(self) -> tuple[str, ...]:
        """The names of ``images``, which name the files and store elements."""
        return tuple(image_name(entry) for entry in self.images)

    @property
    def dimensionalities(self) -> tuple[str, ...]:
        """The files written per image name in writing order, ``("projection", "stack")`` for ``both``."""
        return ("projection", "stack") if self.dimensionality == "both" else (self.dimensionality,)


@dataclass(frozen=True)
class LabelConfig:
    """The label export's scale ``1/k`` in Y and X (default 0.25), used by ``fuse_labels``.

    Raises
    ------
    ValueError
        A scale whose reciprocal is not whole within 1e-9.
    """

    scale_yx: float = 0.25

    def __post_init__(self):
        scale_factor(self.scale_yx, "LabelConfig.scale_yx")

    @property
    def factor_yx(self) -> int:
        """The whole factor ``k`` of ``scale_yx = 1/k``."""
        return scale_factor(self.scale_yx, "LabelConfig.scale_yx")


@dataclass(frozen=True)
class SpatialDataExportConfig:
    """The Zarr format of the SpatialData store: 3 (spatialdata's default, Z2) or 2 (Z1).

    Raises
    ------
    ValueError
        Another value.
    """

    zarr_format: int = 3

    def __post_init__(self):
        if isinstance(self.zarr_format, (bool, np.bool_)) or self.zarr_format not in ZARR_FORMATS:
            raise ValueError(f"SpatialDataExportConfig.zarr_format must be one of {list(ZARR_FORMATS)}; "
                             f"got {self.zarr_format!r}")


@dataclass(frozen=True)
class ExportConfig:
    """Which parts ``Dataset.export`` runs and the configuration of each.

    ``assemble``, ``fuse``, ``fuse_labels`` (off by default) and
    ``export_spatialdata`` (off by default) turn the four parts on or off;
    ``assembly`` and ``table_format`` (``csv`` or ``parquet``) configure the
    assemble part, ``fusion``, ``labels`` and ``spatialdata`` the others. See
    docs/sample-output-contract.md ("The export entry", "Configuration").

    Raises
    ------
    ValueError
        A part key that is not a bool, an unknown ``table_format``, a nested
        config of another type, or ``fusion.ome_zarr`` together with
        ``export_spatialdata`` (the store holds the one OME-Zarr copy).
    """

    assemble: bool = True
    fuse: bool = True
    fuse_labels: bool = False
    export_spatialdata: bool = False
    assembly: AssemblyConfig = field(default_factory=AssemblyConfig)
    table_format: str = "csv"
    fusion: FusionConfig = field(default_factory=FusionConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    spatialdata: SpatialDataExportConfig = field(default_factory=SpatialDataExportConfig)

    def __post_init__(self):
        for name in ("assemble", "fuse", "fuse_labels", "export_spatialdata"):
            _bool(getattr(self, name), f"ExportConfig.{name}")
        if self.table_format not in TABLE_FORMATS:
            raise ValueError(f"ExportConfig.table_format must be one of {list(TABLE_FORMATS)}; "
                             f"got {self.table_format!r}")
        for name, kind in (("assembly", AssemblyConfig), ("fusion", FusionConfig), ("labels", LabelConfig),
                           ("spatialdata", SpatialDataExportConfig)):
            if not isinstance(getattr(self, name), kind):
                raise ValueError(f"ExportConfig.{name} must be a {kind.__name__}; got {getattr(self, name)!r}")
        if self.fusion.ome_zarr and self.export_spatialdata:
            raise ValueError("FusionConfig.ome_zarr cannot be combined with export_spatialdata: the SpatialData "
                             "store holds the one OME-Zarr copy of each fused image")
