"""The configuration of the export entry and its parts (design/specs/sample-output.md, "Configuration")."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np

from starfinder.assembly import AssemblyConfig
from starfinder.assignment import AssignmentConfig

#: The routes to a sample's cells (docs/sample-segmentation-contract.md, "The route setting"): segmentation per
#: FOV followed by assembly, and segmentation of the fused projection.
ROUTES = ("per_fov", "after_fusion")
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
    design/specs/sample-output.md and are not keys.

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


def _segmentation_spec(method):
    """The SEGMENTATION_METHODS spec of a run's method, or None for a LabelImportConfig."""
    from starfinder._registry import spec_for
    from starfinder.segmentation import SEGMENTATION_METHODS, LabelImportConfig
    if isinstance(method, LabelImportConfig):
        return None
    return spec_for(SEGMENTATION_METHODS, method, "segmentation method", TypeError,
                    f"no segmentation method is registered for {type(method).__qualname__}; a run's method is a "
                    "SEGMENTATION_METHODS config or a LabelImportConfig")


@dataclass(frozen=True)
class SampleSegmentationRun:
    """One run of the segmentation of the fused projection: a named label image from one method, or one import.

    ``name`` (snake_case) names the label file ``segmentation_<name>.ome.tif``
    and the run of the label namespace; ``target`` is ``nucleus`` or
    ``cell``; ``inputs`` are ``(role, image)`` pairs in channel order, each
    image a name of ``FusionConfig.images`` whose fused projection is read
    (none for an import); ``method`` is a ``SEGMENTATION_METHODS`` config or a
    :class:`~starfinder.segmentation.LabelImportConfig` whose target equals the
    run's, with its pixel lengths given for the acquired grid (LS2: the route
    converts them to the segmentation grid by the table "Lengths in pixels" of
    docs/sample-segmentation-contract.md and records both values); ``seeds``
    names an earlier run of the same config. See
    docs/sample-segmentation-contract.md ("Where the settings live").

    Raises
    ------
    ValueError
        A name that is not snake_case, an unknown target, a role or image name
        that is not valid, repeated roles, inputs missing or given to an
        import, a target, role or seed requirement the method does not
        accept, or seeds on an import.
    TypeError
        A field of the wrong type or an unregistered method config.
    """

    name: str
    target: str
    inputs: tuple[tuple[str, str], ...]
    method: object
    seeds: str | None = None

    def __post_init__(self):
        from starfinder._registry import check_name
        from starfinder.segmentation._labels import _check_target
        from starfinder.segmentation._methods import ROLES
        check_name(self.name, "sample segmentation run")
        _check_target(self.target)
        if isinstance(self.inputs, str) or not isinstance(self.inputs, (tuple, list)):
            raise TypeError(f"run {self.name!r}: inputs must be a sequence of (role, image) pairs")
        inputs = []
        for item in self.inputs:
            if isinstance(item, str) or not isinstance(item, (tuple, list)) or len(item) != 2 \
                    or not all(isinstance(v, str) for v in item):
                raise TypeError(f"run {self.name!r}: each input is a (role, image) pair of strings; got {item!r}")
            role, image = item
            if role not in ROLES:
                raise ValueError(f"run {self.name!r}: unknown channel role {role!r}; roles are {', '.join(ROLES)}")
            if not _NAME.fullmatch(image):
                raise ValueError(f"run {self.name!r}: image name {image!r} may use letters, digits, '_', '-' and "
                                 "'.' only")
            inputs.append((role, image))
        object.__setattr__(self, "inputs", tuple(inputs))
        roles = [role for role, _ in self.inputs]
        if len(set(roles)) != len(roles):
            raise ValueError(f"run {self.name!r} repeats a channel role: {roles}")
        if self.seeds is not None and (not isinstance(self.seeds, str) or not self.seeds):
            raise TypeError(f"run {self.name!r}: seeds must name an earlier run or be None")
        spec = _segmentation_spec(self.method)
        if spec is None:
            if self.inputs:
                raise ValueError(f"run {self.name!r} imports a mask and takes no inputs")
            if self.method.target != self.target:
                raise ValueError(f"run {self.name!r} has target {self.target!r}, its import {self.method.target!r}")
            if self.seeds is not None:
                raise ValueError(f"run {self.name!r} imports a mask and takes no seeds")
            return
        if not self.inputs:
            raise ValueError(f"run {self.name!r} needs at least one input")
        if self.target not in spec.targets:
            raise ValueError(f"run {self.name!r}: method {spec.name!r} produces "
                             f"{', '.join(sorted(spec.targets))}, not {self.target!r}")
        refused = [role for role in roles if role not in spec.roles]
        if refused:
            raise ValueError(f"run {self.name!r}: method {spec.name!r} does not accept the roles {refused}")
        for required in spec.required_roles:
            if not required & set(roles):
                raise ValueError(f"run {self.name!r}: method {spec.name!r} needs one of the roles "
                                 f"{sorted(required)}")
        if spec.seeds == "required" and self.seeds is None:
            raise ValueError(f"run {self.name!r}: method {spec.name!r} needs seeds, an earlier nucleus run")
        if spec.seeds == "none" and self.seeds is not None:
            raise ValueError(f"run {self.name!r}: method {spec.name!r} takes no seeds")


@dataclass(frozen=True)
class SampleSegmentationConfig:
    """The settings of the route after fusion: the runs, which labels are the cells, and the assignment.

    ``runs`` are the :class:`SampleSegmentationRun` in order (a run's seeds
    name an earlier run, of target ``nucleus``); ``cells`` names the run whose
    labels are the cells (a ``cell`` run, or a ``nucleus`` run as territories);
    ``nuclei`` the ``nucleus`` run for correspondence and compartments, or
    None; ``assignment`` the dataset's
    :class:`~starfinder.assignment.AssignmentConfig`; ``device`` ``cpu`` or
    ``cuda``, which every method of the runs must support. See
    docs/sample-segmentation-contract.md ("Where the settings live").

    Raises
    ------
    ValueError
        No run, a repeated run name, seeds that do not name an earlier
        nucleus run, ``cells`` or ``nuclei`` that do not name a run of the
        targets they take, or a device a method does not run on.
    TypeError
        A field of the wrong type.
    """

    runs: tuple[SampleSegmentationRun, ...]
    cells: str
    nuclei: str | None = None
    assignment: AssignmentConfig = field(default_factory=AssignmentConfig)
    device: str = "cpu"

    def __post_init__(self):
        if isinstance(self.runs, list):
            object.__setattr__(self, "runs", tuple(self.runs))
        if not isinstance(self.runs, tuple) or not all(isinstance(r, SampleSegmentationRun) for r in self.runs):
            raise TypeError("SampleSegmentationConfig.runs must be a tuple of SampleSegmentationRun")
        if not self.runs:
            raise ValueError("SampleSegmentationConfig.runs needs at least one run")
        targets = {}
        for run in self.runs:
            run.__post_init__()
            if run.name in targets:
                raise ValueError(f"run name {run.name!r} appears more than once")
            if run.seeds is not None:
                if run.seeds not in targets:
                    raise ValueError(f"run {run.name!r}: seeds {run.seeds!r} is not an earlier run")
                if targets[run.seeds] != "nucleus":
                    raise ValueError(f"run {run.name!r}: seeds {run.seeds!r} is a {targets[run.seeds]} run, not a "
                                     "nucleus run")
            targets[run.name] = run.target
        if not isinstance(self.cells, str) or self.cells not in targets:
            raise ValueError(f"SampleSegmentationConfig.cells must name a run ({sorted(targets)}); "
                             f"got {self.cells!r}")
        if self.nuclei is not None and (not isinstance(self.nuclei, str) or targets.get(self.nuclei) != "nucleus"):
            raise ValueError(f"SampleSegmentationConfig.nuclei must name a nucleus run "
                             f"({sorted(n for n, t in targets.items() if t == 'nucleus')}) or be None; "
                             f"got {self.nuclei!r}")
        if not isinstance(self.assignment, AssignmentConfig):
            raise TypeError(f"SampleSegmentationConfig.assignment must be an AssignmentConfig; "
                            f"got {self.assignment!r}")
        self.assignment.__post_init__()
        if self.device not in ("cpu", "cuda"):
            raise ValueError(f"SampleSegmentationConfig.device must be 'cpu' or 'cuda'; got {self.device!r}")
        for run in self.runs:
            spec = _segmentation_spec(run.method)
            if spec is not None and self.device not in spec.devices:
                raise ValueError(f"run {run.name!r}: method {spec.name!r} runs on "
                                 f"{', '.join(sorted(spec.devices))}, not device {self.device!r}")


@dataclass(frozen=True)
class ExportConfig:
    """Which parts ``Dataset.export`` runs and the configuration of each.

    ``assemble``, ``fuse``, ``fuse_labels`` (off by default) and
    ``export_spatialdata`` (off by default) turn the four parts on or off;
    ``assembly`` and ``table_format`` (``csv`` or ``parquet``) configure the
    assemble part, ``fusion``, ``labels`` and ``spatialdata`` the others. See
    design/specs/sample-output.md ("The export entry", "Configuration").
    ``route`` chooses the route to the sample's cells (a value of
    :data:`ROUTES`, or None for the default by ``fusion.dimensionality``:
    ``after_fusion`` for ``projection``, ``per_fov`` for ``stack`` and
    ``both``); ``segment`` turns the part ``segment`` of the route after
    fusion on or off (off reuses the saved sample segmentation); and
    ``segmentation`` (a :class:`SampleSegmentationConfig`, or None) holds the
    settings of that route (docs/sample-segmentation-contract.md, "The route
    setting").

    Raises
    ------
    ValueError
        A part key that is not a bool, an unknown ``table_format`` or
        ``route``, a nested config of another type, or ``fusion.ome_zarr``
        together with ``export_spatialdata`` (the store holds the one OME-Zarr
        copy).
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
    route: str | None = None
    segment: bool = True
    segmentation: SampleSegmentationConfig | None = None

    def __post_init__(self):
        for name in ("assemble", "fuse", "fuse_labels", "export_spatialdata", "segment"):
            _bool(getattr(self, name), f"ExportConfig.{name}")
        if self.route is not None and self.route not in ROUTES:
            raise ValueError(f"ExportConfig.route must be one of {list(ROUTES)} or None; got {self.route!r}")
        if self.segmentation is not None and not isinstance(self.segmentation, SampleSegmentationConfig):
            raise ValueError(f"ExportConfig.segmentation must be a SampleSegmentationConfig or None; "
                             f"got {self.segmentation!r}")
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
