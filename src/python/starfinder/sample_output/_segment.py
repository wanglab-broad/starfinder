"""The part ``segment`` of the route after fusion and the route setting (docs/sample-segmentation-contract.md).

:func:`segment_fused` segments the fused projection that ``fuse`` wrote, run by run, and
writes one label OME-TIFF per run and the ``segmentation`` section of ``export.json``;
:func:`load_sample_segmentation` reads them back with their SHA-256 checked. The pixel
lengths of a method's settings are given for the acquired grid and divided by the scale
factor here (LS2, :data:`LENGTH_FIELDS`). :func:`resolve_route` gives a dataset's route,
and :func:`check_route` is check 7 of ``Dataset.export``.
"""
from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from starfinder.image import ImageMetadata
from starfinder.segmentation import (LabelImportConfig, ReferenceGrid, SegmentationBackendUnavailableError,
                                     SegmentationInput, SegmentationResult, import_labels, resolve_model, segment)

from ._config import ROUTES, ExportConfig, SampleSegmentationConfig, _segmentation_spec
from ._record import (ExportIncompleteWarning, Timer, as_json, listed_files, part_status, read_export, remove_listed,
                      update_section)

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
FILE_PREFIX = "segmentation_"
PROJECTION = "max"
#: The method settings the route converts from the acquired grid to the segmentation grid at factor ``k`` (LS2),
#: per registered method, field → rule: ``length`` (pixels: value / k), ``area`` (pixels²: max(1, round(value /
#: k²))), ``factor_yx`` (a factor of Y and X: value · k in Y and X, Z unchanged), ``length_yx`` (pixels per axis:
#: ceil(value / k) in Y and X, Z unchanged) and ``unchanged`` (a count, or µm that ``segment`` converts with the
#: grid's spacing). Each value is recorded as given and as used. A test keeps it equal to SEGMENTATION_METHODS.
LENGTH_FIELDS = {"cellpose": {"diameter": "length", "min_size": "area"},
                 "stardist": {"scale": "factor_yx", "block_size": "length_yx", "min_overlap": "length_yx",
                              "context": "length_yx", "n_tiles": "unchanged"},
                 "seeded_watershed": {"sigma_um": "unchanged"}}
STACK_REFUSED = ("segmentation after fusion is specified for projection samples only; a dataset with "
                 "dimensionality 'stack' is segmented per FOV (route: per_fov)")


def resolve_route(config: ExportConfig) -> tuple[str, str]:
    """``(route, route_source)``: the route set (``set``), or the default by ``fusion.dimensionality`` (RD1).

    ``projection`` gives ``after_fusion``; ``stack`` and ``both`` give ``per_fov``.
    """
    if config.route is not None:
        return config.route, "set"
    return ("after_fusion" if config.fusion.dimensionality == "projection" else "per_fov"), "default"


def segment_off_section(route: str, route_source: str, assignment) -> dict:
    """The ``segmentation`` section of an export whose part ``segment`` does not run.

    Reason ``no_cells`` for a sample without segmentation (``assignment`` None),
    on either route; ``route_per_fov`` on the route per FOV.
    """
    reason = "no_cells" if assignment is None else "route_per_fov"
    return {"status": "off", "reason": reason, "route": route, "route_source": route_source}


# --- LS2: lengths in pixels ---------------------------------------------------------------------------------

def _scaled(value, rule, k):
    if value is None or rule == "unchanged":
        return value
    if rule == "length":
        return value / k
    if rule == "area":
        return max(1, round(value / k ** 2))
    if rule == "factor_yx":
        if isinstance(value, tuple):
            return (value[0], value[1] * k, value[2] * k)
        return value * k
    if rule == "length_yx":
        if isinstance(value, tuple):
            yx = tuple(math.ceil(v / k) for v in value[-2:])
            return tuple(value[:-2]) + yx
        return math.ceil(value / k)
    raise ValueError(f"unknown length rule {rule!r}")


def scale_lengths(method, k: int):
    """``(config used, lengths)``: a run's method with its pixel lengths converted to the factor ``k`` (LS2).

    ``lengths`` maps each field of :data:`LENGTH_FIELDS` to ``given``, ``used``
    and ``rule``. The config is rebuilt with ``dataclasses.replace``, so its own
    checks apply to the values used (StarDist refuses block fields with a scale
    other than 1, so a block run is possible at ``k = 1`` only). An import has
    no lengths.
    """
    spec = _segmentation_spec(method)
    if spec is None:
        return method, {}
    rules = LENGTH_FIELDS[spec.name]
    used = {name: _scaled(getattr(method, name), rule, k) for name, rule in rules.items()}
    changed = {name: value for name, value in used.items() if value != getattr(method, name)}
    lengths = {name: {"given": getattr(method, name), "used": used[name], "rule": rule}
               for name, rule in rules.items()}
    return (replace(method, **changed) if changed else method), as_json(lengths)


# --- Check 7 of Dataset.export --------------------------------------------------------------------------------

def check_models(segmentation: SampleSegmentationConfig) -> list[dict]:
    """Every model of the runs resolves with its SHA-256 (``resolve_model``, which imports no segmentation library).

    Returns the artifacts entries per run; raises the model errors of
    docs/segmentation-contract.md with the run and key named.
    """
    from starfinder.segmentation import MissingModelError, ModelHashMismatchError
    resolved = []
    for run in segmentation.runs:
        spec = _segmentation_spec(run.method)
        if spec is None or not spec.models:
            continue
        model, model_path = getattr(run.method, "model", None), getattr(run.method, "model_path", None)
        key = "model" if model is not None else "model_path"
        try:
            path, artifacts = resolve_model(spec.name, model=model, model_path=model_path,
                                            model_sha256=getattr(run.method, "model_sha256", None))
        except (MissingModelError, ModelHashMismatchError) as error:
            raise type(error)(f"export.segmentation run {run.name!r} {key}: {error}") from error
        resolved.append({"run": run.name, "path": str(path), "artifacts": artifacts})
    return resolved


def check_route(config: ExportConfig, assignment) -> tuple[str, str]:
    """Check 7 of ``Dataset.export`` (docs/sample-segmentation-contract.md, "The entry"); returns the route.

    In this order: the route is a value of :data:`ROUTES` or None; the route
    after fusion, set or by default, refuses ``dimensionality: stack`` and
    ``projection: sum``; on the route after fusion with an ``assignment``
    name, ``segmentation`` is given (RD1), each run's input images are names
    of ``fusion.images``, the projection is written, ``fuse`` and ``assemble``
    are on unless ``segment`` is off, the runs validate (their seeds name
    earlier runs, ``cells`` and ``nuclei`` name runs of their targets) and
    every model resolves with its SHA-256. Each failure raises ``ValueError``
    (or ``MissingModelError``, ``ModelHashMismatchError``) naming the key.
    """
    if config.route is not None and config.route not in ROUTES:
        raise ValueError(f"export.route must be one of {list(ROUTES)} or null; got {config.route!r}")
    route, source = resolve_route(config)
    if route != "after_fusion":
        return route, source
    how = "set" if source == "set" else "the default for dimensionality 'projection'"
    if config.fusion.dimensionality == "stack":
        raise ValueError(STACK_REFUSED)
    if config.fusion.projection == "sum":
        raise ValueError(f"export.projection 'sum' is refused on the route after fusion (route: after_fusion, {how}): "
                         "the route segments the Z maximum, and one projection file cannot be both; set "
                         "projection: max or route: per_fov")
    if assignment is None:
        return route, source
    segmentation = config.segmentation
    if segmentation is None:
        raise ValueError(f"the route after fusion (route: after_fusion, {how}) segments the fused projection and "
                         "needs export.segmentation, the settings of that segmentation; set export.segmentation, "
                         "or route: per_fov to assemble the per-FOV assignments")
    names = config.fusion.image_names
    for run in segmentation.runs:
        missing = [image for _, image in run.inputs if image not in names]
        if missing:
            raise ValueError(f"export.segmentation run {run.name!r} reads the images {missing}, which are not names "
                             f"of export.images {list(names)}")
    if "projection" not in config.fusion.dimensionalities:
        raise ValueError(STACK_REFUSED)
    if config.segment and not config.fuse:
        raise ValueError("export.fuse: false with segment: true is refused: the part segment segments the fused "
                         "images of this export, and fuse: false removes them")
    if config.segment and not config.assemble:
        raise ValueError("export.assemble: false with segment: true is refused: the cells may change, so the tables "
                         "are made again")
    segmentation.__post_init__()
    check_models(segmentation)
    return route, source


# --- The sample segmentation ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleSegmentation:
    """The sample segmentation of the route after fusion: its grid, one label image per run, and its section.

    ``grid`` is the segmentation grid (the fused projection's, frame
    ``sample:<sample>``), None when no run was made; ``runs`` maps each run
    name, in run order, to its
    :class:`~starfinder.segmentation.SegmentationResult` (geometry ``plane``,
    label namespace ``[dataset_id, sample_id, null, null, <run>]``);
    ``record`` is the ``segmentation`` section of ``export.json``.
    """

    grid: ReferenceGrid | None
    runs: Mapping[str, SegmentationResult]
    record: Mapping[str, Any]

    @property
    def status(self) -> str:
        """The status of the part ``segment``, ``written``, ``reused``, ``empty``, ``not_run`` or ``failed``."""
        return self.record.get("status")

    @property
    def cells(self) -> SegmentationResult | None:
        """The run whose labels are the cells."""
        return self.runs.get(self.record.get("cells"))

    @property
    def nuclei(self) -> SegmentationResult | None:
        """The nucleus run, or None."""
        name = self.record.get("nuclei")
        return None if name is None else self.runs.get(name)


def label_file(run: str) -> str:
    """The file name of a run's label image, ``segmentation_<run>.ome.tif``."""
    return f"{FILE_PREFIX}{run}.ome.tif"


def _namespace(identity, sample, run=None):
    dataset_id, sample_id = identity if identity is not None else (None, sample)
    parts = [dataset_id, sample_id, None, None] + ([] if run is None else [run])
    return json.dumps(parts, separators=(",", ":"))


def _grid(entry, sample) -> ReferenceGrid:
    metadata = ImageMetadata(f"sample:{sample}", tuple(float(v) for v in entry["spacing_zyx"]),
                             tuple(float(v) for v in entry["origin_zyx"]), IDENTITY, "µm")
    return ReferenceGrid(tuple(int(v) for v in entry["shape_zyx"]), metadata, "declared")


def _grid_record(grid: ReferenceGrid, k: int) -> dict:
    metadata = grid.metadata
    return {"shape_zyx": list(grid.shape_zyx), "spacing_zyx": list(metadata.spacing_zyx),
            "origin_zyx": list(metadata.origin_zyx), "frame": metadata.frame_id, "unit": metadata.spatial_unit,
            "factor_yx": k, "projection": PROJECTION}


@dataclass
class _Plan:
    status: str
    reason: str | None = None
    grid: ReferenceGrid | None = None
    k: int | None = None
    images: dict = None
    placement: dict = None
    imports: dict = None


def _check(output_dir, config, sample, identity) -> _Plan:
    """Every check before anything is written; the plan of the runs."""
    from starfinder.segmentation._persist import file_sha256
    if not isinstance(config, SampleSegmentationConfig):
        raise TypeError("config must be a SampleSegmentationConfig")
    config.__post_init__()
    if not isinstance(sample, str) or not sample:
        raise ValueError("sample must be a nonempty string")
    if identity is not None and (not isinstance(identity, tuple) or len(identity) != 2
                                 or not all(v is None or isinstance(v, str) for v in identity)):
        raise TypeError("identity must be None or a (dataset_id, sample_id) pair")
    record = read_export(output_dir)
    if record.get("sample") not in (None, sample):
        raise ValueError(f"{output_dir} holds the export of sample {record.get('sample')!r}, not {sample!r}")
    status = part_status(record, "fuse")
    if status == "empty":
        return _Plan("empty", "no_included_fov")
    if status != "written":
        return _Plan("not_run", "part_not_run")
    fusion = record["fusion"]
    if (fusion.get("config") or {}).get("projection") != PROJECTION:
        raise ValueError(f"the fusion section of {output_dir} has projection "
                         f"{(fusion.get('config') or {}).get('projection')!r}; the route after fusion (route: "
                         "after_fusion) segments the Z maximum, projection 'max'")
    outputs = {o["image"]: o for o in fusion.get("outputs") or []
               if o.get("format") == "ome_tiff" and o.get("dimensionality") == "projection"}
    names = list(dict.fromkeys(image for run in config.runs for _, image in run.inputs))
    missing = [name for name in names if name not in outputs]
    if missing:
        raise ValueError(f"the runs read the images {missing}, which have no fused projection in the fusion section "
                         f"of {output_dir} (projections: {sorted(outputs)})")
    used = [outputs[name] for name in names] or list(outputs.values())
    if not used:
        raise ValueError(f"the fusion section of {output_dir} lists no fused projection")
    first = used[0]
    for entry in used[1:]:
        if [entry[key] for key in ("shape_zyx", "spacing_zyx", "origin_zyx")] != \
                [first[key] for key in ("shape_zyx", "spacing_zyx", "origin_zyx")]:
            raise ValueError(f"the fused projections {first['image']!r} and {entry['image']!r} are on different "
                             "grids")
    images = {}
    for name in names:
        entry = outputs[name]
        path = Path(output_dir) / entry["path"]
        if not path.is_file():
            raise ValueError(f"the fused projection of image {name!r}, {path}, is missing")
        digest = file_sha256(path)
        if digest != entry["sha256"]:
            raise ValueError(f"the fused projection of image {name!r}, {path}, has SHA-256 {digest}, the fusion "
                             f"section {entry['sha256']}")
        images[name] = {"image": name, "path": entry["path"], "sha256": digest}
    grid = _grid(first, sample)
    placement = fusion.get("placement") or {}
    plan = _Plan("written", grid=grid, k=int(first["factor_yx"]), images=images,
                 placement={"fovs_sha256": placement.get("fovs_sha256"),
                            "links_sha256": placement.get("links_sha256")}, imports={})
    for run in config.runs:
        if isinstance(run.method, LabelImportConfig):
            method = run.method
            plan.imports[run.name] = import_labels(method.path, grid=grid, target=method.target,
                                                   geometry=method.geometry, relabel=method.relabel,
                                                   label_namespace=_namespace(identity, sample, run.name))
        else:
            scale_lengths(run.method, plan.k)
    check_models(config)
    return plan


def _read_projection(output_dir, entry) -> np.ndarray:
    import tifffile
    array = tifffile.imread(Path(output_dir) / entry["path"])
    return np.ascontiguousarray(array[np.newaxis] if array.ndim == 2 else array)


def _extras(method) -> list[str]:
    spec = _segmentation_spec(method)
    return sorted({d.extra for d in spec.requires if d.extra}) if spec is not None else []


def segment_fused(output_dir, *, config: SampleSegmentationConfig, sample: str,
                  identity: tuple[str | None, str | None] | None = None) -> SampleSegmentation:
    """Segment a sample's fused projection and write its label images and the ``segmentation`` section.

    The part ``segment`` of the route after fusion
    (docs/sample-segmentation-contract.md, "The entry"), callable alone on an
    export folder whose ``fuse`` ran. It reads the fused projection of each
    input image from the ``fusion`` section (its path, checked against the
    recorded SHA-256), builds per run one
    :class:`~starfinder.segmentation.SegmentationInput` on the segmentation
    grid (shape, spacing and origin of the fusion section's output, frame
    ``sample:<sample>``; channels in the run's role order), converts the
    method's pixel lengths by the scale factor (LS2, ``LENGTH_FIELDS``)
    and calls :func:`~starfinder.segmentation.segment`, with the seeds of the
    run it names, or :func:`~starfinder.segmentation.import_labels` with that
    grid for an import. It applies no label operation. Each run's labels are
    written as ``segmentation_<run>.ome.tif`` (uint32, the OME-TIFF layout of
    the fused image, the segmentation grid's geometry) and the section records
    the status, the route, the inputs (the grid, each fused file's name and
    SHA-256 and the placement's SHA-256 values, no hash of the ``fusion``
    section), per run its method record, its lengths as given and as used,
    its label file's path and SHA-256 and its counts, ``cells``, ``nuclei``,
    ``device``, the software, the times and the peak RSS. The files of the
    earlier section are removed first.

    When ``fuse`` did not run, it writes no label file and records
    ``not_run`` (``part_not_run``) with :class:`ExportIncompleteWarning`;
    without an included FOV it records ``empty`` (``no_included_fov``). When a
    method's backend is not importable it records ``not_run``
    (``extra_missing``, naming the extra) with the warning; any other error
    while segmenting or writing records ``failed`` with the files completed
    and is raised again.

    Parameters
    ----------
    output_dir : str or Path
        The sample's export folder ``<OUTPUT>/export/<sample>/``.
    config : SampleSegmentationConfig
        The runs, ``cells``, ``nuclei``, the assignment and the device.
    sample : str
        The sample of the export folder.
    identity : tuple or None
        ``(dataset_id, sample_id)`` of the label namespaces ``[dataset_id,
        sample_id, null, null, <run>]``. ``Dataset.export`` always passes the
        dataset's. None gives ``[null, <sample>, null, null, <run>]``: such
        labels do not carry the dataset's identity and are for use outside a
        dataset.

    Returns
    -------
    SampleSegmentation

    Raises
    ------
    TypeError
        An argument of the wrong type.
    ValueError
        Before anything is written: a folder of another sample, a fusion
        section whose projection is not ``max``, an input image without a fused
        projection, a fused file missing or whose SHA-256 differs from the
        record, or a method config the lengths at the scale factor make
        invalid.
    IncompatibleGeometryError
        Before anything is written: an import file of another shape than the
        segmentation grid.
    MissingModelError, ModelHashMismatchError
        Before anything is written: a model missing or with another SHA-256.
    """
    from starfinder.segmentation._persist import file_sha256

    from . import _tiff
    from ._fuse import _software
    timer = Timer()
    plan = _check(output_dir, config, sample, identity)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_listed(output_dir, listed_files(read_export(output_dir), "segment"))
    runs, outputs, results = [], [], {}

    def finish(status, *, warn=None, **extra):
        section = {"status": status, **extra, "route": "after_fusion",
                   "inputs": None if plan.grid is None else {
                       "grid": _grid_record(plan.grid, plan.k), "images": list(plan.images.values()),
                       "placement": plan.placement},
                   "runs": list(runs), "cells": config.cells, "nuclei": config.nuclei, "device": config.device,
                   "outputs": list(outputs), "software": _software(), **timer.record()}
        section = as_json(section)
        update_section(output_dir, sample, "segment", section)
        if warn:
            warnings.warn(f"export part segment could not run: {warn}", ExportIncompleteWarning, stacklevel=3)
        return SampleSegmentation(plan.grid if status == "written" else None,
                                  dict(results) if status == "written" else {}, section)

    if plan.status == "empty":
        return finish("empty", reason=plan.reason)
    if plan.status == "not_run":
        return finish("not_run", reason=plan.reason, warn=f"{plan.reason}: the part fuse did not write the fused "
                                                           "images")
    grid, k = plan.grid, plan.k
    fusion = read_export(output_dir)["fusion"]
    entries = {o["image"]: o for o in fusion["outputs"]
               if o.get("format") == "ome_tiff" and o.get("dimensionality") == "projection"}
    arrays = {}
    try:
        for run in config.runs:
            if run.name in plan.imports:
                result = plan.imports[run.name]
                method, lengths = None, {}
            else:
                for _, image in run.inputs:
                    if image not in arrays:
                        arrays[image] = _read_projection(output_dir, entries[image])
                image = np.stack([arrays[name] for _, name in run.inputs], axis=-1)
                sources = tuple(dict(plan.images[name]) for _, name in run.inputs)
                used, lengths = scale_lengths(run.method, k)
                try:
                    result = segment(SegmentationInput(image, grid, tuple(role for role, _ in run.inputs), sources),
                                     config=used, target=run.target,
                                     seeds=None if run.seeds is None else results[run.seeds],
                                     device=config.device, label_namespace=_namespace(identity, sample, run.name))
                except SegmentationBackendUnavailableError as error:
                    extras = _extras(run.method)
                    return finish("not_run", reason="extra_missing", extra=extras[0] if len(extras) == 1 else extras,
                                  run=run.name, warn=f"extra_missing: {error}")
                method = (result.record.get("methods") or [None])[0]
            result = replace(result, record=dict(result.record, run=run.name))
            results[run.name] = result
            name = label_file(run.name)
            path = output_dir / name
            annotation = {"sample": sample, "image": name[:-len(".ome.tif")], "run": run.name, "target": run.target,
                          "label_namespace": result.label_namespace, "factor_yx": k,
                          "placement_fovs_sha256": plan.placement["fovs_sha256"],
                          "placement_links_sha256": plan.placement["links_sha256"]}
            layout = _tiff.write_ome_tiff(path, result.labels[0], metadata=grid.metadata, annotation=annotation)
            digest = file_sha256(path)
            metadata = grid.metadata
            outputs.append({"path": name, "format": "ome_tiff", "image": annotation["image"], "run": run.name,
                            "dimensionality": "projection", "factor_yx": k, "shape_zyx": list(grid.shape_zyx),
                            "dtype": "uint32", "spacing_zyx": list(metadata.spacing_zyx),
                            "origin_zyx": list(metadata.origin_zyx), "levels": [], "tile_yx": layout["tile_yx"],
                            "compression": layout["compression"], "axes": layout["axes"], "annotation": annotation,
                            "sha256": digest})
            runs.append({"name": run.name, "target": run.target, "geometry": result.geometry,
                         "label_namespace": result.label_namespace,
                         "inputs": [{"role": role, "image": image} for role, image in run.inputs],
                         "seeds": result.record.get("seeds"), "method": method,
                         "import": result.record.get("import"), "lengths": lengths, "path": name, "sha256": digest,
                         "n_labels": result.n_labels, "max_label": result.max_label,
                         "outcome": result.record.get("outcome")})
    except BaseException as error:
        finish("failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    return finish("written")


def load_sample_segmentation(output_dir) -> SampleSegmentation:
    """The saved sample segmentation of an export folder, each label file checked against its SHA-256.

    Reads the ``segmentation`` section of ``export.json`` (status ``written``
    or ``reused``), rebuilds the segmentation grid from it and reads each run's
    ``segmentation_<run>.ome.tif`` (uint32, the grid's shape).

    Raises
    ------
    ValueError
        The folder holds no saved sample segmentation, or a label file is
        missing, has another SHA-256 than the record (naming the path and both
        values), or is not a uint32 image of the grid's shape.
    """
    import tifffile

    from starfinder.segmentation._persist import file_sha256
    output_dir = Path(output_dir)
    record = read_export(output_dir)
    section = record.get("segmentation")
    status = part_status(record, "segment") if record else None
    if not isinstance(section, Mapping) or status not in ("written", "reused") or not section.get("inputs"):
        raise ValueError(f"{output_dir} holds no saved sample segmentation: its part segment is {status!r}")
    entry = section["inputs"]["grid"]
    metadata = ImageMetadata(entry["frame"], tuple(float(v) for v in entry["spacing_zyx"]),
                             tuple(float(v) for v in entry["origin_zyx"]), IDENTITY, entry["unit"])
    grid = ReferenceGrid(tuple(int(v) for v in entry["shape_zyx"]), metadata, "declared")
    runs = {}
    for run in section["runs"]:
        path = output_dir / run["path"]
        if not path.is_file():
            raise ValueError(f"{output_dir}: the label file {path} of run {run['name']!r} is missing")
        digest = file_sha256(path)
        if digest != run["sha256"]:
            raise ValueError(f"{output_dir}: the label file {path} of run {run['name']!r} has SHA-256 {digest}, the "
                             f"record {run['sha256']}")
        labels = tifffile.imread(path)
        labels = labels[np.newaxis] if labels.ndim == 2 else labels
        if labels.dtype != np.uint32 or labels.shape != grid.shape_zyx:
            raise ValueError(f"{output_dir}: the label file {path} holds {labels.dtype} of shape {labels.shape}, not "
                             f"uint32 of the grid's shape {grid.shape_zyx}")
        runs[run["name"]] = SegmentationResult(np.ascontiguousarray(labels), grid, run["target"], run["geometry"],
                                               run["label_namespace"],
                                               dict(run, run=run["name"], operations=[], input={"projection": None}))
    return SampleSegmentation(grid, runs, dict(section))
