"""The segment entry and the checks the stage wrapper applies to every method (docs/segmentation-contract.md)."""
from __future__ import annotations

import os
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np

from starfinder._execution import THREAD_VARIABLES
from starfinder._registry import provenance, require, spec_for
from starfinder.image import IncompatibleGeometryError, _validate_image

from ._errors import MissingModelError, SegmentationBackendUnavailableError
from ._import import LabelImportConfig, _software
from ._labels import (FORMAT_VERSION, ReferenceGrid, SegmentationResult, _check_namespace, _check_target,
    _grid_record, _json, _namespace_ids, array_sha256, to_label_dtype)
from ._methods import DEVICES, ROLES, SEGMENTATION_METHODS, MethodContext
from ._models import model_dimensions, resolve_model

_WHAT = "segmentation method"


@dataclass(frozen=True)
class SegmentationInput:
    """One ZYXC image on one grid whose channels each carry a declared role.

    ``image`` is finite and real (Z=1 for a plane) with ``image.shape[:3] ==
    grid.shape_zyx``; ``roles`` holds one role per channel, in channel order,
    each of ``nuclear``, ``cytoplasm``, ``membrane``, ``amplicon`` and
    ``composite`` at most once; ``sources`` holds, per channel, a mapping that
    describes where the channel came from (``FOV.segment`` records the
    InputChannel fields and the channel's SHA-256; a direct caller may pass
    empty mappings). Which roles a method needs is checked by :func:`segment`.

    Raises
    ------
    TypeError
        grid is not a ReferenceGrid, or roles or sources is not a tuple.
    IncompatibleGeometryError
        image is not four-dimensional, or its ZYX shape differs from the grid.
    ValueError
        The image is empty or not finite and real (InvalidImageError), a role is
        unknown or repeated, or the number of roles or sources differs from the
        number of channels.
    """

    image: np.ndarray
    grid: ReferenceGrid
    roles: tuple[str, ...]
    sources: tuple[Mapping, ...] = ()

    def __post_init__(self):
        if not isinstance(self.grid, ReferenceGrid):
            raise TypeError("grid must be a ReferenceGrid")
        if not isinstance(self.roles, tuple) or not isinstance(self.sources, tuple):
            raise TypeError("roles and sources must be tuples")
        image = np.asarray(self.image)
        if image.ndim != 4:
            raise IncompatibleGeometryError(f"a segmentation input is a ZYXC image; got {image.ndim} dimensions")
        _validate_image(image, ndim=(4,))
        if image.shape[:3] != self.grid.shape_zyx:
            raise IncompatibleGeometryError(f"input ZYX shape {image.shape[:3]} differs from the grid "
                                            f"{self.grid.shape_zyx}")
        if len(self.roles) != image.shape[3]:
            raise ValueError(f"one role per channel: {image.shape[3]} channels, roles {self.roles}")
        for role in self.roles:
            if role not in ROLES:
                raise ValueError(f"unknown channel role {role!r}; roles are {', '.join(ROLES)}")
        repeated = sorted({r for r in self.roles if self.roles.count(r) > 1})
        if repeated:
            raise ValueError(f"channel role {repeated[0]!r} appears more than once")
        if self.sources and len(self.sources) != len(self.roles):
            raise ValueError("sources must hold one mapping per channel, or be empty")
        if not all(isinstance(s, Mapping) for s in self.sources):
            raise TypeError("each source must be a mapping")


def _check_input(segmentation_input, spec):
    """Check 3: the input's own validity, then the method's accepted and required roles."""
    if not isinstance(segmentation_input, SegmentationInput):
        raise TypeError("segmentation_input must be a SegmentationInput")
    segmentation_input.__post_init__()
    roles = segmentation_input.roles
    for role in roles:
        if role not in spec.roles:
            raise ValueError(f"{_WHAT} {spec.name!r} does not accept the channel role {role!r}; it accepts "
                             f"{', '.join(sorted(spec.roles))}")
    for required in spec.required_roles:
        if not required & set(roles):
            raise ValueError(f"{_WHAT} {spec.name!r} needs a channel with the role "
                             f"{' or '.join(repr(r) for r in sorted(required))}")


def _check_device(device, spec):
    """Check 4: "cpu" or "cuda" (option V1), and one of the method's devices."""
    if not isinstance(device, str) or device not in DEVICES:
        raise ValueError(f"device must be 'cpu' or 'cuda'; got {device!r}")
    if device not in spec.devices:
        raise ValueError(f"{_WHAT} {spec.name!r} runs on {', '.join(sorted(spec.devices))}, not {device!r}")


def _resolve_model(spec, config):
    """Check 6 for a method with a model: the files exist and match their SHA-256, before any library call.

    ``config.model`` names a known model of ``KNOWN_MODELS`` in the weights cache
    (``<root>/<method>/<model>/``), ``config.model_path`` a folder or file given by
    path (see :func:`resolve_model`). Returns the model path, the artifacts entry of
    each file and the model's dimensionality read without the library (a config's
    ``do_3d``, else ``n_dim`` of ``config.json``; None when neither exists). Nothing
    is downloaded.
    """
    model_path, model = getattr(config, "model_path", None), getattr(config, "model", None)
    if model_path is None and model is None:
        raise MissingModelError(f"{_WHAT} {spec.name!r} needs a model: set model or model_path")
    path, artifacts = resolve_model(spec.name, model=model, model_path=model_path,
                                    model_sha256=getattr(config, "model_sha256", None))
    return path, artifacts, model_dimensions(path, config)


def _check_dimensions(shape, spec, n_dim):
    """Check 7: Z=1 needs 2 in spec.dimensions, Z>1 needs 3; the model's dimensionality agrees; minimum shape."""
    if shape[0] == 1 and 2 not in spec.dimensions:
        raise IncompatibleGeometryError(f"{_WHAT} {spec.name!r} needs Z>1; the input has one plane")
    if shape[0] > 1 and 3 not in spec.dimensions:
        raise IncompatibleGeometryError(f"{_WHAT} {spec.name!r} runs on one plane; the input has Z={shape[0]}")
    if n_dim is not None and n_dim != (2 if shape[0] == 1 else 3):
        raise IncompatibleGeometryError(f"{_WHAT} {spec.name!r}: a {n_dim}D model cannot segment an input with "
                                        f"Z={shape[0]}")
    minimum = spec.min_shape_zyx if shape[0] > 1 else (1, *spec.min_shape_zyx[1:])
    if any(n < m for n, m in zip(shape, minimum)):
        raise IncompatibleGeometryError(f"{_WHAT} {spec.name!r} needs every axis at least {minimum}, "
                                        f"not {tuple(shape)}")


def _check_seeds(seeds, spec, grid):
    """Check 8: seeds required, optional or refused; a nucleus result on the input's grid."""
    if seeds is None:
        if spec.seeds == "required":
            raise ValueError(f"{_WHAT} {spec.name!r} grows from seeds: pass the nucleus result as seeds")
        return
    if spec.seeds == "none":
        raise ValueError(f"{_WHAT} {spec.name!r} takes no seeds")
    if not isinstance(seeds, SegmentationResult):
        raise TypeError("seeds must be a SegmentationResult")
    if seeds.target != "nucleus":
        raise ValueError(f"seeds must have target 'nucleus', not {seeds.target!r}")
    if seeds.grid.shape_zyx != grid.shape_zyx or seeds.grid.metadata != grid.metadata:
        raise IncompatibleGeometryError(f"the seeds are on grid {seeds.grid.shape_zyx} {seeds.grid.metadata!r}, "
                                        f"the input on {grid.shape_zyx} {grid.metadata!r}")


def _check_output(labels, shape, spec):
    """Check 10: an integer array on the input's ZYX shape (a dropped Z restored for Z=1), converted to uint32."""
    if not isinstance(labels, np.ndarray):
        raise ValueError(f"{_WHAT} {spec.name!r} returned {type(labels).__name__}, not a label array")
    if labels.dtype.kind not in "iu":
        raise TypeError(f"{_WHAT} {spec.name!r} returned {labels.dtype} labels; labels must be integers")
    if shape[0] == 1 and labels.shape == shape[1:]:
        labels = labels[np.newaxis]
    if labels.shape != shape:
        raise ValueError(f"{_WHAT} {spec.name!r} returned labels of shape {labels.shape}, not the input's {shape}")
    return to_label_dtype(labels)


def _execution(spec, device, framework=None):
    """The execution entry: device, the framework of a method that runs on torch or TensorFlow, thread settings.

    framework is the entry the method reported (version, CUDA build, GPU, thread counts); without one, the
    framework among the method's dependencies that is imported is recorded by name and version.
    """
    threads = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    if framework is not None:
        return {"device": device, "framework": _json(dict(framework)), "threads": threads}
    entry = None
    for module in ("torch", "tensorflow"):
        library = sys.modules.get(module)
        if library is not None and any(d.module == module for d in spec.requires):
            entry = {"name": module, "version": str(library.__version__),
                     "cuda": library.version.cuda if module == "torch" else None}
    return {"device": device, "framework": entry, "threads": threads}


def segment(segmentation_input: SegmentationInput, *, config, target: str, seeds: SegmentationResult | None = None,
            device: str = "cpu", label_namespace: str) -> SegmentationResult:
    """Segment one input with a registered method and return its label image on the input's grid.

    The stage wrapper applies, in order: (1) the config is registered in
    ``SEGMENTATION_METHODS`` by its exact type (a ``LabelImportConfig`` is
    refused: import is not a method); (2) the target is one of the method's;
    (3) the input is valid and its roles are accepted and complete; (4) the
    device is ``"cpu"`` or ``"cuda"`` and one of the method's; (5) the
    method's optional dependencies import; (6) a model is resolved and its
    files hashed before any library call; (7) the dimensionality (Z=1 as a
    plane, Z>1 as a volume, the model's own) and the minimum shape; (8) the
    seeds, a nucleus result on the input's grid when the method takes them;
    then (9) the method runs; (10) its labels are checked against the
    input's ZYX shape (a Z axis dropped for one plane is restored) and
    converted to ``uint32``; (11) the run record is built. There is no
    foreground gate: a result without objects has outcome ``empty``.

    ``segment`` reads and writes no file and applies no label operation; values
    are kept as the method returned them.

    Parameters
    ----------
    segmentation_input : SegmentationInput
        The ZYXC image, its grid and the channel roles.
    config
        A frozen config registered in SEGMENTATION_METHODS.
    target : str
        ``nucleus`` or ``cell``.
    seeds : SegmentationResult | None
        Nucleus labels on the input's grid that the method grows from.
    device : str
        ``"cpu"`` (default) or ``"cuda"``; the method must support it.
    label_namespace : str
        The identity scope of the labels, a JSON list ``[dataset_id,
        sample_id, fov_id, subtile_id, run name]`` for a FOV run.

    Returns
    -------
    SegmentationResult
        The labels on ``segmentation_input.grid`` with geometry ``plane``
        (Z=1) or ``volume``, the run record (``methods`` holds the uniform
        provenance entry with ``artifacts``, ``execution`` and ``effective``;
        ``input`` the input's SHA-256, shape, dtype, metadata and channel
        sources; ``seeds`` the seed run and its labels' SHA-256) and the
        method's details and wall time as diagnostics.

    Raises
    ------
    TypeError
        An unregistered config or a LabelImportConfig (check 1), an input that
        is not a SegmentationInput, seeds that are not a SegmentationResult, or
        non-integer labels from the method.
    ValueError
        A target the method does not produce, a role it does not accept or a
        missing required role, an unknown or unsupported device, seeds given
        to a method without seeds or missing for one that needs them, seeds
        with another target, labels of another shape or with negative values.
    IncompatibleGeometryError
        The input's shape differs from its grid, the dimensionality does not
        fit the method or its model, or the seeds are on another grid.
    SegmentationBackendUnavailableError
        A dependency of the method is not installed.
    MissingModelError, ModelHashMismatchError
        The model's files are missing or differ from their expected SHA-256.
    """
    if isinstance(config, LabelImportConfig):
        raise TypeError("import_labels imports masks; it is not a segmentation method")
    spec = spec_for(SEGMENTATION_METHODS, config, _WHAT, TypeError,
                    f"no {_WHAT} is registered for {type(config).__qualname__} (lookup uses the exact config type)")
    _check_target(target)
    if target not in spec.targets:
        raise ValueError(f"{_WHAT} {spec.name!r} produces {', '.join(sorted(spec.targets))}, not {target!r}")
    _check_input(segmentation_input, spec)
    _check_device(device, spec)
    require(spec, _WHAT, SegmentationBackendUnavailableError)
    model, artifacts, n_dim = _resolve_model(spec, config) if spec.models else (None, [], None)
    image, grid = np.asarray(segmentation_input.image), segmentation_input.grid
    shape = image.shape[:3]
    _check_dimensions(shape, spec, n_dim)
    _check_seeds(seeds, spec, grid)
    _check_namespace(label_namespace)

    context = MethodContext(segmentation_input.roles, None if seeds is None else seeds.labels, grid, device, model)
    start = time.perf_counter()
    output = spec.run(image, config, context)
    seconds = time.perf_counter() - start
    if not isinstance(output, tuple) or len(output) != 2 or not isinstance(output[1], Mapping):
        raise ValueError(f"{_WHAT} {spec.name!r} must return (labels, details)")
    labels, details = output
    labels = _check_output(labels, shape, spec)

    geometry = "plane" if shape[0] == 1 else "volume"
    entry = provenance(spec, config, "segmentation")
    entry.update(artifacts=artifacts, execution=_execution(spec, device, details.get("framework")),
                 effective=_json(details.get("effective", {})))
    n_labels, max_label = int(np.count_nonzero(np.unique(labels))), int(labels.max())
    record = {
        "format_version": FORMAT_VERSION, "stage": "segmentation", **_namespace_ids(label_namespace),
        "target": target, "geometry": geometry, "label_namespace": label_namespace, "grid": _grid_record(grid),
        "input": {"path": None, "sha256": array_sha256(image), "file_sha256": None,
                  "shape_zyxc": list(image.shape), "dtype": str(image.dtype),
                  "metadata": _json(asdict(grid.metadata)), "projection": None,
                  "channels": [{"role": role, **_json(dict(source))} for role, source in
                               zip(segmentation_input.roles,
                                   segmentation_input.sources or ({},) * len(segmentation_input.roles))]},
        "seeds": None if seeds is None else {
            "run": seeds.record.get("run"), "label_namespace": seeds.label_namespace,
            "sha256": array_sha256(seeds.labels), "label_rule": details.get("label_rule")},
        "methods": [entry], "operations": [], "outcome": "ok" if max_label else "empty",
        "labels": {"dtype": str(labels.dtype), "sha256": array_sha256(labels), "n_labels": n_labels,
                   "max_label": max_label},
        "software": _software(),
    }
    diagnostics = {"details": _json({k: v for k, v in details.items() if k not in ("effective", "framework")}),
                   "wall_seconds": seconds}
    return SegmentationResult(labels, grid, target, geometry, label_namespace, record, diagnostics)

