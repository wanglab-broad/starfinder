"""The per-FOV segmentation plan and its coordination (docs/segmentation-contract.md, "Coordination per FOV").

A plan is not a recipe engine: each run names one method or one import, at most one
input function per channel and an ordered tuple of label functions.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace

import numpy as np

from starfinder._registry import check_name, spec_for
from starfinder.dataset.types import UNAVAILABLE
from starfinder.image import IncompatibleGeometryError

from ._import import LabelImportConfig, import_labels
from ._inputs import CompositeConfig, FlamingoEnhancementConfig, composite_nuclei_amplicon, enhance_with_flamingo
from ._labels import SegmentationResult, _check_target, _grid_record, _json, array_sha256
from ._methods import DEVICES, ROLES, SEGMENTATION_METHODS
from ._operations import ExpandLabelsConfig, ZExtensionConfig, expand_labels, extend_labels_through_z
from ._segment import SegmentationInput, segment

_OPERATIONS = (ZExtensionConfig, ExpandLabelsConfig)


def _channel_key(value, name):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or (isinstance(value, str) and not value):
        raise TypeError(f"{name} must be a channel label (a nonempty string) or an index")
    if isinstance(value, int) and value < 0:
        raise ValueError(f"{name} must be a nonnegative index")


@dataclass(frozen=True)
class InputChannel:
    """One channel of a run's segmentation input, taken from the FOV's reference-frame images.

    ``role`` is ``nuclear``, ``cytoplasm``, ``membrane``, ``amplicon`` or
    ``composite``. Either ``round`` and ``channel`` name a loaded round in the
    reference frame and one of its channels (a pattern or a name, resolved by
    ``Dataset.channel_index(round, channel)``, or an index), or ``reference_merged`` is
    True and both are None: the reference round's channel maximum, as
    ``FOV.save_reference_image(reference_image="merged")`` writes it.
    ``prepare`` applies one input function to the channel: a
    :class:`CompositeConfig` combines it (the nuclear stain) with the reference
    merged image (the amplicon) through :func:`composite_nuclei_amplicon`; a
    :class:`FlamingoEnhancementConfig` combines it with the Flamingo channel
    ``prepare_channel`` of the same round (looked up as ``channel``) through
    :func:`enhance_with_flamingo`.

    Raises
    ------
    ValueError
        An unknown role, round or channel given with reference_merged, a missing
        round or channel without it, a prepare function on the reference merged
        image, or prepare_channel without a Flamingo enhancement (or missing
        with one).
    TypeError
        A field of the wrong type.
    """

    role: str
    round: str | None = None
    channel: str | int | None = None
    reference_merged: bool = False
    prepare: CompositeConfig | FlamingoEnhancementConfig | None = None
    prepare_channel: str | int | None = None

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"unknown channel role {self.role!r}; roles are {', '.join(ROLES)}")
        if not isinstance(self.reference_merged, bool):
            raise TypeError("reference_merged must be a bool")
        if self.reference_merged:
            if self.round is not None or self.channel is not None:
                raise ValueError("the reference merged image takes no round and no channel")
            if self.prepare is not None:
                raise ValueError("prepare applies to a round's channel, not to the reference merged image")
        else:
            if not isinstance(self.round, str) or not self.round:
                raise ValueError("an input channel names a round and a channel, or sets reference_merged")
            if self.channel is None:
                raise ValueError(f"an input channel of round {self.round!r} needs a channel")
            _channel_key(self.channel, "channel")
        if self.prepare is not None and not isinstance(self.prepare, (CompositeConfig, FlamingoEnhancementConfig)):
            raise TypeError("prepare must be a CompositeConfig, a FlamingoEnhancementConfig or None")
        if isinstance(self.prepare, FlamingoEnhancementConfig):
            if self.prepare_channel is None:
                raise ValueError("a Flamingo enhancement needs prepare_channel, the round's Flamingo channel")
            _channel_key(self.prepare_channel, "prepare_channel")
        elif self.prepare_channel is not None:
            raise ValueError("prepare_channel names the Flamingo channel of a FlamingoEnhancementConfig")


@dataclass(frozen=True)
class SegmentationRun:
    """One run of a plan: a named label image from one method, or one imported mask.

    ``name`` (snake_case) keys ``FOV.segmentation_results``; ``target`` is
    ``nucleus`` or ``cell``; ``inputs`` are the channels of the segmentation
    input, in channel order (none for an import); ``method`` is a config
    registered in ``SEGMENTATION_METHODS``, or a :class:`LabelImportConfig`
    whose target equals the run's; ``seeds`` names an earlier run of the same
    plan; ``projection`` is an optional Z projection (``ProjectionConfig`` with
    axis ``z``) of the input, which puts the run (an import too) on the
    projected reference grid; ``operations`` are label functions applied in order after the
    method: :class:`ExpandLabelsConfig` (:func:`expand_labels` on the run's grid) and
    :class:`ZExtensionConfig` (which needs a projected run with one input channel
    and extends the plane labels through the unprojected channel). An import
    takes no operations.

    Raises
    ------
    ValueError
        A name that is not snake_case, an unknown target, inputs missing or
        given to an import, repeated roles, a target that differs from the
        import's, seeds or operations on an import, a projection along the
        channels, or an extension without a projection and one input.
    TypeError
        A field of the wrong type or an unregistered method config.
    """

    name: str
    target: str
    inputs: tuple[InputChannel, ...]
    method: object
    seeds: str | None = None
    projection: object | None = None
    operations: tuple = ()

    def __post_init__(self):
        from starfinder.preprocessing import ProjectionConfig
        check_name(self.name, "segmentation run")
        _check_target(self.target)
        if not isinstance(self.inputs, tuple) or not all(isinstance(i, InputChannel) for i in self.inputs):
            raise TypeError("inputs must be a tuple of InputChannel")
        imported = isinstance(self.method, LabelImportConfig)
        if not imported:
            spec_for(SEGMENTATION_METHODS, self.method, "segmentation method", TypeError,
                     f"no segmentation method is registered for {type(self.method).__qualname__}; "
                     "a run's method is a SEGMENTATION_METHODS config or a LabelImportConfig")
        if imported:
            if self.inputs:
                raise ValueError(f"run {self.name!r} imports a mask and takes no inputs")
            if self.method.target != self.target:
                raise ValueError(f"run {self.name!r} has target {self.target!r}, its import {self.method.target!r}")
            if self.seeds is not None:
                raise ValueError(f"run {self.name!r} imports a mask and takes no seeds")
        elif not self.inputs:
            raise ValueError(f"run {self.name!r} needs at least one input channel")
        roles = [i.role for i in self.inputs]
        if len(set(roles)) != len(roles):
            raise ValueError(f"run {self.name!r} repeats a channel role: {roles}")
        if self.seeds is not None and (not isinstance(self.seeds, str) or not self.seeds):
            raise TypeError("seeds must name an earlier run or be None")
        if self.projection is not None:
            if not isinstance(self.projection, ProjectionConfig):
                raise TypeError("projection must be a ProjectionConfig or None")
            if self.projection.axis != "z":
                raise ValueError("a run's projection is along z")
        if not isinstance(self.operations, tuple) or not all(type(o) in _OPERATIONS for o in self.operations):
            raise TypeError("operations must be a tuple of ExpandLabelsConfig and ZExtensionConfig")
        if imported and self.operations:
            raise ValueError(f"run {self.name!r} imports a mask and takes no operations")
        extends = any(type(o) is ZExtensionConfig for o in self.operations)
        if extends and (self.projection is None or len(self.inputs) != 1):
            raise ValueError(f"run {self.name!r}: extend_labels_through_z needs a projected run with one input "
                             "channel, whose unprojected channel is the stain")


@dataclass(frozen=True)
class SegmentationPlan:
    """The ordered runs of one FOV's segmentation; a run's seeds name an earlier run.

    Raises
    ------
    ValueError
        No run, a repeated run name, or seeds that do not name an earlier run.
    TypeError
        runs is not a tuple of SegmentationRun.
    """

    runs: tuple[SegmentationRun, ...]

    def __post_init__(self):
        if not isinstance(self.runs, tuple) or not all(isinstance(r, SegmentationRun) for r in self.runs):
            raise TypeError("runs must be a tuple of SegmentationRun")
        if not self.runs:
            raise ValueError("a segmentation plan needs at least one run")
        seen = []
        for run in self.runs:
            if run.name in seen:
                raise ValueError(f"run name {run.name!r} appears more than once")
            if run.seeds is not None and run.seeds not in seen:
                raise ValueError(f"run {run.name!r}: seeds {run.seeds!r} is not an earlier run of the plan")
            seen.append(run.name)


def _record_sha256(record):
    """SHA-256 of a FOV record as sorted JSON, so the segmentation record refers to it by hash."""
    from starfinder.io._checkpoint import _jsonable
    text = json.dumps(_jsonable(record), sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _check_rounds(fov, plan, grid):
    """Every round an input reads is resident, registered into the reference frame and on the reference grid.

    The reference merged image (an input's own, or the amplicon of a composite)
    is the resident reference round's, else the saved reference image of
    FOV.load_reference_image, which is the grid itself.
    """
    ref = fov.rounds.reference_round
    registered = fov.registration_record.get("rounds", {})
    restored = fov._reference_file is not None and not fov._reference_resident()
    for run in plan.runs:
        names = [ref if i.reference_merged else i.round for i in run.inputs]
        names += [ref for i in run.inputs if isinstance(i.prepare, CompositeConfig)]
        # Only the merged image of the reference round is read: the restored file serves it.
        merged_only = all(i.reference_merged or i.round != ref for i in run.inputs)
        for name in dict.fromkeys(names):
            if name == ref and restored and merged_only:
                continue
            if name not in fov.images or name not in fov.metadata:
                if name == ref:
                    raise ValueError(
                        f"run {run.name!r}: input round {name!r} is not loaded; "
                        + ("a channel of the reference round other than its merged image needs the reference "
                           "round resident (FOV.run or FOV.load_checkpoint('registered')), not only the saved "
                           "reference image of FOV.load_reference_image" if restored else
                           "the reference merged image needs FOV.run, FOV.load_checkpoint('registered') or "
                           "FOV.load_reference_image"))
                raise ValueError(f"run {run.name!r}: input round {name!r} is not loaded")
            if name != ref:
                if name in fov.rounds.sequencing_rounds:
                    if name not in fov.registration_chains:
                        raise ValueError(f"run {run.name!r}: input round {name!r} is not registered to the "
                                         f"reference round {ref!r}")
                elif name not in registered:
                    raise ValueError(f"run {run.name!r}: morphology round {name!r} has no registration entry; "
                                     "call FOV.prepare_morphology (or FOV.register_rounds) or "
                                     "FOV.load_registered_round first")
            shape = np.shape(fov.images[name])[:3]
            if shape != grid.shape_zyx:
                raise IncompatibleGeometryError(f"run {run.name!r}: input round {name!r} has ZYX shape {shape}, "
                                                f"the reference grid {grid.shape_zyx}")
            if fov.metadata[name] != grid.metadata:
                raise ValueError(f"run {run.name!r}: input round {name!r} has metadata {fov.metadata[name]!r}, "
                                 f"unlike the reference round's {grid.metadata!r}")


def _channel_index(fov, round_name, channel):
    """The C index of a channel key: Dataset.channel_index for a pattern or name; an index is checked on the image."""
    image = fov.images[round_name]
    index = fov.dataset._resident_channel_index(round_name, channel) if isinstance(channel, str) else channel
    if np.ndim(image) != 4 or not 0 <= index < np.shape(image)[3]:
        raise ValueError(f"channel {channel!r} is outside round {round_name!r}")
    return index


def _channel_details(fov, round_name, index):
    """name and wavelength (written form) of a round's channel; null and "unavailable" when not configured."""
    channels = fov.dataset._resident_channels(round_name)
    record = channels[index].record() if index < len(channels) else {"name": None, "wavelength": UNAVAILABLE}
    return {"name": record["name"], "wavelength": record["wavelength"]}


def _reference_merged(fov):
    """The reference merged image, ZYX: the reference round's channel maximum, or the saved file restored."""
    return fov._merged_reference()[0]


def _channel(fov, item):
    """One input channel (ZYX) and its source record."""
    ref = fov.rounds.reference_round
    if item.reference_merged:
        image, round_name = _reference_merged(fov), ref
        details = {"name": None, "wavelength": UNAVAILABLE}
    else:
        round_name = item.round
        index = _channel_index(fov, round_name, item.channel)
        image = np.asarray(fov.images[round_name])[..., index]
        details = _channel_details(fov, round_name, index)
    prepared = None
    if isinstance(item.prepare, CompositeConfig):
        image, prepared = composite_nuclei_amplicon(image, _reference_merged(fov), config=item.prepare)
    elif isinstance(item.prepare, FlamingoEnhancementConfig):
        flamingo = np.asarray(fov.images[round_name])[..., _channel_index(fov, round_name, item.prepare_channel)]
        image, prepared = enhance_with_flamingo(image, flamingo, config=item.prepare)
    entry = fov.registration_record.get("rounds", {}).get(round_name) if round_name != ref else None
    source = {"round": item.round, "channel": item.channel, **details, "reference_merged": item.reference_merged,
              "prepare": prepared, "prepare_channel": item.prepare_channel,
              "registration": None if entry is None else {
                  "reference": entry.get("reference"), "reference_sha256": entry.get("reference_sha256"),
                  # The reference stain's relation, and the saved file of an image reloaded by load_registered_round.
                  **{key: entry[key] for key in ("relation", "saved") if key in entry}},
              "sha256": array_sha256(image)}
    path = fov._merged_reference()[1] if item.reference_merged or isinstance(item.prepare, CompositeConfig) else None
    if path is not None:
        # The reference merged image (the channel, or the composite's amplicon) was read from the saved
        # reference image of FOV.load_reference_image: its path below the output root and its file SHA-256.
        source["reference_image"] = {"path": path.relative_to(fov.dataset.output_root).as_posix(),
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return np.ascontiguousarray(image), source


def assemble_input(fov, run, grid):
    """The run's segmentation input from the FOV's resident images, and the unprojected ZYXC stack.

    Each channel is taken (and prepared) in order and stacked as ZYXC (NumPy's
    type promotion gives the dtype); a run with a projection is projected along
    z onto ``grid.projected(method=…)``.
    """
    from starfinder.preprocessing import project_image
    channels, sources = zip(*(_channel(fov, item) for item in run.inputs))
    stack = np.ascontiguousarray(np.stack(channels, axis=-1))
    image, run_grid = stack, grid
    if run.projection is not None:
        image = np.ascontiguousarray(project_image(stack, config=run.projection))
        run_grid = grid.projected(method=run.projection.method)
    return SegmentationInput(image, run_grid, tuple(i.role for i in run.inputs), tuple(sources)), stack


def _extend(result, stain, grid, run, config):
    """extend_labels_through_z on a projected run: the labels return to the reference grid as ``extended``."""
    labels, record = extend_labels_through_z(result.labels, stain, grid.metadata, config=config)
    record = dict(record, source_run=run.name, plane_grid=_grid_record(result.grid))
    max_label = int(labels.max())
    run_record = dict(result.record, geometry="extended", grid=_grid_record(grid),
                      operations=[*result.record["operations"], {"operation": "extend_labels_through_z",
                                                                 "config": record["config"], "record": record}],
                      outcome="ok" if max_label else "empty",
                      labels={"dtype": str(labels.dtype), "sha256": array_sha256(labels),
                              "n_labels": int(np.count_nonzero(np.unique(labels))), "max_label": max_label})
    return SegmentationResult(labels, grid, result.target, "extended", result.label_namespace, run_record,
                              result.diagnostics)


def _expand(result, config):
    """expand_labels on a run's labels, on its own grid; the record lists the operation."""
    labels, record = expand_labels(result.labels, result.grid.metadata, config=config)
    max_label = int(labels.max())
    run_record = dict(result.record,
                      operations=[*result.record["operations"], {"operation": "expand_labels",
                                                                 "config": record["config"], "record": record}],
                      outcome="ok" if max_label else "empty",
                      labels={"dtype": str(labels.dtype), "sha256": array_sha256(labels),
                              "n_labels": int(np.count_nonzero(np.unique(labels))), "max_label": max_label})
    return SegmentationResult(labels, result.grid, result.target, result.geometry, result.label_namespace,
                              run_record, result.diagnostics)


def segment_fov(fov, plan, *, device="cpu", inputs=None):
    """Run a plan on one FOV; return the results by run name (FOV.segment stores them).

    ``inputs``, when a dict, receives each computed run's SegmentationInput by name
    (the saved format writes it); an import has none.
    """
    if not isinstance(plan, SegmentationPlan):
        raise TypeError("plan must be a SegmentationPlan")
    plan.__post_init__()
    if not isinstance(device, str) or device not in DEVICES:
        raise ValueError(f"device must be 'cpu' or 'cuda'; got {device!r}")
    grid = fov.reference_grid()
    _check_rounds(fov, plan, grid)
    upstream = {"preprocessing_record_sha256": _record_sha256(fov.preprocessing_record),
                "registration_record_sha256": _record_sha256(fov.registration_record)}
    results = {}
    for run in plan.runs:
        namespace = json.dumps([fov.dataset.dataset_id, fov.dataset.sample_id, fov.fov_id, fov.subtile_id,
                                run.name], separators=(",", ":"))
        if isinstance(run.method, LabelImportConfig):
            config = run.method
            run_grid = grid if run.projection is None else grid.projected(method=run.projection.method)
            result = import_labels(config.path, grid=run_grid, target=config.target, geometry=config.geometry,
                                   relabel=config.relabel, label_namespace=namespace)
            record = dict(result.record)
        else:
            segmentation_input, stack = assemble_input(fov, run, grid)
            if inputs is not None:
                inputs[run.name] = segmentation_input
            seeds = None if run.seeds is None else results[run.seeds]
            result = segment(segmentation_input, config=run.method, target=run.target, seeds=seeds, device=device,
                             label_namespace=namespace)
            for operation in run.operations:
                if type(operation) is ExpandLabelsConfig:
                    result = _expand(result, operation)
                else:
                    result = _extend(result, stack[..., 0], grid, run, operation)
            record = dict(result.record)
            if run.projection is not None:
                record["input"] = dict(record["input"], projection=_json(asdict(run.projection)))
        results[run.name] = replace(result, record=dict(record, upstream=upstream))
    return results

