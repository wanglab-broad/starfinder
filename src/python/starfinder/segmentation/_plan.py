"""The per-FOV segmentation plan and its coordination (design/specs/segmentation.md, "Coordination per FOV").

Each run names one method or one import, its source channels, an optional input recipe
(design/specs/segmentation-input-recipe.md: the input recipe is the one chain, fixed
stages of registered steps) and an ordered tuple of label functions.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from starfinder._records import UNAVAILABLE, file_sha256
from starfinder._registry import check_name, spec_for
from starfinder.image import IncompatibleGeometryError

from ._import import LabelImportConfig, import_labels
from ._labels import SegmentationResult, _check_target, _grid_record, _json, array_sha256
from ._methods import DEVICES, ROLES, SEGMENTATION_METHODS, SeededWatershedConfig
from ._operations import ExpandLabelsConfig, ZExtensionConfig, expand_labels, extend_labels_through_z
from ._recipe import InputRecipe, apply_input_recipe, check_model_input, method_value, model_factor, model_roles
from ._segment import segment

_OPERATIONS = (ZExtensionConfig, ExpandLabelsConfig)


def _channel_key(value, name):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or (isinstance(value, str) and not value):
        raise TypeError(f"{name} must be a channel label (a nonempty string) or an index")
    if isinstance(value, int) and value < 0:
        raise ValueError(f"{name} must be a nonnegative index")


@dataclass(frozen=True)
class InputChannel:
    """One source channel of a run, taken from the FOV's reference-frame images.

    Parameters
    ----------
    role : str
        Source role: ``nuclear``, ``cytoplasm``, ``membrane``, ``amplicon`` or ``composite``.
    round : str or None
        Name of the loaded round supplying the channel.
    channel : str or int or None
        Channel pattern, name or zero-based index within the round.
    reference_merged : bool
        Select the reference round's channel-maximum image.

    Notes
    -----
    Either ``round`` and ``channel`` name a loaded round in the reference frame
    and one of its channels (a pattern or a name, resolved by
    ``Dataset.channel_index(round, channel)``, or an index), or
    ``reference_merged`` is True and both are None: the reference round's
    channel maximum, as ``FOV.save_reference_image(reference_image="merged")``
    writes it. How the source channels become the model's channels is the
    run's input recipe (:class:`InputRecipe`): the DAPI–amplicon composite is
    two sources, ``nuclear`` and ``amplicon`` (the reference merged image),
    with ``combine`` a :class:`CompositeConfig`; the Flamingo enhancement is
    ``nuclear`` and ``cytoplasm`` (the Flamingo channel) with a
    :class:`FlamingoEnhancementConfig`.

    Raises
    ------
    ValueError
        An unknown role, round or channel given with reference_merged, or a
        missing round or channel without it.
    TypeError
        A field of the wrong type.
    """

    role: str
    round: str | None = None
    channel: str | int | None = None
    reference_merged: bool = False

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"unknown channel role {self.role!r}; roles are {', '.join(ROLES)}")
        if not isinstance(self.reference_merged, bool):
            raise TypeError("reference_merged must be a bool")
        if self.reference_merged:
            if self.round is not None or self.channel is not None:
                raise ValueError("the reference merged image takes no round and no channel")
        else:
            if not isinstance(self.round, str) or not self.round:
                raise ValueError("an input channel names a round and a channel, or sets reference_merged")
            if self.channel is None:
                raise ValueError(f"an input channel of round {self.round!r} needs a channel")
            _channel_key(self.channel, "channel")


@dataclass(frozen=True)
class SegmentationRun:
    """One run of a plan: a named label image from one method, or one imported mask.

    Parameters
    ----------
    name : str
        Snake_case run name used as the key in ``FOV.segmentation_results``.
    target : str
        Label target: ``nucleus`` or ``cell``.
    inputs : tuple[InputChannel, ...]
        Source channels in the order of the run's inputs.
    method : object
        Segmentation method config or ``LabelImportConfig`` for a mask import.
    seeds : str or None
        Name of an earlier run supplying seed labels; ``None`` selects no seed run.
    input_recipe : InputRecipe or None
        How the source channels become the image the method receives; ``None``
        is the empty recipe (the sources stacked as ZYXC in their dtype).
    operations : tuple
        Label operations in application order, using ``ExpandLabelsConfig`` and ``ZExtensionConfig``.

    Notes
    -----
    ``name`` (snake_case) keys ``FOV.segmentation_results``; ``target`` is
    ``nucleus`` or ``cell``; ``inputs`` are the source channels (none for an
    import); ``method`` is a config registered in ``SEGMENTATION_METHODS``, or
    a :class:`LabelImportConfig` whose target equals the run's (its
    ``projection`` puts the import on the projected reference grid);
    ``seeds`` names an earlier run of the same plan; ``input_recipe`` is
    applied by :func:`apply_input_recipe` (its ``projection`` puts the run on
    the projected reference grid, and a method with a model needs its
    ``model_input``); ``operations`` are label functions applied in order after the
    method: :class:`ExpandLabelsConfig` (:func:`expand_labels` on the run's grid) and
    :class:`ZExtensionConfig` (which needs a recipe with a projection and one
    model channel, and extends the plane labels through that model channel
    before the projection). An import takes no recipe and no operations.

    Raises
    ------
    ValueError
        A name that is not snake_case, an unknown target, inputs missing or
        given to an import, repeated roles, a target that differs from the
        import's, seeds, a recipe or operations on an import, source roles
        that differ from the recipe's combine method, a step role the channels
        do not have, a model input missing for a method with a model or given
        to one without, or an extension without a projection and one model
        channel.
    TypeError
        A field of the wrong type or an unregistered method config.
    """

    name: str
    target: str
    inputs: tuple[InputChannel, ...]
    method: object
    seeds: str | None = None
    input_recipe: InputRecipe | None = None
    operations: tuple = ()

    def __post_init__(self):
        check_name(self.name, "segmentation run")
        _check_target(self.target)
        if not isinstance(self.inputs, tuple) or not all(isinstance(i, InputChannel) for i in self.inputs):
            raise TypeError("inputs must be a tuple of InputChannel")
        imported = isinstance(self.method, LabelImportConfig)
        if not imported:
            spec_for(SEGMENTATION_METHODS, self.method, "segmentation method", TypeError,
                     f"no segmentation method is registered for {type(self.method).__qualname__}; "
                     "a run's method is a SEGMENTATION_METHODS config or a LabelImportConfig")
        if self.input_recipe is not None and not isinstance(self.input_recipe, InputRecipe):
            raise TypeError("input_recipe must be an InputRecipe or None")
        if imported:
            if self.inputs:
                raise ValueError(f"run {self.name!r} imports a mask and takes no inputs")
            if self.method.target != self.target:
                raise ValueError(f"run {self.name!r} has target {self.target!r}, its import {self.method.target!r}")
            if self.seeds is not None:
                raise ValueError(f"run {self.name!r} imports a mask and takes no seeds")
            if self.input_recipe is not None:
                raise ValueError(f"run {self.name!r} imports a mask and takes no input_recipe; the import's "
                                 "projection (LabelImportConfig.projection) puts it on a projected grid")
        elif not self.inputs:
            raise ValueError(f"run {self.name!r} needs at least one input channel")
        roles = [i.role for i in self.inputs]
        if len(set(roles)) != len(roles):
            raise ValueError(f"run {self.name!r} repeats a channel role: {roles}")
        if self.seeds is not None and (not isinstance(self.seeds, str) or not self.seeds):
            raise TypeError("seeds must name an earlier run or be None")
        if not isinstance(self.operations, tuple) or not all(type(o) in _OPERATIONS for o in self.operations):
            raise TypeError("operations must be a tuple of ExpandLabelsConfig and ZExtensionConfig")
        if imported:
            if self.operations:
                raise ValueError(f"run {self.name!r} imports a mask and takes no operations")
            return
        modelled = model_roles(self.input_recipe, roles, run=self.name)
        check_model_input(self.method, self.input_recipe, run=self.name)
        extends = any(type(o) is ZExtensionConfig for o in self.operations)
        projection = None if self.input_recipe is None else self.input_recipe.projection
        if extends and (projection is None or len(modelled) != 1):
            raise ValueError(f"run {self.name!r}: extend_labels_through_z needs an input recipe with a projection "
                             "and one model channel, whose channel before the projection is the stain")


@dataclass(frozen=True)
class SegmentationPlan:
    """The ordered runs of one FOV's segmentation; a run's seeds name an earlier run.

    Parameters
    ----------
    runs : tuple of SegmentationRun
        Named segmentation runs in execution order.

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
    from starfinder._records import jsonable
    text = json.dumps(jsonable(record), sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _check_rounds(fov, runs, grid):
    """Every round an input of ``runs`` reads is resident, registered into the reference frame and on the reference
    grid.

    The reference merged image (an input's own, such as the amplicon of a
    composite) is the resident reference round's, else the saved reference
    image of FOV.load_reference_image, which is the grid itself.
    """
    ref = fov.rounds.reference_round
    registered = fov.registration_record.get("rounds", {})
    restored = fov._reference_file is not None and not fov._reference_resident()
    for run in runs:
        names = [ref if i.reference_merged else i.round for i in run.inputs]
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
                                     "call FOV.prepare_morphology (or FOV.register on it) or "
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
    """One source channel (ZYX) and its source record."""
    ref = fov.rounds.reference_round
    if item.reference_merged:
        image, round_name = _reference_merged(fov), ref
        details = {"name": None, "wavelength": UNAVAILABLE}
    else:
        round_name = item.round
        index = _channel_index(fov, round_name, item.channel)
        image = np.asarray(fov.images[round_name])[..., index]
        details = _channel_details(fov, round_name, index)
    entry = fov.registration_record.get("rounds", {}).get(round_name) if round_name != ref else None
    source = {"round": item.round, "channel": item.channel, **details, "reference_merged": item.reference_merged,
              "registration": None if entry is None else {
                  "reference": entry.get("reference"), "reference_sha256": entry.get("reference_sha256"),
                  # The reference stain's relation, and the saved file of an image reloaded by load_registered_round.
                  **{key: entry[key] for key in ("relation", "saved") if key in entry}},
              "sha256": array_sha256(image)}
    path = fov._merged_reference()[1] if item.reference_merged else None
    if path is not None:
        # The reference merged image was read from the saved reference image of FOV.load_reference_image: its path
        # below the output root (an imported or configured file outside it: its absolute path) and its file SHA-256.
        root = Path(fov.dataset.output_root).absolute()
        absolute = Path(path).absolute()
        recorded = (absolute.relative_to(root).as_posix() if absolute.is_relative_to(root) else str(absolute))
        source["reference_image"] = {"path": recorded, "sha256": file_sha256(path)}
    return np.ascontiguousarray(image), source


def assemble_input(fov, run, grid):
    """The run's segmentation input from the FOV's resident images, the unprojected model channels, the recipe
    record and the source channels' records.

    Each source channel is taken in order and the run's input recipe is applied
    by :func:`apply_input_recipe` on the reference grid (a projection puts the
    input on ``grid.projected(method=…)``). The source records are the
    ``input.channels`` of the run record: one per source channel, its role first.
    """
    sources = [(item.role, *_channel(fov, item)) for item in run.inputs]
    segmentation_input, unprojected, recipe = apply_input_recipe(sources, grid, recipe=run.input_recipe, run=run.name)
    return segmentation_input, unprojected, recipe, _json([{"role": role, **record} for role, _, record in sources])


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


def selected_runs(plan, run=None):
    """The runs of ``plan`` that FOV.segment computes: every run, or the one named ``run``."""
    if run is None:
        return plan.runs
    selected = tuple(r for r in plan.runs if r.name == run)
    if not selected:
        raise ValueError(f"run {run!r} is not a run of the plan (runs: {[r.name for r in plan.runs]})")
    return selected


def segment_fov(fov, plan, *, device="cpu", inputs=None, only=None):
    """Run a plan on one FOV; return the results by run name (FOV.segment stores them).

    ``inputs``, when a dict, receives each computed run's SegmentationInput by name
    (the saved format writes it); an import has none. ``only`` names the one run to
    compute (None: every run); the seeds of a run that this call does not compute are
    the FOV's ``segmentation_results`` (a saved run restored by FOV.load_segmentation).
    Each run's record holds ``wall_s``, the wall time of the run in seconds: its input,
    method (or import) and label operations.
    """
    if not isinstance(plan, SegmentationPlan):
        raise TypeError("plan must be a SegmentationPlan")
    plan.__post_init__()
    if not isinstance(device, str) or device not in DEVICES:
        raise ValueError(f"device must be 'cpu' or 'cuda'; got {device!r}")
    runs = selected_runs(plan, only)
    for item in runs:
        if item.seeds is not None and item.seeds not in {r.name for r in runs} and \
                item.seeds not in fov.segmentation_results:
            raise ValueError(f"run {item.name!r}: its seeds {item.seeds!r} are not in segmentation_results; segment "
                             "that run first, or restore it with FOV.load_segmentation")
    grid = fov.reference_grid()
    _check_rounds(fov, runs, grid)
    upstream = {"preprocessing_record_sha256": _record_sha256(fov.preprocessing_record),
                "registration_record_sha256": _record_sha256(fov.registration_record)}
    results = {}
    for run in runs:
        started = time.perf_counter()
        namespace = json.dumps([fov.dataset.dataset_id, fov.dataset.sample_id, fov.fov_id, fov.tile,
                                run.name], separators=(",", ":"))
        if isinstance(run.method, LabelImportConfig):
            config = run.method
            run_grid = grid if config.projection is None else grid.projected(method=config.projection)
            result = import_labels(config.path, grid=run_grid, target=config.target, geometry=config.geometry,
                                   relabel=config.relabel, label_namespace=namespace)
            record = dict(result.record)
            if config.projection is not None:
                record["import"] = dict(record["import"], projection=config.projection)
        else:
            if run.input_recipe is not None and run.input_recipe.model_input is not None:
                # Amendment A and the factor checks, on this grid, before any step runs.
                factor = model_factor(run.input_recipe.model_input, grid.metadata)["factor_zyx"]
                check_model_input(run.method, run.input_recipe, run=run.name, factor_zyx=factor)
            segmentation_input, stack, recipe, channels = assemble_input(fov, run, grid)
            if isinstance(run.method, SeededWatershedConfig) and segmentation_input.grid.metadata.spacing_zyx is None:
                raise ValueError("seeded_watershed needs a spacing: set voxel_size_xy and voxel_size_z "
                                 "(micrometres) before loading raw images, or supply a reference image with "
                                 "known spacing")
            if inputs is not None:
                inputs[run.name] = segmentation_input
            seeds = None if run.seeds is None else results.get(run.seeds, fov.segmentation_results.get(run.seeds))
            result = segment(segmentation_input, config=run.method, target=run.target, seeds=seeds, device=device,
                             label_namespace=namespace)
            if recipe["model_input"] is not None:
                recipe["model_input"]["method_value"] = method_value(result.record)
            result = replace(result, record=dict(result.record, input=dict(result.record["input"], channels=channels,
                                                                            recipe=recipe)))
            for operation in run.operations:
                if type(operation) is ExpandLabelsConfig:
                    result = _expand(result, operation)
                else:
                    result = _extend(result, stack[..., 0], grid, run, operation)
            record = dict(result.record)
        results[run.name] = replace(result, record=dict(record, upstream=upstream,
                                                        wall_s=time.perf_counter() - started))
    return results

