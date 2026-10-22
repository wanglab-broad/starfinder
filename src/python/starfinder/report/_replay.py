"""The replay of the examples (docs/report-contract.md, "The replay"; docs/report-algorithms.md, "The replay of
the examples", ``intensity.csv``).

The report rebuilds the images of the main example from its run record (D1): the raw input
of the reference round and of the first moving round, checked against the SHA-256 the run
recorded (step 1), loaded with ``config.pipeline.load`` through ``io.load_round`` and
rotated with ``FOV.run``'s ``_rotated`` (step 2); the recorded ``steps`` entries rerun with
``preprocessing.run_step``, their fitted values compared with the recorded ones (step 3);
on a moving round the saved transforms restored with ``io/_checkpoint.py``:
``_restore_result``, composed and resampled once with the round's ``application_config``
(step 4; no registration is estimated); then the ``post_registration`` entries (step 5).
The image a round ends with is the image the stage ``registered`` saves (step 6), which
the report does not read.

Of each state of a round the replay keeps only what the report shows: the maximum over Z
of each channel (the Preprocessing example), the maximum over channels and Z (the
Registration example) and the rows of ``intensity.csv``; one round's image is held at a
time. A value it cannot rebuild is :class:`._values.Unavailable` with its reason: the whole
examples (no run record, a run that did not reach the reference round's steps, no recipe,
an input missing or changed, replayed fitted values that differ, or a recorded step the
package cannot rerun on the recorded input, which is ``replay_differs``), a moving round
(its ``steps`` phase not recorded), the outputs of its ``post_registration`` entries and the
Registration example (its transforms: ``step_failed``, ``not_recorded``,
``field_not_saved``, ``input_missing``, ``input_changed``; RE1 and PR1).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ._values import Unavailable, first

#: The phases of a recorded entry, in the order ``FOV.run`` runs them; ``input`` is the loaded and rotated round.
PHASES = ("input", "steps", "post_registration")
#: The number of bins of a float output (docs/report-algorithms.md, ``intensity.csv``).
FLOAT_BINS = 256
# The report's outcome of a round's last registration entry ("Outcome"): succeeded only for a succeeded application.
_SUCCEEDED = ("application", "succeeded")


# --- intensity.csv -------------------------------------------------------------------------------------------

def intensity_bins(image, channels):
    """The rows of ``intensity.csv`` of one step output: per channel and bin, in that order, ``(channel, bin,
    bin_low, bin_high, count)``.

    ``lo`` and ``hi`` are the minimum and maximum over every channel of the output. An integer dtype has bins of
    integer width ``w = max(1, ceil((hi − lo + 1) / 256))`` from ``lo``, bin ``k`` being
    ``[lo + k·w − 0.5, lo + (k + 1)·w − 0.5)``; a float dtype 256 equal bins over ``[lo, hi]``, the last closed
    (one bin ``[lo, hi]`` when ``lo = hi``); a boolean output the two bins 0 and 1. Every bin is written, also with
    a count of 0, and each channel's counts sum to its voxels. The image is ZYXC (or ZYX, one channel); it is
    counted plane by plane, so that no copy of the whole image is made.
    """
    image = np.asarray(image)
    if image.ndim == 3:
        image = image[..., None]
    if image.shape[-1] != len(channels):
        raise ValueError(f"an output of {image.shape[-1]} channels has {len(channels)} channel labels")
    kind = image.dtype.kind
    if kind == "b":
        edges = np.array([-0.5, 0.5, 1.5])
        n_bins = 2
    else:
        lo, hi = image.min(), image.max()
        if kind in "ui":
            lo, hi = int(lo), int(hi)
            width = max(1, math.ceil((hi - lo + 1) / FLOAT_BINS))
            n_bins = math.ceil((hi - lo + 1) / width)
            edges = lo + np.arange(n_bins + 1, dtype=np.float64) * width - 0.5
        else:
            lo, hi = float(lo), float(hi)
            n_bins = 1 if lo == hi else FLOAT_BINS
            edges = np.array([lo, hi]) if lo == hi else np.histogram_bin_edges([], bins=FLOAT_BINS, range=(lo, hi))
    rows = []
    for c, channel in enumerate(channels):
        counts = np.zeros(n_bins, dtype=np.int64)
        for z in range(image.shape[0]):
            plane = image[z, ..., c]
            if kind == "b":
                counts += np.bincount(plane.ravel().astype(np.intp), minlength=2)
            elif kind in "ui":
                index = (plane.astype(np.int64) - lo) // width
                counts += np.bincount(index.ravel(), minlength=n_bins)
            elif n_bins == 1:
                counts[0] += plane.size
            else:
                counts += np.histogram(plane, bins=FLOAT_BINS, range=(lo, hi))[0]
        rows += [(channel, k, float(edges[k]), float(edges[k + 1]), int(counts[k])) for k in range(n_bins)]
    return rows


# --- The replay --------------------------------------------------------------------------------------------------

@dataclass
class State:
    """One state of a replayed round: after the loading (``step_index`` 0) or after a recorded entry.

    ``signal`` holds, per channel, the maximum over Z of the round's image in that state (ZYX: the plane itself
    for ``Z = 1``), at full resolution and in the dtype of the state.
    """

    step_index: int
    phase: str
    step: str
    signal: list = field(repr=False)


@dataclass
class RoundReplay:
    """What the replay keeps of one round: its states, its rows of ``intensity.csv``, the maximum over channels and
    Z after its ``steps`` phase (``before``) and, on a moving round, after its transforms (``after``).

    ``post_missing`` is the reason the outputs of its ``post_registration`` entries cannot be rebuilt (a moving
    round whose transforms cannot be rebuilt), ``transform_missing`` the reason its transforms cannot be applied
    (the Registration example; None on the reference round and when they were applied). ``final`` is the image
    the round ends with, kept only on request.
    """

    name: str
    channels: tuple
    role: str
    states: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    before: np.ndarray | None = field(default=None, repr=False)
    after: np.ndarray | None = field(default=None, repr=False)
    post_missing: Unavailable | None = None
    transform_missing: Unavailable | None = None
    transform_source: str | None = None
    final: np.ndarray | None = field(default=None, repr=False)


@dataclass
class Replay:
    """The replay of the main example: its rounds (the reference round first), and the reasons of what it lacks.

    ``missing`` makes all of it unavailable (the Preprocessing example, the intensity distributions and the
    Registration example); ``round_missing`` holds a moving round that could not be replayed at all, with its
    reason. ``steps`` are the entries rerun, each with whether its fitted values equal the recorded ones (up to
    the first that differs). ``record`` is ``report.json`` ``replay``.
    """

    fov_id: str
    reference: str | None
    moving: list
    rounds: dict = field(default_factory=dict)
    missing: Unavailable | None = None
    round_missing: dict = field(default_factory=dict)
    inputs: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    registration: bool = True

    @property
    def record(self) -> dict:
        steps = self.steps
        sources = {r.transform_source for r in self.rounds.values() if r.transform_source is not None}
        return {"fov_id": self.fov_id, "rounds": {"reference": self.reference, "moving": list(self.moving)},
                "inputs": self.inputs,
                "verified": bool(self.inputs) and all(e["verified"] for e in self.inputs),
                "steps": steps,
                "transform_source": sources.pop() if len(sources) == 1 else (sorted(sources) or None),
                "fitted_equal": all(s["fitted_equal"] for s in steps) if steps else None}


def round_transform(record, saved, fov_id, name):
    """``(source, results, application_config)`` of a round's saved transforms: ``run.json`` ``transforms`` (R1),
    else ``registered/transforms.json`` of format version 2 (``saved()`` reads it, or None); or Unavailable with
    ``not_recorded``."""
    results = _get(record, "transforms", name, "results")
    if results:
        return "run.json", results, _get(record, "transforms", name, "application_config")
    saved = saved()
    if saved is not None and saved.get("format_version") == 2 and _get(saved, "transforms", name):
        return "transforms.json", saved["transforms"][name], _get(saved, "applications", name, "application_config")
    return Unavailable("not_recorded", f"neither checkpoints/{fov_id}/run.json transforms.{name} nor "
                                       f"checkpoints/{fov_id}/registered/transforms.json holds the round")


def replay_fov(output_root, fov_id, rounds=None, *, keep_final=False) -> Replay:
    """The replay of one FOV of an output root, outside a report: ``rounds`` None replays every round of its
    ``preprocessing.rounds`` (the reference round first). The files it reads are checked as the report checks
    them."""
    from ._collect import RUN_RECORD, _Inputs
    root = Path(output_root)
    inputs = _Inputs(root)
    record = inputs.json(root / "checkpoints" / fov_id / RUN_RECORD, "run_record")
    if isinstance(record, Unavailable):
        raise FileNotFoundError(f"no run record of {fov_id} under {root}")
    path = root / "checkpoints" / fov_id / "registered" / "transforms.json"

    def saved():
        data = inputs.json(path, "transforms", strict=False) if path.is_file() else None
        return data if isinstance(data, dict) else None
    if rounds is None:
        reference = reference_round(record)
        rounds = [reference] + [r for r in _get(record, "preprocessing", "rounds") or {} if r != reference]
    return replay_example(inputs, root, fov_id, record, list(rounds),
                          lambda name: round_transform(record, saved, fov_id, name), keep_final=keep_final)


def _get(mapping, *keys):
    for key in keys:
        if not isinstance(mapping, dict) or key not in mapping:
            return None
        mapping = mapping[key]
    return mapping


def _signal(image):
    """Per channel, the maximum over Z of a ZYX(C) image (``The displayed signal``)."""
    image = np.asarray(image)
    if image.ndim == 3:
        image = image[..., None]
    return [image[..., c].max(axis=0) for c in range(image.shape[-1])]


def _composite(image):
    """The maximum over channels and over Z of a ZYX(C) image, as float64 (the Registration example)."""
    image = np.asarray(image)
    return (image.max(axis=(0, 3)) if image.ndim == 4 else image.max(axis=0)).astype(np.float64)


def _jsonable_fitted(fitted):
    from starfinder._records import jsonable
    return json.loads(json.dumps(jsonable(dict(fitted)), allow_nan=False))


def reference_round(record):
    """The round ``FOV.run`` processes first: the reference of the registration entries, else the first key of
    ``preprocessing.rounds``."""
    for entries in (record.get("registration") or {}).values():
        for entry in entries or []:
            if entry.get("reference"):
                return entry["reference"]
    for entry in (record.get("transforms") or {}).values():
        if isinstance(entry, dict) and entry.get("reference"):
            return entry["reference"]
    rounds = list(_get(record, "preprocessing", "rounds") or {})
    return rounds[0] if rounds else None


def _load_config(data):
    from starfinder.image import ImageMetadata
    from starfinder.io import ImageLoadConfig
    from starfinder._records import tuples
    from starfinder.io.conversion import ImageConversionConfig
    data = dict(data)
    if data.get("conversion") is not None:
        data["conversion"] = ImageConversionConfig(**tuples(data["conversion"]))
    if data.get("metadata") is not None:
        data["metadata"] = ImageMetadata(**tuples(data["metadata"]))
    return ImageLoadConfig(**{k: tuples(v) for k, v in data.items() if k not in ("conversion", "metadata")},
                           conversion=data.get("conversion"), metadata=data.get("metadata"))


def _round_files(record, rounds, load):
    """Each round's input files: ``run.json`` ``inputs`` lists them as ``FOV.run`` loads them, one chunk of the
    load's channels per round in the order of ``preprocessing.rounds`` (the order ``FOV.run`` processes them)."""
    order = list(_get(record, "preprocessing", "rounds") or {})
    n = len(load.channel_labels)
    entries = record.get("inputs") or []
    files = {}
    for name in rounds:
        if name not in order or len(entries) < (order.index(name) + 1) * n:
            files[name] = None
            continue
        k = order.index(name)
        files[name] = entries[k * n:(k + 1) * n]
    return files


def _round_directory(entries, load):
    """The folder ``load_round`` reads a round from (its ``round_dir / subdir``), from the round's recorded files."""
    path = Path(entries[0]["path"])
    depth = len(Path(load.source_paths[0]).parts) if load.source_paths is not None else 1
    for _ in range(depth):
        path = path.parent
    return path


class _Replayer:
    def __init__(self, inputs, root, fov_id, record, transform_of, keep_final):
        self.inputs, self.root, self.fov_id, self.record = inputs, Path(root), fov_id, record
        self.transform_of, self.keep_final = transform_of, keep_final
        self.references = {}
        self.supplied = None
        self.steps = []

    def _channels(self, name, load):
        entries = _get(self.record, "channels", name)
        if isinstance(entries, list) and len(entries) == len(load.channel_labels):
            return tuple(str(e.get("channel")) for e in entries)
        return tuple(load.channel_labels)

    def check_inputs(self, files):
        """Step 1: every file of the replayed rounds against its recorded SHA-256; the first reason, or None."""
        reasons = []
        listed = []
        for name, entries in files.items():
            for entry in entries:
                recorded = entry.get("sha256")
                data = self.inputs.read(entry["path"], "raw_image", recorded)
                path = self.inputs.relative(entry["path"])
                if isinstance(data, Unavailable):
                    reasons.append(data)
                    known = self.inputs.entries.get(path)
                    listed.append({"round": name, "path": path, "sha256": known["sha256"] if known else None,
                                   "recorded_sha256": recorded, "verified": recorded is not None,
                                   "equal": False if recorded is not None else None})
                    continue
                digest = self.inputs.entries[path]["sha256"]
                listed.append({"round": name, "path": path, "sha256": digest, "recorded_sha256": recorded,
                               "verified": recorded is not None,
                               "equal": digest == recorded if recorded is not None else None})
        return first(*reasons), listed

    def supplied_statistics(self):
        """The supplied-statistics document of a step with ``fit="supplied"``, checked against its SHA-256."""
        if self.supplied is None:
            from starfinder.preprocessing import read_supplied_statistics
            entry = _get(self.record, "preprocessing", "supplied_statistics") or {}
            if not entry.get("path"):
                self.supplied = Unavailable("not_recorded", "preprocessing.supplied_statistics holds no path")
            else:
                data = self.inputs.read(entry["path"], "supplied_statistics", entry.get("sha256"))
                self.supplied = data if isinstance(data, Unavailable) else read_supplied_statistics(entry["path"])
        return self.supplied

    def load(self, name, entries, load, rotation):
        """Step 2: the round loaded with the run's load configuration from the recorded files, and rotated."""
        from starfinder.dataset.fov import _rotated
        from starfinder.io import load_round
        directory = _round_directory(entries, load)
        loaded = load_round(directory, config=replace(load, subdir=""))
        read = [Path(p).resolve() for p in loaded.source_paths]
        if read != [Path(e["path"]).resolve() for e in entries]:
            return Unavailable("input_changed", f"{self.inputs.relative(directory)} now selects the files "
                                                f"{[self.inputs.relative(p) for p in read]}, not those run.json "
                                                "inputs records"), None
        image, metadata = loaded.image, loaded.metadata
        if rotation is not None:
            image, metadata, _ = _rotated(image, metadata, rotation)
        return image, metadata

    def run_entry(self, entry, image, metadata, name, reference_round):
        """One recorded entry rebuilt from its ``step`` and ``config`` and rerun as ``FOV._preprocess`` runs it;
        returns the output, whether the fitted values equal the recorded ones, or the reason it cannot run."""
        from starfinder.io._checkpoint import _config
        from starfinder.preprocessing import StepContext, run_step, step_config_type, step_spec
        config = _config(entry["config"], step_config_type(entry["step"]))
        spec = step_spec(config)
        reference = section = None
        if getattr(config, "fit", None) == "supplied":
            supplied = self.supplied_statistics()
            if isinstance(supplied, Unavailable):
                return supplied, None
            section = supplied["steps"][entry["step"]]
        elif spec.scope == "needs_reference":
            key = (entry["stage"], entry["index"])
            if key not in self.references:
                if name != reference_round:
                    raise ValueError(f"step {entry['step']!r} needs the reference round {reference_round!r} first")
                self.references[key] = np.asarray(image)[..., config.reference_channel].copy()
            reference = self.references[key]
        result = run_step(image, config, StepContext(name, reference_round, metadata, reference, section))
        return np.asarray(result.image), _jsonable_fitted(result.fitted) == entry.get("fitted")

    def transforms(self, name):
        """Step 4: the round's chain and ``WarpConfig`` from its saved transforms, or the reason it has none."""
        from starfinder.io._checkpoint import _config, _field_loader, _registration_methods, _restore_result
        from starfinder.registration import TransformChain, WarpConfig
        entries = _get(self.record, "registration", name) or []
        last = entries[-1] if entries else {}
        if (last.get("record"), last.get("outcome")) != _SUCCEEDED:
            outcome = last.get("outcome") if entries else "not run"
            return Unavailable("step_failed", f"registration.{name}: outcome {outcome}")
        found = self.transform_of(name)
        if isinstance(found, Unavailable):
            return found
        source, results, application = found
        if application is None:
            return Unavailable("not_recorded", f"{source} holds no application_config of {name}")
        directory = self.root / "checkpoints" / self.fov_id / "registered"
        for entry in results:
            kind = _get(entry, "transform", "kind")
            if kind in ("dense", "bspline"):
                key = "field" if kind == "dense" else "coefficients"
                file_name = _get(entry, "transform", key)
                if not file_name:
                    return Unavailable("field_not_saved", f"{source} transforms.{name}: a {kind} step whose field was "
                                                          "not saved (stage registered off)")
                data = self.inputs.read(directory / file_name, "field")
                if isinstance(data, Unavailable):
                    return data
        try:
            fields = _field_loader(directory)
            methods = _registration_methods()
            warp = _config(application, WarpConfig)
            restored = [_restore_result(e, i, fields, 2, warp, methods) for i, e in enumerate(results)]
        except (KeyError, ValueError, OSError) as error:
            return Unavailable("input_changed", f"{source} transforms.{name} cannot be restored: "
                                                f"{type(error).__name__}: {error}")
        return source, TransformChain(tuple(r.transform for r in restored)), warp

    def replay_round(self, name, entries, recorded, load, rotation, reference_round, registration, role):
        """Steps 2 to 5 on one round; the round's states, rows and examples, or the reason it cannot be replayed."""
        from starfinder.registration._chain import resample
        channels = self._channels(name, load)
        out = RoundReplay(name, channels, role)
        image, metadata = self.load(name, entries, load, rotation)
        if isinstance(image, Unavailable):
            return image
        index = 0

        def keep(phase, step, image):
            nonlocal index
            out.states.append(State(index, phase, step, _signal(image)))
            out.rows += [(name, index, phase, step, *row) for row in intensity_bins(image, channels)]
            index += 1

        keep("input", "input", image)
        for phase in ("steps", "post_registration"):
            if phase == "post_registration" and role == "moving":
                out.before = _composite(image)
                if registration:
                    transform = self.transforms(name)
                    if isinstance(transform, Unavailable):
                        out.transform_missing = transform
                    else:
                        out.transform_source, chain, warp = transform
                        image = resample([image], chain, warp)[0]
                        metadata = chain.reference_metadata
                        out.after = _composite(image)
                else:
                    out.transform_missing = Unavailable("not_configured", "config.pipeline.registration is null")
            elif phase == "post_registration":
                out.before = _composite(image)
            post = [e for e in recorded if e.get("stage") == phase]
            if phase == "post_registration" and post and role == "moving" and registration \
                    and out.transform_missing is not None:
                out.post_missing = out.transform_missing
                break
            for entry in post:
                result, equal = self.run_entry(entry, image, metadata, name, reference_round)
                if isinstance(result, Unavailable):
                    return result
                self.steps.append({"round": name, "step_index": index, "phase": phase, "step": entry["step"],
                                  "fitted_equal": equal})
                if not equal:
                    return Unavailable("replay_differs", f"{name} {phase}[{entry['index']}] {entry['step']}: the "
                                                         "replayed fitted values differ from the recorded ones")
                image = result
                keep(phase, entry["step"], image)
        if self.keep_final and (role == "reference" or not registration or out.transform_source is not None):
            out.final = image
        return out


def _expected_entries(record, name):
    """The round's recorded entries in the order ``FOV.run`` runs them, and whether its ``steps`` phase and its
    ``post_registration`` phase are complete (each holds as many entries as the recipe has steps)."""
    recipe = _get(record, "preprocessing", "recipe") or {}
    entries = sorted(_get(record, "preprocessing", "rounds", name) or [],
                     key=lambda e: (PHASES.index(e.get("stage", "steps")), e.get("index", 0)))
    steps = [e for e in entries if e.get("stage") == "steps"]
    post = [e for e in entries if e.get("stage") == "post_registration"]
    return entries, len(steps) == len(recipe.get("steps") or []), len(post) == len(recipe.get("post_registration")
                                                                                   or [])


def _not_reached(record, name, where):
    status = record.get("status")
    if status == "running":
        return Unavailable("run_not_finished", f"run.json: status running before the {where} of {name}")
    if status in ("failed", "interrupted"):
        error = record.get("error") or {}
        at = error.get("step") if error.get("round") is None else f"{error.get('step')}, {error.get('round')}"
        return Unavailable("step_failed", f"run {status} at {at}")
    return Unavailable("not_recorded", f"preprocessing.rounds.{name} does not hold the {where}")


def replay_example(inputs, root, fov_id, record, rounds, transform_of, *, keep_final=False) -> Replay:
    """Replay the main example's rounds: ``rounds[0]`` is the reference round, the others are moving rounds.

    Parameters
    ----------
    inputs : _collect._Inputs
        The report's reader, which lists every file read in ``report.json`` ``inputs``.
    root : Path
        The output root.
    fov_id : str
        The main example.
    record : dict
        Its ``run.json``.
    rounds : sequence of str
        The reference round, then the moving rounds to replay (the report replays the first moving round).
    transform_of : callable
        ``transform_of(round)``: ``(source, results, application_config)`` of the round's saved transforms (R1,
        else ``registered/transforms.json``), or :class:`Unavailable`.
    keep_final : bool
        Keep the image each round ends with (``RoundReplay.final``), which equals the saved registered image.
    """
    reference, moving = rounds[0], [r for r in rounds[1:] if r is not None]
    pipeline = _get(record, "config", "pipeline") or {}
    replay = Replay(fov_id, reference, moving, registration=pipeline.get("registration") is not None)
    if pipeline.get("preprocessing") is None or not _get(record, "preprocessing", "recipe"):
        replay.missing = Unavailable("not_configured", "config.pipeline.preprocessing is null")
        return replay
    if pipeline.get("load") is None:
        replay.missing = Unavailable("not_configured", "config.pipeline.load is null: the run loaded no file")
        return replay
    load = _load_config(pipeline["load"])
    rotation = pipeline.get("rotation_degrees")
    plans = {}
    for name in [reference, *moving]:
        recorded, steps_done, post_done = _expected_entries(record, name)
        if not steps_done or (name == reference and not post_done):
            missing = _not_reached(record, name, "preprocessing")
            if name == reference:
                replay.missing = missing
                return replay
            replay.round_missing[name] = missing
            continue
        plans[name] = recorded
    files = _round_files(record, list(plans), load)
    absent = [name for name, entries in files.items() if entries is None]
    if absent:
        replay.missing = Unavailable("input_missing", f"run.json inputs list no file of the round {absent[0]}")
        return replay
    replayer = _Replayer(inputs, root, fov_id, record, transform_of, keep_final)
    missing, replay.inputs = replayer.check_inputs(files)
    if missing is not None:
        replay.missing = missing
        return replay
    for name, recorded in plans.items():
        role = "reference" if name == reference else "moving"
        try:
            result = replayer.replay_round(name, files[name], recorded, load, rotation, reference,
                                           replay.registration, role)
        except Exception as error:  # a recorded step the package cannot rerun on the recorded input
            result = Unavailable("replay_differs", f"{name}: {type(error).__name__}: {error}")
        replay.steps = replayer.steps
        if isinstance(result, Unavailable):
            replay.missing = result
            replay.rounds = {}
            return replay
        replay.rounds[name] = result
    return replay
