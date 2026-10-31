"""The plan command: validation, the resolved configuration, input discovery, the task graph and the manifest.

design/specs/orchestration.md, "The plan command" and "The task graph from data dependencies", with the additions
of design/specs/orchestration-engines.md ("Additions to the contract": the groups, the temporary outputs, the
fingerprint). The plan reads one configuration file, refuses a configuration it cannot run (``PlanRefused``, exit 2,
before anything is written) and writes ``workflow/resolved.json`` and ``workflow/manifest.json`` under the output
root, each only when its content changes.
"""
from __future__ import annotations

import copy
import json
import os
from importlib.resources import files
from pathlib import Path, PurePosixPath

import numpy as np

from starfinder import _layout
from starfinder.workflow import _translate
from starfinder.workflow._shape import (CHECKPOINT_STAGES, COMMANDS, EXECUTION_DEFAULTS, ID_FIELDS, KIND_THREADS,
    LEFT_KEYS, MANIFEST_FORMAT, MATLAB_KEYS, PROCESSING_LEFT, RECIPES, RESOLVED_FORMAT, TASK_BLOCKS, TILED_DEFAULTS,
    blocks_sha256, canonical, configuration_sha256, sha256)

#: The message of D7: a configuration without backend that holds MATLAB-dialect keys.
MATLAB_DIALECT = ("this configuration has no backend and holds keys of the MATLAB dialect ({keys}); the default "
                  "backend is python: add backend: matlab to run the MATLAB workflow, or migrate it to the Python "
                  "configuration (docs/migration.md, \"The Python configuration shape\")")
MATLAB_BACKEND = ("backend: matlab runs the frozen MATLAB workflow (workflow/Snakefile with backend: matlab); the plan "
                  "command covers the Python workflow only")
#: The message of the D4 refusal the contract shows: an assignment block without the processing block.
ASSIGNMENT_WITHOUT_MOLECULES = ("the assignment block assigns each FOV's molecules, which only the processing block "
                                "produces (with spot_finding, reads_extraction and reads_filtration): add a processing "
                                "block")


class PlanRefused(ValueError):
    """A configuration the plan refuses before any task runs (exit status 2); the message names the key."""


# --- the shape: V1, V2, V9 ---------------------------------------------------------------------------------------

def check_shape(config):
    """V1 and V2: the backend python (the default), the MATLAB dialect refused (D7), and the keys that left."""
    if not isinstance(config, dict):
        raise PlanRefused("the configuration file must hold a mapping")
    backend = config.get("backend")
    if backend is None:
        matlab = [key for key in config if key in MATLAB_KEYS]
        if matlab:
            raise PlanRefused(MATLAB_DIALECT.format(keys=", ".join(matlab)))
    elif backend == "matlab":
        raise PlanRefused(MATLAB_BACKEND)
    elif backend != "python":
        raise PlanRefused(f"backend must be python (the default) or matlab; got {backend!r}")
    left = [key for key in config if key in LEFT_KEYS]
    if left:
        raise PlanRefused("; ".join(f"{key} left the Python configuration: use {LEFT_KEYS[key]}" for key in left)
                          + " (docs/migration.md, \"The Python configuration shape\")")
    processing = config.get("processing")
    if isinstance(processing, dict):
        moved = [key for key in processing if key in PROCESSING_LEFT]
        if moved:
            raise PlanRefused("; ".join(f"processing.{key} left the Python configuration: use {PROCESSING_LEFT[key]}"
                                        for key in moved))
        flags = [f"processing.{name}.run" for name, value in processing.items()
                 if isinstance(value, dict) and "run" in value]
        if flags:
            raise PlanRefused(f"{', '.join(flags)}: the run flags left the Python configuration; a present block "
                              "enables its step (scoring: false turns scoring off)")
    moved = {("assembly", "previews"): "assignment.previews and export.previews (P5)",
             ("assignment", "checkpoints"): "nothing: the assign task always saves the assignment folder"}
    for (block, key), where in moved.items():
        if isinstance(config.get(block), dict) and key in config[block]:
            raise PlanRefused(f"{block}.{key} left the Python configuration: use {where}")


def schema():
    """The JSON Schema of the Python shape (a package resource), with its JSON pointer of each failed key."""
    return json.loads(files("starfinder.workflow").joinpath("schema.json").read_text(encoding="utf-8"))


def validate_schema(config):
    """V9: the configuration against the Python schema with jsonschema, without inserting defaults."""
    import jsonschema
    validator = jsonschema.Draft202012Validator(schema())
    errors = sorted(validator.iter_errors(config), key=lambda e: (list(e.absolute_path), e.message))
    if errors:
        first = errors[0]
        where = ".".join(str(p) for p in first.absolute_path) or "the top level"
        raise PlanRefused(f"{where}: {first.message}")


# --- the resolved configuration --------------------------------------------------------------------------------

def supplied_steps(r):
    """The preprocessing steps with fit: supplied, by method name, in recipe order."""
    steps = ((r.get("processing") or {}).get("preprocessing") or {}).get("steps") or []
    return [s["method"] for s in steps if isinstance(s, dict) and s.get("fit") == "supplied"]


def supplied_pattern(step):
    """The supplied statistics file the merge tasks of the last supplied step write, per sample."""
    return f"{_layout.STATISTICS}/{{sample}}/{step}/{_layout.SUPPLIED_FILE}"


def two_pass(r):
    """The supplied steps the plan summarizes and merges: those of a recipe that names no supplied file (the
    resolved configuration then names the file of the last step's merge)."""
    steps = supplied_steps(r)
    named = ((r.get("processing") or {}).get("preprocessing") or {}).get("supplied_statistics")
    return steps if steps and named == supplied_pattern(steps[-1]) else []


def sequencing_rounds(r):
    return [f"round{i}" for i in range(1, r["n_rounds"] + 1)]


def sequencing_inputs(r):
    """(run, round) of the segmentation inputs that name a sequencing round (they read its registered image)."""
    sequencing = set(sequencing_rounds(r))
    return [(run["name"], item["round"]) for run in (r.get("segmentation") or {}).get("runs") or []
            for item in run.get("inputs") or [] if item.get("round") in sequencing]


def _merged(base, given):
    """``given`` over ``base``: mappings merged key by key, any other value of ``given`` replacing ``base``'s."""
    if isinstance(base, dict) and isinstance(given, dict):
        return {**base, **{k: _merged(base[k], v) if k in base else v for k, v in given.items()}}
    return given


def expand_recipe(processing):
    """processing.recipe expanded (design/specs/orchestration-tiled.md, "The deep-tissue recipe"): the recipe's keys
    with the block's own keys over them (a key the user also sets wins), ``tiled`` only in mode tiled; the name is
    written null, so that the hash covers the steps, not the name."""
    name = processing.get("recipe")
    if name is None:
        return dict(processing, recipe=None)
    expansion = copy.deepcopy(RECIPES[name])
    if processing.get("mode", "direct") != "tiled":
        expansion.pop("tiled", None)
    return dict(_merged(expansion, {k: v for k, v in processing.items() if k != "recipe"}), recipe=None)


def resolve(config, source, output_root=None):
    """The resolved configuration: the given keys with the contract's defaults written out, and ``api``."""
    r = copy.deepcopy(config)
    r["backend"] = "python"
    r.setdefault("selected_fovs", None)
    r.setdefault("additional_round", [])
    r.setdefault("image_dir_template", _layout.IMAGE_DIR_TEMPLATE)
    r.setdefault("readout_mode", "multiplexed")
    r.setdefault("reference_image_import", None)
    # The two settings exclude each other (reference_settings): with an imported reference image the default pattern
    # is not written, so that every task's dataset reads the imported file.
    r.setdefault("reference_image", None if r["reference_image_import"] is not None else _layout.REFERENCE_IMAGE)
    if "processing" in r:
        p = r["processing"] = expand_recipe(dict(r["processing"] or {}))
        p.setdefault("mode", "direct")
        if p["mode"] == "tiled":
            p["tiled"] = {**TILED_DEFAULTS, **(p.get("tiled") or {})}
        elif p.get("tiled") is not None:
            raise PlanRefused("processing.tiled: the tiled settings are read in processing.mode tiled only; remove "
                              "them or set processing.mode: tiled")
        p.setdefault("checkpoints", list(CHECKPOINT_STAGES))
        # D8: the stage registered is on when a data dependency needs it (a segmentation input of a sequencing round).
        if sequencing_inputs(r) and "registered" not in p["checkpoints"]:
            p["checkpoints"] = ["registered", *p["checkpoints"]]
        steps = supplied_steps(r)
        if steps and not (p.get("preprocessing") or {}).get("supplied_statistics"):
            # The two-pass statistics: the merge tasks write the file, one per sample, relative to the output root.
            p["preprocessing"] = dict(p["preprocessing"], supplied_statistics=supplied_pattern(steps[-1]))
        p.setdefault("reference_image_view", {"mode": "max"})
        p.setdefault("streaming", False)
        p.setdefault("device", "cpu")
    if "morphology" in r:
        m = r["morphology"] = dict(r["morphology"] or {})
        m.setdefault("rounds", [e["round_name"] for e in r.get("additional_round") or []])
        m.setdefault("channel_images", None)
    if "segmentation" in r:
        r["segmentation"] = dict(r["segmentation"] or {})
        r["segmentation"].setdefault("device", "cpu")
    if "assignment" in r:
        a = r["assignment"] = dict(r["assignment"] or {})
        a.setdefault("name", "default")
        a.setdefault("cells", "cell")
        a.setdefault("previews", False)
    if "stitching" in r:
        r["stitching"] = dict(r["stitching"] or {})
        r["stitching"].setdefault("method", "bigstitcher")
        r["stitching"].setdefault("input", "reference_merged")
    if "assembly" in r:
        r["assembly"] = dict(r["assembly"] or {})
        r["assembly"].setdefault("assignment", "default")
    if "export" in r or "assembly" in r:
        r["export"] = dict(r.get("export") or {})
        r["export"].setdefault("previews", False)
    if "report" in r:
        r["report"] = dict(r["report"] or {})
    e = r["execution"] = dict(r.get("execution") or {})
    e["defaults"] = {**EXECUTION_DEFAULTS, **(e.get("defaults") or {})}
    e.setdefault("tasks", {})
    e.setdefault("container", None)
    e.setdefault("fiji", None)
    e.setdefault("resume_from_checkpoint", False)
    e.setdefault("code_version", "keep")
    e.setdefault("log_level", "INFO")
    check_names(r)
    if (r.get("processing") or {}).get("mode") == "tiled":
        check_tiled(r)
    try:
        r["api"] = _translate.api(r)
    except (ValueError, TypeError, KeyError) as error:
        raise PlanRefused(str(error).strip("'\"")) from None
    return {"format": RESOLVED_FORMAT, "source": source, "configuration": r}


def check_tiled(r):
    """Tiled execution's checks (design/specs/orchestration-tiled.md, "The halo rule"): the detector, the operations
    without a finite footprint under ``exact``, and the halo, which the plan computes from the footprints and writes
    into the resolved configuration (a configured halo below the minimum is refused, naming the operation)."""
    from starfinder.dataset.tiled import tile_footprints, unbounded_operations
    from starfinder.spot_finding import LocalMaximaConfig
    p, t = r["processing"], r["processing"]["tiled"]
    if not produces_molecules(r):
        raise PlanRefused("processing.mode tiled detects, extracts and decodes in tiles and merges their molecules: "
                          "it needs spot_finding, reads_extraction and reads_filtration; use processing.mode direct")
    if sequencing_inputs(r):
        raise PlanRefused(f"the segmentation inputs {sequencing_inputs(r)} read registered sequencing rounds, which "
                          "processing.mode tiled keeps in its chunked checkpoint only (OT3); use processing.mode direct")
    try:
        pipeline = _translate.processing(r).workflow.pipeline
        grid = _translate.tiled(dict(r, processing=dict(p, tiled=dict(t, halo=t["halo"] or 0))))
    except (ValueError, TypeError, KeyError) as error:
        raise PlanRefused(str(error).strip("'\"")) from None
    from starfinder.dataset.tiled import _detection_configs
    learned = [c.method for c in _detection_configs(pipeline.spot_finding) if not isinstance(c, LocalMaximaConfig)]
    if learned:
        raise PlanRefused(f"spot_finding method {learned[0]!r} in processing.mode tiled: tiled execution detects with "
                          "local_maxima only (OT2); use processing.mode direct")
    if grid.post_local_registration and pipeline.preprocessing is not None and \
            pipeline.preprocessing.extraction_source is not None:
        raise PlanRefused("processing.tiled.post_local_registration with preprocessing.extraction_source is not "
                          "supported: the tile reads the extraction source from the chunked checkpoint")
    if t["exact"]:
        problems = unbounded_operations(pipeline)
        if problems:
            raise PlanRefused("processing.mode tiled with processing.tiled.exact true: " + "; ".join(problems)
                              + "; these have no finite footprint, so a tile cannot equal direct: change them, or "
                              "set processing.tiled.exact: false to accept a result that may differ from direct")
    footprints = tile_footprints(pipeline, grid.local_registration)
    minimum = max(footprints.values(), default=0)
    if t["halo"] is None:
        t["halo"] = minimum
    elif t["halo"] < minimum:
        names = ", ".join(f"{k} ({v})" for k, v in footprints.items() if v == minimum)
        raise PlanRefused(f"processing.tiled.halo {t['halo']} is below the minimum {minimum} that {names} sets "
                          "(the largest footprint of the tile's operations); leave it null or set at least "
                          f"{minimum}")


# --- input discovery ---------------------------------------------------------------------------------------------

def input_root(r):
    return Path(r["root_input_path"]) / r["dataset_id"] / r["sample_id"]


def default_output_root(r):
    return Path(r["root_output_path"]) / r["dataset_id"] / r["output_id"]


def _stat(path, root):
    """path, bytes and mtime_ns of a file: path relative to ``root`` when inside it (OC4), else absolute."""
    st = os.stat(path)
    return {"path": _relative(path, root), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def _relative(path, root):
    path, root = os.path.abspath(path), os.path.abspath(root)
    if os.path.commonpath([path, root]) == root:
        return PurePosixPath(Path(os.path.relpath(path, root))).as_posix()
    return path


def _channel_file(folder, fov, round_name, label):
    """The one file of a channel in a round folder, as io.load_round finds it (``*<label>*.tif`` or ``.tiff``)."""
    matches = sorted(set(folder.glob(f"*{label}*.tif")) | set(folder.glob(f"*{label}*.tiff")))
    if len(matches) != 1:
        found = ", ".join(p.name for p in matches) if matches else "none"
        raise PlanRefused(f"FOV {fov}, round {round_name}: channel label {label!r} must match exactly one file "
                          f"(*{label}*.tif or .tiff) in {folder}; found {len(matches)} ({found})")
    return matches[0]


def _label(entry):
    return entry if isinstance(entry, str) else entry["channel"]


def codebook_path(r):
    """The codebook (or direct panel) file: the key of the readout mode, relative to the input root, else genes.csv."""
    key = "direct_panel" if r.get("readout_mode") == "direct" else "codebook"
    value = r.get(key)
    path = Path(_layout.CODEBOOK_FILE if value is None else value)
    return path if path.is_absolute() else input_root(r) / path


class Inputs:
    """The plan's input discovery: membership, the files and the images, each with path, bytes and mtime_ns."""

    def __init__(self, r, membership, needs_rounds, needs_codebook):
        self.r, self.root = r, input_root(r)
        self.files, self.images = {}, {}
        self.declared = {}
        self.membership = self.add_file("fovs.csv", _layout.fovs_file(self.root))
        self.codebook = None
        if needs_codebook:
            self.codebook = self.add_file("codebook", codebook_path(r), "the codebook" if r.get("readout_mode") !=
                                          "direct" else "the direct panel")
        rounds = {name: [_label(c) for c in r.get("seq_channel_order") or []] for name in sequencing_rounds(r)}
        others = {e["round_name"]: [_label(c) for c in e.get("channel_order") or []]
                  for e in r.get("additional_round") or []}
        stains = [_label(c) for c in r.get("reference_stains") or []]
        for fov in membership.fovs:
            entry = self.images[fov] = {}
            # The reference stain files lie in the reference round's folder; FOV.run does not read them, the
            # morphology task does, so they are listed apart as the image "reference_stain".
            wanted = [*rounds.items(), *others.items()] + ([("reference_stain", stains)] if stains else [])
            for name, channels in wanted:
                if name not in needs_rounds:
                    continue
                folder_round = r["ref_round"] if name == "reference_stain" else name
                folder = _layout.image_dir(self.root, folder_round, fov, r["image_dir_template"])
                if not folder.is_dir():
                    raise PlanRefused(f"FOV {fov}, round {folder_round}: the round's folder {folder} does not exist")
                entry[name] = {}
                for label in channels:
                    stat = _stat(_channel_file(folder, fov, folder_round, label), self.root)
                    entry[name][label] = stat
                    self.declared[f"input:{stat['path']}"] = stat

    def add_file(self, name, path, what=None):
        """Record a supplied file (a relative path is relative to the input root); returns its declared-input form
        (``input:<path>`` inside the input root, else the absolute path)."""
        path = Path(path)
        path = path if path.is_absolute() else self.root / path
        if not path.is_file():
            raise PlanRefused(f"{what or name}: the file {path} does not exist")
        stat = _stat(path, self.root)
        self.files[name] = stat
        declared = stat["path"] if Path(stat["path"]).is_absolute() else f"input:{stat['path']}"
        self.declared[declared] = stat
        return declared

    def image_inputs(self, fov, names):
        return [f"input:{c['path']}" for name in names for c in self.images[fov].get(name, {}).values()]


# --- the task graph ----------------------------------------------------------------------------------------------

def produces_molecules(r):
    p = r.get("processing")
    return p is not None and all(block in p for block in ("spot_finding", "reads_extraction", "reads_filtration"))


def export_route(r):
    return r["api"]["export"]["route"] if "export" in r["api"] else None


def _reference_inputs(r):
    """The segmentation runs that read the reference merged image (directly, or as a composite's amplicon)."""
    return [run["name"] for run in (r.get("segmentation") or {}).get("runs") or []
            for item in run.get("inputs") or [] if item.get("reference_merged") or item.get("prepare")
            and item["prepare"].get("function") == "composite_nuclei_amplicon"]


def _morphology_needed_for(r):
    """The data products that need the morphology task: the saved forms of reference_stain and of other rounds."""
    needs = []
    sequencing = set(sequencing_rounds(r))
    for run in (r.get("segmentation") or {}).get("runs") or []:
        for item in run.get("inputs") or []:
            if item.get("round") is not None and item["round"] not in sequencing:
                needs.append(f"segmentation run {run['name']!r} reads round {item['round']!r}")
    if (r.get("stitching") or {}).get("input") == "reference_stain":
        needs.append("stitching.input reference_stain")
    export = r["api"].get("export")
    if export and export["config"]["fuse"] and "reference_stain" in export["config"]["fusion"]["images"]:
        needs.append("export.images reference_stain")
    return needs


def check_producers(r):
    """V3 (D4): a step whose input no task produces is refused, naming the missing block."""
    errors = []
    if "assignment" in r and not produces_molecules(r):
        errors.append(ASSIGNMENT_WITHOUT_MOLECULES)
    if "assignment" in r and "segmentation" not in r:
        errors.append("the assignment block reads the cell run of the segmentation block: add a segmentation block")
    if "assignment" in r and "segmentation" in r:
        names = [run["name"] for run in r["segmentation"]["runs"]]
        for key in ("cells", "nuclei"):
            if r["assignment"].get(key) is not None and r["assignment"][key] not in names:
                errors.append(f"assignment.{key} names the run {r['assignment'][key]!r}, which the segmentation block "
                              f"does not have (runs {names})")
    if "processing" not in r and not r.get("reference_image_import"):
        if "stitching" in r:
            errors.append("the stitching block places the FOVs on their reference images, which the processing block "
                          "writes: add a processing block, or reference_image_import")
        if _reference_inputs(r):
            errors.append(f"the segmentation runs {_reference_inputs(r)} read the reference merged image, which the "
                          "processing block writes: add a processing block, or reference_image_import")
    if "processing" not in r and sequencing_inputs(r):
        errors.append(f"the segmentation inputs {sequencing_inputs(r)} read registered sequencing rounds, which the "
                      "processing block writes: add a processing block")
    if "segmentation" in r and "processing" not in r and not r.get("reference_image_import") and "morphology" not in r:
        errors.append("a segmentation run lies on the FOV's reference grid, which the processing block's reference "
                      "image or the morphology block's reference stain gives: add one of them")
    if "assembly" in r and "stitching" not in r:
        errors.append("the assembly block reads the stitching block's placement: add a stitching block")
    route = export_route(r)
    if route == "per_fov" and "assignment" not in r:
        errors.append("the export route per_fov assembles each FOV's saved assignment: add an assignment block")
    if route == "per_fov" and "assignment" in r and r["assembly"]["assignment"] != r["assignment"]["name"]:
        errors.append(f"assembly.assignment {r['assembly']['assignment']!r} is not the name of the assignment block "
                      f"({r['assignment']['name']!r})")
    if route in ("molecules_only", "after_fusion") and not produces_molecules(r):
        errors.append(f"the export route {route} reads each FOV's molecules: add a processing block with "
                      "spot_finding, reads_extraction and reads_filtration")
    morphology = _morphology_needed_for(r)
    if morphology and "morphology" not in r:
        errors.append(f"{'; '.join(morphology)}: the morphology block prepares those images: add a morphology block")
    if errors:
        raise PlanRefused("; ".join(errors))


def morphology_field(r):
    """Whether the saved form of a round that morphology.recipe registers holds ``field.npz``: a step whose transform
    is dense or a B-spline (REGISTRATION_METHODS' transform_kind). A step's recovery alternatives are of its step kind
    (RegistrationStep), and every local method's transform is one of these, so no attempt changes it."""
    from starfinder.registration import REGISTRATION_METHODS
    recipe = _translate.morphology(r).recipe
    return recipe is not None and any(REGISTRATION_METHODS[type(step.config)].transform_kind in ("dense", "bspline")
                                      for step in recipe.steps)


def check_names(r, membership=None):
    """FOV ids, sample names, run and assignment names become parts of file names and task ids: none may hold a
    ``/``. The block names are checked before the translators, the FOV and sample names after the membership."""
    if membership is None:
        named = [("segmentation run name", run.get("name")) for run in (r.get("segmentation") or {}).get("runs")
                 or [] if isinstance(run, dict)]
        if isinstance(r.get("assignment"), dict):
            named.append(("assignment name", r["assignment"].get("name")))
    else:
        named = [("FOV id", fov) for fov in membership.fovs] + [("sample", s) for s in membership.samples]
    bad = [f"{what} {value!r}" for what, value in named if not isinstance(value, str) or not value or "/" in value
           or value in (".", "..")]
    if bad:
        raise PlanRefused(f"{', '.join(bad)}: a name becomes part of a file name and of a task id, so it may not be "
                          "empty or hold a '/'")


def _escape(text):
    """A literal part of a group template: braces doubled, as Snakemake's patterns read them."""
    return str(text).replace("{", "{{").replace("}", "}}")


class Graph:
    """The task graph of one plan: the tasks in an order that respects ``needs``, and their groups."""

    def __init__(self, r, membership, inputs, out, manifest_path):
        self.r, self.membership, self.inputs = r, membership, inputs
        self.out = Path(out)
        self.manifest_path = Path(manifest_path)
        self.tasks, self.by_id, self.groups = [], {}, {}

    # -- paths ------------------------------------------------------------------------------------------------

    def rel(self, path):
        """A declared path relative to the output root (POSIX); a path outside it stays absolute."""
        return _relative(path, self.out)

    @staticmethod
    def _fill(text, identity, wildcards):
        # The value of each wildcard is a path component (or a component's stem): replaced component by component.
        parts = []
        for part in text.split("/"):
            replaced = None
            for field in wildcards:
                value = identity[field]
                if part == value:
                    replaced = f"{{{field}}}"
                elif part.startswith(value + "."):
                    replaced = f"{{{field}}}" + _escape(part[len(value):])
                if replaced is not None:
                    break
            parts.append(_escape(part) if replaced is None else replaced)
        return "/".join(parts)

    def execution(self, kind, gpu=False):
        e = self.r["execution"]
        values = {**e["defaults"], **({"threads": KIND_THREADS[kind]} if kind in KIND_THREADS else {}),
                  **(e["tasks"].get(kind) or {})}
        if gpu:
            values["gpu"] = True
        values = {key: values[key] for key in ("threads", "memory_mb", "runtime_min", "gpu")}
        values["container"] = e["container"]
        if kind == "bigstitcher_engine":
            values["tool"] = {"name": "fiji", "path": e["fiji"]}
        return values

    def add(self, kind, identity, needs, inputs, outputs, *, group=None, gpu=False, tool_command=None, temporary=()):
        """Add one task; ``outputs`` are Paths (or output-root-relative strings ending in ``/`` for a folder), and
        ``temporary`` those of them that the group marks temporary (Snakemake's ``temp()``)."""
        parts = [kind] + [str(identity[k]) for k in ID_FIELDS if k in identity]
        task_id = "/".join(parts)
        if task_id in self.by_id:
            raise PlanRefused(f"two tasks have the id {task_id}")
        record = self.rel(_layout.task_record(self.out, task_id))
        log = self.rel(_layout.task_log(self.out, task_id))
        # The record's and the log's templates: the task's file name, each wildcard's part written {field}.
        name = ".".join([kind] + [f"{{{k}}}" if k in ("sample", "step", "fov", "tile") else _escape(identity[k])
                                  for k in ID_FIELDS if k in identity])
        declared = sorted({self.rel(o) if isinstance(o, Path) else o for o in outputs})
        command = ["starfinder", "workflow", COMMANDS[kind], "--manifest", str(self.out / self.manifest_path),
                   "--task", task_id]
        group_name = group or kind
        wildcards = [k for k in ID_FIELDS if k in identity and k not in ("run", "name")]
        task = {"id": task_id, "kind": kind, "group": group_name, "identity": identity,
                "needs": sorted(set(n for n in needs if n)), "inputs": sorted(set(inputs)), "outputs": declared,
                "record": record, "log": log, "command": command}
        if tool_command is not None:
            task["tool_command"] = tool_command
        task["execution"] = self.execution(kind, gpu)
        task["blocks"] = list(TASK_BLOCKS[kind])
        templates = sorted(self._fill(o, identity, wildcards) for o in declared)
        entry = {"name": group_name, "kind": kind, "wildcards": wildcards, "outputs": templates,
                 "record": f"{_layout.WORKFLOW}/{_layout.TASKS}/{name}.json",
                 "log": f"{_layout.WORKFLOW}/{_layout.LOGS}/{name}.log",
                 "temporary": sorted(self._fill(self.rel(o) if isinstance(o, Path) else o, identity, wildcards)
                                     for o in temporary)}
        known = self.groups.setdefault(group_name, entry)
        if known != entry:
            raise PlanRefused(f"the tasks of group {group_name} declare different output templates")
        self.tasks.append(task)
        self.by_id[task_id] = task
        return task_id

    # -- per-FOV products -------------------------------------------------------------------------------------

    def fov_dir(self, fov):
        return _layout.fov_dir(self.out, fov)

    def reference_image(self, fov):
        return self.rel(_layout.reference_image_path(self.out, fov, self.r["reference_image"],
                                                     self.r["reference_image_import"]))

    def stage_records(self, fov):
        """The stage records the fov (or tile_merge) task declares: the stages of processing.checkpoints its operations
        reach; in mode tiled the stage registered is the chunked checkpoint, which the global task writes."""
        p = self.r.get("processing") or {}
        folder = _layout.checkpoint_dir(self.fov_dir(fov))
        reached = {"candidates": "spot_finding" in p, "pre_qc": "reads_filtration" in p,
                   "registered": p.get("mode") != "tiled"}
        names = {"candidates": "candidates.json", "pre_qc": "pre_qc.json",
                 "registered": f"{_layout.REGISTERED}/{_layout.TRANSFORMS_FILE}"}
        return [self.rel(folder / names[stage]) for stage in p.get("checkpoints", []) if reached[stage]]

    def build(self):
        r, out = self.r, self.out
        samples = self.membership.samples
        sequencing = sequencing_rounds(r)
        codebook, fovs_csv = self.inputs.codebook, self.inputs.membership
        # fov_task: the producer of a FOV's run record and molecules (fov, or tile_merge); reference_task: of its
        # reference image (fov, or tile_global).
        fov_task, reference_task, stats_file = {}, {}, {}
        # The two-pass statistics: per sample and supplied step in recipe order, one summarize per FOV and one merge;
        # a later step is summarized on the earlier step's file.
        for sample, members in samples.items():
            previous = None
            for step in two_pass(r):
                folder = _layout.statistics_dir(out, sample, step)
                earlier = [stats_file[sample]] if previous else []
                summaries = [self.add("summarize", {"sample": sample, "step": step, "fov": fov}, [previous],
                                      self.inputs.image_inputs(fov, sequencing) + earlier,
                                      [folder / "histograms" / f"{fov}.npz"], group=f"summarize.{step}")
                             for fov in members]
                previous = self.add("merge", {"sample": sample, "step": step}, summaries,
                                    [self.by_id[t]["outputs"][0] for t in summaries] + earlier,
                                    [folder / _layout.SUPPLIED_FILE], group=f"merge.{step}")
                stats_file[sample] = self.rel(folder / _layout.SUPPLIED_FILE)
        supplied = ((r.get("processing") or {}).get("preprocessing") or {}).get("supplied_statistics")
        supplied_input = None
        if supplied and not two_pass(r):
            supplied_input = self.inputs.add_file("supplied_statistics", supplied_path(r, supplied),
                                                  "processing.preprocessing.supplied_statistics")
        if "processing" in r:
            for sample, members in samples.items():
                merges = [t["id"] for t in self.tasks if t["kind"] == "merge" and t["identity"]["sample"] == sample]
                for fov in members:
                    stats = [stats_file[sample]] if sample in stats_file else (
                        [supplied_input] if supplied_input else [])
                    inputs = self.inputs.image_inputs(fov, sequencing) + [codebook] + stats
                    if r["processing"]["mode"] == "tiled":
                        fov_task[fov], reference_task[fov] = self.tiled(sample, fov, merges[-1:],
                                                                         [i for i in inputs if i])
                        continue
                    outputs = [_layout.run_record(out, fov)]
                    if not r["reference_image_import"]:
                        outputs.append(self.reference_image(fov))
                    if produces_molecules(r):
                        outputs.append(_layout.spot_table(out, fov))
                    outputs += [_layout.checkpoint_dir(self.fov_dir(fov)) / _layout.REGISTERED / f"{name}.ome.tif"
                                for name in sorted({n for _, n in sequencing_inputs(r)})]
                    outputs += self.stage_records(fov)
                    fov_task[fov] = reference_task[fov] = self.add("fov", {"sample": sample, "fov": fov},
                                                                   merges[-1:], [i for i in inputs if i], outputs)
        morph = {}
        if "morphology" in r:
            m = r["morphology"]
            stains = bool(r.get("reference_stains"))
            field = morphology_field(r)
            for sample, members in samples.items():
                for fov in members:
                    names = (["reference_stain"] if stains else []) + list(m["rounds"])
                    outputs = [_layout.nuclei_registration_record(out, fov)]
                    for name in names:
                        folder = _layout.other_round_dir(self.fov_dir(fov), name)
                        outputs += [folder / "image.ome.tif", folder / "registration.json"]
                        if field and name != "reference_stain":
                            # A dense recipe's displacement field (the reference stain is not registered).
                            outputs.append(folder / "field.npz")
                    if m.get("channel_images"):
                        for e in r.get("additional_round") or []:
                            if e["round_name"] in m["rounds"]:
                                outputs += [_layout.image_file(out, fov, e["round_name"], _label(c))
                                            for c in e.get("channel_order") or []]
                    morph[fov] = self.add("morphology", {"sample": sample, "fov": fov}, [],
                                          self.inputs.image_inputs(fov, ["reference_stain", *m["rounds"]]), outputs)
        seg = {}
        if "segmentation" in r:
            s = r["segmentation"]
            for sample, members in samples.items():
                for fov in members:
                    for run in s["runs"]:
                        needs, inputs = [], []
                        fov_dir = self.fov_dir(fov)
                        # The grid: the processing task's reference image, else the morphology task's saved
                        # reference stain (open point OP4).
                        if r["reference_image_import"]:
                            inputs.append(self.reference_image(fov))
                        elif "processing" in r:
                            needs.append(reference_task.get(fov))
                            inputs.append(self.reference_image(fov))
                        else:
                            needs.append(morph[fov])
                            inputs.append(self.rel(_layout.other_round_dir(fov_dir, "reference_stain")
                                                   / "image.ome.tif"))
                        for item in run.get("inputs") or []:
                            name = item.get("round")
                            if name is None:
                                continue
                            if name in sequencing:
                                needs.append(fov_task.get(fov))
                                inputs.append(self.rel(_layout.checkpoint_dir(fov_dir) / _layout.REGISTERED
                                                       / f"{name}.ome.tif"))
                                continue
                            folder = _layout.other_round_dir(fov_dir, name)
                            inputs += [self.rel(folder / "image.ome.tif"), self.rel(folder / "registration.json")]
                            needs.append(morph[fov])
                        if run.get("seeds"):
                            needs.append(seg[(fov, run["seeds"])])
                            inputs += self.by_id[seg[(fov, run["seeds"])]]["outputs"]
                        if run.get("method") == "import":
                            inputs.append(self.inputs.add_file(f"segmentation.{run['name']}.{fov}",
                                                               str(run["path"]).replace("{fov_id}", fov),
                                                               f"segmentation run {run['name']!r} of FOV {fov}"))
                        folder = _layout.segmentation_dir(fov_dir, run["name"])
                        outputs = [folder / "labels.tif", folder / "segmentation.json"]
                        if run.get("method") != "import":
                            outputs.append(folder / "input.ome.tif")
                        seg[(fov, run["name"])] = self.add(
                            "segment", {"sample": sample, "fov": fov, "run": run["name"]}, needs, inputs, outputs,
                            group=f"segment.{run['name']}", gpu=s["device"] == "cuda")
        assign = {}
        if "assignment" in r:
            a = r["assignment"]
            for sample, members in samples.items():
                for fov in members:
                    runs = [a["cells"]] + ([a["nuclei"]] if a.get("nuclei") else [])
                    needs = [fov_task[fov]] + [seg[(fov, run)] for run in runs]
                    inputs = [self.rel(_layout.spot_table(out, fov)), self.reference_image(fov), codebook]
                    inputs += [o for run in runs for o in self.by_id[seg[(fov, run)]]["outputs"]
                               if o.endswith(("labels.tif", "segmentation.json"))]
                    folder = _layout.assignment_dir(self.fov_dir(fov), a["name"])
                    outputs = [folder / n for n in ("molecules.csv", "cells.csv", "counts.csv",
                                                    _layout.ASSIGNMENT_RECORD)]
                    if a.get("nuclei"):
                        outputs.append(folder / "nuclei.csv")
                    if a.get("expansion"):
                        outputs.append(folder / "territories.tif")
                    if a["previews"]:
                        outputs.append(self.rel(_layout.fov_previews_dir(out, fov)) + "/")
                    assign[fov] = self.add("assign", {"sample": sample, "fov": fov, "name": a["name"]}, needs,
                                           inputs, outputs, group=f"assign.{a['name']}")
        placement = {}
        if "stitching" in r:
            st = r["stitching"]
            positions = st.get("positions") or {}
            position_file = None
            if positions.get("source") in ("table", "stage_file") and positions.get(positions["source"]):
                position_file = self.inputs.add_file("positions", positions[positions["source"]],
                                                     f"stitching.positions.{positions['source']}")
            for sample, members in samples.items():
                folder = _layout.stitching_dir(out, sample)
                saved = [folder / n for n in ("placement.json", "fovs.csv", "links.csv")]
                if st["input"] == "reference_stain":
                    sources = [self.rel(_layout.other_round_dir(self.fov_dir(f), "reference_stain") / "image.ome.tif")
                               for f in members]
                    source_needs = [morph[f] for f in members]
                else:
                    sources = [self.reference_image(f) for f in members]
                    source_needs = [reference_task.get(f) for f in members]
                common = [fovs_csv] + ([position_file] if position_file else [])
                grids = [self.reference_image(f) for f in members]
                needs = source_needs + [reference_task.get(f) for f in members]
                if st["method"] == "bigstitcher" and not st.get("dataset_xml"):
                    save = Path(st["bigstitcher_save_dir"]) / sample
                    macro_outputs = [save / "BigStitcher_macro.ijm", save / "view_map.csv",
                                     self.rel(save / "tiles") + "/"]
                    macro = self.add("bigstitcher_macro", {"sample": sample}, needs, common + sources + grids,
                                     macro_outputs)
                    engine_inputs = [self.rel(p) if isinstance(p, Path) else p for p in macro_outputs]
                    # Today's command of the engine rule (_run_bigstitcher_engine), the tool's path a placeholder.
                    engine = self.add("bigstitcher_engine", {"sample": sample}, [macro], engine_inputs,
                                      [save / "dataset.xml"], tool_command=[
                                          "{fiji}", f"-Djava.io.tmpdir={save}", "--ij2", "--headless", "--run",
                                          str(save / "BigStitcher_macro.ijm")])
                    placement[sample] = self.add(
                        "bigstitcher_import", {"sample": sample}, [engine, macro],
                        common + grids + [self.rel(save / n) for n in ("dataset.xml", "view_map.csv",
                                                                       "BigStitcher_macro.ijm")], saved)
                elif st["method"] == "bigstitcher":
                    # The supplied route (D3): the supplied dataset.xml and its view map, no macro or Fiji task.
                    pair = [self.inputs.add_file(f"{key}.{sample}", str(st[key]).replace("{sample}", sample),
                                                 f"stitching.{key} of sample {sample}")
                            for key in ("dataset_xml", "view_map")]
                    placement[sample] = self.add("bigstitcher_import", {"sample": sample}, needs,
                                                 common + grids + pair, saved)
                else:
                    placement[sample] = self.add("stitch", {"sample": sample}, needs, common + sources + grids, saved)
        export = {}
        if "assembly" in r:
            way = export_route(r)
            api = r["api"]["export"]["config"]
            images = api["fusion"]["images"]
            for sample, members in samples.items():
                needs, inputs = [placement[sample]], list(self.by_id[placement[sample]]["outputs"])
                if way == "per_fov":
                    for fov in members:
                        needs.append(assign[fov])
                        inputs += [o for o in self.by_id[assign[fov]]["outputs"] if not o.endswith("/")]
                else:
                    for fov in members:
                        needs.append(fov_task[fov])
                        inputs += [self.rel(_layout.spot_table(out, fov)), self.reference_image(fov)]
                    if codebook:
                        inputs.append(codebook)
                if api["fuse"]:
                    for fov in members:
                        for name in images:
                            if name == "reference_merged":
                                inputs.append(self.reference_image(fov))
                            elif name == "reference_stain":
                                inputs.append(self.rel(_layout.other_round_dir(self.fov_dir(fov), "reference_stain")
                                                       / "image.ome.tif"))
                                needs.append(morph[fov])
                            elif "=file:" in name:
                                # A relative pattern lies under the output root, where Dataset.export reads it
                                # (dataset/_export.py, source_path).
                                path = Path(name.split("=file:", 1)[1].replace("{fov_id}", fov))
                                inputs.append(self.inputs.add_file(
                                    f"export.{name}.{fov}", str(path if path.is_absolute() else out / path),
                                    f"export.images {name} of FOV {fov}"))
                if way == "after_fusion":
                    for run in ((r.get("export") or {}).get("segmentation") or {}).get("runs") or []:
                        if run.get("method") == "import":
                            inputs.append(self.inputs.add_file(f"export.segmentation.{run['name']}",
                                                               str(run["path"]),
                                                               f"export.segmentation run {run['name']!r}"))
                outputs = [_layout.export_dir(out, sample) / _layout.EXPORT_FILE]
                if r["export"]["previews"] and way == "per_fov" and api["assemble"]:
                    outputs.append(self.rel(_layout.export_previews_dir(out, sample)) + "/")
                device = ((r.get("export") or {}).get("segmentation") or {}).get("device")
                export[sample] = self.add("export", {"sample": sample}, needs, inputs, outputs,
                                          gpu=device == "cuda")
        if "report" in r:
            reports = []
            for sample, members in samples.items():
                needs = [fov_task[f] for f in members if f in fov_task]
                inputs = [self.rel(_layout.run_record(out, f)) for f in members if f in fov_task]
                inputs += [path for f in members if f in fov_task for path in self.stage_records(f)
                           if not path.endswith(_layout.TRANSFORMS_FILE)]
                if sample in export:
                    needs.append(export[sample])
                    inputs += self.by_id[export[sample]]["outputs"][:1]
                elif sample in placement:
                    needs.append(placement[sample])
                    inputs += [o for o in self.by_id[placement[sample]]["outputs"] if o.endswith("placement.json")]
                if assign:
                    needs += [assign[f] for f in members]
                    inputs += [o for f in members for o in self.by_id[assign[f]]["outputs"]
                               if o.endswith(_layout.ASSIGNMENT_RECORD)]
                reports.append(self.add("report", {"sample": sample}, needs, inputs,
                                        [_layout.report_dir(out, sample) / "report.json"]))
            root = _layout.report_root(out)
            self.add("compare", {}, reports, [self.by_id[t]["outputs"][0] for t in reports],
                     [root / "samples.csv", root / "index.html"])
        return self.tasks

    # -- tiled execution --------------------------------------------------------------------------------------

    def image_geometry(self, fov):
        """(Z, Y, X) of a FOV's pipeline rounds after rotation, and the bytes per voxel of its chunked checkpoint,
        from the reference round's first channel file (its header alone, as _volume_header reads it)."""
        import tifffile
        from starfinder.io.tiff import _volume_header
        r = self.r
        rounds = self.inputs.images[fov]
        name = r["ref_round"] if r["ref_round"] in rounds else sequencing_rounds(r)[0]
        entry = next(iter(rounds[name].values()))
        path = self.inputs.root / entry["path"]
        header = _volume_header(path, floating=True)
        if header is None or header[0] is None:
            raise PlanRefused(f"FOV {fov}: the image shape of {path} cannot be read from its header, which tiled "
                              "execution's grid needs")
        z, y, x = header[0]
        angle = r.get("rotate_angle") or 0
        if float(angle) % 180 == 90:
            y, x = x, y
        with tifffile.TiffFile(path) as tif:
            dtype = np.dtype(tif.series[0].dtype)
        pipeline = _translate.processing(r).workflow.pipeline
        for step in (pipeline.preprocessing.steps if pipeline.preprocessing is not None else ()):
            if getattr(step.config, "output_dtype", None) is not None:
                dtype = np.dtype(step.config.output_dtype)
        return (int(z), int(y), int(x)), dtype.itemsize

    def tiled(self, sample, fov, needs, inputs):
        """The tiled tasks of one FOV (design/specs/orchestration-tiled.md, "Declared files of the tiled task kinds"):
        tile_global, one tile per tile of the grid, tile_merge; returns (tile_merge's id, tile_global's id).

        The tile size's predicted peak (the resource model) must fit the tile task's memory: a size that does not is
        refused, naming the largest multiple of 128 that fits."""
        from starfinder.dataset.tiled import largest_tile_size, predicted_tile_memory, tile_grid
        r, out = self.r, self.out
        p, t = r["processing"], r["processing"]["tiled"]
        shape, size = self.image_geometry(fov)
        recipe = (p.get("preprocessing") or {})
        rounds = len(set([r["ref_round"], *sequencing_rounds(r)])) * (2 if recipe.get("extraction_source") else 1)
        geometry = dict(z=shape[0], rounds=rounds, channels=len(r.get("seq_channel_order") or []),
                        bytes_per_voxel=size)
        limit = self.execution("tile")["memory_mb"] * 1e6
        predicted = predicted_tile_memory(t["tile_size_yx"], t["halo"], **geometry)
        if predicted > limit:
            fits = largest_tile_size(limit, t["halo"], **geometry)
            raise PlanRefused(
                f"processing.tiled.tile_size_yx {t['tile_size_yx']}: a tile of FOV {fov} ({shape[0]} planes, "
                f"{geometry['rounds']} rounds, {geometry['channels']} channels, {size} bytes per voxel, halo "
                f"{t['halo']}) is predicted to peak at {predicted / 1e6:.0f} MB, above execution.tasks.tile.memory_mb "
                f"{limit / 1e6:.0f}; " + (f"tile_size_yx {fits} fits" if fits else "no tile size fits that memory")
                + " (the tiled record's resource model, 0.21 GB + 1.93 x the window's bytes)")
        folder = self.fov_dir(fov)
        tiled = _layout.tiled_dir(folder)
        store = tiled / _layout.CHUNKED_CHECKPOINT
        outputs = [store, tiled / _layout.GLOBAL_RECORD, tiled / _layout.RUN_RECORD]
        if not r["reference_image_import"]:
            outputs.append(self.reference_image(fov))
        kept = "registered" in p["checkpoints"]
        identity = {"sample": sample, "fov": fov}
        global_task = self.add("tile_global", identity, needs, inputs, outputs, temporary=[] if kept else [store])
        codebook = self.inputs.codebook
        tiles = []
        for tile in tile_grid(shape[1:], t["tile_size_yx"], t["halo"]):
            directory = _layout.tile_dir(folder, tile.id)
            tiles.append(self.add("tile", {"sample": sample, "fov": fov, "tile": tile.id}, [global_task],
                                  [self.rel(store), self.rel(tiled / _layout.GLOBAL_RECORD)]
                                  + ([codebook] if codebook else []),
                                  [directory / name for name in _layout.TILE_FILES]))
        inputs = [o for task in tiles for o in self.by_id[task]["outputs"]]
        inputs += [self.rel(tiled / _layout.GLOBAL_RECORD), self.rel(tiled / _layout.RUN_RECORD)]
        if codebook:
            inputs.append(codebook)
        outputs = [_layout.run_record(out, fov), _layout.spot_table(out, fov)] + self.stage_records(fov)
        merge = self.add("tile_merge", identity, [global_task, *tiles], inputs, outputs)
        return merge, global_task


def supplied_path(r, supplied):
    """A supplied statistics file the configuration names: absolute, or relative to the input root."""
    path = Path(supplied)
    return path if path.is_absolute() else input_root(r) / path


def _needs_rounds(r):
    rounds = set(sequencing_rounds(r) if "processing" in r else [])
    if "morphology" in r:
        rounds |= {"reference_stain", *r["morphology"]["rounds"]}
    return rounds


def _needs_codebook(r):
    route = export_route(r)
    decodes = "processing" in r and "reads_filtration" in r["processing"]
    return decodes or "assignment" in r or route in ("molecules_only", "after_fusion")


# --- the fingerprint and the manifest ------------------------------------------------------------------------------

def fingerprint_components(task, manifest, fingerprints, code=None):
    """The object a task's fingerprint hashes (design/specs/orchestration-engines.md, "The task fingerprint")."""
    entry = {k: v for k, v in task.items() if k not in ("fingerprint", "execution", "command")}
    stats = {}
    for item in manifest["inputs"]["files"].values():
        stats[item["path"] if Path(item["path"]).is_absolute() else f"input:{item['path']}"] = item
    for rounds in manifest["inputs"]["images"].values():
        for labels in rounds.values():
            for item in labels.values():
                stats[f"input:{item['path']}"] = item
    produced = {o for t in manifest["tasks"] for o in t["outputs"]}
    inputs = [stats[i] for i in task["inputs"] if (i.startswith("input:") or Path(i).is_absolute())
              and i not in produced and i in stats]
    components = {"entry": entry, "blocks_sha256": task["blocks_sha256"],
                  "inputs": [{"path": s["path"], "bytes": s["bytes"], "mtime_ns": s["mtime_ns"]} for s in inputs],
                  "needs": [fingerprints[n] for n in task["needs"]]}
    if code is not None:
        components["package"] = code
    return components


def package_identity():
    """The code identity of the running package: version, git_commit, git_dirty (_records.code)."""
    from starfinder import _records
    code = _records.code()
    return {"version": code["version"], "git_commit": code["git_commit"], "git_dirty": code["git_dirty"]}


def plan(config_path, output_root=None) -> dict:
    """Plan one run: validate, resolve, discover the inputs, build the task graph and write the plan's files.

    Parameters
    ----------
    config_path : str or pathlib.Path
        The configuration file (YAML) in the Python shape.
    output_root : str, pathlib.Path or None
        The run's output root; None is ``<root_output_path>/<dataset_id>/<output_id>``.

    Returns
    -------
    dict
        The manifest, as written to ``<output root>/workflow/manifest.json``.

    Raises
    ------
    PlanRefused
        A configuration the plan refuses (exit status 2 of ``starfinder workflow plan``), before anything is
        written; the message names the key, block, FOV, round or label.

    Notes
    -----
    The checks V1 to V10 of design/specs/orchestration.md: the backend (``python``, the default; a configuration
    without backend that holds MATLAB-dialect keys is refused), the keys that left the Python dialect, the Python
    schema (a package resource), the block translators, ``fovs.csv`` and ``selected_fovs``, and the data
    dependencies (a step whose input no task produces). The resolved configuration writes every default the
    contract defines and ``api``, the API configs of each block; its hash leaves out ``execution`` and the roots.
    The manifest lists the inputs (path relative to the input root, bytes, modification time; no content hash), the
    rule groups and the tasks in an order that respects ``needs``, each with ``blocks_sha256`` and ``fingerprint``.
    Each file is rewritten only when its content changes.
    """
    import yaml
    config_path = Path(config_path)
    try:
        text = config_path.read_bytes()
    except OSError as error:
        raise PlanRefused(f"the configuration file {config_path}: {error.strerror or error}") from None
    try:
        config = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise PlanRefused(f"the configuration file {config_path} is not YAML: {error}") from None
    check_shape(config)
    validate_schema(config)
    source = {"path": str(config_path.resolve()), "sha256": sha256(text)}
    resolved = resolve(config, source)
    r = resolved["configuration"]
    out = Path(output_root).resolve() if output_root is not None else default_output_root(r).absolute()
    from starfinder.dataset.workflow import _membership
    try:
        membership = _membership(r, _layout.fovs_file(input_root(r)))
    except (ValueError, FileNotFoundError) as error:
        raise PlanRefused(str(error)) from None
    check_names(r, membership)
    if "stitching" in r:
        # V6: a stage file names the sample's FOVs by the PositionIDs of fovs.csv (_stage_position_ids), checked
        # before any task, as common.smk checked it when the rules were parsed.
        from starfinder.dataset.workflow import _stage_position_ids
        try:
            _stage_position_ids({"stitching": r["stitching"]}, membership)
        except ValueError as error:
            raise PlanRefused(str(error)) from None
    check_producers(r)
    inputs = Inputs(r, membership, _needs_rounds(r), _needs_codebook(r))
    manifest_path = _layout.manifest_file(out)
    graph = Graph(r, membership, inputs, out, manifest_path.relative_to(out))
    tasks = graph.build()
    for task in tasks:
        task["blocks_sha256"] = blocks_sha256(r, task["kind"])
    text = canonical(resolved) + b"\n"
    manifest = {
        "format": MANIFEST_FORMAT, "backend": "python",
        "configuration": {"path": graph.rel(_layout.resolved_file(out)), "sha256": sha256(text),
                          "configuration_sha256": configuration_sha256(r)},
        "package": package_identity(),
        "roots": {"input": str(input_root(r).absolute()), "output": str(out)},
        "membership": {"fovs": list(membership.fovs), "samples": {k: list(v) for k, v in membership.samples.items()},
                       "selected_fovs": r["selected_fovs"]},
        "inputs": {"files": inputs.files, "images": inputs.images},
        "export_route": export_route(r),
        "groups": list(graph.groups.values()),
        "tasks": tasks}
    code = manifest["package"] if r["execution"]["code_version"] == "rerun" else None
    fingerprints = {}
    for task in tasks:
        fingerprints[task["id"]] = sha256(canonical(fingerprint_components(task, manifest, fingerprints, code)))
        task["fingerprint"] = fingerprints[task["id"]]
    _write_if_changed(_layout.resolved_file(out), text)
    _write_if_changed(manifest_path, (json.dumps(manifest, indent=1) + "\n").encode())
    return manifest


def locate(config_path, output_root=None) -> Path:
    """The path of the manifest that the plan of ``config_path`` writes, without planning.

    An engine's parse in a cluster job (Snakemake's ``workflow.is_main_process`` false) reads the manifest that the
    plan of its main process wrote; planning again there would rescan the inputs and could rewrite the manifest while
    other jobs read it (design/specs/orchestration-engines.md, "The Snakefile over the manifest").

    Parameters
    ----------
    config_path : str or pathlib.Path
        The configuration file (YAML) in the Python shape.
    output_root : str, pathlib.Path or None
        The run's output root; None is ``<root_output_path>/<dataset_id>/<output_id>``, as for ``plan``.

    Returns
    -------
    pathlib.Path
        ``<output root>/workflow/manifest.json``.

    Raises
    ------
    PlanRefused
        The file cannot be read, is not YAML, or lacks a key of the default output root.
    """
    import yaml
    try:
        config = yaml.safe_load(Path(config_path).read_bytes())
    except (OSError, yaml.YAMLError) as error:
        raise PlanRefused(f"the configuration file {config_path}: {error}") from None
    if output_root is not None:
        return _layout.manifest_file(Path(output_root).resolve())
    missing = [k for k in ("root_output_path", "dataset_id", "output_id") if not isinstance(config, dict)
               or k not in config]
    if missing:
        raise PlanRefused(f"the configuration file {config_path} has no {', '.join(missing)}")
    return _layout.manifest_file(default_output_root(config).absolute())


def _write_if_changed(path, data):
    """Write ``data`` atomically unless the file holds it already (its modification time then stays)."""
    from starfinder._records import atomic_file
    path = Path(path)
    if path.is_file() and path.read_bytes() == data:
        return False
    with atomic_file(path) as tmp:
        tmp.write_bytes(data)
    return True
