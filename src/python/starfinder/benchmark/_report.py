"""Private saved-evaluation report behind ``starfinder experiment report``, in W-237's review format.

design/specs/benchmark.md, "Proposed public names and migration"; design/proposals/benchmark-drafts/report-template.md.
The renderer reads a saved evaluation (``evaluation.json``, ``values.csv`` and ``metrics/`` of ``starfinder experiment
evaluate``), the saved records its sources name (a trial's ``record.json``, a workflow task record) and the saved files
a figure names, each checked against the SHA-256 that the evaluation or a checked record holds. An optional protocol
supplies the text a renderer cannot know: purposes, targeted conditions, the fixed cut-offs, supplied precondition,
harm and verdict results, findings and hypotheses. Nothing is processed, detected or generated and no metric is
computed: the only numbers derived are ranges (minimum and maximum) of saved values and, for a list-valued value, the
mean over its items, each labelled where it is shown.

The page puts the summary first (setup; one card per method or mode; cross-cutting findings and anomalies, each
labelled finding or hypothesis with its source table; the reading order), then the method details and figures, then
the appendix with the full tables. A card shows a verdict only when the protocol supplies one, with its failing
clause. Nothing is written when a saved file, the protocol or the revision does not check.
"""
from __future__ import annotations

import html
import io
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
import yaml

REPORT_FORMAT = "starfinder.experiment.report/1"
EVALUATION_FORMAT = "starfinder.experiment.evaluation/1"
TRIAL_FORMAT = "starfinder.benchmark.trial/1"
#: The columns of the evaluation's values.csv, in order.
VALUE_COLUMNS = ["metric", "metric_id", "function", "outcome", "status", "value_name", "value", "unit", "reason"]
#: The files whose change since a clean evaluation's revision refuses the report (relative to the checkout).
EVALUATION_SOURCES = ("src/python/starfinder/benchmark/_evaluate.py", "src/python/starfinder/evaluation")
#: A metric row's identity: the method and mode name its card; the rest place it in the card's tables.
IDENTITY = ("method", "mode", "condition", "seed", "fov", "side")
#: The identity fields a protocol assigns (which card and side a row belongs to); the others are facts of the saved
#: records, which a protocol may fill but not contradict. A row's own trial method is kept as ``trial_method``.
ASSIGNED = ("method", "mode", "side")
SIDES = ("before", "after")
LABELS = ("finding", "hypothesis")
#: The protocol's keys and what each holds; every key is optional.
PROTOCOL_KEYS = {
    "title": "The report's title",
    "qualification": "Pilot or scientific qualification of the evidence, as approved",
    "route": "The task, trial or checkpoint route the evaluated outputs came from",
    "inputs": "Where the inputs came from (dataset, scene preset and version, generation record)",
    "truth": "Where the truth came from",
    "dtype": "The image dtype or dtypes evaluated",
    "resources": "How the resources were measured, or why some are absent",
    "conditions": "A mapping of condition name to one sentence describing it",
    "cutoffs": "The fixed cut-offs as a list of {name, value, unit, note}",
    "missing_prerequisites": "A list of prerequisites that are missing (metrics, adapters, truth)",
    "metrics": "A mapping of short-set ID or function to {meaning, population}, shown beside the built-in meaning",
    "rows": "A mapping of an evaluation metric name to its card's method and mode, its side (before or after), and "
            "its condition, seed and fov where the saved records give none (a value they give cannot be changed)",
    "methods": "The cards in page order: {method, mode, title, purpose, targeted_conditions, key_metrics, "
               "preconditions, harm, verdict, limitations}",
    "findings": "A list of {label: finding or hypothesis, text, source_table}",
    "figures": "A list of {method, mode, title, panels, detections, truth, peaks} drawing saved trial or files images",
}
_METHOD_KEYS = {"method", "mode", "title", "purpose", "targeted_conditions", "key_metrics", "preconditions", "harm",
                "verdict", "limitations"}
_FIGURE_KEYS = {"method", "mode", "title", "panels", "detections", "truth", "peaks"}
_FILE_KEYS = {"label", "source", "output", "input", "file", "channel"}
#: The short-set metrics' meaning and population (design/specs/benchmark.md, "Short-set ownership and semantics").
SHORT_SET = {
    "M01": ("Detection recall: matched truth points over eligible truth points, one-to-one greedy matching at the "
            "frozen radius", "eligible truth points; empty truth is undefined"),
    "M02": ("Detection precision: matched detections over eligible detections", "eligible detections; no detections "
            "is undefined"),
    "M03": ("Detection F1 at the frozen matching radius; zero for no detections with nonempty truth",
            "eligible truth points and detections; empty truth is undefined"),
    "M05": ("AUPRC: recall increments times precision in descending score order, tied scores together; no threshold "
            "is chosen", "scored detections against eligible truth"),
    "M07": ("Reference-round background RMSE and bias against the truth background, in the stage's units",
            "reference-round voxels"),
    "M08": ("Median puncta contrast: centre peak minus shell median, over the quiet-region noise",
            "truth puncta with nonzero noise; the zero-noise fraction is reported"),
    "M09": ("Truth intensity spread: std/mean of the nonempty round and channel medians of truth-neighbourhood sums",
            "truth puncta, grouped by round and channel"),
    "M10": ("Correct-decode fraction: accepted reads matched to truth with the correct gene, over all truth "
            "molecules; wrong-gene and false-detection reads are counted among accepted reads",
            "all truth molecules"),
    "M12": ("Clipped fraction: input-positive voxels that became zero, over all voxels across rounds",
            "all voxels"),
    "M13": ("Saturated fraction: unsigned output voxels at the dtype maximum, over all voxels; zero for signed or "
            "float output", "all voxels"),
    "M15": ("Detected-spot intensity spread: the M09 formula on observed sequences at detected coordinates",
            "every detected spot with an observed sequence, rejected reads included"),
    "M16": ("FOV variation: std/mean across per-FOV values; any missing FOV value makes it undefined",
            "the FOVs of one stratum; edge and dense FOVs are evaluated separately"),
    "M17": ("Unmatched-read rate: unmatched reads over all decoded reads", "all decoded reads, rejected reads "
            "included"),
    "M18": ("Assigned-read count: reads whose call status is assigned, whether accepted or not; a count, not "
            "accuracy", "all decoded reads"),
    "M21": ("Displacement-field error: median, linear p95 and maximum Euclidean error", "voxels of the explicit "
            "valid-overlap mask; an empty mask is undefined"),
    "M24": ("Landmark recall before and after registration under the experiment's frozen matching policy",
            "held-out supplied landmarks"),
    "M26": ("Pearson NCC before, after and their gain", "the common valid overlap"),
    "M29": ("Fold fraction: voxels with det(I + grad u) <= 0", "every voxel of the transform grid"),
    "M34": ("Direct-versus-tiled molecule agreement within the inclusive Chebyshev threshold, both directions, and "
            "the count ratio; agreement, not truth accuracy", "direct (reference) and tiled molecules"),
    "M36": ("Duplicate and spurious detections: unmatched detections near a matched truth point are duplicates, "
            "others spurious", "unmatched detections"),
    "M37": ("Gene accuracy and confusion over spatially matched pairs; unmatched truth affects recall, not this "
            "accuracy", "spatially matched truth-read pairs"),
    "M39": ("Score AUROC with half-credit ties and the Hanley-McNeil standard error", "correct and incorrect calls; "
            "a missing class is undefined"),
    "M40": ("Error at 50, 80, 90 and 100% retention by score, with fractional boundary ties", "scored calls"),
    "M41": ("Cut-off curve over saved scores: retention, unmatched rate, per-gene counts and truth error at each "
            "listed cut-off; no cut-off is selected", "saved scored reads"),
    "M54": ("Instance F1 at IoU 0.5 to 0.9", "truth and predicted labels"),
    "M55": ("Split count: truth nuclei covered by several predicted labels", "truth nuclei"),
    "M56": ("Merge count: predicted labels covering two or more truth nuclei", "predicted labels"),
    "M58": ("Label-size median and IQR", "distinct nonzero label IDs"),
    "M60": ("Nuclei per cell from the assignment's saved records", "cells"),
    "M62": ("Gauge-aligned tile-origin error, median and maximum", "placed tiles"),
    "M63": ("Kept-link residual norm as saved; an unkept link is missing", "stitching links"),
    "M68": ("Overlap molecule distances and directional match rates in a common frame",
            "same-gene molecules in tile overlaps"),
    "M69": ("Per-round shift differences: signed, maximum axis and mean L2", "rounds; a missing round keeps its "
            "missing status"),
    "M70": ("MATLAB-versus-Python molecule agreement: same-gene greedy pairs at the frozen radius, both directions; "
            "agreement, not truth accuracy", "MATLAB good spots (reference) and Python molecules"),
    "M71": ("Python/MATLAB molecule count ratio", "MATLAB molecules; zero MATLAB molecules is undefined"),
    "M73": ("Wall and CPU time as the process recorded them", "one process"),
    "M74": ("Peak RSS in KiB, the larger of the process's and its children's high-water marks", "one process"),
    "M75": ("Task failures and retries from saved attempt records", "task attempts"),
    "M76": ("Completion and output identity across engines", "declared outputs"),
    "M77": ("Attributed billing total", "billing rows attributed to this run"),
    "M78": ("Counted human steps", "recorded actions"),
    "M79": ("Resource-prediction error on held-out dataset records; a zero measurement is undefined",
            "held-out task records"),
}
#: The fixed qualification sentences of every report.
LIMITATIONS = (
    "This report renders saved values: it processes, detects and generates nothing and computes no metric.",
    "It selects no threshold, default or optimum, and pools no methods, modes, units or parameter sets.",
    "A verdict appears only where the protocol supplied one, with its failing clause.",
    "Agreement and assigned-read counts are not accuracy; accuracy needs truth.",
    "Resource measurements of different scopes are shown side by side, never combined.",
)
#: Display limits of the full tables: column groups fit FULL_WIDTH_PX, row blocks BLOCK_ROWS rows (one capture).
FULL_WIDTH_PX = 1300
BLOCK_ROWS = 40
CHAR_PX = 8
CELL_PX = 12
WRAP_CHARS = 40
KEY_COLUMNS = ("row", "metric", "value_name", "source", "task", "method", "mode", "condition", "seed", "fov", "side",
               "path", "key")
MAX_PANELS = 6


class ReportError(ValueError):
    """The protocol, the evaluation or the request cannot be rendered as given."""


class SourceMismatch(ReportError):
    """A saved file is missing or differs from the SHA-256 or record that names it."""


class RevisionMismatch(ReportError):
    """The evaluation's code revision cannot be checked against this checkout."""


def _sha(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def _bare(digest):
    return digest.removeprefix("sha256:") if isinstance(digest, str) else digest


# --- Saved files ----------------------------------------------------------------------------------------------------

class Saved:
    """Every file the report reads, checked as it is read; ``verified`` lists each with what it was checked against."""

    def __init__(self):
        self.verified = {}

    def read(self, path, what, sha256=None, size=None):
        path = Path(path)
        if not path.is_file():
            raise SourceMismatch(f"{what} {path} is missing")
        data = path.read_bytes()
        digest = _sha(data)
        if size is not None and len(data) != size:
            raise SourceMismatch(f"{what} {path} has {len(data)} bytes, its record {size}")
        if sha256 is not None and digest != _bare(sha256):
            raise SourceMismatch(f"{what} {path} differs from its recorded SHA-256 ({digest}, recorded "
                                 f"{_bare(sha256)})")
        self.verified[str(path)] = {"path": str(path), "sha256": digest, "bytes": len(data), "what": what,
                                    "checked_against": "recorded SHA-256" if sha256 is not None else "read only"}
        return data


def value_rows(metrics):
    """The rows of values.csv that ``experiment evaluate`` writes for these metric records, derived again here."""
    rows = []
    for record in metrics:
        common = {"metric": record["name"], "metric_id": record.get("metric"), "function": record.get("function"),
                  "outcome": record.get("outcome")}
        for key, value in (record.get("values") or {}).items():
            rows.append({**common, "status": record.get("status"), "value_name": key,
                         "value": value if value is None or isinstance(value, (bool, int, float, str))
                         else json.dumps(value), "unit": (record.get("units") or {}).get(key),
                         "reason": (record.get("reasons") or {}).get(key)})
        if record.get("outcome") != "computed":
            rows.append({**common, "status": None, "value_name": None, "value": None, "unit": None,
                         "reason": record.get("reason") or (record.get("error") or {}).get("message")})
    return rows


def _csv_frame(text):
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)


class SavedEvaluation:
    """A saved evaluation folder, checked: its format, values.csv against evaluation.json and each details file."""

    def __init__(self, folder, saved):
        self.folder = Path(folder).resolve()
        if not self.folder.is_dir():
            raise SourceMismatch(f"the evaluation folder {self.folder} is missing")
        self.saved = saved
        data = saved.read(self.folder / "evaluation.json", "the saved evaluation")
        try:
            record = json.loads(data)
        except ValueError as error:
            raise SourceMismatch(f"{self.folder / 'evaluation.json'} is not JSON: {error}") from None
        if not isinstance(record, dict) or record.get("format") != EVALUATION_FORMAT:
            raise SourceMismatch(f"{self.folder / 'evaluation.json'} is not a {EVALUATION_FORMAT} record")
        for key in ("sources", "metrics", "software", "inputs", "status", "counts"):
            if key not in record:
                raise SourceMismatch(f"evaluation.json has no {key!r}")
        names = [m.get("name") for m in record["metrics"]]
        if len(set(names)) != len(names):
            raise SourceMismatch("evaluation.json names a metric twice")
        self.record, self.sha256 = record, _sha(data)
        values = saved.read(self.folder / "values.csv", "the evaluation's value table").decode()
        expected = pd.DataFrame(value_rows(record["metrics"]), columns=VALUE_COLUMNS).to_csv(index=False)
        try:
            found = _csv_frame(values)
        except (ValueError, pd.errors.ParserError) as error:
            raise SourceMismatch(f"values.csv cannot be read: {error}") from None
        if list(found.columns) != VALUE_COLUMNS or not found.equals(_csv_frame(expected)):
            raise SourceMismatch("values.csv differs from evaluation.json's metrics")
        self.values_sha256 = self.saved.verified[str(self.folder / "values.csv")]["sha256"]
        self.details = {}
        for metric in record["metrics"]:
            name = metric.get("details_file")
            if name is None:
                continue
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise SourceMismatch(f"the details file {name!r} is outside the evaluation folder")
            try:
                self.details[metric["name"]] = json.loads(saved.read(self.folder / name, "a metric's details file"))
            except ValueError as error:
                raise SourceMismatch(f"{name} is not JSON: {error}") from None
        self.sources = {s["name"]: s for s in record["sources"]}

    def roots(self):
        """The folders of the evaluation's sources, which the report must not write into."""
        roots = [self.folder]
        for source in self.sources.values():
            if source.get("kind") in ("trial", "checkpoint") and source.get("path"):
                roots.append(Path(source["path"]))
            roots += [Path(r) for r in (source.get("roots") or {}).values() if r]
        return roots


# --- Revision -------------------------------------------------------------------------------------------------------

def _git(repo, *args):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if result.returncode != 0:
        raise RevisionMismatch(f"git {' '.join(args)} failed in {repo}: {result.stderr.decode().strip()}")
    return result.stdout.decode().strip()


def checkout():
    """The starfinder checkout that holds this package, or RevisionMismatch."""
    package = Path(__file__).resolve().parents[1]
    try:
        top = Path(_git(package, "rev-parse", "--show-toplevel"))
    except (RevisionMismatch, OSError):
        raise RevisionMismatch(f"the renderer's package {package} is not in a Git checkout, so the evaluation's "
                               "revision cannot be checked") from None
    if (top / "src" / "python" / "starfinder").resolve() != package:
        raise RevisionMismatch(f"{top} is not the starfinder checkout of the renderer's package")
    return top


def check_revision(code, repo=None, sources=EVALUATION_SOURCES):
    """Check the evaluation's recorded code identity against this checkout; return what was checked.

    The recorded commit must be a commit of this checkout and an ancestor of HEAD. For an evaluation of a clean tree,
    ``sources`` must be unchanged from that commit to the working tree. The saved evaluation records whether the tree
    was dirty but no checksum of the changes, so a dirty evaluation cannot be resolved to a commit: it is labelled
    unresolved, and the report shows that label.
    """
    repo = checkout() if repo is None else Path(repo)
    revision = (code or {}).get("git_commit")
    if not revision:
        raise RevisionMismatch("the evaluation records no source revision (software.code.git_commit)")
    if subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{revision}^{{commit}}"],
                      capture_output=True).returncode != 0:
        raise RevisionMismatch(f"the evaluation's revision {revision} is not a commit of {repo}")
    head = _git(repo, "rev-parse", "HEAD")
    if subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", revision, head],
                      capture_output=True).returncode != 0:
        raise RevisionMismatch(f"the evaluation's revision {revision} is not an ancestor of HEAD {head}")
    identity = {"revision": revision, "head": head, "dirty": bool(code.get("git_dirty")), "version":
                code.get("version"), "sources": list(sources)}
    if code.get("git_dirty"):
        return dict(identity, resolved=False, changed=None,
                    match=f"unresolved: the evaluation ran on a modified tree at {revision[:12]}, and the saved "
                          "evaluation records no checksum of the changes, so no commit can be shown to hold them")
    changed = [p for p in _git(repo, "diff", "--name-only", revision, "--", *sources).splitlines() if p]
    changed += [p for p in _git(repo, "ls-files", "--others", "--exclude-standard", "--", *sources).splitlines()
                if p]
    if changed:
        raise RevisionMismatch(f"the evaluation sources changed since {revision}: {', '.join(sorted(changed))}")
    return dict(identity, resolved=True, changed=[],
                match=f"the evaluation's tree was clean at {revision[:12]}, an ancestor of HEAD, and the evaluation "
                      "sources are unchanged since")


# --- Protocol -------------------------------------------------------------------------------------------------------

def _strings(value, where):
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        raise ReportError(f"protocol {where} is a list of strings")
    return value


def _evidence(value, where):
    if not isinstance(value, list):
        raise ReportError(f"protocol {where} is a list of {{text, source_table}}")
    for item in value:
        if not isinstance(item, dict) or set(item) != {"text", "source_table"} or \
                not all(isinstance(item[k], str) and item[k] for k in item):
            raise ReportError(f"protocol {where}: each entry has exactly a text and the source_table it comes from")
    return value


def load_protocol(path, saved):
    """The protocol's mapping (empty without one), checked key by key; ReportError for an unknown or malformed key."""
    if path is None:
        return {}
    data = saved.read(path, "the report protocol")
    try:
        protocol = yaml.safe_load(data)
    except yaml.YAMLError as error:
        raise ReportError(f"{path}: not YAML or JSON: {error}") from None
    if not isinstance(protocol, dict):
        raise ReportError(f"{path} must be a mapping")
    unknown = set(protocol) - set(PROTOCOL_KEYS)
    if unknown:
        raise ReportError(f"protocol: unsupported keys {sorted(unknown)}")
    for key in ("title", "qualification", "route", "inputs", "truth", "dtype", "resources"):
        if key in protocol and not isinstance(protocol[key], str):
            raise ReportError(f"protocol {key} is a string")
    if not isinstance(protocol.get("conditions", {}), dict) or \
            any(not isinstance(v, str) for v in protocol.get("conditions", {}).values()):
        raise ReportError("protocol conditions maps a condition name to one sentence")
    for cutoff in protocol.get("cutoffs", []):
        if not isinstance(cutoff, dict) or not {"name", "value", "unit"} <= set(cutoff) <= \
                {"name", "value", "unit", "note"}:
            raise ReportError("protocol cutoffs: each is {name, value, unit} with an optional note")
    _strings(protocol.get("missing_prerequisites", []), "missing_prerequisites")
    for key, entry in protocol.get("metrics", {}).items():
        if not isinstance(entry, dict) or set(entry) - {"meaning", "population"}:
            raise ReportError(f"protocol metrics {key!r} is {{meaning, population}}")
    for name, entry in protocol.get("rows", {}).items():
        if not isinstance(entry, dict) or set(entry) - set(IDENTITY):
            raise ReportError(f"protocol rows {name!r} maps to keys of {list(IDENTITY)}")
        if entry.get("side") not in (None, *SIDES):
            raise ReportError(f"protocol rows {name!r}: side is before or after")
    seen = set()
    for card in protocol.get("methods", []):
        if not isinstance(card, dict) or set(card) - _METHOD_KEYS or not isinstance(card.get("method"), str):
            raise ReportError(f"protocol methods: each names a method and keys of {sorted(_METHOD_KEYS)}")
        key = (card["method"], card.get("mode"))
        if key in seen:
            raise ReportError(f"protocol methods: {key} appears twice")
        seen.add(key)
        for field in ("targeted_conditions", "key_metrics", "limitations"):
            _strings(card.get(field, []), f"methods {key} {field}")
        for field in ("preconditions", "harm"):
            _evidence(card.get(field, []), f"methods {key} {field}")
        verdict = card.get("verdict")
        if verdict is not None and (not isinstance(verdict, dict) or
                                    set(verdict) != {"value", "failing_clause", "source_table"}):
            raise ReportError(f"protocol methods {key}: a verdict is {{value, failing_clause, source_table}}; "
                              "failing_clause is null only when no clause failed")
    for finding in protocol.get("findings", []):
        if not isinstance(finding, dict) or set(finding) != {"label", "text", "source_table"} or \
                finding["label"] not in LABELS:
            raise ReportError("protocol findings: each is {label: finding or hypothesis, text, source_table}")
    for figure in protocol.get("figures", []):
        if not isinstance(figure, dict) or set(figure) - _FIGURE_KEYS or not isinstance(figure.get("method"), str) \
                or not isinstance(figure.get("panels"), list) or not figure["panels"]:
            raise ReportError(f"protocol figures: each names a method and nonempty panels, keys of "
                              f"{sorted(_FIGURE_KEYS)}")
        for item in [*figure["panels"], *(figure[k] for k in ("detections", "truth") if k in figure)]:
            if not isinstance(item, dict) or set(item) - _FILE_KEYS or not isinstance(item.get("source"), str) \
                    or sum(k in item for k in ("output", "input", "file")) != 1:
                raise ReportError("protocol figures: each file names a source and one of output, input or file")
    return protocol


# --- Records the sources name ---------------------------------------------------------------------------------------

def _method_label(method):
    if not isinstance(method, dict):
        return None
    if method.get("module") == "preprocessing":
        return "+".join(str(step.get("method")) for step in method.get("steps") or [])
    return method.get("method")


class Records:
    """The trial records and workflow task records the evaluation's sources name, each checked against its hash."""

    def __init__(self, evaluation, saved):
        self.evaluation, self.saved = evaluation, saved
        self.trials, self.tasks = {}, {}
        for name, source in evaluation.sources.items():
            if source.get("kind") == "trial" and source.get("record"):
                entry = source["record"]
                record = json.loads(saved.read(entry["path"], f"trial record of source {name!r}", entry["sha256"]))
                if record.get("format") != TRIAL_FORMAT:
                    raise SourceMismatch(f"{entry['path']} is not a {TRIAL_FORMAT} record")
                self.trials[name] = record
            if source.get("kind") == "workflow":
                for task_id, task in (source.get("tasks") or {}).items():
                    entry = task.get("record")
                    if entry:
                        self.tasks[(name, task_id)] = json.loads(saved.read(
                            entry["path"], f"task record of {task_id} in source {name!r}", entry["sha256"]))

    def trial_identity(self, name):
        record = self.trials.get(name)
        if record is None:
            return {}
        configuration = record.get("configuration") or {}
        condition = configuration.get("condition")
        return {"method": _method_label(configuration.get("method")), "seed": record.get("seed"),
                "condition": condition.get("name") if isinstance(condition, dict) else None}

    def trial_setup(self, name):
        """The trial's resolved method and condition as canonical JSON: rows of different setups are never
        pooled."""
        configuration = self.trials[name].get("configuration") or {}
        return json.dumps({"method": configuration.get("method"), "condition": configuration.get("condition") or {}},
                          sort_keys=True)

    def file(self, spec):
        """(path, bytes) of a figure's file: a trial's listed output or recorded input, or a files source's file."""
        source = self.evaluation.sources.get(spec["source"])
        if source is None:
            raise ReportError(f"a figure names the source {spec['source']!r}, which the evaluation does not have")
        if source.get("kind") == "files":
            entry = (source.get("files") or {}).get(spec.get("file"))
            if entry is None:
                raise ReportError(f"files source {spec['source']!r} has no file {spec.get('file')!r}")
            path = Path(entry["path"])
            return path, self.saved.read(path, f"file {spec['file']!r} of source {spec['source']!r}",
                                         entry["sha256"])
        if source.get("kind") != "trial" or spec["source"] not in self.trials:
            raise ReportError(f"a figure reads a trial or files source; {spec['source']!r} is a {source.get('kind')} "
                              "source or has no checked record")
        record = self.trials[spec["source"]]
        folder = Path(source["path"])
        if "output" in spec:
            entry = next((o for o in record.get("outputs") or [] if o.get("path") == spec["output"]), None)
            if entry is None:
                raise SourceMismatch(f"trial {folder} lists no output {spec['output']!r}")
            path = folder / entry["path"]
            return path, self.saved.read(path, f"output {spec['output']!r} of trial {spec['source']!r}",
                                         entry["sha256"], entry.get("bytes"))
        if "input" in spec:
            entry = (record.get("inputs") or {}).get(spec["input"])
            if not isinstance(entry, dict) or "path" not in entry:
                raise SourceMismatch(f"trial {folder} has no file input {spec['input']!r}")
            path = Path(entry["path"])
            return path, self.saved.read(path, f"input {spec['input']!r} of trial {spec['source']!r}",
                                         entry["sha256"])
        raise ReportError("a trial figure file names an output or an input")


def _arguments(described):
    """The (source, task) pairs a metric's recorded arguments read."""
    found = []
    if isinstance(described, dict):
        if "mapping" in described:
            for item in described["mapping"].values():
                found += _arguments(item)
        elif "source" in described:
            found.append((described["source"], described.get("task")))
    return found


def identities(evaluation, records, protocol):
    """{metric name: {method, mode, condition, seed, fov, side, provenance}} from the saved records and protocol.

    A trial source gives the method, seed and condition name; a workflow task record gives the FOV. A field that the
    sources read by one metric give differently stays empty and is noted. The protocol's rows assign the card's
    method and mode and the side, and fill an empty condition, seed or FOV; one that contradicts a saved condition,
    seed or FOV is refused. The trial's own method stays in ``trial_method``.
    """
    rows = protocol.get("rows", {})
    unknown = set(rows) - {m["name"] for m in evaluation.record["metrics"]}
    if unknown:
        raise ReportError(f"protocol rows name metrics the evaluation does not have: {sorted(unknown)}")
    out = {}
    for metric in evaluation.record["metrics"]:
        found = {key: set() for key in IDENTITY}
        setups = set()
        # A metric skipped at its first argument records no arguments, but names every source it used.
        read = [pair for argument in metric.get("arguments", {}).values() for pair in _arguments(argument)]
        read += [(source, None) for source in metric.get("sources") or []]
        for source, task in read:
            for key, value in records.trial_identity(source).items():
                if value is not None:
                    found[key].add(json.dumps(value))
            if source in records.trials:
                setups.add(records.trial_setup(source))
            if task is not None and (source, task) in records.tasks:
                fov = (records.tasks[(source, task)].get("identity") or {}).get("fov")
                found["fov"].add(json.dumps(fov if fov is not None else task))
        identity, provenance, notes = {}, {}, []
        for key in IDENTITY:
            values = sorted(found[key])
            if len(values) == 1:
                identity[key], provenance[key] = json.loads(values[0]), "saved record"
            else:
                identity[key] = None
                if len(values) > 1:
                    notes.append(f"its sources give different {key} values ({', '.join(values)})")
            if key == "method":
                trial_method = identity[key]
            supplied = rows.get(metric["name"], {}).get(key)
            if supplied is not None:
                if key not in ASSIGNED and identity[key] is not None and identity[key] != supplied:
                    raise ReportError(f"protocol rows {metric['name']!r} gives {key} {supplied!r}, the saved "
                                      f"record {identity[key]!r}")
                identity[key], provenance[key] = supplied, "protocol"
        out[metric["name"]] = dict(identity, trial_method=trial_method, setup=setups.pop() if len(setups) == 1
                                   else None, provenance=provenance, notes=notes)
    return out


def resource_rows(evaluation, records):
    """One row per source, and per workflow task, with its resource scope; missing values keep a reason."""
    rows = []
    fields = ("wall_s", "cpu_s", "peak_rss_kib")
    for name, source in evaluation.sources.items():
        kind = source.get("kind")
        base = {"source": name, "kind": kind, "task": None, "source_status": source.get("status")}
        if kind == "trial":
            record = records.trials.get(name)
            identity = records.trial_identity(name)
            row = dict(base, method=identity.get("method"), condition=identity.get("condition"),
                       seed=identity.get("seed"), status=None if record is None else record.get("status"),
                       scope="trial process: wall time after the output folder is reserved; CPU and peak RSS of the "
                             "process and its reaped children")
            for field in fields:
                row[field] = None if record is None else record.get(field)
            row["reason"] = ("no checked trial record: " + str((source.get("error") or {}).get("message"))
                             if record is None else None if all(row[f] is not None for f in fields)
                             else "the trial record holds no value")
            rows.append(row)
        elif kind == "workflow":
            for task_id, task in (source.get("tasks") or {}).items():
                record = records.tasks.get((name, task_id))
                section = (record or {}).get("workflow") or {}
                row = dict(base, task=task_id, method=None, condition=None, seed=None,
                           status=None if record is None else record.get("status"),
                           scope="workflow task process: the task runner's wall time, CPU and peak RSS of the task "
                                 "process and its reaped children")
                for field in fields:
                    row[field] = section.get(field)
                row["reason"] = ("no task record" if record is None else None
                                 if all(row[f] is not None for f in fields) else "the task record holds no value")
                rows.append(row)
        else:
            rows.append(dict(base, method=None, condition=None, seed=None, status=None, scope=None,
                             **dict.fromkeys(fields), reason=f"a {kind} source holds no resource record"))
    columns = ["source", "kind", "task", "method", "condition", "seed", "source_status", "status", "scope", *fields,
               "reason"]
    return pd.DataFrame(rows, columns=columns, dtype=object)


# --- Tables of values -----------------------------------------------------------------------------------------------

def _number(value):
    """A saved value as a float for ranges: a number itself, the mean of a list's or mapping's numeric items (with
    their count), else None; booleans are not numbers here."""
    if isinstance(value, bool) or value is None:
        return None, None
    if isinstance(value, (int, float)):
        return (float(value), None) if math.isfinite(value) else (None, None)
    items = list(value.values()) if isinstance(value, dict) else value if isinstance(value, list) else None
    if not items or any(isinstance(v, bool) or not isinstance(v, (int, float)) or v is None for v in items):
        return None, None
    if not all(math.isfinite(v) for v in items):
        return None, None
    return float(np.mean(items)), len(items)


def complete_rows(evaluation, ids):
    """Every row of values.csv with its identity, parameter set, counts and saved value, in values.csv order."""
    rows = []
    sets = {}
    for metric in evaluation.record["metrics"]:
        key = metric.get("metric") or metric.get("function")
        parameters = json.dumps(metric.get("parameters", {}), sort_keys=True)
        sets.setdefault(key, [])
        if parameters not in sets[key]:
            sets[key].append(parameters)
    by_name = {m["name"]: m for m in evaluation.record["metrics"]}
    setups = list(dict.fromkeys(ids[m["name"]]["setup"] for m in evaluation.record["metrics"]
                                if ids[m["name"]]["setup"] is not None))
    for i, row in enumerate(value_rows(evaluation.record["metrics"]), start=1):
        metric = by_name[row["metric"]]
        key = metric.get("metric") or metric.get("function")
        parameters = json.dumps(metric.get("parameters", {}), sort_keys=True)
        raw = (metric.get("values") or {}).get(row["value_name"]) if row["value_name"] is not None else None
        number, items = _number(raw)
        identity = ids[row["metric"]]
        rows.append({"row": i, **row, "selector": key,
                     "parameter_set": f"p{sets[key].index(parameters) + 1}" if len(sets[key]) > 1 else "",
                     **{k: identity[k] for k in IDENTITY}, "trial_method": identity["trial_method"],
                     "setup": None if identity["setup"] is None else f"setup {setups.index(identity['setup']) + 1}",
                     "setup_json": identity["setup"],
                     "number": number, "items": items, "raw": raw,
                     "counts": json.dumps(metric.get("counts") or {}, sort_keys=True),
                     "parameters": parameters, "sources": ", ".join(metric.get("sources") or []),
                     "identity_from": ", ".join(f"{k}: {v}" for k, v in identity["provenance"].items()),
                     "identity_notes": "; ".join(identity["notes"])})
    # Object columns keep integer seeds and None as saved (a float column would turn seed 12345 into 1.234e+04).
    return pd.DataFrame(rows, dtype=object)


def _fmt(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "undefined"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


def _cell(value):
    """Display text of a table cell; empty for a missing identity field."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return _fmt(value)


def _span(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    low, high = min(values), max(values)
    return _fmt(low) if low == high else f"{_fmt(low)} to {_fmt(high)}"


def _labels(values):
    values = sorted({str(_cell(v)) for v in values if _cell(v) != ""}, key=lambda s: (len(s), s))
    return ", ".join(values)


def card_key(method, mode):
    return (method, mode)


def card_anchor(key):
    method, mode = key
    text = "unassigned" if method is None else method if mode is None else f"{method}-{mode}"
    return "card-" + re.sub(r"[^A-Za-z0-9_-]+", "-", str(text))


def card_order(frame, protocol):
    """The cards: the protocol's methods in page order, then every other (method, mode) in values.csv order."""
    keys = [card_key(c["method"], c.get("mode")) for c in protocol.get("methods", [])]
    for method, mode in zip(frame["method"], frame["mode"]):
        key = card_key(None if pd.isna(method) else method, None if pd.isna(mode) else mode)
        if key not in keys:
            keys.append(key)
    return keys


def _in_card(frame, key):
    method, mode = key
    rows = frame[frame["method"].isna()] if method is None else frame[frame["method"] == method]
    return rows[rows["mode"].isna()] if mode is None else rows[rows["mode"] == mode]


def _selected(rows, selectors):
    """The rows of the selected key metrics: ``ID`` or ``function`` for every value, ``ID:value`` for one."""
    if not selectors:
        return rows
    keep = np.zeros(len(rows), dtype=bool)
    for selector in selectors:
        key, _, value = selector.partition(":")
        match = (rows["selector"] == key).to_numpy(dtype=bool, copy=True)
        if value:
            match = match & ((rows["value_name"] == value) | rows["value_name"].isna()).to_numpy(dtype=bool)
        keep |= match
    return rows[keep]


def key_table(rows):
    """The card's key metrics: one row per metric value, unit, parameter set and condition, the before and after
    ranges across seeds and FOVs side by side, with the denominators, nulls, failures, missing seeds or FOVs and
    the values.csv rows they come from. Nothing is pooled across units or parameter sets."""
    out = []
    computed = rows[rows["value_name"].notna()]
    failed = rows[rows["value_name"].isna()]
    groups = computed.groupby(["selector", "value_name", "parameter_set", "condition"], dropna=False, sort=False)
    seen = set()
    for (selector, value_name, parameter_set, condition), group in groups:
        units = [u for u in group["unit"].unique()]
        for unit in units:
            part = group[group["unit"].isna()] if pd.isna(unit) else group[group["unit"] == unit]
            same = (failed["selector"] == selector) & (failed["parameter_set"] == parameter_set) & \
                (failed["condition"].isna() if pd.isna(condition) else failed["condition"] == condition)
            lost = failed[same]
            seen.update(lost.index)
            out.append(_key_row(rows, selector, value_name, unit, parameter_set, condition, part, lost))
    rest = failed.drop(index=[i for i in seen if i in failed.index])
    for (selector, parameter_set, condition), lost in rest.groupby(["selector", "parameter_set", "condition"],
                                                                   dropna=False, sort=False):
        out.append(_key_row(rows, selector, None, None, parameter_set, condition, lost.iloc[0:0], lost))
    return out


def _key_row(card_rows, selector, value_name, unit, parameter_set, condition, part, lost):
    universe = card_rows if pd.isna(condition) else card_rows[card_rows["condition"] == condition]
    sides = {}
    for side in (*SIDES, None):
        here = part[part["side"].isna()] if side is None else part[part["side"] == side]
        gone = lost[lost["side"].isna()] if side is None else lost[lost["side"] == side]
        if here.empty and gone.empty:
            continue
        # Rows of different trial setups (method, configuration, condition) on one side are shown apart, never
        # pooled into one range.
        setups = list(dict.fromkeys([*here["setup"], *gone["setup"]]))
        cells = []
        for setup in setups:
            mine = here[here["setup"].isna()] if setup is None else here[here["setup"] == setup]
            failed = gone[gone["setup"].isna()] if setup is None else gone[gone["setup"] == setup]
            label = None
            if len(setups) > 1:
                methods = _labels([*mine["trial_method"], *failed["trial_method"]])
                label = f"{setup or 'no trial setup'}" + (f" (trial method {methods})" if methods else "")
            cells.append(_side_cell(universe, mine, failed, label))
        sides["single" if side is None else side] = cells
    return {"selector": selector, "value_name": value_name, "unit": None if pd.isna(unit) else unit,
            "parameter_set": parameter_set, "condition": None if pd.isna(condition) else condition, "sides": sides,
            "rows": sorted(int(r) for r in [*part["row"], *lost["row"]])}


def _side_cell(universe, here, gone, setup):
    numbers = [n for n in here["number"] if n is not None and not pd.isna(n)]
    booleans = [v for v in here["raw"] if isinstance(v, bool)]
    nulls = here[here["raw"].isna() & here["value"].isna()]
    present_seeds = set(here["seed"].dropna()) | set(gone["seed"].dropna())
    present_fovs = set(here["fov"].dropna()) | set(gone["fov"].dropna())
    # A seed or FOV with no row at all is missing; one whose metric was not computed is shown as that failure.
    missing_seeds = set(universe["seed"].dropna()) - present_seeds
    missing_fovs = set(universe["fov"].dropna()) - present_fovs
    text = _span(numbers)
    if booleans:
        text = ", ".join(f"{v} {sum(b is v for b in booleans)}" for v in (True, False) if any(b is v for b in booleans))
    return {"setup": setup, "range": text, "numeric": bool(numbers), "n": len(here),
            "defined": len(numbers) + len(booleans), "seeds": _labels(present_seeds), "fovs": _labels(present_fovs),
            "missing_seeds": _labels(missing_seeds), "missing_fovs": _labels(missing_fovs),
            "items": sorted({int(i) for i in here["items"].dropna()}),
            "nulls": [(r.seed, r.fov, r.reason) for r in nulls.itertuples()],
            "failures": [(r.seed, r.fov, r.outcome, r.reason) for r in gone.itertuples()],
            "counts": _count_ranges(here["counts"])}


def _count_ranges(counts):
    found = {}
    for text in counts:
        for key, value in json.loads(text).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                found.setdefault(key, []).append(value)
    return {key: _span(values) for key, values in found.items()}


def pair_rows(frame):
    """The complete before/after table: every row with a side, joined on everything but the side; a row without
    its partner is kept and marked unpaired."""
    sided = frame[frame["side"].isin(SIDES)].copy()
    columns = ["selector", "value_name", "unit", "parameter_set", "method", "mode", "condition", "seed", "fov"]
    out = []
    keyed = {}
    for row in sided.itertuples(index=False):
        key = tuple(None if (isinstance(getattr(row, c), float) and math.isnan(getattr(row, c)))
                    else getattr(row, c) for c in columns)
        keyed.setdefault(key, {}).setdefault(row.side, []).append(row)
    for key, sides in keyed.items():
        before, after = sides.get("before", []), sides.get("after", [])
        for i in range(max(len(before), len(after))):
            b = before[i] if i < len(before) else None
            a = after[i] if i < len(after) else None
            pairing = "paired" if b is not None and a is not None else \
                "unpaired: no after row" if a is None else "unpaired: no before row"
            out.append({**dict(zip(columns, key)),
                        "before": None if b is None else b.value, "before_outcome": None if b is None else b.outcome,
                        "before_reason": None if b is None else b.reason,
                        "after": None if a is None else a.value, "after_outcome": None if a is None else a.outcome,
                        "after_reason": None if a is None else a.reason, "pairing": pairing,
                        "rows": ", ".join(str(r.row) for r in (b, a) if r is not None)})
    return pd.DataFrame(out, columns=[*columns, "before", "before_outcome", "before_reason", "after",
                                      "after_outcome", "after_reason", "pairing", "rows"])


def collapsed(points):
    """True when every point of a precision-recall curve is the same: the curve collapsed to one point."""
    finite = points.dropna(subset=["precision", "recall"])
    return len(finite) > 1 and len({(round(p, 12), round(r, 12)) for p, r in zip(finite.precision,
                                                                                    finite.recall)}) == 1


def curves(evaluation):
    """{metric name: curve table} of every computed metric whose details hold a curve of precision and recall."""
    out = {}
    for name, details in evaluation.details.items():
        curve = details.get("curve") if isinstance(details, dict) else None
        if isinstance(curve, list) and curve and all(isinstance(r, dict) for r in curve):
            frame = pd.DataFrame(curve)
            if {"precision", "recall"} <= set(frame.columns):
                out[name] = frame
    return out


# --- Figures --------------------------------------------------------------------------------------------------------

class _pyplot:
    def __enter__(self):
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        self.plt = plt
        return plt

    def __exit__(self, *args):
        self.plt.close("all")


def _png(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=80, metadata={"Software": None})
    return buffer.getvalue()


def metric_figure(table):
    """Dot plot of the card's numeric key metrics: one panel per metric value, one column per condition and side,
    one dot per saved row; undefined values are counted in the panel title, never drawn as zero."""
    panels = [r for r in table if any(c["numeric"] for cells in r["sides"].values() for c in cells)]
    shown, omitted = panels[:MAX_PANELS], panels[MAX_PANELS:]
    if not shown:
        return None, {"panels": [], "omitted": [(r["selector"], r["value_name"]) for r in omitted]}
    return shown, {"panels": [(r["selector"], r["value_name"], r["condition"]) for r in shown],
                   "omitted": [(r["selector"], r["value_name"], r["condition"]) for r in omitted]}


def draw_metric_figure(rows, panels):
    with _pyplot() as plt:
        figure, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 2.8), squeeze=False,
                                    layout="constrained")
        for axis, panel in zip(axes[0], panels):
            labels, nulls = [], 0
            for side in [s for s in ("before", "after", "single") if s in panel["sides"]]:
                here = rows[(rows["selector"] == panel["selector"]) & (rows["value_name"] == panel["value_name"])
                            & ((rows["side"].isna()) if side == "single" else (rows["side"] == side))
                            & (rows["condition"].isna() if panel["condition"] is None
                               else rows["condition"] == panel["condition"])]
                here = here[here["unit"].isna()] if panel["unit"] is None else here[here["unit"] == panel["unit"]]
                here = here[here["parameter_set"] == panel["parameter_set"]]
                methods = list(dict.fromkeys(here["setup"]))
                for method in methods:
                    mine = here[here["setup"].isna()] if method is None else here[here["setup"] == method]
                    values = [n for n in mine["number"] if n is not None and not pd.isna(n)]
                    nulls += len(mine) - len(values)
                    axis.plot([len(labels)] * len(values), values, "o", color="C1" if side == "after" else "C0",
                              alpha=0.8)
                    labels.append(side if len(methods) == 1 else f"{side}\n{method}")
            axis.set_xticks(range(len(labels)), labels)
            axis.set_xlim(-0.5, len(labels) - 0.5)
            title = f"{panel['selector']} {panel['value_name']}"
            if panel["condition"] is not None:
                title += f"\n{panel['condition']}"
            if nulls:
                title += f" ({nulls} undefined)"
            axis.set_title(title, fontsize=8)
            axis.set_ylabel(panel["unit"] or "", fontsize=8)
            axis.tick_params(labelsize=7)
        return _png(figure)


def _plane(array, channel):
    array = np.asarray(array)
    if array.ndim == 4:
        array = array[..., channel or 0]
    if array.ndim == 3:
        return array.max(axis=0)
    if array.ndim == 2:
        return array
    raise ReportError(f"a figure panel is ZYX, ZYXC or YX, not shape {array.shape}")


def _volume(array, channel):
    array = np.asarray(array)
    return array[..., channel or 0] if array.ndim == 4 else array


def realized_peaks(points, volume):
    """Each point's realized peak: the image maximum over the 3 × 3 × 3 voxels around its rounded centre, clipped at
    the image edge, in point order."""
    volume = np.asarray(volume)
    peaks = []
    for z, y, x in np.asarray(points, dtype=float):
        center = [int(round(v)) for v in (z, y, x)]
        box = tuple(slice(max(c - 1, 0), max(min(c + 2, n), 0)) for c, n in zip(center, volume.shape))
        part = volume[box]
        peaks.append(float(part.max()) if part.size else math.nan)
    return np.asarray(peaks, dtype=np.float64)


def dim_punctum(points, volume):
    """The point with the lowest realized peak (ties to the first in order), or None when there is none."""
    peaks = realized_peaks(points, volume)
    finite = np.where(np.isfinite(peaks))[0]
    if not len(finite):
        return None
    index = int(finite[np.argmin(peaks[finite])])
    center = tuple(int(round(v)) for v in np.asarray(points, dtype=float)[index])
    return {"index": index, "center": center, "peak": float(peaks[index])}


def representative_puncta(peaks, count=4):
    """Point indices shown in the figure: the dim punctum, then those at the 25th, 50th and 90th percentiles of the
    realized peak."""
    peaks = np.asarray(peaks, dtype=float)
    valid = np.where(np.isfinite(peaks))[0]
    if not len(valid):
        return []
    order = valid[np.argsort(peaks[valid], kind="stable")]
    chosen = [int(order[0])]
    for q in (0.25, 0.5, 0.9):
        index = int(order[int(round(q * (len(order) - 1)))])
        if index not in chosen:
            chosen.append(index)
    return chosen[:count]


def _array(path, data):
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        return np.load(io.BytesIO(data), allow_pickle=False)
    if suffix in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(io.BytesIO(data))
    raise ReportError(f"a figure panel reads .npy or .tif images, not {Path(path).name}")


def _points(path, data):
    table = pd.read_csv(io.BytesIO(data))
    if not {"z", "y", "x"} <= set(table.columns):
        raise ReportError(f"{Path(path).name} has no z, y and x columns")
    return table[["z", "y", "x"]].to_numpy(dtype=float)


def method_figure(panels, detections=None, truth=None, peaks=0):
    """One saved-image figure: a maximum projection per panel on one shared linear display range, the saved
    detections, and with truth the representative puncta, the dim punctum first. Returns (PNG bytes, facts)."""
    planes = [_plane(array, channel) for _label, array, channel in panels]
    low = float(min(p.min() for p in planes))
    high = float(max(p.max() for p in planes))
    facts = {"display_range": [low, high], "panels": [label for label, _a, _c in panels], "detections": None,
             "dim_punctum": None, "puncta": []}
    chosen = []
    if truth is not None:
        label, array, channel = panels[peaks]
        volume = _volume(array, channel)
        values = realized_peaks(truth, volume)
        facts["dim_punctum"] = dim_punctum(truth, volume)
        facts["peaks_from"] = label
        chosen = representative_puncta(values)
        facts["puncta"] = [{"index": i, "center": [int(round(v)) for v in truth[i]], "peak": float(values[i])}
                           for i in chosen]
    if detections is not None:
        facts["detections"] = int(len(detections))
    with _pyplot() as plt:
        figure, axes = plt.subplots(1, len(planes), figsize=(2.8 * len(planes), 3.0), squeeze=False,
                                    layout="constrained")
        for axis, plane, (label, _a, _c) in zip(axes[0], planes, panels):
            axis.imshow(plane, cmap="gray", vmin=low, vmax=high if high > low else low + 1, interpolation="nearest")
            if detections is not None and len(detections):
                axis.plot(detections[:, 2], detections[:, 1], "o", mfc="none", mec="#e8590c", ms=6, mew=1)
            for n, index in enumerate(chosen, start=1):
                z, y, x = truth[index]
                axis.plot([x], [y], "+", color="#2f9e44", ms=9, mew=1.5)
                axis.annotate(str(n), (x, y), xytext=(3, 3), textcoords="offset points", color="#2f9e44",
                              fontsize=8)
            axis.set_title(label, fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
        return _png(figure), facts


# --- HTML -----------------------------------------------------------------------------------------------------------

def _e(value):
    return html.escape(str(value), quote=True)


def _table(header, rows, cls=""):
    head = "".join(f"<th>{_e(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table class=\"{cls}\"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def full_table(frame, source, anchor, caption="", row_ids=None):
    """Every row and column of a table, naming its source; wide tables split into column groups that fit
    FULL_WIDTH_PX (the key columns repeat in each), long ones into blocks of BLOCK_ROWS rows, each piece one capture
    with the sub-anchor ``<anchor>-r<i>c<j>`` when there is more than one."""
    frame = frame.reset_index(drop=True)
    columns = list(frame.columns)
    texts = {c: [_cell(v) for v in frame[c].astype(object).where(frame[c].notna(), None)] for c in columns}
    widths = {c: int(min(max(max((len(t) for t in texts[c]), default=0), min(len(str(c)), 12)), WRAP_CHARS)
                     * CHAR_PX + CELL_PX) for c in columns}
    keys = [c for c in columns if c in KEY_COLUMNS]
    chunks, current, used = [], [], sum(widths[c] for c in keys)
    for c in columns:
        if c in keys:
            continue
        if current and used + widths[c] > FULL_WIDTH_PX:
            chunks.append(current)
            current, used = [], sum(widths[k] for k in keys)
        current.append(c)
        used += widths[c]
    chunks.append(current)
    blocks = [range(i, min(i + BLOCK_ROWS, len(frame))) for i in range(0, max(len(frame), 1), BLOCK_ROWS)]
    pieces = []
    many = len(chunks) * len(blocks) > 1
    for i, block in enumerate(blocks):
        for j, chunk in enumerate(chunks):
            shown = keys + chunk
            width = sum(widths[c] for c in shown)
            head = "".join(f"<th>{_e(c)}</th>" for c in shown)
            body = []
            for r in block:
                rid = f" id=\"{row_ids[r]}\"" if row_ids is not None and j == 0 else ""
                cells = "".join(f"<td class=\"{'undefined' if texts[c][r] == 'undefined' else ''}\">"
                                f"{_e(texts[c][r])}</td>" for c in shown)
                body.append(f"<tr{rid}>{cells}</tr>")
            piece_id = f"{anchor}-r{i}c{j}" if many else anchor
            label = f"rows {block.start + 1} to {block.stop} of {len(frame)}, columns {j + 1} of {len(chunks)}" \
                if many else f"{len(frame)} rows"
            pieces.append(f"<div class=\"full\" id=\"{piece_id}\" style=\"width:{width}px\">"
                          f"<p class=\"source\">{_e(caption)} Source: <code>{_e(source)}</code>, {label}.</p>"
                          f"<table><colgroup>{''.join(f'<col style=\"width:{widths[c]}px\">' for c in shown)}"
                          f"</colgroup><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>")
    return "".join(pieces)


CSS = """
:root { color-scheme: light dark; --fg: #1b1b1b; --bg: #ffffff; --muted: #5c5c5c; --line: #d0d0d0; --card: #f6f7f9;
        --undef: #8a4b00; --error: #b00020; --link: #0b57d0; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #e6e6e6; --bg: #16181c; --muted: #a8a8a8; --line: #3a3d44; --card: #1f2228; --undef: #f0b35a;
          --error: #ff7a8a; --link: #8ab4f8; } }
body { font: 14px/1.45 system-ui, sans-serif; color: var(--fg); background: var(--bg); margin: 24px;
       max-width: 1360px; }
a { color: var(--link); } h1 { font-size: 22px; } h2 { font-size: 18px; border-bottom: 1px solid var(--line); }
h3 { font-size: 15px; margin-bottom: 4px; } code { font-size: 12px; }
table { border-collapse: collapse; margin: 6px 0 12px; } th, td { border: 1px solid var(--line); padding: 2px 6px;
  vertical-align: top; text-align: left; font-size: 12px; } th { background: var(--card); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: 8px 14px; margin: 12px 0; }
.undefined { color: var(--undef); font-style: italic; } .error { color: var(--error); font-weight: 600; }
.label { font-weight: 700; text-transform: uppercase; font-size: 11px; letter-spacing: .04em; }
.full table { table-layout: fixed; width: 100%; } .full td, .full th { overflow-wrap: anywhere; }
.source { color: var(--muted); font-size: 12px; margin: 2px 0; }
figure { margin: 10px 0; } figcaption { color: var(--muted); font-size: 12px; max-width: 1100px; }
img { max-width: 100%; background: #ffffff; }
"""


def _side_text(cells):
    """One side of a key-metric row: each trial method's cell apart (labelled when there are several)."""
    if not cells:
        return "<span class=\"undefined\">no row</span>"
    texts = [_cell_text(cell) for cell in cells]
    if len(cells) == 1:
        return texts[0]
    return "<span class=\"error\">different trial setups, not pooled</span><hr>" + "<hr>".join(
        f"<a href=\"#setups\">{_e(cell['setup'])}</a>:<br>{text}" for cell, text in zip(cells, texts))


def _cell_text(side):
    parts = []
    if side["range"] is None:
        parts.append("<span class=\"undefined\">undefined</span>")
    else:
        parts.append(f"<b>{_e(side['range'])}</b>")
    if side["items"]:
        parts.append(f"(mean over {', '.join(map(str, side['items']))} items per row)")
    where = []
    if side["seeds"]:
        where.append(f"seeds {_e(side['seeds'])}")
    if side["fovs"]:
        where.append(f"FOVs {_e(side['fovs'])}")
    parts.append(f"{side['defined']} of {side['n']} rows defined" + (f"; {'; '.join(where)}" if where else ""))
    if side["missing_seeds"] or side["missing_fovs"]:
        missing = [f"seeds {side['missing_seeds']}" if side["missing_seeds"] else "",
                   f"FOVs {side['missing_fovs']}" if side["missing_fovs"] else ""]
        parts.append(f"<span class=\"error\">missing {_e(' and '.join(m for m in missing if m))}</span>")
    for seed, fov, reason in side["nulls"]:
        at = ", ".join(x for x in (f"seed {_cell(seed)}" if _cell(seed) else "", f"FOV {_cell(fov)}"
                                   if _cell(fov) else "") if x)
        parts.append(f"<span class=\"undefined\">undefined{(' at ' + _e(at)) if at else ''}: "
                     f"{_e(reason or 'no reason saved')}</span>")
    for seed, fov, outcome, reason in side["failures"]:
        at = ", ".join(x for x in (f"seed {_cell(seed)}" if _cell(seed) else "", f"FOV {_cell(fov)}"
                                   if _cell(fov) else "") if x)
        parts.append(f"<span class=\"error\">{_e(outcome)}{(' at ' + _e(at)) if at else ''}: {_e(reason)}</span>")
    if side["counts"]:
        parts.append("denominators: " + _e("; ".join(f"{k} {v}" for k, v in side["counts"].items())))
    return "<br>".join(parts)


def _value_cell(row):
    """A values.csv row's value: name = value, undefined shown as such, a metric not computed as its outcome."""
    if row.value_name is None:
        return f"<span class=\"error\">{_e(row.outcome)}</span>"
    value = "<span class=\"undefined\">undefined</span>" if row.value is None else _e(_cell(row.value))
    return f"{_e(row.value_name)} = {value}"


def _rows_link(rows):
    return ", ".join(f"<a href=\"#v{r}\">{r}</a>" for r in rows)


class Page:
    """The report's HTML, built in order; ``anchors`` lists every id in page order."""

    def __init__(self):
        self.parts, self.anchors = [], []

    def add(self, text):
        self.parts.append(text)

    def section(self, anchor, title, level=2):
        self.anchors.append(anchor)
        self.add(f"<section id=\"{anchor}\"><h{level}>{_e(title)}</h{level}>")

    def end(self):
        self.add("</section>")

    def html(self, title):
        return ("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>" + _e(title) +
                "</title><style>" + CSS + "</style></head><body>" + "".join(self.parts) + "</body></html>\n")


# --- The report -----------------------------------------------------------------------------------------------------

def _metric_meaning(selector, protocol):
    built = SHORT_SET.get(selector)
    supplied = protocol.get("metrics", {}).get(selector, {})
    return (built[0] if built else None), (built[1] if built else None), supplied.get("meaning"), \
        supplied.get("population")


def _findings(evaluation, frame, resources, pairs, card_tables, curve_tables, revision, protocol):
    """Cross-cutting anomalies shown by the tables (findings), then the protocol's findings and hypotheses."""
    out = []
    for source in evaluation.record["sources"]:
        if source.get("status") == "failed":
            out.append(("finding", f"Source {source['name']} ({source.get('kind')}) failed: "
                        f"{(source.get('error') or {}).get('message')}", "tables/sources.csv"))
        for limitation in source.get("limitations") or []:
            out.append(("finding", f"Source {source['name']}: {limitation}.", "tables/sources.csv"))
    not_computed = frame[frame["value_name"].isna()]
    for row in not_computed.itertuples():
        out.append(("finding", f"Metric {row.metric} was {row.outcome}: {row.reason}", "values.csv"))
    undefined = frame[frame["value_name"].notna() & frame["value"].isna()]
    for row in undefined.itertuples():
        out.append(("finding", f"{row.metric} {row.value_name} is undefined: {row.reason}", "values.csv"))
    for key, table in card_tables.items():
        for entry in table:
            for side, cell in ((side, cell) for side, cells in entry["sides"].items() for cell in cells):
                missing = [f"seeds {cell['missing_seeds']}" if cell["missing_seeds"] else "",
                           f"FOVs {cell['missing_fovs']}" if cell["missing_fovs"] else ""]
                if any(missing):
                    where = side + (f", {entry['condition']}" if entry["condition"] else "")
                    out.append(("finding", f"{card_anchor(key)[5:]}: {entry['selector']} {entry['value_name']} "
                                f"({where}) has no row for {' and '.join(m for m in missing if m)}.", "values.csv"))
    unpaired = pairs[pairs["pairing"] != "paired"]
    if len(unpaired):
        out.append(("finding", f"{len(unpaired)} before/after rows have no partner and are kept as unpaired.",
                    "tables/pairs.csv"))
    scopes = sorted({s for s in resources["scope"].dropna()})
    if len(scopes) > 1:
        out.append(("finding", f"Resources come in {len(scopes)} scopes ({'; '.join(s.split(':')[0] for s in scopes)})"
                    " and are not combined.", "tables/resources.csv"))
    missing = resources[resources[["wall_s", "cpu_s", "peak_rss_kib"]].isna().any(axis=1)]
    if len(missing):
        out.append(("finding", f"Resource rows with missing values: {len(missing)}; "
                    f"{'; '.join(sorted({str(r) for r in missing['reason']}))}.", "tables/resources.csv"))
    ambiguous = frame[frame["identity_notes"] != ""]
    for name in ambiguous["metric"].unique():
        note = ambiguous[ambiguous["metric"] == name]["identity_notes"].iloc[0]
        out.append(("finding", f"Metric {name}: {note}; it is not assigned by those fields.", "values.csv"))
    for name, curve in curve_tables.items():
        if collapsed(curve):
            out.append(("finding", f"The precision-recall curve of {name} collapses to one point.",
                        "tables/curves.csv"))
    sets = frame[frame["parameter_set"] != ""]
    for selector in sets["selector"].unique():
        count = sets[sets["selector"] == selector]["parameter_set"].nunique()
        out.append(("finding", f"{selector} was computed with {count} parameter sets, shown apart.",
                    "tables/metrics.csv"))
    if not revision.get("resolved"):
        out.append(("finding", f"Revision {revision['match']}.", "report.json"))
    for item in protocol.get("findings", []):
        out.append((item["label"], item["text"], item["source_table"]))
    return out


def build(evaluation, records, protocol, revision, protocol_path=None):
    """The report's HTML, its figures and tables and its summary record, in memory."""
    ids = identities(evaluation, records, protocol)
    frame = complete_rows(evaluation, ids)
    resources = resource_rows(evaluation, records)
    pairs = pair_rows(frame)
    curve_tables = curves(evaluation)
    keys = card_order(frame, protocol)
    cards = {card_key(c["method"], c.get("mode")): c for c in protocol.get("methods", [])}
    card_tables = {}
    for key in keys:
        rows = _in_card(frame, key)
        card_tables[key] = key_table(_selected(rows, cards.get(key, {}).get("key_metrics", [])))
    findings = _findings(evaluation, frame, resources, pairs, card_tables, curve_tables, revision, protocol)
    figures, figure_files = {key: [] for key in keys}, {}
    for key in keys:
        panels, facts = metric_figure(card_tables[key])
        if panels:
            name = f"figures/{card_anchor(key)[5:]}-metrics.png"
            figure_files[name] = draw_metric_figure(_in_card(frame, key), panels)
            figures[key].append((f"fig-{card_anchor(key)[5:]}-metrics", name,
                                 "Saved values of the key metrics, one dot per row; before and after are separate "
                                 "columns, undefined values are counted in each panel's title, not drawn.", facts))
    for n, figure in enumerate(protocol.get("figures", []), start=1):
        key = card_key(figure["method"], figure.get("mode"))
        if key not in figures:
            raise ReportError(f"protocol figure {n} names {key}, which has no card")
        panels = []
        for spec in figure["panels"]:
            path, data = records.file(spec)
            panels.append((spec.get("label") or Path(path).name, _array(path, data), spec.get("channel")))
        detections = truth = None
        if "detections" in figure:
            detections = _points(*records.file(figure["detections"]))
        if "truth" in figure:
            truth = _points(*records.file(figure["truth"]))
        peaks = figure.get("peaks", 0)
        if not isinstance(peaks, int) or not 0 <= peaks < len(panels):
            raise ReportError(f"protocol figure {n}: peaks is the index of a panel")
        png, facts = method_figure(panels, detections, truth, peaks)
        name = f"figures/{card_anchor(key)[5:]}-{n}.png"
        figure_files[name] = png
        caption = (f"{figure.get('title') or 'Saved images'}: maximum projections over Z on one linear display range "
                   f"{_fmt(facts['display_range'][0])} to {_fmt(facts['display_range'][1])}.")
        if facts["detections"] is not None:
            caption += f" Orange circles: the {facts['detections']} saved detections."
        if facts["dim_punctum"] is not None:
            d = facts["dim_punctum"]
            caption += (f" Green crosses: representative truth puncta by realized peak (3 × 3 × 3 maximum) in "
                        f"{facts['peaks_from']}; 1 is the dim punctum, truth row {d['index']} at ZYX {d['center']} "
                        f"with peak {_fmt(d['peak'])}.")
        figures[key].append((f"fig-{card_anchor(key)[5:]}-{n}", name, caption, facts))

    title = protocol.get("title") or "Saved-evaluation report"
    page = Page()
    page.add(f"<h1>{_e(title)}</h1>")
    page.add(f"<p class=\"source\">Rendered from <code>{_e(evaluation.folder / 'evaluation.json')}</code> "
             f"(SHA-256 {evaluation.sha256[:16]}…, status <b>{_e(evaluation.record['status'])}</b>).</p>")
    page.section("summary", "Summary")
    _setup(page, evaluation, records, protocol, frame, resources, revision, protocol_path)
    page.section("cards", "Methods and modes", 3)
    for key in keys:
        _card(page, key, cards.get(key, {}), card_tables[key], frame, evaluation, figures[key])
    page.end()
    page.section("findings", "Cross-cutting findings and anomalies", 3)
    page.add("<p>Each item names the table it comes from. A <span class=\"label\">finding</span> is shown by the "
             "tables; a <span class=\"label\">hypothesis</span> is an explanation this evaluation did not test.</p>")
    page.anchors.append("findings-list")
    page.add("<div id=\"findings-list\"><h4>Findings</h4><ul>" + "".join(
        f"<li><span class=\"label\">finding</span> {_e(text)} <span class=\"source\">Source: <code>{_e(source)}"
        f"</code></span></li>" for label, text, source in findings if label == "finding") + "</ul></div>")
    hypotheses = [f for f in findings if f[0] == "hypothesis"]
    page.anchors.append("hypotheses")
    page.add("<div id=\"hypotheses\"><h4>Hypotheses</h4>" + ("<ul>" + "".join(
        f"<li><span class=\"label\">hypothesis</span> {_e(text)} <span class=\"source\">Source: <code>{_e(source)}"
        f"</code></span></li>" for _l, text, source in hypotheses) + "</ul>" if hypotheses else
        "<p>None supplied.</p>") + "</div>")
    page.end()
    page.section("reading-order", "Reading order", 3)
    order = [("setup", "Setup")] + [(card_anchor(k), f"Card: {_card_title(k, cards.get(k, {}))}") for k in keys] + \
        [("findings", "Findings and anomalies"), ("hypotheses", "Hypotheses")]
    for key in keys:
        order.append((f"details-{card_anchor(key)[5:]}", f"Details: {_card_title(key, cards.get(key, {}))}"))
        order += [(anchor, f"Figure: {Path(name).name}") for anchor, name, _c, _f in figures[key]]
    order += [("appendix", "Appendix"), ("identity", "Identity, revision and checksums"),
              ("a-values", "Every value, per seed and FOV"), ("a-pairs", "Before and after pairs"),
              ("a-resources", "Resources and their scopes"), ("a-diagnostics", "Diagnostics"),
              ("a-omitted", "Omitted or reduced")]
    page.add("<ol>" + "".join(f"<li><a href=\"#{a}\">{_e(t)}</a></li>" for a, t in order) + "</ol>")
    page.end()
    page.end()

    page.section("method-details", "Method details and figures")
    for key in keys:
        anchor = f"details-{card_anchor(key)[5:]}"
        page.section(anchor, f"Details: {_card_title(key, cards.get(key, {}))}", 3)
        page.add(f"<p><a href=\"#{card_anchor(key)}\">Back to the card</a>.</p>")
        for fig_anchor, name, caption, _facts in figures[key]:
            page.anchors.append(fig_anchor)
            page.add(f"<figure id=\"{fig_anchor}\"><img src=\"{_e(name)}\" alt=\"{_e(caption)}\">"
                     f"<figcaption>{_e(caption)}</figcaption></figure>")
        if not figures[key]:
            page.add("<p>No figure: the card has no numeric key metric and the protocol names no saved image.</p>")
        rows = _in_card(frame, key)
        page.add(_table(["row", "metric", "value", "unit", "condition", "seed", "FOV", "side", "outcome", "reason"],
                        [[f"<a href=\"#v{r.row}\">{r.row}</a>", _e(r.metric), _value_cell(r), _e(_cell(r.unit)),
                          _e(_cell(r.condition)), _e(_cell(r.seed)), _e(_cell(r.fov)), _e(_cell(r.side)),
                          _e(r.outcome), _e(_cell(r.reason))] for r in rows.itertuples()]))
        page.end()
    page.end()

    tables = _tables(evaluation, frame, pairs, resources, curve_tables, records)
    page.section("appendix", "Appendix")
    _identity(page, evaluation, revision, protocol_path, records)
    page.section("a-sources", "Sources", 3)
    page.add(full_table(tables["tables/sources.csv"], "tables/sources.csv", "a-sources-t"))
    page.add(full_table(tables["tables/consumed.csv"], "tables/consumed.csv", "a-consumed",
                        "Every file the evaluation consumed, with its SHA-256."))
    page.end()
    page.section("a-values", "Every value, per seed and FOV", 3)
    page.add("<p>Every row of <code>values.csv</code>, failed and undefined rows included, with its identity.</p>")
    page.add(full_table(tables["tables/values.csv"], "tables/values.csv", "a-values-t",
                        row_ids=[f"v{r}" for r in frame["row"]]))
    page.end()
    page.section("a-pairs", "Before and after pairs", 3)
    page.add(full_table(pairs, "tables/pairs.csv", "a-pairs-t", "Rows without a partner are kept as unpaired.")
             if len(pairs) else "<p>No row has a before or after side.</p>")
    page.end()
    page.section("a-resources", "Resources and their scopes", 3)
    page.add(f"<p>{_e(protocol.get('resources', 'No resource note supplied.'))} Each row keeps its scope; rows of "
             "different scopes are not combined.</p>")
    page.add(full_table(resources, "tables/resources.csv", "a-resources-t"))
    page.end()
    page.section("a-diagnostics", "Diagnostics", 3)
    page.add(full_table(tables["tables/metrics.csv"], "tables/metrics.csv", "a-metrics",
                        "Each metric's function, parameters, arguments, counts, configuration and details file."))
    if curve_tables:
        page.add(full_table(tables["tables/curves.csv"], "tables/curves.csv", "a-curves",
                            "Every saved curve point; no point is selected."))
    page.end()
    omitted = _omitted(frame, card_tables, figures, keys, evaluation)
    page.section("a-omitted", "Omitted or reduced", 3)
    page.add("<ul>" + "".join(f"<li>{_e(o)}</li>" for o in omitted) + "</ul>")
    page.end()
    page.end()
    summary = {"format": REPORT_FORMAT, "title": title,
               "evaluation": {"path": str(evaluation.folder), "sha256": evaluation.sha256,
                              "values_sha256": evaluation.values_sha256, "status": evaluation.record["status"],
                              "format": evaluation.record["format"]},
               "revision": revision, "anchors": page.anchors,
               "cards": [{"anchor": card_anchor(k), "method": k[0], "mode": k[1],
                          "metrics": sorted(_in_card(frame, k)["metric"].unique().tolist()),
                          "verdict": cards.get(k, {}).get("verdict")} for k in keys],
               "findings": [{"label": label, "text": text, "source_table": source}
                            for label, text, source in findings],
               "figures": [{"anchor": anchor, "path": name, "card": card_anchor(key), "facts": facts}
                           for key in keys for anchor, name, _c, facts in figures[key]],
               "omitted": omitted}
    return page.html(title), figure_files, tables, summary


def _card_title(key, card):
    method, mode = key
    if card.get("title"):
        return card["title"]
    if method is None:
        return "Metrics without a method"
    return method if mode is None else f"{method}, mode {mode}"


def _setup(page, evaluation, records, protocol, frame, resources, revision, protocol_path):
    page.section("setup", "Setup", 3)
    qualification = protocol.get("qualification")
    page.add("<p><b>Qualification.</b> " + (_e(qualification) if qualification else
                                             "<span class=\"undefined\">Not stated: no protocol supplied a pilot or "
                                             "scientific qualification.</span>") + "</p>")
    page.add("<ul>" + "".join(f"<li>{_e(s)}</li>" for s in LIMITATIONS) + "</ul>")
    kinds = pd.Series([s.get("kind") for s in evaluation.record["sources"]]).value_counts()
    route = ", ".join(f"{n} {k} source{'s' if n > 1 else ''}" for k, n in kinds.items())
    page.add(f"<p><b>Route.</b> {_e(protocol.get('route', 'Not described by a protocol.'))} The evaluation read "
             f"{_e(route)}.</p>")
    page.add(f"<p><b>Inputs.</b> {_e(protocol.get('inputs', 'Not described by a protocol.'))} "
             f"<b>Truth.</b> {_e(protocol.get('truth', 'Not described by a protocol.'))}</p>")
    files = [(s["name"], key, entry) for s in evaluation.record["sources"] if s.get("kind") == "files"
             for key, entry in (s.get("files") or {}).items()]
    if files:
        page.add(_table(["files source", "key", "path", "SHA-256"],
                        [[_e(n), _e(k), f"<code>{_e(e['path'])}</code>", f"<code>{_e(str(e['sha256'])[:16])}…</code>"]
                         for n, k, e in files]))
    trials = []
    for name in records.trials:
        record = records.trials[name]
        configuration = record.get("configuration") or {}
        trials.append([_e(name), _e(_cell(records.trial_identity(name).get("method"))),
                       f"<code>{_e(json.dumps(configuration.get('method'), sort_keys=True))}</code>",
                       _e(json.dumps(configuration.get("condition") or {}, sort_keys=True)),
                       _e(_cell(record.get("seed"))), _e(record.get("status"))])
    page.add("<p><b>Method configuration, conditions and seeds.</b> "
             + ("Each trial's resolved method, condition and seed as its checked record holds them.</p>" + _table(
                 ["trial source", "method", "configuration", "condition", "seed", "status"], trials)
                if trials else "No trial source; the protocol's rows name methods.</p>"))
    setups = frame[["setup", "setup_json"]].dropna().drop_duplicates()
    page.anchors.append("setups")
    page.add("<div id=\"setups\"><p><b>Trial setups.</b> Rows whose trials differ in method, configuration or "
             "condition have different setups and are never pooled into one range.</p>" +
             (_table(["setup", "method and condition"], [[_e(r.setup), f"<code>{_e(r.setup_json)}</code>"]
                                                        for r in setups.itertuples()]) if len(setups)
              else "<p>No metric reads a trial source.</p>") + "</div>")
    conditions = sorted({str(c) for c in frame["condition"].dropna()} | set(protocol.get("conditions", {})))
    page.add("<p><b>Conditions.</b></p>" + (_table(["condition", "description"], [
        [_e(c), _e(protocol.get("conditions", {}).get(c)) if c in protocol.get("conditions", {}) else
         "<span class=\"undefined\">no description supplied</span>"] for c in conditions]) if conditions
        else "<p>No condition is named.</p>"))
    seeds = _labels(frame["seed"].dropna())
    modes = _labels(frame["mode"].dropna())
    page.add(f"<p><b>Seeds</b> {_e(seeds or 'none named')}; <b>modes</b> {_e(modes or 'none named')}; <b>dtype</b> "
             f"{_e(protocol.get('dtype', 'not stated'))}.</p>")
    metrics = []
    for selector, group in frame.groupby("selector", sort=False):
        meaning, population, note, supplied = _metric_meaning(selector, protocol)
        values = ", ".join(f"{v} ({u})" if isinstance(u, str) else str(v) for v, u in
                           group[["value_name", "unit"]].dropna(subset=["value_name"]).drop_duplicates()
                           .itertuples(index=False))
        parameters = "; ".join(sorted(set(group["parameters"])))
        metrics.append([_e(selector), _e(values or "none computed"),
                        _e(meaning) if meaning else "<span class=\"undefined\">no built-in meaning</span>",
                        _e(population or ""), _e(" ".join(x for x in (note, supplied) if x)),
                        f"<code>{_e(parameters)}</code>"])
    page.add("<p><b>Metrics, their meaning and population, and the fixed parameters they were computed with.</b></p>"
             + _table(["metric", "values (unit)", "meaning", "population", "protocol note", "parameters"], metrics))
    cutoffs = protocol.get("cutoffs", [])
    page.add("<p><b>Fixed cut-offs.</b> " + ("The protocol's fixed cut-offs; none was selected from these results."
                                             "</p>" + _table(["name", "value", "unit", "note"], [
                                                 [_e(c["name"]), _e(c["value"]), _e(c["unit"]), _e(c.get("note", ""))]
                                                 for c in cutoffs]) if cutoffs else
                                             "None supplied beyond the parameters above.</p>"))
    missing = list(protocol.get("missing_prerequisites", []))
    missing += [f"{r.metric}: {r.outcome}, {r.reason}" for r in frame[frame["value_name"].isna()].itertuples()]
    page.add("<p><b>Missing prerequisites.</b></p>" + ("<ul>" + "".join(f"<li class=\"error\">{_e(m)}</li>"
                                                                       for m in missing) + "</ul>"
                                                      if missing else "<p>None.</p>"))
    scopes = []
    for scope, group in resources.groupby("scope", dropna=False, sort=False):
        scopes.append([_e(_cell(scope) or "no resource record"), str(len(group)),
                       *(_e(_span([v for v in group[f] if not pd.isna(v)]) or "missing")
                         for f in ("wall_s", "cpu_s", "peak_rss_kib"))])
    page.add("<p><b>Resources</b>, by scope (ranges across rows; see the "
             "<a href=\"#a-resources\">resource table</a>).</p>" +
             _table(["scope", "rows", "wall s", "CPU s", "peak RSS KiB"], scopes))
    page.add(f"<p><b>Revision.</b> {_e(revision['match'])}." + ("" if revision.get("resolved") else
                                                                " <span class=\"error\">Unresolved.</span>") +
             f" Evaluation status <b>{_e(evaluation.record['status'])}</b>: "
             f"{_e(json.dumps(evaluation.record['counts']))}.</p>")
    page.end()


def _card(page, key, card, table, frame, evaluation, figures):
    anchor = card_anchor(key)
    page.anchors.append(anchor)
    rows = _in_card(frame, key)
    page.add(f"<article class=\"card\" id=\"{anchor}\"><h3>{_e(_card_title(key, card))}</h3>")
    page.add(f"<p class=\"source\">method <code>{_e(_cell(key[0]) or 'none')}</code>, mode "
             f"<code>{_e(_cell(key[1]) or 'none')}</code>; {len(rows)} rows of values.csv.</p>")
    page.add("<p><b>Purpose.</b> " + (_e(card["purpose"]) if card.get("purpose") else
                                       "<span class=\"undefined\">No purpose supplied by a protocol.</span>") + "</p>")
    present = sorted({str(c) for c in rows["condition"].dropna()})
    targeted = card.get("targeted_conditions", [])
    text = ", ".join(targeted) if targeted else "none supplied"
    absent = [c for c in targeted if c not in present]
    page.add(f"<p><b>Targeted conditions.</b> {_e(text)}. In this evaluation: {_e(', '.join(present) or 'none named')}"
             + (f"; <span class=\"error\">absent: {_e(', '.join(absent))}</span>" if absent else "") + ".</p>")
    body = []
    for entry in table:
        meaning = SHORT_SET.get(entry["selector"])
        label = f"<b>{_e(entry['selector'])}</b> {_e(_cell(entry['value_name']) or '(not computed)')}"
        if entry["parameter_set"]:
            label += f" <span class=\"source\">parameter set {_e(entry['parameter_set'])}</span>"
        if meaning:
            label += f"<br><span class=\"source\">population: {_e(meaning[1])}</span>"
        body.append([label, _e(_cell(entry["unit"])), _e(_cell(entry["condition"]) or "—"),
                     _side_text(entry["sides"].get("before")) if any(s in entry["sides"] for s in SIDES)
                     else _side_text(entry["sides"].get("single")),
                     _side_text(entry["sides"].get("after")) if any(s in entry["sides"] for s in SIDES) else "—",
                     "values.csv rows " + _rows_link(entry["rows"])])
    page.add("<p><b>Key metrics</b>, before and after, as ranges across seeds and FOVs.</p>" +
             (_table(["metric", "unit", "condition", "before (or the value)", "after", "source"], body, "key")
              if body else "<p class=\"undefined\">No key metric has a row in this card.</p>"))
    for field, title in (("preconditions", "Preconditions supplied"), ("harm", "Harm results supplied")):
        items = card.get(field, [])
        page.add(f"<p><b>{title}.</b> " + ("</p><ul>" + "".join(
            f"<li>{_e(i['text'])} <span class=\"source\">Source: <code>{_e(i['source_table'])}</code></span></li>"
            for i in items) + "</ul>" if items else "None supplied.</p>"))
    verdict = card.get("verdict")
    if verdict is not None:
        clause = verdict["failing_clause"]
        page.add(f"<p><b>Verdict (supplied by the protocol).</b> {_e(verdict['value'])}; failing clause: "
                 f"{_e(clause) if clause else 'none failed'}. <span class=\"source\">Source: "
                 f"<code>{_e(verdict['source_table'])}</code></span></p>")
    else:
        page.add("<p><b>Verdict.</b> None: the protocol supplied no verdict, and the report makes none.</p>")
    notes = []
    for row in rows[rows["value_name"].isna()].itertuples():
        notes.append(f"<span class=\"error\">{_e(row.metric)} {_e(row.outcome)}</span>: {_e(row.reason)}")
    for row in rows[rows["value_name"].notna() & rows["value"].isna()].itertuples():
        notes.append(f"<span class=\"undefined\">{_e(row.metric)} {_e(row.value_name)} undefined</span>: "
                     f"{_e(row.reason)}")
    used = set(", ".join(rows["sources"]).split(", ")) - {""}
    for source in evaluation.record["sources"]:
        if source["name"] in used:
            notes += [f"source {_e(source['name'])}: {_e(t)}" for t in source.get("limitations") or []]
            if source.get("status") == "failed":
                notes.append(f"<span class=\"error\">source {_e(source['name'])} failed</span>: "
                             f"{_e((source.get('error') or {}).get('message'))}")
    notes += [_e(t) for t in card.get("limitations", [])]
    page.add("<p><b>Failures, null reasons and limitations.</b></p>" +
             ("<ul>" + "".join(f"<li>{n}</li>" for n in notes) + "</ul>" if notes else "<p>None.</p>"))
    links = [f"<a href=\"#details-{anchor[5:]}\">details</a>"] + [f"<a href=\"#{a}\">{_e(Path(n).name)}</a>"
                                                                  for a, n, _c, _f in figures]
    page.add(f"<p>Read next: {', '.join(links)}.</p></article>")


def _identity(page, evaluation, revision, protocol_path, records):
    page.section("identity", "Identity, revision and checksums", 3)
    code = evaluation.record["software"].get("code") or {}
    rows = [["evaluation.json", f"<code>{_e(evaluation.folder / 'evaluation.json')}</code>",
             f"<code>{evaluation.sha256}</code>"],
            ["values.csv", f"<code>{_e(evaluation.folder / 'values.csv')}</code>",
             f"<code>{evaluation.values_sha256}</code>"]]
    for key in ("sources", "metrics"):
        entry = evaluation.record["inputs"].get(key) or {}
        rows.append([f"{key} file", f"<code>{_e(entry.get('path'))}</code>", f"<code>{_e(entry.get('sha256'))}</code>"])
    if protocol_path is not None:
        entry = records.saved.verified[str(protocol_path)]
        rows.append(["protocol", f"<code>{_e(protocol_path)}</code>", f"<code>{entry['sha256']}</code>"])
    page.add(_table(["record", "path", "SHA-256"], rows))
    page.add(f"<p>Format <code>{_e(evaluation.record['format'])}</code>, generated "
             f"{_e(evaluation.record.get('generated_at'))}; code {_e(json.dumps(code, sort_keys=True))}; "
             f"environment {_e(json.dumps(evaluation.record['software'].get('environment'), sort_keys=True))}.</p>")
    page.anchors.append("revision")
    page.add(f"<p id=\"revision\"><b>Revision check.</b> {_e(revision['match'])}. Checked against HEAD "
             f"<code>{_e(revision['head'])}</code>; the evaluation sources are "
             f"<code>{_e(', '.join(revision['sources']))}</code>.</p>")
    page.anchors.append("a-verified")
    verified = pd.DataFrame(list(records.saved.verified.values()))
    page.add(full_table(verified, "report.json", "a-verified", "Every file this report read, checked as read."))
    page.end()


def _tables(evaluation, frame, pairs, resources, curve_tables, records):
    values = frame.drop(columns=["number", "items", "raw"]).copy()
    sources = pd.DataFrame([{
        "source": s["name"], "kind": s.get("kind"), "status": s.get("status"),
        "error": (s.get("error") or {}).get("message"), "limitations": "; ".join(s.get("limitations") or []),
        "path": s.get("path") or (s.get("manifest") or {}).get("path"), "consumed": len(s.get("consumed") or []),
        **{k: records.trial_identity(s["name"]).get(k) for k in ("method", "condition", "seed")}}
        for s in evaluation.record["sources"]])
    consumed = pd.DataFrame([{"source": s["name"], **c} for s in evaluation.record["sources"]
                             for c in s.get("consumed") or []],
                            columns=["source", "path", "sha256", "bytes", "checked_against"])
    metrics = pd.DataFrame([{
        "metric": m["name"], "metric_id": m.get("metric"), "function": m.get("function"), "outcome": m.get("outcome"),
        "status": m.get("status"), "parameters": json.dumps(m.get("parameters", {}), sort_keys=True),
        "arguments": json.dumps(m.get("arguments", {}), sort_keys=True), "counts": json.dumps(m.get("counts") or {}),
        "config": json.dumps(m.get("config") or {}, sort_keys=True, default=str),
        "details_file": m.get("details_file"), "sources": ", ".join(m.get("sources") or [])}
        for m in evaluation.record["metrics"]])
    tables = {"tables/values.csv": values, "tables/pairs.csv": pairs, "tables/resources.csv": resources,
              "tables/sources.csv": sources, "tables/consumed.csv": consumed, "tables/metrics.csv": metrics}
    if curve_tables:
        tables["tables/curves.csv"] = pd.concat([c.assign(metric=name) for name, c in curve_tables.items()],
                                                ignore_index=True)
        columns = ["metric", *[c for c in tables["tables/curves.csv"].columns if c != "metric"]]
        tables["tables/curves.csv"] = tables["tables/curves.csv"][columns]
    return tables


def _omitted(frame, card_tables, figures, keys, evaluation):
    out = ["Ranges show the minimum and maximum of the defined saved values; every value is in the appendix and "
           "tables/values.csv."]
    if frame["items"].notna().any():
        out.append("A list- or mapping-valued value enters a card's range as the mean over its items, the number of "
                   "items stated; its items are in tables/values.csv.")
    for key in keys:
        for _a, _n, _c, facts in figures[key]:
            for item in facts.get("omitted", []):
                out.append(f"{card_anchor(key)[5:]}: the metric figure shows {MAX_PANELS} panels; "
                           f"{' '.join(str(i) for i in item if i is not None)} is in the card table only.")
    if evaluation.details:
        out.append("Each metric's details file is listed in the diagnostics with its path; only saved curves are "
                   "drawn as tables.")
    return out


def _check_output(output_dir, evaluation):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"the report output must be a new or empty folder: {output_dir}")
    for root in evaluation.roots():
        root = Path(root).resolve()
        if output_dir == root or output_dir.is_relative_to(root):
            raise ReportError(f"the report output {output_dir} is inside the evaluation or one of its sources "
                              f"({root})")
    return output_dir


def render(evaluation_dir, output_dir, protocol=None, revision_check=None):
    """Write the report of a saved evaluation in ``output_dir`` (new or empty, outside the evaluation and its
    sources); return the folder. Everything is checked and built before anything is written."""
    saved = Saved()
    evaluation = SavedEvaluation(evaluation_dir, saved)
    output_dir = _check_output(output_dir, evaluation)
    protocol_path = None if protocol is None else Path(protocol).resolve()
    protocol = load_protocol(protocol_path, saved)
    revision = (revision_check or check_revision)(evaluation.record["software"].get("code"))
    records = Records(evaluation, saved)
    text, figure_files, tables, summary = build(evaluation, records, protocol, revision, protocol_path)
    files = {"report.html": text.encode()}
    files.update(figure_files)
    for name, table in tables.items():
        files[name] = table.to_csv(index=False).encode()
    summary["verified"] = list(saved.verified.values())
    summary["protocol"] = None if protocol_path is None else {
        "path": str(protocol_path), "sha256": saved.verified[str(protocol_path)]["sha256"]}
    summary["outputs"] = [{"path": name, "sha256": _sha(data), "bytes": len(data)} for name, data in files.items()]
    files["report.json"] = (json.dumps(summary, indent=1, default=str) + "\n").encode()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        path = output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return output_dir
