"""Real-engine fixtures, completion checks and retained evidence for X1–X3 and X7/X10/X11.

X3's named workflow token and provenance fields are checked separately before
normalization. Directory markers are ignored only for Snakemake's declared
directory outputs; X11's global hashes vary only on tasks outside its rerun set. Its report Hash
cell varies only when no FOV task reran, and is checked against every FOV
record of each tree's sample; X11 also compares all file names outside workflow/.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import pytest

from starfinder.workflow import plan, verify

from . import workflow_examples as X
from . import workflow_nextflow as N
from .test_workflow_snakemake_dryrun import TRIGGERS, _snakemake, downstream, snakemake


def evidence(work, name, value):
    """Keep evidence locally and, when requested, under the controller's artifact directory."""
    path = Path(work) / "evidence" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, indent=2) + "\n")
    destination = os.environ.get("STARFINDER_WORKFLOW_EVIDENCE")
    if destination:
        target = Path(destination) / Path(work).parent.name / Path(work).name / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if os.environ.get("STEP_WRITES"):
            with Path(os.environ["STEP_WRITES"]).open("a") as handle:
                handle.write(str(target) + "\n")
        return target
    return path


def require(engine):
    if engine == "nextflow":
        N.executable()
    else:
        _snakemake()
    if any(importlib.util.find_spec(m) is None for m in ("anndata", "multiview_stitcher")):
        pytest.skip("real engine validation needs the anndata and multiview-stitcher extras")


def supplied(work):
    """A hand-written XML and view map for tiny_sample's four views, without Fiji."""
    import pandas as pd
    folder = Path(work) / "supplied" / X.SAMPLE
    folder.mkdir(parents=True, exist_ok=True)
    registrations = []
    setups = []
    views = []
    for index, fov in enumerate(X.FOVS):
        column, row = divmod(index, 2)
        setups.append(f'<ViewSetup><id>{index}</id><size>36 36 5</size>'
                      '<voxelSize><unit>µm</unit><size>0.5 0.5 1.0</size></voxelSize>'
                      f'<attributes><tile>{index}</tile></attributes></ViewSetup>')
        registrations.append(
            f'<ViewRegistration timepoint="0" setup="{index}"><ViewTransform type="affine">'
            '<Name>Stitching Transform</Name>'
            '<affine>1 0 0 0 0 1 0 0 0 0 1 0</affine></ViewTransform>'
            '<ViewTransform type="affine"><Name>Translation to Regular Grid</Name>'
            f'<affine>1 0 0 {column * 24} 0 1 0 {row * 24} 0 0 1 0</affine>'
            '</ViewTransform><ViewTransform type="affine"><Name>calibration</Name>'
            '<affine>1 0 0 0 0 1 0 0 0 0 2 0</affine></ViewTransform></ViewRegistration>')
        views.append(dict(view=index, grid_column=column, grid_row=row, fov_id=fov, kind="fov"))
    links = []
    for a, b in ((0, 1), (0, 2), (1, 3), (2, 3)):
        links.append(f'<PairwiseResult view_setup_a="{a}" timepoint_a="0" view_setup_b="{b}" timepoint_b="0">'
                     '<shift>1 0 0 0 0 1 0 0 0 0 1 0</shift>'
                     f'<correlation>0.99</correlation><hash>synthetic-{a}-{b}</hash>'
                     '<overlap_boundingbox>0 0 0 36 36 5</overlap_boundingbox></PairwiseResult>')
    (folder / "dataset.xml").write_text('<SpimData><SequenceDescription><ViewSetups>' + ''.join(setups) +
                                       '</ViewSetups></SequenceDescription><ViewRegistrations>' + ''.join(registrations) +
                                       '</ViewRegistrations><StitchingResults>' + ''.join(links) +
                                       '</StitchingResults></SpimData>\n')
    pd.DataFrame(views).to_csv(folder / "view_map.csv", index=False)


class EngineRun:
    """One persistent engine work folder; every invocation saves its job list and records."""

    def __init__(self, work, engine, variant="per-fov", *, fixture=True):
        require(engine)
        self.work, self.engine, self.variant = Path(work), engine, variant
        if fixture:
            X.golden().write_fixture(self.work)
        if variant == "supplied-xml":
            supplied(self.work)
        self.config = N.small_resources(X.variants(self.work)[variant])
        self.path = X.write_config(self.work, "validation", self.config)
        sources = [p for p in self.work.rglob("*") if p.is_file()]
        evidence(self.work, "inputs.json", [{"path": str(p), "bytes": p.stat().st_size,
                  "mtime_ns": p.stat().st_mtime_ns, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in sorted(sources)])
        self.data = plan(self.path)
        self.out = Path(self.data["roots"]["output"])
        self.manifest = self.out / "workflow/manifest.json"
        self.invocations = 0

    def records(self):
        return {t["id"]: json.loads((self.out / t["record"]).read_text()) for t in self.data["tasks"]}

    def invoke(self, expected=None, *, environment=None, held=False):
        """Run the engine on the plan; ``held``: an export held for review (exit 4) ends the run, which fails and is
        not checked for completion."""
        tag = f"invocation-{self.invocations:02d}"
        self.invocations += 1
        before = {t["id"]: (self.out / t["record"]).read_bytes() for t in self.data["tasks"]
                  if (self.out / t["record"]).is_file()}
        if self.engine == "nextflow":
            done, rows = N.run(getattr(self, "engine_work", self.work), self.path, *(["-resume"] if before else []), stub=False,
                               environment=environment)
            actual = {r["tag"] for r in N.tasks(rows) if r["status"] == "COMPLETED"}
        else:
            # The subprocess environment is inherited by Snakemake and its real task commands.
            from unittest.mock import patch
            with patch.dict(os.environ, environment or {}):
                done = snakemake(getattr(self, "engine_work", self.work), self.path, "--rerun-incomplete", *TRIGGERS)
            rows = []
            actual = set(re.findall(r" --task (\S+) ", done.stdout + done.stderr))
        evidence(self.work, tag + ".json", {"engine": self.engine, "returncode": done.returncode,
                 "stdout": done.stdout, "stderr": done.stderr, "trace": rows,
                 "expected": sorted(expected) if expected is not None else None, "actual": sorted(actual)})
        if held:
            assert done.returncode != 0, done.stdout[-6000:] + done.stderr[-6000:]
            evidence(self.work, tag + "-records.json", {t["id"]: json.loads((self.out / t["record"]).read_text())
                                                        for t in self.data["tasks"]
                                                        if (self.out / t["record"]).is_file()})
            return actual
        assert done.returncode == 0, done.stdout[-6000:] + done.stderr[-6000:]
        self.data = json.loads(self.manifest.read_text())
        if self.engine == "nextflow":
            checks = [r for r in rows if r["name"] == "VERIFY_AFTER"]
            assert len(checks) == 1 and checks[0]["status"] == "COMPLETED", rows
            task_rows = N.tasks(rows)
            assert Counter(r["tag"] for r in task_rows) == Counter(t["id"] for t in self.data["tasks"])
            assert all(r["status"] in ("COMPLETED", "CACHED") for r in task_rows)
        if expected is not None:
            assert actual == set(expected), (actual, expected)
            after = self.records()
            for task_id, payload in before.items():
                if task_id not in expected:
                    assert_unchanged(payload, (self.out / next(t["record"] for t in self.data["tasks"]
                                                               if t["id"] == task_id)).read_bytes(), task_id)
                else:
                    assert json.loads(payload) != after[task_id], task_id
        self.check()
        evidence(self.work, tag + "-records.json", self.records())
        return actual

    def check(self):
        result = verify(self.manifest, hash=True, final=True)
        evidence(self.work, f"completion-{self.invocations:02d}.json", result)
        assert result["passed"], result
        for task_id, record in self.records().items():
            assert record["status"] == "succeeded" and not record.get("stub", False), task_id
            # A task a person ran by its command (the placement task after an approval, under Snakemake) says manual.
            assert record["workflow"]["execution_context"]["engine"] in (
                self.engine, *(("manual",) if task_id in getattr(self, "manual", ()) else ())), task_id
        assert f'<a href="{X.SAMPLE}/report.html">{X.SAMPLE}</a>' in (self.out / "report/index.html").read_text()

    def replan(self, config):
        X.write_config(self.work, "validation", config)
        paths = [self.path, *sorted(p for p in X.input_root(self.work).rglob("*") if p.is_file())]
        evidence(self.work, f"replan-{self.invocations:02d}-inputs.json",
                 [{"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
                   "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths])
        self.data = plan(self.path)
        return self.data


def snapshot(run):
    """Every declared file, directory member and file declared through export/report records."""
    result = OutputSnapshot()
    result.file_names = tree_names(run.out)
    inventory = []
    for task in run.data["tasks"]:
        for declared in [*task["outputs"], task["record"]]:
            path = Path(declared) if Path(declared).is_absolute() else run.out / declared
            paths = [path]
            if path.name == "report.json" and task["kind"] == "report":
                paths += [path.parent / entry["path"] for entry in json.loads(path.read_text())["files"]]
            elif path.name == "export.json" and task["kind"] == "export":
                from starfinder.export._record import PARTS, listed_files
                record = json.loads(path.read_text())
                paths += [path.parent / name for part in PARTS for name in listed_files(record, part)]
            files = [file for output in paths
                     for file in (sorted(output.rglob("*")) if output.is_dir() else [output])]
            for file in files:
                if file.is_file():
                    name = file.relative_to(run.out).as_posix()
                    result[name] = file.read_bytes()
                    inventory.append({"path": str(file), "bytes": len(result[name]),
                                      "sha256": hashlib.sha256(result[name]).hexdigest()})
    # X3 removes its first tree and X11 moves it: keep each identity before that
    # happens, rather than reconstructing a missing file's digest at handoff.
    evidence(run.work, f"snapshot-{run.invocations:02d}.json", inventory)
    return result


class OutputSnapshot(dict):
    """Declared contents plus the file names captured before a recovery changes the tree."""


def tree_names(root):
    """All file names under an output root outside workflow/, without reading extra contents."""
    return {p.relative_to(root).as_posix() for p in Path(root).rglob("*")
            if p.is_file() and not p.relative_to(root).as_posix().startswith("workflow/")}


def is_marker(run, path, engine):
    """Only Snakemake's marker directly inside a currently declared directory is undeclared."""
    return engine == "snakemake" and Path(path).name == ".snakemake_timestamp" and any(
        o.endswith("/") and Path(path).parent.as_posix() == o.rstrip("/")
        for t in run.data["tasks"] for o in t["outputs"])


def file_name_differences(run, left, right, *, left_engine=None, right_engine=None):
    """Compare tree names only, with the same narrow marker rule as declared contents."""
    left_engine = left_engine or run.engine
    right_engine = right_engine or run.engine
    a = {p for p in left if not is_marker(run, p, left_engine)}
    b = {p for p in right if not is_marker(run, p, right_engine)}
    return [{"path": p, "side": "left" if p in a else "right"} for p in sorted(a ^ b)]


def report_hash_exception(run, rerun_tasks):
    """X11's rendered provenance exception applies only when no FOV task reran."""
    return rerun_tasks is not None and not any(
        t.get("kind") == "fov" and t["id"] in rerun_tasks for t in run.data["tasks"])


REPORT_HASH_LABEL = "Hash (FOVs ran under)"
REPORT_HASH_SEVERAL = "several; see each FOV's run.json"


def report_hash_cell(run, path, text, tree):
    """Validate the one Hash cell against this sample's FOV records, then replace its whole value.

    The cell holds ``<code>hash</code>`` when every FOV's record holds one hash, and exactly the short text when
    they differ. The returned value is the hash or the short text; the page keeps the placeholder in either case."""
    sample = Path(path).parent.name
    hashes = {json.loads(tree[f"fovs/{fov}/run.json"])["workflow_config"]["sha256"]
              for fov in run.data["membership"]["samples"][sample]}
    value = hashes.pop() if len(hashes) == 1 else REPORT_HASH_SEVERAL
    markup = REPORT_HASH_SEVERAL if value == REPORT_HASH_SEVERAL else f"<code>{value}</code>"
    blocks = list(re.finditer(r'(<div\b[^>]*\sid="b-files-configuration"[^>]*>)(.*?)(</table>)', text, re.S))
    assert len(blocks) == 1, (path, "expected one Configuration table", len(blocks))
    block = blocks[0]
    cells = list(re.finditer(rf'(<tr><td>{re.escape(REPORT_HASH_LABEL)}</td><td>)(.*?)(</td></tr>)', block[2], re.S))
    assert len(cells) == 1, (path, "expected one Hash cell", len(cells))
    cell = cells[0]
    assert cell[2] == markup, (path, "Hash cell differs from the FOVs' workflow_config.sha256", cell[2], markup)
    start, end = block.start(2) + cell.start(2), block.start(2) + cell.end(2)
    return value, text[:start] + "<workflow-config-sha256>" + text[end:]


def compare(run, left, right, *, name="comparison.json", enforce=True, left_engine=None, right_engine=None, rerun_tasks=None):
    """Compare declared bytes outside X3's named fields and X11's scoped global hashes.

    Tokens obey their engine's contract before removal. Only Snakemake's marker
    directly inside a declared directory is omitted. Cached records retain both
    global hashes when compared with the pre-rerun tree, checked by invoke.
    X11's report Hash cell is validated against each tree's FOV records before
    normalization, only if no FOV reran. Every other page value stays exact.
    Every binary file remains byte-equal, including H5AD and TIFF files.
    """
    original_left, original_right = left, right
    left_engine = left_engine or getattr(run, "engine", "snakemake")
    right_engine = right_engine or getattr(run, "engine", "snakemake")
    tasks = getattr(run, "data", {}).get("tasks", [])
    def owner(path):
        return next((t["id"] for t in tasks if path == t["record"] or
                     any(path == o or (o.endswith("/") and path.startswith(o)) for o in t["outputs"])), None)
    left = {p: v for p, v in left.items() if not is_marker(run, p, left_engine)}
    right = {p: v for p, v in right.items() if not is_marker(run, p, right_engine)}
    mask_report_hash = report_hash_exception(run, rerun_tasks) if rerun_tasks is not None else False
    if Path(name).name.startswith(("x7-", "x10-")) and hasattr(left_snapshot := original_left, "file_names"):
        evidence(run.work, Path(name).stem + "-file-names.json",
                 file_name_differences(run, left_snapshot.file_names, original_right.file_names))
    for tree, engine in ((left, left_engine), (right, right_engine)):
        for path, payload in tree.items():
            if path.endswith(".json"):
                workflow = json.loads(payload).get("workflow")
                if workflow is not None:
                    assert "token" in workflow, (path, "missing workflow.token")
                    token = workflow["token"]
                    assert token is None if engine == "snakemake" else isinstance(token, str) and bool(token), (path, engine, token)
    golden = X.golden()
    varying = (*golden.VARYING_EXPORT, "workflow/tasks/", "workflow/logs/")
    def normalize(value, path=""):
        omit_hashes = rerun_tasks is not None and owner(path) is not None and owner(path) not in rerun_tasks
        def context(node):
            if isinstance(node, dict):
                result = {k: context(v) for k, v in node.items()
                          if k not in {"execution_context", *golden.VARYING_COLUMNS}}
                if isinstance(result.get("workflow"), dict):
                    result["workflow"].pop("token", None)
                    if omit_hashes:
                        for key in ("configuration_sha256", "manifest_sha256"):
                            result["workflow"].pop(key, None)
                return result
            if isinstance(node, list):
                return [context(v) for v in node]
            return node
        return golden._normalized(context(value), str(run.work), varying, golden.VARYING_EXPORT_DIGESTS)

    def canonical(path, payload, tree):
        if path.endswith(".json"):
            return json.dumps(normalize(json.loads(payload), path), sort_keys=True, separators=(",", ":")).encode()
        if path.endswith(".csv"):
            rows = list(csv.reader(io.StringIO(payload.decode())))
            if not rows:
                return payload
            if not set(rows[0]) & golden.VARYING_COLUMNS:
                return payload.replace(str(run.work).encode(), b"<work>")
            indices = [i for i, key in enumerate(rows[0]) if key not in golden.VARYING_COLUMNS]
            return json.dumps([[row[i].replace(str(run.work), "<work>") for i in indices] for row in rows]).encode()
        if path.endswith(".html"):
            text = payload.decode()
            if mask_report_hash and Path(path).name == "report.html":
                _, text = report_hash_cell(run, path, text, tree)
            text = text.replace(str(run.work), "<work>")
            # The two spans contain only generated_at and run start/end/duration.
            text = re.sub(r'<span id="(?:generated-at|run-span)"[^>]*>.*?</span>',
                          '<span>time</span>', text, flags=re.S)
            # This labelled plot contains only measured seconds, already removed
            # from tables/steps.csv; every other embedded scientific image stays.
            text = re.sub(r'(<div class="blk" id="b-time"[^>]*>.*?<img[^>]*src=")data:image/png;base64,[^"]*',
                          r'\1<measured-seconds>', text, flags=re.S)
            def script(match):
                return match[1] + json.dumps(normalize(json.loads(match[2])), sort_keys=True) + match[3]
            text = re.sub(r'(<script[^>]*type="application/json"[^>]*>)(.*?)(</script>)', script, text, flags=re.S)
            text = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\+00:00|Z)', '<timestamp>', text)
            return text.encode()
        return payload
    rows = []
    for path in sorted(left.keys() | right.keys()):
        a, b = left.get(path), right.get(path)
        row = {"path": path, "left_sha256": hashlib.sha256(a).hexdigest() if a is not None else None,
               "right_sha256": hashlib.sha256(b).hexdigest() if b is not None else None}
        row["equal"] = a is not None and b is not None and canonical(path, a, left) == canonical(path, b, right)
        row["left_normalized_sha256"] = hashlib.sha256(canonical(path, a, left)).hexdigest() if a is not None else None
        row["right_normalized_sha256"] = hashlib.sha256(canonical(path, b, right)).hexdigest() if b is not None else None
        if not row["equal"] and path.endswith((".h5ad", ".tif")):
            prefix = Path(name).stem + "-files"
            row["binary_evidence"] = {
                side: str(evidence(run.work, f"{prefix}/{side}/{path}", payload))
                for side, payload in (("left", a), ("right", b)) if payload is not None}
        if not row["equal"] and a is not None and b is not None and path.endswith(".json"):
            def different(x, y, location=""):
                if type(x) is not type(y):
                    return [{"field": location, "left": x, "right": y}]
                if isinstance(x, dict):
                    return [d for k in sorted(x.keys() | y.keys())
                            for d in different(x.get(k), y.get(k), location + "." + k)]
                if isinstance(x, list):
                    if len(x) != len(y):
                        return [{"field": location + ".length", "left": len(x), "right": len(y)}]
                    return [d for i, (u, v) in enumerate(zip(x, y))
                            for d in different(u, v, location + f"[{i}]")]
                return [] if x == y else [{"field": location, "left": x, "right": y}]
            row["differences"] = different(json.loads(canonical(path, a, left)), json.loads(canonical(path, b, right)))[:40]
        rows.append(row)
    evidence(run.work, name, {"exceptions": sorted(golden.NORMALIZED | {"workflow.execution_context", "workflow.token"}),
                             "report_hash_cell_exception": mask_report_hash,
                             "global_hash_exceptions_tasks": sorted({t["id"] for t in tasks} - set(rerun_tasks)) if rerun_tasks is not None else [],
                             "rows": rows, "passed": all(row["equal"] for row in rows)})
    differences = [row["path"] for row in rows if not row["equal"]]
    if enforce:
        assert not differences, differences
    return differences


def fresh_comparison(run, expected, *, name, rerun_tasks):
    """Recompute from scratch at the same output path, then restore the tested engine tree.

    This keeps source and output roots identical, so paths or fingerprint hashes
    cannot mask a numerical difference. The fresh run uses a new engine cache,
    including Snakemake's directory timestamps, and the tested cache is retained.
    """
    backup = run.out.with_name(run.out.name + "-saved")
    shutil.move(run.out, backup)
    try:
        data = plan(run.path)
        run.engine_work = run.work / f"fresh-{run.invocations:02d}"
        run.engine_work.mkdir()
        run.invoke({t["id"] for t in data["tasks"]})
        if "export" in run.work.name:
            from starfinder.export._record import PARTS, listed_files
            def unlisted(root):
                folder = root / "export" / X.SAMPLE
                record = json.loads((folder / "export.json").read_text())
                listed = {"export.json", *(p for part in PARTS for p in listed_files(record, part))}
                return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*")
                              if p.is_file() and p.relative_to(folder).as_posix() not in listed)
            evidence(run.work, "x11-export-unlisted.json", {"rerun": unlisted(backup), "fresh": unlisted(run.out)})
        fresh = snapshot(run)
        if report_hash_exception(run, rerun_tasks):
            resolved_sha256 = "sha256:" + hashlib.sha256((run.out / "workflow/resolved.json").read_bytes()).hexdigest()
            for path, payload in fresh.items():
                if Path(path).name == "report.html":
                    value, _ = report_hash_cell(run, path, payload.decode(), fresh)
                    assert value == resolved_sha256, (path, "fresh Hash cell differs from resolved.json", value, resolved_sha256)
        one_sided = file_name_differences(run, tree_names(backup), tree_names(run.out))
        evidence(run.work, Path(name).stem + "-file-names.json", one_sided)
        differences = compare(run, expected, fresh, name=name, enforce=False, rerun_tasks=rerun_tasks)
        # Declared paths are already compared by contents; the rest are checked by name only.
        differences += [row["path"] for row in one_sided if row["path"] not in expected.keys() | fresh.keys()]
        failures = sorted(set(differences))
        evidence(run.work, Path(name).stem + "-result.json", {"file_names": one_sided,
                 "failures": failures, "passed": not failures})
        return failures
    finally:
        del run.engine_work
        shutil.rmtree(run.out)
        shutil.move(backup, run.out)


def assert_unchanged(before, after, task_id):
    """Untouched records remain byte-equal, including tokens and global hashes."""
    assert before == after, task_id
