"""The fixture ``tiny_sample`` of the workflow contract and its plumbing (design/validation/orchestration.md, "Fixtures").

``tiny_sample`` is W-642's fixture (``scripts/workflow_golden.py``, ``write_fixture``: ``sample-tiny-v1``, seed 42,
one sample of 2 × 2 FOVs of 5 × 36 × 36, four rounds of four channels, the reference stain ``ch04`` and the
additional round ``flamingo``). ``write_supplied`` writes the supplied files the worked-example variants name
(``workflow_variants.py``: the BigStitcher result of ``supplied-xml`` and the import file of ``after-fusion``); their
content is never read by the plan. ``write_config`` writes a configuration, ``fix_code`` fixes the package's code
identity and ``run_task`` runs one task of a manifest in this process.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATASET_ID = "sample-tiny-v1"
SAMPLE = "sample"
FOVS = ("FOV_001", "FOV_002", "FOV_003", "FOV_004")


def golden():
    """scripts/workflow_golden.py as a module (its fixture writer and comparison)."""
    spec = importlib.util.spec_from_file_location("workflow_golden", ROOT / "src" / "python" / "scripts" /
                                                  "workflow_golden.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def input_root(work):
    return Path(work) / "input" / DATASET_ID / SAMPLE


def output_root(work, output_id):
    return Path(work) / "output" / DATASET_ID / output_id


def write_supplied(work):
    """The supplied files the variants name: supplied-xml's dataset.xml and view map, after-fusion's import file."""
    work = Path(work)
    folder = work / "supplied" / SAMPLE
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "dataset.xml").write_text("<SpimData/>\n")
    (folder / "view_map.csv").write_text("view,grid_column,grid_row,fov_id,kind\n")
    (work / "labels" / "fused-cells.ome.tif").write_bytes(b"supplied")


def write_config(work, name, config):
    import yaml
    path = Path(work) / "configs" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


#: A clean code identity of the package, for the tests of verified resume (a dirty tree refuses a resume).
CLEAN_CODE = {"version": "0.2.0", "version_source": "checkout", "git_commit": "0123456789abcdef0123456789abcdef01234567", "git_dirty": False}


def fix_code(monkeypatch, **code):
    """Make ``starfinder._records.code`` (the run records' and the plan's code identity) return ``code`` (by default
    CLEAN_CODE), its cache cleared, as test_workflow_orchestration_golden.py patches it."""
    import starfinder._records as records
    if hasattr(records.code, "cache_clear"):
        records.code.cache_clear()
    identity = dict(CLEAN_CODE, **code)
    monkeypatch.setattr(records, "code", lambda: dict(identity))


def run_task(manifest, task, *extra):
    """Run one task's command in this process (``starfinder workflow <command> --manifest M --task T``)."""
    from starfinder.__main__ import main
    from starfinder.workflow._shape import COMMANDS
    kind = next(t["kind"] for t in json.loads(Path(manifest).read_text())["tasks"] if t["id"] == task)
    return main(["workflow", COMMANDS[kind], "--manifest", str(manifest), "--task", task, *extra])
