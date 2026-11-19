"""W-814 and W-875: real engines on tiny_sample when the placement needs review.

The molecules-only variant with review bounds of zero (``stitching.review``), so that criterion V1 marks the
multiview placement ``required`` and flags FOVs (under X2 the supplied-xml variant's placement is ``not_needed``).
Each engine runs twice: held (the export exits 4 after writing the held page, no report is made, the run fails and
``verify --final`` fails), then after ``approve_placement`` and ``save_placement``, which write ``approvals.json``
alone (the export, the report and the comparison run, the placement task does not, and ``workflow verify --final``
exits 0). Over both invocations the stitch task runs once and the report task runs once.
"""
import json
import subprocess
import sys

import pytest

from starfinder import _layout

from . import workflow_engine_validation as E
from . import workflow_examples as X

pytestmark = [pytest.mark.workflow, pytest.mark.integration, pytest.mark.extended, pytest.mark.slow]

REVIEW = {"max_correction_fraction": 0.0, "max_residual_um": 0.0}
PLACEMENT_FILES = ("placement.json", "fovs.csv", "links.csv")


def _ids(run, *kinds):
    return {t["id"]: t for t in run.data["tasks"] if t["kind"] in kinds}


def _held_run(tmp_path, engine):
    run = E.EngineRun(tmp_path / f"review-{engine}", engine, "molecules-only")
    run.config["stitching"]["review"] = dict(REVIEW)
    run.replan(run.config)
    return run


def _placement_files(run):
    """name → (bytes, inode, mtime_ns) of the stitch task's declared outputs."""
    folder = run.out / "stitching" / X.SAMPLE
    return {n: ((folder / n).read_bytes(), (folder / n).stat().st_ino, (folder / n).stat().st_mtime_ns)
            for n in PLACEMENT_FILES}


def _check_held(run, ran):
    """The held run: the stitch task ran, the export's failed record says held, the held page shows the review and
    the flagged FOVs, there is no export.json, no report and no report record, and neither verify --final nor a
    record says the run succeeded."""
    from starfinder.workflow import verify
    (stitch_id,), (export_id,), (report_id,) = _ids(run, "stitch"), _ids(run, "export"), _ids(run, "report")
    assert stitch_id in ran and report_id not in ran, ran
    failed = json.loads(_layout.failed_record(run.out, export_id).read_text())
    assert failed["status"] == "held" and failed["error"]["type"] == "TaskHeld", failed
    assert not _layout.task_record(run.out, export_id).exists()
    assert not (run.out / "export" / X.SAMPLE / "molecules_only" / "export.json").exists()
    review = json.loads((run.out / "stitching" / X.SAMPLE / "placement.json").read_text())["review"]
    assert review["state"] == "required" and review["flagged"] and review["approvals"] == [], review
    assert not (run.out / "stitching" / X.SAMPLE / "approvals.json").exists()
    page = _layout.workflow_held_page(run.out, X.SAMPLE, "molecules_only").read_text(encoding="utf-8")
    flagged = "<br>".join(f"{reason}: {', '.join(fovs)}" for reason, fovs in review["flagged"].items())
    assert f"<tr><td>Reasons</td><td>{', '.join(review['reasons'])}</td></tr>" in page
    assert f"<tr><td>Flagged FOVs</td><td>{flagged}</td></tr>" in page
    assert not (run.out / "report" / X.SAMPLE).exists()
    assert not _layout.task_record(run.out, report_id).exists()
    result = verify(run.manifest, hash=True, final=True)
    assert not result["passed"] and export_id in result["incomplete"]
    assert result["tasks"][stitch_id]["complete"], result["tasks"][stitch_id]
    assert result["tasks"][export_id]["problems"] == [
        "held for review (a failed record that says held) and no task record"]
    E.evidence(run.work, "held.json", {"failed": failed, "review": review, "verify": result})
    E.evidence(run.work, "held-page.html", page.encode("utf-8"))
    return review


def _approve(run):
    """The guide's route: load_placement, inspect the flags and links, approve_placement and save_placement, which
    writes approvals.json and changes none of the placement task's outputs."""
    from starfinder.stitching import approve_placement, load_placement, save_placement
    from starfinder.workflow import verify
    folder = run.out / "stitching" / X.SAMPLE
    before = _placement_files(run)
    result = load_placement(folder)
    assert result.record["review"]["state"] == "required" and len(result.links) > 0
    assert any(result.fovs["flags"] != "")
    save_placement(approve_placement(result, reviewer="W-814 test", note="the synthetic placement is as expected"),
                   folder)
    assert _placement_files(run) == before
    (stitch_id,), (export_id,) = _ids(run, "stitch"), _ids(run, "export")
    result = verify(run.manifest, hash=True)
    assert result["tasks"][stitch_id]["complete"] and not result["tasks"][export_id]["complete"], result
    E.evidence(run.work, "approved-verify.json", result)


def _check_final(run, review, held, ran):
    """After the rerun: the stitch task did not run again and its record is unchanged, the export, the report and
    the comparison ran once, the approval is in approvals.json, and `workflow verify --final` exits 0."""
    (stitch_id,), (report_id,) = _ids(run, "stitch"), _ids(run, "report")
    assert stitch_id not in ran and report_id in ran, ran
    assert _layout.task_record(run.out, stitch_id).read_bytes() == held
    placement = json.loads((run.out / "stitching" / X.SAMPLE / "placement.json").read_text())
    assert placement["review"]["approvals"] == [] and placement["review"]["flagged"] == review["flagged"]
    approvals = json.loads((run.out / "stitching" / X.SAMPLE / "approvals.json").read_text())["approvals"]
    assert [a["reviewer"] for a in approvals] == ["W-814 test"]
    assert approvals[0]["fovs_sha256"] == placement["tables"]["fovs"]["sha256"]
    assert (run.out / "export" / X.SAMPLE / "molecules_only" / "export.json").is_file()
    assert not _layout.workflow_held_page(run.out, X.SAMPLE, "molecules_only").exists()
    page = (run.out / "report" / X.SAMPLE / "molecules_only" / "report.html").read_text(encoding="utf-8")
    assert 'id="place-review"' not in page
    command = [sys.executable, "-m", "starfinder", "workflow", "verify", "--manifest", str(run.manifest), "--hash",
               "--final"]
    done = subprocess.run(command, capture_output=True, text=True, timeout=60)
    E.evidence(run.work, "verify-final-cli.json", {"command": command, "returncode": done.returncode,
                                                "stdout": done.stdout, "stderr": done.stderr})
    assert done.returncode == 0, done.stdout + done.stderr
    E.evidence(run.work, "runs.json", {"stitch": [stitch_id in a for a in (run.held_ran, ran)].count(True),
                                       "report": [report_id in a for a in (run.held_ran, ran)].count(True),
                                       "held_invocation": sorted(run.held_ran), "approved_invocation": sorted(ran)})


def _review_route(tmp_path, engine):
    run = _held_run(tmp_path, engine)
    run.held_ran = run.invoke(held=True)
    review = _check_held(run, run.held_ran)
    (stitch_id,) = _ids(run, "stitch")
    held = _layout.task_record(run.out, stitch_id).read_bytes()
    _approve(run)
    # Neither engine reruns the placement task: its outputs and record are unchanged (approvals.json is no declared
    # output). The export, the report and the comparison run.
    ran = run.invoke({*_ids(run, "export", "report", "compare")})
    _check_final(run, review, held, ran)


def test_held_export_and_approved_rerun_snakemake(tmp_path):
    _review_route(tmp_path, "snakemake")


@pytest.mark.nextflow
def test_held_export_and_approved_rerun_nextflow(tmp_path):
    _review_route(tmp_path, "nextflow")
