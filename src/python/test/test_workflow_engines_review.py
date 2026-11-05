"""W-814: real engines on tiny_sample when the placement needs review.

The molecules-only variant with review bounds of zero (``stitching.review``), so that criterion V1 marks the
multiview placement ``required`` and flags FOVs (under X2 the supplied-xml variant's placement is ``not_needed``).
Each engine runs twice: held (the export exits 4 after making the sample report, the run fails and ``verify
--final`` fails), then after ``approve_placement`` and ``save_placement`` (the export completes and ``workflow verify
--final`` exits 0).
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


def _ids(run, *kinds):
    return {t["id"]: t for t in run.data["tasks"] if t["kind"] in kinds}


def _held_run(tmp_path, engine):
    run = E.EngineRun(tmp_path / f"review-{engine}", engine, "molecules-only")
    run.config["stitching"]["review"] = dict(REVIEW)
    run.replan(run.config)
    return run


def _check_held(run):
    """The held run: the export's failed record says held, the report shows the review and the flagged FOVs, there is
    no export.json, and neither verify --final nor a record says the run succeeded."""
    from starfinder.workflow import verify
    (export_id,), (report_id,) = _ids(run, "export"), _ids(run, "report")
    failed = json.loads(_layout.failed_record(run.out, export_id).read_text())
    assert failed["status"] == "held" and failed["error"]["type"] == "TaskHeld", failed
    assert not _layout.task_record(run.out, export_id).exists()
    assert not (run.out / "export" / X.SAMPLE / "export.json").exists()
    review = json.loads((run.out / "stitching" / X.SAMPLE / "placement.json").read_text())["review"]
    assert review["state"] == "required" and review["flagged"] and review["approvals"] == [], review
    page = (run.out / "report" / X.SAMPLE / "report.html").read_text(encoding="utf-8")
    flagged = "<br>".join(f"{reason}: {', '.join(fovs)}" for reason, fovs in review["flagged"].items())
    assert f"<td>Reasons</td><td>{', '.join(review['reasons'])}</td>" in page
    assert f"<td>Flagged FOVs</td><td>{flagged}</td>" in page
    assert (run.out / "report" / X.SAMPLE / "report.json").is_file()
    assert not _layout.task_record(run.out, report_id).exists()
    result = verify(run.manifest, hash=True, final=True)
    assert not result["passed"] and export_id in result["incomplete"]
    assert result["tasks"][export_id]["problems"] == [
        "held for review (a failed record that says held) and no task record"]
    E.evidence(run.work, "held.json", {"failed": failed, "review": review, "verify": result})
    E.evidence(run.work, "held-report.html", page.encode("utf-8"))
    return review


def _approve(run):
    """The guide's route: load_placement, inspect the flags and links, approve_placement and save_placement."""
    from starfinder.stitching import approve_placement, load_placement, save_placement
    folder = run.out / "stitching" / X.SAMPLE
    result = load_placement(folder)
    assert result.record["review"]["state"] == "required" and len(result.links) > 0
    assert any(result.fovs["flags"] != "")
    save_placement(approve_placement(result, reviewer="W-814 test", note="the synthetic placement is as expected"),
                   folder)


def _check_final(run, review):
    """After the rerun: the approval was kept, the export completed and `workflow verify --final` exits 0."""
    placement = json.loads((run.out / "stitching" / X.SAMPLE / "placement.json").read_text())
    assert [a["reviewer"] for a in placement["review"]["approvals"]] == ["W-814 test"]
    assert placement["review"]["flagged"] == review["flagged"]
    assert (run.out / "export" / X.SAMPLE / "export.json").is_file()
    page = (run.out / "report" / X.SAMPLE / "report.html").read_text(encoding="utf-8")
    assert 'id="place-review"' not in page
    command = [sys.executable, "-m", "starfinder", "workflow", "verify", "--manifest", str(run.manifest), "--hash",
               "--final"]
    done = subprocess.run(command, capture_output=True, text=True, timeout=60)
    E.evidence(run.work, "verify-final-cli.json", {"command": command, "returncode": done.returncode,
                                                "stdout": done.stdout, "stderr": done.stderr})
    assert done.returncode == 0, done.stdout + done.stderr


def test_held_export_and_approved_rerun_snakemake(tmp_path):
    run = _held_run(tmp_path, "snakemake")
    run.invoke(held=True)
    review = _check_held(run)
    _approve(run)
    # Snakemake removes a job's outputs before it runs it again, approval included, and does not rerun a task whose
    # output changed: the placement task is run by its command, which keeps the approval and records the task again.
    (stitch_id, stitch), = _ids(run, "stitch").items()
    done = subprocess.run([sys.executable, "-m", "starfinder", *stitch["command"][1:]], capture_output=True,
                          text=True, timeout=600)
    E.evidence(run.work, "stitch-by-command.json", {"command": stitch["command"], "returncode": done.returncode,
                                                    "stdout": done.stdout, "stderr": done.stderr})
    assert done.returncode == 0, done.stderr
    run.manual = {stitch_id}
    run.invoke({*_ids(run, "export", "report", "compare")})
    _check_final(run, review)


@pytest.mark.nextflow
def test_held_export_and_approved_rerun_nextflow(tmp_path):
    run = _held_run(tmp_path, "nextflow")
    run.invoke(held=True)
    review = _check_held(run)
    _approve(run)
    # verify finds the placement task not complete (placement.json changed); -resume reruns it, keeping the approval.
    run.invoke({*_ids(run, "stitch", "export", "report", "compare")})
    _check_final(run, review)
