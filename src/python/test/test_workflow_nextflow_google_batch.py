"""Google Batch profile resolution without credentials, a plugin download or a cloud call."""
from __future__ import annotations

import os
import subprocess

import pytest

from . import workflow_examples as X
from . import workflow_nextflow as N

pytestmark = [pytest.mark.workflow, pytest.mark.contract, pytest.mark.nextflow, pytest.mark.slow]


VERIFY = "process.withName:VERIFY_BEFORE|VERIFY_AFTER.executor"


def resolve(tmp_path, profile):
    done = subprocess.run(
        [N.executable(), "config", "-flat", "-profile", profile, str(X.ROOT / "nextflow/main.nf")],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
        env=dict(os.environ, NXF_OFFLINE="true", NXF_PLUGINS_DEFAULT="false"),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return {key.replace("'", ""): value for key, value in (
        line.split(" = ", 1) for line in done.stdout.splitlines() if " = " in line)}


@pytest.mark.parametrize("profile, executor", [("local", "local"), ("sge", "sge")])
def test_verify_processes_keep_the_local_executor_outside_google_batch(tmp_path, profile, executor):
    values = resolve(tmp_path, profile)
    assert values["process.executor"] == repr(executor)
    # The processes declare executor 'local' themselves; no profile but google-batch moves them (W-834).
    assert "google-batch" not in values.get(VERIFY, "")
    assert values.get(VERIFY, "'local'") == "'local'"


@pytest.mark.parametrize("profile, executor, verify", [
    ("google-batch", "google-batch", "google-batch"),
    ("google-batch,local", "local", "local"),
])
def test_google_batch_profile_resolves_offline(tmp_path, profile, executor, verify):
    values = resolve(tmp_path, profile)
    assert values["process.executor"] == repr(executor)
    # A local executor refuses a gs:// work directory and the Batch head has no starfinder, so under google-batch the
    # VERIFY processes run in the image on Batch; the local profile still brings everything back to the head (W-834).
    assert values[VERIFY] == repr(verify)
    assert values["google.project"] == "'<project>'"
    assert values["google.location"] == "'us-central1'"
    assert values["google.batch.spot"] == "true"
    assert values["workDir"] == "'gs://<bucket>/work/<run>'"
    assert values["process.container"] == (
        "'us-central1-docker.pkg.dev/<project>/<repository>/starfinder@sha256:<cpu-image-digest>'")
    assert values["process.maxRetries"] == "2"
    assert values["process.machineType"] == "'n2-standard-4'"
    assert values["process.withName:FOV|TILE_GLOBAL.machineType"] == "'n2-highmem-4'"
    assert values["process.withLabel:gpu.machineType"] == "'g2-standard-4'"
    assert values["process.withLabel:gpu.container"].endswith("sha256:<gpu-image-digest>'")
    assert "nvidia-l4" in values["process.withLabel:gpu.accelerator"]
