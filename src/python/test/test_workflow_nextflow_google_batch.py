"""Google Batch profile resolution without credentials, a plugin download or a cloud call."""
from __future__ import annotations

import os
import subprocess

import pytest

from . import workflow_examples as X
from . import workflow_nextflow as N

pytestmark = [pytest.mark.workflow, pytest.mark.contract, pytest.mark.nextflow, pytest.mark.slow]


@pytest.mark.parametrize("profile, executor", [
    ("google-batch", "google-batch"),
    ("google-batch,local", "local"),
])
def test_google_batch_profile_resolves_offline(tmp_path, profile, executor):
    done = subprocess.run(
        [N.executable(), "config", "-flat", "-profile", profile, str(X.ROOT / "nextflow/main.nf")],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
        env=dict(os.environ, NXF_OFFLINE="true", NXF_PLUGINS_DEFAULT="false"),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    values = {key.replace("'", ""): value for key, value in (
        line.split(" = ", 1) for line in done.stdout.splitlines() if " = " in line)}
    assert values["process.executor"] == repr(executor)
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
