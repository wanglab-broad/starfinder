"""The X9 operator must supply the resource grant explicitly."""
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.workflow, pytest.mark.contract]
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/x9_local_registration.py"


@pytest.mark.parametrize("limit, message", [(None, "required: --rss-limit-gib"), ("0", "must be positive"), ("-1", "must be positive")])
def test_x9_requires_positive_explicit_rss_grant(tmp_path, limit, message):
    command = [sys.executable, str(SCRIPT), "--output", str(tmp_path / "result.json"),
               "--scratch", str(tmp_path / "scratch")]
    if limit is not None:
        command += ["--rss-limit-gib", limit]
    done = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert done.returncode == 2
    assert message in done.stderr
    assert not (tmp_path / "scratch").exists()
    assert not (tmp_path / "result.json").exists()
