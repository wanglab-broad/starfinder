"""``run_measured`` (test/measured_process.py, W-920): a child's exit status, output and rusage without /usr/bin/time."""
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from .measured_process import run_measured

pytestmark = [pytest.mark.core, pytest.mark.contract]

MIB_KIB = 1024
# The interpreter's own pages in a child that only allocates (about 10 MiB on GP099-29C), with margin.
INTERPRETER_KIB = 64 * MIB_KIB


def python(source, *arguments):
    return [sys.executable, "-c", source, *arguments]


def fill(mib):
    return python(f"block = b'\\x01' * ({mib} * 1024 * 1024)")


def test_peak_rss_counts_a_known_allocation():
    """A child that fills 128 MiB peaks at 128 MiB plus the interpreter's own pages."""
    done = run_measured(fill(128))
    assert done.returncode == 0, done.stderr
    assert 128 * MIB_KIB <= done.peak_rss_kib <= 128 * MIB_KIB + INTERPRETER_KIB, done.peak_rss_kib


def test_a_large_parent_does_not_raise_the_childs_peak():
    """The child's peak excludes what it shared with this process before its exec: with 256 MiB held here, a child
    that allocates nothing still peaks at its interpreter's pages (spawned from this process directly, it reported
    at least this process's RSS)."""
    ballast = b"\x01" * (256 * 1024 * 1024)
    done = run_measured(python("pass"))
    assert len(ballast) and done.returncode == 0, done.stderr
    assert done.peak_rss_kib <= INTERPRETER_KIB, done.peak_rss_kib


def test_exit_status_and_output_are_returned():
    done = run_measured(python("import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"))
    assert (done.returncode, done.stdout, done.stderr) == (3, "out\n", "err\n")
    assert done.usage() == dict(returncode=3, wall_s=done.wall_s, user_s=done.user_s, system_s=done.system_s,
                                peak_rss_kib=done.peak_rss_kib)


def test_a_signal_gives_a_negative_status():
    done = run_measured(python("import os, signal; os.kill(os.getpid(), signal.SIGTERM)"))
    assert done.returncode == -signal.SIGTERM


def test_a_missing_program_raises():
    with pytest.raises(FileNotFoundError):
        run_measured(["starfinder-no-such-program-w920"])


def test_wall_and_cpu_seconds():
    """0.3 s asleep, then 0.3 s of CPU: the wall time covers both, the CPU time the second (tolerance 0.1 s below)."""
    done = run_measured(python("import time\ntime.sleep(0.3)\nstart = time.process_time()\n"
                               "while time.process_time() - start < 0.3:\n    pass"))
    assert done.returncode == 0, done.stderr
    assert done.wall_s >= 0.6 and done.cpu_s >= 0.2, done.usage()
    assert done.cpu_s == done.user_s + done.system_s


def running(pid):
    """Whether ``pid`` is alive: its /proc entry exists and is not a zombie awaiting its reaper."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def test_timeout_kills_the_child(tmp_path):
    marker = tmp_path / "pid"
    source = "import os, sys, time\nopen(sys.argv[1], 'w').write(str(os.getpid()))\ntime.sleep(60)"
    with pytest.raises(subprocess.TimeoutExpired):
        run_measured(python(source, str(marker)), timeout=2)
    pid = int(marker.read_text())
    deadline = time.monotonic() + 5
    while running(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not running(pid), pid
