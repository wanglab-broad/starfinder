"""Run a command in its own process and measure it from the child's resource usage (W-920).

``run_measured`` does what GNU ``time -v`` does, without the ``/usr/bin/time`` binary, which the learned
SIF and RHEL 8 nodes lack: a small launcher process (Python without ``site``, standard library only)
spawns the command, reaps it with ``os.wait4`` and reports the child's rusage: wall clock, user and
system time, and the maximum resident set size, in KiB on Linux, of the child or of a descendant it
waited for.

The launcher is needed because a child's ``ru_maxrss`` includes the memory it shared with its parent
between the fork and the exec: the kernel keeps the high-water mark of the address space that the exec
replaces. Spawned from the test process directly, every child would report at least the test
process's own RSS (a trivial child of a 500 MiB process reported 527 MB on GP099-29C); spawned from
the launcher, it reports at least the launcher's few MiB, as under GNU time.

The tests that measure a step's cost in its own process use it: GN24, GA24, the cellular chain's
steps, the small sample preset and the medium generation check.
"""
from __future__ import annotations

from dataclasses import dataclass
import errno
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

# Polling interval while a timeout is set; without one the parent blocks in wait4.
POLL_S = 0.01

# argv: the result file, the program's path, then the command (its argv[0] and arguments). Writes one line: the raw wait status, wall
# seconds, user and system seconds, and ru_maxrss.
_LAUNCHER = """\
import os, sys, time
start = time.monotonic()
pid = os.posix_spawn(sys.argv[2], sys.argv[3:], os.environ)
_, status, usage = os.wait4(pid, 0)
wall = time.monotonic() - start
with open(sys.argv[1], "w") as out:
    out.write(f"{status} {wall!r} {usage.ru_utime!r} {usage.ru_stime!r} {usage.ru_maxrss}\\n")
"""


@dataclass(frozen=True)
class Measured:
    """A finished child process and its resource usage.

    Parameters
    ----------
    returncode : int
        The exit status; negative when a signal ended the child, as in ``subprocess``.
    stdout, stderr : str
        The child's output, decoded as UTF-8 with replacement.
    wall_s : float
        Seconds from the spawn of the child to its reaping.
    user_s, system_s : float
        User and system CPU seconds of the child and the descendants it waited for.
    peak_rss_kib : int
        Maximum resident set size of the child or of one such descendant, in KiB.
    """

    returncode: int
    stdout: str
    stderr: str
    wall_s: float
    user_s: float
    system_s: float
    peak_rss_kib: int

    @property
    def cpu_s(self):
        """User plus system CPU seconds."""
        return self.user_s + self.system_s

    def usage(self):
        """The measurement without the output, as a dictionary (written beside a step's output)."""
        return dict(returncode=self.returncode, wall_s=self.wall_s, user_s=self.user_s, system_s=self.system_s,
                    peak_rss_kib=self.peak_rss_kib)


def _reap(pid, deadline):
    """Wait for ``pid``; return its wait status, or raise TimeoutError once ``deadline`` (monotonic) passes."""
    if deadline is None:
        return os.wait4(pid, 0)[1]
    while True:
        found, status, _ = os.wait4(pid, os.WNOHANG)
        if found:
            return status
        if time.monotonic() >= deadline:
            raise TimeoutError
        time.sleep(POLL_S)


def run_measured(command, *, env=None, cwd=None, timeout=None):
    """Run ``command`` in its own process, capture its output, and measure it from its rusage.

    Parameters
    ----------
    command : sequence of str
        The program and its arguments; the program is looked up on the ``PATH`` of ``env``.
    env : mapping, optional
        The child's environment; the parent's when omitted.
    cwd : path-like, optional
        The child's working directory.
    timeout : float, optional
        Seconds after which the child and its launcher are killed (their process group) and
        ``subprocess.TimeoutExpired`` is raised.

    Returns
    -------
    Measured
        The exit status, the output and the resource usage.

    Raises
    ------
    FileNotFoundError
        When the program is not found.
    """
    command = [str(part) for part in command]
    environment = dict(os.environ if env is None else env)
    # A name without a slash is looked up on the child's PATH, here, so that a missing program raises as in
    # ``subprocess``; the path is not resolved (a virtual environment's ``python`` is a symbolic link).
    program = command[0] if os.sep in command[0] else shutil.which(command[0], path=environment.get("PATH"))
    if program is None:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), command[0])
    with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        result = Path(folder) / "usage"
        launcher = subprocess.Popen([sys.executable, "-I", "-S", "-c", _LAUNCHER, str(result), program, *command],
                                    stdout=out, stderr=err, env=environment, cwd=cwd, start_new_session=True)
        start = time.monotonic()
        try:
            _reap(launcher.pid, None if timeout is None else start + timeout)
        except BaseException as error:
            os.killpg(launcher.pid, signal.SIGKILL)
            os.wait4(launcher.pid, 0)
            launcher.returncode = -signal.SIGKILL
            if isinstance(error, TimeoutError):
                raise subprocess.TimeoutExpired(command, timeout) from None
            raise
        launcher.returncode = 0
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read().decode("utf-8", "replace"), err.read().decode("utf-8", "replace")
        if not result.is_file():
            raise RuntimeError(f"the measuring launcher wrote no usage for {command}:\n{stderr[-2000:]}")
        status, wall, user, system, rss = result.read_text().split()
        return Measured(returncode=os.waitstatus_to_exitcode(int(status)), stdout=stdout, stderr=stderr,
                        wall_s=float(wall), user_s=float(user), system_s=float(system), peak_rss_kib=int(rss))
