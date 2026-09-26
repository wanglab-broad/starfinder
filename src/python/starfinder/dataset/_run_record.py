"""Small per-FOV run record (run.json) written while checkpoints are active."""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import logging
from pathlib import Path
import platform
import subprocess
import traceback

from starfinder.io._checkpoint import write_json

logger = logging.getLogger("starfinder")

FORMAT_VERSION = 1
_PACKAGES = ("starfinder", "numpy", "pandas", "scipy", "scikit-image", "tifffile", "pyarrow", "SimpleITK")


def _now():
    return datetime.now(timezone.utc).isoformat()


@lru_cache(maxsize=1)
def _code():
    from importlib.metadata import PackageNotFoundError, version
    try:
        package_version = version("starfinder")
    except PackageNotFoundError:
        package_version = None
    root = Path(__file__).resolve().parents[1]
    try:
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip()
        # An installed package can sit inside another project's repository; only
        # a starfinder checkout (src/python/starfinder at its root) is recorded.
        if not (Path(top) / "src" / "python" / "starfinder").resolve() == root:
            raise subprocess.SubprocessError("package is not inside a starfinder checkout")
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=10, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, timeout=30, check=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit = dirty = None
    return {"version": package_version, "git_commit": commit or None, "git_dirty": dirty}


@lru_cache(maxsize=1)
def _environment():
    from importlib.metadata import PackageNotFoundError, version
    packages = {}
    for name in _PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def _sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


class _RunRecord:
    """Mutable run.json content; rewritten atomically at step boundaries."""

    def __init__(self, fov, directory, checkpoints, config, execution):
        self.fov = fov
        self.directory = Path(directory)
        self.checkpoints = checkpoints
        self.failure = None
        dataset = fov.dataset
        self.data = {
            "format_version": FORMAT_VERSION, "dataset_id": dataset.dataset_id,
            "sample_id": dataset.sample_id, "fov_id": fov.fov_id, "subtile_id": fov.subtile_id,
            "status": "running", "started_at": _now(), "ended_at": None, "error": None,
            "code": _code(), "environment": _environment(),
            "config": {"pipeline": config, "execution": execution, "checkpoints": checkpoints},
            "inputs": [], "steps": [], "registration": {}, "counts": {},
            "checkpoint_directory": str(self.directory), "checkpoints": {},
        }

    @property
    def path(self):
        return self.directory / "run.json"

    def add_inputs(self, paths):
        """Record loaded source files, hashing them unless hash_inputs is False."""
        for path in paths:
            self.data["inputs"].append({"path": str(path),
                "sha256": _sha256(path) if self.checkpoints.hash_inputs else None})

    def add_checkpoint(self, stage, files):
        self.data["checkpoints"].setdefault(stage, []).extend(files)

    def add_step(self, name, round_name, seconds, status):
        self.data["steps"].append({"name": name, "round": round_name, "seconds": seconds, "status": status})

    def fail_step(self, name, round_name):
        # The innermost failing step is reported; enclosing steps keep it.
        if self.failure is None:
            self.failure = (name, round_name)

    def write(self):
        fov = self.fov
        counts = {}
        if fov.spot_result is not None:
            counts["spots"] = len(fov.spot_result.spots)
        if fov.intensity_result is not None:
            counts["intensities"] = len(fov.intensity_result.spot_ids)
        if fov.decoding_result is not None:
            counts["call_status"] = {str(k): int(v) for k, v in fov.decoding_result.table.call_status.value_counts().items()}
        if fov.filtering_result is not None:
            counts["filtering"] = fov.filtering_result.counts
        self.data.update(registration=fov.registration_attempts, counts=counts)
        write_json(self.data, self.path)

    def finish(self, status, error=None, round_name=None):
        """Record the final status; a failed write never masks ``error``."""
        self.data.update(status=status, ended_at=_now())
        if error is not None:
            step, failed_round = self.failure or ("run", round_name)
            self.data["error"] = {"step": step, "round": failed_round, "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__))}
            try:
                self.write()
            except Exception:
                logger.exception(f"[{self.fov.fov_id}] Could not write {self.path}")
        else:
            try:
                self.write()
            except BaseException:
                # The success was never recorded on disk; let the caller record the failure.
                self.data.update(status="running", ended_at=None)
                raise
