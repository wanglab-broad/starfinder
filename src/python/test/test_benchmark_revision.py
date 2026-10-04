"""Default-tier tests of the benchmark revision record, benchmarks/revision.py (W-288), on temporary Git repositories."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess

import pytest

pytestmark = pytest.mark.benchmark

ROOT = Path(__file__).resolve().parents[3]


def module():
    spec = importlib.util.spec_from_file_location("benchmark_revision", ROOT / "benchmarks" / "revision.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


rev = module()


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True).stdout


def _repository(path):
    """A repository with one commit, independent of the host's Git configuration."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-c", "init.defaultBranch=main", "init", "-q", str(path)], capture_output=True, check=True)
    for key, value in (("user.name", "test"), ("user.email", "test@example.org"), ("commit.gpgsign", "false")):
        _git(path, "config", key, value)
    (path / "tracked.txt").write_text("version 1\n")
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "first")
    return path


def _expected(repo, entries):
    """sha256 of git diff HEAD --binary, then each (name, contents) in sorted order as name, NUL, contents."""
    digest = hashlib.sha256(_git(repo, "diff", "HEAD", "--binary"))
    for name, contents in sorted(entries):
        digest.update(name.encode() + b"\0" + contents)
    return digest.hexdigest()


def test_regular_untracked_files_keep_the_digest_meaning(tmp_path):
    repo = _repository(tmp_path / "repo")
    clean = rev.revision(repo)
    assert clean["revision"] == _git(repo, "rev-parse", "HEAD").decode().strip()
    assert clean["dirty"] is False and clean["uncommitted_diff_sha256"] is None and clean["untracked_files"] == []
    (repo / "tracked.txt").write_text("version 2\n")
    (repo / "a file.txt").write_bytes(b"spaced\n")
    (repo / "notes").mkdir()
    (repo / "notes" / "b.bin").write_bytes(bytes(range(256)))
    record = rev.revision(repo)
    assert record["dirty"] is True
    assert record["untracked_files"] == ["a file.txt", "notes/b.bin"]
    assert record["uncommitted_diff_sha256"] == _expected(
        repo, [("a file.txt", b"spaced\n"), ("notes/b.bin", bytes(range(256)))])


def test_an_untracked_nested_repository_contributes_its_name_only(tmp_path):
    repo = _repository(tmp_path / "repo")
    nested = _repository(repo / "nested")
    (repo / "a file.txt").write_bytes(b"spaced\n")
    record = rev.revision(repo)
    assert _git(repo, "ls-files", "--others", "--exclude-standard").decode().split("\n").count("nested/") == 1
    assert record["dirty"] is True
    assert record["untracked_files"] == ["a file.txt", "nested/"]
    assert record["uncommitted_diff_sha256"] == _expected(repo, [("a file.txt", b"spaced\n"), ("nested/", b"")])
    (nested / "tracked.txt").write_text("changed inside the nested repository\n")
    (nested / "new.txt").write_text("new inside the nested repository\n")
    assert rev.revision(repo)["uncommitted_diff_sha256"] == record["uncommitted_diff_sha256"]
    assert "directory entry" in record["uncommitted_diff"] and "name and the NUL byte only" in record["uncommitted_diff"]
