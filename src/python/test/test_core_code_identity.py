"""The code identity of a run, ``starfinder._records.code()`` (W-827): one commit has one ``version`` and ``git_commit``
on every install route.

Each case runs ``code()`` in a fresh interpreter over a copy of ``_records.py`` in a temporary tree, so that the route
under test is the only one visible: a worktree first on ``PYTHONPATH`` with the stale metadata of another checkout, an
installed package with build information, and an installed package with neither. The tree is a throwaway Git
repository with ``src/python/starfinder/`` at its root, as a starfinder checkout has.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.core, pytest.mark.contract]

RECORDS = Path(__file__).resolve().parents[1] / "starfinder" / "_records.py"
CONTAINERFILE = Path(__file__).resolve().parents[3] / "Containerfile"
STALE = "0.1.0"
BUILD_VERSION = "0.1.dev763+g023125826"
BUILD_COMMIT = "023125826fe73c59ba4a50e613da7570e4bb1db0"
#: setuptools-scm stamps a tree with uncommitted changes with this date (SOURCE_DATE_EPOCH, 2026-11-10 UTC).
EPOCH = "1794312000"
DATE = "20261110"
# The version rule of pyproject.toml's [tool.hatch.version].
RULE = {"root": None, "tag_regex": r"^v(?P<version>[0-9.]+)$",
        "git_describe_command": "git describe --dirty --tags --long --match v*"}

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not on PATH")


def git(repo, *args):
    env = dict(os.environ, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null", GIT_AUTHOR_NAME="t",
               GIT_AUTHOR_EMAIL="t@example.org", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.org")
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
                          env=env).stdout.strip()


def commit(repo, name):
    (repo / "notes.txt").write_text(name)
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", name)


def package(root):
    """``root/src/python/starfinder`` with a copy of this tree's ``_records.py``."""
    folder = root / "src" / "python" / "starfinder"
    folder.mkdir(parents=True)
    (folder / "__init__.py").write_text("")
    shutil.copy(RECORDS, folder / "_records.py")
    return folder


def dist_info(site, version):
    folder = site / f"starfinder-{version}.dist-info"
    folder.mkdir(parents=True)
    (folder / "METADATA").write_text(f"Metadata-Version: 2.1\nName: starfinder\nVersion: {version}\n")
    return site


def repository(tmp_path, tags=(), commits=1):
    """A checkout with ``commits`` commits; ``tags`` maps a tag name to the number of the commit it marks."""
    repo = tmp_path / "worktree"
    package(repo)
    git(repo.parent, "init", "-q", str(repo))
    for number in range(1, commits + 1):
        commit(repo, f"commit {number}")
        for name, at in dict(tags).items():
            if at == number:
                git(repo, "tag", name)
    return repo


def run_code(src, *extra_path):
    """``code()`` of the copy under ``src``, with the stale metadata of another checkout further down the path."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(src), *map(str, extra_path)]), SOURCE_DATE_EPOCH=EPOCH,
               PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([sys.executable, "-c", "import json; from starfinder import _records; "
                           "print(json.dumps(_records.code()))"], capture_output=True, text=True, env=env,
                          cwd=src.parent, check=True)
    return json.loads(done.stdout)


@needs_git
@pytest.mark.parametrize("tags, commits, expected", [
    ({}, 3, "0.1.dev3+g{short}"),
    ({"v0.3.1": 1}, 3, "0.3.2.dev2+g{short}"),
    ({"v0.3.1": 3}, 3, "0.3.1"),
    ({"v0.3.1": 1, "archive/v9.9.9": 2}, 3, "0.3.2.dev2+g{short}"),
])
def test_a_worktree_first_on_the_path_is_identified_by_its_own_commit(tmp_path, tags, commits, expected):
    # The environment's editable install belongs to another checkout, so its metadata says 0.1.0.
    repo = repository(tmp_path, tags, commits)
    head = git(repo, "rev-parse", "HEAD")
    found = run_code(repo / "src" / "python", dist_info(tmp_path / "site", STALE))
    # setuptools-scm's node: the first nine digits of the commit when no tag is reachable, else `git describe`'s own
    # abbreviation (seven digits in a repository this small), cut to nine.
    short = git(repo, "rev-parse", "--short", "HEAD")[:9] if tags else head[:9]
    assert found == {"version": expected.format(short=short),
                     "version_source": "checkout", "git_commit": head, "git_dirty": False}
    assert found["version"] != STALE


@needs_git
def test_uncommitted_changes_add_the_dirty_suffix_of_the_rule(tmp_path):
    repo = repository(tmp_path, {"v0.3.1": 1}, 3)
    short = git(repo, "rev-parse", "--short", "HEAD")[:9]
    (repo / "notes.txt").write_text("changed")
    found = run_code(repo / "src" / "python", dist_info(tmp_path / "site", STALE))
    assert (found["version"], found["git_dirty"]) == (f"0.3.2.dev2+g{short}.d{DATE}", True)
    # Untracked files are not changes (the rule's `git status` and `git describe --dirty` agree).
    (repo / "notes.txt").write_text("commit 3")
    (repo / "untracked.txt").write_text("x")
    assert run_code(repo / "src" / "python", dist_info(tmp_path / "site2", STALE))["git_dirty"] is False


@needs_git
def test_a_modified_tagged_tree_is_the_next_development_version(tmp_path):
    repo = repository(tmp_path, {"v0.3.1": 2}, 2)
    short = git(repo, "rev-parse", "--short", "HEAD")[:9]
    (repo / "notes.txt").write_text("changed")
    found = run_code(repo / "src" / "python", dist_info(tmp_path / "site", STALE))
    assert (found["version"], found["git_dirty"]) == (f"0.3.2.dev0+g{short}.d{DATE}", True)


@needs_git
def test_the_checkout_version_equals_the_one_the_build_computes(tmp_path):
    # `uv sync --no-editable` builds with hatch-vcs, which calls setuptools-scm with the options of pyproject.toml.
    setuptools_scm = pytest.importorskip("setuptools_scm", reason="the build's version rule is not installed")
    for number, (tags, commits) in enumerate([({}, 3), ({"v0.3.1": 1}, 3), ({"v0.3.1": 3}, 2 + 1)]):
        repo = repository(tmp_path / str(number), tags, commits)
        old = os.environ.get("SOURCE_DATE_EPOCH")
        os.environ["SOURCE_DATE_EPOCH"] = EPOCH
        try:
            for dirty in (False, True):
                (repo / "notes.txt").write_text("changed" if dirty else "commit " + str(commits))
                built = setuptools_scm.get_version(**{**RULE, "root": str(repo)})
                found = run_code(repo / "src" / "python", dist_info(tmp_path / f"site{number}{dirty}", STALE))
                assert found["version"] == built
        finally:
            if old is None:
                del os.environ["SOURCE_DATE_EPOCH"]
            else:
                os.environ["SOURCE_DATE_EPOCH"] = old


@needs_git
def test_an_installed_package_records_the_build_information(tmp_path):
    # An image build writes starfinder/_build_info.py and installs without .git; the metadata carries the version.
    site = dist_info(tmp_path / "site", BUILD_VERSION)
    folder = package(site.parent / "image")
    (folder / "_build_info.py").write_text(f'COMMIT = "{BUILD_COMMIT}"\nDIRTY = False\n')
    found = run_code(folder.parent, site)
    assert found == {"version": BUILD_VERSION, "version_source": "build", "git_commit": BUILD_COMMIT,
                     "git_dirty": False}
    (folder / "_build_info.py").write_text(f'COMMIT = "{BUILD_COMMIT}"\nDIRTY = True\n')
    assert run_code(folder.parent, site)["git_dirty"] is True


@needs_git
def test_a_package_inside_another_repository_does_not_use_it(tmp_path):
    # Only src/python/starfinder at a checkout's root counts; an enclosing project's commit is never recorded.
    git(tmp_path, "init", "-q", str(tmp_path))
    commit(tmp_path, "other project")
    site = dist_info(tmp_path / "site", BUILD_VERSION)
    folder = tmp_path / "venv" / "lib" / "starfinder"
    folder.mkdir(parents=True)
    (folder / "__init__.py").write_text("")
    shutil.copy(RECORDS, folder / "_records.py")
    found = run_code(folder.parent, site)
    assert found == {"version": BUILD_VERSION, "version_source": "dist", "git_commit": None, "git_dirty": None}


def test_without_a_checkout_or_build_information_only_the_metadata_remains(tmp_path):
    site = dist_info(tmp_path / "site", BUILD_VERSION)
    folder = package(site.parent / "installed")
    assert run_code(folder.parent, site) == {"version": BUILD_VERSION, "version_source": "dist",
                                             "git_commit": None, "git_dirty": None}


def build_step(stage):
    """The shell command of the RUN that writes ``_build_info.py`` in a Containerfile build stage (continuations joined)."""
    text = CONTAINERFILE.read_text().replace("\\\n", " ")
    section = text.split(f"AS {stage}\n")[1].split("\nFROM ")[0]
    steps = [line[len("RUN "):] for line in section.splitlines() if line.startswith("RUN ") and "_build_info.py" in line]
    assert len(steps) == 1, stage
    return steps[0]


@pytest.mark.skipif(not CONTAINERFILE.is_file(), reason="the Containerfile is not in this tree")
@pytest.mark.parametrize("stage", ["build-cpu", "build-learned", "build-learned-cuda"])
def test_every_image_build_stage_writes_the_build_information_from_its_arguments(tmp_path, stage):
    command = build_step(stage)
    folder = package(tmp_path / "image")

    def run(revision, dirty):
        env = dict(os.environ, STARFINDER_REVISION=revision, STARFINDER_DIRTY=dirty)
        return subprocess.run(["sh", "-c", command], cwd=folder.parent, env=env, capture_output=True)
    assert run(BUILD_COMMIT, "false").returncode == 0
    site = dist_info(tmp_path / "site", BUILD_VERSION)
    assert run_code(folder.parent, site) == {"version": BUILD_VERSION, "version_source": "build",
                                        "git_commit": BUILD_COMMIT, "git_dirty": False}
    assert run(BUILD_COMMIT, "true").returncode == 0
    assert run_code(folder.parent, site)["git_dirty"] is True
    # The build stops, and writes nothing new, without a full commit or a dirty flag.
    (folder / "_build_info.py").unlink()
    for revision, dirty in [("", "false"), (BUILD_COMMIT[:12], "false"), (BUILD_COMMIT.upper(), "false"),
                            (BUILD_COMMIT, ""), (BUILD_COMMIT, "yes")]:
        assert run(revision, dirty).returncode != 0, (revision, dirty)
        assert not (folder / "_build_info.py").exists()
