"""The marker registration and the collection check that every test has a subsystem marker (W-289)."""
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from .conftest import SUBSYSTEM_MARKERS, files_without_subsystem_marker, pytest_collection_modifyitems

pytestmark = [pytest.mark.core, pytest.mark.contract]

COST = {"slow", "learned"}
KIND = {"contract", "golden", "validation", "e2e"}


def item(path, *names):
    return SimpleNamespace(location=(path, 0, "test"), iter_markers=lambda: [SimpleNamespace(name=n) for n in names])


def test_registered_markers_are_the_agreed_set():
    options = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    options = options["tool"]["pytest"]["ini_options"]
    registered = {line.split(":")[0] for line in options["markers"]}
    assert registered == SUBSYSTEM_MARKERS | COST | KIND | {"extended"}
    assert "--strict-markers" in options["addopts"].split()


def test_a_test_without_a_subsystem_marker_is_reported_by_file():
    items = [item("test/test_a.py", "registration", "slow"), item("test/test_b.py", "slow", "contract"),
             item("test/test_b.py", "parametrize"), item("test/test_c.py")]
    assert files_without_subsystem_marker(items) == ["test/test_b.py", "test/test_c.py"]
    assert files_without_subsystem_marker(items[:1]) == []


def test_collection_stops_on_a_missing_subsystem_marker():
    pytest_collection_modifyitems([item("test/test_a.py", "io")])
    with pytest.raises(pytest.UsageError, match=r"set pytestmark in: test/test_new\.py"):
        pytest_collection_modifyitems([item("test/test_a.py", "io"), item("test/test_new.py", "golden")])
