"""The marker registration and the collection check that every test has a subsystem marker (W-289); the
selection coverage and the marker drift report of scripts/ (W-530)."""
import importlib.util
import json
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


# --- The selection coverage and the marker drift report (scripts/, W-530) -----------------------------------------

def script(name):
    """The module of scripts/<name>.py."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_coverage_reports_an_overlap_a_missing_and_a_repeated_node_id():
    coverage = script("selection_coverage").coverage
    ids = {"Q": ["a", "b"], "V": ["c"], "R": ["a", "b", "c"], "slow": ["d"], "learned": ["e"], "extended": []}
    report = coverage(ids, ["a", "b", "c", "d", "e"])
    assert report["ok"] and report["partition_each_once"] and report["q_union_v_equals_r"]
    assert report["q_and_v_overlap"] == []
    report = coverage(dict(ids, V=["b", "c"], learned=["e", "d"], extended=["x"]), ["a", "b", "c", "d", "e", "f"])
    assert not report["ok"] and report["q_and_v_overlap"] == ["b"]
    assert report["partition_missing"] == ["f"] and report["partition_repeated"] == ["d"]
    assert report["partition_outside_collection"] == ["x"]


DRIFT_LOG = """\
test/test_a.py::test_at_the_bound PASSED                                    [ 12%]
test/test_a.py::test_under_the_bound PASSED                                 [ 25%]
test/test_b.py::test_quick_but_slow PASSED                                  [ 37%]
test/test_b.py::test_two_seconds PASSED                                     [ 50%]
test/test_b.py::test_cached PASSED                                          [ 62%]
test/test_b.py::test_hidden[1] PASSED                                       [ 75%]
test/test_b.py::test_skipped SKIPPED (the extra is not installed)           [ 87%]
test/test_c.py::test_renamed PASSED                                         [100%]
============================== slowest durations ===============================
4.50s call     test/test_a.py::test_at_the_bound
0.50s setup    test/test_a.py::test_at_the_bound
4.99s call     test/test_a.py::test_under_the_bound
1.99s call     test/test_b.py::test_quick_but_slow
2.00s call     test/test_b.py::test_two_seconds
0.10s call     test/test_b.py::test_cached
0.01s setup    test/test_b.py::test_skipped
0.20s call     test/test_c.py::test_renamed

(3 durations < 0.005s hidden.  Use -vv to show these durations.)
"""
DRIFT_ITEMS = {"test/test_a.py::test_at_the_bound": {"markers": ["io"]},
               "test/test_a.py::test_under_the_bound": {"markers": ["io", "contract"]},
               "test/test_b.py::test_quick_but_slow": {"markers": ["io", "slow"], "shared_fixtures": []},
               "test/test_b.py::test_two_seconds": {"markers": ["io", "slow"]},
               "test/test_b.py::test_cached": {"markers": ["io", "slow"], "shared_fixtures": ["scene"]},
               "test/test_b.py::test_hidden[1]": {"markers": ["io", "slow", "parametrize"]},
               "test/test_b.py::test_skipped": {"markers": ["io", "slow"]}}


def test_marker_drift_lists_unmarked_tests_at_5_s_and_slow_tests_under_2_s():
    drift = script("marker_drift")
    durations, outcomes = drift.parse_log(DRIFT_LOG)
    assert durations["test/test_a.py::test_at_the_bound"] == {"call": 4.5, "setup": 0.5}
    assert outcomes["test/test_b.py::test_skipped"] == "SKIPPED" and "test/test_b.py::test_hidden[1]" not in durations
    report = drift.drift(durations, outcomes, DRIFT_ITEMS)
    assert report["tests"] == 7
    assert [(r["nodeid"], r["seconds"]) for r in report["unmarked_at_or_over"]] == [
        ("test/test_a.py::test_at_the_bound", 5.0)]
    assert [(r["nodeid"], r["seconds"]) for r in report["slow_under"]] == [
        ("test/test_b.py::test_quick_but_slow", 1.99), ("test/test_b.py::test_hidden[1]", 0)]
    assert [(r["nodeid"], r["shared_fixtures"]) for r in report["slow_under_shared_fixture"]] == [
        ("test/test_b.py::test_cached", ["scene"])]
    assert report["not_collected"] == ["test/test_c.py::test_renamed"]


def test_marker_drift_reads_a_log_and_a_markers_file(tmp_path, capsys):
    drift = script("marker_drift")
    (tmp_path / "durations.log").write_text(DRIFT_LOG)
    (tmp_path / "markers.jsonl").write_text("".join(json.dumps(dict(item, nodeid=node)) + "\n"
                                                    for node, item in DRIFT_ITEMS.items()))
    assert drift.main([str(tmp_path / "durations.log"), "--markers", str(tmp_path / "markers.jsonl"),
                       "--json", str(tmp_path / "drift.json")]) == 0
    out = capsys.readouterr().out
    assert "Without slow, 5 s or more: 1\n     5.00 s  test/test_a.py::test_at_the_bound" in out
    assert "slow, under 2 s: 2\n" in out and "sharing a wider-scoped fixture): 1\n" in out
    assert json.loads((tmp_path / "drift.json").read_text())["slow_under"][0]["seconds"] == 1.99
