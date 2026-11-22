"""The runner checks' inputs by import closure (W-893's H3, W-900): ``test/check_inputs.json``, written by
``scripts/check_inputs.py``, names for each check the test files it runs and the modules they import."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core, pytest.mark.contract]

# Loads scripts/check_inputs.py as the fixture `script` does, in a fresh interpreter, and prints the files under
# src/python of the modules then loaded (repository paths).
LOADED = """
import importlib.util, json, sys
from pathlib import Path
path, root = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("check_inputs", path)
spec.loader.exec_module(importlib.util.module_from_spec(spec))
files = {Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, "__file__", None)}
print(json.dumps(sorted(str(f.relative_to(root.parents[1])) for f in files if f.is_relative_to(root))))
"""

# The checks of the workflow recovery and engine tests (W-893's H1 and H2 took the task, plan and dry-run helpers out
# of their imports).
RECOVERY_AND_ENGINES = ("pytest-extended-recovery-1", "pytest-extended-recovery-2", "pytest-nextflow-recovery",
                        "pytest-nextflow-engines")


@pytest.fixture(scope="module")
def script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_inputs.py"
    spec = importlib.util.spec_from_file_location("check_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def committed(script):
    return json.loads(script.OUTPUT.read_text())


def test_the_file_names_the_checks_of_the_script(script, committed):
    assert {name: check["arguments"] for name, check in committed["checks"].items()} == script.CHECKS
    assert committed["always"] == ["src/python/test/conftest.py", "src/python/test/__init__.py"]


@pytest.mark.slow
def test_each_check_names_the_modules_its_test_files_import(script, committed):
    """Reads every module of the closures once (about 13 s on one CPU)."""
    graph = script.ImportGraph()
    for name, check in committed["checks"].items():
        expected = script.inputs_of(check["test_files"], graph)
        assert {key: check[key] for key in expected} == expected, name


def test_every_test_file_is_in_a_check_or_listed_as_uncollected(script, committed):
    selected = {path for check in committed["checks"].values() for path in check["test_files"]}
    on_disk = script.test_files_on_disk()
    assert selected <= set(on_disk)
    assert [path for path in on_disk if path not in selected] == committed["uncollected"]


def test_the_recovery_and_engine_checks_import_no_other_test_module(committed):
    for name in RECOVERY_AND_ENGINES:
        check = committed["checks"][name]
        assert check["test_files"], name
        assert [path for path in check["modules"] if Path(path).name.startswith("test_")] == [], name
        helpers = {"src/python/test/workflow_tasks.py", "src/python/test/tiled_fixtures.py"}
        assert helpers.isdisjoint(check["modules"]), name


def test_the_checks_that_run_this_file_name_every_module_loading_the_script_executes(committed):
    """An oracle apart from the import graph: what the interpreter has loaded after it executes check_inputs.py."""
    root = Path(__file__).resolve().parents[1]
    done = subprocess.run([sys.executable, "-c", LOADED, str(root / "scripts" / "check_inputs.py"), str(root)],
                          capture_output=True, text=True, check=True)
    loaded = json.loads(done.stdout)
    assert "src/python/scripts/learned_closure.py" in loaded
    this = "src/python/test/test_check_inputs.py"
    checks = [name for name, check in committed["checks"].items() if this in check["test_files"]]
    assert checks
    for name in checks:
        check = committed["checks"][name]
        inputs = set(check["test_files"]) | set(check["modules"]) | set(committed["always"])
        assert [path for path in loaded if path not in inputs] == [], name
