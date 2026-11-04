"""The public experiment commands over the private entries: ``experiment expand``, ``trial run``, ``experiment evaluate``.

design/specs/benchmark.md, "Proposed public names and migration". Each command has ``--help`` and explicit exit
statuses (0 done, 1 a method, source or metric failure with its record written, 2 a usage or input error with nothing
written), and writes the records the specification names: ``manifest.json`` (``starfinder.benchmark.experiment/1``),
``record.json`` (``starfinder.benchmark.trial/1``) and ``evaluation.json``. No scheduler, retry or workflow task kind is
added. W-781 removed the old ``benchmark`` commands and lifecycle objects: they are refused, with no alias.
"""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pytest
import yaml

from starfinder.__main__ import main
from starfinder.benchmark._experiment import expand_experiment

pytestmark = [pytest.mark.benchmark, pytest.mark.contract]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _inputs(folder):
    z, y, x = np.indices((8, 24, 24))
    image = (100 * np.exp(-((z - 3)**2 / 2 + (y - 9)**2 / 8 + (x - 10)**2 / 8))).astype(np.float32)
    np.save(folder / "reference.npy", image)
    np.save(folder / "moving.npy", np.roll(image, 1, axis=2))


def _experiment(folder):
    _inputs(folder)
    path = folder / "experiment.yaml"
    path.write_text(yaml.safe_dump(dict(
        methods=[dict(module="registration", method="translation")],
        conditions=[dict(name="scipy"), dict(name="skimage", method=dict(config=dict(backend="skimage")))],
        seeds=[7], inputs=[dict(reference="reference.npy", moving="moving.npy")])))
    return path


def _exit(capsys, *argv):
    with pytest.raises(SystemExit) as error:
        main(list(argv))
    return error.value.code, capsys.readouterr()


@pytest.mark.parametrize("command", [
    ("experiment",), ("experiment", "expand"), ("experiment", "evaluate"), ("experiment", "report"), ("trial",),
    ("trial", "run"), ("synthetic", "generate"),
])
def test_commands_help_and_exit_status(capsys, command):
    status, captured = _exit(capsys, *command, "--help")
    assert status == 0 and captured.out.startswith("usage: starfinder " + " ".join(command))
    if command in [("experiment", "expand"), ("experiment", "evaluate"), ("experiment", "report"), ("trial", "run")]:
        text = " ".join(captured.out.split())
        assert "Exit 0" in text and "2 for" in text
        if command not in [("experiment", "expand"), ("experiment", "report")]:
            assert "1 when" in text


@pytest.mark.parametrize("command", [("benchmark", "run"), ("benchmark", "evaluate"), ("benchmark", "report")])
def test_old_benchmark_commands_are_refused(capsys, command):
    """W-781 removed ``starfinder benchmark``: usage error 2, nothing run."""
    status, captured = _exit(capsys, *command, "--help")
    assert status == 2 and "invalid choice: 'benchmark'" in captured.err


def test_expand_writes_the_private_manifest_and_refuses_a_nonempty_output(tmp_path, capsys):
    path = _experiment(tmp_path)
    assert main(["experiment", "expand", "--config", str(path), "--output", str(tmp_path / "expanded")]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "expanded" / "manifest.json")
    manifest = json.loads((tmp_path / "expanded" / "manifest.json").read_text())
    assert manifest["format"] == "starfinder.benchmark.experiment/1" and len(manifest["trials"]) == 2
    assert manifest["source"] == {"path": str(path), "sha256": _sha(path)}
    for entry in manifest["trials"]:
        assert entry["sha256"] == "sha256:" + _sha(tmp_path / "expanded" / entry["path"])
    # The same records as the private expansion into the same folder, byte for byte.
    before = {p.name: p.read_bytes() for p in (tmp_path / "expanded").iterdir()}
    assert expand_experiment(path, tmp_path / "expanded") == manifest
    assert {p.name: p.read_bytes() for p in (tmp_path / "expanded").iterdir()} == before
    status, captured = _exit(capsys, "experiment", "expand", "--config", str(path), "--output",
                             str(tmp_path / "expanded"))
    assert status == 2 and "new or empty" in captured.err
    assert {p.name: p.read_bytes() for p in (tmp_path / "expanded").iterdir()} == before
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(dict(yaml.safe_load(path.read_text()), schedule="nightly")))
    status, captured = _exit(capsys, "experiment", "expand", "--config", str(bad), "--output", str(tmp_path / "new"))
    assert status == 2 and "unsupported experiment keys" in captured.err and not (tmp_path / "new").exists()
    bad.write_text("methods: [")
    status, captured = _exit(capsys, "experiment", "expand", "--config", str(bad), "--output", str(tmp_path / "new"))
    assert status == 2 and "not YAML or JSON" in captured.err


def test_trial_run_writes_the_record_and_exits_with_the_trial_status(tmp_path, capsys):
    path = _experiment(tmp_path)
    assert main(["experiment", "expand", "--config", str(path), "--output", str(tmp_path / "expanded")]) == 0
    manifest = json.loads((tmp_path / "expanded" / "manifest.json").read_text())
    capsys.readouterr()
    config = tmp_path / "expanded" / manifest["trials"][0]["path"]
    assert main(["trial", "run", "--config", str(config)]) == 0
    record_path = tmp_path / "expanded" / "trial-000000" / "record.json"
    assert capsys.readouterr().out.strip() == str(record_path)
    record = json.loads(record_path.read_text())
    assert record["format"] == "starfinder.benchmark.trial/1" and record["status"] == "succeeded"
    assert record["configuration_sha256"] == manifest["trials"][0]["configuration_sha256"]
    assert {o["path"] for o in record["outputs"]} == {"transform.json", "image.npy"}
    # One method per process: a second run into the same output is refused, and the first record is kept.
    before = record_path.read_bytes()
    status, captured = _exit(capsys, "trial", "run", "--config", str(config))
    assert status == 2 and "must be empty" in captured.err and record_path.read_bytes() == before
    failing = tmp_path / "failing.json"
    failing.write_text(json.dumps(dict(method=dict(module="registration", method="translation"),
                                       inputs=dict(reference="reference.npy", moving="absent.npy"), seed=1,
                                       root_output_path="failing")))
    assert main(["trial", "run", "--config", str(failing)]) == 1
    record = json.loads((tmp_path / "failing" / "record.json").read_text())
    assert record["status"] == "failed" and record["error"]["type"] == "FileNotFoundError"
    assert [p.name for p in (tmp_path / "failing").iterdir()] == ["record.json"]
    unreadable = tmp_path / "unreadable.yaml"
    unreadable.write_text("method: [")
    status, captured = _exit(capsys, "trial", "run", "--config", str(unreadable))
    assert status == 2 and "not YAML or JSON" in captured.err
    unreadable.write_text(json.dumps(dict(method=dict(module="registration", method="translation"))))
    status, captured = _exit(capsys, "trial", "run", "--config", str(unreadable))
    assert status == 2 and "root_output_path" in captured.err


def test_no_second_scheduler_and_no_new_task_kind():
    """The commands add no workflow task kind; the trial and expansion entries hold no scheduler or retry."""
    import inspect
    import starfinder.benchmark._evaluate as evaluate
    import starfinder.benchmark._experiment as experiment
    import starfinder.benchmark._trial as trial
    from starfinder.workflow._shape import COMMANDS
    assert set(COMMANDS) == {"summarize", "merge", "fov", "tile_global", "tile", "tile_merge", "morphology",
                             "segment", "assign", "bigstitcher_macro", "bigstitcher_engine", "bigstitcher_import",
                             "stitch", "export", "report", "compare"}
    for module in (trial, experiment, evaluate):
        source = inspect.getsource(module)
        for word in ("subprocess", "retry", "Snakefile", "nextflow", "qsub", "concurrent.futures", "multiprocessing"):
            assert word not in source, (module.__name__, word)


DOCS = Path(__file__).resolve().parents[3] / "docs"


def test_the_documented_example_runs(tmp_path, monkeypatch, capsys):
    """The Quick start of docs/benchmark/experiments.md, command by command, in a new folder outside Git."""
    import pandas as pd
    import runpy
    page = (DOCS / "benchmark" / "experiments.md").read_text()
    block = page.split("```bash\n", 1)[1].split("```", 1)[0]
    lines = [line.replace("/tmp/starfinder-experiment", str(tmp_path / "example")).split()
             for line in block.strip().splitlines()]
    assert [line[:3] for line in lines] == [["uv", "run", "python"]] + [["uv", "run", "starfinder"]] * 4
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr("sys.argv", lines[0][3:])
    runpy.run_path(lines[0][3], run_name="__main__")
    for line in lines[1:]:
        assert main(line[3:]) == 0, line
    capsys.readouterr()
    values = pd.read_csv(tmp_path / "example" / "evaluation" / "values.csv")
    assert set(values.metric) == {"translation-scipy", "translation-skimage"} and set(values.outcome) == {"computed"}
    assert set(values[values.value_name == "max_error"].value.astype(float)) == {0.0}
    assert set(values[values.value_name == "passed"].value) == {"True"}


def test_the_reference_names_every_metric_module_and_outcome():
    from starfinder.benchmark._evaluate import METRIC_MODULES, OUTCOMES
    reference = (DOCS / "benchmark" / "commands.md").read_text()
    guide = (DOCS / "benchmark" / "experiments.md").read_text()
    assert all(f"`{module}`" in reference for module in METRIC_MODULES)
    assert all(f"`{outcome}`" in guide for outcome in OUTCOMES)


def test_removed_api_and_invalid_config(tmp_path, capsys):
    """The removed lifecycle names stay removed, with no alias, and unknown fields fail explicitly."""
    import starfinder.benchmark as benchmark
    for name in ("BenchmarkSuite", "BenchmarkResult", "RegistrationBenchmarkRunner", "RegistrationResult", "benchmark",
                 "measure", "run_comparison", "BenchmarkCase", "BenchmarkTrialResult", "run_benchmark",
                 "evaluate_benchmark", "report_benchmark"):
        assert not hasattr(benchmark, name), name
    import starfinder
    assert not any("benchmark" in name.lower() for name in starfinder.__all__)
    _inputs(tmp_path)
    good = dict(method=dict(module="registration", method="translation"),
                inputs=dict(reference="reference.npy", moving="moving.npy"), seed=1)
    # An unknown trial key and an unknown method field: the trial fails with its record saying why (exit 1).
    for name, change, error in (("key", dict(typo=True), ("ValueError", "unsupported trial keys")),
                                ("field", dict(method=dict(good["method"], config=dict(typo=1))),
                                 ("TypeError", "unexpected keyword argument 'typo'"))):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(dict(good, root_output_path=name, **change)))
        assert main(["trial", "run", "--config", str(path)]) == 1
        record = json.loads((tmp_path / name / "record.json").read_text())
        assert record["status"] == "failed" and record["outputs"] == []
        assert record["error"]["type"] == error[0] and error[1] in record["error"]["message"]
    # An unknown experiment key is refused before anything is written (exit 2).
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(yaml.safe_dump(dict(methods=[good["method"]], conditions=[{}], seeds=[1],
                                              inputs=[good["inputs"]], cases=[])))
    status, captured = _exit(capsys, "experiment", "expand", "--config", str(experiment), "--output",
                             str(tmp_path / "expanded"))
    assert status == 2 and "unsupported experiment keys" in captured.err and not (tmp_path / "expanded").exists()


#: The archive links of docs/migration.md: the GitHub blob of a removed file at an immutable revision.
ARCHIVE = re.compile(r"https://github\.com/wanglab-broad/starfinder/blob/([0-9a-f]+)/([^)#\s]+)")


def test_explicit_config_and_archive_links(tmp_path, monkeypatch, capsys):
    """The documented replacement configuration is explicit and expands with no processing; each archive link of
    docs/migration.md names a file that is gone from the tree, at a full immutable revision that holds it."""
    import runpy
    import subprocess
    monkeypatch.setattr("sys.argv", ["experiment_inputs.py", str(tmp_path / "example")])
    runpy.run_path(str(DOCS / "examples" / "experiment_inputs.py"), run_name="__main__")
    config = yaml.safe_load((tmp_path / "example" / "experiment.yaml").read_text())
    assert set(config) <= {"methods", "conditions", "seeds", "inputs", "metadata", "root_output_path"}
    manifest = expand_experiment(tmp_path / "example" / "experiment.yaml", tmp_path / "trials")
    assert len(manifest["trials"]) == len(config["methods"]) * len(config["conditions"]) * len(config["seeds"]) * len(
        config["inputs"])
    assert not any(p.name == "record.json" for p in tmp_path.rglob("*"))
    capsys.readouterr()
    links = ARCHIVE.findall((DOCS / "migration.md").read_text())
    removed = {"benchmarks/recipes.py", "benchmarks/report_saved.py", "benchmarks/preprocessing_synthetic.py",
               "benchmarks/preprocessing_report.py", "src/python/starfinder/benchmark/_lifecycle.py",
               "design/notes/benchmark-recipes.md"}
    assert removed <= {path for _, path in links}
    root = DOCS.parent
    git = (root / ".git").exists()
    for revision, path in links:
        assert len(revision) == 40, (revision, path)
        assert not (root / path).exists(), f"{path} is a live file, not an archived one"
        if git:
            subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{revision}:{path}"], check=True)
