"""Default-tier tests of the §2.5 report renderer and example (W-234)."""
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def module():
    spec = importlib.util.spec_from_file_location("preprocessing_report",
                                                  ROOT / "benchmarks" / "preprocessing_report.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


report = module()
harness = report.harness


def test_comparison_example_runs():
    results = runpy.run_path(str(ROOT / "docs/examples/preprocessing_comparison.py"))["main"]()
    assert set(results) == {(seed, arm) for seed in (0, 100) for arm in ("none", "scalar", "pct", "r2_scalar")}
    for record in results.values():
        assert 0 <= record["max_f1"] <= 1 and 0 <= record["auprc"] <= 1
        assert record["f1_t5"] <= record["max_f1"]


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.org",
                           *args], capture_output=True, check=True).stdout


def test_revision_check_resolves_a_dirty_run_to_the_commit_that_holds_its_diff(tmp_path):
    source = tmp_path / "benchmarks" / "preprocessing_synthetic.py"
    source.parent.mkdir()
    source.write_text("version = 1\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "first")
    first = _git(tmp_path, "rev-parse", "HEAD").decode().strip()
    source.write_text("version = 2\n")
    recorded = hashlib.sha256(_git(tmp_path, "diff", "HEAD", "--binary")).hexdigest()
    _git(tmp_path, "commit", "-q", "-am", "repair")
    second = _git(tmp_path, "rev-parse", "HEAD").decode().strip()
    (tmp_path / "notes.md").write_text("later, unrelated\n")
    _git(tmp_path, "add", "notes.md")
    _git(tmp_path, "commit", "-q", "-m", "later")
    software = dict(revision=first, dirty=True, uncommitted_diff_sha256=recorded, untracked_files=[])
    identity = report.check_revision(software, tmp_path)
    assert identity["evaluated_revision"] == second and identity["manifest_revision"] == first
    assert report.check_revision(dict(revision=second, dirty=False), tmp_path)["evaluated_revision"] == second
    with pytest.raises(report.RevisionMismatch, match="no commit"):
        report.check_revision(dict(software, uncommitted_diff_sha256="0" * 64), tmp_path)
    with pytest.raises(report.RevisionMismatch, match="untracked"):
        report.check_revision(dict(software, untracked_files=["x.py"]), tmp_path)
    source.write_text("version = 3\n")
    with pytest.raises(report.RevisionMismatch, match="evaluation sources changed"):
        report.check_revision(software, tmp_path)


def test_revision_check_rejects_a_commit_outside_the_history(tmp_path):
    (tmp_path / "a.txt").write_text("a\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "a")
    _git(tmp_path, "checkout", "-q", "-b", "side")
    (tmp_path / "a.txt").write_text("b\n")
    _git(tmp_path, "commit", "-q", "-am", "side")
    side = _git(tmp_path, "rev-parse", "HEAD").decode().strip()
    _git(tmp_path, "checkout", "-q", "-")
    with pytest.raises(report.RevisionMismatch, match="not an ancestor"):
        report.check_revision(dict(revision=side, dirty=False), tmp_path)
    with pytest.raises(report.RevisionMismatch, match="no source revision"):
        report.check_revision(dict(revision=None), tmp_path)


def test_sources_are_verified_against_the_manifest(tmp_path):
    table = tmp_path / "tables" / "flags.csv"
    table.parent.mkdir()
    table.write_bytes(b"method,low_benefit\nscalar_background,False\n")
    data = table.read_bytes()
    manifest = dict(files=[dict(path="tables/flags.csv", bytes=len(data), sha256=hashlib.sha256(data).hexdigest())])
    sources = report.Sources(tmp_path, report.verify_sources(tmp_path, manifest))
    rows = sources.table("tables/flags.csv").to_dict("records")
    columns = [("method", lambda r: report.code(r["method"]), False)]
    rendered = report.html_table(rows, columns, "tables/flags.csv", sources)
    assert "<code>tables/flags.csv</code>" in rendered and hashlib.sha256(data).hexdigest()[:16] in rendered
    assert sources.used == {"tables/flags.csv"}
    with pytest.raises(report.SourceMismatch, match="not listed"):
        sources.table("tables/other.csv")
    table.write_bytes(data.replace(b"False", b"True "))
    with pytest.raises(report.SourceMismatch, match="differs from the manifest"):
        report.verify_sources(tmp_path, manifest)
    table.unlink()
    with pytest.raises(report.SourceMismatch, match="missing"):
        report.verify_sources(tmp_path, manifest)


def test_every_method_mode_and_condition_has_report_text():
    methods = {m for m, *_ in harness.COMPARISONS + harness.SAMPLE_COMPARISONS}
    assert set(report.METHODS) == methods
    for spec in report.METHODS.values():
        assert all(spec[key] for key in ("title", "problem", "cause", "why_small", "panel"))
    assert set(report.CONDITIONS) == set(harness.CONDITIONS) | set(harness.MULTI_FOV)
    for sentence in report.CONDITIONS.values():
        assert sentence.endswith(".") and ". " not in sentence
    limitations = " ".join(report.LIMITATIONS)
    for phrase in ("Synthetic presets only", "Uncalibrated backgrounds", "uint8 export scaling unconfirmed",
                   "No recommended default", "provisional"):
        assert phrase in limitations


def test_method_figure_renders_panels_for_a_tiny_scene(tmp_path):
    book, config = harness.scene_config("baseline", dtype="uint8", seed=100, shape=(10, 32, 32), count=12)
    scene, background, signal, truth = harness.generate(book, config)
    recipes = harness.arms(harness.background_radius(config))
    ref = book.round_labels[0]
    outputs = {arm: harness.preprocess(harness.make_fov(scene, book, config.FOV_id, tmp_path), recipes[arm],
                                       False).images[ref] for arm in ("none", "scalar")}
    data = dict(book=book, truth=truth, background=background[ref], signal=signal[ref], outputs=outputs)
    cutoffs = {side: harness._noise_thresholds(outputs[arm], 5.0) for side, arm in (("before", "none"),
                                                                                    ("after", "scalar"))}
    report._check_cutoffs(outputs["none"], cutoffs["before"], "before")
    with pytest.raises(report.SourceMismatch):
        report._check_cutoffs(outputs["none"], cutoffs["after"], "after")
    png, facts = report.method_figure(data, "none", "scalar", dict(before="none", after="scalar"), cutoffs)
    assert png.startswith(b"\x89PNG") and facts["channel"] in book.channel_labels
    dim = report.dim_punctum(truth, signal[ref], book)
    assert (facts["z"], facts["y"], facts["x"]) == dim["center"]
    peaks = [signal[ref][tuple(int(round(v)) for v in (r.z, r.y, r.x)) + (book.color_to_channel[r.color_sequence[0]],)]
             for r in truth.itertuples()]
    assert dim["peak"] >= min(peaks) and np.isfinite(dim["peak"])
    assert json.dumps(facts, default=str)
