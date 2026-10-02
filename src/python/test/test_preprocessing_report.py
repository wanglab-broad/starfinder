"""Default-tier tests of the §2.5 report renderer and example (W-234; calibrated report, W-249)."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import runpy
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.preprocessing, pytest.mark.benchmark]

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


# --- Calibrated rerun report (W-249) ---------------------------------------------------------------

@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    """A calibrated smoke evaluation (clean, uint8, seeds 0 and 100, both threshold modes)."""
    output = tmp_path_factory.mktemp("w249") / "evaluation"
    manifest = harness.run_calibrated(output, scope="smoke", shape=(8, 32, 32), count=12, log=lambda m: None)
    return output, manifest


def _sources(output, manifest):
    records = report.verify_sources(output, manifest)
    data = (output / "manifest.json").read_bytes()
    records["manifest.json"] = dict(path="manifest.json", bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    return report.Sources(output, records)


def test_calibrated_cards_follow_the_page_order_and_cover_every_method():
    methods = {m for m, *_ in harness.CALIBRATED_COMPARISONS + harness.SAMPLE_COMPARISONS}
    assert list(report.CARDS) == ["scalar_background", "background_3d", "percentile_normalization",
                                  "sample_level_fitting", "extraction_source", "min_max_normalization",
                                  "histogram_matching", "reconstruction", "white_tophat"]
    assert set(report.CARDS) == methods
    for method, card in report.CARDS.items():
        _before, _after, targeted = report.comparison_spec(method, card["comparison"])
        names = [row[0] if isinstance(row, tuple) else row for row in card["rows"]]
        assert names[0] in targeted and report._role(method, names[0]) == "targeted"
        roles = [report._role(method, n) for n in names]
        assert "harm_check" in roles and set(roles) <= {"targeted", "harm_check", "combined", "harm_test"}
        assert ("combined" in roles) == (method != "sample_level_fitting")
    assert report._role("histogram_matching", "clean_unbalanced") == "harm_test"
    assert set(report.CALIBRATED_TEXT) == set(harness.CALIBRATED_ALL) | set(harness.MULTI_FOV)
    for sentence in report.CALIBRATED_TEXT.values():
        assert sentence.endswith(".") and ". " not in sentence


@pytest.mark.slow
def test_calibrated_scenes_outputs_and_detections_match_the_saved_run(smoke, tmp_path):
    output, manifest = smoke
    sources = _sources(output, manifest)
    scenes = report.CalibratedScenes(manifest, tmp_path)
    fov = scenes.processed("clean", "scalar")
    assert [v["what"] for v in scenes.verified][-1] == "output"
    truth = scenes.single("clean")["truth"]
    points, curves = sources.table("tables/operating_points.csv"), sources.table("curves/pr_curves.csv")
    for mode in report.MODES:
        value, _edge = report.selected_threshold(points, "clean", "uint8", "scalar", mode, False)
        found = report.detect(fov, truth, mode, value)
        selector = dict(condition="clean", dtype="uint8", seed=100, arm="scalar", threshold_mode=mode, threshold=value)
        assert report.check_detection(curves, selector, found, "test")["n_matched"] == found["counts"]["n_matched"]
        wrong = dict(found, counts=dict(found["counts"], n_detected=found["counts"]["n_detected"] + 1))
        with pytest.raises(report.DetectionMismatch, match="n_detected"):
            report.check_detection(curves, selector, wrong, "test")
        with pytest.raises(report.DetectionMismatch, match="cutoffs"):
            report.check_detection(curves, selector, dict(found, cutoffs=[c + 1 for c in found["cutoffs"]]), "test")
    tampered = json.loads(json.dumps(manifest))
    next(r for r in tampered["runs"] if r["seed"] == 100 and r["arm"] == "pct")["output_sha256"] = "0" * 64
    with pytest.raises(report.SourceMismatch, match="output_sha256"):
        report.CalibratedScenes(tampered, tmp_path).processed("clean", "pct")
    scene = next(s for s in tampered["scenes"] if s["seed"] == 100)
    scene["image_sha256"] = {k: "0" * 64 for k in scene["image_sha256"]}
    with pytest.raises(report.SourceMismatch, match="differs from the manifest"):
        report.CalibratedScenes(tampered, tmp_path).single("clean")


@pytest.mark.slow
def test_dim_punctum_is_the_lowest_realized_peak_and_leads_the_colour_view(smoke, tmp_path):
    _output, manifest = smoke
    data = report.CalibratedScenes(manifest, tmp_path).single("clean")
    book, truth = data["book"], data["truth"]
    signal = data["signal"][book.round_labels[0]]
    peaks = report.realized_peaks(truth, signal, book)
    dim = report.calibrated_dim_punctum(truth, signal, book)
    assert dim["index"] == int(np.argmin(peaks)) and dim["peak"] == peaks.min()
    row = truth.iloc[dim["index"]]
    assert dim["channel"] == book.color_to_channel[row.color_sequence[0]]
    center = tuple(int(round(v)) for v in (row.z, row.y, row.x))
    box = signal[..., dim["channel"]][harness._box(signal.shape[:3], center, (1, 1, 1))]
    assert dim["peak"] == box.max()
    chosen = report.colour_puncta(peaks)
    assert chosen[0] == dim["index"] and len(set(chosen)) == len(chosen) <= 4
    assert report.signal_channels(truth, book) == sorted({book.color_to_channel[s[0]] for s in truth.color_sequence})


def test_collapsed_pr_curves_are_detected():
    flat = pd.DataFrame(dict(threshold=[2.0, 3.0, 4.0], precision=[0.5] * 3, recall=[0.4] * 3))
    assert report.collapsed(flat)
    assert not report.collapsed(flat.assign(recall=[0.4, 0.3, 0.2]))
    assert not report.collapsed(flat.iloc[:1])


@pytest.mark.slow
def test_calibrated_report_puts_the_summary_first_and_refuses_unverified_inputs(smoke, tmp_path, monkeypatch):
    output, manifest = smoke
    real_check = report.check_revision
    # The smoke run has only clean, so every single-FOV card shows clean; the revision check is tested above.
    cards = {m: dict(c, rows=("clean",)) for m, c in report.CARDS.items() if m != "sample_level_fitting"}
    monkeypatch.setattr(report, "CARDS", cards)
    identity = dict(manifest_revision="0" * 40, dirty=False, uncommitted_diff_sha256=None, evaluated_revision="0" * 40,
                    match="test", sources_unchanged=list(report.CALIBRATED_SOURCES))
    monkeypatch.setattr(report, "check_revision", lambda *args, **kwargs: identity)
    path = tmp_path / "report.html"
    summary = report.render(output, path)
    text = path.read_text()
    anchors = ["summary", "setup", "cards", *(f"card-{m}" for m in cards), "findings", "reading-order",
               "method-figures", *(f"fig-{m}" for m in cards), "appendix", "identity", "render-checks", "a-flags",
               "a-comparisons-uint8", "sources"]
    positions = [text.index(f"id=\"{a}\"") for a in anchors]
    assert positions == sorted(positions)
    for phrase in ("not D04", "no preprocessing default", "provisional", "noise mode", "adaptive mode",
                   "label finding", "label hypothesis", "fig-scalar_background-background",
                   "identical in both threshold modes and is drawn once"):
        assert phrase in text
    diagnostics = pd.read_csv(output / "tables" / "diagnostics.csv")
    section = text[text.index("id=\"a-diagnostics\""):text.index("id=\"a-statistics\"")]
    assert section.count("<tr>") - section.count("<thead><tr>") == len(diagnostics) and diagnostics["round"].nunique() > 1
    assert summary["anchors"] == [a for a in summary["anchors"] if f"id=\"{a}\"" in text]
    assert {"setup", "findings", "fig-scalar_background-histograms", "render-checks-detections"} <= set(summary["anchors"])
    assert "src=\"http" not in text and "href=\"http" not in text
    assert summary["detection_checks"] == len(cards) * 2 * 2 and summary["files_verified"] == len(manifest["files"])
    assert summary["schema"] == report.CALIBRATED_REPORT_SCHEMA
    # Nothing is written when a saved file or the revision does not match.
    copy = tmp_path / "copy"
    shutil.copytree(output, copy)
    table = copy / "tables" / "endpoints.csv"
    table.write_bytes(table.read_bytes().replace(b"clean", b"CLEAN", 1))
    with pytest.raises(report.SourceMismatch, match="endpoints.csv differs"):
        report.render(copy, tmp_path / "refused.html")
    monkeypatch.setattr(report, "check_revision", real_check)
    stale = json.loads((output / "manifest.json").read_text())
    stale["software"] = dict(stale["software"], revision="0" * 40)
    (copy / "manifest.json").write_text(json.dumps(stale))
    with pytest.raises(report.RevisionMismatch):
        report.render(copy, tmp_path / "refused.html")
    assert not (tmp_path / "refused.html").exists()


def test_full_tables_keep_every_row_and_column_within_one_capture():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"method": ["percentile_normalization"] * 400, "dtype": ["uint8"] * 400,
                          "seed": np.arange(400),
                          **{f"metric_{k:02d}_before_mean": rng.normal(size=400) for k in range(60)},
                          "reason": ["no eligible targeted condition shows a benefit; clean does not hold"] * 400})
    sources = report.Sources(Path("."), {"t.csv": dict(path="t.csv", bytes=1, sha256="0" * 64)})
    text = report.full_table(frame, "t.csv", sources, anchor="a-t")
    widths = [int(w) for w in re.findall(r"class=\"full\" style=\"width:(\d+)px\"", text)]
    anchors = re.findall(r"id=\"(a-t-r\d+c\d+)\"", text)
    assert len(widths) == len(anchors) > 1 and len(set(anchors)) == len(anchors)
    assert max(widths) <= report.FULL_WIDTH_PX
    groups = len({a.split("c")[-1] for a in anchors})
    assert text.count("<tr>") - text.count("<thead><tr>") == len(frame) * groups
    for column in frame.columns:
        assert f"<th>{column}</th>" in text
    assert text.count("<th>method</th>") == len(anchors)  # the key columns repeat in every piece
