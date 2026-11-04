"""``starfinder experiment report``: a saved evaluation rendered in W-237's review format, without processing.

design/proposals/benchmark-drafts/report-template.md (W-782); design/baselines/benchmark.md, "Test disposition rules",
names the tests replaced here. The study is synthetic and hand-built: eight saved trial records (format
``starfinder.benchmark.trial/1``) of one detector, before (``raw``) and after (``tophat``) a preprocessing method, in
two fixed threshold modes (``noise``, ``adaptive``) and seeds 0 and 1, each listing a detection table and an image
(the adaptive mode has no before trial of seed 1), and a failed noise-mode trial of seed 2. ``starfinder experiment
evaluate`` computes F1 (M03) against hand-placed truth, and the correct-decode fraction (M10) and the assigned-read
count (M18) of the noise mode. The known values follow from the coordinates.
Every rendering runs with the processing and evaluation entry points patched to raise; the source files are hashed
before and after.
"""
import hashlib
import importlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
import yaml

from starfinder.__main__ import main
from starfinder.benchmark import _report as R
from starfinder.workflow._shape import configuration_sha256

pytestmark = [pytest.mark.benchmark, pytest.mark.contract]

DOCS = Path(__file__).resolve().parents[3] / "docs"
#: The study's card identifiers: a readable slug and the first 8 hex characters of the exact key's SHA-256.
NOISE, ADAPTIVE = R.card_slug(("white_tophat", "noise")), R.card_slug(("white_tophat", "adaptive"))
UNASSIGNED = R.card_slug((None, None))
SHAPE = (4, 32, 32)
TRUTH = [(2, 8, 8), (2, 8, 24), (2, 24, 8), (2, 24, 24)]
GENES = ["a", "b", "a", "b"]
#: Detections per (mode, arm, seed): indices of the truth points found, and spurious points.
FOUND = {
    ("noise", "raw", 0): ([0, 1], [(1, 16, 16), (3, 30, 2)]),             # recall .5, precision .5, F1 .5
    ("noise", "raw", 1): ([0], [(1, 16, 16)]),                             # recall .25, precision .5, F1 1/3
    ("noise", "tophat", 0): ([0, 1, 2, 3], []),                            # F1 1
    ("noise", "tophat", 1): ([0, 1, 2], [(1, 16, 16)]),                    # recall .75, precision .75, F1 .75
    ("adaptive", "raw", 0): ([0, 1], [(1, 16, 16), (3, 30, 2)]),           # F1 .5
    ("adaptive", "tophat", 0): ([0, 1, 2, 3], []),                         # F1 1
    ("adaptive", "tophat", 1): ([0, 1, 2], []),                            # against empty truth: undefined
}
#: The processing entry points a renderer must never call, and the evaluation entries.
PROCESSING = [
    ("starfinder.dataset.fov", "FOV.run"), ("starfinder.registration", "estimate_transform"),
    ("starfinder.registration", "apply_transform"), ("starfinder.preprocessing.steps", "run_step"),
    ("starfinder.spot_finding", "find_spots"), ("starfinder.segmentation", "segment"),
    *(("starfinder.readout", name) for name in ("extract_intensities", "decode_reads", "filter_reads",
                                                  "score_reads")),
    ("starfinder.assignment", "assign_molecules"),
    *(("starfinder.synthetic", name) for name in ("generate_amplicon_dataset", "generate_amplicon_scene",
                                                    "generate_registration_pair", "generate_cellular_scene")),
    ("starfinder.benchmark._trial", "_run"), ("starfinder.benchmark._trial", "_main"),
    ("starfinder.benchmark._evaluate", "evaluate"),
    ("starfinder.evaluation.spot_finding", "f1_score"), ("starfinder.evaluation.readout", "correct_fraction"),
    ("starfinder.evaluation.readout", "read_status_summary"),
    ("starfinder.evaluation.registration", "evaluate_translation"),
    ("starfinder.evaluation.matching", "match_points"),
]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tree(*roots):
    return {str(p): _sha(p) for root in roots for p in sorted(Path(root).rglob("*")) if p.is_file()}


def _no_processing(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("the report called a processing or evaluation entry point")
    for module, attribute in PROCESSING:
        target = importlib.import_module(module)
        *owner, name = attribute.split(".")
        for part in owner:
            target = getattr(target, part)
        monkeypatch.setattr(target, name, boom)


def _image(seed, arm):
    image = np.zeros(SHAPE, dtype=np.float32) + (5.0 if arm == "raw" else 0.0)
    for k, (z, y, x) in enumerate(TRUTH):
        image[z, y, x] += 40 + 10 * k + seed
    return image


def _write_trial(folder, mode, arm, seed, status="succeeded"):
    """A saved trial folder: an input image, and when it succeeded, the detections and the image it saved."""
    folder.mkdir(parents=True)
    raw = folder / "input.npy"
    np.save(raw, _image(seed, "raw"))
    outputs = []
    if status == "succeeded":
        np.save(folder / "image.npy", _image(seed, arm))
        found, spurious = FOUND[(mode, arm, seed)]
        points = [TRUTH[i] for i in found] + spurious
        genes = [GENES[i] for i in found] + ["a"] * len(spurious)
        # Truth gene of the first found point is miscalled in the noise-mode tophat seed 1, one wrong-gene read.
        if (mode, arm, seed) == ("noise", "tophat", 1):
            genes[0] = "b"
        pd.DataFrame({"z": [float(p[0]) for p in points], "y": [float(p[1]) for p in points],
                      "x": [float(p[2]) for p in points], "gene_id": genes, "accepted": [True] * len(points),
                      "call_status": ["assigned"] * (len(points) - 1) + ["unmatched"] if points else []}
                     ).to_csv(folder / "spots.csv", index=False)
        outputs = [{"path": name, "sha256": _sha(folder / name), "bytes": (folder / name).stat().st_size}
                   for name in ("image.npy", "spots.csv")]
    configuration = {"method": {"module": "spot_finding", "method": "log",
                                "config": {"threshold_mode": mode, "input": arm}},
                     "inputs": {"image": {"path": str(raw), "sha256": _sha(raw)}},
                     "metadata": {"frame_id": "report/voxel"}, "seed": seed, "condition": {"name": "clean"},
                     "root_output_path": str(folder)}
    record = {"format": "starfinder.benchmark.trial/1", "configuration": configuration,
              "configuration_sha256": configuration_sha256(configuration), "seed": seed,
              "inputs": configuration["inputs"], "outputs": outputs, "status": status,
              "error": None if status == "succeeded" else {"type": "ValueError", "message": "the detector failed"},
              "wall_s": 1.5 + seed, "cpu_s": 1.25 + seed, "peak_rss_kib": 50000 + 1000 * seed}
    (folder / "record.json").write_text(json.dumps(record, indent=1))
    return folder


VOXEL = {"metadata": {"frame_id": "report/voxel"}}
MATCHING = {"policy": "greedy", "threshold": 0.5, "boundary": "inclusive", "units": "voxel"}


def _yaml(path, value):
    Path(path).write_text(yaml.safe_dump(value, sort_keys=False))
    return Path(path)


def _study(root):
    """The saved study: trials, truth, the evaluation and the protocol; returns their paths."""
    trials = root / "trials"
    names = {}
    for (mode, arm, seed) in FOUND:
        names[(mode, arm, seed)] = _write_trial(trials / f"{mode}-{arm}-{seed}", mode, arm, seed)
    names[("noise", "tophat", 2)] = _write_trial(trials / "noise-tophat-2", "noise", "tophat", 2, status="failed")
    truth = root / "truth"
    truth.mkdir()
    pd.DataFrame({"z": [float(p[0]) for p in TRUTH], "y": [float(p[1]) for p in TRUTH],
                  "x": [float(p[2]) for p in TRUTH], "gene_id": GENES}).to_csv(truth / "points.csv", index=False)
    pd.DataFrame(columns=["z", "y", "x", "gene_id"]).to_csv(truth / "empty.csv", index=False)
    sources = [{"name": f"{m}-{a}-{s}", "kind": "trial", "path": str(path)} for (m, a, s), path in names.items()]
    sources.append({"name": "truth", "kind": "files", "files": {
        key: {"path": str(truth / f"{key}.csv"), "sha256": _sha(truth / f"{key}.csv")} for key in ("points", "empty")}})
    metrics, rows = [], {}
    for (mode, arm, seed) in names:
        name = f"f1-{mode}-{arm}-{seed}"
        empty = (mode, arm, seed) == ("adaptive", "tophat", 1)
        metrics.append({"name": name, "metric": "M03", "function": "spot_finding.f1_score", "parameters": MATCHING,
                        "arguments": {"detected": {"source": f"{mode}-{arm}-{seed}", "output": "spots.csv",
                                                   "columns": ["z", "y", "x"]},
                                      "truth": {"source": "truth", "file": "empty" if empty else "points",
                                                "columns": ["z", "y", "x"]},
                                      "reference_metadata": VOXEL, "observed_metadata": VOXEL}})
        rows[name] = {"method": "white_tophat", "mode": mode, "side": "before" if arm == "raw" else "after"}
    for seed in (0, 1):
        source = f"noise-tophat-{seed}"
        metrics.append({"name": f"decode-{seed}", "metric": "M10", "function": "readout.correct_fraction",
                        "parameters": MATCHING,
                        "arguments": {"detected": {"source": source, "output": "spots.csv"},
                                      "truth": {"source": "truth", "file": "points"},
                                      "reference_metadata": VOXEL, "observed_metadata": VOXEL}})
        metrics.append({"name": f"assigned-{seed}", "metric": "M18", "function": "readout.read_status_summary",
                        "arguments": {"reads": {"source": source, "output": "spots.csv"}}})
        rows[f"decode-{seed}"] = rows[f"assigned-{seed}"] = {"method": "white_tophat", "mode": "noise",
                                                             "side": "after"}
    status = main(["experiment", "evaluate", "--sources", str(_yaml(root / "sources.yaml", {"sources": sources})),
                   "--metrics", str(_yaml(root / "metrics.yaml", {"metrics": metrics})),
                   "--output", str(root / "evaluation")])
    assert status == 1  # a failed trial and an undefined F1: the evaluation is written and incomplete
    protocol = {
        "title": "White top-hat before and after, two fixed threshold modes",
        "qualification": "Pilot: synthetic development evidence; no default is recommended.",
        "route": "Single-stack detection trials through the private trial entry.",
        "inputs": "Hand-built 4 × 32 × 32 scenes.", "truth": "Four hand-placed points with genes a and b.",
        "dtype": "float32",
        "conditions": {"clean": "No added noise."},
        "cutoffs": [{"name": "matching radius", "value": 0.5, "unit": "voxel", "note": "inclusive, frozen"}],
        "missing_prerequisites": ["No segmentation adapter is evaluated here."],
        "rows": rows,
        "methods": [
            {"method": "white_tophat", "mode": "noise", "purpose": "Removes the flat background before detection.",
             "targeted_conditions": ["clean", "high_background"], "key_metrics": ["M03:f1", "M10", "M18:assigned"],
             "preconditions": [{"text": "Background is flat in the raw images.", "source_table": "values.csv"}],
             "harm": [{"text": "No harm test was run.", "source_table": "values.csv"}],
             "verdict": {"value": "provisional", "failing_clause": "seed 2 failed", "source_table": "values.csv"},
             "limitations": ["Two seeds only."]},
            {"method": "white_tophat", "mode": "adaptive", "purpose": "The same method in the adaptive mode.",
             "targeted_conditions": ["clean"], "key_metrics": ["M03:f1"]}],
        "findings": [{"label": "finding", "text": "F1 after is at least F1 before in every saved seed.",
                      "source_table": "values.csv"},
                     {"label": "hypothesis", "text": "The flat background raises spurious detections.",
                      "source_table": "values.csv"}],
        "figures": [{"method": "white_tophat", "mode": "noise", "title": "Seed 0 before and after",
                     "panels": [{"label": "before", "source": "noise-tophat-0", "input": "image"},
                                {"label": "after", "source": "noise-tophat-0", "output": "image.npy"}],
                     "detections": {"source": "noise-tophat-0", "output": "spots.csv"},
                     "truth": {"source": "truth", "file": "points"}, "peaks": 1}],
    }
    return {"root": root, "trials": trials, "truth": truth, "evaluation": root / "evaluation",
            "protocol": _yaml(root / "protocol.yaml", protocol)}


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    return _study(tmp_path_factory.mktemp("report") / "study")


@pytest.fixture
def rendered(study, tmp_path, monkeypatch):
    """The study's report, rendered with every processing and evaluation entry point patched to raise."""
    _no_processing(monkeypatch)
    out = R.render(study["evaluation"], tmp_path / "report", study["protocol"])
    return out, (out / "report.html").read_text(), json.loads((out / "report.json").read_text())


def _card(text, anchor):
    start = text.index(f"id=\"{anchor}\"")
    return text[start:text.index("</article>", start)]


def test_calibrated_cards_follow_the_page_order_and_cover_every_method(rendered):
    out, text, summary = rendered
    cards = [f"card-{NOISE}", f"card-{ADAPTIVE}"]
    figures = [f"fig-{NOISE}-metrics", f"fig-{NOISE}-1", f"fig-{ADAPTIVE}-metrics"]
    anchors = ["summary", "setup", "setups", "cards", *cards, "findings", "findings-list", "hypotheses",
               "reading-order", "method-details", f"details-{NOISE}", f"fig-{NOISE}-metrics",
               f"fig-{NOISE}-1", f"details-{ADAPTIVE}", f"fig-{ADAPTIVE}-metrics",
               "appendix", "identity", "revision", "a-verified", "a-sources", "a-values", "a-pairs", "a-resources",
               "a-diagnostics", "a-omitted"]
    positions = [text.index(f"id=\"{a}\"") for a in anchors]
    assert positions == sorted(positions)
    assert [a for a in summary["anchors"] if a in anchors] == anchors
    # Every card covers the (method, mode) pairs of the evaluation's rows, and the protocol's page order.
    assert [c["anchor"] for c in summary["cards"]] == cards
    values = pd.read_csv(out / "tables" / "values.csv")
    assert {(m, d) for m, d in zip(values["method"], values["mode"])} == {("white_tophat", "noise"),
                                                                          ("white_tophat", "adaptive")}
    # The reading order links every card and every figure, and each figure file is written.
    order = text[text.index("id=\"reading-order\""):text.index("id=\"method-details\"")]
    for anchor in [*cards, *figures, "setup", "findings", "hypotheses", "appendix"]:
        assert f"href=\"#{anchor}\"" in order
    for figure in summary["figures"]:
        assert (out / figure["path"]).read_bytes().startswith(b"\x89PNG")
        assert f"href=\"#{figure['anchor']}\"" in order
    assert "src=\"http" not in text and "href=\"http" not in text
    _links_resolve(text)


def _links_resolve(text):
    """Every id is unique and every in-page link names one."""
    ids = re.findall(r"\sid=\"([^\"]+)\"", text)
    assert len(ids) == len(set(ids))
    targets = set(re.findall(r"href=\"#([^\"]+)\"", text))
    assert targets and targets <= set(ids), targets - set(ids)


def test_every_method_mode_and_condition_has_report_text(study, rendered, tmp_path, monkeypatch):
    _out, text, _summary = rendered
    for anchor in (f"card-{NOISE}", f"card-{ADAPTIVE}"):
        card = _card(text, anchor)
        for phrase in ("Purpose.", "Targeted conditions.", "Key metrics", "Preconditions supplied.",
                       "Harm results supplied.", "Verdict", "Failures, null reasons and limitations."):
            assert phrase in card
    assert "Removes the flat background before detection." in text and "No added noise." in text
    assert "absent: high_background" in _card(text, f"card-{NOISE}")
    for sentence in R.LIMITATIONS:
        assert html_text(sentence) in text
    # Without a protocol every missing text is stated, not left out.
    _no_processing(monkeypatch)
    bare = (R.render(study["evaluation"], tmp_path / "bare") / "report.html").read_text()
    for phrase in ("No purpose supplied by a protocol.", "Not stated: no protocol supplied a pilot or scientific "
                   "qualification.", "no description supplied", "None: the protocol supplied no verdict, and the "
                   "report makes none."):
        assert phrase in bare
    # Without rows, the different threshold modes stay apart as trial setups in one card.
    assert "different setups, not pooled" in bare


def html_text(sentence):
    import html
    return html.escape(sentence, quote=True)


def test_known_values_keep_units_denominators_conditions_seeds_and_sources(rendered):
    out, text, _summary = rendered
    card = _card(text, f"card-{NOISE}")
    row = card[card.index("<b>M03</b> f1"):card.index("</tr>", card.index("<b>M03</b> f1"))]
    before, after = row.split("<td>")[3], row.split("<td>")[4]
    # F1 before: .5 (seed 0) and 1/3 (seed 1); after: 1 (seed 0) and .75 (seed 1); seed 2 failed.
    assert "<b>0.3333 to 0.5</b>" in before and "2 of 2 rows defined; seeds 0, 1" in before
    assert "missing seeds 2" in before
    assert "<b>0.75 to 1</b>" in after and "2 of 2 rows defined; seeds 0, 1, 2" in after
    assert "skipped at seed 2" in after and "the detector failed" in after
    assert "denominators: " in after and "total_reference 4" in after and "matched 3 to 4" in after
    assert "fraction" in row and "clean" in row and "values.csv rows" in row
    # The CSV keeps every row of values.csv, the failed one included, with its identity and unit.
    values = pd.read_csv(out / "tables" / "values.csv")
    saved = pd.read_csv(Path(_summary["evaluation"]["path"]) / "values.csv")
    assert len(values) == len(saved) and list(values["row"]) == list(range(1, len(saved) + 1))
    f1 = values[(values.value_name == "f1")].set_index("metric")
    assert f1.loc["f1-noise-raw-1", "value"] == pytest.approx(1 / 3, rel=1e-9, abs=1e-12)
    assert f1.loc["f1-noise-tophat-1", "value"] == pytest.approx(0.75, rel=1e-9, abs=1e-12)
    assert (f1.loc["f1-noise-raw-0", "unit"], f1.loc["f1-noise-raw-0", "seed"], f1.loc["f1-noise-raw-0", "side"],
            f1.loc["f1-noise-raw-0", "condition"]) == ("fraction", 0, "before", "clean")
    assert pd.isna(f1.loc["f1-adaptive-tophat-1", "value"])
    assert f1.loc["f1-adaptive-tophat-1", "reason"] == "zero denominator or no eligible observations"
    failed = values[values.metric == "f1-noise-tophat-2"]
    assert len(failed) == 1 and failed.outcome.item() == "skipped" and "the detector failed" in failed.reason.item()
    assert json.loads(f1.loc["f1-noise-tophat-1", "counts"])["matched"] == 3
    # The pairs table keeps the unpaired rows: adaptive seed 1 has no before row; noise seed 2 no before row.
    pairs = pd.read_csv(out / "tables" / "pairs.csv")
    unpaired = pairs[(pairs.value_name == "f1") & (pairs.pairing != "paired")]
    assert {(m, s, p) for m, s, p in zip(unpaired["mode"], unpaired["seed"], unpaired["pairing"])} == {
        ("adaptive", 1, "unpaired: no before row")}
    assert len(pairs[(pairs.selector == "M03") & pairs.value_name.isna()]) == 1  # the skipped seed 2 metric


def test_resource_scopes_missing_and_failures(study, rendered, tmp_path):
    out, text, _summary = rendered
    resources = pd.read_csv(out / "tables" / "resources.csv")
    trial = resources[resources.kind == "trial"].set_index("source")
    assert len(resources) == 9  # eight trials and the files source, outer-joined
    assert trial.loc["noise-tophat-1", "wall_s"] == pytest.approx(2.5, rel=1e-9, abs=1e-12)
    assert trial.loc["noise-tophat-2", "status"] == "failed" and trial.loc["noise-tophat-2", "peak_rss_kib"] == 52000
    assert trial["scope"].str.startswith("trial process").all()
    files = resources[resources.kind == "files"].iloc[0]
    assert pd.isna(files.wall_s) and files.reason == "a files source holds no resource record"
    assert "trial process" in text and "no resource record" in text
    # A workflow task's scope is kept apart from a trial's, and a task without a record keeps its row and reason.

    class Evaluation:
        sources = {"run": {"name": "run", "kind": "workflow", "status": "verified",
                           "tasks": {"fov/s/FOV_001": {"record": {}}, "fov/s/FOV_002": {"record": None}}},
                   "t": {"name": "t", "kind": "trial", "status": "failed", "error": {"message": "record.json is "
                                                                                               "missing"}}}

    class Records:
        trials = {}
        tasks = {("run", "fov/s/FOV_001"): {"status": "succeeded",
                                             "workflow": {"wall_s": 3.0, "cpu_s": 2.0, "peak_rss_kib": 9}}}

        def trial_identity(self, name):
            return {}
    table = R.resource_rows(Evaluation(), Records())
    assert list(table["task"]) == ["fov/s/FOV_001", "fov/s/FOV_002", None]
    assert table["scope"].iloc[0].startswith("workflow task process") and table["wall_s"].iloc[0] == 3.0
    assert table["reason"].iloc[1] == "no task record" and table["wall_s"].iloc[1] is None
    assert table["reason"].iloc[2] == "no checked trial record: record.json is missing"
    assert table["scope"].iloc[2].startswith("trial process")
    # Output refusal: a nonempty folder, a folder inside the evaluation, or inside a source.
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "x").write_text("x")
    with pytest.raises(FileExistsError):
        R.render(study["evaluation"], busy)
    with pytest.raises(R.ReportError, match="inside the evaluation or one of its sources"):
        R.render(study["evaluation"], study["evaluation"] / "report")
    with pytest.raises(R.ReportError, match="inside the evaluation or one of its sources"):
        R.render(study["evaluation"], study["trials"] / "noise-raw-0" / "report")


def test_ranges_after_channel_aggregation():
    """A list-valued value (one item per channel) enters the range as its mean, per row; the range is across seeds
    after that aggregation, as the preprocessing comparison did."""
    rows = []
    for seed in (0, 100):
        for side, items in (("before", [9.0, 10.0]), ("after", [2.0, 4.0 + seed / 100])):
            number, count = R._number(items)
            rows.append({"row": len(rows) + 1, "metric": f"m-{seed}-{side}", "selector": "M09", "value_name": "mad",
                         "unit": "intensity", "parameter_set": "", "condition": "baseline", "seed": seed,
                         "fov": None, "side": side, "setup": "setup 1", "trial_method": "log", "number": number,
                         "items": count, "raw": items, "value": json.dumps(items), "reason": None,
                         "outcome": "computed", "counts": "{}"})
    frame = pd.DataFrame(rows, dtype=object)
    (entry,) = R.key_table(frame)
    before, after = entry["sides"]["before"][0], entry["sides"]["after"][0]
    assert before["range"] == "9.5" and before["items"] == [2] and before["seeds"] == "0, 100"
    assert after["range"] == "3 to 3.5"
    assert "(mean over 2 items per row)" in R._side_text(entry["sides"]["before"])
    assert R._number([1.0, None]) == (None, None) and R._number([True, False]) == (None, None)


def test_saved_only_rendering_and_unchanged_sources(study, tmp_path, monkeypatch):
    before = _tree(study["trials"], study["truth"], study["evaluation"])
    _no_processing(monkeypatch)
    out = R.render(study["evaluation"], tmp_path / "report", study["protocol"])
    assert _tree(study["trials"], study["truth"], study["evaluation"]) == before
    summary = json.loads((out / "report.json").read_text())
    # Every file the report read was checked as it was read; each trial file against its record's SHA-256.
    verified = {Path(v["path"]).name: v for v in summary["verified"]}
    assert verified["evaluation.json"]["sha256"] == _sha(study["evaluation"] / "evaluation.json")
    shown = [v for v in summary["verified"] if v["what"].startswith(("output", "input", "file"))]
    assert {Path(v["path"]).name for v in shown} == {"input.npy", "image.npy", "spots.csv", "points.csv"}
    assert all(v["checked_against"] == "recorded SHA-256" and v["sha256"] == _sha(v["path"]) for v in shown)
    assert {o["path"] for o in summary["outputs"]} >= {"report.html", "tables/values.csv", "tables/pairs.csv"}
    for entry in summary["outputs"]:
        assert _sha(out / entry["path"]) == entry["sha256"]


def _copy(study, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(study["root"], copy)
    for path in [copy / "evaluation" / "evaluation.json", copy / "evaluation" / "values.csv",
                 copy / "protocol.yaml",
                 *(copy / "trials").glob("*/record.json")]:
        path.write_text(path.read_text().replace(str(study["root"]), str(copy)))
    return copy


def test_calibrated_report_puts_the_summary_first_and_refuses_unverified_inputs(study, rendered, tmp_path,
                                                                                monkeypatch):
    _out, text, _summary = rendered
    assert text.index("id=\"summary\"") < text.index("id=\"method-details\"") < text.index("id=\"appendix\"")
    for phrase in ("Pilot: synthetic development evidence", "<span class=\"label\">finding</span>",
                   "<span class=\"label\">hypothesis</span>", "provisional; failing clause: seed 2 failed"):
        assert phrase in text
    _no_processing(monkeypatch)
    # The copied evaluation renders, then each tampered file refuses the report and nothing is written.
    copy = _copy(study, tmp_path)
    evaluation = json.loads((copy / "evaluation" / "evaluation.json").read_text())
    for source in evaluation["sources"]:
        if source.get("record"):
            source["record"]["sha256"] = _sha(source["record"]["path"])
    (copy / "evaluation" / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
    R.render(copy / "evaluation", tmp_path / "fine", copy / "protocol.yaml")
    damages = [
        (copy / "evaluation" / "values.csv", lambda b: b.replace(b"0.75", b"0.76", 1), "values.csv differs"),
        (copy / "trials" / "noise-raw-0" / "record.json", lambda b: b.replace(b"1.5", b"1.6", 1),
         "differs from its recorded SHA-256"),
        (copy / "trials" / "noise-tophat-0" / "image.npy", lambda b: b[:-4] + b"\0\0\0\1",
         "differs from its recorded SHA-256"),
        (copy / "truth" / "points.csv", lambda b: b.replace(b"24", b"25", 1), "differs from its recorded SHA-256"),
    ]
    for path, damage, message in damages:
        original = path.read_bytes()
        path.write_bytes(damage(original))
        with pytest.raises(R.SourceMismatch, match=message):
            R.render(copy / "evaluation", tmp_path / "refused", copy / "protocol.yaml")
        assert not (tmp_path / "refused").exists()
        path.write_bytes(original)
    (copy / "evaluation" / "evaluation.json").write_text("{\"format\": \"other\"}")
    with pytest.raises(R.SourceMismatch, match="is not a starfinder.experiment.evaluation/1 record"):
        R.render(copy / "evaluation", tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


def test_sources_are_verified_against_the_manifest(study, tmp_path):
    """The report reads a file only through a record that names its SHA-256; an unlisted, missing or changed file
    refuses the report, and each table names its source."""
    saved = R.Saved()
    data = (study["truth"] / "points.csv").read_bytes()
    assert saved.read(study["truth"] / "points.csv", "truth", "sha256:" + _sha(study["truth"] / "points.csv")) == data
    with pytest.raises(R.SourceMismatch, match="missing"):
        saved.read(study["truth"] / "absent.csv", "truth", "0" * 64)
    with pytest.raises(R.SourceMismatch, match="has 3 bytes"):
        (tmp_path / "small.csv").write_bytes(b"a,b")
        saved.read(tmp_path / "small.csv", "a table", size=4)
    evaluation = R.SavedEvaluation(study["evaluation"], saved)
    records = R.Records(evaluation, saved)
    with pytest.raises(R.SourceMismatch, match="lists no output 'labels.npy'"):
        records.file({"source": "noise-raw-0", "output": "labels.npy"})
    with pytest.raises(R.ReportError, match="the evaluation does not have"):
        records.file({"source": "other", "output": "image.npy"})
    with pytest.raises(R.ReportError, match="has no file 'codebook'"):
        records.file({"source": "truth", "file": "codebook"})
    frame = pd.DataFrame({"metric": ["a"], "value": [1.0]})
    rendered = R.full_table(frame, "tables/values.csv", "t")
    assert "Source: <code>tables/values.csv</code>" in rendered


def test_calibrated_scenes_outputs_and_detections_match_the_saved_run(rendered, study):
    _out, text, summary = rendered
    (figure,) = [f for f in summary["figures"] if f["anchor"] == f"fig-{NOISE}-1"]
    facts = figure["facts"]
    detections = pd.read_csv(study["trials"] / "noise-tophat-0" / "spots.csv")
    assert facts["detections"] == len(detections) == 4
    raw, after = _image(0, "raw"), _image(0, "tophat")
    assert facts["display_range"] == [float(min(raw.min(), after.min())), float(max(raw.max(), after.max()))]
    assert facts["panels"] == ["before", "after"] and facts["peaks_from"] == "after"
    assert "the 4 saved detections" in text


def test_dim_punctum_is_the_lowest_realized_peak_and_leads_the_colour_view(rendered):
    volume = _image(1, "tophat")
    points = np.asarray(TRUTH, dtype=float)
    peaks = R.realized_peaks(points, volume)
    assert list(peaks) == [41.0, 51.0, 61.0, 71.0]
    dim = R.dim_punctum(points, volume)
    assert dim == {"index": 0, "center": (2, 8, 8), "peak": 41.0}
    chosen = R.representative_puncta(peaks)
    assert chosen[0] == dim["index"] and len(set(chosen)) == len(chosen) <= 4
    # A punctum at the edge uses the clipped box; ties go to the first point.
    edge = np.zeros((2, 4, 4))
    edge[0, 0, 0] = 3
    edge[1, 3, 3] = 3
    assert R.dim_punctum([(0, 0, 0), (1, 3, 3)], edge)["index"] == 0
    _out, _text, summary = rendered
    (figure,) = [f for f in summary["figures"] if f["anchor"] == f"fig-{NOISE}-1"]
    assert figure["facts"]["dim_punctum"] == {"index": 0, "center": [2, 8, 8], "peak": 40.0}
    assert figure["facts"]["puncta"][0]["index"] == 0


def test_method_figure_renders_panels_for_a_tiny_scene():
    before, after = _image(0, "raw"), _image(0, "tophat")
    detections = np.asarray(TRUTH[:2], dtype=float)
    png, facts = R.method_figure([("before", before, None), ("after", after, None)], detections,
                                 np.asarray(TRUTH, dtype=float), peaks=0)
    assert png.startswith(b"\x89PNG") and facts["detections"] == 2
    assert facts["display_range"] == [0.0, float(before.max())]
    assert facts["dim_punctum"]["center"] == (2, 8, 8) and json.dumps(facts, default=str)
    with pytest.raises(R.ReportError, match="ZYX, ZYXC or YX"):
        R.method_figure([("bad", np.zeros(4), None)])


def test_collapsed_pr_curves_are_detected(tmp_path):
    flat = pd.DataFrame(dict(threshold=[2.0, 3.0, 4.0], precision=[0.5] * 3, recall=[0.4] * 3))
    assert R.collapsed(flat)
    assert not R.collapsed(flat.assign(recall=[0.4, 0.3, 0.2]))
    assert not R.collapsed(flat.iloc[:1])

    class Evaluation:
        details = {"auprc": {"curve": flat.to_dict(orient="records")}, "f1": {"pairs": []}}
    assert list(R.curves(Evaluation())) == ["auprc"]


def test_full_tables_keep_every_row_and_column_within_one_capture():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"metric": ["white_tophat"] * 400, "seed": np.arange(400),
                          **{f"value_{k:02d}_before": rng.normal(size=400) for k in range(60)},
                          "reason": ["zero denominator or no eligible observations"] * 400})
    text = R.full_table(frame, "t.csv", "a-t")
    widths = [int(w) for w in re.findall(r"class=\"full\" id=\"a-t-r\d+c\d+\" style=\"width:(\d+)px\"", text)]
    anchors = re.findall(r"id=\"(a-t-r\d+c\d+)\"", text)
    assert len(widths) == len(anchors) > 1 and len(set(anchors)) == len(anchors)
    assert max(widths) <= R.FULL_WIDTH_PX
    groups = len({a.split("c")[-1] for a in anchors})
    blocks = len({a.split("r")[-1].split("c")[0] for a in anchors})
    assert blocks == 400 // R.BLOCK_ROWS and text.count("<tr>") - text.count("<thead><tr>") == len(frame) * groups
    for column in frame.columns:
        assert f"<th>{column}</th>" in text
    assert text.count("<th>metric</th>") == len(anchors)  # the key columns repeat in every piece


def test_undefined_is_visible(tmp_path, monkeypatch):
    """A translation evaluated without a tolerance has an undefined pass; the report shows it, with its reason."""
    folder = tmp_path / "files"
    folder.mkdir()
    (folder / "shift.json").write_text(json.dumps({"detected": [0.125, 0, 0], "truth": [0, 0, 0]}))
    sources = {"sources": [{"name": "shift", "kind": "files", "files": {"shift": {
        "path": str(folder / "shift.json"), "sha256": _sha(folder / "shift.json")}}}]}
    metrics = {"metrics": [{"name": "translation", "function": "registration.evaluate_translation",
                            "parameters": {"units": "voxel", "tolerance": None},
                            "arguments": {"detected": {"mapping": {"r": {"source": "shift", "file": "shift",
                                                                          "key": "detected"}}},
                                          "truth": {"mapping": {"r": {"source": "shift", "file": "shift",
                                                                       "key": "truth"}}},
                                          "reference_metadata": VOXEL, "observed_metadata": VOXEL}}]}
    assert main(["experiment", "evaluate", "--sources", str(_yaml(tmp_path / "s.yaml", sources)),
                 "--metrics", str(_yaml(tmp_path / "m.yaml", metrics)), "--output", str(tmp_path / "ev")]) == 0
    _no_processing(monkeypatch)
    text = (R.render(tmp_path / "ev", tmp_path / "report") / "report.html").read_text()
    card = _card(text, f"card-{UNASSIGNED}")
    assert "<span class=\"undefined\">undefined</span>" in card and "tolerance gate not requested" in card
    assert "<b>0.125</b>" in card
    assert "passed is undefined: tolerance gate not requested" in text


def test_report_never_chooses_an_optimum_pools_methods_hides_fovs_or_takes_assigned_reads_as_accuracy(
        study, rendered, tmp_path, monkeypatch):
    out, text, summary = rendered
    noise, adaptive = _card(text, f"card-{NOISE}"), _card(text, f"card-{ADAPTIVE}")
    # Two fixed modes, two cards: each holds its own rows only, and no range crosses the modes.
    values = pd.read_csv(out / "tables" / "values.csv")
    assert set(values[values.metric.str.contains("adaptive")]["mode"]) == {"adaptive"}
    assert "seeds 0, 1, 2" not in adaptive and "<b>1</b>" in adaptive  # adaptive after: seed 0 is 1, seed 1 null
    # The missing seed is shown, never dropped: adaptive has no before row for seed 1.
    assert "missing seeds 1" in adaptive and "undefined at seed 1: zero denominator" in adaptive
    assert any("adaptive" in f["text"] and "has no row for seeds 1" in f["text"] for f in summary["findings"])
    # No optimum, best or selected threshold anywhere; the cut-offs are the protocol's fixed ones.
    lowered = re.sub(r"<[^>]+>", " ", text).lower()
    for word in ("optimum threshold", "optimal", "best ", "selected threshold", "recommended default"):
        assert word not in lowered.replace("selects no threshold, default or optimum", "")
    assert "none was selected from these results" in text
    # Assigned reads are a count over all decoded reads, not accuracy; accuracy (M10) has its truth population.
    assert "a count, not accuracy" in text and "population: all decoded reads" in noise
    assert "population: all truth molecules" in noise and "reads_wrong_gene 0 to 1" in noise
    # Differing resource scopes are kept apart, and a protocol row cannot contradict a saved seed.
    protocol = yaml.safe_load(study["protocol"].read_text())
    protocol["rows"]["f1-noise-raw-0"]["seed"] = 7
    with pytest.raises(R.ReportError, match="gives seed 7, the saved record 0"):
        _no_processing(monkeypatch)
        R.render(study["evaluation"], tmp_path / "x", _yaml(tmp_path / "p.yaml", protocol))
    # A row of another trial setup assigned to a card's side is shown apart, not pooled.
    protocol = yaml.safe_load(study["protocol"].read_text())
    protocol["rows"]["f1-adaptive-raw-0"] = {"method": "white_tophat", "mode": "noise", "side": "before"}
    text = (R.render(study["evaluation"], tmp_path / "y", _yaml(tmp_path / "q.yaml", protocol)) /
            "report.html").read_text()
    assert "different setups, not pooled" in _card(text, f"card-{NOISE}")
    # A verdict without its failing clause is refused.
    protocol = yaml.safe_load(study["protocol"].read_text())
    del protocol["methods"][0]["verdict"]["failing_clause"]
    with pytest.raises(R.ReportError, match="failing_clause"):
        R.render(study["evaluation"], tmp_path / "z", _yaml(tmp_path / "r.yaml", protocol))
    protocol["methods"][0].pop("verdict")
    protocol["findings"][0]["label"] = "fact"
    with pytest.raises(R.ReportError, match="finding or hypothesis"):
        R.render(study["evaluation"], tmp_path / "z", _yaml(tmp_path / "r.yaml", protocol))
    # Two workflow sources of different configurations are not pooled, and a FOV with no row is named missing.
    folder = _workflow_evaluation(tmp_path / "workflow")
    text = (R.render(folder, tmp_path / "w", revision_check=_revision) / "report.html").read_text()
    card = _card(text, f"card-{UNASSIGNED}")
    assert "different setups, not pooled" in card and "0.2 to 0.8" not in card
    assert "<b>0.2</b>" in card and "<b>0.8</b>" in card and "missing FOVs FOV_002" in card


# --- Identity, sides and links (the review of 8f57c63) ----------------------------------------------------------

def _revision(code):
    return {"revision": "0" * 40, "head": "0" * 40, "dirty": False, "version": None, "sources": [],
            "resolved": True, "changed": [], "match": "not checked in this test"}


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1))
    return {"path": str(path), "sha256": _sha(path)}


def _workflow_evaluation(root, specification="kept"):
    """A saved evaluation in the approved schema, written by hand: two workflow sources (runs A and B) whose
    manifests and task records name different configuration and blocks hashes; M17's unmatched rate 0.2 for run A's
    FOV_001 and 0.8 for run B's FOV_001; and run A's FOV_002 metric, skipped before it recorded its arguments (its
    task is incomplete), whose task only the recorded metrics specification names. ``specification`` is ``kept``,
    ``changed`` (its bytes differ from the recorded SHA-256) or ``missing``."""
    sources, configuration = [], {"A": "a" * 64, "B": "b" * 64}
    for run in ("A", "B"):
        folder = root / f"run{run}"
        # blocks_sha256 depends on the configuration and the task kind only (workflow/_shape.py), so one per run.
        tasks = [{"id": f"fov/s/{fov}", "kind": "fov", "identity": {"sample": "s", "fov": fov},
                  "blocks_sha256": f"{run.lower()}1" * 32} for fov in ("FOV_001", "FOV_002")]
        manifest = _write_json(folder / "workflow" / "manifest.json", {
            "format": "test", "roots": {"output": str(folder)}, "tasks": tasks,
            "configuration": {"configuration_sha256": configuration[run]}})
        summary = {}
        for task in tasks:
            complete = task["identity"]["fov"] == "FOV_001"
            record = None
            if complete:
                record = _write_json(folder / "records" / f"{task['identity']['fov']}.json", {
                    "id": task["id"], "kind": "fov", "identity": task["identity"], "status": "succeeded",
                    "workflow": {"configuration_sha256": configuration[run], "blocks_sha256": task["blocks_sha256"],
                                 "wall_s": 1.0, "cpu_s": 1.0, "peak_rss_kib": 10}})
                record["status"] = "succeeded"
            summary[task["id"]] = {"kind": "fov", "complete": complete,
                                   "problems": [] if complete else ["the task record is missing"], "record": record}
        sources.append({"name": f"run{run}", "kind": "workflow", "status": "verified", "error": None,
                        "limitations": [], "consumed": [], "manifest": manifest, "roots": {"output": str(folder)},
                        "tasks": summary, "incomplete": ["fov/s/FOV_002"]})

    def computed(name, run, value):
        return {"name": name, "metric": "M17", "function": "readout.read_status_summary", "parameters": {},
                "arguments": {"reads": {"source": f"run{run}", "task": "fov/s/FOV_001", "output": "reads.csv"}},
                "sources": [f"run{run}"], "outcome": "computed", "status": "ok",
                "values": {"unmatched_rate": value}, "units": {"unmatched_rate": "fraction"},
                "counts": {"total": 10}, "reasons": {}, "config": {}}
    metrics = [computed("rate-A-1", "A", 0.2), computed("rate-B-1", "B", 0.8),
               {"name": "rate-A-2", "metric": "M17", "function": "readout.read_status_summary", "parameters": {},
                "arguments": {}, "sources": ["runA"], "outcome": "skipped",
                "reason": "workflow task fov/s/FOV_002 is not complete: the task record is missing"}]
    specified = {"metrics": [{"name": m["name"], "metric": "M17", "function": "readout.read_status_summary",
                              "arguments": {"reads": {"source": "runA" if "A" in m["name"] else "runB",
                                                      "task": "fov/s/FOV_00" + m["name"][-1],
                                                      "output": "reads.csv"}}} for m in metrics]}
    path = root / "metrics.yaml"
    spec = _write_json(path, specified)  # JSON is YAML
    if specification == "changed":
        path.write_text(path.read_text().replace("FOV_002", "FOV_003"))
    elif specification == "missing":
        path.unlink()
    folder = root / "evaluation"
    record = {"format": R.EVALUATION_FORMAT, "generated_at": "2026-11-04T00:00:00+00:00",
              "software": {"code": {"git_commit": None}}, "inputs": {"sources": spec, "metrics": spec},
              "sources": sources, "metrics": metrics, "status": "incomplete",
              "counts": {"sources": 2, "failed_sources": 0, "metrics": 3, "computed": 2, "skipped": 1,
                         "unavailable": 0, "failed": 0}}
    _write_json(folder / "evaluation.json", record)
    pd.DataFrame(R.value_rows(metrics), columns=R.VALUE_COLUMNS).to_csv(folder / "values.csv", index=False)
    return folder


def test_workflow_setups_are_not_pooled_and_a_skipped_metric_keeps_its_fov(tmp_path, monkeypatch):
    _no_processing(monkeypatch)
    out = R.render(_workflow_evaluation(tmp_path / "kept"), tmp_path / "report", revision_check=_revision)
    text = (out / "report.html").read_text()
    values = pd.read_csv(out / "tables" / "values.csv").set_index("metric")
    # The skipped metric's FOV comes from the checked specification and the checked manifest's task entry.
    assert values.loc["rate-A-2", "fov"] == "FOV_002" and values.loc["rate-A-2", "outcome"] == "skipped"
    assert values.loc["rate-A-1", "fov"] == "FOV_001" and values.loc["rate-B-1", "fov"] == "FOV_001"
    assert values.loc["rate-A-1", "setup"] == values.loc["rate-A-2", "setup"] != values.loc["rate-B-1", "setup"]
    card = _card(text, f"card-{UNASSIGNED}")
    assert "skipped at FOV FOV_002" in card and "missing FOVs FOV_002" in card  # run B has no FOV_002 row
    assert "different setups, not pooled" in card and "0.2 to 0.8" not in card
    setups = text[text.index("id=\"setups\""):text.index("<b>Conditions.</b>")]
    assert setups.count("configuration_sha256") == 2 and "a" * 64 in setups and "b" * 64 in setups
    summary = json.loads((out / "report.json").read_text())
    assert any(v["what"] == "the evaluation's metrics specification" and v["checked_against"] == "recorded SHA-256"
               for v in summary["verified"])
    assert any(v["what"].startswith("workflow manifest") for v in summary["verified"])
    _links_resolve(text)
    # A changed or missing specification leaves the FOV empty, with a finding that says why; nothing is refused.
    for case, why in (("changed", "differs from its recorded SHA-256"), ("missing", "is missing")):
        out = R.render(_workflow_evaluation(tmp_path / case, case), tmp_path / f"report-{case}",
                       revision_check=_revision)
        values = pd.read_csv(out / "tables" / "values.csv").set_index("metric")
        assert pd.isna(values.loc["rate-A-2", "fov"])
        findings = json.loads((out / "report.json").read_text())["findings"]
        assert any(f["text"].startswith("Metric rate-A-2: it was skipped before recording which workflow task")
                   and why in f["text"] and "its FOV is unknown" in f["text"] for f in findings)
    # A changed manifest refuses the report, as a changed trial record does.
    folder = _workflow_evaluation(tmp_path / "manifest")
    manifest = tmp_path / "manifest" / "runA" / "workflow" / "manifest.json"
    manifest.write_text(manifest.read_text().replace("FOV_002", "FOV_009"))
    with pytest.raises(R.SourceMismatch, match="differs from its recorded SHA-256"):
        R.render(folder, tmp_path / "refused", revision_check=_revision)


def test_a_value_without_a_side_is_shown_beside_before_and_after(tmp_path, monkeypatch):
    """The review's probe: before 0.2 and a selected value without a side, 0.8, in one card and condition."""
    folder = _workflow_evaluation(tmp_path / "study")
    protocol = _yaml(tmp_path / "protocol.yaml", {
        "rows": {"rate-A-1": {"method": "m", "side": "before"}, "rate-B-1": {"method": "m"},
                 "rate-A-2": {"method": "m", "side": "after"}},
        "methods": [{"method": "m", "key_metrics": ["M17:unmatched_rate"]}]})
    _no_processing(monkeypatch)
    out = R.render(folder, tmp_path / "report", protocol, revision_check=_revision)
    card = _card((out / "report.html").read_text(), f"card-{R.card_slug(('m', None))}")
    row = card[card.index("<b>M17</b> unmatched_rate"):]
    row = row[:row.index("</tr>")]
    before, after, single = row.split("<td>")[3:6]
    assert "<b>0.2</b>" in before and "skipped at FOV FOV_002" in after and "<b>0.8</b>" in single
    assert "without a side" in card


def test_card_identifiers_stay_distinct_and_every_link_resolves(tmp_path, monkeypatch):
    keys = [("x-y", "z"), ("x", "y-z"), ("a b", None), ("a-b", None)]
    slugs = [R.card_slug(k) for k in keys]
    assert len(set(slugs)) == 4 and slugs[0].startswith("x-y-z-") and slugs[2].startswith("a-b-")
    folder = _workflow_evaluation(tmp_path / "study")
    rows = {"rate-A-1": {"method": "x-y", "mode": "z"}, "rate-B-1": {"method": "x", "mode": "y-z"},
            "rate-A-2": {"method": "a b"}}
    _no_processing(monkeypatch)
    out = R.render(folder, tmp_path / "report", _yaml(tmp_path / "p.yaml", {"rows": rows}),
                   revision_check=_revision)
    text = (out / "report.html").read_text()
    summary = json.loads((out / "report.json").read_text())
    anchors = [c["anchor"] for c in summary["cards"]]
    assert anchors == [f"card-{R.card_slug(k)}" for k in keys[:3]]
    figures = [f["path"] for f in summary["figures"]]
    assert len(figures) == len(set(figures)) == 2 and all((out / f).is_file() for f in figures)
    for slug in slugs[:3]:
        assert f"id=\"details-{slug}\"" in text and f"href=\"#card-{slug}\"" in text
    _links_resolve(text)
    # The backstop: two distinct keys that would share an identifier refuse the report.
    monkeypatch.setattr(R, "card_slug", lambda key: "same")
    with pytest.raises(R.ReportError, match="would share the identifier 'same'"):
        R.render(folder, tmp_path / "refused", _yaml(tmp_path / "p.yaml", {"rows": rows}),
                 revision_check=_revision)
    assert not (tmp_path / "refused").exists()


# --- Revision -------------------------------------------------------------------------------------------------------

def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.org",
                           *args], capture_output=True, check=True).stdout.decode().strip()


def _repo(tmp_path):
    source = tmp_path / "src" / "python" / "starfinder" / "evaluation" / "metric.py"
    source.parent.mkdir(parents=True)
    source.write_text("version = 1\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "first")
    return source, _git(tmp_path, "rev-parse", "HEAD")


def test_revision_check_labels_a_dirty_evaluation_unresolved_and_refuses_changed_sources(tmp_path):
    """The saved evaluation records git_dirty but no checksum of the changes (``_records.code``), so a dirty
    evaluation cannot be resolved to the commit that holds its diff, as the old preprocessing report did: it is
    labelled unresolved instead. A clean evaluation's sources must be unchanged since its commit."""
    source, first = _repo(tmp_path)
    (tmp_path / "notes.md").write_text("later, unrelated\n")
    _git(tmp_path, "add", "notes.md")
    _git(tmp_path, "commit", "-q", "-m", "later")
    clean = R.check_revision({"git_commit": first, "git_dirty": False}, tmp_path)
    assert clean["resolved"] and clean["changed"] == [] and clean["head"] != first
    dirty = R.check_revision({"git_commit": first, "git_dirty": True}, tmp_path)
    assert not dirty["resolved"] and dirty["match"].startswith("unresolved")
    source.write_text("version = 2\n")
    with pytest.raises(R.RevisionMismatch, match="evaluation sources changed"):
        R.check_revision({"git_commit": first, "git_dirty": False}, tmp_path)
    source.write_text("version = 1\n")
    (source.parent / "new.py").write_text("x = 1\n")
    with pytest.raises(R.RevisionMismatch, match="new.py"):
        R.check_revision({"git_commit": first, "git_dirty": False}, tmp_path)


def test_revision_check_rejects_a_commit_outside_the_history(tmp_path):
    _source, _first = _repo(tmp_path)
    _git(tmp_path, "checkout", "-q", "-b", "side")
    (tmp_path / "a.txt").write_text("b\n")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-q", "-m", "side")
    side = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "checkout", "-q", "-")
    with pytest.raises(R.RevisionMismatch, match="not an ancestor"):
        R.check_revision({"git_commit": side, "git_dirty": False}, tmp_path)
    with pytest.raises(R.RevisionMismatch, match="not a commit"):
        R.check_revision({"git_commit": "0" * 40, "git_dirty": False}, tmp_path)
    with pytest.raises(R.RevisionMismatch, match="no source revision"):
        R.check_revision({"git_commit": None}, tmp_path)


def test_a_revision_failure_refuses_the_report(study, tmp_path):
    def refuse(code):
        raise R.RevisionMismatch("the evaluation records no source revision")
    with pytest.raises(R.RevisionMismatch):
        R.render(study["evaluation"], tmp_path / "report", revision_check=refuse)
    assert not (tmp_path / "report").exists()


# --- Command and example --------------------------------------------------------------------------------------------

def _exit(capsys, *argv):
    with pytest.raises(SystemExit) as error:
        main(list(argv))
    return error.value.code, capsys.readouterr()


def test_report_command_help_and_exit_status(study, tmp_path, capsys):
    status, captured = _exit(capsys, "experiment", "report", "--help")
    text = " ".join(captured.out.split())
    assert status == 0 and captured.out.startswith("usage: starfinder experiment report")
    assert "Exit 0" in text and "2 for" in text and "--protocol" in text
    assert main(["experiment", "report", "--evaluation", str(study["evaluation"]), "--output",
                 str(tmp_path / "report"), "--protocol", str(study["protocol"])]) == 0
    assert capsys.readouterr().out.strip() == str((tmp_path / "report" / "report.html").resolve())
    status, captured = _exit(capsys, "experiment", "report", "--evaluation", str(study["evaluation"]), "--output",
                             str(tmp_path / "report"))
    assert status == 2 and "new or empty" in captured.err
    bad = _yaml(tmp_path / "bad.yaml", {"verdicts": []})
    status, captured = _exit(capsys, "experiment", "report", "--evaluation", str(study["evaluation"]), "--output",
                             str(tmp_path / "other"), "--protocol", str(bad))
    assert status == 2 and "unsupported keys ['verdicts']" in captured.err and not (tmp_path / "other").exists()


def test_saved_comparison_example(tmp_path, monkeypatch, capsys):
    """The bounded example of docs/benchmark/report.md, command by command, in a new folder outside Git."""
    import runpy
    page = (DOCS / "benchmark" / "report.md").read_text()
    block = page.split("```bash\n", 1)[1].split("```", 1)[0]
    lines = [line.replace("/tmp/starfinder-report", str(tmp_path / "example")).split()
             for line in block.strip().splitlines()]
    assert [line[:3] for line in lines] == [["uv", "run", "python"]] + [["uv", "run", "starfinder"]] * 6
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr("sys.argv", lines[0][3:])
    runpy.run_path(lines[0][3], run_name="__main__")
    statuses = [main(line[3:]) for line in lines[1:]]
    # Expansion and two trials succeed, one trial fails, the evaluation is incomplete, the report is written.
    assert statuses == [0, 0, 0, 1, 1, 0]
    capsys.readouterr()
    text = (tmp_path / "example" / "report" / "report.html").read_text()
    assert "<span class=\"undefined\">undefined</span>" in text and "tolerance gate not requested" in text
    assert "class=\"error\">skipped" in text and "card-translation" in text
