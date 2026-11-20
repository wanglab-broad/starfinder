"""The segmentation background and the object sizes of the report (W-884).

design/specs/report-algorithms.md ("Segmentation previews", "Object sizes"). The example FOVs are drawn over the
first channel of the segmentation input the run saved (route per FOV: ``fovs/<fov>/segmentation/<run>/input.ome.tif``;
route after fusion: the fused image ``export.json`` ``segmentation.runs[*].inputs`` names), not over ``reference.tif``
or the first fused image. "Object sizes" gives the equivalent diameter of the nuclei and cells across the sample and
the fraction of nuclei below a lower limit, a display rule that marks FOVs under Problematic FOVs.

The hand-built FOVs hold squares of known side on a 0.5 µm grid, so the equivalent diameter of a square of ``s``
pixels is ``s / sqrt(pi)`` µm: 4 px 2.2568, 6 px 3.3851, 8 px 4.5135, 10 px 5.6419 µm. The chain tests use the
synthetic scene (``scene_chain_stain``) and skip without the export extras.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from starfinder.io import save_volume
from starfinder.report import PROBLEM_RULES, ReportConfig, ReportLimits, RuleLimit
from starfinder.report._collect import report_examples
from starfinder.report._rules import RuleInputs, evaluate_rules
from starfinder.report._sizes import centroids_yx, equivalent_diameters, size_row
from starfinder.report._values import NA, Unavailable

from . import hash_oracles
from .report_fixtures import report_records
from .report_scene import (require_extras, scene_chain_stain,  # noqa: F401 (module-scoped fixtures)
                           scene_chain_stain_merged)
from .report_stain_runs import export_after_fusion_on_stain
from .test_report_collection import collect, keyed

pytestmark = [pytest.mark.report, pytest.mark.contract]

TOL = 1e-9
SPACING = (1.0, 0.5, 0.5)
FOVS = ("F1", "F2", "F3", "F4")
#: The side in pixels of each nucleus of each FOV; those of 4 px (2.2568 µm) are below 3 µm.
SIDES = {"F1": (6, 8, 10), "F2": (6, 8, 10, 8), "F3": (4, 4, 4, 8, 6), "F4": (8, 10, 8, 6)}
SHAPE = (1, 16, 64)
NUCLEUS_UM = {s: s / math.sqrt(math.pi) for s in (4, 6, 8, 10)}


# --- Equivalent diameters --------------------------------------------------------------------------------------

def squares(sides, shape=SHAPE, pad=0):
    """A label image with one square per side, labels 1…n in a row, 12 px apart, from row 2 and column 1."""
    labels = np.zeros(shape, dtype=np.uint32)
    for k, side in enumerate(sides):
        labels[0, 2:2 + side + pad, 1 + 12 * k:1 + 12 * k + side + pad] = k + 1
    return labels


def test_equivalent_diameters_of_squares_on_a_plane():
    labels = squares((4, 8, 12))
    labels[labels == 2] = 5          # the labels need not be consecutive; ids come back in increasing order
    ids, diameters = equivalent_diameters(labels, SPACING)
    assert ids.tolist() == [1, 3, 5]
    # Areas 4 (label 1), 36 (label 3) and 16 µm² (label 5): the circle of the same area has d = 2 sqrt(A / pi).
    assert diameters == pytest.approx([2.256758334191025, 6.770275002573076, 4.51351666838205], abs=1e-9)
    assert (equivalent_diameters(np.zeros(SHAPE, dtype=np.uint32), SPACING)[1] == []).all()


def test_equivalent_diameters_of_a_volume_use_the_volume():
    labels = np.zeros((6, 8, 8), dtype=np.uint32)
    labels[1:5, 2:6, 2:6] = 7        # 4 x 4 x 4 voxels of 2 x 1 x 1 µm: 128 µm³
    ids, diameters = equivalent_diameters(labels, (2.0, 1.0, 1.0))
    assert ids.tolist() == [7]
    assert diameters == pytest.approx([6.252741134374175], abs=1e-9)     # (6 V / pi)^(1/3)


def test_centroids_are_in_the_frame_of_the_grid():
    labels = squares((4, 8))
    ids, _ = equivalent_diameters(labels, SPACING)
    centres = centroids_yx(labels, ids, SPACING, (0.0, 100.0, 200.0))
    # Object 1: rows 2-5, columns 1-4: pixel centres at their mean index 3.5 and 2.5.
    # Object 2: rows 2-9, columns 13-20: mean index 5.5 and 16.5.
    np.testing.assert_allclose(centres, [[100.0 + 0.5 * 3.5, 200.0 + 0.5 * 2.5],
                                         [100.0 + 0.5 * 5.5, 200.0 + 0.5 * 16.5]], atol=TOL, rtol=0)


def test_size_row_percentiles_and_the_limit_is_strict():
    diameters = np.array([1.0, 2.0, 2.9, 3.0, 4.0, 8.0])
    row = size_row("sample", NA, "nucleus", diameters, 3.0)
    assert (row["scope"], row["object"], row["n_objects"], row["below_limit"]) == ("sample", "nucleus", 6, 3)
    assert row["below_fraction"] == pytest.approx(0.5, abs=TOL)      # 3.0 itself is not below 3.0
    # NumPy's linear percentile of the sorted values at positions 0.25, 1.25, 2.5, 3.75 and 4.75.
    assert [row[c] for c in ("diameter_p5", "diameter_p25", "diameter_median", "diameter_p75", "diameter_p95")] == \
        pytest.approx([1.25, 2.225, 2.95, 3.75, 7.0], abs=TOL)
    empty = size_row("fov", "F9", "cell", np.array([]), 3.0)
    assert empty["n_objects"] == 0 and empty["below_limit"] == 0
    assert all(isinstance(empty[c], Unavailable) and empty[c].reason == "undefined"
               for c in ("diameter_median", "below_fraction"))


# --- The small-nucleus rule ------------------------------------------------------------------------------------

def hand_fovs(ids):
    from starfinder.report._schema import TABLE_DTYPES
    rows = []
    for i, fov in enumerate(ids):
        row = {c: NA for c in TABLE_DTYPES["fovs"]}
        row.update(fov_id=fov, list_index=i)
        rows.append(row)
    return rows


def evaluate(fractions, limits=None, ids=None):
    ids = ids or list(fractions)
    inputs = RuleInputs(ids, {f: {"status": "succeeded"} for f in ids}, Unavailable("no_placement", "x"),
                        Unavailable("not_configured", "no registration"), {}, fractions)
    rows, rules, part_rows = evaluate_rules(hand_fovs(ids), [], limits or ReportLimits(), inputs)
    return [r for r in rows if r["rule"] == "small_nucleus_fraction"], rules["small_nucleus_fraction"], [
        p for p in part_rows if p[0] == "rule:small_nucleus_fraction"]


def test_small_nucleus_rule_marks_the_high_side():
    rows, rule, part_rows = evaluate({"H1": 0.02, "H2": 0.03, "H3": 0.30, "H4": 0.025})
    # Median 0.0275, MAD 0.005: the relative limit max(3 x 1.4826 x 0.005, 0.05) = 0.05.
    assert [(r["fov_id"], r["section"], r["measured"], r["unit"]) for r in rows] == [
        ("H3", "Segmentation and assembly", 0.30, "fraction")]
    assert rows[0]["sample_median"] == pytest.approx(0.0275, abs=TOL) and rows[0]["limit"] == pytest.approx(0.05)
    assert "3 µm" in rows[0]["reason"] and rule == {"evaluated": True, "fovs_marked": 1} and part_rows == []
    # An absolute limit is that limit alone, strictly above, with no sample median.
    rows, _, _ = evaluate({"H1": 0.02, "H2": 0.10, "H3": 0.30}, ReportLimits(small_nucleus_fraction=RuleLimit(
        absolute=0.10)))
    assert [(r["fov_id"], r["sample_median"], r["limit"]) for r in rows] == [("H3", NA, 0.10)]
    # The lower limit of the diameter is in the reason.
    rows, _, _ = evaluate({"H1": 0.0, "H2": 0.4}, ReportLimits(small_nucleus_fraction=RuleLimit(absolute=0.3),
                                                                small_nucleus_diameter_um=2.5))
    assert "2.5 µm" in rows[0]["reason"]


def test_small_nucleus_rule_skips_fovs_without_a_value_and_reports_the_unevaluated_sample():
    rows, rule, _ = evaluate({"H1": 0.0, "H2": Unavailable("undefined", "no nucleus"), "H3": 0.0, "H4": 0.5})
    assert [r["fov_id"] for r in rows] == ["H4"] and rule["fovs_marked"] == 1
    reason = Unavailable("no_segmentation", "export assembled molecules_only")
    for nothing in (reason, None):
        rows, rule, part_rows = evaluate(nothing, ids=["H1", "H2"])
        assert rows == [] and rule == {"evaluated": False, "fovs_marked": NA}
        assert [p[1].reason for p in part_rows] == ["no_segmentation"]
    rows, rule, part_rows = evaluate({"H1": Unavailable("undefined", "no nucleus")})
    assert rule["evaluated"] is False and part_rows[0][1].reason == "undefined"
    rows, rule, part_rows = evaluate({"H1": 0.3}, ReportLimits(small_nucleus_fraction=RuleLimit()))
    assert rule["evaluated"] is False and part_rows[0][1].reason == "not_configured"


def test_the_limits_of_the_sizes():
    limits = ReportLimits()
    assert (limits.small_nucleus_fraction, limits.small_nucleus_diameter_um) == (RuleLimit(k=3.0, floor=0.05), 3.0)
    assert PROBLEM_RULES[-1] == "small_nucleus_fraction"
    assert ReportLimits(small_nucleus_diameter_um=5).small_nucleus_diameter_um == 5.0
    for bad, error in ((0, ValueError), (-1.0, ValueError), (float("inf"), ValueError), (True, TypeError),
                       ("3", TypeError), (None, TypeError)):
        with pytest.raises(error):
            ReportLimits(small_nucleus_diameter_um=bad)
    with pytest.raises(TypeError):
        ReportLimits(small_nucleus_fraction=0.1)


# --- The route per FOV on hand-built folders --------------------------------------------------------------------

def stain_pattern(fov, channel):
    """The input channel of a FOV: an integer ramp that differs per FOV and channel."""
    base = 1 + int(fov[1:]) + 10 * channel
    return (base + np.arange(SHAPE[1] * SHAPE[2], dtype=np.float32).reshape(SHAPE) % 7).astype(np.float32)


def write_labels(path, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, labels)
    return hash_oracles.file_sha256(path)


def write_fov(case, fov, *, nucleus_input=True, cell_input=False):
    """The saved runs ``nucleus`` and ``cell`` of a FOV and its assignment folder, a ``reference.tif`` unlike every
    input. A run with an input holds ``input.ome.tif`` of two channels (the first is ``stain_pattern(fov, 0)``)."""
    folder = case.fov_folder(fov)
    sides = SIDES[fov]
    runs = {"nucleus": (squares(sides), nucleus_input), "cell": (squares(sides, pad=2), cell_input)}
    entries = {}
    for name, (labels, has_input) in runs.items():
        directory = folder / "segmentation" / name
        record = {"format_version": 1, "stage": "segmentation", "run": name, "target": name, "geometry": "plane",
                  "methods": [{"stage": "segmentation", "method": "watershed"}], "import": None, "operations": [],
                  "outcome": "ok", "input": None}
        labels_sha = write_labels(directory / "labels.tif", labels)
        record["labels"] = {"path": "labels.tif", "file_sha256": labels_sha}
        if has_input:
            stack = np.stack([stain_pattern(fov, 0), stain_pattern(fov, 1)], axis=-1)
            save_volume(stack, directory / "input.ome.tif")
            record["input"] = {"path": "input.ome.tif", "file_sha256": hash_oracles.file_sha256(
                directory / "input.ome.tif"), "shape_zyxc": list(stack.shape),
                "channels": [{"role": "nuclear"}, {"role": "cytoplasm"}]}
        (directory / "segmentation.json").write_text(json.dumps(record, indent=2) + "\n")
        entries[name] = {"run": name, "target": name, "geometry": "plane", "saved_under_run": True,
                         "record_sha256": hash_oracles.record_sha256(record), "record": record,
                         "file": {"path": f"../../segmentation/{name}/labels.tif", "sha256": labels_sha}}
    assignment = folder / "assignment" / "default"
    assignment.mkdir(parents=True, exist_ok=True)
    (assignment / "cells.csv").write_text("cell_id,status\n1,kept\n")
    record = {"format_version": 1, "stage": "assignment", "name": "default",
              "grid": {"shape_zyx": list(SHAPE), "metadata": {"spacing_zyx": list(SPACING)}},
              "inputs": {"cells": entries["cell"], "nuclei": entries["nucleus"]},
              "files": {"cells": {"path": "cells.csv", "sha256": hash_oracles.file_sha256(assignment / "cells.csv"),
                                  "dtypes": {}}}}
    (assignment / "assignment.json").write_text(json.dumps(record, indent=2) + "\n")
    tifffile.imwrite(folder / "reference.tif", np.full(SHAPE, 99, dtype=np.uint8))


def case_with_runs(tmp_path, **options):
    case = report_records(tmp_path / "root")
    case.config = ReportConfig(example_fovs=("F1", "F2", "F3"), **options.pop("config", {}))
    for fov in FOVS:
        write_fov(case, fov, **options)
    return case


def previews_of(case):
    _, previews = report_examples(case.root, case.sample, case.fov_ids, dataset_id=case.dataset_id,
                                  sample_id=case.sample_id, config=case.config)
    return previews


def test_background_is_the_first_channel_of_the_nucleus_run_input_not_the_reference_image(tmp_path):
    previews = previews_of(case_with_runs(tmp_path))
    assert {p.fov_id for p in previews.values()} == {"F1", "F2", "F3"}
    for preview in previews.values():
        assert preview.source == "segmentation_input"
        assert (preview.background_run, preview.background_role) == ("nucleus", "nuclear")
        assert preview.background_name == f"fovs/{preview.fov_id}/segmentation/nucleus/input.ome.tif"
        np.testing.assert_array_equal(preview.background, stain_pattern(preview.fov_id, 0)[0])
        assert not (preview.background == 99).any()          # reference.tif is a constant 99
        np.testing.assert_array_equal(preview.nuclei, squares(SIDES[preview.fov_id])[0])


def test_background_is_the_cell_run_input_when_the_nucleus_run_has_none(tmp_path):
    previews = previews_of(case_with_runs(tmp_path, nucleus_input=False, cell_input=True))
    for preview in previews.values():
        assert (preview.source, preview.background_run) == ("segmentation_input", "cell")
        np.testing.assert_array_equal(preview.background, stain_pattern(preview.fov_id, 0)[0])


def test_background_is_the_reference_image_when_no_run_has_an_input(tmp_path):
    previews = previews_of(case_with_runs(tmp_path, nucleus_input=False))
    for preview in previews.values():
        assert (preview.source, preview.background_run) == ("reference_image", None)
        assert preview.background_name == f"fovs/{preview.fov_id}/reference.tif"
        assert (preview.background == 99).all()


def test_a_changed_run_input_is_input_changed_not_a_silent_reference_background(tmp_path):
    case = case_with_runs(tmp_path)
    path = case.fov_folder("F2") / "segmentation" / "nucleus" / "input.ome.tif"
    path.write_bytes(path.read_bytes() + b"\0")
    previews = previews_of(case)
    changed = [p for p in previews.values() if isinstance(p, Unavailable)]
    assert [p.reason for p in changed] == ["input_changed"]
    _, _, tables = collect(case)
    assert ("assembly.examples", None, None, "fov_id=F2;role=example_2", "input_changed") in keyed(
        tables["unavailable"])
    missing = case.fov_folder("F2") / "segmentation" / "nucleus" / "input.ome.tif"
    missing.unlink()
    assert [p.reason for p in previews_of(case).values() if isinstance(p, Unavailable)] == ["input_missing"]


def test_the_report_shows_the_background_caption_and_the_size_panel(tmp_path):
    case = case_with_runs(tmp_path)
    folder, record, tables = collect(case)
    page = (folder / "report.html").read_text(encoding="utf-8")
    caption = page[page.index('id="seg-background"'):]
    caption = caption[:caption.index("</p>")]
    assert "segmentation input of run nucleus, its first channel (nuclear)" in caption
    assert "fovs/F1/segmentation/nucleus/input.ome.tif" in caption
    assert 'id="b-seg-sizes"' in page and 'id="t-sizes"' in page and 'data-figure="size:nucleus"' in page
    assert 'data-figure="size:cell"' in page
    assert not [u for u in record["unavailable"] if u["part"] in ("assembly.examples", "assembly.sizes")]


def test_the_size_table_of_the_hand_built_sample(tmp_path):
    folder, record, tables = collect(case_with_runs(tmp_path))
    sizes = tables["sizes"]
    sample = sizes[sizes["scope"].eq("sample").to_numpy(bool)].set_index("object")
    every = sorted(NUCLEUS_UM[s] for f in FOVS for s in SIDES[f])
    assert sample.loc["nucleus", ["n_objects", "below_limit"]].tolist() == [16, 3]
    assert sample.loc["nucleus", "below_fraction"] == pytest.approx(3 / 16, abs=TOL)
    assert sample.loc["nucleus", "diameter_median"] == pytest.approx(float(np.median(every)), abs=1e-9)
    assert sample.loc["nucleus", "diameter_p5"] == pytest.approx(float(np.percentile(every, 5)), abs=1e-9)
    assert sample.loc["nucleus", "diameter_p95"] == pytest.approx(float(np.percentile(every, 95)), abs=1e-9)
    # A cell is its nucleus grown by 2 px on two sides: 6, 8, 10 px become 8, 10, 12 px.
    cell = [(s + 2) / math.sqrt(math.pi) for f in FOVS for s in SIDES[f]]
    assert sample.loc["cell", "n_objects"] == 16
    assert sample.loc["cell", "diameter_median"] == pytest.approx(float(np.median(cell)), abs=1e-9)
    by_fov = sizes[sizes["scope"].eq("fov").to_numpy(bool) & sizes["object"].eq("nucleus").to_numpy(bool)]
    assert by_fov["fov_id"].tolist() == list(FOVS)
    assert by_fov["n_objects"].tolist() == [3, 4, 5, 4] and by_fov["below_limit"].tolist() == [0, 0, 3, 0]
    assert by_fov["below_fraction"].tolist() == pytest.approx([0.0, 0.0, 0.6, 0.0], abs=TOL)
    assert list(sizes.columns)[:3] == ["scope", "fov_id", "object"]
    assert record["rules"]["small_nucleus_fraction"] == {
        "limit": {"k": 3.0, "floor": 0.05, "absolute": None}, "unit": "fraction", "evaluated": True,
        "fovs_marked": 1}
    problems = tables["problematic_fovs"]
    row = problems[problems["rule"].eq("small_nucleus_fraction").to_numpy(bool)]
    assert row[["fov_id", "section", "unit"]].values.tolist() == [["F3", "Segmentation and assembly", "fraction"]]
    assert row["measured"].tolist() == pytest.approx([0.6], abs=TOL) and row["limit"].tolist() == [0.05]
    assert pd.isna(row["round"]).all() and pd.isna(row["axis"]).all()


def test_the_lower_limit_of_the_diameter_is_configurable(tmp_path):
    # Below 4 µm: the 4 px (2.26 µm) and 6 px (3.39 µm) nuclei; the 8 px one (4.51 µm) is not.
    case = case_with_runs(tmp_path, config={"limits": ReportLimits(small_nucleus_diameter_um=4.0)})
    _, record, tables = collect(case)
    sizes = tables["sizes"]
    by_fov = sizes[sizes["scope"].eq("fov").to_numpy(bool) & sizes["object"].eq("nucleus").to_numpy(bool)]
    assert by_fov["below_limit"].tolist() == [1, 1, 4, 1]
    assert record["config"]["limits"]["small_nucleus_diameter_um"] == 4.0
    assert record["rules"]["small_nucleus_fraction"]["fovs_marked"] == 1     # F3: 4 of 5; the others 1 of 3 or 4


def test_a_sample_without_a_nucleus_label_file_has_cell_sizes_only(tmp_path):
    case = case_with_runs(tmp_path)
    for fov in FOVS:
        path = case.fov_folder(fov) / "assignment" / "default" / "assignment.json"
        record = json.loads(path.read_text())
        record["inputs"]["nuclei"] = None
        path.write_text(json.dumps(record, indent=2) + "\n")
    _, record, tables = collect(case, assignment="default")
    assert set(tables["sizes"]["object"]) == {"cell"}
    assert record["rules"]["small_nucleus_fraction"]["evaluated"] is False
    assert ("rule:small_nucleus_fraction", None, None, "", "no_segmentation") in keyed(tables["unavailable"])


def test_sizes_need_the_label_images(tmp_path):
    case = case_with_runs(tmp_path)
    for fov in FOVS:
        (case.fov_folder(fov) / "segmentation" / "cell" / "labels.tif").unlink()
    _, record, tables = collect(case)
    assert tables["sizes"].empty
    rows = keyed(tables["unavailable"])
    assert ("assembly.sizes", None, None, "", "input_missing") in rows
    assert ("rule:small_nucleus_fraction", None, None, "", "input_missing") in rows


# --- The chain on the synthetic scene (both routes, four FOVs) --------------------------------------------------

@pytest.mark.slow
@pytest.mark.integration
def test_the_background_and_sizes_of_the_stain_chain_on_the_route_per_fov(scene_chain_stain):  # noqa: F811
    chain = scene_chain_stain["per_fov"]
    assert len(chain.fov_ids) == 4
    folder, record = chain.collect()
    assert record["route"] == "per_fov"
    assert [u for u in record["unavailable"] if u["part"] in ("assembly.examples", "assembly.sizes")] == []
    _, previews = report_examples(chain.root, chain.sample, chain.fov_ids, dataset_id=chain.dataset_id,
                                  sample_id=chain.sample)
    reference = {f: tifffile.imread(chain.root / "fovs" / f / "reference.tif") for f in chain.fov_ids}
    for preview in previews.values():
        fov = preview.fov_id
        saved = tifffile.imread(chain.root / "fovs" / fov / "segmentation" / "nucleus" / "input.ome.tif")
        assert preview.source == "segmentation_input" and preview.background_run == "nucleus"
        np.testing.assert_array_equal(preview.background, np.asarray(saved).reshape(preview.background.shape))
        np.testing.assert_array_equal(preview.background, chain.scene.stains[fov].reshape(preview.background.shape))
        assert not np.array_equal(preview.background, reference[fov].reshape(preview.background.shape).astype(
            preview.background.dtype))
    sizes = pd.read_csv(folder / "tables" / "sizes.csv")
    nuclei = sizes[(sizes["scope"] == "sample") & (sizes["object"] == "nucleus")].iloc[0]
    labels = {f: tifffile.imread(chain.root / "fovs" / f / "segmentation" / "nucleus" / "labels.tif")
              for f in chain.fov_ids}
    assert nuclei["n_objects"] == sum(len(np.unique(a[a != 0])) for a in labels.values())
    spacing = chain.scene.fov_metadata[chain.fov_ids[0]].spacing_zyx
    every = np.concatenate([equivalent_diameters(a.reshape(1, *a.shape[-2:]), spacing)[1] for a in labels.values()])
    assert nuclei["diameter_median"] == pytest.approx(float(np.median(every)), abs=1e-9)
    page = (folder / "report.html").read_text(encoding="utf-8")
    assert "segmentation input of run nucleus" in page and 'id="t-sizes"' in page


@pytest.mark.slow
@pytest.mark.integration
def test_the_background_and_sizes_of_the_stain_chain_after_fusion(scene_chain_stain_merged):  # noqa: F811
    chain = scene_chain_stain_merged
    folder, record = chain.collect()
    assert record["route"] == "after_fusion"
    assert [u for u in record["unavailable"] if u["part"] == "assembly.examples"] == []
    # A FOV that owns no cell has no percentile of cell diameters: undefined, as every column of a count of zero.
    assert {(u["reason"], u["row_key"].split("object=")[-1]) for u in record["unavailable"]
            if u["part"] == "assembly.sizes"} <= {("undefined", "cell")}
    _, previews = report_examples(chain.root, chain.sample, chain.fov_ids, dataset_id=chain.dataset_id,
                                  sample_id=chain.sample)
    export_dir = chain.root / "export" / chain.sample
    export = json.loads((export_dir / "export.json").read_text())
    outputs = {o["image"]: o for o in export["fusion"]["outputs"] if o["dimensionality"] == "projection"}
    first = export["fusion"]["outputs"][0]["image"]
    named = export["segmentation"]["runs"][1]["inputs"][0]["image"]       # the nucleus run is an import
    assert named == "reference_merged" and first != named                 # the old background was the first image
    fovs = pd.read_csv(export_dir / "tables" / "fovs.csv", float_precision="round_trip").set_index("fov_id")
    for preview in previews.values():
        assert (preview.source, preview.background_name, preview.background_run) == (
            "segmentation_input", named, "cell")
        entry = outputs[named]
        image = tifffile.imread(export_dir / entry["path"])
        old = tifffile.imread(export_dir / outputs[first]["path"])
        row = fovs.loc[preview.fov_id]
        (_, sy, sx), (_, oy, ox) = entry["spacing_zyx"], entry["origin_zyx"]
        y0, y1 = (int(np.floor((row[k] - oy) / sy + 0.5)) for k in ("footprint_y0", "footprint_y1"))
        x0, x1 = (int(np.floor((row[k] - ox) / sx + 0.5)) for k in ("footprint_x0", "footprint_x1"))
        y0, y1, x0, x1 = max(0, y0), min(image.shape[-2], y1), max(0, x0), min(image.shape[-1], x1)
        fused = image.reshape(-1, *image.shape[-2:]).max(axis=0)
        np.testing.assert_array_equal(preview.background, fused[y0:y1, x0:x1])
        assert not np.array_equal(preview.background, old.reshape(-1, *old.shape[-2:]).max(axis=0)[y0:y1, x0:x1])
    sizes = pd.read_csv(folder / "tables" / "sizes.csv")
    nuclei = sizes[(sizes["scope"] == "sample") & (sizes["object"] == "nucleus")].iloc[0]
    runs = {r["name"]: r for r in export["segmentation"]["runs"]}
    labels = tifffile.imread(export_dir / runs["nucleus"]["path"])
    assert nuclei["n_objects"] == len(np.unique(labels[labels != 0]))
    per_fov = sizes[(sizes["scope"] == "fov") & (sizes["object"] == "nucleus")]
    assert set(per_fov["fov_id"]) <= set(chain.fov_ids) and 0 < per_fov["n_objects"].sum() <= nuclei["n_objects"]


@pytest.fixture(scope="module")
def stain_fusion_chain(scene_chain_stain, tmp_path_factory):  # noqa: F811
    """The stain chain exported after fusion with the cell run segmenting the fused reference stain (not
    ``report_scene.export_after_fusion``'s ``reference_merged``), into ``export_on_stain/`` beside the copied export."""
    require_extras()
    return export_after_fusion_on_stain(scene_chain_stain["per_fov"], tmp_path_factory.mktemp("stain_fusion") / "chain",
                                        export_name="export_on_stain")


@pytest.mark.slow
@pytest.mark.integration
def test_the_background_after_fusion_is_the_reference_stain_when_the_cell_run_segments_it(stain_fusion_chain, tmp_path):
    chain = stain_fusion_chain
    export_dir = chain.root / "export_on_stain" / chain.sample
    folder, record = chain.collect(export_dir=export_dir, output_dir=tmp_path / "report")
    assert [u for u in record["unavailable"] if u["part"] == "assembly.examples"] == []
    _, previews = report_examples(chain.root, chain.sample, chain.fov_ids, dataset_id=chain.dataset_id,
                                  sample_id=chain.sample, export_dir=export_dir)
    export = json.loads((export_dir / "export.json").read_text())
    cell = next(r for r in export["segmentation"]["runs"] if r["name"] == "cell")
    assert cell["inputs"] == [{"role": "cytoplasm", "image": "reference_stain"}]
    outputs = {o["image"]: o for o in export["fusion"]["outputs"] if o["dimensionality"] == "projection"}
    fovs = pd.read_csv(export_dir / "tables" / "fovs.csv", float_precision="round_trip").set_index("fov_id")

    def crop(name, fov):
        entry = outputs[name]
        image = tifffile.imread(export_dir / entry["path"])
        image = image.reshape(-1, *image.shape[-2:]).max(axis=0)
        (_, sy, sx), (_, oy, ox) = entry["spacing_zyx"], entry["origin_zyx"]
        row = fovs.loc[fov]
        y0, y1 = (int(np.floor((row[k] - oy) / sy + 0.5)) for k in ("footprint_y0", "footprint_y1"))
        x0, x1 = (int(np.floor((row[k] - ox) / sx + 0.5)) for k in ("footprint_x0", "footprint_x1"))
        return image[max(0, y0):min(image.shape[0], y1), max(0, x0):min(image.shape[1], x1)]
    for preview in previews.values():
        assert (preview.source, preview.background_name, preview.background_run, preview.background_role) == (
            "segmentation_input", "reference_stain", "cell", "cytoplasm")
        np.testing.assert_array_equal(preview.background, crop("reference_stain", preview.fov_id))
        assert not np.array_equal(preview.background, crop("reference_merged", preview.fov_id))
    page = (folder / "report.html").read_text(encoding="utf-8")
    assert "segmentation input of run cell, its first channel (cytoplasm)" in page and "reference_stain" in page
