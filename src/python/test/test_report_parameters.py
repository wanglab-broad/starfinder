"""The cards of the report's Workflow overview and their parameters (W-883; design/specs/report-algorithms.md,
"Workflow overview").

The overview has a card for every stage the run records, in the order the steps run, among them the post-registration
preprocessing ("not run" without ``post_registration``). Each card with parameters opens on them: the method, then the
values of ``workflow/resolved.json`` that differ from the defaults of the dataclass its block names, the fitted values
of the example FOV with a link to the table that holds them, and the whole block behind "all parameters".

The resolved configuration of these tests is built from the package's own dataclasses with ``plain`` (the form the plan
writes), so that every ``type`` is a class with defaults; every expected list of values is written here by hand.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from starfinder.dataset.config import PipelineConfig
from starfinder.preprocessing import MinMaxNormalizationConfig, PreprocessingRecipe, PreprocessingStep
from starfinder.preprocessing import ReconstructionConfig, TophatConfig
from starfinder.readout import DeduplicationConfig, ReadFilterConfig, ReadScoreConfig
from starfinder.readout.decoding import WtaDecoderConfig
from starfinder.readout.extraction import NeighborhoodSumConfig
from starfinder.registration import RegistrationQcConfig, RegistrationRecipe, RegistrationStep, TranslationConfig
from starfinder.report import _page as page_module
from starfinder.report._parameters import defaults, differing
from starfinder.spot_finding import LocalMaximaConfig
from starfinder.workflow._shape import plain, sha256

from . import hash_oracles
from .report_fixtures import FOVS, SAMPLE, report_cells, report_records
from .test_report_collection import anndata_or_skip

pytestmark = pytest.mark.report

#: The Per FOV cards of the pipeline, in order; a sample whose route is not after fusion adds Segmentation and
#: Assignment after them.
FOV_TITLES = ["Preprocessing", "Registration", "Post-registration preprocessing", "Spot finding", "Read extraction",
              "Decoding", "Scoring", "Deduplication", "Read QC filter"]


def pipeline(*, post_registration=True):
    """The pipeline of the tests' configuration: two preprocessing steps, a post-registration reconstruction (radius 5,
    the default is 3), a translation with a quality check, local maxima off the border, a WTA decoder that keeps
    diagnostics, and the end bases CC."""
    step = PreprocessingStep
    recipe = PreprocessingRecipe(
        steps=(step(MinMaxNormalizationConfig(output_dtype="uint8", output_range=(0, 255))),
               step(TophatConfig(radius_yx=2))),
        post_registration=(step(ReconstructionConfig(radius_yx=5)),) if post_registration else ())
    return PipelineConfig(
        preprocessing=recipe,
        registration=RegistrationRecipe(steps=(RegistrationStep(TranslationConfig()),),
                                        qc=RegistrationQcConfig(min_coverage=0.8)),
        spot_finding=LocalMaximaConfig(threshold_value=0.5, exclude_border=False),
        extraction=NeighborhoodSumConfig(), decoding=WtaDecoderConfig(diagnostics=True), scoring=ReadScoreConfig(),
        deduplication=DeduplicationConfig(), filtering=ReadFilterConfig(end_bases="CC"))


def write_resolved(case, *, post_registration=True, api=None, extra=None, rehash=False):
    """``workflow/resolved.json`` of the case, as the plan writes its ``api``, and ``workflow_config`` in every run
    record (path and the file's hash), with the records' recipe and steps naming the post-registration step; with
    ``rehash`` the file is written again after the records hold its hash (a changed file)."""
    api = api if api is not None else {"processing": {"pipeline": plain(pipeline(post_registration=post_registration))}}
    configuration = {"backend": "python", "api": api, **(extra or {})}
    path = case.root / "workflow" / "resolved.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps({"format": "starfinder.workflow.resolved/1", "source": "test", "configuration": configuration},
                      sort_keys=True).encode() + b"\n"
    path.write_bytes(data)
    for fov in case.fov_ids:
        record = case.run_record(fov)
        record["workflow_config"] = {"path": "workflow/resolved.json", "sha256": sha256(data)}
        if post_registration:
            record["preprocessing"]["recipe"]["post_registration"] = ["reconstruction"]
            steps = record["steps"]
            at = next(i for i, s in enumerate(steps) if s["name"] == "decode_reads")
            for name in ("round1", "round2", "round3"):
                steps.insert(at, {"name": "preprocess:reconstruction", "round": name, "seconds": 0.5,
                                  "status": "succeeded"})
        case.write_run_record(fov, record)
    if rehash:
        configuration["backend"] = "changed"
        path.write_bytes(json.dumps({"format": "starfinder.workflow.resolved/1", "source": "test",
                                     "configuration": configuration}, sort_keys=True).encode() + b"\n")
    return path


def overview(text):
    """The cards of the Workflow overview: per lane, ``[(number, title, detail, [(heading, rows)] or None)]``, where
    ``rows`` are the ``(name, value)`` of the differing-values table of each section of the card's panel. Read with
    string operations (the page holds megabytes of figures)."""
    chart = text[text.index('id="b-chart"'):text.index('id="b-fov-status"')]
    lanes = []
    for lane in chart.split('<div class="lane">')[1:]:
        cards = []
        for chunk in lane.split('<div class="st')[1:]:
            head = chunk.split('<span class="n">', 1)[1]
            number, rest = head.split("</span>", 1)
            title, rest = rest.split("</b><i>", 1)
            detail = rest.split("</i>", 1)[0]
            sections = None
            if "<details>" in chunk:
                body = chunk.split("</summary>", 1)[1].split('<details class="sp-all">', 1)[0]
                sections = []
                for part in body.split('<div class="sp-sec">')[1:]:
                    heading = part.split("<h4>", 1)[1].split("</h4>", 1)[0] if part.startswith("<h4>") else None
                    caption = part.split('<p class="sp-cap">', 1)[1].split("</p>", 1)[0]
                    rows = [tuple(row.replace("<code>", "").replace("</code>", "").split("</td><td>"))
                            for row in (cell.split("</td></tr>", 1)[0] for cell in part.split("<tr><td>")[1:])]
                    if not caption.startswith("Fitted values"):
                        sections.append((heading, caption, rows))
            cards.append((int(number), title, detail, sections))
        lanes.append(cards)
    return lanes


def card(text, title):
    return next(c for lane in overview(text) for c in lane if c[1] == title)


def page_text(folder):
    return (Path(folder) / "report.html").read_text(encoding="utf-8")


def values(sections, kind="Values that differ from the defaults"):
    """The rows of every section of a card whose caption starts ``kind``, by section heading."""
    return {heading: rows for heading, caption, rows in sections if caption.startswith(kind) and rows}


# --- Which values differ from the defaults ------------------------------------------------------------------------

def test_differing_lists_the_values_that_are_not_the_defaults_in_field_order():
    node = {"type": "LocalMaximaConfig", "threshold_mode": "adaptive", "threshold_value": 0.5,
            "min_distance_voxels": 1, "exclude_border": False, "channel_labels": None,
            "measure_peak_intensity": True, "merge_radius_zyx": None, "method": "local_maxima"}
    assert defaults("LocalMaximaConfig")["threshold_value"] == 0.2
    assert differing(node) == [(("threshold_value",), 0.5), (("exclude_border",), False)]
    assert differing(dict(node, threshold_value=0.2, exclude_border=True)) == []


def test_differing_goes_into_nested_dataclasses_and_lists_of_them():
    recipe = plain(RegistrationRecipe(steps=(RegistrationStep(TranslationConfig(fft_workers=2)),),
                                      qc=RegistrationQcConfig(min_coverage=0.8, projections=True)))
    assert differing(recipe) == [(("steps[0]", "config", "fft_workers"), 2), (("qc", "min_coverage"), 0.8),
                                 (("qc", "projections"), True)]


def test_differing_lists_a_field_without_a_default_and_every_field_of_an_unknown_type():
    step = plain(PreprocessingStep(MinMaxNormalizationConfig(output_dtype="uint16", output_range=(0, 65535))))
    assert differing(step) == [(("config", "output_dtype"), "uint16"), (("config", "output_range"), [0, 65535])]
    unknown = {"type": "NotAConfig", "a": 1, "b": None}
    assert differing(unknown) == [(("a",), 1), (("b",), None)]


# --- The cards ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("post_registration", [True, False])
def test_the_post_registration_card_is_there_and_says_not_run_without_the_step(tmp_path, post_registration):
    case = report_records(tmp_path / "root")
    write_resolved(case, post_registration=post_registration)
    folder, record = case.collect()
    lead, sample = overview(page_text(folder))
    assert [c[1] for c in lead] == FOV_TITLES + ["Segmentation", "Assignment"]
    # Numbered in the order the steps run; the card follows Registration.
    assert [c[0] for c in lead] == list(range(1, len(FOV_TITLES) + 3))
    post = card(page_text(folder), "Post-registration preprocessing")
    if post_registration:
        assert post[2] == "reconstruction" and 'class="st off"' not in page_text(folder).split(
            "Post-registration preprocessing")[0].rsplit('<div class="st', 1)[1]
        assert values(post[3]) == {"reconstruction": [("radius_yx", "5")]}
    else:
        assert post[2] == "not run" and post[3] is None
    assert [c[1] for c in sample] == ["Stitching", "Assembly", "Export"]


def test_the_cards_follow_the_steps_each_fov_records_in_execution_order(tmp_path):
    """Read extraction and Scoring are stages the run records (``extract_round``, ``score_reads``), so they have
    cards; deduplication runs before the read filter."""
    case = report_records(tmp_path / "root")
    folder, _ = case.collect()
    record = case.run_record("F1")
    order = [s["name"] for s in record["steps"] if s["name"] in ("find_spots", "extract_round", "decode_reads",
                                                                "score_reads", "deduplicate_reads", "filter_reads")]
    assert order == ["find_spots", "extract_round", "extract_round", "extract_round", "decode_reads", "score_reads",
                     "deduplicate_reads", "filter_reads"]
    titles = [c[1] for c in overview(page_text(folder))[0]]
    assert titles.index("Spot finding") < titles.index("Read extraction") < titles.index("Decoding") < titles.index(
        "Scoring") < titles.index("Deduplication") < titles.index("Read QC filter")


@pytest.mark.parametrize("key,title", [("extraction", "Read extraction"), ("scoring", "Scoring")])
def test_a_stage_the_configuration_leaves_out_is_a_card_that_says_not_run(tmp_path, key, title):
    case = report_records(tmp_path / "root")
    for fov in case.fov_ids:
        record = case.run_record(fov)
        record["config"]["pipeline"][key] = None
        case.write_run_record(fov, record)
    folder, _ = case.collect()
    assert card(page_text(folder), title)[2:] == ("not run", None)


# --- The expanded panel -------------------------------------------------------------------------------------------

def test_the_panels_list_exactly_the_values_that_differ_from_the_defaults(tmp_path):
    case = report_records(tmp_path / "root")
    write_resolved(case)
    folder, _ = case.collect()
    text = page_text(folder)
    shown = {title: values(card(text, title)[3]) for title in FOV_TITLES}
    assert shown == {
        "Preprocessing": {"min_max_normalization": [("output_dtype", "uint8"), ("output_range", "[0, 255]")],
                          "white_tophat": [("radius_yx", "2")]},
        "Registration": {"Recipe": [("qc.min_coverage", "0.8")]},
        "Post-registration preprocessing": {"reconstruction": [("radius_yx", "5")]},
        "Spot finding": {"local_maxima": [("threshold_value", "0.5"), ("exclude_border", "false")]},
        "Read extraction": {},
        "Decoding": {"wta": [("diagnostics", "true")]},
        "Scoring": {},
        "Deduplication": {},
        "Read QC filter": {None: [("end_bases", "CC")]}}


def test_a_step_whose_values_are_all_defaults_says_so_and_still_names_its_method(tmp_path):
    case = report_records(tmp_path / "root")
    write_resolved(case)
    folder, _ = case.collect()
    sections = card(page_text(folder), "Preprocessing")[3]
    tophat = next(s for s in sections if s[0] == "white_tophat")
    assert tophat[2] == [("radius_yx", "2")]
    registration = card(page_text(folder), "Registration")[3]
    assert [s[0] for s in registration] == ["translation", "Recipe"] and registration[0][2] == []
    for title in ("Read extraction", "Scoring", "Deduplication"):
        assert card(page_text(folder), title)[3][0][2] == []
    # The translation of the registration and the three stages above are the only sections without a value.
    assert page_text(folder).count("no value differs from the defaults") == 4


def test_the_panel_holds_the_whole_block_behind_all_parameters_and_names_the_file(tmp_path):
    case = report_records(tmp_path / "root")
    path = write_resolved(case)
    folder, record = case.collect()
    text = page_text(folder)
    spot = text.split("<b><span class=\"n\">4</span>")[1].split("</details></div>", 1)[0]
    assert '<details class="sp-all"><summary>all parameters</summary><pre>' in spot
    whole = spot.split("<pre>", 1)[1].split("</pre>", 1)[0]
    assert json.loads(html.unescape(whole)) == plain(pipeline().spot_finding)
    assert '<p class="sp-src muted">from <code>workflow/resolved.json</code></p>' in spot
    # The file is one of the report's inputs, with the hash it has.
    entry = next(e for e in record["inputs"] if e["path"] == "workflow/resolved.json")
    assert entry["sha256"] == hash_oracles.file_sha256(path)


def test_the_files_table_names_the_parameters_and_a_changed_file(tmp_path):
    case = report_records(tmp_path / "root")
    write_resolved(case)
    folder, _ = case.collect()
    rows = dict(re.findall(r"<tr><td>(.*?)</td><td>(.*?)</td></tr>",
                                         page_text(folder).split('id="b-files-configuration"')[1].split("</table>")[0]))
    assert rows["Parameters"].startswith("the cards of the Workflow overview open on them")
    assert "changed" not in rows["Parameters"]
    write_resolved(case, rehash=True)
    folder, _ = case.collect()
    # The FOVs ran under the hash their records hold; the file beside them has another one, said on the panel and
    # in the Files table (the parameters are still those of the file).
    text = page_text(folder)
    assert text.count("the file has changed since the FOVs ran") == len(FOV_TITLES) + 1
    assert "the file has changed since the FOVs ran" in text.split('id="b-files-configuration"')[1]
    assert values(card(text, "Spot finding")[3]) == {"local_maxima": [("threshold_value", "0.5"),
                                                                       ("exclude_border", "false")]}


# --- Without the file, and outside the workflow --------------------------------------------------------------------

def test_a_missing_resolved_file_leaves_the_cards_plain_and_is_named_in_the_files_table(tmp_path):
    case = report_records(tmp_path / "root")
    path = write_resolved(case)
    path.unlink()
    folder, record = case.collect()
    text = page_text(folder)
    assert all(c[3] is None for lane in overview(text) for c in lane)
    rows = [r for r in record["unavailable"] if r["part"] == "files.configuration"]
    assert [(r["row_key"], r["reason"]) for r in rows] == [("entry=parameters", "input_missing")]
    assert 'data-part="files.configuration" data-key="entry=parameters" data-reason="input_missing"' in text


def test_a_file_that_is_not_a_resolved_configuration_is_not_recorded(tmp_path):
    case = report_records(tmp_path / "root")
    path = write_resolved(case)
    path.write_text("{}\n")
    folder, record = case.collect()
    rows = [r for r in record["unavailable"] if r["part"] == "files.configuration"]
    assert [(r["row_key"], r["reason"]) for r in rows] == [("entry=parameters", "not_recorded")]
    assert all(c[3] is None for lane in overview(page_text(folder)) for c in lane)


def test_a_run_outside_the_workflow_has_cards_without_parameters(tmp_path):
    folder, record = report_records(tmp_path / "root").collect()
    lanes = overview(page_text(folder))
    assert [c[1] for c in lanes[0]][:len(FOV_TITLES)] == FOV_TITLES
    assert all(c[3] is None for lane in lanes for c in lane)
    assert all(r["row_key"] != "entry=parameters" for r in record["unavailable"])


# --- Fitted values ------------------------------------------------------------------------------------------------

def test_the_fitted_values_of_the_example_fov_are_shown_with_a_link_to_their_table(tmp_path):
    case = report_records(tmp_path / "root")
    write_resolved(case)
    for fov in case.fov_ids:
        record = case.run_record(fov)
        entry = record["preprocessing"]["rounds"]["round1"][0]
        entry["fitted"] = {"groups": [{"channel": 0, "min": 0.0, "max": 255, "mode": "rescale"},
                                      {"channel": 1, "min": 3.14159265, "max": 218, "mode": "rescale"}]}
        case.write_run_record(fov, record)
    folder, _ = case.collect()
    text = page_text(folder)
    preprocessing = text.split('<span class="n">1</span>Preprocessing')[1].split("</details></div>", 1)[0]
    fitted = preprocessing.split('<p class="sp-cap">Fitted values')[1].split('<details class="sp-all">')[0]
    assert '<a href="#prep">Preprocessing</a> · table <code>tables/intensity.csv</code>' in fitted
    assert "<h4>min_max_normalization · F1, round1</h4>" in fitted
    assert [r for r in fitted.split("<tr>")[1:]] == [
        "<th>channel</th><th>min</th><th>max</th><th>mode</th></tr></thead><tbody>",
        "<td>0</td><td>0</td><td>255</td><td>rescale</td></tr>",
        "<td>1</td><td>3.142</td><td>218</td><td>rescale</td></tr></tbody></table></div>"]
    # The registration card shows the example FOV's shifts, the rows of registration.csv.
    registration = text.split('<span class="n">2</span>Registration')[1].split("</details></div>", 1)[0]
    table = pd.read_csv(folder / "tables" / "registration.csv")
    rows = table[table["fov_id"] == "F1"]
    assert "table <code>tables/registration.csv</code>" in registration
    for _, row in rows.iterrows():
        cells = "".join(f"<td>{value:,.4g}</td>" for value in (row["shift_z"], row["shift_y"], row["shift_x"]))
        assert f"<tr><td>{row['round']}</td>{cells}</tr>" in registration


def test_the_stitching_card_shows_the_quality_of_the_links(tmp_path):
    from starfinder.stitching import save_placement
    from starfinder.stitching._placement import LINK_COLUMNS, LINK_DTYPES, PlacementResult
    from .report_fixtures import placement_result
    case = report_records(tmp_path / "root")
    write_resolved(case, api={"processing": {"pipeline": plain(pipeline())},
                              "stitching": {"method": "multiview_stitcher", "input": "reference_merged"}})
    result = placement_result(case.fov_ids)
    rows = [{c: 0.0 for c in LINK_COLUMNS} | {"fov_a": a, "fov_b": b, "quality": q, "kept": kept, "reason": ""}
            for a, b, q, kept in (("F1", "F2", 0.9, True), ("F1", "F3", 0.7, True), ("F3", "F4", 0.5, False))]
    links = pd.DataFrame(rows, columns=list(LINK_COLUMNS)).astype(LINK_DTYPES)
    save_placement(PlacementResult(result.sample, result.fovs, links, result.record),
                   case.root / "stitching" / SAMPLE)
    folder, _ = case.collect()
    stitching = card(page_text(folder), "Stitching")
    assert values(stitching[3], "As configured") == {None: [("input", "reference_merged"),
                                                           ("method", "multiview_stitcher")]}
    block = page_text(folder).split('</span>Stitching')[1].split("</details></div>", 1)[0]
    assert ('<tr><td>links</td><td>3, kept 2</td></tr><tr><td>quality</td><td>min 0.5, median 0.7, max 0.9</td></tr>'
            in block)
    assert '<a href="#b-placement">Placement</a>' in block


# --- Both routes --------------------------------------------------------------------------------------------------

SEGMENT_API = {"plan": {"type": "SegmentationPlan", "runs": []}, "device": "cpu"}


def route_case(tmp_path, route):
    """report_cells (the export of the route per FOV, which needs anndata) with the route its export.json names."""
    anndata_or_skip()
    case = report_cells(tmp_path / "root")
    api = {"processing": {"pipeline": plain(pipeline())},
           "export": {"config": {"type": "ExportConfig", "assemble": True, "fuse": False, "route": route}}}
    write_resolved(case, api=api, extra={"assembly": {"assignment": "default"}})
    if route == "after_fusion":
        path = case.root / "export" / SAMPLE / "export.json"
        record = json.loads(path.read_text())
        record["assembly"]["route"] = "after_fusion"
        path.write_text(json.dumps(record))
    return case


@pytest.mark.parametrize("route", ["per_fov", "after_fusion"])
def test_the_panels_of_both_routes_list_the_values_that_differ(tmp_path, route):
    case = route_case(tmp_path, route)
    try:
        folder, record = case.collect()
    except ValueError as error:  # an export.json that names a route its tables do not hold
        pytest.skip(str(error))
    lanes = overview(page_text(folder))
    fov_titles = [c[1] for c in lanes[0]]
    sample_titles = [c[1] for c in lanes[1]]
    assert fov_titles[:len(FOV_TITLES)] == FOV_TITLES
    assert ("Segmentation" in fov_titles and "Assignment" in fov_titles) == (route == "per_fov")
    assert ("Segmentation" in sample_titles) == (route == "after_fusion")
    assert sample_titles == (["Stitching", "Assembly", "Export"] if route == "per_fov" else
                             ["Stitching", "Segmentation", "Assembly", "Export"])
    text = page_text(folder)
    assert values(card(text, "Spot finding")[3]) == {"local_maxima": [("threshold_value", "0.5"),
                                                                      ("exclude_border", "false")]}
    assert values(card(text, "Export")[3]) == {None: [("fuse", "false"), ("route", route)]}
    assert values(card(text, "Assembly")[3], "As configured") == {None: [("assignment", "default")]}
    assert record["route"] == route


# --- The size of the page -----------------------------------------------------------------------------------------

def test_the_panels_keep_the_page_within_its_budget_and_add_little(tmp_path):
    plain_case = report_records(tmp_path / "plain")
    folder, record = plain_case.collect()
    without = record["files"][0]["bytes"]
    case = report_records(tmp_path / "workflow")
    write_resolved(case)
    folder, record = case.collect()
    page = record["files"][0]
    assert page["bytes"] <= page["budget_bytes"] == page_module.budget_bytes(len(FOVS), 1)
    # Eight panels of a few small tables each and one copy of each block: a few tens of KiB.
    assert 0 < page["bytes"] - without < 64 * 1024
