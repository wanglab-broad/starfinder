"""The §2.11 report end to end on the synthetic sample scene (W-459).

Checks of docs/report-algorithms.md ("Validation design") that need the whole chain, through
the public entries ``Dataset.report`` and ``Dataset.compare_samples``: the report after the
per-FOV processing without and with the placement (CH1), after the assignment (CH2) and after
the export (CH3), each replacing the earlier in the sample's default folder; both routes on
the one-plane scene (RT1, RT2); ``Z = 1`` (Z1); an export without run records, on both roots
of ``scene_no_run_json``, with and without ``example_fovs`` (A2); the comparison page of two
synthetic samples (CP1); and the determinism of the chain in two single-thread processes
(DT1). PG1 to PG6 and RC12 hold on every page written here. The table "Implemented checks
(W-459)" of the algorithms page names the tests of every check; a default-tier test here
keeps it true.

The chains are those of ``report_scene.py`` (``sample_scene_preset("sample-tiny-v1")``, seed 42),
each built once per module; every chain test is ``slow`` and skips without the ``anndata`` and
``multiview-stitcher`` extras. Expected values are written here from docs/report-contract.md
("The report at each stage of a sample's processing", "Parts, their inputs and their reasons")
and docs/report-algorithms.md, or computed here from the chain's own records and export (read
with the checkpoint table reader and with ``anndata``), with ``ownership_map`` for the cells and
pixels a FOV owns; never from the report's own output. The page, null-value and measurement
checks are those of the earlier issues' test modules, called on these pages.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from starfinder.report import REPORT_PARTS, MarkerGroup, ReportConfig, load_report

from .conftest import SUBSYSTEM_MARKERS
from . import test_report_collection as collection_checks
from . import test_report_comparison as comparison_checks
from . import test_report_measurements as measurement_checks
from . import test_report_page as page_checks
from .report_scene import Chain, scene_chain, scene_chain_plane  # noqa: F401 (module-scoped fixtures)
from .report_sizes import EXAMPLE_TILE, INTENSITY_SIZE, preprocessing_tile, tiles_width

pytestmark = [pytest.mark.report, pytest.mark.integration, pytest.mark.validation]

TOL = 1e-9
DESIGN = Path(__file__).resolve().parents[3] / "docs" / "report-algorithms.md"
SECTION = "### Implemented checks (W-459)"
ROW = r"(?:RC|DF|PR|EF|RP|PG|CH|RT|Z|A|CP|RR|WF|DT)\d+"
TIERS = {"default": (False,), "slow": (True,), "default and slow": (False, True)}
KIB, MIB = 1024, 1024 * 1024
ROLES = ("example_1", "example_2", "example_3")
# The columns of fovs.csv "up to the readout": the run's status and the read levels (docs/report-contract.md, A2).
RUN_COLUMNS = ("status", "failed_step", "failed_round", "started_at", "ended_at", "run_seconds", "spots",
               "correct_form", "matched", "merged_by_deduplication", "passed_filter", "correct_form_fraction",
               "matched_fraction", "merged_fraction", "passed_filter_fraction")
# The parts of "Segmentation and assembly" and "Sample measurements".
LATER_PARTS = ("assembly.cells_per_fov", "assembly.placement", "assembly.examples", "assembly.accounting",
               "sample.per_cell", "sample.markers", "sample.marker_zoom", "sample.mecr", "sample.reads_along_z")
# The parts up to the readout that are unavailable for the whole sample when no FOV has a run record (A2).
RUN_RECORD_PARTS = [("preprocessing.example", ""), ("preprocessing.intensity", ""), ("registration.maps", ""),
                    ("registration.heatmap", ""), ("registration.example", ""), ("readout.distributions", ""),
                    ("readout.table", ""), ("files.input", "entry=images"), ("files.input", "entry=codebook"),
                    ("files.configuration", "entry=workflow"), ("summary.identity", "entry=acquisition"),
                    ("summary.identity", "entry=configuration"), ("summary.identity", "entry=run_time")]
REPLAY_PARTS = ("preprocessing.example", "preprocessing.intensity", "registration.example")
# A reason that a configuration or a record explains at every stage; any other reason makes its part unavailable.
STANDING_REASONS = ("not_configured", "not_recorded")
# The run-record steps that have a method in steps.csv (docs/report-contract.md, "steps.csv", method).
STEPS_WITH_A_METHOD = ("preprocess:", "register", "find_spots", "decode_reads", "filter_reads")
DT1_SCRIPT = """
import json
import sys
from starfinder.dataset import Dataset
from starfinder.report import MarkerGroup, ReportConfig
a = json.loads(sys.argv[1])
config = ReportConfig(markers=tuple(MarkerGroup(name, tuple(genes)) for name, genes in a["markers"]),
                      marker_pairs=tuple(tuple(pair) for pair in a["pairs"]))
Dataset(a["input_root"], a["root"], a["dataset_id"], a["sample"], "output").report(a["sample"], a["fov_ids"],
                                                                                  config=config)
"""


# --- Helpers ----------------------------------------------------------------------------------------------------

class ChainCase:
    """The arguments of a chain's report, as the page and null-value checks read them from a case."""

    def __init__(self, chain: Chain, config):
        self.chain, self.config = chain, config
        self.fov_ids, self.root = list(chain.fov_ids), chain.root

    def run_record(self, fov):
        return self.chain.run_record(fov)


def dataset_of(chain: Chain):
    from starfinder.dataset import Dataset
    return Dataset(chain.input_root, chain.root, chain.dataset_id, chain.sample, "output")


def genes_of(chain: Chain):
    return json.loads((chain.root / "export" / chain.sample / "export.json").read_text())["assembly"]["genes"]


def marker_config(genes, **changes):
    """Two groups of the scene's genes and two pairs (CH3)."""
    return ReportConfig(markers=(MarkerGroup("first", tuple(genes[:2])), MarkerGroup("second", tuple(genes[2:5]))),
                        marker_pairs=((genes[0], genes[1]), (genes[2], genes[3])), **changes)


def report(chain: Chain, config, aside: Path, **changes):
    """``Dataset.report`` into the sample's default folder; the folder is copied to ``aside`` and read from there as
    a :class:`test_report_page.Page`, so that a later report of the same folder leaves it as it was."""
    result = dataset_of(chain).report(chain.sample, chain.fov_ids, config=config, **changes)
    assert result.path == chain.root / "report" / chain.sample
    assert result.record == load_report(result.path)
    shutil.copytree(result.path, aside)
    return page_checks.Page(ChainCase(chain, config), aside, result.record)


def keyed(page):
    return collection_checks.keyed(page.tables["unavailable"])


def present_parts(page):
    """The parts of REPORT_PARTS that no row of unavailable.csv names with a reason a stage of the processing gives
    (every reason but ``not_configured`` and ``not_recorded``)."""
    named = {u[0] for u in keyed(page) if u[4] not in STANDING_REASONS}
    return [p for p in REPORT_PARTS if p not in named]


def expected_roles(chain: Chain, cells=None):
    """The example FOVs of "The example FOVs" (A1, M1) from the run records' read counts (``passed_filter``, the
    filter's ``accepted``), or from ``n_cells_retained`` when ``cells`` gives it: role → FOV."""
    order = list(chain.fov_ids)
    reads = {f: int(chain.run_record(f)["counts"]["filtering"]["accepted"]) for f in order}
    m = float(np.median(list(reads.values())))
    q = max(1, math.ceil(len(order) / 5))
    near = sorted(sorted(order, key=lambda f: (abs(reads[f] - m), order.index(f)))[:q], key=order.index)
    roles = {"example_1": near[int(np.random.default_rng(0).integers(len(near)))]}
    values = cells if cells is not None else reads
    left = [f for f in order if f != roles["example_1"] and values.get(f) is not None]
    roles["example_2"] = min(left, key=lambda f: (values[f], order.index(f)))
    left.remove(roles["example_2"])
    roles["example_3"] = max(left, key=lambda f: (values[f], -order.index(f)))
    return roles


def figure_sizes(page, chain: Chain):
    """The sizes of "Figure sizes" for the figures test_report_page.check_figures does not know."""
    record = page.record
    replay = record["replay"] or {}

    def sizes(name, image):
        if name == "preprocessing":
            run = chain.run_record(replay["fov_id"])
            reference = replay["rounds"]["reference"]
            columns = len(run["preprocessing"]["rounds"][reference]) + 1
            assert image.size[1] > len(run["channels"][reference]) * preprocessing_tile(columns)
            return (tiles_width(columns, preprocessing_tile(columns)), image.size[1])
        if name in ("registration", "segmentation"):
            assert image.size[1] > EXAMPLE_TILE
            return (tiles_width(3, EXAMPLE_TILE), image.size[1])
        if name == "intensity":
            return INTENSITY_SIZE
        if re.fullmatch(r"marker:\d+:whole", name):
            assert image.size[0] == page.case.config.figure_width_px
            return "checked"
        if re.fullmatch(r"marker:\d+:example_\d:(xy|xz)", name):
            assert image.size[0] == 512
            return "checked"
        return None
    return sizes


def check_null_values(page):
    """RC12 with test_report_collection.check_null_values, which reads the null rules of the hand-written records,
    and three rules of docs/report-contract.md ("The tables") that only a chain exercises:

    * ``steps.csv`` ``method``: n/a for a step that has none. The steps with a method are the preprocessing steps,
      registration, detection, decoding and the filter; a chain's run records also hold steps such as
      ``write_checkpoint:<stage>``, which have none.
    * ``population.csv``: n/a for a reason the export's ``assembly.counts.<cells|molecules>.excluded`` does not hold
      (``duplicate_observation`` after fusion, or in a ``molecules_only`` export).
    * ``mecr.csv``: each cell of each pair unavailable on its own; the helper does not read this table, so its null
      cells are checked here: each named by exactly one row, with a reason ``sample.mecr`` can have.
    """
    names = {s["name"] for f in page.record["stages"]["run_records"] for s in page.case.run_record(f)["steps"]}
    without = tuple(sorted(n for n in names if not n.startswith(STEPS_WITH_A_METHOD)))
    export = page.case.root / "export" / page.case.chain.sample / "export.json"
    counts = json.loads(export.read_text())["assembly"]["counts"] if export.is_file() else None
    listed, rule = collection_checks.WITHOUT_METHOD, collection_checks._not_applicable

    def not_applicable(table, column, row, case, context):
        if rule(table, column, row, case, context):
            return True
        if table == "population" and counts is not None and row["population"].startswith("excluded_"):
            return row["population"][len("excluded_"):] not in (counts.get(column) or {}).get("excluded", {})
        return False
    unavailable = page.tables["unavailable"]
    mecr_rows = unavailable["table"].eq("mecr").fillna(False).to_numpy(bool)
    tables = dict(page.tables, unavailable=unavailable[~mecr_rows].reset_index(drop=True))
    collection_checks.WITHOUT_METHOD = tuple(dict.fromkeys(listed + without))
    collection_checks._not_applicable = not_applicable
    try:
        collection_checks.check_null_values(page.case, page.record, tables)
    finally:
        collection_checks.WITHOUT_METHOD, collection_checks._not_applicable = listed, rule
    named = collection_checks.keyed(unavailable[mecr_rows])
    assert len(set(named)) == len(named) and all(
        r[0] == "sample.mecr" and r[4] in collection_checks.PART_REASONS["sample.mecr"] for r in named), named
    cells = set()
    for row in page.tables["mecr"].to_dict("records"):
        key = collection_checks.key_of("mecr", row)
        cells |= {("sample.mecr", "mecr", c, key) for c in ("cells_a", "cells_b", "both", "either", "mecr")
                  if pd.isna(row[c])}
    assert {r[:4] for r in named} == cells
    # The tables whose every column is never null: cells.csv, reads_along_z.csv and intensity.csv.
    for name in ("cells", "reads_along_z", "intensity"):
        assert not page.tables[name].isna().any().any(), name


def check_page(page, chain: Chain, tmp_path):
    """PG1 to PG6 and RC12 on one page of the chain, with the checks of test_report_page.py and
    test_report_collection.py."""
    as_case = (lambda maker, variant: page)  # noqa: E731
    tmp_path.mkdir(parents=True, exist_ok=True)
    page_checks.test_pg1_embedded_data_equal_the_tables(as_case, "chain", None)
    page_checks.test_pg2_sections_and_blocks(as_case, "chain", None)
    page_checks.test_pg3_every_unavailable_part_carries_its_reason(as_case, "chain", None)
    if shutil.which("node") or shutil.which("nodejs"):
        page_checks.test_pg3_cells_drawn_by_the_script(as_case, tmp_path, "chain", None)
    page_checks.test_pg4_no_external_resource(as_case, "chain", None)
    # PG5: the bytes within the budget and equal to the record's; every figure decodes and has its size.
    size = (page.folder / "report.html").stat().st_size
    entry = next(e for e in page.record["files"] if e["path"] == "report.html")
    budget = 6 * MIB + 16 * KIB * len(chain.fov_ids) + 3 * MIB * (len(page.case.config.markers) or 1)
    assert entry["bytes"] == size == len(page.text.encode("utf-8")) and entry["budget_bytes"] == budget
    assert size <= budget
    names = page_checks.check_figures(page, figure_sizes(page, chain))
    assert len(names) == len(set(names)) and "time" in names
    assert ("heatmap" in names) == (not page.tables["registration"].empty)
    assert load_report(page.folder) == page.record
    page_checks.test_pg6_no_verdict_and_no_score(as_case, "chain", None)
    # RC12: every null cell, summary key and rule key is n/a by its rule or named by exactly one row.
    check_null_values(page)
    assert keyed(page) == collection_checks.ordered(keyed(page))
    return names


def export_table(chain: Chain, name):
    return measurement_checks.export_table(chain.root, chain.sample, name)


def by_fov(frame):
    return frame.set_index("fov_id")


def null_and_named(page, table, column, key, reason):
    """The cell is null and named by exactly one row of unavailable.csv, with this reason."""
    frame = page.tables[table]
    keys = [collection_checks.key_of(table, row) for row in frame.to_dict("records")]
    value = frame[column].iloc[keys.index(key)]
    rows = [u for u in keyed(page) if u[1:4] == (table, column, key)]
    assert pd.isna(value) and [u[4] for u in rows] == [reason], (table, column, key, value, rows)


def whole_rows(page):
    return [u for u in keyed(page) if u[1] is None]


# --- CH1 to CH3: the report at each stage, in the same default folder -------------------------------------------

def _move(path: Path, holder: Path):
    holder.mkdir(parents=True, exist_ok=True)
    target = holder / "__".join(path.relative_to(path.parents[2]).parts)
    path.rename(target)
    return target, path


@pytest.fixture(scope="module")
def stages(scene_chain, tmp_path_factory):  # noqa: F811
    """A copy of ``scene_chain`` reported four times through ``Dataset.report`` into its default folder: after the
    per-FOV processing without the placement (CH1, ``ch1_unplaced``) and with it (``ch1``), after the assignment
    (CH2) and after the export (CH3, two marker groups and two pairs); the later outputs are moved aside and back,
    so that each stage's files are those the chain wrote. Each report's folder is copied aside before the next."""
    base = tmp_path_factory.mktemp("stages")
    chain = scene_chain.copy(base / "chain")
    genes = genes_of(chain)
    holder = base / "held"
    moved = {"export": [_move(chain.root / "export" / chain.sample, holder)],
             "stitching": [_move(chain.root / "stitching" / chain.sample, holder)],
             "assignment": [_move(chain.root / "checkpoints" / f / "assignment", holder) for f in chain.fov_ids]}
    pages, folders = {}, []

    def stage(name, restore, config):
        for target, place in moved.get(restore, []):
            target.rename(place)
        pages[name] = report(chain, config, base / "pages" / name)
        folders.append(page_checks.snapshot(chain.root / "report" / chain.sample))
    stage("ch1_unplaced", None, ReportConfig())
    stage("ch1", "stitching", ReportConfig())
    stage("ch2", "assignment", ReportConfig())
    stage("ch3", "export", marker_config(genes))
    return chain, pages, folders


def stage_rows(chain: Chain, *, placement, assignment, export):
    """The rows of unavailable.csv the contract's stage table gives for this chain (docs/report-contract.md, "The
    report at each stage of a sample's processing"), with the reasons of "Parts, their inputs and their reasons"."""
    fovs = list(chain.fov_ids)
    rows = [collection_checks.ALIGNMENT_RULE]  # the default configuration sets no alignment-score limit
    n_steps = {f: len(chain.run_record(f)["steps"]) for f in fovs}
    if placement:
        rows.append(collection_checks.STITCH_TIME)  # no record holds the time of stitching
    else:
        # The FOV maps and the placement block; the placement rules; the placement table of Files. The Z size is the
        # reference image's (test_ch1_ch2_ch3_the_report_at_each_stage checks it), so no row names it.
        rows += [("registration.maps", None, None, "", "no_placement"),
                 ("readout.per_fov", None, None, "", "no_placement"),
                 ("rule:not_placed", None, None, "", "no_placement"),
                 ("rule:stage_position_fallback", None, None, "", "no_placement"),
                 ("files.input", None, None, "entry=positions", "no_placement")]
        rows += [("assembly.placement", "fovs", c, f"fov_id={f}", "no_placement")
                 for c in collection_checks.PLACEMENT_COLUMNS for f in fovs]
    if assignment:
        # The per-FOV lane's segmentation and assignment: steps whose time no record holds.
        rows += [("workflow.time", "steps", "seconds", f"scope=fov;fov_id={f};index={n_steps[f] + i}", "not_recorded")
                 for f in fovs for i in (0, 1)]
    else:
        # The segmentation examples read the assignment folder on the route per FOV; it does not exist yet.
        roles = expected_roles(chain)
        rows += [("assembly.examples", None, None, f"fov_id={roles[r]};role={r}", "input_missing") for r in ROLES]
    if export:
        rows.append(("workflow.time", "steps", "seconds", "scope=sample;fov_id=;index=1", "not_recorded"))  # assemble
    else:
        rows.append(collection_checks.READS_NO_CELL_RULE)
        for row in collection_checks.no_export_rows(fovs):
            if not placement and row[2] in ("owned_area_um2", "cell_density_per_mm2"):
                row = (*row[:4], "no_placement")  # the first reason that applies (P1): no_placement before no_export
            rows.append(row)
    return rows


@pytest.mark.slow
def test_ch1_ch2_ch3_the_report_at_each_stage(stages, tmp_path):
    chain, pages, folders = stages
    expected = {"ch1_unplaced": dict(placement=False, assignment=False, export=False),
                "ch1": dict(placement=True, assignment=False, export=False),
                "ch2": dict(placement=True, assignment=True, export=False),
                "ch3": dict(placement=True, assignment=True, export=True)}
    for name, stage in expected.items():
        page = pages[name]
        assert sorted(keyed(page)) == sorted(stage_rows(chain, **stage)), name
        record = page.record
        assert record["stages"]["run_records"] == list(chain.fov_ids) and record["stages"]["placement"] is \
            stage["placement"] and record["stages"]["export"] is stage["export"], name
        assert record["route"] == ("per_fov" if stage["export"] else None), name
        # Every stage holds the Summary's key numbers up to the matched fraction, the readout and the examples.
        summary = record["summary"]
        assert summary["fovs_succeeded"] == len(chain.fov_ids) and summary["spots_median_per_fov"] is not None
        assert {r: (e or {}).get("fov_id") for r, e in record["examples"].items() if r in ROLES} == \
            (expected_roles(chain) if not stage["export"] else expected_roles(chain, cells=dict(
                by_fov(export_table(chain, "fovs"))["n_cells_retained"].items()))), name
        # The per-FOV lane: segment and assign_molecules rows once the assignment exists.
        steps = page.tables["steps"]
        lane = steps[steps["scope"].eq("fov").to_numpy(bool)
                     & steps["name"].isin(["segment", "assign_molecules"]).to_numpy(bool)]
        assert len(lane) == (2 * len(chain.fov_ids) if stage["assignment"] else 0), name
        check_page(page, chain, tmp_path / name)
    # The Z size at every stage: without an export or a placement, the shape of the first FOV's reference image, read
    # from its header and listed in inputs; the acquisition says "3D" for the scene's 5 planes.
    reference = f"images/ref_merged/{chain.fov_ids[0]}.tif"
    planes = tifffile.imread(chain.root / reference).shape[0]
    assert planes == 5
    for name, page in pages.items():
        facts = re.search(r'<section[^>]* id="summary".*?</section>', page.text, re.S).group(0)
        assert re.search(r"<dt>Acquisition</dt><dd>4 rounds × 4 channels, 3D</dd>", facts), name
        listed = {e["path"]: e for e in page.record["inputs"]}
        if name == "ch1_unplaced":
            assert listed[reference]["kind"] == "reference_image" and listed[reference]["sha256"] == \
                hashlib.sha256((chain.root / reference).read_bytes()).hexdigest()
    # The placement block with the placement: every placement column holds a value.
    for name in ("ch1", "ch2", "ch3"):
        fovs = pages[name].tables["fovs"]
        assert fovs[list(collection_checks.PLACEMENT_COLUMNS)].notna().all().all(), name
    # The present parts grow with the stages: CH1 ⊆ CH2 ⊆ CH3, and CH3 holds every part.
    present = {name: set(present_parts(page)) for name, page in pages.items()}
    assert present["ch1_unplaced"] <= present["ch1"] <= present["ch2"] <= present["ch3"] == set(REPORT_PARTS)
    assert set(LATER_PARTS) - {"assembly.placement"} <= set(REPORT_PARTS) - present["ch1"]
    assert "assembly.examples" in present["ch2"] and "assembly.cells_per_fov" not in present["ch2"]
    # Each later report replaced the earlier in the same folder: the folder held only the latest report's files.
    for (name, page), files in zip(pages.items(), folders):
        assert files == page_checks.snapshot(page.folder), name
        assert sorted(files) == sorted(["report.html", "report.json"] + [e["path"] for e in page.record["files"]
                                                                         if e["path"] != "report.html"])
    assert all(a["report.json"] != b["report.json"] for a, b in zip(folders, folders[1:]))
    assert sorted(p.name for p in (chain.root / "report").iterdir()) == [chain.sample]


@pytest.mark.slow
def test_ch3_the_reads_assigned_to_cells(stages):
    """CH3's reads check: reads_in_cells equals Σ X of sample.h5ad (read with anndata) and the molecules of the counted
    cells (identity 4 of docs/assembly-contract.md), per FOV as this test counts them from the export's tables."""
    chain, pages, _ = stages
    page = pages["ch3"]
    _, _, _, X = measurement_checks.matrix(chain.root, chain.sample)
    cells, molecules = export_table(chain, "cells"), export_table(chain, "molecules")
    counted = cells["assembly_status"].eq("retained").to_numpy(bool) & cells["status"].eq("kept").to_numpy(bool)
    summary = page.record["summary"]
    assert summary["reads_in_cells"] == summary["h5ad_whole_count_sum"] == int(np.rint(X.sum())) == \
        int(cells["n_molecules"][counted].sum())
    assert summary["reads_in_cells_equal"] is True and summary["counted_cells"] == X.shape[0] == int(counted.sum())
    per_fov = measurement_checks.expected_reads_in_cells(molecules, cells, chain.fov_ids)
    assert by_fov(page.tables["fovs"])["reads_in_cells"].to_dict() == per_fov
    assert page.tables["readout_totals"].set_index("level").loc["reads_in_cells", "count"] == sum(per_fov.values())
    # The cells and MECR of the export, on the stack.
    measurement_checks.check_cells_and_mecr(chain, page.case.config, page.tables, "stack")


# --- RT1, RT2, Z1: both routes on the one-plane scene ------------------------------------------------------------

@pytest.fixture(scope="module")
def plane_pages(scene_chain_plane, tmp_path_factory):  # noqa: F811
    """The report of each route of ``scene_chain_plane``, with two marker groups and two pairs."""
    base = tmp_path_factory.mktemp("plane_pages")
    return {route: report(chain, marker_config(genes_of(chain)), base / route)
            for route, chain in scene_chain_plane.items()}


def owned(chain: Chain):
    """The owner of each export cell's ownership centroid and each FOV's owned pixel centres, with the package's
    ``ownership_map`` of the placement, the export's FOV grids and its assembly configuration (as DF5 counts them)."""
    from starfinder.assembly import AssemblyConfig, ownership_map
    from starfinder.image import ImageMetadata
    from starfinder.segmentation import ReferenceGrid
    from starfinder.stitching import load_placement
    export = json.loads((chain.root / "export" / chain.sample / "export.json").read_text())
    grids = {e["fov_id"]: ReferenceGrid(tuple(e["grid"]["shape_zyx"]), ImageMetadata(**e["grid"]["metadata"]),
                                        "declared") for e in export["assembly"]["fovs"]}
    stitching = chain.root / "stitching" / chain.sample
    owner = ownership_map(load_placement(stitching), grids, config=AssemblyConfig(**export["assembly"]["config"]))
    cells = export_table(chain, "cells")
    centroids = np.stack([cells[f"sample_ownership_centroid_{a}"].astype("float64").to_numpy(np.float64,
                                                                                            na_value=np.nan)
                          for a in "zyx"], axis=1)
    cell_owner = owner.owner(centroids)
    origins = pd.read_csv(stitching / "fovs.csv", float_precision="round_trip").set_index("fov_id")
    pixels = {}
    for fov, grid in grids.items():
        (_, sy, sx), (_, ny, nx) = grid.metadata.spacing_zyx, grid.shape_zyx
        i, j = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        oz, oy, ox = (float(origins.loc[fov, f"origin_{a}"]) for a in "zyx")
        points = np.stack([np.full(i.size, oz), oy + sy * i.ravel(), ox + sx * j.ravel()], axis=1)
        pixels[fov] = (int((owner.owner(points) == fov).sum()), sy * sx)
    return {f: int((cell_owner == f).sum()) for f in chain.fov_ids}, pixels


@pytest.mark.slow
def test_rt1_segmentation_and_assembly_on_both_routes(scene_chain_plane, plane_pages, tmp_path):  # noqa: F811
    pages = plane_pages
    for route, page in pages.items():
        assert page.record["route"] == route
        check_page(page, scene_chain_plane[route], tmp_path / route)
    # The same columns, with the same dtypes, on both routes.
    for name in ("fovs", "population"):
        assert pages["per_fov"].dtypes[name] == pages["after_fusion"].dtypes[name], name
    for route, page in pages.items():
        chain = scene_chain_plane[route]
        fovs, export_fovs = by_fov(page.tables["fovs"]), by_fov(export_table(chain, "fovs"))
        for column in ("n_cells", "n_cells_retained", "n_molecules", "n_molecules_retained"):
            assert fovs[column].to_dict() == export_fovs[column].astype("int64").to_dict(), (route, column)
        cell_counts, pixels = owned(chain)
        if route == "after_fusion":
            # After fusion each FOV's cells are those whose ownership centroid it owns, so the two columns are equal.
            assert fovs["n_cells"].to_dict() == fovs["n_cells_retained"].to_dict() == cell_counts
        # The area a FOV owns and the cell density, by the same formulas on both routes.
        for fov in chain.fov_ids:
            count, pixel = pixels[fov]
            assert fovs.loc[fov, "owned_area_um2"] == pytest.approx(count * pixel, abs=TOL), (route, fov)
            assert fovs.loc[fov, "cell_density_per_mm2"] == pytest.approx(
                fovs.loc[fov, "n_cells_retained"] / (count * pixel) * 1e6, abs=TOL), (route, fov)
        # Population accounting: the export's counts, counted the rows of sample.h5ad and the retained molecules with a
        # counted cell; after fusion the duplicate rows are n/a (null, named by no row).
        population = page.tables["population"].set_index("population")
        counts = json.loads((chain.root / "export" / chain.sample / "export.json").read_text())["assembly"]["counts"]
        for column in ("cells", "molecules"):
            assert population.loc["input", column] == counts[column]["input"], (route, column)
            assert population.loc["retained", column] == counts[column]["retained"], (route, column)
            for reason, count in counts[column]["excluded"].items():
                assert population.loc[f"excluded_{reason}", column] == count, (route, column, reason)
        _, _, _, X = measurement_checks.matrix(chain.root, chain.sample)
        reads = measurement_checks.expected_reads_in_cells(export_table(chain, "molecules"),
                                                           export_table(chain, "cells"), chain.fov_ids)
        assert population.loc["counted", "cells"] == X.shape[0]
        assert population.loc["counted", "molecules"] == sum(reads.values()) == int(np.rint(X.sum()))
        named = {(u[2], u[3]) for u in keyed(page) if u[1] == "population"}
        assert not named, route
        duplicates = [("molecules", "excluded_duplicate_observation")]  # a reason of cells only, on both routes
        if route == "after_fusion":
            duplicates += [("cells", "excluded_duplicate_observation"),
                           ("molecules", "excluded_cell_duplicate_observation")]
        for column, row in duplicates:
            assert pd.isna(population.loc[row, column]), (route, row, column)
        if route == "per_fov":
            assert not pd.isna(population.loc["excluded_duplicate_observation", "cells"])
        # The example tiles: the segmentation examples are drawn for the three roles (their labels against the label
        # images: test_report_replay.py::test_rt1_segmentation_previews).
        assert not [u for u in keyed(page) if u[0] == "assembly.examples"]
        assert 'data-figure="segmentation"' in page.text


@pytest.mark.slow
def test_rt2_sample_measurements_on_both_routes(scene_chain_plane, plane_pages):  # noqa: F811
    for route, page in plane_pages.items():
        chain = scene_chain_plane[route]
        measurement_checks.check_cells_and_mecr(chain, page.case.config, page.tables, route)
        # The marker figures drawn on both routes: each group's whole view and three XY zooms, decoded at their size
        # (check_figures, in test_rt1's check_page); no XZ zoom on a plane.
        names = page_checks.check_figures(page, figure_sizes(page, chain))
        for k in range(len(page.case.config.markers)):
            assert f"marker:{k}:whole" in names
            assert [n for n in names if n.startswith(f"marker:{k}:example_")] == [
                f"marker:{k}:{r}:xy" for r in ROLES]


@pytest.mark.slow
def test_z1_one_plane(scene_chain_plane, plane_pages):  # noqa: F811
    chain, page = scene_chain_plane["per_fov"], plane_pages["per_fov"]
    registration = page.tables["registration"]
    assert len(registration) == len(chain.fov_ids) * (len(chain.rounds) - 1)
    assert registration[["shift_z", "residual_z"]].abs().le(TOL).all().all()  # 0, within the float tolerance
    along = page.tables["reads_along_z"]
    molecules = export_table(chain, "molecules")
    assert along[["fov_id", "z"]].values.tolist() == [[f, 0] for f in chain.fov_ids]
    assert along["reads"].tolist() == [int(molecules["fov_id"].eq(f).sum()) for f in chain.fov_ids]
    roles = expected_roles(chain, cells=dict(by_fov(export_table(chain, "fovs"))["n_cells_retained"].items()))
    assert [u for u in keyed(page) if u[0].startswith("sample.")] == [
        ("sample.marker_zoom", None, None, f"fov_id={roles[r]};role={r};view=xz", "no_stack") for r in ROLES]
    images = [i["data-figure"] for i in page_checks.Elements(page.text).images]
    assert [n for n in images if n.endswith(":xy")] == [f"marker:{k}:{r}:xy" for k in range(2) for r in ROLES]
    assert not [n for n in images if n.endswith(":xz")]
    facts = re.search(r'<section[^>]* id="summary".*?</section>', page.text, re.S).group(0)
    assert re.search(r">\d+ rounds × \d+ channels, 2D \(Z = 1\)<", facts)
    assert not [u for u in keyed(page) if u[3] == "entry=z_size"]
    # PG1 to PG5 hold on this page: test_rt1_segmentation_and_assembly_on_both_routes checks every page of the routes.


# --- A2: an export and no run.json --------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scene_no_run_json(scene_chain, tmp_path_factory):  # noqa: F811
    """``scene_chain``'s output root with every ``checkpoints/<fov>/run.json`` removed (``cells``), and a second root
    with no checkpoint folder, exported with ``assignment=None`` from the rule outputs ``signal/<fov>_goodSpots.csv``
    alone (``molecules_only``; the reference images ``images/ref_merged`` fused, since the stains live under the
    checkpoint folder)."""
    from starfinder.dataset import Dataset, RoundState
    from starfinder.export import ExportConfig, FusionConfig
    base = tmp_path_factory.mktemp("scene_no_run_json")
    cells = scene_chain.copy(base / "cells")
    for fov in cells.fov_ids:
        (cells.root / "checkpoints" / fov / "run.json").unlink()
    molecules = scene_chain.copy(base / "molecules_only")
    shutil.rmtree(molecules.root / "checkpoints")
    shutil.rmtree(molecules.root / "export")
    dataset = Dataset(molecules.input_root, molecules.root, molecules.dataset_id, molecules.sample, "output",
                      rounds=RoundState(sequencing_rounds=molecules.rounds, reference_round=molecules.rounds[0]),
                      channel_order=molecules.channels)
    dataset.codebook = molecules.book
    dataset.export(molecules.sample, placement=molecules.root / "stitching" / molecules.sample, assignment=None,
                   output_dir=molecules.root / "export" / molecules.sample,
                   config=ExportConfig(route="per_fov", fusion=FusionConfig(images=("reference_merged",),
                                                                            dimensionality="both")))
    return {"cells": cells, "molecules_only": molecules}


@pytest.mark.slow
@pytest.mark.parametrize("root", ["cells", "molecules_only"])
@pytest.mark.parametrize("named", [False, True], ids=["chosen", "example_fovs"])
def test_a2_an_export_and_no_run_json(scene_no_run_json, tmp_path, root, named):
    chain = scene_no_run_json[root]
    example_fovs = (chain.fov_ids[2], chain.fov_ids[0], chain.fov_ids[1]) if named else None
    config = marker_config(genes_of(chain), example_fovs=example_fovs)
    assignment = "default" if root == "cells" else None
    page = report(chain, config, tmp_path / "page", assignment=assignment)
    check_page(page, chain, tmp_path / "checks")
    record, rows, fovs = page.record, keyed(page), by_fov(page.tables["fovs"])
    assert record["stages"]["run_records"] == [] and record["stages"]["export"] is True
    assert record["route"] == "per_fov" and not fovs["run_record"].any()
    # Every column up to the readout null with no_run_record, one row each.
    for fov in chain.fov_ids:
        for column in RUN_COLUMNS:
            null_and_named(page, "fovs", column, f"fov_id={fov}", "no_run_record")
    for level in ("spots", "correct_form", "matched", "merged_by_deduplication", "passed_filter"):
        null_and_named(page, "readout_totals", "count", f"level={level}", "no_run_record")
    assert page.tables["registration"].empty
    assert not page.tables["steps"]["name"].isin(["load_images", "register", "find_spots"]).any()
    for part, key in RUN_RECORD_PARTS:
        if named and part in REPLAY_PARTS:
            key = f"fov_id={example_fovs[0]};role=example_1"  # the replay of the named main example
        assert [u[4] for u in whole_rows(page) if u[0] == part and u[3] == key] == ["no_run_record"], (part, key)
    # The Summary: the Z size from the export (the acquisition's rounds and channels need a run record).
    assert not [u for u in rows if u[3] == "entry=z_size"]
    facts = re.search(r'<section[^>]* id="summary".*?</section>', page.text, re.S).group(0)
    assert re.search(r"<dt>Acquisition</dt><dd>.*?, 3D<", facts)
    # Segmentation and assembly and Sample measurements present, with the export's values.
    export_fovs = by_fov(export_table(chain, "fovs"))
    for column in ("n_molecules", "n_molecules_retained"):
        assert fovs[column].to_dict() == export_fovs[column].astype("int64").to_dict(), column
    assert fovs[list(collection_checks.PLACEMENT_COLUMNS)].notna().all().all()
    for part in ("assembly.placement", "sample.markers", "sample.reads_along_z"):
        assert not [u for u in rows if u[0] == part], part
    molecules = export_table(chain, "molecules")
    assert int(page.tables["reads_along_z"]["reads"].sum()) == len(molecules)
    if root == "cells":
        for column in ("n_cells", "n_cells_retained"):
            assert fovs[column].to_dict() == export_fovs[column].astype("int64").to_dict(), column
        cells = export_table(chain, "cells")
        assert fovs["reads_in_cells"].to_dict() == measurement_checks.expected_reads_in_cells(molecules, cells,
                                                                                              chain.fov_ids)
        _, _, _, X = measurement_checks.matrix(chain.root, chain.sample)
        summary = record["summary"]
        assert summary["reads_in_cells"] == int(np.rint(X.sum())) and summary["reads_in_cells_equal"] is True
        measurement_checks.check_cells_and_mecr(chain, config, page.tables, "stack")
        for part in ("assembly.cells_per_fov", "assembly.accounting", "sample.per_cell", "sample.mecr"):
            assert not [u for u in rows if u[0] == part], part
    else:
        for fov in chain.fov_ids:
            for column in ("n_cells", "n_cells_retained", "reads_in_cells", "reads_in_cells_fraction",
                           "cell_density_per_mm2"):
                null_and_named(page, "fovs", column, f"fov_id={fov}", "no_segmentation")
        assert [u[4] for u in whole_rows(page) if u[0] == "sample.per_cell"] == ["no_segmentation"]
        assert page.tables["cells"].empty
        # MECR: each cell of each pair unavailable on its own (docs/report-contract.md, "mecr.csv").
        mecr = page.tables["mecr"]
        assert list(zip(mecr["gene_a"], mecr["gene_b"])) == list(config.marker_pairs)
        for a, b in config.marker_pairs:
            for column in ("cells_a", "cells_b", "both", "either", "mecr"):
                null_and_named(page, "mecr", column, f"gene_a={a};gene_b={b}", "no_segmentation")
    # The examples: without example_fovs no read count chooses them (no_run_record); with them, the named FOVs are the
    # roles in the given order and the zoom-in regions of the marker figures.
    examples = [u for u in whole_rows(page) if u[0] in ("assembly.examples", "sample.marker_zoom")]
    if not named:
        assert examples == [("assembly.examples", None, None, "", "no_run_record"),
                            ("sample.marker_zoom", None, None, "", "no_run_record")]
        assert all(record["examples"][r] is None for r in ROLES)
    else:
        assert {r: record["examples"][r]["fov_id"] for r in ROLES} == dict(zip(ROLES, example_fovs))
        assert {record["examples"][r]["chosen_by"] for r in ROLES} == {"configuration"}
        assert [u for u in examples if u[0] == "sample.marker_zoom"] == []
        if root == "molecules_only":
            assert examples == [("assembly.examples", None, None, f"fov_id={f};role={r}", "no_segmentation")
                                for r, f in zip(ROLES, example_fovs)]
        else:
            assert examples == []
        from starfinder.report._collect import report_measurements
        drawn = report_measurements(chain.root, chain.sample, chain.fov_ids, dataset_id=chain.dataset_id,
                                    sample_id=chain.sample, assignment=assignment, config=config)
        for figure in drawn.markers:
            for k, (role, fov) in enumerate(zip(ROLES, example_fovs), start=1):
                assert figure.zooms[(role, "xy")].tiles[0]["title"].startswith(f"Region {k} · {fov} · ")


# --- CP1: the comparison page of two synthetic samples -------------------------------------------------------------

@pytest.fixture(scope="module")
def scene_two_samples(scene_chain, scene_chain_plane, tmp_path_factory):  # noqa: F811
    """``scene_chain`` reported as sample ``s1`` (into ``report/s1``), and ``scene_chain_plane``'s route per FOV saved
    as sample ``s2`` under the same output root (its placement and export folders), without a report. The scene's
    sample id is fixed by the preset, so ``s1`` and ``s2`` are the names of their folders."""
    base = tmp_path_factory.mktemp("scene_two_samples")
    chain = scene_chain.copy(base / "chain")
    plane = scene_chain_plane["per_fov"]
    for folder in ("stitching", "export"):
        shutil.copytree(plane.root / folder / plane.sample, chain.root / folder / "s2")
    config = marker_config(genes_of(chain))
    result = dataset_of(chain).report(chain.sample, chain.fov_ids, config=config,
                                      output_dir=chain.root / "report" / "s1")
    page = page_checks.Page(ChainCase(chain, config), result.path, result.record)
    return chain, page


@pytest.mark.slow
def test_cp1_the_comparison_page(scene_two_samples, tmp_path):
    chain, page = scene_two_samples
    check_page(page, chain, tmp_path)
    report_root = chain.root / "report"
    s1 = page_checks.snapshot(report_root / "s1")
    result = dataset_of(chain).compare_samples(["s1", "s2"])
    assert result.path == report_root
    frame, text, dtypes = comparison_checks.written(result)
    assert dtypes == {c: d for c, d, _ in comparison_checks.SAMPLES_COLUMNS}
    rows = frame.to_dict("records")
    assert [(r["sample"], r["report"]) for r in rows] == [("s1", "present"), ("s2", "missing")]
    record = json.loads((report_root / "s1" / "report.json").read_text(encoding="utf-8"))
    assert rows[0]["report_sha256"] == hashlib.sha256((report_root / "s1" / "report.json").read_bytes()).hexdigest()
    for column, _, path in comparison_checks.SAMPLES_COLUMNS:
        if path is not None:
            assert comparison_checks.equal(rows[0][column], comparison_checks.key(record, path)), column
    assert all(comparison_checks.missing_cell(rows[1][c]) for c, _, _ in comparison_checks.SAMPLES_COLUMNS
               if c not in ("sample", "report"))
    # The box plots' embedded values equal s1's fovs.csv columns; s2 is an empty slot labelled "missing".
    _, data = comparison_checks.data_block(text)
    assert list(data["fovs"]) == ["s1"]
    fovs = page.tables["fovs"]
    for row, saved in zip(data["fovs"]["s1"], fovs.to_dict("records")):
        assert row["fov_id"] == saved["fov_id"]
        for column in comparison_checks.BOX_COLUMNS:
            assert row[column] == pytest.approx(float(saved[column]), abs=TOL), column
    assert [r["fov_id"] for r in data["fovs"]["s1"]] == list(chain.fov_ids)
    for column in comparison_checks.BOX_COLUMNS:
        assert [(s["sample"], s["fovs"], s["label"]) for s in result.record["box_plots"][column]] == [
            ("s1", len(chain.fov_ids), None), ("s2", 0, "missing")]
    # PG4 on index.html: nothing loaded, the one link a relative s1/report.html.
    assert not re.search(r"https?:", text, re.I) and not re.search(r'(src|href)="//', text)
    for word in ("<link", "<iframe", "<object", "@import", "prefers-color-scheme"):
        assert word not in text.lower(), word
    assert all(s.startswith("data:") for s in re.findall(r'\ssrc="([^"]*)"', text))
    assert all(text[m.end():].startswith("data:") for m in re.finditer(r"url\(", text))
    assert re.findall(r'\shref="([^"]*)"', text) == ["s1/report.html"]
    # A table of s1 changed after its report raises ValueError naming the file; nothing is written.
    earlier = {name: (report_root / name).read_bytes() for name in ("index.html", "samples.csv")}
    path = report_root / "s1" / "tables" / "fovs.csv"
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match=re.escape(str(path))):
        dataset_of(chain).compare_samples(["s1", "s2"])
    assert {name: (report_root / name).read_bytes() for name in earlier} == earlier
    path.write_bytes(original)
    # Nothing under s1/ was written, and no folder of s2 was made.
    assert page_checks.snapshot(report_root / "s1") == s1
    assert sorted(p.name for p in report_root.iterdir()) == ["index.html", "s1", "samples.csv"]


# --- DT1: determinism of the chain ----------------------------------------------------------------------------------

@pytest.mark.slow
def test_dt1_the_chain_in_two_single_thread_processes(scene_chain, tmp_path):  # noqa: F811
    chain = scene_chain
    genes = genes_of(chain)
    config = marker_config(genes)
    arguments = json.dumps({"input_root": str(chain.input_root), "root": str(chain.root),
                            "dataset_id": chain.dataset_id, "sample": chain.sample, "fov_ids": list(chain.fov_ids),
                            "markers": [[g.name, list(g.genes)] for g in config.markers],
                            "pairs": [list(p) for p in config.marker_pairs]})
    source = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(source), os.environ.get("PYTHONPATH", "")]),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               NUMBA_NUM_THREADS="1")
    folder = chain.root / "report" / chain.sample
    copies = []
    for run in range(2):
        subprocess.run([sys.executable, "-c", DT1_SCRIPT, arguments], check=True, env=env, cwd=source)
        aside = tmp_path / f"run{run}"
        shutil.copytree(folder, aside)
        copies.append(aside)
    first, second = (page_checks.snapshot(c) for c in copies)
    assert sorted(first) == sorted(second) == sorted(["report.html", "report.json"]
                                                     + [f"tables/{t}.csv" for t in collection_checks.COLLECTED])
    for name in first:
        if name not in ("report.json", "report.html"):
            assert first[name] == second[name], name
    records = [json.loads(s["report.json"]) for s in (first, second)]
    assert records[0]["generated_at"] != records[1]["generated_at"]
    for record in records:
        for key in collection_checks.LEFT_OUT:
            del record[key]
        entry = next(e for e in record["files"] if e["path"] == "report.html")
        del entry["sha256"], entry["bytes"]
    assert records[0] == records[1]
    pages = [collection_checks.GENERATED.sub(r"\1\2", s["report.html"].decode("utf-8")) for s in (first, second)]
    assert pages[0] != first["report.html"].decode("utf-8") and pages[0] == pages[1]
    # The chain's report holds every part: the replayed examples, the segmentation examples and the marker figures.
    assert records[1]["replay"]["fov_id"] in chain.fov_ids
    assert [u["part"] for u in records[1]["unavailable"] if u["table"] is None] == ["rule:alignment_score"]


# --- The Z size from a reference image (DF12; docs/report-contract.md, "Where each record is looked for") ---------

def z_size_case(tmp_path):
    """``report_records`` ``no_placement``: run records only, so no record holds a grid."""
    from .report_fixtures import report_records
    return report_records(tmp_path / "root", "no_placement")


def z_size_rows(record):
    return [(u["part"], u["row_key"], u["reason"], u["detail"]) for u in record["unavailable"]
            if u["row_key"] == "entry=z_size"]


def reference_inputs(record, root):
    """(path, kind, sha256, bytes, verified) of every reference_image input, with the hash expected of its file."""
    listed = [(e["path"], e["kind"], e["sha256"], e["bytes"], e["verified"]) for e in record["inputs"]
              if e["kind"] == "reference_image"]
    expected = [(e[0], "reference_image", hashlib.sha256((Path(root) / e[0]).read_bytes()).hexdigest(),
                 (Path(root) / e[0]).stat().st_size, False) for e in listed]
    assert listed == expected
    return [e[0] for e in listed]


def acquisition_text(case):
    page = (case.folder / "report.html").read_text(encoding="utf-8")
    return re.search(r"<dt>Acquisition</dt><dd>(.*?)</dd>", page, re.S).group(1)


def direct_z_size(root, fov_ids, pattern=None):
    """reference_image_z_size called on its own, with a fresh register of inputs."""
    from starfinder.report._collect import _Inputs
    from starfinder.report._values import reference_image_z_size
    return reference_image_z_size(_Inputs(Path(root)), Path(root), fov_ids, pattern)


def stack(path, planes, dtype):
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (planes, 16, 16) if planes > 1 else (16, 16)
    tifffile.imwrite(path, np.zeros(shape, dtype), photometric="minisblack")


def test_z_size_without_a_reference_image_is_not_recorded(tmp_path):
    """No record holds a grid and no FOV has a reference image: the Z size is unavailable with not_recorded."""
    case = z_size_case(tmp_path)
    _, record = case.collect()
    assert [r[:3] for r in z_size_rows(record)] == [("summary.identity", "entry=z_size", "not_recorded")]
    assert reference_inputs(record, case.root) == []
    assert acquisition_text(case).startswith("3 rounds × 4 channels <span")
    assert direct_z_size(case.root, case.fov_ids) is None


@pytest.mark.parametrize("dtype", [np.float32, np.uint16], ids=["float32", "uint16"])
def test_z_size_from_a_reference_image_without_a_grid(tmp_path, dtype):
    """Only F1 has a reference image (3 planes; float32 or uint16), no record holds a grid: the Z size is its shape,
    read from the header, and the file is listed in inputs. With one plane, 2D (Z = 1)."""
    case = z_size_case(tmp_path)
    image = case.root / "images" / "ref_merged" / "F1.tif"
    stack(image, 3, dtype)
    assert direct_z_size(case.root, case.fov_ids) == 3
    _, record = case.collect()
    assert z_size_rows(record) == []
    assert reference_inputs(record, case.root) == ["images/ref_merged/F1.tif"]
    assert acquisition_text(case) == "3 rounds × 4 channels, 3D"
    image.unlink()
    stack(image, 1, dtype)
    assert direct_z_size(case.root, case.fov_ids) == 1
    case.collect()
    assert acquisition_text(case) == "3 rounds × 4 channels, 2D (Z = 1)"


def test_z_size_from_the_first_fov_with_a_reference_image(tmp_path):
    """F1 has none, F2 a uint16 image of 7 planes, F3 a float32 image of 3: F2's, the first in list order, is read and
    is the only reference image listed."""
    case = z_size_case(tmp_path)
    images = case.root / "images" / "ref_merged"
    stack(images / "F2.tif", 7, np.uint16)
    stack(images / "F3.tif", 3, np.float32)
    assert direct_z_size(case.root, case.fov_ids) == 7
    _, record = case.collect()
    assert z_size_rows(record) == [] and reference_inputs(record, case.root) == ["images/ref_merged/F2.tif"]


def test_z_size_when_no_reference_image_can_be_read(tmp_path):
    """F1's file is not a TIFF and F2's is a TIFF of three channels (not a ZYX or YX image): the Z size is unavailable
    with input_changed naming F1's file, and both files are listed. With a readable F3, F3's is read."""
    from starfinder.report._values import Unavailable
    case = z_size_case(tmp_path)
    images = case.root / "images" / "ref_merged"
    images.mkdir(parents=True)
    (images / "F1.tif").write_bytes(b"not a tiff\n" * 8)
    tifffile.imwrite(images / "F2.tif", np.zeros((2, 3, 16, 16), np.uint16), imagej=True, metadata={"axes": "ZCYX"})
    found = direct_z_size(case.root, case.fov_ids)
    assert isinstance(found, Unavailable) and found.reason == "input_changed"
    _, record = case.collect()
    (row,) = z_size_rows(record)
    assert row[:3] == ("summary.identity", "entry=z_size", "input_changed")
    assert row[3].startswith("images/ref_merged/F1.tif cannot be read as a TIFF")
    assert reference_inputs(record, case.root) == ["images/ref_merged/F1.tif", "images/ref_merged/F2.tif"]
    (images / "F1.tif").unlink()
    (row,) = z_size_rows(case.collect()[1])
    assert row[2] == "input_changed" and row[3].startswith("images/ref_merged/F2.tif is not a ZYX or YX image")
    stack(images / "F3.tif", 5, np.uint16)
    _, record = case.collect()
    assert z_size_rows(record) == [] and acquisition_text(case) == "3 rounds × 4 channels, 3D"
    assert reference_inputs(record, case.root) == ["images/ref_merged/F2.tif", "images/ref_merged/F3.tif"]


def test_z_size_from_a_tiff_of_several_series_or_a_file_gone_while_read(tmp_path):
    """A TIFF of two series (load_volume asks for an index, so the header gives no image) is input_changed; a file that
    is gone between the look-up and its hash is input_missing, the reason the register of inputs gives."""
    from starfinder.report._collect import _Inputs
    from starfinder.report._values import Unavailable, reference_image_z_size
    case = z_size_case(tmp_path)
    image = case.root / "images" / "ref_merged" / "F1.tif"
    image.parent.mkdir(parents=True)
    with tifffile.TiffWriter(image) as writer:
        writer.write(np.zeros((2, 16, 16), np.uint16), photometric="minisblack")
        writer.write(np.zeros((8, 8), np.uint8), photometric="minisblack")
    with tifffile.TiffFile(image) as tif:
        assert len(tif.series) == 2
    (row,) = z_size_rows(case.collect()[1])
    assert row[2] == "input_changed" and row[3].startswith("images/ref_merged/F1.tif is not a ZYX or YX image")

    class Gone(_Inputs):
        def verify(self, path, kind, recorded=None):
            return Unavailable("input_missing", self.relative(path))
    found = reference_image_z_size(Gone(case.root), case.root, case.fov_ids)
    assert found == Unavailable("input_missing", "images/ref_merged/F1.tif")


def test_z_size_from_the_dataset_reference_image_pattern(tmp_path):
    """Through Dataset.report with ``reference_image="refs/{fov_id}_ref.tif"``: the Z size comes from F2's float32
    image of 4 planes there, and that file is listed; a one-plane image under images/ref_merged is not read. Without
    the pattern's files the Z size is not_recorded, although images/ref_merged holds an image."""
    from starfinder.dataset import Dataset

    from .report_fixtures import DATASET, SAMPLE_ID
    case = z_size_case(tmp_path)
    stack(case.root / "refs" / "F2_ref.tif", 4, np.float32)
    stack(case.root / "images" / "ref_merged" / "F1.tif", 1, np.uint16)
    dataset = Dataset(case.root / "input", case.root, DATASET, SAMPLE_ID, "output",
                      reference_image="refs/{fov_id}_ref.tif")
    result = dataset.report(case.sample, list(case.fov_ids), config=case.config)
    assert z_size_rows(result.record) == []
    assert reference_inputs(result.record, case.root) == ["refs/F2_ref.tif"]
    assert acquisition_text(case) == "3 rounds × 4 channels, 3D"
    assert direct_z_size(case.root, case.fov_ids, "refs/{fov_id}_ref.tif") == 4
    shutil.rmtree(case.root / "refs")
    result = dataset.report(case.sample, list(case.fov_ids), config=case.config)
    assert [r[2] for r in z_size_rows(result.record)] == ["not_recorded"]
    assert "refs/{fov_id}_ref.tif" in z_size_rows(result.record)[0][3]
    assert reference_inputs(result.record, case.root) == []


# --- The table "Implemented checks (W-459)" -------------------------------------------------------------------------

def design_rows():
    """The row ids of the table "Checks" of the validation design."""
    text = DESIGN.read_text(encoding="utf-8").split("### Checks\n", 1)[1].split("\n### ", 1)[0]
    return [m.group(1) for m in re.finditer(rf"^\| ({ROW}) \|", text, flags=re.MULTILINE)]


def implemented_checks():
    """{row: (the backticked names of its "Module and tests" cell, its tier)}."""
    section = DESIGN.read_text(encoding="utf-8").split(SECTION, 1)[1].split("\n## ", 1)[0]
    rows = {}
    for line in section.splitlines():
        match = re.match(rf"\| ({ROW}) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|$", line)
        if match:
            assert match.group(1) not in rows, match.group(1)
            rows[match.group(1)] = (re.findall(r"`([^`]+)`", match.group(2)), match.group(3).strip())
    return rows


def named_tests(names):
    """(module, test name) of a row, each name under the module named before it."""
    module, named = None, []
    for name in names:
        if name.endswith(".py"):
            module = name
        elif name.startswith("test_"):
            assert module is not None, name
            named.append((module, name))
    return named


def collected(session, module):
    """The items pytest collects from one test module."""
    items, nodes = [], [pytest.Module.from_parent(session, path=Path(__file__).resolve().parent / module)]
    while nodes:
        node = nodes.pop()
        for child in node.collect():
            (items if isinstance(child, pytest.Item) else nodes).append(child)
    return items


def test_every_check_names_collected_tests(request):
    """Every row of the table "Checks" has a row in "Implemented checks (W-459)", in the same order, naming tests that
    pytest collects, none an expected failure, each with a subsystem marker and the tier the row states."""
    rows, expected = implemented_checks(), design_rows()
    assert len(expected) == len(set(expected)) == 54
    assert list(rows) == expected
    modules = sorted({module for names, _ in rows.values() for module, _ in named_tests(names)})
    items_of = {module: collected(request.session, module) for module in modules}
    for row, (names, tier) in rows.items():
        named = named_tests(names)
        assert named and tier in TIERS, row
        items = []
        for module, name in named:
            found = [item for item in items_of[module] if (item.originalname.startswith(name[:-1])
                                                           if name.endswith("*") else item.originalname == name)]
            assert found, (row, module, name)
            items += found
        for item in items:
            marks = {mark.name for mark in item.iter_markers()}
            assert "xfail" not in marks and SUBSYSTEM_MARKERS & marks, item.nodeid
        slow = {item.get_closest_marker("slow") is not None for item in items}
        assert tuple(sorted(slow)) == TIERS[tier], (row, tier)
