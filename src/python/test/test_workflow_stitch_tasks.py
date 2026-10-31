"""The stitch, bigstitcher-macro, bigstitcher-check and bigstitcher-import commands (moved by W-750 from the tests of
the stitching rules' adapters: test_stitching_workflow.py, test_fov_membership.py and
test_stitching_reference_image.py).

Each test plans a configuration in the Python shape on an input root whose raw images are empty files
(``workflow_tasks``): the reference images the placement reads are supplied (``reference_image_import``) or written by
the test at the default path, and the task runs by its command, or its kind's function is called on the task's
``Run`` with ``Dataset.stitch``, ``stitching.stitch`` or ``import_bigstitcher`` spied, as the adapter tests spied
them. Fiji never runs: a test writes ``dataset.xml`` where Fiji would (the engine task runs the tool, then the
``bigstitcher-check`` command, D3). dask runs with the synchronous scheduler.
"""
import copy
import importlib.util
import json
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import tifffile

from starfinder.dataset import Dataset
from starfinder.io import save_volume
from starfinder.stitching import (BigStitcherConfig, MultiviewStitcherConfig, ReviewPolicy, import_bigstitcher,
                                  load_placement, placement_from_table)
from starfinder.workflow._plan import PlanRefused

from . import workflow_tasks as T
from .stitching_fixtures import LN_SPACING, STAGE_SPACING, grid, maf_text
from .test_stitching_bigstitcher import (LINK_IDS, LINK_SHAPE, golden_stitching_setups,  # noqa: F401
                                         links_table, links_view_map, links_xml)
from .test_stitching_declared_grids import plain_pages
from .test_stitching_reference_image import MOVED, GOLDEN_SHAPE, GOLDEN_SPACING, assert_same_placement, fov_ids
from .test_stitching_reference_image import scene_layouts  # noqa: F401  (the module fixture)
from .test_workflow_stitching_golden import stage_file

pytestmark = [pytest.mark.workflow, pytest.mark.stitching, pytest.mark.contract]

needs_library = pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None,
                                   reason="the multiview-stitcher extra (starfinder[multiview-stitcher]) is not "
                                          "installed")
SAMPLE = "sample1"
GRID = {"source": "regular_grid", "regular_grid": {"columns": 2, "rows": 2, "order": "column_major_down_right",
                                                   "overlap": 0.25}}
MULTIVIEW = {"method": "multiview_stitcher", "input": "reference_stain", "positions": GRID,
             "config": {"binning_zyx": [1, 2, 2], "quality_threshold": -1.0},
             "review": {"max_residual_um": 2.0}, "fallback": "nominal"}
MERGED = dict(MULTIVIEW, input="reference_merged")
STAGE_AXES = {"x": "+x", "y": "+y"}
STAGE_POSITIONS = {"source": "stage_file", "stage_axes": STAGE_AXES, "snap_to_grid": {"overlap": 0.1}}
# A sample of FOVs 9 to 12 on a 2 × 2 stage grid (step 261 µm, tiles of 16 × 24 px at 18 × 12 µm), the stage file in
# the order 9, 10, 12, 11 (test_stitching_workflow.py's stage_dataset).
STAGE_POINTS = ((9, "0.076", "0.048"), (10, "0.076", "0.048261"), (12, "0.076261", "0.048261"),
                (11, "0.076261", "0.048"))
STAGE_SHAPE = (2, 16, 24)


@pytest.fixture(autouse=True)
def synchronous():
    import dask
    with dask.config.set(scheduler="synchronous"):
        yield


def rows(sample, fovs, position_ids=None):
    """fovs.csv: the FOVs of one sample, with their PositionIDs when given."""
    if position_ids is None:
        return "sample,fov_id\n" + "".join(f"{sample},{fov}\n" for fov in fovs)
    return "sample,fov_id,position_id\n" + "".join(f"{sample},{f},{p}\n" for f, p in zip(fovs, position_ids))


def imported(work, **keys):
    """The dataset keys with the supplied reference images ``<work>/refs/<fov>.tif`` (reference_image_import)."""
    return T.dataset_keys(work, reference_image_import=str(work / "refs" / "{fov_id}.tif"), **keys)


def reference(work, fov, image, spacing):
    """A supplied reference image with its stored metadata."""
    save_volume(image, work / "refs" / f"{fov}.tif", metadata=grid(image.shape, spacing).metadata)


def spy(monkeypatch, owner, name, result=None, error=None):
    """Replace ``owner.name`` by a recorder of its calls; it returns ``result`` or raises ``error``."""
    calls = []

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        if error is not None:
            raise error
        return result
    monkeypatch.setattr(owner, name, record)
    return calls


# --- stitch: the block's values and the sample's FOVs ------------------------------------------------------------

def test_the_stitch_command_calls_dataset_stitch_with_the_configured_values(tmp_path, monkeypatch):
    """From test_wf3_run_stitching_calls_dataset_stitch_with_the_configured_values: Dataset.stitch of the sample's
    FOVs (fovs.csv) with the block's table, config, input, review and fallback, into stitching/<sample>/."""
    calls = []
    monkeypatch.setattr(Dataset, "stitch", lambda self, *args, **kwargs: calls.append((self, args, kwargs)))
    fovs = [f"Position00{i}" for i in range(1, 5)]
    config = imported(tmp_path, reference_stains=[{"channel": "ch04", "name": "DAPI"}], morphology={},
                      additional_round=[{"round_name": "morph", "channel_order": ["ch00"]}], stitching=MULTIVIEW)
    T.write_inputs(config, rows(SAMPLE, fovs))
    manifest, _ = T.plan(tmp_path, "stitch", config)
    out = T.output_root(config)
    T.call(manifest, f"stitch/{SAMPLE}")
    (dataset, args, kwargs), = calls
    assert dataset.output_root == out and dataset.fov_pattern == "Position{i:03d}"
    assert args == (SAMPLE, fovs)
    assert kwargs == {"table": GRID, "config": MultiviewStitcherConfig(binning_zyx=(1, 2, 2), quality_threshold=-1.0),
                      "input": "reference_stain", "review": ReviewPolicy(max_residual_um=2.0), "fallback": "nominal",
                      "output_dir": out / "stitching" / SAMPLE, "position_ids": None}


def test_a_stage_file_without_stage_axes_is_refused_before_any_task(tmp_path, monkeypatch):
    """From test_wf3_a_stage_file_without_stage_axes_raises_in_the_rule: the plan refuses the block, naming
    stitching.positions.stage_axes, and no stitching runs."""
    monkeypatch.setattr(Dataset, "stitch", lambda *a, **k: pytest.fail("Dataset.stitch must not be called"))
    config = imported(tmp_path, stitching=dict(MERGED, positions={"source": "stage_file"}))
    T.write_inputs(config, rows(SAMPLE, ["Position001"]))
    with pytest.raises(PlanRefused, match=r"stitching\.positions\.stage_axes"):
        T.plan(tmp_path, "refused", config)
    assert not T.output_root(config).exists()


def stage_dataset(work, pattern, points=STAGE_POINTS, **blocks):
    """A configuration with the stage file of ``points``, fovs.csv with position_id and a supplied reference image per
    FOV 9 to 12 (test_stitching_workflow.py's stage_dataset)."""
    stage = work / "stage" / "acquisition.maf"
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.write_text(maf_text(points))
    blocks = copy.deepcopy(blocks)
    blocks["stitching"]["positions"]["stage_file"] = str(stage)
    config = imported(work, voxel_size_xy=STAGE_SPACING[2], voxel_size_z=STAGE_SPACING[0], **blocks)
    T.write_inputs(config, rows(SAMPLE, [pattern.format(i=i) for i in range(9, 13)], list(range(9, 13))))
    for i in range(9, 13):
        reference(work, pattern.format(i=i), np.full(STAGE_SHAPE, i, np.uint16), STAGE_SPACING)
    return config


@pytest.mark.parametrize("pattern", ["tile_{i:02d}", "tile_{i}"])
def test_the_stitch_command_names_the_stage_file_s_fovs_by_fovs_csv(tmp_path, monkeypatch, pattern):
    """From test_wf3_stage_file_names_in_process: stitching.stitch receives the table of the stage file's order,
    each position named by fovs.csv."""
    import starfinder.stitching as stitching
    config = stage_dataset(tmp_path, pattern, stitching=dict(MERGED, positions=dict(STAGE_POSITIONS)))
    manifest, planned = T.plan(tmp_path, "stage", config)
    calls = []
    monkeypatch.setattr(stitching, "stitch", lambda inputs, table, **kw: calls.append((inputs, table, kw)) or "result")
    monkeypatch.setattr(stitching, "save_placement", lambda result, folder: None)
    T.call(manifest, f"stitch/{SAMPLE}")
    (inputs, table, kwargs), = calls
    names = [pattern.format(i=p) for p, *_ in STAGE_POINTS]
    assert table.fov_ids == names                                      # stage-file order
    assert table.positions["position_id"].tolist() == [9, 10, 12, 11]
    assert [item.fov_id for item in inputs] == names and kwargs["sample"] == SAMPLE
    assert set(names) == set(planned["membership"]["samples"][SAMPLE])
    assert table.record["fov_ids"] == {str(p): pattern.format(i=p) for p, *_ in STAGE_POINTS}


@pytest.mark.parametrize("pattern", ["tile_{i:02d}", "tile_{i}"])
def test_the_bigstitcher_commands_name_the_stage_file_s_fovs_by_fovs_csv(tmp_path, monkeypatch, pattern):
    """From test_wf3_stage_file_names_bigstitcher: the macro task's view map and the import task's table carry the
    names of fovs.csv."""
    import starfinder.stitching as stitching
    save = tmp_path / "bigstitcher"
    config = stage_dataset(tmp_path, pattern, stitching={"positions": dict(STAGE_POSITIONS),
                                                         "bigstitcher_save_dir": str(save)})
    manifest, planned = T.plan(tmp_path, "stage", config)
    assert T.run(manifest, f"bigstitcher_macro/{SAMPLE}") == 0
    views = pd.read_csv(save / SAMPLE / "view_map.csv", keep_default_na=False)
    expected = [pattern.format(i=i) for i in (9, 10, 11, 12)]          # column-major cells: 9, 10 | 11, 12
    assert views["fov_id"].tolist() == expected and views["kind"].tolist() == ["fov"] * 4
    assert expected == planned["membership"]["samples"][SAMPLE]
    (save / SAMPLE / "dataset.xml").write_text(links_xml())             # Fiji's stand-in: views 0 to 3 placed
    tables = []

    def stop(xml, *, table, **kwargs):
        tables.append(table)
        raise StopIteration

    monkeypatch.setattr(stitching, "import_bigstitcher", stop)
    with pytest.raises(StopIteration):
        T.call(manifest, f"bigstitcher_import/{SAMPLE}")
    assert tables[0].fov_ids == [pattern.format(i=p) for p, *_ in STAGE_POINTS]


def test_a_stage_file_that_does_not_hold_a_fov_fails_the_stitch_task(tmp_path):
    """From test_wf3_a_stage_file_that_does_not_match_the_sample_raises: a FOV whose PositionID the stage file does
    not hold fails the task (exit 1), naming it; no placement is left."""
    config = stage_dataset(tmp_path, "tile_{i}", points=STAGE_POINTS[:3],
                           stitching=dict(MERGED, positions=dict(STAGE_POSITIONS)))
    manifest, _ = T.plan(tmp_path, "stage", config)
    assert T.run(manifest, f"stitch/{SAMPLE}") == 1
    error = T.failure(manifest, f"stitch/{SAMPLE}")
    assert error["type"] == "ValueError" and "PositionID 11 is not in stage file" in error["message"]
    assert not (T.output_root(config) / "stitching" / SAMPLE / "placement.json").exists()


# --- the BigStitcher route --------------------------------------------------------------------------------------------

def test_the_bigstitcher_route_by_its_commands(tmp_path):
    """From test_wf3_the_bigstitcher_path_end_to_end: xml_links as a dataset; the macro task writes the tiles, the
    macro and the view map; Fiji (a stand-in here) writes dataset.xml; the check passes; the import task imports it
    and saves the placement, and nothing else (the legacy tile configuration writer is retired, D6)."""
    save = tmp_path / "bigstitcher"
    config = imported(tmp_path, stitching={"positions": GRID, "bigstitcher_save_dir": str(save)})
    T.write_inputs(config, rows(SAMPLE, LINK_IDS))
    for k, fov in enumerate(LINK_IDS):
        reference(tmp_path, fov, np.full(LINK_SHAPE, k + 1, np.uint16), LN_SPACING)
    manifest, _ = T.plan(tmp_path, "links", config)
    folder = save / SAMPLE
    macro, view_map = folder / "BigStitcher_macro.ijm", folder / "view_map.csv"
    assert T.run(manifest, f"bigstitcher_macro/{SAMPLE}") == 0
    assert f"dataset_save_path={folder} " in macro.read_text() and f"path={folder}/tiles/*.tif" in macro.read_text()
    assert sorted(p.name for p in (folder / "tiles").iterdir()) == [f"tile_{j}.tif" for j in range(4)]
    assert [int(tifffile.imread(folder / "tiles" / f"tile_{j}.tif").max()) for j in range(4)] == [1, 2, 3, 4]
    assert pd.read_csv(view_map, keep_default_na=False).equals(links_view_map())
    (folder / "dataset.xml").write_text(links_xml())
    assert T.run(manifest, f"bigstitcher_engine/{SAMPLE}") == 0
    assert T.run(manifest, f"bigstitcher_import/{SAMPLE}") == 0
    root = T.output_root(config)
    placement = root / "stitching" / SAMPLE
    saved = load_placement(placement)
    expected = import_bigstitcher(folder / "dataset.xml", table=links_table(), view_map=links_view_map(),
                                  grids={fov: grid(LINK_SHAPE, LN_SPACING) for fov in LINK_IDS},
                                  config=BigStitcherConfig(), sample=SAMPLE, macro=macro.read_text())
    pd.testing.assert_frame_equal(saved.fovs, expected.fovs, check_exact=True)
    pd.testing.assert_frame_equal(saved.links, expected.links, check_exact=True)
    assert saved.record["method"]["macro_sha256"] == expected.record["method"]["macro_sha256"]
    assert sorted(p.name for p in placement.iterdir()) == ["fovs.csv", "links.csv", "placement.json", "view_map.csv"]
    assert not (root / "output").exists()


def without_stitching_transform(text, view):
    """The xml_links text with view ``view``'s Stitching Transform removed."""
    start = text.index(f'<ViewRegistration timepoint="0" setup="{view}">')
    begin = text.index('      <ViewTransform type="affine">\n        <Name>Stitching Transform</Name>', start)
    end = text.index("</ViewTransform>\n", begin) + len("</ViewTransform>\n")
    return text[:begin] + text[end:]


@pytest.fixture
def engine_task(tmp_path):
    """The links dataset planned for the BigStitcher route with its macro task run: (manifest, dataset.xml)."""
    save = tmp_path / "bigstitcher"
    config = imported(tmp_path, stitching={"positions": GRID, "bigstitcher_save_dir": str(save)},
                      execution={"fiji": "/opt/Fiji.app/ImageJ-linux64"})
    T.write_inputs(config, rows(SAMPLE, LINK_IDS))
    for k, fov in enumerate(LINK_IDS):
        reference(tmp_path, fov, np.full(LINK_SHAPE, k + 1, np.uint16), LN_SPACING)
    manifest, _ = T.plan(tmp_path, "engine", config)
    assert T.run(manifest, f"bigstitcher_macro/{SAMPLE}") == 0
    return manifest, save / SAMPLE / "dataset.xml"


def test_the_check_accepts_the_complete_file_and_the_engine_task_holds_fijis_command(engine_task):
    """From test_wf3_engine_check_accepts_the_complete_file: the check after Fiji passes on a dataset.xml that places
    every view; the engine task's tool command is the legacy engine rule's (Fiji headless on the macro, the
    BigStitcher folder as Java's temporary folder), with the configured Fiji."""
    manifest, xml = engine_task
    xml.write_text(links_xml())
    assert T.run(manifest, f"bigstitcher_engine/{SAMPLE}") == 0
    task = T.task(manifest, "bigstitcher_engine", sample=SAMPLE)
    assert task["tool_command"] == ["{fiji}", f"-Djava.io.tmpdir={xml.parent}", "--ij2", "--headless", "--run",
                                    str(xml.parent / "BigStitcher_macro.ijm")]
    assert task["execution"]["tool"] == {"name": "fiji", "path": "/opt/Fiji.app/ImageJ-linux64"}


def test_the_check_fails_without_view_3_s_stitching_transform(engine_task, capsys):
    """From test_wf3_engine_check_fails_without_view_3_s_stitching_transform: the check fails (exit 1) naming the
    view, and removes dataset.xml. Fiji's exit status is not the check's to name: the engine runs the tool and the
    check in one job (D3), and the task fails whatever the tool's status."""
    manifest, xml = engine_task
    text = without_stitching_transform(links_xml(), 3)
    assert text.count("Stitching Transform") == 3
    xml.write_text(text)
    assert T.run(manifest, f"bigstitcher_engine/{SAMPLE}") == 1
    error = T.failure(manifest, f"bigstitcher_engine/{SAMPLE}")
    assert error["type"] == "RuntimeError"
    assert "no 'Stitching Transform' for views 3; the BigStitcher engine task failed" in error["message"]
    assert not xml.exists()


def test_the_check_fails_without_a_file(engine_task):
    """From test_wf3_engine_check_fails_without_a_file: no dataset.xml fails the check (exit 1), naming the path."""
    manifest, xml = engine_task
    assert T.run(manifest, f"bigstitcher_engine/{SAMPLE}") == 1
    error = T.failure(manifest, f"bigstitcher_engine/{SAMPLE}")
    assert error["message"] == f"the BigStitcher engine task wrote no dataset.xml at {xml}"


def test_fiji_is_read_only_by_the_engine_task(tmp_path):
    """From test_wf3_fiji_path_is_read_only_when_the_engine_rule_runs: the stitching block needs no Fiji; the plan
    gives the Fiji path (execution.fiji) to the engine task alone, as its tool, and no command reads it."""
    save = tmp_path / "bigstitcher"
    config = imported(tmp_path, stitching={"positions": GRID, "bigstitcher_save_dir": str(save)})
    T.write_inputs(config, rows(SAMPLE, LINK_IDS))
    manifest, planned = T.plan(tmp_path, "nofiji", config)
    tools = {t["kind"]: t["execution"].get("tool") for t in planned["tasks"]}
    assert tools == {"bigstitcher_macro": None, "bigstitcher_engine": {"name": "fiji", "path": None},
                     "bigstitcher_import": None}
    _, configured = T.plan(tmp_path, "fiji", dict(config, execution={"fiji": "/opt/Fiji"}))
    assert {t["kind"]: t["execution"].get("tool") for t in configured["tasks"]}["bigstitcher_engine"] == {
        "name": "fiji", "path": "/opt/Fiji"}
    assert [t["kind"] for t in configured["tasks"] if "tool_command" in t] == ["bigstitcher_engine"]
    # No command reads it: the name fiji is in no code token of the commands (only in their docstrings).
    import io
    import tokenize
    from pathlib import Path
    commands = (Path(__file__).resolve().parents[1] / "starfinder" / "workflow" / "_commands.py").read_text()
    code = [t.string.lower() for t in tokenize.generate_tokens(io.StringIO(commands).readline)
            if t.type == tokenize.NAME]
    assert "fiji" not in code


@pytest.mark.slow
def test_the_engine_task_fails_naming_execution_fiji_in_a_real_job(tmp_path):
    """From test_wf3_fiji_path_is_read_only_when_the_engine_rule_runs (pytest.raises(ValueError, match="fiji_path")
    when the engine rule ran without the path): a real Snakemake job of the engine task without execution.fiji fails
    at once naming execution.fiji; no tool runs, the shell's || echo does not hide it and bigstitcher-check does not
    run (the task has no record), while the plan and the macro task succeed. With a stand-in Fiji that fails, the job
    removes a stale dataset.xml (one the check would pass) before the tool runs, and bigstitcher-check reports the
    missing file; with one that writes a complete dataset.xml the task succeeds (the job runs the tool: before W-755
    the rule's shell, evaluated again when the job ran, took the last group's and left the tool out)."""
    import yaml
    from starfinder import _layout
    from starfinder.workflow._commands import _check_dataset_xml, _read_view_map
    from .test_workflow_snakemake_dryrun import _snakemake, snakemake
    _snakemake()
    save = tmp_path / "bigstitcher"
    for k, fov in enumerate(LINK_IDS):
        reference(tmp_path, fov, np.full(LINK_SHAPE, k + 1, np.uint16), LN_SPACING)
    config = imported(tmp_path, stitching={"positions": GRID, "bigstitcher_save_dir": str(save)})
    T.write_inputs(config, rows(SAMPLE, LINK_IDS))
    out, engine = T.output_root(config), f"bigstitcher_engine/{SAMPLE}"
    path = tmp_path / "configs" / "engine.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    done = snakemake(tmp_path, path, "--", str(_layout.task_record(out, engine)))
    message = done.stdout + done.stderr
    assert done.returncode != 0
    assert f"task {engine} runs fiji, but execution.fiji is not set: set execution.fiji" in message, message
    assert "the tool exited with status" not in message
    assert _layout.task_record(out, f"bigstitcher_macro/{SAMPLE}").is_file()
    assert not _layout.task_record(out, engine).exists() and not _layout.failed_record(out, engine).exists()
    assert T.task(_layout.manifest_file(out), "bigstitcher_engine")["execution"]["tool"] == {"name": "fiji",
                                                                                            "path": None}
    # A stand-in Fiji that records whether dataset.xml was there when it ran, and fails.
    stale = save / SAMPLE / "dataset.xml"
    stale.write_text(links_xml())
    _check_dataset_xml(stale, _read_view_map(save / SAMPLE / "view_map.csv"))      # the stale file passes the check
    fiji = tmp_path / "fiji.sh"
    fiji.write_text(f"#!/bin/sh\nif [ -e {stale} ]; then echo present; else echo absent; fi > {tmp_path / 'seen'}\n"
                    "exit 3\n")
    fiji.chmod(0o755)
    path.write_text(yaml.safe_dump(dict(config, execution={"fiji": str(fiji)}), sort_keys=False))
    done = snakemake(tmp_path, path, "--", str(_layout.task_record(out, engine)))
    assert done.returncode != 0
    assert (tmp_path / "seen").read_text() == "absent\n"
    assert "the tool exited with status 3" in done.stdout + done.stderr
    message = T.failure(_layout.manifest_file(out), engine)["message"]
    assert message.startswith("the BigStitcher engine task wrote no dataset.xml at ") and message.endswith(
        f"/bigstitcher/{SAMPLE}/dataset.xml"), message
    assert not stale.exists()
    # A stand-in Fiji that writes a complete dataset.xml: the job runs it, and the check passes.
    (tmp_path / "complete.xml").write_text(links_xml())
    fiji.write_text(f"#!/bin/sh\ncp {tmp_path / 'complete.xml'} {stale}\n")
    done = snakemake(tmp_path, path, "--", str(_layout.task_record(out, engine)))
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(_layout.task_record(out, engine).read_text())["status"] == "succeeded"
    assert stale.read_text() == links_xml()


# --- the sample's FOVs under selected_fovs --------------------------------------------------------------------------

# Two samples whose rows interleave (test_fov_membership.py's ROWS).
MEMBERS = [("s2", "F10", 7), ("s1", "F2", 2), ("s2", "F3", 3), ("s1", "F1", 1)]


def member_rows(positions=True):
    header = "sample,fov_id,position_id\n" if positions else "sample,fov_id\n"
    return header + "".join(f"{s},{f},{p}\n" if positions else f"{s},{f}\n" for s, f, p in MEMBERS)


def test_the_per_sample_commands_read_the_selected_fovs_only(tmp_path, monkeypatch):
    """From test_a_per_sample_rule_reads_the_selected_fovs_only: the stitch and report commands read the sample's
    selected FOVs (fovs.csv restricted by selected_fovs); a regular grid reads no PositionID."""
    calls = []
    monkeypatch.setattr(Dataset, "stitch", lambda self, *args, **kwargs: calls.append(("stitch", args, kwargs)))
    monkeypatch.setattr(Dataset, "report", lambda self, *args, **kwargs: calls.append(("report", args, kwargs)))
    config = imported(tmp_path, selected_fovs=["F1", "F10"], voxel_size_xy=1.0, voxel_size_z=1.0,
                      stitching=dict(MERGED, review=None, fallback=None), report={})
    T.write_inputs(config, member_rows())
    reference(tmp_path, "F1", np.zeros((2, 16, 24), np.uint16), STAGE_SPACING)  # the selected grid's geometry
    manifest, planned = T.plan(tmp_path, "selected", config)
    assert planned["membership"]["samples"] == {"s2": ["F10"], "s1": ["F1"]}
    T.call(manifest, "stitch/s1")
    T.call(manifest, "report/s1")
    (_, stitch_args, stitch_kwargs), (_, report_args, _) = calls
    assert stitch_args == ("s1", ["F1"]) and report_args == ("s1", ["F1"])
    assert stitch_kwargs["position_ids"] is None       # a regular grid reads no PositionID
    assert stitch_kwargs["table"].fov_ids == ["F1"]


def test_the_stitch_task_needs_only_fovs_csv_and_a_stage_file(tmp_path, monkeypatch):
    """From test_the_python_stitching_parse_needs_only_fovs_csv_and_a_stage_file: with a whole-acquisition stage file
    the plan needs fovs.csv and the stage file (no sample annotation, no documents/ folder); the stitch task declares
    the stage file and the sample's reference images, and places the sample's positions, selected from the file;
    without the position_id column the stage file names no FOV of the run, and the plan refuses it."""
    import starfinder.stitching as stitching
    stage = tmp_path / "acquisition" / "positions.maf"
    stage.parent.mkdir()
    # Sample s1 (PositionIDs 1, 2) and sample s2 (7, 3), in one acquisition file, in the order 7, 1, 3, 2.
    stage.write_text(maf_text(((7, "0.07", "0.04"), (1, "0.076", "0.048"), (3, "0.07", "0.040261"),
                               (2, "0.076", "0.048261"))))
    positions = {"source": "stage_file", "stage_axes": STAGE_AXES, "stage_file": str(stage)}
    config = imported(tmp_path, voxel_size_xy=12.0, voxel_size_z=1.0,
                      stitching={"method": "multiview_stitcher", "positions": positions})
    T.write_inputs(config, member_rows())
    for fov in ("F2", "F1"):
        reference(tmp_path, fov, np.zeros((2, 16, 24), np.uint16), STAGE_SPACING)
    manifest, planned = T.plan(tmp_path, "stage", config)
    task = T.task(manifest, "stitch", sample="s1")
    assert str(stage) in task["inputs"]
    assert {i for i in task["inputs"] if i.startswith(str(tmp_path / "refs"))} == {
        str(tmp_path / "refs" / "F2.tif"), str(tmp_path / "refs" / "F1.tif")}
    assert planned["inputs"]["files"]["fovs.csv"]["path"] == "fovs.csv"
    tables = []
    monkeypatch.setattr(stitching, "stitch", lambda inputs, table, **kw: tables.append(table) or "result")
    monkeypatch.setattr(stitching, "save_placement", lambda result, folder: None)
    T.call(manifest, "stitch/s1")
    table, = tables
    assert table.fov_ids == ["F1", "F2"] and table.positions["position_id"].tolist() == [1, 2]   # file order
    assert table.positions["nominal_y"].tolist() == [0.0, 261.0] and table.positions["nominal_x"].tolist() == [0, 0]
    assert not (T.output_root(config) / "documents").exists()
    T.write_inputs(config, member_rows(positions=False))
    with pytest.raises(PlanRefused, match="position_id column of fovs.csv"):
        T.plan(tmp_path, "stage-without-ids", config)


def test_selected_fovs_select_from_a_supplied_position_table(tmp_path, monkeypatch):
    """From the test of the same name: the stitch command under selected_fovs [F2] with a supplied table that holds F1
    and F2 places F2 as a table that holds only F2 does."""
    import starfinder.stitching as stitching
    monkeypatch.setattr(stitching, "stitch", lambda inputs, table, *, sample, **kw: placement_from_table(
        table, {item.fov_id: item.grid for item in inputs}, sample=sample))
    saved = []
    monkeypatch.setattr(stitching, "save_placement", lambda result, folder: saved.append(result))
    for fov in ("F1", "F2"):
        reference(tmp_path, fov, np.zeros((2, 16, 24), np.uint16), STAGE_SPACING)
    table = [{"fov_id": "F1", "nominal_z": 0.0, "nominal_y": 0.0, "nominal_x": 0.0},
             {"fov_id": "F2", "nominal_z": 0.0, "nominal_y": 259.2, "nominal_x": 0.0}]
    for name, entries in (("both.csv", table), ("f2.csv", table[1:])):
        pd.DataFrame(entries).to_csv(tmp_path / name, index=False)
        config = imported(tmp_path, output_id=name[:-4], selected_fovs=["F2"],
                          stitching=dict(MERGED, positions={"source": "table", "table": str(tmp_path / name)}))
        T.write_inputs(config, "sample,fov_id\ns1,F1\ns1,F2\n")
        manifest, _ = T.plan(tmp_path, name, config)
        T.call(manifest, "stitch/s1")
    both, alone = saved
    assert both.fovs.fov_id.tolist() == ["F2"]
    pd.testing.assert_frame_equal(both.fovs, alone.fovs, check_exact=True)


@pytest.mark.parametrize("method", ["multiview_stitcher", "bigstitcher"])
def test_a_selection_keeps_each_fov_s_regular_grid_cell(tmp_path, monkeypatch, method):
    """From the test of the same name (test_fov_membership.py): a regular grid is laid out over the sample's FOVs of
    fovs.csv: under selected_fovs [F2, F4] the placement table holds F2 and F4 with the cells and nominal positions
    they have without the selection (the stitch command, and the BigStitcher import command)."""
    import starfinder.stitching as stitching
    for fov in ("F1", "F2", "F3", "F4"):
        reference(tmp_path, fov, np.zeros((2, 16, 24), np.uint16), STAGE_SPACING)
    tables = []

    def stand_in(inputs, table, *, sample, **kw):
        tables.append(table)
        return placement_from_table(table, {item.fov_id: item.grid for item in inputs}, sample=sample)

    def stop(xml, *, table, **kwargs):
        tables.append(table)
        raise StopIteration

    monkeypatch.setattr(stitching, "stitch", stand_in)
    monkeypatch.setattr(stitching, "save_placement", lambda result, folder: None)
    monkeypatch.setattr(stitching, "import_bigstitcher", stop)
    for selected in (None, ["F2", "F4"]):
        name = "every" if selected is None else "chosen"
        save = tmp_path / name / "bigstitcher"
        block = (dict(MERGED, positions=dict(GRID, regular_grid=dict(GRID["regular_grid"])))
                 if method == "multiview_stitcher" else {"positions": GRID, "bigstitcher_save_dir": str(save)})
        config = imported(tmp_path, output_id=name, selected_fovs=selected, stitching=block)
        T.write_inputs(config, "sample,fov_id\ns1,F1\ns1,F2\ns1,F3\ns1,F4\n")
        manifest, _ = T.plan(tmp_path, name, config)
        if method == "bigstitcher":
            (save / "s1").mkdir(parents=True)
            (save / "s1" / "dataset.xml").write_text(links_xml())       # views 0 to 3 placed: the check passes
            links_view_map().to_csv(save / "s1" / "view_map.csv", index=False)
            (save / "s1" / "BigStitcher_macro.ijm").write_text("// macro\n")
            with pytest.raises(StopIteration):
                T.call(manifest, "bigstitcher_import/s1")
        else:
            T.call(manifest, "stitch/s1")
    every, chosen = (table.positions.set_index("fov_id") for table in tables)
    assert list(every.index) == ["F1", "F2", "F3", "F4"] and list(chosen.index) == ["F2", "F4"]
    pd.testing.assert_frame_equal(chosen, every.loc[["F2", "F4"]], check_exact=True)
    assert chosen.loc["F4", "grid_column"] == 1 and chosen.loc["F4", "grid_row"] == 1     # column-major 2 × 2


# --- files without stored metadata (W-397) ------------------------------------------------------------------------

def layout_config(dataset, work, **blocks):
    """A configuration whose output root is the dataset's (<root>/data/output), whose input root holds fovs.csv, and
    whose reference_image is the dataset's pattern (the default layout's, or the moved layout's
    images/flamingo/DAPI/{fov_id}_reference.tif), under a processing block, whose fov tasks write the files."""
    from . import workflow_examples as X
    root = dataset.output_root.parent.parent
    config = T.dataset_keys(work, root_output_path=str(root), output_id="output",
                            reference_image=dataset.reference_image, processing=copy.deepcopy(X.PROCESSING), **blocks)
    if dataset.voxel_size_xy is not None:
        config.update(voxel_size_xy=dataset.voxel_size_xy, voxel_size_z=dataset.voxel_size_z)
    return config


def stitch_layout(dataset, layouts, work):
    """The stitch command with a multiview_stitcher block on the scene's positions table (the fov tasks, whose
    reference images the layout holds, say succeeded); the saved placement."""
    positions = {"source": "table", "table": str(layouts["saved"] / "positions.csv")}
    block = {"method": "multiview_stitcher", "input": "reference_merged", "positions": positions,
             "config": {"binning_zyx": [1, 2, 2], "quality_threshold": -1.0}}
    config = layout_config(dataset, work, stitching=block)
    T.write_inputs(config, rows("scene", fov_ids(layouts)))
    manifest, _ = T.plan(work, "layout", config)
    task = T.task(manifest, "stitch", sample="scene")
    assert {i for i in task["inputs"] if i.endswith(".tif")} == {
        dataset.reference_image.replace("{fov_id}", fov) for fov in fov_ids(layouts)}
    T.succeed(manifest, "fov")
    status = T.run(manifest, "stitch/scene")
    if status:
        raise ValueError(T.failure(manifest, "stitch/scene")["message"])
    placement = load_placement(dataset.output_root / "stitching" / "scene")
    shutil.rmtree(dataset.output_root / "stitching")
    shutil.rmtree(dataset.output_root / "workflow")
    return placement


@needs_library
@pytest.mark.slow
@pytest.mark.parametrize("layout", ["default", "moved"])
def test_the_stitch_command_on_files_without_metadata(scene_layouts, tmp_path, layout):  # noqa: F811
    """From test_the_workflow_stitching_adapter_on_files_without_metadata: the stitch command with a
    multiview_stitcher block, reading the reference images at reference_image (the default layout's, or the moved
    layout's images/flamingo/DAPI/{fov_id}_reference.tif, the base's pattern): plain pages with the configured voxel
    sizes give the stored copy's placement; without them the task fails naming them."""
    stored, plain = scene_layouts[("stored", layout)], scene_layouts[("plain", layout)]
    via = {form: stitch_layout(dataset, scene_layouts, tmp_path / form)
           for form, dataset in (("stored", stored), ("plain", plain))}
    assert_same_placement(via["stored"], via["plain"])
    assert set(via["plain"].fovs.status) == {"image_supported"} and len(via["plain"].links) > 0
    with pytest.raises(ValueError, match="lacks voxel_size_xy and voxel_size_z"):
        stitch_layout(replace(plain, voxel_size_xy=None, voxel_size_z=None), scene_layouts, tmp_path / "unconfigured")


def golden_route(work, form, setups, *, voxel_sizes=True, fallback=None, reference_image=None):
    """The golden fixture as a configuration of the BigStitcher route: one-plane reference images Position1 to
    Position8 (stored metadata or plain pages; at reference_image's pattern under a processing block when given, else
    supplied), its stage file and fovs.csv (position_id 1 to 8), and the BigStitcher folder with the hand-built
    dataset.xml, view map and macro; (manifest, configuration, folder)."""
    from . import workflow_examples as X
    xml, _, _, table, view_map, _ = setups
    save = work / "bigstitcher"
    stage = work / "stage" / f"{SAMPLE}.maf"
    stage.parent.mkdir(parents=True)
    stage.write_text(stage_file())
    positions = {"source": "stage_file", "stage_axes": STAGE_AXES, "snap_to_grid": {"overlap": 0.1},
                 "stage_file": str(stage)}
    block = {"positions": positions, "bigstitcher_save_dir": str(save), "fallback": fallback}
    if reference_image is None:
        config = imported(work, stitching=block)
        path = lambda fov: work / "refs" / f"{fov}.tif"  # noqa: E731
    else:
        config = T.dataset_keys(work, reference_image=reference_image, processing=copy.deepcopy(X.PROCESSING),
                                stitching=block)
        path = lambda fov: T.output_root(config) / reference_image.replace("{fov_id}", fov)  # noqa: E731
    if form == "plain" and voxel_sizes:
        config.update(voxel_size_xy=GOLDEN_SPACING[2], voxel_size_z=GOLDEN_SPACING[0])
    T.write_inputs(config, rows(SAMPLE, list(table.fov_ids), list(range(1, 9))))
    for k, fov in enumerate(table.fov_ids):
        image = (np.arange(np.prod(GOLDEN_SHAPE)).reshape(GOLDEN_SHAPE) % 251 + 3 * k).astype(np.uint16)
        if form == "stored":
            save_volume(image, path(fov), metadata=grid(GOLDEN_SHAPE, GOLDEN_SPACING).metadata)
        else:
            plain_pages(path(fov), image)
    folder = save / SAMPLE
    folder.mkdir(parents=True)
    shutil.copy(xml, folder / "dataset.xml")
    view_map.to_csv(folder / "view_map.csv", index=False)
    (folder / "BigStitcher_macro.ijm").write_text("// macro\n")
    manifest, _ = T.plan(work, f"golden-{fallback}", config)
    if reference_image is not None:
        T.succeed(manifest, "fov")
    return manifest, config, folder


def import_route(manifest, config):
    """The import command (its tasks before it say succeeded); the saved placement, or ValueError with the failure."""
    T.succeed(manifest, "bigstitcher_macro", "bigstitcher_engine")
    if T.run(manifest, f"bigstitcher_import/{SAMPLE}"):
        raise ValueError(T.failure(manifest, f"bigstitcher_import/{SAMPLE}")["message"])
    return load_placement(T.output_root(config) / "stitching" / SAMPLE)


def test_the_bigstitcher_commands_on_the_golden_fixture_without_metadata(tmp_path, golden_stitching_setups):  # noqa: F811
    """From test_the_bigstitcher_adapters_on_the_golden_fixture_without_metadata: the import command (with each
    fallback) and the macro command, which reads the images, on plain pages with the voxel sizes give the stored
    copy's placement, tiles and view map; without the voxel sizes they fail naming them. The golden dataset.xml holds
    no pairwise result, so every FOV is failed without a fallback and placed at its nominal position with fallback
    nominal. (The legacy tile configuration the adapter also wrote is retired, D6.)"""
    results = {}
    for form in ("stored", "plain"):
        results[form] = []
        for fallback in (None, "nominal"):
            manifest, config, _ = golden_route(tmp_path / form / str(fallback), form, golden_stitching_setups,
                                               fallback=fallback)
            results[form].append(import_route(manifest, config))
        manifest, config, folder = golden_route(tmp_path / form / "macro", form, golden_stitching_setups)
        assert T.run(manifest, f"bigstitcher_macro/{SAMPLE}") == 0
        results[form].append((folder, tmp_path / form / "macro"))
    for stored, plain in zip(results["stored"][:2], results["plain"][:2]):
        assert_same_placement(stored, plain)
        assert len(plain.fovs) == 8
    assert set(results["plain"][0].fovs.status) == {"failed"}
    assert results["plain"][1].fovs.origin_x.notna().all()
    (stored_macro, stored_work), (plain_macro, plain_work) = results["stored"][2], results["plain"][2]
    assert (plain_macro / "view_map.csv").read_text() == (stored_macro / "view_map.csv").read_text()
    assert (plain_macro / "BigStitcher_macro.ijm").read_text().replace(str(plain_work), "") == (
        stored_macro / "BigStitcher_macro.ijm").read_text().replace(str(stored_work), "")
    for tile in sorted((stored_macro / "tiles").iterdir()):
        assert np.array_equal(tifffile.imread(tile), tifffile.imread(plain_macro / "tiles" / tile.name)), tile.name
    manifest, config, _ = golden_route(tmp_path / "unconfigured", "plain", golden_stitching_setups, voxel_sizes=False)
    with pytest.raises(ValueError, match="lacks voxel_size_xy and voxel_size_z"):
        import_route(manifest, config)
    assert T.run(manifest, f"bigstitcher_macro/{SAMPLE}") == 1
    assert "lacks voxel_size_xy and voxel_size_z" in T.failure(manifest, f"bigstitcher_macro/{SAMPLE}")["message"]


def links_route(work, form, *, voxel_size_xy=True):
    """xml_links as a configuration of the BigStitcher route: four 4×16×24 supplied reference images L0 to L3 on a
    2 × 2 grid at LN's spacing, and the BigStitcher folder with the hand-built dataset.xml, view map and macro."""
    save = work / "bigstitcher"
    config = imported(work, stitching={"positions": GRID, "bigstitcher_save_dir": str(save)})
    if form == "plain":
        config.update(voxel_size_z=LN_SPACING[0], **({"voxel_size_xy": LN_SPACING[2]} if voxel_size_xy else {}))
    T.write_inputs(config, rows(SAMPLE, LINK_IDS))
    for k, fov in enumerate(LINK_IDS):
        image = np.full(LINK_SHAPE, k + 1, np.uint16)
        if form == "stored":
            reference(work, fov, image, LN_SPACING)
        else:
            plain_pages(work / "refs" / f"{fov}.tif", image)
    folder = save / SAMPLE
    folder.mkdir(parents=True)
    (folder / "dataset.xml").write_text(links_xml())
    links_view_map().to_csv(folder / "view_map.csv", index=False)
    (folder / "BigStitcher_macro.ijm").write_text("// macro\n")
    manifest, _ = T.plan(work, "links", config)
    return manifest, config


def test_the_import_command_on_the_2x2_links_fixture_without_metadata(tmp_path):
    """From test_the_import_adapter_on_the_2x2_links_fixture_without_metadata: the hand-built xml_links dataset.xml
    (pairwise results for its six links): plain pages with the voxel sizes give the stored copy's placement and links;
    without voxel_size_xy the import fails naming it."""
    stored, plain = (import_route(*links_route(tmp_path / form, form)) for form in ("stored", "plain"))
    assert_same_placement(stored, plain)
    assert set(plain.fovs.status) == {"image_supported"} and len(plain.links) == 6
    with pytest.raises(ValueError, match="lacks voxel_size_xy: set voxel_size_xy"):
        import_route(*links_route(tmp_path / "unconfigured", "plain", voxel_size_xy=False))


def test_the_import_command_reads_the_configured_reference_image(tmp_path, golden_stitching_setups):  # noqa: F811
    """From test_the_import_adapter_reads_the_configured_reference_image: with reference_image set
    (images/flamingo/DAPI/{fov_id}_reference.tif, the base's pattern), the import task declares and reads the moved
    files and gives the default layout's placement."""
    manifest, config, _ = golden_route(tmp_path / "default", "plain", golden_stitching_setups,
                                       reference_image="fovs/{fov_id}/reference.tif")
    expected = import_route(manifest, config)
    manifest, config, _ = golden_route(tmp_path / "moved", "plain", golden_stitching_setups, reference_image=MOVED)
    task = T.task(manifest, "bigstitcher_import", sample=SAMPLE)
    assert "images/flamingo/DAPI/Position1_reference.tif" in task["inputs"]
    assert not [i for i in task["inputs"] if i.endswith("/reference.tif")]
    assert_same_placement(expected, import_route(manifest, config))
    assert json.loads((T.output_root(config) / "stitching" / SAMPLE / "placement.json").read_text())
