"""WF1 to WF3 of docs/stitching-assembly-validation.md: the §2.10 workflow blocks and rules (W-372).

The Python-only ``stitching``, ``assembly`` and ``export`` blocks of
``workflow/schemas/config.schema.yaml``, the adapter functions of
``starfinder.dataset.workflow`` that the rules of ``stitching-py.smk``, ``export.smk``
and ``reads-assignment.smk`` call, and the legacy translation. Fixtures, all written in
the test: ``wf_configs`` (the repository's example configurations, and copies with the
blocks); ``assembly_golden`` written as a dataset (``fusion_fixtures.export_folder``);
``xml_links``, ``golden_stitching_setups`` and ``hand_records_setups`` of
``test_stitching_bigstitcher.py``. The adapters run through a stub ``snakemake`` object, as
in ``test_assignment_workflow.py``. Fiji never runs: the engine rule's command is replaced
by a stand-in that writes (or does not write) ``dataset.xml``. dask runs with the
synchronous scheduler.
"""
import copy
import importlib.util
import inspect
import json
import re
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml

import starfinder.dataset.workflow as workflow
from starfinder.assembly import AssemblyConfig
from starfinder.assignment._persist import read_assignment
from starfinder.dataset import Dataset
from starfinder.image import ImageMetadata
from starfinder.io import save_volume
from starfinder.sample_output import (ExportConfig, ExportIncompleteWarning, FusionConfig, LabelConfig,
                                      SpatialDataExportConfig)
from starfinder.segmentation import ReferenceGrid, reference_grid_from_file
from starfinder.segmentation._labels import same_grid
from starfinder.stitching import (STITCHING_METHODS, BigStitcherConfig, MultiviewStitcherConfig, ReviewPolicy,
                                  StageAxisMapping, import_bigstitcher, load_placement)
from starfinder.stitching._dataset import POSITION_KEYS

from .assembly_fixtures import SAMPLE
from .fusion_fixtures import export_folder, golden_codebook
from .stitching_fixtures import LN_SPACING, STAGE_SPACING, grid, maf_text
from .test_assignment_golden import SHAPE_ZYX, label_fixture
from .test_assignment_workflow import KEEP, LEGACY_OBS, LEGACY_READS, NEW_OBS, NEW_READS, run_rule
from .test_stitching_bigstitcher import (HAND_RECORDS, LINK_IDS, LINK_SHAPE, golden_stitching_setups,  # noqa: F401
                                         hand_records_setups, links_table, links_view_map, links_xml)
from .test_workflow_stitching_golden import TILE_COL, TILE_CONFIG, TILE_ROW, legacy_tile_config

pytestmark = [pytest.mark.workflow, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = yaml.safe_load((ROOT / "workflow/schemas/config.schema.yaml").read_text())
DEFS = SCHEMA["$defs"]
WF_CONFIGS = ("workflow-full.yaml", "workflow-minimal.yaml", "workflow-recipe-2.yaml")
needs_anndata = pytest.mark.skipif(importlib.util.find_spec("anndata") is None,
                                   reason="the anndata extra (starfinder[anndata]) is not installed")
needs_library = pytest.mark.skipif(importlib.util.find_spec("multiview_stitcher") is None,
                                   reason="the multiview-stitcher extra (starfinder[multiview-stitcher]) is not "
                                          "installed")

# The blocks of the contracts' "Workflow configuration" sections, completed where a key has no default.
STITCHING = yaml.safe_load("""
method: bigstitcher
input: reference_merged
positions:
  source: stage_file
  stage_axes: {x: "+x", y: "+y"}
  snap_to_grid: {overlap: 0.1}
  regular_grid: {columns: 7, rows: 10, order: column_major_down_right, overlap: 0.1}
  tile_shape_yx: null
  spacing_yx: null
  table: null
config: {}
review: {max_correction_fraction: 0.5, max_residual_um: 1.0}
fallback: null
bigstitcher_save_dir: /local/bigstitcher
""")
ASSEMBLY = yaml.safe_load("""
assignment: default
ownership: nearest_center
centroid: territory
tie: table_order
conflicts: true
table_format: csv
""")
EXPORT = yaml.safe_load("""
assemble: true
fuse: true
fuse_labels: false
export_spatialdata: false
images: [reference_stain]
dimensionality: projection
projection: max
scale_yx: 0.5
ome_zarr: false
ome_zarr_version: "0.5"
pyramid_levels: 0
labels_scale_yx: 0.25
spatialdata_zarr_format: 3
""")
MULTIVIEW = {"method": "multiview_stitcher", "input": "reference_stain",
             "positions": {"source": "regular_grid",
                           "regular_grid": {"columns": 2, "rows": 2, "order": "column_major_down_right",
                                            "overlap": 0.25}},
             "config": {"binning_zyx": [1, 2, 2], "quality_threshold": -1.0},
             "review": {"max_residual_um": 2.0}, "fallback": "nominal"}


class Wildcards(dict):
    """Snakemake wildcards: read both as attributes and by key."""
    __getattr__ = dict.__getitem__


def wf_config(name):
    return yaml.safe_load((ROOT / "docs/examples" / name).read_text())


def with_blocks(config, *, stitching=STITCHING, assembly=ASSEMBLY, export=EXPORT):
    out = copy.deepcopy(config)
    out["backend"] = "python"
    for key, block in (("stitching", stitching), ("assembly", assembly), ("export", export)):
        if block is not None:
            out[key] = copy.deepcopy(block)
    return out


def stub(config, *, inputs, outputs, **wildcards):
    return SimpleNamespace(config=config, input=[str(p) for p in inputs], output=[str(p) for p in outputs],
                           wildcards=Wildcards(wildcards))


def dataset_config(tmp_path, *, pattern="Position{i:03d}", **blocks):
    """A Python configuration whose output root is tmp_path/data/output (dataset_id data, output_id output)."""
    return {"backend": "python", "dataset_id": "data", "sample_id": "sample", "output_id": "output",
            "root_input_path": str(tmp_path / "input"), "root_output_path": str(tmp_path), "fov_id_pattern": pattern,
            "n_rounds": 3, "ref_round": "round1", "rotate_angle": 0, "img_col": 64, "img_row": 48,
            "voxel_size_xy": 1.0, "voxel_size_z": 1.0, "rules": {}, **blocks}


def annotation(tmp_path, start, end, sample=SAMPLE):
    path = tmp_path / "data" / "output" / "documents" / "sample-annotation.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"sample_id,fov_start,fov_end\n{sample},{start},{end}\n")
    return path


# --- WF1: the schema equals the configs ----------------------------------------------------------------------

def test_wf1_stitching_methods_equal_the_registry_plus_bigstitcher():
    declared = DEFS["stitching_block"]["properties"]["method"]["enum"]
    assert len(declared) == len(set(declared))
    assert set(declared) == {spec.name for spec in STITCHING_METHODS.values()} | {"bigstitcher"}
    assert set(workflow._stitching_methods()) == set(declared)


def test_wf1_stitching_keys_equal_the_adapter_and_the_package():
    block = DEFS["stitching_block"]["properties"]
    assert set(block) == set(workflow._STITCHING_KEYS)
    assert set(DEFS["stitching_positions"]["properties"]) == set(POSITION_KEYS)
    assert set(block["review"]["properties"]) == {f.name for f in fields(ReviewPolicy) if f.init}
    stitch = set(inspect.signature(Dataset.stitch).parameters)
    assert {"input", "review", "fallback", "config", "table"} <= stitch


def test_wf1_assembly_keys_equal_the_config_fields_the_assignment_and_the_table_format():
    declared = set(DEFS["assembly_block"]["properties"])
    assert declared == {f.name for f in fields(AssemblyConfig) if f.init} | {"assignment", "table_format"}
    assert "assignment" in inspect.signature(Dataset.export).parameters
    assert "table_format" in {f.name for f in fields(ExportConfig)}


def test_wf1_export_keys_equal_the_export_fusion_label_and_spatialdata_fields():
    declared = set(DEFS["export_block"]["properties"])
    nested = {"assembly", "table_format", "fusion", "labels", "spatialdata"}
    expected = ({f.name for f in fields(ExportConfig) if f.name not in nested}
                | {f.name for f in fields(FusionConfig) if f.init}
                | {f"labels_{f.name}" for f in fields(LabelConfig) if f.init}
                | {f"spatialdata_{f.name}" for f in fields(SpatialDataExportConfig) if f.init})
    assert declared == expected == set(workflow._export_keys())
    assert {"labels_scale_yx", "spatialdata_zarr_format"} <= declared


@pytest.mark.parametrize("name", WF_CONFIGS)
def test_wf1_every_configuration_validates(name):
    """wf_configs: each example as it is, and its copy with the three blocks (and with the multiview block)."""
    config = wf_config(name)
    jsonschema.validate(config, SCHEMA)
    jsonschema.validate(with_blocks(config), SCHEMA)
    jsonschema.validate(with_blocks(config, stitching=MULTIVIEW, export=None), SCHEMA)
    jsonschema.validate(with_blocks(config, assembly=None, export=None), SCHEMA)


@pytest.mark.parametrize("change", [
    {"backend": "matlab"},                                   # Python-only
    {"stitching": dict(STITCHING, method="grid_stitcher")},  # an unregistered method
    {"stitching": dict(STITCHING, tile_filter=True)},        # an unknown key
    {"assembly": dict(ASSEMBLY, ownership="voronoi")},
    {"export": dict(EXPORT, labels_scale=0.25)},             # the block's names, not LabelConfig's
    {"stitching": None},                                     # assembly without stitching
    {"assembly": None},                                      # export without assembly
])
def test_wf1_invalid_blocks_fail_the_schema(change):
    config = with_blocks(wf_config("workflow-full.yaml"))
    for key, value in change.items():
        if value is None:
            config.pop(key)
        else:
            config[key] = value
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(config, SCHEMA)


def test_wf1_the_blocks_of_the_contracts_translate():
    config = with_blocks(wf_config("workflow-full.yaml"))
    block = workflow._stitching_block(config)
    assert (block.method, block.config, block.input, block.fallback) == ("bigstitcher", BigStitcherConfig(),
                                                                        "reference_merged", None)
    assert block.review == ReviewPolicy(0.5, 1.0) and block.bigstitcher_save_dir == Path("/local/bigstitcher")
    assert block.positions["stage_axes"] == {"x": "+x", "y": "+y"}
    export, assembly = workflow._export_config(config)
    assert export == ExportConfig()
    assert (assembly.config, assembly.assignment, assembly.table_format) == (AssemblyConfig(), "default", "csv")
    multiview = workflow._stitching_block(with_blocks(config, stitching=MULTIVIEW))
    assert multiview.config == MultiviewStitcherConfig(binning_zyx=(1, 2, 2), quality_threshold=-1.0)
    changed = dict(EXPORT, dimensionality="stack", labels_scale_yx=0.5, spatialdata_zarr_format=2, fuse_labels=True,
                   images=["reference_stain", "reference_merged"])
    export, _ = workflow._export_config(with_blocks(config, assembly=dict(ASSEMBLY, table_format="parquet",
                                                                          conflicts=False), export=changed))
    assert export == ExportConfig(fuse_labels=True, assembly=AssemblyConfig(conflicts=False), table_format="parquet",
                                  fusion=FusionConfig(images=("reference_stain", "reference_merged"),
                                                      dimensionality="stack"),
                                  labels=LabelConfig(scale_yx=0.5), spatialdata=SpatialDataExportConfig(zarr_format=2))


# --- WF2: legacy configurations ------------------------------------------------------------------------------

@pytest.mark.parametrize("name", WF_CONFIGS)
def test_wf2_the_legacy_translation(name):
    config = wf_config(name)
    assert not {"stitching", "assembly", "export"} & set(config)
    translated = workflow._legacy_stitching(config)
    assert translated["stage_axes"] == StageAxisMapping(x="+x", y="+y", source="legacy")
    assert translated["rotate_angle"] == config["rotate_angle"]
    assert (translated["voxel_size_xy"], translated["voxel_size_z"]) == (config.get("voxel_size_xy"),
                                                                         config.get("voxel_size_z"))
    with pytest.raises(ValueError, match="stage_axes"):
        workflow._legacy_stitching(with_blocks(config))


def rule_names(text):
    return re.findall(r"^\s*rule (\w+):", text, flags=re.MULTILINE)


def test_wf2_legacy_configurations_include_the_legacy_rules_only():
    """Without the blocks the Snakefile includes stitching.smk (the four legacy rules) and no export rule; the
    legacy outputs of reads_assignment and the sample rules stay in reads-assignment.smk."""
    snakefile = (ROOT / "workflow/Snakefile").read_text()
    assert re.search(r'if STITCHING_BLOCK:\n\s+include: "rules/stitching-py.smk"\nelse:\n\s+include: '
                     r'"rules/stitching.smk"', snakefile)
    assert re.search(r'if ASSEMBLY_BLOCK:\n\s+include: "rules/export.smk"', snakefile)
    assert rule_names((ROOT / "workflow/rules/stitching.smk").read_text()) == [
        "stitching_preparation", "create_BigStitcher_macro", "run_BigStitcher_macro", "create_tile_config"]
    assert rule_names((ROOT / "workflow/rules/reads-assignment.smk").read_text()) == [
        "reads_assignment", "create_sample_h5ad", "create_sample_reads_assignment"]
    assert rule_names((ROOT / "workflow/rules/stitching-py.smk").read_text()) == [
        "bigstitcher_macro_sample", "bigstitcher_engine", "stitch_sample", "stitch_sample"]
    assert rule_names((ROOT / "workflow/rules/export.smk").read_text()) == ["export_sample"]
    common = (ROOT / "workflow/rules/common.smk").read_text()
    assert 'STITCHING_BLOCK = "stitching" in config' in common and 'ASSEMBLY_BLOCK = "assembly" in config' in common


# --- WF3: the new blocks -------------------------------------------------------------------------------------

def test_wf3_run_stitching_calls_dataset_stitch_with_the_configured_values(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(Dataset, "stitch", lambda self, *args, **kwargs: calls.append((self, args, kwargs)))
    config = dataset_config(tmp_path, stitching=MULTIVIEW)
    placement = tmp_path / "data/output/stitching" / SAMPLE / "placement.json"
    workflow._run_stitching(stub(config, inputs=[annotation(tmp_path, 1, 4)], outputs=[placement], sample=SAMPLE))
    (dataset, args, kwargs), = calls
    assert dataset.output_root == tmp_path / "data" / "output" and dataset.fov_pattern == "Position{i:03d}"
    assert args == (SAMPLE, ["Position001", "Position002", "Position003", "Position004"])
    assert kwargs == {"table": {"source": "regular_grid",
                                "regular_grid": {"columns": 2, "rows": 2, "order": "column_major_down_right",
                                                 "overlap": 0.25}},
                      "config": MultiviewStitcherConfig(binning_zyx=(1, 2, 2), quality_threshold=-1.0),
                      "input": "reference_stain", "review": ReviewPolicy(max_residual_um=2.0), "fallback": "nominal",
                      "output_dir": placement.parent}


BIGSTITCHER_DEFAULT = {k: v for k, v in STITCHING.items() if k != "method"}   # bigstitcher is the default method


@pytest.mark.parametrize("block, match", [
    (dict(MULTIVIEW, positions={"source": "stage_file"}), r"stitching\.positions\.stage_axes"),
    (dict(STITCHING, positions={"source": "stage_file", "stage_axes": None}), r"stitching\.positions\.stage_axes"),
    (dict(BIGSTITCHER_DEFAULT, bigstitcher_save_dir=None), r"stitching\.bigstitcher_save_dir"),
    ({k: v for k, v in STITCHING.items() if k != "bigstitcher_save_dir"}, r"stitching\.bigstitcher_save_dir"),
    (dict(MULTIVIEW, bigstitcher_save_dir="/local"), r"bigstitcher_save_dir is read with method bigstitcher only"),
    (dict(MULTIVIEW, fallback="metadata"), "fallback"),
    (dict(MULTIVIEW, input="dapi"), "input must be"),
    (dict(MULTIVIEW, config={"binning": 2}), r"unknown stitching\.config keys"),
    (dict(MULTIVIEW, positions={"source": "regular_grid", "grid": {}}), r"unknown stitching\.positions keys"),
])
def test_wf3_block_errors_name_the_key(tmp_path, block, match):
    with pytest.raises(ValueError, match=match):
        workflow._stitching_block(dataset_config(tmp_path, stitching=block))


def test_wf3_a_stage_file_without_stage_axes_raises_in_the_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(Dataset, "stitch", lambda *a, **k: pytest.fail("Dataset.stitch must not be called"))
    config = dataset_config(tmp_path, stitching=dict(MULTIVIEW, positions={"source": "stage_file"}))
    with pytest.raises(ValueError, match=r"stitching\.positions\.stage_axes"):
        workflow._run_stitching(stub(config, inputs=[annotation(tmp_path, 1, 4)],
                                     outputs=[tmp_path / "placement.json"], sample=SAMPLE))


def test_wf3_the_blocks_are_python_only_and_ordered(tmp_path):
    with pytest.raises(ValueError, match="Python-only"):
        workflow._stitching_block(dict(dataset_config(tmp_path, stitching=MULTIVIEW), backend="matlab"))
    with pytest.raises(ValueError, match="add a stitching block"):
        workflow._assembly_block(dataset_config(tmp_path, assembly=ASSEMBLY))
    with pytest.raises(ValueError, match="add an assembly block"):
        workflow._export_config(dataset_config(tmp_path, stitching=MULTIVIEW, export=EXPORT))
    with pytest.raises(ValueError, match=r"unknown export keys: \['labels_scale'\]"):
        workflow._export_config(dataset_config(tmp_path, stitching=MULTIVIEW, assembly=ASSEMBLY,
                                               export={"labels_scale": 0.25}))
    with pytest.raises(ValueError, match="unknown assembly keys"):
        workflow._assembly_block(dataset_config(tmp_path, stitching=MULTIVIEW, assembly={"filter": True}))


STAGE_AXES = {"x": "+x", "y": "+y"}
POSITION_CASES = {   # positions -> the position file the placement rules declare (OUT is the output root)
    "stage_file_default": ({"source": "stage_file", "stage_axes": STAGE_AXES}, ["OUT/documents/maf/sample1.maf"]),
    "stage_file_path": ({"source": "stage_file", "stage_axes": STAGE_AXES, "stage_file": "/data/stage/LN.maf"},
                        ["/data/stage/LN.maf"]),
    "table": ({"source": "table", "table": "/data/positions.csv", "unit": "um"}, ["/data/positions.csv"]),
    "regular_grid": (MULTIVIEW["positions"], []),
}


@pytest.mark.parametrize("case", sorted(POSITION_CASES))
def test_wf3_placement_rules_declare_the_position_file(tmp_path, case):
    """The input function of the placement rules gives the stage file (default or given) or the supplied table,
    and nothing for a regular grid."""
    positions, expected = POSITION_CASES[case]
    config = dataset_config(tmp_path, stitching=dict(MULTIVIEW, positions=positions))
    workflow._stitching_block(config)
    root = tmp_path / "data" / "output"
    assert workflow._stitching_position_inputs(config, root, SAMPLE) == [p.replace("OUT", str(root)) for p in expected]


@pytest.mark.parametrize("case", ["stage_file_default", "stage_file_path", "table"])
def test_wf3_the_declared_position_file_is_the_file_dataset_stitch_reads(tmp_path, monkeypatch, case):
    """table_from_positions (what Dataset.stitch and the BigStitcher rules call) opens the declared file."""
    import starfinder.stitching._dataset as positions_module
    positions, _ = POSITION_CASES[case]
    opened = []

    def spy(path, **kwargs):
        opened.append(str(path))
        raise StopIteration

    monkeypatch.setattr(positions_module, "positions_from_leica", spy)
    monkeypatch.setattr(positions_module, "positions_from_table", spy)
    config = dataset_config(tmp_path, stitching=dict(MULTIVIEW, positions=positions))
    root = tmp_path / "data" / "output"
    with pytest.raises(StopIteration):
        positions_module.table_from_positions(positions, ["Position001"], {}, sample=SAMPLE, output_root=root,
                                              fov_pattern="Position{i:03d}")
    assert opened == workflow._stitching_position_inputs(config, root, SAMPLE)


def test_wf3_the_placement_rules_use_the_position_input_function():
    text = (ROOT / "workflow/rules/stitching-py.smk").read_text()
    assert re.search(r"def get_stitching_positions\(wildcards\):\n(?:    .*\n)+?    return _stitching_position_inputs"
                     r"\(config, OUTPUT_DIR, wildcards\.sample\)", text)
    rules = re.split(r"\n\s*rule ", text)[1:]
    placing = [r for r in rules if r.startswith(("bigstitcher_macro_sample", "stitch_sample"))]
    assert len(placing) == 3 and all("get_stitching_positions" in r.split("output:")[0] for r in placing)
    engine_rule = next(r for r in rules if r.startswith("bigstitcher_engine"))
    assert "get_stitching_positions" not in engine_rule
    # With an assembly block reads_assignment reads the grid of the file reference_image names: an input then.
    reads = (ROOT / "workflow/rules/reads-assignment.smk").read_text()
    assert 'READS_ASSIGNMENT_GEOMETRY = _reference_image_inputs(config, OUTPUT_DIR, ["{fovID}"])' in reads


def test_wf3_export_sample_inputs_and_declared_output(tmp_path):
    config = dataset_config(tmp_path, stitching=MULTIVIEW, assembly=ASSEMBLY)
    root = Path("/out")
    inputs, output = workflow._export_sample_files(config, root, ["Position001", "Position002"], SAMPLE)
    assert inputs == ["/out/stitching/sample1/placement.json",
                      "/out/expr/Position001/raw.h5ad", "/out/expr/Position002/raw.h5ad",
                      "/out/expr/Position001/reads_assignment.csv", "/out/expr/Position002/reads_assignment.csv",
                      "/out/checkpoints/Position001/assignment/default/assignment.json",
                      "/out/checkpoints/Position002/assignment/default/assignment.json"]
    assert output == "/out/export/sample1/export.json"
    assert workflow._export_sample_files(config, root, [])[1] == "/out/export/{sample}/export.json"
    molecules = dataset_config(tmp_path, stitching=MULTIVIEW, assembly=dict(ASSEMBLY, assignment=None))
    assert workflow._export_sample_files(molecules, root, ["Position001"], SAMPLE)[0] == [
        "/out/stitching/sample1/placement.json", "/out/signal/Position001_goodSpots.csv"]
    # export.smk declares exactly these: its input function and its one output.
    text = (ROOT / "workflow/rules/export.smk").read_text()
    rule = text[text.index("rule export_sample:"):]
    assert re.search(r"input:\s+get_export_input\s+output:\s+_export_sample_files\(config, OUTPUT_DIR, \[\]\)\[1\]",
                     rule)
    assert 'script:\n        "../scripts/export_sample.py"' in rule


@pytest.fixture
def synchronous():
    import dask
    with dask.config.set(scheduler="synchronous"):
        yield


def run_export(tmp_path, config):
    folder = export_folder(tmp_path / "data")
    output = folder.export_dir / "export.json"
    workflow._run_export(stub(config, inputs=[folder.placement_dir / "placement.json"], outputs=[output],
                              sample=SAMPLE))
    return folder, json.loads(output.read_text())


@needs_anndata
@needs_library
def test_wf3_run_export_writes_the_export_folder_with_the_defaults(tmp_path, monkeypatch, synchronous):
    """Without an export block the defaults run, on the constructed placement saved in stitching/<sample>/."""
    calls = []
    real = Dataset.export
    monkeypatch.setattr(Dataset, "export", lambda self, *a, **k: calls.append((a, k)) or real(self, *a, **k))
    config = dataset_config(tmp_path, stitching=MULTIVIEW, assembly={"table_format": "csv"})
    folder, record = run_export(tmp_path, config)
    (args, kwargs), = calls
    assert args == (SAMPLE,)
    assert kwargs == {"placement": folder.placement_dir, "assignment": "default", "config": ExportConfig(),
                      "output_dir": folder.export_dir}
    assert record["parts"] == {"assemble": "written", "fuse": "written", "fuse_labels": "off",
                               "export_spatialdata": "off"}
    assert sorted(p.relative_to(folder.export_dir).as_posix() for p in folder.export_dir.rglob("*") if p.is_file()) \
        == ["export.json", "reference_stain_projection.ome.tif", "sample.h5ad", "tables/cells.csv",
            "tables/counts.csv", "tables/fovs.csv", "tables/molecules.csv"]
    # Nothing is written outside export/<sample>/ and stitching/<sample>/ is the saved placement, unchanged.
    assert load_placement(folder.placement_dir).fovs.equals(folder.placement.fovs)


@needs_anndata
def test_wf3_run_export_succeeds_without_multiview_stitcher(tmp_path, monkeypatch, synchronous):
    monkeypatch.setitem(sys.modules, "multiview_stitcher", None)
    config = dataset_config(tmp_path, stitching=MULTIVIEW, assembly=ASSEMBLY, export={"dimensionality": "stack"})
    with pytest.warns(ExportIncompleteWarning, match="fuse could not run: extra_missing"):
        _, record = run_export(tmp_path, config)
    assert record["parts"] == {"assemble": "written", "fuse": "not_run", "fuse_labels": "off",
                               "export_spatialdata": "off"}
    assert record["fusion"]["reason"] == "extra_missing" and record["complete"] is False


@needs_anndata
def test_wf3_run_export_molecules_only_loads_the_codebook(tmp_path, synchronous):
    """assembly.assignment null: the goodSpots tables are assembled, with the genes of the configured codebook."""
    config = dataset_config(tmp_path, stitching=MULTIVIEW, assembly=dict(ASSEMBLY, assignment=None),
                            export={"fuse": False}, seq_channel_order=["ch00", "ch01", "ch02", "ch03"])
    codebook = tmp_path / "input" / "data" / "sample" / "genes.csv"
    codebook.parent.mkdir(parents=True)
    golden_codebook().table[["entry_id", "gene_id", "color_sequence"]].to_csv(codebook, index=False)
    _, record = run_export(tmp_path, config)
    assert record["parts"] == {"assemble": "written", "fuse": "off", "fuse_labels": "off",
                               "export_spatialdata": "off"}
    assert record["assembly"]["mode"] == "molecules_only"
    assert record["assembly"]["genes"] == list(golden_codebook().genes)


@needs_anndata
def test_wf3_with_an_assembly_block_the_per_fov_files_have_no_filter_and_no_global(tmp_path, monkeypatch):
    """Option W1: no tile configuration is read, every kept cell and every molecule is written, no global_*, and
    the FOV's assignment folder is saved on the grid of images/ref_merged, for export_sample."""
    monkeypatch.setattr(workflow, "_tile_record", lambda *a: pytest.fail("the tile configuration must not be read"))
    labels = label_fixture()
    metadata = grid(SHAPE_ZYX, (0.35, 0.1, 0.1)).metadata
    reference = tmp_path / "output/data/out/images/ref_merged/Position001.tif"
    save_volume(np.zeros(SHAPE_ZYX, np.uint8), reference, metadata=metadata)
    adata, reads, _ = run_rule(tmp_path, labels, KEEP, stitching=MULTIVIEW, assembly=ASSEMBLY)
    record = json.loads(adata.uns["assignment"])
    counts = record["counts"]
    assert list(adata.obs.columns) == [c for c in LEGACY_OBS["3d"] if not c.startswith("global_")] + NEW_OBS
    assert len(adata.obs) == counts["cells_kept"]
    assert list(reads.columns) == [c for c in LEGACY_READS if not c.startswith("global_")] + NEW_READS
    assert len(reads) == counts["molecules"] == len(KEEP)
    assert reads.spot_id.tolist() == [f"csv:{i}" for i in range(len(KEEP))]       # input order, unfiltered
    folder = tmp_path / "output/data/out/checkpoints/Position001/assignment/default"
    saved = read_assignment(folder, "default")
    assert saved.record["name"] == record["name"] == "default"
    recorded = ReferenceGrid(tuple(saved.record["grid"]["shape_zyx"]), ImageMetadata(**saved.record["grid"]["metadata"]),
                             "declared")
    assert same_grid(recorded, reference_grid_from_file(reference))   # the grid check of assemble passes
    log = (tmp_path / "output/data/out/expr/Position001/log.txt").read_text().splitlines()[0]
    assert log.endswith(f"{len(adata.obs)} cells written without the tile filter (export_sample assembles the sample)")


@needs_anndata
def test_wf3_without_an_assembly_block_reads_assignment_is_the_legacy_rule(tmp_path):
    """The same run without the blocks reads the tile configuration and writes the legacy global_* columns."""
    adata, reads, _ = run_rule(tmp_path, label_fixture(), KEEP)
    assert list(adata.obs.columns) == LEGACY_OBS["3d"] + NEW_OBS and list(reads.columns) == LEGACY_READS + NEW_READS
    assert not (tmp_path / "output/data/out/checkpoints").exists()


# --- WF3: the BigStitcher rules ------------------------------------------------------------------------------

def without_stitching_transform(text, view):
    """The xml_links text with view ``view``'s Stitching Transform removed."""
    start = text.index(f'<ViewRegistration timepoint="0" setup="{view}">')
    begin = text.index('      <ViewTransform type="affine">\n        <Name>Stitching Transform</Name>', start)
    end = text.index("</ViewTransform>\n", begin) + len("</ViewTransform>\n")
    return text[:begin] + text[end:]


def engine(tmp_path, monkeypatch, *, text, returncode, config=None):
    """bigstitcher_engine through a stub snakemake object; Fiji is a stand-in that writes ``text`` (None: nothing)."""
    folder = tmp_path / "bigstitcher" / SAMPLE
    folder.mkdir(parents=True, exist_ok=True)
    macro, view_map, xml = folder / "BigStitcher_macro.ijm", folder / "view_map.csv", folder / "dataset.xml"
    macro.write_text("// macro\n")
    links_view_map().to_csv(view_map, index=False)
    commands = []

    def fiji(command, check):
        commands.append(command)
        if text is not None:
            xml.write_text(text)
        return subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr(subprocess, "run", fiji)
    config = {"fiji_path": "/opt/Fiji.app/ImageJ-linux64"} if config is None else config
    workflow._run_bigstitcher_engine(stub(config, inputs=[macro, view_map], outputs=[xml], sample=SAMPLE))
    return commands, xml


@pytest.mark.parametrize("returncode", [0, 1])
def test_wf3_engine_check_accepts_the_complete_file(tmp_path, monkeypatch, returncode):
    commands, xml = engine(tmp_path, monkeypatch, text=links_xml(), returncode=returncode)
    assert commands == [["/opt/Fiji.app/ImageJ-linux64", f"-Djava.io.tmpdir={xml.parent}", "--ij2", "--headless",
                         "--run", str(xml.parent / "BigStitcher_macro.ijm")]]


@pytest.mark.parametrize("returncode", [0, 1])
def test_wf3_engine_check_fails_without_view_3_s_stitching_transform(tmp_path, monkeypatch, returncode):
    text = without_stitching_transform(links_xml(), 3)
    assert text.count("Stitching Transform") == 3
    with pytest.raises(RuntimeError, match=rf"no 'Stitching Transform' for views 3; .*status {returncode}"):
        engine(tmp_path, monkeypatch, text=text, returncode=returncode)


@pytest.mark.parametrize("returncode", [0, 1])
def test_wf3_engine_check_fails_without_a_file(tmp_path, monkeypatch, returncode):
    xml = tmp_path / "bigstitcher" / SAMPLE / "dataset.xml"
    with pytest.raises(RuntimeError, match=rf"no dataset.xml at {re.escape(str(xml))} \(Fiji exited with status "
                                           rf"{returncode}\)"):
        engine(tmp_path, monkeypatch, text=None, returncode=returncode)


def test_wf3_engine_check_removes_a_stale_file_first(tmp_path, monkeypatch):
    folder = tmp_path / "bigstitcher" / SAMPLE
    folder.mkdir(parents=True)
    (folder / "dataset.xml").write_text(links_xml())
    with pytest.raises(RuntimeError, match="no dataset.xml"):
        engine(tmp_path, monkeypatch, text=None, returncode=0)


def test_wf3_fiji_path_is_read_only_when_the_engine_rule_runs(tmp_path, monkeypatch):
    """The block and the rule files need no fiji_path; the engine rule raises naming it when it runs."""
    config = dataset_config(tmp_path, stitching=dict(STITCHING, bigstitcher_save_dir=str(tmp_path / "bigstitcher")))
    assert "fiji_path" not in config
    workflow._stitching_block(config)
    rules = (ROOT / "workflow/rules/stitching-py.smk").read_text()
    assert not re.search(r"config\[['\"]fiji_path['\"]\]|config\.get\(['\"]fiji_path", rules)
    with pytest.raises(ValueError, match="fiji_path"):
        engine(tmp_path, monkeypatch, text=links_xml(), returncode=0, config=config)


def test_wf3_the_bigstitcher_path_end_to_end(tmp_path, monkeypatch):
    """xml_links as a dataset: the macro rule writes the tiles, the macro and the view map; the engine (a stand-in)
    writes dataset.xml; stitch_sample imports it, saves the placement and writes tile_config_{sample}.csv."""
    root = tmp_path / "data" / "output"
    save = tmp_path / "bigstitcher"
    for k, fov in enumerate(LINK_IDS):
        save_volume(np.full(LINK_SHAPE, k + 1, np.uint16), root / "images/ref_merged" / f"{fov}.tif",
                    metadata=grid(LINK_SHAPE, LN_SPACING).metadata)
    positions = {"source": "regular_grid", "regular_grid": {"columns": 2, "rows": 2,
                                                            "order": "column_major_down_right", "overlap": 0.25}}
    config = dataset_config(tmp_path, pattern="L{i}", stitching={"positions": positions,
                                                                  "bigstitcher_save_dir": str(save)},
                            img_col=LINK_SHAPE[2], img_row=LINK_SHAPE[1])
    sheet = annotation(tmp_path, 0, 3)
    folder = save / SAMPLE
    macro, view_map = folder / "BigStitcher_macro.ijm", folder / "view_map.csv"
    folder.mkdir(parents=True)
    workflow._run_bigstitcher_macro(stub(config, inputs=[sheet], outputs=[macro, view_map], sample=SAMPLE))
    assert f"dataset_save_path={folder} " in macro.read_text() and f"path={folder}/tiles/*.tif" in macro.read_text()
    assert sorted(p.name for p in (folder / "tiles").iterdir()) == [f"tile_{j}.tif" for j in range(4)]
    assert [int(tifffile.imread(folder / "tiles" / f"tile_{j}.tif").max()) for j in range(4)] == [1, 2, 3, 4]
    assert pd.read_csv(view_map, keep_default_na=False).equals(links_view_map())
    commands, xml = engine(tmp_path, monkeypatch, text=links_xml(), returncode=0)
    placement, tile_config = root / "stitching" / SAMPLE / "placement.json", root / "output/tile_config_sample1.csv"
    workflow._run_stitching(stub(config, inputs=[sheet, xml, view_map, macro], outputs=[placement, tile_config],
                                 sample=SAMPLE))
    saved = load_placement(placement.parent)
    expected = import_bigstitcher(xml, table=links_table(), view_map=links_view_map(),
                                  grids={fov: grid(LINK_SHAPE, LN_SPACING) for fov in LINK_IDS},
                                  config=BigStitcherConfig(), sample=SAMPLE, macro=macro.read_text())
    pd.testing.assert_frame_equal(saved.fovs, expected.fovs, check_exact=True)
    pd.testing.assert_frame_equal(saved.links, expected.links, check_exact=True)
    assert saved.record["method"]["macro_sha256"] == expected.record["method"]["macro_sha256"]
    assert sorted(p.name for p in placement.parent.iterdir()) == ["fovs.csv", "links.csv", "placement.json",
                                                                  "view_map.csv"]
    assert tile_config.read_text().startswith(",x,y,z,fov_index,col,row,id,grid,start_x,")


# --- WF3: the legacy writer from the record ------------------------------------------------------------------

@pytest.mark.parametrize("fallback", [None, "nominal"])
def test_wf3_legacy_writer_golden(tmp_path, golden_stitching_setups, fallback):  # noqa: F811
    setups, plain, grid_csv, table, view_map, grids = golden_stitching_setups
    result = import_bigstitcher(setups, table=table, view_map=view_map, grids=grids, config=BigStitcherConfig(),
                                fallback=fallback)
    path = tmp_path / "output" / "tile_config_sample1.csv"
    workflow._write_tile_config(result, path, {"img_col": TILE_COL, "img_row": TILE_ROW})
    frame, _ = legacy_tile_config(plain, grid_csv, TILE_COL, TILE_ROW)
    assert path.read_bytes() == frame.to_csv().encode()
    written = pd.read_csv(path, index_col=0)
    columns = ["x", "y", "z", "start_x", "start_y", "end_x", "end_y",
               "start_x_norm", "start_y_norm", "end_x_norm", "end_y_norm"]
    assert [tuple(int(v) for v in row) for row in written[columns].itertuples(index=False)] == TILE_CONFIG


@pytest.mark.parametrize("fallback", [None, "nominal"])
@pytest.mark.parametrize("name", sorted(HAND_RECORDS))
def test_wf3_legacy_writer_hand_records(tmp_path, hand_records_setups, name, fallback):  # noqa: F811
    setups, plain, grid_csv, table, view_map, grids = hand_records_setups[name]
    result = import_bigstitcher(setups, table=table, view_map=view_map, grids=grids, config=BigStitcherConfig(),
                                fallback=fallback)
    path = tmp_path / f"tile_config_{name}.csv"
    workflow._write_tile_config(result, path, {"img_col": 100, "img_row": 100})
    frame, _ = legacy_tile_config(plain, grid_csv, 100, 100)
    assert path.read_bytes() == frame.to_csv().encode()


# --- WF3: FOV names of a stage file at the workflow boundary ----------------------------------------------------
# A sample of FOVs 9 to 12 on a 2 × 2 stage grid (step 261 µm, tiles of 16 × 24 px at 18 × 12 µm, so 288 µm), the
# stage file in the order 9, 10, 12, 11. Its PositionIDs are the FOV numbers i, as create_sample_maf.py keeps them.
STAGE_POINTS = ((9, "0.076", "0.048"), (10, "0.076", "0.048261"), (12, "0.076261", "0.048261"),
                (11, "0.076261", "0.048"))
STAGE_SHAPE = (2, 16, 24)
STAGE_POSITIONS = {"source": "stage_file", "stage_axes": {"x": "+x", "y": "+y"}, "snap_to_grid": {"overlap": 0.1}}


def stage_dataset(tmp_path, pattern, points=STAGE_POINTS, start=9, end=12, **blocks):
    """A configuration, the sample annotation, the default stage file and a ref_merged image per FOV of the sample."""
    config = dataset_config(tmp_path, pattern=pattern, **blocks)
    sheet = annotation(tmp_path, start, end)
    root = tmp_path / "data" / "output"
    (root / "documents" / "maf").mkdir(parents=True, exist_ok=True)
    (root / "documents" / "maf" / f"{SAMPLE}.maf").write_text(maf_text(points))
    for i in range(start, end + 1):
        save_volume(np.full(STAGE_SHAPE, i, np.uint16), root / "images/ref_merged" / f"{pattern.format(i=i)}.tif",
                    metadata=grid(STAGE_SHAPE, STAGE_SPACING).metadata)
    return config, sheet, root


@pytest.mark.parametrize("pattern", ["tile_{i:02d}", "tile_{i}"])
def test_wf3_stage_file_names_in_process(tmp_path, monkeypatch, pattern):
    """_run_stitching reaches Dataset.stitch (run for real up to stitch) with a table named by fov_id_pattern."""
    import starfinder.stitching as stitching
    config, sheet, root = stage_dataset(tmp_path, pattern, stitching=dict(MULTIVIEW, positions=STAGE_POSITIONS,
                                                                           input="reference_merged"))
    calls = []
    monkeypatch.setattr(stitching, "stitch", lambda inputs, table, **kw: calls.append((inputs, table, kw)) or "result")
    monkeypatch.setattr(stitching, "save_placement", lambda result, folder: None)
    placement = root / "stitching" / SAMPLE / "placement.json"
    workflow._run_stitching(stub(config, inputs=[sheet], outputs=[placement], sample=SAMPLE))
    (inputs, table, kwargs), = calls
    names = [pattern.format(i=p) for p, *_ in STAGE_POINTS]
    assert table.fov_ids == names                                      # stage-file order
    assert table.positions["position_id"].tolist() == [9, 10, 12, 11]
    assert [item.fov_id for item in inputs] == names and kwargs["sample"] == SAMPLE
    assert set(names) == set(workflow._sample_fovs(sheet, config, SAMPLE))
    assert table.record["fov_ids"] == {str(p): pattern.format(i=p) for p, *_ in STAGE_POINTS}


@pytest.mark.parametrize("pattern", ["tile_{i:02d}", "tile_{i}"])
def test_wf3_stage_file_names_bigstitcher(tmp_path, monkeypatch, pattern):
    """The macro rule's view map and the import rule's table carry the names of fov_id_pattern."""
    import starfinder.stitching as stitching
    save = tmp_path / "bigstitcher"
    config, sheet, root = stage_dataset(tmp_path, pattern, stitching={"positions": STAGE_POSITIONS,
                                                                      "bigstitcher_save_dir": str(save)})
    folder = save / SAMPLE
    folder.mkdir(parents=True)
    macro, view_map = folder / "BigStitcher_macro.ijm", folder / "view_map.csv"
    workflow._run_bigstitcher_macro(stub(config, inputs=[sheet], outputs=[macro, view_map], sample=SAMPLE))
    views = pd.read_csv(view_map, keep_default_na=False)
    expected = [pattern.format(i=i) for i in (9, 10, 11, 12)]          # column-major cells: 9, 10 | 11, 12
    assert views["fov_id"].tolist() == expected and views["kind"].tolist() == ["fov"] * 4
    assert expected == workflow._sample_fovs(sheet, config, SAMPLE)
    tables = []

    def spy(xml, *, table, **kwargs):
        tables.append(table)
        raise StopIteration

    monkeypatch.setattr(stitching, "import_bigstitcher", spy)
    with pytest.raises(StopIteration):
        workflow._run_stitching(stub(config, inputs=[sheet, folder / "dataset.xml", view_map, macro],
                                     outputs=[root / "stitching" / SAMPLE / "placement.json",
                                              root / "output" / "tile_config_sample1.csv"], sample=SAMPLE))
    assert tables[0].fov_ids == [pattern.format(i=p) for p, *_ in STAGE_POINTS]


@pytest.mark.parametrize("points, match", [
    (STAGE_POINTS + ((13, "0.076522", "0.048"),), r"missing \[\], extra \['tile_13'\]"),   # not a FOV of the sample
    (STAGE_POINTS[:3], r"missing \['tile_11'\], extra \[\]"),                                # a FOV not listed
])
def test_wf3_a_stage_file_that_does_not_match_the_sample_raises(tmp_path, points, match):
    """As Dataset.stitch for a table that does not match its FOVs: ValueError naming both lists."""
    config, sheet, root = stage_dataset(tmp_path, "tile_{i}", points=points,
                                        stitching=dict(MULTIVIEW, positions=STAGE_POSITIONS))
    with pytest.raises(ValueError, match=match):
        workflow._run_stitching(stub(config, inputs=[sheet], outputs=[root / "stitching/sample1/placement.json"],
                                     sample=SAMPLE))
