"""WF1 to WF3 of design/validation/stitching-assembly.md: the §2.10 workflow blocks (W-372).

The Python-only ``stitching``, ``assembly`` and ``export`` blocks: their keys in the MATLAB schema's definitions
(``workflow/schemas/config.schema.yaml``), their validation by the package (the Python schema, the block translators
and the data dependencies, as the plan validates them), the translators of ``starfinder.dataset.workflow``, the
legacy reads_assignment adapter (``starfinder._legacy_workflow``) on the route of an assembly block, and the frozen
MATLAB rule set's stitching and assignment rules. Fixtures, all written in the test: ``wf_configs`` (the repository's
example configurations, and copies with the blocks) and configurations in the Python shape. The adapter runs
through a stub ``snakemake`` object, as in ``test_assignment_workflow.py``. The tests of the stitching, BigStitcher,
export and assign rules' adapters moved to the commands (test_workflow_stitch_tasks.py, test_workflow_export_tasks.py,
test_workflow_cell_tasks.py) and their declarations to the plan (test_workflow_declared_outputs.py), W-750.
"""
import copy
import importlib.util
import inspect
import json
import re
from dataclasses import fields
from pathlib import Path

import jsonschema
import numpy as np
import pytest
import yaml

import starfinder._legacy_workflow as legacy
import starfinder.dataset.workflow as workflow
from starfinder.assembly import AssemblyConfig
from starfinder.dataset import Dataset
from starfinder.dataset._stitching import POSITION_KEYS
from starfinder.io import save_volume
from starfinder.export import (ExportConfig, FusionConfig, LabelConfig,
                                      SpatialDataExportConfig)
from starfinder.stitching import (STITCHING_METHODS, BigStitcherConfig, MultiviewStitcherConfig, ReviewPolicy)

from .stitching_fixtures import grid
from .test_assignment_golden import SHAPE_ZYX, label_fixture
from .test_assignment_workflow import (ASSIGNMENT_READS, KEEP, LEGACY_OBS, MOLECULE_READS, NEW_OBS,
                                       run_rule)
from .test_stitching_bigstitcher import (HAND_RECORDS, LINK_IDS, LINK_SHAPE, golden_stitching_setups,  # noqa: F401
                                         hand_records_setups, links_table, links_view_map, links_xml)

pytestmark = [pytest.mark.workflow, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = yaml.safe_load((ROOT / "workflow/schemas/config.schema.yaml").read_text())
DEFS = SCHEMA["$defs"]
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
  stage_file: /data/stage/acquisition.maf
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
duplicate_threshold: 0.5
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
    return yaml.safe_load((ROOT / "src/python/test/data/workflow_examples" / name).read_text())


def with_blocks(config, *, stitching=STITCHING, assembly=ASSEMBLY, export=EXPORT):
    out = copy.deepcopy(config)
    out["backend"] = "python"
    for key, block in (("stitching", stitching), ("assembly", assembly), ("export", export)):
        if block is not None:
            out[key] = copy.deepcopy(block)
    return out


def dataset_config(tmp_path, **blocks):
    """A Python configuration whose input root is tmp_path/input/data/sample and output root tmp_path/data/output
    (dataset_id data, sample_id sample, output_id output)."""
    return {"backend": "python", "dataset_id": "data", "sample_id": "sample", "output_id": "output",
            "root_input_path": str(tmp_path / "input"), "root_output_path": str(tmp_path),
            "n_rounds": 3, "ref_round": "round1", "rotate_angle": 0, "img_col": 64, "img_row": 48,
            "voxel_size_xy": 1.0, "voxel_size_z": 1.0, "rules": {}, **blocks}


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
    assert declared == {f.name for f in fields(AssemblyConfig) if f.init} | {"assignment", "table_format", "previews"}
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


def python_shape(tmp_path, *, stitching=STITCHING, assembly=ASSEMBLY, export=EXPORT):
    """The worked example's dataset keys and processing block in the Python shape with the three blocks (a block None
    is left out); an assignment name of the assembly block gets the assignment block and the segment run it reads (the
    route per FOV, which the export block then names)."""
    from . import workflow_examples as X
    config = dict(X.dataset(tmp_path, "blocks"), processing=copy.deepcopy(X.PROCESSING), morphology={})
    for key, block in (("stitching", stitching), ("assembly", assembly), ("export", export)):
        if block is not None:
            config[key] = copy.deepcopy(block)
    if assembly is not None and assembly.get("assignment") is not None:
        config["segmentation"] = {"runs": [{"name": "cell", "target": "cell", "method": "import",
                                            "path": str(tmp_path / "{fov_id}.tif")}]}
        config["assignment"] = {"cells": "cell", "name": assembly["assignment"]}
        if export is not None:
            config["export"]["route"] = "per_fov"
    return config


def test_wf1_every_configuration_validates(tmp_path):
    """The package validates the configuration with the three blocks, with the multiview block and an export block
    of only the route, and with the stitching block alone (the Python schema, the translators and the data
    dependencies, as the plan validates them; W-750). Without the export block the export's defaults apply (the route
    after fusion for dimensionality projection), which an assignment name refuses without export.segmentation."""
    from . import workflow_tasks as T
    assert T.validation_error(python_shape(tmp_path)) is None
    assert T.validation_error(python_shape(tmp_path, stitching=MULTIVIEW, export={"route": "per_fov"})) is None
    assert "export.segmentation" in T.validation_error(python_shape(tmp_path, stitching=MULTIVIEW, export=None))
    assert T.validation_error(python_shape(tmp_path, assembly=None, export=None)) is None


@pytest.mark.parametrize("change", [
    {"backend": "matlab"},                                   # Python-only
    {"stitching": dict(STITCHING, method="grid_stitcher")},  # an unregistered method
    {"stitching": dict(STITCHING, tile_filter=True)},        # an unknown key
    {"assembly": dict(ASSEMBLY, ownership="voronoi")},
    {"assembly": dict(ASSEMBLY, duplicate_threshold=0)},     # the duplicate rule: above 0 and at most 1, or null
    {"assembly": dict(ASSEMBLY, duplicate_threshold=1.5)},
    {"assembly": dict(ASSEMBLY, duplicate_threshold="0.5")},
    {"assembly": dict(ASSEMBLY, duplicate_threshold=True)},
    {"export": dict(EXPORT, labels_scale=0.25)},             # the block's names, not LabelConfig's
    {"stitching": None},                                     # assembly without stitching
    {"assembly": None},                                      # export without assembly
])
def test_wf1_invalid_blocks_fail_the_schema(tmp_path, change):
    """The package refuses each (the Python schema, the translators or the data dependencies, as the plan validates
    them; W-750)."""
    from . import workflow_tasks as T
    config = python_shape(tmp_path)
    for key, value in change.items():
        if value is None:
            config.pop(key)
        else:
            config[key] = value
    assert T.validation_error(config)


@pytest.mark.parametrize("block", ["stitching", "assembly", "export"])
def test_wf1_each_block_is_python_only_in_the_matlab_schema(tmp_path, block):
    """From test_wf1_invalid_blocks_fail_the_schema, its backend: matlab case: the MATLAB workflow validates its
    configuration with the MATLAB schema before any rule (workflow/rules/legacy.smk), whose Python-only clause refuses
    the stitching, assembly and export blocks: a valid MATLAB configuration passes with backend matlab and with the
    backend omitted (the schema's default), and with the block it is refused by that clause, which names the block,
    in both cases."""
    from .test_fov_membership import matlab_config
    config = matlab_config(tmp_path)
    validator = jsonschema.Draft202012Validator(SCHEMA)
    value = {"stitching": STITCHING, "assembly": ASSEMBLY, "export": EXPORT}[block]
    assert SCHEMA["properties"]["backend"]["default"] == "matlab"
    for backend in ({"backend": "matlab"}, {}):
        plain = {**{k: v for k, v in config.items() if k != "backend"}, **backend}
        assert validator.is_valid(plain), list(validator.iter_errors(plain))
        errors = list(validator.iter_errors(dict(plain, **{block: copy.deepcopy(value)})))
        clause = [e for e in errors if e.validator == "not"]
        assert len(clause) == 1 and {"required": [block]} in clause[0].validator_value["anyOf"], (block, errors)
        # The rest are the blocks' order (an assembly block needs stitching, export needs assembly).
        assert all(e.validator == "dependentRequired" for e in errors if e not in clause), (block, errors)


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


def test_wf1_assembly_previews_is_an_opt_in_flag(tmp_path):
    """W-568 (OM-16, A): assembly.previews is a Boolean, false by default, in the schema and the adapter."""
    assert DEFS["assembly_block"]["properties"]["previews"] == {
        "description": DEFS["assembly_block"]["properties"]["previews"]["description"], "type": "boolean"}
    config = dataset_config(tmp_path, stitching=MULTIVIEW, assembly=ASSEMBLY)
    assert "previews" not in ASSEMBLY and workflow._assembly_block(config).previews is False
    assert workflow._assembly_block(dict(config, assembly=dict(ASSEMBLY, previews=True))).previews is True
    with pytest.raises(ValueError, match=r"assembly\.previews must be Boolean"):
        workflow._assembly_block(dict(config, assembly=dict(ASSEMBLY, previews="yes")))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(ASSEMBLY, previews="yes"), DEFS["assembly_block"])


# --- WF2: legacy configurations ------------------------------------------------------------------------------

def rule_names(text):
    return re.findall(r"^\s*rule (\w+):", text, flags=re.MULTILINE)


def test_wf2_legacy_configurations_include_the_legacy_rules_only():
    """The dispatcher includes rules/legacy.smk for backend matlab only, which includes stitching.smk (the four legacy
    rules) and no export rule; the legacy outputs of reads_assignment and the sample rules stay in
    reads-assignment.smk. (That the Python route includes no rule file of the old Python workflow is C8,
    test_workflow_boundary.py, W-750.)"""
    snakefile = (ROOT / "workflow/Snakefile").read_text()
    legacy_rules = (ROOT / "workflow/rules/legacy.smk").read_text()
    assert re.search(r'if config.get\("backend"\) == "matlab":\n\s+include: "rules/legacy.smk"\n', snakefile)
    assert 'include: "stitching.smk"' in legacy_rules
    assert rule_names((ROOT / "workflow/rules/stitching.smk").read_text()) == [
        "stitching_preparation", "create_BigStitcher_macro", "run_BigStitcher_macro", "create_tile_config"]
    assert rule_names((ROOT / "workflow/rules/reads-assignment.smk").read_text()) == [
        "reads_assignment", "create_sample_h5ad", "create_sample_reads_assignment"]


# --- WF3: the new blocks -------------------------------------------------------------------------------------

BIGSTITCHER_DEFAULT = {k: v for k, v in STITCHING.items() if k != "method"}   # bigstitcher is the default method


@pytest.mark.parametrize("block, match", [
    (dict(MULTIVIEW, positions={"source": "stage_file"}), r"stitching\.positions\.stage_axes"),
    (dict(STITCHING, positions={"source": "stage_file", "stage_axes": None}), r"stitching\.positions\.stage_axes"),
    (dict(MULTIVIEW, positions={"source": "stage_file", "stage_axes": {"x": "+x", "y": "+y"}}),
     r"stitching\.positions\.stage_file .* is required with source stage_file; it has no default"),
    (dict(MULTIVIEW, positions={"source": "table"}), r"stitching\.positions\.table .* is required with source table"),
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


@needs_anndata
def test_wf3_molecules_only_writes_the_per_fov_files_without_filter_and_global(tmp_path, monkeypatch):
    """Option W1 with assembly.assignment null (unchanged by W-568): no tile configuration is read, every kept cell
    and every molecule is written, no global_*, log.txt, and no assignment folder."""
    monkeypatch.setattr(legacy, "_tile_record", lambda *a: pytest.fail("the tile configuration must not be read"))
    labels = label_fixture()
    metadata = grid(SHAPE_ZYX, (0.35, 0.1, 0.1)).metadata
    reference = tmp_path / "output/data/out/fovs/Position001/reference.tif"
    save_volume(np.zeros(SHAPE_ZYX, np.uint8), reference, metadata=metadata)
    adata, reads, _ = run_rule(tmp_path, labels, KEEP, stitching=MULTIVIEW, assembly=dict(ASSEMBLY, assignment=None))
    record = json.loads(adata.uns["assignment"])
    counts = record["counts"]
    assert record["name"] == "reads_assignment"
    assert list(adata.obs.columns) == [c for c in LEGACY_OBS["3d"] if not c.startswith("global_")] + NEW_OBS
    assert len(adata.obs) == counts["cells_kept"]
    assert list(reads.columns) == [c for c in MOLECULE_READS if not c.startswith("global_")] + ASSIGNMENT_READS
    assert len(reads) == counts["molecules"] == len(KEEP)
    assert reads.spot_id.tolist() == [f"m{i}" for i in range(len(KEEP))]       # input order, unfiltered
    assert not (tmp_path / "output/data/out/fovs/Position001/assignment").exists()
    log = (tmp_path / "output/data/out/expr/Position001/log.txt").read_text().splitlines()[0]
    assert log.endswith(f"{len(adata.obs)} cells written without the tile filter (export_sample assembles the sample)")


@needs_anndata
def test_wf3_without_an_assembly_block_reads_assignment_is_the_legacy_rule(tmp_path):
    """The same run without the blocks reads the tile configuration and writes the legacy global_* columns."""
    adata, reads, _ = run_rule(tmp_path, label_fixture(), KEEP)
    assert list(adata.obs.columns) == LEGACY_OBS["3d"] + NEW_OBS and list(reads.columns) == MOLECULE_READS + ASSIGNMENT_READS
    assert not (tmp_path / "output/data/out/checkpoints").exists()
