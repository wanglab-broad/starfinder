"""The §2.9 assignment workflow key: the legacy reads_assignment translation (row A18), the adapter's
raw.h5ad and reads_assignment.csv, the intentional changes, the compartment layers, the Python-only
assignment block and the static schema (W-317; docs/assignment-contract.md, "Workflow configuration")."""
import inspect
import io
import json
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml
from pandas.testing import assert_frame_equal

from starfinder.assignment import AssignmentConfig, CorrespondenceConfig
from starfinder.dataset import FOV, CheckpointConfig
from starfinder.dataset.workflow import _assignment_block, _reads_assignment, _run_reads_assignment, _write_raw_h5ad
from starfinder.segmentation import ExpandLabelsConfig

from .test_assignment import assign_boxes
from .test_assignment_golden import DISTANCE, GENES, MOLECULES, SHAPE_ZYX, label_fixture, legacy_assignment

pytestmark = [pytest.mark.segmentation, pytest.mark.workflow, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = yaml.safe_load((ROOT / "workflow/schemas/config.schema.yaml").read_text())
KEEP = tuple(m for m in MOLECULES if m[1] != "Z")
LEGACY_OBS = {"3d": ["sample", "fov_id", "volume", "fov_x", "fov_y", "fov_z", "seg_label", "global_x", "global_y",
                     "global_z"],
              "2d": ["sample", "fov_id", "volume", "fov_x", "fov_y", "seg_label", "global_x", "global_y"]}
NEW_OBS = ["size_voxels", "expanded_size_voxels", "size_physical", "centroid_z", "centroid_y", "centroid_x",
           "n_molecules", "n_nuclei", "correspondence", "correspondence_flags", "compartments"]
LEGACY_READS = ["x", "y", "z", "gene", "global_x", "global_y", "global_z", "seg_label"]
NEW_READS = ["spot_id", "assignment_status", "cell_id", "in_expansion", "original_cell_id", "nucleus_id",
             "compartment"]


def workflow(tmp_path, *, expand=False, seg_expand=False, backend="python", **extra):
    return {"backend": backend, "dataset_id": "data", "sample_id": "sample", "output_id": "out",
            "root_input_path": str(tmp_path / "input"), "root_output_path": str(tmp_path / "output"),
            "fov_id_pattern": "Position{i:03d}", "img_z": SHAPE_ZYX[0], "voxel_size_z": 0.35, "voxel_size_xy": 0.1,
            "rules": {"stardist_segmentation": {"parameters": {"segmentation_input_folder": "overlay",
                                                               "expand_labels": seg_expand, "distance": 4}},
                      "reads_assignment": {"parameters": {"expand_labels": expand, "dilation_distance": DISTANCE}}},
            **extra}


def write_reads(path, molecules, *, floats=False):
    rows = [(x + 1, y + 1, z + 1, gene) for (z, y, x), gene in molecules]
    frame = pd.DataFrame(rows, columns=["x", "y", "z", "gene"])
    if floats:  # as starfinder.io.export_spots writes them
        frame[["x", "y", "z"]] = frame[["x", "y", "z"]].astype(np.float64)
    frame.to_csv(path, index=False)
    return path


def run_rule(tmp_path, labels, molecules=KEEP, *, floats=False, codebook=GENES, **options):
    """reads_assignment through a stub snakemake object: one FOV, tile offsets 0, a box covering the FOV."""
    anndata = pytest.importorskip("anndata")
    config = workflow(tmp_path, **options)
    out = tmp_path / "output" / "data" / "out"
    for folder in ("documents", "output", "signal", "images/DAPI", "images/stardist_segmentation"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    (out / "documents" / "sample-annotation.csv").write_text("sample_id,fov_start,fov_end\nsample1,1,1\n")
    (out / "documents" / "genes.csv").write_text("".join(f"{g},{b}\n" for g, b in GENES))
    codebook_path = tmp_path / "input" / "data" / "sample" / "genes.csv"
    codebook_path.parent.mkdir(parents=True)
    codebook_path.write_text("".join(f"{g},{b}\n" for g, b in codebook))
    ny, nx = labels.shape[-2:]
    pd.DataFrame([{"id": 1, "x": 0, "y": 0, "z": 0, "start_x_norm": 0, "end_x_norm": nx, "start_y_norm": 0,
                   "end_y_norm": ny}]).to_csv(out / "output" / "tile_config_sample1.csv")
    tifffile.imwrite(out / "images" / "DAPI" / "Position001.tif", np.zeros(labels.shape, np.uint8))
    tifffile.imwrite(out / "images" / "stardist_segmentation" / "Position001.tif", labels)
    reads = write_reads(out / "signal" / "Position001_goodSpots.csv", molecules, floats=floats)
    inputs = [out / "documents" / "sample-annotation.csv", out / "images" / "DAPI" / "Position001.tif",
              out / "images" / "stardist_segmentation" / "Position001.tif", reads, out / "documents" / "genes.csv",
              out / "output" / "tile_config_sample1.csv"]
    outputs = [out / "expr" / "Position001" / "raw.h5ad", out / "expr" / "Position001" / "reads_assignment.csv"]
    _run_reads_assignment(SimpleNamespace(input=[str(p) for p in inputs], output=[str(p) for p in outputs],
                                          config=config, wildcards=SimpleNamespace(fovID="Position001")))
    return anndata.read_h5ad(outputs[0]), pd.read_csv(outputs[1]), reads


def legacy_reads_assignment(labels, reads_csv, genes_csv, *, expand):
    """legacy_assignment, then the script's overlap filter (lines 155-165) with tile offsets 0 and a covering box."""
    table, counts, meta = legacy_assignment(labels, reads_csv, genes_csv, expand=expand)
    for axis in ("x", "y", "z"):
        table[f"global_{axis}"] = table[axis]  # lines 78-80 with offsets 0
    table = table[["x", "y", "z", "gene", "global_x", "global_y", "global_z", "seg_label"]]
    for axis in ("x", "y", "z") if "fov_z" in meta else ("x", "y"):
        meta[f"global_{axis}"] = meta[f"fov_{axis}"]  # lines 143-149 with offsets 0
    cells = table[table.seg_label.isin(meta.seg_label)]
    background = table[table.seg_label == 0]
    return pd.concat([cells, background]), counts, meta


def roundtrip(frame):
    return pd.read_csv(io.StringIO(frame.to_csv(index=False)))


# --- A18: the translation of the legacy keys -----------------------------------------------------------

def test_a18_without_expansion_dilation_distance_is_ignored_and_recorded(tmp_path):
    config, target, record = _reads_assignment(workflow(tmp_path))
    assert config == AssignmentConfig() and target == "cell"
    assert record["dilation_distance_ignored"] and record["target_source"] == "legacy_default"


def test_a18_dilation_distance_is_one_legacy_pixel_expansion(tmp_path):
    config, _, record = _reads_assignment(workflow(tmp_path, expand=True))
    assert config == AssignmentConfig(expansion=ExpandLabelsConfig(DISTANCE, "pixel", "planar"),
                                      legacy_pixel_expansion=True)
    assert not record["dilation_distance_ignored"]


@pytest.mark.parametrize("expand", [False, True])
def test_a18_an_expansion_in_segmentation_raises_naming_the_keys(tmp_path, expand):
    with pytest.raises(ValueError) as error:
        _reads_assignment(workflow(tmp_path, expand=expand, seg_expand=True))
    message = str(error.value)
    assert "rules.stardist_segmentation.parameters.expand_labels" in message
    assert ("and rules.reads_assignment.parameters.expand_labels are true" in message) == expand
    assert "dilation_distance" in message


def test_a18_the_label_target_is_the_segmentation_adapter_s(tmp_path):
    config = workflow(tmp_path)
    config["rules"]["stardist_segmentation"]["parameters"]["segmentation_input_folder"] = "DAPI"
    assert _reads_assignment(config)[1] == "nucleus"
    config["rules"]["stardist_segmentation"]["parameters"]["target"] = "cell"
    assert _reads_assignment(config)[1:] == ("cell", {**_reads_assignment(config)[2], "target_source": "config"})


def test_a18_unknown_keys_and_the_block_raise(tmp_path):
    config = workflow(tmp_path)
    config["rules"]["reads_assignment"]["parameters"]["tile_filter"] = False
    with pytest.raises(ValueError, match="unknown reads_assignment parameter keys"):
        _reads_assignment(config)
    with pytest.raises(ValueError, match="FOV.assign"):
        _reads_assignment(dict(workflow(tmp_path), assignment={"cells": "cell"}))


def test_a18_a_gene_mismatch_with_the_codebook_raises(tmp_path):
    with pytest.raises(ValueError, match=r"only in genes.csv \['E'\], only in the codebook \['F'\]"):
        run_rule(tmp_path, label_fixture(), codebook=GENES[:4] + (("F", "TTTT"),))


# --- A18: raw.h5ad and reads_assignment.csv against the frozen legacy helper ---------------------------

@pytest.mark.parametrize("floats", [False, True])
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("dimensions", ["3d", "2d"])
def test_a18_the_legacy_outputs_equal_the_frozen_helper(tmp_path, dimensions, expand, floats):
    """X, the legacy obs columns and reads_assignment.csv's legacy columns equal the legacy values."""
    labels = label_fixture() if dimensions == "3d" else label_fixture().max(axis=0)
    adata, reads, reads_csv = run_rule(tmp_path, labels, floats=floats, expand=expand)
    legacy_csv = write_reads(tmp_path / "legacy.csv", KEEP)  # the legacy script raises on float coordinates
    expected, counts, meta = legacy_reads_assignment(labels, legacy_csv,
                                                     tmp_path / "output/data/out/documents/genes.csv", expand=expand)
    assert adata.X.dtype == np.float64 and np.array_equal(adata.X, counts)
    assert adata.var_names.tolist() == [g for g, _ in GENES]
    obs = adata.obs.reset_index(drop=True)
    assert list(obs.columns) == LEGACY_OBS[dimensions] + NEW_OBS
    # anndata stores the string columns as categoricals, the legacy file's included.
    legacy_obs = obs[LEGACY_OBS[dimensions]].astype({"sample": str, "fov_id": str})
    assert_frame_equal(legacy_obs, meta[LEGACY_OBS[dimensions]].astype({"sample": str, "fov_id": str}),
                       check_dtype=False)
    assert obs.volume.dtype == np.float64 and obs.seg_label.dtype == np.int64
    assert list(reads.columns) == LEGACY_READS + NEW_READS
    observed = reads[LEGACY_READS]
    if floats:
        assert (observed[["x", "y", "z"]].dtypes == np.float64).all()
        observed = observed.astype({c: np.int64 for c in ("x", "y", "z", "global_x", "global_y", "global_z")})
    assert_frame_equal(observed, roundtrip(expected))
    assert set(reads.assignment_status) <= {"assigned", "unassigned"}
    assert reads.spot_id.tolist() == [f"csv:{i}" for i in expected.index]  # the legacy row order
    record = json.loads(adata.uns["assignment"])
    assert record["name"] == "reads_assignment" and record["workflow"]["target"] == "cell"
    assert record["config"]["legacy_pixel_expansion"] == expand and record["grid"]["check"] == "declared_checked"
    assert (tmp_path / "output/data/out/expr/Position001/assignment.png").stat().st_size > 0


def test_a18_volume_and_position_come_from_the_expanded_territory(tmp_path):
    adata, _, _ = run_rule(tmp_path, label_fixture(), expand=True)
    cell_3 = adata.obs.iloc[0]
    assert (cell_3.seg_label, cell_3.volume, cell_3.size_voxels, cell_3.expanded_size_voxels) == (3, 1866, 771, 1866)


# --- Intentional changes ----------------------------------------------------------------------------------

def test_cells_are_kept_when_every_molecule_is_in_the_background(tmp_path):
    adata, reads, _ = run_rule(tmp_path, label_fixture(), MOLECULES[10:12])
    assert adata.shape == (5, len(GENES)) and not adata.X.any()
    assert adata.obs.seg_label.tolist() == [3, 7, 12, 20, 25] and adata.obs.n_molecules.tolist() == [0] * 5
    assert reads.assignment_status.tolist() == ["unassigned", "unassigned"]


def test_molecules_off_the_grid_are_outside_grid(tmp_path):
    off_grid = (((5, 10, -1), "A"), ((5, 10, SHAPE_ZYX[2]), "A"))  # one-based x = 0 and x = 65
    adata, reads, _ = run_rule(tmp_path, label_fixture(), KEEP + off_grid)
    counts = json.loads(adata.uns["assignment"])["counts"]
    assert (counts["molecules"], counts["outside_grid"]) == (len(KEEP) + 2, 2)
    # Neither wraps to the far edge nor raises: both are rows of the CSV with status outside_grid.
    assert len(reads) == len(KEEP) + 2
    rows = reads[reads.assignment_status == "outside_grid"]
    assert rows.spot_id.tolist() == [f"csv:{len(KEEP)}", f"csv:{len(KEEP) + 1}"]
    assert rows.seg_label.tolist() == [0, 0] and rows.cell_id.isna().all()
    assert rows.x.tolist() == [-1, SHAPE_ZYX[2]] and rows.global_x.tolist() == [-1, SHAPE_ZYX[2]]
    # The rows of the other molecules are those of the run without the two.
    _, without, _ = run_rule(tmp_path / "without", label_fixture(), KEEP)
    assert_frame_equal(reads[reads.assignment_status != "outside_grid"], without)


def test_a_gene_outside_the_codebook_raises_before_assignment(tmp_path):
    with pytest.raises(ValueError, match=r"genes outside the gene list: \['Z'\]"):
        run_rule(tmp_path, label_fixture(), MOLECULES)
    assert not (tmp_path / "output/data/out/expr").exists()


# --- Compartment layers -----------------------------------------------------------------------------------

def test_compartment_layers_are_nan_where_not_available(tmp_path):
    """Row A10 on boxes with nuclei and the exclusion off: 0.0 is a measured zero, NaN is not available."""
    anndata = pytest.importorskip("anndata")
    result = assign_boxes(exclude_cells_without_nucleus=False)
    tile = pd.Series({"x": 0, "y": 0, "z": 0, "start_x_norm": 0, "end_x_norm": 32, "start_y_norm": 0,
                      "end_y_norm": 32})
    path = tmp_path / "raw.h5ad"
    written = _write_raw_h5ad(result, path, sample="sample1", fov_id="boxes", tile=tile)
    adata = anndata.read_h5ad(path)
    assert written.tolist() == adata.obs.seg_label.tolist() == list(range(1, 9))
    cells = result.cells.set_index(result.cells.cell_id.astype(int))
    for compartment in ("nucleus", "cytoplasm"):
        layer = adata.layers[compartment]
        assert layer.dtype == np.float64
        values, rows = result.matrix(compartment)
        assert rows.cell_id.astype(int).tolist() == [1, 2]
        assert np.array_equal(layer[:2], values) and (layer[:2] == 0).any()
        assert np.isnan(layer[2:]).all()
    assert np.array_equal(adata.layers["nucleus"][:2] + adata.layers["cytoplasm"][:2], adata.X[:2])
    for column in ("compartments", "correspondence"):
        assert adata.obs[column].astype(str).tolist() == cells[column].astype(str).tolist()
    assert cells.compartments.tolist()[2:] == ["withheld"] * 4 + ["no_nucleus"] * 2


# --- The Python-only block and the static schema ------------------------------------------------------

BLOCK = yaml.safe_load("""
assignment:
  cells: cell              # a segmentation run name
  nuclei: nucleus          # a segmentation run name, or null
  population: final        # final (accepted reads) or called
  expansion: null          # or {distance: 0.78, unit: um, mode: planar}
  correspondence: {match_fraction: 0.5, outside_tolerance: 0.0}
  exclude_cells_without_nucleus: null   # null: on when nuclei is set
""")


def test_the_block_of_the_contract_gives_the_assign_call():
    config, call = _assignment_block({"backend": "python", **BLOCK})
    assert config == AssignmentConfig(correspondence=CorrespondenceConfig(0.5, 0.0))
    assert call == {"cells": "cell", "nuclei": "nucleus", "population": "final"}
    block = dict(BLOCK["assignment"], expansion={"distance": 0.78, "unit": "um", "mode": "planar"}, name="main",
                 checkpoints={"table_format": "parquet"}, exclude_cells_without_nucleus=False)
    config, call = _assignment_block({"backend": "python", "assignment": block})
    assert config.expansion == ExpandLabelsConfig(0.78, "um", "planar") and not config.exclude_cells_without_nucleus
    assert call["name"] == "main" and call["checkpoints"] == CheckpointConfig(table_format="parquet")


@pytest.mark.parametrize("change, error, match", [
    ({"backend": "matlab"}, ValueError, "Python-only"),
    ({"tile_filter": True}, ValueError, "unknown assignment keys"),
    ({"correspondence": {"match_fraction": 0.5, "tolerance": 0}}, ValueError, "unknown assignment.correspondence"),
    ({"expansion": {"distance": 1}}, TypeError, "unit"),
])
def test_invalid_assignment_blocks_raise(change, error, match):
    config = {"backend": change.get("backend", "python"),
              "assignment": dict(BLOCK["assignment"], **{k: v for k, v in change.items() if k != "backend"})}
    with pytest.raises(error, match=match):
        _assignment_block(config)


def test_schema_assignment_keys_equal_the_config_fields_and_the_assign_arguments():
    declared = set(SCHEMA["$defs"]["assignment_block"]["properties"])
    arguments = set(inspect.signature(FOV.assign).parameters) - {"self", "config"}
    assert declared == {f.name for f in fields(AssignmentConfig) if f.init} | arguments


def test_the_assignment_block_validates_on_the_python_backend_only():
    config = yaml.safe_load((ROOT / "docs/examples/workflow-full.yaml").read_text())
    config.update(backend="python", **BLOCK)
    jsonschema.validate(config, SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(config, backend="matlab"), SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(config, assignment=dict(BLOCK["assignment"], tile_filter=True)), SCHEMA)
