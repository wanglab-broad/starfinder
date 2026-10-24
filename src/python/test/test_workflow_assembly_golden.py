"""Golden baseline for the current sample assembly (W-326, §2.10).

Pins, on one hand-built sample generated in the test (three FOVs of 8×48×64 voxels on a
2 × 2 grid whose fourth cell is blank, the transforms of a BigStitcher ``dataset.xml`` for
it, and per FOV a ZYX label image and a one-based goodSpots CSV; nothing outside the test
is read and Fiji never runs):

(a) the tile offsets and non-overlap rectangles of ``workflow/scripts/create_tile_config.py``
    for the sample, through ``legacy_tile_config`` of the W-325 golden test; the tiles are
    not square, so ``img_col`` stands for the Y extent;
(b) the legacy per-FOV assembly of ``workflow/scripts/reads_assignment.py`` at 6b384cd
    (blob 26a7eab): global coordinates, the overlap filter of cells and molecules and the
    empty branch, through ``legacy_fov_assembly``, which follows the script's lines 78-80,
    111-124 and 141-165 with the cited lines (``SCRIPT_LINES``) on the output of
    ``legacy_assignment`` of the §2.9 golden test;
(c) today's workflow adapter, ``_run_reads_assignment`` through a stub ``snakemake`` object,
    whose ``raw.h5ad`` and ``reads_assignment.csv`` carry the same legacy values as (b)
    except where (b) takes the empty branch, which the adapter no longer has;
(d) the two sample-level scripts on the adapter's files: ``create_sample_reads_assignment.py``
    unchanged, and ``create_sample_h5ad.py`` unchanged with a stand-in ``scanpy`` module
    whose ``read_h5ad`` is ``anndata.read_h5ad`` (the function scanpy re-exports; scanpy
    is not in the lock) and pandas' ``future.infer_string`` off, the pandas 2 behavior it
    was written for: under pandas 3.0.0 it fails at ``write_h5ad``, which a test pins.

The scene is cut from one global label volume at the integer offsets of (a), so every FOV
sees the same cells where tiles overlap; each FOV numbers its own labels 1, 2, … in the
order of the scene's cells. ``reads_assignment.py`` at 6b384cd is no longer in the tree
(W-317 made it an adapter call), so its cited lines cannot be checked against the file;
they are checked against the blob in the W-326 notes. The values were produced in the
locked project environment (Python 3.12.12, pandas 3.0.0, NumPy 2.2.6, scikit-image
0.26.0, anndata 0.12.6) and are identical over repeated single-thread runs; see
docs/assembly-baseline.md. When §2.10 replaces these steps, the helpers stay as the frozen
legacy reference, and docs/assembly-contract.md names what the assemble entry must
reproduce and what it changes on purpose.
"""
import runpy
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile

from starfinder._snakemake import _run_reads_assignment

from .coordination_helpers import write_molecules_csv
from .test_assignment_golden import GENES, legacy_assignment, write_genes
from .test_workflow_stitching_golden import _affine, legacy_tile_config

pytestmark = [pytest.mark.workflow, pytest.mark.dataset, pytest.mark.golden]

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "workflow" / "scripts"

# ---------------------------------------------------------------------------------------
# The placement: a 2 × 2 grid scanned down each column, the cell (row 1, column 1) blank.
TILE_ZYX = (8, 48, 64)            # one FOV: Z, Y (img_row), X (img_col); not square
IMG_ROW, IMG_COL = TILE_ZYX[1], TILE_ZYX[2]
GRID_CSV = (",col,row,id,grid\n"
            "0,0,0,1,tile_0_0\n1,0,1,2,tile_1_0\n2,1,0,3,tile_0_1\n3,1,1,0,tile_1_1\n")
FOVS = {1: "Position001", 2: "Position002", 3: "Position003"}   # grid id -> FOV id
STEP_X, STEP_Y = 56.0, 36.0       # Translation to Regular Grid: 8 px overlap in x, 12 px in y
# setup (= grid row j) -> Stitching Transform translation (x, y, z); setup 3 is the blank.
# Setup 1's y sum 37.4 casts to 37; setup 2's x sum 55.4 casts to 55 and its y −0.3 to 0.
STITCHING = {0: (0.0, 0.0, 0.0), 1: (0.7, 1.4, 1.3), 2: (-0.6, -0.3, -0.6), 3: (0.0, 0.0, 0.0)}
Z_CALIBRATION = 1.7894736842105263


def dataset_xml(stitching=STITCHING):
    """ViewRegistrations of a BigStitcher dataset.xml for the 2 × 2 grid, as in the W-325 test."""
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', '<SpimData version="0.2">', '  <ViewRegistrations>']
    for setup, (sx, sy, sz) in stitching.items():
        col, row = divmod(setup, 2)
        parts.append(f'    <ViewRegistration timepoint="0" setup="{setup}">')
        for name, affine in (("Stitching Transform", _affine(sx, sy, sz)),
                             ("Translation to Regular Grid", _affine(STEP_X * col, STEP_Y * row, 0.0)),
                             ("calibration", _affine(0.0, 0.0, 0.0, Z_CALIBRATION))):
            parts += ['      <ViewTransform type="affine">', f'        <Name>{name}</Name>',
                      f'        <affine>{affine}</affine>', '      </ViewTransform>']
        parts.append('    </ViewRegistration>')
    parts += ['  </ViewRegistrations>', '</SpimData>']
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------------------
# The scene, in global voxels (the frame of the cast offsets): scene cell -> (z, y, x)
# centre and (z, y, x) semi-axes. Global rectangles (pinned below): FOV1 x [0, 60) ×
# y [0, 51), FOV2 x [0, 64) × y [43, 101), FOV3 x [60, 119) × y [0, 64).
SCENE_ZYX = (9, 85, 119)
CELLS = {
    1: ((4, 14, 16), (2, 5, 5)),     # FOV1 only
    2: ((4, 22, 58), (2, 4, 4)),     # FOV1 whole, FOV3 a part; centroid x 58: FOV1 keeps it
    3: ((4, 34, 60), (2, 2, 2)),     # whole in FOV1 and FOV3; centroid x 60.0, on the edge: FOV3 keeps it
    4: ((4, 45, 20), (2, 2, 4)),     # whole in FOV1 and FOV2; centroid y 45 in the band [43, 51): both keep it
    5: ((4, 40, 40), (2, 2, 3)),     # whole in FOV1 and FOV2; centroid y 40: FOV1 keeps it, FOV2 drops it
    6: ((5, 64, 30), (2, 5, 5)),     # FOV2 only
    7: ((3, 20, 92), (2, 5, 6)),     # FOV3 only
    8: ((4, 46, 100), (2, 4, 4)),    # FOV3, cut by its lower edge y = 48 (blank below): a partial cell
    9: ((4, 8, 55), (2, 3, 10)),     # x 45-65: past FOV1's edge x = 64; each FOV sees a part
}
# Global (z, y, x) and gene of each molecule; every FOV that images it detects it.
MOLECULES = (
    ((4, 14, 16), "A"), ((3, 12, 18), "B"),   # cell 1
    ((4, 22, 58), "C"), ((4, 22, 61), "A"),   # cell 2; x 61 also lies in FOV3's part of it
    ((4, 34, 60), "D"), ((4, 33, 59), "D"),   # cell 3
    ((4, 45, 20), "B"), ((4, 45, 23), "E"),   # cell 4
    ((4, 40, 40), "C"),                       # cell 5
    ((5, 64, 30), "A"), ((5, 66, 28), "B"),   # cell 6
    ((3, 20, 92), "E"), ((3, 18, 95), "C"),   # cell 7
    ((4, 46, 100), "D"),                      # cell 8
    ((4, 8, 48), "E"), ((4, 8, 64), "A"),     # cell 9; x 64 is outside FOV1
    ((2, 5, 40), "A"),                        # background, FOV1 only
    ((2, 28, 57), "B"),                       # background, x 57 in both FOV1 and FOV3
    ((2, 28, 62), "C"),                       # background, x 62 in both FOV1 and FOV3
    ((6, 45, 50), "D"),                       # background, y 45 in both FOV1 and FOV2
    ((6, 39, 10), "E"),                       # background, y 39 in both FOV1 and FOV2
    ((2, 75, 50), "A"),                       # background, FOV2 only
    ((6, 40, 110), "B"),                      # background, FOV3 only
    ((7, 14, 16), "C"),                       # above cell 1: background in 3D, in cell 1 in 2D
)
FOV_ID_PATTERN = "Position{i:03d}"


def scene_labels(cells=CELLS):
    """The uint16 global label volume of the cells (scene cell ids)."""
    z, y, x = np.ogrid[:SCENE_ZYX[0], :SCENE_ZYX[1], :SCENE_ZYX[2]]
    labels = np.zeros(SCENE_ZYX, np.uint16)
    for value, ((cz, cy, cx), (rz, ry, rx)) in cells.items():
        labels[((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1] = value
    return labels


def tile_frame(folder, stitching=STITCHING):
    """The tile_config frame of the sample: legacy_tile_config on the fixture's dataset.xml and grid.csv."""
    folder.mkdir(parents=True, exist_ok=True)
    xml, grid = folder / "dataset.xml", folder / "grid.csv"
    xml.write_text(dataset_xml(stitching))
    grid.write_text(GRID_CSV)
    frame, _ = legacy_tile_config(xml, grid, IMG_COL, IMG_ROW)
    return frame


def offsets(frame):
    """FOV number -> integer (z, y, x) offset of its tile_config row."""
    return {int(r.id): (int(r.z), int(r.y), int(r.x)) for r in frame.itertuples() if r.id != 0}


def fov_inputs(frame, cells=CELLS, molecules=MOLECULES):
    """FOV number -> (ZYX label image, scene ids in local label order, local molecules ((z, y, x), gene)).

    Each FOV is cut from the scene at its tile_config offset and numbers the scene cells it
    sees 1, 2, … in scene order; it detects every molecule inside its tile.
    """
    scene = scene_labels(cells)
    out = {}
    for number, (oz, oy, ox) in offsets(frame).items():
        crop = scene[oz:oz + TILE_ZYX[0], oy:oy + TILE_ZYX[1], ox:ox + TILE_ZYX[2]]
        assert crop.shape == TILE_ZYX
        present = [int(v) for v in np.unique(crop) if v]
        local = np.zeros(TILE_ZYX, np.uint16)
        for k, value in enumerate(present, start=1):
            local[crop == value] = k
        inside = tuple(((z - oz, y - oy, x - ox), gene) for (z, y, x), gene in molecules
                       if 0 <= z - oz < TILE_ZYX[0] and 0 <= y - oy < TILE_ZYX[1] and 0 <= x - ox < TILE_ZYX[2])
        out[number] = (local, present, inside)
    return out


def write_reads(path, molecules):
    """The goodSpots CSV: one-based integer x, y, z and the gene (header only when empty)."""
    rows = [(x + 1, y + 1, z + 1, gene) for (z, y, x), gene in molecules]
    pd.DataFrame(rows, columns=["x", "y", "z", "gene"]).to_csv(path, index=False)
    return path


def legacy_fov_assembly(label_img, reads_csv, genes_csv, record, *, sample="sample1", fov_id="Position001"):
    """workflow/scripts/reads_assignment.py at 6b384cd: the FOV-local steps, then the §2.10 lines.

    legacy_assignment (test_assignment_golden) runs the FOV-local lines without expansion;
    this helper adds the global coordinates (:78-80, :141-149), the empty branch (:111-124)
    and the overlap filter (:155-165), with record the FOV's tile_config row (:39-42).
    Returns the cell metadata and the molecule table the script writes (raw.h5ad's obs and
    reads_assignment.csv), and whether the empty branch was taken.
    """
    reads_df, cell_by_gene, meta = legacy_assignment(label_img, reads_csv, genes_csv, expand=False,
                                                     sample=sample, fov_id=fov_id)
    reads_df['global_x'] = reads_df['x'] + record['x']                                    # :78
    reads_df['global_y'] = reads_df['y'] + record['y']                                    # :79
    reads_df['global_z'] = reads_df['z'] + record['z']                                    # :80
    three_d = len(label_img.shape) == 3
    if len(meta) == 0:                           # the empty branch (:111): written unfiltered (:124)
        for axis in ('x', 'y', 'z') if three_d else ('x', 'y'):
            meta[f'global_{axis}'] = 0                                                    # :117-121
        return meta, reads_df, True
    if three_d:                                                                            # :142
        meta['global_x'] = meta['fov_x'] + record['x']                                    # :143-145
        meta['global_y'] = meta['fov_y'] + record['y']
        meta['global_z'] = meta['fov_z'] + record['z']
    else:
        meta['global_x'] = meta['fov_x'] + record['x']                                    # :147-149
        meta['global_y'] = meta['fov_y'] + record['y']
    meta = meta[meta['fov_x'].isin(range(record['start_x_norm'], record['end_x_norm']))]  # :156
    meta = meta[meta['fov_y'].isin(range(record['start_y_norm'], record['end_y_norm']))]  # :157
    meta = meta.reset_index(drop=True)                                                     # :158
    seg_label_left = list(meta['seg_label'].unique())                                      # :160
    cells = reads_df.loc[reads_df['seg_label'].isin(seg_label_left), :].copy()            # :161
    background = reads_df.loc[reads_df['seg_label'] == 0, :].copy()                       # :162
    background = background.loc[background['x'].isin(range(record['start_x_norm'], record['end_x_norm'])), ]  # :163
    background = background.loc[background['y'].isin(range(record['start_y_norm'], record['end_y_norm'])), ]  # :164
    return meta, pd.concat([cells, background]), False                                     # :165


# The lines of reads_assignment.py at 6b384cd (blob 26a7eab) that legacy_fov_assembly follows.
SCRIPT_LINES = {
    40: ("tile_config_df = pd.read_csv(os.path.join(output_path, f'tile_config_{current_sample}.csv'), "
         "index_col=0)"),
    41: "current_record = tile_config_df[tile_config_df['id'] == int(current_id)]",
    78: "reads_df['global_x'] = reads_df['x'] + current_record['x']",
    79: "reads_df['global_y'] = reads_df['y'] + current_record['y']",
    80: "reads_df['global_z'] = reads_df['z'] + current_record['z']",
    111: ("if total_cells == 0 or (len(np.unique(reads_assignment)) == 1 and "
          "np.unique(reads_assignment)[0] == 0):"),
    124: 'reads_df.to_csv(os.path.join(current_expr_path, "reads_assignment.csv"), index=False)',
    141: "cell_locs = np.array(cell_locs).astype(int)",
    143: ("global_cell_locs = cell_locs + np.array([current_record['z'], current_record['y'], "
          "current_record['x']])"),
    147: "global_cell_locs = cell_locs + np.array([current_record['y'], current_record['x']])",
    156: ("adata = adata[adata.obs['fov_x'].isin(range(current_record['start_x_norm'], "
          "current_record['end_x_norm'])), ]"),
    157: ("adata = adata[adata.obs['fov_y'].isin(range(current_record['start_y_norm'], "
          "current_record['end_y_norm'])), ]"),
    161: "reads_df_cells = reads_df.loc[reads_df['seg_label'].isin(seg_label_left), :].copy()",
    162: "reads_df_background = reads_df.loc[reads_df['seg_label'] == 0, :].copy()",
    163: ("reads_df_background = reads_df_background.loc[reads_df_background['x'].isin("
          "range(current_record['start_x_norm'], current_record['end_x_norm'])), ]"),
    165: "reads_df_filtered = pd.concat([reads_df_cells, reads_df_background])",
}


class _Wildcards(dict):
    """Snakemake wildcards: the scripts read them both as attributes and by key."""
    __getattr__ = dict.__getitem__


def _config(root, *, legacy=True):
    """The workflow configuration of the sample; ``legacy`` names the legacy route (backend matlab), whose
    rsf_single_fov writes the one-based goodSpots CSVs, else the Python backend and fovs/<fov>/molecules.csv."""
    return {"backend": "matlab" if legacy else "python", "dataset_id": "data", "sample_id": "sample", "output_id": "out",
            "root_input_path": str(root / "input"), "root_output_path": str(root / "output"),
            "fov_id_pattern": FOV_ID_PATTERN, "img_z": TILE_ZYX[0], "img_row": IMG_ROW, "img_col": IMG_COL,
            "voxel_size_z": 0.35, "voxel_size_xy": 0.2,
            "rules": {"stardist_segmentation": {"parameters": {"segmentation_input_folder": "overlay",
                                                               "expand_labels": False}},
                      "reads_assignment": {"parameters": {"expand_labels": False, "dilation_distance": 0}}}}


def build_sample(root, dimensions, *, cells=CELLS, placement=STITCHING, molecules=None, legacy=True):
    """The sample's output folder: annotation, genes, codebook, tile_config and per FOV labels and molecules: the
    legacy route's goodSpots CSV, or with ``legacy`` False fovs/<fov>/molecules.csv (row ``i`` the spot ``m<i>``).

    The FOVs are always cut at the offsets of STITCHING; placement gives the dataset.xml from
    which tile_config is computed, so a changed placement moves the rectangles and global
    coordinates but not the images. molecules maps a FOV number to the local molecules that
    replace its own. Returns the tile_config frame, the per-FOV inputs and the output folder.
    """
    out = root / "output" / "data" / "out"
    for folder in ("documents", "output", "signal", "images/DAPI", "images/stardist_segmentation"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    (out / "documents" / "sample-annotation.csv").write_text("sample_id,fov_start,fov_end\nsample1,1,3\n")
    write_genes(out / "documents" / "genes.csv")
    codebook = root / "input" / "data" / "sample" / "genes.csv"
    codebook.parent.mkdir(parents=True, exist_ok=True)
    write_genes(codebook)
    frame = tile_frame(root / "placement", placement)
    frame.to_csv(out / "output" / "tile_config_sample1.csv")
    inputs = fov_inputs(tile_frame(root / "cut"), cells)
    for number, (labels, _, local) in inputs.items():
        fov_id = FOVS[number]
        image = labels if dimensions == "3d" else labels.max(axis=0)
        tifffile.imwrite(out / "images" / "stardist_segmentation" / f"{fov_id}.tif", image)
        tifffile.imwrite(out / "images" / "DAPI" / f"{fov_id}.tif", np.zeros(image.shape, np.uint8))
        rows = (molecules or {}).get(number, local)
        if legacy:
            write_reads(out / "signal" / f"{fov_id}_goodSpots.csv", rows)
        else:
            write_molecules_csv(out / "fovs" / fov_id / "molecules.csv", f'["data","sample","{fov_id}",null]',
                                [p for p, _ in rows], [g for _, g in rows])
    return frame, inputs, out


def run_adapter(root, out, fov_id):
    """reads_assignment for one FOV through a stub snakemake object; returns raw.h5ad and the CSV."""
    anndata = pytest.importorskip("anndata")
    inputs = [out / "documents" / "sample-annotation.csv", out / "images" / "DAPI" / f"{fov_id}.tif",
              out / "images" / "stardist_segmentation" / f"{fov_id}.tif", out / "signal" / f"{fov_id}_goodSpots.csv",
              out / "documents" / "genes.csv", out / "output" / "tile_config_sample1.csv"]
    outputs = [out / "expr" / fov_id / "raw.h5ad", out / "expr" / fov_id / "reads_assignment.csv"]
    _run_reads_assignment(SimpleNamespace(input=[str(p) for p in inputs], output=[str(p) for p in outputs],
                                          config=_config(root), wildcards=SimpleNamespace(fovID=fov_id)))
    return anndata.read_h5ad(outputs[0]), pd.read_csv(outputs[1])


def run_legacy(out, frame, dimensions):
    """legacy_fov_assembly for every FOV of the sample folder: FOV number -> (meta, reads, empty)."""
    results = {}
    for number, fov_id in FOVS.items():
        labels = tifffile.imread(out / "images" / "stardist_segmentation" / f"{fov_id}.tif")
        record = frame[frame['id'] == number].iloc[0]
        results[number] = legacy_fov_assembly(labels, out / "signal" / f"{fov_id}_goodSpots.csv",
                                               out / "documents" / "genes.csv", record, fov_id=fov_id)
    return results


def run_sample_script(root, out, script, monkeypatch, *, pandas_2_strings=True):
    """One sample-level script, unchanged, through a stub snakemake object; returns its output path.

    create_sample_h5ad.py imports scanpy, which the lock does not hold: a stand-in module
    gives it scanpy.read_h5ad, which is anndata.read_h5ad re-exported. Under pandas 3.0.0 the
    script's string concatenation gives an Arrow string column that anndata 0.12.6 cannot
    write, so by default the script runs with future.infer_string off, the pandas 2
    behavior it was written for; nothing else is changed.
    """
    anndata = pytest.importorskip("anndata")
    scanpy = types.ModuleType("scanpy")
    scanpy.read_h5ad = anndata.read_h5ad
    monkeypatch.setitem(sys.modules, "scanpy", scanpy)
    name = {"create_sample_reads_assignment.py": "sample1_reads_assignment.csv",
            "create_sample_h5ad.py": "sample1_raw.h5ad"}[script]
    path = out / "expr" / name
    snakemake = SimpleNamespace(input=[str(out / "documents" / "sample-annotation.csv")], output=[str(path)],
                                config=_config(root), wildcards=_Wildcards(sample="sample1"))
    with pd.option_context("future.infer_string", not pandas_2_strings):
        runpy.run_path(str(SCRIPTS / script), init_globals={"snakemake": snakemake})
    return path


def run_sample_scripts(root, out, monkeypatch):
    """The sample CSV (create_sample_reads_assignment.py) and the sample AnnData (create_sample_h5ad.py)."""
    anndata = pytest.importorskip("anndata")
    reads = run_sample_script(root, out, "create_sample_reads_assignment.py", monkeypatch)
    adata = run_sample_script(root, out, "create_sample_h5ad.py", monkeypatch)
    return pd.read_csv(reads, index_col=0), anndata.read_h5ad(adata)


def cell_rows(meta, three_d):
    """(seg_label, fov_x, fov_y[, fov_z], global_x, global_y[, global_z]) per row, as ints."""
    columns = (["seg_label", "fov_x", "fov_y", "fov_z", "global_x", "global_y", "global_z"] if three_d
               else ["seg_label", "fov_x", "fov_y", "global_x", "global_y"])
    return [tuple(int(v) for v in row) for row in meta[columns].itertuples(index=False)]


def read_rows(reads):
    """(global_x, global_y, global_z, gene, seg_label) per molecule row, in file order."""
    return [(int(r.global_x), int(r.global_y), int(r.global_z), r.gene, int(r.seg_label))
            for r in reads.itertuples(index=False)]


# ---------------------------------------------------------------------------------------
# Pinned values.
# tile_config per grid row j: x, y, z, start_x, start_y, end_x, end_y,
# start_x_norm, start_y_norm, end_x_norm, end_y_norm
TILE_CONFIG = [
    (0, 0, 0, 0, 0, 60, 51, 0, 0, 60, 51),
    (0, 37, 1, 0, 43, 64, 101, 0, 6, 64, 64),
    (55, 0, 0, 60, 0, 119, 64, 5, 0, 64, 64),
    (56, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0),          # the blank: offsets kept, rectangle zero
]
# FOV number -> the scene cells it sees, in local label order (local label k is the k-th)
SCENE_CELLS = {1: [1, 2, 3, 4, 5, 9], 2: [4, 5, 6], 3: [2, 3, 7, 8, 9]}
# dimensions -> FOV number -> kept cells: (seg_label, fov_x, fov_y[, fov_z], global_x, global_y[, global_z])
KEPT_CELLS = {
    "3d": {1: [(1, 16, 14, 4, 16, 14, 4), (2, 58, 22, 4, 58, 22, 4), (4, 20, 45, 4, 20, 45, 4),
               (5, 40, 40, 4, 40, 40, 4), (6, 54, 8, 4, 54, 8, 4)],
           2: [(1, 20, 8, 3, 20, 45, 4), (3, 30, 27, 4, 30, 64, 5)],
           3: [(2, 5, 34, 4, 60, 34, 4), (3, 37, 20, 3, 92, 20, 3), (4, 45, 45, 4, 100, 45, 4)]},
    "2d": {1: [(1, 16, 14, 16, 14), (2, 58, 22, 58, 22), (4, 20, 45, 20, 45), (5, 40, 40, 40, 40),
               (6, 54, 8, 54, 8)],
           2: [(1, 20, 8, 20, 45), (3, 30, 27, 30, 64)],
           3: [(2, 5, 34, 60, 34), (3, 37, 20, 92, 20), (4, 45, 45, 100, 45)]},
}
# dimensions -> FOV number -> kept molecules in file order: (global_x, global_y, global_z, gene, seg_label)
_FOV2 = [(20, 45, 4, "B", 1), (23, 45, 4, "E", 1), (30, 64, 5, "A", 3), (28, 66, 5, "B", 3),
         (50, 45, 6, "D", 0), (50, 75, 2, "A", 0)]
_FOV3 = [(60, 34, 4, "D", 2), (59, 33, 4, "D", 2), (92, 20, 3, "E", 3), (95, 18, 3, "C", 3),
         (100, 46, 4, "D", 4), (62, 28, 2, "C", 0), (110, 40, 6, "B", 0)]
KEPT_READS = {
    "3d": {1: [(16, 14, 4, "A", 1), (18, 12, 3, "B", 1), (58, 22, 4, "C", 2), (61, 22, 4, "A", 2),
               (20, 45, 4, "B", 4), (23, 45, 4, "E", 4), (40, 40, 4, "C", 5), (48, 8, 4, "E", 6),
               (40, 5, 2, "A", 0), (57, 28, 2, "B", 0), (50, 45, 6, "D", 0), (10, 39, 6, "E", 0),
               (16, 14, 7, "C", 0)],
           2: _FOV2, 3: _FOV3},
    "2d": {1: [(16, 14, 4, "A", 1), (18, 12, 3, "B", 1), (58, 22, 4, "C", 2), (61, 22, 4, "A", 2),
               (20, 45, 4, "B", 4), (23, 45, 4, "E", 4), (40, 40, 4, "C", 5), (48, 8, 4, "E", 6),
               (16, 14, 7, "C", 1), (40, 5, 2, "A", 0), (57, 28, 2, "B", 0), (50, 45, 6, "D", 0),
               (10, 39, 6, "E", 0)],
           2: _FOV2, 3: _FOV3},
}
# FOV number -> the scene cells it keeps (both dimensions): cell 4 twice, every cell at least once
KEPT_SCENE_CELLS = {1: [1, 2, 4, 5, 9], 2: [4, 6], 3: [3, 7, 8]}
# Scene molecule (MOLECULES order) -> how many FOVs keep it (both dimensions): the two of cell 4
# and the background one at y 45 twice; cell 9's at x 64 (outside FOV1, in FOV3's dropped part) never
RETAINED = [1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1]
# The empty branch: FOV3 with its three background molecules only (local x 2, 7 and 55)
EMPTY_LEGACY_READS = [(57, 28, 2, "B", 0), (62, 28, 2, "C", 0), (110, 40, 6, "B", 0)]   # unfiltered
EMPTY_ADAPTER_READS = [(62, 28, 2, "C", 0), (110, 40, 6, "B", 0)]                       # x 2 < 5 dropped
# The sample tables: the sample H5AD's obs index, and the sample CSV's rows per unique_index
SAMPLE_OBS = ["sample1_Position001_1", "sample1_Position001_2", "sample1_Position001_4", "sample1_Position001_5",
              "sample1_Position001_6", "sample1_Position002_1", "sample1_Position002_3", "sample1_Position003_2",
              "sample1_Position003_3", "sample1_Position003_4"]
SAMPLE_READS = {
    "3d": {"sample1_Position001_0": 5, "sample1_Position001_1": 2, "sample1_Position001_2": 2,
           "sample1_Position001_4": 2, "sample1_Position001_5": 1, "sample1_Position001_6": 1,
           "sample1_Position002_0": 2, "sample1_Position002_1": 2, "sample1_Position002_3": 2,
           "sample1_Position003_0": 2, "sample1_Position003_2": 2, "sample1_Position003_3": 2,
           "sample1_Position003_4": 1},
}
SAMPLE_READS["2d"] = dict(SAMPLE_READS["3d"], sample1_Position001_0=4, sample1_Position001_1=3)


def scene_index(rows):
    """Scene molecule index of each kept molecule row (global position and gene)."""
    where = {((z, y, x), gene): i for i, ((z, y, x), gene) in enumerate(MOLECULES)}
    return [where[((gz, gy, gx), gene)] for gx, gy, gz, gene, _ in rows]


@pytest.fixture(scope="module", params=["3d", "2d"])
def sample(request, tmp_path_factory):
    """The fixture sample of one dimension: the legacy helper and the adapter on every FOV."""
    dimensions = request.param
    root = tmp_path_factory.mktemp(f"assembly_{dimensions}")
    frame, inputs, out = build_sample(root, dimensions)
    adapter = {number: run_adapter(root, out, fov_id) for number, fov_id in FOVS.items()}
    return SimpleNamespace(dimensions=dimensions, root=root, out=out, frame=frame, inputs=inputs,
                           legacy=run_legacy(out, frame, dimensions), adapter=adapter)


def test_tile_offsets_and_rectangles(tmp_path):
    """create_tile_config.py's offsets (cast, then shifted) and rectangles for the fixture placement."""
    frame = tile_frame(tmp_path)
    columns = ["x", "y", "z", "start_x", "start_y", "end_x", "end_y",
               "start_x_norm", "start_y_norm", "end_x_norm", "end_y_norm"]
    assert [tuple(int(v) for v in row) for row in frame[columns].itertuples(index=False)] == TILE_CONFIG
    assert offsets(frame) == {1: (0, 0, 0), 2: (1, 37, 0), 3: (0, 0, 55)}


def test_img_col_is_the_y_extent():
    """FOV1's lower edge uses img_col (64) where its height is img_row (48): 51 instead of 43.

    The band y in [43, 51) is then owned by FOV1 and by FOV2 (start 43), and both keep
    scene cell 4, whose centroid is at y 45.
    """
    y1, y2 = 0, 37
    assert int((y1 + IMG_COL - y2) / 2 + 0.5 + y2) == TILE_CONFIG[0][6] == 51
    assert int((y1 + IMG_ROW - y2) / 2 + 0.5 + y2) == 43
    assert int((y1 + IMG_ROW - y2) / 2 + 0.5) + y2 == TILE_CONFIG[1][4] == 43


def test_fixture(tmp_path):
    frame = tile_frame(tmp_path)
    inputs = fov_inputs(frame)
    assert {number: present for number, (_, present, _) in inputs.items()} == SCENE_CELLS
    assert {number: len(local) for number, (_, _, local) in inputs.items()} == {1: 16, 2: 8, 3: 11}
    for labels, present, _ in inputs.values():
        assert labels.dtype == np.uint16 and sorted(np.unique(labels).tolist()) == list(range(len(present) + 1))


def test_cells_and_their_global_coordinates(sample):
    """Kept cells: truncated centroid in the FOV's rectangle (XY only), global = truncated centroid + offset."""
    for number in FOVS:
        meta, _, empty = sample.legacy[number]
        assert not empty
        assert cell_rows(meta, sample.dimensions == "3d") == KEPT_CELLS[sample.dimensions][number]


def test_molecules_and_their_global_coordinates(sample):
    """Kept molecules: every molecule of a kept cell, then background molecules inside the rectangle."""
    for number in FOVS:
        _, reads, _ = sample.legacy[number]
        assert read_rows(reads) == KEPT_READS[sample.dimensions][number]


def test_the_adapter_writes_the_legacy_values(sample):
    """raw.h5ad and reads_assignment.csv of the workflow adapter carry the legacy helper's values."""
    for number in FOVS:
        meta, reads, _ = sample.legacy[number]
        adata, csv = sample.adapter[number]
        assert cell_rows(adata.obs, sample.dimensions == "3d") == cell_rows(meta, sample.dimensions == "3d")
        assert read_rows(csv) == read_rows(reads)
        assert np.array_equal(np.asarray(adata.X).sum(axis=1), adata.obs.n_molecules.to_numpy())
        assert set(csv.assignment_status) <= {"assigned", "unassigned"}


def test_duplicates_and_losses_across_fovs(sample):
    """Scene cell 4 is kept by FOV1 and FOV2; cell 9's molecule at x 64 is kept by no FOV."""
    kept = {}
    for number, (labels, present, _) in sample.inputs.items():
        meta, _, _ = sample.legacy[number]
        kept[number] = [present[int(label) - 1] for label in meta.seg_label]
    assert kept == KEPT_SCENE_CELLS
    counts = [0] * len(MOLECULES)
    for number in FOVS:
        for i in scene_index(read_rows(sample.legacy[number][1])):
            counts[i] += 1
    assert counts == RETAINED


@pytest.mark.parametrize("dimensions", ["3d", "2d"])
def test_the_empty_branch(tmp_path, dimensions):
    """FOV3 with background molecules only: the script keeps no cell and writes every molecule
    unfiltered; the adapter keeps the cells and filters the molecules."""
    frame = tile_frame(tmp_path / "cut")
    labels, _, local = fov_inputs(frame)[3]
    background = tuple(m for m in local if labels[m[0]] == 0)
    frame, _, out = build_sample(tmp_path, dimensions, molecules={3: background})
    meta, reads, empty = run_legacy(out, frame, dimensions)[3]
    assert empty and len(meta) == 0
    assert list(meta.columns)[-3 if dimensions == "3d" else -2:] == (
        ["global_x", "global_y", "global_z"] if dimensions == "3d" else ["global_x", "global_y"])
    assert read_rows(reads) == EMPTY_LEGACY_READS
    adata, csv = run_adapter(tmp_path, out, "Position003")
    assert cell_rows(adata.obs, dimensions == "3d") == KEPT_CELLS[dimensions][3]
    assert not np.asarray(adata.X).any()
    assert read_rows(csv) == EMPTY_ADAPTER_READS


def test_the_sample_tables(sample, monkeypatch):
    """The two sample-level scripts concatenate the per-FOV files; keys are sample_fov_id_seg_label."""
    reads, adata = run_sample_scripts(sample.root, sample.out, monkeypatch)
    assert adata.obs_names.tolist() == SAMPLE_OBS == adata.obs.unique_index.tolist()
    assert adata.var_names.tolist() == [gene for gene, _ in GENES]
    assert reads.unique_index.value_counts().sort_index().to_dict() == SAMPLE_READS[sample.dimensions]
    assert len(reads) == sum(len(KEPT_READS[sample.dimensions][n]) for n in FOVS)
    # Every background molecule of a FOV shares the key <sample>_<fov>_0.
    assert (reads.seg_label == 0).sum() == sum(n for k, n in SAMPLE_READS[sample.dimensions].items()
                                                if k.endswith("_0"))
    # The counts of the sample AnnData equal the kept cells' molecules in the sample CSV.
    in_cells = reads[reads.seg_label > 0].unique_index.value_counts()
    assert np.asarray(adata.X).sum(axis=1).tolist() == [float(in_cells[k]) for k in SAMPLE_OBS]
    # One scene cell, two observations: the same global position in two FOVs.
    axes = ["global_x", "global_y", "global_z"] if sample.dimensions == "3d" else ["global_x", "global_y"]
    position = adata.obs[axes].astype(int).apply(tuple, axis=1)
    repeated = adata.obs_names[position.duplicated(keep=False).to_numpy()].tolist()
    assert repeated == ["sample1_Position001_4", "sample1_Position002_1"]


def test_the_sample_h5ad_script_needs_pandas_2_strings(sample, monkeypatch):
    """Under pandas 3.0.0 the unique_index column is an Arrow string array, which anndata 0.12.6 cannot write."""
    from anndata._io.specs.registry import IORegistryError
    with pytest.raises(IORegistryError, match="ArrowStringArray"):
        run_sample_script(sample.root, sample.out, "create_sample_h5ad.py", monkeypatch, pandas_2_strings=False)


@pytest.mark.parametrize("dimensions", ["3d", "2d"])
def test_moving_one_centroid_across_an_edge_changes_the_pins(tmp_path, dimensions):
    """Scene cell 3 one voxel left (centroid x 59): FOV1 now keeps it and FOV3 drops it."""
    cells = dict(CELLS)
    cells[3] = ((4, 34, 59), (2, 2, 2))
    frame, _, out = build_sample(tmp_path, dimensions, cells=cells)
    legacy = run_legacy(out, frame, dimensions)
    three_d = dimensions == "3d"
    first, third = cell_rows(legacy[1][0], three_d), cell_rows(legacy[3][0], three_d)
    assert first != KEPT_CELLS[dimensions][1] and third != KEPT_CELLS[dimensions][3]
    assert [row[0] for row in first] == [1, 2, 3, 4, 5, 6] and [row[0] for row in third] == [3, 4]


@pytest.mark.parametrize("dimensions", ["3d", "2d"])
def test_changing_one_tile_offset_changes_the_pins(tmp_path, dimensions):
    """Setup 2's stitching x from −0.6 to 0.7: FOV3's offset becomes 56 and its global x moves by 1."""
    placement = dict(STITCHING)
    placement[2] = (0.7, -0.3, -0.6)
    frame, _, out = build_sample(tmp_path, dimensions, placement=placement)
    assert offsets(frame)[3] == (0, 0, 56)
    rows = cell_rows(run_legacy(out, frame, dimensions)[3][0], dimensions == "3d")
    assert rows != KEPT_CELLS[dimensions][3]
    width = 3 if dimensions == "3d" else 2
    assert [row[-width] for row in rows[:3]] == [row[-width] + 1 for row in KEPT_CELLS[dimensions][3]]
