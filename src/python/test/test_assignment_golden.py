"""Golden baseline for the current molecule-to-cell assignment (W-308, §2.9).

Pins, with exact SHA-256 digests, the FOV-local behavior of
``workflow/scripts/reads_assignment.py`` on one hand-built fixture: a 16×64×64 ``uint16``
label image with five cells (labels 3, 7, 12, 20 and 25, two of them 2 voxels apart),
its Z maximum as the 2D case, a molecule table of 19 rows written as a one-based
``x,y,z,gene`` CSV, and a headerless five-gene ``genes.csv``:

(a) the per-molecule label (``seg_label``), the cell-by-gene count matrix and the cell
    metadata (``volume``, ``fov_x``, ``fov_y``, ``fov_z``, ``seg_label``), in 3D and in 2D,
    with and without the script's per-plane label expansion;
(b) the empty branch: a FOV with no cell, and a FOV whose molecules all fall outside
    every cell, both of which give a count matrix with no row;
(c) legacy behaviors that §2.9 changes: a molecule at the one-based coordinate 0 reads
    the label at the far edge (negative index), one beyond the grid raises
    ``IndexError``, float coordinates (as ``starfinder.io.export_spots`` writes them)
    raise ``IndexError``, a gene absent from ``genes.csv`` keeps its label but is not
    counted, a FOV whose molecules all miss the cells loses every cell, and a planar
    expansion by 4 followed by one by 2 (segmentation, then assignment) is not one
    expansion by 6.

The script cannot run in the locked environment (it imports ``parse`` and ``anndata``,
neither of which is installed there), so ``legacy_assignment`` follows its lines, which
are cited, and a test checks that those lines are still in the script. The tile
configuration, the global coordinates, the overlap filter (lines 39-42, 78-80, 141-165),
the plots and the H5AD and CSV writing are §2.10 or output plumbing and are not pinned;
see docs/assignment-baseline.md.

The digests were produced with the locked project environment (NumPy 2.2.6,
scikit-image 0.26.0, pandas 3.0.0) and are bit-identical over repeated single-thread
runs. The §2.9 work may replace the body of ``legacy_assignment`` with calls to the
assignment entry; every pinned digest stays, except where docs/assignment-contract.md
names an edit.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from skimage.measure import regionprops
from skimage.segmentation import expand_labels

pytestmark = [pytest.mark.workflow, pytest.mark.golden]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "workflow" / "scripts" / "reads_assignment.py"
SHAPE_ZYX = (16, 64, 64)
DISTANCE = 4  # dilation_distance
# label value -> (z, y, x) centre and (z, y, x) semi-axes in voxels. Cells 3 and 7 lie
# 2 voxels apart, so their expansions meet; cell 25 touches the z = 0 plane.
CELLS = {3: ((6, 16, 16), (3, 8, 8)), 7: ((8, 16, 34), (3, 7, 7)),
         12: ((9, 44, 20), (4, 10, 9)), 20: ((10, 46, 50), (2, 6, 6)),
         25: ((0, 58, 58), (1, 3, 3))}
GENES = (("A", "AACG"), ("B", "ACGT"), ("C", "CGTA"), ("D", "GTAC"), ("E", "TACG"))
# Zero-based (z, y, x) and gene of each molecule; the CSV holds x + 1, y + 1, z + 1.
MOLECULES = (
    ((6, 16, 16), "A"), ((6, 12, 18), "B"), ((5, 20, 14), "A"),   # cell 3
    ((8, 16, 34), "C"), ((8, 18, 38), "C"),                       # cell 7
    ((9, 44, 20), "A"), ((9, 40, 24), "D"), ((11, 48, 16), "B"),  # cell 12
    ((10, 46, 50), "D"),                                          # cell 20
    ((0, 58, 58), "E"),                                           # cell 25
    ((2, 60, 4), "A"), ((14, 4, 60), "B"),                        # background
    ((6, 16, 26), "C"),   # 2 voxels right of cell 3, 3 left of cell 7 in its plane
    ((10, 46, 58), "D"),  # 2 voxels right of cell 20
    ((9, 44, 8), "B"),    # 3 voxels left of cell 12
    ((8, 14, 32), "Z"),   # inside cell 7; gene Z is not in genes.csv
    ((3, 30, 0), "A"),    # background, on the x = 0 face
    ((6, 16, 16), "B"),   # the position of the first molecule again
    ((13, 16, 16), "C"),  # above cell 3: background in 3D, inside cell 3 in 2D
)


def label_fixture():
    """The uint16 ZYX label image of CELLS."""
    z, y, x = np.ogrid[:SHAPE_ZYX[0], :SHAPE_ZYX[1], :SHAPE_ZYX[2]]
    labels = np.zeros(SHAPE_ZYX, np.uint16)
    for value, ((cz, cy, cx), (rz, ry, rx)) in CELLS.items():
        labels[((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1] = value
    return labels


def write_reads(path, molecules=MOLECULES):
    """The goodSpots CSV: one-based integer x, y, z and the gene."""
    rows = [(x + 1, y + 1, z + 1, gene) for (z, y, x), gene in molecules]
    pd.DataFrame(rows, columns=["x", "y", "z", "gene"]).to_csv(path, index=False)
    return path


def write_genes(path):
    """documents/genes.csv: headerless gene,barcode."""
    path.write_text("".join(f"{gene},{barcode}\n" for gene, barcode in GENES))
    return path


def digest(array):
    """SHA-256 over dtype, shape and C-order bytes."""
    array = np.ascontiguousarray(array)
    h = hashlib.sha256(f"{array.dtype.str}|{array.shape}|".encode())
    h.update(array.tobytes())
    return h.hexdigest()


def frame_digest(frame):
    """SHA-256 over the column names, their dtypes and the CSV text of the frame."""
    header = json.dumps([[str(c), str(t)] for c, t in frame.dtypes.items()])
    text = frame.to_csv(index=False, float_format="%.17g")
    return hashlib.sha256((header + "|" + text).encode()).hexdigest()


def legacy_assignment(label_img, reads_csv, genes_csv, *, expand, distance=DISTANCE,
                      sample="sample1", fov_id="Position001"):
    """workflow/scripts/reads_assignment.py at 6b384cd, FOV-local steps, line by line.

    label_img is the array the script reads with imread (line 50); reads_csv and genes_csv
    are snakemake.input[3] and [4]; expand and distance stand for the rule parameters
    expand_labels and dilation_distance. The tile record, the global coordinates and the
    overlap filter are left out (§2.10); the cell metadata keeps the FOV-local columns.
    Returns the reads table with seg_label, the count matrix and the cell metadata.
    """
    current_label_img = label_img.copy()
    if len(current_label_img.shape) == 3:                                                  # :52
        if expand:                                                                         # :55
            for z in range(current_label_img.shape[0]):                                    # :56
                current_label_img[z, :, :] = expand_labels(current_label_img[z, :, :],
                                                           distance=distance)              # :57
    else:
        if expand:                                                                         # :66
            current_label_img = expand_labels(current_label_img, distance=distance)        # :67
    reads_df = pd.read_csv(reads_csv)                                                      # :74
    reads_df['x'] = reads_df['x'] - 1                                                      # :75
    reads_df['y'] = reads_df['y'] - 1                                                      # :76
    reads_df['z'] = reads_df['z'] - 1                                                      # :77
    if reads_df.shape[0] != 0:                                                             # :82
        points = reads_df.loc[:, ["x", "y", "z"]].values                                   # :84
        bases = reads_df['gene'].values                                                    # :85
        if len(current_label_img.shape) == 3:                                              # :86
            reads_assignment = current_label_img[points[:, 2], points[:, 1], points[:, 0]]  # :87
        else:
            reads_assignment = current_label_img[points[:, 1], points[:, 0]]               # :89
        reads_df['seg_label'] = reads_assignment                                           # :91
    else:
        reads_assignment = np.array([0])                                                   # :93
    cell_locs, areas, seg_labels = [], [], []                                              # :96-99
    total_cells = len(np.unique(current_label_img)) - 1                                    # :97
    genes_df = pd.read_csv(genes_csv, header=None)                                         # :102
    genes_df.columns = ['gene', 'barcode']                                                 # :103
    genes = genes_df['gene'].values                                                        # :105
    gene_seq_to_index = {k: i for i, k in enumerate(genes)}                            # :106-109
    if total_cells == 0 or (len(np.unique(reads_assignment)) == 1
                            and np.unique(reads_assignment)[0] == 0):                     # :111
        cell_by_gene = np.zeros((0, len(genes)))                                           # :112
        if len(current_label_img.shape) == 3:                                              # :116
            meta = pd.DataFrame({'sample': sample, 'fov_id': fov_id, 'volume': 0, 'fov_x': 0,
                                 'fov_y': 0, 'fov_z': 0, 'seg_label': 0}, index=[])       # :117-118
        else:
            meta = pd.DataFrame({'sample': sample, 'fov_id': fov_id, 'volume': 0, 'fov_x': 0,
                                 'fov_y': 0, 'seg_label': 0}, index=[])                   # :120-121
        return reads_df, cell_by_gene, meta
    cell_by_gene = np.zeros((total_cells, len(genes)))                                     # :127
    for i, region in enumerate(regionprops(current_label_img)):                            # :131
        areas.append(region.area)                                                          # :132
        cell_locs.append(region.centroid)                                                  # :133
        seg_labels.append(region.label)                                                    # :134
        assigned_reads = bases[np.argwhere(reads_assignment == region.label).flatten()]    # :136
        for j in assigned_reads:                                                           # :137
            if j in gene_seq_to_index:                                                     # :138
                cell_by_gene[i, gene_seq_to_index[j]] += 1                                 # :139
    cell_locs = np.array(cell_locs).astype(int)                                            # :141
    if len(current_label_img.shape) == 3:                                                  # :142
        meta = pd.DataFrame({'sample': sample, 'fov_id': fov_id, 'volume': areas,
                             'fov_x': cell_locs[:, 2], 'fov_y': cell_locs[:, 1],
                             'fov_z': cell_locs[:, 0], 'seg_label': seg_labels})          # :144
    else:
        meta = pd.DataFrame({'sample': sample, 'fov_id': fov_id, 'volume': areas,
                             'fov_x': cell_locs[:, 1], 'fov_y': cell_locs[:, 0],
                             'seg_label': seg_labels})                                     # :148
    return reads_df, cell_by_gene, meta


# The lines of reads_assignment.py that legacy_assignment follows.
_PARAMETERS = "snakemake.config['rules']['reads_assignment']['parameters']"
SCRIPT_LINES = {
    52: "if len(current_label_img.shape) == 3:",
    55: f"if {_PARAMETERS}['expand_labels']:",
    56: "for z in range(current_label_img.shape[0]):",
    57: ("current_label_img[z,:,:] = expand_labels(current_label_img[z,:,:], "
         f"distance={_PARAMETERS}['dilation_distance'])"),
    66: f"if {_PARAMETERS}['expand_labels']:",
    67: ("current_label_img = expand_labels(current_label_img, "
         f"distance={_PARAMETERS}['dilation_distance'])"),
    74: "reads_df = pd.read_csv(snakemake.input[3])",
    75: "reads_df['x'] = reads_df['x'] - 1",
    76: "reads_df['y'] = reads_df['y'] - 1",
    77: "reads_df['z'] = reads_df['z'] - 1",
    82: "if reads_df.shape[0] != 0:",
    84: 'points = reads_df.loc[:, ["x", "y", "z"]].values',
    85: "bases = reads_df['gene'].values",
    87: "reads_assignment = current_label_img[points[:, 2], points[:, 1], points[:, 0]]",
    89: "reads_assignment = current_label_img[points[:, 1], points[:, 0]]",
    91: "reads_df['seg_label'] = reads_assignment",
    93: "reads_assignment = np.array([0])",
    97: "total_cells = len(np.unique(current_label_img)) - 1",
    102: "genes_df = pd.read_csv(snakemake.input[4], header=None)",
    103: "genes_df.columns = ['gene', 'barcode']",
    111: ("if total_cells == 0 or (len(np.unique(reads_assignment)) == 1 and "
          "np.unique(reads_assignment)[0] == 0):"),
    112: "cell_by_gene = np.zeros((0, len(genes)))",
    116: "if len(current_label_img.shape) == 3:",
    117: ("current_meta = pd.DataFrame({'sample': current_sample, 'fov_id': current_fov_id, "
          "'volume': 0, 'fov_x': 0, 'fov_y': 0, 'fov_z': 0, 'seg_label': 0,"),
    120: ("current_meta = pd.DataFrame({'sample': current_sample, 'fov_id': current_fov_id, "
          "'volume': 0, 'fov_x': 0, 'fov_y': 0, 'seg_label': 0,"),
    127: "cell_by_gene = np.zeros((total_cells, len(genes)))",
    131: "for i, region in enumerate(regionprops(current_label_img)):",
    132: "areas.append(region.area)",
    133: "cell_locs.append(region.centroid)",
    134: "seg_labels.append(region.label)",
    136: "assigned_reads = bases[np.argwhere(reads_assignment == region.label).flatten()]",
    138: "if j in gene_seq_to_index:",
    139: "cell_by_gene[i, gene_seq_to_index[j]] += 1",
    141: "cell_locs = np.array(cell_locs).astype(int)",
    144: ("current_meta = pd.DataFrame({'sample': current_sample, 'fov_id': current_fov_id, "
          "'volume': areas, 'fov_x': cell_locs[:, 2], 'fov_y': cell_locs[:, 1], "
          "'fov_z': cell_locs[:, 0], 'seg_label': seg_labels,"),
    148: ("current_meta = pd.DataFrame({'sample': current_sample, 'fov_id': current_fov_id, "
          "'volume': areas, 'fov_x': cell_locs[:, 1], 'fov_y': cell_locs[:, 0], "
          "'seg_label': seg_labels,"),
}

LABELS_DIGEST = "d4023267d11f4f1e238fbfa320a88e02c3ddf230667a8e8b2e724e51a796b6a7"
# (dimensions, expand) -> digests of the per-molecule labels, the count matrix and the cell metadata
PINS = {
    ("3d", False): ("c78d741141b61bf3d3b4b1c1e45a3d44e90e7f99c48a0d68d06a934fb46fe9e9",
                    "dc0804ba501126c4fe0a0eb5b1d56b1eb1371ef9a2c41bba0d11f81fe1737f13",
                    "1fc6532b64b21d7aed9c4b7065f5a53a3725679e36e84716ed4e13ade96a389d"),
    ("3d", True): ("fe4ae60cb6ba5afe4c773d80701c0dd4a2ceb6f2142a16b4f68fb04c10da912e",
                   "8f25ccc3a40de3528ac0836e1d21ea3f706328e4248669054a49aede3a18c132",
                   "fcb61757df5b9cc889316fa3d9277215d3aad715c62ffa8dee74ec222f10a805"),
    ("2d", False): ("2e57ab8052b30e75cc3576c53c2038364a974d1f62bc3f72f2043db32f81d86e",
                    "e187daf861c24b05aff233f6cfb0aa5079a831d2c5d4750a255a1abfa2e0b709",
                    "d4a6b07fc003f0b78ed681c0dc4b4435edc7a05b84c95857f0b2f95f485772c2"),
    ("2d", True): ("53d0fdb95001b762f549243b77e73016895c35eff6b8326b828a10374525818f",
                   "fa2da938c61f2515336df415fc2908ceed137ada3a18c99b7ab0aef41328a830",
                   "0605e2e234588e6c5d7bb6b6e429c675176b97f92f1a163633eed727a646a14f"),
}
# (dimensions, expand) -> per-molecule labels, in MOLECULES order
SEG_LABELS = {
    ("3d", False): [3, 3, 3, 7, 7, 12, 12, 12, 20, 25, 0, 0, 0, 0, 0, 7, 0, 3, 0],
    ("3d", True): [3, 3, 3, 7, 7, 12, 12, 12, 20, 25, 0, 0, 3, 20, 12, 7, 0, 3, 0],
    ("2d", False): [3, 3, 3, 7, 7, 12, 12, 12, 20, 25, 0, 0, 0, 0, 0, 7, 0, 3, 3],
    ("2d", True): [3, 3, 3, 7, 7, 12, 12, 12, 20, 25, 0, 0, 7, 20, 12, 7, 0, 3, 3],
}
# dimensions -> digest of the empty cell metadata
EMPTY_META_DIGESTS = {"3d": "6255dcaada831c4983663981a5672e0e30bfaf103a32add4531bcc7aa88284ad",
                      "2d": "633abac4ebf542373a795970e930f65c11aa2edb971ad7bededed21ca2096533"}
# Planar expansion by 4 (segmentation) then by 2 (assignment), and by 6 once
TWO_EXPANSIONS = {"twice": "1e683ba4caa86c83b29f8692d1be9339394aa64b33fa8c6d78ad17aad6df490b",
                  "once": "1a84d65b8f83a686345ad1aeea70c45498167722427c106ff7a3f36ebc4713f3"}


@pytest.fixture(scope="module")
def labels():
    return label_fixture()


@pytest.fixture
def inputs(tmp_path):
    return write_reads(tmp_path / "goodSpots.csv"), write_genes(tmp_path / "genes.csv")


def _case(labels, dimensions):
    return labels if dimensions == "3d" else labels.max(axis=0)


def test_the_helper_follows_the_script():
    lines = SCRIPT.read_text().splitlines()
    assert {n: lines[n - 1].strip() for n in SCRIPT_LINES} == SCRIPT_LINES


def test_fixture(labels):
    assert labels.shape == SHAPE_ZYX and labels.dtype == np.uint16
    assert sorted(np.unique(labels).tolist()) == [0, 3, 7, 12, 20, 25]
    assert digest(labels) == LABELS_DIGEST


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("dimensions", ["3d", "2d"])
def test_assignment(labels, inputs, dimensions, expand):
    reads, counts, meta = legacy_assignment(_case(labels, dimensions), *inputs, expand=expand)
    assert reads["seg_label"].tolist() == SEG_LABELS[(dimensions, expand)]
    assert counts.dtype == np.float64 and counts.shape == (5, len(GENES))
    assert meta["seg_label"].tolist() == [3, 7, 12, 20, 25]  # regionprops order, by label value
    observed = (digest(reads["seg_label"].to_numpy()), digest(counts), frame_digest(meta))
    assert observed == PINS[(dimensions, expand)]


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("dimensions", ["3d", "2d"])
def test_counts_follow_the_labels(labels, inputs, dimensions, expand):
    """Every counted molecule has its cell's label; label 0 and the gene Z are never counted."""
    reads, counts, meta = legacy_assignment(_case(labels, dimensions), *inputs, expand=expand)
    genes = [gene for gene, _ in GENES]
    expected = np.zeros_like(counts)
    for label, gene in zip(reads["seg_label"], reads["gene"]):
        if label and gene in genes:
            expected[meta["seg_label"].tolist().index(label), genes.index(gene)] += 1
    assert np.array_equal(counts, expected)
    in_cells = int((reads["seg_label"] > 0).sum())
    uncounted_gene = int(((reads["seg_label"] > 0) & (reads["gene"] == "Z")).sum())
    assert counts.sum() == in_cells - uncounted_gene


@pytest.mark.parametrize("dimensions", ["3d", "2d"])
def test_a_fov_without_cells(labels, inputs, dimensions):
    empty = np.zeros_like(_case(labels, dimensions))
    reads, counts, meta = legacy_assignment(empty, *inputs, expand=True)
    assert counts.shape == (0, len(GENES)) and len(meta) == 0
    assert not reads["seg_label"].any()
    assert frame_digest(meta) == EMPTY_META_DIGESTS[dimensions]


def test_cells_without_any_molecule_are_dropped(labels, tmp_path):
    """Legacy: when no molecule lands in a cell, the FOV gives no cell at all (line 111)."""
    background = tuple(m for m in MOLECULES[10:12])
    reads_csv = write_reads(tmp_path / "goodSpots.csv", background)
    reads, counts, meta = legacy_assignment(labels, reads_csv, write_genes(tmp_path / "genes.csv"),
                                            expand=False)
    assert not reads["seg_label"].any()
    assert counts.shape == (0, len(GENES)) and len(meta) == 0


def test_coordinate_zero_reads_the_far_edge(labels, tmp_path):
    """Legacy: one-based x = 0 becomes index -1 and reads the label at x = 63 (no bounds check)."""
    probe = np.zeros(SHAPE_ZYX, np.uint16)
    probe[5, 10, 63] = 9
    reads_csv = tmp_path / "goodSpots.csv"
    reads_csv.write_text("x,y,z,gene\n0,11,6,A\n")
    genes_csv = write_genes(tmp_path / "genes.csv")
    reads, _, _ = legacy_assignment(probe, reads_csv, genes_csv, expand=False)
    assert reads["seg_label"].tolist() == [9]


def test_a_coordinate_beyond_the_grid_raises(labels, tmp_path):
    reads_csv = tmp_path / "goodSpots.csv"
    reads_csv.write_text(f"x,y,z,gene\n{SHAPE_ZYX[2] + 1},11,6,A\n")
    with pytest.raises(IndexError):
        legacy_assignment(labels, reads_csv, write_genes(tmp_path / "genes.csv"), expand=False)


def test_float_coordinates_raise(labels, tmp_path):
    """Legacy: export_spots writes float coordinates (8.0); indexing with them raises (line 87)."""
    reads_csv = tmp_path / "goodSpots.csv"
    reads_csv.write_text("x,y,z,gene\n17.0,17.0,7.0,A\n")
    with pytest.raises(IndexError, match="integer"):
        legacy_assignment(labels, reads_csv, write_genes(tmp_path / "genes.csv"), expand=False)


def test_two_expansions_are_not_one(labels):
    """Legacy: segmentation and assignment each expand per plane; 4 then 2 is not 6."""
    twice, once = labels.copy(), labels.copy()
    for z in range(SHAPE_ZYX[0]):
        twice[z] = expand_labels(expand_labels(labels[z], distance=4), distance=2)
        once[z] = expand_labels(labels[z], distance=6)
    assert {"twice": digest(twice), "once": digest(once)} == TWO_EXPANSIONS
    assert (twice > 0).sum() < (once > 0).sum()


@pytest.mark.parametrize("change", ["distance", "coordinate"])
def test_changing_a_pinned_input_changes_a_digest(labels, tmp_path, change):
    genes_csv = write_genes(tmp_path / "genes.csv")
    if change == "distance":
        reads, counts, _ = legacy_assignment(labels, write_reads(tmp_path / "reads.csv"), genes_csv,
                                             expand=True, distance=DISTANCE - 2)
    else:
        moved = (((2, 16, 16), "A"),) + MOLECULES[1:]  # out of cell 3, in the background
        reads_csv = write_reads(tmp_path / "reads.csv", moved)
        reads, counts, _ = legacy_assignment(labels, reads_csv, genes_csv, expand=True)
    pinned = PINS[("3d", True)]
    assert digest(reads["seg_label"].to_numpy()) != pinned[0]
    assert digest(counts) != pinned[1]
