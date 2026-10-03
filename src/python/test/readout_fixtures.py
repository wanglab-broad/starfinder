"""Hand-built §2.8 readout fixtures `crosstalk` and `two_seg` (docs/readout-algorithms.md, "Fixtures").

Both are 16×64×64 uint16 images with four channels and four rounds, built like the golden
fixture of test_readout_golden.py: a background of 20 grey levels with noise sd 3, and
per amplicon and round a Gaussian of amplitude 800 (sd 1.0 in Z, 1.2 in Y and X) in the
channel of its color, all values from the seed (100 to 102). The candidates are placed by
hand at the planted positions, with their detection channel in round 1 (no detection
runs), so every expected answer is known.

crosstalk: 20 amplicons of the eight golden genes, each with a 5 % copy of its signal in
the next channel, centred at a known offset: 8 at 0, 4 at 1, 4 at √2 and 4 at 2 voxels;
each has a candidate in its own channel and one in the copy's. Six pairs of amplicons of
different genes 1 voxel apart, in different channels. One more amplicon of the color
sequence 2311 (one color from GeneX 2341 and from GeneY 2314) has its copy 1 voxel along
Z, and two added Gaussians make round 3 ambiguous at the copy (GeneX's channel, centred
2 voxels beyond the source) and round 4 ambiguous at the source (GeneY's channel, centred
1 voxel before it): the codebook-aware decoder rescues the source to GeneY and the copy
to GeneX, so their group has conflicting calls.

two_seg: a codebook of eight 6-base barcodes in two 3-base segments (the layout of
load_codebook.split_index [3], 2 + 2 colors) with allowed ends CC (segment A) and TT
(segment B); 24 amplicons (three of each entry), 4 reads one color from an entry, and
4 reads whose segment A colors are an entry's and whose segment B colors decode from T
to ends outside TT.

golden_seed and golden_rounds give the golden fixture of test_readout_golden.py (unchanged)
with a seed in place of its pinned SEED: the same spots and features, other noise.
GOLDEN_SEEDS is the pinned seed followed by the hand-built seeds 100 to 102.
"""
from contextlib import contextmanager
from unittest import mock

import numpy as np
import pandas as pd

from starfinder.barcode import BarcodeLayout, Segment, decode_color_sequence, load_codebook
from starfinder.dataset import Dataset, RoundState
from starfinder.image import ImageMetadata
from starfinder.spot_finding import LocalMaximaConfig, SpotFindingResult

from . import test_readout_golden as golden

SHAPE_ZYX = (16, 64, 64)
CHANNELS = ("ch00", "ch01", "ch02", "ch03")
ROUNDS = ("round1", "round2", "round3", "round4")
SEEDS = (100, 101, 102)
GOLDEN_SEEDS = (golden.SEED,) + SEEDS
AMPLITUDE = 800.0
COPY_FRACTION = 0.05
# Candidate slots, 12 voxels apart laterally and 6 along Z.
SLOTS = [(z, y, x) for z in (5.0, 11.0) for y in (8.0, 20.0, 32.0, 44.0, 56.0) for x in (8.0, 20.0, 32.0, 44.0, 56.0)]


@contextmanager
def golden_seed(seed):
    """Within the block, the golden helpers (fixture_rounds, extract) draw their noise from seed."""
    with mock.patch.object(golden, "SEED", seed):
        yield


def golden_rounds(seed):
    """The golden fixture's images with seed in place of its SEED."""
    with golden_seed(seed):
        return golden.fixture_rounds()


# --- crosstalk ------------------------------------------------------------------------------------

CROSSTALK_CODEWORDS = {"1234": "GeneA", "2143": "GeneB", "3412": "GeneC", "4321": "GeneD", "2222": "GeneE",
                       "3131": "GeneF", "4242": "GeneG", "1313": "GeneH", "2341": "GeneX", "2314": "GeneY"}
# Copy offsets (dz, dy, dx) of the 20 amplicons: 8 at 0, 4 at 1, 4 at √2 and 4 at 2 voxels.
COPY_OFFSETS = ([(0, 0, 0)] * 8 + [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, -1, 0)]
                + [(0, 1, 1), (1, 1, 0), (0, 1, -1), (1, 0, 1)] + [(0, 2, 0), (0, 0, 2), (0, -2, 0), (0, 0, -2)])
DIFFERENT_GENE_PAIRS = [("1234", "2143"), ("3412", "4321"), ("2222", "3131"), ("4242", "1313"), ("1234", "3412"),
                        ("2143", "4321")]
CONFLICT_SEQUENCE, CONFLICT_X, CONFLICT_Y = "2311", "2341", "2314"
CONFLICT_OFFSET = (1, 0, 0)
# Amplitudes (x AMPLITUDE) of the added Gaussians that make the conflicting reads rescue apart.
CONFLICT_X_AMPLITUDE, CONFLICT_Y_AMPLITUDE = 0.85, 1.1


def _gaussian(center):
    z, y, x = np.meshgrid(*(np.arange(n, dtype=np.float64) for n in SHAPE_ZYX), indexing="ij")
    cz, cy, cx = center
    return np.exp(-((z - cz) / 1.0) ** 2 / 2 - ((y - cy) / 1.2) ** 2 / 2 - ((x - cx) / 1.2) ** 2 / 2)


def crosstalk_amplicons():
    """(amplicon_id, centre, color sequence, copy offset or None, kind) of every planted amplicon."""
    genes = list(CROSSTALK_CODEWORDS)[:8]
    amplicons, slot = [], 0
    for i, offset in enumerate(COPY_OFFSETS):
        amplicons.append((f"a{i}", SLOTS[slot], genes[i % 8], offset, "copy"))
        slot += 1
    for p, (first, second) in enumerate(DIFFERENT_GENE_PAIRS):
        z, y, x = SLOTS[slot]
        slot += 1
        amplicons.append((f"p{p}a", (z, y, x), first, None, "pair"))
        amplicons.append((f"p{p}b", (z, y, x + 1.0), second, None, "pair"))
    amplicons.append(("c0", SLOTS[slot], CONFLICT_SEQUENCE, CONFLICT_OFFSET, "conflict"))
    return amplicons


def crosstalk_truth():
    """One row per candidate: z, y, x, channel (round-1 detection channel), source amplicon, role
    (source or copy), copy distance (voxels) and kind (copy, pair, conflict)."""
    rows = []
    for amplicon, center, colors, offset, kind in crosstalk_amplicons():
        channel = int(colors[0]) - 1
        rows.append(dict(z=center[0], y=center[1], x=center[2], channel=channel, source=amplicon, role="source",
                         distance=np.nan if offset is None else float(np.linalg.norm(offset)), kind=kind))
        if offset is not None:
            z, y, x = np.add(center, offset)
            rows.append(dict(z=float(z), y=float(y), x=float(x), channel=(channel + 1) % 4, source=amplicon,
                             role="copy", distance=float(np.linalg.norm(offset)), kind=kind))
    return pd.DataFrame(rows)


def crosstalk_rounds(seed):
    """uint16 ZYXC image per round of the crosstalk fixture."""
    rng = np.random.RandomState(seed)
    amplicons = crosstalk_amplicons()
    cache = {}

    def blob(center):
        key = tuple(float(c) for c in center)
        if key not in cache:
            cache[key] = _gaussian(key)
        return cache[key]

    conflict = next(center for _, center, _, _, kind in amplicons if kind == "conflict")
    rounds = {}
    for r, label in enumerate(ROUNDS):
        image = 20.0 + rng.normal(0.0, 3.0, SHAPE_ZYX + (len(CHANNELS),))
        for _, center, colors, offset, _ in amplicons:
            channel = int(colors[r]) - 1
            image[..., channel] += AMPLITUDE * blob(center)
            if offset is not None:
                image[..., (channel + 1) % 4] += COPY_FRACTION * AMPLITUDE * blob(np.add(center, offset))
        z, y, x = conflict
        if r == 2:
            image[..., int(CONFLICT_X[2]) - 1] += CONFLICT_X_AMPLITUDE * AMPLITUDE * blob((z + 2, y, x))
        if r == 3:
            image[..., int(CONFLICT_Y[3]) - 1] += CONFLICT_Y_AMPLITUDE * AMPLITUDE * blob((z - 1, y, x))
        rounds[label] = np.clip(np.rint(image), 0, 65535).astype(np.uint16)
    return rounds


def candidates(truth, name):
    """The hand-placed candidates of a truth table as a SpotFindingResult with their detection channels."""
    spots = pd.DataFrame({"spot_id": pd.array([str(i) for i in range(len(truth))], dtype="string"),
                          "z": truth.z.astype(np.float64).to_numpy(), "y": truth.y.astype(np.float64).to_numpy(),
                          "x": truth.x.astype(np.float64).to_numpy(),
                          "channel": truth.channel.astype(np.int64).to_numpy()})
    return SpotFindingResult(spots, ImageMetadata(f"{name}/FOV_001"), f'["{name}", "sample", "FOV_001", null]',
                             LocalMaximaConfig(), {"channel_labels": list(CHANNELS)})


def write_codebook(directory, codewords, name="codebook.csv"):
    """gene,barcode CSV of 5-base two_base barcodes (start base C, reversed) for 4-color sequences."""
    path = directory / name
    path.write_text("\n".join(f"{gene},{decode_color_sequence(colors, 'C')[::-1]}"
                              for colors, gene in codewords.items()) + "\n")
    return path


def crosstalk_codebook(directory):
    return load_codebook(write_codebook(directory, CROSSTALK_CODEWORDS), round_labels=ROUNDS,
                         channel_labels=CHANNELS)


def dataset(root, name):
    root.mkdir(parents=True, exist_ok=True)
    return Dataset(root, root / "out", name, "sample", "out",
                   rounds=RoundState(sequencing_rounds=list(ROUNDS), reference_round="round1"),
                   channel_order=list(CHANNELS))


def crosstalk_fov(root, seed=100):
    """A FOV of the crosstalk fixture with resident rounds, the hand-placed candidates and the codebook."""
    data = dataset(root, "crosstalk")
    data.codebook = crosstalk_codebook(root)
    fov = data.fov("FOV_001")
    fov.images = crosstalk_rounds(seed)
    fov.metadata = {label: ImageMetadata("crosstalk/FOV_001") for label in ROUNDS}
    fov.spot_result = candidates(crosstalk_truth(), "crosstalk")
    return fov


# --- two_seg --------------------------------------------------------------------------------------

TWO_SEG_LAYOUT = BarcodeLayout((Segment("A", 3, (("C", "C"),)), Segment("B", 3, (("T", "T"),))), ("A", "B"))
# Segment A in read orientation C?C gives colors 22, 11, 44, 33; segment B T?T gives 44, 33, 22, 11.
_SEGMENT_A = {"22": "CAC", "11": "CCC", "44": "CGC", "33": "CTC"}
_SEGMENT_B = {"44": "TAT", "33": "TCT", "22": "TGT", "11": "TTT"}
TWO_SEG_ENTRIES = {"2244": "G1", "1133": "G2", "4422": "G3", "3311": "G4", "2211": "G5", "1122": "G6",
                   "4433": "G7", "3344": "G8"}
# One color from exactly one entry (2244, 1133, 4422, 3311).
TWO_SEG_ONE_OFF = ("2144", "1233", "4322", "3312")
# Segment A colors of an entry; segment B colors decode from T to TAG, TCG, TTG and TGG.
TWO_SEG_WRONG_ENDS = ("2243", "1134", "4412", "3321")


def two_seg_codebook(directory):
    """The two_seg codebook: gene,barcode rows of 6-base barcodes, both segments palindromic."""
    path = directory / "two_seg.csv"
    path.write_text("\n".join(f"{gene},{_SEGMENT_A[colors[:2]]}{_SEGMENT_B[colors[2:]]}"
                              for colors, gene in TWO_SEG_ENTRIES.items()) + "\n")
    return load_codebook(path, round_labels=ROUNDS, channel_labels=CHANNELS, layout=TWO_SEG_LAYOUT)


def two_seg_truth():
    """One row per candidate: z, y, x, channel, planted color sequence and kind (entry, one_off, wrong_ends)."""
    planted = ([(colors, "entry") for colors in TWO_SEG_ENTRIES for _ in range(3)]
               + [(colors, "one_off") for colors in TWO_SEG_ONE_OFF]
               + [(colors, "wrong_ends") for colors in TWO_SEG_WRONG_ENDS])
    return pd.DataFrame([dict(z=z, y=y, x=x, channel=int(colors[0]) - 1, colors=colors, kind=kind)
                         for (z, y, x), (colors, kind) in zip(SLOTS, planted)])


def two_seg_rounds(seed):
    """uint16 ZYXC image per round of the two_seg fixture (5 % of each spot in the next channel)."""
    rng = np.random.RandomState(seed)
    truth = two_seg_truth()
    blobs = [_gaussian((z, y, x)) for z, y, x in zip(truth.z, truth.y, truth.x)]
    rounds = {}
    for r, label in enumerate(ROUNDS):
        image = 20.0 + rng.normal(0.0, 3.0, SHAPE_ZYX + (len(CHANNELS),))
        for blob, colors in zip(blobs, truth.colors):
            channel = int(colors[r]) - 1
            image[..., channel] += AMPLITUDE * blob
            image[..., (channel + 1) % 4] += COPY_FRACTION * AMPLITUDE * blob
        rounds[label] = np.clip(np.rint(image), 0, 65535).astype(np.uint16)
    return rounds


def two_seg_fov(root, seed=100):
    data = dataset(root, "two_seg")
    data.codebook = two_seg_codebook(root)
    fov = data.fov("FOV_001")
    fov.images = two_seg_rounds(seed)
    fov.metadata = {label: ImageMetadata("two_seg/FOV_001") for label in ROUNDS}
    fov.spot_result = candidates(two_seg_truth(), "two_seg")
    return fov
