"""Checks of the worked examples in docs/readout-algorithms.md ("Worked examples", W-279).

Each test follows one example of the page, step by step. The ``two_base`` examples run
against the current code. The ``one_base`` and direct-readout examples describe behavior
that §2.8 adds: the parts the current code can check (codebook validation and decoding of
explicit color sequences, multi-round detection with its ``round`` and ``channel``
columns, single-round extraction) are checked against it; the rest is written here as
small reference functions of the expected behavior, which the implementation issues'
tests check against the real functions.
"""
import numpy as np
import pandas as pd
import pytest

from starfinder.barcode import (Codebook, EncodingConfig, WtaDecoderConfig, decode_barcodes,
                                decode_color_sequence, encode_bases, extract_intensities, load_codebook)
from starfinder.dataset import Dataset, RoundState
from starfinder.image import ImageMetadata
from starfinder.io import ImageLoadResult
from starfinder.spot_finding import LocalMaximaConfig, SpotFindingPlan

from .barcode_cases import intensity, tensor

CHANNELS = ("ch00", "ch01", "ch02", "ch03")


def rounds(n):
    return tuple(f"round{i}" for i in range(1, n + 1))


def assign(codebook, sequence):
    """WTA call of one read whose brightest channel in each round is the given color."""
    reads = intensity(tensor([sequence]), channels=CHANNELS, rounds=codebook.round_labels)
    return decode_barcodes(reads, codebook, config=WtaDecoderConfig()).table.iloc[0]


# --- Example 1: two_base, one segment ------------------------------------------------

def test_two_base_one_segment(tmp_path):
    barcode = "CTGACC"
    # Barcode to colors: reverse, then one color per adjacent base pair.
    assert barcode[::-1] == "CCAGTC"
    assert [encode_bases(barcode[::-1][i:i + 2]) for i in range(5)] == ["1", "2", "3", "2", "3"]
    assert EncodingConfig(reverse_bases=True).encode(barcode) == "12323"
    path = tmp_path / "genes.csv"
    path.write_text(f"Slc17a7,{barcode}\n")
    book = load_codebook(path, round_labels=rounds(5), channel_labels=CHANNELS)
    assert book.gene_to_seq == {"Slc17a7": "12323"}
    # Observed colors back to the barcode: the first base of the read orientation (C) is known.
    read = assign(book, "12323")
    assert (read.observed_color_sequence, read.gene_id, read.call_status) == ("12323", "Slc17a7", "assigned")
    assert decode_color_sequence("12323", "C") == "CCAGTC"
    assert decode_color_sequence("12323", "C")[::-1] == barcode
    # The current end-base check reads the decoded (reversed) bases: first C, last C.
    decoded = decode_color_sequence("12323", "C")
    assert decoded[0] + decoded[-1] == "CC"


# --- Example 2: two_base, two segments (11 bases, 5 + 4 colors over 9 rounds) --------

def test_two_base_two_segments(tmp_path):
    segment_a, segment_b = "CAGTAC", "TGCAT"
    barcode = segment_a + segment_b
    assert len(barcode) == 11
    reversed_bases = barcode[::-1]
    assert reversed_bases == "TACGTCATGAC"
    colors = encode_bases(reversed_bases)
    assert colors == "4242324232"  # c1..c10
    # c5 is the pair (T, C) across the junction: b7 of segment B and b6 of segment A.
    assert reversed_bases[4:6] == segment_b[0] + segment_a[-1] and colors[4] == "3"
    # Python split_index 4 (zero-based color index; MATLAB split_index 5) drops c5 and puts
    # c6..c10 (segment A, rounds 1-5) before c1..c4 (segment B, rounds 6-9).
    assert colors[5:] + colors[:4] == "24232" + "4242"
    assert EncodingConfig(reverse_bases=True, split_index=4).encode(barcode) == "242324242"
    path = tmp_path / "genes.csv"
    path.write_text(f"Gfap_probe1,{barcode}\n")
    book = load_codebook(path, round_labels=rounds(9), channel_labels=CHANNELS,
                         encoding=EncodingConfig(reverse_bases=True, split_index=4))
    assert book.gene_to_seq == {"Gfap_probe1": "242324242"}
    read = assign(book, "242324242")
    assert (read.gene_id, read.call_status) == ("Gfap_probe1", "assigned")
    # Observed colors back to the barcode: cut 5 | 4, decode each segment from its own
    # first base in the read orientation, reverse each, and join A then B.
    first, second = "242324242"[:5], "242324242"[5:]
    assert decode_color_sequence(first, "C") == "CATGAC" == segment_a[::-1]
    assert decode_color_sequence(second, "T") == "TACGT" == segment_b[::-1]
    assert decode_color_sequence(first, "C")[::-1] + decode_color_sequence(second, "T")[::-1] == barcode
    # Each segment's end bases in the read orientation: A is C...C, B is T...T.
    assert [s[0] + s[-1] for s in ("CATGAC", "TACGT")] == ["CC", "TT"]


def layout_encode(bases, lengths, order, reverse_bases):
    """Expected segment-layout encoding: each segment encoded on its own bases, then the
    segments' colors joined in acquisition order (docs/readout-contract.md)."""
    starts = np.cumsum((0,) + tuple(lengths))
    segments = [bases[a:b] for a, b in zip(starts[:-1], starts[1:])]
    colors = [encode_bases(s[::-1] if reverse_bases else s) for s in segments]
    return "".join(colors[k] for k in order)


@pytest.mark.parametrize("reverse_bases", [True, False])
@pytest.mark.parametrize("matlab_split", range(2, 10))
def test_split_index_translation_equals_the_segment_layout(matlab_split, reverse_bases):
    # The contract's translation of the shared one-based split_index s for an n-base
    # barcode, checked against the current zero-based EncodingConfig.split_index s - 1.
    barcode = "CAGTACTGCAT"
    n, s = len(barcode), matlab_split
    if reverse_bases:
        expected = layout_encode(barcode, (n - s, s), (0, 1), reverse_bases)
    else:
        expected = layout_encode(barcode, (s, n - s), (1, 0), reverse_bases)
    assert EncodingConfig(reverse_bases=reverse_bases, split_index=s - 1).encode(barcode) == expected


# --- Example 3: one_base -----------------------------------------------------------

ONE_BASE_COLORS = {"A": "1", "C": "2", "G": "3", "T": "4"}  # the example's explicit mapping


def one_base_encode(bases, base_to_color, reverse_bases=False):
    """Expected one_base encoding: one color per base through the explicit mapping."""
    bases = bases[::-1] if reverse_bases else bases
    return "".join(base_to_color[b] for b in bases)


def one_base_decode(colors, base_to_color, reverse_bases=False):
    """Expected one_base decoding: the inverse mapping, which must be one-to-one."""
    inverse = {c: b for b, c in base_to_color.items()}
    assert len(inverse) == len(base_to_color)
    bases = "".join(inverse[c] for c in colors)
    return bases[::-1] if reverse_bases else bases


def test_one_base():
    barcode = "GATC"
    # Expected behavior (the one_base registry entry): each base is one round's color.
    assert one_base_encode(barcode, ONE_BASE_COLORS) == "3142"
    assert one_base_decode("3142", ONE_BASE_COLORS) == barcode
    # Checked against current code: a codebook of explicit color sequences validates, and
    # WTA assigns the gene from the observed colors.
    book = Codebook(pd.DataFrame({"gene_id": ["Mbp"], "color_sequence": [one_base_encode(barcode, ONE_BASE_COLORS)]}),
                    rounds(4), CHANNELS)
    read = assign(book, "3142")
    assert (read.gene_id, read.call_status, read.call_type) == ("Mbp", "assigned", "exact")
    # The current two_base encoding of the same 4 bases has 3 colors, not 4.
    assert EncodingConfig(reverse_bases=False).encode(barcode) == "343"


# --- Example 4: direct readout, two rounds -----------------------------------------

DIRECT_MAPPING = pd.DataFrame({
    "round": ["round1"] * 4 + ["round2"] * 4,
    "channel": list(CHANNELS) * 2,
    "gene_id": ["Gfap", "Slc17a7", "Gad1", "Mbp", "Pvalb", "Sst", "Vip", "Aqp4"],
})
# (round, channel index, z, y, x) of the planted spots; the first and last share a position.
DIRECT_SPOTS = [("round1", 0, 3, 8, 8), ("round1", 2, 3, 16, 16), ("round2", 1, 3, 8, 16), ("round2", 3, 3, 8, 8)]


def check_direct_mapping(mapping):
    """Expected validation: every (round, channel) once and every gene once."""
    if mapping.duplicated(["round", "channel"]).any():
        raise ValueError("a (round, channel) appears more than once")
    if mapping.gene_id.duplicated().any():
        raise ValueError(f"genes {sorted(set(mapping.gene_id[mapping.gene_id.duplicated()]))} appear more than once")


def direct_images():
    """Two uint16 rounds of 6×24×24 voxels: background 100 with noise sd 3 (seed 100), and
    each planted spot a Gaussian of amplitude 1000 and sd 1 voxel in its round and channel."""
    shape = (6, 24, 24)
    rng = np.random.RandomState(100)
    z, y, x = np.meshgrid(*(np.arange(n, dtype=np.float64) for n in shape), indexing="ij")
    images = {name: 100 + rng.normal(0, 3, shape + (4,)) for name in rounds(2)}
    for name, channel, cz, cy, cx in DIRECT_SPOTS:
        images[name][..., channel] += 1000 * np.exp(-((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) / 2)
    return {name: np.rint(image).astype(np.uint16) for name, image in images.items()}


def test_direct_readout_two_rounds(tmp_path):
    check_direct_mapping(DIRECT_MAPPING)
    with pytest.raises(ValueError, match="Gfap"):
        check_direct_mapping(DIRECT_MAPPING.replace({"gene_id": {"Vip": "Gfap"}}))
    # Checked against current code: detection in both rounds gives one candidate table with
    # its round and channel; coincident candidates of different rounds stay separate rows.
    dataset = Dataset(tmp_path, tmp_path / "out", "example", "sample", "out",
                      rounds=RoundState(sequencing_rounds=list(rounds(2)), reference_round="round1"),
                      channel_order=list(CHANNELS))
    fov = dataset.fov("FOV_001")
    fov.images = direct_images()
    fov.metadata = {name: ImageMetadata("example/FOV_001") for name in rounds(2)}
    fov.find_spots(config=SpotFindingPlan(LocalMaximaConfig("noise", 10.0), rounds=rounds(2)))
    spots = fov.spot_result.spots
    assert list(zip(spots["round"], spots.channel, spots.y, spots.x)) == [
        (name, channel, float(cy), float(cx)) for name, channel, _, cy, cx in DIRECT_SPOTS]
    # Expected behavior: identity from the candidate's own round and channel, one read each.
    calls = spots.assign(channel=[CHANNELS[c] for c in spots.channel]).merge(
        DIRECT_MAPPING, on=["round", "channel"], how="left", validate="many_to_one")
    assert list(calls.gene_id) == ["Gfap", "Gad1", "Sst", "Aqp4"]
    assert list(calls.spot_id) == list(spots.spot_id)
    # Checked against current code: extracting only each candidate's own round gives its
    # detection channel as the brightest channel of that round.
    for name in rounds(2):
        own = fov.spot_result.spots["round"] == name
        result = fov.spot_result
        one_round = type(result)(result.spots[own].reset_index(drop=True), result.metadata,
                                 result.spot_namespace, result.config, {})
        loaded = {name: ImageLoadResult(fov.images[name], fov.metadata[name], CHANNELS, (), {})}
        values = extract_intensities(loaded, one_round).values
        assert values.shape[2] == 1
        assert list(values[:, :, 0].argmax(axis=1)) == list(one_round.spots.channel)
