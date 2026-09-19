"""Canonical codebook validation, encoding and public API inventory."""

from pathlib import Path
import importlib
import numpy as np
import pandas as pd
import pytest
from starfinder.barcode import (
    Codebook,
    EncodingConfig,
    load_codebook,
    encode_bases,
    decode_color_sequence,
    decode_barcodes,
    WtaDecoderConfig,
    filter_reads,
)
from starfinder.synthetic._presets import _TEST_CODEBOOK
from starfinder.barcode import EncodingConfig
from .barcode_cases import CHANNELS, intensity, tensor

ROUNDS = tuple(f"r{i}" for i in range(4))
SMALL = Path(__file__).resolve().parents[3] / "tests/fixtures/synthetic/small/codebook.csv"


def test_encoding_roundtrips_and_synthetic_reversal():
    assert encode_bases("CGCAC") == "4422"
    assert decode_color_sequence("4422", "C") == "CGCAC"
    for gene, bases in _TEST_CODEBOOK:
        seq = EncodingConfig().encode(bases)
        assert seq == EncodingConfig(reverse_bases=True).encode(bases) == encode_bases(bases[::-1])
        assert decode_color_sequence(seq, bases[-1]) == bases[::-1]
    assert EncodingConfig(False).encode("CACGC") == "2244"
    bases = "ACGTAC"
    raw = encode_bases(bases)
    assert EncodingConfig(False, 2).encode(bases) == raw[3:] + raw[:2]


def test_load_canonical_order_and_ground_truth_calls():
    cb = load_codebook(SMALL, round_labels=ROUNDS, channel_labels=CHANNELS)
    assert isinstance(cb, Codebook) and cb.n_genes == 8
    assert cb.gene_to_seq["GeneA"] == "4422"
    assert cb.seq_to_gene["4422"] == "GeneA"
    assert cb.genes == [g for g, _ in _TEST_CODEBOOK]
    result = decode_barcodes(
        intensity(tensor(cb.table.color_sequence.tolist()), rounds=ROUNDS),
        cb,
        config=WtaDecoderConfig(),
    )
    assert filter_reads(result).accepted.gene_id.tolist() == cb.genes


@pytest.mark.parametrize("header", ["", "gene,barcode\n", "\ufeffgene,barcode\n"])
def test_csv_headers_and_bom(tmp_path, header):
    path = tmp_path / "book.csv"
    path.write_text(header + "A,CACGC\nB,CATGC\n")
    cb = load_codebook(path, round_labels=ROUNDS, channel_labels=CHANNELS)
    assert cb.genes == ["A", "B"] and cb.gene_to_seq["A"] == "4422"


@pytest.mark.parametrize(
    "rows",
    [
        "A,CACGC\nA,CATGC\n",
        "A,CACGC\nB,CACGC\n",
        "A,CACGC\nA,CACGC\n",
        "A,ACNTG\n",
        "A,AC\n",
        ",CACGC\n",
    ],
)
def test_invalid_csv_has_row_context(tmp_path, rows):
    path = tmp_path / "bad.csv"
    path.write_text("gene,barcode\n" + rows)
    with pytest.raises(ValueError, match="row [23]"):
        load_codebook(path, round_labels=ROUNDS, channel_labels=CHANNELS)


@pytest.mark.parametrize("split", [-1, 0, True, 1.5])
def test_invalid_split_config(split):
    with pytest.raises(ValueError):
        EncodingConfig(split_index=split)


def test_split_range_and_mapping_errors(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("A,CACGC\n")
    with pytest.raises(ValueError, match="row 1.*split"):
        load_codebook(
            path,
            round_labels=ROUNDS,
            channel_labels=CHANNELS,
            encoding=EncodingConfig(split_index=3),
        )
    for mapping in ({"1": 0}, dict(zip("1234", [0, 0, 2, 3])), dict(zip("1234", [0, 1, 2, 4]))):
        with pytest.raises(ValueError, match="map"):
            load_codebook(
                SMALL, round_labels=ROUNDS, channel_labels=CHANNELS, color_to_channel=mapping
            )


def test_no_old_public_interfaces():
    import starfinder.barcode as barcode
    import starfinder.dataset as dataset

    for name in (
        "extract_from_location",
        "extract_intensity_tensor",
        "decode_codebook_aware",
        "decode_color_seq",
        "BASE_PAIR_TO_COLOR",
        "COLOR_TO_CHANNEL",
        "build_one_error_index",
        "candidate_sequences",
        "score_candidates",
        "channel_probabilities",
        "wta_color_sequences",
    ):
        assert not hasattr(barcode, name)
    assert not hasattr(dataset, "Codebook")
    for module in ("starfinder.barcode.encoding", "starfinder.barcode.codebook_aware"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)


def test_distinct_bases_with_identical_encoding_collide(tmp_path):
    path = tmp_path / "collision.csv"
    path.write_text("A,AA\nB,CC\n")
    with pytest.raises(ValueError, match="row 2.*collision"):
        load_codebook(path, round_labels=("r",), channel_labels=CHANNELS)


def test_canonical_csv_and_optional_bases_validated(tmp_path):
    path = tmp_path / "canonical.csv"
    path.write_text("gene_id,color_sequence,base_sequence\nA,4422,CACGC\n")
    cb = load_codebook(path, round_labels=ROUNDS, channel_labels=CHANNELS)
    assert cb.gene_to_seq == {"A": "4422"}
    path.write_text("gene_id,color_sequence,base_sequence\nA,4422,CATGC\n")
    with pytest.raises(ValueError, match="row 2.*disagrees"):
        load_codebook(path, round_labels=ROUNDS, channel_labels=CHANNELS)
