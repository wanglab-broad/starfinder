"""The §2.8 encoding and decoding registries (W-292, docs/readout-contract.md, "Encoding registry").

Check R1 of docs/readout-algorithms.md: round trips of both encodings on random barcodes,
the two_base pair table against src/matlab/EncodeBases.m, and a non-bijective one_base
mapping. The registry tests check the shared fields, the exact-type lookup and the
decoder's declared encoding kinds.
"""
import re
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from starfinder._registry import config_type_for, names
from starfinder.barcode import (DECODING_METHODS, ENCODINGS, Codebook, CodebookAwareDecoderConfig, DecodingSpec,
                                EncodingConfig, EncodingSpec, OneBaseEncodingConfig, WtaDecoderConfig,
                                decode_barcodes)

from .barcode_cases import CHANNELS, intensity, tensor

pytestmark = pytest.mark.barcode

ROOT = Path(__file__).resolve().parents[3]
ONE_BASE_COLORS = {"A": "1", "C": "2", "G": "3", "T": "4"}


def random_barcodes(seed, n=1000):
    """n random barcodes of 6 to 11 bases."""
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list("ACGT"), size=rng.integers(6, 12))) for _ in range(n)]


def read_first_base(barcode, config):
    return (barcode[::-1] if config.reverse_bases else barcode)[0]


@pytest.mark.validation
@pytest.mark.parametrize("seed", [100, 101, 102])
def test_r1_round_trips_are_exact(seed):
    configs = [EncodingConfig(reverse_bases=True), EncodingConfig(reverse_bases=False),
               OneBaseEncodingConfig(ONE_BASE_COLORS), OneBaseEncodingConfig(ONE_BASE_COLORS, reverse_bases=True),
               OneBaseEncodingConfig({"A": "3", "C": "1", "G": "4", "T": "2"})]
    for barcode in random_barcodes(seed):
        for config in configs:
            spec = ENCODINGS[type(config)]
            colors = spec.encode(barcode, config)
            assert len(colors) == spec.colors_for(len(barcode))
            first = read_first_base(barcode, config) if spec.needs_first_base else None
            assert spec.decode(colors, config, first) == barcode


@pytest.mark.validation
def test_r1_two_base_pairs_equal_matlab_encode_bases():
    source = (ROOT / "src/matlab/EncodeBases.m").read_text()
    # The active table only: lines that are not comments.
    active = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))
    matlab = dict(re.findall(r"base_to_color\{'([ACGT]{2})'\} = '([1-4])';", active))
    assert len(matlab) == 16
    spec, config = ENCODINGS[EncodingConfig], EncodingConfig(reverse_bases=False)
    assert {"".join(pair): spec.encode("".join(pair), config) for pair in product("ACGT", repeat=2)} == matlab


@pytest.mark.validation
@pytest.mark.parametrize("mapping", [
    {"A": "1", "C": "1", "G": "3", "T": "4"},      # two bases, one color
    {"A": "1", "C": "2", "G": "3"},                # a missing base
    {"A": "1", "C": "2", "G": "3", "T": "5"},      # a color outside 1-4
    {"A": "1", "C": "2", "G": "3", "T": "4", "N": "1"},
    {"A": 1, "C": 2, "G": 3, "T": 4},              # colors are symbols, not integers
])
def test_r1_a_non_bijective_base_to_color_raises(mapping):
    with pytest.raises(ValueError, match="one-to-one"):
        OneBaseEncodingConfig(mapping)


@pytest.mark.contract
def test_encoding_registry_names_and_capabilities():
    assert names(ENCODINGS) == ("two_base", "one_base")
    for config_type, spec in ENCODINGS.items():
        assert config_type_for(ENCODINGS, spec.name, "encoding") is config_type
        assert (spec.alphabet, spec.symbols) == ("1234", "color")
    two, one = ENCODINGS[EncodingConfig], ENCODINGS[OneBaseEncodingConfig]
    assert (two.needs_first_base, two.junction_colors, two.colors_for(11)) == (True, 1, 10)
    assert (one.needs_first_base, one.junction_colors, one.colors_for(11)) == (False, 0, 11)
    assert EncodingConfig().method == "two_base" and OneBaseEncodingConfig(ONE_BASE_COLORS).method == "one_base"
    # The positional constructor of EncodingConfig is unchanged.
    assert EncodingConfig(False, 2) == EncodingConfig(reverse_bases=False, split_index=2)


@pytest.mark.contract
def test_decoding_registry_names_and_capabilities():
    # W-293 adds the direct entry, which supports readout mode direct and no encoding.
    assert names(DECODING_METHODS) == ("wta", "codebook_aware", "direct")
    for config_type, spec in DECODING_METHODS.items():
        assert config_type().method == spec.name
        if spec.name == "direct":
            assert spec.modes == frozenset({"direct"}) and spec.encodings == frozenset() and not spec.rescue
        else:
            assert spec.modes == frozenset({"multiplexed"}) and spec.encodings == frozenset({"color"})
    assert DECODING_METHODS[WtaDecoderConfig].rescue is False
    assert DECODING_METHODS[CodebookAwareDecoderConfig].rescue is True
    assert DECODING_METHODS[WtaDecoderConfig].score_columns == ("wta_l2_nll",)


@pytest.mark.contract
@pytest.mark.parametrize("config", [WtaDecoderConfig(), CodebookAwareDecoderConfig()], ids=["wta", "codebook_aware"])
def test_declared_score_columns_are_the_numeric_columns_the_decoder_writes(config):
    book = Codebook(pd.DataFrame({"gene_id": ["A", "B"], "color_sequence": ["1234", "4321"]}),
                    tuple(f"r{i}" for i in range(4)), CHANNELS)
    table = decode_barcodes(intensity(tensor(["1234", "4321"])), book, config=config).table
    numeric = [c for c in table.columns if pd.api.types.is_float_dtype(table[c])]
    assert sorted(numeric) == sorted(DECODING_METHODS[type(config)].score_columns)


@pytest.mark.contract
def test_one_base_codebook_decodes_with_both_decoders():
    rounds = tuple(f"r{i}" for i in range(4))
    path_rows = pd.DataFrame({"gene_id": ["Mbp", "Gfap"], "base_sequence": ["GATC", "CCTA"],
                              "color_sequence": ["3142", "2241"]})
    book = Codebook(path_rows, rounds, CHANNELS, encoding=OneBaseEncodingConfig(ONE_BASE_COLORS))
    assert book.layout.segments[0].bases == 4
    for config in (WtaDecoderConfig(), CodebookAwareDecoderConfig()):
        table = decode_barcodes(intensity(tensor(["3142", "2241"])), book, config=config).table
        assert table.gene_id.tolist() == ["Mbp", "Gfap"] and table.entry_id.tolist() == ["3142", "2241"]
    with pytest.raises(ValueError, match="disagrees"):
        Codebook(path_rows.assign(base_sequence=["GATC", "CCTT"]), rounds, CHANNELS,
                 encoding=OneBaseEncodingConfig(ONE_BASE_COLORS))


@dataclass(frozen=True)
class BinaryFixtureConfig:
    """A fixture encoding of another symbol kind (as a binary on/off code would be)."""
    reverse_bases: bool = False
    method: str = field(default="binary_fixture", init=False)

    def __post_init__(self):
        pass


@pytest.mark.contract
def test_decoding_a_codebook_of_an_undeclared_encoding_kind_raises_type_error(monkeypatch):
    one_base = ENCODINGS[OneBaseEncodingConfig]
    spec = replace(one_base, name="binary_fixture", symbols="binary",
                   encode=lambda bases, config: one_base.encode(bases, OneBaseEncodingConfig(ONE_BASE_COLORS)))
    monkeypatch.setitem(ENCODINGS, BinaryFixtureConfig, spec)
    book = Codebook(pd.DataFrame({"gene_id": ["A"], "color_sequence": ["1234"]}),
                    tuple(f"r{i}" for i in range(4)), CHANNELS, encoding=BinaryFixtureConfig())
    for config in (WtaDecoderConfig(), CodebookAwareDecoderConfig()):
        with pytest.raises(TypeError, match="does not decode 'binary' encodings"):
            decode_barcodes(intensity(tensor(["1234"])), book, config=config)


@pytest.mark.contract
def test_unregistered_configs_raise_type_error():
    book = Codebook(pd.DataFrame({"gene_id": ["A"], "color_sequence": ["1234"]}),
                    tuple(f"r{i}" for i in range(4)), CHANNELS)

    @dataclass(frozen=True)
    class Subclass(WtaDecoderConfig):
        pass

    with pytest.raises(TypeError, match="unsupported decoder config"):
        decode_barcodes(intensity(tensor(["1234"])), book, config=Subclass())
    with pytest.raises(TypeError, match="ENCODINGS"):
        Codebook(book.table, book.round_labels, CHANNELS, encoding=BinaryFixtureConfig())


@pytest.mark.contract
def test_spec_fields_are_validated():
    two = ENCODINGS[EncodingConfig]
    with pytest.raises(ValueError, match="snake_case"):
        replace(two, name="Two-Base")
    with pytest.raises(ValueError, match="junction_colors"):
        replace(two, junction_colors=-1)
    wta = DECODING_METHODS[WtaDecoderConfig]
    with pytest.raises(TypeError, match="modes"):
        replace(wta, modes={"multiplexed"})
    with pytest.raises(TypeError, match="rescue"):
        replace(wta, rescue=1)
    assert isinstance(two, EncodingSpec) and isinstance(wta, DecodingSpec)


@pytest.mark.contract
def test_encodings_reject_invalid_inputs():
    two, one = ENCODINGS[EncodingConfig], ENCODINGS[OneBaseEncodingConfig]
    with pytest.raises(ValueError, match="at least 2"):
        two.encode("A", EncodingConfig())
    with pytest.raises(ValueError, match="A/C/G/T"):
        one.encode("ACNT", OneBaseEncodingConfig(ONE_BASE_COLORS))
    with pytest.raises(ValueError, match="first base"):
        two.decode("123", EncodingConfig(), None)
    with pytest.raises(ValueError, match="1234"):
        one.decode("12M4", OneBaseEncodingConfig(ONE_BASE_COLORS), None)
