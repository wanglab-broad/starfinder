"""The two_base encoding table: visible, configurable and recorded (W-304, docs/readout-contract.md,
"Encoding registry").

EncodingConfig.pair_to_color holds the 16 ordered base pairs mapped to the colors 1 to 4, with
the active table of src/matlab/EncodeBases.m as its default. Two changed tables are written out
here: RELABELLED, the second table kept as comments in EncodeBases.m (the default with the colors
relabelled 1 → 3, 2 → 2, 3 → 4, 4 → 1), and REPAIRED, which is not a relabelling of the default:
with A, C, G, T at positions 0 to 3 a pair has the color ((second − first) mod 4) + 1.

The checks: the default and the module functions are unchanged; invalid tables raise naming the
problem; both changed tables round-trip through ENCODINGS on 1,000 random barcodes per seed
(100 to 102); the relabelled golden fixture (codebook rebuilt under RELABELLED, image channels
permuted to match) and a hand-built REPAIRED scene decode, filter and explain like their
default-table runs, with no default-table code path reached; the table is visible through
Codebook, Dataset and FOV and in the summaries; the workflow key, the schema entry and the
recorded encoding of pre_qc.json and run.json, with the rerun check.
"""
from dataclasses import fields
import inspect
from itertools import permutations, product
import json
from pathlib import Path
import re

import jsonschema
import numpy as np
import pandas as pd
import pytest
import yaml

from starfinder.barcode import (ENCODINGS, BarcodeLayout, Codebook, EncodingConfig, EncodingSpec,
                                NeighborhoodSumConfig, OneBaseEncodingConfig, ReadFilterConfig, Segment,
                                WtaDecoderConfig, decode_barcodes, decode_color_sequence, encode_bases,
                                explain_read, extract_intensities, filter_reads, inspect_read, load_codebook)
from starfinder.barcode import _encoding, filtering
from starfinder.dataset import CheckpointConfig, Dataset, PipelineConfig, RoundState
from starfinder.dataset.workflow import from_workflow_config
from starfinder.image import ImageMetadata
from starfinder.io import ImageLoadResult
from starfinder.io._checkpoint import FORMAT_VERSION

from . import readout_fixtures as fx
from .test_readout_direct import direct_dataset, direct_fov
from .test_readout_direct import pipeline as direct_pipeline
from .test_readout_golden import (CHANNELS, METADATA, ROUNDS, candidates, golden_dataset, readout_config,
                                  write_codebook)

pytestmark = pytest.mark.barcode

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = yaml.safe_load((ROOT / "workflow/schemas/config.schema.yaml").read_text())
SEEDS = (100, 101, 102)
PAIRS = ["".join(p) for p in product("ACGT", repeat=2)]

DEFAULT = {"AA": "1", "CC": "1", "GG": "1", "TT": "1",
           "AC": "2", "CA": "2", "GT": "2", "TG": "2",
           "AG": "3", "CT": "3", "GA": "3", "TC": "3",
           "AT": "4", "CG": "4", "GC": "4", "TA": "4"}
# src/matlab/EncodeBases.m, the commented table (lines 41 to 60).
RELABELLED = {"AT": "1", "TA": "1", "GC": "1", "CG": "1",
              "AC": "2", "CA": "2", "GT": "2", "TG": "2",
              "AA": "3", "TT": "3", "GG": "3", "CC": "3",
              "AG": "4", "GA": "4", "CT": "4", "TC": "4"}
# ((second - first) mod 4) + 1 with A, C, G, T at positions 0 to 3.
REPAIRED = {"AA": "1", "CC": "1", "GG": "1", "TT": "1",
            "AC": "2", "CG": "2", "GT": "2", "TA": "2",
            "AG": "3", "CT": "3", "GA": "3", "TC": "3",
            "AT": "4", "CA": "4", "GC": "4", "TG": "4"}
# The relabelling that takes the default colors to RELABELLED's.
RELABEL = {"1": "3", "2": "2", "3": "4", "4": "1"}


def random_barcodes(seed, n=1000):
    """n random barcodes of 6 to 11 bases."""
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list("ACGT"), size=rng.integers(6, 12))) for _ in range(n)]


def matlab_tables():
    """(active, commented) pair tables of src/matlab/EncodeBases.m."""
    lines = (ROOT / "src/matlab/EncodeBases.m").read_text().splitlines()
    pattern = r"base_to_color\{'([ACGT]{2})'\} = '([1-4])';"
    active = dict(re.findall(pattern, "\n".join(l for l in lines if not l.lstrip().startswith("%"))))
    commented = dict(re.findall(pattern, "\n".join(l for l in lines if l.lstrip().startswith("%"))))
    return active, commented


# --- the default is unchanged ---------------------------------------------------------------------

@pytest.mark.contract
def test_the_default_table_is_the_active_matlab_table_and_the_module_functions_are_unchanged():
    active, commented = matlab_tables()
    assert len(active) == 16 and active == DEFAULT and commented == RELABELLED
    assert EncodingConfig() == EncodingConfig(pair_to_color=DEFAULT)
    assert EncodingConfig().pair_to_color == DEFAULT and list(EncodingConfig().pair_to_color) == PAIRS
    assert EncodingConfig(False, 2) == EncodingConfig(reverse_bases=False, split_index=2, pair_to_color=DEFAULT)
    assert str(inspect.signature(encode_bases)) == "(sequence: str) -> str"
    assert str(inspect.signature(decode_color_sequence)) == "(color_sequence: str, start_base: str) -> str"
    assert encode_bases("CGCAC") == "4422" and decode_color_sequence("4422", "C") == "CGCAC"
    two = ENCODINGS[EncodingConfig]
    for barcode in random_barcodes(100):
        colors = encode_bases(barcode)
        assert colors == "".join(DEFAULT[barcode[i:i + 2]] for i in range(len(barcode) - 1))
        assert decode_color_sequence(colors, barcode[0]) == barcode
        assert two.encode(barcode, EncodingConfig(reverse_bases=False)) == colors
        assert EncodingConfig().encode(barcode) == encode_bases(barcode[::-1])
    assert hash(EncodingConfig()) == hash(EncodingConfig(pair_to_color=DEFAULT))


# --- validation -----------------------------------------------------------------------------------

def _without(key):
    return {k: v for k, v in DEFAULT.items() if k != key}


@pytest.mark.contract
@pytest.mark.parametrize("table, message", [
    (_without("TT"), "exactly the 16 ordered pairs.*missing TT"),
    ({**DEFAULT, "AN": "1"}, "exactly the 16 ordered pairs.*unexpected 'AN'"),
    ({**_without("AC"), "ac": "2"}, "exactly the 16 ordered pairs.*missing AC; unexpected 'ac'"),
    ({**_without("AA"), ("A", "A"): "1"}, "exactly the 16 ordered pairs"),
    ({**DEFAULT, "GG": "5"}, r"colors must be the strings '1' to '4'; got \{'GG': '5'\}"),
    ({**DEFAULT, "GG": 1}, r"colors must be the strings '1' to '4'; got \{'GG': 1\}"),
    ({**DEFAULT, "CG": "M"}, "colors must be the strings '1' to '4'"),
    ({**DEFAULT, "CA": "1"}, "the four pairs that start with C four different colors.*CA 1, CC 1"),
    ({p: "1" for p in PAIRS}, "the four pairs that start with A four different colors"),
    ([("AA", "1")], "must be a mapping"),
])
def test_invalid_tables_raise_naming_the_problem(table, message):
    with pytest.raises(ValueError, match=message):
        EncodingConfig(pair_to_color=table)


@pytest.mark.contract
def test_the_table_may_be_given_in_any_order_and_is_stored_as_a_copy():
    given = dict(reversed(list(RELABELLED.items())))
    config = EncodingConfig(pair_to_color=given)
    assert config == EncodingConfig(pair_to_color=RELABELLED) and list(config.pair_to_color) == PAIRS
    assert config.pair_to_color is not given
    given["AT"] = "4"
    assert config.pair_to_color["AT"] == "1"
    assert config != EncodingConfig() and config != EncodingConfig(pair_to_color=REPAIRED)
    table = ENCODINGS[EncodingConfig].table(config)
    table["AT"] = "4"
    assert config.pair_to_color["AT"] == "1"


# --- round trips ----------------------------------------------------------------------------------

@pytest.mark.contract
def test_the_fixture_tables_are_a_relabelling_and_a_re_pairing():
    assert {p: RELABEL[c] for p, c in DEFAULT.items()} == RELABELLED
    assert REPAIRED == {a + b: str((("ACGT".index(b) - "ACGT".index(a)) % 4) + 1) for a, b in product("ACGT", repeat=2)}
    # No relabelling of the four colors turns the default into REPAIRED.
    for image in permutations("1234"):
        relabel = dict(zip("1234", image))
        assert {p: relabel[c] for p, c in DEFAULT.items()} != REPAIRED


@pytest.mark.validation
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("table", [RELABELLED, REPAIRED], ids=["relabelled", "repaired"])
def test_changed_tables_round_trip_through_encodings(seed, table):
    spec = ENCODINGS[EncodingConfig]
    for reverse_bases in (True, False):
        config = EncodingConfig(reverse_bases=reverse_bases, pair_to_color=table)
        for barcode in random_barcodes(seed):
            colors = spec.encode(barcode, config)
            read = barcode[::-1] if reverse_bases else barcode
            assert colors == "".join(table[read[i:i + 2]] for i in range(len(read) - 1))
            assert spec.decode(colors, config, read[0]) == barcode


# --- the relabelled golden fixture ----------------------------------------------------------------

def permuted(rounds):
    """The images with each default color's channel moved to its RELABELLED color's channel."""
    out = {}
    for label, image in rounds.items():
        moved = np.empty_like(image)
        for color, new in RELABEL.items():
            moved[..., int(new) - 1] = image[..., int(color) - 1]
        out[label] = moved
    return out


def golden_csv(root):
    """The golden gene,barcode file (written with the default table) and the allowed (first, last) ends of
    one 5-base segment: those of every golden entry, in read orientation."""
    root.mkdir(parents=True, exist_ok=True)
    path = write_codebook(root)
    barcodes = sorted(line.split(",")[1] for line in path.read_text().split())
    return path, tuple(dict.fromkeys((b[::-1][0], b[::-1][-1]) for b in barcodes))


def golden_run(root, seed, table, decoder, ends, csv):
    """FOV.run of the golden fixture with its codebook built under table (images permuted for RELABELLED);
    ends: "legacy" (ReadFilterConfig.end_bases "CC") or "layout" (one segment with declared ends)."""
    path, allowed = csv
    layout = None if ends == "legacy" else BarcodeLayout((Segment("A", 5, allowed),))
    dataset = golden_dataset(root)
    dataset.load_codebook(path, encoding=EncodingConfig(pair_to_color=table), layout=layout)
    fov = dataset.fov("FOV_001")
    images = fx.golden_rounds(seed)
    fov.images = images if table == DEFAULT else permuted(images)
    fov.metadata = {label: METADATA for label in ROUNDS}
    fov.spot_result = candidates()
    pipeline = readout_config("pipeline", decoder=decoder, end_bases="CC" if ends == "legacy" else None)
    fov.run(pipeline, checkpoints=CheckpointConfig(stages=("candidates", "pre_qc"), directory=root / "ck"))
    return dataset, fov


class _Cleared:
    """Within the block the default pair table is empty in place, in every module that imported it, so
    any default-table use (encode_bases, decode_color_sequence, a default EncodingConfig or the legacy
    end-base fallback of filter_reads) raises."""

    def __enter__(self):
        self.saved = [(d, dict(d)) for d in (_encoding._BASE_PAIR_TO_COLOR, _encoding._COLOR_TO_BASE_PAIRS)]
        for d, _ in self.saved:
            d.clear()

    def __exit__(self, *exc):
        for d, saved in self.saved:
            d.update(saved)


@pytest.mark.contract
def test_the_guard_catches_every_default_table_use():
    with _Cleared():
        with pytest.raises(KeyError):
            encode_bases("AC")
        with pytest.raises(KeyError):
            decode_color_sequence("1", "A")
        with pytest.raises(ValueError, match="pair_to_color"):
            EncodingConfig()
        with pytest.raises(KeyError):
            _encoding.decode_pairs("12", "C", filtering._BASE_PAIR_TO_COLOR)
    assert encode_bases("AC") == "2" and EncodingConfig().pair_to_color == DEFAULT


@pytest.mark.validation
@pytest.mark.parametrize("ends", ["legacy", "layout"])
@pytest.mark.parametrize("decoder", ["wta", "codebook_aware"])
@pytest.mark.parametrize("seed", fx.GOLDEN_SEEDS)
def test_relabelled_golden_decodes_filters_and_explains_like_the_default(tmp_path, seed, decoder, ends):
    csv = golden_csv(tmp_path)
    _, default = golden_run(tmp_path / "default", seed, DEFAULT, decoder, ends, csv)
    with _Cleared():
        dataset, changed = golden_run(tmp_path / "relabelled", seed, RELABELLED, decoder, ends, csv)
        explained = {spot: explain_read(changed.results, spot, intensity_result=changed.intensity_result,
                                        reference=dataset.codebook) for spot in changed.spot_result.spots.spot_id}
        inspected = {spot: inspect_read(changed.intensity_result, changed.results, spot, reference=dataset.codebook)
                     for spot in changed.spot_result.spots.spot_id}
    columns = ["spot_id", "gene_id", "entry_id", "call_status", "call_type"]
    pd.testing.assert_frame_equal(changed.decoding_result.table[columns], default.decoding_result.table[columns],
                                  check_exact=True)
    observed = changed.decoding_result.table.observed_color_sequence
    assert observed.tolist() == ["".join(RELABEL.get(c, c) for c in s) if isinstance(s, str) else s
                                 for s in default.decoding_result.table.observed_color_sequence]
    outcome = [c for c in changed.filtering_result.table if c.startswith("endpoint_valid")] + [
        "accepted", "rejection_reasons"]
    assert "endpoint_valid" in outcome
    pd.testing.assert_frame_equal(changed.filtering_result.table[outcome], default.filtering_result.table[outcome],
                                  check_exact=True)
    assert changed.filtering_result.counts == default.filtering_result.counts
    assert 0 < changed.filtering_result.table.endpoint_valid.sum() < len(changed.filtering_result.table)
    for spot, table in explained.items():
        reference = explain_read(default.results, spot, intensity_result=default.intensity_result,
                                 reference=default.dataset.codebook)
        a, b = (t[t.stage == "end_bases"][["item", "passed"]].reset_index(drop=True) for t in (table, reference))
        pd.testing.assert_frame_equal(a, b)
        a, b = (t.groupby("round")[["observed_bases", "assigned_bases"]].first() for t in (
            inspected[spot], inspect_read(default.intensity_result, default.results, spot,
                                          reference=default.dataset.codebook)))
        pd.testing.assert_frame_equal(a, b)


# --- the re-paired hand-built scene ---------------------------------------------------------------

# Eight 5-base barcodes (four colors), three amplicons each; read orientation (reversed) starts with C,
# and three of them also end with C.
SCENE_BARCODES = {"Gfap": "CAGTC", "Mbp": "GGTAC", "Sst": "ATCGC", "Gad1": "CTAAC",
                  "Aqp4": "TTCAC", "Olig2": "GACTC", "Pvalb": "CCGAC", "Snap25": "AGGTC"}


def scene_codebook(root, table):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "scene.csv"
    path.write_text("".join(f"{gene},{barcode}\n" for gene, barcode in SCENE_BARCODES.items()))
    return load_codebook(path, round_labels=fx.ROUNDS, channel_labels=fx.CHANNELS,
                         encoding=EncodingConfig(pair_to_color=table))


def scene(book, seed):
    """Images of four rounds (16×64×64, four channels) with the amplicons planted from book's colors."""
    genes = [gene for gene in SCENE_BARCODES for _ in range(3)]
    slots = fx.SLOTS[:len(genes)]
    colors = dict(zip(book.table.gene_id, book.table.color_sequence))
    rng = np.random.RandomState(seed)
    blobs = [fx._gaussian(slot) for slot in slots]
    rounds = {}
    for r, label in enumerate(book.round_labels):
        image = 20.0 + rng.normal(0.0, 3.0, fx.SHAPE_ZYX + (4,))
        for blob, gene in zip(blobs, genes):
            channel = book.color_to_channel[colors[gene][r]]
            image[..., channel] += fx.AMPLITUDE * blob
            image[..., (channel + 1) % 4] += fx.COPY_FRACTION * fx.AMPLITUDE * blob
        rounds[label] = np.clip(np.rint(image), 0, 65535).astype(np.uint16)
    truth = pd.DataFrame([dict(z=z, y=y, x=x, channel=0) for z, y, x in slots])
    return rounds, fx.candidates(truth, "scene"), genes


def scene_reads(book, seed):
    rounds, spots, genes = scene(book, seed)
    loaded = {label: ImageLoadResult(image, ImageMetadata("scene/FOV_001"), fx.CHANNELS, (), {})
              for label, image in rounds.items()}
    intensities = extract_intensities(loaded, spots, config=NeighborhoodSumConfig())
    decoded = decode_barcodes(intensities, book, config=WtaDecoderConfig())
    filtered = filter_reads(decoded, config=ReadFilterConfig(end_bases="CC", exclude_invalid_endpoints=True),
                            codebook=book)
    return decoded, filtered, genes


@pytest.mark.validation
@pytest.mark.parametrize("seed", SEEDS)
def test_repaired_scene_decodes_every_planted_spot_to_its_gene(tmp_path, seed):
    default, repaired = scene_codebook(tmp_path / "default", DEFAULT), scene_codebook(tmp_path / "repaired", REPAIRED)
    assert repaired.table.entry_id.tolist() == default.table.entry_id.tolist()
    differs = [a != b for a, b in zip(repaired.table.color_sequence, default.table.color_sequence)]
    assert any(differs)
    _, expected, _ = scene_reads(default, seed)
    with _Cleared():
        decoded, filtered, genes = scene_reads(repaired, seed)
    assert decoded.table.gene_id.tolist() == genes
    assert decoded.table.call_status.eq("assigned").all()
    outcome = ["endpoint_valid", "accepted", "rejection_reasons"]
    pd.testing.assert_frame_equal(filtered.table[outcome], expected.table[outcome], check_exact=True)
    # The reads whose barcode ends in C (read orientation C...C) pass the end-base check.
    ends = [SCENE_BARCODES[g][::-1][-1] == "C" for g in genes]
    assert filtered.table.endpoint_valid.tolist() == ends and 0 < sum(ends) < len(ends)


# --- visibility -----------------------------------------------------------------------------------

def book(encoding, color_to_channel=None, n_rounds=4):
    rounds = tuple(f"r{i}" for i in range(1, n_rounds + 1))
    return Codebook(pd.DataFrame(columns=["gene_id", "color_sequence"]), rounds, ("a", "b", "c", "d"),
                    color_to_channel or {"1": 0, "2": 1, "3": 2, "4": 3}, encoding)


@pytest.mark.contract
def test_codebook_encoding_table_has_the_mapping_and_channels():
    mapping = {"1": 2, "2": 0, "3": 3, "4": 1}
    for table in (DEFAULT, RELABELLED, REPAIRED):
        shown = book(EncodingConfig(pair_to_color=table), mapping).encoding_table()
        assert list(shown.columns) == ["bases", "color", "channel"] and len(shown) == 16
        assert dict(zip(shown.bases, shown.color)) == table and shown.bases.tolist() == PAIRS
        assert shown.channel.tolist() == ["abcd"[mapping[c]] for c in shown.color]
        assert all(str(t) == "string" for t in shown.dtypes)
    colors = {"A": "3", "C": "1", "G": "4", "T": "2"}
    shown = book(OneBaseEncodingConfig(colors), mapping).encoding_table()
    assert len(shown) == 4 and dict(zip(shown.bases, shown.color)) == colors
    assert shown.channel.tolist() == ["abcd"[mapping[c]] for c in shown.color]
    for config in (EncodingConfig(pair_to_color=REPAIRED), OneBaseEncodingConfig(colors)):
        assert ENCODINGS[type(config)].table(config) == book(config).encoding_table().set_index(
            "bases").color.to_dict()
    assert "table" in {f.name for f in fields(EncodingSpec)}


@pytest.mark.contract
def test_dataset_and_fov_encoding_table(tmp_path):
    dataset = golden_dataset(tmp_path)
    with pytest.raises(ValueError, match="no codebook is loaded"):
        dataset.encoding_table()
    with pytest.raises(ValueError, match="no codebook is loaded"):
        dataset.fov("FOV_001").encoding_table()
    dataset.load_codebook(write_codebook(tmp_path), encoding=EncodingConfig(pair_to_color=RELABELLED))
    expected = dataset.codebook.encoding_table()
    pd.testing.assert_frame_equal(dataset.encoding_table(), expected)
    pd.testing.assert_frame_equal(dataset.fov("FOV_001").encoding_table(), expected)
    assert dict(zip(expected.bases, expected.color)) == RELABELLED
    direct = direct_dataset(tmp_path / "direct")
    direct.codebook = dataset.codebook
    for owner in (direct, direct.fov("FOV_001")):
        with pytest.raises(ValueError, match="readout_mode='direct' has no barcode encoding"):
            owner.encoding_table()


@pytest.mark.contract
def test_summaries_name_the_encoding_and_the_segment_layout(tmp_path):
    two = BarcodeLayout((Segment("A", 6), Segment("B", 5)), ("A", "B"))
    cases = [(book(EncodingConfig()), "two_base, one segment of 4 colors"),
             (book(EncodingConfig(pair_to_color=RELABELLED)),
              "two_base, one segment of 4 colors, non-default pair_to_color table"),
             (Codebook(pd.DataFrame(columns=["gene_id", "color_sequence"]), tuple(f"r{i}" for i in range(9)),
                       ("a", "b", "c", "d"), layout=two), "two_base, two segments of 5 and 4 colors"),
             (book(EncodingConfig(split_index=4), n_rounds=9), "two_base, two segments of 5 and 4 colors"),
             (book(OneBaseEncodingConfig({"A": "1", "C": "2", "G": "3", "T": "4"})),
              "one_base, one segment of 4 colors")]
    for codebook, text in cases:
        assert [line for line in repr(codebook).splitlines() if "encoding" in line] == [
            repr(codebook)] and repr(codebook).endswith(f"; encoding {text}")
    dataset = golden_dataset(tmp_path)
    assert "encoding" not in repr(dataset)
    dataset.load_codebook(write_codebook(tmp_path), encoding=EncodingConfig(pair_to_color=REPAIRED))
    lines = [line for line in repr(dataset).splitlines() if "encoding" in line]
    assert lines == ["    encoding:          two_base, one segment of 4 colors, non-default pair_to_color table"]


# --- workflow configuration -----------------------------------------------------------------------

def workflow(load_codebook, backend="python"):
    config = {"config_path": "config.yaml", "starfinder_path": str(ROOT), "root_input_path": "in",
              "root_output_path": "out", "dataset_id": "d", "sample_id": "s", "output_id": "o",
              "fov_id_pattern": "%s", "n_fovs": 1, "n_rounds": 4, "ref_round": "round1", "rotate_angle": 0,
              "img_col": 48, "img_row": 48, "subset_range": False, "subset_start": 1, "subset_end": 1,
              "subset_random": False, "n_random_tests": 1, "seq_channel_order": list(CHANNELS),
              "rules": {"rsf_single_fov": {"run": True, "parameters": {
                  "load_raw_images": {"run": False},
                  "load_codebook": {"run": True, **load_codebook},
                  "reads_filtration": {"run": True}}}}}
    if backend is not None:
        config["backend"] = backend
    return config


YAML_TABLE = "\n".join(f"        {pair}: {color}" for pair, color in RELABELLED.items())
YAML_CONFIG = f"""
load_codebook:
  encoding:
    method: two_base
    reverse_bases: true
    pair_to_color:
{YAML_TABLE}
"""


@pytest.mark.contract
def test_yaml_pair_to_color_gives_the_python_codebook(tmp_path):
    block = yaml.safe_load(YAML_CONFIG.replace("\n        ", "\n      "))["load_codebook"]
    assert block["encoding"]["pair_to_color"]["AT"] == 1  # unquoted YAML colors are integers
    adapted = from_workflow_config(workflow(block))
    assert adapted.encoding == EncodingConfig(pair_to_color=RELABELLED)
    path = write_codebook(tmp_path)
    adapted.dataset.load_codebook(path, encoding=adapted.encoding, layout=adapted.layout)
    python = load_codebook(path, round_labels=ROUNDS, channel_labels=CHANNELS,
                           encoding=EncodingConfig(pair_to_color=RELABELLED))
    pd.testing.assert_frame_equal(adapted.dataset.codebook.table, python.table, check_exact=True)
    assert adapted.dataset.codebook.encoding == python.encoding and adapted.dataset.codebook.layout == python.layout
    quoted = {k: str(v) for k, v in block["encoding"]["pair_to_color"].items()}
    assert from_workflow_config(workflow({"encoding": {"pair_to_color": quoted}})).encoding == python.encoding
    assert from_workflow_config(workflow({})).encoding == EncodingConfig()
    assert from_workflow_config(workflow({"encoding": {"method": "two_base"}})).encoding == EncodingConfig()
    jsonschema.validate(workflow(block), SCHEMA)
    jsonschema.validate(workflow({"encoding": {"pair_to_color": quoted}}), SCHEMA)


@pytest.mark.contract
def test_invalid_yaml_tables_raise_the_python_error():
    for table in ({**RELABELLED, "CC": "1"}, _without("TT"), {**RELABELLED, "AA": "7"}):
        with pytest.raises(ValueError) as python:
            EncodingConfig(pair_to_color=table)
        with pytest.raises(ValueError) as adapter:
            from_workflow_config(workflow({"encoding": {"pair_to_color": table}}))
        assert str(adapter.value) == str(python.value)


@pytest.mark.contract
def test_schema_accepts_pair_to_color_for_the_python_backend_only():
    block = {"encoding": {"pair_to_color": RELABELLED}}
    jsonschema.validate(workflow(block), SCHEMA)
    jsonschema.validate(workflow({"encoding": {"reverse_bases": True}}, backend="matlab"), SCHEMA)
    description = SCHEMA["$defs"]["load_codebook_params"]["properties"]["encoding"]["properties"][
        "pair_to_color"]["description"]
    assert description.startswith("Python only")
    for invalid in (workflow(block, backend="matlab"), workflow(block, backend=None),
                    workflow({"encoding": {"pair_to_color": _without("TT")}}),
                    workflow({"encoding": {"pair_to_color": {**RELABELLED, "AA": "5"}}}),
                    workflow({"encoding": {"pair_to_color": {**RELABELLED, "AN": "1"}}})):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, SCHEMA)


# --- provenance -----------------------------------------------------------------------------------

def recorded(table, reverse_bases=True):
    """The recorded two_base encoding (the table in AA, AC, ..., TT order, as stored)."""
    return {"method": "two_base", "reverse_bases": reverse_bases, "pair_to_color": {p: table[p] for p in PAIRS}}


def other_dataset(root, encoding, color_sequences=False):
    """The golden dataset with its codebook loaded under encoding (color_sequences: the golden color
    sequences as a canonical file, for one_base)."""
    root.mkdir(parents=True, exist_ok=True)
    dataset = golden_dataset(root)
    if color_sequences:
        path = root / "canonical.csv"
        path.write_text("gene_id,color_sequence\n" + "".join(
            f"Gene{c},{c}\n" for c in ("1234", "2143", "3412", "4321", "2222", "3131", "4242", "1313")))
        dataset.load_codebook(path, encoding=encoding)
    else:
        dataset.load_codebook(write_codebook(root), encoding=encoding)
    return dataset


def checkpoint_run(root, seed, encoding, color_sequences=False):
    """FOV.run of the golden fixture with checkpoints and the codebook under encoding (other_dataset)."""
    dataset = other_dataset(root, encoding, color_sequences)
    fov = dataset.fov("FOV_001")
    images = fx.golden_rounds(seed)
    fov.images = permuted(images) if getattr(encoding, "pair_to_color", None) == RELABELLED else images
    fov.metadata = {label: METADATA for label in ROUNDS}
    fov.spot_result = candidates()
    checkpoints = CheckpointConfig(stages=("candidates", "pre_qc"), directory=root / "ck")
    fov.run(readout_config("pipeline"), checkpoints=checkpoints)
    return dataset, fov, checkpoints, root / "ck" / "FOV_001"


@pytest.mark.validation
@pytest.mark.parametrize("seed", SEEDS)
def test_pre_qc_and_run_json_record_the_encoding_and_reruns_check_it(tmp_path, seed):
    dataset, fov, checkpoints, directory = checkpoint_run(tmp_path / "run", seed,
                                                          EncodingConfig(pair_to_color=RELABELLED))
    header = json.loads((directory / "pre_qc.json").read_text())
    assert header["format_version"] == FORMAT_VERSION == 2
    assert header["encoding"] == recorded(RELABELLED)
    assert header["layout"]["segments"] == [{"name": "A", "bases": 5, "ends": []}]
    assert "pair_to_color" not in header["decoding_config"]
    assert "encoding" not in json.loads((directory / "candidates.json").read_text())
    run = json.loads((directory / "run.json").read_text())
    assert run["config"]["encoding"] == recorded(RELABELLED) and run["config"]["readout_mode"] == "multiplexed"
    # The same encoding reloads; a different one raises naming both.
    reloaded = dataset.fov("FOV_001").load_checkpoint("pre_qc", checkpoints=checkpoints)
    pd.testing.assert_frame_equal(reloaded.decoding_result.table, fov.decoding_result.table, check_exact=True)
    for encoding in (EncodingConfig(), EncodingConfig(reverse_bases=False, pair_to_color=RELABELLED),
                     EncodingConfig(pair_to_color=REPAIRED)):
        other = other_dataset(tmp_path / "other", encoding)
        with pytest.raises(ValueError, match="pre_qc checkpoint encoding") as error:
            other.fov("FOV_001").load_checkpoint("pre_qc", checkpoints=checkpoints)
        message = str(error.value)
        assert str(recorded(RELABELLED)) in message
        assert str(recorded(encoding.pair_to_color, encoding.reverse_bases)) in message
    # Candidates carry no encoding: a rerun from them decodes with the loaded codebook.
    other = other_dataset(tmp_path / "other", EncodingConfig())
    assert other.fov("FOV_001").load_checkpoint("candidates", checkpoints=checkpoints).spot_result is not None


@pytest.mark.validation
@pytest.mark.parametrize("seed", SEEDS)
def test_a_checkpoint_without_the_encoding_key_loads_as_today(tmp_path, seed):
    dataset, fov, checkpoints, directory = checkpoint_run(tmp_path / "run", seed,
                                                          EncodingConfig(pair_to_color=RELABELLED))
    path = directory / "pre_qc.json"
    header = json.loads(path.read_text())
    header.pop("encoding")
    path.write_text(json.dumps(header, indent=2) + "\n")
    # Written at 9aeb220 or 141c093: no key, no check, whatever the loaded codebook.
    for encoding in (EncodingConfig(pair_to_color=RELABELLED), EncodingConfig()):
        other = other_dataset(tmp_path / "other", encoding)
        reloaded = other.fov("FOV_001").load_checkpoint("pre_qc", checkpoints=checkpoints)
        pd.testing.assert_frame_equal(reloaded.decoding_result.table, fov.decoding_result.table, check_exact=True)
        pd.testing.assert_frame_equal(reloaded.scoring_result.table, fov.scoring_result.table, check_exact=True)
    # Without a loaded codebook the recorded encoding is not checked either.
    header["encoding"] = recorded(REPAIRED)
    path.write_text(json.dumps(header, indent=2) + "\n")
    bare = golden_dataset(tmp_path / "bare")
    assert bare.fov("FOV_001").load_checkpoint("pre_qc", checkpoints=checkpoints).decoding_result is not None


@pytest.mark.validation
@pytest.mark.parametrize("seed", SEEDS)
def test_one_base_records_its_encoding_too(tmp_path, seed):
    colors = {"A": "1", "C": "2", "G": "3", "T": "4"}
    dataset, fov, checkpoints, directory = checkpoint_run(tmp_path / "run", seed, OneBaseEncodingConfig(colors),
                                                          color_sequences=True)
    expected = {"method": "one_base", "base_to_color": colors, "reverse_bases": False}
    assert json.loads((directory / "pre_qc.json").read_text())["encoding"] == expected
    assert json.loads((directory / "run.json").read_text())["config"]["encoding"] == expected
    reloaded = dataset.fov("FOV_001").load_checkpoint("pre_qc", checkpoints=checkpoints)
    pd.testing.assert_frame_equal(reloaded.decoding_result.table, fov.decoding_result.table, check_exact=True)
    for encoding in (OneBaseEncodingConfig({"A": "2", "C": "1", "G": "3", "T": "4"}),
                     OneBaseEncodingConfig(colors, reverse_bases=True)):
        other = other_dataset(tmp_path / "other", encoding, color_sequences=True)
        with pytest.raises(ValueError, match="pre_qc checkpoint encoding") as error:
            other.fov("FOV_001").load_checkpoint("pre_qc", checkpoints=checkpoints)
        assert str(expected) in str(error.value) and "'one_base'" in str(error.value)


@pytest.mark.validation
@pytest.mark.parametrize("seed", SEEDS)
def test_direct_mode_records_no_encoding(tmp_path, seed):
    dataset = direct_dataset(tmp_path)
    checkpoints = CheckpointConfig(stages=("candidates", "pre_qc"), directory=tmp_path / "ck")
    direct_fov(dataset, seed).run(direct_pipeline(), checkpoints=checkpoints)
    assert json.loads((tmp_path / "ck" / "FOV_001" / "pre_qc.json").read_text())["encoding"] is None
    assert json.loads((tmp_path / "ck" / "FOV_001" / "run.json").read_text())["config"]["encoding"] is None
