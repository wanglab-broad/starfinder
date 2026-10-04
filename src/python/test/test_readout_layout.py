"""The §2.8 segment layout and its workflow translation (W-292, docs/readout-contract.md, "Segment layout").

The split_index fix: the shared load_codebook.split_index is MATLAB's one-based position,
so [5] on the 11-base example CAGTACTGCAT gives 242324242 (141c093 gave 423242423; see
docs/readout-baseline.md). Check R2 (layout part) of docs/readout-algorithms.md: the
translated BarcodeLayout gives the codebook of EncodingConfig(split_index=s - 1) for every
valid split, with and without reverse_bases; the reads_filtration keys must agree with the
layout; a list end_base gives the allowed ends per segment. The schema's decoding-method
and encoding lists stay equal to the registries.
"""
import copy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from starfinder._registry import names
from starfinder.barcode import (DECODING_METHODS, ENCODINGS, BarcodeLayout, Codebook, CodebookAwareDecoderConfig,
                                EncodingConfig, OneBaseEncodingConfig, ReadFilterConfig, Segment, WtaDecoderConfig,
                                decode_barcodes, filter_reads, load_codebook)
from starfinder.dataset.workflow import from_workflow_config

from .barcode_cases import intensity, tensor

pytestmark = pytest.mark.barcode

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = yaml.safe_load((ROOT / "workflow/schemas/config.schema.yaml").read_text())
CHANNELS = ("ch00", "ch01", "ch02", "ch03")
BARCODE = "CAGTACTGCAT"  # W-279 worked example 2: segment A CAGTAC + segment B TGCAT


def workflow(n_rounds, *, load_codebook=None, reads_filtration=None, **parameters):
    return {"n_rounds": n_rounds, "ref_round": "round1", "dataset_id": "d", "sample_id": "s", "output_id": "o",
            "root_input_path": "in", "root_output_path": "out", "seq_channel_order": list(CHANNELS),
            "rules": {"rsf_single_fov": {"parameters": {
                "load_raw_images": {"run": False},
                "load_codebook": {"run": True, **(load_codebook or {})},
                "reads_filtration": {"run": True, **(reads_filtration or {})}, **parameters}}}}


def write(tmp_path, rows):
    path = tmp_path / "genes.csv"
    path.write_text("".join(f"{gene},{barcode}\n" for gene, barcode in rows))
    return path


def rounds(n):
    return tuple(f"round{i}" for i in range(1, n + 1))


# --- the split_index fix -------------------------------------------------------------

@pytest.mark.contract
def test_the_shared_split_index_is_one_based(tmp_path):
    adapted = from_workflow_config(workflow(9, load_codebook={"split_index": [5]}))
    dataset = adapted.dataset
    dataset.load_codebook(write(tmp_path, [("Gfap_probe1", BARCODE)]), split_index=adapted.split_index)
    assert dataset.codebook.table.color_sequence.tolist() == ["242324242"]
    assert adapted.split_index == 4
    # The workflow's own route: the translated encoding and layout.
    dataset.load_codebook(write(tmp_path, [("Gfap_probe1", BARCODE)]), encoding=adapted.encoding,
                          layout=adapted.layout)
    assert dataset.codebook.table.color_sequence.tolist() == ["242324242"]
    assert adapted.layout == BarcodeLayout((Segment("A", 6), Segment("B", 5)), ("A", "B"))


# --- R2: the translated layout equals the zero-based legacy split --------------------

@pytest.mark.validation
@pytest.mark.parametrize("reverse_bases", [True, False])
@pytest.mark.parametrize("split", range(2, 10))
def test_r2_translated_layout_equals_the_legacy_split(tmp_path, split, reverse_bases):
    n = len(BARCODE)
    adapted = from_workflow_config(workflow(n - 2, load_codebook={
        "split_index": [split], "encoding": {"method": "two_base", "reverse_bases": reverse_bases}}))
    assert adapted.encoding == EncodingConfig(reverse_bases=reverse_bases)
    path = write(tmp_path, [("Gfap_probe1", BARCODE), ("Mbp", "CTTGACCAGTT")])
    by_layout = load_codebook(path, round_labels=rounds(n - 2), channel_labels=CHANNELS,
                              encoding=adapted.encoding, layout=adapted.layout)
    by_split = load_codebook(path, round_labels=rounds(n - 2), channel_labels=CHANNELS,
                             encoding=EncodingConfig(reverse_bases=reverse_bases, split_index=split - 1))
    pd.testing.assert_frame_equal(by_layout.table, by_split.table, check_exact=True)
    assert by_layout.layout == by_split.layout == adapted.layout
    # The contract's segment lengths and acquisition order.
    if reverse_bases:
        assert adapted.layout == BarcodeLayout((Segment("A", n - split), Segment("B", split)), ("A", "B"))
    else:
        assert adapted.layout == BarcodeLayout((Segment("A", split), Segment("B", n - split)), ("B", "A"))


@pytest.mark.validation
def test_r2_worked_example_segments_and_ends(tmp_path):
    layout = BarcodeLayout((Segment("A", 6, (("C", "C"),)), Segment("B", 5, (("T", "T"),))), ("A", "B"))
    book = load_codebook(write(tmp_path, [("Gfap_probe1", BARCODE)]), round_labels=rounds(9),
                         channel_labels=CHANNELS, layout=layout)
    assert book.table.color_sequence.tolist() == ["242324242"]
    assert book.table.entry_id.tolist() == [BARCODE]
    # An entry whose segment ends are not declared raises, naming the entry and segment.
    with pytest.raises(ValueError, match="row 1.*CAGTACTGCAG.*segment 'B'"):
        load_codebook(write(tmp_path, [("Gfap_probe1", "CAGTACTGCAG")]), round_labels=rounds(9),
                      channel_labels=CHANNELS, layout=layout)


@pytest.mark.validation
@pytest.mark.parametrize("filtration, message", [
    ({"n_barcode_segments": 1}, "n_barcode_segments 1 differs from the 2 segment"),
    ({"n_barcode_segments": 3}, "n_barcode_segments 3 differs from the 2 segment"),
    ({"split_index": [4]}, "must equal load_codebook.split_index"),
    ({"split_index": []}, "must equal load_codebook.split_index"),
])
def test_r2_reads_filtration_keys_must_agree_with_the_layout(filtration, message):
    with pytest.raises(ValueError, match=message):
        from_workflow_config(workflow(9, load_codebook={"split_index": [5]}, reads_filtration=filtration))
    # Without a split, a second segment cannot be stated either.
    with pytest.raises(ValueError, match="n_barcode_segments 2 differs from the 1 segment"):
        from_workflow_config(workflow(9, reads_filtration={"n_barcode_segments": 2}))


@pytest.mark.validation
def test_r2_the_matlab_two_segment_configuration_is_accepted():
    adapted = from_workflow_config(workflow(9, load_codebook={"split_index": [5]}, reads_filtration={
        "n_barcode_segments": 2, "split_index": [5], "end_base": ["CC", "TT"]}))
    # A list end_base of two pairs gives the allowed ends of each segment, in acquisition order.
    assert adapted.layout == BarcodeLayout(
        (Segment("A", 6, (("C", "C"),)), Segment("B", 5, (("T", "T"),))), ("A", "B"))
    assert adapted.pipeline.filtering == ReadFilterConfig()
    # Without reverse_bases the colors after the junction (segment B) come first.
    adapted = from_workflow_config(workflow(9, load_codebook={"split_index": [5], "encoding": {"reverse_bases": False}},
                                            reads_filtration={"end_base": ["CA", ["TT", "CC"]]}))
    assert adapted.layout == BarcodeLayout(
        (Segment("A", 5, (("T", "T"), ("C", "C"))), Segment("B", 6, (("C", "A"),))), ("B", "A"))


@pytest.mark.contract
def test_end_base_lists_and_strings():
    # One segment: every listed pair is an allowed end of that segment.
    adapted = from_workflow_config(workflow(4, reads_filtration={"end_base": ["CC", "TT"],
                                                                 "exclude_invalid_endpoints": True}))
    assert adapted.layout == BarcodeLayout((Segment("A", 5, (("C", "C"), ("T", "T"))),))
    assert adapted.pipeline.filtering == ReadFilterConfig(exclude_invalid_endpoints=True)
    # A string stays the one-segment shortcut of ReadFilterConfig.
    adapted = from_workflow_config(workflow(4, reads_filtration={"end_base": "CC"}))
    assert adapted.layout is None and adapted.pipeline.filtering == ReadFilterConfig(end_bases="CC")
    for bad in (["CC", "TT", "AA"], ["C", "TT"]):
        with pytest.raises(ValueError):
            from_workflow_config(workflow(9, load_codebook={"split_index": [5]}, reads_filtration={"end_base": bad}))
    with pytest.raises(ValueError, match="at least 2 bases"):
        from_workflow_config(workflow(9, load_codebook={"split_index": [10]}))
    with pytest.raises(ValueError, match="two_base encoding only"):
        from_workflow_config(workflow(9, load_codebook={"split_index": [5], "encoding": {
            "method": "one_base", "base_to_color": {"A": 1, "C": 2, "G": 3, "T": 4}}}))


@pytest.mark.contract
def test_encoding_and_decoding_keys():
    adapted = from_workflow_config(workflow(4, load_codebook={"encoding": {
        "method": "one_base", "base_to_color": {"A": 1, "C": 2, "G": 3, "T": 4}}}))
    assert adapted.encoding == OneBaseEncodingConfig({"A": "1", "C": "2", "G": "3", "T": "4"})
    assert from_workflow_config(workflow(4)).pipeline.decoding == WtaDecoderConfig(diagnostics=True)
    adapted = from_workflow_config(workflow(4, decoding={"method": "codebook_aware", "min_score_delta": 0.5}))
    assert adapted.pipeline.decoding == CodebookAwareDecoderConfig(min_score_delta=0.5, allow_rescue=False,
                                                                   diagnostics=True)
    adapted = from_workflow_config(workflow(4, decoding={"method": "codebook_aware", "allow_rescue": True}))
    assert adapted.pipeline.decoding.allow_rescue is True
    with pytest.raises(ValueError, match="unknown decoding method"):
        from_workflow_config(workflow(4, decoding={"method": "viterbi"}))
    # W-293 registers direct, which readout mode multiplexed (the default) does not support.
    with pytest.raises(TypeError, match="readout mode 'multiplexed' does not support decoder 'direct'"):
        from_workflow_config(workflow(4, decoding={"method": "direct"}))
    with pytest.raises(ValueError, match="unknown decoding keys"):
        from_workflow_config(workflow(4, decoding={"method": "wta", "max_hamming": 1}))
    with pytest.raises(ValueError, match="unknown load_codebook.encoding keys"):
        from_workflow_config(workflow(4, load_codebook={"encoding": {"split_index": 4}}))
    config = workflow(4, decoding={"method": "wta"})
    config["rules"]["rsf_single_fov"]["parameters"]["reads_filtration"]["run"] = False
    with pytest.raises(ValueError, match="requires reads_filtration.run"):
        from_workflow_config(config)


@pytest.mark.contract
def test_schema_lists_equal_the_registries():
    defs = SCHEMA["$defs"]
    assert defs["decoding_params"]["properties"]["method"]["enum"] == list(names(DECODING_METHODS))
    encoding = defs["load_codebook_params"]["properties"]["encoding"]["properties"]["method"]["enum"]
    assert encoding == list(names(ENCODINGS))
    for rule in ("rsf_single_fov_config", "lrsf_single_fov_subtile_config", "deep_rsf_subtile_config"):
        properties = defs[rule]["properties"]["parameters"]["properties"]
        assert properties["decoding"] == {"$ref": "#/$defs/decoding_params"}


# --- layouts on the codebook and per-segment end bases in filter_reads ----------------

@pytest.mark.contract
def test_layout_validation():
    with pytest.raises(ValueError, match="unique"):
        BarcodeLayout((Segment("A", 3), Segment("A", 3)))
    with pytest.raises(ValueError, match="acquisition_order"):
        BarcodeLayout((Segment("A", 3), Segment("B", 3)), ("A",))
    with pytest.raises(ValueError, match="ends"):
        Segment("A", 3, ("CC",))
    table = pd.DataFrame({"gene_id": ["A"], "color_sequence": ["1234"]})
    with pytest.raises(ValueError, match="5 colors; the codebook has 4 rounds"):
        Codebook(table, rounds(4), CHANNELS, layout=BarcodeLayout((Segment("A", 3), Segment("B", 4))))
    with pytest.raises(ValueError, match="too few bases"):
        Codebook(table, rounds(4), CHANNELS, layout=BarcodeLayout((Segment("A", 1), Segment("B", 5))))
    with pytest.raises(ValueError, match="mutually exclusive"):
        Codebook(table, rounds(4), CHANNELS, encoding=EncodingConfig(split_index=2),
                 layout=BarcodeLayout((Segment("A", 3), Segment("B", 3)), ("B", "A")))
    # The default is one segment over the whole barcode.
    assert Codebook(table, rounds(4), CHANNELS).layout == BarcodeLayout((Segment("A", 5),))


@pytest.mark.contract
def test_filter_reads_checks_each_segment(tmp_path):
    # Two 3-base segments of 6-base barcodes, 2 + 2 colors (the two_seg shape), ends CC and TT.
    layout = BarcodeLayout((Segment("A", 3, (("C", "C"),)), Segment("B", 3, (("T", "T"),))), ("A", "B"))
    book = load_codebook(write(tmp_path, [("G1", "CACTGT"), ("G2", "CGCTAT")]), round_labels=rounds(4),
                         channel_labels=CHANNELS, layout=layout)
    entries = book.table.color_sequence.tolist()
    assert entries == ["2222", "4444"]
    # An entry's colors; an entry's segment A with segment B "43", which decodes from T to TAG;
    # and an M in segment A, whose segment B is an entry's.
    reads = [entries[0], entries[1][:2] + "43", "M" + entries[0][1:]]
    decoded = decode_barcodes(intensity(tensor([r.replace("M", "1") for r in reads]), channels=CHANNELS,
                                        rounds=rounds(4)), book, config=WtaDecoderConfig())
    table = decoded.table.copy()
    table["observed_color_sequence"] = pd.array(reads, dtype="string")
    decoded = type(decoded)(table, decoded.spot_namespace, decoded.channel_labels, decoded.round_labels,
                            decoded.config, decoded.diagnostics)
    filtered = filter_reads(decoded, codebook=book)
    assert filtered.table.endpoint_valid_A.tolist() == [True, True, False]
    assert filtered.table.endpoint_valid_B.tolist() == [True, False, True]
    assert filtered.table.endpoint_valid.tolist() == [True, False, False]
    assert filtered.diagnostics["endpoint_valid_segment_counts"] == {"A": 2, "B": 2}
    assert filtered.table.accepted.tolist() == decoded.table.call_status.eq("assigned").tolist()
    excluded = filter_reads(decoded, config=ReadFilterConfig(exclude_invalid_endpoints=True), codebook=book)
    assert excluded.table.accepted.tolist() == [True, False, False]
    with pytest.raises(ValueError, match="one-segment shortcut"):
        filter_reads(decoded, config=ReadFilterConfig(end_bases="CC"), codebook=book)
    with pytest.raises(ValueError, match="requires end_bases or segment ends"):
        filter_reads(decoded, config=ReadFilterConfig(exclude_invalid_endpoints=True))


def test_dataset_load_codebook_takes_an_encoding_or_the_legacy_arguments(tmp_path):
    adapted = from_workflow_config(workflow(4))
    with pytest.raises(ValueError, match="not both"):
        adapted.dataset.load_codebook(write(tmp_path, [("A", "CACGC")]), split_index=2, encoding=EncodingConfig())
    adapted.dataset.load_codebook(write(tmp_path, [("A", "CACGC")]), encoding=EncodingConfig(reverse_bases=False))
    assert adapted.dataset.codebook.table.color_sequence.tolist() == ["2244"]
    copy.deepcopy(adapted.dataset.codebook)
