"""The §2.8 readout modes and direct readout (W-293, docs/readout-contract.md, "Readout modes" and
"Direct readout").

Fixture ``direct2`` is worked example 4 of docs/readout-algorithms.md: two rounds of 6×24×24
voxels (seed 100) with four planted spots and the eight-gene panel. Check R7: the reads are
Gfap, Gad1, Sst and Aqp4; the other round of each candidate is valid=False with values 0; an
unmapped (round, channel) is unmatched with unmapped_channel; a zero own round is no_signal; a
repeated gene or (round, channel) raises naming it. Check R16 (direct part): the candidates and
pre_qc checkpoints of the direct2 run round-trip exactly in CSV and Parquet with readout_mode in
the headers, and a checkpoint without the key loads as multiplexed. The mode checks, the YAML
route through workflow/scripts/rsf_single_fov.py and the schema key are checked too.
"""
from dataclasses import replace
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import jsonschema
import numpy as np
import pandas as pd
import pytest
import yaml

from starfinder.barcode import (DECODING_METHODS, Codebook, CodebookAwareDecoderConfig, DirectAssignmentConfig,
                                DirectPanel, NeighborhoodSumConfig, ReadFilterConfig, WtaDecoderConfig, assign_direct,
                                decode_barcodes, extract_intensities, load_direct_panel)
from starfinder.barcode.decoding import BarcodeDecodingResult
from starfinder.dataset import CheckpointConfig, Dataset, PipelineConfig, RoundState
from starfinder.dataset.workflow import from_workflow_config
from starfinder.image import ImageMetadata
from starfinder.io import ImageLoadResult, save_volume
from starfinder.io._checkpoint import candidates_frame, read_checkpoint, read_header, readout_mode
from starfinder.spot_finding import LocalMaximaConfig, SpotFindingPlan

from .test_readout_examples import CHANNELS, DIRECT_MAPPING, DIRECT_SPOTS, direct_images

pytestmark = pytest.mark.barcode

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = yaml.safe_load((ROOT / "workflow/schemas/config.schema.yaml").read_text())
ROUNDS = ("round1", "round2")
METADATA = ImageMetadata("example/FOV_001")
DETECTION = LocalMaximaConfig("noise", 10.0)
PLAN = SpotFindingPlan(DETECTION, rounds=ROUNDS)
GENES = ["Gfap", "Gad1", "Sst", "Aqp4"]
OWN = [0, 0, 1, 1]  # own round index of each direct2 candidate, in DIRECT_SPOTS order


def direct_dataset(root, *, mode="direct", panel=DIRECT_MAPPING):
    dataset = Dataset(root, root / "out", "example", "sample", "out",
                      rounds=RoundState(sequencing_rounds=list(ROUNDS), reference_round="round1"),
                      channel_order=list(CHANNELS), readout_mode=mode)
    if panel is not None:
        root.mkdir(parents=True, exist_ok=True)
        path = root / "panel.csv"
        panel.to_csv(path, index=False)
        dataset.load_direct_panel(path)
    return dataset


def direct_fov(dataset):
    fov = dataset.fov("FOV_001")
    fov.images = direct_images()
    fov.metadata = {name: METADATA for name in ROUNDS}
    return fov


def pipeline(**changes):
    return replace(PipelineConfig(spot_finding=PLAN, extraction=NeighborhoodSumConfig(),
                                  decoding=DirectAssignmentConfig(), filtering=ReadFilterConfig()), **changes)


def loaded(fov):
    return {name: ImageLoadResult(fov.images[name], fov.metadata[name], CHANNELS, (), {}) for name in ROUNDS}


# --- R7: direct assignment, known answer -------------------------------------------------

@pytest.mark.validation
def test_r7_direct2_reads_and_unavailable_other_rounds(tmp_path):
    fov = direct_fov(direct_dataset(tmp_path)).run(pipeline())
    table = fov.decoding_result.table
    assert fov.decoding_result.readout_mode == "direct"
    assert table.gene_id.tolist() == GENES
    assert table.call_status.tolist() == ["assigned"] * 4 and table.call_type.tolist() == ["direct"] * 4
    assert table.failure_reason.tolist() == [""] * 4
    assert table.entry_id.tolist() == ["round1/ch00", "round1/ch02", "round2/ch01", "round2/ch03"]
    assert table["round"].tolist() == [r for r, *_ in DIRECT_SPOTS]
    assert table.channel.tolist() == [CHANNELS[c] for _, c, *_ in DIRECT_SPOTS]
    assert table.observed_color_sequence.isna().all() and table.decoded_color_sequence.isna().all()
    assert table.own_channel_rank.tolist() == [1.0] * 4
    assert (table.own_channel_fraction > 0.25).all()
    assert fov.filtering_result.counts == {"total": 4, "accepted": 4, "rejected": 0}
    # Only the own round is extracted: the other round is 0.0 with valid False and box_voxels 0.
    result = fov.intensity_result
    full = extract_intensities(loaded(fov), fov.spot_result)
    for i, own in enumerate(OWN):
        other = 1 - own
        assert result.valid[i, own] and not result.valid[i, other]
        assert (result.values[i, :, other] == 0).all() and result.box_voxels[i, other] == 0
        np.testing.assert_array_equal(result.values[i, :, own], full.values[i, :, own], strict=True)
        assert result.box_voxels[i, own] == 3 * 5 * 5
    assert full.valid.all() and (full.box_voxels == 75).all()


@pytest.mark.validation
def test_r7_unmapped_channel_and_zero_own_round(tmp_path):
    # (round2, ch01) is absent from the panel: its candidate is unmatched, the others unchanged.
    without = DIRECT_MAPPING[DIRECT_MAPPING.gene_id != "Sst"]
    table = direct_fov(direct_dataset(tmp_path, panel=without)).run(pipeline()).decoding_result.table
    assert table.call_status.tolist() == ["assigned", "assigned", "unmatched", "assigned"]
    assert table.failure_reason.tolist() == ["", "", "unmapped_channel", ""]
    assert table.gene_id.tolist()[:2] + table.gene_id.tolist()[3:] == ["Gfap", "Gad1", "Aqp4"]
    assert table.gene_id.isna().tolist() == [False, False, True, False] and table.entry_id.isna()[2]
    assert table.call_type.tolist() == ["direct", "direct", "no_call", "direct"]
    # A zero own round: round2 set to 0 after detection gives no_signal for its two candidates.
    fov = direct_fov(direct_dataset(tmp_path / "zero"))
    fov.find_spots(config=PLAN)
    fov.images["round2"][:] = 0
    fov.extract_intensities().decode_barcodes(config=DirectAssignmentConfig())
    table = fov.decoding_result.table
    assert table.call_status.tolist() == ["assigned", "assigned", "no_signal", "no_signal"]
    assert table.failure_reason.tolist() == ["", "", "zero_signal_round", "zero_signal_round"]
    assert table.gene_id.tolist()[:2] == ["Gfap", "Gad1"] and table.gene_id.isna().tolist()[2:] == [True, True]
    assert table.own_channel_rank.isna().tolist() == [False, False, True, True]


@pytest.mark.validation
@pytest.mark.parametrize("change, name", [
    ({"gene_id": {"Vip": "Gfap"}}, "Gfap"),
    ({"channel": {"ch03": "ch02"}}, r"\('round1', 'ch02'\)"),
])
def test_r7_repeated_gene_or_round_channel_raises_naming_it(tmp_path, change, name):
    invalid = DIRECT_MAPPING.replace(change)
    with pytest.raises(ValueError, match=name):
        DirectPanel(invalid)
    path = tmp_path / "panel.csv"
    invalid.to_csv(path, index=False)
    with pytest.raises(ValueError, match=name):
        load_direct_panel(path, round_labels=ROUNDS, channel_labels=CHANNELS)


@pytest.mark.contract
def test_r7_invalid_own_round_is_invalid_measurement(tmp_path):
    fov = direct_fov(direct_dataset(tmp_path)).run(pipeline(filtering=None))
    valid = fov.intensity_result.valid.copy()
    valid[1, 0] = False
    reads = assign_direct(replace(fov.intensity_result, valid=valid), fov.spot_result, fov.dataset.direct_panel)
    assert reads.table.call_status.tolist() == ["assigned", "unmatched", "assigned", "assigned"]
    assert reads.table.failure_reason[1] == "invalid_measurement" and pd.isna(reads.table.gene_id[1])
    # The brightest channel never changes the identity: a brighter other channel only lowers the rank.
    values = fov.intensity_result.values.copy()
    values[0, 1, 0] = 10 * values[0, 0, 0]
    reads = assign_direct(replace(fov.intensity_result, values=values), fov.spot_result, fov.dataset.direct_panel)
    assert reads.table.gene_id[0] == "Gfap" and reads.table.own_channel_rank[0] == 2.0
    assert reads.table.own_channel_fraction[0] < 0.1


@pytest.mark.contract
def test_panel_labels_and_header(tmp_path):
    path = tmp_path / "panel.csv"
    for text, message in [("round,channel,gene_id\nround3,ch00,A\n", "round 'round3' is not a sequencing round"),
                          ("round,channel,gene_id\nround1,ch09,A\n", "channel 'ch09' is not a channel label"),
                          ("round,channel,gene\nround1,ch00,A\n", "header must be round,channel,gene_id"),
                          ("round,channel,gene_id\nround1,ch00,\n", "row 2: gene_id is empty")]:
        path.write_text(text)
        with pytest.raises(ValueError, match=message):
            load_direct_panel(path, round_labels=ROUNDS, channel_labels=CHANNELS)
    DIRECT_MAPPING.to_csv(path, index=False)
    panel = load_direct_panel(path, round_labels=ROUNDS, channel_labels=CHANNELS)
    assert panel.n_genes == 8 and panel.round_labels == ROUNDS and panel.gene_of[("round2", "ch03")] == "Aqp4"
    assert repr(panel) == "DirectPanel: 8 genes over 2 rounds"


@pytest.mark.contract
def test_direct_declares_its_mode_and_score_columns(tmp_path):
    spec = DECODING_METHODS[DirectAssignmentConfig]
    assert (spec.name, spec.modes, spec.encodings, spec.rescue) == ("direct", frozenset({"direct"}), frozenset(), False)
    table = direct_fov(direct_dataset(tmp_path)).run(pipeline()).decoding_result.table
    numeric = [c for c in table.columns if pd.api.types.is_float_dtype(table[c])]
    assert sorted(numeric) == sorted(spec.score_columns)


# --- Mode checks -----------------------------------------------------------------------------

@pytest.mark.contract
def test_multiplexed_mode_with_a_round_column_names_the_direct_mode(tmp_path):
    fov = direct_fov(direct_dataset(tmp_path, mode="multiplexed", panel=None))
    with pytest.raises(ValueError) as raised:
        fov.run(pipeline(decoding=WtaDecoderConfig(), filtering=None))
    assert "readout mode (§2.8)" in str(raised.value) and "readout_mode='direct'" in str(raised.value)
    assert fov.spot_result is None
    fov.find_spots(config=PLAN)
    with pytest.raises(ValueError) as raised:
        fov.decode_barcodes()
    assert "readout mode (§2.8)" in str(raised.value) and "readout_mode='direct'" in str(raised.value)


@pytest.mark.contract
def test_direct_mode_without_a_round_column_raises(tmp_path):
    fov = direct_fov(direct_dataset(tmp_path))
    with pytest.raises(ValueError, match="needs candidates with a round column"):
        fov.run(pipeline(spot_finding=DETECTION))
    with pytest.raises(ValueError, match="needs candidates with a round column"):
        fov.run(pipeline(spot_finding=DETECTION, decoding=None, filtering=None))
    assert fov.spot_result is None
    fov.find_spots(config=DETECTION)
    with pytest.raises(ValueError, match="round column"):
        fov.extract_intensities()
    multiplexed = extract_intensities(loaded(fov), fov.spot_result)
    with pytest.raises(ValueError, match="round and channel columns"):
        assign_direct(multiplexed, fov.spot_result, fov.dataset.direct_panel)


@pytest.mark.contract
@pytest.mark.parametrize("decoder", [WtaDecoderConfig(), CodebookAwareDecoderConfig()], ids=["wta", "codebook_aware"])
def test_barcode_decoders_raise_in_direct_mode(tmp_path, decoder):
    fov = direct_fov(direct_dataset(tmp_path))
    name = DECODING_METHODS[type(decoder)].name
    with pytest.raises(TypeError, match=f"readout mode 'direct' does not support decoder '{name}'"):
        fov.run(pipeline(decoding=decoder))
    assert fov.spot_result is None
    fov.run(pipeline(decoding=None, filtering=None))
    with pytest.raises(TypeError, match=f"readout mode 'direct' does not support decoder '{name}'"):
        fov.decode_barcodes(config=decoder)
    with pytest.raises(TypeError, match=f"readout mode 'direct' does not support decoder '{name}'"):
        assign_direct(fov.intensity_result, fov.spot_result, fov.dataset.direct_panel, config=decoder)
    with pytest.raises(TypeError, match="a DirectPanel is the reference of readout mode 'direct'"):
        decode_barcodes(fov.intensity_result, fov.dataset.direct_panel, config=decoder)


@pytest.mark.contract
def test_direct_assignment_raises_in_multiplexed_mode(tmp_path):
    dataset = direct_dataset(tmp_path, mode="multiplexed", panel=None)
    fov = direct_fov(dataset)
    with pytest.raises(TypeError, match="readout mode 'multiplexed' does not support decoder 'direct'"):
        fov.run(pipeline(spot_finding=DETECTION))
    fov.run(pipeline(spot_finding=DETECTION, decoding=None, filtering=None))
    with pytest.raises(TypeError, match="readout mode 'multiplexed' does not support decoder 'direct'"):
        fov.decode_barcodes(config=DirectAssignmentConfig())
    book = Codebook(pd.DataFrame({"gene_id": ["A"], "color_sequence": ["12"]}), ROUNDS, CHANNELS)
    with pytest.raises(TypeError, match="readout mode 'multiplexed' does not support decoder 'direct'"):
        decode_barcodes(fov.intensity_result, book, config=DirectAssignmentConfig())


@pytest.mark.contract
def test_direct_mode_needs_its_panel_and_a_known_mode(tmp_path):
    fov = direct_fov(direct_dataset(tmp_path, panel=None))
    with pytest.raises(ValueError, match="requires a loaded direct panel"):
        fov.run(pipeline())
    with pytest.raises(ValueError, match="readout_mode must be one of"):
        direct_dataset(tmp_path, mode="sequential", panel=None)
    table = pd.DataFrame({c: pd.array([], dtype="string") for c in (
        "spot_id", "spot_namespace", "observed_color_sequence", "decoded_color_sequence", "gene_id",
        "call_status", "failure_reason")})
    with pytest.raises(ValueError, match="readout mode 'direct' does not support decoder 'wta'"):
        BarcodeDecodingResult(table, "ns", CHANNELS, ROUNDS, WtaDecoderConfig(), {}, readout_mode="direct")
    assert "readout mode:      direct" in repr(direct_dataset(tmp_path)).splitlines()[4]


# --- R16 (direct part): checkpoints ------------------------------------------------------------

@pytest.mark.validation
@pytest.mark.parametrize("table_format", ["csv", "parquet"])
def test_r16_direct_checkpoints_round_trip_exactly(tmp_path, table_format):
    dataset = direct_dataset(tmp_path)
    checkpoints = CheckpointConfig(stages=("candidates", "pre_qc"), directory=tmp_path / "checkpoints",
                                   table_format=table_format, hash_inputs=False)
    fov = direct_fov(dataset).run(pipeline(), checkpoints=checkpoints)
    directory = tmp_path / "checkpoints" / "FOV_001"
    for stage in ("candidates", "pre_qc"):
        assert read_header(directory, stage)["readout_mode"] == "direct"
    assert json.loads((directory / "run.json").read_text())["config"]["readout_mode"] == "direct"
    reloaded = dataset.fov("FOV_001").load_checkpoint("candidates", checkpoints=checkpoints)
    pd.testing.assert_frame_equal(candidates_frame(reloaded.spot_result, reloaded.intensity_result),
                                  candidates_frame(fov.spot_result, fov.intensity_result), check_exact=True)
    reloaded.load_checkpoint("pre_qc", checkpoints=checkpoints)
    pd.testing.assert_frame_equal(reloaded.decoding_result.table, fov.decoding_result.table, check_exact=True)
    assert reloaded.decoding_result.config == DirectAssignmentConfig()
    assert reloaded.decoding_result.readout_mode == "direct"
    # Rerun without images: assign again from the reloaded candidates.
    rerun = dataset.fov("FOV_001").load_checkpoint("candidates", checkpoints=checkpoints)
    rerun.run(PipelineConfig(decoding=DirectAssignmentConfig(), filtering=ReadFilterConfig()))
    pd.testing.assert_frame_equal(rerun.decoding_result.table, fov.decoding_result.table, check_exact=True)
    # A dataset of the other mode does not load them.
    other = direct_dataset(tmp_path / "other", mode="multiplexed", panel=None)
    with pytest.raises(ValueError, match="readout mode 'direct' differs from the dataset"):
        other.fov("FOV_001").load_checkpoint("candidates", checkpoints=checkpoints)


@pytest.mark.validation
def test_r16_a_checkpoint_without_the_key_loads_as_multiplexed(tmp_path):
    dataset = direct_dataset(tmp_path, mode="multiplexed", panel=None)
    dataset.codebook = Codebook(pd.DataFrame({"gene_id": ["A", "B"], "color_sequence": ["11", "33"]}),
                                ROUNDS, CHANNELS)
    checkpoints = CheckpointConfig(stages=("candidates", "pre_qc"), directory=tmp_path / "checkpoints",
                                   hash_inputs=False)
    fov = direct_fov(dataset).run(pipeline(spot_finding=DETECTION, decoding=WtaDecoderConfig()),
                                  checkpoints=checkpoints)
    directory = tmp_path / "checkpoints" / "FOV_001"
    for stage in ("candidates", "pre_qc"):
        path = directory / f"{stage}.json"
        header = json.loads(path.read_text())
        assert header.pop("readout_mode") == "multiplexed"
        path.write_text(json.dumps(header))
        assert readout_mode(read_header(directory, stage)) == "multiplexed"
    assert read_checkpoint(directory, "pre_qc")["decoding_result"].readout_mode == "multiplexed"
    reloaded = dataset.fov("FOV_001").load_checkpoint("candidates", checkpoints=checkpoints)
    reloaded.load_checkpoint("pre_qc", checkpoints=checkpoints)
    pd.testing.assert_frame_equal(reloaded.decoding_result.table, fov.decoding_result.table, check_exact=True)
    with pytest.raises(ValueError, match="readout mode 'multiplexed' differs from the dataset"):
        direct_dataset(tmp_path / "direct").fov("FOV_001").load_checkpoint("pre_qc", checkpoints=checkpoints)


# --- The YAML route --------------------------------------------------------------------------

def yaml_config(tmp_path, **top):
    parameters = {"spot_finding": {"run": True, "intensity_estimation": "noise", "intensity_threshold": 10.0,
                                   "rounds": list(ROUNDS)},
                  "reads_extraction": {"run": True}, "reads_filtration": {"run": True}}
    config = dict(config_path="config.yaml", starfinder_path=str(ROOT), root_input_path=str(tmp_path),
                  root_output_path=str(tmp_path / "out"), dataset_id="data", sample_id="sample", output_id="run",
                  fov_id_pattern="%s", n_fovs=1, n_rounds=2, ref_round="round1", rotate_angle=0, img_col=24,
                  img_row=24, subset_range=False, subset_start=1, subset_end=1, subset_random=False,
                  n_random_tests=1, seq_channel_order=list(CHANNELS),
                  backend="python", readout_mode="direct",
                  rules={"rsf_single_fov": {"run": True, "parameters": parameters}})
    config.update(top)
    return config


@pytest.mark.contract
def test_yaml_direct_route_writes_the_four_genes(tmp_path):
    for name, image in direct_images().items():
        for c, channel in enumerate(CHANNELS):
            save_volume(image[..., c], tmp_path / "data" / "sample" / name / "FOV" / f"{channel}.tif",
                        metadata=METADATA)
    panel = tmp_path / "panel.csv"
    DIRECT_MAPPING.to_csv(panel, index=False)
    config = yaml_config(tmp_path)
    jsonschema.validate(config, SCHEMA)
    snakemake = SimpleNamespace(config=config, input=["unused", str(panel)], wildcards=SimpleNamespace(fovID="FOV"))
    runpy.run_path(str(ROOT / "workflow" / "scripts" / "rsf_single_fov.py"), init_globals={"snakemake": snakemake})
    adapted = from_workflow_config(config)
    assert adapted.dataset.readout_mode == "direct" and adapted.pipeline.decoding == DirectAssignmentConfig()
    good = pd.read_csv(adapted.dataset.fov("FOV").paths.signal_csv("goodSpots"))
    assert list(good.columns) == ["x", "y", "z", "gene"]
    assert good.gene.tolist() == GENES
    assert list(zip(good.x, good.y, good.z)) == [(x + 1, y + 1, z + 1) for _, _, z, y, x in DIRECT_SPOTS]


@pytest.mark.contract
def test_yaml_readout_mode_keys(tmp_path):
    config = yaml_config(tmp_path)
    shared = dict(config, rules={"rsf_single_fov": {"run": True, "parameters": {"reads_filtration": {"run": True}}}})
    for backend in ("matlab", "python"):
        jsonschema.validate(dict(shared, readout_mode="multiplexed", backend=backend), SCHEMA)
    jsonschema.validate(config, SCHEMA)
    for invalid in (dict(config, backend="matlab"), {k: v for k, v in config.items() if k != "backend"},
                    dict(config, readout_mode="sequential")):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, SCHEMA)
    with pytest.raises(ValueError, match="readout_mode must be one of"):
        from_workflow_config(dict(config, readout_mode="sequential"))
    parameters = config["rules"]["rsf_single_fov"]["parameters"]
    with pytest.raises(TypeError, match="readout mode 'direct' does not support decoder 'wta'"):
        from_workflow_config(dict(config, rules={"rsf_single_fov": {"parameters": dict(
            parameters, decoding={"method": "wta"})}}))
    assert from_workflow_config(dict(config, rules={"rsf_single_fov": {"parameters": dict(
        parameters, decoding={"method": "direct"})}})).pipeline.decoding == DirectAssignmentConfig()
    with pytest.raises(ValueError, match=r"remove \['reads_filtration.end_base'\]"):
        from_workflow_config(dict(config, rules={"rsf_single_fov": {"parameters": dict(
            parameters, reads_filtration={"run": True, "end_base": "CC"})}}))
    assert from_workflow_config(dict(config, readout_mode="multiplexed")).dataset.readout_mode == "multiplexed"
    assert from_workflow_config({k: v for k, v in config.items() if k != "readout_mode"}).pipeline.decoding == \
        WtaDecoderConfig(diagnostics=True)
