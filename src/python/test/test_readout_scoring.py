"""§2.8 shared read-QC score, its checkpoints and reruns (W-294; checks R11, R12, R16, R17).

docs/readout-contract.md, "Shared read-QC score" and "Checkpoints and reruns";
docs/readout-algorithms.md, "Shared read-QC score". R11 compares qc_score and its
components with the W-278 formula (scripts/w278_lib.py, components) written out in the
test, on the golden fixture of test_readout_golden.py and on the calibrated scenes of
seed 103. R12 (extended tier) checks the ranking on the W-278 held-out scenes (seeds 103
to 105) with the tolerances of W-278 scores.csv (heldout, all_calibrated). R16 and R17
check the candidates and pre_qc checkpoints with background and score columns, reruns
from them, and a checkpoint written before §2.8 (as at 141c093). The golden fixture runs with its
pinned seed and with seeds 100 to 102, direct2 with seeds 100 to 102 (readout_fixtures.golden_seed,
test_readout_direct.direct_fov).
"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from starfinder.barcode import (DirectAssignmentConfig, NeighborhoodSumConfig, ReadFilterConfig, ReadScoreConfig,
                                ReadScoringResult, WtaDecoderConfig, decode_barcodes, filter_reads, score_reads)
from starfinder.barcode.scoring import SCORE_COLUMNS
from starfinder.dataset import CheckpointConfig, PipelineConfig
from starfinder.dataset.workflow import from_workflow_config
from starfinder.evaluation.barcode import ranking_quality
from starfinder.io import read_checkpoint
from starfinder.io._checkpoint import candidates_frame

from . import readout_fixtures as fx
from . import readout_scenes as scenes
from .test_readout_direct import HAND_SEEDS, direct_dataset, direct_fov
from .test_readout_direct import pipeline as direct_pipeline
from .test_readout_golden import (ROUNDS, SEED, candidates, extract, fixture_rounds, golden_codebook,
                                  golden_dataset, readout_config, METADATA)

pytestmark = [pytest.mark.barcode]

ROOT = Path(__file__).resolve().parents[3]

IDENTITY = ["spot_id", "gene_id", "entry_id", "call_status", "call_type"]
COMPONENTS = ("qc_score", "qc_ambiguity_max", "qc_signal_to_background")
# R11 tolerance, provisional: no W-278 row applies; it is floating-point agreement between the
# implementation and the W-278 formula computed in the test (a formula identity, exact up to rounding).
FORMULA_TOLERANCE = 1e-12


def assert_formula(table, reference):
    """Each score column equals the W-278 reference within FORMULA_TOLERANCE, with infinities and NaN at the
    same rows."""
    for column, expected in zip(COMPONENTS, reference):
        actual = table[column].to_numpy(dtype=float)
        assert np.array_equal(np.isnan(actual), np.isnan(expected)), column
        assert np.array_equal(np.isinf(actual), np.isinf(expected)), column
        finite = np.isfinite(actual)
        assert np.array_equal(actual[~finite & ~np.isnan(actual)], expected[~finite & ~np.isnan(expected)]), column
        assert np.max(np.abs(actual[finite] - expected[finite]), initial=0.0) <= FORMULA_TOLERANCE, column


# --- R11: the score definition ----------------------------------------------------------------

@pytest.mark.validation
@pytest.mark.parametrize("decoder", ["wta", "codebook_aware"])
@pytest.mark.parametrize("seed", fx.GOLDEN_SEEDS)
def test_r11_golden_score_equals_the_w278_formula(tmp_path, seed, decoder):
    book = golden_codebook(tmp_path)
    with fx.golden_seed(seed):
        intensities, rounds = extract(), fixture_rounds()
    decoded = decode_barcodes(intensities, book, config=readout_config("decoding", decoder=decoder))
    scored = score_reads(decoded, intensities, reference=book)
    table = scored.table
    reference = scenes.w278_components(intensities.values, intensities.box_voxels, intensities.background,
                                       scenes.assigned_channels(table, book, len(ROUNDS)))
    assert_formula(table, reference)
    assigned = table.call_status.eq("assigned")
    assert table.qc_reason[~assigned].eq("no_assignment").all() and table.qc_reason[assigned].eq("").all()
    assert table.loc[~assigned, list(SCORE_COLUMNS[:-1])].isna().all().all()
    assert (table.qc_rounds[assigned] == len(ROUNDS)).all()
    # Every row is kept and no identity changes; the decoding columns are untouched.
    pd.testing.assert_frame_equal(table[IDENTITY], decoded.table[IDENTITY])
    pd.testing.assert_frame_equal(table.drop(columns=list(SCORE_COLUMNS)), decoded.table)
    assert list(table.columns[-len(SCORE_COLUMNS):]) == list(SCORE_COLUMNS)
    assert scored.counts == {"total": 10, "scored": int(assigned.sum()), "no_assignment": int((~assigned).sum()),
                             "background_unavailable": 0}
    # The local background of the golden fixture is the W-278 ring estimate.
    centers = np.floor(candidates().spots[["z", "y", "x"]].to_numpy(float) + 0.5).astype(np.int64)
    for j, image in enumerate(rounds.values()):
        median, mad, voxels = scenes.ring_estimate(image, centers)
        np.testing.assert_array_equal(intensities.background[:, :, j], median)
        np.testing.assert_array_equal(intensities.noise[:, :, j], mad)
        np.testing.assert_array_equal(intensities.background_voxels[:, j], voxels)


@pytest.mark.validation
def test_r11_calibrated_seed_103_score_equals_the_w278_formula():
    for condition in scenes.CAL_CONDITIONS:
        book, scene = scenes.scene(condition, 103)
        spots, intensities, tables, _ = scenes.pipeline(condition, 103)
        centers = np.floor(spots.spots[["z", "y", "x"]].to_numpy(float) + 0.5).astype(np.int64)
        for j, label in enumerate(scene.round_labels):
            median, mad, voxels = scenes.ring_estimate(scene.rounds[label], centers)
            np.testing.assert_array_equal(intensities.background[:, :, j], median)
            np.testing.assert_array_equal(intensities.noise[:, :, j], mad)
            np.testing.assert_array_equal(intensities.background_voxels[:, j], voxels)
        for name, table in tables.items():
            decoded = decode_barcodes(intensities, book, config=scenes.DECODERS[name])
            assert_formula(table, scenes.w278_components(intensities.values, intensities.box_voxels,
                                                         intensities.background,
                                                         scenes.assigned_channels(table, book, 4)))
            pd.testing.assert_frame_equal(table.drop(columns=list(SCORE_COLUMNS)), decoded.table)
            assigned = table.call_status.eq("assigned")
            assert table.qc_reason[~assigned].eq("no_assignment").all() and table.qc_score[assigned].notna().all()


@pytest.mark.validation
@pytest.mark.parametrize("seed", HAND_SEEDS)
def test_r11_direct_reads_score_their_own_round(tmp_path, seed):
    fov = direct_fov(direct_dataset(tmp_path), seed).run(direct_pipeline(scoring=ReadScoreConfig()))
    table, intensities = fov.scoring_result.table, fov.intensity_result
    assert fov.scoring_result.readout_mode == "direct" and table.qc_reason.eq("").all()
    assert (table.qc_rounds == 1).all()
    own = np.array([intensities.round_labels.index(r) for r in table["round"]])
    channel = np.array([intensities.channel_labels.index(c) for c in table.channel])
    rows = np.arange(len(table))
    reference = scenes.w278_components(intensities.values[rows, :, own][:, :, None],
                                       intensities.box_voxels[rows, own][:, None],
                                       intensities.background[rows, :, own][:, :, None], channel[:, None])
    assert_formula(table, reference)
    pd.testing.assert_frame_equal(table.drop(columns=list(SCORE_COLUMNS)), fov.decoding_result.table)
    # Filtering keeps the score columns of the scored reads.
    assert list(fov.filtering_result.table.columns[:len(table.columns)]) == list(table.columns)


@pytest.mark.contract
def test_score_reads_checks_its_inputs(tmp_path):
    book = golden_codebook(tmp_path)
    intensities = extract()
    decoded = decode_barcodes(intensities, book, config=WtaDecoderConfig())
    with pytest.raises(TypeError, match="Codebook"):
        score_reads(decoded, intensities, reference=None)
    with pytest.raises(ValueError, match="extraction"):
        score_reads(decoded, replace(intensities, background=None, noise=None, background_voxels=None,
                                     image_background=None, image_noise=None), reference=book)
    with pytest.raises(ValueError, match="identities"):
        score_reads(replace(decoded, table=decoded.table.iloc[::-1].reset_index(drop=True)), intensities,
                    reference=book)
    scored = score_reads(decoded, intensities, reference=book)
    assert repr(scored) == "ReadScoringResult: 6 of 10 reads scored — no_assignment 4"
    # Rescoring a scored table replaces its score columns.
    again = score_reads(replace(decoded, table=scored.table), intensities, reference=book)
    pd.testing.assert_frame_equal(again.table, scored.table)
    # A scored result filters like its decoding result, keeping the score columns.
    filtered = filter_reads(scored, config=ReadFilterConfig())
    pd.testing.assert_frame_equal(filtered.table.drop(columns=["accepted", "rejection_reasons"]), scored.table)
    assert filtered.counts == filter_reads(decoded, config=ReadFilterConfig()).counts


# --- R12: ranking on the calibrated scenes (extended tier) ------------------------------------

@pytest.fixture(scope="module")
def r12_calls():
    """Exact and rescued assigned calls per decoder and condition, seeds 103 to 105.

    matched calls have a matched amplicon (correct when its gene is the call's); the others are
    unmatched detections, which W-278 kept as a separate population.
    """
    rows = []
    for condition in scenes.CONDITIONS:
        for seed in scenes.HELDOUT_SEEDS:
            _, _, tables, truth_gene = scenes.pipeline(condition, seed)
            matched = np.array([g is not None for g in truth_gene])
            for decoder, table in tables.items():
                correct = np.array([t is not None and t == g for t, g in zip(truth_gene, table.gene_id.fillna(""))])
                keep = table.call_status.eq("assigned").to_numpy()
                rows.append(pd.DataFrame({
                    "condition": condition, "decoder": decoder,
                    "call_class": np.where(table.call_type.eq("exact"), "exact", "rescued")[keep],
                    "matched": matched[keep],
                    "correct": correct[keep],
                    "qc_score": table.qc_score.to_numpy()[keep],
                    "decoder_score": table[scenes.DECODER_SCORE[decoder]].to_numpy()[keep]}))
    return pd.concat(rows, ignore_index=True)


RETENTION_KEYS = ("error_at_50", "error_at_80", "error_at_90", "error_at_100")


def _quality(calls, column):
    """ranking_quality of one score on calls (correct ranked above the others), lower is more reliable."""
    return ranking_quality(calls[column].to_numpy(), calls.correct.to_numpy(dtype=bool), orientation="lower")


def _auroc(calls, column):
    return _quality(calls, column).values["auroc"]


def _report(label, calls):
    """One printed line per score: AUROC, its Hanley-McNeil SE and the error at each retention level."""
    for column in ("qc_score", "decoder_score"):
        result = _quality(calls, column)
        values = result.values
        errors = " ".join(f"{key[9:]}% {values[key]}" for key in RETENTION_KEYS)
        print(f"  {label} {column}: AUROC {values['auroc']} SE {values['auroc_se']}; error at {errors}; "
              f"calls {result.counts['scored']}, incorrect {result.counts['incorrect']}")


@pytest.mark.extended
@pytest.mark.slow
@pytest.mark.validation
def test_r12_qc_score_ranks_above_the_decoder_scores(r12_calls):
    calls = r12_calls[r12_calls.matched]
    print("\nR12 pooled AUROC (qc_score, decoder score, calls, incorrect):")
    # W-278 scores.csv (heldout, all_calibrated) pools the nine conditions; "cal" is the eight
    # without dense. Both pools are gated.
    for pool, conditions in (("all_calibrated", scenes.CONDITIONS), ("cal", scenes.CAL_CONDITIONS)):
        population = calls[calls.condition.isin(conditions)]
        for decoder in scenes.DECODERS:
            exact = population[(population.decoder == decoder) & (population.call_class == "exact")]
            qc, reference = _auroc(exact, "qc_score"), _auroc(exact, "decoder_score")
            print(f"  {pool} {decoder} exact: {qc:.4f} {reference:.4f} {len(exact)} {int((~exact.correct).sum())}")
            # call_class=exact: 0.897 against 0.782 (WTA) and 0.743 (codebook-aware).
            assert qc >= reference + 0.05, (pool, decoder)
        rescued = population[(population.decoder == "codebook_aware") & (population.call_class == "rescued")]
        qc, reference = _auroc(rescued, "qc_score"), _auroc(rescued, "decoder_score")
        print(f"  {pool} codebook_aware rescued: {qc:.4f} {reference:.4f} {len(rescued)} "
              f"{int((~rescued.correct).sum())}")
        # call_class=rescued: 0.851 against 0.670 (12 incorrect); ordering only.
        assert qc > reference, pool
    # Reported, not gated: the full ranking_quality of each score per call_type (AUROC, Hanley-McNeil
    # SE, error at 50, 80, 90 and 100 % retention), per condition for exact and rescued calls, and
    # the unmatched detections (correct calls against unmatched detections, as W-278
    # auroc_correct_vs_unmatched).
    print("R12 ranking_quality per call_type, pooled (reported, not gated):")
    for pool, conditions in (("all_calibrated", scenes.CONDITIONS), ("cal", scenes.CAL_CONDITIONS)):
        population = calls[calls.condition.isin(conditions)]
        for (decoder, call_class), group in population.groupby(["decoder", "call_class"], sort=False):
            _report(f"{pool} {decoder} {call_class}", group)
    for call_class in ("exact", "rescued"):
        print(f"R12 per condition, {call_class} calls (reported, not gated):")
        for (condition, decoder), group in calls[calls.call_class == call_class].groupby(
                ["condition", "decoder"], sort=False):
            qc, reference = _auroc(group, "qc_score"), _auroc(group, "decoder_score")
            print(f"  {condition} {decoder}: qc_score {qc} decoder {reference} calls {len(group)} incorrect "
                  f"{int((~group.correct).sum())}")
    print("R12 unmatched detections, correct calls against unmatched detections (reported, not gated):")
    detections = r12_calls[r12_calls.correct | ~r12_calls.matched]
    for pool, conditions in (("all_calibrated", scenes.CONDITIONS), ("cal", scenes.CAL_CONDITIONS)):
        population = detections[detections.condition.isin(conditions)]
        for (decoder, call_class), group in population.groupby(["decoder", "call_class"], sort=False):
            print(f"  {pool} {decoder} {call_class}: unmatched {int((~group.matched).sum())}")
            _report(f"{pool} {decoder} {call_class} vs unmatched", group)


# --- R16 and R17: checkpoints and reruns --------------------------------------------------------

def golden_run(root, table_format="csv", decoder="wta", seed=SEED, **changes):
    """FOV.run of the golden fixture (with the noise of seed, by default its pinned SEED) with extraction,
    decoding, scoring and filtering, and checkpoints."""
    dataset = golden_dataset(root)
    dataset.codebook = golden_codebook(root)
    fov = dataset.fov("FOV_001")
    fov.images = fx.golden_rounds(seed)
    fov.metadata = {label: METADATA for label in ROUNDS}
    fov.spot_result = candidates()
    checkpoints = CheckpointConfig(stages=("candidates", "pre_qc"), directory=root / "checkpoints",
                                   table_format=table_format)
    fov.run(replace(readout_config("pipeline", decoder=decoder), **changes), checkpoints=checkpoints)
    return dataset, fov, checkpoints


def assert_reads_equal(a, b):
    for name in ("decoding_result", "scoring_result", "filtering_result"):
        x, y = getattr(a, name), getattr(b, name)
        assert (x is None) == (y is None), name
        if x is not None:
            pd.testing.assert_frame_equal(x.table, y.table, check_exact=True)
            assert x.config == y.config, name


@pytest.mark.validation
@pytest.mark.parametrize("table_format", ["csv", "parquet"])
@pytest.mark.parametrize("seed", fx.GOLDEN_SEEDS)
def test_r16_background_and_score_columns_round_trip_exactly(tmp_path, seed, table_format):
    dataset, fov, checkpoints = golden_run(tmp_path, table_format, seed=seed)
    directory = tmp_path / "checkpoints" / "FOV_001"
    loaded = read_checkpoint(directory, "candidates")
    a, b = loaded["intensity_result"], fov.intensity_result
    for name in ("values", "valid", "box_voxels", "background", "noise", "background_voxels", "image_background",
                 "image_noise"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name, strict=True)
    assert a.config == b.config and a.config.background is not None
    pd.testing.assert_frame_equal(candidates_frame(loaded["spot_result"], a),
                                  candidates_frame(fov.spot_result, fov.intensity_result), check_exact=True)
    header = json.loads((directory / "candidates.json").read_text())
    # The 141c093 fields stay as they were; the background settings are new top-level keys.
    assert set(header["signals"]["extraction_config"]) == {"neighborhood_radius_zyx", "sampling", "boundary"}
    assert header["background_config"] == {"inner_radius_zyx": [1, 3, 3], "outer_radius_zyx": [1, 6, 6],
                                           "min_voxels": 16}
    assert header["image_background"]["round1"]["ch00"] == b.image_background[0, 0]
    assert header["readout_mode"] == "multiplexed"
    columns = list(header["dtypes"])
    assert columns[columns.index("valid_round4") + 1:] == (
        [f"bg_{r}_{c}" for r in ROUNDS for c in b.channel_labels]
        + [f"noise_{r}_{c}" for r in ROUNDS for c in b.channel_labels]
        + [f"bgvox_{r}" for r in ROUNDS] + [f"boxvox_{r}" for r in ROUNDS])
    pre_qc = dataset.fov("FOV_001").load_checkpoint("pre_qc", checkpoints=checkpoints)
    pd.testing.assert_frame_equal(pre_qc.scoring_result.table, fov.scoring_result.table, check_exact=True)
    pd.testing.assert_frame_equal(pre_qc.decoding_result.table, fov.decoding_result.table, check_exact=True)
    assert pre_qc.scoring_result.config == ReadScoreConfig() and pre_qc.scoring_result.counts == fov.scoring_result.counts
    header = json.loads((directory / "pre_qc.json").read_text())
    assert header["scoring_config"] == {"method": "bgcorr_probability"} and header["deduplication_config"] is None
    assert header["stages_applied"] == ["decoding", "scoring"] and header["readout_mode"] == "multiplexed"
    assert header["layout"]["segments"] == [{"name": "A", "bases": 5, "ends": []}]
    assert list(header["dtypes"])[-len(SCORE_COLUMNS):] == list(SCORE_COLUMNS)


@pytest.mark.validation
@pytest.mark.parametrize("table_format", ["csv", "parquet"])
@pytest.mark.parametrize("seed", HAND_SEEDS)
def test_r16_direct_background_and_score_round_trip_exactly(tmp_path, seed, table_format):
    dataset = direct_dataset(tmp_path)
    checkpoints = CheckpointConfig(stages=("candidates", "pre_qc"), directory=tmp_path / "ck", table_format=table_format)
    fov = direct_fov(dataset, seed).run(direct_pipeline(scoring=ReadScoreConfig()), checkpoints=checkpoints)
    reloaded = dataset.fov("FOV_001").load_checkpoint("candidates", checkpoints=checkpoints)
    for name in ("background", "noise", "background_voxels", "box_voxels"):
        np.testing.assert_array_equal(getattr(reloaded.intensity_result, name), getattr(fov.intensity_result, name))
    reloaded = dataset.fov("FOV_001").load_checkpoint("pre_qc", checkpoints=checkpoints)
    pd.testing.assert_frame_equal(reloaded.scoring_result.table, fov.scoring_result.table, check_exact=True)
    assert reloaded.scoring_result.readout_mode == "direct"
    assert json.loads((tmp_path / "ck" / "FOV_001" / "pre_qc.json").read_text())["layout"] is None


@pytest.mark.validation
@pytest.mark.parametrize("decoder", ["wta", "codebook_aware"])
@pytest.mark.parametrize("seed", fx.GOLDEN_SEEDS)
def test_r17_reruns_from_retained_measurements_equal_the_full_run(tmp_path, seed, decoder):
    dataset, full, checkpoints = golden_run(tmp_path, decoder=decoder, seed=seed)
    stages = readout_config("pipeline", decoder=decoder)
    # From candidates: decode, score and filter without images.
    rerun = dataset.fov("FOV_001").load_checkpoint("candidates", checkpoints=checkpoints)
    assert not rerun.images
    rerun.run(PipelineConfig(decoding=stages.decoding, scoring=stages.scoring, filtering=stages.filtering))
    assert_reads_equal(rerun, full)
    # From candidates and pre_qc: rescore without decoding, then filter.
    rescored = dataset.fov("FOV_001").load_checkpoint("candidates", checkpoints=checkpoints)
    rescored.load_checkpoint("pre_qc", checkpoints=checkpoints)
    rescored.run(PipelineConfig(scoring=stages.scoring, filtering=stages.filtering))
    assert_reads_equal(rescored, full)
    # From pre_qc: filter only.
    filtered = dataset.fov("FOV_001").load_checkpoint("pre_qc", checkpoints=checkpoints)
    filtered.run(PipelineConfig(filtering=stages.filtering))
    assert_reads_equal(filtered, full)


def _as_141c093(directory):
    """Rewrite the headers of a background-free, unscored checkpoint into the 141c093 layout (no §2.8 keys)."""
    for stage, keys in (("candidates", ("background_config", "image_background", "image_noise", "readout_mode")),
                        ("pre_qc", ("scoring_config", "deduplication_config", "layout", "readout_mode",
                                    "stages_applied"))):
        path = directory / f"{stage}.json"
        header = json.loads(path.read_text())
        for key in keys:
            header.pop(key)
        path.write_text(json.dumps(header, indent=2) + "\n")


@pytest.mark.validation
@pytest.mark.parametrize("seed", fx.GOLDEN_SEEDS)
def test_r16_a_141c093_checkpoint_loads_without_background_or_score_and_scoring_raises(tmp_path, seed):
    no_background = NeighborhoodSumConfig(background=None)
    dataset, fov, checkpoints = golden_run(tmp_path, extraction=no_background, scoring=None, seed=seed)
    directory = tmp_path / "checkpoints" / "FOV_001"
    _as_141c093(directory)
    loaded = dataset.fov("FOV_001").load_checkpoint("candidates", checkpoints=checkpoints)
    intensity = loaded.intensity_result
    assert intensity.config == no_background and intensity.background is None and intensity.box_voxels is None
    np.testing.assert_array_equal(intensity.values, fov.intensity_result.values)
    pre_qc = read_checkpoint(directory, "pre_qc")
    assert pre_qc["scoring_result"] is None
    pd.testing.assert_frame_equal(pre_qc["decoding_result"].table, fov.decoding_result.table)
    with pytest.raises(ValueError, match="rerun extraction"):
        loaded.run(PipelineConfig(decoding=WtaDecoderConfig(), scoring=ReadScoreConfig()))
    assert loaded.decoding_result is None
    loaded.run(PipelineConfig(decoding=WtaDecoderConfig()))
    with pytest.raises(ValueError, match="rerun extraction"):
        loaded.score_reads()
    with pytest.raises(ValueError, match="rerun extraction"):
        score_reads(loaded.decoding_result, intensity, reference=dataset.codebook)


@pytest.mark.contract
def test_run_checks_the_scoring_stage_before_processing(tmp_path):
    dataset = golden_dataset(tmp_path)
    dataset.codebook = golden_codebook(tmp_path)
    fov = dataset.fov("FOV_001")
    fov.images = fixture_rounds()
    fov.metadata = {label: METADATA for label in ROUNDS}
    fov.spot_result = candidates()
    # Without reads there is nothing to score: the stage is skipped.
    fov.run(PipelineConfig(extraction=NeighborhoodSumConfig(), scoring=ReadScoreConfig()))
    assert fov.scoring_result is None and fov.intensity_result is not None
    fov.intensity_result = None
    with pytest.raises(ValueError, match="rerun extraction"):
        fov.run(PipelineConfig(extraction=NeighborhoodSumConfig(background=None), decoding=WtaDecoderConfig(),
                               scoring=ReadScoreConfig()))
    assert fov.intensity_result is None
    with pytest.raises(TypeError):
        PipelineConfig(scoring={"method": "bgcorr_probability"})
    # Scoring is opt-in in the Python API; the results mapping lists it after decoding.
    assert PipelineConfig().scoring is None
    fov.run(PipelineConfig(extraction=NeighborhoodSumConfig(), decoding=WtaDecoderConfig(), scoring=ReadScoreConfig()))
    assert list(fov.results) == ["spot_finding", "extraction", "decoding", "scoring"]
    assert isinstance(fov.results["scoring"], ReadScoringResult)
    # Decoding again drops the score of the earlier reads.
    fov.decode_barcodes(config=WtaDecoderConfig())
    assert fov.scoring_result is None


# --- workflow adapter -------------------------------------------------------------------------------

def workflow(**parameters):
    return {"n_rounds": 4, "ref_round": "round1", "dataset_id": "d", "sample_id": "s", "output_id": "o",
            "root_input_path": "in", "root_output_path": "out", "seq_channel_order": ["ch00", "ch01", "ch02", "ch03"],
            "rules": {"rsf_single_fov": {"parameters": {"load_raw_images": {"run": False}, **parameters}}}}


@pytest.mark.contract
def test_workflow_adapter_scores_whenever_it_decodes():
    extraction = {"reads_extraction": {"run": True}}
    adapted = from_workflow_config(workflow(**extraction, reads_filtration={"run": True}))
    assert adapted.pipeline.scoring == ReadScoreConfig()
    assert adapted.pipeline.extraction == NeighborhoodSumConfig()
    assert from_workflow_config(workflow(**extraction)).pipeline.scoring is None
    off = from_workflow_config(workflow(**extraction, reads_filtration={"run": True}, scoring={"run": False}))
    assert off.pipeline.scoring is None
    with pytest.raises(ValueError, match="requires reads_filtration.run"):
        from_workflow_config(workflow(**extraction, scoring={"run": True}))
    with pytest.raises(ValueError, match="unknown scoring keys"):
        from_workflow_config(workflow(**extraction, reads_filtration={"run": True}, scoring={"cutoff": 1.0}))


@pytest.mark.contract
def test_workflow_adapter_background_key():
    def extraction(**keys):
        parameters = workflow(reads_extraction={"run": True, **keys}, reads_filtration={"run": True},
                              scoring={"run": False})
        return from_workflow_config(parameters).pipeline.extraction

    assert extraction(background=False).background is None
    assert extraction(background={"inner_radius_zyx": [1, 4, 4], "outer_radius_zyx": [1, 8, 8],
                                  "min_voxels": 20}).background.outer_radius_zyx == (1, 8, 8)
    # Without the key, the default ring grows along the axes where the box exceeds its inner box.
    grown = extraction(voxel_size=[2, 2, 1]).background
    assert (grown.inner_radius_zyx, grown.outer_radius_zyx, grown.min_voxels) == ((2, 3, 3), (2, 6, 6), 16)
    with pytest.raises(ValueError, match="unknown reads_extraction.background keys"):
        extraction(background={"radius": 3})
    with pytest.raises(ValueError, match="needs the local background"):
        from_workflow_config(workflow(reads_extraction={"run": True, "background": False},
                                      reads_filtration={"run": True}))
    schema = yaml.safe_load((ROOT / "workflow/schemas/config.schema.yaml").read_text())["$defs"]
    assert schema["scoring_params"]["properties"]["method"]["enum"] == [ReadScoreConfig().method]
    assert "background" in schema["reads_extraction_params"]["properties"]
    for rule in ("rsf_single_fov_config", "lrsf_single_fov_subtile_config", "deep_rsf_subtile_config"):
        assert schema[rule]["properties"]["parameters"]["properties"]["scoring"] == {"$ref": "#/$defs/scoring_params"}


@pytest.mark.contract
def test_direct_assignment_config_scores_in_direct_mode_through_the_adapter_default(tmp_path):
    adapted = from_workflow_config({**workflow(spot_finding={"run": True, "rounds": ["round1"]},
                                               reads_extraction={"run": True}, reads_filtration={"run": True}),
                                    "readout_mode": "direct", "backend": "python"})
    assert adapted.pipeline.decoding == DirectAssignmentConfig() and adapted.pipeline.scoring == ReadScoreConfig()
