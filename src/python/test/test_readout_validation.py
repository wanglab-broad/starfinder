"""§2.8 engineering validation checks not covered by the task-group modules (W-296).

docs/readout-algorithms.md, "Engineering validation design". The task groups implemented
R1 (test_readout_encodings.py), R2 (test_readout_layout.py), R4 (test_readout_entries.py),
R7 (test_readout_direct.py), R9 and R10 (test_readout_background.py), R11, R12, R16 and
R17 (test_readout_scoring.py) and R13 to R15, R17 and R18 (test_readout_deduplication.py).
This module adds:

* R2 on ``two_seg``: gene calls equal those of the same colors in a one-segment codebook,
  and the per-segment end checks of entry reads and of the planted wrong-end reads;
* R3 on ``one_base``: the golden geometry with a one_base codebook;
* R5 on ``dropout``: required rounds and acquisition-local failures on the golden geometry;
* R6: the golden pins and the 141c093 output on ``cal`` seed 103, without the new columns (on
  ``cal`` the float columns of the decoding tables within 1e-12, every other column exact);
* R8 (extended tier) on ``cal_direct``: the ranking of direct calls;
* R16 on ``two_seg``: checkpoints with background, score, deduplication and two segments;
* R19: SHA-256 of every table of ``golden`` and ``cal`` seed 103 in three single-thread
  processes.

Hand-built fixtures use seeds 100 to 102 (the golden geometry with each seed in place of
the golden test's SEED; R6 and R19 use the pinned golden fixture itself), and the
W-278-derived scenes seeds 103 to 105 with W-278's conditions, pipeline and matching
(readout_scenes.py). Every tolerance below is the design's, unchanged; each cites its
W-278 row or is marked provisional with its reason.

Run as ``python -m test.test_readout_validation OUT.json`` from src/python, the module
writes the R19 digests of one process to OUT.json.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

from starfinder.barcode import (Codebook, CodebookAwareDecoderConfig, DeduplicationConfig, DirectAssignmentConfig,
                                DirectPanel, NeighborhoodSumConfig, OneBaseEncodingConfig, ReadFilterConfig,
                                ReadScoreConfig, WtaDecoderConfig, assign_direct, decode_barcodes,
                                deduplicate_reads, extract_intensities, filter_reads, load_codebook, score_reads)
from starfinder.barcode._codebook_aware import _channel_probabilities
from starfinder.barcode.deduplication import DEDUPLICATION_COLUMNS
from starfinder.barcode.scoring import SCORE_COLUMNS
from starfinder.dataset import CheckpointConfig, PipelineConfig
from starfinder.evaluation.barcode import evaluate_decoding, ranking_quality
from starfinder.evaluation.matching import match_points
from starfinder.io import ImageLoadResult, read_checkpoint
from starfinder.io._checkpoint import candidates_frame
from starfinder.spot_finding import LocalMaximaConfig, SpotFindingResult, find_spots

from . import readout_fixtures as fx
from . import readout_scenes as scenes
from . import test_readout_golden as golden

pytestmark = [pytest.mark.barcode]

SRC = Path(__file__).resolve().parents[1]
HAND_SEEDS = (100, 101, 102)
DECODERS = {"wta": WtaDecoderConfig(), "codebook_aware": CodebookAwareDecoderConfig()}
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "NUMBA_NUM_THREADS")


golden_rounds = fx.golden_rounds


def loaded(rounds, channels=golden.CHANNELS):
    return {label: ImageLoadResult(image, golden.METADATA, tuple(channels), (), {}) for label, image in rounds.items()}


def extract_golden(rounds, spots=None):
    return extract_intensities(loaded(rounds), golden.candidates() if spots is None else spots,
                               config=NeighborhoodSumConfig((1, 2, 2)))


def without(table, columns):
    return table.drop(columns=[c for c in columns if c in table])


# --- R2: two_seg -----------------------------------------------------------------------------------

@pytest.mark.validation
@pytest.mark.parametrize("decoder", list(DECODERS))
@pytest.mark.parametrize("seed", HAND_SEEDS)
def test_r2_two_seg_calls_equal_one_segment_and_segment_ends(tmp_path, seed, decoder):
    truth = fx.two_seg_truth()
    fov = fx.two_seg_fov(tmp_path, seed).run(PipelineConfig(
        extraction=NeighborhoodSumConfig(), decoding=DECODERS[decoder], scoring=ReadScoreConfig(),
        filtering=ReadFilterConfig()))
    assert [s.name for s in fov.codebook.layout.segments] == ["A", "B"]
    # The same colors in a one-segment codebook (no layout ends) give the same calls.
    one = Codebook(fov.codebook.table[["gene_id", "color_sequence"]], fx.ROUNDS, fx.CHANNELS)
    assert len(one.layout.segments) == 1 and not one.layout.segments[0].ends
    plain = decode_barcodes(fov.intensity_result, one, config=DECODERS[decoder]).table
    pd.testing.assert_frame_equal(without(fov.decoding_result.table, ["entry_id"]), without(plain, ["entry_id"]),
                                  check_exact=True)
    reads = fov.filtering_result.table
    assert reads.observed_color_sequence.tolist() == truth.colors.tolist()
    entries = truth.kind.eq("entry").to_numpy()
    wrong = truth.kind.eq("wrong_ends").to_numpy()
    assert reads.gene_id[entries].tolist() == [fx.TWO_SEG_ENTRIES[c] for c in truth.colors[entries]]
    # Every read equal to an entry passes both segments' end checks; the 4 wrong-end reads pass
    # segment A and fail segment B.
    assert reads.endpoint_valid_A[entries].all() and reads.endpoint_valid_B[entries].all()
    assert wrong.sum() == 4
    assert reads.endpoint_valid_A[wrong].all() and not reads.endpoint_valid_B[wrong].any()


# --- R3: one_base ----------------------------------------------------------------------------------

ONE_BASE_MAPPING = {"A": "1", "C": "2", "G": "3", "T": "4"}
# The golden spots whose planted colors are a codeword without a planted perturbation.
CLEAN_SPOTS = ("0", "1", "2", "3", "8", "9")


def one_base_codebook(directory):
    """The golden codewords as 4-base one_base barcodes (A→1, C→2, G→3, T→4), gene,barcode rows."""
    inverse = {color: base for base, color in ONE_BASE_MAPPING.items()}
    path = directory / "one_base.csv"
    path.write_text("".join(f"{gene},{''.join(inverse[c] for c in colors)}\n"
                            for colors, gene in golden.CODEWORDS.items()))
    return load_codebook(path, round_labels=golden.ROUNDS, channel_labels=golden.CHANNELS,
                         encoding=OneBaseEncodingConfig(ONE_BASE_MAPPING))


@pytest.mark.validation
@pytest.mark.parametrize("decoder", list(DECODERS))
@pytest.mark.parametrize("seed", HAND_SEEDS)
def test_r3_one_base_decoding_equals_the_two_base_run(tmp_path, seed, decoder):
    book = one_base_codebook(tmp_path)
    assert book.table.color_sequence.tolist() == list(golden.CODEWORDS)
    assert book.table.entry_id.tolist() == ["ACGT", "CATG", "GTAC", "TGCA", "CCCC", "GAGA", "TCTC", "AGAG"]
    intensities = extract_golden(golden_rounds(seed))
    table = decode_barcodes(intensities, book, config=DECODERS[decoder]).table
    # Truth: the golden spot positions with their planted genes, for the clean reads.
    spots = golden.candidates().spots
    clean = spots.spot_id.isin(CLEAN_SPOTS).to_numpy()
    truth = pd.DataFrame({"gene_id": [golden.CODEWORDS[golden.SPOTS[s][3]] for s in spots.spot_id[clean]],
                          "color_sequence": [golden.SPOTS[s][3] for s in spots.spot_id[clean]]})
    points = spots[["z", "y", "x"]].to_numpy(float)
    matches = match_points(points[clean], points, reference_metadata=golden.METADATA,
                           observed_metadata=golden.METADATA, **scenes.MATCH)
    evaluation = evaluate_decoding(table, truth, matches=matches)
    assert evaluation.counts["eligible_gene_id"] == len(CLEAN_SPOTS)
    assert evaluation.values["gene_id_accuracy"] == 1.0
    assert table.call_status[clean].eq("assigned").all()
    # Statuses (and every other column but entry_id) equal the golden two-base run of the same colors.
    two_base = decode_barcodes(intensities, golden.golden_codebook(tmp_path), config=DECODERS[decoder]).table
    pd.testing.assert_frame_equal(without(table, ["entry_id"]), without(two_base, ["entry_id"]), check_exact=True)
    statuses = dict(zip(table.spot_id, zip(table.call_status, table.failure_reason)))
    assert statuses == golden.PINNED_STATUS[decoder]


# --- R5: dropout -------------------------------------------------------------------------------------

DROPOUT_ROUND = 2                      # round 3: zero in a 9×20×20 region
DROPOUT_REGION = (slice(2, 11), slice(0, 20), slice(8, 28))
DROPOUT_COVERED = ("0", "3", "9")
MASKED_ROUND, MASKED_SPOTS = 1, ("1", "8")   # round 2: valid=False (a caller mask)
BAND_ROUND, BAND_WIDTH = 3, 4          # round 4: a zero band at the x=47 face (registration fill)
# A supplementary candidate inside the band, beyond the design's ten: no golden candidate's box
# reaches the band (the largest x is 40, box to 42), so it makes part (iii) a nonempty known answer.
BAND_CANDIDATE = (6.0, 24.0, 46.0)


def dropout_rounds(seed):
    rounds = golden_rounds(seed)
    zeroed, banded = golden.ROUNDS[DROPOUT_ROUND], golden.ROUNDS[BAND_ROUND]
    rounds[zeroed] = rounds[zeroed].copy()
    rounds[zeroed][DROPOUT_REGION] = 0
    rounds[banded] = rounds[banded].copy()
    rounds[banded][:, :, -BAND_WIDTH:, :] = 0
    return rounds


def box_overlap(points, region, radius=(1, 2, 2), shape=golden.SHAPE_ZYX):
    """(inside, touching): indices of the points whose extraction box (clipped to the image) lies
    inside region, and of those whose box intersects it."""
    lo = np.array([s.start for s in region])
    hi = np.array([s.stop for s in region])
    centers = np.floor(points + 0.5).astype(np.int64)
    box_lo = np.maximum(centers - radius, 0)
    box_hi = np.minimum(centers + np.asarray(radius) + 1, shape)
    inside = ((box_lo >= lo) & (box_hi <= hi)).all(axis=1)
    touching = (np.maximum(box_lo, lo) < np.minimum(box_hi, hi)).all(axis=1)
    return np.flatnonzero(inside), np.flatnonzero(touching)


@pytest.mark.validation
@pytest.mark.parametrize("decoder", list(DECODERS))
@pytest.mark.parametrize("seed", HAND_SEEDS)
def test_r5_required_rounds_and_acquisition_local_failures(tmp_path, seed, decoder):
    book = golden.golden_codebook(tmp_path)
    spots = golden.candidates()
    ids = spots.spots.spot_id.tolist()
    points = spots.spots[["z", "y", "x"]].to_numpy(float)
    band = (slice(0, golden.SHAPE_ZYX[0]), slice(0, golden.SHAPE_ZYX[1]),
            slice(golden.SHAPE_ZYX[2] - BAND_WIDTH, golden.SHAPE_ZYX[2]))
    # The fixture's geometry: the region covers exactly 3 candidates' boxes and touches no other;
    # no golden candidate's round-4 box touches the band.
    inside, touching = box_overlap(points, DROPOUT_REGION)
    assert [ids[i] for i in inside] == [ids[i] for i in touching] == list(DROPOUT_COVERED)
    assert len(box_overlap(points, band)[1]) == 0
    assert not set(MASKED_SPOTS) & set(DROPOUT_COVERED)

    config = DECODERS[decoder]
    clean = extract_golden(golden_rounds(seed))
    reference = decode_barcodes(clean, book, config=config).table
    perturbed = extract_golden(dropout_rounds(seed))
    valid = perturbed.valid.copy()
    masked = np.isin(ids, MASKED_SPOTS)
    valid[masked, MASKED_ROUND] = False
    perturbed = replace(perturbed, valid=valid)
    table = decode_barcodes(perturbed, book, config=config).table
    covered = np.isin(ids, DROPOUT_COVERED)
    # (i) exactly the 3 covered candidates are no_signal / zero_signal_round.
    zero_round = table.failure_reason.eq("zero_signal_round").to_numpy()
    assert zero_round.tolist() == (covered | (table.spot_id == golden.ZERO_SPOT).to_numpy()).tolist()
    assert table.call_status[covered].eq("no_signal").all()
    # (ii) the 2 masked candidates are unmatched / invalid_measurement, never rescued.
    assert table.call_status[masked].eq("unmatched").all()
    assert table.failure_reason[masked].eq("invalid_measurement").all()
    assert table.call_type[masked].eq("no_call").all() and table.gene_id[masked].isna().all()
    assert table.decoded_color_sequence[masked].isna().all() and table.entry_id[masked].isna().all()
    # (iii) no golden candidate's box lies in the band, so the band changes no read; every other row
    # equals the unperturbed run.
    other = ~(covered | masked)
    pd.testing.assert_frame_equal(table[other], reference[other], check_exact=True)
    # The score is NaN with no_assignment for every perturbed read.
    scored = score_reads(decode_barcodes(perturbed, book, config=config), perturbed, reference=book).table
    assert scored.loc[covered | masked, "qc_score"].isna().all()
    assert scored.qc_reason[covered | masked].eq("no_assignment").all()

    # (iii), supplementary: a candidate whose round-4 box lies in the band is no_signal; the other
    # ten rows are unchanged by it.
    frame = spots.spots
    extra = pd.concat([frame, pd.DataFrame({"spot_id": pd.array(["10"], dtype="string"),
                                            "z": [BAND_CANDIDATE[0]], "y": [BAND_CANDIDATE[1]],
                                            "x": [BAND_CANDIDATE[2]]})], ignore_index=True)
    assert box_overlap(extra[["z", "y", "x"]].to_numpy(float), band)[0].tolist() == [10]
    with_band = SpotFindingResult(extra, spots.metadata, spots.spot_namespace, spots.config, spots.diagnostics)
    banded = extract_golden(dropout_rounds(seed), with_band)
    banded = replace(banded, valid=np.vstack([valid, banded.valid[-1:]]))
    band_table = decode_barcodes(banded, book, config=config).table
    assert tuple(band_table.iloc[10][["call_status", "failure_reason", "call_type"]]) == (
        "no_signal", "zero_signal_round", "no_call")
    pd.testing.assert_frame_equal(band_table.iloc[:10], table, check_exact=True)
    band_scored = score_reads(decode_barcodes(banded, book, config=config), banded, reference=book).table
    assert np.isnan(band_scored.qc_score.iloc[10]) and band_scored.qc_reason.iloc[10] == "no_assignment"


# --- R6: multiplexed regression ------------------------------------------------------------------------

# Digests of the W-278 cal pipeline (readout_scenes.pipeline: LocalMaximaConfig() on round 1,
# NeighborhoodSumConfig((1, 2, 2)), both decoders at their defaults) at seed 103, computed with
# the 141c093 source tree (W-296 run directory, scripts/w296_r6_pins.py and
# logs/r6-141c093-digests.json): condition -> (spots table, values, valid, WTA table,
# codebook-aware table), with test_readout_golden's digest and table_digest. Provisional: no
# W-278 row covers it; the bound is the 141c093 behavior.
PINNED_CAL_141C093 = {
    "noise": ("f9c05b5412241288578389f3088c0e85c1a8e3dbf23d7a6288a4ac4eb4cb99c0",
              "f6b4d7b045b1e4ce9534ea7a1a7c583000cb202903842d6be9bbcd8f119dcf31",
              "b323ab77e11846fcc8e89d8c91d5f8923ed5e6a3dbadc0bc4bed3acfd7738ab5",
              "49396d89cb666119c2ebf17609f8115ca1ef23d90c99830f4ca26a36590fdfbc",
              "1224da512110588d7c8b46312efea933345fc113495b00e243fa32ee009ddc2e"),
    "mixing": ("a5694e3403ce5e601b8620643b32dac19bf68c2f02bd221cb736ff0f71f667c2",
               "ffb315e69a267b4edd8ff28267c652cf01aee582fa74a824c904dbf4056b67fe",
               "7bd539e41210589b64a27caf31703d8cb9d67bd2d9c924d417ba67f83b99c01b",
               "c8ca7c77c8cc7b7d64b53c3f1655385ed5fd729442139bf4df011561b3c9b696",
               "a9923d51d8b8762f540289a473cec6a84d848b41fa313719c6f2ec9ffad322f7"),
    "weakening": ("bb5d7ee84439c505d1aa4a465a67636499a6d318730bb772128503741625c3aa",
                  "351bf3c237e615f5f7b00b145af1d90b5c826a4f69c98a87cd078ae6da62cae5",
                  "7bd539e41210589b64a27caf31703d8cb9d67bd2d9c924d417ba67f83b99c01b",
                  "51b644fc6db9185f8bc8e1baf98ea2b7a1a8613645f59f45afd2dc6908bb6041",
                  "1a46358adb7f510ad87e52f833d08e87d9101c5000bf1914033c1c71ebd53bf8"),
    "gain": ("bde11b805e9f68dfc7a1bc4587f4c21ebd89a6773c31a54b4df8c0df5036e4e8",
             "f2dda07e6899ca72b82b7fe3cd09c18139650dd9e746f0ee125b330329332553",
             "4319753fafca708cab5a74f7daa3af9aa27b1fbfcfdf1e149c1d76f6f54da649",
             "ad44779c5673902564b5fab47957541d65fe13b7fbc098aa34343014989acbfb",
             "c6d1ec840ff8097b787c7b21604d753332a11506d880e035a30a65aa774978d0"),
    "trend": ("82dc5af3b41af23f9aa05c83f3c40230ce04585737dfb1154608e02285ece004",
              "ff424fcd5025965b463be57e8ab6491041b2da4f5ed50a400b7858422e69a245",
              "b323ab77e11846fcc8e89d8c91d5f8923ed5e6a3dbadc0bc4bed3acfd7738ab5",
              "2c43791617bc8178a0b4e8b4efd97af03e3c5cc4da248c364bc21545c528c305",
              "9237fa15d051f3832805f308d16d5ab569418a783208fcd3f5fb20f92777b829"),
    "round_effect_only": ("279d93f5f5a05739ec01d4dd5eba05e16d5ebb7a2e91e239673cb255c208fa6b",
                          "68ef6e27aa46d177028457f78b23f34604c0686e20a6afc6647b2fa592e7c510",
                          "b18621f4d3e148abd10f12c20836d9cb2c5de7c7dd0318f40f63b615f977fb07",
                          "2f285f0327ea1ccde740381071603b60bfc01db27acbb89a09257e48cecb98d1",
                          "309155f713ea16c5fd1c150fe2b58cf3c189809cc0ea273d9706d145dce5d986"),
    "background_only": ("e722433c77efd7e694c55947cf63285a4433d009cb1155ecb11e02b9458e8586",
                        "8123b6fccc3f17848b100c16c9122129e46b60c3299765536e329bd6f6fc4dcc",
                        "4319753fafca708cab5a74f7daa3af9aa27b1fbfcfdf1e149c1d76f6f54da649",
                        "376f826221406a4fe3cec1ad46bb39574d1215dc1c5ffa6ad36ecd256e4029cc",
                        "e9fa5b67e980c7cac3052eff862c83295453471bb86d736f6dcbdf9c607a116f"),
    "combined": ("ed7b1ec4739e118a014d24320a90668460682a4f7dda9c8bef75eaa84c7d7611",
                 "3338a998ddff0c6df783a8ce4388182c12e2e8c1679d13a4e3bb3fb55e8b725d",
                 "b18621f4d3e148abd10f12c20836d9cb2c5de7c7dd0318f40f63b615f977fb07",
                 "8b74d844b0e4bfdfefb1a89ede06cb089ed2cdebf68e6a33e7b760deaed2ce51",
                 "c0b342d1d84a81d772593e5eee4341121f8849c1f6d8a6de1763b9cea04c7ebf"),
}


#: R6 on ``cal`` compares the float columns of the decoding tables numerically. Provisional: no
#: W-278 row covers it. The last bit of np.log and np.sqrt depends on the CPU, so the %.17g digest
#: of a float column holds on the host that wrote the pins and not on every host; 1e-12 is far
#: below any digit a decoding decision or a printed score depends on.
R6_FLOAT_TOLERANCE = 1e-12
R6_DECODERS = ("wta", "codebook_aware")
#: The float columns of the 141c093 decoding tables on ``cal`` seed 103 (written by write_r6_reference).
R6_REFERENCE = Path(__file__).parent / "data" / "readout_r6_cal_141c093"


def cal_tables(condition):
    """(spots, intensities, decoding tables without the §2.8 columns) of the W-278 pipeline at seed 103."""
    spots, intensities, tables, _ = scenes.pipeline(condition, 103)
    return spots, intensities, {name: without(tables[name], ["entry_id", *SCORE_COLUMNS]) for name in R6_DECODERS}


def float_columns(table):
    return list(table.select_dtypes(include="floating").columns)


def r6_reference(condition, decoder):
    """The 141c093 float columns of one decoding table; %.17g text round-trips float64 exactly."""
    return pd.read_csv(R6_REFERENCE / f"{condition}_{decoder}.csv", dtype=np.float64, float_precision="round_trip")


def write_r6_reference(directory=R6_REFERENCE):
    """Write the float columns of the ``cal`` decoding tables as the R6 reference.

    Run from src/python on a host where the %.17g table digests equal PINNED_CAL_141C093, which
    this function checks first: there the tables are the 141c093 output bit for bit.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for condition in scenes.CAL_CONDITIONS:
        _, _, tables = cal_tables(condition)
        for name, pin in zip(R6_DECODERS, PINNED_CAL_141C093[condition][3:]):
            if golden.table_digest(tables[name]) != pin:
                raise RuntimeError(f"{condition} {name}: this host does not reproduce the 141c093 digest")
            tables[name][float_columns(tables[name])].to_csv(
                directory / f"{condition}_{name}.csv", index=False, float_format="%.17g", na_rep="nan")


@pytest.mark.validation
@pytest.mark.parametrize("decoder", list(DECODERS))
def test_r6_golden_tables_without_the_new_columns_equal_the_pins(tmp_path, decoder):
    intensities = golden.extract()
    assert (golden.digest(intensities.values), golden.digest(intensities.valid)) == golden.PINNED_EXTRACTION[(1, 2, 2)]
    decoded = decode_barcodes(intensities, golden.golden_codebook(tmp_path), config=DECODERS[decoder])
    assert golden.table_digest(golden.without_entry(decoded.table)) == golden.PINNED_DECODING[decoder]
    for name in golden.FILTERS:
        options = golden.filter_options(decoder, name)
        result = filter_reads(decoded, config=golden.readout_config("filtering", decoder=decoder, **options))
        assert golden.table_digest(golden.without_entry(result.table)) == golden.PINNED_FILTERING[(decoder, name)][1]
    scored = score_reads(decoded, intensities, reference=golden.golden_codebook(tmp_path))
    assert golden.table_digest(golden.without_entry(golden.without_score(scored.table))) == (
        golden.PINNED_DECODING[decoder])


@pytest.mark.validation
@pytest.mark.parametrize("condition", scenes.CAL_CONDITIONS)
def test_r6_cal_seed_103_without_the_new_columns_equals_141c093(condition):
    spots, intensities, tables = cal_tables(condition)
    pinned = PINNED_CAL_141C093[condition]
    assert (golden.table_digest(spots.spots), golden.digest(intensities.values),
            golden.digest(intensities.valid)) == pinned[:3]
    for name, pin in zip(R6_DECODERS, pinned[3:]):
        table, reference = tables[name], r6_reference(condition, name)
        columns = float_columns(table)
        assert list(reference.columns) == columns and len(reference) == len(table)
        # Every other column is exact: with the 141c093 float columns in place of the computed
        # ones, the table has the 141c093 digest.
        with_reference = table.copy()
        with_reference[columns] = reference.to_numpy()
        assert golden.table_digest(with_reference) == pin, name
        np.testing.assert_allclose(table[columns].to_numpy(), reference.to_numpy(), rtol=R6_FLOAT_TOLERANCE,
                                   atol=R6_FLOAT_TOLERANCE, equal_nan=True, err_msg=name)


# --- R8: direct assignment on the calibrated scenes (extended tier) --------------------------------------

def direct_panel(book, label):
    """The W-278 direct.csv panel of one round: genes color-1 to color-4 at their colors' channels."""
    channel = {k: book.channel_labels[book.color_to_channel[k]] for k in "1234"}
    return DirectPanel(pd.DataFrame({"round": [label] * 4, "channel": [channel[k] for k in "1234"],
                                     "gene_id": [f"color-{k}" for k in "1234"]}))


def direct_calls(condition, seed):
    """Assigned, matched direct calls of every round of one scene: correct, qc_score, probability_nll."""
    book, the_scene = scenes.scene(condition, seed)
    rows = []
    for r, label in enumerate(the_scene.round_labels):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            found = find_spots(the_scene.rounds[label], config=LocalMaximaConfig(), metadata=the_scene.metadata,
                               spot_namespace=f"w296/{condition}/{seed}/{label}")
        frame = found.spots.assign(round=pd.array([label] * len(found.spots), dtype="string"))
        spots = SpotFindingResult(frame, found.metadata, found.spot_namespace, found.config, found.diagnostics)
        image = {label: scenes.loaded(the_scene.rounds[label], the_scene.metadata, book.channel_labels)}
        intensities = extract_intensities(image, spots, config=scenes.EXTRACTION, readout_mode="direct")
        panel = direct_panel(book, label)
        decoded = assign_direct(intensities, spots, panel, config=DirectAssignmentConfig())
        table = score_reads(decoded, intensities, reference=panel).table
        # The decoder's probability NLL of the same call: the codebook-aware decoder's channel
        # probabilities (sums clipped at 0, + 1e-6) at the assigned channel, floored at 1e-12.
        probabilities = _channel_probabilities(intensities.values)[:, :, 0]
        channel = frame.channel.to_numpy()
        nll = -np.log(np.maximum(probabilities[np.arange(len(frame)), channel], 1e-12))
        truth = scenes.truth(the_scene, label)
        colors = the_scene.formed.set_index("amplicon_id").loc[list(the_scene.amplicon_ids)].color_sequence.str[r]
        matches = match_points(truth[["z", "y", "x"]].to_numpy(float), frame[["z", "y", "x"]].to_numpy(float),
                               reference_metadata=the_scene.metadata, observed_metadata=the_scene.metadata,
                               eligible_reference=truth.center_in_bounds.to_numpy(bool), **scenes.MATCH)
        truth_gene = np.full(len(frame), None, dtype=object)
        for i, j, _ in matches.details["matched_pairs"]:
            truth_gene[j] = f"color-{colors.iloc[i]}"
        keep = table.call_status.eq("assigned").to_numpy() & np.array([g is not None for g in truth_gene])
        rows.append(pd.DataFrame({
            "condition": condition, "seed": seed, "round": label,
            "correct": (table.gene_id.astype(object).to_numpy() == truth_gene)[keep],
            "qc_score": table.qc_score.to_numpy()[keep], "probability_nll": nll[keep]}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def r8_calls():
    return pd.concat([direct_calls(c, s) for c in scenes.CONDITIONS for s in scenes.HELDOUT_SEEDS],
                     ignore_index=True)


def auroc(calls, column):
    result = ranking_quality(calls[column].to_numpy(), calls.correct.to_numpy(dtype=bool), orientation="lower")
    return result.values["auroc"], result


@pytest.mark.extended
@pytest.mark.slow
@pytest.mark.validation
def test_r8_direct_qc_score_ranks_above_the_decoder_score(r8_calls):
    print("\nR8 pooled AUROC of direct calls (qc_score, probability_nll, calls, incorrect):")
    # Pooled over conditions and seeds: the nine W-278 conditions (all_calibrated, as direct.csv)
    # and the eight cal conditions. Provisional: W-278 direct.csv (D1 0.930 ± 0.004 against 0.878
    # and 0.849) measured decoder calls on a one-round panel, not this assignment.
    for pool, conditions in (("all_calibrated", scenes.CONDITIONS), ("cal", scenes.CAL_CONDITIONS)):
        calls = r8_calls[r8_calls.condition.isin(conditions)]
        (qc, qc_result), (reference, _) = auroc(calls, "qc_score"), auroc(calls, "probability_nll")
        print(f"  {pool}: {qc:.4f} (SE {qc_result.values['auroc_se']:.4f}) {reference:.4f} {len(calls)} "
              f"{int((~calls.correct).sum())}")
        assert qc > reference, pool
    print("R8 per condition (reported, not gated; undefined without an incorrect call):")
    for condition, group in r8_calls.groupby("condition", sort=False):
        print(f"  {condition}: qc_score {auroc(group, 'qc_score')[0]} "
              f"probability_nll {auroc(group, 'probability_nll')[0]} incorrect {int((~group.correct).sum())}")


# --- R16: two_seg checkpoints ----------------------------------------------------------------------------

@pytest.mark.validation
@pytest.mark.parametrize("table_format", ["csv", "parquet"])
@pytest.mark.parametrize("seed", HAND_SEEDS)
def test_r16_two_seg_checkpoints_round_trip_exactly(tmp_path, seed, table_format):
    checkpoints = CheckpointConfig(stages=("candidates", "pre_qc"), directory=tmp_path / "checkpoints",
                                   table_format=table_format)
    fov = fx.two_seg_fov(tmp_path, seed).run(PipelineConfig(
        extraction=NeighborhoodSumConfig(), decoding=CodebookAwareDecoderConfig(), scoring=ReadScoreConfig(),
        deduplication=DeduplicationConfig(), filtering=ReadFilterConfig()), checkpoints=checkpoints)
    directory = tmp_path / "checkpoints" / "FOV_001"
    loaded_candidates = read_checkpoint(directory, "candidates")
    a, b = loaded_candidates["intensity_result"], fov.intensity_result
    for name in ("values", "valid", "box_voxels", "background", "noise", "background_voxels", "image_background",
                 "image_noise"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name, strict=True)
    assert a.config == b.config and a.config.background is not None
    pd.testing.assert_frame_equal(candidates_frame(loaded_candidates["spot_result"], a),
                                  candidates_frame(fov.spot_result, b), check_exact=True)
    reloaded = fov.dataset.fov("FOV_001").load_checkpoint("pre_qc", checkpoints=checkpoints)
    for name in ("decoding_result", "scoring_result", "deduplication_result"):
        pd.testing.assert_frame_equal(getattr(reloaded, name).table, getattr(fov, name).table, check_exact=True)
        assert getattr(reloaded, name).config == getattr(fov, name).config, name
    assert reloaded.deduplication_result.counts == fov.deduplication_result.counts
    header = json.loads((directory / "pre_qc.json").read_text())
    assert header["layout"]["segments"] == [{"name": "A", "bases": 3, "ends": [["C", "C"]]},
                                            {"name": "B", "bases": 3, "ends": [["T", "T"]]}]
    assert header["stages_applied"] == ["decoding", "scoring", "deduplication"]
    assert list(header["dtypes"])[-len(DEDUPLICATION_COLUMNS):] == list(DEDUPLICATION_COLUMNS)
    # Filtering from the reloaded reads with the codebook's segment ends equals the full run.
    again = filter_reads(reloaded.deduplication_result, codebook=fov.codebook).table
    pd.testing.assert_frame_equal(again, fov.filtering_result.table, check_exact=True)


# --- R19: determinism --------------------------------------------------------------------------------------

def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def intensity_digests(prefix, result):
    return {f"{prefix}/{name}": golden.digest(getattr(result, name))
            for name in ("values", "valid", "box_voxels", "background", "noise", "background_voxels",
                         "image_background", "image_noise")}


def determinism_digests():
    """SHA-256 of every table of golden (FOV.run with checkpoints, both decoders) and of cal seed 103
    (the W-278 pipeline of each cal condition: decode, score, deduplicate and filter, both decoders)."""
    out = {}
    with tempfile.TemporaryDirectory(prefix="w296-r19-") as workspace:
        root = Path(workspace)
        for decoder in DECODERS:
            directory = root / decoder
            directory.mkdir()
            dataset = golden.golden_dataset(directory)
            dataset.codebook = golden.golden_codebook(directory)
            fov = dataset.fov("FOV_001")
            fov.images = golden.fixture_rounds()
            fov.metadata = {label: golden.METADATA for label in golden.ROUNDS}
            fov.spot_result = golden.candidates()
            fov.run(golden.readout_config("pipeline", decoder=decoder),
                    checkpoints=golden.readout_config("checkpoints", directory=directory / "checkpoints"))
            prefix = f"golden/{decoder}"
            out.update(intensity_digests(prefix, fov.intensity_result))
            out[f"{prefix}/candidates"] = golden.table_digest(candidates_frame(fov.spot_result, fov.intensity_result))
            for name in ("decoding_result", "scoring_result", "filtering_result"):
                out[f"{prefix}/{name}"] = golden.table_digest(getattr(fov, name).table)
            for name in ("candidates.csv", "pre_qc.csv"):
                out[f"{prefix}/{name}"] = golden.file_digest(directory / "checkpoints" / "FOV_001" / name)
    for condition in scenes.CAL_CONDITIONS:
        book, _ = scenes.scene(condition, 103)
        spots, intensities, _, _ = scenes.pipeline(condition, 103)
        prefix = f"cal/{condition}/103"
        out[f"{prefix}/spots"] = golden.table_digest(spots.spots)
        out.update(intensity_digests(prefix, intensities))
        for decoder, config in DECODERS.items():
            decoded = decode_barcodes(intensities, book, config=config)
            scored = score_reads(decoded, intensities, reference=book)
            deduplicated = deduplicate_reads(scored, spots, intensities, config=DeduplicationConfig())
            filtered = filter_reads(deduplicated, codebook=book)
            for name, result in (("decoding", decoded), ("scoring", scored), ("deduplication", deduplicated),
                                 ("filtering", filtered)):
                out[f"{prefix}/{decoder}/{name}"] = golden.table_digest(result.table)
    return out


def single_thread_command(output):
    """A child process of this interpreter on one CPU (taskset -c 0 where CPU 0 is allowed)."""
    command = [sys.executable, "-m", "test.test_readout_validation", str(output)]
    taskset = shutil.which("taskset")
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    if taskset and allowed:
        cpu = 0 if 0 in allowed else allowed[0]
        command = [taskset, "-c", str(cpu)] + command
    return command


@pytest.mark.slow
@pytest.mark.validation
def test_r19_three_single_thread_processes_give_identical_digests(tmp_path):
    env = {**os.environ, **{name: "1" for name in THREAD_VARIABLES}, "CUDA_VISIBLE_DEVICES": "",
           "PYTHONPATH": os.pathsep.join([str(SRC)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
                                                      if p])}
    digests = []
    for k in range(3):
        output = tmp_path / f"digests-{k}.json"
        subprocess.run(single_thread_command(output), cwd=SRC, env=env, check=True, timeout=600)
        digests.append(output.read_bytes())
    assert digests[0] == digests[1] == digests[2]
    tables = json.loads(digests[0])
    # Every table of golden (per decoder: 8 intensity arrays, candidates, 3 read tables, 2 files) and
    # of the 8 cal conditions (spots, 8 intensity arrays, 4 read tables per decoder) is covered.
    assert len(tables) == 2 * 14 + 8 * (9 + 2 * 4)
    print(f"\nR19: {len(tables)} digests identical in three processes; sha256 of the digest file "
          f"{hashlib.sha256(digests[0]).hexdigest()}")


if __name__ == "__main__":
    result = determinism_digests()
    Path(sys.argv[1]).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
