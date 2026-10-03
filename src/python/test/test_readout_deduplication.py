"""§2.8 optional deduplication and the extended read filter (W-295; checks R13 to R15, R17, R18).

docs/readout-contract.md, "Optional deduplication", "Runtime order and defaults" and
"Checkpoints and reruns"; docs/readout-algorithms.md, "Deduplication" and "Engineering
validation design". R13 checks the known answers on the crosstalk fixture (seeds 100 to
102). R14 (extended tier) checks the false merges on the W-278 held-out scenes (cal and
dense, seeds 103 to 105) against W-278 duplicates.csv (heldout, all_calibrated,
same_sequence, d = 1: 5 of 763); the grouping, representative and conflict rule are
provisional there, and missed duplicates are reported, not gated. R15 checks that the
stage is off by default and unavailable in direct mode, R17 the reruns from checkpoints
with deduplication and R18 the filter on golden, two_seg and crosstalk.
"""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from starfinder.barcode import (BarcodeDecodingResult, CodebookAwareDecoderConfig, DeduplicationConfig,
                                NeighborhoodSumConfig, ReadDeduplicationResult, ReadFilterConfig, ReadScoreConfig,
                                WtaDecoderConfig, decode_barcodes, deduplicate_reads, filter_reads)
from starfinder.barcode.deduplication import DEDUPLICATION_COLUMNS
from starfinder.barcode.filtering import score_columns
from starfinder.dataset import CheckpointConfig, PipelineConfig
from starfinder.dataset.workflow import from_workflow_config
from starfinder.evaluation.barcode import evaluate_deduplication
from starfinder.spot_finding import LocalMaximaConfig, SpotFindingResult

from . import readout_fixtures as fx
from . import readout_scenes as scenes
from .barcode_cases import META, codebook, intensity, tensor
from .test_readout_direct import direct_dataset, direct_fov
from .test_readout_direct import pipeline as direct_pipeline
from .test_readout_golden import (METADATA, PINNED_FILTERING_SCORED, PINNED_PRE_QC_SCORED, ROUNDS, candidates,
                                  fixture_rounds, golden_codebook, golden_dataset, readout_config, table_digest)
from .test_readout_scoring import ROOT, golden_run, workflow

pytestmark = [pytest.mark.barcode]

MERGED, CONFLICTING = "same_sequence_within_distance", "conflicting_calls"
READS = ("decoding_result", "scoring_result", "deduplication_result", "filtering_result")


def merge_labels(table):
    """Each read's merged-group label for evaluate_deduplication: reads with equal labels were merged.

    Reads of a conflicting group stay representatives and are not merged, so they get no label.
    """
    merged = table.duplicate_reason.eq(MERGED).to_numpy(bool)
    return [str(g) if m else None for g, m in zip(table.duplicate_group, merged)]


def crosstalk_pipeline(decoder=CodebookAwareDecoderConfig(), **changes):
    return replace(PipelineConfig(extraction=NeighborhoodSumConfig(), decoding=decoder, scoring=ReadScoreConfig(),
                                  deduplication=DeduplicationConfig(), filtering=ReadFilterConfig()), **changes)


def copy_pairs(truth, select):
    """(source row, copy row) of every copy amplicon whose copy distance passes select."""
    pairs = []
    for source, rows in truth[truth.kind == "copy"].groupby("source", sort=False):
        a, b = rows.index[rows.role.eq("source")][0], rows.index[rows.role.eq("copy")][0]
        if select(rows.distance.iloc[0]):
            pairs.append((int(a), int(b)))
    return pairs


def assert_reads_equal(a, b):
    for name in READS:
        x, y = getattr(a, name), getattr(b, name)
        assert (x is None) == (y is None), name
        if x is not None:
            pd.testing.assert_frame_equal(x.table, y.table, check_exact=True)
            assert x.config == y.config, name
            if name != "filtering_result":
                assert x.readout_mode == y.readout_mode, name
    assert a.deduplication_result is None or a.deduplication_result.counts == b.deduplication_result.counts


# --- R13: known answers on crosstalk ------------------------------------------------------------

@pytest.mark.validation
@pytest.mark.parametrize("seed", fx.SEEDS)
def test_r13_crosstalk_copies_pairs_and_the_conflicting_group(tmp_path, seed):
    truth = fx.crosstalk_truth()
    fov = fx.crosstalk_fov(tmp_path, seed).run(crosstalk_pipeline())
    result = fov.deduplication_result
    table = result.table
    labels, source = merge_labels(table), truth.source.tolist()
    near = copy_pairs(truth, lambda d: d <= 1.0)
    far = copy_pairs(truth, lambda d: d > 1.0)
    assert (len(near), len(far)) == (12, 8)
    assert sorted(truth[truth.kind == "copy"].groupby("source").distance.first().round(6).tolist()) == (
        [0.0] * 8 + [1.0] * 4 + [round(np.sqrt(2), 6)] * 4 + [2.0] * 4)
    # The 12 copies at 0 and 1 voxel are merged; the 8 at √2 and 2 voxels are not, as planted.
    merged = evaluate_deduplication(labels, source, pairs=near)
    assert (merged.values["missed_duplicates"], merged.counts["true_duplicate_pairs"]) == (0, 12)
    kept = evaluate_deduplication(labels, source, pairs=far)
    assert (kept.values["missed_duplicates"], kept.counts["merged_pairs"]) == (8, 0)
    # The six pairs of different genes 1 voxel apart are not merged.
    pair_rows = truth.index[truth.kind.eq("pair")].tolist()
    different = list(zip(pair_rows[0::2], pair_rows[1::2]))
    assert len(different) == 6 and all(np.isclose(np.linalg.norm(
        truth.loc[i, ["z", "y", "x"]].to_numpy(float) - truth.loc[j, ["z", "y", "x"]].to_numpy(float)), 1.0)
        and truth.channel[i] != truth.channel[j] for i, j in different)
    distinct = evaluate_deduplication(labels, source, pairs=different)
    assert (distinct.values["false_merges"], distinct.counts["distinct_pairs"]) == (0, 6)
    assert table.duplicate_group.iloc[pair_rows].isna().all()
    # Every representative is the source amplicon's own-channel candidate.
    for a, b in near:
        assert bool(table.is_representative[a]) and not bool(table.is_representative[b])
        assert table.duplicate_of[b] == table.spot_id[a] == table.duplicate_group[a] == table.duplicate_group[b]
        assert truth.role[a] == "source" and truth.channel[b] == (truth.channel[a] + 1) % 4
    # The conflicting group is kept: both reads stay representatives with conflicting_calls.
    conflict = truth.index[truth.kind.eq("conflict")].tolist()
    rows = table.iloc[conflict]
    assert rows.duplicate_reason.tolist() == [CONFLICTING] * 2 and rows.is_representative.all()
    assert rows.duplicate_of.isna().all() and rows.duplicate_group.tolist() == [table.spot_id[conflict[0]]] * 2
    assert rows.gene_id.tolist() == ["GeneY", "GeneX"] and rows.call_type.tolist() == ["rescued_h1"] * 2
    assert rows.observed_color_sequence.tolist() == [fx.CONFLICT_SEQUENCE] * 2
    assert result.counts == {"total": len(truth), "groups": 12, "merged_reads": 12, "conflicting_groups": 1}
    assert (result.diagnostics["cross_channel_pairs"], result.diagnostics["linked_pairs"]) == (19, 13)
    # Nothing but the deduplication columns changes, and no read is removed.
    pd.testing.assert_frame_equal(table.drop(columns=list(DEDUPLICATION_COLUMNS)), fov.scoring_result.table)
    # With WTA the conflicting reads are both unmatched, so their group merges into the source too.
    wta = fx.crosstalk_fov(tmp_path / "wta", seed).run(crosstalk_pipeline(WtaDecoderConfig())).deduplication_result
    assert wta.counts == {"total": len(truth), "groups": 13, "merged_reads": 13, "conflicting_groups": 0}
    for a, b in near + [tuple(conflict)]:
        assert wta.table.duplicate_of[b] == wta.table.spot_id[a] and bool(wta.table.is_representative[a])


# --- R14: false merges on the W-278 held-out scenes ---------------------------------------------

@pytest.mark.extended
@pytest.mark.slow
@pytest.mark.validation
def test_r14_false_merges_on_calibrated_scenes():
    totals = {"candidates": 0, "distinct": 0, "true_duplicates": 0, "measured_rule_links": 0, "links": 0}
    merges = {name: {"false": 0, "missed": 0} for name in scenes.DECODERS}
    for condition in scenes.CONDITIONS:
        for seed in scenes.HELDOUT_SEEDS:
            book, _ = scenes.scene(condition, seed)
            spots, intensities, tables, _ = scenes.pipeline(condition, seed)
            source, pairs = scenes.duplicate_population(condition, seed)
            distinct = pairs[pairs.pair_class == "distinct"]
            true = pairs[pairs.pair_class == "true_duplicate"]
            observed = tables["wta"].observed_color_sequence.astype(str).to_numpy()
            totals["candidates"] += len(spots.spots)
            totals["distinct"] += len(distinct)
            totals["true_duplicates"] += len(true)
            # W-278's measured rule: same sequence within d = 1, without the M/N exclusion.
            totals["measured_rule_links"] += int(((distinct.distance <= 1 + 1e-9)
                                                  & (observed[distinct.i] == observed[distinct.j])).sum())
            for name, config in scenes.DECODERS.items():
                result = deduplicate_reads(decode_barcodes(intensities, book, config=config), spots, intensities,
                                           config=DeduplicationConfig(1.0))
                ids = result.table.spot_id.astype(str).to_numpy()
                if name == "wta":
                    linked = result.diagnostics["pairs"]
                    linked = set(zip(linked.spot_id_a[linked.linked], linked.spot_id_b[linked.linked]))
                    totals["links"] += sum((ids[i], ids[j]) in linked for i, j in zip(distinct.i, distinct.j))
                labels, groups = merge_labels(result.table), list(source)
                population = list(zip(distinct.i.tolist(), distinct.j.tolist())) + list(zip(true.i.tolist(),
                                                                                           true.j.tolist()))
                evaluation = evaluate_deduplication(labels, groups, pairs=population)
                merges[name]["false"] += evaluation.values["false_merges"]
                merges[name]["missed"] += evaluation.values["missed_duplicates"]
    print(f"\nR14 pooled over {len(scenes.CONDITIONS)} conditions x seeds {scenes.HELDOUT_SEEDS}: {totals}; "
          f"after grouping {merges}")
    # The W-278 population is reproduced: 1927 candidates, 763 distinct pairs within 5 voxels, 5 links.
    assert (totals["candidates"], totals["distinct"], totals["measured_rule_links"]) == (1927, 763, 5)
    # (a) Pairwise links: at most 0.7 % of the distinct pairs (W-278 duplicates.csv: 5 of 763).
    assert totals["links"] <= totals["measured_rule_links"]
    assert totals["links"] / totals["distinct"] <= 0.007
    # (b) After grouping, representative selection and the conflict rule (provisional): at most 0.7 %.
    for name, counts in merges.items():
        assert counts["false"] / totals["distinct"] <= 0.007, name


# --- R15: off by default; unavailable in direct mode --------------------------------------------

@pytest.mark.validation
@pytest.mark.parametrize("decoder", ["wta", "codebook_aware"])
def test_r15_without_a_deduplication_config_every_table_equals_a_run_without_the_stage(tmp_path, decoder):
    assert PipelineConfig().deduplication is None
    dataset, fov, checkpoints = golden_run(tmp_path, decoder=decoder)
    assert fov.deduplication_result is None and "deduplication" not in fov.results
    for name in ("decoding_result", "scoring_result", "filtering_result"):
        assert not set(DEDUPLICATION_COLUMNS) & set(getattr(fov, name).table)
    # The tables equal the pins of the golden run, which has no deduplication stage.
    assert table_digest(fov.scoring_result.table) == PINNED_PRE_QC_SCORED[decoder][0]
    assert table_digest(fov.filtering_result.table) == PINNED_FILTERING_SCORED[(decoder, "default")]
    pd.testing.assert_frame_equal(fov.filtering_result.table,
                                  filter_reads(fov.scoring_result, config=ReadFilterConfig(),
                                               codebook=dataset.codebook).table, check_exact=True)
    header = json.loads((tmp_path / "checkpoints" / "FOV_001" / "pre_qc.json").read_text())
    assert header["deduplication_config"] is None and header["stages_applied"] == ["decoding", "scoring"]
    record = json.loads((tmp_path / "checkpoints" / "FOV_001" / "run.json").read_text())
    assert record["config"]["pipeline"]["deduplication"] is None and "deduplication" not in record["counts"]
    # A run that decodes again without the stage drops an earlier deduplication.
    crosstalk = fx.crosstalk_fov(tmp_path / "crosstalk").run(crosstalk_pipeline())
    assert crosstalk.deduplication_result is not None
    crosstalk.run(crosstalk_pipeline(deduplication=None, extraction=None))
    fresh = fx.crosstalk_fov(tmp_path / "fresh").run(crosstalk_pipeline(deduplication=None))
    assert_reads_equal(crosstalk, fresh)


@pytest.mark.validation
def test_r15_deduplication_raises_in_direct_mode(tmp_path):
    dataset = direct_dataset(tmp_path)
    fov = direct_fov(dataset).run(direct_pipeline(scoring=ReadScoreConfig()))
    with pytest.raises(ValueError, match="not available in readout mode 'direct'"):
        deduplicate_reads(fov.decoding_result, fov.spot_result, fov.intensity_result)
    with pytest.raises(ValueError, match="not available in readout mode 'direct'"):
        deduplicate_reads(fov.scoring_result, fov.spot_result, fov.intensity_result, config=DeduplicationConfig())
    with pytest.raises(ValueError, match="not available in readout mode 'direct'"):
        fov.deduplicate_reads()
    other = direct_fov(direct_dataset(tmp_path / "other"))
    with pytest.raises(ValueError, match="not available in readout mode 'direct'"):
        other.run(direct_pipeline(deduplication=DeduplicationConfig()))
    # The check comes before any processing.
    assert other.spot_result is None and other.intensity_result is None
    with pytest.raises(ValueError, match="readout_mode direct"):
        from_workflow_config({**workflow(spot_finding={"run": True, "rounds": ["round1"]},
                                         reads_extraction={"run": True}, reads_filtration={"run": True},
                                         deduplication={"run": True}),
                              "readout_mode": "direct", "backend": "python"})


# --- R17: reruns with deduplication -------------------------------------------------------------

def crosstalk_checkpointed(root, table_format, seed=100):
    checkpoints = CheckpointConfig(stages=("candidates", "pre_qc"), directory=root / "checkpoints",
                                   table_format=table_format)
    fov = fx.crosstalk_fov(root, seed).run(crosstalk_pipeline(), checkpoints=checkpoints)
    return fov, checkpoints


@pytest.mark.validation
@pytest.mark.parametrize("table_format", ["csv", "parquet"])
@pytest.mark.parametrize("seed", fx.SEEDS)
def test_r17_reruns_with_deduplication_equal_the_full_run(tmp_path, seed, table_format):
    full, checkpoints = crosstalk_checkpointed(tmp_path, table_format, seed)
    dataset = full.dataset
    stages = crosstalk_pipeline()
    header = json.loads((tmp_path / "checkpoints" / "FOV_001" / "pre_qc.json").read_text())
    assert header["deduplication_config"] == {"distance_voxels": 1.0, "compatibility": "same_sequence"}
    assert header["stages_applied"] == ["decoding", "scoring", "deduplication"]
    assert list(header["dtypes"])[-4:] == list(DEDUPLICATION_COLUMNS)
    # pre_qc holds the deduplicated reads and reloads them exactly.
    reloaded = dataset.fov("FOV_001").load_checkpoint("pre_qc", checkpoints=checkpoints)
    for name in READS[:3]:
        pd.testing.assert_frame_equal(getattr(reloaded, name).table, getattr(full, name).table, check_exact=True)
    assert reloaded.deduplication_result.counts == full.deduplication_result.counts
    assert reloaded.deduplication_result.config == DeduplicationConfig()
    # From candidates and pre_qc: deduplicate again and filter, without decoding or images.
    rerun = dataset.fov("FOV_001").load_checkpoint("candidates", checkpoints=checkpoints)
    rerun.load_checkpoint("pre_qc", checkpoints=checkpoints)
    assert not rerun.images
    rerun.run(PipelineConfig(deduplication=stages.deduplication, filtering=stages.filtering))
    assert_reads_equal(rerun, full)
    # The same with rescoring.
    rescored = dataset.fov("FOV_001").load_checkpoint("candidates", checkpoints=checkpoints)
    rescored.load_checkpoint("pre_qc", checkpoints=checkpoints)
    rescored.run(PipelineConfig(scoring=stages.scoring, deduplication=stages.deduplication,
                                filtering=stages.filtering))
    assert_reads_equal(rescored, full)
    # From candidates: decode, score, deduplicate and filter.
    decoded = dataset.fov("FOV_001").load_checkpoint("candidates", checkpoints=checkpoints)
    decoded.run(replace(stages, extraction=None))
    assert_reads_equal(decoded, full)
    # From pre_qc: filtering alone rejects the stored duplicates.
    filtered = dataset.fov("FOV_001").load_checkpoint("pre_qc", checkpoints=checkpoints)
    filtered.run(PipelineConfig(filtering=stages.filtering))
    assert_reads_equal(filtered, full)
    # Deduplication needs the candidates, which pre_qc alone does not hold.
    alone = dataset.fov("FOV_001").load_checkpoint("pre_qc", checkpoints=checkpoints)
    with pytest.raises(ValueError, match="load the candidates checkpoint"):
        alone.run(PipelineConfig(deduplication=stages.deduplication))


@pytest.mark.validation
def test_r17_golden_rerun_with_deduplication(tmp_path):
    # The golden candidates with their round-1 detection channels; no two of them are linked.
    dataset = golden_dataset(tmp_path)
    dataset.codebook = golden_codebook(tmp_path)
    spots = candidates()
    frame = spots.spots.assign(channel=np.array([0, 1, 2, 3, 1, 2, 0, 1, 3, 0], dtype=np.int64))
    full = dataset.fov("FOV_001")
    full.images, full.metadata = fixture_rounds(), {label: METADATA for label in ROUNDS}
    full.spot_result = SpotFindingResult(frame, spots.metadata, spots.spot_namespace, spots.config, spots.diagnostics)
    checkpoints = CheckpointConfig(stages=("candidates", "pre_qc"), directory=tmp_path / "checkpoints")
    pipeline = replace(readout_config("pipeline", decoder="codebook_aware"), deduplication=DeduplicationConfig(2.0))
    full.run(pipeline, checkpoints=checkpoints)
    assert full.deduplication_result.counts == {"total": 10, "groups": 0, "merged_reads": 0,
                                                "conflicting_groups": 0}
    assert table_digest(full.filtering_result.table.drop(columns=list(DEDUPLICATION_COLUMNS))) == (
        PINNED_FILTERING_SCORED[("codebook_aware", "default")])
    rerun = dataset.fov("FOV_001").load_checkpoint("candidates", checkpoints=checkpoints)
    rerun.load_checkpoint("pre_qc", checkpoints=checkpoints)
    rerun.run(PipelineConfig(deduplication=pipeline.deduplication, filtering=pipeline.filtering))
    assert_reads_equal(rerun, full)


# --- R18: the extended filter -------------------------------------------------------------------

@pytest.mark.validation
@pytest.mark.parametrize("decoder", ["wta", "codebook_aware"])
def test_r18_golden_score_bounds_and_the_default_filter(tmp_path, decoder):
    _, fov, _ = golden_run(tmp_path, decoder=decoder)
    scored = fov.scoring_result
    table = scored.table
    assigned = table.call_status.eq("assigned")
    # The default filter keeps every assigned read, with no score bound.
    assert ReadFilterConfig().score_bounds == {}
    default = filter_reads(scored)
    assert default.table.accepted.tolist() == assigned.tolist()
    assert default.table.rejection_reasons[~assigned].eq("call_status").all()
    # A bound on qc_score, and on every declared score column the reads have.
    bound = float(table.qc_score[assigned].median())
    bounded = filter_reads(scored, config=ReadFilterConfig(score_bounds={"qc_score": (None, bound)}))
    expected = assigned & table.qc_score.le(bound)
    assert bounded.table.accepted.tolist() == expected.tolist()
    assert bounded.table.rejection_reasons[assigned & ~expected].eq("score:qc_score").all()
    present = [c for c in score_columns() if c in table]
    assert {"qc_score", "qc_ambiguity_max", "qc_signal_to_background", "qc_rounds"} <= set(present)
    for column in present:
        values = table[column][assigned]
        finite = values[np.isfinite(values)]
        if finite.empty:
            continue
        lower = float(finite.min())
        result = filter_reads(scored, config=ReadFilterConfig(score_bounds={column: (lower, None)}))
        assert result.table.accepted.tolist() == (assigned & table[column].ge(lower)).tolist(), column
    with pytest.raises(ValueError, match="unknown score predicate"):
        ReadFilterConfig(score_bounds={"qc_reason": (0, 1)})
    with pytest.raises(ValueError, match="unavailable for this decoder"):
        filter_reads(fov.decoding_result, config=ReadFilterConfig(score_bounds={"qc_score": (None, 1.0)}))
    # exclude_duplicates has no effect on reads that were not deduplicated.
    for exclude in (True, False):
        pd.testing.assert_frame_equal(
            filter_reads(scored, config=ReadFilterConfig(exclude_duplicates=exclude)).table, default.table)


@pytest.mark.validation
def test_r18_every_declared_score_column_is_a_valid_bound():
    from starfinder.barcode import DECODING_METHODS
    declared = {c for spec in DECODING_METHODS.values() for c in spec.score_columns}
    assert set(score_columns()) == declared | {"qc_score", "qc_ambiguity_max", "qc_signal_to_background",
                                               "qc_rounds"}
    for column in score_columns():
        assert ReadFilterConfig(score_bounds={column: (0.0, 1.0)}).score_bounds == {column: (0.0, 1.0)}


@pytest.mark.validation
@pytest.mark.parametrize("seed", fx.SEEDS)
def test_r18_crosstalk_exclude_duplicates(tmp_path, seed):
    fov = fx.crosstalk_fov(tmp_path, seed).run(crosstalk_pipeline())
    table = fov.deduplication_result.table
    duplicate = ~table.is_representative
    assert int(duplicate.sum()) == 12
    # The default rejects exactly the non-representatives, with reason duplicate.
    filtered = fov.filtering_result.table
    assigned = filtered.call_status.eq("assigned")
    assert filtered.accepted.tolist() == (assigned & ~duplicate).tolist()
    assert set(filtered.rejection_reasons[duplicate]) == {"duplicate"}
    assert filtered.rejection_reasons[~duplicate].eq("").all()
    assert fov.filtering_result.diagnostics["duplicate_count"] == 12
    kept = filter_reads(fov.deduplication_result, config=ReadFilterConfig(exclude_duplicates=False))
    assert kept.table.accepted.tolist() == assigned.tolist()
    # Without deduplication the default filter keeps every assigned read.
    plain = filter_reads(fov.scoring_result)
    assert plain.table.accepted.tolist() == assigned.tolist() and "duplicate_count" not in plain.diagnostics
    with pytest.raises(ValueError, match="exclude_duplicates must be Boolean"):
        ReadFilterConfig(exclude_duplicates=1)


@pytest.mark.validation
@pytest.mark.parametrize("seed", fx.SEEDS)
def test_r18_two_seg_per_segment_end_bases(tmp_path, seed):
    truth = fx.two_seg_truth()
    fov = fx.two_seg_fov(tmp_path, seed).run(PipelineConfig(
        extraction=NeighborhoodSumConfig(), decoding=WtaDecoderConfig(), scoring=ReadScoreConfig(),
        filtering=ReadFilterConfig()))
    table = fov.filtering_result.table
    assert table.observed_color_sequence.tolist() == truth.colors.tolist()
    entries = truth.kind.eq("entry")
    wrong = truth.kind.eq("wrong_ends")
    # Every read equal to an entry passes both segments' end checks; the wrong-end reads pass
    # segment A and fail segment B; endpoint_valid is the conjunction.
    assert table.endpoint_valid_A[entries].all() and table.endpoint_valid_B[entries].all()
    assert table.endpoint_valid_A[wrong].all() and not table.endpoint_valid_B[wrong].any()
    assert table.endpoint_valid.tolist() == (table.endpoint_valid_A & table.endpoint_valid_B).tolist()
    # The check is diagnostic: the default keeps every assigned read.
    assert table.accepted.tolist() == table.call_status.eq("assigned").tolist()
    assert fov.filtering_result.config.score_bounds == {}
    excluded = filter_reads(fov.scoring_result, config=ReadFilterConfig(exclude_invalid_endpoints=True),
                            codebook=fov.codebook).table
    assert excluded.accepted.tolist() == (table.call_status.eq("assigned") & table.endpoint_valid).tolist()
    assert all("endpoint" in r.split(";") for r in excluded.rejection_reasons[~table.endpoint_valid])
    with pytest.raises(ValueError, match="one-segment shortcut"):
        filter_reads(fov.scoring_result, config=ReadFilterConfig(end_bases="CC"), codebook=fov.codebook)


# --- contract -----------------------------------------------------------------------------------

def manual(sequences, points, channels, *, own=None, rounds=None):
    """Hand-built reads: WTA decoding of tensor(sequences) at the given points and detection channels.

    own (optional) overrides each candidate's sum in its own channel in the first round.
    """
    values = tensor(sequences)
    if own is not None:
        for i, value in enumerate(own):
            values[i, channels[i], 0] = value
    labels = tuple(f"r{i}" for i in range(len(sequences[0])))
    data = intensity(values, ids=[str(i) for i in range(len(sequences))], rounds=labels)
    frame = pd.DataFrame({"spot_id": pd.array([str(i) for i in range(len(points))], dtype="string"),
                          "z": [float(p[0]) for p in points], "y": [float(p[1]) for p in points],
                          "x": [float(p[2]) for p in points], "channel": np.asarray(channels, dtype=np.int64)})
    if rounds is not None:
        frame["round"] = pd.array(rounds, dtype="string")
    spots = SpotFindingResult(frame, META, "test/sample/fov", LocalMaximaConfig(),
                              {"channel_labels": ["red", "green", "blue", "yellow"]})
    book = codebook({"1234": "G1", "2143": "G2", "4321": "G3"}, rounds=labels)
    return decode_barcodes(data, book, config=WtaDecoderConfig()), spots, data


@pytest.mark.contract
def test_config_and_result_validation():
    assert DeduplicationConfig() == DeduplicationConfig(1.0, "same_sequence")
    for bad in ({"distance_voxels": -1.0}, {"distance_voxels": float("nan")}, {"distance_voxels": True},
                {"compatibility": "trace_cosine"}):
        with pytest.raises(ValueError):
            DeduplicationConfig(**bad)
    with pytest.raises(TypeError):
        PipelineConfig(deduplication={"distance_voxels": 1.0})
    reads, spots, data = manual(["1234", "1234"], [(0, 0, 0), (0, 0, 1)], [0, 1])
    result = deduplicate_reads(reads, spots, data)
    assert repr(result) == "ReadDeduplicationResult: 2 reads — 1 groups, 1 merged, 0 conflicting"
    with pytest.raises(ValueError, match="direct"):
        replace(result, readout_mode="direct")
    with pytest.raises(TypeError, match="decoded reads"):
        deduplicate_reads(result, spots, data)
    with pytest.raises(TypeError, match="decoded reads"):
        deduplicate_reads(reads.table, spots, data)
    no_channel = SpotFindingResult(spots.spots.drop(columns="channel"), META, "test/sample/fov",
                                   LocalMaximaConfig(), {})
    with pytest.raises(ValueError, match="channel column"):
        deduplicate_reads(reads, no_channel, data)
    with pytest.raises(ValueError, match="identities must match"):
        deduplicate_reads(reads, SpotFindingResult(spots.spots.iloc[::-1].reset_index(drop=True), META,
                                                   "test/sample/fov", LocalMaximaConfig(), spots.diagnostics), data)
    with pytest.raises(ValueError, match="not rounds of the intensities"):
        deduplicate_reads(reads, spots, data, detection_round="round9")


@pytest.mark.contract
def test_links_are_inclusive_cross_channel_and_exclude_m_and_n():
    # Distance exactly d links; just beyond it does not; the same channel never links.
    reads, spots, data = manual(["1234"] * 4, [(0, 0, 0), (0, 0, 1.5), (0, 10, 0), (0, 10, 1.5000001)], [0, 1, 0, 1])
    result = deduplicate_reads(reads, spots, data, config=DeduplicationConfig(1.5))
    assert result.table.duplicate_group.tolist()[:2] == ["0", "0"] and result.table.duplicate_group[2:].isna().all()
    same, spots_same, data_same = manual(["1234"] * 2, [(0, 0, 0), (0, 0, 0)], [2, 2])
    assert deduplicate_reads(same, spots_same, data_same).counts["groups"] == 0
    # A sequence with M (a tie) or N is never compatible, even when both reads have it.
    table = reads.table.copy()
    table["observed_color_sequence"] = pd.array(["1M34", "1M34", "N234", "N234"], dtype="string")
    tied = BarcodeDecodingResult(table, reads.spot_namespace, reads.channel_labels, reads.round_labels,
                                 reads.config, reads.diagnostics)
    result = deduplicate_reads(tied, spots, data, config=DeduplicationConfig(5.0))
    assert result.counts["groups"] == 0 and result.diagnostics["cross_channel_pairs"] == 2
    assert result.diagnostics["linked_pairs"] == 0
    # Different sequences are not linked.
    other, spots_other, data_other = manual(["1234", "2143"], [(0, 0, 0), (0, 0, 0)], [0, 1])
    assert deduplicate_reads(other, spots_other, data_other).counts["groups"] == 0


@pytest.mark.contract
def test_groups_are_connected_components_with_one_representative():
    # Three reads chained by 1-voxel links (the ends 2 voxels apart, in one channel) form one
    # group, represented by the largest sum in its own detection channel (read 2).
    reads, spots, data = manual(["1234"] * 3, [(0, 0, 0), (0, 0, 1), (0, 0, 2)], [0, 1, 0], own=[100.0, 90.0, 120.0])
    result = deduplicate_reads(reads, spots, data)
    assert result.table.duplicate_group.tolist() == ["2", "2", "2"]
    assert result.table.is_representative.tolist() == [False, False, True]
    assert result.table.duplicate_of.isna().tolist() == [False, False, True]
    assert result.table.duplicate_of[:2].tolist() == ["2", "2"]
    assert result.table.duplicate_reason.eq(MERGED).all()
    # Equal own-channel sums: the earliest row represents the group.
    reads, spots, data = manual(["1234"] * 2, [(0, 0, 0), (0, 0, 1)], [1, 2], own=[50.0, 50.0])
    assert deduplicate_reads(reads, spots, data).table.is_representative.tolist() == [True, False]
    # The representative is chosen among the assigned members; an unassigned member is a duplicate of it.
    reads, spots, data = manual(["1234"] * 2, [(0, 0, 0), (0, 0, 1)], [1, 2], own=[90.0, 50.0])
    table = reads.table.copy()
    table.loc[0, ["call_status", "failure_reason", "call_type"]] = ["unmatched", "invalid_measurement", "no_call"]
    table.loc[0, ["gene_id", "entry_id", "decoded_color_sequence"]] = pd.NA
    mixed = BarcodeDecodingResult(table, reads.spot_namespace, reads.channel_labels, reads.round_labels,
                                  reads.config, reads.diagnostics)
    result = deduplicate_reads(mixed, spots, data)
    assert result.table.is_representative.tolist() == [False, True] and result.table.duplicate_of[0] == "1"
    assert deduplicate_reads(reads, spots, data).table.is_representative.tolist() == [True, False]
    # Candidates of different detection rounds are never paired.
    reads, spots, data = manual(["1234"] * 2, [(0, 0, 0), (0, 0, 0)], [0, 1], rounds=["r0", "r1"])
    assert deduplicate_reads(reads, spots, data).counts["groups"] == 0
    reads, spots, data = manual(["1234"] * 2, [(0, 0, 0), (0, 0, 0)], [0, 1], rounds=["r1", "r1"])
    assert deduplicate_reads(reads, spots, data).counts["groups"] == 1
    # Deduplicating reads that already carry the columns replaces them.
    reads, spots, data = manual(["1234"] * 2, [(0, 0, 0), (0, 0, 1)], [0, 1])
    once = deduplicate_reads(reads, spots, data)
    again = deduplicate_reads(replace(reads, table=once.table), spots, data)
    pd.testing.assert_frame_equal(again.table, once.table)


@pytest.mark.contract
def test_run_checks_the_deduplication_stage_before_processing(tmp_path):
    fov = fx.crosstalk_fov(tmp_path)
    with pytest.raises(ValueError, match="deduplication requires decoding"):
        fov.run(PipelineConfig(extraction=NeighborhoodSumConfig(), deduplication=DeduplicationConfig()))
    assert fov.intensity_result is None
    fov.run(crosstalk_pipeline())
    assert list(fov.results) == ["spot_finding", "extraction", "decoding", "scoring", "deduplication", "filtering"]
    assert isinstance(fov.results["deduplication"], ReadDeduplicationResult)
    # Rescoring or decoding again drops the deduplication of the earlier reads.
    fov.score_reads()
    assert fov.deduplication_result is None
    fov.deduplicate_reads()
    assert fov.deduplication_result is not None
    fov.decode_barcodes(config=CodebookAwareDecoderConfig())
    assert fov.deduplication_result is None and fov.scoring_result is None
    # Without scoring, deduplication reads the decoded reads.
    fov.deduplicate_reads(config=DeduplicationConfig(0.0))
    assert fov.deduplication_result.counts["groups"] == 8
    assert not {"qc_score"} & set(fov.deduplication_result.table)


# --- workflow adapter -------------------------------------------------------------------------------

@pytest.mark.contract
def test_workflow_adapter_deduplication_block():
    extraction = {"reads_extraction": {"run": True}, "reads_filtration": {"run": True}}
    assert from_workflow_config(workflow(**extraction)).pipeline.deduplication is None
    assert from_workflow_config(workflow(**extraction, deduplication={"run": False})).pipeline.deduplication is None
    on = from_workflow_config(workflow(**extraction, deduplication={"run": True}))
    assert on.pipeline.deduplication == DeduplicationConfig()
    assert on.pipeline.filtering.exclude_duplicates
    wider = from_workflow_config(workflow(**extraction, deduplication={"run": True, "distance_voxels": 2}))
    assert wider.pipeline.deduplication == DeduplicationConfig(2)
    with pytest.raises(ValueError, match="unknown deduplication keys"):
        from_workflow_config(workflow(**extraction, deduplication={"run": True, "rule": "distance_only"}))
    with pytest.raises(ValueError, match="requires reads_filtration.run"):
        from_workflow_config(workflow(reads_extraction={"run": True}, deduplication={"run": True}))
    with pytest.raises(ValueError, match="deduplication.run must be Boolean"):
        from_workflow_config(workflow(**extraction, deduplication={"run": "yes"}))
    schema = yaml.safe_load((ROOT / "workflow/schemas/config.schema.yaml").read_text())
    definitions = schema["$defs"]
    assert definitions["deduplication_params"]["properties"]["compatibility"]["enum"] == ["same_sequence"]
    for rule in ("rsf_single_fov_config", "lrsf_single_fov_subtile_config", "deep_rsf_subtile_config"):
        assert definitions[rule]["properties"]["parameters"]["properties"]["deduplication"] == {
            "$ref": "#/$defs/deduplication_params"}
    assert '"required": ["deduplication"]' in json.dumps(schema)
