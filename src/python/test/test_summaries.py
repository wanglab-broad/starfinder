"""Plain-text Dataset/FOV/result summaries and the read-only FOV.results mapping."""
from dataclasses import replace
import re

import numpy as np
import pytest

from starfinder.barcode import NeighborhoodSumConfig, ReadFilterConfig, WtaDecoderConfig, filter_reads
from starfinder.dataset import CheckpointConfig, Dataset, PipelineConfig, RegistrationRecipe, RegistrationStep, RoundState
from starfinder.evaluation import EvaluationResult
from starfinder.io import ImageLoadConfig
from starfinder.preprocessing import MinMaxNormalizationConfig, PreprocessingRecipe, PreprocessingStep
from starfinder.registration import DemonsConfig, TranslationConfig
from starfinder.spot_finding import LocalMaximaConfig

from .test_checkpoints import DECODE, DETECT, dataset, full, resident

pytestmark = pytest.mark.integration

STAGES = ['registration', 'spot_finding', 'extraction', 'decoding', 'filtering']


def assert_no_content(text, fov):
    """No array values or table rows: genes, sequences, namespaces or coordinates.

    Spot IDs are row numbers here, indistinguishable from counts, so rows are
    detected through their other columns. The only decimal is the accepted share.
    """
    assert 'array(' not in text and '[[' not in text and 'dtype=' not in text and ' rows x ' not in text
    assert fov.spot_result.spot_namespace not in text
    tables = [fov.decoding_result.table, fov.codebook.table]
    for token in {str(v) for t in tables for c in ('gene_id', 'color_sequence', 'observed_color_sequence')
                  if c in t for v in t[c].dropna()}:
        assert not re.search(rf'(?<![\w.]){re.escape(token)}(?![\w.])', text), token
    assert set(re.findall(r'\d+\.\d+', text)) <= {f"{fov.filtering_result.fractions['accepted'] * 100:.1f}"}


def test_fov_and_dataset_summaries_on_small_synthetic_run(small_dataset, tmp_path):
    for round_dir in (small_dataset / 'FOV_001').iterdir():
        if round_dir.is_dir():
            target = tmp_path / round_dir.name / 'FOV_001'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(round_dir)
    rounds = ['round1', 'round2', 'round3', 'round4']
    ds = Dataset(tmp_path, tmp_path / 'output', 'test', 'small', 'out',
                 RoundState(rounds, reference_round='round1'), ('ch00', 'ch01', 'ch02', 'ch03'))
    ds.load_codebook(small_dataset / 'codebook.csv')
    fov = ds.fov('FOV_001').run(PipelineConfig(
        load=ImageLoadConfig(channel_labels=ds.channel_order),
        preprocessing=PreprocessingRecipe((PreprocessingStep(MinMaxNormalizationConfig('uint8', (0, 255), snr_threshold=5.0)),)),
        registration=RegistrationRecipe((RegistrationStep(TranslationConfig()),)), spot_finding=LocalMaximaConfig(),
        extraction=NeighborhoodSumConfig(), decoding=WtaDecoderConfig(diagnostics=True),
        filtering=ReadFilterConfig()))

    text = repr(fov)
    lines = text.splitlines()
    assert len(lines) <= 15
    spots, reads = fov.spot_result.spots, fov.decoding_result.table
    statuses = reads.call_status.value_counts()
    counts = fov.filtering_result.counts
    reasons = fov.filtering_result.table.rejection_reasons.str.split(';').explode()
    reasons = reasons[reasons.ne('') & reasons.notna()].value_counts()
    assert counts['total'] == len(spots) > 0
    image = fov.images['round1']
    assert lines[:4] == [
        "FOV 'FOV_001' of Dataset 'test' (sample 'small')",
        f"    images:   round1*, round2, round3, round4 — {image.shape} {image.dtype} ZYXC   (* reference)",
        "    channels: ch00, ch01, ch02, ch03",
        "    results:  " + ", ".join(STAGES),
    ]
    assert image.shape == (16, 256, 256, 4)
    status_text = ", ".join(f"{s} {statuses[s]}" for s in ('assigned', 'ambiguous', 'no_signal', 'unmatched')
                            if s in statuses.index)
    filtering = (f"{counts['accepted']} accepted / {counts['total']} "
                 f"({counts['accepted'] / counts['total']:.1%}), rejected {counts['rejected']}")
    if len(reasons):
        filtering += " — " + ", ".join(f"{r} {n}" for r, n in reasons.items())
    assert lines[4:] == [
        "      registration  3 moving rounds, translation",
        f"      spot_finding  {len(spots)} spots × [spot_id, z, y, x, ...]",
        f"      extraction    {len(spots)} spots × 4 channels × 4 rounds",
        f"      decoding      {len(reads)} reads — {status_text}",
        f"      filtering     {filtering}",
    ]
    assert 'precision' not in text and 'accuracy' not in text
    assert_no_content(text, fov)

    summary = repr(ds)
    assert summary.splitlines() == [
        "Dataset 'test' (sample 'small', output 'out')",
        "    sequencing rounds: round1*, round2, round3, round4   (* reference)",
        "    other rounds:      none",
        "    channels:          ch00, ch01, ch02, ch03",
        f"    codebook:          {ds.codebook.n_genes} genes × 4 rounds",
        "    encoding:          two_base, one segment of 4 colors",
        f"    input root:        {tmp_path}",
        f"    output root:       {tmp_path / 'output'}",
    ]
    # Temporary directory names may contain digits that look like color sequences.
    assert_no_content(summary.replace(str(tmp_path), '<root>'), fov)
    assert 'FOV' not in summary.replace(str(tmp_path), '<root>')


def test_fov_summary_is_independent_of_image_size_and_marks_unloaded_images(tmp_path):
    ds = Dataset(tmp_path, tmp_path / 'out', 'data', 'sample', 'run',
                 RoundState(['round1', 'round2'], ['dapi'], reference_round='round1'), ('a', 'b', 'c', 'd'))
    fov = ds.fov('FOV')
    assert repr(fov).splitlines() == [
        "FOV 'FOV' of Dataset 'data' (sample 'sample')",
        "    images:   not loaded (round1*, round2, dapi)   (* reference)",
        "    channels: a, b, c, d",
        "    results:  none",
    ]
    # Read-only broadcast views stand in for very large volumes without allocating them.
    fov.images = {'round1': np.broadcast_to(np.uint16(4321), (64, 4096, 4096, 4)),
                  'dapi': np.broadcast_to(np.float32(0.5), (64, 4096, 4096))}
    fov.subtile_id = 2
    assert repr(fov).splitlines()[:2] == [
        "FOV 'FOV' subtile 2 of Dataset 'data' (sample 'sample')",
        "    images:   round1* (64, 4096, 4096, 4) uint16 ZYXC, dapi (64, 4096, 4096) float32 ZYX; "
        "not loaded: round2   (* reference)",
    ]
    assert '4321' not in repr(fov) and '0.5' not in repr(fov)
    assert repr(ds).splitlines()[1:5] == [
        "    sequencing rounds: round1*, round2   (* reference)",
        "    other rounds:      dapi",
        "    channels:          a, b, c, d",
        "    codebook:          not loaded",
    ]


def test_result_classes_have_one_line_summaries(tmp_path):
    ds = dataset(tmp_path)
    fov = resident(ds).run(full())
    summaries = [
        (ds.codebook, "Codebook: 2 genes × 2 rounds, channels a, b, c, d; encoding two_base, one segment of 2 colors"),
        (fov.spot_result, "SpotFindingResult: 4 spots × [spot_id, z, y, x, ...]"),
        (fov.intensity_result, "IntensityExtractionResult: 4 spots × 4 channels × 2 rounds"),
        (fov.decoding_result, "BarcodeDecodingResult: 4 reads — assigned 1, ambiguous 2, unmatched 1"),
        (fov.filtering_result, "ReadFilteringResult: 1 accepted / 4 (25.0%), rejected 3 — call_status 3"),
        (fov.registration_results['round2'][0],
         "RegistrationResult: translation (scipy_fft), displacement_zyx (0, 0, 0)"),
    ]
    for result, expected in summaries:
        assert repr(result) == expected

    # Several reasons are counted per read; the accepted share never claims truth metrics.
    config = ReadFilterConfig(call_statuses=('assigned', 'unmatched'),
                              score_bounds={'wta_l2_nll': (None, 1.0)})
    text = repr(filter_reads(fov.decoding_result, config=config))
    assert re.fullmatch(r'ReadFilteringResult: \d+ accepted / 4 \(\d+\.\d%\), rejected \d+ — '
                        r'(call_status|score:wta_l2_nll) \d+, (call_status|score:wta_l2_nll) \d+', text)
    assert 'precision' not in text.lower() and 'accuracy' not in text.lower()
    empty = replace(fov.decoding_result, table=fov.decoding_result.table.iloc[:0])
    assert repr(empty) == "BarcodeDecodingResult: 0 reads"
    assert repr(filter_reads(empty)) == "ReadFilteringResult: 0 accepted / 0 (undefined), rejected 0"

    dense = resident(ds).run(PipelineConfig(registration=RegistrationRecipe((RegistrationStep(DemonsConfig(iterations=(1,))),))))
    text = repr(dense.registration_results['round2'][0])
    assert re.fullmatch(r'RegistrationResult: demons \(\w+\), dense field \(4, 12, 14, 3\) float(32|64)'
                        r'(, (not )?converged)?', text)

    metrics = EvaluationResult({'recall': 0.95, 'precision': None, 'mean_distance': 1.234},
                               {'recall': 'fraction', 'precision': 'fraction', 'mean_distance': 'voxel'},
                               {'total': 3}, 'undefined', {'precision': 'zero denominator'}, {})
    assert repr(metrics) == "EvaluationResult: undefined — recall 0.95, precision undefined, mean_distance 1.23 voxel"
    many = EvaluationResult({f'm{i}': float(i) for i in range(8)} | {'passed': True}, {}, {}, 'ok', {}, {})
    assert repr(many) == "EvaluationResult: ok — m0 0, m1 1, m2 2, m3 3, m4 4, m5 5, +3 more"
    assert repr(EvaluationResult({}, {}, {}, 'missing', {}, {})) == "EvaluationResult: missing"
    for result in [*(r for r, _ in summaries), metrics, many, dense.registration_results['round2'][0]]:
        assert '\n' not in repr(result)


def test_results_are_read_only_ordered_and_the_stored_objects(tmp_path):
    ds = dataset(tmp_path)
    fov = resident(ds)
    assert dict(fov.results) == {}
    fov.run(PipelineConfig(**DETECT))
    assert list(fov.results) == ['spot_finding', 'extraction']
    fov.run(PipelineConfig(**DECODE))
    assert list(fov.results) == STAGES[1:]
    fov = resident(ds).run(full())
    results = fov.results
    assert list(results) == STAGES
    assert results['spot_finding'] is fov.spot_result
    assert results['extraction'] is fov.intensity_result
    assert results['decoding'] is fov.decoding_result
    assert results['filtering'] is fov.filtering_result
    assert dict(results['registration']) == fov.registration_results
    assert results['registration']['round2'] is fov.registration_results['round2']
    with pytest.raises(TypeError):
        results['filtering'] = None
    with pytest.raises(TypeError):
        del results['decoding']
    with pytest.raises(TypeError):
        results['registration']['round2'] = []
    with pytest.raises(AttributeError):
        fov.results = {}
    assert list(fov.results) == STAGES


def test_results_after_checkpoint_reload(tmp_path):
    ds = dataset(tmp_path)
    saved = resident(ds).run(full(), checkpoints=CheckpointConfig())

    fov = ds.fov('FOV').load_checkpoint('registered')
    assert list(fov.results) == ['registration']
    assert fov.results['registration']['round2'] is fov.registration_results['round2']
    assert dict(fov.results['registration']) == saved.registration_results

    fov = ds.fov('FOV').load_checkpoint('candidates')
    assert list(fov.results) == ['spot_finding', 'extraction']
    assert fov.results['spot_finding'] is fov.spot_result
    assert fov.results['extraction'] is fov.intensity_result
    assert repr(fov).splitlines()[1] == "    images:   not loaded (round1*, round2)   (* reference)"
    fov.run(PipelineConfig(**DECODE))
    assert list(fov.results) == STAGES[1:]
    assert fov.results['filtering'] is fov.filtering_result
    lines = repr(fov).splitlines()
    assert lines[3] == "    results:  " + ", ".join(STAGES[1:])
    assert lines[4:] == repr(saved).splitlines()[5:]

    fov = ds.fov('FOV').load_checkpoint('pre_qc')
    assert list(fov.results) == ['decoding']
    assert fov.results['decoding'] is fov.decoding_result
    assert repr(fov.decoding_result) == repr(saved.decoding_result)
