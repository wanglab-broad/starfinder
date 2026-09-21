"""Save a bounded synthetic delivery; reload and render without generator state.

Run ``saved_synthetic.py create /external/fresh-directory``, then ``reload`` and
``report`` on that directory. Report uses saved outputs only; no live kernel.
"""
from dataclasses import asdict, fields, replace
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from starfinder.barcode import (NeighborhoodSumConfig, ReadFilterConfig,
    WtaDecoderConfig, decode_barcodes, extract_intensities, filter_reads)
from starfinder.dataset import Dataset, ExecutionConfig, PipelineConfig, RegistrationStep, RoundState
from starfinder.io import (ImageLayer, ImageLoadResult, ImageProcessingState,
    load_candidate_checkpoint, load_image_checkpoint, save_image_checkpoint)
from starfinder.provenance import RunRecorder, read_run
from starfinder.registration import TranslationConfig
from starfinder.spot_finding import LocalMaximaConfig


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def source_identity():
    repository = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=repository)
    paths = sorted((repository / 'src/python/starfinder').rglob('*.py')) + [Path(__file__)]
    hashes = {str(p.relative_to(repository)): checksum(p) for p in paths}
    return dict(commit=git('rev-parse', 'HEAD').decode().strip(),
        dirty=bool(git('status', '--porcelain')), patch_sha256=hashlib.sha256(git('diff', '--binary', 'HEAD')).hexdigest(),
        source_sha256=hashes, snapshot_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        package_version='0.1.0', source_location=str(repository / 'src/python/starfinder'))


def pipeline():
    return PipelineConfig(registration=(RegistrationStep(TranslationConfig()),),
        detection=LocalMaximaConfig('adaptive', .1), extraction=NeighborhoodSumConfig((0, 0, 0)),
        decoding=WtaDecoderConfig(), filtering=ReadFilterConfig())


def restore_config(cls, payload):
    """Restore constructor fields and verify fixed discriminator/default fields."""
    arguments = {f.name: payload[f.name] for f in fields(cls) if f.init}
    if cls is ReadFilterConfig:
        arguments['call_statuses'] = tuple(arguments['call_statuses'])
    result = cls(**arguments)
    if json.loads(json.dumps(asdict(result))) != payload:
        raise ValueError(f'inconsistent saved {cls.__name__}')
    return result


def make_scene(depth):
    # Generator is imported only by create, never by reload/report.
    from starfinder.synthetic import ScalarDistribution, formed_scene_preset, generate_formed_scene
    book, config = formed_scene_preset('formed-z1-v1' if depth == 1 else 'formed-small-v1')
    z = depth // 2
    config = replace(config, dataset_version=f'saved-formed-z{depth}-v1',
        shape_zyx=(depth, 7, 9), coordinates=((z, 2, 2), (z, 4, 6)),
        amplicon_ids=('formed-A', 'formed-B'), gene_ids={'formed-A': 'gene-A', 'formed-B': 'gene-B'},
        brightness=ScalarDistribution(parameters=(8,)), axial_width=ScalarDistribution(parameters=(.25,)),
        lateral_width=ScalarDistribution(parameters=(.25,)))
    return generate_formed_scene(book, config=config)


def assert_truth(formed, per_round, arrays, depth):
    """Literal two-object oracle, independent of renderer and pipeline outputs."""
    indexed = formed.set_index('amplicon_id')
    assert len(formed) == 2 and set(indexed.index) == {'formed-A', 'formed-B'}
    np.testing.assert_array_equal(indexed.loc[['formed-A', 'formed-B'], ['z', 'y', 'x']],
                                  [[depth // 2, 2, 2], [depth // 2, 4, 6]])
    assert indexed.loc['formed-A', 'gene_id'] == 'gene-A'
    assert indexed.loc['formed-B', 'gene_id'] == 'gene-B'
    assert len(per_round) == 6
    for label in ('round10', 'round2', 'round1'):
        rows = per_round[per_round.round_label == label].set_index('amplicon_id')
        assert set(rows.index) == {'formed-A', 'formed-B'} and len(rows) == 2
        np.testing.assert_array_equal(rows.loc[['formed-A', 'formed-B'], ['z', 'y', 'x']],
                                      [[depth // 2, 2, 2], [depth // 2, 4, 6]])
        assert rows[['emitting', 'center_in_bounds', 'support_intersects']].all().all()
        assert not rows[['dropped', 'weakened', 'lost']].any().any()
        assert rows.first_loss_round.isna().all()
    expected = np.array([[[0, 8, 0], [8, 0, 0], [0, 0, 0], [0, 0, 8]],
                         [[8, 0, 0], [0, 8, 0], [0, 0, 8], [0, 0, 0]]], dtype=np.float64)
    for name in ('intended', 'pre_mix', 'realized'):
        np.testing.assert_array_equal(arrays[name], expected, strict=True)
    return expected


def create(directory):
    directory.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    code = source_identity()
    cases = []
    for depth in (3, 1):
        root = directory / f'z{depth}'
        root.mkdir()
        scene = make_scene(depth)
        expected = assert_truth(scene.formed, scene.round_truth,
            {n: getattr(scene, n) for n in ('intended', 'pre_mix', 'realized')}, depth)
        scene.formed.to_parquet(root / 'formed.parquet', index=False)
        scene.round_truth.to_parquet(root / 'round-truth.parquet', index=False)
        np.savez(root / 'truth-signals.npz', intended=scene.intended, pre_mix=scene.pre_mix, realized=scene.realized)
        write_json(root / 'synthetic.json', scene.provenance)
        source = dict(scene.provenance, uri=str(root / 'synthetic.json'),
            sha256=checksum(root / 'synthetic.json'), unverified_reason=None,
            catalog='docs/datasets.md#saved-formed-development-v1')
        rounds = RoundState(list(scene.round_labels), reference_round=scene.round_labels[0])
        dataset = Dataset(root, root / 'unused', f'saved-formed-z{depth}-v1', 'sample', 'output', rounds, scene.channel_labels)
        dataset.codebook = scene.codebook
        fov = dataset.fov('FOV_001')
        fov.images.update(scene.rounds)
        fov.metadata.update({r: scene.metadata for r in scene.round_labels})
        record = RunRecorder(root / 'run', dataset_id=dataset.dataset_id, sample_id='sample',
            code=code, sources=(source,), seed={'root': 42}, owner='Jiahao',
            retention='thesis/project handoff and associated publication')
        context = dict(rounds=rounds, dataset_id=dataset.dataset_id, sample_id='sample', FOV='FOV_001',
            run_id=record.run_id, config={'pipeline': asdict(pipeline()), 'fixture': source['dataset_version']},
            code=code, sources=(source,))
        def roles(r):
            return ('sequencing', 'registration_reference') if r == rounds.reference_round else ('sequencing',)
        prepared = save_image_checkpoint(root / 'prepared', tuple(ImageLayer(r, roles(r),
            ImageLoadResult(a, scene.metadata, scene.channel_labels, ()), ImageProcessingState(scene.metadata))
            for r, a in scene.rounds.items()), stage='prepared_input', **context)
        fov.run(pipeline(), execution=ExecutionConfig('batch', retain_images=True), provenance=record)
        assert read_run(record.path)['status'] == 'succeeded'
        # Spatial correspondence is explicit; generated IDs are never detector IDs.
        table = fov.spot_result.spots
        assert len(table) == 2 and fov.spot_result.spot_namespace != scene.formed.namespace.iloc[0]
        for i, row in table.iterrows():
            truth_index = {(depth // 2, 2, 2): 0, (depth // 2, 4, 6): 1}[(row.z, row.y, row.x)]
            np.testing.assert_array_equal(fov.intensity_result.values[i], expected[truth_index])
        assert sorted(fov.filtering_result.accepted.gene_id) == ['gene-A', 'gene-B']
        registered_layers = []
        for r in rounds.all_rounds:
            results = tuple(fov.registration_results.get(r, ()))
            for result in results:
                assert result.transform.correction_zyx == (0, 0, 0)
            state = ImageProcessingState(scene.metadata,
                operations=tuple({'operation': 'apply_transform', 'config': result.application_config} for result in results),
                registrations=results, attempts=tuple(fov.registration_attempts.get(r, ())),
                terminal_state='applied' if results else 'reference_unchanged',
                reason=None if results else 'reference selected; no warp')
            registered_layers.append(ImageLayer(r, roles(r), ImageLoadResult(fov.images[r], fov.metadata[r],
                scene.channel_labels, ()), state))
        registered = save_image_checkpoint(root / 'registered', tuple(registered_layers), stage='registered_images',
            **context, provenance={'uri': str(record.path), 'sha256': checksum(record.path)},
            parents=({'run_id': record.run_id, 'artifact_id': read_json(prepared)['artifact_id'], 'sha256': checksum(prepared)},))
        fov.decoding_result.table.to_parquet(root / 'uninterrupted-decoded.parquet', index=False)
        fov.filtering_result.table.to_parquet(root / 'uninterrupted-filtered.parquet', index=False)
        write_json(root / 'uninterrupted-counts.json', fov.filtering_result.counts)
        case = dict(depth=depth, root=root.name, shape=[depth, 7, 9, 4],
            candidates=str(fov.candidate_checkpoint_save.path.relative_to(directory)),
            registered=str(registered.relative_to(directory)), prepared=str(prepared.relative_to(directory)),
            settings=dict(decoding=asdict(pipeline().decoding), filtering=asdict(pipeline().filtering)),
            config_sha256=scene.provenance['extensions']['starfinder.synthetic']['config_sha256'])
        cases.append(case)
    env_names = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS',
        'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'CUDA_VISIBLE_DEVICES', 'TMPDIR',
        'MPLBACKEND', 'PYTHONDONTWRITEBYTECODE', 'UV_LOCKED', 'UV_NO_SYNC', 'UV_OFFLINE', 'UV_PYTHON', 'UV_CACHE_DIR']
    write_json(directory / 'delivery.json', dict(contract='starfinder.artifacts/1',
        specification_commit='db3667bd9eef7ee6bdea6fe09fae8af3eb1158cc', code=code, cases=cases,
        seed=42, host=platform.node(), python=sys.version, environment={k: os.environ.get(k) for k in env_names},
        cpu_affinity=sorted(os.sched_getaffinity(0)), create_seconds=perf_counter()-started,
        command=sys.argv, owner='Jiahao', retention='thesis/project handoff and publication', backup='unverified',
        files={str(p.relative_to(directory)): checksum(p) for p in sorted(directory.rglob('*')) if p.is_file()},
        limitations=['Development processed images; not calibrated D04, RNA truth or scientific accuracy.',
            'No round effects/noise/background/deformation; unknown physical calibration.',
            'No MATLAB, SpatialData, napari, Fiji, cloud or public reproducibility verification.',
            'Controller full pytest/Sphinx/reference gates are separate; this example does not certify them.',
            'RSS/storage targets are measured, not enforced cgroup caps; billing/token totals unknown.']))
    print('Saved both scenes; next run reload in a fresh process.')


def verify_files(directory, manifest):
    for name, digest in manifest['files'].items():
        if checksum(directory / name) != digest:
            raise ValueError(f'delivery checksum mismatch: {name}')


def reload(directory):
    started = perf_counter()
    manifest = read_json(directory / 'delivery.json')
    verify_files(directory, manifest)
    evidence = []
    for case in manifest['cases']:
        root = directory / case['root']
        prepared = load_image_checkpoint(directory / case['prepared'])
        registered = load_image_checkpoint(directory / case['registered'])
        saved = load_candidate_checkpoint(directory / case['candidates'])
        for before, after in zip(prepared.layers, registered.layers, strict=True):
            assert before.loaded.image.dtype == after.loaded.image.dtype
            assert before.loaded.image.tobytes() == after.loaded.image.tobytes()
            assert before.loaded.metadata == after.loaded.metadata == saved.spots.metadata
            assert before.loaded.channel_labels == after.loaded.channel_labels == saved.intensities.channel_labels
        images = registered.sequencing_images(require_registered=True)
        extracted = extract_intensities(images, saved.spots, config=saved.intensities.config)
        assert extracted.spot_ids == saved.intensities.spot_ids
        assert extracted.spot_namespace == saved.intensities.spot_namespace
        assert extracted.round_labels == saved.intensities.round_labels
        assert extracted.values.dtype == saved.intensities.values.dtype
        assert extracted.values.tobytes() == saved.intensities.values.tobytes()
        np.testing.assert_array_equal(extracted.valid, saved.intensities.valid, strict=True)
        for signals in (extracted, saved.intensities):
            decoded = decode_barcodes(signals, saved.codebook, config=restore_config(WtaDecoderConfig, case['settings']['decoding']))
            filtered = filter_reads(decoded, config=restore_config(ReadFilterConfig, case['settings']['filtering']))
            pd.testing.assert_frame_equal(decoded.table, pd.read_parquet(root / 'uninterrupted-decoded.parquet'), check_exact=True)
            pd.testing.assert_frame_equal(filtered.table, pd.read_parquet(root / 'uninterrupted-filtered.parquet'), check_exact=True)
            assert filtered.counts == read_json(root / 'uninterrupted-counts.json')
        with np.load(root / 'truth-signals.npz', allow_pickle=False) as arrays:
            assert_truth(pd.read_parquet(root / 'formed.parquet'), pd.read_parquet(root / 'round-truth.parquet'), arrays, case['depth'])
        trace = saved.source_trace(run_id=saved.artifact['run_id'], candidate_artifact_id=saved.artifact['artifact_id'],
            spot_namespace=saved.intensities.spot_namespace, spot_id=saved.intensities.spot_ids[0])
        np.testing.assert_array_equal(trace['values'], saved.intensities.values[0])
        evidence.append(dict(depth=case['depth'], image_bytes_geometry_labels='exact', extraction='exact',
            decoding_filtering_counts='exact', independent_truth='passed', source_trace='exact'))
    result = dict(pid=os.getpid(), command=sys.argv, delivery_sha256=checksum(directory / 'delivery.json'),
        seconds=perf_counter()-started, cases=evidence)
    # Preserve earlier verification records instead of overwriting them.
    path = directory / f'reload-{os.getpid()}.json'
    if path.exists():
        raise FileExistsError(path)
    write_json(path, result)
    print(json.dumps(result, sort_keys=True))


def overlay(image, formed):
    """Inline SVG: explicit max-over-Z-and-channel display, no image conversion."""
    plane = image.max(axis=(0, 3))
    scale = 32
    h, w = plane.shape
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="XY maximum projection with formed truth centers" viewBox="0 0 {w*scale} {h*scale}" width="{w*scale}" height="{h*scale}">']
    peak = float(plane.max()) or 1
    for y in range(h):
        for x in range(w):
            level = round(255 * float(plane[y, x]) / peak)
            parts.append(f'<rect x="{x*scale}" y="{y*scale}" width="32" height="32" fill="rgb({level},{level},{level})"/>')
    for row in formed.itertuples():
        x, y = (row.x+.5)*scale, (row.y+.5)*scale
        parts.append(f'<circle cx="{x}" cy="{y}" r="12" fill="none" stroke="#ff9566" stroke-width="2"/>')
        anchor, offset = ('end', -14) if x > w*scale/2 else ('start', 14)
        parts.append(f'<text x="{x+offset}" y="{y-10}" text-anchor="{anchor}" fill="#ff9566" font-size="12">{html.escape(row.amplicon_id)} z={row.z:g}</text>')
    return ''.join(parts) + '</svg>'


def report(directory):
    manifest = read_json(directory / 'delivery.json')
    verify_files(directory, manifest)
    reloads = sorted(directory.glob('reload-*.json'))
    if not reloads:
        raise ValueError('fresh-process reload evidence required before reporting')
    for path in reloads:
        assert read_json(path)['delivery_sha256'] == checksum(directory / 'delivery.json')
    destination = directory / 'review.html'
    if destination.exists():
        raise FileExistsError('Preserve report versions; use a fresh delivery directory')
    sections = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>W-160 saved synthetic review</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#172536}table{display:block;overflow-x:auto;border-collapse:collapse;margin:16px 0}td,th{padding:8px;border:1px solid #aaa;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef3f7;padding:16px}svg{max-width:100%;height:auto}h2{margin-top:40px}</style>',
        '<h1>W-160: saved 3D and Z=1 development example</h1>',
        '<p>Decision requested from Jiahao: review this exact source/packet identity for batch 1. W-173 remains open. Technical checks do not grant human approval or authorize batch 2.</p>',
        '<h2>Catalog and assumptions</h2><p>Selected inputs: saved-formed-z3-v1 and saved-formed-z1-v1, two explicit formed amplicons, seed 42, no external inputs or historical TIFFs. Parent model: formed development v1; no calibrated D04 status. Full catalog: docs/datasets.md. Historical D01–D08 remain governed by their existing qualification limits.</p>',
        '<p>Float32 ZYXC images; float64 NCR signals; unknown physical calibration. Three ordered rounds (round10, round2, round1), channels (ch02, ch00, ch03, ch01), color mapping 1→1, 2→0, 3→3, 4→2. A=8, axial/lateral sigma=.25, elongation=1, angle=0; no effects/noise/background. Z=1 samples the same 3D kernel. Truth and detector namespaces are distinct.</p>',
        '<p>Independent oracle: centers (Z//2,2,2) and (Z//2,4,6), gene-A=123 and gene-B=214, two formed rows and six round rows; active amplitudes exactly 8, inactive 0. Registration corrections are exactly zero. Persistence and deterministic downstream comparisons require exact equality, with no relaxed tolerance.</p>']
    for case in manifest['cases']:
        root = directory / case['root']
        formed = pd.read_parquet(root / 'formed.parquet')
        saved = load_candidate_checkpoint(directory / case['candidates'])
        images = load_image_checkpoint(directory / case['registered']).sequencing_images(require_registered=True)
        sections.append(f'<h2>Z={case["depth"]}: saved images and truth</h2><p>Display only: maximum over Z and channels, per-round peak-scaled grayscale; circles mark truth XY centers and labels retain Z. Stored volumes are unchanged.</p>')
        for name, loaded in images.items():
            sections.append(f'<h3>{html.escape(name)}</h3>' + overlay(loaded.image, formed))
        sections.append('<h3>Formed population</h3>' + formed[['amplicon_id', 'gene_id', 'codeword', 'z', 'y', 'x', 'A', 'sz', 'sl']].to_html(index=False, escape=True))
        history = pd.read_parquet(root / 'round-truth.parquet')
        sections.append('<h3>Per-round truth history</h3>' + history[['amplicon_id', 'round_label', 'z', 'y', 'x', 'emitting', 'dropped', 'lost', 'support_truncated']].to_html(index=False, escape=True))
        rows = []
        for i, identity in enumerate(saved.intensities.spot_ids):
            for r, label in enumerate(saved.intensities.round_labels):
                rows.append(dict(spot_id=identity, round=label, valid=bool(saved.intensities.valid[i, r]),
                    **{c: saved.intensities.values[i, k, r] for k, c in enumerate(saved.intensities.channel_labels)}))
        sections.append('<h3>Saved candidate signals (all channels, each round)</h3>' + pd.DataFrame(rows).to_html(index=False))
        sections.append('<h3>Saved decoding and QC</h3>' + pd.read_parquet(root / 'uninterrupted-filtered.parquet').to_html(index=False, escape=True))
        sections.append('<details><summary>Full configuration, order, seeds and generator provenance</summary><pre>' + html.escape((root / 'synthetic.json').read_text()) + '</pre></details>')
    sections.append('<h2>Fresh-process evidence</h2><pre>' + html.escape(json.dumps([read_json(p) for p in reloads], indent=2)) + '</pre>')
    sections.append('<h2>Tests, failures, limits and open decisions</h2><p>This packet records successful create/reload assertions. Focused test logs and measured RSS/storage are in the enclosing implementation manifest. Full pytest, strict Sphinx and reference gates are controller-owned and pending at implementation handoff. No failed or skipped test is inferred to pass. Scientific calibration, later effects/exporters, token billing and backup coverage remain unverified.</p><pre>' + html.escape(json.dumps(manifest['limitations'], indent=2)) + '</pre>')
    context = directory / 'review-context.json'
    if context.exists():
        sections.append('<h3>Saved validation and batch context</h3><p>Context SHA-256: ' + checksum(context) + '</p><pre>' + html.escape(context.read_text()) + '</pre>')
    sections.append('<h2>Revision and artifact identity</h2><p>Delivery manifest SHA-256: ' + checksum(directory / 'delivery.json') + '</p><p>Report hash is recorded externally in report-identity.json, avoiding a self-referential hash. All essential figures and tables are inline; opening this HTML needs no kernel, server, CDN or rerun.</p><pre>' + html.escape(json.dumps(manifest, indent=2)) + '</pre></html>')
    destination.write_text(''.join(sections))
    write_json(directory / 'report-identity.json', dict(report_sha256=checksum(destination),
        delivery_sha256=checksum(directory / 'delivery.json'), reloads={p.name: checksum(p) for p in reloads},
        review_context_sha256=checksum(context) if context.exists() else None))
    print(destination)


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] not in ('create', 'reload', 'report'):
        raise SystemExit('Usage: saved_synthetic.py {create|reload|report} /external/directory')
    globals()[sys.argv[1]](Path(sys.argv[2]).resolve())
