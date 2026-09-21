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


def process_identity():
    # PID alone is insufficient across sandbox PID namespaces or PID reuse.
    return dict(pid=os.getpid(), pid_namespace=os.readlink('/proc/self/ns/pid'),
                start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])


def source_identity():
    repository = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=repository)
    paths = sorted((repository / 'src/python/starfinder').rglob('*.py')) + sorted(Path(__file__).parent.glob('saved_synthetic*.py')) + [Path(__file__).with_name('inspect_saved_synthetic_fiji.py')]
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
    config = replace(config, dataset_version=f'saved-formed-z{depth}-v3',
        shape_zyx=(depth, 32, 32), coordinates=((z, 10, 10), (z, 22, 22)),
        amplicon_ids=('gt-A', 'gt-B'), gene_ids={'gt-A': 'gene-A', 'gt-B': 'gene-B'},
        brightness=ScalarDistribution(parameters=(8,)), axial_width=ScalarDistribution(parameters=(1,)),
        lateral_width=ScalarDistribution(parameters=(1.25,)))
    return generate_formed_scene(book, config=config)


def assert_truth(formed, per_round, arrays, depth):
    """Literal two-object oracle, independent of renderer and pipeline outputs."""
    indexed = formed.set_index('amplicon_id')
    assert len(formed) == 2 and set(indexed.index) == {'gt-A', 'gt-B'}
    np.testing.assert_array_equal(indexed.loc[['gt-A', 'gt-B'], ['z', 'y', 'x']],
                                  [[depth // 2, 10, 10], [depth // 2, 22, 22]])
    assert indexed.loc['gt-A', 'gene_id'] == 'gene-A'
    assert indexed.loc['gt-B', 'gene_id'] == 'gene-B'
    assert indexed.loc['gt-A', 'codeword'] == '123'
    assert indexed.loc['gt-B', 'codeword'] == '214'
    if 'barcode' in indexed:
        assert indexed.loc['gt-A', 'barcode'] == 'CCAG'
        assert indexed.loc['gt-B', 'barcode'] == 'CAAT'
    assert len(per_round) == 6
    for label in ('round10', 'round2', 'round1'):
        rows = per_round[per_round.round_label == label].set_index('amplicon_id')
        assert set(rows.index) == {'gt-A', 'gt-B'} and len(rows) == 2
        np.testing.assert_array_equal(rows.loc[['gt-A', 'gt-B'], ['z', 'y', 'x']],
                                      [[depth // 2, 10, 10], [depth // 2, 22, 22]])
        assert rows[['emitting', 'center_in_bounds', 'support_intersects']].all().all()
        assert not rows[['dropped', 'weakened', 'lost']].any().any()
        assert rows.first_loss_round.isna().all()
        assert (rows.support_truncated == (depth == 1)).all()
    expected = np.array([[[0, 8, 0], [8, 0, 0], [0, 0, 0], [0, 0, 8]],
                         [[8, 0, 0], [0, 8, 0], [0, 0, 8], [0, 0, 0]]], dtype=np.float64)
    for name in ('intended', 'pre_mix', 'realized'):
        np.testing.assert_array_equal(arrays[name], expected, strict=True)
    return expected


def assert_sampled_images(images, depth):
    """Independent full-grid Gaussian oracle, including the closed 4-sigma support.

    The singleton-Z fixture samples z=0; it does not integrate/project the kernel.
    """
    z, y, x = np.indices((depth, 32, 32), dtype=np.float64)
    for r, name in enumerate(('round10', 'round2', 'round1')):
        expected = np.zeros((depth, 32, 32, 4), dtype=np.float64)
        for center, channels in ((10, (1, 0, 3)), (22, (0, 1, 2))):
            radius = (z-depth//2)**2 + ((y-center)/1.25)**2 + ((x-center)/1.25)**2
            expected[..., channels[r]] = np.where(radius <= 16, 8*np.exp(-radius/2), 0)
        assert images[name].dtype == np.float32
        np.testing.assert_allclose(images[name], expected, rtol=0, atol=1e-6)
        # Closed support includes x offset 5; offset 6 is exactly zero.
        np.testing.assert_allclose(images[name][depth//2, 10, [10,11,15,16], (1,0,3)[r]],
                                   [8, 5.809192296589527, .002683701023220095, 0], rtol=0, atol=1e-6)


def create(directory):
    directory.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    code = source_identity()
    cases = []
    for depth in (9, 1):
        root = directory / f'z{depth}'
        root.mkdir()
        scene = make_scene(depth)
        expected = assert_truth(scene.formed, scene.round_truth,
            {n: getattr(scene, n) for n in ('intended', 'pre_mix', 'realized')}, depth)
        assert_sampled_images(scene.rounds, depth)
        # Independent literal nucleotide expectations under the saved start-base C
        # convention: C-C-A-G (123), C-A-A-T (214). The model truth is color-space.
        formed = scene.formed.copy()
        formed['barcode'] = formed.amplicon_id.map({'gt-A': 'CCAG', 'gt-B': 'CAAT'}).astype('string')
        formed.to_parquet(root / 'formed.parquet', index=False)
        scene.round_truth.to_parquet(root / 'round-truth.parquet', index=False)
        np.savez(root / 'truth-signals.npz', intended=scene.intended, pre_mix=scene.pre_mix, realized=scene.realized)
        write_json(root / 'synthetic.json', scene.provenance)
        source = dict(scene.provenance, uri=str(root / 'synthetic.json'),
            sha256=checksum(root / 'synthetic.json'), unverified_reason=None,
            catalog='docs/datasets.md#saved-formed-development-v3')
        rounds = RoundState(list(scene.round_labels), reference_round=scene.round_labels[0])
        dataset = Dataset(root, root / 'unused', f'saved-formed-z{depth}-v3', 'sample', 'output', rounds, scene.channel_labels)
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
            truth_index = {(depth // 2, 10, 10): 0, (depth // 2, 22, 22): 1}[(row.z, row.y, row.x)]
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
        case = dict(depth=depth, root=root.name, shape=[depth, 32, 32, 4],
            candidates=str(fov.candidate_checkpoint_save.path.relative_to(directory)),
            registered=str(registered.relative_to(directory)), prepared=str(prepared.relative_to(directory)),
            settings=dict(decoding=asdict(pipeline().decoding), filtering=asdict(pipeline().filtering)),
            config_sha256=scene.provenance['extensions']['starfinder.synthetic']['config_sha256'])
        from saved_synthetic_inspection import export_images
        export_images(root, load_image_checkpoint(registered).sequencing_images(require_registered=True))
        cases.append(case)
    from saved_synthetic_inspection import export_recipe
    export_recipe(directory)
    env_names = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS',
        'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'CUDA_VISIBLE_DEVICES', 'TMPDIR',
        'MPLBACKEND', 'PYTHONDONTWRITEBYTECODE', 'UV_LOCKED', 'UV_NO_SYNC', 'UV_OFFLINE', 'UV_PYTHON', 'UV_CACHE_DIR', 'UV_PROJECT_ENVIRONMENT', 'PYTHONPATH', 'MPLCONFIGDIR']
    write_json(directory / 'delivery.json', dict(contract='starfinder.artifacts/1',
        specification_commit='db3667bd9eef7ee6bdea6fe09fae8af3eb1158cc', code=code, cases=cases, create_pid=os.getpid(), create_process=process_identity(),
        seed=42, host=platform.node(), python=sys.version, environment={k: os.environ.get(k) for k in env_names},
        cpu_affinity=sorted(os.sched_getaffinity(0)), create_seconds=perf_counter()-started,
        command=sys.argv, owner='Jiahao', retention='thesis/project handoff and publication', backup='unverified',
        files={str(p.relative_to(directory)): checksum(p) for p in sorted(directory.rglob('*')) if p.is_file()},
        limitations=['Development processed images; not calibrated D04, RNA truth or scientific accuracy.',
            'No round effects/noise/background/deformation; unknown physical calibration.',
            'No MATLAB, SpatialData, napari, cloud or public reproducibility verification; Fiji evidence is separate.',
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
        assert_sampled_images({r: v.image for r, v in images.items()}, case['depth'])
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
        from saved_synthetic_inspection import verify_exports
        verify_exports(root, images)
        trace = saved.source_trace(run_id=saved.artifact['run_id'], candidate_artifact_id=saved.artifact['artifact_id'],
            spot_namespace=saved.intensities.spot_namespace, spot_id=saved.intensities.spot_ids[0])
        np.testing.assert_array_equal(trace['values'], saved.intensities.values[0])
        evidence.append(dict(depth=case['depth'], image_bytes_geometry_labels='exact', extraction='exact',
            decoding_filtering_counts='exact', independent_truth='passed', source_trace='exact', gaussian='rtol=0, atol=1e-6', inspection_exports='exact'))
    assert process_identity() != manifest['create_process'], 'reload must use a fresh process'
    result = dict(pid=os.getpid(), process=process_identity(), command=sys.argv, delivery_sha256=checksum(directory / 'delivery.json'),
        seconds=perf_counter()-started, cases=evidence)
    # Preserve earlier verification records instead of overwriting them.
    path = directory / f'reload-{os.getpid()}.json'
    if path.exists():
        raise FileExistsError(path)
    write_json(path, result)
    print(json.dumps(result, sort_keys=True))


def report(directory):
    from saved_synthetic_report import render_report
    render_report(directory)


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] not in ('create', 'reload', 'report'):
        raise SystemExit('Usage: saved_synthetic.py {create|reload|report} /external/directory')
    globals()[sys.argv[1]](Path(sys.argv[2]).resolve())
