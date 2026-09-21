"""Saved 3D/Z=1 image example with literal expectations and measured I/O cost."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from starfinder.barcode import Codebook, NeighborhoodSumConfig, ReadFilterConfig, WtaDecoderConfig
from starfinder.dataset import Dataset, ExecutionConfig, PipelineConfig, RegistrationStep, RoundState
from starfinder.image import ImageMetadata
from starfinder.io import ImageLayer, ImageLoadResult, ImageProcessingState, load_image_checkpoint, save_image_checkpoint
from starfinder.provenance import RunRecorder, read_run
from starfinder.registration import TranslationConfig
from starfinder.spot_finding import LocalMaximaConfig


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()
    # The example records the actual dirty source identity without changing Git.
    patch = subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=repository)
    source_hashes = {str(p.relative_to(repository)): checksum(p) for p in
                     sorted((repository / 'src/python/starfinder').rglob('*.py'))}
    snapshot = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    code = dict(commit=revision, dirty=bool(patch) or bool(subprocess.check_output(
        ['git', 'ls-files', '--others', '--exclude-standard'], cwd=repository)),
        patch_sha256=hashlib.sha256(patch).hexdigest(), snapshot_sha256=snapshot,
        package_version='0.1.0', source_location=str(repository / 'src/python/starfinder'))
    costs = []
    for depth in (1, 3):
        root = directory / f'z{depth}'
        channels = ('ch02', 'ch00', 'ch03', 'ch01')
        rounds = RoundState(['round10', 'round2'], reference_round='round10')
        dataset = Dataset(root, root / 'unused', 'image-contract-v1', 'sample', 'output', rounds, channels)
        dataset.codebook = Codebook(pd.DataFrame({'gene_id': ['gene-A'], 'color_sequence': ['12']}),
            tuple(rounds.sequencing_rounds), channels, {'1': 1, '2': 0, '3': 3, '4': 2})
        fov = dataset.fov(f'FOV-Z{depth}')
        shape = (depth, 4, 5, 4)
        first, second = np.zeros(shape, dtype=np.uint16), np.zeros(shape, dtype=np.uint16)
        first[depth // 2, 2, 1, 1] = 7
        second[depth // 2, 1, 2, 0] = 9
        inputs = {'round10': first, 'round2': second}
        geometry = {r: ImageMetadata(r) for r in rounds.all_rounds}
        source_records = tuple(dict(source_id=r, catalog='image-contract-v1', uri=f'memory:{r}',
            sha256=hashlib.sha256(a.tobytes()).hexdigest(), unverified_reason=None,
            selection=dict(round=r, axes='ZYXC', shape=list(a.shape), dtype=a.dtype.str,
                           checksum_scope='C-order bytes', metadata=asdict(geometry[r]))) for r, a in inputs.items())
        arguments = dict(rounds=rounds, dataset_id=dataset.dataset_id, sample_id=dataset.sample_id,
            FOV=fov.fov_id, config={'fixture': 'image-contract-v1', 'seed': None}, code=code, sources=source_records)
        record = RunRecorder(root / 'run', dataset_id=dataset.dataset_id, sample_id=dataset.sample_id,
            code=code, sources=source_records, owner='Jiahao', retention='thesis/project handoff and publication')
        def roles(name):
            return ('sequencing', 'registration_reference') if name == rounds.reference_round else ('sequencing',)
        prepared = tuple(ImageLayer(r, roles(r), ImageLoadResult(a, geometry[r], channels, ()),
                                   ImageProcessingState(geometry[r])) for r, a in inputs.items())
        started = perf_counter()
        prepared_path = save_image_checkpoint(root / 'prepared', prepared, stage='prepared_input',
                                             run_id=record.run_id, **arguments)
        prepared_seconds = perf_counter() - started
        loaded = load_image_checkpoint(prepared_path)
        # Prepared rounds have different frames; alignment is deliberately explicit.
        fov.load_image_checkpoint(prepared_path)
        pipeline = PipelineConfig(registration=(RegistrationStep(TranslationConfig()),),
            detection=LocalMaximaConfig('adaptive', .1), extraction=NeighborhoodSumConfig((0, 0, 0)),
            decoding=WtaDecoderConfig(), filtering=ReadFilterConfig())
        fov.run(pipeline, execution=ExecutionConfig('batch', retain_images=True), provenance=record)
        run = read_run(record.path)
        assert run['status'] == 'succeeded' and not run['failures']
        assert fov.registration_results['round2'][0].transform.correction_zyx == (0, 1, -1)
        np.testing.assert_array_equal(fov.intensity_result.values, [[[0, 9], [7, 0], [0, 0], [0, 0]]])
        assert fov.filtering_result.accepted.gene_id.tolist() == ['gene-A']
        registered = []
        for name in rounds.all_rounds:
            results = tuple(fov.registration_results.get(name, ()))
            # This example has exactly one registration and no preprocessing.
            state = ImageProcessingState(geometry[name],
                operations=tuple({'operation': 'apply_transform', 'config': r.application_config} for r in results),
                registrations=results, attempts=tuple(fov.registration_attempts.get(name, ())),
                terminal_state='applied' if results else 'reference_unchanged',
                reason=None if results else 'reference selected; no warp')
            registered.append(ImageLayer(name, roles(name),
                ImageLoadResult(fov.images[name], fov.metadata[name], channels, ()), state))
        started = perf_counter()
        registered_path = save_image_checkpoint(root / 'registered', tuple(registered),
            stage='registered_images', run_id=record.run_id, **arguments,
            provenance={'uri': str(record.path), 'sha256': checksum(record.path)},
            parents=({'run_id': record.run_id, 'artifact_id': loaded.artifact['artifact_id'],
                      'sha256': checksum(prepared_path)},))
        write_seconds = perf_counter() - started
        started = perf_counter()
        reloaded = dataset.fov(fov.fov_id).load_image_checkpoint(registered_path, require_registered=True)
        read_seconds = perf_counter() - started
        for name in rounds.all_rounds:
            assert reloaded.images[name].tobytes() == fov.images[name].tobytes()
        reloaded.find_spots(config=pipeline.detection).extract_intensities(config=pipeline.extraction)
        reloaded.decode_barcodes(config=pipeline.decoding).filter_reads(config=pipeline.filtering)
        np.testing.assert_array_equal(reloaded.intensity_result.values, fov.intensity_result.values, strict=True)
        pd.testing.assert_frame_equal(reloaded.decoding_result.table, fov.decoding_result.table, check_exact=True)
        pd.testing.assert_frame_equal(reloaded.filtering_result.table, fov.filtering_result.table, check_exact=True)
        assert reloaded.filtering_result.counts == fov.filtering_result.counts
        costs.append(dict(depth=depth, shape=list(shape), raw_bytes=sum(a.nbytes for a in inputs.values()),
            prepared_hdf5_bytes=(prepared_path.parent / 'images.h5').stat().st_size,
            registered_hdf5_bytes=(registered_path.parent / 'images.h5').stat().st_size,
            prepared_write_seconds=prepared_seconds, registered_write_seconds=write_seconds,
            registered_read_seconds=read_seconds,
            image_sha256={r: hashlib.sha256(a.tobytes()).hexdigest() for r, a in fov.images.items()}))
    summary = dict(fixture='image-contract-v1', contract='starfinder.artifacts/1',
        specification_commit='db3667bd9eef7ee6bdea6fe09fae8af3eb1158cc', code=code,
        sources=source_hashes, seed=None, costs=costs,
        limitations=['Tiny-fixture costs only; no E10 storage superiority claim.',
                     'Development arithmetic, not scientific qualification.'])
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('Image checkpoint examples passed (3D and Z=1).')


if __name__ == '__main__':
    main(sys.argv[1])
