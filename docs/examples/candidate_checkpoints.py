"""Saved C1/C2/C3/X1 literal expectations, 3D and Z=1; no historical data."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from artifact_contracts import signal_example
from starfinder.barcode import decode_barcodes, filter_reads, WtaDecoderConfig
from starfinder.io import save_candidate_checkpoint, load_candidate_checkpoint, export_spots


def main(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    code = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            'dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True)),
            'patch_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=root)).hexdigest()}
    code['source_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root/'src/python/starfinder/io/candidates.py', Path(__file__),
                  Path(__file__).with_name('artifact_contracts.py'))}
    rows = []
    for depth in (3, 1):
        spots, result, book, decoded, filtered = signal_example(depth)
        for variant in ('normal', 'empty', 'invalid'):
            current_spots, current = spots, result
            if variant == 'empty':
                current_spots = replace(spots, spots=spots.spots.iloc[:0].copy())
                current = replace(result, values=result.values[:0], spot_ids=(), valid=result.valid[:0])
            elif variant == 'invalid':
                valid = result.valid.copy()
                valid[0, 1] = False
                current = replace(result, valid=valid)
            start = perf_counter()
            report = save_candidate_checkpoint(directory/f'z{depth}-{variant}', current_spots, current,
                codebook=book, dataset_id='artifact-contract-v1', sample_id='sample', FOV=f'FOV-Z{depth}',
                run_id=f'literal-z{depth}-{variant}', config={'fixture': 'artifact-contract-v1'}, code=code,
                sources=({'source_id': 'literal', 'catalog': 'docs/datasets.md#artifact-contract-example-v1',
                    'uri': str(Path(__file__).with_name('artifact_contracts.py')),
                    'sha256': hashlib.sha256(Path(__file__).with_name('artifact_contracts.py').read_bytes()).hexdigest(),
                    'unverified_reason': None, 'selection': {'depth': depth, 'seed': None}},))
            write_seconds = perf_counter() - start
            checksum = hashlib.sha256(report.path.read_bytes()).hexdigest()
            start = perf_counter()
            loaded = load_candidate_checkpoint(report.path, sha256=checksum)
            read_seconds = perf_counter() - start
            assert loaded.intensities.values.tobytes() == current.values.tobytes()
            np.testing.assert_array_equal(loaded.intensities.valid, current.valid, strict=True)
            pd.testing.assert_frame_equal(loaded.spots.spots, current_spots.spots, check_exact=True)
            before = filter_reads(decode_barcodes(current, book, config=WtaDecoderConfig()))
            after = filter_reads(decode_barcodes(loaded.intensities, loaded.codebook, config=WtaDecoderConfig()))
            pd.testing.assert_frame_equal(before.table, after.table, check_exact=True)
            assert before.counts == after.counts
            if variant == 'normal':
                np.testing.assert_array_equal(loaded.intensities.values[0], [[0, 9], [7, 0], [0, 0], [0, 0]])
                assert after.accepted.spot_id.tolist() == ['A']
                trace = loaded.source_trace(run_id=loaded.artifact['run_id'],
                    candidate_artifact_id=loaded.artifact['artifact_id'], spot_namespace=spots.spot_namespace, spot_id='A')
                assert trace['candidate']['x'] == 2.
                export_spots(current_spots, None, directory/f'z{depth}-candidate-diagnostic.csv',
                             columns=['spot_namespace', 'spot_id', 'x', 'y', 'z'])
            rows.append(dict(depth=depth, variant=variant, path=str(report.path), sha256=checksum,
                size_bytes=report.size_bytes, write_seconds=write_seconds, read_seconds=read_seconds,
                counts=after.counts, signal_sha256=hashlib.sha256(current.values.tobytes()).hexdigest()))
    (directory/'summary.json').write_text(json.dumps(dict(fixture='artifact-contract-v1',
        contract='starfinder.artifacts/1', specification_commit='db3667bd9eef7ee6bdea6fe09fae8af3eb1158cc',
        code=code, seed=None, cases=rows, limitations=['Tiny development fixtures; storage qualification remains W-171.',
        'No molecular or scientific qualification.']), indent=2))
    print('Candidate checkpoint examples passed (3D, Z=1, empty and invalid).')


if __name__ == '__main__':
    main(sys.argv[1])
