"""D1/S1: saved literal calls/QC and sample access; no historical inputs."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from artifact_contracts import signal_example
from starfinder.barcode import decode_barcodes, filter_reads, WtaDecoderConfig, ReadFilterConfig
from starfinder.io import (save_candidate_checkpoint, checkpoint_reference,
    save_decoded_checkpoint, load_decoded_checkpoint, save_final_checkpoint,
    load_final_checkpoint, save_molecule_index, load_molecule_index)


def main(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[2]
    code = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
        dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=repository, text=True)))
    sources, cases = [], []
    for depth, variant in ((3, 'normal'), (1, 'normal'), (1, 'empty'), (1, 'rejected'), (1, 'omitted')):
        name = f'z{depth}-{variant}'
        spots, intensities, book, *_ = signal_example(depth)
        namespace = f'literal/sample/{name}'
        spots = replace(spots, spot_namespace=namespace)
        intensities = replace(intensities, spot_namespace=namespace)
        if variant == 'empty':
            spots = replace(spots, spots=spots.spots.iloc[:0])
            intensities = replace(intensities, values=intensities.values[:0], valid=intensities.valid[:0], spot_ids=())
        context = dict(dataset_id='molecular-contract-v1', sample_id='sample', FOV=name,
            run_id=f'literal-{name}', config={'variant': variant, 'depth': depth, 'seed': None}, code=code)
        candidate = save_candidate_checkpoint(root/name/'candidates', spots, intensities,
            codebook=book, enabled=variant != 'omitted', **context)
        decoded = decode_barcodes(intensities, book, config=WtaDecoderConfig(diagnostics=True))
        pre = save_decoded_checkpoint(root/name/'decoded', spots, decoded, book,
            candidate_source=checkpoint_reference(candidate.path) if candidate.path else None,
            trace_unavailable_reason=candidate.reason, **context)
        saved = load_decoded_checkpoint(pre)
        filtered = filter_reads(saved.decoded,
            config=ReadFilterConfig(call_statuses=()) if variant == 'rejected' else ReadFilterConfig())
        final = save_final_checkpoint(root/name/'final', filtered,
            decoded_source=checkpoint_reference(pre), config={'filter': variant}, code=code)
        loaded = load_final_checkpoint(final)
        pd.testing.assert_frame_equal(loaded.filtering.table, filtered.table, check_exact=True)
        expected = {'total': 2, 'accepted': 1, 'rejected': 1}
        if variant == 'empty': expected = {'total': 0, 'accepted': 0, 'rejected': 0}
        if variant == 'rejected': expected = {'total': 2, 'accepted': 0, 'rejected': 2}
        assert loaded.filtering.counts == expected
        if variant != 'empty':
            trace = loaded.pre_qc.source_trace(spot_namespace=namespace, spot_id='A')
            if variant == 'omitted':
                assert trace == {'available': False, 'reason': 'candidates_signals_disabled'}
            else:
                np.testing.assert_array_equal(trace['values'], [[0, 9], [7, 0], [0, 0], [0, 0]])
                assert loaded.filtering.table.gene_id.iloc[0] == 'gene-A'
        sources.append(checkpoint_reference(final))
        cases.append(dict(FOV=name, counts=filtered.counts, final=str(final),
            sha256=hashlib.sha256(final.read_bytes()).hexdigest()))
    index_path = save_molecule_index(root/'sample.json', tuple(reversed(sources)),
        dataset_id='molecular-contract-v1', sample_id='sample', section_id='supplied-section')
    index = load_molecule_index(index_path)
    for accepted in (True, False):
        batches = list(index.iter_batches(batch_size=1, accepted_only=accepted))
        pd.testing.assert_frame_equal(pd.concat([b.table for b in batches], ignore_index=True),
            index.read_table(accepted_only=accepted), check_exact=True)
    assert len(index.read_table()) == 3 and len(index.read_table(accepted_only=False)) == 8
    (root/'summary.json').write_text(json.dumps(dict(fixture='molecular-contract-v1',
        contract='starfinder.artifacts/1', specification_commit='db3667bd9eef7ee6bdea6fe09fae8af3eb1158cc',
        code=code, seed=None, cases=cases, sample_index=str(index_path),
        limitations=['Development arithmetic only; no calibrated molecular truth.',
            'FOV frames remain distinct; section is grouping only.',
            'Batch memory is bounded by one FOV; MATLAB and external H5AD consumers are unexecuted.']), indent=2))
    print('Molecular checkpoint examples passed (3D, Z=1, empty, rejected, omitted, sample batches).')


if __name__ == '__main__':
    main(sys.argv[1])
