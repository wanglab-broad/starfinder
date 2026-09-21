"""Bounded saved successful/failed runs; render in a separate invocation."""
from dataclasses import replace
import json
from pathlib import Path
import sys

from saved_synthetic import create, pipeline, source_identity, checksum, write_json
from starfinder.dataset import Dataset, ExecutionConfig, RegistrationStep
from starfinder.io import load_image_checkpoint
from starfinder.provenance import RunRecorder, read_run
from starfinder.registration import TpsConfig, InsufficientLandmarksError
from starfinder.reporting import write_run_summary


def prepare(root):
    create(root / 'saved')
    checkpoint = load_image_checkpoint(root/'saved/z9/prepared')
    ds = Dataset(root, root/'unused', 'saved-formed-z9-v3', 'sample', 'output',
                 checkpoint.rounds, checkpoint.layers[0].loaded.channel_labels)
    from starfinder.io import load_candidate_checkpoint
    ds.codebook = load_candidate_checkpoint(root/'saved/z9/run/candidates-signals').codebook
    fov = ds.fov('FOV_001')
    for layer in checkpoint.layers:
        fov.images[layer.round_label] = layer.loaded.image
        fov.metadata[layer.round_label] = layer.loaded.metadata
    rec = RunRecorder(root/'failed', dataset_id=ds.dataset_id, sample_id='sample',
        code=source_identity(), owner='Jiahao', retention='thesis/project handoff and publication',
        sources=(dict(source_id='prepared', catalog='docs/datasets.md#saved-formed-development-v3',
            uri=str(root/'saved/z9/prepared/artifact.json'), sha256=checksum(root/'saved/z9/prepared/artifact.json'),
            unverified_reason=None, selection={'role': 'intentional TPS failure; only two landmarks'}),))
    try:
        fov.run(replace(pipeline(), registration=(RegistrationStep(TpsConfig()),)),
                execution=ExecutionConfig('batch'), provenance=rec)
    except InsufficientLandmarksError:
        pass
    else:
        raise AssertionError('Two landmarks must not satisfy TPS')
    run = read_run(rec.path)
    assert run['status'] == 'failed'
    assert run['extensions']['starfinder.provenance']['stage_state']['candidates_signals'] == 'partial'
    assert run['extensions']['starfinder.provenance']['final_state']['counts'] is None
    write_json(root/'inputs.json', dict(fixture='saved-summary-v1', seed=42,
        sources={str(p.relative_to(root)): checksum(p) for p in sorted(root.rglob('*')) if p.is_file()},
        limits='Two formed objects; ZYXC (9,32,32,4)/(1,32,32,4); three rounds; development only'))


def render(root):
    manifest = json.loads((root/'inputs.json').read_text())
    for name, expected in manifest['sources'].items():
        assert checksum(root/name) == expected, name
    for name, path in [('z9', root/'saved/z9/run'), ('z1', root/'saved/z1/run'), ('failed', root/'failed')]:
        write_run_summary(path, root/(name+'-summary.html'), title=name+' saved run')
    from checkpoint_inspection import inspect_saved
    inspect_saved(root/'saved', root/'inspection.html')
    print('Saved success/failure summaries and checkpoint inspection passed.')


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] not in ('prepare', 'render'):
        raise SystemExit('Usage: run_summaries.py {prepare|render} /external/new-directory')
    globals()[sys.argv[1]](Path(sys.argv[2]).resolve())
