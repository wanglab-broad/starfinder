"""Pipeline adapter: explicit TIFF sources, shared FOV execution, saved counts.

New STARfinder implementation; historical scripts supply configuration facts,
not vendored code. No MATLAB/Postcode imports or implicit dataset discovery.
"""
from dataclasses import asdict
from pathlib import Path
import re
import tempfile

from ._storage import _verified, _reference, _write, _read, _save_transform


def _adapt(case, input_root, output_root):
    from starfinder.dataset import from_workflow_config
    workflow = dict(case.config['workflow'])
    # Paths belong to the lifecycle, never to a historical host-specific config.
    if set(workflow) & {'root_input_path', 'root_output_path'}:
        raise ValueError('pipeline roots are supplied by the benchmark lifecycle')
    workflow.update(root_input_path=str(input_root), root_output_path=str(output_root))
    return from_workflow_config(workflow)


def _validate_pipeline(case):
    if set(case.config) != {'workflow', 'sources', 'fov_id', 'evaluation'}:
        raise ValueError('pipeline config requires workflow, sources, fov_id and evaluation')
    if case.config['evaluation'] != {'counts': True} or case.truth:
        raise ValueError('pipeline adapter evaluates saved counts only; no inferred truth')
    adapted = _adapt(case, Path('/input'), Path('/output'))
    ds = adapted.dataset
    safe = lambda s: isinstance(s, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', s)
    if not all(safe(s) for s in (ds.dataset_id, ds.sample_id, ds.output_id, case.config['fov_id'], *ds.channel_order)):
        raise ValueError('pipeline identifiers must be safe path components')
    sources = case.config['sources']
    if set(sources) != set(ds.rounds.sequencing_rounds):
        raise ValueError('sources must cover every configured sequencing round')
    names = []
    for channels in sources.values():
        if set(channels) != set(ds.channel_order):
            raise ValueError('sources must cover every configured channel')
        names.extend(channels.values())
    if set(names) != set(case.inputs) or len(names) != len(set(names)):
        raise ValueError('each round/channel requires one distinct named input')
    if any(Path(p).suffix.lower() not in ('.tif', '.tiff') for p in case.inputs.values()):
        raise ValueError('pipeline inputs must be single-channel TIFFs')
    if not adapted.pipeline.load or adapted.dataset.rounds.other_rounds:
        raise ValueError('pipeline adapter requires raw sequencing-round loading')
    if adapted.pipeline.load.subdir:
        raise ValueError('pipeline sources replace load subdir selection')
    if not all((adapted.pipeline.detection, adapted.pipeline.extraction,
                adapted.pipeline.decoding, adapted.pipeline.filtering)):
        raise ValueError('pipeline count recipe requires detection, extraction, decoding and filtering')
    if set(case.artifacts) != {'codebook'}:
        raise ValueError('pipeline requires an explicit codebook artifact')


def _process_pipeline(case, root, directory, trial):
    # Links are disposable local scratch; persisted artifacts are ordinary files.
    with tempfile.TemporaryDirectory(prefix='starfinder-pipeline-') as scratch:
        adapted = _adapt(case, Path(scratch), directory)
        ds = adapted.dataset
        for round_name, channels in case.config['sources'].items():
            folder = ds.input_root / round_name / case.config['fov_id']
            folder.mkdir(parents=True)
            for channel, name in channels.items():
                (folder / (channel + '.tif')).symlink_to(_verified(root, trial.artifacts[name]))
        ds.load_codebook(_verified(root, trial.artifacts['artifacts:codebook']),
                         split_index=adapted.split_index)
        fov = ds.fov(case.config['fov_id'])
        try:
            fov.run(adapted.pipeline, execution=adapted.execution)
        finally:
            trial.attempts = [dict(round=round_name, **attempt)
                for round_name, attempts in fov.registration_attempts.items() for attempt in attempts]
        for round_name, results in fov.registration_results.items():
            for index, result in enumerate(results):
                _save_transform(root, directory, f'transform-{round_name}-{index}', result, trial.artifacts)
        trial.actual_method = 'pipeline'
        effective = asdict(adapted.pipeline)
        for step in effective['registration']:
            if step['recovery']:
                step['recovery']['allowed_errors'] = [e.__name__ for e in step['recovery']['allowed_errors']]
        trial.effective_configs = {**case.config, 'pipeline': effective,
                                   'execution': asdict(adapted.execution)}
        for name, table in [('spots', fov.spot_result.spots), ('reads', fov.filtering_result.table)]:
            path = directory / (name + '.csv')
            table.to_csv(path, index=False, mode='x')
            trial.artifacts[name] = _reference(root, path)
        _write(directory / 'pipeline.json', {
            'detected': len(fov.spot_result.spots), 'counts': fov.filtering_result.counts,
            'fractions': fov.filtering_result.fractions,
            'metadata': {k: asdict(v) for k, v in fov.metadata.items()},
            'rounds': asdict(ds.rounds), 'channel_order': ds.channel_order,
            'spot_namespace': fov.spot_result.spot_namespace,
            'retained_rounds': list(fov.images), 'registration_attempts': fov.registration_attempts})
        trial.artifacts['pipeline'] = _reference(root, directory / 'pipeline.json')
        trial.status['processing'] = ('fallback_success' if any(a['outcome'] == 'failed' for a in trial.attempts) else 'success')


def _evaluate_pipeline(root, trial):
    import pandas as pd
    saved = _read(_verified(root, trial.artifacts['pipeline']))
    spots = pd.read_csv(_verified(root, trial.artifacts['spots']))
    reads = pd.read_csv(_verified(root, trial.artifacts['reads']))
    if len(spots) != saved['detected'] or set(spots.spot_id) != set(reads.spot_id):
        raise ValueError('saved pipeline identity/count mismatch')
    values = {'detected': len(spots), 'accepted': int(reads.accepted.sum())}
    return {'counts': {'status': 'ok', 'values': values,
                       'units': {k: 'count' for k in values}, 'reasons': {},
                       'counts': values, 'config': {'source': 'saved tables'}, 'details': {}}}
