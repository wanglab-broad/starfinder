"""Build explicit cases from source-derived profiles; never discover or run data.

New maintained STARfinder code under the package's declared MIT terms.
Historical external source code and imregdemons snapshots are not distributed.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from starfinder.benchmark import BenchmarkCase

CONFIG_ROOT = Path(__file__).parent / 'configs'


def profile(task, name):
    return deepcopy(json.loads((CONFIG_ROOT / (task + '.json')).read_text())['profiles'][name])


def registration_case(name, *, case_id, reference, moving, evaluation,
                      reference_metadata, moving_metadata):
    """Select a named method profile, with explicit inputs/geometry/metrics."""
    return BenchmarkCase(case_id, 'registration', {'reference': reference, 'moving': moving},
        {'registration': profile('registration', name)['registration'],
         'reference_metadata': reference_metadata, 'moving_metadata': moving_metadata,
         'evaluation': evaluation})


def pipeline_case(name, *, case_id, fov_id, sources, codebook, n_rounds=None):
    """Translate a preserved pipeline profile to the shared workflow adapter.

    sources is {round: {channel: input-relative TIFF path}}. Supplying it is an
    explicit layout decision; neither round/FOV layout is guessed from a name.
    n_rounds is required for profiles whose count came from ground_truth.json.
    """
    recipe = profile('pipeline', name)
    p = recipe['parameters']
    count = p['n_rounds'] if n_rounds is None else n_rounds
    if p['n_rounds'] is not None and count != p['n_rounds']:
        raise ValueError('round count differs from selected historical profile')
    if type(count) is not int or count < 1:
        raise ValueError('explicit positive n_rounds required')
    params = {
        'streaming': recipe['execution'] == 'streaming',
        'load_raw_images': {'run': True},
        'load_codebook': {'run': True, 'split_index': p['split_index']},
        'enhance_contrast': {'run': True, 'snr_threshold': p['snr_threshold']},
        'spot_finding': {'run': True, 'intensity_estimation': p['threshold_mode'],
                         'intensity_threshold': p['threshold_value']},
        'reads_extraction': {'run': True, 'voxel_size': p['neighborhood_radius_zyx']},
        'reads_filtration': {'run': True, 'end_base': p['end_bases'], 'start_base': p['start_base']}}
    if len(p['registration']) > 2:
        raise ValueError('recipe supports at most global and local stages')
    for key, step in zip(('global_registration', 'local_registration'), p['registration']):
        params[key] = {'run': True, **step}
    workflow = {'dataset_id': 'benchmark', 'sample_id': case_id, 'output_id': 'out',
        'n_rounds': count, 'ref_round': p['reference_round'], 'rotate_angle': p['rotation_degrees'],
        'seq_channel_order': p['channel_order'], 'fov_id_pattern': p['fov_pattern'],
        'rules': {'rsf_single_fov': {'parameters': params}}}
    inputs, source_names = {}, {}
    for i, (round_name, channels) in enumerate(sources.items()):
        source_names[round_name] = {}
        for j, (channel, path) in enumerate(channels.items()):
            key = f'input-{i}-{j}'
            inputs[key] = path
            source_names[round_name][channel] = key
    return BenchmarkCase(case_id, 'pipeline', inputs,
        {'workflow': workflow, 'sources': source_names, 'fov_id': fov_id, 'evaluation': {'counts': True}},
        artifacts={'codebook': codebook})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True,
        help='JSON with task, profile and explicit keyword arguments for the case builder')
    parser.add_argument('--output', type=Path, required=True, help='new CLI configuration JSON')
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    task, name = request.pop('task'), request.pop('profile')
    builders = {'registration': registration_case, 'pipeline': pipeline_case}
    case = builders[task](name, **request)
    from starfinder.benchmark._adapters import _validate
    _validate(case)
    with args.output.open('x') as stream:
        json.dump({'schema_version': 1, 'repetitions': 1, 'cases': [case.to_dict()],
            'provenance': {'recipe': name, 'source': profile(task, name)['source'],
                           'qualification': 'configuration only; inputs/science not qualified'}}, stream, indent=2)


if __name__ == '__main__':
    main()
