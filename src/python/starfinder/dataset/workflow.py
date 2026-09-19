"""Single translation boundary for shared MATLAB/Snakemake configuration."""
from dataclasses import dataclass, fields
from pathlib import Path

from .config import PipelineConfig, ExecutionConfig, RegistrationStep, RecoveryConfig
from .dataset import Dataset
from .types import RoundState, SubtileConfig
from starfinder.io import ImageLoadConfig
from starfinder.preprocessing import (MinMaxNormalizationConfig, HistogramMatchingConfig,
    ReconstructionConfig, TophatConfig, ProjectionConfig)
from starfinder.registration import (TranslationConfig, DemonsConfig, TpsConfig, CpdConfig,
    InsufficientLandmarksError, RegistrationEstimationError, WarpConfig)
from starfinder.barcode import NeighborhoodSumConfig, ReadFilterConfig, WtaDecoderConfig
from starfinder.spot_finding import LocalMaximaConfig

_RULES = ('rsf_single_fov', 'gr_single_fov_subtile', 'lrsf_single_fov_subtile',
          'deep_create_subtile', 'deep_rsf_subtile')


def _known(values, allowed, context):
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f'unknown {context} keys: {sorted(unknown)}')


def _operation(params, name, allowed):
    values = params.get(name, {})
    if not isinstance(values, dict):
        raise TypeError(f'{name} must be a mapping')
    _known(values, {'run', *allowed}, name)
    if 'run' in values and not isinstance(values['run'], bool):
        raise ValueError(f'{name}.run must be Boolean')
    return {k: v for k, v in values.items() if k != 'run'}, values.get('run', False)


def _registration(values, *, local=False):
    values = dict(values)
    method = values.pop('method', 'demons' if local else 'translation')
    mode = lambda v: {'merged-image': 'merged', 'merged': 'merged', 'single-channel': 'single-channel'}[v]
    try:
        reference = mode(values.pop('ref_img', 'single-channel' if local else 'merged-image'))
        moving = mode(values.pop('mov_img', 'single-channel' if local else 'merged-image'))
    except KeyError as error:
        raise ValueError('unknown registration image representation') from error
    channel = values.pop('ref_channel', 0)
    boundary = values.pop('boundary_mode', None)
    recovery_values = values.pop('recovery', None)
    mapping = {'detection_threshold': 'detection_noise_sigma', 'match_distance': 'match_distance_voxels',
        'tps_smoothing': 'smoothing', 'grid_spacing': 'grid_spacing_voxels', 'beta': 'kernel_width_voxels',
        'lmbda': 'regularization_weight', 'cpd_w': 'outlier_fraction', 'candidate_radius': 'candidate_radius_voxels',
        'k_neighbors': 'neighbors_per_anchor'}
    translated = {}
    for key, value in values.items():
        target = mapping.get(key, key)
        if target in translated:
            raise ValueError(f'duplicate registration setting {target}')
        translated[target] = value
    if method == 'translation':
        config = TranslationConfig(**translated)
    elif method == 'tps':
        config = TpsConfig(**translated)
    elif method == 'cpd':
        translated.setdefault('detection_noise_sigma', 3.0)
        translated.setdefault('grid_spacing_voxels', 32)
        config = CpdConfig(**translated)
    elif method in ('demons', 'diffeomorphic', 'symmetric', 'fast_symmetric'):
        if 'iterations' in translated:
            translated['iterations'] = tuple(translated['iterations'])
        config = DemonsConfig(variant=method, **translated)
    else:
        raise ValueError(f'unknown registration method {method}')
    recovery = None
    if recovery_values is not None:
        _known(recovery_values, ('allowed_errors', 'alternatives'), 'recovery')
        errors = {'InsufficientLandmarksError': InsufficientLandmarksError, 'RegistrationEstimationError': RegistrationEstimationError}
        try:
            allowed = tuple(errors[e] for e in recovery_values['allowed_errors'])
        except KeyError as error:
            raise ValueError('invalid recovery error category') from error
        alternatives = []
        for alternative in recovery_values['alternatives']:
            # Alternatives configure estimators only; no hidden warp/reduction policy.
            if set(alternative) & {'recovery', 'boundary_mode', 'ref_img', 'mov_img', 'ref_channel'}:
                raise ValueError('recovery alternatives configure estimators only')
            alternatives.append(_registration(alternative, local=local).config)
        recovery = RecoveryConfig(allowed, tuple(alternatives))
    warp = None
    if boundary is not None:
        backend = 'translation' if method == 'translation' else 'scipy' if method in ('tps', 'cpd') else 'simpleitk'
        warp = WarpConfig(backend=backend, boundary_mode=boundary)
    return RegistrationStep(config, reference, moving, channel, recovery, warp)


@dataclass(frozen=True)
class WorkflowConfig:
    """Translated dataset, scientific pipeline and execution/output policies."""
    dataset: Dataset
    pipeline: PipelineConfig
    execution: ExecutionConfig
    split_index: int | None = None
    reference_projection: ProjectionConfig | None = None


def from_workflow_config(config: dict, rule: str = 'rsf_single_fov') -> WorkflowConfig:
    """Validate and translate one shared rule; never mutate shared MATLAB keys.

    Other rules/top-level acquisition/downstream keys are shared with MATLAB.
    Unknown fields in this Python rule's parameters raise instead of disappearing.
    Direct Python callers construct Dataset/PipelineConfig (no legacy aliases).
    """
    if rule not in _RULES:
        raise ValueError(f'unsupported Python workflow rule {rule}')
    n = config['n_rounds']
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError('n_rounds must be a positive integer')
    rounds = RoundState(sequencing_rounds=[f'round{i}' for i in range(1, n + 1)],
                        other_rounds=list(config.get('additional_round', [])), reference_round=config['ref_round'])
    rounds.validate()
    if set(config) & {'channel_order', 'fov_pattern'}:
        raise ValueError('direct Python field names belong to Dataset; use shared seq_channel_order/fov_id_pattern here')
    channels = tuple(config.get('seq_channel_order', ()))
    dataset = Dataset(Path(config['root_input_path']) / config['dataset_id'] / config['sample_id'],
        Path(config['root_output_path']) / config['dataset_id'] / config['output_id'],
        config['dataset_id'], config['sample_id'], config['output_id'], rounds=rounds,
        channel_order=channels, fov_pattern=config.get('fov_id_pattern', 'Position%03d'))
    params = config.get('rules', {}).get(rule, {}).get('parameters', {})
    _known(params, ('streaming', 'snr_threshold', 'load_codebook', 'load_raw_images', 'enhance_contrast',
        'hist_equalize', 'morph_recon', 'tophat', 'global_registration', 'local_registration',
        'spot_finding', 'reads_extraction', 'reads_filtration', 'create_subtiles'), 'Python workflow parameter')
    norm, do_norm = _operation(params, 'enhance_contrast', ('snr_threshold',))
    hist, do_hist = _operation(params, 'hist_equalize', ('reference_channel',))
    morph, do_morph = _operation(params, 'morph_recon', ('radius',))
    top, do_top = _operation(params, 'tophat', ('radius',))
    spot, do_spot = _operation(params, 'spot_finding', ('ref_round', 'intensity_estimation', 'intensity_threshold', 'min_distance', 'min_distance_voxels'))
    extract, do_extract = _operation(params, 'reads_extraction', ('voxel_size',))
    filt, do_filter = _operation(params, 'reads_filtration', ('end_base', 'start_base', 'exclude_invalid_endpoints', 'score_bounds', 'n_barcode_segments', 'split_index'))
    book, _ = _operation(params, 'load_codebook', ('split_index',))
    load, do_load = _operation(params, 'load_raw_images', ('subdir',))
    steps = []
    registration_keys = {f.name for cls in (TranslationConfig, DemonsConfig, TpsConfig, CpdConfig) for f in fields(cls) if f.init}
    registration_keys |= {'ref_round', 'method', 'ref_img', 'mov_img', 'ref_channel', 'boundary_mode', 'recovery',
        'detection_threshold', 'match_distance', 'tps_smoothing', 'grid_spacing', 'beta', 'lmbda', 'cpd_w', 'candidate_radius', 'k_neighbors'}
    for name in ('global_registration', 'local_registration'):
        values, enabled = _operation(params, name, registration_keys)
        if values.pop('ref_round', rounds.reference_round) != rounds.reference_round:
            raise ValueError('rule reference round differs from dataset reference')
        if enabled:
            steps.append(_registration(values, local=name == 'local_registration'))
    if spot.get('ref_round', rounds.reference_round) != rounds.reference_round:
        raise ValueError('detection reference differs from dataset reference')
    if filt.get('n_barcode_segments', 1) != 1 or filt.get('split_index') not in (None, []):
        raise ValueError('segmented endpoint filtering is not supported by ReadFilterConfig')
    resident = rule in ('lrsf_single_fov_subtile', 'deep_rsf_subtile')
    # The adapter loads raw input unless this is a saved-subtile job or explicitly disabled.
    load_config = ImageLoadConfig(channel_labels=channels, **load) if channels and not resident and ('load_raw_images' not in params or do_load) else None
    pipeline = PipelineConfig(load=load_config, rotation_degrees=None if resident else config.get('rotate_angle'),
        normalization=MinMaxNormalizationConfig('uint8', (0, 255), snr_threshold=norm.get('snr_threshold', params.get('snr_threshold'))) if do_norm else None,
        histogram=HistogramMatchingConfig() if do_hist else None, histogram_reference_channel=hist.get('reference_channel', 0),
        reconstruction=ReconstructionConfig(radius_yx=morph.get('radius', 3)) if do_morph else None,
        reconstruction_after_registration=resident,
        tophat=TophatConfig(radius_yx=top.get('radius', 3)) if do_top else None,
        registration=tuple(steps),
        detection=LocalMaximaConfig(threshold_mode=spot.get('intensity_estimation', 'noise'), threshold_value=spot.get('intensity_threshold', 5.0), min_distance_voxels=spot.get('min_distance_voxels', spot.get('min_distance', 1))) if do_spot else None,
        extraction=NeighborhoodSumConfig(tuple(extract.get('voxel_size', (1, 2, 2)))) if do_extract else None,
        decoding=WtaDecoderConfig(diagnostics=True) if do_filter else None,
        filtering=ReadFilterConfig(end_bases=filt.get('end_base'), start_base=filt.get('start_base', 'C'), exclude_invalid_endpoints=filt.get('exclude_invalid_endpoints', False), score_bounds=filt.get('score_bounds', {})) if do_filter else None)
    creation_rules = (('deep_create_subtile',) if rule.startswith('deep_') else
                      ('gr_single_fov_subtile',) if rule == 'lrsf_single_fov_subtile' else
                      (rule,) if rule == 'gr_single_fov_subtile' else
                      ('gr_single_fov_subtile', 'deep_create_subtile'))
    for subtile_rule in creation_rules:
        subtile_params = config.get('rules', {}).get(subtile_rule, {}).get('parameters', {})
        values, enabled = _operation(subtile_params, 'create_subtiles', ('sqrt_pieces', 'overlap_ratio'))
        if enabled or values:
            dataset.subtile = SubtileConfig(values.get('sqrt_pieces', 4), values.get('overlap_ratio', .1))
            dataset.subtile.compute_windows(config['img_row'], config['img_col'])
            break
    streaming = params.get('streaming', False)
    if not isinstance(streaming, bool):
        raise ValueError('streaming must be Boolean')
    split_index = book.get('split_index') or None
    if isinstance(split_index, list):
        if len(split_index) != 1:
            raise ValueError('split_index requires one two-segment boundary')
        split_index = split_index[0]
    return WorkflowConfig(dataset, pipeline,
        ExecutionConfig('streaming' if streaming else 'batch', rule in ('gr_single_fov_subtile', 'deep_create_subtile')),
        split_index, ProjectionConfig() if config.get('maximum_projection', False) else None)


def _run_workflow(snakemake, rule):
    """Thin common runner used by the five Snakemake script adapters."""
    from .fov import FOV
    adapted = from_workflow_config(snakemake.config, rule)
    dataset = adapted.dataset
    fov_id = snakemake.wildcards.fovID
    resident = rule in ('lrsf_single_fov_subtile', 'deep_rsf_subtile')
    if adapted.pipeline.decoding:
        dataset.load_codebook(Path(snakemake.input[1]), split_index=adapted.split_index)
    fov = FOV.from_subtile(Path(snakemake.input[2]), dataset, fov_id) if resident else dataset.fov(fov_id)
    fov.run(adapted.pipeline, execution=adapted.execution)
    if resident:
        number = snakemake.wildcards.n_subtile
        fov.save_spots(path=Path(snakemake.input[2]).parent / f'subtile_goodSpots_{number}.csv')
        fov.save_diagnostics(suffix=f'_{number}')
    else:
        fov.save_reference_image(projection=adapted.reference_projection)
        if rule == 'rsf_single_fov':
            fov.save_spots()
            fov.save_diagnostics()
        else:
            fov.create_subtiles()
        fov.save_processing_log('rsf' if rule == 'rsf_single_fov' else 'gr')
