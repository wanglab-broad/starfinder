"""Single translation boundary for shared MATLAB/Snakemake configuration."""
from dataclasses import dataclass, fields
from pathlib import Path

from .config import PipelineConfig, ExecutionConfig, RegistrationRecipe, RegistrationStep, RecoveryConfig
from .dataset import Dataset
from .types import RoundState, SubtileConfig
from starfinder.io import ImageLoadConfig
from starfinder.preprocessing import (MinMaxNormalizationConfig, HistogramMatchingConfig,
    ReconstructionConfig, TophatConfig, ProjectionConfig, PreprocessingRecipe, PreprocessingStep, step_config_type)
from starfinder._registry import config_type_for
from starfinder.registration import (REGISTRATION_METHODS, CpdConfig, DemonsConfig, InsufficientLandmarksError,
    RegistrationEstimationError, RegistrationQcConfig, RegistrationRejectedError, RegistrationSignalConfig, WarpConfig)
from starfinder.barcode import NeighborhoodSumConfig, ReadFilterConfig, WtaDecoderConfig
from starfinder.spot_finding import LocalMaximaConfig

_RULES = ('rsf_single_fov', 'gr_single_fov_subtile', 'lrsf_single_fov_subtile',
          'deep_create_subtile', 'deep_rsf_subtile')
# Keys replaced by the explicit preprocessing key (snr_threshold only feeds min-max).
_LEGACY_PREPROCESSING = ('enhance_contrast', 'hist_equalize', 'morph_recon', 'tophat', 'snr_threshold')
# Legacy method names: the demons variants select method demons with that variant.
_DEMONS_VARIANTS = ('diffeomorphic', 'symmetric', 'fast_symmetric')


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


_RECOVERY_ERRORS = {'InsufficientLandmarksError': InsufficientLandmarksError,
                    'RegistrationEstimationError': RegistrationEstimationError,
                    'RegistrationRejectedError': RegistrationRejectedError}
# Legacy image representations: merged (either spelling) is the channel maximum, as in MATLAB.
_SIGNAL_MODES = {'merged-image': 'max', 'merged': 'max', 'single-channel': 'channel'}


def _recovery(values, alternative):
    """RecoveryConfig from a recovery mapping; alternative(entry) builds one alternative config."""
    if not isinstance(values, dict):
        raise TypeError('recovery must be a mapping')
    _known(values, ('allowed_errors', 'alternatives'), 'recovery')
    try:
        allowed = tuple(_RECOVERY_ERRORS[e] for e in values['allowed_errors'])
    except KeyError as error:
        raise ValueError('invalid recovery error category') from error
    return RecoveryConfig(allowed, tuple(alternative(entry) for entry in values['alternatives']))


def _registration(values, *, local=False):
    """One legacy block: (config, signal, boundary_mode, recovery) of its registration step."""
    values = dict(values)
    method = values.pop('method', 'demons' if local else 'translation')
    ref_img, mov_img = values.pop('ref_img', 'merged-image'), values.pop('mov_img', 'merged-image')
    if ref_img not in _SIGNAL_MODES or mov_img not in _SIGNAL_MODES:
        raise ValueError('unknown registration image representation')
    if _SIGNAL_MODES[ref_img] != _SIGNAL_MODES[mov_img]:
        raise ValueError(f'ref_img {ref_img!r} and mov_img {mov_img!r} differ; one registration signal is used for both rounds')
    channel = values.pop('ref_channel', 0)
    signal = (RegistrationSignalConfig('channel', channel) if _SIGNAL_MODES[ref_img] == 'channel'
              else RegistrationSignalConfig('max'))
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
    config_type = config_type_for(REGISTRATION_METHODS, 'demons' if method in _DEMONS_VARIANTS else method,
                                  'registration method', message=f'unknown registration method {method}')
    block = 'local' if local else 'global'
    if REGISTRATION_METHODS[config_type].step_kind != block:
        raise ValueError(f'{method} is not a {block} registration method')
    if config_type is CpdConfig:
        translated.setdefault('detection_noise_sigma', 3.0)
        translated.setdefault('grid_spacing_voxels', 32)
    if config_type is DemonsConfig:
        if 'iterations' in translated:
            translated['iterations'] = tuple(translated['iterations'])
        config = DemonsConfig(variant=method, **translated)
    else:
        config = config_type(**translated)
    recovery = None
    if recovery_values is not None:
        def alternative(entry):
            # Alternatives configure estimators only; no hidden warp/reduction policy.
            if set(entry) & {'recovery', 'boundary_mode', 'ref_img', 'mov_img', 'ref_channel'}:
                raise ValueError('recovery alternatives configure estimators only')
            return _registration(entry, local=local)[0]
        recovery = _recovery(recovery_values, alternative)
    return config, signal, boundary, recovery


def _legacy_registration(blocks):
    """The recipe of the enabled legacy blocks, global then local; None when neither runs.

    The first block's signal is the recipe signal and a different one becomes
    the other step's own signal. The blocks' boundary_mode is the recipe's one
    final resampling, so different values are rejected; nearest selects the
    SciPy backend, and constant (or none) keeps the derived default.
    """
    if not blocks:
        return None
    steps = [_registration(values, local=local) for values, local in blocks]
    boundaries = {boundary for _, _, boundary, _ in steps if boundary is not None}
    if len(boundaries) > 1:
        raise ValueError(f'boundary_mode differs between the registration blocks: {sorted(boundaries)}')
    boundary = boundaries.pop() if boundaries else None
    signal = steps[0][1]
    warp = None if boundary in (None, 'constant') else WarpConfig(backend='scipy', boundary_mode=boundary)
    return RegistrationRecipe(tuple(RegistrationStep(config, recovery, None if own == signal else own)
                                    for config, own, _, recovery in steps), signal=signal, warp=warp)


def _registration_step(entry):
    """One step of the Python-only registration key: a method name, its config fields, recovery and signal."""
    if not isinstance(entry, dict) or 'method' not in entry:
        raise ValueError('each registration step must be a mapping with a method')
    entry = dict(entry)
    config_type = config_type_for(REGISTRATION_METHODS, entry.pop('method'), 'registration method')
    recovery, signal = entry.pop('recovery', None), entry.pop('signal', None)
    _known(entry, [f.name for f in fields(config_type) if f.init], f'registration step {config_type.__name__}')
    config = config_type(**{k: _tuples(v) for k, v in entry.items()})

    def alternative(values):
        if isinstance(values, dict) and set(values) & {'recovery', 'signal'}:
            raise ValueError('recovery alternatives configure estimators only')
        return _registration_step(values).config
    return RegistrationStep(config, None if recovery is None else _recovery(recovery, alternative),
                            None if signal is None else _typed(RegistrationSignalConfig, signal, 'registration signal'))


def _typed(cls, values, context):
    if not isinstance(values, dict):
        raise TypeError(f'{context} must be a mapping')
    _known(values, [f.name for f in fields(cls) if f.init], context)
    return cls(**{k: _tuples(v) for k, v in values.items()})


def _explicit_registration(values):
    """Recipe from the Python-only registration key: steps named by their REGISTRATION_METHODS names,
    plus the recipe's signal, warp, qc and reference_round.
    """
    if not isinstance(values, dict):
        raise TypeError('registration must be a mapping')
    _known(values, ('steps', 'signal', 'warp', 'qc', 'reference_round'), 'registration')
    if not isinstance(values.get('steps'), list):
        raise ValueError('registration.steps must be a list')
    options = {}
    for name, cls in (('signal', RegistrationSignalConfig), ('warp', WarpConfig), ('qc', RegistrationQcConfig)):
        if values.get(name) is not None:
            options[name] = _typed(cls, values[name], f'registration {name}')
    return RegistrationRecipe(tuple(map(_registration_step, values['steps'])),
                              reference_round=values.get('reference_round'), **options)


def _legacy_recipe(params, norm, do_norm, hist, do_hist, morph, do_morph, top, do_top, resident):
    """Recipe 1 from the legacy keys, in the legacy order; None when no key runs.

    Resident subtile rules run reconstruction after registration.
    """
    steps, post = [], []
    if do_norm:
        steps.append(MinMaxNormalizationConfig('uint8', (0, 255), rounding='truncate',
            snr_threshold=norm.get('snr_threshold', params.get('snr_threshold'))))
    if do_hist:
        steps.append(HistogramMatchingConfig(reference_channel=hist.get('reference_channel', 0)))
    if do_morph:
        (post if resident else steps).append(ReconstructionConfig(radius_yx=morph.get('radius', 3)))
    if do_top:
        steps.append(TophatConfig(radius_yx=top.get('radius', 3)))
    if not steps and not post:
        return None
    return PreprocessingRecipe(tuple(map(PreprocessingStep, steps)), tuple(map(PreprocessingStep, post)))


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


def _explicit_recipe(values):
    """Recipe from the preprocessing key: steps named by their PREPROCESSING_METHODS names, plus the recipe fields.

    A step's keys other than method and save_as are the fields of its config
    dataclass; YAML lists become tuples.
    """
    if not isinstance(values, dict):
        raise TypeError('preprocessing must be a mapping')
    _known(values, ('steps', 'extraction_source', 'registration_source', 'supplied_statistics'), 'preprocessing')
    if not isinstance(values.get('steps'), list):
        raise ValueError('preprocessing.steps must be a list')
    steps = []
    for entry in values['steps']:
        if not isinstance(entry, dict) or 'method' not in entry:
            raise ValueError('each preprocessing step must be a mapping with a method')
        entry = dict(entry)
        config_type = step_config_type(entry.pop('method'))
        save_as = entry.pop('save_as', None)
        _known(entry, [f.name for f in fields(config_type) if f.init], f'preprocessing step {config_type.__name__}')
        steps.append(PreprocessingStep(config_type(**{k: _tuples(v) for k, v in entry.items()}), save_as))
    supplied = values.get('supplied_statistics')
    return PreprocessingRecipe(tuple(steps), extraction_source=values.get('extraction_source'),
                               registration_source=values.get('registration_source'),
                               supplied_statistics=None if supplied is None else Path(supplied))


@dataclass(frozen=True)
class WorkflowConfig:
    """Translated dataset, scientific pipeline and execution/output policies.

    reference_projection, reference_image and reference_channel are passed to
    FOV.save_reference_image for ``images/ref_merged``.
    """
    dataset: Dataset
    pipeline: PipelineConfig
    execution: ExecutionConfig
    split_index: int | None = None
    reference_projection: ProjectionConfig | None = None
    reference_image: str = 'merged'
    reference_channel: int = 0


def from_workflow_config(config: dict, rule: str = 'rsf_single_fov') -> WorkflowConfig:
    """Validate and translate one shared rule; never mutate shared MATLAB keys.

    Other rules/top-level acquisition/downstream keys are shared with MATLAB.
    Unknown fields in this Python rule's parameters raise instead of disappearing.
    The Python-only preprocessing key declares an explicit recipe and is
    mutually exclusive with the legacy keys enhance_contrast, hist_equalize,
    morph_recon, tophat and snr_threshold, which map to recipe 1. The
    Python-only registration key declares a RegistrationRecipe and is
    mutually exclusive with an enabled global_registration or
    local_registration; those legacy blocks map to one recipe (global step,
    then local step; merged-image is the channel maximum).
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
        'hist_equalize', 'morph_recon', 'tophat', 'preprocessing', 'registration', 'global_registration', 'local_registration',
        'spot_finding', 'reads_extraction', 'reads_filtration', 'create_subtiles'), 'Python workflow parameter')
    legacy = [key for key in _LEGACY_PREPROCESSING if key in params]
    if 'preprocessing' in params and legacy:
        raise ValueError(f'preprocessing is mutually exclusive with the legacy keys {legacy}')
    norm, do_norm = _operation(params, 'enhance_contrast', ('snr_threshold',))
    hist, do_hist = _operation(params, 'hist_equalize', ('reference_channel',))
    morph, do_morph = _operation(params, 'morph_recon', ('radius',))
    top, do_top = _operation(params, 'tophat', ('radius',))
    spot, do_spot = _operation(params, 'spot_finding', ('ref_round', 'intensity_estimation', 'intensity_threshold', 'min_distance', 'min_distance_voxels'))
    extract, do_extract = _operation(params, 'reads_extraction', ('voxel_size',))
    filt, do_filter = _operation(params, 'reads_filtration', ('end_base', 'start_base', 'exclude_invalid_endpoints', 'score_bounds', 'n_barcode_segments', 'split_index'))
    book, _ = _operation(params, 'load_codebook', ('split_index',))
    load, do_load = _operation(params, 'load_raw_images', ('subdir',))
    registration_keys = {f.name for cls in REGISTRATION_METHODS for f in fields(cls) if f.init}
    registration_keys |= {'ref_round', 'method', 'ref_img', 'mov_img', 'ref_channel', 'boundary_mode', 'recovery',
        'detection_threshold', 'match_distance', 'tps_smoothing', 'grid_spacing', 'beta', 'lmbda', 'cpd_w', 'candidate_radius', 'k_neighbors'}
    blocks, enabled_registration = [], []
    for name in ('global_registration', 'local_registration'):
        values, enabled = _operation(params, name, registration_keys)
        if values.pop('ref_round', rounds.reference_round) != rounds.reference_round:
            raise ValueError('rule reference round differs from dataset reference')
        if enabled:
            blocks.append((values, name == 'local_registration'))
            enabled_registration.append(name)
    if 'registration' in params and enabled_registration:
        raise ValueError(f'registration is mutually exclusive with the enabled legacy blocks {enabled_registration}')
    registration = (_explicit_registration(params['registration']) if 'registration' in params
                    else _legacy_registration(blocks))
    if registration is not None and registration.reference_round not in (None, rounds.reference_round):
        raise ValueError('registration reference round differs from dataset reference')
    # ref_merged is what the MATLAB script saves: the reference image of its last
    # registration. Only rsf_single_fov passes ref_img, to global registration;
    # its local registration and the other scripts use the channel maximum.
    reference_view = ('merged', 0)
    if rule == 'rsf_single_fov' and enabled_registration == ['global_registration']:
        signal = registration.signal
        reference_view = ('single-channel', signal.reference_channel) if signal.mode == 'channel' else ('merged', 0)
    if spot.get('ref_round', rounds.reference_round) != rounds.reference_round:
        raise ValueError('detection reference differs from dataset reference')
    if filt.get('n_barcode_segments', 1) != 1 or filt.get('split_index') not in (None, []):
        raise ValueError('segmented endpoint filtering is not supported by ReadFilterConfig')
    resident = rule in ('lrsf_single_fov_subtile', 'deep_rsf_subtile')
    # The adapter loads raw input unless this is a saved-subtile job or explicitly disabled.
    load_config = ImageLoadConfig(channel_labels=channels, **load) if channels and not resident and ('load_raw_images' not in params or do_load) else None
    pipeline = PipelineConfig(load=load_config, rotation_degrees=None if resident else config.get('rotate_angle'),
        preprocessing=(_explicit_recipe(params['preprocessing']) if 'preprocessing' in params else
                       _legacy_recipe(params, norm, do_norm, hist, do_hist, morph, do_morph, top, do_top, resident)),
        registration=registration,
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
        split_index, ProjectionConfig() if config.get('maximum_projection', False) else None, *reference_view)


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
        fov.save_reference_image(projection=adapted.reference_projection, reference_image=adapted.reference_image,
                                 reference_channel=adapted.reference_channel)
        if rule == 'rsf_single_fov':
            fov.save_spots()
            fov.save_diagnostics()
        else:
            fov.create_subtiles()
        fov.save_processing_log('rsf' if rule == 'rsf_single_fov' else 'gr')
