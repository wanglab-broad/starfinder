"""Single translation boundary for shared MATLAB/Snakemake configuration."""
from dataclasses import MISSING, dataclass, fields, replace
from pathlib import Path

from .config import ExternalReference, PipelineConfig, ExecutionConfig, RegistrationRecipe, RegistrationStep, RecoveryConfig
from .dataset import Dataset
from .types import RoundState, SubtileConfig
from starfinder.io import ImageLoadConfig
from starfinder.preprocessing import (MinMaxNormalizationConfig, HistogramMatchingConfig,
    ReconstructionConfig, TophatConfig, ProjectionConfig, PreprocessingRecipe, PreprocessingStep, step_config_type)
from starfinder._registry import config_type_for
from starfinder.registration import (REGISTRATION_METHODS, CpdConfig, DemonsConfig, InsufficientLandmarksError,
    RegistrationEstimationError, RegistrationQcConfig, RegistrationRejectedError, RegistrationSignalConfig,
    TranslationConfig, WarpConfig)
from starfinder.barcode import (DECODING_METHODS, ENCODINGS, BarcodeLayout, EncodingConfig, NeighborhoodSumConfig,
    OneBaseEncodingConfig, ReadFilterConfig, Segment, WtaDecoderConfig)
from starfinder.barcode.decoding import READOUT_MODES, _mode_mismatch
from starfinder.spot_finding import SPOT_FINDING_METHODS, ChannelOverride, LocalMaximaConfig, SpotFindingPlan

_RULES = ('rsf_single_fov', 'gr_single_fov_subtile', 'lrsf_single_fov_subtile',
          'deep_create_subtile', 'deep_rsf_subtile')
# Keys replaced by the explicit preprocessing key (snr_threshold only feeds min-max).
_LEGACY_PREPROCESSING = ('enhance_contrast', 'hist_equalize', 'morph_recon', 'tophat', 'snr_threshold')
# Legacy method names: the demons variants select method demons with that variant.
_DEMONS_VARIANTS = ('diffeomorphic', 'symmetric', 'fast_symmetric')
# Legacy spot_finding keys: aliases of LocalMaximaConfig fields, for local_maxima only.
_SPOT_ALIASES = {'intensity_estimation': 'threshold_mode', 'intensity_threshold': 'threshold_value',
                 'min_distance': 'min_distance_voxels'}


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


def _spot_fields(values, config_type, context):
    """Init fields of config_type from one spot_finding mapping (YAML lists become tuples).

    The legacy keys are aliases of LocalMaximaConfig fields for local_maxima
    only, and raise together with their field; for any other method a key
    must be an init field of its config (min_distance is then that config's
    native field, passed unchanged). Unknown keys raise ValueError.
    """
    if not isinstance(values, dict):
        raise TypeError(f'{context} must be a mapping')
    names = {f.name for f in fields(config_type) if f.init}
    aliases = _SPOT_ALIASES if config_type is LocalMaximaConfig else {}
    _known(values, names | set(aliases), context)
    for key, field_name in aliases.items():
        if key in values and field_name in values:
            raise ValueError(f'{context}: {key} and {field_name} set the same field; give one of them')
    return {aliases.get(key, key): _tuples(value) for key, value in values.items()}


def _detection(values, channels):
    """The detection config of the spot_finding block, or a SpotFindingPlan when it has channel_overrides or rounds.

    method names a SPOT_FINDING_METHODS method with pipeline=True (default
    local_maxima); the other keys are its config's fields (_spot_fields),
    and a config field without a default (such as the four starfish_log
    scale and threshold settings) must be given, else ValueError.
    channel_overrides maps a channel label to config fields that replace the
    block's for that channel. rounds lists the round labels to detect in
    (omitted: the reference round only); FOV.run checks them.
    """
    values = dict(values)
    values.pop('ref_round', None)
    method = values.pop('method', 'local_maxima')
    overrides = values.pop('channel_overrides', None)
    rounds = values.pop('rounds', None)
    config_type = config_type_for(SPOT_FINDING_METHODS, method, 'spot-finding method')
    if not SPOT_FINDING_METHODS[config_type].pipeline:
        raise ValueError(f'spot-finding method {method!r} is not a pipeline method')
    values = _spot_fields(values, config_type, 'spot_finding')
    missing = [f.name for f in fields(config_type)
               if f.init and f.default is MISSING and f.default_factory is MISSING and f.name not in values]
    if missing:
        raise ValueError(f'spot_finding: method {method!r} requires {", ".join(missing)}')
    config = config_type(**values)
    if rounds is not None and (not isinstance(rounds, (list, tuple)) or not all(isinstance(r, str) for r in rounds)):
        raise TypeError('spot_finding.rounds must be a list of round labels')
    if overrides is None:
        return config if rounds is None else SpotFindingPlan(config, rounds=tuple(rounds))
    if not isinstance(overrides, dict):
        raise TypeError('spot_finding.channel_overrides must be a mapping from channel label to settings')
    unknown = [label for label in overrides if label not in channels]
    if channels and unknown:
        raise ValueError(f'spot_finding.channel_overrides names unknown channels {unknown}; '
                         f'seq_channel_order is {list(channels)}')
    return SpotFindingPlan(config, tuple(
        ChannelOverride(label, replace(config, **_spot_fields(entry, config_type, f'spot_finding.channel_overrides.{label}')))
        for label, entry in overrides.items()), None if rounds is None else tuple(rounds))


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


def _ends(value, context):
    """Allowed (first, last) pairs from one end_base item: a pair string or a list of them."""
    items = [value] if isinstance(value, str) else value
    if (not isinstance(items, list) or not items
            or any(not isinstance(v, str) or len(v) != 2 or any(b not in 'ACGT' for b in v) for v in items)):
        raise ValueError(f'invalid endpoint bases in {context}: {value!r}')
    return tuple(dict.fromkeys((v[0], v[1]) for v in items))


def _split_layout(split_index, n_bases, *, reverse_bases=True, ends=((), ())):
    """Two-segment BarcodeLayout of the shared split_index (docs/readout-contract.md, "Segment layout").

    split_index is MATLAB's one-based position s, in the encoded color string, of
    the junction color that LoadCodebook.m removes; the colors after it are acquired
    first. ends holds the allowed end pairs of each segment in acquisition order.
    The layout equals the zero-based EncodingConfig(split_index=s - 1).
    """
    s = split_index
    if not 2 <= s <= n_bases - 2:
        raise ValueError(f'load_codebook.split_index {s} must leave two segments of at least 2 bases '
                         f'in a {n_bases}-base barcode')
    if reverse_bases:
        return BarcodeLayout((Segment('A', n_bases - s, ends[0]), Segment('B', s, ends[1])), ('A', 'B'))
    return BarcodeLayout((Segment('A', s, ends[1]), Segment('B', n_bases - s, ends[0])), ('B', 'A'))


def _encoding(values):
    """Encoding config of the Python-only load_codebook.encoding mapping (method default two_base)."""
    if not isinstance(values, dict):
        raise TypeError('load_codebook.encoding must be a mapping')
    values = dict(values)
    config_type = config_type_for(ENCODINGS, values.pop('method', 'two_base'), 'encoding')
    # split_index stays the shared load_codebook key; it is translated into the layout.
    _known(values, {f.name for f in fields(config_type) if f.init} - {'split_index'}, 'load_codebook.encoding')
    if isinstance(values.get('base_to_color'), dict):
        values['base_to_color'] = {str(k): str(v) for k, v in values['base_to_color'].items()}
    return config_type(**values)


def _decoding(values, mode='multiplexed'):
    """Decoder config of the Python-only decoding mapping (method default wta, or direct in direct mode).

    The adapter's defaults are diagnostics=True and, for a decoder that can
    rescue, allow_rescue=False unless the block sets them. A decoder that does
    not support the readout mode raises TypeError naming both.
    """
    if not isinstance(values, dict):
        raise TypeError('decoding must be a mapping')
    values = dict(values)
    config_type = config_type_for(DECODING_METHODS, values.pop('method', 'wta' if mode == 'multiplexed' else 'direct'),
                                  'decoding method')
    if mode not in DECODING_METHODS[config_type].modes:
        raise TypeError(_mode_mismatch(mode, DECODING_METHODS[config_type]))
    names = {f.name for f in fields(config_type) if f.init}
    _known(values, names, 'decoding')
    defaults = {'diagnostics': True, 'allow_rescue': False}
    return config_type(**{**{k: v for k, v in defaults.items() if k in names}, **values})


# Barcode keys that have no meaning in readout mode direct (no encoding, layout or end-base check).
_BARCODE_KEYS = (('load_codebook', ('split_index', 'encoding')),
                 ('reads_filtration', ('end_base', 'split_index', 'n_barcode_segments', 'exclude_invalid_endpoints')))


def _readout_mode(config, params):
    """The top-level readout_mode (default multiplexed); direct rejects the barcode keys of _BARCODE_KEYS."""
    mode = config.get('readout_mode', 'multiplexed')
    if mode not in READOUT_MODES:
        raise ValueError(f'readout_mode must be one of {list(READOUT_MODES)}; got {mode!r}')
    if mode == 'direct':
        given = [f'{block}.{key}' for block, keys in _BARCODE_KEYS for key in keys
                 if isinstance(params.get(block), dict) and key in params[block]]
        if given:
            raise ValueError(f'readout_mode direct has no barcode encoding, segment layout or end-base check; '
                             f'remove {given}')
    return mode


def _one_split(value, key):
    """One shared split_index from an integer list (empty or missing: None) or an integer."""
    value = value or None
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError('split_index requires one two-segment boundary')
        value = value[0]
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise ValueError(f'{key} must be an integer list')
    return value


def _codebook_layout(book, filt, n_rounds):
    """(encoding, layout, zero-based split_index, filter end_bases) from the shared keys.

    load_codebook.split_index is MATLAB's one-based position s, translated into a
    two-segment layout equal to EncodingConfig(split_index=s - 1). The stated
    reads_filtration.n_barcode_segments and reads_filtration.split_index must agree
    with it. A string end_base is the one-segment shortcut of ReadFilterConfig; a
    list gives the allowed ends of the layout's segments: with one segment every
    listed pair, with two segments item k for segment k in acquisition order.
    """
    encoding = _encoding(book.get('encoding', {}))
    split = _one_split(book.get('split_index'), 'load_codebook.split_index')
    if split is not None and type(encoding) is not EncodingConfig:
        raise ValueError('load_codebook.split_index applies to the two_base encoding only')
    segments = 1 if split is None else 2
    count = filt.get('n_barcode_segments')
    if count is not None and count != segments:
        raise ValueError(f'reads_filtration.n_barcode_segments {count} differs from the {segments} segment(s) '
                         'of the codebook layout (load_codebook.split_index)')
    if 'split_index' in filt and _one_split(filt['split_index'], 'reads_filtration.split_index') != split:
        raise ValueError('reads_filtration.split_index must equal load_codebook.split_index')
    end_base = filt.get('end_base')
    ends = None
    if isinstance(end_base, list):
        if segments == 1:
            ends = (_ends(end_base, 'reads_filtration.end_base'),)
        elif len(end_base) != 2:
            raise ValueError('with two segments reads_filtration.end_base lists one item per segment')
        else:
            ends = tuple(_ends(item, 'reads_filtration.end_base') for item in end_base)
        end_base = None
    spec = ENCODINGS[type(encoding)]
    if split is not None:
        # n bases give n - 2 colors with the junction color removed.
        layout = _split_layout(split, n_rounds + 2 * spec.junction_colors, reverse_bases=encoding.reverse_bases,
                               ends=ends or ((), ()))
    elif ends is not None:
        layout = BarcodeLayout((Segment('A', n_rounds + spec.junction_colors, ends[0]),))
    else:
        layout = None
    return encoding, layout, None if split is None else split - 1, end_base


@dataclass(frozen=True)
class WorkflowConfig:
    """Translated dataset, scientific pipeline and execution/output policies.

    reference_projection, reference_image and reference_channel are passed to
    FOV.save_reference_image for ``images/ref_merged``. split_index is the
    zero-based EncodingConfig.split_index converted from the shared one-based
    load_codebook.split_index (None without a split); encoding and layout are
    the codebook's encoding config and segment layout (None: one segment), which
    Dataset.load_codebook(path, encoding=..., layout=...) takes.
    """
    dataset: Dataset
    pipeline: PipelineConfig
    execution: ExecutionConfig
    split_index: int | None = None
    reference_projection: ProjectionConfig | None = None
    reference_image: str = 'merged'
    reference_channel: int = 0
    encoding: EncodingConfig | OneBaseEncodingConfig = EncodingConfig()
    layout: BarcodeLayout | None = None


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
    then local step; merged-image is the channel maximum). The spot_finding
    block's Python-only method key names a pipeline SPOT_FINDING_METHODS
    method (default local_maxima) whose config fields are the other keys;
    the legacy keys intensity_estimation, intensity_threshold and
    min_distance are aliases for local_maxima only, channel_overrides
    gives per-channel settings and rounds the rounds to detect in. The rule-level Python-only device key sets
    ExecutionConfig.device. The shared load_codebook.split_index is MATLAB's
    one-based position and becomes the two-segment layout (zero-based
    EncodingConfig.split_index s - 1); reads_filtration.n_barcode_segments,
    reads_filtration.split_index and a list end_base are checked against and
    translated into that layout. The Python-only load_codebook.encoding names an
    ENCODINGS method (default two_base) with its config fields, and the
    Python-only decoding key a DECODING_METHODS method (default wta) with its
    config fields; the adapter decodes with diagnostics and without rescue
    unless that key sets them. The Python-only top-level readout_mode
    (multiplexed, the default, or direct) becomes Dataset.readout_mode; in
    direct mode decoding is the direct method (DirectAssignmentConfig), the
    rule's codebook input is the panel CSV (round,channel,gene_id), the
    candidates need spot_finding.rounds, and the barcode keys
    load_codebook.split_index and encoding and reads_filtration.end_base,
    split_index, n_barcode_segments and exclude_invalid_endpoints raise.
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
    params = config.get('rules', {}).get(rule, {}).get('parameters', {})
    mode = _readout_mode(config, params)
    dataset = Dataset(Path(config['root_input_path']) / config['dataset_id'] / config['sample_id'],
        Path(config['root_output_path']) / config['dataset_id'] / config['output_id'],
        config['dataset_id'], config['sample_id'], config['output_id'], rounds=rounds,
        channel_order=channels, fov_pattern=config.get('fov_id_pattern', 'Position%03d'), readout_mode=mode)
    _known(params, ('streaming', 'snr_threshold', 'load_codebook', 'load_raw_images', 'enhance_contrast',
        'hist_equalize', 'morph_recon', 'tophat', 'preprocessing', 'registration', 'global_registration', 'local_registration',
        'spot_finding', 'reads_extraction', 'reads_filtration', 'decoding', 'create_subtiles', 'device'),
        'Python workflow parameter')
    legacy = [key for key in _LEGACY_PREPROCESSING if key in params]
    if 'preprocessing' in params and legacy:
        raise ValueError(f'preprocessing is mutually exclusive with the legacy keys {legacy}')
    norm, do_norm = _operation(params, 'enhance_contrast', ('snr_threshold',))
    hist, do_hist = _operation(params, 'hist_equalize', ('reference_channel',))
    morph, do_morph = _operation(params, 'morph_recon', ('radius',))
    top, do_top = _operation(params, 'tophat', ('radius',))
    spot_keys = {f.name for cls, spec in SPOT_FINDING_METHODS.items() if spec.pipeline for f in fields(cls) if f.init}
    spot, do_spot = _operation(params, 'spot_finding', {'ref_round', 'method', 'channel_overrides', 'rounds', *_SPOT_ALIASES,
                                                        *spot_keys})
    extract, do_extract = _operation(params, 'reads_extraction', ('voxel_size',))
    filt, do_filter = _operation(params, 'reads_filtration', ('end_base', 'start_base', 'exclude_invalid_endpoints', 'score_bounds', 'n_barcode_segments', 'split_index'))
    book, _ = _operation(params, 'load_codebook', ('split_index', 'encoding'))
    if mode == 'direct':
        encoding, layout, split_index, end_base = EncodingConfig(), None, None, None
    else:
        encoding, layout, split_index, end_base = _codebook_layout(book, filt, n)
    if 'decoding' in params and not do_filter:
        raise ValueError('decoding requires reads_filtration.run')
    decoding = (_decoding(params['decoding'], mode) if 'decoding' in params else
                WtaDecoderConfig(diagnostics=True) if mode == 'multiplexed' else _decoding({}, mode))
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
    resident = rule in ('lrsf_single_fov_subtile', 'deep_rsf_subtile')
    # The adapter loads raw input unless this is a saved-subtile job or explicitly disabled.
    load_config = ImageLoadConfig(channel_labels=channels, **load) if channels and not resident and ('load_raw_images' not in params or do_load) else None
    pipeline = PipelineConfig(load=load_config, rotation_degrees=None if resident else config.get('rotate_angle'),
        preprocessing=(_explicit_recipe(params['preprocessing']) if 'preprocessing' in params else
                       _legacy_recipe(params, norm, do_norm, hist, do_hist, morph, do_morph, top, do_top, resident)),
        registration=registration,
        spot_finding=_detection(spot, channels) if do_spot else None,
        extraction=NeighborhoodSumConfig(tuple(extract.get('voxel_size', (1, 2, 2)))) if do_extract else None,
        decoding=decoding if do_filter else None,
        filtering=ReadFilterConfig(end_bases=end_base, start_base=filt.get('start_base', 'C'), exclude_invalid_endpoints=filt.get('exclude_invalid_endpoints', False), score_bounds=filt.get('score_bounds', {})) if do_filter else None)
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
    return WorkflowConfig(dataset, pipeline,
        ExecutionConfig('streaming' if streaming else 'batch', rule in ('gr_single_fov_subtile', 'deep_create_subtile'),
                        params.get('device', 'cpu')),
        split_index, ProjectionConfig() if config.get('maximum_projection', False) else None, *reference_view,
        encoding=encoding, layout=layout)


def _run_workflow(snakemake, rule):
    """Thin common runner used by the five Snakemake script adapters."""
    from .fov import FOV
    adapted = from_workflow_config(snakemake.config, rule)
    dataset = adapted.dataset
    fov_id = snakemake.wildcards.fovID
    resident = rule in ('lrsf_single_fov_subtile', 'deep_rsf_subtile')
    if adapted.pipeline.decoding and dataset.readout_mode == 'direct':
        # In direct mode the rule's codebook input is the panel CSV.
        dataset.load_direct_panel(Path(snakemake.input[1]))
    elif adapted.pipeline.decoding:
        dataset.load_codebook(Path(snakemake.input[1]), encoding=adapted.encoding, layout=adapted.layout)
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


# The reference stain nuclei_registration.m reads: the reference round's *ch04.tif.
_NUCLEI_REFERENCE_CHANNEL = 'ch04'


def _nuclei_registration(config):
    """(dataset, stain label per other round, output folder per channel per other round) of nuclei_registration.

    Reads the shared keys as workflow/scripts/nuclei_registration.m does. Each
    additional_round entry is an other round with its channel_order: entry
    ``channel`` is the filename pattern and the round's channel label, entry
    ``name`` the output folder. The round's shared stain is its one channel
    whose name contains the top-level ref_channel (MATLAB single-channel
    matching); none or several is a ValueError.
    """
    n = config['n_rounds']
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError('n_rounds must be a positive integer')
    stain = config.get('ref_channel')
    if not isinstance(stain, str) or not stain:
        raise ValueError('nuclei_registration requires the top-level ref_channel naming the shared stain')
    entries = config.get('additional_round') or []
    if not isinstance(entries, list) or not entries:
        raise ValueError('nuclei_registration requires additional_round entries')
    orders, stains, folders = {}, {}, {}
    for entry in entries:
        name = entry.get('round_name') if isinstance(entry, dict) else None
        order = entry.get('channel_order') if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name:
            raise ValueError('each additional_round entry requires a round_name')
        if (not isinstance(order, list) or not order or
                any(not isinstance(c, dict) or not isinstance(c.get('channel'), str) or not isinstance(c.get('name'), str)
                    for c in order)):
            raise ValueError(f'additional round {name!r} requires a channel_order of channel and name entries')
        matches = [c['channel'] for c in order if stain in c['name']]
        if len(matches) != 1:
            raise ValueError(f'expected one channel of additional round {name!r} whose name contains ref_channel '
                             f'{stain!r}; found {len(matches)}')
        orders[name], stains[name] = tuple(c['channel'] for c in order), matches[0]
        folders[name] = tuple(c['name'] for c in order)
    rounds = RoundState(sequencing_rounds=[f'round{i}' for i in range(1, n + 1)], other_rounds=list(orders),
                        reference_round=config['ref_round'])
    rounds.validate()
    dataset = Dataset(Path(config['root_input_path']) / config['dataset_id'] / config['sample_id'],
        Path(config['root_output_path']) / config['dataset_id'] / config['output_id'],
        config['dataset_id'], config['sample_id'], config['output_id'], rounds=rounds,
        channel_order=tuple(config.get('seq_channel_order', ())), fov_pattern=config.get('fov_id_pattern', 'Position%03d'),
        other_channel_order=orders)
    return dataset, stains, folders


def _run_nuclei_registration(snakemake):
    """Python counterpart of nuclei_registration.m for the Python backend.

    Loads each additional round with its channel_order and rotates it by
    rotate_angle; reads the reference round's ch04 image, rotated the same
    way, as an ExternalReference; registers each round by one translation
    step on its shared stain and transfers the transform to all its
    channels. Writes the MATLAB names: log/<fov>_nr.txt (the attempts),
    log/gr_shifts/<fov>_nr.txt and images/<round>/<channel name>/<fov>.tif
    (ZYX, or the Z maximum YX with maximum_projection), keeping the dtype.
    Unlike MATLAB, it does not min-max stretch the other rounds.
    """
    from dataclasses import asdict
    import tifffile
    from starfinder.io import load_round, save_volume
    from starfinder.preprocessing import project_image
    from .fov import _rotated
    config = snakemake.config
    dataset, stains, folders = _nuclei_registration(config)
    fov = dataset.fov(snakemake.wildcards.fovID)
    angle, ref = config.get('rotate_angle'), dataset.rounds.reference_round
    for name in dataset.rounds.other_rounds:
        fov.load_images(rounds=[name], config=ImageLoadConfig(channel_labels=dataset.channel_labels(name)))
        if angle is not None:
            fov._rotate_round(round_name=name, angle=angle)
    loaded = load_round(fov.input_dir(ref), config=ImageLoadConfig(channel_labels=(_NUCLEI_REFERENCE_CHANNEL,)))
    image, metadata = loaded.image[..., 0], loaded.metadata
    if angle is not None:
        image, metadata, _ = _rotated(image, metadata, angle)
    reference = ExternalReference(image, metadata, f'{ref}:{_NUCLEI_REFERENCE_CHANNEL}')
    for name in dataset.rounds.other_rounds:
        signal = RegistrationSignalConfig('channel', _NUCLEI_REFERENCE_CHANNEL, stains[name])
        fov.register_rounds(RegistrationRecipe((RegistrationStep(TranslationConfig()),), signal=signal),
                            rounds=[name], reference=reference)
    fov.save_processing_log('nr')
    projection = ProjectionConfig() if config.get('maximum_projection', False) else None
    for name in dataset.rounds.other_rounds:
        # Channels that share a name write the same file, the last one winning, as in MATLAB.
        for c, folder in enumerate(folders[name]):
            path = dataset.output_root / 'images' / name / folder / f'{fov.fov_id}.tif'
            volume, metadata = fov.images[name][..., c], fov.metadata[name]
            if projection is None:
                save_volume(volume, path, metadata=metadata)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(path, project_image(volume, config=projection)[0], photometric='minisblack',
                                 metadata={'axes': 'YX',
                                           'starfinder_metadata': asdict(metadata.projected(method=projection.method))})
