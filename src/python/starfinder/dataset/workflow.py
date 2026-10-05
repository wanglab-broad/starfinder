"""Single translation boundary for shared MATLAB/Snakemake configuration."""
import json
from dataclasses import MISSING, asdict, dataclass, field, fields, replace
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
from starfinder.barcode import (DECODING_METHODS, ENCODINGS, BarcodeLayout, EncodingConfig, LocalBackgroundConfig,
    NeighborhoodSumConfig, OneBaseEncodingConfig, ReadFilterConfig, ReadScoreConfig, Segment, WtaDecoderConfig,
    DeduplicationConfig)
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
    """Encoding config of the Python-only load_codebook.encoding mapping (method default two_base).

    two_base takes reverse_bases and pair_to_color (the 16 ordered base pairs to
    colors 1-4; without it the default table), one_base base_to_color and
    reverse_bases. YAML integer colors are read as the color strings.
    """
    if not isinstance(values, dict):
        raise TypeError('load_codebook.encoding must be a mapping')
    values = dict(values)
    config_type = config_type_for(ENCODINGS, values.pop('method', 'two_base'), 'encoding')
    # split_index stays the shared load_codebook key; it is translated into the layout.
    _known(values, {f.name for f in fields(config_type) if f.init} - {'split_index'}, 'load_codebook.encoding')
    for key in ('base_to_color', 'pair_to_color'):
        if isinstance(values.get(key), dict):
            values[key] = {str(k): str(v) for k, v in values[key].items()}
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


def _extraction(values):
    """NeighborhoodSumConfig of the reads_extraction keys voxel_size and the Python-only background.

    background is false (off) or a mapping of LocalBackgroundConfig fields. When
    it is not given, the background is on with the default ring, whose inner and
    outer boxes grow by the same number of voxels along any axis where the
    extraction box is larger than the default inner box (so the inner box
    contains it and the ring keeps its width).
    """
    radius = tuple(values.get('voxel_size', (1, 2, 2)))
    background = values.get('background')
    if background is False:
        return NeighborhoodSumConfig(radius, background=None)
    if background is None:
        default = LocalBackgroundConfig()
        grow = tuple(max(r - i, 0) if isinstance(r, int) and not isinstance(r, bool) else 0
                     for r, i in zip(radius, default.inner_radius_zyx)) if len(radius) == 3 else (0, 0, 0)
        return NeighborhoodSumConfig(radius, background=LocalBackgroundConfig(
            tuple(i + g for i, g in zip(default.inner_radius_zyx, grow)),
            tuple(o + g for o, g in zip(default.outer_radius_zyx, grow))))
    if not isinstance(background, dict):
        raise TypeError('reads_extraction.background must be false or a mapping')
    names = {f.name for f in fields(LocalBackgroundConfig) if f.init}
    _known(background, names, 'reads_extraction.background')
    return NeighborhoodSumConfig(radius, background=LocalBackgroundConfig(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in background.items()}))


def _scoring(params, decoding, extraction):
    """ReadScoreConfig whenever the adapter decodes, unless the Python-only scoring block sets run false."""
    values = params.get('scoring', {})
    if not isinstance(values, dict):
        raise TypeError('scoring must be a mapping')
    _known(values, ('run', 'method'), 'scoring')
    run = values.get('run', True)
    if not isinstance(run, bool):
        raise ValueError('scoring.run must be Boolean')
    if values.get('method', 'bgcorr_probability') != 'bgcorr_probability':
        raise ValueError("scoring.method must be 'bgcorr_probability'")
    if 'scoring' in params and run and decoding is None:
        raise ValueError('scoring requires reads_filtration.run')
    if not run or decoding is None:
        return None
    if extraction is not None and extraction.background is None:
        raise ValueError('scoring needs the local background: remove reads_extraction.background: false '
                         'or set scoring: {run: false}')
    return ReadScoreConfig()


def _deduplication(params, decoding, mode):
    """DeduplicationConfig of the Python-only deduplication block; off unless it sets run true.

    The block holds run and the DeduplicationConfig fields (distance_voxels,
    compatibility). It needs reads_filtration.run and readout mode multiplexed.
    """
    if 'deduplication' not in params:
        return None
    values = params['deduplication']
    if not isinstance(values, dict):
        raise TypeError('deduplication must be a mapping')
    names = {f.name for f in fields(DeduplicationConfig) if f.init}
    _known(values, ('run', *names), 'deduplication')
    run = values.get('run', False)
    if not isinstance(run, bool):
        raise ValueError('deduplication.run must be Boolean')
    if not run:
        return None
    if mode == 'direct':
        raise ValueError("deduplication is not available with readout_mode direct")
    if decoding is None:
        raise ValueError('deduplication requires reads_filtration.run')
    return DeduplicationConfig(**{k: v for k, v in values.items() if k != 'run'})


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
    ENCODINGS method (default two_base) with its config fields (two_base:
    reverse_bases and pair_to_color, the default table when absent), and the
    Python-only decoding key a DECODING_METHODS method (default wta) with its
    config fields; the adapter decodes with diagnostics and without rescue
    unless that key sets them. Whenever it decodes, the adapter also scores
    (ReadScoreConfig) unless the Python-only scoring key sets run false; the
    Python-only deduplication key (run and the DeduplicationConfig fields)
    turns on deduplication, which is off otherwise and raises in direct mode; and
    extraction measures the local background unless the Python-only
    reads_extraction.background is false (or a mapping of LocalBackgroundConfig
    fields). The Python-only top-level readout_mode
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
        'spot_finding', 'reads_extraction', 'reads_filtration', 'decoding', 'scoring', 'deduplication',
        'create_subtiles', 'device'),
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
    extract, do_extract = _operation(params, 'reads_extraction', ('voxel_size', 'background'))
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
    extraction = _extraction(extract) if do_extract else None
    scoring = _scoring(params, decoding if do_filter else None, extraction)
    deduplication = _deduplication(params, decoding if do_filter else None, mode)
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
        extraction=extraction,
        decoding=decoding if do_filter else None, scoring=scoring, deduplication=deduplication,
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


# --- §2.9 segmentation and assignment (docs/segmentation-contract.md, docs/assignment-contract.md, ---------
# --- "Workflow configuration"). The rules of segmentation.smk and reads-assignment.smk run for both backends.

# segmentation_input_folder -> the role of the input file's one channel; any other folder is nuclear.
_SEGMENTATION_ROLES = {'overlay': 'composite', 'DAPI': 'nuclear', 'flamingo/enhanced_DAPI': 'nuclear'}
_STARDIST_KEYS = ('stardist_base_path', 'stardist_model_name', 'segmentation_input_folder', 'prob_thresh',
                  'nms_thresh', 'rescale', 'expand_labels', 'distance', 'target', 'device')
_READS_ASSIGNMENT_KEYS = ('expand_labels', 'dilation_distance')
# The label namespace run name of the legacy label file, which reads_assignment imports.
_LEGACY_SEGMENTATION_RUN = 'stardist_segmentation'
# Keys of a segmentation block run besides the method config's fields.
_SEGMENTATION_RUN_KEYS = ('name', 'target', 'inputs', 'seeds', 'projection', 'operations', 'method')
# Keys of the assignment block besides the AssignmentConfig fields: the FOV.assign arguments.
_ASSIGNMENT_CALL_KEYS = ('cells', 'nuclei', 'name', 'population', 'correspondence', 'checkpoints')


def _rule_parameters(config, rule):
    params = config.get('rules', {}).get(rule, {}).get('parameters', {})
    if not isinstance(params, dict):
        raise TypeError(f'rules.{rule}.parameters must be a mapping')
    return params


def _python_only(config, keys):
    """Raise ValueError naming the Python-only keys given without backend: python."""
    if keys and config.get('backend', 'matlab') != 'python':
        raise ValueError(f'{", ".join(keys)} {"is" if len(keys) == 1 else "are"} Python-only and need '
                         'backend: python')


def _flag(params, key, context):
    value = params.get(key, False)
    if not isinstance(value, bool):
        raise ValueError(f'{context}.{key} must be Boolean')
    return value


def _legacy_target(config):
    """(target, source) of the legacy label file: the Python-only target key, else cell for overlay, nucleus otherwise."""
    params = _rule_parameters(config, 'stardist_segmentation')
    if 'target' in params:
        _python_only(config, ['rules.stardist_segmentation.parameters.target'])
        if params['target'] not in ('nucleus', 'cell'):
            raise ValueError(f"rules.stardist_segmentation.parameters.target must be nucleus or cell; "
                             f"got {params['target']!r}")
        return params['target'], 'config'
    folder = params.get('segmentation_input_folder', 'overlay')
    return ('cell' if folder == 'overlay' else 'nucleus'), 'legacy_default'


def _label_namespace(config, fov_id, run=None):
    """The JSON identity list of a FOV ([dataset_id, sample_id, fov_id, subtile_id]), with a run name for labels."""
    ids = [config['dataset_id'], config['sample_id'], fov_id, None]
    return json.dumps(ids if run is None else [*ids, run], separators=(',', ':'))


def _declared_metadata(config, fov_id):
    """ImageMetadata of a legacy file without stored metadata: the configured voxel_size_z and voxel_size_xy (µm)."""
    from starfinder.image import ImageMetadata
    frame = f"declared:{config['dataset_id']}/{config['output_id']}/{fov_id}"
    z, xy = config.get('voxel_size_z'), config.get('voxel_size_xy')
    if z is None or xy is None:
        return ImageMetadata(frame)
    return ImageMetadata(frame, spacing_zyx=(float(z), float(xy), float(xy)), spatial_unit='micrometer')


@dataclass(frozen=True)
class _LegacySegmentation:
    """The segment call translated from rules.stardist_segmentation.parameters (see _segmentation)."""
    config: object
    target: str
    role: str
    operations: tuple = ()
    projection: object = None
    device: str = 'cpu'
    record: dict = field(default_factory=dict)


def _stored_thresholds(folder):
    """{'prob', 'nms'} of a StarDist model folder's thresholds.json, or None when the file is absent."""
    path = Path(folder) / 'thresholds.json'
    if not path.is_file():
        return None
    values = json.loads(path.read_text())
    return {'prob': float(values['prob']), 'nms': float(values['nms'])}


def _segmentation(config) -> _LegacySegmentation:
    """The stardist_segmentation rule's one segment call, from rules.stardist_segmentation.parameters.

    The translation table of docs/segmentation-contract.md ("Workflow
    configuration"): model_path = <stardist_base_path>/<stardist_model_name>, or
    the known model of that name when the base path is the weights cache's
    stardist folder; prob_thresh and nms_thresh pass as None (the stored
    thresholds, threshold_source stored) when they equal the model's
    thresholds.json, else as overrides; rescale true is StarDist scale 0.5 in Y
    and X on the input grid, false 1.0; expand_labels with distance is one planar
    pixel expand_labels operation; segmentation_input_folder gives the channel
    role (overlay: composite; DAPI, flamingo/enhanced_DAPI and any other
    folder: nuclear). The Python-only target (default cell for overlay,
    nucleus otherwise, recorded as legacy_default) and device keys need
    backend: python. The maximum projection that made a YX input (the overlay's
    maximum_projection or the top-level one) is recorded as the run's projection.
    rotate_angle and dapi_round are not read. The segmentation block is run by
    FOV.segment, not by this rule, so it raises here.
    """
    from starfinder.preprocessing import ProjectionConfig
    from starfinder.segmentation import KNOWN_MODELS, ExpandLabelsConfig, StarDistConfig
    from starfinder.spot_finding._weights import weights_directory
    if 'segmentation' in config:
        _python_only(config, ['segmentation'])
        raise ValueError('the segmentation block is run by FOV.segment (the §2.13 rule); the stardist_segmentation '
                         'rule translates rules.stardist_segmentation.parameters only: remove the block or the rule')
    params = _rule_parameters(config, 'stardist_segmentation')
    _known(params, _STARDIST_KEYS, 'stardist_segmentation parameter')
    _python_only(config, [f'rules.stardist_segmentation.parameters.{k}' for k in ('target', 'device') if k in params])
    base, name = params.get('stardist_base_path'), params.get('stardist_model_name')
    if not isinstance(base, str) or not base or not isinstance(name, str) or not name:
        raise ValueError('stardist_segmentation needs stardist_base_path and stardist_model_name')
    cache = weights_directory() / 'stardist'
    known = ('stardist', name) in KNOWN_MODELS and Path(base).expanduser().resolve() == cache.resolve()
    model = {'model': name} if known else {'model_path': str(Path(base) / name)}
    stored = _stored_thresholds(cache / name if known else Path(base).expanduser() / name)
    thresholds = {}
    for key, short in (('prob_thresh', 'prob'), ('nms_thresh', 'nms')):
        value = params.get(key)
        thresholds[key] = None if value is None or (stored is not None and float(value) == stored[short]) else value
    rescale, expand = _flag(params, 'rescale', 'stardist_segmentation'), _flag(params, 'expand_labels',
                                                                                'stardist_segmentation')
    operations = ()
    if expand:
        if 'distance' not in params:
            raise ValueError('rules.stardist_segmentation.parameters.expand_labels needs distance')
        operations = (ExpandLabelsConfig(params['distance'], 'pixel', 'planar'),)
    folder = params.get('segmentation_input_folder', 'overlay')
    target, target_source = _legacy_target(config)
    projected = bool(config.get('maximum_projection', False)) or (folder == 'overlay' and bool(
        _rule_parameters(config, 'create_nuclei_amplicon_overlay').get('maximum_projection', False)))
    stardist = StarDistConfig(scale=0.5 if rescale else 1.0, **model, **thresholds)
    record = {'source': 'rules.stardist_segmentation.parameters',
              'legacy': {k: params[k] for k in _STARDIST_KEYS if k in params},
              'model': 'known' if known else 'path', 'stored_thresholds': stored,
              'threshold_source': 'stored' if all(v is None for v in thresholds.values()) else 'override',
              'segmentation_input_folder': folder, 'target_source': target_source}
    return _LegacySegmentation(stardist, target, _SEGMENTATION_ROLES.get(folder, 'nuclear'), operations,
                               ProjectionConfig() if projected else None, params.get('device', 'cpu'), record)


def _read_stack(path):
    """A legacy TIFF as ZYX (a YX file gains a singleton Z) and whether the file was YX."""
    import tifffile
    image = tifffile.imread(path)
    return (image[None] if image.ndim == 2 else image), image.ndim == 2


def _run_nuclei_amplicon_overlay(snakemake):
    """create_nuclei_amplicon_overlay: composite_nuclei_amplicon of the DAPI and reference merged images.

    The composite of create_nuclei_amplicon_overlay.py bit for bit; with the
    rule's maximum_projection the output is its Z maximum (YX). YX inputs are
    read as one plane and give a YX output. Written as the script wrote it.
    """
    import tifffile
    from starfinder.preprocessing import ProjectionConfig, project_image
    from starfinder.segmentation import composite_nuclei_amplicon
    params = _rule_parameters(snakemake.config, 'create_nuclei_amplicon_overlay')
    projection = _flag(params, 'maximum_projection', 'create_nuclei_amplicon_overlay')
    nuclear, plane = _read_stack(snakemake.input['dapi_img'])
    amplicon, _ = _read_stack(snakemake.input['amplicon_img'])
    composite, _ = composite_nuclei_amplicon(nuclear, amplicon)
    if projection:
        composite = project_image(composite, config=ProjectionConfig())
    tifffile.imwrite(snakemake.output[0], composite[0] if projection or plane else composite)


def _run_enhance_dapi_with_flamingo(snakemake):
    """enhance_dapi_with_flamingo: enhance_with_flamingo of the DAPI and Flamingo images, as the script wrote it."""
    import tifffile
    from starfinder.segmentation import enhance_with_flamingo
    nuclear, plane = _read_stack(snakemake.input['dapi_img'])
    flamingo, _ = _read_stack(snakemake.input['flamingo_img'])
    enhanced, _ = enhance_with_flamingo(nuclear, flamingo)
    tifffile.imwrite(snakemake.output[0], enhanced[0] if plane else enhanced)


def _run_stardist_segmentation(snakemake):
    """stardist_segmentation: the translated run (_segmentation) on the rule's one input file.

    The input is read with load_volume; its grid is reference_grid_from_file,
    whose metadata is the file's own or, for a file without one, declared from
    voxel_size_z and voxel_size_xy (the projection of it for a YX file with a
    translated projection). The labels are segment's uint32 result after the
    translated operations, written to the rule's output in the file's
    dimensionality (zlib), with the run record beside it as {fovID}.json. There
    is no foreground gate: an image without objects gives an all-zero image
    (outcome empty).
    """
    import tifffile
    from starfinder.io import load_volume
    from starfinder.io._checkpoint import _jsonable
    from starfinder.segmentation import SegmentationInput, reference_grid_from_file, segment
    from starfinder.segmentation._import import _file_sha256
    from starfinder.segmentation._plan import _expand
    config, fov_id = snakemake.config, snakemake.wildcards.fovID
    legacy = _segmentation(config)
    path = Path(snakemake.input[0])
    loaded = load_volume(path)
    declared = loaded.diagnostics['metadata_source'] != 'stored'
    with tifffile.TiffFile(path) as tif:
        plane = len(tif.series[0].shape) == 2
    metadata = None
    if declared:
        metadata = _declared_metadata(config, fov_id)
        if plane and legacy.projection is not None:
            metadata = metadata.projected(method=legacy.projection.method)
    grid = reference_grid_from_file(path, metadata=metadata)
    source = {'path': str(path), 'segmentation_input_folder': legacy.record['segmentation_input_folder']}
    result = segment(SegmentationInput(loaded.image[..., None], grid, (legacy.role,), (source,)),
                     config=legacy.config, target=legacy.target, device=legacy.device,
                     label_namespace=_label_namespace(config, fov_id, _LEGACY_SEGMENTATION_RUN))
    for operation in legacy.operations:
        result = _expand(result, operation)
    output = Path(snakemake.output[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output, result.labels[0] if plane else result.labels, compression='zlib',
                     photometric='minisblack')
    record = dict(result.record, run=_LEGACY_SEGMENTATION_RUN,
                  input=dict(result.record['input'], path=str(path), file_sha256=_file_sha256(path),
                             projection=None if legacy.projection is None else asdict(legacy.projection)),
                  workflow=dict(legacy.record, metadata_source='declared' if declared else 'stored',
                                output=str(output), output_sha256=_file_sha256(output)))
    output.with_suffix('.json').write_text(json.dumps(_jsonable(record), indent=2, default=str))


def _input_channel(values):
    """InputChannel of one segmentation block input; prepare names an input function with its config fields."""
    from starfinder.segmentation import CompositeConfig, FlamingoEnhancementConfig, InputChannel
    if not isinstance(values, dict):
        raise TypeError('each segmentation input must be a mapping')
    values = dict(values)
    prepare = values.pop('prepare', None)
    if prepare is not None:
        functions = {'composite_nuclei_amplicon': CompositeConfig, 'enhance_with_flamingo': FlamingoEnhancementConfig}
        if not isinstance(prepare, dict) or prepare.get('function') not in functions:
            raise ValueError(f'prepare must be a mapping whose function is one of {sorted(functions)}')
        prepare = dict(prepare)
        prepare = _typed(functions[prepare.pop('function')], prepare, 'segmentation input prepare')
    _known(values, [f.name for f in fields(InputChannel) if f.init and f.name != 'prepare'], 'segmentation input')
    return InputChannel(**values, prepare=prepare)


def _segmentation_run(values):
    """SegmentationRun of one segmentation block run (docs/segmentation-contract.md, "Workflow configuration")."""
    from starfinder.preprocessing import ProjectionConfig
    from starfinder.segmentation import (SEGMENTATION_METHODS, ExpandLabelsConfig, LabelImportConfig, SegmentationRun,
                                         ZExtensionConfig)
    if not isinstance(values, dict) or 'method' not in values or 'name' not in values:
        raise ValueError('each segmentation run must be a mapping with a name and a method')
    values = dict(values)
    run = {key: values.pop(key) for key in _SEGMENTATION_RUN_KEYS if key in values}
    name, method = run['name'], run['method']
    if method == 'import':
        names = [f.name for f in fields(LabelImportConfig) if f.init and f.name != 'target']
        _known(values, names, f'segmentation run {name!r} (import)')
        config = LabelImportConfig(target=run.get('target'), **values)
    else:
        config_type = config_type_for(SEGMENTATION_METHODS, method, 'segmentation method')
        _known(values, [f.name for f in fields(config_type) if f.init], f'segmentation run {name!r} ({method})')
        missing = [f.name for f in fields(config_type)
                   if f.init and f.default is MISSING and f.default_factory is MISSING and f.name not in values]
        if missing:
            raise ValueError(f'segmentation run {name!r}: method {method!r} requires {", ".join(missing)}')
        config = config_type(**{k: _tuples(v) for k, v in values.items()})
    projection = run.get('projection')
    if projection is True:
        projection = ProjectionConfig()
    elif projection is False:
        projection = None
    elif projection is not None:
        projection = _typed(ProjectionConfig, projection, f'segmentation run {name!r} projection')
    operations = []
    for entry in run.get('operations') or []:
        types = {'expand_labels': ExpandLabelsConfig, 'extend_labels_through_z': ZExtensionConfig}
        if not isinstance(entry, dict) or entry.get('operation') not in types:
            raise ValueError(f'segmentation run {name!r}: each operation is a mapping whose operation is one of '
                             f'{sorted(types)}')
        entry = dict(entry)
        operations.append(_typed(types[entry.pop('operation')], entry, f'segmentation run {name!r} operation'))
    inputs = run.get('inputs') or []
    if not isinstance(inputs, list):
        raise TypeError(f'segmentation run {name!r}: inputs must be a list')
    return SegmentationRun(name, run.get('target'), tuple(map(_input_channel, inputs)), config, run.get('seeds'),
                           projection, tuple(operations))


def _segmentation_block(config):
    """(SegmentationPlan, device) of the Python-only top-level segmentation block.

    Each run has name, target, inputs (InputChannel fields; prepare names
    composite_nuclei_amplicon or enhance_with_flamingo with its config fields),
    seeds, projection (true or ProjectionConfig fields), operations
    (expand_labels or extend_labels_through_z with their config fields) and
    method, a SEGMENTATION_METHODS name whose config fields are the run's other
    keys, or import with the LabelImportConfig fields. device is cpu (default)
    or cuda. The block needs backend: python; the §2.13 rule runs it with
    FOV.segment (docs/segmentation-contract.md, "Workflow configuration").
    """
    from starfinder.segmentation import SegmentationPlan
    _python_only(config, ['segmentation'])
    values = config['segmentation']
    if not isinstance(values, dict):
        raise TypeError('segmentation must be a mapping')
    _known(values, ('device', 'runs'), 'segmentation')
    runs = values.get('runs')
    if not isinstance(runs, list) or not runs:
        raise ValueError('segmentation.runs must be a nonempty list')
    device = values.get('device', 'cpu')
    if device not in ('cpu', 'cuda'):
        raise ValueError(f"segmentation.device must be 'cpu' or 'cuda'; got {device!r}")
    return SegmentationPlan(tuple(map(_segmentation_run, runs))), device


def _assignment_block(config):
    """(AssignmentConfig, FOV.assign keyword arguments) of the Python-only top-level assignment block.

    Its keys are the AssignmentConfig fields (expansion and correspondence as
    mappings of ExpandLabelsConfig and CorrespondenceConfig fields) and the
    FOV.assign arguments cells, nuclei, name, population and checkpoints (a
    mapping of CheckpointConfig fields); a supplied correspondence table is not
    a YAML value, so correspondence configures the overlap rule. The block needs
    backend: python; the §2.13 rule runs it with FOV.assign
    (docs/assignment-contract.md, "Workflow configuration").
    """
    from starfinder.assignment import AssignmentConfig, CorrespondenceConfig
    from starfinder.segmentation import ExpandLabelsConfig
    from .config import CheckpointConfig
    _python_only(config, ['assignment'])
    values = config['assignment']
    if not isinstance(values, dict):
        raise TypeError('assignment must be a mapping')
    names = [f.name for f in fields(AssignmentConfig) if f.init]
    _known(values, {*names, *_ASSIGNMENT_CALL_KEYS}, 'assignment')
    options = {}
    if values.get('expansion') is not None:
        options['expansion'] = _typed(ExpandLabelsConfig, values['expansion'], 'assignment.expansion')
    if values.get('correspondence') is not None:
        options['correspondence'] = _typed(CorrespondenceConfig, values['correspondence'],
                                           'assignment.correspondence')
    for key in ('legacy_pixel_expansion', 'exclude_cells_without_nucleus'):
        if key in values:
            options[key] = values[key]
    call = {key: values[key] for key in ('cells', 'nuclei', 'name', 'population') if key in values}
    if values.get('checkpoints') is not None:
        call['checkpoints'] = _typed(CheckpointConfig, values['checkpoints'], 'assignment.checkpoints')
    return AssignmentConfig(**options), call


def _reads_assignment(config):
    """(AssignmentConfig, cell target, record) of the reads_assignment rule, from its legacy keys.

    The translation table of docs/assignment-contract.md ("Workflow
    configuration"): expand_labels false is no expansion (dilation_distance is
    ignored and recorded); true with dilation_distance d is
    ExpandLabelsConfig(d, pixel, planar) with legacy_pixel_expansion=True, so
    assign expands once and keeps both masks. A label file expanded by
    stardist_segmentation (its expand_labels true) keeps no original mask and
    raises ValueError naming that key, and the assignment key when both are
    true. The label file's target is the segmentation adapter's
    (_legacy_target). The assignment block is run by FOV.assign, not by this
    rule, so it raises here.
    """
    from starfinder.assignment import AssignmentConfig
    from starfinder.segmentation import ExpandLabelsConfig
    if 'assignment' in config:
        _python_only(config, ['assignment'])
        raise ValueError('the assignment block is run by FOV.assign (the §2.13 rule); the reads_assignment rule '
                         'translates rules.reads_assignment.parameters only: remove the block or the rule')
    params = _rule_parameters(config, 'reads_assignment')
    _known(params, _READS_ASSIGNMENT_KEYS, 'reads_assignment parameter')
    expand = _flag(params, 'expand_labels', 'reads_assignment')
    if _flag(_rule_parameters(config, 'stardist_segmentation'), 'expand_labels', 'stardist_segmentation'):
        keys = ['rules.stardist_segmentation.parameters.expand_labels']
        if expand:
            keys.append('rules.reads_assignment.parameters.expand_labels')
        raise ValueError(f'{" and ".join(keys)} {"are" if len(keys) > 1 else "is"} true, but a label image is '
                         'expanded once, by assign, which keeps the original mask: set '
                         'rules.stardist_segmentation.parameters.expand_labels to false and give the distance as '
                         'rules.reads_assignment.parameters.dilation_distance with expand_labels: true')
    expansion = None
    if expand:
        if 'dilation_distance' not in params:
            raise ValueError('rules.reads_assignment.parameters.expand_labels needs dilation_distance')
        expansion = ExpandLabelsConfig(params['dilation_distance'], 'pixel', 'planar')
    target, target_source = _legacy_target(config)
    record = {'source': 'rules.reads_assignment.parameters', 'legacy': dict(params),
              'dilation_distance_ignored': not expand and 'dilation_distance' in params,
              'target': target, 'target_source': target_source}
    return AssignmentConfig(expansion=expansion, legacy_pixel_expansion=expand), target, record


def _fov_sample(annotation, config, fov_id):
    """(sample, FOV number) of fov_id: the sample-annotation row whose fov_start..fov_end holds it."""
    import pandas as pd
    table = pd.read_csv(annotation)
    for row in table.itertuples():
        for i in range(int(row.fov_start), int(row.fov_end) + 1):
            if config['fov_id_pattern'].format(i=i) == fov_id:
                return row.sample_id, i
    raise ValueError(f'{fov_id} is in no sample of {annotation}')


def _tile_record(path, number):
    """The FOV's row of output/tile_config_<sample>.csv (offsets x, y, z and the start/end_*_norm box)."""
    import pandas as pd
    table = pd.read_csv(path, index_col=0)
    rows = table[table['id'] == int(number)]
    if rows.empty:
        raise ValueError(f'{path} has no row with id {number}')
    return rows.iloc[0]


def _codebook_genes(path, mode):
    """The codebook's genes in first-appearance order: the gene (gene_id) column, or a direct panel's gene_id."""
    import csv
    with open(path, newline='', encoding='utf-8-sig') as handle:
        rows = [row for row in csv.reader(handle) if row]
    header = [v.strip() for v in rows[0]] if rows else []
    if mode == 'direct':
        if header != ['round', 'channel', 'gene_id']:
            raise ValueError(f'{path}: the direct panel header must be round,channel,gene_id')
        column = 2
    elif header and header[0] in ('gene', 'gene_id', 'entry_id'):
        column = header.index('gene_id') if 'gene_id' in header else 0
    else:
        column, header = 0, None
    body = rows[1:] if header else rows
    return list(dict.fromkeys(row[column].strip() for row in body))


def _assignment_genes(genes_csv, config):
    """The gene list of the count matrix: documents/genes.csv, checked against the codebook's genes.

    The codebook is the sequencing rules' input genes.csv (a direct panel in
    readout_mode direct). A gene on one list only raises ValueError naming it.
    The order is that of documents/genes.csv, the legacy matrix's columns.
    """
    import pandas as pd
    codebook = Path(config['root_input_path']) / config['dataset_id'] / config['sample_id'] / 'genes.csv'
    listed = list(dict.fromkeys(pd.read_csv(genes_csv, header=None, dtype=str)[0].str.strip()))
    known = _codebook_genes(codebook, config.get('readout_mode', 'multiplexed'))
    only_listed, only_known = [g for g in listed if g not in known], [g for g in known if g not in listed]
    if only_listed or only_known:
        raise ValueError(f'{genes_csv} and the codebook {codebook} list different genes: only in genes.csv '
                         f'{only_listed}, only in the codebook {only_known}')
    return tuple(listed)


def _legacy_cells(path, config, fov_id, target):
    """(molecule grid, cell run) of a legacy label file, imported on a grid declared from the configuration.

    A ZYX file is on the grid of its shape; a YX file is a plane on the
    projection of the img_z × Y × X grid. The metadata is _declared_metadata.
    """
    import tifffile
    from starfinder.segmentation import ReferenceGrid, import_labels
    with tifffile.TiffFile(path) as tif:
        shape = tuple(int(n) for n in tif.series[0].shape)
    metadata = _declared_metadata(config, fov_id)
    if len(shape) == 2:
        if 'img_z' not in config:
            raise ValueError(f'{path} is a YX label image; img_z gives the Z size of the molecule grid')
        grid = ReferenceGrid((int(config['img_z']), *shape), metadata, 'declared')
        label_grid = grid.projected(method='max') if grid.shape_zyx[0] > 1 else grid
    else:
        grid = label_grid = ReferenceGrid(shape, metadata, 'declared')
    cells = import_labels(path, grid=label_grid, target=target,
                          label_namespace=_label_namespace(config, fov_id, _LEGACY_SEGMENTATION_RUN))
    return grid, cells


def _in_box(x, y, tile):
    """Whether integer positions lie in the tile's start/end_x_norm, start/end_y_norm box (legacy range test)."""
    return ((x >= tile['start_x_norm']) & (x < tile['end_x_norm']) & (y >= tile['start_y_norm'])
            & (y < tile['end_y_norm']))


def _cell_obs(result, rows, *, sample, fov_id, tile):
    """The raw.h5ad obs of the kept cells ``rows``: the legacy columns, then the cell table's.

    Legacy meaning, computed on the territories assign samples: volume is
    expanded_size_voxels (size_voxels without expansion), fov_x/y/z the
    truncated (expanded_)centroid, seg_label the cell_id, global_* fov_* plus
    the tile offsets; plane cells have no fov_z and global_z.
    """
    import numpy as np
    import pandas as pd
    expanded = result.territories is not None
    prefix = 'expanded_' if expanded else ''
    plane = result.cell_labels.shape[0] == 1
    n = len(rows)
    columns = {'sample': np.array([sample] * n, dtype=object), 'fov_id': np.array([fov_id] * n, dtype=object),
               'volume': rows[f'{prefix}size_voxels'].to_numpy(np.float64)}
    axes = ('x', 'y') if plane else ('x', 'y', 'z')
    for axis in axes:
        columns[f'fov_{axis}'] = rows[f'{prefix}centroid_{axis}'].to_numpy(np.float64).astype(np.int64)
    columns['seg_label'] = rows.cell_id.to_numpy(np.int64)
    for axis in axes:
        columns[f'global_{axis}'] = columns[f'fov_{axis}'] + np.int64(tile[axis])
    columns.update(size_voxels=rows.size_voxels.to_numpy(np.int64),
                   expanded_size_voxels=pd.array(rows.expanded_size_voxels, dtype='Int64'),
                   size_physical=rows.size_physical.to_numpy(np.float64))
    for axis in ('z', 'y', 'x'):
        columns[f'centroid_{axis}'] = rows[f'centroid_{axis}'].to_numpy(np.float64)
    columns.update(n_molecules=rows.n_molecules.to_numpy(np.int64), n_nuclei=pd.array(rows.n_nuclei, dtype='Int64'))
    for name in ('correspondence', 'correspondence_flags', 'compartments'):
        # Categorical, as anndata stores strings; an all-null column (no nuclei) has no categories.
        columns[name] = pd.Categorical(np.array([None if pd.isna(v) else str(v) for v in rows[name]], dtype=object))
    return pd.DataFrame(columns, index=pd.Index([str(i) for i in range(n)], dtype=object))


def _write_raw_h5ad(result, path, *, sample, fov_id, tile):
    """expr/{fovID}/raw.h5ad from an AssignmentResult: the kept cells inside the tile box.

    X is the float64 whole-cell counts in result.genes order and obs the
    legacy and cell-table columns (_cell_obs); the legacy overlap filter keeps
    the cells whose fov_x and fov_y lie in the tile box. With nuclei, the
    float64 layers nucleus and cytoplasm hold the compartment counts, 0.0 for
    a measured zero and NaN for every gene of a cell whose compartments are not
    available. uns["assignment"] is the record as JSON text. Returns the
    seg_label values of the cells written.
    """
    import anndata
    import numpy as np
    import pandas as pd
    from starfinder.io._checkpoint import _jsonable
    matrix, rows = result.matrix()
    layers = {}
    if result.nucleus_labels is not None:
        row_of = {int(c): i for i, c in enumerate(rows.cell_id)}
        for compartment in ('nucleus', 'cytoplasm'):
            values, available = result.matrix(compartment)
            layers[compartment] = np.full(matrix.shape, np.nan)
            layers[compartment][[row_of[int(c)] for c in available.cell_id], :] = values
    # Object strings throughout: anndata 0.12 writes no pandas Arrow string arrays, which pandas 3 infers.
    with pd.option_context('future.infer_string', False):
        obs = _cell_obs(result, rows, sample=sample, fov_id=fov_id, tile=tile)
        keep = _in_box(obs.fov_x.to_numpy(), obs.fov_y.to_numpy(), tile)
        layers = {name: layer[keep] for name, layer in layers.items()}
        obs = obs[keep]
        obs.index = pd.Index([str(i) for i in range(len(obs))], dtype=object)
        adata = anndata.AnnData(X=matrix[keep].astype(np.float64), obs=obs,
                                var=pd.DataFrame(index=pd.Index(list(result.genes), dtype=object)), layers=layers)
        adata.uns['assignment'] = json.dumps(_jsonable(result.record), default=str)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        adata.write(path)
    return obs.seg_label.to_numpy()


def _write_reads_assignment(result, reads_csv, path, *, tile, cells):
    """expr/{fovID}/reads_assignment.csv: the goodSpots rows with the legacy and the assignment columns.

    The legacy columns are those of reads_assignment.py (the CSV's columns with
    zero-based x, y, z, global_x/y/z and seg_label, the cell at the sampled
    voxel or 0); the new ones are spot_id, assignment_status, cell_id,
    in_expansion, original_cell_id, nucleus_id and compartment. The legacy
    overlap filter keeps the molecules assigned to the written cells, then the
    unassigned and excluded_cell molecules whose sampled voxel lies in the tile
    box. Every outside_grid molecule follows them, in input order, unfiltered
    (it has no sampled voxel, and the script wrapped or raised on it): seg_label
    0, null cell columns and coordinates computed as for the other rows.
    """
    import numpy as np
    import pandas as pd
    reads = pd.read_csv(reads_csv)
    for axis in ('x', 'y', 'z'):
        reads[axis] = reads[axis] - 1
        reads[f'global_{axis}'] = reads[axis] + tile[axis]
    molecules = result.molecules
    reads['seg_label'] = molecules.cell_id.astype('Int64').fillna(0).to_numpy(np.int64)
    for column in ('spot_id', 'assignment_status', 'cell_id', 'in_expansion', 'original_cell_id', 'nucleus_id',
                   'compartment'):
        reads[column] = molecules[column].to_numpy(dtype=object, na_value=None)
    status = molecules.assignment_status.to_numpy(dtype=object)
    assigned = (status == 'assigned') & np.isin(reads.seg_label.to_numpy(), cells)
    x = molecules.voxel_x.astype('Int64').fillna(-1).to_numpy(np.int64)
    y = molecules.voxel_y.astype('Int64').fillna(-1).to_numpy(np.int64)
    off_grid = status == 'outside_grid'
    background = (status != 'assigned') & ~off_grid & (x >= 0) & (y >= 0) & _in_box(x, y, tile)
    pd.concat([reads[assigned], reads[background], reads[off_grid]]).to_csv(path, index=False)


def _run_reads_assignment(snakemake):
    """reads_assignment: assign_molecules on the legacy label file, written as raw.h5ad and reads_assignment.csv.

    Inputs (as the rule declares them): the sample annotation, the DAPI image
    (diagnostic plot only), images/stardist_segmentation/{fovID}.tif (imported
    with the inferred target on the declared grid, _legacy_cells), the
    goodSpots CSV (molecule_table_from_csv; integer and float coordinates),
    documents/genes.csv (checked against the codebook, _assignment_genes) and
    the tile configuration, read for the sample of the FOV and applied outside
    the package: global coordinates and the overlap filter, as the script does
    (§2.10 boundary); outside_grid molecules are kept in reads_assignment.csv.
    Also writes expr/{fovID}/assignment.png (plot_assignment) and log.txt.
    """
    import matplotlib.pyplot as plt
    import tifffile
    from starfinder.assignment import assign_molecules, molecule_table_from_csv, plot_assignment
    config, fov_id = snakemake.config, snakemake.wildcards.fovID
    assignment, target, translation = _reads_assignment(config)
    sample, number = _fov_sample(snakemake.input[0], config, fov_id)
    output_root = Path(config['root_output_path']) / config['dataset_id'] / config['output_id']
    tile = _tile_record(output_root / 'output' / f'tile_config_{sample}.csv', number)
    genes = _assignment_genes(snakemake.input[4], config)
    grid, cells = _legacy_cells(Path(snakemake.input[2]), config, fov_id, target)
    molecules = molecule_table_from_csv(snakemake.input[3], spot_namespace=_label_namespace(config, fov_id),
                                        genes=genes)
    result = assign_molecules(molecules, cells, grid=grid, config=assignment)
    result = replace(result, record=dict(result.record, name='reads_assignment', workflow=translation))
    h5ad, csv_path = Path(snakemake.output[0]), Path(snakemake.output[1])
    written = _write_raw_h5ad(result, h5ad, sample=sample, fov_id=fov_id, tile=tile)
    _write_reads_assignment(result, snakemake.input[3], csv_path, tile=tile, cells=written)
    image = tifffile.imread(snakemake.input[1])
    image = image if image.shape[-2:] == result.cell_labels.shape[1:] else None
    figure = plot_assignment(result, image=image)
    figure.savefig(h5ad.parent / 'assignment.png')
    plt.close(figure)
    counts = result.record['counts']
    (h5ad.parent / 'log.txt').write_text(
        f"{counts['assigned']} of {counts['molecules']} molecules assigned to {counts['cells_kept']} cells; "
        f"{counts['unassigned']} unassigned, {counts['excluded_cell']} in excluded cells, "
        f"{counts['outside_grid']} outside the grid; {len(written)} cells inside the tile box\n")
