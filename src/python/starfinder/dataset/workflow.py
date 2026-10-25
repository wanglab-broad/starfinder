"""Single translation boundary for shared MATLAB/Snakemake configuration."""
import json
from dataclasses import MISSING, dataclass, field, fields, replace
from pathlib import Path

from .config import PipelineConfig, ExecutionConfig
from ._declared import reference_settings
from .dataset import Dataset
from .types import ChannelInfo, RoundState, SubtileConfig
from starfinder.io import ImageLoadConfig
from starfinder.preprocessing import ProjectionConfig, PreprocessingRecipe, PreprocessingStep, step_config_type
from starfinder import _layout
from starfinder._registry import config_type_for
from starfinder.registration import (REGISTRATION_METHODS, InsufficientLandmarksError, RecoveryConfig,
    RegistrationEstimationError, RegistrationQcConfig, RegistrationRecipe, RegistrationRejectedError,
    RegistrationSignalConfig, RegistrationStep, WarpConfig)
from starfinder.readout import (DECODING_METHODS, ENCODINGS, BarcodeLayout, EncodingConfig, LocalBackgroundConfig,
    NeighborhoodSumConfig, OneBaseEncodingConfig, ReadFilterConfig, ReadScoreConfig, Segment, WtaDecoderConfig,
    DeduplicationConfig)
from starfinder.readout.decoding import READOUT_MODES, _mode_mismatch
from starfinder.spot_finding import SPOT_FINDING_METHODS, ChannelOverride, LocalMaximaConfig, SpotFindingPlan

_RULES = ('rsf_single_fov', 'gr_single_fov_subtile', 'lrsf_single_fov_subtile',
          'deep_create_subtile', 'deep_rsf_subtile')
_RESIDENT_RULES = ('lrsf_single_fov_subtile', 'deep_rsf_subtile')
# The MATLAB keys of the Python processing rules and their native forms. The Python backend accepts the
# native keys only and raises naming the form; the MATLAB backend's configurations are translated by
# _legacy_config (docs/migration.md, "One Python configuration dialect").
_NATIVE_FORMS = {
    'enhance_contrast': 'a preprocessing.steps entry {method: min_max_normalization, output_dtype: uint8, '
                        'output_range: [0, 255], rounding: truncate}',
    'hist_equalize': 'a preprocessing.steps entry {method: histogram_matching, reference_channel: 0}',
    'morph_recon': 'a preprocessing.steps entry {method: reconstruction, radius_yx: 3} (in the resident subtile '
                   'rules a preprocessing.post_registration entry)',
    'tophat': 'a preprocessing.steps entry {method: white_tophat, radius_yx: 3}',
    'snr_threshold': 'the snr_threshold of the preprocessing.steps entry {method: min_max_normalization}',
    'global_registration': 'a registration.steps entry of a global method (default translation), with the '
                           'recipe signal and warp',
    'local_registration': 'a registration.steps entry of a local method (default demons), with the recipe signal '
                          'and warp',
}
# The MATLAB spot_finding keys of local_maxima and the LocalMaximaConfig fields they name.
_NATIVE_SPOT_FIELDS = {'intensity_estimation': 'threshold_mode', 'intensity_threshold': 'threshold_value',
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


def _spot_fields(values, config_type, context, aliases=None):
    """Init fields of config_type from one spot_finding mapping (YAML lists become tuples).

    A key must be an init field of its config (min_distance is the native
    field of spotiflow and piscis, passed unchanged). For local_maxima the
    MATLAB keys intensity_estimation, intensity_threshold and min_distance
    raise naming their LocalMaximaConfig fields; aliases (the MATLAB backend's
    translation, _legacy_config) maps them to those fields instead, and an
    alias raises together with its field. Unknown keys raise ValueError.
    """
    if not isinstance(values, dict):
        raise TypeError(f'{context} must be a mapping')
    names = {f.name for f in fields(config_type) if f.init}
    if config_type is LocalMaximaConfig and aliases is None:
        given = [key for key in _NATIVE_SPOT_FIELDS if key in values]
        if given:
            keys = 'is a MATLAB key' if len(given) == 1 else 'are MATLAB keys'
            raise ValueError(f'{context}: {", ".join(given)} {keys} of local_maxima; the Python backend names the '
                             f'LocalMaximaConfig fields {", ".join(_NATIVE_SPOT_FIELDS[key] for key in given)}')
    aliases = aliases if aliases is not None and config_type is LocalMaximaConfig else {}
    _known(values, names | set(aliases), context)
    for key, field_name in aliases.items():
        if key in values and field_name in values:
            raise ValueError(f'{context}: {key} and {field_name} set the same field; give one of them')
    return {aliases.get(key, key): _tuples(value) for key, value in values.items()}


def _detection(values, channels, aliases=None):
    """The detection config of the spot_finding block, or a SpotFindingPlan when it has channel_overrides or rounds.

    method names a SPOT_FINDING_METHODS method with pipeline=True (default
    local_maxima); the other keys are its config's fields (_spot_fields, with
    the MATLAB backend's aliases),
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
    values = _spot_fields(values, config_type, 'spot_finding', aliases)
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
        ChannelOverride(label, replace(config, **_spot_fields(
            entry, config_type, f'spot_finding.channel_overrides.{label}', aliases)))
        for label, entry in overrides.items()), None if rounds is None else tuple(rounds))


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


def _preprocessing_steps(entries, context):
    """PreprocessingSteps of one list of the preprocessing key: each a method name, its config fields and save_as."""
    if not isinstance(entries, list):
        raise ValueError(f'{context} must be a list')
    steps = []
    for entry in entries:
        if not isinstance(entry, dict) or 'method' not in entry:
            raise ValueError('each preprocessing step must be a mapping with a method')
        entry = dict(entry)
        config_type = step_config_type(entry.pop('method'))
        save_as = entry.pop('save_as', None)
        _known(entry, [f.name for f in fields(config_type) if f.init], f'preprocessing step {config_type.__name__}')
        steps.append(PreprocessingStep(config_type(**{k: _tuples(v) for k, v in entry.items()}), save_as))
    return tuple(steps)


def _explicit_recipe(values):
    """Recipe from the preprocessing key: steps named by their PREPROCESSING_METHODS names, plus the recipe fields.

    A step's keys other than method and save_as are the fields of its config
    dataclass; YAML lists become tuples. post_registration (absent: none)
    lists the steps run after registration, in the same form
    (PreprocessingRecipe.post_registration: reconstruction only).
    """
    if not isinstance(values, dict):
        raise TypeError('preprocessing must be a mapping')
    _known(values, ('steps', 'post_registration', 'extraction_source', 'registration_source', 'supplied_statistics'),
           'preprocessing')
    steps = _preprocessing_steps(values.get('steps'), 'preprocessing.steps')
    post = _preprocessing_steps(values.get('post_registration', []), 'preprocessing.post_registration')
    supplied = values.get('supplied_statistics')
    return PreprocessingRecipe(steps, post, extraction_source=values.get('extraction_source'),
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
    FOV.save_reference_image, which writes the file of the dataset's
    reference_image (default ``fovs/{fov_id}/reference.tif``); reference_image
    and reference_channel are the rule's reference_image_view (``merged`` and 0
    for mode max, ``single-channel`` and the channel's index for mode channel). split_index is the
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
    With backend: python the rule takes one dialect, the native keys: the
    preprocessing key declares an explicit recipe (steps, and post_registration
    for the steps run after registration), the registration key a
    RegistrationRecipe, and the spot_finding block's method key names a
    pipeline SPOT_FINDING_METHODS method (default local_maxima) whose config
    fields are the other keys, channel_overrides gives per-channel settings and
    rounds the rounds to detect in. A MATLAB key of these rules
    (enhance_contrast, hist_equalize, morph_recon, tophat, snr_threshold,
    global_registration, local_registration, and local_maxima's
    intensity_estimation, intensity_threshold and min_distance) raises
    ValueError naming its native form (_NATIVE_FORMS). The rule parameter
    reference_image_view ({mode: max | channel, channel: index or label},
    default max) selects the reference image FOV.save_reference_image writes.
    A MATLAB-backend configuration (backend absent or matlab), which the
    adapters of the shared rules read, is translated by _legacy_config. The rule-level Python-only device key sets
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
    direct mode decoding is the direct method (DirectDecoderConfig), the
    rule's codebook input is the panel CSV (round,channel,gene_id), the
    candidates need spot_finding.rounds, and the barcode keys
    load_codebook.split_index and encoding and reads_filtration.end_base,
    split_index, n_barcode_segments and exclude_invalid_endpoints raise.
    The Python-only checkpoints key ({stages: [...]}) is checked here and read
    by the rule's runner (_rule_checkpoints); the top-level config_path is kept
    on the Dataset, whose run.json records it as workflow_config. img_row and
    img_col are read only for the subtile windows of create_subtiles.
    Direct Python callers construct Dataset/PipelineConfig (no legacy aliases).
    """
    if rule not in _RULES:
        raise ValueError(f'unsupported Python workflow rule {rule}')
    if set(config) & {'channel_order', 'fov_pattern'}:
        raise ValueError('direct Python field names belong to Dataset; use seq_channel_order here (the FOVs are those '
                         'of fovs.csv)')
    params = config.get('rules', {}).get(rule, {}).get('parameters', {})
    python = config.get('backend', 'matlab') == 'python'
    mode = _readout_mode(config, params)
    dataset = _workflow_dataset(config, readout_mode=mode)
    rounds, channels = dataset.rounds, dataset.channel_order
    _known(params, ('streaming', 'load_codebook', 'load_raw_images', 'preprocessing', 'registration',
        'reference_image_view', 'spot_finding', 'reads_extraction', 'reads_filtration', 'decoding', 'scoring',
        'deduplication', 'create_subtiles', 'device', _layout.CHECKPOINTS, *_NATIVE_FORMS),
        'Python workflow parameter')
    _rule_checkpoints(config, rule)
    if config.get('config_path') is not None:
        # run.json records the configuration file a rule ran with (workflow_config).
        dataset._workflow_config = Path(config['config_path']).absolute()
    if python:
        replaced = [key for key in _NATIVE_FORMS if key in params]
        if replaced:
            raise ValueError(f'rules.{rule}.parameters: ' + '; '.join(
                f'{key} is a MATLAB key, which the Python backend replaces by {_NATIVE_FORMS[key]}' for key in replaced)
                + ' (docs/migration.md, "One Python configuration dialect")')
        preprocessing = _explicit_recipe(params['preprocessing']) if 'preprocessing' in params else None
        registration = _explicit_registration(params['registration']) if 'registration' in params else None
        reference_view = _reference_image_view(params, dataset, rule)
        spot_aliases = None
    else:
        _python_only(config, [f'rules.{rule}.parameters.reference_image_view'] if 'reference_image_view' in params
                     else [])
        from ._legacy_config import legacy_processing
        preprocessing, registration, reference_view, spot_aliases = legacy_processing(params, rule,
                                                                                      rounds.reference_round)
    spot_keys = {f.name for cls, spec in SPOT_FINDING_METHODS.items() if spec.pipeline for f in fields(cls) if f.init}
    spot, do_spot = _operation(params, 'spot_finding', {'ref_round', 'method', 'channel_overrides', 'rounds',
                                                        *_NATIVE_SPOT_FIELDS, *spot_keys})
    extract, do_extract = _operation(params, 'reads_extraction', ('voxel_size', 'background'))
    filt, do_filter = _operation(params, 'reads_filtration', ('end_base', 'start_base', 'exclude_invalid_endpoints', 'score_bounds', 'n_barcode_segments', 'split_index'))
    book, _ = _operation(params, 'load_codebook', ('split_index', 'encoding'))
    if mode == 'direct':
        encoding, layout, split_index, end_base = EncodingConfig(), None, None, None
    else:
        encoding, layout, split_index, end_base = _codebook_layout(book, filt, len(rounds.sequencing_rounds))
    if 'decoding' in params and not do_filter:
        raise ValueError('decoding requires reads_filtration.run')
    decoding = (_decoding(params['decoding'], mode) if 'decoding' in params else
                WtaDecoderConfig(diagnostics=True) if mode == 'multiplexed' else _decoding({}, mode))
    extraction = _extraction(extract) if do_extract else None
    scoring = _scoring(params, decoding if do_filter else None, extraction)
    deduplication = _deduplication(params, decoding if do_filter else None, mode)
    load, do_load = _operation(params, 'load_raw_images', ('subdir',))
    if registration is not None and registration.reference_round not in (None, rounds.reference_round):
        raise ValueError('registration reference round differs from dataset reference')
    if spot.get('ref_round', rounds.reference_round) != rounds.reference_round:
        raise ValueError('detection reference differs from dataset reference')
    resident = rule in _RESIDENT_RULES
    # The adapter loads raw input unless this is a saved-subtile job or explicitly disabled.
    load_config = ImageLoadConfig(channel_labels=channels, **load) if channels and not resident and ('load_raw_images' not in params or do_load) else None
    pipeline = PipelineConfig(load=load_config, rotation_degrees=None if resident else config.get('rotate_angle'),
        preprocessing=preprocessing,
        registration=registration,
        spot_finding=_detection(spot, channels, spot_aliases) if do_spot else None,
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
            dataset.subtile.compute_windows(*_image_shape(config, f'rules.{subtile_rule}.parameters.create_subtiles'))
            break
    streaming = params.get('streaming', False)
    if not isinstance(streaming, bool):
        raise ValueError('streaming must be Boolean')
    return WorkflowConfig(dataset, pipeline,
        ExecutionConfig('streaming' if streaming else 'batch', rule in ('gr_single_fov_subtile', 'deep_create_subtile'),
                        params.get('device', 'cpu')),
        split_index, ProjectionConfig() if config.get('maximum_projection', False) else None, *reference_view,
        encoding=encoding, layout=layout)


def _reference_image_view(params, dataset, rule):
    """(reference_image, reference_channel) of FOV.save_reference_image from the rule parameter reference_image_view.

    {mode: max} (the default, also when the key is absent) is the channel
    maximum; {mode: channel, channel: c} is one channel of the reference
    round, c a zero-based index or a channel pattern or name
    (Dataset.channel_index). The resident subtile rules write no reference
    image, so the key raises there.
    """
    context = f'rules.{rule}.parameters.reference_image_view'
    values = params.get('reference_image_view')
    if values is None:
        return 'merged', 0
    if not isinstance(values, dict):
        raise TypeError(f'{context} must be a mapping {{mode, channel}}')
    _known(values, ('mode', 'channel'), context)
    if rule in _RESIDENT_RULES:
        raise ValueError(f'{context}: the resident subtile rule {rule} writes no reference image')
    view = values.get('mode', 'max')
    if view == 'max':
        if values.get('channel') is not None:
            raise ValueError(f'{context}.channel applies to mode channel only')
        return 'merged', 0
    if view != 'channel':
        raise ValueError(f"{context}.mode must be 'max' or 'channel'; got {view!r}")
    channel = values.get('channel')
    if isinstance(channel, bool) or not isinstance(channel, (int, str)):
        raise ValueError(f'{context}: mode channel needs channel, a zero-based index or a channel label; got '
                         f'{channel!r}')
    if isinstance(channel, int):
        if channel < 0:
            raise ValueError(f'{context}.channel must be a zero-based index; got {channel}')
        return 'single-channel', channel
    return 'single-channel', dataset.channel_index(dataset.rounds.reference_round, channel)


def _image_shape(config, consumer):
    """(img_row, img_col) of the configuration, which only their consumers require (the subtile windows, the legacy
    tile configuration writer); missing, ValueError naming the keys and the consumer."""
    missing = [key for key in ('img_row', 'img_col') if config.get(key) is None]
    if missing:
        raise ValueError(f'{" and ".join(missing)} (the image height and width in pixels) '
                         f'{"is" if len(missing) == 1 else "are"} required by {consumer}')
    return config['img_row'], config['img_col']


def _rule_checkpoints(config, rule):
    """The CheckpointConfig of a processing rule: run.json alone, or the stages of rules.<rule>.parameters.checkpoints.

    The Python-only key is a mapping with the one key stages (absent or null:
    {stages: []}). overwrite is on, so a rerun of the rule replaces its record
    and its checkpoint files; the FOV folders are under <output_root>/fovs and
    the inputs are hashed.
    """
    from .config import CheckpointConfig
    context = f'rules.{rule}.parameters.{_layout.CHECKPOINTS}'
    values = _rule_parameters(config, rule).get(_layout.CHECKPOINTS)
    if values is None:
        values = {}
    else:
        _python_only(config, [context])
        if not isinstance(values, dict):
            raise TypeError(f'{context} must be a mapping')
        _known(values, ('stages',), context)
    return CheckpointConfig(stages=_tuples(values.get('stages', [])), overwrite=True)


def _additional_rounds(config):
    """(round names, channel_order per round that has one) of the shared additional_round entries.

    Each entry names an other round by round_name; its optional channel_order
    lists the round's channels (wavelength, channel and name entries, or
    patterns), which Dataset validates.
    """
    entries = config.get('additional_round') or []
    if not isinstance(entries, list):
        raise ValueError('additional_round must be a list of round entries')
    names, orders = [], {}
    for entry in entries:
        name = entry.get('round_name') if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name:
            raise ValueError('each additional_round entry requires a round_name')
        names.append(name)
        if entry.get('channel_order') is not None:
            orders[name] = entry['channel_order']
    return names, orders


def _reference_stains(config):
    """Dataset.reference_stains: the Python-only reference_stains, or the MATLAB backend's dapi_round stain.

    With backend: python the reference round's stain files are declared by
    reference_stains, a list of {channel, name} entries (channel the file
    pattern and label, name the content; absent or null: none), and
    dapi_round, which names MATLAB's hard-coded *ch04.tif stain, raises
    naming reference_stains. A MATLAB-backend configuration keeps the
    translation of its dapi_round (_legacy_config.matlab_reference_stains).
    """
    if config.get('backend', 'matlab') != 'python':
        _python_only(config, ['reference_stains'] if config.get('reference_stains') is not None else [])
        from ._legacy_config import matlab_reference_stains
        return matlab_reference_stains(config)
    if config.get('dapi_round') is not None:
        raise ValueError('dapi_round names the *ch04.tif stain file of the MATLAB rules; the Python backend declares '
                         "the reference round's stain files with reference_stains: [{channel, name}]")
    values = config.get('reference_stains')
    if values is None:
        return ()
    if not isinstance(values, list) or not values or any(not isinstance(v, dict) for v in values):
        raise ValueError('reference_stains must be a nonempty list of {channel, name} entries')
    return tuple(ChannelInfo.from_value(v) for v in values)


def _nuclear_stain(config):
    """The reference stain that the Python rules rotate_nuclei and nuclei_registration read (ChannelInfo).

    The one entry of reference_stains; with several, the one whose name
    contains the top-level ref_channel (the match nuclei_registration makes
    in each other round). None or several raise ValueError.
    """
    stains = _reference_stains(config)
    if not stains:
        raise ValueError("rotate_nuclei and nuclei_registration read the reference round's stain file: declare it "
                         'with reference_stains: [{channel, name}]')
    if len(stains) == 1:
        return stains[0]
    stain = config.get('ref_channel')
    matches = [c for c in stains if isinstance(stain, str) and stain and c.name is not None and stain in c.name]
    if len(matches) != 1:
        raise ValueError(f'expected one of the reference_stains {[c.channel for c in stains]} whose name contains '
                         f'ref_channel {stain!r}; found {len(matches)}')
    return matches[0]


def _workflow_dataset(config, **fields):
    """The Dataset of every rule: sequencing rounds, other rounds by round_name, channels and the reference stain.

    seq_channel_order is a list of patterns or of wavelength/channel/name
    entries (MATLAB's channel_order_dict); each additional_round entry's
    channel_order gives that round its own channels. voxel_size_xy and
    voxel_size_z become the Dataset fields of the same names, the declared
    calibration of saved files that store none, and reference_image (absent
    or null: fovs/{fov_id}/reference.tif) and reference_image_import (absent
    or null: none) the fields of those names, which exclude each other. The
    Python-only image_dir_template is the field of that name
    (_image_dir_template).
    """
    n = config['n_rounds']
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError('n_rounds must be a positive integer')
    names, orders = _additional_rounds(config)
    rounds = RoundState(sequencing_rounds=[f'round{i}' for i in range(1, n + 1)], other_rounds=names,
                        reference_round=config['ref_round'])
    rounds.validate()
    channels = config.get('seq_channel_order') or ()
    if not isinstance(channels, (list, tuple)):
        raise ValueError('seq_channel_order must be a list of channel patterns or of wavelength/channel/name entries')
    reference = reference_settings(config)
    fields.setdefault('image_dir_template', _image_dir_template(config))
    return Dataset(Path(config['root_input_path']) / config['dataset_id'] / config['sample_id'],
        Path(config['root_output_path']) / config['dataset_id'] / config['output_id'],
        config['dataset_id'], config['sample_id'], config['output_id'], rounds=rounds,
        channel_order=tuple(channels), other_channel_order=orders, reference_stains=_reference_stains(config),
        voxel_size_xy=config.get('voxel_size_xy'), voxel_size_z=config.get('voxel_size_z'),
        reference_image=reference['reference_image'], reference_image_import=reference['reference_image_import'],
        **fields)


def _image_dir_template(config):
    """The Python-only image_dir_template: "{round}/{fov}" (absent or null) or "{fov}/{round}"; ValueError otherwise.

    The MATLAB rules read the round-first layout only, so the key needs
    backend: python.
    """
    value = config.get('image_dir_template')
    if value is None:
        return _layout.IMAGE_DIR_TEMPLATE
    _python_only(config, ['image_dir_template'])
    return _layout.check_image_dir_template(value)


# The Python-only keys naming the reference file of the sequencing rules, by readout mode.
_REFERENCE_KEYS = {'multiplexed': 'codebook', 'direct': 'direct_panel'}
# The channel labels of a codebook loaded only for its genes when the configuration names no seq_channel_order
# (MATLAB's default channel order): the four colours, which the genes do not depend on.
_COLOUR_LABELS = ('1', '2', '3', '4')


def _reference_file(config):
    """The codebook the sequencing rules read, or the direct panel in readout_mode direct.

    The Python-only key codebook (direct_panel in readout_mode direct) names
    the file, absolute or relative to the input root; absent or null, it is
    <input_root>/genes.csv, the file the MATLAB rules read. The key of the
    other readout mode raises ValueError naming both keys.
    """
    mode = config.get('readout_mode', 'multiplexed')
    key = _REFERENCE_KEYS.get(mode, 'codebook')
    given = [k for k in _REFERENCE_KEYS.values() if config.get(k) is not None]
    _python_only(config, given)
    for other in given:
        if other != key:
            raise ValueError(f'{other} names the file of readout_mode '
                             f'{next(m for m, k in _REFERENCE_KEYS.items() if k == other)}; readout_mode {mode} '
                             f'reads {key}')
    value = config.get(key)
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f'{key} must be a file path, absolute or relative to the input root; got {value!r}')
    return _input_root(config) / (_layout.CODEBOOK_FILE if value is None else value)


def _load_reference(adapted, path):
    """adapted.dataset with path loaded: the direct panel in readout_mode direct, else the codebook with the
    encoding and layout of the adapted rule."""
    dataset = adapted.dataset
    if dataset.readout_mode == 'direct':
        dataset.load_direct_panel(path)
    else:
        dataset.load_codebook(path, encoding=adapted.encoding, layout=adapted.layout)
    return dataset


def _first_rule(config):
    """The first Python processing rule the configuration names (rsf_single_fov when none)."""
    return next((r for r in _RULES if r in config.get('rules', {})), 'rsf_single_fov')


def _reference_genes(config, path=None):
    """The genes of the count matrix in their order: the loaded codebook's (Codebook.genes, first-appearance order)
    or direct panel's (DirectPanel.genes, table order).

    path is the reference file (None: _reference_file), loaded as
    _export_dataset loads it, with the encoding and layout of the first Python
    rule the configuration names; for the MATLAB backend that is the file
    MATLAB decodes with. A configuration without seq_channel_order (MATLAB's
    default channel order) labels the codebook's channels by its four colours,
    on which the genes do not depend.
    """
    adapted = from_workflow_config(config, _first_rule(config))
    if not adapted.dataset.channel_order:
        adapted = replace(adapted, dataset=replace(adapted.dataset, channel_order=_COLOUR_LABELS))
    dataset = _load_reference(adapted, _reference_file(config) if path is None else path)
    return tuple(dataset.direct_panel.genes if dataset.readout_mode == 'direct' else dataset.codebook.genes)


def _nuclei_registration(config):
    """(dataset, stain label per other round, output folder per channel per other round) of nuclei_registration.

    Reads the shared keys as workflow/scripts/nuclei_registration.m does, with
    the Dataset of every rule (_workflow_dataset). Each additional_round entry
    is an other round with its channel_order: entry ``channel`` is the filename
    pattern and the round's channel label, which names the channel's output
    folder (MATLAB names it by entry ``name``, which may repeat).
    The round's shared stain is its one channel whose name contains the
    top-level ref_channel (MATLAB single-channel matching, kept here instead of
    Dataset.channel_index); none or several is a ValueError. The reference
    side is the reference stain of reference_stains (_nuclear_stain), which
    must be declared.
    """
    stain = config.get('ref_channel')
    if not isinstance(stain, str) or not stain:
        raise ValueError('nuclei_registration requires the top-level ref_channel naming the shared stain')
    names, orders = _additional_rounds(config)
    if not names:
        raise ValueError('nuclei_registration requires additional_round entries')
    for name in names:
        order = orders.get(name)
        if (not isinstance(order, list) or not order or
                any(not isinstance(c, dict) or not isinstance(c.get('channel'), str) or not isinstance(c.get('name'), str)
                    for c in order)):
            raise ValueError(f'additional round {name!r} requires a channel_order of channel and name entries')
    _nuclear_stain(config)
    dataset = _workflow_dataset(config)
    stains, folders = {}, {}
    for name in names:
        channels = dataset.channel_info(name)
        matches = [c.channel for c in channels if stain in c.name]
        if len(matches) != 1:
            raise ValueError(f'expected one channel of additional round {name!r} whose name contains ref_channel '
                             f'{stain!r}; found {len(matches)}')
        stains[name], folders[name] = matches[0], tuple(c.channel for c in channels)
    return dataset, stains, folders


def _nuclei_checkpoints(config):
    """The CheckpointConfig of the Python-only rules.nuclei_registration.parameters.checkpoints, or None.

    A mapping of CheckpointConfig fields ({} for the defaults) enables the saved
    form of FOV.prepare_morphology; absent or null writes none.
    """
    from .config import CheckpointConfig
    values = _rule_parameters(config, 'nuclei_registration').get(_layout.CHECKPOINTS)
    if values is None:
        return None
    _python_only(config, ['rules.nuclei_registration.parameters.checkpoints'])
    return _typed(CheckpointConfig, values, 'rules.nuclei_registration.parameters.checkpoints')


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
# Keys of a run of the export block's segmentation besides its method's config fields.
_SAMPLE_SEGMENTATION_RUN_KEYS = ('name', 'target', 'inputs', 'seeds', 'method')
# Keys of the assignment block besides the AssignmentConfig fields: the FOV.assign_molecules arguments.
_ASSIGNMENT_CALL_KEYS = ('cells', 'nuclei', 'name', 'population', 'correspondence', _layout.CHECKPOINTS)


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


def _legacy_molecules(config):
    """Whether the workflow's molecule table of a FOV is the legacy one-based ``signal/{fovID}_goodSpots.csv``.

    The configuration names the producer: the backend's ``rsf_single_fov``, or
    ``stitch_subtile``, which the ruleorder of workflow/Snakefile puts first in
    the subtile and deep modes and in free mode with ``gr_single_fov_subtile``
    or ``deep_create_subtile`` on. MATLAB's ``rsf_single_fov`` and
    ``stitch_subtile`` (either backend) write the legacy file; the Python
    backend's ``rsf_single_fov`` writes ``fovs/{fovID}/molecules.csv``.
    """
    if config.get('backend', 'matlab') != 'python':
        return True
    mode = config.get('workflow_mode', 'free')
    if mode in ('subtile', 'deep'):
        return True
    rules = config.get('rules') or {}
    return mode == 'free' and any((rules.get(rule) or {}).get('run', False)
                                  for rule in ('gr_single_fov_subtile', 'deep_create_subtile'))


def _molecule_table(config, output_root, fov_id):
    """The molecule table the workflow's consumers read for a FOV, by producer (_legacy_molecules)."""
    paths = _layout._FovPaths(Path(output_root), fov_id)
    return paths.signal_csv('goodSpots') if _legacy_molecules(config) else paths.spot_table('molecules')


def _label_namespace(config, fov_id, run=None):
    """The JSON identity list of a FOV ([dataset_id, sample_id, fov_id, subtile_id]), with a run name for labels."""
    ids = [config['dataset_id'], config['sample_id'], fov_id, None]
    return json.dumps(ids if run is None else [*ids, run], separators=(',', ':'))


def _declared_metadata(config, fov_id):
    """ImageMetadata of a legacy file without stored metadata: the configured voxel_size_z and voxel_size_xy (µm).

    starfinder.dataset._declared.declared_metadata, the rule Dataset.export uses for such a file.
    """
    from ._declared import declared_metadata
    return declared_metadata(config['dataset_id'], config['output_id'], fov_id,
                             voxel_size_xy=config.get('voxel_size_xy'), voxel_size_z=config.get('voxel_size_z'))


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
    from starfinder._weights import weights_directory
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


def _run_method(values, name, method, target, what='segmentation run'):
    """The method config of one run: import with the LabelImportConfig fields, or a SEGMENTATION_METHODS name whose
    config fields are the run's other keys."""
    from starfinder.segmentation import SEGMENTATION_METHODS, LabelImportConfig
    if method == 'import':
        names = [f.name for f in fields(LabelImportConfig) if f.init and f.name != 'target']
        _known(values, names, f'{what} {name!r} (import)')
        return LabelImportConfig(target=target, **values)
    config_type = config_type_for(SEGMENTATION_METHODS, method, 'segmentation method')
    _known(values, [f.name for f in fields(config_type) if f.init], f'{what} {name!r} ({method})')
    missing = [f.name for f in fields(config_type)
               if f.init and f.default is MISSING and f.default_factory is MISSING and f.name not in values]
    if missing:
        raise ValueError(f'{what} {name!r}: method {method!r} requires {", ".join(missing)}')
    return config_type(**{k: _tuples(v) for k, v in values.items()})


def _segmentation_run(values):
    """SegmentationRun of one segmentation block run (docs/segmentation-contract.md, "Workflow configuration")."""
    from starfinder.preprocessing import ProjectionConfig
    from starfinder.segmentation import ExpandLabelsConfig, SegmentationRun, ZExtensionConfig
    if not isinstance(values, dict) or 'method' not in values or 'name' not in values:
        raise ValueError('each segmentation run must be a mapping with a name and a method')
    values = dict(values)
    run = {key: values.pop(key) for key in _SEGMENTATION_RUN_KEYS if key in values}
    name, method = run['name'], run['method']
    config = _run_method(values, name, method, run.get('target'))
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
    """(AssignmentConfig, FOV.assign_molecules keyword arguments) of the Python-only top-level assignment block.

    Its keys are the AssignmentConfig fields (expansion and correspondence as
    mappings of ExpandLabelsConfig and CorrespondenceConfig fields) and the
    FOV.assign_molecules arguments cells, nuclei, name, population and checkpoints (a
    mapping of CheckpointConfig fields); a supplied correspondence table is not
    a YAML value, so correspondence configures the overlap rule. The block needs
    backend: python; the §2.13 rule runs it with FOV.assign_molecules
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
    if values.get(_layout.CHECKPOINTS) is not None:
        call[_layout.CHECKPOINTS] = _typed(CheckpointConfig, values[_layout.CHECKPOINTS], 'assignment.checkpoints')
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
    (_legacy_target). The assignment block is run by FOV.assign_molecules, not by this
    rule, so it raises here.
    """
    from starfinder.assignment import AssignmentConfig
    from starfinder.segmentation import ExpandLabelsConfig
    if 'assignment' in config:
        _python_only(config, ['assignment'])
        raise ValueError('the assignment block is run by FOV.assign_molecules (the §2.13 rule); the reads_assignment '
                         'rule translates rules.reads_assignment.parameters only: remove the block or the rule')
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


# --- §2.10: the Python-only stitching, assembly and export blocks (docs/stitching-contract.md,
# --- docs/assembly-contract.md and docs/sample-output-contract.md, "Workflow configuration").

_STITCHING_KEYS = ('method', 'input', 'positions', 'config', 'review', 'fallback', 'bigstitcher_save_dir')
# BigStitcher runs in Fiji as an engine task, so it is a method of the block but not of STITCHING_METHODS.
_BIGSTITCHER = 'bigstitcher'
# Keys of the assembly block besides the AssemblyConfig fields: the Dataset.export argument, the table format and the
# workflow's opt-in assignment previews.
_ASSEMBLY_CALL_KEYS = ('assignment', 'table_format', 'previews')
# ExportConfig fields the export block does not name: the assembly block gives them, or they are its parts.
_EXPORT_PARTS = ('assembly', 'table_format', 'fusion', 'labels', 'spatialdata')


@dataclass(frozen=True)
class _StitchingBlock:
    """The stitching block translated (see _stitching_block)."""
    method: str
    config: object
    input: str
    positions: dict
    review: object
    fallback: str | None
    bigstitcher_save_dir: Path | None


@dataclass(frozen=True)
class _AssemblyBlock:
    """The assembly block translated (see _assembly_block)."""
    config: object
    assignment: str | None
    table_format: str
    previews: bool = False


def _stitching_methods():
    """The block's method names: those of STITCHING_METHODS, then bigstitcher."""
    from starfinder.stitching import STITCHING_METHODS
    return (*dict.fromkeys(spec.name for spec in STITCHING_METHODS.values()), _BIGSTITCHER)


def _stitching_block(config) -> _StitchingBlock:
    """The Python-only stitching block (docs/stitching-contract.md, "Workflow configuration").

    method is a STITCHING_METHODS name, run in process by Dataset.stitch, or
    bigstitcher (the default), run by the macro, engine and import rules; config
    holds the fields of that method's config (MultiviewStitcherConfig or
    BigStitcherConfig); input is reference_merged (default), reference_stain or
    file:<pattern>; positions is the positions mapping of Dataset.stitch, whose
    stage_axes is required with source stage_file (never inferred from
    rotate_angle); review holds the ReviewPolicy fields; fallback is null or
    nominal; bigstitcher_save_dir is a local folder, required with bigstitcher
    and refused with another method. Raises ValueError naming the key.
    """
    from starfinder.stitching import STITCHING_METHODS, BigStitcherConfig, ReviewPolicy
    from ._stitching import POSITION_KEYS, _check_input
    _python_only(config, ['stitching'])
    values = config['stitching']
    if not isinstance(values, dict):
        raise TypeError('stitching must be a mapping')
    _known(values, _STITCHING_KEYS, 'stitching')
    method = values.get('method', _BIGSTITCHER)
    if method not in _stitching_methods():
        raise ValueError(f'stitching.method must be one of {list(_stitching_methods())}; got {method!r}')
    config_type = (BigStitcherConfig if method == _BIGSTITCHER
                   else config_type_for(STITCHING_METHODS, method, 'stitching method'))
    method_config = _typed(config_type, values.get('config') or {}, 'stitching.config')
    input_name = values.get('input', 'reference_merged')
    _check_input(input_name)
    positions = values.get('positions')
    if not isinstance(positions, dict):
        raise ValueError('stitching.positions (the positions mapping of Dataset.stitch) is required')
    _known(positions, POSITION_KEYS, 'stitching.positions')
    if positions.get('source') == 'stage_file' and positions.get('stage_axes') is None:
        raise ValueError('stitching.positions.stage_axes ({x, y}) is required with source stage_file; it has no '
                         'default and is never inferred from rotate_angle')
    if positions.get('source') == 'stage_file' and not positions.get('stage_file'):
        raise ValueError("stitching.positions.stage_file (the stage file's path) is required with source stage_file; "
                         'it has no default')
    if positions.get('source') == 'table' and not positions.get('table'):
        raise ValueError('stitching.positions.table (a CSV path) is required with source table')
    review = _typed(ReviewPolicy, values.get('review') or {}, 'stitching.review')
    fallback = values.get('fallback')
    if fallback not in (None, 'nominal'):
        raise ValueError(f"stitching.fallback must be null or 'nominal'; got {fallback!r}")
    save_dir = values.get('bigstitcher_save_dir')
    if method == _BIGSTITCHER and not save_dir:
        raise ValueError('stitching.bigstitcher_save_dir (a local folder for the tiles, the macro and dataset.xml) is '
                         'required with method bigstitcher')
    if method != _BIGSTITCHER and save_dir is not None:
        raise ValueError(f'stitching.bigstitcher_save_dir is read with method bigstitcher only; method is {method!r}')
    if method == _BIGSTITCHER and 'assembly' not in config:
        # Without an assembly block the import writes output/tile_config_{sample}.csv with the legacy writer.
        _image_shape(config, 'the legacy tile configuration writer of stitching method bigstitcher without an '
                             'assembly block')
    return _StitchingBlock(method, method_config, input_name, {k: _tuples(v) for k, v in positions.items()}, review,
                           fallback, None if save_dir is None else Path(save_dir))


@dataclass(frozen=True)
class _Membership:
    """The Python workflow's FOV membership (_membership): every FOV of the run and each sample's FOV IDs, in the
    order of fovs.csv, and each FOV's PositionID (None without the position_id column)."""
    fovs: tuple
    samples: dict
    position_ids: dict | None


_MEMBERSHIP_COLUMNS = ('sample', 'fov_id')


def _membership(config, path=None) -> _Membership:
    """The FOVs of a Python run: <input_root>/fovs.csv (or path), restricted by the top-level selected_fovs.

    fovs.csv has the columns sample and fov_id and an optional position_id
    (the stage file's PositionID of the FOV, an integer), read in row order:
    the samples in the order of their first row, each sample's FOVs in row
    order. A repeated row, a FOV in two rows (two samples or two
    PositionIDs), a PositionID of two FOVs, an empty value or another column
    raises ValueError naming them. selected_fovs (absent or null: every FOV) is
    a nonempty list of FOV IDs of the file; it restricts every rule, and a
    sample with none of them is left out. The sample column groups FOVs; it is
    not the acquisition sample_id. The MATLAB backend reads its sample
    annotation instead (common.smk).
    """
    import csv
    from collections import Counter
    path = _layout.fovs_file(_input_root(config)) if path is None else Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'the FOV membership file {path} (columns sample,fov_id and optionally position_id) '
                                'does not exist')
    with open(path, newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.reader(handle))
    header = [v.strip() for v in rows[0]] if rows else []
    if header[:2] != list(_MEMBERSHIP_COLUMNS) or header[2:] not in ([], ['position_id']):
        raise ValueError(f'{path} must have the columns sample,fov_id and optionally position_id; got {header}')
    body = [[v.strip() for v in row] for row in rows[1:] if any(v.strip() for v in row)]
    if not body:
        raise ValueError(f'{path} lists no FOV')
    bad = [n for n, row in enumerate(body, start=2) if len(row) != len(header) or not all(row)]
    if bad:
        raise ValueError(f'{path}: rows {bad} do not give a value for each of {header}')
    with_ids = len(header) == 3
    if with_ids:
        unparsed = [row[2] for row in body if not _is_integer(row[2])]
        if unparsed:
            raise ValueError(f'{path}: position_id must be an integer PositionID; got {unparsed}')
    seen, repeated, conflicting = {}, [], []
    for row in body:
        key = tuple(row)
        if row[1] in seen:
            (repeated if seen[row[1]] == key else conflicting).append(row[1])
        seen.setdefault(row[1], key)
    if repeated:
        raise ValueError(f'{path} repeats the rows of the FOVs {sorted(set(repeated))}')
    if conflicting:
        raise ValueError(f'{path} gives the FOVs {sorted(set(conflicting))} conflicting rows (another sample or '
                         'position_id); a FOV has one row')
    positions = {row[1]: int(row[2]) for row in body} if with_ids else None
    if positions is not None:
        shared = sorted(p for p, n in Counter(positions.values()).items() if n > 1)
        if shared:
            raise ValueError(f'{path} gives the position_id {shared} to more than one FOV')
    selected = config.get('selected_fovs')
    if selected is not None:
        if (not isinstance(selected, list) or not selected or
                any(not isinstance(f, str) or not f for f in selected)):
            raise ValueError('selected_fovs must be a nonempty list of FOV IDs of fovs.csv (omit it to run every FOV)')
        unknown = [f for f in selected if f not in seen]
        twice = sorted(f for f, n in Counter(selected).items() if n > 1)
        if unknown or twice:
            raise ValueError(f'selected_fovs must name FOVs of {path} once each; unknown {unknown}, repeated {twice}')
    rows = [row[:2] for row in body if selected is None or row[1] in selected]
    samples = {}
    for sample, fov in rows:
        samples.setdefault(sample, []).append(fov)
    return _Membership(tuple(fov for _, fov in rows), {sample: tuple(fovs) for sample, fovs in samples.items()},
                       None if positions is None else {fov: positions[fov] for _, fov in rows})


def _is_integer(text):
    try:
        int(text)
    except ValueError:
        return False
    return True


def _sample_fovs(config, sample, path=None):
    """The FOV IDs of a sample, in the order of fovs.csv (_membership, restricted by selected_fovs)."""
    membership = _membership(config, path)
    if sample not in membership.samples:
        raise ValueError(f'sample {sample!r} has no FOV of the run; its samples are {list(membership.samples)}')
    return list(membership.samples[sample])


def _stage_position_ids(config, membership):
    """The PositionID of each FOV (fovs.csv's position_id) with a stitching block whose positions come from a stage
    file, else None. The column is then required: the workflow names no FOV by a pattern."""
    positions = (config.get('stitching') or {}).get('positions') or {}
    if positions.get('source') != 'stage_file':
        return None
    if membership.position_ids is None:
        raise ValueError('stitching.positions.source stage_file needs the position_id column of fovs.csv (the stage '
                         "file's PositionID of each FOV)")
    return membership.position_ids


def _check_sample_rules(config):
    """Refuse, for the Python backend, the legacy sample rules, which run with backend matlab only.

    They read MATLAB's sample annotation: the four legacy stitching rules
    (and the per-sample stage file create_sample_maf writes) are replaced by
    the stitching block, create_sample_h5ad and create_sample_reads_assignment
    by the assembly block, and reads_assignment without an assembly block reads
    the tile configuration of the stitching block's bigstitcher import.
    Raises ValueError naming the rules and the blocks to add.
    """
    rules = config.get('rules') or {}
    enabled = [rule for rule, values in rules.items() if isinstance(values, dict) and values.get('run', False)]
    needs = []
    stitching = [r for r in enabled if r in _LEGACY_STITCHING_RULES]
    if stitching and 'stitching' not in config:
        needs.append(f'{stitching} need a stitching block')
    sample = [r for r in enabled if r in _LEGACY_SAMPLE_RULES]
    if sample and 'assembly' not in config:
        needs.append(f'{sample} need a stitching block and an assembly block')
    if 'reads_assignment' in enabled and 'assembly' not in config and 'stitching' not in config:
        needs.append("['reads_assignment'] needs an assembly block, or a stitching block with method bigstitcher, "
                     'which writes the tile configuration it reads')
    if needs:
        raise ValueError('the legacy sample rules run with backend matlab only; for the Python backend '
                         + '; '.join(needs))


_LEGACY_STITCHING_RULES = ('stitching_preparation', 'create_BigStitcher_macro', 'run_BigStitcher_macro',
                           'create_tile_config')
_LEGACY_SAMPLE_RULES = ('create_sample_h5ad', 'create_sample_reads_assignment')


def _input_root(config):
    return Path(config['root_input_path']) / config['dataset_id'] / config['sample_id']


def _sample_grid_fovs(config, sample, path=None):
    """Every FOV of a sample in fovs.csv, selected_fovs aside: the FOVs a regular grid places in its cells, so that a
    selection keeps each selected FOV's cell."""
    return _sample_fovs(dict(config, selected_fovs=None), sample, path)


def _sample_table(block, dataset, sample, fov_ids, grids, position_ids, grid_fov_ids=None):
    """The tile position table of the block's positions mapping, holding exactly fov_ids (as Dataset.stitch builds it).

    A stage file's positions are those of position_ids (_stage_position_ids),
    selected before the conversion; a regular grid places grid_fov_ids
    (_sample_grid_fovs) and keeps the rows of fov_ids. A FOV of the sample that the positions do
    not list, or a listed FOV that is not one of the sample, raises ValueError
    naming both lists, as Dataset.stitch does for a table that does not match
    its FOVs.
    """
    from ._stitching import table_from_positions
    table = table_from_positions(block.positions, fov_ids, grids, fov_pattern=dataset.fov_pattern,
                                 position_ids=position_ids, grid_fov_ids=grid_fov_ids)
    if set(table.fov_ids) != set(fov_ids):
        missing = [f for f in fov_ids if f not in table.fov_ids]
        extra = [f for f in table.fov_ids if f not in fov_ids]
        raise ValueError(f'the tile position table must hold exactly the FOVs of sample {sample!r}; missing {missing}, '
                         f'extra {extra}')
    return table


def _assembly_block(config) -> _AssemblyBlock:
    """The Python-only assembly block (docs/assembly-contract.md, "Workflow configuration").

    Its keys are the AssemblyConfig fields, assignment (the assignment name
    read per FOV, default ``default``; null for molecules only),
    table_format (csv or parquet) and previews (default false: on the route
    of a saved assignment, reads_assignment draws the local assignment
    figures into fovs/<fov>/previews/ and export_sample the ownership figures
    into export/<sample>/previews/<fov>/). It needs backend: python and a
    stitching block, whose placement it reads.
    """
    from starfinder._registry import check_name
    from starfinder.assembly import AssemblyConfig
    from starfinder.export._config import TABLE_FORMATS
    _python_only(config, ['assembly'])
    if 'stitching' not in config:
        raise ValueError("the assembly block reads the stitching block's placement; add a stitching block")
    values = config['assembly']
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise TypeError('assembly must be a mapping')
    names = [f.name for f in fields(AssemblyConfig) if f.init]
    _known(values, {*names, *_ASSEMBLY_CALL_KEYS}, 'assembly')
    assignment = values.get('assignment', 'default')
    if assignment is not None:
        check_name(assignment, 'assignment')
    table_format = values.get('table_format', 'csv')
    if table_format not in TABLE_FORMATS:
        raise ValueError(f'assembly.table_format must be one of {list(TABLE_FORMATS)}; got {table_format!r}')
    previews = _flag(values, 'previews', 'assembly')
    return _AssemblyBlock(AssemblyConfig(**{k: values[k] for k in names if k in values}), assignment, table_format,
                          previews)


def _export_keys():
    """export block key → (ExportConfig part or None, field): the fields of ExportConfig (its flags), FusionConfig,
    LabelConfig as labels_<field> and SpatialDataExportConfig as spatialdata_<field>."""
    from starfinder.export import ExportConfig, FusionConfig, LabelConfig, SpatialDataExportConfig
    keys = {f.name: (None, f.name) for f in fields(ExportConfig) if f.init and f.name not in _EXPORT_PARTS}
    keys.update({f.name: ('fusion', f.name) for f in fields(FusionConfig) if f.init})
    keys.update({f'labels_{f.name}': ('labels', f.name) for f in fields(LabelConfig) if f.init})
    keys.update({f'spatialdata_{f.name}': ('spatialdata', f.name) for f in fields(SpatialDataExportConfig) if f.init})
    return keys


def _sample_segmentation_block(values, config):
    """SampleSegmentationConfig of the export block's segmentation (docs/sample-segmentation-contract.md, "Where the
    settings live"): device, cells, nuclei and runs, each run with name, target, inputs (a list of {role, image}),
    seeds and method, whose config fields are the run's other keys (import: the LabelImportConfig fields); the
    assignment settings are those of the top-level assignment block."""
    from starfinder.assignment import AssignmentConfig
    from starfinder.export import SampleSegmentationConfig, SampleSegmentationRun
    if not isinstance(values, dict):
        raise TypeError('export.segmentation must be a mapping')
    _known(values, ('device', 'cells', 'nuclei', 'runs'), 'export.segmentation')
    runs = values.get('runs')
    if not isinstance(runs, list) or not runs:
        raise ValueError('export.segmentation.runs must be a nonempty list')
    parsed = []
    for entry in runs:
        if not isinstance(entry, dict) or 'method' not in entry or 'name' not in entry:
            raise ValueError('each export.segmentation run must be a mapping with a name and a method')
        entry = dict(entry)
        run = {key: entry.pop(key) for key in _SAMPLE_SEGMENTATION_RUN_KEYS if key in entry}
        inputs = run.get('inputs') or []
        if not isinstance(inputs, list) or not all(isinstance(i, dict) for i in inputs):
            raise TypeError(f"export.segmentation run {run['name']!r}: inputs must be a list of {{role, image}}")
        for item in inputs:
            _known(item, ('role', 'image'), f"export.segmentation run {run['name']!r} input")
        method = _run_method(entry, run['name'], run['method'], run.get('target'), 'export.segmentation run')
        parsed.append(SampleSegmentationRun(run['name'], run.get('target'),
                                            tuple((i.get('role'), i.get('image')) for i in inputs), method,
                                            run.get('seeds')))
    assignment = _assignment_block(config)[0] if config.get('assignment') is not None else AssignmentConfig()
    return SampleSegmentationConfig(tuple(parsed), values.get('cells'), values.get('nuclei'), assignment,
                                    values.get('device', 'cpu'))


def _export_config(config):
    """(ExportConfig, assembly block) of the export block and the assembly block (docs/sample-output-contract.md).

    The export block's keys turn the parts on and off and set FusionConfig,
    LabelConfig (labels_scale_yx) and SpatialDataExportConfig
    (spatialdata_zarr_format), the route (route, segment) and the settings of
    the route after fusion (segmentation, a SampleSegmentationConfig); without
    the block the defaults apply. The
    assembly and table_format of the ExportConfig come from the assembly block,
    which the export block needs.
    """
    from starfinder.export import ExportConfig, FusionConfig, LabelConfig, SpatialDataExportConfig
    _python_only(config, ['export'] if 'export' in config else [])
    if 'assembly' not in config:
        raise ValueError('the export block configures the export_sample rule, which runs with an assembly block only; '
                         'add an assembly block')
    block = _assembly_block(config)
    values = config.get('export')
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise TypeError('export must be a mapping')
    keys = _export_keys()
    _known(values, keys, 'export')
    parts = {'fusion': {}, 'labels': {}, 'spatialdata': {}}
    flags = {}
    for key, value in values.items():
        part, name = keys[key]
        if key == 'segmentation':
            flags[name] = None if value is None else _sample_segmentation_block(value, config)
            continue
        (flags if part is None else parts[part])[name] = _tuples(value)
    export = ExportConfig(**flags, assembly=block.config, table_format=block.table_format,
                          fusion=FusionConfig(**parts['fusion']), labels=LabelConfig(**parts['labels']),
                          spatialdata=SpatialDataExportConfig(**parts['spatialdata']))
    return export, block


def _after_fusion(export, block):
    """Whether export_sample runs the route after fusion: the route resolved by Dataset.export's rule (set, or by
    default after_fusion for dimensionality projection) with an assignment name; without one the export is
    molecules_only on either route (docs/sample-segmentation-contract.md, "The route setting")."""
    from starfinder.export._segment import resolve_route
    return block.assignment is not None and resolve_route(export)[0] == 'after_fusion'


def _check_export_route(config):
    """Check 7 of Dataset.export as far as the configuration alone decides it, when the rule files are parsed.

    The route value, the stack and sum refusals, the default without
    export.segmentation (RD1), the runs' image names, fuse and assemble with
    segment on, and the runs themselves raise here naming the key. The reuse
    conditions (segment: false) and the models are checked by Dataset.export
    before anything is written: they need the export folder and the model
    files, and a model is not hashed at every parse.
    """
    from starfinder.export._segment import check_route
    export, block = _export_config(config)
    check_route(export, block.assignment, models=False)


def _export_dataset(config, block, export=None):
    """The Dataset of export_sample; with the codebook loaded for its genes when the export reads the FOVs'
    molecule tables: molecules only (assignment null), or the route after fusion (``export``, the ExportConfig;
    None reads it from the configuration).

    The codebook is the sequencing rules' reference file (_reference_file: the
    file codebook names, or direct_panel in readout_mode direct, else
    <input_root>/genes.csv), loaded with the encoding and layout of the first
    Python rule the configuration names (rsf_single_fov when none).
    """
    if block.assignment is not None and not _after_fusion(_export_config(config)[0] if export is None else export,
                                                          block):
        return _workflow_dataset(config)
    adapted = from_workflow_config(config, _first_rule(config))
    # The molecule tables of the legacy route are read with the legacy reader (_legacy_molecules).
    adapted.dataset._legacy_molecules = _legacy_molecules(config)
    return _load_reference(adapted, _reference_file(config))


# --- §2.11: the Python-only report block and the rules report_sample and report_dataset
# --- (docs/report-contract.md, "Workflow configuration", option WR1).

def _report_block(config):
    """ReportConfig of the Python-only top-level report block (docs/report-contract.md, "Workflow configuration").

    Its keys are the ReportConfig fields: markers a list of {name, genes}
    (MarkerGroup), marker_pairs a list of gene pairs, limits the ReportLimits
    fields with each rule a mapping of RuleLimit fields (k, floor, absolute; an
    omitted one is null) and spots_near_zero. Lists become tuples; an empty or
    null block is ReportConfig(). An unknown key raises ValueError naming it;
    the block needs backend: python.
    """
    from starfinder.report import MarkerGroup, ReportConfig, ReportLimits, RuleLimit
    _python_only(config, ['report'])
    values = config['report']
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise TypeError('report must be a mapping')
    _known(values, [f.name for f in fields(ReportConfig) if f.init], 'report')
    options = {key: _tuples(value) for key, value in values.items() if key not in ('markers', 'limits')}
    if values.get('markers') is not None:
        if not isinstance(values['markers'], list):
            raise TypeError('report.markers must be a list of {name, genes}')
        options['markers'] = tuple(_typed(MarkerGroup, entry, 'report.markers') for entry in values['markers'])
    if values.get('limits') is not None:
        limits = values['limits']
        if not isinstance(limits, dict):
            raise TypeError('report.limits must be a mapping')
        _known(limits, [f.name for f in fields(ReportLimits) if f.init], 'report.limits')
        options['limits'] = ReportLimits(**{
            key: (_typed(RuleLimit, value, f'report.limits.{key}') if key != 'spots_near_zero' else value)
            for key, value in limits.items()})
    return ReportConfig(**options)


def _report_assignment(config):
    """The assignment name report_sample reads per FOV: the assembly block's assignment; without an assembly block
    no rule saves an assignment folder, so None (a sample without cells)."""
    return _assembly_block(config).assignment if 'assembly' in config else None
