"""The MATLAB-key translation that remains for MATLAB-backend configurations.

The Python backend accepts one dialect, the native keys of from_workflow_config; a legacy key under
``backend: python`` raises naming its native form. The shared rules (reads_assignment, the legacy
stitching and segmentation rules) also run for the MATLAB backend, and their Python adapters call
from_workflow_config on a MATLAB configuration, which holds the keys the MATLAB scripts read. This
module translates those keys, as the Python path did before the configuration-dialect change of
2026-10-25: only from_workflow_config's MATLAB branch imports it (docs/migration.md, "One Python
configuration dialect").
"""
# SM-16, W-569; before W-569
from dataclasses import fields

from starfinder._registry import config_type_for
from starfinder.preprocessing import (HistogramMatchingConfig, MinMaxNormalizationConfig, PreprocessingRecipe,
    PreprocessingStep, ReconstructionConfig, TophatConfig)
from starfinder.registration import (REGISTRATION_METHODS, CpdConfig, DemonsConfig, RegistrationRecipe,
    RegistrationSignalConfig, RegistrationStep, WarpConfig)

from .types import ChannelInfo
from .workflow import _NATIVE_PARAMETERS, _explicit_recipe, _explicit_registration, _operation, _recovery

# Keys replaced by the explicit preprocessing key (snr_threshold only feeds min-max).
_LEGACY_PREPROCESSING = ('enhance_contrast', 'hist_equalize', 'morph_recon', 'tophat', 'snr_threshold')
# Legacy method names: the demons variants select method demons with that variant.
_DEMONS_VARIANTS = ('diffeomorphic', 'symmetric', 'fast_symmetric')
# Legacy spot_finding keys: aliases of LocalMaximaConfig fields, for local_maxima only.
_SPOT_ALIASES = {'intensity_estimation': 'threshold_mode', 'intensity_threshold': 'threshold_value',
                 'min_distance': 'min_distance_voxels'}
# Legacy image representations: merged (either spelling) is the channel maximum, as in MATLAB.
_SIGNAL_MODES = {'merged-image': 'max', 'merged': 'max', 'single-channel': 'channel'}
# The registration parameter aliases of the legacy blocks and the fields they name.
_PARAMETER_ALIASES = _NATIVE_PARAMETERS
# The reference stain nuclei_registration.m reads: the reference round's *ch04.tif.
_MATLAB_STAIN_CHANNEL = 'ch04'
_RESIDENT = ('lrsf_single_fov_subtile', 'deep_rsf_subtile')


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
    translated = {}
    for key, value in values.items():
        target = _PARAMETER_ALIASES.get(key, key)
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


def legacy_processing(params, rule, reference_round):
    """(preprocessing, registration, reference view, spot aliases) of one rule's MATLAB-backend parameters.

    The translation of the Python path before the configuration-dialect change
    of 2026-10-25: the native preprocessing key is mutually exclusive with
    enhance_contrast, hist_equalize, morph_recon, tophat and snr_threshold, which map to recipe 1 (resident
    subtile rules run reconstruction after registration); the native
    registration key is mutually exclusive with an enabled global_registration
    or local_registration, which map to one recipe (global step, then local
    step; merged-image is the channel maximum; CPD detection_noise_sigma 3.0 and
    grid_spacing_voxels 32). The reference view is what the MATLAB script saves:
    single-channel only for rsf_single_fov with global_registration alone and
    a single-channel signal. The spot aliases are those of _spot_fields for
    local_maxima.
    """
    # Before W-569
    legacy = [key for key in _LEGACY_PREPROCESSING if key in params]
    if 'preprocessing' in params and legacy:
        raise ValueError(f'preprocessing is mutually exclusive with the legacy keys {legacy}')
    norm, do_norm = _operation(params, 'enhance_contrast', ('snr_threshold',))
    hist, do_hist = _operation(params, 'hist_equalize', ('reference_channel',))
    morph, do_morph = _operation(params, 'morph_recon', ('radius',))
    top, do_top = _operation(params, 'tophat', ('radius',))
    registration_keys = {f.name for cls in REGISTRATION_METHODS for f in fields(cls) if f.init}
    registration_keys |= {'ref_round', 'method', 'ref_img', 'mov_img', 'ref_channel', 'boundary_mode', 'recovery',
                          *_PARAMETER_ALIASES}
    blocks, enabled_registration = [], []
    for name in ('global_registration', 'local_registration'):
        values, enabled = _operation(params, name, registration_keys)
        if values.pop('ref_round', reference_round) != reference_round:
            raise ValueError('rule reference round differs from dataset reference')
        if enabled:
            blocks.append((values, name == 'local_registration'))
            enabled_registration.append(name)
    if 'registration' in params and enabled_registration:
        raise ValueError(f'registration is mutually exclusive with the enabled legacy blocks {enabled_registration}')
    registration = (_explicit_registration(params['registration']) if 'registration' in params
                    else _legacy_registration(blocks))
    # The reference image is what the MATLAB script saves: the reference image of its last
    # registration. Only rsf_single_fov passes ref_img, to global registration;
    # its local registration and the other scripts use the channel maximum.
    reference_view = ('merged', 0)
    if rule == 'rsf_single_fov' and enabled_registration == ['global_registration']:
        signal = registration.signal
        reference_view = ('single-channel', signal.reference_channel) if signal.mode == 'channel' else ('merged', 0)
    preprocessing = (_explicit_recipe(params['preprocessing']) if 'preprocessing' in params else
                     _legacy_recipe(params, norm, do_norm, hist, do_hist, morph, do_morph, top, do_top,
                                    rule in _RESIDENT))
    return preprocessing, registration, reference_view, _SPOT_ALIASES


def matlab_reference_stains(config):
    """Dataset.reference_stains of the MATLAB keys: the reference round's *ch04.tif, named by ref_channel.

    dapi_round names the round whose ch04 file is the stain (MATLAB's
    nuclei_registration rule); it must be ref_round. Without dapi_round there
    is no reference stain.
    """
    dapi, ref = config.get('dapi_round'), config['ref_round']
    if dapi is None:
        return ()
    if dapi != ref:
        raise ValueError(f'dapi_round {dapi!r} differs from ref_round {ref!r}: the stain file '
                         f'(*{_MATLAB_STAIN_CHANNEL}.tif) is a file of the reference round')
    return (ChannelInfo(_MATLAB_STAIN_CHANNEL, config.get('ref_channel')),)
