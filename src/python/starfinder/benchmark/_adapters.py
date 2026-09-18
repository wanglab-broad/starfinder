"""Explicit registration processing and saved-array evaluation adapters."""
from dataclasses import asdict
import numpy as np
from starfinder.image import ImageMetadata
from starfinder.registration import (TranslationConfig, DemonsConfig, TpsConfig, CpdConfig,
    InsufficientLandmarksError, RegistrationEstimationError)

_CONFIGS = {'translation': TranslationConfig, 'demons': DemonsConfig,
            'tps': TpsConfig, 'cpd': CpdConfig}
_ERRORS = {'InsufficientLandmarksError': InsufficientLandmarksError,
           'RegistrationEstimationError': RegistrationEstimationError}


def _config(value):
    value = dict(value)
    method = value.pop('method')
    if method not in _CONFIGS:
        raise ValueError(f'unsupported registration method: {method}')
    return _CONFIGS[method](**value)


def _validate(case):
    if case.task == 'pipeline':
        from ._pipeline import _validate_pipeline
        return _validate_pipeline(case)
    if case.task != 'registration':
        raise ValueError(f'unsupported benchmark task: {case.task}')
    if set(case.inputs) != {'reference', 'moving'}:
        raise ValueError('registration requires reference and moving inputs')
    required = {'registration', 'reference_metadata', 'moving_metadata', 'evaluation'}
    if not required <= set(case.config) or set(case.config) - required - {'fallback'}:
        raise ValueError('case config requires registration, metadata and evaluation; unknown keys rejected')
    _config(case.config['registration'])
    ImageMetadata(**case.config['reference_metadata'])
    ImageMetadata(**case.config['moving_metadata'])
    fallback = case.config.get('fallback', {'on_errors': [], 'configs': []})
    if set(fallback) != {'on_errors', 'configs'} or any(x not in _ERRORS for x in fallback['on_errors']):
        raise ValueError('fallback requires explicit supported estimation error categories and configs')
    for config in fallback['configs']:
        _config(config)
    evaluation = case.config['evaluation']
    if not evaluation or set(evaluation) - {'ncc', 'ssim', 'translation'}:
        raise ValueError('evaluation must explicitly select ncc, ssim and/or translation')
    if 'ncc' in evaluation and evaluation['ncc'] is not True:
        raise ValueError('ncc must be true when selected')
    if 'ssim' in evaluation:
        options = evaluation['ssim']
        if not {'data_range', 'policy'} <= set(options) or set(options) - {'data_range', 'policy', 'win_size', 'slice_index'}:
            raise ValueError('SSIM requires data_range and policy; unknown options rejected')
        if not np.isfinite(options['data_range']) or options['data_range'] <= 0:
            raise ValueError('SSIM data_range must be finite and positive')
        if options['policy'] not in ('mip', 'volume', 'slice'):
            raise ValueError('registration SSIM policy must be mip, volume or slice')
        window = options.get('win_size')
        if window is not None and (type(window) is not int or window < 3 or window % 2 == 0):
            raise ValueError('SSIM win_size must be odd and >=3')
        index = options.get('slice_index')
        if options['policy'] == 'slice':
            if type(index) is not int or index < 0:
                raise ValueError('SSIM slice requires nonnegative slice_index')
        elif index is not None:
            raise ValueError('SSIM slice_index requires slice policy')
    if 'translation' in evaluation:
        if set(evaluation['translation']) != {'tolerance'} or 'correction' not in case.truth:
            raise ValueError('translation evaluation requires tolerance and correction truth JSON')
        tol = evaluation['translation']['tolerance']
        if tol is not None and (not np.isfinite(tol) or tol < 0):
            raise ValueError('translation tolerance must be nonnegative or null')


def _process(case, reference, moving, attempts):
    from starfinder.registration import estimate_transform, apply_transform
    fallback = case.config.get('fallback', {'on_errors': [], 'configs': []})
    configs = [case.config['registration'], *fallback['configs']]
    allowed = tuple(_ERRORS[x] for x in fallback['on_errors'])
    for index, value in enumerate(configs):
        config = _config(value)
        attempt = {'method': config.method, 'config': asdict(config), 'status': 'failed'}
        attempts.append(attempt)
        try:
            result = estimate_transform(reference, moving, config=config,
                reference_metadata=ImageMetadata(**case.config['reference_metadata']),
                moving_metadata=ImageMetadata(**case.config['moving_metadata']))
        except Exception as exc:
            attempt['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            if isinstance(exc, allowed) and index + 1 < len(configs):
                continue
            raise
        # Application errors never select a fallback estimator.
        try:
            registered = apply_transform(moving, result.transform, config=result.application_config)
        except Exception as exc:
            attempt['error'] = {**{'type': type(exc).__name__, 'message': str(exc)}, 'stage': 'application'}
            raise
        attempt['status'] = 'success'
        return registered, result


def _evaluate(case, arrays, transform, truth):
    from starfinder.evaluation.registration import (
        normalized_cross_correlation, structural_similarity, evaluate_translation)
    metrics = {}
    for label in ('moving', 'registered'):
        if 'ncc' in case.config['evaluation']:
            metrics['ncc_' + label] = asdict(normalized_cross_correlation(arrays['reference'], arrays[label]))
        if 'ssim' in case.config['evaluation']:
            metrics['ssim_' + label] = asdict(structural_similarity(arrays['reference'], arrays[label],
                **case.config['evaluation']['ssim']))
    if 'translation' in case.config['evaluation']:
        if 'correction_zyx' not in transform:
            raise ValueError('translation evaluation requires a translation transform')
        metadata = ImageMetadata(**case.config['reference_metadata'])
        metrics['translation'] = asdict(evaluate_translation(
            {'moving': transform['correction_zyx']}, {'moving': truth['correction']},
            reference_metadata=metadata, observed_metadata=metadata, units='voxel',
            **case.config['evaluation']['translation']))
    return metrics
