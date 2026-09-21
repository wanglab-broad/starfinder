"""Independent, bounded audit of controlled-development-v1 saved outputs.

The oracle imports no production renderer, geometry, observation or preset helper.
Its constants are the published development preset specification. The optional
repeat command imports the generator solely to compare persisted results.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from starfinder.io import load_image_checkpoint
from saved_synthetic import checksum, read_json, write_json, process_identity
from synthetic_specification import stream_descriptor

SIZES = {'z1': (1, 32, 32), 'small': (9, 32, 32), 'wide': (9, 48, 48)}
FACTORS = ('brightness', 'axial_width', 'lateral_width', 'elongation', 'placement',
           'dropout', 'weakening', 'trend', 'loss', 'gain', 'mixing', 'baseline',
           'gradient', 'regions', 'texture', 'dependent_noise', 'independent_noise',
           'translation', 'local')
COMBINED = set(FACTORS[5:]) - {'dropout', 'loss'}
ROUNDS = ('round10', 'round2', 'round1')
CHANNELS = ('ch02', 'ch00', 'ch03', 'ch01')


def rng(component, entity=None, round_label=None, channel=None):
    descriptor = stream_descriptor(component, scene='controlled-development-v1',
        entity=entity, round_label=round_label, channel_label=channel)
    return np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(descriptor).digest(), 'big')))


def inverse_grid(points, center, vector, translation):
    """Independent scalar bisection for the preset's one RBF per round.

    q=p-t-v*a with a=exp(-||q-c||²/128). The root is bracketed by
    a in [0,1]; 55 halvings, unlike the producer's fixed-point iteration.
    """
    base = points - translation
    if not np.any(vector):
        return base
    lo = np.zeros(points.shape[:-1])
    hi = np.ones_like(lo)
    for _ in range(55):
        mid = (lo + hi) / 2
        q = base - mid[..., None] * vector
        residual = mid - np.exp(-np.sum((q-center)**2, axis=-1)/128)
        hi = np.where(residual >= 0, mid, hi)
        lo = np.where(residual < 0, mid, lo)
    result = base - ((lo+hi)/2)[..., None]*vector
    residual = result + np.exp(-np.sum((result-center)**2, axis=-1)/128)[..., None]*vector + translation - points
    assert np.max(np.abs(residual)) <= 2e-10
    return result


def expected_case(size, condition):
    """Return independent full-grid values and latent expectations for one preset."""
    shape = SIZES[size]
    factors = COMBINED if condition == 'combined' else {condition}
    centers = np.array([[shape[0]//2, 10, 10], [shape[0]//2, 22, 22]], dtype=float)
    anchor = centers[0].copy()
    if condition == 'placement':
        for i, identity in enumerate(('gt-A', 'gt-B')):
            draw = rng('placement', identity)
            assert draw.choice(1, p=[1.]) == 0
            for _ in range(10000):
                offset = draw.normal(0, [1, 2, 2], 3)
                if shape[0] == 1:
                    offset[0] = 0
                candidate = anchor + offset
                if np.all(candidate >= 0) and np.all(candidate <= np.array(shape)-1):
                    centers[i] = candidate
                    break
            else:
                raise AssertionError('oracle placement exhausted')
    amplitude = 16 if condition == 'brightness' else 8
    widths = np.array([1.5 if condition == 'axial_width' else 1,
                       2 if condition == 'lateral_width' else 1.25,
                       2 if condition == 'lateral_width' else 1.25])
    if condition == 'elongation':
        widths[1] *= 2
    intended = np.zeros((2, 4, 3))
    for i, active in enumerate(((1, 0, 3), (0, 1, 2))):
        intended[i, active, np.arange(3)] = amplitude
    pre_mix = intended.copy()
    if 'trend' in factors:
        pre_mix *= [1, .5, .25]
    if 'weakening' in factors:
        pre_mix *= [1, .25, 1]
    if 'dropout' in factors:
        pre_mix *= [1, 0, 1]
    if 'loss' in factors:
        pre_mix *= [1, 0, 0]
    if 'gain' in factors:
        pre_mix *= .5
    realized = pre_mix.copy()
    if 'mixing' in factors:
        realized[:, 0] += .25*pre_mix[:, 1]
        realized[:, 2] += .25*pre_mix[:, 3]
    points = np.moveaxis(np.indices(shape, dtype=float), 0, -1)
    images, histories = {}, []
    for r, label in enumerate(ROUNDS):
        translation = np.array(((0, 0, 0), (0, .5, -.5), (0, -1, 1))[r]) if 'translation' in factors else np.zeros(3)
        vector = np.array(((0, 0, 0), (0, 0, .5), (0, .25, 0))[r]) if 'local' in factors else np.zeros(3)
        moved = centers + np.exp(-np.sum((centers-anchor)**2, axis=1)/128)[:, None]*vector + translation
        image = np.zeros((*shape, 4))
        for i, center in enumerate(moved):
            squared_radius = np.sum(((points-center)/widths)**2, axis=-1)
            kernel = np.where(squared_radius <= 16, np.exp(-squared_radius/2), 0)
            image += kernel[..., None]*realized[i, :, r]
            histories.append(dict(amplicon_id=('gt-A', 'gt-B')[i], round_label=label,
                z=center[0], y=center[1], x=center[2],
                dropped='dropout' in factors and r == 1,
                weakened='weakening' in factors and r == 1,
                lost='loss' in factors and r >= 1,
                emitting=bool(realized[i, :, r].any()),
                center_in_bounds=bool(np.all(center >= 0) and np.all(center <= np.array(shape)-1)),
                support_intersects=bool(np.any(squared_radius <= 16)),
                support_truncated=bool(np.any(center-4*widths < 0) or np.any(center+4*widths > np.array(shape)-1))))
        reference = inverse_grid(points, anchor, vector, translation)
        background = np.zeros(shape)
        if 'gradient' in factors:
            background += np.maximum(0, 1 + 2*reference[..., 2]/(shape[2]-1))
        if 'regions' in factors:
            background += 3*np.exp(-.5*np.sum(((reference-anchor)/[2, 5, 5])**2, axis=-1))
        if 'texture' in factors:
            for i in range(2):
                center = rng('background.placement', f'blob-{i}').uniform(0, np.array(shape)-1, 3)
                background += 5*np.exp(-.5*np.sum(((reference-center)/[1, 3, 3])**2, axis=-1))
        image += background[..., None]*[1, .5, .25, 0]
        if 'baseline' in factors:
            image += [1, 2, 3, 4]
        for c, channel in enumerate(CHANNELS):
            if 'dependent_noise' in factors:
                image[..., c] += np.sqrt(.25*image[..., c])*rng('noise.dependent', round_label=label, channel=channel).standard_normal(shape)
            if 'independent_noise' in factors:
                image[..., c] += .5*rng('noise.independent', round_label=label, channel=channel).standard_normal(shape)
        images[label] = image
    return dict(images=images, centers=centers, intended=intended, pre_mix=pre_mix,
                realized=realized, histories=pd.DataFrame(histories), amplitude=amplitude)


def audit(directory, destination):
    manifest = read_json(directory/'manifest.json')
    assert manifest['version'] == 'controlled-development-v1'
    assert {c['name'] for c in manifest['cases']} == {f'{s}-{c}' for s in SIZES for c in ('clean', *FACTORS, 'combined')}
    for path, digest in manifest['files'].items():
        assert checksum(directory/path) == digest, path
    evidence = []
    for case in manifest['cases']:
        size, condition = case['name'].split('-', 1)
        oracle = expected_case(size, condition)
        root = directory/case['name']
        ext = read_json(root/'synthetic.json')['extensions']['starfinder.synthetic']
        assert ext['config_sha256'] == case['config_sha256']
        assert ext['effective_config']['split'] == 'development'
        formed = pd.read_parquet(root/'formed.parquet')
        assert formed.amplicon_id.tolist() == ['gt-A', 'gt-B']
        assert formed.gene_id.tolist() == ['gene-A', 'gene-B']
        assert formed.codeword.tolist() == ['123', '214']
        np.testing.assert_array_equal(formed[['z', 'y', 'x']], oracle['centers'])
        np.testing.assert_array_equal(formed.A, [oracle['amplitude']]*2)
        for column, value in dict(sz=1.5 if condition == 'axial_width' else 1,
                                  sl=2 if condition == 'lateral_width' else 1.25,
                                  e=2 if condition == 'elongation' else 1, theta=0).items():
            np.testing.assert_array_equal(formed[column], [value]*2)
        truth = pd.read_parquet(root/'round-truth.parquet').set_index(['amplicon_id', 'round_label']).sort_index()
        expected_truth = oracle['histories'].set_index(['amplicon_id', 'round_label']).sort_index()
        for col in expected_truth:
            if col in ('z', 'y', 'x'):
                np.testing.assert_allclose(truth[col], expected_truth[col], rtol=0, atol=1e-12)
            else:
                np.testing.assert_array_equal(truth[col], expected_truth[col])
        assert truth.first_loss_round.eq(1).all() if condition == 'loss' else truth.first_loss_round.isna().all()
        with np.load(root/'signals.npz') as saved:
            for key in ('intended', 'pre_mix', 'realized'):
                np.testing.assert_array_equal(saved[key], oracle[key], strict=True)
        loaded = load_image_checkpoint(root/'prepared')
        assert loaded.artifact['payload']['config'] == ext['requested_config']
        errors = []
        for layer in loaded.layers:
            assert layer.loaded.channel_labels == CHANNELS
            assert layer.loaded.image.dtype == np.float32
            assert layer.loaded.metadata.spacing_zyx is None
            # Float32 rounding bound scales with the independently calculated intensity.
            expected = oracle['images'][layer.round_label]
            error = np.abs(layer.loaded.image.astype(float)-expected)
            bound = np.maximum(1e-6, np.abs(expected)*np.finfo(np.float32).eps/2 + 2e-10)
            assert np.all(error <= bound), (case['name'], float(error.max()))
            errors.append(float(error.max()))
        evidence.append(dict(case=case['name'], status='pass', voxels=3*4*int(np.prod(SIZES[size])),
            max_absolute_error=max(errors), truth_rows=len(truth), formed=len(formed),
            emitting=int(truth.emitting.sum()), lost=int(truth.lost.sum()),
            invisible_support=int((~truth.support_intersects).sum()),
            config_sha256=case['config_sha256']))
    write_json(destination, dict(preset_manifest=str(directory/'manifest.json'),
        preset_manifest_sha256=checksum(directory/'manifest.json'), process=process_identity(),
        cases=evidence, tolerance='rtol=0; max(1e-6, abs(oracle)*float32_eps/2+2e-10); truth coordinates 1e-12; persistence exact',
        oracle='Published constants; full-grid Gaussian; independent SHA256/PCG64; scalar bisection inverse',
        limitations=['Development model only; no calibration, evaluation eligibility or historical provenance qualification.']))


def repeat(directory, destination):
    """Fresh-process exact comparison, distinct from the independent oracle."""
    from starfinder.synthetic import development_scene_preset, generate_formed_scene
    results = {}
    for case in read_json(directory/'manifest.json')['cases']:
        size, condition = case['name'].split('-', 1)
        book, config = development_scene_preset(condition, size=size)
        scene = generate_formed_scene(book, config=config)
        root = directory/case['name']
        assert scene.provenance == read_json(root/'synthetic.json')
        pd.testing.assert_frame_equal(scene.formed, pd.read_parquet(root/'formed.parquet'), check_exact=True)
        pd.testing.assert_frame_equal(scene.round_truth, pd.read_parquet(root/'round-truth.parquet'), check_exact=True)
        with np.load(root/'signals.npz') as saved:
            for name in saved.files:
                assert saved[name].dtype == getattr(scene, name).dtype
                assert saved[name].tobytes() == getattr(scene, name).tobytes()
        for layer in load_image_checkpoint(root/'prepared').layers:
            assert layer.loaded.image.tobytes() == scene.rounds[layer.round_label].tobytes()
            assert layer.loaded.metadata == scene.round_metadata[layer.round_label]
        results[case['name']] = scene.provenance['extensions']['starfinder.synthetic']['observation']['image_sha256']
    # Reserved splits are descriptors only: never create calibration/evaluation scenes.
    split_digests = {s: hashlib.sha256(stream_descriptor('noise.independent', split=s)).hexdigest()
                     for s in ('development', 'calibration', 'evaluation')}
    assert len(set(split_digests.values())) == 3
    book, config = development_scene_preset()
    for split in ('calibration', 'evaluation'):
        try:
            generate_formed_scene(book, config=replace(config, split=split))
        except ValueError:
            pass
        else:
            raise AssertionError('reserved split must reject generation')
    write_json(destination, dict(process=process_identity(), pythonhashseed=os.environ.get('PYTHONHASHSEED'),
        cases=results, exact_images_truth_config=True, reserved_split_digests=split_digests,
        reserved_generation='rejected', preset_manifest_sha256=checksum(directory/'manifest.json')))


if __name__ == '__main__':
    command, source, destination = sys.argv[1:]
    target = Path(destination)
    if target.exists():
        raise FileExistsError(target)
    {'audit': audit, 'repeat': repeat}[command](Path(source), target)
