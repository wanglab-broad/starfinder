"""Inputs of the bounded report example (docs/benchmark/report.md); the folder must be new and outside Git.

Writes an 8 × 16 × 16 impulse pair whose moving image is the reference shifted by (1, -2, 1) voxels, a moving image of
another shape, the truth shift, an experiment of one translation method under three conditions (the scipy and
skimage backends, and a mismatched moving image whose trial fails), the sources and metrics files that ``starfinder
experiment evaluate`` reads once the trials have run, and the report protocol. The skimage metric requests no
tolerance, so its pass is undefined; the mismatched trial's metric is skipped. No processing runs here.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def main(folder):
    folder.mkdir(parents=True, exist_ok=False)
    inputs = folder / 'inputs'
    inputs.mkdir()
    image = np.zeros((8, 16, 16), dtype=np.float32)
    image[3, 7, 8] = 10
    np.save(inputs / 'reference.npy', image)
    np.save(inputs / 'moving.npy', np.roll(image, (1, -2, 1), axis=(0, 1, 2)))
    np.save(inputs / 'moving-small.npy', image[:, :8, :8])
    (inputs / 'shift.json').write_text(json.dumps({'displacement': [1.0, -2.0, 1.0]}))
    _write(folder / 'experiment.yaml', {
        'methods': [{'module': 'registration', 'method': 'translation'}],
        'conditions': [{'name': 'scipy'}, {'name': 'skimage', 'method': {'config': {'backend': 'skimage'}}},
                       {'name': 'mismatched', 'inputs': {'moving': 'inputs/moving-small.npy'}}],
        'seeds': [42],
        'inputs': [{'reference': 'inputs/reference.npy', 'moving': 'inputs/moving.npy'}],
        'metadata': {'frame_id': 'example/reference'}})
    trials = ('scipy', 'skimage', 'mismatched')
    _write(folder / 'sources.yaml', {'sources': [
        *({'name': trial, 'kind': 'trial', 'path': f'trials/trial-{i:06d}'} for i, trial in enumerate(trials)),
        {'name': 'truth', 'kind': 'files',
         'files': {'shift': {'path': 'inputs/shift.json', 'sha256': _sha(inputs / 'shift.json')}}}]})
    _write(folder / 'metrics.yaml', {'metrics': [
        {'name': f'translation-{trial}', 'function': 'registration.evaluate_translation',
         'parameters': {'units': 'voxel', 'tolerance': None if trial == 'skimage' else 0.01},
         'arguments': {
             'detected': {'mapping': {'moving': {'source': trial, 'output': 'transform.json',
                                                 'key': 'transform.displacement_zyx'}}},
             'truth': {'mapping': {'moving': {'source': 'truth', 'file': 'shift', 'key': 'displacement'}}},
             'reference_metadata': {'source': trial, 'metadata': True},
             'observed_metadata': {'source': trial, 'metadata': True}}}
        for trial in trials]})
    _write(folder / 'protocol.yaml', {
        'title': 'Translation on an impulse pair (software example)',
        'qualification': 'Software example on a hand-built pair; not an accuracy experiment.',
        'route': 'Single-stack registration trials through starfinder trial run.',
        'inputs': 'One 8 x 16 x 16 impulse pair written by experiment_report_inputs.py.',
        'truth': 'The known shift (1, -2, 1) voxels, as a files source.',
        'dtype': 'float32',
        'conditions': {'scipy': 'The scipy backend.', 'skimage': 'The skimage backend, evaluated without a tolerance.',
                       'mismatched': 'A moving image of another shape; its trial fails.'},
        'cutoffs': [{'name': 'tolerance', 'value': 0.01, 'unit': 'voxel', 'note': 'scipy and mismatched only'}],
        'methods': [{'method': 'translation', 'purpose': 'Estimate one rigid shift between two stacks.',
                     'targeted_conditions': ['scipy', 'skimage', 'mismatched'],
                     'key_metrics': ['registration.evaluate_translation'],
                     'limitations': ['One seed and one pair.']}],
        'findings': [{'label': 'hypothesis', 'text': 'The mismatched trial fails because the two images differ in '
                      'shape.', 'source_table': 'tables/sources.csv'}],
        'figures': [{'method': 'translation', 'title': 'The scipy trial',
                     'panels': [{'label': 'reference', 'source': 'scipy', 'input': 'reference'},
                                {'label': 'moving', 'source': 'scipy', 'input': 'moving'},
                                {'label': 'registered', 'source': 'scipy', 'output': 'image.npy'}]}]})
    print(folder)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
