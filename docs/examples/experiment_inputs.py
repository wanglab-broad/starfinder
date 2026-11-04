"""Inputs of the bounded experiment example (docs/benchmark/experiments.md); the folder must be new and outside Git.

Writes an 8 × 16 × 16 impulse pair whose moving image is the reference shifted by (1, -2, 1) voxels, the truth shift,
an experiment of one translation method under two conditions (the scipy and skimage backends), and the sources and
metrics files that ``starfinder experiment evaluate`` reads once both trials have run. No processing runs here.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(folder):
    folder.mkdir(parents=True, exist_ok=False)
    inputs = folder / 'inputs'
    inputs.mkdir()
    image = np.zeros((8, 16, 16), dtype=np.float32)
    image[3, 7, 8] = 10
    np.save(inputs / 'reference.npy', image)
    np.save(inputs / 'moving.npy', np.roll(image, (1, -2, 1), axis=(0, 1, 2)))
    (inputs / 'shift.json').write_text(json.dumps({'displacement': [1.0, -2.0, 1.0]}))
    experiment = {'methods': [{'module': 'registration', 'method': 'translation'}],
                  'conditions': [{'name': 'scipy'}, {'name': 'skimage', 'method': {'config': {'backend': 'skimage'}}}],
                  'seeds': [42],
                  'inputs': [{'reference': 'inputs/reference.npy', 'moving': 'inputs/moving.npy'}],
                  'metadata': {'frame_id': 'example/reference'}}
    (folder / 'experiment.yaml').write_text(yaml.safe_dump(experiment, sort_keys=False))
    sources = {'sources': [
        {'name': 'scipy', 'kind': 'trial', 'path': 'trials/trial-000000'},
        {'name': 'skimage', 'kind': 'trial', 'path': 'trials/trial-000001'},
        {'name': 'truth', 'kind': 'files',
         'files': {'shift': {'path': 'inputs/shift.json', 'sha256': _sha(inputs / 'shift.json')}}}]}
    (folder / 'sources.yaml').write_text(yaml.safe_dump(sources, sort_keys=False))
    metrics = {'metrics': [
        {'name': f'translation-{trial}', 'function': 'registration.evaluate_translation',
         'parameters': {'units': 'voxel', 'tolerance': 0.01},
         'arguments': {
             'detected': {'mapping': {'moving': {'source': trial, 'output': 'transform.json',
                                                 'key': 'transform.displacement_zyx'}}},
             'truth': {'mapping': {'moving': {'source': 'truth', 'file': 'shift', 'key': 'displacement'}}},
             'reference_metadata': {'source': trial, 'metadata': True},
             'observed_metadata': {'source': trial, 'metadata': True}}}
        for trial in ('scipy', 'skimage')]}
    (folder / 'metrics.yaml').write_text(yaml.safe_dump(metrics, sort_keys=False))
    print(folder)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
