"""Bounded one-case CLI lifecycle; output must be a new external directory."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / 'inputs'
    inputs.mkdir()
    image = np.zeros((8, 16, 16), dtype=np.float32)
    image[3, 7, 8] = 10
    np.save(inputs / 'reference.npy', image)
    np.save(inputs / 'moving.npy', np.roll(image, (1, -2, 1), axis=(0, 1, 2)))
    (inputs / 'correction.json').write_text('[-1.0, 2.0, -1.0]')
    config = {'schema_version': 1, 'repetitions': 1,
        'provenance': {'seed': None, 'fixture': 'deterministic impulse; no RNG',
                       'limitations': 'software smoke only; no scientific qualification'},
        'cases': [{'case_id': 'tiny', 'task': 'registration',
            'inputs': {'reference': 'reference.npy', 'moving': 'moving.npy'},
            'truth': {'correction': 'correction.json'},
            'config': {'registration': {'method': 'translation'},
                'reference_metadata': {'frame_id': 'reference'},
                'moving_metadata': {'frame_id': 'moving'},
                'evaluation': {'ncc': True, 'translation': {'tolerance': 0.01}}}}]}
    config_path = output / 'cases.json'
    config_path.write_text(json.dumps(config, indent=2))
    commands = []
    def command(*args):
        argv = [sys.executable, '-m', 'starfinder', *map(str, args)]
        completed = subprocess.run(argv, capture_output=True, text=True, check=True)
        commands.append({'argv': argv, 'exit_code': completed.returncode,
                         'stdout': completed.stdout, 'stderr': completed.stderr})
        return Path(completed.stdout.strip())
    run = command('benchmark', 'run', '--config', config_path, '--input-root', inputs,
                  '--output-root', output / 'runs', '--owner', 'Jiahao')
    trial_path = run / 'tiny/0000/trial.json'
    before = hashlib.sha256(trial_path.read_bytes()).hexdigest()
    evaluation = command('benchmark', 'evaluate', '--run-dir', run)
    report = command('benchmark', 'report', '--evaluation-dir', evaluation)
    records = json.loads((evaluation / 'results.json').read_text())
    assert records[0]['metrics']['translation']['values']['passed'] is True
    assert np.isclose(records[0]['metrics']['ncc_registered']['values']['ncc'], 1)
    assert hashlib.sha256(trial_path.read_bytes()).hexdigest() == before
    (output / 'commands.json').write_text(json.dumps(commands, indent=2))
    print(report)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
