"""Stream-key isolation, cross-process repeatability and package boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from starfinder.synthetic import GeometryConfig, NoiseConfig, ReadoutEffectsConfig

from .amplicon_repeatability import every_effect, paths
from .formed_oracle import COMPONENTS, generator, stream_descriptor, stream_digest

pytestmark = pytest.mark.synthetic


def test_descriptor_bytes():
    assert stream_descriptor('count') == (
        b'["starfinder.synthetic/1","development",42,"formed-v1","count",null,null,null]')


def test_namespace_separation():
    base = stream_digest('placement')
    variants = [stream_digest('identity'), stream_digest('placement', seed=43)]
    for key, value in (('split', 'calibration'), ('split', 'evaluation'),
                       ('scene', 'other'), ('entity', '0'),
                       ('round_label', 'round2'), ('channel_label', 'ch00')):
        variants.append(stream_digest('placement', **{key: value}))
    assert len({base, *variants}) == 9
    # JSON field boundaries avoid collisions from concatenating labels.
    assert stream_digest('placement', scene='ab', entity='c') != stream_digest(
        'placement', scene='a', entity='bc')


def test_every_component_uses_its_documented_key():
    scene = every_effect()
    streams = scene.provenance['stream_scheme']['streams']
    assert {s[4] for s in streams} == set(COMPONENTS)
    assert len({tuple(s) for s in streams}) == len(streams)
    # Replay the placement and noise draws from the independent key derivation.
    for row in scene.formed.itertuples():
        expected = generator('placement', entity=row.amplicon_id).uniform(0, [2, 8, 10], 3)
        np.testing.assert_array_equal([row.z, row.y, row.x], expected)
    count = generator('count').poisson(.02 * 3 * 9 * 11)
    assert len(scene.formed) == count


def test_unrelated_changes_preserve_other_streams():
    base = every_effect()
    # Noise strength, readout probabilities and geometry magnitude each change
    # one factor; formed latents and every standardized draw key stay fixed.
    for changes in (dict(noise=NoiseConfig(True, 3, True, 2)),
                    dict(readout=ReadoutEffectsConfig()),
                    dict(geometry=GeometryConfig())):
        other = every_effect(**changes)
        pd.testing.assert_frame_equal(base.formed, other.formed)
        assert (base.provenance['effective_config']['background']['components']
                == other.provenance['effective_config']['background']['components'])
        kept = {tuple(s) for s in base.provenance['stream_scheme']['streams']
                if s[4].startswith(('noise.', 'background.', 'placement', 'identity'))}
        assert kept <= {tuple(s) for s in other.provenance['stream_scheme']['streams']}
    # Changing the scene key intentionally redraws the whole scene.
    moved = every_effect(scene_key='another-scene')
    assert not base.formed[['z', 'y', 'x']].equals(moved.formed[['z', 'y', 'x']])


@pytest.mark.slow
def test_cross_process_repeatability_with_hash_seeds(tmp_path):
    # One child per hash seed generates every path of amplicon_repeatability (the every-effect scene, the six
    # presets and development fixtures, the singleton random geometry, background and noise, readout trend and
    # mixing, and the bounded dataset) and prints one JSON document per path.
    code = '''
import sys
sys.path.insert(0, sys.argv[1])
from test.amplicon_repeatability import main
main()
'''
    root = str(Path(__file__).resolve().parents[1])
    outputs = []
    for value in ('1', '7654321'):
        run = subprocess.run([sys.executable, '-c', code, root], cwd=tmp_path, capture_output=True, text=True,
                             timeout=600, env=dict(os.environ, PYTHONHASHSEED=value))
        assert run.returncode == 0, f'the child with PYTHONHASHSEED={value} failed:\n{run.stderr}'
        lines = run.stdout.splitlines()
        outputs.append(({json.loads(line)[0]: line for line in lines[:-1]}, json.loads(lines[-1])))
    # Generating writes no file, for every path.
    assert list(tmp_path.iterdir()) == []
    (documents, probe), (other, other_probe) = outputs
    # The Python hash differs between processes; generated content does not.
    assert probe[0] == other_probe[0] == 'probe' and probe[1] != other_probe[1]
    assert list(documents) == list(other) == list(paths())
    for path in documents:
        assert documents[path] == other[path], f'path {path} differs between the two processes'
    scenes = [json.loads(line)[1] for path, line in documents.items() if path.startswith('scene/')]
    assert len(scenes) == 6
    # Clean fixtures have explicit coordinates and constant/supplied properties,
    # so they draw nothing; every recorded stream uses the documented namespace.
    streams = [s for scene in scenes for s in scene['provenance']['stream_scheme']['streams']]
    assert streams and all(s[0] == 'starfinder.synthetic/1' for s in streams)
    assert not json.loads(documents['scene/z1-clean'])[1]['provenance']['stream_scheme']['streams']


@pytest.mark.parametrize('keys', [
    {'seed': True}, {'seed': -1}, {'seed': 2**64}, {'seed': 1.5},
    {'split': ''}, {'scene': ''}, {'scene': None}, {'entity': 'é'},
    {'round_label': 1},
])
def test_invalid_reference_stream_keys(keys):
    with pytest.raises(ValueError):
        stream_descriptor('placement', **keys)


def test_package_imports_no_persistence_or_checkpoint_modules():
    code = '''
import sys, tempfile
import starfinder.synthetic as s
book, config = s.development_scene_preset('combined', size='z1')
with tempfile.TemporaryDirectory() as root:
    s.save_amplicon_scene(s.generate_amplicon_scene(book, config=config), root + '/fixture')
bad = sorted(m for m in sys.modules if m == 'starfinder.provenance' or m.startswith(
    ('starfinder.provenance.', 'starfinder.artifacts', 'starfinder.io.checkpoints',
     'starfinder.io.candidates', 'starfinder.io.molecules')))
print(bad)
'''
    assert subprocess.check_output([sys.executable, '-c', code], text=True, timeout=120).strip() == '[]'
