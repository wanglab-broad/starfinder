"""Stream-key isolation, cross-process repeatability and package boundaries."""
from dataclasses import replace
import os
import re
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from starfinder.synthetic import (BackgroundConfig, GeometryConfig, NoiseConfig,
                                  ReadoutEffectsConfig, ScalarDistribution, TextureConfig,
                                  formed_scene_preset, generate_formed_scene)

from .formed_oracle import COMPONENTS, generator, stream_descriptor, stream_digest

PACKAGE = Path(__file__).resolve().parents[1] / 'starfinder/synthetic'
DOCS = Path(__file__).resolve().parents[3] / 'docs'


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


def every_effect(**changes):
    """A scene drawing from every registered component."""
    book, config = formed_scene_preset()
    config = replace(config, shape_zyx=(3, 9, 11), density=.02, dtype='float64',
        brightness=ScalarDistribution('lognormal', (2, .2)),
        axial_width=ScalarDistribution('uniform', (.8, 1.2)),
        lateral_width=ScalarDistribution('uniform', (.8, 1.2)),
        elongation=ScalarDistribution('folded_lognormal', (0, .2)),
        angle=ScalarDistribution('uniform', (0, np.pi)),
        readout=ReadoutEffectsConfig(dropout_enabled=True, dropout_probability=(.3, .3, .3),
            weakening_enabled=True, weakening_probability=(.3, .3, .3), weak_factor=(.5, .5, .5),
            loss_enabled=True, loss_probability=.3),
        background=BackgroundConfig(texture_enabled=True, tissue_weights=np.ones((3, 4)),
            texture=TextureConfig(density=.01, axial_width=ScalarDistribution('uniform', (1, 2)),
                                  brightness=ScalarDistribution('uniform', (2, 4)))),
        noise=NoiseConfig(True, .5, True, .5),
        geometry=GeometryConfig(translation_enabled=True, translation_max_zyx=(.2, .5, .5),
                                local_enabled=True, centers_zyx=((1, 4, 5),), strength=.05,
                                affine_enabled=True, affine_max_zyx=(.01, .05, .05),
                                polynomial_enabled=True, polynomial_max_zyx=(.01, .05, .05)))
    return generate_formed_scene(book, config=replace(config, **changes))


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


def test_cross_process_repeatability_with_hash_seeds():
    code = '''
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
from test.test_formed_streams import every_effect
s = every_effect()
h = hashlib.sha256()
for a in [*s.rounds.values(), s.intended, s.pre_mix, s.realized]: h.update(a.tobytes())
for t in [s.formed, s.round_truth]: h.update(t.to_json(orient='table').encode())
h.update(json.dumps(s.provenance, sort_keys=True).encode())
print(h.hexdigest(), hash('probe'))
'''
    root = str(Path(__file__).resolve().parents[1])
    outputs = [subprocess.check_output([sys.executable, '-c', code, root], text=True, timeout=120,
                                       env=dict(os.environ, PYTHONHASHSEED=value)).split()
               for value in ('1', '7654321')]
    # The Python hash differs between processes; generated content does not.
    assert outputs[0][1] != outputs[1][1]
    assert outputs[0][0] == outputs[1][0]


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
    s.save_formed_scene(s.generate_formed_scene(book, config=config), root + '/fixture')
bad = sorted(m for m in sys.modules if m == 'starfinder.provenance' or m.startswith(
    ('starfinder.provenance.', 'starfinder.artifacts', 'starfinder.io.checkpoints',
     'starfinder.io.candidates', 'starfinder.io.molecules')))
print(bad)
'''
    assert subprocess.check_output([sys.executable, '-c', code], text=True, timeout=120).strip() == '[]'


def test_no_artifact_contract_or_issue_references():
    sources = list(PACKAGE.glob('*.py'))
    docs = [DOCS / 'synthetic-specification.md', DOCS / 'api/synthetic.rst',
            *(DOCS / 'examples' / f'{name}.py' for name in
              ('formed_scene', 'readout_effects', 'background_noise', 'formed_geometry'))]
    for path in sources + docs:
        text = path.read_text()
        assert 'starfinder.artifacts/1' not in text, path
        assert 'starfinder.provenance' not in text and 'RunRecorder' not in text, path
    for path in docs:
        assert not re.search(r'\bW-\d+', path.read_text()), path
