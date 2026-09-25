"""Independent readout expectations and component invariance."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from starfinder.synthetic import (ReadoutEffectsConfig, ScalarDistribution,
                                  formed_scene_preset, generate_formed_scene)

EXAMPLE = Path(__file__).resolve().parents[3] / 'docs/examples/readout_effects.py'


def generate(effects=ReadoutEffectsConfig(), **kwargs):
    book, config = formed_scene_preset()
    return generate_formed_scene(book, config=replace(config, readout=effects, **kwargs))


@pytest.mark.parametrize('depth', [3, 1])
def test_a4_a5_example(depth):
    runpy.run_path(str(EXAMPLE))['check_readout'](depth)


def test_disabled_controls_and_identity_mixing():
    baseline = generate()
    disabled = ReadoutEffectsConfig(dropout_probability=(1, 1, 1), weakening_probability=(1, 1, 1),
                                   weak_factor=(0, 0, 0), trend_base=0, loss_probability=1,
                                   loss_start=0, gains=np.zeros((3, 4)), mixing=np.zeros((3, 4, 4)))
    scenes = [generate(disabled), generate(ReadoutEffectsConfig(
        dropout_enabled=True, weakening_enabled=True, loss_enabled=True, trend_enabled=True,
        gain_enabled=True, mixing_enabled=True))]
    for scene in scenes:
        pd.testing.assert_frame_equal(scene.formed, baseline.formed)
        pd.testing.assert_frame_equal(scene.round_truth, baseline.round_truth)
        for name in ('intended', 'pre_mix', 'realized'):
            np.testing.assert_array_equal(getattr(scene, name), getattr(baseline, name))
        for label in scene.rounds:
            np.testing.assert_array_equal(scene.rounds[label], baseline.rounds[label])
    payload = scenes[0].provenance
    assert payload['requested_config']['readout']['loss_probability'] == 1
    assert payload['effective_config']['readout']['loss_probability'] == 0
    assert not payload['effective_config']['effects_enabled']
    assert payload['stream_scheme'] == baseline.provenance['stream_scheme']


def test_full_state_overlap_loss_endpoints_and_empty():
    effects = ReadoutEffectsConfig(dropout_enabled=True, dropout_probability=(0, 1, 0),
        weakening_enabled=True, weakening_probability=(0, 1, 0), weak_factor=(1, .25, 1),
        loss_enabled=True, loss_probability=1, loss_start=1, trend_enabled=True, trend_base=0)
    scene = generate(effects, count=1)
    truth = scene.round_truth
    assert truth.dropped.tolist() == [False, True, False]
    assert truth.weakened.tolist() == [False, True, False]
    assert truth.lost.tolist() == [False, True, True]
    assert truth.weak_multiplier.tolist() == [1, .25, 1]
    assert truth.trend_multiplier.tolist() == [1, 0, 0]  # 0**0 is 1
    assert truth.emitting.tolist() == [True, False, False]
    for start, expected in [(0, [True]*3), (2, [False, False, True])]:
        assert generate(replace(effects, loss_start=start), count=1).round_truth.lost.tolist() == expected
    empty = generate(effects, count=0)
    assert empty.realized.shape == (0, 4, 3)
    assert str(empty.round_truth.first_loss_round.dtype) == 'Int64'
    assert empty.round_truth.lost.dtype == bool
    assert not np.shares_memory(scene.intended, scene.pre_mix)
    assert not np.shares_memory(scene.pre_mix, scene.realized)


def test_independent_random_laws_isolation_and_scheduling():
    effects = ReadoutEffectsConfig(dropout_enabled=True, dropout_probability=(.2, .5, .8),
        weakening_enabled=True, weakening_probability=(.3, .7, .4), weak_factor=(.2, .4, .6),
        loss_enabled=True, loss_probability=.5, loss_start=1)
    scene = generate(effects, brightness=ScalarDistribution('lognormal', (3, .2)))
    # Derive draws from the frozen descriptor, independently of production helpers.
    def draw(component, identity, label=None):
        descriptor = ['starfinder.synthetic/1', 'development', 42, 'formed-v1',
                      component, identity, label, None]
        seed = int.from_bytes(hashlib.sha256(json.dumps(descriptor, ensure_ascii=False,
                              separators=(',', ':')).encode()).digest(), 'big')
        return np.random.Generator(np.random.PCG64(seed)).random()
    for row in scene.round_truth.itertuples():
        r = row.round_index
        assert row.dropped == (draw('round.dropout', row.amplicon_id, row.round_label) < (.2, .5, .8)[r])
        assert row.weakened == (draw('round.weakening', row.amplicon_id, row.round_label) < (.3, .7, .4)[r])
        selected = draw('round.loss', row.amplicon_id) < .5
        assert row.lost == (selected and r >= 1)
        assert (row.first_loss_round == 1) if selected else pd.isna(row.first_loss_round)
    changed = generate(replace(effects, dropout_probability=(1, 1, 1), weak_factor=(0, 0, 0),
                              gain_enabled=True, gains=np.full((3, 4), 2)),
                       brightness=ScalarDistribution('lognormal', (3, .2)))
    pd.testing.assert_frame_equal(scene.formed, changed.formed)
    pd.testing.assert_frame_equal(scene.round_truth[['weakened', 'lost', 'first_loss_round']],
                                  changed.round_truth[['weakened', 'lost', 'first_loss_round']])
    assert scene.provenance['stream_scheme'] == changed.provenance['stream_scheme']
    for ids in [scene.amplicon_ids[::-1], scene.amplicon_ids + ('extra',)]:
        other = generate(effects, count=len(ids), amplicon_ids=ids,
                         brightness=ScalarDistribution('lognormal', (3, .2)))
        keys = ['amplicon_id', 'round_label']
        expected = scene.round_truth.set_index(keys).sort_index()
        pd.testing.assert_frame_equal(expected, other.round_truth.set_index(keys).loc[expected.index])
        if len(ids) == len(scene.amplicon_ids):
            for label in scene.rounds:
                np.testing.assert_array_equal(scene.rounds[label], other.rounds[label])


@pytest.mark.parametrize('kwargs', [
    {'dropout_enabled': 1}, {'mixing_enabled': 'yes'},
    {'dropout_probability': (.1,)}, {'weakening_probability': (0, -1, 0)},
    {'weak_factor': (1, 2, 1)}, {'trend_base': 1.1}, {'trend_base': float('nan')},
    {'trend_base': [1]}, {'loss_probability': -1}, {'loss_probability': True},
    {'loss_start': True}, {'loss_start': 3}, {'loss_start': -1},
    {'gains': np.ones((4, 3))}, {'gains': np.full((3, 4), np.inf)},
    {'mixing': np.eye(4)}, {'mixing': np.full((3, 4, 4), -1)},
    {'mixing': np.full((3, 4, 4), np.nan)},
])
def test_invalid_even_disabled_empty(kwargs):
    with pytest.raises(ValueError):
        generate(ReadoutEffectsConfig(**kwargs), count=0)


def test_overflow_and_zero_mixing():
    with pytest.raises(ValueError, match='nonfinite readout'):
        generate(ReadoutEffectsConfig(gain_enabled=True, gains=np.full((3, 4), 1e308)))
    scene = generate(ReadoutEffectsConfig(mixing_enabled=True, mixing=np.zeros((3, 4, 4))))
    assert scene.intended.any() and scene.pre_mix.any() and not scene.realized.any()
    assert not scene.round_truth.emitting.any() and not scene.round_truth.lost.any()


def test_cross_process_repeatability():
    code = '''
import hashlib, json
from starfinder.synthetic import *
from dataclasses import replace
b,c = formed_scene_preset()
e = ReadoutEffectsConfig(dropout_enabled=True, dropout_probability=(.2,.5,.8),
    weakening_enabled=True, weakening_probability=(.3,.7,.4), weak_factor=(.2,.4,.6),
    loss_enabled=True, loss_probability=.5, trend_enabled=True, trend_base=.9,
    mixing_enabled=True, mixing=[[[1,.1,0,0],[.2,1,0,0],[0,0,1,.3],[0,0,0,1]]]*3)
s = generate_formed_scene(b, config=replace(c,readout=e))
h=hashlib.sha256()
for a in [*s.rounds.values(),s.intended,s.pre_mix,s.realized]: h.update(a.tobytes())
for t in [s.formed,s.round_truth]: h.update(t.to_json(orient='table').encode())
h.update(json.dumps(s.provenance,sort_keys=True).encode())
print(h.hexdigest())
'''
    outputs = [subprocess.check_output([sys.executable, '-c', code],
               env=dict(os.environ, PYTHONHASHSEED=value), text=True) for value in ('1', '987')]
    assert outputs[0] == outputs[1]


def test_acquisition_order_mapping_and_single_round_loss_default():
    from starfinder.barcode import Codebook
    book, config = formed_scene_preset()
    matrix = np.repeat(np.eye(4)[None], 3, axis=0)
    matrix[0, 2, 1] = .5
    matrix[1, 3, 0] = .25
    matrix[2, 0, 3] = 2
    scene = generate_formed_scene(book, config=replace(config, count=1,
        gene_ids={'amplicon-0': 'gene-A'}, brightness=ScalarDistribution(parameters=(8,)),
        readout=ReadoutEffectsConfig(mixing_enabled=True, mixing=matrix,
            gain_enabled=True, gains=((1, .5, 1, 1), (.25, 1, 1, 1), (1, 1, 1, 2)))))
    np.testing.assert_array_equal(scene.intended[0], [[0, 8, 0], [8, 0, 0], [0, 0, 0], [0, 0, 8]])
    np.testing.assert_array_equal(scene.pre_mix[0], [[0, 2, 0], [4, 0, 0], [0, 0, 0], [0, 0, 16]])
    np.testing.assert_array_equal(scene.realized[0], [[0, 2, 32], [4, 0, 0], [2, 0, 0], [0, .5, 16]])
    one = Codebook(pd.DataFrame({'gene_id': ['gene-A'], 'color_sequence': ['1']}),
                   ('only',), book.channel_labels, book.color_to_channel)
    lost = generate_formed_scene(one, config=replace(config, count=1,
        readout=ReadoutEffectsConfig(loss_enabled=True, loss_probability=1)))
    assert lost.round_truth.first_loss_round.tolist() == [0]
    assert lost.round_truth.lost.tolist() == [True]
    assert len(lost.formed) == 1 and not lost.realized.any()
