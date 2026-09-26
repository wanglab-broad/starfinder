"""The four packaged development fixtures against an independent full-grid oracle.

Fixtures are generated once per session, saved as TIFF images and CSV truth,
reloaded from disk and compared with formed_oracle, which imports no producer
code. Retained edge populations use hand-built literals.
"""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from starfinder.io import ImageLoadConfig, load_volume
from starfinder.synthetic import (DEVELOPMENT_FIXTURES, ReadoutEffectsConfig, ScalarDistribution,
                                  development_scene_preset, formed_scene_preset,
                                  generate_formed_scene, save_formed_scene)

from .formed_oracle import CHANNELS, FIXTURES, ROUNDS, expected_case


@pytest.fixture(scope='module')
def fixture_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('development-fixtures')
    for name, (condition, size) in DEVELOPMENT_FIXTURES.items():
        book, config = development_scene_preset(condition, size=size)
        save_formed_scene(generate_formed_scene(book, config=config), root / name)
    return root


def load_round(path):
    channels = [load_volume(path, config=ImageLoadConfig(channel_labels=(label,), channel_index=c))
                for c, label in enumerate(CHANNELS)]
    return np.stack([result.image for result in channels], axis=-1), channels[0].metadata


def test_fixture_names_match_oracle():
    assert tuple(DEVELOPMENT_FIXTURES) == FIXTURES


@pytest.mark.parametrize('name', FIXTURES)
def test_saved_fixture_matches_independent_oracle(fixture_root, name):
    size, condition = name.split('-')
    oracle = expected_case(size, condition)
    root = fixture_root / name
    provenance = json.loads((root / 'provenance.json').read_text())
    assert provenance['effective_config']['dataset_version'] == f'controlled-development-v1-{name}'
    for label in ROUNDS:
        image, metadata = load_round(root / 'images' / f'{label}.tif')
        assert image.dtype == np.float32 and image.shape == oracle['images'][label].shape
        assert hashlib.sha256(image.tobytes()).hexdigest() == provenance['image_sha256'][label]
        expected_frame = [t['destination_frame'] for t in provenance['transforms'].values()
                          if t['round_label'] == label]
        assert [metadata.frame_id] == expected_frame
        # Every voxel/channel: float32 rounding of the float64 oracle plus the
        # geometry inverse tolerance; nothing is renderer-derived.
        expected = oracle['images'][label]
        bound = np.maximum(1e-6, np.abs(expected) * np.finfo(np.float32).eps / 2 + 2e-10)
        assert np.all(np.abs(image.astype(np.float64) - expected) <= bound), label
    formed = pd.read_csv(root / 'formed.csv')
    for column in ('amplicon_id', 'gene_id', 'formed_index'):
        assert formed[column].tolist() == oracle['formed'][column].tolist()
    assert formed.codeword.astype(str).tolist() == ['123', '214']
    for column in ('z', 'y', 'x', 'A', 'sz', 'sl', 'e', 'theta'):
        np.testing.assert_array_equal(formed[column], oracle['formed'][column], err_msg=column)
    assert formed.theta.iloc[1] != 0
    truth = pd.read_csv(root / 'round_truth.csv').set_index(['amplicon_id', 'round_label']).sort_index()
    expected_truth = oracle['histories'].set_index(['amplicon_id', 'round_label']).sort_index()
    assert truth.index.equals(expected_truth.index)
    for column in expected_truth:
        if column in ('z', 'y', 'x'):
            np.testing.assert_allclose(truth[column], expected_truth[column], rtol=0, atol=1e-12)
        else:
            np.testing.assert_array_equal(truth[column], expected_truth[column], err_msg=column)
    assert truth.first_loss_round.isna().all()
    signals = pd.read_csv(root / 'signals.csv')
    assert len(signals) == 2 * 4 * 3
    for row in signals.itertuples():
        i, c, r = (('gt-A', 'gt-B').index(row.amplicon_id), CHANNELS.index(row.channel_label),
                   ROUNDS.index(row.round_label))
        for key in ('intended', 'pre_mix', 'realized'):
            assert getattr(row, key) == oracle[key][i, c, r], (key, i, c, r)


@pytest.mark.parametrize('size', ['small', 'z1'])
def test_combined_in_memory_float64_oracle(size):
    book, config = development_scene_preset('combined', size=size)
    actual = generate_formed_scene(book, config=replace(config, dtype='float64'))
    expected = expected_case(size, 'combined')
    for label in actual.round_labels:
        # Covers every voxel/channel, background advection, both noise streams
        # and their order. Bisection is independent of the production inverse.
        np.testing.assert_allclose(actual.rounds[label], expected['images'][label], rtol=0, atol=2e-10)
    for name in ('intended', 'pre_mix', 'realized'):
        np.testing.assert_array_equal(getattr(actual, name), expected[name], strict=True)


def test_save_refuses_nonempty_directory_and_unsafe_labels(tmp_path):
    book, config = formed_scene_preset('formed-z1-v1')
    scene = generate_formed_scene(book, config=replace(config, count=0, shape_zyx=(1, 2, 2)))
    (tmp_path / 'used').mkdir()
    (tmp_path / 'used' / 'keep.txt').write_text('x')
    with pytest.raises(FileExistsError):
        save_formed_scene(scene, tmp_path / 'used')
    assert (tmp_path / 'used' / 'keep.txt').read_text() == 'x'
    saved = save_formed_scene(scene, tmp_path / 'empty')
    assert sorted(p.name for p in saved.rglob('*') if p.is_file()) == [
        'formed.csv', 'provenance.json', 'round1.tif', 'round10.tif', 'round2.tif',
        'round_truth.csv', 'signals.csv']
    scene.round_labels = ('../escape', 'round2', 'round1')
    with pytest.raises(ValueError, match='file name'):
        save_formed_scene(scene, tmp_path / 'unsafe')
    assert not (tmp_path / 'unsafe').exists()


@pytest.mark.parametrize('depth', [1, 3])
def test_overlap_boundary_invisible_and_loss_keep_full_truth(depth):
    book, base = formed_scene_preset()
    z = depth//2
    ids = ('edge', 'far', 'overlap-A', 'overlap-B')
    config = replace(base, shape_zyx=(depth, 7, 9), dtype='float64',
        coordinates=((z, 2, -.5), (z, 2, -20), (z, 4, 6), (z, 4, 6)),
        amplicon_ids=ids, gene_ids={i: 'gene-A' for i in ids},
        brightness=ScalarDistribution(parameters=(8,)),
        readout=ReadoutEffectsConfig(dropout_enabled=True, dropout_probability=(0, 1, 0),
            loss_enabled=True, loss_probability=1, loss_start=2))
    scene = generate_formed_scene(book, config=config)
    assert len(scene.formed) == 4 and len(scene.round_truth) == 12
    assert scene.round_truth.groupby('amplicon_id').size().eq(3).all()
    edge = scene.round_truth[scene.round_truth.amplicon_id == 'edge']
    far = scene.round_truth[scene.round_truth.amplicon_id == 'far']
    assert not edge.center_in_bounds.any() and edge.support_intersects.all()
    assert not far.support_intersects.any()
    assert scene.round_truth[scene.round_truth.round_index == 1].dropped.all()
    assert scene.round_truth[scene.round_truth.round_index == 2].lost.all()
    assert scene.round_truth.first_loss_round.eq(2).all()
    assert scene.rounds['round10'][z, 2, 0, 1] == pytest.approx(7.059975220676764, abs=1e-12)
    assert scene.rounds['round10'][z, 4, 6, 1] == 16
    assert not scene.rounds['round2'].any() and not scene.rounds['round1'].any()
    assert scene.intended.sum() == 4*3*8
    # N=0 retains typed empty truth and NCR arrays with dropout and loss enabled.
    empty = generate_formed_scene(book, config=replace(config, coordinates=(), amplicon_ids=(), gene_ids={}))
    assert empty.intended.shape == (0, 4, 3)
    assert len(empty.formed) == len(empty.round_truth) == 0
    assert empty.round_truth.lost.dtype == bool
    assert all(not image.any() for image in empty.rounds.values())
