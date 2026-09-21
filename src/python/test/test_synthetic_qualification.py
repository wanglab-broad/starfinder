"""Independent full-combination image oracle and retained edge populations."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from starfinder.synthetic import (ReadoutEffectsConfig, ScalarDistribution,
    development_scene_preset, formed_scene_preset, generate_formed_scene)


@pytest.mark.parametrize('size', ['small', 'z1'])
def test_combined_full_grid_independent_oracle(size, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]/'docs/examples'))
    from qualify_synthetic import expected_case
    book, config = development_scene_preset('combined', size=size)
    actual = generate_formed_scene(book, config=replace(config, dtype='float64'))
    expected = expected_case(size, 'combined')
    for label in actual.round_labels:
        # Covers every pixel/channel, including background advection, both noise
        # streams and their order. Bisection is independent of production inverse.
        np.testing.assert_allclose(actual.rounds[label], expected['images'][label],
                                   rtol=0, atol=2e-10)
    for name in ('intended', 'pre_mix', 'realized'):
        np.testing.assert_array_equal(getattr(actual, name), expected[name], strict=True)


@pytest.mark.parametrize('depth', [1, 3])
def test_overlap_boundary_invisible_and_loss_keep_full_truth(depth):
    book, base = formed_scene_preset()
    z = depth//2
    ids = ('edge', 'far', 'overlap-A', 'overlap-B')
    config = replace(base, shape_zyx=(depth, 7, 9), dtype='float64',
        coordinates=((z, 2, -.5), (z, 2, -20), (z, 4, 6), (z, 4, 6)),
        amplicon_ids=ids, gene_ids={i:'gene-A' for i in ids},
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
