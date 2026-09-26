"""Independent readout-history and mixing arithmetic at rendered 3D/Z=1 centers."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd

from starfinder.barcode import Codebook, NeighborhoodSumConfig, extract_intensities
from starfinder.io import ImageLoadResult
from starfinder.spot_finding import SpotFindingResult, LocalMaximaConfig
from starfinder.synthetic import (
    FormedSceneConfig, ReadoutEffectsConfig, ScalarDistribution, generate_formed_scene,
)


def check_readout(depth=3):
    """Check exact binary-representable expectations; no renderer-derived oracle."""
    # Deliberately nonlexical labels and nonidentity color mapping.
    book = Codebook(pd.DataFrame(dict(gene_id=['gene-A'], color_sequence=['222'])),
                    ('round10', 'round2', 'round1'), ('ch02', 'ch00', 'ch03', 'ch01'),
                    {'1': 1, '2': 0, '3': 3, '4': 2})
    config = FormedSceneConfig(
        dataset_version=f'readout-contract-z{depth}-v1', shape_zyx=(depth, 7, 9),
        coordinates=((depth // 2, 3, 4),), amplicon_ids=('gt-A',),
        gene_ids={'gt-A': 'gene-A'}, brightness=ScalarDistribution(parameters=(8,)))
    trend = ReadoutEffectsConfig(trend_enabled=True, trend_base=.5)
    cases = {
        'clean': (ReadoutEffectsConfig(), [8, 8, 8]),
        'trend': (trend, [8, 4, 2]),
        'drop_recover': (replace(trend, dropout_enabled=True, dropout_probability=(0, 1, 0)), [8, 0, 2]),
        'weak_recover': (replace(trend, weakening_enabled=True, weakening_probability=(0, 1, 0),
                                 weak_factor=(1, .25, 1)), [8, 1, 2]),
        'persistent_loss': (replace(trend, loss_enabled=True, loss_probability=1, loss_start=1), [8, 0, 0]),
    }
    results = {}
    for name, (effects, expected) in cases.items():
        scene = generate_formed_scene(book, config=replace(config, readout=effects))
        np.testing.assert_array_equal(scene.intended[0, 0], [8, 8, 8])
        np.testing.assert_array_equal(scene.realized[0, 0], expected)
        assert len(scene.formed) == 1 and len(scene.round_truth) == 3
        assert scene.formed.codeword.tolist() == ['222']
        np.testing.assert_array_equal([a[depth // 2, 3, 4, 0] for a in scene.rounds.values()], expected)
        results[name] = scene
    assert results['drop_recover'].round_truth.dropped.tolist() == [False, True, False]
    assert results['drop_recover'].round_truth.first_loss_round.isna().all()
    assert results['persistent_loss'].round_truth.lost.tolist() == [False, True, True]
    assert results['persistent_loss'].round_truth.first_loss_round.tolist() == [1, 1, 1]
    assert results['persistent_loss'].round_truth.support_intersects.all()
    matrix = np.repeat(np.eye(4)[None], 3, axis=0)
    matrix[:, 1, 0] = .25
    mixing = ReadoutEffectsConfig(mixing_enabled=True, mixing=matrix)
    mixed = generate_formed_scene(book, config=replace(config, readout=mixing))
    np.testing.assert_array_equal(mixed.realized[0], [[8]*3, [2]*3, [0]*3, [0]*3])
    combined = generate_formed_scene(book, config=replace(config, readout=replace(
        mixing, gain_enabled=True, gains=np.full((3, 4), .5), trend_enabled=True,
        trend_base=.5, weakening_enabled=True, weakening_probability=(0, 1, 0), weak_factor=(1, .25, 1))))
    expected = np.array([[4, .5, 1], [1, .125, .25], [0, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(combined.realized[0], expected)
    spots = SpotFindingResult(pd.DataFrame({'spot_id': pd.Series(['spot-1'], dtype='string'),
                                           'z': [float(depth // 2)], 'y': [3.], 'x': [4.]}),
                              combined.metadata, 'detected/readout', LocalMaximaConfig(), {})
    rounds = {label: ImageLoadResult(image, combined.metadata, combined.channel_labels, ())
              for label, image in combined.rounds.items()}
    extracted = extract_intensities(rounds, spots, config=NeighborhoodSumConfig((0, 0, 0)))
    np.testing.assert_array_equal(extracted.values[0], expected)
    results.update(mixed=mixed, combined=combined)
    return results


if __name__ == '__main__':
    summary = {}
    for depth in (3, 1):
        scenes = check_readout(depth)
        summary[str(depth)] = {name: {'signals': scene.realized.tolist(),
                                     'provenance': scene.provenance,
                                     'history': json.loads(scene.round_truth.to_json(orient='records'))}
                               for name, scene in scenes.items()}
    print(json.dumps(summary, indent=2))
