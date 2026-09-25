"""Shared molecule/background geometry, with independent expected values."""
from dataclasses import replace
import json

import numpy as np

from starfinder.synthetic import (BackgroundConfig, GeometryConfig,
    ScalarDistribution, formed_scene_preset, generate_formed_scene)


def run():
    records = []
    for depth in (3, 1):
        z = depth // 2
        cb, cfg = formed_scene_preset()
        center = (z, 2, 3)
        geometry = GeometryConfig(translation_enabled=True,
            translations_zyx=((0, 0, .5), (0, 0, 0), (0, 0, -3.5)),
            local_enabled=True, centers_zyx=(center,), scales=(2,),
            vectors_zyx=(((0, 0, .25),), ((0, 0, 0),), ((0, 0, 0),)))
        background = BackgroundConfig(regions_enabled=True, region_centers=(center,),
            region_sigma_zyx=((1, 1, 1),), region_heights=(8,),
            tissue_weights=((1, 0, 0, 0),)*3)
        cfg = replace(cfg, dataset_version=f'geometry-contract-z{depth}-v1',
            shape_zyx=(depth, 7, 9), dtype='float64', coordinates=(center,),
            amplicon_ids=('gt-A',), gene_ids={'gt-A': 'gene-A'},
            brightness=ScalarDistribution(parameters=(8,)), geometry=geometry,
            background=background)
        scene = generate_formed_scene(cb, config=cfg)
        np.testing.assert_array_equal(scene.round_truth[['z', 'y', 'x']],
                                      [[z, 2, 3.75], [z, 2, 3], [z, 2, -.5]])
        np.testing.assert_allclose(scene.rounds['round1'][z, 2, 0, 3],
                                   7.059975220676764, atol=1e-12, rtol=0)
        assert scene.round_truth.center_in_bounds.tolist() == [True, True, False]
        assert scene.round_truth.support_intersects.all()
        transforms = scene.provenance['transforms']
        assert all(t['inverse']['max_residual'] <= 2e-10 for t in transforms.values())
        records.append(dict(dataset=cfg.dataset_version, provenance=scene.provenance,
                            formed=scene.formed.to_dict('records'),
                            positions=scene.round_truth[['amplicon_id','round_label','z','y','x',
                                'center_in_bounds','support_intersects']].to_dict('records')))
    return records


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, allow_nan=False))
