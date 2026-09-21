"""Clean formed scenes and independent downstream checks; no file writes."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd

from starfinder.barcode import NeighborhoodSumConfig, WtaDecoderConfig, extract_intensities, decode_barcodes
from starfinder.io import ImageLoadResult
from starfinder.spot_finding import SpotFindingResult, LocalMaximaConfig
from starfinder.synthetic import ScalarDistribution, formed_scene_preset, generate_formed_scene


def check_scene(depth):
    codebook, config = formed_scene_preset('formed-z1-v1' if depth == 1 else 'formed-small-v1')
    z = depth // 2
    config = replace(config, shape_zyx=(depth, 7, 9), coordinates=((z, 2, 2), (z, 4, 6)),
                     amplicon_ids=('formed-A', 'formed-B'), gene_ids={'formed-B': 'gene-B', 'formed-A': 'gene-A'},
                     brightness=ScalarDistribution(parameters=(8,)),
                     axial_width=ScalarDistribution(parameters=(.25,)),
                     lateral_width=ScalarDistribution(parameters=(.25,)))
    scene = generate_formed_scene(codebook, config=config)
    truth_namespace = scene.formed.namespace.iloc[0]
    # Independently supplied candidate coordinates in REVERSE order, distinct IDs
    # and namespace. This is an extraction oracle, not a detector/truth ID join.
    candidates = SpotFindingResult(pd.DataFrame(dict(spot_id=['candidate-B', 'candidate-A'],
        z=[float(z)]*2, y=[4., 2.], x=[6., 2.])), scene.metadata,
        json.dumps([config.dataset_version, 'sample', config.FOV_id, 'independent-candidates']),
        LocalMaximaConfig(), {})
    assert candidates.spot_namespace != truth_namespace
    rounds = {label: ImageLoadResult(image, scene.metadata, scene.channel_labels, ())
              for label, image in scene.rounds.items()}
    extraction = extract_intensities(rounds, candidates, config=NeighborhoodSumConfig((0, 0, 0)))
    expected = np.array([[[8, 0, 0], [0, 8, 0], [0, 0, 8], [0, 0, 0]],
                         [[0, 8, 0], [8, 0, 0], [0, 0, 0], [0, 0, 8]]], dtype=np.float64)
    np.testing.assert_array_equal(extraction.values, expected)
    assert extraction.spot_ids == ('candidate-B', 'candidate-A')
    decoded = decode_barcodes(extraction, codebook, config=WtaDecoderConfig())
    by_id = decoded.table.set_index('spot_id')
    assert by_id.loc['candidate-B', 'gene_id'] == 'gene-B'
    assert by_id.loc['candidate-A', 'gene_id'] == 'gene-A'
    assert len(scene.formed) == 2 and len(scene.round_truth) == 6
    return scene


if __name__ == '__main__':
    for depth in (3, 1):
        check_scene(depth)
    for name in ('formed-small-v1', 'formed-z1-v1'):
        codebook, config = formed_scene_preset(name)
        result = generate_formed_scene(codebook, config=config)
        print(json.dumps(result.provenance, sort_keys=True))
    print('Formed-scene analytic/downstream examples passed (3D and Z=1).')
