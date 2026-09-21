"""Bounded run provenance examples; no historical images or random streams."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from starfinder.barcode import Codebook, NeighborhoodSumConfig, WtaDecoderConfig, ReadFilterConfig
from starfinder.dataset import Dataset, RoundState, PipelineConfig, ExecutionConfig
from starfinder.image import ImageMetadata
from starfinder.provenance import RunRecorder, read_run
from starfinder.spot_finding import LocalMaximaConfig


def main(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for z in (1, 3):
        for mode in ('batch', 'streaming'):
            dataset = Dataset(directory, directory/'unused', 'literal', 'sample', 'output',
                RoundState(['round10', 'round2'], reference_round='round10'), ('b', 'a', 'd', 'c'))
            dataset.codebook = Codebook(pd.DataFrame({'gene_id': ['gene'], 'color_sequence': ['11']}),
                ('round10', 'round2'), dataset.channel_order)
            fov = dataset.fov(f'FOV-Z{z}')
            image = np.zeros((z, 7, 9, 4), dtype=np.uint16)
            image[z//2, 3, 4, 0] = 7
            fov.images = {'round10': image, 'round2': image.copy()}
            fov.metadata = {r: ImageMetadata('common') for r in fov.images}
            config = PipelineConfig(detection=LocalMaximaConfig('adaptive', .1),
                extraction=NeighborhoodSumConfig((0, 0, 0)), decoding=WtaDecoderConfig(),
                filtering=ReadFilterConfig())
            record = RunRecorder(directory/f'z{z}-{mode}', dataset_id='literal', sample_id='sample',
                owner='example caller', retention='retain with associated analysis',
                save_candidates_signals=False)  # This example isolates provenance.
            fov.run(config, execution=ExecutionConfig(mode), provenance=record)
            run = read_run(record.path)
            assert run['status'] == 'succeeded'
            assert not run['failures']
            assert run['extensions']['starfinder.provenance']['final_state']['detected'] == 1
            np.testing.assert_array_equal(fov.intensity_result.values, [[[7, 7], [0, 0], [0, 0], [0, 0]]])
            assert fov.filtering_result.accepted['gene_id'].tolist() == ['gene']
            assert all(a['status'] == 'omitted' for a in run['artifacts'])
    print('Provenance examples passed.')


if __name__ == '__main__':
    main(sys.argv[1])
