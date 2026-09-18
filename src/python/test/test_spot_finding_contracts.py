"""Bounded detector policy, typed schema and identity acceptance cases."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from starfinder.image import ImageMetadata
from starfinder.spot_finding import (
    LocalMaximaConfig, NoiseLandmarkConfig, PercentileCentroidConfig, find_spots,
)


def detect(image, config, namespace='dataset/sample/FOV'):
    return find_spots(image, config=config, metadata=ImageMetadata('frame'),
                      spot_namespace=namespace)


@pytest.mark.parametrize('config', [LocalMaximaConfig(), NoiseLandmarkConfig(), PercentileCentroidConfig()])
def test_empty_schema_and_singleton_z(config):
    image = np.zeros((1, 9, 9, 2), dtype=np.uint16)
    empty = detect(image, config)
    assert empty.spots.empty
    assert isinstance(empty.spots.spot_id.dtype, pd.StringDtype)
    assert all(empty.spots[c].dtype == np.float64 for c in ('z', 'y', 'x'))
    image[0, 4, 5, 0] = 100
    before = image.copy()
    result = detect(image, config)
    assert result.spots[['z', 'y', 'x']].values.tolist() == [[0, 4, 5]]
    assert result.metadata.frame_id == 'frame'
    assert result.metadata.spacing_zyx is None
    assert result.config == config
    np.testing.assert_array_equal(image, before)


def test_distinct_channel_and_centroid_policies():
    image = np.zeros((5, 9, 9, 2))
    image[2, 4, 4, :] = [10, 20]
    image[2, 4, 5, 0] = 10
    pipeline = detect(image, LocalMaximaConfig(channel_labels=('green', 'red')))
    noise = detect(image, NoiseLandmarkConfig())
    centroid = detect(image, PercentileCentroidConfig(90))
    assert set(pipeline.spots.channel) == {0, 1}
    assert len(noise.spots) == 1  # original pairwise channel deduplication
    assert set(noise.spots.columns) == {'spot_id', 'z', 'y', 'x'}
    assert centroid.spots[['z', 'y', 'x']].values.tolist() == [[2, 4, 4.25]]
    assert pipeline.diagnostics['channel_labels'] == ('green', 'red')
    assert pipeline.spots.channel.dtype == np.int64
    assert pipeline.spots.peak_intensity.dtype == np.float64
    optional = detect(image, LocalMaximaConfig(measure_peak_intensity=False))
    assert 'peak_intensity' not in optional.spots
    assert optional.diagnostics['measurements'] == {}


def test_border_and_fraction_threshold():
    image = np.zeros((1, 9, 9), dtype=np.uint8)
    image[0, 0, 4] = 200
    image[0, 4, 4] = 40
    assert len(detect(image, LocalMaximaConfig('global', .1)).spots) == 1
    result = detect(image, LocalMaximaConfig('global', .5, exclude_border=False))
    assert result.spots[['z', 'y', 'x']].values.tolist() == [[0, 0, 4]]


@pytest.mark.parametrize('factory,kwargs', [
    (LocalMaximaConfig, {'threshold_mode': 'unknown'}),
    (LocalMaximaConfig, {'threshold_mode': 'adaptive', 'threshold_value': 1.1}),
    (LocalMaximaConfig, {'threshold_value': -1}),
    (LocalMaximaConfig, {'threshold_value': np.nan}),
    (LocalMaximaConfig, {'min_distance_voxels': 0}),
    (NoiseLandmarkConfig, {'min_distance_voxels': 1.5}),
    (NoiseLandmarkConfig, {'noise_sigma': -1}),
    (PercentileCentroidConfig, {'threshold_percentile': 101}),
    (PercentileCentroidConfig, {'threshold_percentile': -1}),
    (LocalMaximaConfig, {'channel_labels': ('a', 'a')}),
    (LocalMaximaConfig, {'channel_labels': ('',)}),
])
def test_invalid_configs(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)


@pytest.mark.parametrize('image', [np.zeros((4, 4)), np.zeros((0, 4, 4)),
                                  np.full((1, 4, 4), np.nan), np.ones((1, 4, 4), complex)])
def test_invalid_images(image):
    with pytest.raises(ValueError):
        detect(image, LocalMaximaConfig())


def test_labels_global_dtype_namespace_validation():
    image = np.zeros((1, 4, 4, 2))
    for config in [LocalMaximaConfig(channel_labels=('one',)), LocalMaximaConfig('global', .5)]:
        with pytest.raises(ValueError):
            detect(image, config)
    with pytest.raises(ValueError):
        detect(image, LocalMaximaConfig(), '')


def test_ids_survive_subset_join_and_reject_collision():
    image = np.zeros((5, 9, 9))
    image[2, 3, 3], image[2, 6, 6] = 10, 20
    result = detect(image, LocalMaximaConfig())
    other = detect(image, LocalMaximaConfig(), 'dataset/sample/FOV/subtile:1')
    left = result.spots.assign(spot_namespace=result.spot_namespace)
    right = other.spots.assign(spot_namespace=other.spot_namespace)
    combined = pd.concat([left, right], ignore_index=True)
    accepted = combined.iloc[[3, 0]][['spot_namespace', 'spot_id']].assign(gene=['a', 'b'])
    joined = accepted.merge(combined, on=['spot_namespace', 'spot_id'], validate='one_to_one')
    assert joined[['z', 'y', 'x']].values.tolist() == combined.iloc[[3, 0]][['z', 'y', 'x']].values.tolist()
    with pytest.raises(ValueError, match='unique'):
        replace(result, spots=pd.concat([result.spots, result.spots]))
    # Reusing a namespace for an independent result is a collision, not new IDs.
    collision = pd.concat([left, left])
    with pytest.raises(pd.errors.MergeError):
        accepted.merge(collision, on=['spot_namespace', 'spot_id'], validate='one_to_one')


def test_fov_namespace_and_downstream_identity(tmp_path):
    from starfinder.dataset import STARMapDataset, LayerState
    dataset = STARMapDataset(tmp_path, tmp_path, 'dataset', 'sample', 'out',
                            layers=LayerState(seq=['round1'], ref='round1'),
                            channel_order=['a','b','c','d'])
    fov = dataset.fov('FOV')
    image = np.zeros((5, 9, 9, 4), dtype=np.uint8)
    image[2, 4, 4, 0] = 100
    fov.images['round1'] = image
    fov.find_spots()
    ids = fov.all_spots.spot_id.copy()
    namespace = fov.spot_result.spot_namespace
    fov.extract_intensities()
    pd.testing.assert_series_equal(ids, fov.all_spots.spot_id)
    assert fov.all_spots.spot_namespace.tolist() == [namespace]
    from starfinder.barcode import Codebook
    dataset.codebook = Codebook(pd.DataFrame({'gene_id':['gene'],'color_sequence':['1']}),
                               ('round1',), ('a','b','c','d'))
    fov.decode_barcodes().filter_reads()
    accepted = fov.filtering_result.accepted
    pd.testing.assert_series_equal(ids, accepted.spot_id)
    assert accepted.spot_namespace.tolist() == [namespace]
    fov.subtile_id = 1
    fov.find_spots()
    assert fov.spot_result.spot_namespace != namespace


def test_mip_caller_keeps_two_spatial_axes():
    from starfinder.benchmark.evaluate import evaluate_registration
    image = np.zeros((3, 24, 24), dtype=np.uint8)
    image[1, 10, 8] = 100
    image[1, 10, 16] = 200  # same Y, distinct X must not collapse into one spot
    report = evaluate_registration(image, image, image, use_mip=True)
    assert report['n_spots_ref'] == 2
    assert report['match_rate_after'] == 1.0
