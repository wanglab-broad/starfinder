"""W-790 known answers for regional maxima and their discrete representatives."""
from dataclasses import replace
import warnings

import numpy as np
import pytest

from starfinder.image import ImageMetadata
from starfinder.spot_finding import LocalMaximaConfig, NoiseLandmarkConfig, find_spots

pytestmark = pytest.mark.spot_finding

CONFIG = LocalMaximaConfig(threshold_mode="absolute", threshold_value=1, exclude_border=False)


def detect(image, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return find_spots(image, config=replace(CONFIG, **options), metadata=ImageMetadata("plateaus"),
                          spot_namespace="plateaus")


def coords(result):
    return [tuple(p) for p in result.spots[["z", "y", "x"]].to_numpy(dtype=int)]


def test_cubes_and_single_peak_round_half_up():
    image = np.zeros((12, 32, 32), dtype=np.uint16)
    image[2:5, 3:6, 3:6] = 10
    image[7:9, 12:14, 12:14] = 9
    image[8, 24, 24] = 8
    result = detect(image)
    assert coords(result) == [(3, 4, 4), (8, 13, 13), (8, 24, 24)]
    assert result.diagnostics["plateaus"] == {"0": {"maxima": 2, "removed_voxels": 33}}


def test_centroid_outside_uses_nearest_then_zyx():
    image = np.zeros((1, 12, 12), dtype=np.uint16)
    image[0, 3:6, 3:6] = 10
    image[0, 4, 4] = 0
    assert coords(detect(image)) == [(0, 3, 4)]


def test_ramp_shoulder_is_not_a_regional_maximum():
    image = np.zeros((1, 12, 16), dtype=np.uint16)
    image[0, 4, 3:10] = 5
    image[0, 4, 10] = 6
    assert coords(detect(image)) == [(0, 4, 10)]


def test_plane_uses_eight_connectivity_and_half_up():
    image = np.zeros((1, 12, 12), dtype=np.uint16)
    image[0, 3, 3] = image[0, 4, 4] = 10
    assert coords(detect(image)) == [(0, 4, 4)]


def test_border_exclusion_uses_representative():
    image = np.zeros((1, 12, 12), dtype=np.uint16)
    image[0, 0:3, 4:7] = 10
    assert coords(detect(image, exclude_border=True)) == [(0, 1, 5)]


def test_distance_window_and_spacing_apply_after_representatives():
    image = np.zeros((1, 16, 16), dtype=np.uint16)
    image[0, 3:5, 3:5] = 10
    image[0, 6, 6] = 9  # within the distance-3 window of the representative
    image[0, 10:12, 10:12] = 8
    assert coords(detect(image, min_distance_voxels=3)) == [(0, 4, 4), (0, 11, 11)]


def test_merge_visits_the_representatives():
    image = np.zeros((1, 16, 16), dtype=np.uint16)
    image[0, 3:5, 3:5] = 10
    image[0, 4, 7] = 9
    result = detect(image, merge_radius_zyx=(1, 1, 3))
    assert coords(result) == [(0, 4, 4)]
    assert result.diagnostics["merged"] == {"0": 1}


def test_noise_landmarks_share_regional_maxima():
    # A 0/1 checkerboard keeps fewer than half of the voxels at the minimum, so the landmarks' noise is the all-voxel
    # median 1 and MAD 1 (W-836): threshold 1 + 5 x 1.4826 = 8.413, below the plateau of 10.
    image = (np.indices((8, 12, 12)).sum(axis=0) % 2).astype(np.uint16)
    image[2:4, 3:5, 3:5] = 10
    result = find_spots(image, config=NoiseLandmarkConfig(), metadata=ImageMetadata("landmarks"),
                        spot_namespace="landmarks")
    assert coords(result) == [(3, 4, 4)]
