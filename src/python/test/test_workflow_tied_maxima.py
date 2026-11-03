"""W-790 tile-task checks on bounded, hand-built plateaus, including internal window edges."""
import warnings

import numpy as np
import pandas as pd
import pytest

from starfinder.dataset import PipelineConfig
from starfinder.dataset import tiled as T
from starfinder.image import ImageMetadata
from starfinder.io.chunked import ChunkedCheckpointWriter
from starfinder.spot_finding import LocalMaximaConfig, find_spots

from .test_spot_finding_golden import CHANNELS, golden_dataset

pytestmark = [pytest.mark.workflow, pytest.mark.dataset]


def fixture(tmp_path, long):
    image = np.zeros((4, 24, 24, 4), dtype=np.uint16)
    image[1:3, 4:6, 10:(18 if long else 14), 0] = 10
    image[2, 18, 18, 1] = 9
    dataset = golden_dataset(tmp_path)
    pipeline = PipelineConfig(spot_finding=LocalMaximaConfig(threshold_mode="absolute", threshold_value=1))
    store = tmp_path / "registered.h5"
    with ChunkedCheckpointWriter(store, identity={}, channel_labels=CHANNELS) as writer:
        writer.write_round("round1", image, ImageMetadata("tile-ties"))
    statistics = {"detection": T._detection_statistics(dataset.fov("FOV_001"), pipeline.spot_finding, image)}
    return image, dataset, pipeline, store, statistics


def run(dataset, pipeline, store, statistics, tile, folder, halo, exact=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return T.run_tile(dataset.fov("FOV_001"), pipeline, tile, store=store, statistics=statistics,
                          directory=folder, halo=halo, exact=exact)


def test_truncated_maximum_fails_exact_and_counts_approximate(tmp_path):
    _, dataset, pipeline, store, statistics = fixture(tmp_path, True)
    tile = T.tile_grid((24, 24), 12, 2)[0]
    with pytest.raises(ValueError, match=r"truncated regional maxim.*halo.*12"):
        run(dataset, pipeline, store, statistics, tile, tmp_path / "exact", 2)
    record = run(dataset, pipeline, store, statistics, tile, tmp_path / "approx", 2, exact=False)
    assert record["diagnostics"]["spot_finding"]["truncated_maxima"] == {
        "ch00": 1, "ch01": 0, "ch02": 0, "ch03": 0}


def test_constant_window_of_nonconstant_fov_is_truncated(tmp_path):
    image, dataset, pipeline, store, _ = fixture(tmp_path, True)
    image[:, :16, :16, 0] = 10
    with ChunkedCheckpointWriter(store, identity={}, channel_labels=CHANNELS) as writer:
        writer.write_round("round1", image, ImageMetadata("constant-window"))
    statistics = {"detection": T._detection_statistics(dataset.fov("FOV_001"), pipeline.spot_finding, image)}
    tile = T.tile_grid((24, 24), 12, 2)[0]
    with pytest.raises(ValueError, match="truncated regional maximum"):
        run(dataset, pipeline, store, statistics, tile, tmp_path / "constant", 2)
    record = run(dataset, pipeline, store, statistics, tile, tmp_path / "approx-constant", 2, exact=False)
    assert record["diagnostics"]["spot_finding"]["truncated_maxima"]["ch00"] == 1


def test_fitting_plateau_tiles_equal_direct_and_sum_diagnostics(tmp_path):
    image, dataset, pipeline, store, statistics = fixture(tmp_path, False)
    tiles = T.tile_grid((24, 24), 12, 4)
    records, tables = [], []
    for tile in tiles:
        folder = tmp_path / tile.id
        records.append(run(dataset, pipeline, store, statistics, tile, folder, 4))
        tables.append(pd.read_csv(folder / "candidates.csv"))
    actual = T.detector_order(pd.concat(tables, ignore_index=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = find_spots(image, config=pipeline.spot_finding, metadata=ImageMetadata("tile-ties"),
                            spot_namespace="direct")
    columns = ["z", "y", "x", "channel", "peak_intensity"]
    np.testing.assert_allclose(actual[columns].to_numpy(dtype=float), direct.spots[columns].to_numpy(dtype=float),
                               rtol=1e-9, atol=1e-12)
    # Only the tile owning the representative contributes the plateau's diagnostics.
    assert sum(r["diagnostics"]["spot_finding"].get("plateaus", {}).get("ch00", {}).get("maxima", 0)
               for r in records) == 1
    first = records[0]["diagnostics"]["spot_finding"]
    statistics["reference_metadata"] = {"frame_id": "tile-ties"}
    merged = T._merged_detection(dataset.fov("FOV_001"), pipeline, statistics, first, direct.spots,
                                 records=records)
    assert merged.diagnostics["plateaus"]["ch00"] == {"maxima": 1, "removed_voxels": 15}
