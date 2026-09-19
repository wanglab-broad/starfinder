"""Neighborhood, geometry and subpixel extraction contracts."""

from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from starfinder.barcode import (
    extract_intensities,
    NeighborhoodSumConfig,
    decode_barcodes,
    WtaDecoderConfig,
)
from starfinder.io import ImageLoadResult
from starfinder.spot_finding import SpotFindingResult, LocalMaximaConfig
from starfinder.image import ImageMetadata
from .barcode_cases import CHANNELS, META, codebook


def spots(points, ids=None):
    table = pd.DataFrame(points, columns=["z", "y", "x"], dtype="float64")
    table["spot_id"] = pd.Series(ids or [f"s{i}" for i in range(len(table))], dtype="string")
    return SpotFindingResult(
        table, META, "test/sample/fov", LocalMaximaConfig(), {"channel_labels": CHANNELS}
    )


def loaded(image, **kwargs):
    return ImageLoadResult(
        image, kwargs.get("metadata", META), kwargs.get("channel_labels", CHANNELS), (), {}
    )


def test_integer_sums_edges_round_order_and_immutability():
    rng = np.random.default_rng(141)
    image = rng.integers(0, 65536, (4, 7, 8, 4), dtype=np.uint16)
    original = image.copy()
    found = spots([(0, 0, 0), (2, 3, 4), (3, 6, 7)], ["corner", "middle", "last"])
    result = extract_intensities(
        {"later": loaded(image), "earlier": loaded(image)},
        found,
        config=NeighborhoodSumConfig((1, 2, 1)),
    )
    for i, (z, y, x) in enumerate(found.spots[["z", "y", "x"]].to_numpy().astype(int)):
        expected = image[max(0, z - 1) : z + 2, max(0, y - 2) : y + 3, max(0, x - 1) : x + 2].sum(
            (0, 1, 2)
        )
        np.testing.assert_array_equal(result.values[i, :, 0], expected)
    assert result.round_labels == ("later", "earlier")
    assert result.spot_ids == ("corner", "middle", "last")
    assert result.values.dtype == np.float64 and result.valid.all()
    np.testing.assert_array_equal(image, original)


def test_half_voxel_rounds_up_and_no_coordinate_truncation():
    image = np.zeros((1, 3, 4, 4), dtype=np.uint8)
    image[0, 1, 2, 2] = 200
    found = spots([(0, 0.5, 1.5)])
    result = extract_intensities(
        {"r0": loaded(image)}, found, config=NeighborhoodSumConfig((0, 0, 0))
    )
    np.testing.assert_array_equal(result.values[0, :, 0], [0, 0, 200, 0])
    assert found.spots.x[0] == 1.5
    decoded = decode_barcodes(result, codebook({"3": "gene"}), config=WtaDecoderConfig())
    assert decoded.table.gene_id[0] == "gene"


@pytest.mark.parametrize("point", [(-0.01, 0, 0), (0, -1, 0), (0, 0, 3.01), (1, 0, 0)])
def test_out_of_bounds_rejected_before_rounding(point):
    with pytest.raises(ValueError, match="bounds"):
        extract_intensities({"r": loaded(np.ones((1, 3, 4, 4)))}, spots([point]))


@pytest.mark.parametrize("change", ["shape", "frame", "spacing", "channels", "nonfinite"])
def test_round_mismatch_rejected(change):
    a = loaded(np.ones((1, 3, 4, 4)))
    b = a
    if change == "shape":
        b = loaded(np.ones((1, 3, 5, 4)))
    if change == "frame":
        b = replace(a, metadata=ImageMetadata("other"))
    if change == "spacing":
        b = replace(a, metadata=ImageMetadata(META.frame_id, (1, 1, 1)))
    if change == "channels":
        b = replace(a, channel_labels=CHANNELS[::-1])
    if change == "nonfinite":
        b = loaded(np.full((1, 3, 4, 4), np.nan))
    with pytest.raises(ValueError):
        extract_intensities({"a": a, "b": b}, spots([(0, 1, 1)]))


def test_empty_and_nonempty_axes():
    result = extract_intensities({"r": loaded(np.ones((1, 3, 4, 4)))}, spots([]))
    assert result.values.shape == (0, 4, 1) and result.valid.shape == (0, 1)
    with pytest.raises(ValueError):
        extract_intensities({}, spots([]))


@pytest.mark.parametrize("radius", [(-1, 0, 0), (0.5, 0, 0), (True, 0, 0), (1, 2)])
def test_invalid_neighborhood(radius):
    with pytest.raises(ValueError):
        NeighborhoodSumConfig(radius)
