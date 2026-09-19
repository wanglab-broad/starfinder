"""Tiny explicit labeled barcode fixtures shared by contract tests."""

import numpy as np
import pandas as pd
from starfinder.barcode import Codebook, IntensityExtractionResult, NeighborhoodSumConfig
from starfinder.image import ImageMetadata

CHANNELS = ("red", "green", "blue", "yellow")
META = ImageMetadata("test/sample/fov")


def intensity(values, ids=None, channels=CHANNELS, rounds=None):
    values = np.asarray(values, dtype=np.float64)
    n, c, r = values.shape
    return IntensityExtractionResult(
        values,
        tuple(ids if ids is not None else (f"s{i}" for i in range(n))),
        "test/sample/fov",
        channels,
        rounds or tuple(f"r{i}" for i in range(r)),
        META,
        NeighborhoodSumConfig(),
        np.ones((n, r), bool),
        {},
    )


def codebook(mapping, r=None, channels=CHANNELS, rounds=None, **kwargs):
    r = r if r is not None else len(next(iter(mapping)))
    return Codebook(
        pd.DataFrame({"gene_id": list(mapping.values()), "color_sequence": list(mapping)}),
        rounds or tuple(f"r{i}" for i in range(r)),
        channels,
        **kwargs,
    )


def tensor(sequences, high=100.0, low=1.0):
    values = np.full((len(sequences), 4, len(sequences[0])), low)
    for i, seq in enumerate(sequences):
        for j, c in enumerate(seq):
            values[i, int(c) - 1, j] = high
    return values
