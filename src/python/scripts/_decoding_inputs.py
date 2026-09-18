"""Adapters for historical saved tensors with explicitly positional acquisition axes.

Saved NPY tensors predate label metadata. These scripts declare round1..R and
channel1..4 in their stored axis order; they do not claim recovered provenance.
Scientific reevaluation of historical artifacts remains separate from housekeeping.
"""

import csv
import numpy as np
import pandas as pd
from starfinder.barcode import (
    Codebook,
    EncodingConfig,
    IntensityExtractionResult,
    NeighborhoodSumConfig,
    CodebookAwareDecoderConfig,
    decode_barcodes,
    load_codebook,
    extract_intensities,
)
from starfinder.image import ImageMetadata
from starfinder.io import ImageLoadResult
from starfinder.spot_finding import SpotFindingResult, LocalMaximaConfig

CHANNELS = tuple(f"channel{i}" for i in range(1, 5))


def saved_codebook(path, split_index=None):
    """Declare positional labels for a legacy gene/barcode CSV and saved tensor."""
    encoding = EncodingConfig(reverse_bases=True, split_index=split_index)
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = csv.reader(handle)
        first = next(rows)
        if first[0].strip() == "gene":
            first = next(rows)
        rounds = tuple(f"round{i + 1}" for i in range(len(encoding.encode(first[1].strip()))))
    return load_codebook(path, round_labels=rounds, channel_labels=CHANNELS, encoding=encoding)


def saved_decoding(
    values, seq_to_gene, *, spot_ids=None, namespace="historical/saved-tensor", **options
):
    """Decode with public results/diagnostics under the saved-axis convention."""
    values = np.asarray(values, dtype=np.float64)
    n, c, r = values.shape
    if c != 4:
        raise ValueError("saved STARmap tensor requires four channels")
    rounds = tuple(f"round{i + 1}" for i in range(r))
    ids = tuple(str(i) for i in (range(n) if spot_ids is None else spot_ids))
    extracted = IntensityExtractionResult(
        values,
        ids,
        namespace,
        CHANNELS,
        rounds,
        ImageMetadata(namespace),
        NeighborhoodSumConfig(),
        np.ones((n, r), bool),
        {"provenance": "historical positional axes; physical geometry unverified"},
    )
    book = Codebook(
        pd.DataFrame({"gene_id": list(seq_to_gene.values()), "color_sequence": list(seq_to_gene)}),
        rounds,
        CHANNELS,
    )
    options.setdefault("diagnostics", True)
    return decode_barcodes(extracted, book, config=CodebookAwareDecoderConfig(**options))


def tensor_diagnostics(values):
    """Public probability diagnostics with an empty codebook (no assignments)."""
    return saved_decoding(values, {}).diagnostics


def observed_candidates(sequence, seq_to_gene):
    """Inspect candidates by explicitly constructing the observed color pattern."""
    values = np.zeros((1, 4, len(sequence)), dtype=float)
    for i, color in enumerate(sequence):
        if color in "1234":
            values[0, int(color) - 1, i] = 1
        else:
            values[0, :, i] = 1
    return saved_decoding(values, seq_to_gene).diagnostics["candidates"]


def saved_extraction(images, spots, round_order, neighborhood_radius_zyx=(1, 2, 2), *, namespace):
    """Extract registered saved images under explicitly declared common geometry."""
    table = spots[["z", "y", "x"]].astype("float64").copy()
    table["spot_id"] = pd.Series(
        spots["spot_id"].astype(str).to_numpy()
        if "spot_id" in spots
        else [str(i) for i in range(len(spots))],
        index=table.index,
        dtype="string",
    )
    metadata = ImageMetadata(namespace)
    found = SpotFindingResult(table, metadata, namespace, LocalMaximaConfig(), {})
    return extract_intensities(
        {r: ImageLoadResult(images[r], metadata, CHANNELS, (), {}) for r in round_order},
        found,
        config=NeighborhoodSumConfig(tuple(neighborhood_radius_zyx)),
    ).values


def report_table(result):
    """Translate public columns to the existing saved-report vocabulary."""
    return result.table.rename(
        columns={
            "observed_color_sequence": "color_seq_wta",
            "decoded_color_sequence": "decoded_seq",
            "gene_id": "gene",
            "failure_reason": "reject_reason",
            "probability_nll": "score",
            "geomean_probability": "geomean_prob",
        }
    ).copy()
