"""Independent small numerical expectations for starfinder.artifacts/1.

No storage adapter is implemented here. Future checkpoint acceptance must run
these same expectations on reloaded payloads as well as uninterrupted results.
"""

from dataclasses import replace
import math

import numpy as np
import pandas as pd

from starfinder.barcode import (
    Codebook, NeighborhoodSumConfig, WtaDecoderConfig,
    decode_barcodes, extract_intensities, filter_reads,
)
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.io import ImageLoadResult
from starfinder.registration import (
    DenseDisplacementTransform, TranslationTransform, WarpConfig, apply_transform,
)
from starfinder.spot_finding import LocalMaximaConfig, SpotFindingResult


def geometry_example():
    metadata = ImageMetadata(
        "development/reference", (2, 3, 4), (10, 20, 30),
        ((1, 0, 0), (0, -1, 0), (0, 0, -1)), "um",
    )
    np.testing.assert_array_equal(metadata.index_to_world((1, 2, 3)), (12, 14, 18))
    np.testing.assert_array_equal(metadata.world_to_index((12, 14, 18)), (1, 2, 3))
    assert metadata.cropped((1, 1, 2), frame_id="crop").origin_zyx == (12, 17, 22)
    unknown = ImageMetadata("development/unknown")
    assert unknown.spacing_zyx is None and unknown.origin_zyx is None
    assert unknown.direction_zyx is None and unknown.spatial_unit is None
    try:
        unknown.index_to_world((0, 0, 0))
    except IncompatibleGeometryError:
        pass
    else:
        raise AssertionError("Unknown calibration must reject physical conversion")


def image_example(depth):
    shape = (depth, 4, 5)
    z = depth // 2
    reference, moving = ImageMetadata("reference"), ImageMetadata("moving")
    source = np.zeros(shape, dtype=np.uint16)
    source[z, 1, 2] = 7
    transform = TranslationTransform((0, 1, -1), shape, shape, reference, moving)
    result = apply_transform(source, transform, config=WarpConfig())
    expected = np.zeros(shape, dtype=np.uint16)
    expected[z, 2, 1] = 7
    np.testing.assert_array_equal(result, expected, strict=True)
    assert source[z, 1, 2] == 7  # no mutation

    ramp = np.broadcast_to(np.array([0, 1, 2, 3, 4], dtype=np.uint16), shape).copy()
    field = np.zeros((*shape, 3), dtype=np.float64)
    field[..., 2] = 0.5
    pull = DenseDisplacementTransform(field, shape, shape, reference, moving)
    integer = apply_transform(ramp, pull, config=WarpConfig(backend="scipy"))
    floating = apply_transform(
        ramp, pull, config=WarpConfig(backend="scipy", output_dtype="float64"),
    )
    np.testing.assert_array_equal(integer[z, 1], np.array([0, 2, 2, 4, 0], dtype=np.uint16), strict=True)
    np.testing.assert_array_equal(floating[z, 1], [0.5, 1.5, 2.5, 3.5, 0])


def signal_example(depth):
    """Return reusable in-memory payloads with independently asserted values."""
    metadata = ImageMetadata("development/reference")
    channels = ("ch02", "ch00", "ch03", "ch01")
    rounds = ("round10", "round2")
    namespace = f'["artifact-contract-v1","sample","FOV-Z{depth}",null]'
    z = depth // 2
    table = pd.DataFrame({
        "spot_id": pd.Series(["A", "B"], dtype="string"),
        "z": [float(z), float(z)], "y": [1.0, 2.0], "x": [2.0, 3.0],
    })
    spots = SpotFindingResult(table, metadata, namespace, LocalMaximaConfig(), {})
    first = np.zeros((depth, 4, 5, 4), dtype=np.uint16)
    second = np.zeros_like(first)
    first[z, 1, 2, 1] = 7
    second[z, 1, 2, 0] = 9
    images = {
        rounds[0]: ImageLoadResult(first, metadata, channels, ()),
        rounds[1]: ImageLoadResult(second, metadata, channels, ()),
    }
    config = NeighborhoodSumConfig((0, 0, 0))
    extracted = extract_intensities(images, spots, config=config)
    expected = np.array([[[0, 9], [7, 0], [0, 0], [0, 0]], np.zeros((4, 2))], dtype=np.float64)
    np.testing.assert_array_equal(extracted.values, expected, strict=True)
    np.testing.assert_array_equal(extracted.valid, np.ones((2, 2), dtype=bool), strict=True)
    assert extracted.spot_ids == ("A", "B") and extracted.spot_namespace == namespace
    assert extracted.round_labels == rounds and extracted.channel_labels == channels
    codebook = Codebook(
        pd.DataFrame({"gene_id": ["gene-A"], "color_sequence": ["12"]}),
        rounds, channels, {"1": 1, "2": 0, "3": 3, "4": 2},
    )
    decoded = decode_barcodes(extracted, codebook, config=WtaDecoderConfig())
    assert decoded.table.call_status.tolist() == ["assigned", "no_signal"]
    assert decoded.table.loc[0, "gene_id"] == "gene-A"
    assert pd.isna(decoded.table.loc[1, "gene_id"])
    assert decoded.table.loc[0, "observed_color_sequence"] == "12"
    # log1p gives an independent stable expression for the one-hot WTA score.
    expected_score = math.log1p(1e-6 / 7) + math.log1p(1e-6 / 9)
    np.testing.assert_allclose(decoded.table.loc[0, "wta_l2_nll"], expected_score, rtol=0, atol=1e-12)
    filtered = filter_reads(decoded)
    assert filtered.accepted.spot_id.tolist() == ["A"]
    assert filtered.table.spot_id.tolist() == ["A", "B"]
    assert filtered.table.rejection_reasons.tolist() == ["", "call_status"]
    assert filtered.counts == {"total": 2, "accepted": 1, "rejected": 1}
    unavailable = extracted.valid.copy()
    unavailable[0, 1] = False
    invalid = decode_barcodes(replace(extracted, valid=unavailable), codebook, config=WtaDecoderConfig())
    assert invalid.table.loc[0, "failure_reason"] == "invalid_measurement"
    assert invalid.table.spot_id.tolist() == ["A", "B"]
    assert pd.isna(invalid.table.loc[0, "gene_id"])
    empty = extract_intensities(images, replace(spots, spots=table.iloc[:0].copy()), config=config)
    assert empty.values.shape == (0, 4, 2) and empty.valid.shape == (0, 2)
    empty_qc = filter_reads(decode_barcodes(empty, codebook, config=WtaDecoderConfig()))
    assert empty_qc.counts == {"total": 0, "accepted": 0, "rejected": 0}
    assert empty_qc.fractions == {"accepted": None}
    assert empty_qc.diagnostics["undefined_fraction_reasons"] == {"accepted": "empty_population"}
    return spots, extracted, codebook, decoded, filtered


def main():
    geometry_example()
    for depth in (1, 3):
        image_example(depth)
        signal_example(depth)
    print("Artifact contract examples passed.")


if __name__ == "__main__":
    main()
