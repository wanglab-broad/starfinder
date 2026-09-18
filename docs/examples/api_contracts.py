"""Bounded API contract examples; pass an external output directory."""

from starfinder.preprocessing import MinMaxNormalizationConfig
from starfinder.preprocessing import ProjectionConfig

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from starfinder.barcode import (
    Codebook, NeighborhoodSumConfig, CodebookAwareDecoderConfig,
    decode_barcodes, extract_intensities, filter_reads,
)
from starfinder.benchmark import measure
from starfinder.synthetic import generate_volume
from starfinder.dataset import RoundState, Dataset, RegistrationStep
from starfinder.io import ImageLoadResult, load_volume, save_volume
from starfinder.preprocessing import normalize_intensity
from starfinder.registration import estimate_transform, apply_transform, TranslationConfig, TpsConfig, DemonsConfig, WarpConfig, DenseDisplacementTransform, InsufficientLandmarksError
from starfinder.spot_finding import find_spots, LocalMaximaConfig
from starfinder.image import ImageMetadata
from starfinder.preprocessing import project_image


def main(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    fixed = np.zeros((12, 24, 24), dtype=np.uint16)
    fixed[5, 10, 10] = 1000
    displacement = (1, -2, 3)
    moving = np.roll(fixed, displacement, axis=(0, 1, 2))
    result = estimate_transform(fixed, moving, config=TranslationConfig(), reference_metadata=ImageMetadata("reference"), moving_metadata=ImageMetadata("moving"))
    detected = tuple(-x for x in result.transform.correction_zyx)
    assert detected == displacement
    np.testing.assert_array_equal(
        apply_transform(moving, result.transform, config=result.application_config), fixed
    )
    field = np.empty(fixed.shape + (3,), dtype=np.float32)
    field[...] = displacement
    np.testing.assert_array_equal(apply_transform(moving, DenseDisplacementTransform(field, fixed.shape, moving.shape, ImageMetadata("reference"), ImageMetadata("moving")), config=WarpConfig(backend="scipy")), fixed)

    normalized = normalize_intensity(fixed, config=MinMaxNormalizationConfig('uint8', (0, 255)))
    assert normalized.dtype == np.uint8 and normalized.max() == 255
    projection = project_image(fixed, config=ProjectionConfig(method='max'))
    assert projection.shape == (1, 24, 24) and projection.dtype == np.uint16
    save_volume(fixed, output / "volume.tif")
    np.testing.assert_array_equal(
        load_volume(output / "volume.tif").image, fixed
    )

    image = np.zeros(fixed.shape + (4,), dtype=np.uint16)
    image[..., 0] = fixed
    spots = find_spots(
        image, config=LocalMaximaConfig("adaptive", 0.2),
        metadata=ImageMetadata("example/sample/FOV/round1"),
        spot_namespace="example/sample/FOV",
    )
    assert spots.spots[["z", "y", "x"]].values.tolist() == [[5, 10, 10]]
    channels = ('ch00', 'ch01', 'ch02', 'ch03')
    extracted = extract_intensities(
        {'round1': ImageLoadResult(image, spots.metadata, channels, (), {})}, spots,
        config=NeighborhoodSumConfig((0, 0, 0)))
    assert extracted.values.shape == (1, 4, 1) and extracted.values.dtype == np.float64
    book = Codebook(pd.DataFrame({'gene_id':['GeneA'], 'color_sequence':['1']}),
                    ('round1',), channels)
    decoded = decode_barcodes(extracted, book, config=CodebookAwareDecoderConfig(diagnostics=True))
    np.testing.assert_allclose(decoded.diagnostics['probabilities'].sum(axis=1), 1)
    assert decoded.table.loc[0, 'gene_id'] == 'GeneA'
    assert decoded.table.loc[0, 'call_type'] == 'exact'

    dataset = Dataset(
        input_root=output, output_root=output,
        dataset_id="example", sample_id="example", output_id="example",
        rounds=RoundState(sequencing_rounds=["round1", "round2"], reference_round="round1"),
    )
    fov = dataset.fov("Position000")
    fov.spot_result = spots
    fov.decoding_result = decoded
    fov.filtering_result = filter_reads(decoded)
    csv = pd.read_csv(fov.save_spots())
    assert csv[["x", "y", "z"]].values.tolist() == [[11, 11, 6]]
    assert fov.spot_result.spots[["z", "y", "x"]].values.tolist() == [[5, 10, 10]]

    if importlib.util.find_spec("SimpleITK") is None:
        try:
            estimate_transform(fixed, moving, config=DemonsConfig(iterations=(1,)), reference_metadata=ImageMetadata("reference"), moving_metadata=ImageMetadata("moving"))
        except ImportError as exc:
            assert "SimpleITK" in str(exc)
        else:
            raise AssertionError("Expected missing SimpleITK error")
    fov.images = {r: np.zeros_like(image) for r in dataset.rounds.sequencing_rounds}
    fov.metadata = {r: ImageMetadata(r) for r in dataset.rounds.sequencing_rounds}
    try:
        fov.register(RegistrationStep(TpsConfig(), "single-channel", "single-channel"))
    except InsufficientLandmarksError:
        pass
    else:
        raise AssertionError("Expected insufficient landmarks without fallback")

    synthetic = generate_volume((12, 24, 24), n_spots=3, seed=97)
    assert synthetic.shape == fixed.shape and synthetic.dtype == np.uint8
    result, seconds, memory_mib = measure(lambda: int(synthetic.sum()))
    assert result > 0 and seconds >= 0 and memory_mib >= 0
    print("All API contract examples passed (synthetic seed=97).")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
