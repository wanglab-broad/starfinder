"""Small development recipes; reuse quickstart inputs and write new outputs."""

from starfinder.barcode import NeighborhoodSumConfig, ReadFilterConfig
from starfinder.io import ImageLoadConfig

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from starfinder.dataset import RegistrationStep, FOV, RoundState, Dataset
from starfinder.io import load_round, load_volume, save_volume
from starfinder.registration import TranslationConfig, estimate_transform, apply_transform, TranslationConfig
from starfinder.spot_finding import find_spots, LocalMaximaConfig
from starfinder.image import ImageMetadata


def image_io(output: Path) -> np.ndarray:
    volume = np.zeros((12, 24, 24), dtype=np.uint16)
    volume[5, 10, 10] = 1000
    save_volume(volume, output / "channels" / "ch00.tif")
    save_volume(volume // 2, output / "channels" / "ch01.tif")
    loaded = load_volume(output / "channels" / "ch00.tif").image
    np.testing.assert_array_equal(loaded, volume)
    loaded_round = load_round(output / "channels", config=ImageLoadConfig(channel_labels=tuple(["ch00", "ch01"])))
    image = loaded_round.image
    metadata = loaded_round.diagnostics
    assert image.shape == (12, 24, 24, 2) and image.dtype == np.uint16
    assert not metadata["cropped"]
    np.testing.assert_array_equal(image[..., 1], volume // 2)
    print("I/O: uint16 round-trip; stacked shape", image.shape)
    return loaded


def register_volumes(fixed: np.ndarray) -> None:
    moving = np.roll(fixed, (1, -2, 3), axis=(0, 1, 2))
    assert fixed.ndim == moving.ndim == 3 and fixed.shape == moving.shape
    result = estimate_transform(fixed, moving, config=TranslationConfig(), reference_metadata=ImageMetadata("reference"), moving_metadata=ImageMetadata("moving"))
    detected = tuple(-x for x in result.transform.correction_zyx)
    corrected = apply_transform(moving, result.transform, config=result.application_config)
    assert detected == (1, -2, 3)
    np.testing.assert_array_equal(corrected, fixed)
    print("Registration: detected", detected, "; correction (-1, 2, -3)")


def detect_spots(volume: np.ndarray) -> None:
    # Add a channel axis to this known single-channel ZYX volume.
    image = volume[..., np.newaxis]
    for mode, threshold in [("noise", 5.0), ("adaptive", 0.2)]:
        spots = find_spots(
            image,
            config=LocalMaximaConfig(threshold_mode=mode, threshold_value=threshold),
            metadata=ImageMetadata("example/sample/FOV/round1"),
            spot_namespace="example/sample/FOV",
        ).spots
        assert spots[["z", "y", "x", "channel"]].values.tolist() == [[5, 10, 10, 0]]
    print("Detection: one spot at internal (z,y,x)=(5,10,10), channel 0")


def decode_fov(quickstart: Path, output: Path) -> FOV:
    dataset = Dataset(
        input_root=quickstart / "input", output_root=output / "results",
        dataset_id="tiny", sample_id="synthetic", output_id="recipes",
        rounds=RoundState(sequencing_rounds=["round1", "round2", "round3", "round4"], reference_round="round1"),
        channel_order=["ch00", "ch01", "ch02", "ch03"],
    )
    dataset.rounds.validate()
    dataset.load_codebook(quickstart / "synthetic" / "codebook.csv", reverse_bases=True)
    fov = dataset.fov("FOV_001")
    fov.load_images()
    assert all(image.shape == (8, 128, 128, 4) for image in fov.images.values())
    fov.register(RegistrationStep(TranslationConfig()))
    fov.find_spots(config=LocalMaximaConfig(threshold_mode="noise", threshold_value=5.0))
    fov.extract_intensities(config=NeighborhoodSumConfig((1, 2, 2)))
    fov.decode_barcodes().filter_reads()
    assert 0 < len(fov.filtering_result.accepted) <= len(fov.spot_result.spots)
    assert set(fov.filtering_result.accepted["gene_id"]) <= set(dataset.codebook.genes)
    print(f"FOV_001: {len(fov.spot_result.spots)} detected, {len(fov.filtering_result.accepted)} retained")
    return fov


def inspect_outputs(fov: FOV) -> dict:
    all_path = fov.save_spots("allSpots", columns=["spot_namespace", "spot_id", "x", "y", "z", "gene", "color_seq", "call_status"])
    good_path = fov.save_spots("goodSpots")
    log_path = fov.save_processing_log()
    candidates = pd.read_csv(all_path, dtype={"color_seq": str})
    molecules = pd.read_csv(good_path)
    assert list(molecules.columns) == ["x", "y", "z", "gene"]
    assert candidates["color_seq"].str.fullmatch(r"[1-4MN]{4}").all()
    assert len(candidates) == len(fov.spot_result.spots)
    assert len(molecules) == len(fov.filtering_result.accepted)
    # CSV columns are Cartesian XYZ, but NumPy indexing requires zero-based ZYX.
    coordinates = molecules[["z", "y", "x"]].to_numpy(dtype=int) - 1
    np.testing.assert_array_equal(coordinates, fov.spot_result.spots.merge(fov.filtering_result.accepted[["spot_id"]], on="spot_id", validate="one_to_one")[["z", "y", "x"]])
    assert (coordinates >= 0).all() and (coordinates < (8, 128, 128)).all()
    reference = fov.images[fov.rounds.reference_round]
    intensities = reference[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], :]
    assert intensities.shape == (len(molecules), 4)
    print(molecules.head().to_string(index=False))
    print("Logs:", log_path, "and", fov.dataset.output_root / "log" / "gr_shifts")
    return {
        "detected": len(candidates), "retained": len(molecules),
        "gene_counts": molecules.groupby("gene").size().to_dict(),
        "molecules_csv": str(good_path),
    }


def main(quickstart: Path, output: Path) -> None:
    quickstart, output = quickstart.resolve(), output.resolve()
    if not (quickstart / "summary.json").is_file():
        raise ValueError("Run the quickstart successfully first; summary.json is missing")
    output.mkdir(parents=True, exist_ok=False)
    volume = image_io(output)
    register_volumes(volume)
    detect_spots(volume)
    summary = inspect_outputs(decode_fov(quickstart, output))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("All development recipes passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quickstart", type=Path, help="Completed quickstart output directory")
    parser.add_argument("output", type=Path, help="New external output directory")
    args = parser.parse_args()
    main(args.quickstart, args.output)
