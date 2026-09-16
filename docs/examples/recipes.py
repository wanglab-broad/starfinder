"""Small development recipes; reuse quickstart inputs and write new outputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from starfinder.dataset import FOV, LayerState, STARMapDataset
from starfinder.io import load_image_stacks, load_multipage_tiff, save_stack
from starfinder.registration import apply_shift, phase_correlate
from starfinder.spotfinding import find_spots_3d


def image_io(output: Path) -> np.ndarray:
    volume = np.zeros((12, 24, 24), dtype=np.uint16)
    volume[5, 10, 10] = 1000
    save_stack(volume, output / "channels" / "ch00.tif")
    save_stack(volume // 2, output / "channels" / "ch01.tif")
    loaded = load_multipage_tiff(output / "channels" / "ch00.tif", convert_uint8=False)
    np.testing.assert_array_equal(loaded, volume)
    image, metadata = load_image_stacks(
        output / "channels", channel_order=["ch00", "ch01"], convert_uint8=False,
    )
    assert image.shape == (12, 24, 24, 2) and image.dtype == np.uint16
    assert not metadata["cropped"]
    np.testing.assert_array_equal(image[..., 1], volume // 2)
    print("I/O: uint16 round-trip; stacked shape", image.shape)
    return loaded


def register_volumes(fixed: np.ndarray) -> None:
    moving = apply_shift(fixed, (1, -2, 3))
    assert fixed.ndim == moving.ndim == 3 and fixed.shape == moving.shape
    detected = phase_correlate(fixed, moving, workers=1)
    corrected = apply_shift(moving, tuple(-s for s in detected))
    assert detected == (1, -2, 3)
    np.testing.assert_array_equal(corrected, fixed)
    print("Registration: detected", detected, "; correction (-1, 2, -3)")


def detect_spots(volume: np.ndarray) -> None:
    # Add a channel axis to this known single-channel ZYX volume.
    image = volume[..., np.newaxis]
    for mode, threshold in [("noise", 5.0), ("adaptive", 0.2)]:
        spots = find_spots_3d(
            image, intensity_estimation=mode, intensity_threshold=threshold,
            min_distance=1,
        )
        assert spots[["z", "y", "x", "channel"]].values.tolist() == [[5, 10, 10, 0]]
    print("Detection: one spot at internal (z,y,x)=(5,10,10), channel 0")


def decode_fov(quickstart: Path, output: Path) -> FOV:
    dataset = STARMapDataset(
        input_root=quickstart / "input", output_root=output / "results",
        dataset_id="tiny", sample_id="synthetic", output_id="recipes",
        layers=LayerState(seq=["round1", "round2", "round3", "round4"], ref="round1"),
        channel_order=["ch00", "ch01", "ch02", "ch03"],
    )
    dataset.layers.validate()
    dataset.load_codebook(quickstart / "synthetic" / "codebook.csv", do_reverse=True)
    fov = dataset.fov("FOV_001")
    fov.load_raw_images(convert_uint8=False)
    assert all(image.shape == (8, 128, 128, 4) for image in fov.images.values())
    fov.global_registration(ref_img="merged", mov_img="merged", save_shifts=True)
    fov.spot_finding(intensity_estimation="noise", intensity_threshold=5.0)
    fov.reads_extraction(voxel_size=(1, 2, 2))
    fov.reads_filtration()
    assert 0 < len(fov.good_spots) <= len(fov.all_spots)
    assert set(fov.good_spots["gene"]) <= set(dataset.codebook.genes)
    print(f"FOV_001: {len(fov.all_spots)} detected, {len(fov.good_spots)} retained")
    return fov


def inspect_outputs(fov: FOV) -> dict:
    all_path = fov.save_signal("allSpots", columns=list(fov.all_spots.columns))
    good_path = fov.save_signal("goodSpots")
    log_path = fov.save_log()
    candidates = pd.read_csv(all_path, dtype={"color_seq": str})
    molecules = pd.read_csv(good_path)
    assert list(molecules.columns) == ["x", "y", "z", "gene"]
    assert candidates["color_seq"].str.fullmatch(r"[1-4MN]{4}").all()
    assert len(candidates) == len(fov.all_spots)
    assert len(molecules) == len(fov.good_spots)
    # CSV columns are Cartesian XYZ, but NumPy indexing requires zero-based ZYX.
    coordinates = molecules[["z", "y", "x"]].to_numpy(dtype=int) - 1
    np.testing.assert_array_equal(coordinates, fov.good_spots[["z", "y", "x"]])
    assert (coordinates >= 0).all() and (coordinates < (8, 128, 128)).all()
    reference = fov.images[fov.layers.ref]
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
