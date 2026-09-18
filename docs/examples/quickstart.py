"""Generate tiny synthetic TIFFs and process them to molecule-level CSVs.

Run from src/python with a new external output directory; see getting-started.md.
"""

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from starfinder.benchmark.synthetic import generate_synthetic_dataset, get_preset_config
from starfinder.dataset import LayerState, STARMapDataset


def main(output: Path) -> None:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = get_preset_config("tiny")
    config.seed = 42
    (output / "synthetic_config.json").write_text(json.dumps(asdict(config), indent=2))
    truth = generate_synthetic_dataset(output / "synthetic", config=config, preset="tiny")

    rounds = [f"round{i}" for i in range(1, config.n_rounds + 1)]
    channels = [f"ch{i:02d}" for i in range(config.n_channels)]
    # The generator writes FOV/round; the dataset loader expects round/FOV.
    # Copy these small inputs so both layouts are inspectable and portable.
    for fov_id in truth["fovs"]:
        for round_name in rounds:
            shutil.copytree(
                output / "synthetic" / fov_id / round_name,
                output / "input" / round_name / fov_id,
            )

    dataset = STARMapDataset(
        input_root=output / "input",
        output_root=output / "results",
        dataset_id="tiny",
        sample_id="synthetic",
        output_id="quickstart",
        layers=LayerState(seq=rounds, ref="round1"),
        channel_order=channels,
        fov_pattern="FOV_%03d",
    )
    dataset.layers.validate()
    dataset.load_codebook(output / "synthetic" / "codebook.csv", do_reverse=True)
    summary = {"preset": "tiny", "seed": config.seed, "fovs": {}}
    for fov_id in dataset.fov_ids(config.n_fovs, start=1):
        fov = dataset.fov(fov_id)
        fov.load_raw_images()
        for volume in fov.images.values():
            assert volume.shape == (8, 128, 128, 4)
            assert volume.dtype == np.uint8
        fov.global_registration(ref_img="merged", mov_img="merged", save_shifts=True)
        fov.spot_finding(
            intensity_estimation="noise", intensity_threshold=5.0, min_distance=1,
        )
        fov.reads_extraction(voxel_size=(1, 2, 2))
        fov.reads_filtration()

        # These are completion checks, not a benchmark of detection accuracy.
        assert 0 < len(fov.good_spots) <= len(fov.all_spots)
        assert set(fov.good_spots["gene"]) <= set(dataset.codebook.genes)
        assert fov.good_spots["color_seq"].str.fullmatch(r"[1-4]{4}").all()
        detected_shifts = {}
        for round_name, shift in fov.global_shifts.items():
            expected = truth["fovs"][fov_id]["shifts"][round_name]
            np.testing.assert_allclose(shift, expected, atol=1, rtol=0)
            detected_shifts[round_name] = [float(value) for value in shift]

        # Keep diagnostic columns in allSpots; the standard goodSpots CSV is x,y,z,gene.
        all_path = fov.save_signal("allSpots", columns=list(fov.all_spots.columns))
        good_path = fov.save_signal("goodSpots")
        fov.save_log()
        all_saved = pd.read_csv(all_path, dtype={"color_seq": str})
        good_saved = pd.read_csv(good_path)
        assert list(good_saved.columns) == ["x", "y", "z", "gene"]
        assert len(all_saved) == len(fov.all_spots)
        assert len(good_saved) == len(fov.good_spots)
        np.testing.assert_array_equal(
            good_saved[["x", "y", "z"]], fov.good_spots[["x", "y", "z"]] + 1,
        )
        for axis, size in zip(("z", "y", "x"), (8, 128, 128)):
            assert all_saved[axis].between(1, size).all()
        record = {
            "generated_spots": len(truth["fovs"][fov_id]["spots"]),
            "detected_spots": len(fov.all_spots),
            "retained_reads": len(fov.good_spots),
            "detected_shifts_zyx": detected_shifts,
            "gene_counts": {gene: int(n) for gene, n in good_saved["gene"].value_counts().items()},
            "molecules_csv": str(good_path.relative_to(output)),
        }
        summary["fovs"][fov_id] = record
        print(f"{fov_id}: {record['detected_spots']} detected, {record['retained_reads']} retained")
        print(good_saved.head().to_string(index=False))
        del fov  # Process FOVs sequentially without retaining their image arrays.

    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Quickstart checks passed. Outputs: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New output directory outside the checkout")
    main(parser.parse_args().output)
