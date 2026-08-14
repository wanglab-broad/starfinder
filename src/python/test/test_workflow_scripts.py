"""Regression tests for standalone Snakemake workflow scripts."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd

from starfinder.io import save_stack

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_stitch_subtile_plots_4d_reference_image(tmp_path):
    """The subtile stitcher should render a (Z, Y, X, C) reference TIFF."""
    base_path = tmp_path / "dataset" / "output"
    subtile_path = base_path / "output" / "subtile" / "FOV_001"
    signal_path = base_path / "signal"
    image_path = base_path / "images" / "ref_merged" / "FOV_001.tif"
    subtile_path.mkdir(parents=True)
    signal_path.mkdir(parents=True)

    coords_path = subtile_path / "subtile_coords.csv"
    pd.DataFrame(
        [
            {
                "t": 1,
                "scoords_x": 1,
                "scoords_y": 1,
                "ecoords_x": 8,
                "ecoords_y": 8,
            }
        ]
    ).to_csv(coords_path, index=False)
    pd.DataFrame([{"x": 4, "y": 5, "z": 1, "gene": "Gene1"}]).to_csv(
        subtile_path / "subtile_goodSpots_1.csv", index=False
    )

    reference_image = np.zeros((2, 8, 8, 4), dtype=np.uint8)
    reference_image[1, 4, 3, 2] = 255
    save_stack(reference_image, image_path)

    reads_output = signal_path / "FOV_001_goodSpots.csv"
    preview_output = signal_path / "FOV_001_goodSpots.png"
    snakemake = SimpleNamespace(
        wildcards=SimpleNamespace(fovID="FOV_001"),
        config={
            "root_output_path": str(tmp_path),
            "dataset_id": "dataset",
            "output_id": "output",
        },
        input=[str(coords_path)],
        output=[str(reads_output), str(preview_output)],
    )

    runpy.run_path(
        str(REPO_ROOT / "workflow" / "scripts" / "stitch_subtile.py"),
        init_globals={"snakemake": snakemake},
    )

    assert reads_output.exists()
    assert preview_output.exists()
