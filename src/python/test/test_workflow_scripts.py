"""Regression tests for standalone Snakemake workflow scripts."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

from starfinder.io import save_volume

pytestmark = pytest.mark.workflow

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
    save_volume(reference_image, image_path)

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


def ref_merged_fov(tmp_path, reference_image):
    """An FOV whose saved reference merged image has the W-235 content."""
    from starfinder.dataset import Dataset, RoundState
    from starfinder.image import ImageMetadata

    ds = Dataset(tmp_path, tmp_path / "dataset" / "output", "dataset", "sample", "output",
                 RoundState(["round1"], reference_round="round1"), ("a", "b", "c", "d"))
    fov = ds.fov("FOV_001")
    fov.images = {"round1": reference_image}
    fov.metadata = {"round1": ImageMetadata("common")}
    return fov


@pytest.mark.parametrize("maximum_projection", [False, True])
def test_stitch_subtile_plots_the_reference_merged_image(tmp_path, maximum_projection):
    """The stitch preview reads the ZYX, or projected YX, reference merged image."""
    from starfinder.preprocessing import ProjectionConfig

    base_path = tmp_path / "dataset" / "output"
    subtile_path = base_path / "output" / "subtile" / "FOV_001"
    signal_path = base_path / "signal"
    subtile_path.mkdir(parents=True)
    signal_path.mkdir(parents=True)
    coords_path = subtile_path / "subtile_coords.csv"
    pd.DataFrame([{"t": 1, "scoords_x": 1, "scoords_y": 1, "ecoords_x": 8, "ecoords_y": 8}]).to_csv(
        coords_path, index=False)
    pd.DataFrame([{"x": 4, "y": 5, "z": 1, "gene": "Gene1"}]).to_csv(
        subtile_path / "subtile_goodSpots_1.csv", index=False)
    reference_image = np.zeros((2, 8, 8, 4), dtype=np.uint16)
    reference_image[1, 4, 3, 2] = 4000
    fov = ref_merged_fov(tmp_path, reference_image)
    fov.save_reference_image(projection=ProjectionConfig() if maximum_projection else None)

    reads_output = signal_path / "FOV_001_goodSpots.csv"
    preview_output = signal_path / "FOV_001_goodSpots.png"
    snakemake = SimpleNamespace(
        wildcards=SimpleNamespace(fovID="FOV_001"),
        config={"root_output_path": str(tmp_path), "dataset_id": "dataset", "output_id": "output"},
        input=[str(coords_path)],
        output=[str(reads_output), str(preview_output)],
    )
    namespace = runpy.run_path(
        str(REPO_ROOT / "workflow" / "scripts" / "stitch_subtile.py"),
        init_globals={"snakemake": snakemake},
    )

    np.testing.assert_array_equal(namespace["ref_merged_img"], reference_image.max(axis=(0, 3)))
    assert preview_output.stat().st_size > 0


@pytest.mark.parametrize("overlay_projection", [False, True])
def test_nuclei_amplicon_overlay_reads_the_reference_merged_image(tmp_path, overlay_projection):
    """The overlay pairs a ZYX DAPI stack with the ZYX reference merged image."""
    import tifffile

    rng = np.random.default_rng(0)
    reference_image = rng.integers(0, 256, size=(5, 8, 9, 4), dtype=np.uint8)
    fov = ref_merged_fov(tmp_path, reference_image)
    amplicon_path = fov.save_reference_image()
    dapi = rng.integers(0, 4096, size=(5, 8, 9), dtype=np.uint16)
    dapi_path = tmp_path / "DAPI.tif"
    tifffile.imwrite(dapi_path, dapi, photometric="minisblack")
    output = tmp_path / "overlay.tif"
    snakemake = SimpleNamespace(
        input={"dapi_img": str(dapi_path), "amplicon_img": str(amplicon_path)},
        output=[str(output)],
        config={"rules": {"create_nuclei_amplicon_overlay": {"parameters": {"maximum_projection": overlay_projection}}}},
    )
    runpy.run_path(
        str(REPO_ROOT / "workflow" / "scripts" / "create_nuclei_amplicon_overlay.py"),
        init_globals={"snakemake": snakemake},
    )

    overlay = tifffile.imread(output)
    assert overlay.dtype == np.uint8
    assert overlay.shape == ((8, 9) if overlay_projection else (5, 8, 9))
