"""Tests for FOV pipeline on mini synthetic dataset."""

import json

import numpy as np
import pandas as pd
import pytest

from starfinder.dataset import FOV, STARMapDataset, SubtileConfig


@pytest.fixture
def mini_pipeline_dataset(mini_dataset, tmp_path):
    """Create a STARMapDataset pointing at the mini synthetic data.

    The mini dataset layout is {base}/{fov}/{round}/ but FOV.input_dir()
    expects {input_root}/{round}/{fov}/. This fixture creates symlinks
    in the expected layout.
    """
    # Restructure: mini/FOV_001/round1/ -> tmp/round1/FOV_001/
    fov_dir = mini_dataset / "FOV_001"
    for round_dir in fov_dir.iterdir():
        if round_dir.is_dir():
            target = tmp_path / round_dir.name / "FOV_001"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(round_dir)

    ds = STARMapDataset(
        input_root=tmp_path,
        output_root=tmp_path / "output",
        dataset_id="test",
        sample_id="mini",
        output_id="out",
        layers=__import__("starfinder.dataset.types", fromlist=["LayerState"]).LayerState(
            seq=["round1", "round2", "round3", "round4"],
            ref="round1",
        ),
        channel_order=["ch00", "ch01", "ch02", "ch03"],
        fov_pattern="FOV_%03d",
    )
    return ds


class TestFOVPipeline:
    """Integration test: full pipeline on mini synthetic dataset."""

    def test_load_raw_images(self, mini_pipeline_dataset):
        fov = mini_pipeline_dataset.fov("FOV_001")
        fov.load_raw_images()

        assert len(fov.images) == 4
        for r in ["round1", "round2", "round3", "round4"]:
            assert r in fov.images
            assert fov.images[r].ndim == 4  # (Z, Y, X, C)
            assert fov.images[r].shape[3] == 4  # 4 channels

    def test_enhance_contrast(self, mini_pipeline_dataset):
        fov = mini_pipeline_dataset.fov("FOV_001")
        fov.load_raw_images().enhance_contrast()

        for r in fov.images:
            assert fov.images[r].dtype == np.uint8

    def test_global_registration(self, mini_pipeline_dataset):
        fov = mini_pipeline_dataset.fov("FOV_001")
        fov.load_raw_images().enhance_contrast().global_registration()

        # Reference round should not be in global_shifts
        assert "round1" not in fov.global_shifts
        # Other rounds should have shifts
        for r in ["round2", "round3", "round4"]:
            assert r in fov.global_shifts
            assert len(fov.global_shifts[r]) == 3  # (dz, dy, dx)

        # Shift log should be saved
        shift_path = fov.paths.shift_log()
        assert shift_path.exists()
        shift_df = pd.read_csv(shift_path)
        assert list(shift_df.columns) == ["fov_id", "round", "row", "col", "z"]
        assert len(shift_df) == 3

    def test_spot_finding(self, mini_pipeline_dataset):
        fov = mini_pipeline_dataset.fov("FOV_001")
        fov.load_raw_images().enhance_contrast().global_registration()
        fov.spot_finding()

        assert fov.all_spots is not None
        assert len(fov.all_spots) > 0
        for col in ["z", "y", "x", "intensity", "channel"]:
            assert col in fov.all_spots.columns

    def test_reads_extraction(self, mini_pipeline_dataset):
        fov = mini_pipeline_dataset.fov("FOV_001")
        (
            fov.load_raw_images()
            .enhance_contrast()
            .global_registration()
            .spot_finding()
            .reads_extraction()
        )

        assert "color_seq" in fov.all_spots.columns
        for r in ["round1", "round2", "round3", "round4"]:
            assert f"{r}_color" in fov.all_spots.columns
            assert f"{r}_score" in fov.all_spots.columns

        # color_seq should be 4 characters (one per round)
        for seq in fov.all_spots["color_seq"]:
            assert len(seq) == 4

    def test_reads_filtration(self, mini_pipeline_dataset, mini_dataset):
        fov = mini_pipeline_dataset.fov("FOV_001")
        (
            fov.load_raw_images()
            .enhance_contrast()
            .global_registration()
            .spot_finding()
            .reads_extraction()
        )
        mini_pipeline_dataset.load_codebook(mini_dataset / "codebook.csv")
        fov.reads_filtration()

        assert fov.good_spots is not None
        assert "gene" in fov.good_spots.columns

    def test_save_signal(self, mini_pipeline_dataset, mini_dataset):
        fov = mini_pipeline_dataset.fov("FOV_001")
        (
            fov.load_raw_images()
            .enhance_contrast()
            .global_registration()
            .spot_finding()
            .reads_extraction()
        )
        mini_pipeline_dataset.load_codebook(mini_dataset / "codebook.csv")
        fov.reads_filtration()

        # Only save if there are good spots
        if len(fov.good_spots) > 0:
            path = fov.save_signal(slot="goodSpots")
            assert path.exists()

            df = pd.read_csv(path)
            assert "x" in df.columns
            assert "y" in df.columns
            assert "z" in df.columns
            assert "gene" in df.columns

            # Coordinates should be 1-based (minimum >= 1)
            assert df["x"].min() >= 1
            assert df["y"].min() >= 1
            assert df["z"].min() >= 1

    def test_save_ref_merged(self, mini_pipeline_dataset):
        fov = mini_pipeline_dataset.fov("FOV_001")
        fov.load_raw_images()
        path = fov.save_ref_merged()
        assert path.exists()

    def test_fluent_chaining(self, mini_pipeline_dataset):
        """Verify methods return self for chaining."""
        fov = mini_pipeline_dataset.fov("FOV_001")
        result = fov.load_raw_images().enhance_contrast()
        assert result is fov


class TestFOVSubtile:
    """Tests for subtile operations."""

    def test_create_and_load_subtiles(self, mini_pipeline_dataset):
        fov = mini_pipeline_dataset.fov("FOV_001")
        fov.load_raw_images()

        # Configure subtiles (2x2 grid)
        mini_pipeline_dataset.subtile = SubtileConfig(
            sqrt_pieces=2, overlap_ratio=0.1
        )
        h, w = fov.images["round1"].shape[1:3]
        mini_pipeline_dataset.subtile.compute_windows(h, w)

        coords_df = fov.create_subtiles()
        assert len(coords_df) == 4
        assert set(coords_df.columns) == {
            "t", "scoords_x", "scoords_y", "ecoords_x", "ecoords_y",
        }

        # Coordinates should be 1-based
        assert coords_df["scoords_x"].min() >= 1
        assert coords_df["scoords_y"].min() >= 1

        # Load back from NPZ
        npz_path = fov.paths.subtile_dir / "subtile_00000.npz"
        assert npz_path.exists()

        loaded = FOV.from_subtile(
            npz_path, mini_pipeline_dataset, "FOV_001"
        )
        assert len(loaded.images) == 4
        for r in ["round1", "round2", "round3", "round4"]:
            assert r in loaded.images


class TestFOVPaths:
    """Tests for FOV path helpers."""

    def test_paths(self, mini_pipeline_dataset):
        fov = mini_pipeline_dataset.fov("FOV_001")
        p = fov.paths

        assert "ref_merged" in str(p.ref_merged_tif)
        assert p.ref_merged_tif.name == "FOV_001.tif"
        assert "subtile" in str(p.subtile_dir)
        assert p.signal_csv("goodSpots").name == "FOV_001_goodSpots.csv"
        assert p.shift_log().name == "FOV_001.txt"
