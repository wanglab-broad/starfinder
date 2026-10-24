from starfinder.registration import RegistrationRecipe, RegistrationStep, TranslationConfig
from .coordination_helpers import spot_table, detected_shifts
"""Tests for FOV pipeline on small synthetic dataset."""

from starfinder.spot_finding import LocalMaximaConfig
import json

import numpy as np
import pandas as pd
import pytest

from starfinder.dataset import FOV, Dataset, SubtileConfig

pytestmark = pytest.mark.dataset


@pytest.fixture
def small_pipeline_dataset(small_dataset, tmp_path):
    """Create a Dataset pointing at the small synthetic data.

    The small dataset layout is {base}/{fov}/{round}/ but FOV.input_dir()
    expects {input_root}/{round}/{fov}/. This fixture creates symlinks
    in the expected layout.
    """
    # Restructure: small/FOV_001/round1/ -> tmp/round1/FOV_001/
    fov_dir = small_dataset / "FOV_001"
    for round_dir in fov_dir.iterdir():
        if round_dir.is_dir():
            target = tmp_path / round_dir.name / "FOV_001"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(round_dir)

    ds = Dataset(
        input_root=tmp_path,
        output_root=tmp_path / "output",
        dataset_id="test",
        sample_id="small",
        output_id="out",
        rounds=__import__("starfinder.dataset.types", fromlist=["RoundState"]).RoundState(
            sequencing_rounds=["round1", "round2", "round3", "round4"],
            reference_round="round1",
        ),
        channel_order=["ch00", "ch01", "ch02", "ch03"],
        fov_pattern="FOV_%03d",
    )
    ds.load_codebook(small_dataset / "codebook.csv")
    return ds


class TestFOVPipeline:
    """Integration test: full pipeline on small synthetic dataset."""

    def test_load_raw_images(self, small_pipeline_dataset):
        fov = small_pipeline_dataset.fov("FOV_001")
        fov.load_images()

        assert len(fov.images) == 4
        for r in ["round1", "round2", "round3", "round4"]:
            assert r in fov.images
            assert fov.images[r].ndim == 4  # (Z, Y, X, C)
            assert fov.images[r].shape[3] == 4  # 4 channels

    def test_enhance_contrast(self, small_pipeline_dataset):
        fov = small_pipeline_dataset.fov("FOV_001")
        fov.load_images().normalize_min_max()

        for r in fov.images:
            assert fov.images[r].dtype == np.uint8

    def test_global_registration(self, small_pipeline_dataset):
        fov = small_pipeline_dataset.fov("FOV_001")
        fov.load_images().normalize_min_max().register(RegistrationRecipe((RegistrationStep(TranslationConfig()),)))

        # Reference round should not be in global_shifts
        assert "round1" not in detected_shifts(fov)
        # Other rounds should have shifts
        for r in ["round2", "round3", "round4"]:
            assert r in detected_shifts(fov)
            assert len(detected_shifts(fov)[r]) == 3  # (dz, dy, dx)

    def test_spot_finding(self, small_pipeline_dataset):
        fov = small_pipeline_dataset.fov("FOV_001")
        fov.load_images().normalize_min_max().register(RegistrationRecipe((RegistrationStep(TranslationConfig()),)))
        fov.find_spots(config=LocalMaximaConfig())

        assert spot_table(fov) is not None
        assert len(spot_table(fov)) > 0
        for col in ["spot_id", "spot_namespace", "z", "y", "x", "peak_intensity", "channel"]:
            assert col in spot_table(fov).columns

    def test_reads_extraction(self, small_pipeline_dataset):
        fov = small_pipeline_dataset.fov("FOV_001")
        (
            fov.load_images()
            .normalize_min_max()
            .register(RegistrationRecipe((RegistrationStep(TranslationConfig()),))).find_spots(config=LocalMaximaConfig())
            .extract_intensities().decode_reads()
        )

        assert "observed_color_sequence" in spot_table(fov).columns
        assert fov.decoding_result.diagnostics['wta_round_l2_nll'].shape == (len(fov.spot_result.spots), 4)

        # Each observed color sequence has 4 characters (one per round)
        for seq in spot_table(fov)["observed_color_sequence"]:
            assert len(seq) == 4

    def test_reads_filtration(self, small_pipeline_dataset, small_dataset):
        fov = small_pipeline_dataset.fov("FOV_001")
        (
            fov.load_images()
            .normalize_min_max()
            .register(RegistrationRecipe((RegistrationStep(TranslationConfig()),))).find_spots(config=LocalMaximaConfig())
            .extract_intensities().decode_reads()
        )
        small_pipeline_dataset.load_codebook(small_dataset / "codebook.csv")
        fov.filter_reads()

        assert spot_table(fov, accepted=True) is not None
        assert "gene_id" in spot_table(fov, accepted=True).columns

    def test_save_signal(self, small_pipeline_dataset, small_dataset):
        fov = small_pipeline_dataset.fov("FOV_001")
        (
            fov.load_images()
            .normalize_min_max()
            .register(RegistrationRecipe((RegistrationStep(TranslationConfig()),))).find_spots(config=LocalMaximaConfig())
            .extract_intensities().decode_reads()
        )
        small_pipeline_dataset.load_codebook(small_dataset / "codebook.csv")
        fov.filter_reads()

        # Only save if there are good spots
        if len(spot_table(fov, accepted=True)) > 0:
            path = fov.export_spots(slot="molecules")
            assert path.exists()

            df = pd.read_csv(path)
            assert "x" in df.columns
            assert "y" in df.columns
            assert "z" in df.columns
            assert "gene_id" in df.columns

            # Coordinates should be 0-based (minimum >= 0)
            assert df["x"].min() >= 0
            assert df["y"].min() >= 0
            assert df["z"].min() >= 0

    def test_save_ref_merged(self, small_pipeline_dataset):
        fov = small_pipeline_dataset.fov("FOV_001")
        fov.load_images()
        path = fov.save_reference_image()
        assert path.exists()

    def test_fluent_chaining(self, small_pipeline_dataset):
        """Verify methods return self for chaining."""
        fov = small_pipeline_dataset.fov("FOV_001")
        result = fov.load_images().normalize_min_max()
        assert result is fov


    def test_rotate(self, small_pipeline_dataset):
        """FOV.rotate() should rotate all volumes in the YX plane."""
        fov = small_pipeline_dataset.fov("FOV_001")
        fov.load_images()
        original_shape = fov.images["round1"].shape
        # Capture a pixel before rotation
        original_val = fov.images["round1"].copy()
        fov.rotate(angle=-90)
        # Shape should be preserved (reshape=False)
        assert fov.images["round1"].shape == original_shape
        # Content should have changed (non-trivial rotation)
        assert not np.array_equal(fov.images["round1"], original_val)
        # All rounds should be rotated
        for r in ["round1", "round2", "round3", "round4"]:
            assert fov.images[r].shape == original_shape


class TestFOVSubtile:
    """Tests for subtile operations."""

    def test_create_and_load_subtiles(self, small_pipeline_dataset):
        fov = small_pipeline_dataset.fov("FOV_001")
        fov.load_images()

        # Configure subtiles (2x2 grid)
        small_pipeline_dataset.subtile = SubtileConfig(
            sqrt_pieces=2, overlap_ratio=0.1
        )
        h, w = fov.images["round1"].shape[1:3]
        small_pipeline_dataset.subtile.compute_windows(h, w)

        coords_df = fov.save_subtiles()
        assert len(coords_df) == 4
        assert set(coords_df.columns) == {
            "t", "scoords_x", "scoords_y", "ecoords_x", "ecoords_y",
        }

        # Coordinates should be 1-based
        assert coords_df["scoords_x"].min() >= 1
        assert coords_df["scoords_y"].min() >= 1

        # Load back from NPZ
        npz_path = fov.paths.subtile_dir / "subtile_data_1.npz"
        assert npz_path.exists()

        loaded = FOV.from_subtile(
            npz_path, small_pipeline_dataset, "FOV_001"
        )
        assert len(loaded.images) == 4
        for r in ["round1", "round2", "round3", "round4"]:
            assert r in loaded.images


class TestFOVPaths:
    """Tests for FOV path helpers."""

    def test_paths(self, small_pipeline_dataset):
        fov = small_pipeline_dataset.fov("FOV_001")
        p = fov.paths

        assert p.fov_dir == small_pipeline_dataset.output_root / "fovs" / "FOV_001"
        assert p.checkpoint_dir == p.fov_dir / "checkpoints" and p.run_record == p.fov_dir / "run.json"
        assert "subtile" in str(p.subtile_dir)
        assert p.signal_csv("goodSpots").name == "FOV_001_goodSpots.csv"

    def test_paths_keep_the_attributes_of_the_base_revision(self, tmp_path):
        """FOV.paths keeps its class name and every public attribute it had before the path object moved to
        starfinder._layout (W-548; the list recorded at 81fead1), except the unused signal PNG path helper, which
        W-562 deleted (docs/migration.md), with spot_table, which W-562 added, without the log paths gr_log,
        nr_log, rsf_log, score_log and shift_log, whose files W-563 removed (docs/migration.md), and with fov_dir
        and run_record in place of ref_merged_tif, which W-564 replaced by the one reference-image resolver
        (docs/migration.md)."""
        fov = Dataset(tmp_path / "in", tmp_path / "out", "d", "s", "o").fov("FOV_001")
        assert type(fov.paths).__name__ == "_FovPaths"
        assert sorted(name for name in dir(fov.paths) if not name.startswith("_")) == [
            "checkpoint_dir", "fov_dir", "fov_id", "output_root", "run_record", "signal_csv", "spot_table",
            "subtile_dir"]
