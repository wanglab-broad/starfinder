from starfinder.dataset import from_workflow_config
"""Tests for Dataset class."""

from pathlib import Path

import pytest

from starfinder.dataset import CropWindow, Dataset


@pytest.fixture
def sample_config():
    """Minimal config dict for Dataset."""
    return {
        "root_input_path": "/data/input",
        "root_output_path": "/data/output",
        "dataset_id": "test-dataset",
        "sample_id": "sample-01",
        "output_id": "output-01",
        "n_rounds": 4,
        "ref_round": "round1",
        "fov_id_pattern": "FOV_%03d",
        "rotate_angle": -90.0,
        "maximum_projection": False,
    }


class TestSTARMapDataset:
    """Tests for Dataset creation and methods."""

    def test_from_config(self, sample_config):
        ds = from_workflow_config(sample_config).dataset
        assert ds.dataset_id == "test-dataset"
        assert ds.input_root == Path("/data/input/test-dataset/sample-01")
        assert ds.output_root == Path("/data/output/test-dataset/output-01")
        assert ds.rounds.sequencing_rounds == ["round1", "round2", "round3", "round4"]
        assert ds.rounds.reference_round == "round1"
        assert from_workflow_config(sample_config).pipeline.rotation_degrees == -90.0

    def test_fov_factory(self, sample_config):
        ds = from_workflow_config(sample_config).dataset
        fov = ds.fov("FOV_001")
        assert fov.fov_id == "FOV_001"
        assert fov.dataset is ds
        assert fov.rounds is ds.rounds

    def test_fov_ids(self, sample_config):
        ds = from_workflow_config(sample_config).dataset
        ids = ds.fov_ids(3, start=1)
        assert ids == ["FOV_001", "FOV_002", "FOV_003"]

    def test_load_codebook(self, sample_config, small_dataset):
        sample_config["seq_channel_order"] = ["ch00", "ch01", "ch02", "ch03"]
        ds = from_workflow_config(sample_config).dataset
        assert ds.codebook is None
        ds.load_codebook(small_dataset / "codebook.csv")
        assert ds.codebook is not None
        assert ds.codebook.n_genes == 8

    def test_channel_order_default(self, sample_config):
        ds = from_workflow_config(sample_config).dataset
        assert ds.channel_order == ()

    def test_channel_order_from_config(self, sample_config):
        sample_config["seq_channel_order"] = ["ch00", "ch01", "ch02", "ch03"]
        ds = from_workflow_config(sample_config).dataset
        assert ds.channel_order == ("ch00", "ch01", "ch02", "ch03")

    @pytest.mark.parametrize(
        "rule_name", ["gr_single_fov_subtile", "deep_create_subtile"]
    )
    def test_from_config_builds_subtile_windows(self, sample_config, rule_name):
        sample_config.update(
            {
                "img_row": 100,
                "img_col": 100,
                "rules": {
                    rule_name: {
                        "parameters": {
                            "create_subtiles": {"run": True, "sqrt_pieces": 2}
                        }
                    }
                },
            }
        )

        ds = from_workflow_config(sample_config).dataset

        assert ds.subtile is not None
        assert ds.subtile.sqrt_pieces == 2
        assert ds.subtile.windows == [
            CropWindow(0, 52, 0, 52),
            CropWindow(0, 52, 48, 100),
            CropWindow(48, 100, 0, 52),
            CropWindow(48, 100, 48, 100),
        ]
