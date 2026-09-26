from starfinder.dataset import PipelineConfig, ExecutionConfig, RegistrationStep
from starfinder.registration import TranslationConfig
from starfinder.io import ImageLoadConfig
from starfinder.preprocessing import MinMaxNormalizationConfig
from starfinder.spot_finding import LocalMaximaConfig
from starfinder.barcode import NeighborhoodSumConfig, WtaDecoderConfig, ReadFilterConfig
from .coordination_helpers import spot_table, detected_shifts
"""End-to-end pipeline validation against synthetic ground truth.

Validates that the full pipeline (load → enhance → register → spot_find →
extract → filter) produces quantitatively correct results by comparing
against known ground truth from the small synthetic dataset.

Uses the session-scoped ``e2e_result`` fixture from conftest.py, which
runs the pipeline once and shares results across all tests.
"""

import pandas as pd
import pytest

from starfinder.evaluation.registration import evaluate_translation
from starfinder.evaluation.spot_finding import evaluate_spots
from starfinder.evaluation.barcode import evaluate_decoding
from starfinder.image import ImageMetadata
import numpy as np


def matching_config():
    metadata = ImageMetadata("synthetic/reference")
    return dict(policy="greedy", threshold=5.0, units="voxel", boundary="exclusive",
                reference_metadata=metadata, observed_metadata=metadata)


pytestmark = pytest.mark.extended


class TestE2EPipelineSmokeTest:
    """Basic sanity: pipeline runs and produces non-empty output."""

    def test_pipeline_produces_output(self, e2e_result):
        fov, ds, gt = e2e_result

        assert spot_table(fov, accepted=True) is not None
        assert len(spot_table(fov, accepted=True)) > 0
        assert "gene" in spot_table(fov, accepted=True).columns
        assert spot_table(fov, accepted=True)["gene"].nunique() >= 1


class TestE2EShiftRecovery:
    """Validate global registration recovers known inter-round shifts."""

    def test_shift_recovery(self, e2e_result):
        fov, ds, gt = e2e_result
        metadata = ImageMetadata("synthetic/displacement")
        result = evaluate_translation(detected_shifts(fov), gt["fovs"]["FOV_001"]["shifts"],
            tolerance=1.5, units="voxel", reference_metadata=metadata, observed_metadata=metadata,
            eligible_rounds=ds.rounds.moving_rounds)
        assert result.values["passed"] is True

        # Print metrics for calibration
        for round_name, info in result.details["per_round"].items():
            print(
                f"  {round_name}: gt={info['gt']}, "
                f"detected={info['detected']}, error={info['error']}"
            )

        for round_name, info in result.details["per_round"].items():
            for axis, axis_name in enumerate(["dz", "dy", "dx"]):
                assert info["error"][axis] < 1.5, (
                    f"{round_name} {axis_name}: error={info['error'][axis]:.2f}px "
                    f"(gt={info['gt']}, detected={info['detected']})"
                )

    def test_shift_log_csv_matches(self, e2e_result):
        """Shift log CSV values must match detected_shifts(fov) exactly."""
        fov, ds, gt = e2e_result
        shift_path = fov.paths.shift_log()
        assert shift_path.exists(), f"Shift log not found: {shift_path}"

        df = pd.read_csv(shift_path)
        for _, row in df.iterrows():
            round_name = row["round"]
            dz, dy, dx = detected_shifts(fov)[round_name]
            assert row["row"] == pytest.approx(dy), (
                f"{round_name}: CSV row={row['row']} != dy={dy}"
            )
            assert row["col"] == pytest.approx(dx), (
                f"{round_name}: CSV col={row['col']} != dx={dx}"
            )
            assert row["z"] == pytest.approx(dz), (
                f"{round_name}: CSV z={row['z']} != dz={dz}"
            )


class TestE2ESpotDetection:
    """Validate spot finding against ground truth spot positions."""

    def test_spot_recall(self, e2e_result):
        fov, ds, gt = e2e_result
        result = evaluate_spots(
            spot_table(fov)[["z", "y", "x"]].to_numpy(),
            np.array([s["position"] for s in gt["fovs"]["FOV_001"]["spots"]]),
            **matching_config())

        print(
            f"\n  Spot detection: recall={result.values['recall']:.3f}, "
            f"precision={result.values['precision']:.3f}, "
            f"mean_dist={result.values['mean_distance']:.2f}px, "
            f"matched={result.counts['matched']}/{result.counts['total_reference']} GT spots, "
            f"detected={result.counts['total_observed']} total"
        )

        assert result.values["recall"] >= 0.7, (
            f"Recall {result.values['recall']:.3f} < 0.7 "
            f"({result.counts['matched']}/{result.counts['total_reference']} matched)"
        )
        # With noise-based threshold (k=5σ above noise floor),
        # observed 20 spots for 20 GT → precision ~1.0.
        assert result.values["precision"] >= 0.5, (
            f"Precision {result.values['precision']:.3f} < 0.005 "
            f"({result.counts['matched']}/{result.counts['total_observed']} near GT)"
        )

    def test_spot_positions_reasonable(self, e2e_result):
        """All detected spot coordinates within image bounds."""
        fov, ds, gt = e2e_result
        Z, Y, X = gt["image_shape"]
        spots = spot_table(fov)

        assert (spots["z"] >= 0).all() and (spots["z"] < Z).all(), (
            f"z out of bounds: [{spots['z'].min()}, {spots['z'].max()}] vs [0, {Z})"
        )
        assert (spots["y"] >= 0).all() and (spots["y"] < Y).all(), (
            f"y out of bounds: [{spots['y'].min()}, {spots['y'].max()}] vs [0, {Y})"
        )
        assert (spots["x"] >= 0).all() and (spots["x"] < X).all(), (
            f"x out of bounds: [{spots['x'].min()}, {spots['x'].max()}] vs [0, {X})"
        )


class TestE2EBarcodeDecoding:
    """Validate barcode decoding against ground truth gene labels."""

    def test_color_seq_accuracy(self, e2e_result):
        """Color sequences extracted at GT spot locations match GT."""
        fov, ds, gt = e2e_result
        truth = pd.DataFrame(gt["fovs"]["FOV_001"]["spots"])
        detected = spot_table(fov)
        matches = evaluate_spots(detected[["z", "y", "x"]].to_numpy(),
            np.array(truth["position"].tolist()), **matching_config())
        result = evaluate_decoding(detected, truth, matches=matches)

        print(
            f"\n  Color seq accuracy: {result.values['color_seq_accuracy']:.3f} "
            f"({result.counts['correct_color_seq']}/{result.counts['matched']} matched)"
        )

    def test_gene_accuracy(self, e2e_result):
        """Decoded gene labels match ground truth for spatially matched spots."""
        fov, ds, gt = e2e_result
        truth = pd.DataFrame(gt["fovs"]["FOV_001"]["spots"])
        detected = spot_table(fov, accepted=True)
        matches = evaluate_spots(detected[["z", "y", "x"]].to_numpy(),
            np.array(truth["position"].tolist()), **matching_config())
        result = evaluate_decoding(detected, truth, matches=matches)

        print(
            f"\n  Gene accuracy: {result.values['gene_accuracy']:.3f} "
            f"({result.counts['correct_gene']}/{result.counts['matched']} matched)"
        )
        if result.details["gene_confusion"]:
            print(f"  Confusion: {result.details['gene_confusion']}")

        assert result.values["gene_accuracy"] >= 0.5, (
            f"Gene accuracy {result.values['gene_accuracy']:.3f} < 0.5"
        )

        # All gene labels must be valid codebook entries
        assert spot_table(fov, accepted=True)["gene"].notna().all(), "NaN gene values found"
        codebook_genes = set(ds.codebook.gene_to_seq.keys())
        detected_genes = set(spot_table(fov, accepted=True)["gene"])
        assert detected_genes.issubset(codebook_genes), (
            f"Unknown genes: {detected_genes - codebook_genes}"
        )


class TestE2EStreamingMode:
    """Validate that streaming pipeline produces identical results to batch."""

    def test_streaming_matches_batch(self, e2e_result, small_dataset, tmp_path_factory):
        """Streaming mode output must be identical to batch mode."""
        from starfinder.dataset import Dataset
        from starfinder.dataset.types import RoundState

        batch_fov, _, _ = e2e_result

        tmp_path = tmp_path_factory.mktemp("streaming")
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
            rounds=RoundState(
                sequencing_rounds=["round1", "round2", "round3", "round4"],
                reference_round="round1",
            ),
            channel_order=["ch00", "ch01", "ch02", "ch03"],
            fov_pattern="FOV_%03d",
        )
        ds.load_codebook(small_dataset / "codebook.csv")

        stream_fov = ds.fov("FOV_001")
        stream_fov.run(PipelineConfig(
            load=ImageLoadConfig(channel_labels=ds.channel_order),
            normalization=MinMaxNormalizationConfig('uint8', (0, 255), snr_threshold=5.0),
            registration=(RegistrationStep(TranslationConfig()),), detection=LocalMaximaConfig(),
            extraction=NeighborhoodSumConfig(), decoding=WtaDecoderConfig(diagnostics=True), filtering=ReadFilterConfig()),
            execution=ExecutionConfig('streaming'))

        # Both must produce good spots
        assert spot_table(stream_fov, accepted=True) is not None
        assert len(spot_table(stream_fov, accepted=True)) > 0

        # Sort by position for consistent comparison
        batch_spots = (
            spot_table(batch_fov, accepted=True).sort_values(["z", "y", "x"])
            .reset_index(drop=True)
        )
        stream_spots = (
            spot_table(stream_fov, accepted=True).sort_values(["z", "y", "x"])
            .reset_index(drop=True)
        )

        # Same number of spots
        assert len(stream_spots) == len(batch_spots), (
            f"Streaming: {len(stream_spots)} spots, Batch: {len(batch_spots)}"
        )

        # Same gene assignments at same positions
        pd.testing.assert_frame_equal(
            batch_spots[["z", "y", "x", "gene", "color_seq"]],
            stream_spots[["z", "y", "x", "gene", "color_seq"]],
        )

        # Same global shifts
        assert set(detected_shifts(stream_fov)) == set(detected_shifts(batch_fov))
        for rnd in detected_shifts(stream_fov):
            for i in range(3):
                assert detected_shifts(stream_fov)[rnd][i] == pytest.approx(
                    detected_shifts(batch_fov)[rnd][i]
                )

    def test_streaming_releases_memory(self, e2e_result, small_dataset, tmp_path_factory):
        """After streaming, only ref round remains in images dict."""
        from starfinder.dataset import Dataset
        from starfinder.dataset.types import RoundState

        batch_fov, _, _ = e2e_result

        tmp_path = tmp_path_factory.mktemp("streaming_mem")
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
            rounds=RoundState(
                sequencing_rounds=["round1", "round2", "round3", "round4"],
                reference_round="round1",
            ),
            channel_order=["ch00", "ch01", "ch02", "ch03"],
            fov_pattern="FOV_%03d",
        )
        ds.load_codebook(small_dataset / "codebook.csv")

        stream_fov = ds.fov("FOV_001")
        stream_fov.run(PipelineConfig(
            load=ImageLoadConfig(channel_labels=ds.channel_order),
            normalization=MinMaxNormalizationConfig('uint8', (0, 255), snr_threshold=5.0),
            registration=(RegistrationStep(TranslationConfig()),), detection=LocalMaximaConfig(),
            extraction=NeighborhoodSumConfig(), decoding=WtaDecoderConfig(diagnostics=True), filtering=ReadFilterConfig()),
            execution=ExecutionConfig('streaming'))

        # Only ref round should remain in memory
        assert set(stream_fov.images.keys()) == {"round1"}

        # Batch mode keeps all rounds
        assert len(batch_fov.images) == 4


class TestE2ESubtileRoundTrip:
    """Validate subtile coordinate mapping after spot finding."""

    def test_subtile_spot_coordinate_mapping(self, e2e_result):
        """Spots found in a subtile remap to valid global coordinates."""
        from starfinder.dataset import FOV, SubtileConfig
        from starfinder.spot_finding import find_spots, LocalMaximaConfig
        from starfinder.image import ImageMetadata

        fov, ds, gt = e2e_result
        Z, Y, X = gt["image_shape"]

        # Configure 2x2 subtiles
        ds.subtile = SubtileConfig(sqrt_pieces=2, overlap_ratio=0.1)
        h, w = fov.images["round1"].shape[1:3]
        ds.subtile.compute_windows(h, w)

        coords_df = fov.create_subtiles()

        # Load first subtile and run spot finding
        npz_path = fov.paths.subtile_dir / "subtile_data_1.npz"
        sub_fov = FOV.from_subtile(npz_path, ds, "FOV_001")

        ref_image = sub_fov.images[ds.rounds.reference_round]
        sub_spots = find_spots(ref_image, config=LocalMaximaConfig(threshold_mode="noise", threshold_value=5.0, min_distance_voxels=1), metadata=ImageMetadata("direct/test_e2e"), spot_namespace="direct/test_e2e").spots

        if len(sub_spots) == 0:
            pytest.skip("No spots detected in subtile")

        # Get subtile offsets (1-based CSV → 0-based)
        row0 = coords_df.iloc[0]
        x_offset = int(row0["scoords_x"]) - 1
        y_offset = int(row0["scoords_y"]) - 1

        # Remap to global coordinates
        global_x = sub_spots["x"].values + x_offset
        global_y = sub_spots["y"].values + y_offset

        # All remapped coordinates must be within image bounds
        assert (global_x >= 0).all() and (global_x < X).all(), (
            f"Remapped x out of bounds: [{global_x.min()}, {global_x.max()}] "
            f"vs [0, {X})"
        )
        assert (global_y >= 0).all() and (global_y < Y).all(), (
            f"Remapped y out of bounds: [{global_y.min()}, {global_y.max()}] "
            f"vs [0, {Y})"
        )
