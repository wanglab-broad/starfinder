"""End-to-end pipeline validation against synthetic ground truth.

Validates that the full pipeline (load → enhance → register → spot_find →
extract → filter) produces quantitatively correct results by comparing
against known ground truth from the mini synthetic dataset.

Uses the session-scoped ``e2e_result`` fixture from conftest.py, which
runs the pipeline once and shares results across all tests.
"""

import pandas as pd
import pytest

from starfinder.testdata.validation import compare_genes, compare_shifts, compare_spots


class TestE2EPipelineSmokeTest:
    """Basic sanity: pipeline runs and produces non-empty output."""

    def test_pipeline_produces_output(self, e2e_result):
        fov, ds, gt = e2e_result

        assert fov.good_spots is not None
        assert len(fov.good_spots) > 0
        assert "gene" in fov.good_spots.columns
        assert fov.good_spots["gene"].nunique() >= 1


class TestE2EShiftRecovery:
    """Validate global registration recovers known inter-round shifts."""

    def test_shift_recovery(self, e2e_result):
        fov, ds, gt = e2e_result
        result = compare_shifts(fov.global_shifts, gt, "FOV_001", tolerance=1.5)

        # Print metrics for calibration
        for round_name, info in result["per_round"].items():
            print(
                f"  {round_name}: gt={info['gt']}, "
                f"detected={info['detected']}, error={info['error']}"
            )

        for round_name, info in result["per_round"].items():
            for axis, axis_name in enumerate(["dz", "dy", "dx"]):
                assert info["error"][axis] < 1.5, (
                    f"{round_name} {axis_name}: error={info['error'][axis]:.2f}px "
                    f"(gt={info['gt']}, detected={info['detected']})"
                )

    def test_shift_log_csv_matches(self, e2e_result):
        """Shift log CSV values must match fov.global_shifts exactly."""
        fov, ds, gt = e2e_result
        shift_path = fov.paths.shift_log()
        assert shift_path.exists(), f"Shift log not found: {shift_path}"

        df = pd.read_csv(shift_path)
        for _, row in df.iterrows():
            round_name = row["round"]
            dz, dy, dx = fov.global_shifts[round_name]
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
        result = compare_spots(
            fov.all_spots, gt, "FOV_001", position_tolerance=5.0
        )

        print(
            f"\n  Spot detection: recall={result['recall']:.3f}, "
            f"precision={result['precision']:.3f}, "
            f"mean_dist={result['mean_distance']:.2f}px, "
            f"matched={result['n_matched']}/{result['n_gt']} GT spots, "
            f"detected={result['n_detected']} total"
        )

        assert result["recall"] >= 0.7, (
            f"Recall {result['recall']:.3f} < 0.7 "
            f"({result['n_matched']}/{result['n_gt']} matched)"
        )
        # With noise-based threshold (k=5σ above noise floor),
        # observed 20 spots for 20 GT → precision ~1.0.
        assert result["precision"] >= 0.5, (
            f"Precision {result['precision']:.3f} < 0.005 "
            f"({result['n_matched']}/{result['n_detected']} near GT)"
        )

    def test_spot_positions_reasonable(self, e2e_result):
        """All detected spot coordinates within image bounds."""
        fov, ds, gt = e2e_result
        Z, Y, X = gt["image_shape"]
        spots = fov.all_spots

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
        result = compare_genes(
            fov.all_spots, gt, "FOV_001", position_tolerance=5.0
        )

        print(
            f"\n  Color seq accuracy: {result['color_seq_accuracy']:.3f} "
            f"({result['correct_color_seq']}/{result['n_matched']} matched)"
        )

    def test_gene_accuracy(self, e2e_result):
        """Decoded gene labels match ground truth for spatially matched spots."""
        fov, ds, gt = e2e_result
        result = compare_genes(
            fov.good_spots, gt, "FOV_001", position_tolerance=5.0
        )

        print(
            f"\n  Gene accuracy: {result['gene_accuracy']:.3f} "
            f"({result['correct_genes']}/{result['n_matched']} matched)"
        )
        if result["gene_confusion"]:
            print(f"  Confusion: {result['gene_confusion']}")

        assert result["gene_accuracy"] >= 0.5, (
            f"Gene accuracy {result['gene_accuracy']:.3f} < 0.5"
        )

        # All gene labels must be valid codebook entries
        assert fov.good_spots["gene"].notna().all(), "NaN gene values found"
        codebook_genes = set(ds.codebook.gene_to_seq.keys())
        detected_genes = set(fov.good_spots["gene"])
        assert detected_genes.issubset(codebook_genes), (
            f"Unknown genes: {detected_genes - codebook_genes}"
        )


class TestE2ESubtileRoundTrip:
    """Validate subtile coordinate mapping after spot finding."""

    def test_subtile_spot_coordinate_mapping(self, e2e_result):
        """Spots found in a subtile remap to valid global coordinates."""
        from starfinder.dataset import FOV, SubtileConfig
        from starfinder.spotfinding import find_spots_3d

        fov, ds, gt = e2e_result
        Z, Y, X = gt["image_shape"]

        # Configure 2x2 subtiles
        ds.subtile = SubtileConfig(sqrt_pieces=2, overlap_ratio=0.1)
        h, w = fov.images["round1"].shape[1:3]
        ds.subtile.compute_windows(h, w)

        coords_df = fov.create_subtiles()

        # Load first subtile and run spot finding
        npz_path = fov.paths.subtile_dir / "subtile_00000.npz"
        sub_fov = FOV.from_subtile(npz_path, ds, "FOV_001")

        ref_image = sub_fov.images[ds.layers.ref]
        sub_spots = find_spots_3d(ref_image)

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
