"""Ground truth comparison utilities for e2e validation.

Compares pipeline output (shifts, spots, genes) against ground_truth.json
from synthetic datasets. All functions operate in 0-based coordinates
matching the internal pipeline convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def compare_shifts(
    detected_shifts: dict,
    ground_truth: dict,
    fov_id: str,
    tolerance: float = 1.5,
) -> dict:
    """Compare detected global shifts against ground truth shifts.

    Parameters
    ----------
    detected_shifts : dict[str, tuple[float, float, float]]
        ``fov.global_shifts`` — maps round name to (dz, dy, dx).
        Reference round is absent (no self-shift).
    ground_truth : dict
        Loaded ``ground_truth.json``.
    fov_id : str
        FOV key in ground_truth (e.g. ``"FOV_001"``).
    tolerance : float
        Maximum allowed per-axis error in pixels.

    Returns
    -------
    dict with ``per_round``, ``max_error``, ``passed``, ``tolerance``.
    """
    gt_shifts = ground_truth["fovs"][fov_id]["shifts"]
    results = {}
    max_error = 0.0

    for round_name, gt_shift in gt_shifts.items():
        if round_name not in detected_shifts:
            # Reference round — not in detected_shifts
            continue
        gt = np.array(gt_shift, dtype=float)
        det = np.array(detected_shifts[round_name], dtype=float)
        error = np.abs(det - gt)
        max_axis_error = float(error.max())
        max_error = max(max_error, max_axis_error)
        results[round_name] = {
            "gt": gt.tolist(),
            "detected": det.tolist(),
            "error": error.tolist(),
            "max_axis_error": max_axis_error,
        }

    return {
        "per_round": results,
        "max_error": max_error,
        "passed": max_error < tolerance,
        "tolerance": tolerance,
    }


def compare_spots(
    detected_spots: pd.DataFrame,
    ground_truth: dict,
    fov_id: str,
    position_tolerance: float = 5.0,
) -> dict:
    """Match detected spots to ground truth spots by proximity.

    Uses greedy nearest-neighbor matching with exclusion:
    sort all (gt, det) pairs by distance, greedily assign closest
    unmatched pairs. A match requires distance < position_tolerance.

    Parameters
    ----------
    detected_spots : DataFrame
        Must have columns ``z, y, x`` (0-based).
    ground_truth : dict
        Loaded ``ground_truth.json``.
    fov_id : str
        FOV key in ground_truth.
    position_tolerance : float
        Max Euclidean distance for a valid match (pixels).

    Returns
    -------
    dict with ``recall``, ``precision``, ``mean_distance``,
    ``matched_pairs`` (list of (gt_idx, det_idx, distance)),
    ``n_gt``, ``n_detected``, ``n_matched``.
    """
    gt_spots = ground_truth["fovs"][fov_id]["spots"]

    if len(gt_spots) == 0 or len(detected_spots) == 0:
        return {
            "recall": 0.0,
            "precision": 0.0,
            "mean_distance": float("inf"),
            "matched_pairs": [],
            "n_gt": len(gt_spots),
            "n_detected": len(detected_spots),
            "n_matched": 0,
        }

    # Build coordinate arrays: GT is [z, y, x], DataFrame has z/y/x columns
    gt_coords = np.array(
        [s["position"] for s in gt_spots], dtype=float
    )  # (N_gt, 3)
    det_coords = detected_spots[["z", "y", "x"]].values.astype(float)  # (N_det, 3)

    # Pairwise Euclidean distances
    dists = cdist(gt_coords, det_coords)  # (N_gt, N_det)

    # Collect candidate pairs within tolerance, sorted by distance
    pairs = []
    for i in range(dists.shape[0]):
        for j in range(dists.shape[1]):
            if dists[i, j] < position_tolerance:
                pairs.append((dists[i, j], i, j))
    pairs.sort()

    # Greedy assignment with exclusion
    matched_pairs = []
    used_gt: set[int] = set()
    used_det: set[int] = set()

    for dist, gt_idx, det_idx in pairs:
        if gt_idx in used_gt or det_idx in used_det:
            continue
        matched_pairs.append((gt_idx, det_idx, float(dist)))
        used_gt.add(gt_idx)
        used_det.add(det_idx)

    n_matched = len(matched_pairs)
    recall = n_matched / len(gt_spots)
    precision = n_matched / len(detected_spots)
    mean_distance = (
        float(np.mean([d for _, _, d in matched_pairs]))
        if matched_pairs
        else float("inf")
    )

    return {
        "recall": recall,
        "precision": precision,
        "mean_distance": mean_distance,
        "matched_pairs": matched_pairs,
        "n_gt": len(gt_spots),
        "n_detected": len(detected_spots),
        "n_matched": n_matched,
    }


def compare_genes(
    spots: pd.DataFrame,
    ground_truth: dict,
    fov_id: str,
    position_tolerance: float = 5.0,
) -> dict:
    """Compare decoded gene labels and color sequences against ground truth.

    First matches ``spots`` to GT spots by position (same algorithm as
    ``compare_spots``). Then for matched pairs, compares gene labels
    (if ``gene`` column present) and color sequences (if ``color_seq``
    column present).

    Color sequence format: GT stores as string ``"4422"``, pipeline
    ``reads_extraction`` also produces a string ``"4422"`` — compared
    directly.

    Returns
    -------
    dict with ``gene_accuracy``, ``color_seq_accuracy``,
    ``gene_confusion``, ``n_matched``, ``spot_match``.
    """
    gt_spots = ground_truth["fovs"][fov_id]["spots"]

    # Match by position first
    spot_result = compare_spots(spots, ground_truth, fov_id, position_tolerance)
    matched_pairs = spot_result["matched_pairs"]

    if not matched_pairs:
        return {
            "gene_accuracy": 0.0,
            "color_seq_accuracy": 0.0,
            "gene_confusion": {},
            "n_matched": 0,
            "correct_genes": 0,
            "correct_color_seq": 0,
            "spot_match": spot_result,
        }

    has_gene = "gene" in spots.columns
    has_color_seq = "color_seq" in spots.columns
    correct_genes = 0
    correct_color_seq = 0
    gene_confusion: dict[str, int] = {}

    for gt_idx, det_idx, _dist in matched_pairs:
        # Gene comparison
        if has_gene:
            gt_gene = gt_spots[gt_idx]["gene"]
            pred_gene = spots.iloc[det_idx]["gene"]
            if gt_gene == pred_gene:
                correct_genes += 1
            else:
                key = f"{gt_gene}->{pred_gene}"
                gene_confusion[key] = gene_confusion.get(key, 0) + 1

        # Color sequence comparison
        if has_color_seq:
            gt_cs = gt_spots[gt_idx]["color_seq"]  # string "4422"
            pred_cs = str(spots.iloc[det_idx]["color_seq"])
            if gt_cs == pred_cs:
                correct_color_seq += 1

    n = len(matched_pairs)
    return {
        "gene_accuracy": correct_genes / n if has_gene else None,
        "color_seq_accuracy": correct_color_seq / n if has_color_seq else None,
        "gene_confusion": gene_confusion,
        "n_matched": n,
        "correct_genes": correct_genes,
        "correct_color_seq": correct_color_seq,
        "spot_match": spot_result,
    }


def e2e_summary(
    shift_result: dict, spot_result: dict, gene_result: dict
) -> dict:
    """Combine comparison results into a single summary dict."""
    return {
        "shift_max_error": shift_result["max_error"],
        "shift_passed": shift_result["passed"],
        "spot_recall": spot_result["recall"],
        "spot_precision": spot_result["precision"],
        "spot_mean_distance": spot_result["mean_distance"],
        "gene_accuracy": gene_result["gene_accuracy"],
        "color_seq_accuracy": gene_result["color_seq_accuracy"],
    }
