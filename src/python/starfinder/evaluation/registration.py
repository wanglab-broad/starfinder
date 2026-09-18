"""Registration quality metrics for sparse fluorescence images.

This module provides metrics specifically designed for evaluating registration
quality in spot-based fluorescence microscopy (e.g., STARmap, MERFISH).

For STARmap barcode decoding, spot matching accuracy is the most critical metric
because a spot must match across ALL sequencing rounds to decode its barcode.
"""

from __future__ import annotations


import numpy as np
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity as _ssim


def structural_similarity(
    img1: np.ndarray,
    img2: np.ndarray,
    win_size: int | None = None,
    data_range: float | None = None,
) -> float:
    """Compute Structural Similarity Index (SSIM) between two images.

    SSIM is a perceptual metric that considers luminance, contrast, and
    structure. It's particularly good at detecting localized distortions
    that affect image quality.

    Parameters
    ----------
    img1 : np.ndarray
        First image (2D or 3D).
    img2 : np.ndarray
        Second image (same shape as img1).
    win_size : int | None, optional
        Size of the sliding window for local statistics. Must be odd.
        If None, uses min(7, smallest_dimension) and ensures it's odd.
    data_range : float | None, optional
        Data range of the images. If None, computed from img1.

    Returns
    -------
    float
        SSIM value in range [-1, 1]. 1 = identical, 0 = no similarity.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if data_range is None:
        data_range = img1.max() - img1.min()
        if data_range == 0:
            data_range = 1.0  # Avoid division by zero for constant images

    # Determine appropriate window size
    if win_size is None:
        min_dim = min(img1.shape)
        win_size = min(7, min_dim)
        # Ensure odd
        if win_size % 2 == 0:
            win_size = max(3, win_size - 1)

    return float(_ssim(img1, img2, win_size=win_size, data_range=data_range))


def normalized_cross_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute normalized cross-correlation between two images.

    NCC is intensity-invariant, making it robust to photobleaching and
    exposure differences between rounds.

    Parameters
    ----------
    img1 : np.ndarray
        First image (any shape).
    img2 : np.ndarray
        Second image (same shape as img1).

    Returns
    -------
    float
        NCC value in range [-1, 1]. 1 = perfect correlation, 0 = uncorrelated.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Normalize to zero mean, unit variance
    img1_norm = (img1 - img1.mean()) / (img1.std() + 1e-10)
    img2_norm = (img2 - img2.mean()) / (img2.std() + 1e-10)

    return float(np.mean(img1_norm * img2_norm))


def evaluate_mask_overlap(
    ref: np.ndarray,
    img: np.ndarray,
) -> dict[str, float]:
    """Compute IoU and Dice from supplied equal-shaped Boolean masks.

    No thresholding/detection is performed here. Returns iou, dice,
    n_ref_pixels and n_img_pixels. Existing empty-mask values remain zero;
    undefined-metric result redesign belongs to evaluation consolidation.
    """
    if ref.dtype != bool or img.dtype != bool or ref.shape != img.shape:
        raise ValueError("mask overlap requires equal-shaped Boolean masks")
    ref_spots, img_spots = ref, img

    intersection = np.logical_and(ref_spots, img_spots).sum()
    union = np.logical_or(ref_spots, img_spots).sum()

    iou = intersection / union if union > 0 else 0.0
    dice = (
        2 * intersection / (ref_spots.sum() + img_spots.sum())
        if (ref_spots.sum() + img_spots.sum()) > 0
        else 0.0
    )

    return {
        "iou": float(iou),
        "dice": float(dice),
        "n_ref_pixels": int(ref_spots.sum()),
        "n_img_pixels": int(img_spots.sum()),
    }


def evaluate_landmark_alignment(
    ref_spots: np.ndarray,
    mov_spots: np.ndarray,
    max_distance: float = 2.0,
) -> dict[str, float | int]:
    """Compute spot matching accuracy between two spot sets.

    This is the most critical metric for STARmap barcode decoding because
    a spot must be matched across ALL sequencing rounds to decode its barcode.

    The matching rate has exponential impact on decoding success:

    - 90% match/round × 4 rounds = 65% decoded

    - 99% match/round × 4 rounds = 96% decoded

    Parameters
    ----------
    ref_spots : np.ndarray
        Reference spot positions, shape (N, ndim).
    mov_spots : np.ndarray
        Moving/registered spot positions, shape (M, ndim).
    max_distance : float, optional
        Maximum distance (pixels) for a valid match. Default is 2.0.

    Returns
    -------
    dict with keys:

        - matched: Number of matched spots

        - match_rate: Fraction of reference spots that were matched

        - mean_distance: Mean distance of matched pairs

        - unmatched_ref: Number of unmatched reference spots

        - unmatched_mov: Number of unmatched moving spots

        - total_ref: Total reference spots

        - total_mov: Total moving spots
    """
    if len(ref_spots) == 0 or len(mov_spots) == 0:
        return {
            "matched": 0,
            "match_rate": 0.0,
            "mean_distance": float("nan"),
            "unmatched_ref": len(ref_spots),
            "unmatched_mov": len(mov_spots),
            "total_ref": len(ref_spots),
            "total_mov": len(mov_spots),
        }

    # Compute pairwise distances
    distances = cdist(ref_spots, mov_spots)

    # Greedy matching: assign closest pairs within threshold
    matched_pairs = []
    used_ref: set[int] = set()
    used_mov: set[int] = set()

    # Collect all valid pairs and sort by distance
    pairs = []
    for i in range(len(ref_spots)):
        for j in range(len(mov_spots)):
            if distances[i, j] <= max_distance:
                pairs.append((distances[i, j], i, j))
    pairs.sort()

    # Greedy assignment
    for dist, i, j in pairs:
        if i not in used_ref and j not in used_mov:
            matched_pairs.append((i, j, dist))
            used_ref.add(i)
            used_mov.add(j)

    n_matched = len(matched_pairs)
    match_rate = n_matched / len(ref_spots) if len(ref_spots) > 0 else 0.0
    mean_distance = (
        float(np.mean([d for _, _, d in matched_pairs])) if matched_pairs else float("nan")
    )

    return {
        "matched": n_matched,
        "match_rate": float(match_rate),
        "mean_distance": mean_distance,
        "unmatched_ref": len(ref_spots) - n_matched,
        "unmatched_mov": len(mov_spots) - n_matched,
        "total_ref": len(ref_spots),
        "total_mov": len(mov_spots),
    }


def evaluate_registration(
    ref: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    *,
    reference_spots: np.ndarray,
    before_spots: np.ndarray,
    after_spots: np.ndarray,
    reference_mask: np.ndarray,
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    match_tolerance: float = 2.0,
) -> dict[str, dict[str, float]]:
    """Compute before/after metrics using supplied detections and masks.

    Parameters
    ----------
    ref, before, after : np.ndarray
        Reference, moving and registered images on an equal grid.
    reference_spots, before_spots, after_spots : np.ndarray
        Supplied point arrays, with consistent coordinate axes and units.
    reference_mask, before_mask, after_mask : np.ndarray
        Supplied Boolean masks on the image grid; no detection is performed.
    match_tolerance : float
        Matching threshold in the point coordinate units (default 2 voxels).

    Returns
    -------
    dict
        Existing nested before/after NCC, SSIM, overlap and matching metrics.
        Numerical/undefined-value policies are unchanged by this relocation.
    """
    # Image-based metrics
    ncc_before = normalized_cross_correlation(ref, before)
    ncc_after = normalized_cross_correlation(ref, after)

    ssim_before = structural_similarity(ref, before)
    ssim_after = structural_similarity(ref, after)

    coloc_before = evaluate_mask_overlap(reference_mask, before_mask)
    coloc_after = evaluate_mask_overlap(reference_mask, after_mask)

    ref_spots = reference_spots
    match_before = evaluate_landmark_alignment(ref_spots, before_spots, match_tolerance)
    match_after = evaluate_landmark_alignment(ref_spots, after_spots, match_tolerance)

    return {
        "ncc": {"before": ncc_before, "after": ncc_after},
        "ssim": {"before": ssim_before, "after": ssim_after},
        "spot_iou": {"before": coloc_before["iou"], "after": coloc_after["iou"]},
        "spot_dice": {"before": coloc_before["dice"], "after": coloc_after["dice"]},
        "match_rate": {"before": match_before["match_rate"], "after": match_after["match_rate"]},
        "match_distance": {
            "before": match_before["mean_distance"],
            "after": match_after["mean_distance"],
        },
        "n_spots": {
            "ref": len(ref_spots),
            "before": len(before_spots),
            "after": len(after_spots),
        },
    }


__all__ = ["structural_similarity", "normalized_cross_correlation", "evaluate_mask_overlap", "evaluate_landmark_alignment", "evaluate_registration"]
