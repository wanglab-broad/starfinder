"""Registration quality metrics for sparse fluorescence images.

This module provides metrics specifically designed for evaluating registration
quality in spot-based fluorescence microscopy (e.g., STARmap, MERFISH).

For STARmap barcode decoding, spot matching accuracy is the most critical metric
because a spot must match across ALL sequencing rounds to decode its barcode.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max
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


def spot_colocalization(
    ref: np.ndarray,
    img: np.ndarray,
    threshold_percentile: float = 99.0,
) -> dict[str, float]:
    """Compute spot colocalization metrics (IoU and Dice).

    Measures how well bright spots overlap between two images. This is more
    relevant than MAE for sparse fluorescence images because it focuses on
    spot alignment rather than background pixels.

    Parameters
    ----------
    ref : np.ndarray
        Reference image.
    img : np.ndarray
        Image to compare (same shape as ref).
    threshold_percentile : float, optional
        Percentile threshold for defining "spots". Default is 99 (top 1%).

    Returns
    -------
    dict with keys:
        - iou: Intersection over Union (Jaccard index)
        - dice: Dice coefficient (F1 score)
        - n_ref_pixels: Number of spot pixels in reference
        - n_img_pixels: Number of spot pixels in image
    """
    ref_threshold = np.percentile(ref, threshold_percentile)
    img_threshold = np.percentile(img, threshold_percentile)

    ref_spots = ref > ref_threshold
    img_spots = img > img_threshold

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


def detect_spots(
    volume: np.ndarray,
    threshold_percentile: float = 99.5,
    threshold_mode: str = "percentile",
    noise_k: float = 5.0,
    min_distance: int = 1,
) -> np.ndarray:
    """Detect spot centroids in a 3D or 4D volume.

    Supports two workflows:

    - ``threshold_mode="percentile"`` (default): percentile-based threshold
      with connected-component centroids. Used by quality metrics.
    - ``threshold_mode="noise"``: per-channel MAD-based noise-floor threshold
      with ``peak_local_max`` suppression. Matches ``find_spots_3d`` workflow.

    For 4D input ``(Z, Y, X, C)``, noise mode detects spots per channel and
    deduplicates by spatial proximity. For 3D input, treats as single channel.

    Parameters
    ----------
    volume : np.ndarray
        Input volume ``(Z, Y, X)`` or ``(Z, Y, X, C)``.
    threshold_percentile : float, optional
        Percentile threshold. Only used when ``threshold_mode="percentile"``.
    threshold_mode : str, optional
        ``"percentile"``: top percentile of voxel intensities.
        ``"noise"``: MAD-based noise floor with ``peak_local_max``.
    noise_k : float, optional
        Number of sigma above noise floor (default 5.0).
        Only used when ``threshold_mode="noise"``.
    min_distance : int, optional
        Minimum pixel distance between detected peaks (default 1).
        Only used when ``threshold_mode="noise"``.

    Returns
    -------
    np.ndarray
        Array of spot centroids with shape ``(N, 3)`` — ``(z, y, x)``.
    """
    if threshold_mode == "noise":
        return _detect_spots_noise(volume, noise_k, min_distance)

    # Legacy percentile mode: CCA on the full volume (3D only)
    vol = volume if volume.ndim == 3 else volume.sum(axis=-1)
    threshold = np.percentile(vol, threshold_percentile)
    binary = vol > threshold
    labeled, n_spots = label(binary)

    if n_spots == 0:
        return np.array([]).reshape(0, 3)

    centroids = center_of_mass(vol, labeled, range(1, n_spots + 1))
    return np.array(centroids)


def _detect_spots_noise(
    volume: np.ndarray,
    noise_k: float,
    min_distance: int,
) -> np.ndarray:
    """Noise-floor spot detection: per-channel MAD threshold + peak_local_max.

    Mirrors the ``find_spots_3d`` workflow from ``starfinder.spotfinding``.
    For 3D input, treats as single channel. For 4D ``(Z, Y, X, C)``, detects
    per channel and deduplicates spatially.
    """
    # Normalize to 4D
    if volume.ndim == 3:
        channels = [volume]
    elif volume.ndim == 4:
        channels = [volume[:, :, :, c] for c in range(volume.shape[3])]
    else:
        raise ValueError(f"Expected 3D or 4D volume, got {volume.ndim}D")

    all_coords = []
    for ch in channels:
        if ch.max() == 0:
            continue

        ch_float = ch.astype(np.float64)
        med = np.median(ch_float)
        mad = np.median(np.abs(ch_float - med))
        threshold = med + noise_k * mad * 1.4826

        coords = peak_local_max(
            ch, min_distance=min_distance, threshold_abs=threshold,
        )
        if len(coords) > 0:
            all_coords.append(coords)

    if not all_coords:
        return np.array([]).reshape(0, 3)

    all_coords = np.concatenate(all_coords)

    # Deduplicate: spots from different channels at the same location
    if len(all_coords) > 1 and volume.ndim == 4:
        from scipy.spatial import cKDTree

        tree = cKDTree(all_coords)
        # Group points within min_distance of each other, keep one per group
        pairs = tree.query_pairs(r=max(min_distance, 1))
        remove = set()
        for i, j in pairs:
            remove.add(max(i, j))  # keep lower index
        if remove:
            keep = np.array(sorted(set(range(len(all_coords))) - remove))
            all_coords = all_coords[keep]

    return all_coords


def spot_matching_accuracy(
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


def registration_quality_report(
    ref: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    spot_threshold: float = 99.5,
    match_tolerance: float = 2.0,
) -> dict[str, dict[str, float]]:
    """Generate comprehensive registration quality report.

    Parameters
    ----------
    ref : np.ndarray
        Reference image/volume.
    before : np.ndarray
        Image before registration (same shape as ref).
    after : np.ndarray
        Image after registration (same shape as ref).
    spot_threshold : float, optional
        Percentile threshold for spot detection. Default is 99.5.
    match_tolerance : float, optional
        Maximum distance for spot matching. Default is 2.0 pixels.

    Returns
    -------
    dict
        Dictionary with metrics for "before" and "after", plus "improvement".
    """
    # Image-based metrics
    ncc_before = normalized_cross_correlation(ref, before)
    ncc_after = normalized_cross_correlation(ref, after)

    ssim_before = structural_similarity(ref, before)
    ssim_after = structural_similarity(ref, after)

    coloc_before = spot_colocalization(ref, before, spot_threshold)
    coloc_after = spot_colocalization(ref, after, spot_threshold)

    # Spot matching
    ref_spots = detect_spots(ref, spot_threshold)
    before_spots = detect_spots(before, spot_threshold)
    after_spots = detect_spots(after, spot_threshold)

    match_before = spot_matching_accuracy(ref_spots, before_spots, match_tolerance)
    match_after = spot_matching_accuracy(ref_spots, after_spots, match_tolerance)

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


def print_quality_report(report: dict[str, dict[str, float]]) -> None:
    """Print a formatted registration quality report.

    Parameters
    ----------
    report : dict
        Output from registration_quality_report().
    """
    print("=" * 60)
    print("REGISTRATION QUALITY REPORT")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'Before':>12} {'After':>12} {'Change':>12}")
    print("-" * 60)

    # NCC
    b, a = report["ncc"]["before"], report["ncc"]["after"]
    print(f"{'NCC (↑ better)':<25} {b:>12.4f} {a:>12.4f} {(a-b)/abs(b+1e-10)*100:>+11.1f}%")

    # SSIM
    b, a = report["ssim"]["before"], report["ssim"]["after"]
    print(f"{'SSIM (↑ better)':<25} {b:>12.4f} {a:>12.4f} {(a-b)/max(abs(b),0.001)*100:>+11.1f}%")

    # Spot IoU
    b, a = report["spot_iou"]["before"], report["spot_iou"]["after"]
    print(f"{'Spot IoU (↑ better)':<25} {b:>12.4f} {a:>12.4f} {(a-b)/max(b,0.001)*100:>+11.1f}%")

    # Spot Dice
    b, a = report["spot_dice"]["before"], report["spot_dice"]["after"]
    print(f"{'Spot Dice (↑ better)':<25} {b:>12.4f} {a:>12.4f} {(a-b)/max(b,0.001)*100:>+11.1f}%")

    # Match rate
    b, a = report["match_rate"]["before"], report["match_rate"]["after"]
    print(f"{'Match Rate (↑ better)':<25} {b*100:>11.1f}% {a*100:>11.1f}% {(a-b)/max(b,0.001)*100:>+11.1f}%")

    # Match distance
    b, a = report["match_distance"]["before"], report["match_distance"]["after"]
    b_str = f"{b:.2f}" if not np.isnan(b) else "N/A"
    a_str = f"{a:.2f}" if not np.isnan(a) else "N/A"
    print(f"{'Match Distance (↓ better)':<25} {b_str:>12} {a_str:>12}")

    print("-" * 60)

    # Barcode decoding projection
    b_rate = report["match_rate"]["before"]
    a_rate = report["match_rate"]["after"]
    print(f"\n📈 Projected barcode decoding (4 rounds):")
    print(f"   Before: {b_rate*100:.1f}%/round → {b_rate**4*100:.1f}% decoded")
    print(f"   After:  {a_rate*100:.1f}%/round → {a_rate**4*100:.1f}% decoded")

    print("=" * 60)
