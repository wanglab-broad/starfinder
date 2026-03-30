"""Spot-based local registration: TPS and Coherent Point Drift (CPD).

TPS: matched spot correspondences → RBF interpolation.
CPD: simultaneous correspondence + transformation via EM on GMM.

No SimpleITK dependency — pure numpy/scipy.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import map_coordinates, zoom
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def sanitize_displacement_field(
    field: np.ndarray,
    *,
    clamp: bool = True,
    margin: int = 0,
    smooth_sigma: float | None = None,
    detect_folds: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Post-process a displacement field to reduce interpolation artifacts.

    Parameters
    ----------
    field : np.ndarray
        Displacement field with shape (Z, Y, X, 3), (dz, dy, dx).
    clamp : bool
        If True, clip ``position + displacement`` to ``[margin, dim-1-margin]``
        so that source coordinates never go out of bounds. Eliminates black
        bands from ``map_coordinates(cval=0)`` or SimpleITK zero-fill.
    margin : int
        Safety margin for clamping (pixels from each edge). Default 0.
    smooth_sigma : float or None
        If set, apply Gaussian smoothing to each displacement component.
        Removes cubic-zoom ringing and softens sharp gradients that cause
        field folding.
    detect_folds : bool
        If True, also return a boolean mask where det(J) < 0 (non-injective
        mapping). Diagnostic only — the caller decides what to do.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        Sanitized field (same shape). If ``detect_folds=True``, returns
        ``(field, fold_mask)`` where ``fold_mask`` has shape (Z, Y, X).
    """
    from scipy.ndimage import gaussian_filter

    field = field.copy()
    shape = field.shape[:3]  # (Z, Y, X)

    # Gaussian smoothing (before clamping so clamp sees smoothed values)
    if smooth_sigma is not None and smooth_sigma > 0:
        for d in range(3):
            field[..., d] = gaussian_filter(field[..., d], sigma=smooth_sigma)

    # Coordinate clamping
    if clamp:
        for d in range(3):
            # Build coordinate grid for this axis
            ax_len = shape[d]
            lo = float(margin)
            hi = float(ax_len - 1 - margin)

            # Create position array matching field shape
            slices = [None, None, None]
            slices[d] = slice(None)
            pos = np.arange(ax_len, dtype=np.float32)
            # Reshape for broadcasting: e.g. for d=1 → (1, Y, 1)
            bcast_shape = [1, 1, 1]
            bcast_shape[d] = ax_len
            pos = pos.reshape(bcast_shape)

            src = pos + field[..., d]
            field[..., d] = np.clip(src, lo, hi) - pos

    # Fold detection via Jacobian determinant
    if detect_folds:
        fold_mask = _detect_folds(field)
        return field, fold_mask

    return field


def _detect_folds(field: np.ndarray) -> np.ndarray:
    """Compute fold mask where det(Jacobian of deformation) < 0.

    The deformation is phi(x) = x + d(x). The Jacobian of phi is
    J = I + grad(d), so det(J) < 0 means the mapping folds (non-injective).
    Uses finite differences on the displacement field.

    Parameters
    ----------
    field : np.ndarray
        Displacement field, shape (Z, Y, X, 3).

    Returns
    -------
    np.ndarray
        Boolean mask, shape (Z, Y, X), True where det(J) < 0.
    """
    # Gradient of each displacement component w.r.t. each spatial axis
    # grad[i][j] = d(field[..., i]) / d(axis_j)
    grad = np.empty((3, 3) + field.shape[:3], dtype=np.float32)
    for i in range(3):
        for j in range(3):
            grad[i, j] = np.gradient(field[..., i], axis=j)

    # Jacobian of deformation = I + grad(displacement)
    for i in range(3):
        grad[i, i] += 1.0

    # det(J) via explicit 3x3 formula
    det = (
        grad[0, 0] * (grad[1, 1] * grad[2, 2] - grad[1, 2] * grad[2, 1])
        - grad[0, 1] * (grad[1, 0] * grad[2, 2] - grad[1, 2] * grad[2, 0])
        + grad[0, 2] * (grad[1, 0] * grad[2, 1] - grad[1, 1] * grad[2, 0])
    )

    return det < 0


def detect_and_match_spots(
    fixed: np.ndarray,
    moving: np.ndarray,
    detection_threshold: float = 3.0,
    match_distance: float = 10.0,
    min_matches: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect spots in both volumes and match by nearest-neighbor.

    Parameters
    ----------
    fixed : np.ndarray
        Fixed (reference) volume, shape (Z, Y, X).
    moving : np.ndarray
        Moving volume, shape (Z, Y, X).
    detection_threshold : float
        ``k`` for noise-floor detection: ``median + k * MAD * 1.4826``.
        Lower values detect more spots (more control points).
    match_distance : float
        Maximum Euclidean distance (pixels) for a valid match.
    min_matches : int
        Minimum number of matched pairs required.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(positions, displacements)`` where:
        - positions: (N, 3) matched spot positions in **fixed** image coords
        - displacements: (N, 3) displacement vectors (dz, dy, dx) at each position

    Raises
    ------
    ValueError
        If fewer than ``min_matches`` pairs are found.
    """
    from starfinder.registration.metrics import detect_spots

    fixed_spots = detect_spots(
        fixed, threshold_mode="noise", noise_k=detection_threshold,
    )
    moving_spots = detect_spots(
        moving, threshold_mode="noise", noise_k=detection_threshold,
    )

    if len(fixed_spots) == 0 or len(moving_spots) == 0:
        raise ValueError(
            f"Spot detection found {len(fixed_spots)} fixed, "
            f"{len(moving_spots)} moving spots — need at least {min_matches}."
        )

    # Greedy nearest-neighbor matching via KDTree
    tree = cKDTree(moving_spots)
    distances, indices = tree.query(fixed_spots, k=1)

    # Filter by match distance
    valid = distances <= match_distance
    matched_fixed = fixed_spots[valid]
    matched_moving = moving_spots[indices[valid]]

    # Remove duplicate moving matches (keep closest)
    used_moving: set[int] = set()
    keep = []
    # Sort by distance so we keep closest matches
    order = np.argsort(distances[valid])
    matched_indices = indices[valid]
    for i in order:
        mov_idx = matched_indices[i]
        if mov_idx not in used_moving:
            used_moving.add(mov_idx)
            keep.append(i)
    keep = np.array(keep) if keep else np.array([], dtype=int)

    if len(keep) < min_matches:
        raise ValueError(
            f"Only {len(keep)} spot matches found (need {min_matches}). "
            f"Detected {len(fixed_spots)} fixed, {len(moving_spots)} moving spots. "
            f"Try lowering detection_threshold or increasing match_distance."
        )

    positions = matched_fixed[keep]
    # Backward-mapping convention: displacement points from fixed-space to
    # moving-space so map_coordinates(moving, pos + disp) recovers fixed.
    displacements = matched_moving[keep] - matched_fixed[keep]

    return positions, displacements


def subsample_control_points(
    positions: np.ndarray,
    displacements: np.ndarray,
    max_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample to max_points using farthest-point sampling.

    Greedy farthest-point sampling ensures uniform spatial coverage,
    unlike random sampling which can leave gaps.

    Parameters
    ----------
    positions : np.ndarray
        (N, 3) point positions.
    displacements : np.ndarray
        (N, 3) displacement vectors at each position.
    max_points : int
        Maximum number of control points to keep.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Subsampled ``(positions, displacements)``.
    """
    n = len(positions)
    if n <= max_points:
        return positions, displacements

    # Farthest-point sampling: O(N * max_points)
    selected = np.zeros(max_points, dtype=np.intp)
    # Start from the point closest to the centroid (most central)
    centroid = positions.mean(axis=0)
    selected[0] = np.argmin(np.sum((positions - centroid) ** 2, axis=1))

    # Track minimum distance from each point to the selected set
    min_dist = np.full(n, np.inf)
    for i in range(1, max_points):
        # Update min distances with the last selected point
        d = np.sum((positions - positions[selected[i - 1]]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, d)
        # Select the farthest point from the current set
        selected[i] = np.argmax(min_dist)

    return positions[selected], displacements[selected]


def tps_displacement_field(
    positions: np.ndarray,
    displacements: np.ndarray,
    shape: tuple[int, int, int],
    smoothing: float = 1.0,
    grid_spacing: int = 32,
    zoom_order: int = 3,
    field_smooth_sigma: float | None = None,
) -> np.ndarray:
    """Interpolate sparse displacements into a dense field via TPS.

    Fits a multi-output RBFInterpolator (thin-plate-spline kernel) to the
    control points, evaluates on a coarse grid, and zooms to full resolution.

    Parameters
    ----------
    positions : np.ndarray
        (N, 3) control point positions (z, y, x).
    displacements : np.ndarray
        (N, 3) displacement vectors (dz, dy, dx) at each position.
    shape : tuple[int, int, int]
        Output field shape (Z, Y, X).
    smoothing : float
        RBF smoothing parameter. Higher = smoother field.
        0 = exact interpolation (may overfit noisy matches).
    grid_spacing : int
        Stride for the coarse evaluation grid in YX dimensions.
    zoom_order : int
        Spline order for coarse→full zoom. Default 3 (cubic).
        Use 1 (linear) to eliminate ringing artifacts.
    field_smooth_sigma : float or None
        If set, Gaussian-smooth the displacement field after zoom.
        Removes ringing and softens sharp gradients.

    Returns
    -------
    np.ndarray
        Displacement field with shape (Z, Y, X, 3), dtype float32.
    """
    # Fit multi-output TPS: positions → displacements (all 3 axes at once)
    rbf = RBFInterpolator(
        positions,
        displacements,
        kernel="thin_plate_spline",
        smoothing=smoothing,
    )

    # Build coarse evaluation grid
    z_stride = max(1, shape[0] // max(15, shape[0]))  # ≥15 Z points
    y_stride = min(grid_spacing, max(1, shape[1] // 4))
    x_stride = min(grid_spacing, max(1, shape[2] // 4))

    zz = np.arange(0, shape[0], z_stride)
    yy = np.arange(0, shape[1], y_stride)
    xx = np.arange(0, shape[2], x_stride)

    # Meshgrid for coarse evaluation points
    gz, gy, gx = np.meshgrid(zz, yy, xx, indexing="ij")
    coarse_points = np.column_stack([gz.ravel(), gy.ravel(), gx.ravel()])

    # Evaluate TPS at coarse grid
    coarse_displacements = rbf(coarse_points)  # (M, 3)
    coarse_shape = (len(zz), len(yy), len(xx))
    coarse_field = coarse_displacements.reshape(*coarse_shape, 3).astype(np.float32)

    # Zoom each component to full resolution
    zoom_factors = (
        shape[0] / coarse_shape[0],
        shape[1] / coarse_shape[1],
        shape[2] / coarse_shape[2],
    )
    field = np.empty((*shape, 3), dtype=np.float32)
    for d in range(3):
        field[..., d] = zoom(coarse_field[..., d], zoom_factors, order=zoom_order)

    # Sanitize: smooth + clamp to prevent OOB and reduce ringing
    field = sanitize_displacement_field(
        field, clamp=True, smooth_sigma=field_smooth_sigma,
    )

    return field


def apply_tps_deformation(
    volume: np.ndarray,
    displacement_field: np.ndarray,
    boundary_mode: str = "constant",
) -> np.ndarray:
    """Warp a volume using a displacement field, slice-by-slice.

    Processes one Z-slice at a time to limit memory usage to ~113 MB
    (for 3072x3072) instead of ~3.4 GB for the full coordinate grid.

    Parameters
    ----------
    volume : np.ndarray
        Input volume, shape (Z, Y, X).
    displacement_field : np.ndarray
        Displacement field, shape (Z, Y, X, 3) with (dz, dy, dx).
    boundary_mode : str
        How to handle out-of-bounds source coordinates:
        - ``"constant"`` (default): Fill with 0 (black bands at edges).
        - ``"nearest"``: Extend edge pixels (no black bands).

    Returns
    -------
    np.ndarray
        Warped volume, same shape and dtype as input.
    """
    nz, ny, nx = volume.shape
    result = np.empty_like(volume)

    # Pre-compute YX coordinate grids (reused across slices)
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)

    for z in range(nz):
        dz = displacement_field[z, :, :, 0]
        dy = displacement_field[z, :, :, 1]
        dx = displacement_field[z, :, :, 2]

        # Source coordinates = current position + displacement
        src_z = np.full_like(yy, z, dtype=np.float32) + dz
        src_y = yy + dy
        src_x = xx + dx

        coords = np.array([src_z, src_y, src_x])
        result[z] = map_coordinates(
            volume, coords, order=1, mode=boundary_mode, cval=0
        )

    return result


def tps_register(
    fixed: np.ndarray,
    moving: np.ndarray,
    detection_threshold: float = 3.0,
    match_distance: float = 10.0,
    min_matches: int = 50,
    max_control_points: int = 1000,
    smoothing: float = 1.0,
    grid_spacing: int = 32,
    zoom_order: int = 3,
    field_smooth_sigma: float | None = None,
) -> np.ndarray:
    """Compute TPS displacement field between fixed and moving volumes.

    End-to-end: detect spots → match → subsample → fit TPS → dense field.

    Parameters
    ----------
    fixed : np.ndarray
        Fixed (reference) volume, shape (Z, Y, X).
    moving : np.ndarray
        Moving volume, shape (Z, Y, X).
    detection_threshold : float
        Spot detection sensitivity (lower = more spots).
    match_distance : float
        Maximum pixel distance for spot matching.
    min_matches : int
        Minimum matched pairs required (raises ValueError if not met).
    max_control_points : int
        Maximum control points for TPS fitting.
    smoothing : float
        TPS smoothing parameter.
    grid_spacing : int
        Coarse grid stride for TPS evaluation.
    zoom_order : int
        Spline order for coarse→full zoom (default 3). Use 1 for no ringing.
    field_smooth_sigma : float or None
        Gaussian smoothing sigma for the displacement field after zoom.

    Returns
    -------
    np.ndarray
        Displacement field, shape (Z, Y, X, 3) with (dz, dy, dx), float32.

    Raises
    ------
    ValueError
        If insufficient spot matches for reliable TPS fitting.
    """
    positions, displacements = detect_and_match_spots(
        fixed, moving,
        detection_threshold=detection_threshold,
        match_distance=match_distance,
        min_matches=min_matches,
    )

    positions, displacements = subsample_control_points(
        positions, displacements, max_points=max_control_points
    )

    return tps_displacement_field(
        positions, displacements, fixed.shape,
        smoothing=smoothing, grid_spacing=grid_spacing,
        zoom_order=zoom_order, field_smooth_sigma=field_smooth_sigma,
    )


def register_volume_tps(
    images: np.ndarray,
    ref_image: np.ndarray,
    mov_image: np.ndarray,
    boundary_mode: str = "constant",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Register multi-channel volume using TPS.

    Mirrors the signature of ``register_volume_local()`` from demons.py.

    Parameters
    ----------
    images : np.ndarray
        Multi-channel volume, shape (Z, Y, X, C).
    ref_image : np.ndarray
        Reference image, shape (Z, Y, X).
    mov_image : np.ndarray
        Moving image, shape (Z, Y, X).
    boundary_mode : str
        How to handle out-of-bounds source coordinates:
        ``"constant"`` (default) fills with 0; ``"nearest"`` extends edges.
    **kwargs
        Passed to ``tps_register()``: detection_threshold, match_distance,
        min_matches, max_control_points, smoothing, grid_spacing,
        zoom_order, field_smooth_sigma.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(registered_images, displacement_field)`` where:
        - registered_images: shape (Z, Y, X, C), same dtype as input
        - displacement_field: shape (Z, Y, X, 3), float32
    """
    displacement_field = tps_register(ref_image, mov_image, **kwargs)

    n_channels = images.shape[-1]
    registered = np.empty_like(images)
    for c in range(n_channels):
        registered[:, :, :, c] = apply_tps_deformation(
            images[:, :, :, c], displacement_field,
            boundary_mode=boundary_mode,
        )

    return registered, displacement_field


# ─── Coherent Point Drift (CPD) ─────────────────────────────────────────


def _gaussian_kernel(Y: np.ndarray, beta: float) -> np.ndarray:
    """Compute N×N Gaussian kernel matrix.

    G(i,j) = exp(-||y_i - y_j||² / (2β²))
    """
    dist_sq = cdist(Y, Y, "sqeuclidean")
    return np.exp(-dist_sq / (2 * beta**2))


def _subsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    """Farthest-point sampling on a single point cloud.

    Greedy FPS ensures uniform spatial coverage. O(N * max_points).
    """
    n = len(points)
    if n <= max_points:
        return points

    selected = np.zeros(max_points, dtype=np.intp)
    centroid = points.mean(axis=0)
    selected[0] = np.argmin(np.sum((points - centroid) ** 2, axis=1))

    min_dist = np.full(n, np.inf)
    for i in range(1, max_points):
        d = np.sum((points - points[selected[i - 1]]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, d)
        selected[i] = np.argmax(min_dist)

    return points[selected]


def _subsample_with_neighbors(
    points: np.ndarray,
    max_anchors: int = 250,
    k_neighbors: int = 3,
) -> np.ndarray:
    """FPS for spatial anchors, then expand with K nearest neighbors.

    Preserves local cluster structure so CPD can disambiguate
    nearby spots during soft assignment.

    Parameters
    ----------
    points : np.ndarray
        (N, D) point cloud.
    max_anchors : int
        Number of FPS anchor points.
    k_neighbors : int
        Number of nearest neighbors to include per anchor.

    Returns
    -------
    np.ndarray
        (M, D) enriched point cloud where M <= max_anchors * (1 + k_neighbors).
    """
    anchors = _subsample_points(points, max_anchors)
    if len(points) <= max_anchors:
        return anchors  # all points already included

    tree = cKDTree(points)
    k = min(k_neighbors + 1, len(points))  # +1 because nearest is self
    _, indices = tree.query(anchors, k=k)
    all_idx = set(indices.ravel())
    return points[sorted(all_idx)]


def _gather_candidates(
    fixed_sub: np.ndarray,
    moving_all: np.ndarray,
    radius: float = 15.0,
) -> np.ndarray:
    """Gather moving points within radius of subsampled fixed points.

    Ensures every fixed control point has candidate correspondences
    for CPD's EM, without hard pre-matching.

    Parameters
    ----------
    fixed_sub : np.ndarray
        (M, D) subsampled fixed points.
    moving_all : np.ndarray
        (N, D) full moving point cloud.
    radius : float
        Search radius in pixels.

    Returns
    -------
    np.ndarray
        (K, D) moving candidates, K <= N.
    """
    tree = cKDTree(moving_all)
    nearby = tree.query_ball_point(fixed_sub, r=radius)
    idx: set[int] = set()
    for group in nearby:
        idx.update(group)
    return moving_all[sorted(idx)] if idx else moving_all[:0]


def _cpd_e_step(
    X: np.ndarray,
    T: np.ndarray,
    sigma2: float,
    w: float,
    M: int,
    N: int,
    D: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """E-step: compute soft assignment matrix P.

    P(m,n) = probability that fixed point m was generated by moving centroid n.
    Includes uniform outlier component weighted by ``w``.
    """
    dist_sq = cdist(X, T, "sqeuclidean")  # (M, N)
    P = np.exp(-dist_sq / (2 * sigma2))
    c = (2 * np.pi * sigma2) ** (D / 2) * w / (1 - w) * M / N
    denom = P.sum(axis=1, keepdims=True) + c
    P /= denom

    P1 = P.sum(axis=0)  # (N,) responsibility per centroid
    Pt1 = P.sum(axis=1)  # (M,) responsibility per data point
    Np = P1.sum()  # expected number of matches
    return P, P1, Pt1, Np


def _cpd_sigma2(
    X: np.ndarray,
    T: np.ndarray,
    P: np.ndarray,
    P1: np.ndarray,
    Pt1: np.ndarray,
    Np: float,
    D: int,
) -> float:
    """Update variance σ² from current assignments and transformed positions.

    σ² = (1/(Np·D)) · Σ_{m,n} P(m,n) ||x_m - t_n||²
    Computed via trace decomposition to avoid (M, N, D) intermediate.
    """
    PX = P.T @ X  # (N, D)
    val = (
        np.dot(Pt1, np.sum(X**2, axis=1))
        - 2 * np.sum(PX * T)
        + np.dot(P1, np.sum(T**2, axis=1))
    ) / (Np * D)
    return max(val, 1e-10)


def cpd_affine(
    X: np.ndarray,
    Y: np.ndarray,
    w: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Affine CPD: find B, t such that T(Y) = Y @ B.T + t.

    Internally normalizes coordinates for numerical stability (anisotropic
    3D data with Z << YX would otherwise make the M-step ill-conditioned).

    Parameters
    ----------
    X : np.ndarray
        (M, D) fixed points.
    Y : np.ndarray
        (N, D) moving points (GMM centroids).
    w : float
        Outlier weight in [0, 1). 0 = no outliers.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Relative convergence tolerance on σ².

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(transformed_Y, B, t)`` where T(Y) = Y @ B.T + t
        (in original coordinate space).
    """
    # Per-axis normalization for numerical stability.
    # Global std fails when Z range << YX range (e.g., 32 vs 512),
    # leaving Z ~17x smaller than YX after normalization.
    all_pts = np.concatenate([X, Y])
    center = all_pts.mean(axis=0)
    scale = all_pts.std(axis=0)
    scale = np.where(scale < 1e-10, 1.0, scale)  # avoid div-by-zero
    Xn = (X - center) / scale
    Yn = (Y - center) / scale

    M, D = Xn.shape
    N = Yn.shape[0]

    Tn = Yn.copy()
    sigma2 = np.sum(cdist(Xn, Yn, "sqeuclidean")) / (D * M * N)

    Bn = np.eye(D)
    tn = np.zeros(D)

    for _ in range(max_iter):
        # E-step
        P, P1, Pt1, Np = _cpd_e_step(Xn, Tn, sigma2, w, M, N, D)

        # M-step: weighted means
        mu_x = Xn.T @ Pt1 / Np  # (D,)
        mu_y = Yn.T @ P1 / Np  # (D,)

        X_hat = Xn - mu_x
        Y_hat = Yn - mu_y

        # B = (X_hat^T P Y_hat) @ inv(Y_hat^T diag(P1) Y_hat)
        A = X_hat.T @ P @ Y_hat  # (D, D)
        YPY = Y_hat.T @ (P1[:, None] * Y_hat)  # (D, D)
        Bn = np.linalg.solve(YPY.T, A.T).T

        tn = mu_x - Bn @ mu_y
        Tn = Yn @ Bn.T + tn

        # Update σ²
        sigma2_new = _cpd_sigma2(Xn, Tn, P, P1, Pt1, Np, D)

        if abs(sigma2_new - sigma2) / sigma2 < tol:
            break
        sigma2 = sigma2_new

    # Denormalize with per-axis scale: T(y) = y @ B.T + t in original space.
    # Normalized: y_n = (y - c) @ diag(1/s), T_n = y_n @ Bn.T + tn.
    # Denorm: T = T_n @ diag(s) + c = (y-c) @ diag(1/s) @ Bn.T @ diag(s) + tn@diag(s) + c
    # So: B = diag(s) @ Bn @ diag(1/s), t = c @ (I - B.T) + tn * s
    B = (scale[:, None] / scale[None, :]) * Bn
    t = center @ (np.eye(D) - B.T) + tn * scale
    T = Y @ B.T + t

    return T, B, t


def cpd_nonrigid(
    X: np.ndarray,
    Y: np.ndarray,
    beta: float = 3.0,
    lmbda: float = 2.0,
    w: float = 0.1,
    max_iter: int = 150,
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Non-rigid CPD: find W such that T(Y) = Y + G @ W.

    Parameters
    ----------
    X : np.ndarray
        (M, D) fixed points.
    Y : np.ndarray
        (N, D) moving points (GMM centroids).
    beta : float
        Gaussian kernel width (pixels). Controls deformation smoothness.
    lmbda : float
        Regularization weight. Higher = smoother, less flexible.
    w : float
        Outlier weight in [0, 1). 0 = no outliers.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Relative convergence tolerance on σ².

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(transformed_Y, W, G)`` where T(Y) = Y + G @ W.
    """
    M, D = X.shape
    N = Y.shape[0]

    G = _gaussian_kernel(Y, max(beta, 1.0))
    T = Y.copy()
    sigma2 = np.sum(cdist(X, Y, "sqeuclidean")) / (D * M * N)

    for _ in range(max_iter):
        # E-step
        P, P1, Pt1, Np = _cpd_e_step(X, T, sigma2, w, M, N, D)
        PX = P.T @ X  # (N, D)

        # M-step: (G + λσ² diag(1/P1)) W = diag(1/P1) PX - Y
        P1_inv = 1.0 / (P1 + 1e-10)
        A = G + lmbda * sigma2 * np.diag(P1_inv)
        B = P1_inv[:, None] * PX - Y
        W = np.linalg.solve(A, B)  # (N, D)

        # Update transform
        T = Y + G @ W

        # Update σ²
        sigma2_new = _cpd_sigma2(X, T, P, P1, Pt1, Np, D)

        if abs(sigma2_new - sigma2) / sigma2 < tol:
            break
        sigma2 = sigma2_new

    return T, W, G


def cpd_displacement_field(
    Y: np.ndarray,
    W: np.ndarray,
    beta: float,
    shape: tuple[int, int, int],
    grid_spacing: int = 16,
    B: np.ndarray | None = None,
    t: np.ndarray | None = None,
    zoom_order: int = 3,
    field_smooth_sigma: float | None = None,
) -> np.ndarray:
    """Generate dense backward displacement field from CPD results.

    For each position p in fixed space, computes d(p) such that
    ``moving[p + d(p)] ≈ fixed[p]``.

    Non-rigid backward: ``p_nr = p - K(p, Y) @ W``
    Affine backward (if B, t given): ``p_orig = (p_nr - t) @ inv(B).T``
    Total displacement: ``d = p_orig - p``

    Parameters
    ----------
    Y : np.ndarray
        (N, D) kernel centers (affine-corrected moving point positions).
    W : np.ndarray
        (N, D) CPD non-rigid weight matrix.
    beta : float
        Gaussian kernel width.
    shape : tuple[int, int, int]
        Output field shape (Z, Y, X).
    grid_spacing : int
        Coarse grid stride in YX dimensions.
    B : np.ndarray, optional
        (D, D) affine matrix from cpd_affine.
    t : np.ndarray, optional
        (D,) affine translation from cpd_affine.
    zoom_order : int
        Spline order for coarse→full zoom. Default 3 (cubic).
        Use 1 (linear) to eliminate ringing artifacts.
    field_smooth_sigma : float or None
        If set, Gaussian-smooth the displacement field after zoom.

    Returns
    -------
    np.ndarray
        Displacement field with shape (Z, Y, X, 3), dtype float32.
    """
    N, D = W.shape
    beta = max(beta, 1.0)

    # Precompute affine inverse
    has_affine = B is not None and t is not None
    if has_affine:
        B_inv_T = np.linalg.inv(B).T

    # Coarse grid
    y_stride = min(grid_spacing, max(1, shape[1] // 4))
    x_stride = min(grid_spacing, max(1, shape[2] // 4))

    zz = np.arange(shape[0])  # every Z slice (Z is typically small)
    yy = np.arange(0, shape[1], y_stride)
    xx = np.arange(0, shape[2], x_stride)

    coarse_shape = (len(zz), len(yy), len(xx))
    coarse_field = np.zeros((*coarse_shape, D), dtype=np.float32)

    # Evaluate Z-slice by Z-slice to limit memory
    two_beta_sq = 2 * beta**2
    for iz, z in enumerate(zz):
        gy, gx = np.meshgrid(yy, xx, indexing="ij")
        gz = np.full_like(gy, z, dtype=np.float64)
        points = np.column_stack(
            [gz.ravel(), gy.ravel(), gx.ravel()]
        )  # (n_pts, 3)

        # Non-rigid backward: p_nr = p - K @ W
        K = cdist(points, Y, "sqeuclidean")
        np.exp(-K / two_beta_sq, out=K)
        p_nr = points - K @ W

        # Affine backward
        if has_affine:
            p_orig = (p_nr - t) @ B_inv_T
        else:
            p_orig = p_nr

        disp = (p_orig - points).reshape(len(yy), len(xx), D)
        coarse_field[iz] = disp.astype(np.float32)

    # Zoom each component to full resolution
    zoom_factors = (
        shape[0] / coarse_shape[0],
        shape[1] / coarse_shape[1],
        shape[2] / coarse_shape[2],
    )
    field = np.empty((*shape, D), dtype=np.float32)
    for d in range(D):
        field[..., d] = zoom(coarse_field[..., d], zoom_factors, order=zoom_order)

    # Sanitize: smooth + clamp to prevent OOB and reduce ringing
    field = sanitize_displacement_field(
        field, clamp=True, smooth_sigma=field_smooth_sigma,
    )

    return field


def cpd_register(
    fixed: np.ndarray,
    moving: np.ndarray,
    detection_threshold: float = 5.0,
    max_control_points: int = 1000,
    beta: float | None = None,
    lmbda: float = 2.0,
    w: float = 0.15,
    affine_first: bool = True,
    grid_spacing: int = 16,
    candidate_radius: float = 15.0,
    k_neighbors: int = 3,
    zoom_order: int = 3,
    field_smooth_sigma: float | None = None,
) -> np.ndarray:
    """End-to-end CPD registration.

    1. Detect spots in both volumes
    2. Subsample fixed with FPS anchors + K nearest neighbors
    3. Gather moving candidates within radius of fixed points
    4. Trim fixed points with no nearby moving candidate
    5. Optional affine CPD for global alignment
    6. Non-rigid CPD for local deformation
    7. Generate dense backward displacement field

    Parameters
    ----------
    fixed : np.ndarray
        Fixed (reference) volume, shape (Z, Y, X).
    moving : np.ndarray
        Moving volume, shape (Z, Y, X).
    detection_threshold : float
        Spot detection sensitivity (lower = more spots).
    max_control_points : int
        Target total fixed point count. FPS anchors are computed as
        ``max_control_points // (1 + k_neighbors)`` so that after
        neighbor expansion the total stays near this value.
    beta : float or None
        Gaussian kernel width (pixels). Controls deformation smoothness.
        If None (default), auto-computed as 5× median nearest-neighbor
        distance in the moving point cloud.
    lmbda : float
        Regularization weight. Higher = smoother.
    w : float
        Expected outlier fraction in [0, 1). Default 0.15 balances
        tolerance to boundary dropout with registration accuracy.
    affine_first : bool
        If True, run affine CPD before non-rigid.
    grid_spacing : int
        Coarse grid stride for field evaluation.
    candidate_radius : float
        Radius (pixels) for gathering moving candidates around each
        fixed point. Ensures true correspondences survive subsampling.
    k_neighbors : int
        Number of nearest neighbors per FPS anchor in the fixed cloud.
        Preserves local cluster structure for CPD soft assignment.
    zoom_order : int
        Spline order for coarse→full zoom (default 3). Use 1 for no ringing.
    field_smooth_sigma : float or None
        Gaussian smoothing sigma for the displacement field after zoom.

    Returns
    -------
    np.ndarray
        Displacement field, shape (Z, Y, X, 3) with (dz, dy, dx), float32.

    Raises
    ------
    ValueError
        If fewer than 10 spots detected in either volume, or too few
        moving candidates within the search radius.
    """
    from starfinder.registration.metrics import detect_spots

    fixed_spots = detect_spots(
        fixed, threshold_mode="noise", noise_k=detection_threshold,
    )
    moving_spots = detect_spots(
        moving, threshold_mode="noise", noise_k=detection_threshold,
    )

    if len(fixed_spots) < 10 or len(moving_spots) < 10:
        raise ValueError(
            f"Too few spots for CPD: {len(fixed_spots)} fixed, "
            f"{len(moving_spots)} moving (need >= 10 each)."
        )

    # Correspondence-aware subsampling:
    # FPS anchors + K neighbors for fixed (preserves local clusters),
    # radius-based candidate gathering for moving (preserves true matches).
    max_anchors = max(1, max_control_points // (1 + k_neighbors))
    X = _subsample_with_neighbors(
        fixed_spots, max_anchors=max_anchors, k_neighbors=k_neighbors,
    )
    Y = _gather_candidates(X, moving_spots, radius=candidate_radius)

    # Trim fixed points with no moving candidate within radius.
    # Orphaned fixed points (from boundary dropout under large deformations)
    # poison the affine EM with phantom correspondences.
    if len(Y) > 0:
        tree_y = cKDTree(Y)
        dists, _ = tree_y.query(X, k=1)
        matched = dists <= candidate_radius
        X = X[matched]

    if len(Y) < 10:
        raise ValueError(
            f"Too few moving candidates within {candidate_radius}px "
            f"radius: {len(Y)} (need >= 10)."
        )

    # Stage 1: Affine alignment
    B_affine, t_affine = None, None
    if affine_first:
        Y, B_affine, t_affine = cpd_affine(X, Y, w=w)

    # Auto-compute beta from point cloud density if not specified.
    # 5× median NN distance ensures the Gaussian kernel couples enough
    # neighbors for coherent motion (too-small beta → overfitting).
    if beta is None:
        tree = cKDTree(Y)
        k = min(6, len(Y))
        dists, _ = tree.query(Y, k=k)
        beta = 5.0 * float(np.median(dists[:, 1:])) if k > 1 else 10.0
        beta = max(beta, 1.0)

    # Stage 2: Non-rigid deformation
    _, W, _ = cpd_nonrigid(X, Y, beta=beta, lmbda=lmbda, w=w)

    # Stage 3: Dense field
    return cpd_displacement_field(
        Y, W, beta, fixed.shape,
        grid_spacing=grid_spacing,
        B=B_affine, t=t_affine,
        zoom_order=zoom_order, field_smooth_sigma=field_smooth_sigma,
    )


def register_volume_cpd(
    images: np.ndarray,
    ref_image: np.ndarray,
    mov_image: np.ndarray,
    boundary_mode: str = "constant",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Register multi-channel volume using CPD.

    Mirrors the signature of ``register_volume_tps()``.

    Parameters
    ----------
    images : np.ndarray
        Multi-channel volume, shape (Z, Y, X, C).
    ref_image : np.ndarray
        Reference image, shape (Z, Y, X).
    mov_image : np.ndarray
        Moving image, shape (Z, Y, X).
    boundary_mode : str
        How to handle out-of-bounds source coordinates:
        ``"constant"`` (default) fills with 0; ``"nearest"`` extends edges.
    **kwargs
        Passed to ``cpd_register()``: detection_threshold, max_control_points,
        beta, lmbda, w, affine_first, grid_spacing, candidate_radius,
        k_neighbors, zoom_order, field_smooth_sigma.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(registered_images, displacement_field)`` where:
        - registered_images: shape (Z, Y, X, C), same dtype as input
        - displacement_field: shape (Z, Y, X, 3), float32
    """
    displacement_field = cpd_register(ref_image, mov_image, **kwargs)

    n_channels = images.shape[-1]
    registered = np.empty_like(images)
    for c in range(n_channels):
        registered[:, :, :, c] = apply_tps_deformation(
            images[:, :, :, c], displacement_field,
            boundary_mode=boundary_mode,
        )

    return registered, displacement_field
