"""Private registration numerical implementation."""
from __future__ import annotations
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import zoom
from ._fields import sanitize_displacement_field
from ._landmarks import detect_and_match_spots, subsample_control_points
def tps_displacement_field(
    positions: np.ndarray,
    displacements: np.ndarray,
    shape: tuple[int, int, int],
    smoothing: float = 1.0,
    grid_spacing: int = 32,
    zoom_order: int = 3,
    field_smooth_sigma: float | None = None,
    clamp_sampling_coordinates: bool = True,
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
    field = sanitize_displacement_field(field, clamp=clamp_sampling_coordinates, smooth_sigma=field_smooth_sigma).field

    return field
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
    clamp_sampling_coordinates: bool = True,
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
        clamp_sampling_coordinates=clamp_sampling_coordinates,
    )
