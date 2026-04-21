# Plan V2: Point Cloud Registration via Coherent Point Drift (CPD)

## Context

The current TPS registration detects spots independently in both images, matches by nearest-neighbor, and fits a TPS surface. This fails because:
1. NN matching commits to wrong correspondences when deformation exceeds inter-spot spacing
2. TPS has no model bias — overfits noise, extrapolates wildly

The user's insight: the two point clouds share **global structure** (same constellation of spots, just deformed). We should exploit this by first aligning globally, then refining locally — matching correspondence and transformation **simultaneously**, not sequentially.

**Coherent Point Drift (CPD)** [Myronenko & Song, NIPS 2006] does exactly this. It models the moving point set as GMM centroids and fits them to the fixed point set via EM. Correspondence emerges naturally from soft probabilistic assignments — no pre-matching needed.

## Algorithm: Two-Stage CPD

### Stage 1: Affine CPD (global alignment)

Captures translation, rotation, scaling, and shearing that the existing global `phase_correlate` may not fully correct (e.g., rotation, anisotropic scaling).

1. Model moving spots Y as GMM centroids, fixed spots X as observations
2. E-step: compute soft assignments P(m,n) = probability that fixed spot m corresponds to moving spot n
3. M-step: estimate affine transform (B, t) that maximizes likelihood
4. Iterate until convergence
5. Apply affine correction: `T(Y) = Y @ B.T + t`

### Stage 2: Non-rigid CPD (local deformation)

After affine alignment, residual deformation is small and smooth. Non-rigid CPD captures it via Gaussian-regularized displacements.

1. E-step: soft assignments between fixed spots and affine-corrected moving spots
2. M-step: solve for displacement weights W where `T(Y) = Y + G @ W`
   - G is N×N Gaussian kernel: `G(i,j) = exp(-||y_i - y_j||² / (2β²))`
   - Regularization: `(G + λσ² diag(1/P1)) W = diag(1/P1) PX - Y`
3. Iterate until σ² converges
4. Output: weight matrix W, kernel parameter β, point positions Y

### Stage 3: Dense field generation

CPD gives displacements at spot locations. To warp the full image:

1. Displacement at any point p: `d(p) = Σ_n W_n · exp(-||p - Y_n||² / (2β²))`
2. Evaluate on coarse 3D grid (stride 16-32 in YX) → O(grid_points × N)
3. Zoom to full resolution with cubic interpolation
4. Apply via existing `apply_tps_deformation()` (slice-by-slice warping)

### Why CPD over the current approach

| Aspect | Current TPS | CPD |
|--------|-------------|-----|
| Correspondence | Hard NN matching (commits to wrong matches) | Soft probabilistic (correct matches emerge) |
| Outlier handling | None (all matches weighted equally) | Uniform noise component absorbs unmatched spots |
| Model bias | None (TPS extrapolates freely) | Gaussian kernel regularization (smooth, bounded) |
| Global structure | Ignored (purely local matching) | Global GMM fitting exploits constellation shape |
| Iterations | Single-shot | EM converges from coarse to fine (σ² annealing) |

### Computational Cost

For N=1000 spots (after subsampling), 150 EM iterations:
- G kernel: 1000² × 8 bytes = 8 MB (computed once)
- P matrix: M×N per iteration → 8 MB
- Linear solve: 1000×1000 system → ~1ms per iteration
- Total: **< 1 second** for the CPD itself
- Dense field evaluation: O(grid_points × N) → ~1 second
- Main cost is spot detection + image warping (reuses existing code)

## Implementation

### Files to Modify

| File | Change |
|------|--------|
| `src/python/starfinder/registration/pointset.py` | Add CPD functions (keep all existing TPS code) |
| `src/python/starfinder/registration/__init__.py` | Add `cpd_register` to exports |
| `src/python/starfinder/dataset/fov.py` | Add `method="cpd"` routing in `local_registration()` |
| `src/python/test/test_pointset.py` | Add `TestCPDRegistration` (5 tests) |

No new dependencies — implemented in pure numpy/scipy.

### New Functions in `pointset.py`

#### 1. `_gaussian_kernel(Y, beta)`
```python
def _gaussian_kernel(Y: np.ndarray, beta: float) -> np.ndarray:
    """Compute N×N Gaussian kernel matrix.
    G(i,j) = exp(-||y_i - y_j||² / (2β²))
    """
    # Y: (N, D)
    diff = Y[:, None, :] - Y[None, :, :]  # (N, N, D)
    return np.exp(-np.sum(diff**2, axis=2) / (2 * beta**2))
```

#### 2. `cpd_affine(X, Y, w=0.1, max_iter=100, tol=1e-5)`
```python
def cpd_affine(
    X: np.ndarray,  # (M, D) fixed points
    Y: np.ndarray,  # (N, D) moving points
    w: float = 0.1,  # outlier weight
    max_iter: int = 100,
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Affine CPD: find B, t such that T(Y) = Y @ B.T + t.
    Returns (transformed_Y, B, t).
    """
    # Initialize sigma² from mean pairwise distance
    # E-step: P(m,n) ∝ exp(-||x_m - t_n||²/(2σ²)), normalized + outlier term
    # M-step: solve for B, t from weighted correspondences
    # Update σ²
    # Iterate until convergence
```

#### 3. `cpd_nonrigid(X, Y, beta=3.0, lmbda=2.0, w=0.1, max_iter=150, tol=1e-5)`
```python
def cpd_nonrigid(
    X: np.ndarray,  # (M, D) fixed points
    Y: np.ndarray,  # (N, D) moving points
    beta: float = 3.0,  # Gaussian kernel width (pixels)
    lmbda: float = 2.0,  # regularization weight
    w: float = 0.1,  # outlier weight
    max_iter: int = 150,
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Non-rigid CPD: find W such that T(Y) = Y + G @ W.

    Returns (transformed_Y, W, G).

    Key equations per iteration:
      E-step: P(m,n) = exp(-||x_m - t_n||²/(2σ²)) / (Σ_k exp(...) + c)
              where c = (2πσ²)^(D/2) · w/(1-w) · M/N
      M-step: (G + λσ² diag(1/P1)) W = diag(1/P1) P^T X - Y
      Update: T(Y) = Y + G @ W
      σ² = Σ P(m,n)||x_m - t_n||² / (Np · D)
    """
```

Core loop (~40 lines):
```python
G = _gaussian_kernel(Y, beta)  # (N, N), computed once
T = Y.copy()
sigma2 = np.sum((X[None] - Y[:, None])**2) / (D * M * N)

for _ in range(max_iter):
    # E-step: responsibilities
    diff = X[:, None] - T[None]       # (M, N, D)
    exp_term = np.exp(-np.sum(diff**2, axis=2) / (2 * sigma2))  # (M, N)
    c = (2 * np.pi * sigma2) ** (D/2) * w / (1 - w) * M / N
    denom = exp_term.sum(axis=1, keepdims=True) + c
    P = exp_term / denom              # (M, N)

    # M-step: solve for W
    P1 = P.sum(axis=0)                # (N,)
    PX = P.T @ X                      # (N, D)
    Np = P1.sum()

    diag_inv = np.diag(1.0 / (P1 + 1e-10))
    A = G + lmbda * sigma2 * diag_inv
    B = diag_inv @ PX - Y
    W = np.linalg.solve(A, B)         # (N, D)

    # Update
    T = Y + G @ W
    sigma2_new = np.sum(P * np.sum(diff**2, axis=2)) / (Np * D)
    sigma2_new = max(sigma2_new, 1e-10)

    if abs(sigma2_new - sigma2) / sigma2 < tol:
        break
    sigma2 = sigma2_new

return T, W, G
```

#### 4. `cpd_displacement_field(Y, W, beta, shape, grid_spacing=16)`
```python
def cpd_displacement_field(
    Y: np.ndarray,       # (N, D) original moving point positions
    W: np.ndarray,       # (N, D) CPD weight matrix
    beta: float,         # Gaussian kernel width
    shape: tuple[int, int, int],  # (Z, Y, X) volume shape
    grid_spacing: int = 16,
) -> np.ndarray:
    """Generate dense displacement field from CPD weights.

    Displacement at point p: d(p) = Σ_n W_n · exp(-||p - Y_n||² / (2β²))
    Evaluates on coarse grid, zooms to full resolution.
    Returns (Z, Y, X, 3) float32 field.
    """
    # Build coarse grid
    # For each coarse point p, compute: d(p) = Σ_n W_n · K(p, Y_n)
    # where K(p, y) = exp(-||p - y||² / (2β²))
    # This is a matrix-vector product: coarse_displacements = K_coarse @ W
    # K_coarse shape: (n_grid_points, N)
    # Zoom each displacement component to full resolution
```

#### 5. `cpd_register(fixed, moving, ...)`
```python
def cpd_register(
    fixed: np.ndarray,       # (Z, Y, X) fixed volume
    moving: np.ndarray,      # (Z, Y, X) moving volume
    detection_threshold: float = 3.0,
    max_control_points: int = 1000,
    beta: float = 3.0,
    lmbda: float = 2.0,
    w: float = 0.1,
    affine_first: bool = True,
    grid_spacing: int = 16,
) -> np.ndarray:
    """End-to-end CPD registration.

    1. Detect spots in both volumes (reuse detect_spots from metrics.py)
    2. Subsample to max_control_points (reuse subsample_control_points)
    3. Optional: affine CPD for global alignment
    4. Non-rigid CPD for local deformation
    5. Generate dense displacement field

    Returns (Z, Y, X, 3) displacement field.
    """
```

#### 6. Update `register_volume_tps` — add `method="cpd"` routing

```python
def register_volume_tps(images, ref_image, mov_image, *, method="cpd", **kwargs):
    if method == "cpd":
        field = cpd_register(ref_image, mov_image, **cpd_kwargs)
    elif method == "tps":
        field = tps_register(ref_image, mov_image, **tps_kwargs)
    # ... apply field to all channels
```

### Key Parameters and Tuning

| Parameter | Default | Meaning | Tuning guidance |
|-----------|---------|---------|-----------------|
| `beta` | 3.0 | Gaussian kernel width (px). Controls deformation smoothness. | Increase for smoother fields; decrease for more local flexibility. Scale with image size: ~1-5% of image width. |
| `lmbda` | 2.0 | Regularization weight. Higher = smoother, less flexible. | Increase if field is noisy; decrease if under-fitting. |
| `w` | 0.1 | Expected outlier fraction. | 0 = no outliers; 0.3 = 30% unmatched spots. |
| `max_control_points` | 1000 | Cap on point cloud size (farthest-point sampling). | 500 for speed; 2000 for accuracy on dense data. |
| `detection_threshold` | 3.0 | k-sigma for spot detection (lower = more spots). | 2.0 for sparse volumes; 3.0 for dense. |
| `grid_spacing` | 16 | Coarse grid stride for field evaluation. | 8 for fine detail; 32 for speed. |

### Edge Cases

1. **Too few spots** (< 10 in either volume): Raise `ValueError`, FOV falls back to demons
2. **Point clouds with no overlap** (completely different spots): CPD's σ² stays large, W → 0 (identity transform). Harmless — returns near-zero field.
3. **Very large point clouds** (> 2000): Subsample with existing `subsample_control_points` before CPD
4. **Beta too small**: Displacement field becomes spiky. Guard: `beta >= 1.0`
5. **Convergence failure**: If max_iter reached without convergence, log warning and use current state (still a valid approximation)

### FOV Integration

`fov.py:local_registration(method="cpd")` routes to `register_volume_tps(method="cpd")`.

```python
# In FOV.local_registration():
if method == "cpd":
    from starfinder.registration.pointset import cpd_register, apply_tps_deformation
    field = cpd_register(ref_3d, mov_3d, **cpd_kwargs)
    for c in range(n_channels):
        registered[:,:,:,c] = apply_tps_deformation(images[:,:,:,c], field)
```

Fallback to demons on `ValueError` when `fallback=True` (same pattern as existing TPS).

## Tests

5 new tests in `TestCPDRegistration`:

1. **`test_cpd_nonrigid_identity`** — Identical point clouds → near-zero displacements (W ≈ 0)
2. **`test_cpd_affine_recovery`** — Known affine transform → affine CPD recovers B and t within tolerance
3. **`test_cpd_nonrigid_polynomial`** — Point cloud with polynomial deformation → CPD reduces RMS displacement error > 50%
4. **`test_cpd_register_improves_ncc`** — Full pipeline on synthetic volume with polynomial deformation → NCC improves
5. **`test_cpd_register_too_few_spots`** — Dark volume → raises ValueError

## Verification

1. `cd src/python && uv run pytest test/ -v` — all existing + 5 new tests pass
2. Run on polynomial_small benchmark: target NCC improvement > 5× over current TPS (> 0.13)
3. Compare with demons on polynomial_small: target > 50% of demons NCC
4. Memory: no SimpleITK, peak RSS < 20% of demons
5. Speed: CPD point cloud registration < 5 seconds; total (including field generation + warping) < 50% of demons time
