# Plan: Redesign Spot-Based Registration → Block Polynomial + TPS Residual

## Context

The current TPS registration (`pointset.py`) achieves NCC 0.027 on polynomial_small — essentially no improvement. Demons achieves 0.386. The root causes are:

1. **NN spot matching fails** on smooth global deformations — spots shift in similar directions, causing wrong matches
2. **TPS has no model bias** — extrapolates poorly in empty regions, overfits noise
3. **Single-shot fitting** — no iterative refinement, no recovery from bad matches

The polynomial_small deformation is a 6-term polynomial per axis (18 parameters total), max 15px displacement. This is actually the *easiest* deformation to model parametrically — we just need the right estimator.

## Approach: Block Phase Correlation + Polynomial Fit + TPS Residual

Replace spot-based NN matching with **block-based phase correlation** as the measurement step, and replace TPS with **polynomial model fitting** as the interpolation step.

### Two-Pass Pipeline

**Pass 1 — Coarse polynomial recovery:**
1. Divide volume into 64×64 YX blocks (50% overlap, full Z depth)
2. Phase-correlate each block pair → local (dz, dy, dx) integer shifts
3. RANSAC polynomial fit (degree 1: `c0 + c1*x + c2*y + c3*z + c4*xy + c5*yz`) → dense field
4. Apply polynomial correction to moving image

**Pass 2 — Fine TPS residual (optional):**
1. Re-correlate with 32×32 blocks on the corrected image
2. Fit 2D TPS (YX only, Z is constant for blocks) to non-zero residual shifts
3. Compose: `total_field = poly_field + residual_field`
4. Apply total field to *original* moving image (avoid double interpolation)

### Why This Works

- Phase correlation uses **all image content** (not just spots) → reliable shift measurement
- Polynomial model has **6 terms** matching the ground truth structure → perfect extrapolation
- RANSAC rejects **outlier blocks** (low contrast, ambiguous correlation peaks)
- TPS residual captures **non-polynomial residuals** at sub-pixel precision
- No SimpleITK dependency → ~6x less memory than demons

### Validated Performance (from prototyping)

| Method | NCC (large) | Match Rate | Time |
|--------|-------------|------------|------|
| Current TPS | 0.027 | 0.009 | 51s |
| **Block+Poly** | **0.230** | **0.324** | 9s |
| **Block+Poly+TPS** | **0.267** | **0.478** | 20s |
| Demons | 0.386 | 0.503 | 58s |

## Implementation

### Files to Modify

| File | Change |
|------|--------|
| `src/python/starfinder/registration/pointset.py` | Add 5 new functions, update `register_volume_tps` routing |
| `src/python/starfinder/registration/__init__.py` | Add `block_register` to exports |
| `src/python/starfinder/dataset/fov.py` | Add `method="block"` routing in `local_registration()` |
| `src/python/test/test_pointset.py` | Add `TestBlockRegistration` (5 tests) |

### New Functions in `pointset.py`

All existing functions (TPS pipeline) remain unchanged for backward compatibility.

#### 1. `block_phase_correlate(fixed, moving, block_size=64, step=None, min_signal=50)`
- Divides volume into overlapping YX blocks (full Z depth)
- Calls existing `phase_correlate()` per block
- Skips dark blocks (`max < min_signal`)
- Returns `(positions, shifts)` — block centers and detected integer shifts

#### 2. `fit_polynomial_field(positions, displacements, shape, degree=1, ransac=True, ransac_threshold=2.0)`
- Normalizes positions to [-1, 1]
- Builds design matrix: degree 1 = `[1, x, y, z, xy, yz]` (6 terms)
- RANSAC or ordinary least-squares fit
- Evaluates polynomial on dense meshgrid → `(Z, Y, X, 3)` field

#### 3. `_ransac_polynomial(A, displacements, threshold=2.0, n_iterations=100, min_samples=10)`
- Manual RANSAC (no sklearn dependency)
- Sample → fit → count inliers → refit on best inlier set
- Returns `(coefficients, inlier_mask)`

#### 4. `_tps_residual_2d(fixed, corrected, shape, block_size=32, smoothing=3.0)`
- Block correlate on corrected image
- Filter to non-zero residual shifts
- 2D TPS fit (YX positions only, avoids constant-Z singularity)
- Zoom + broadcast to full 3D field

#### 5. `block_register(fixed, moving, coarse_block_size=64, fine_block_size=32, polynomial_degree=1, use_ransac=True, tps_residual=True, tps_smoothing=3.0, min_signal=50)`
- Main entry point: pass 1 (polynomial) + pass 2 (TPS residual)
- Composes fields additively
- Returns `(Z, Y, X, 3)` displacement field

#### 6. Update `register_volume_tps` — add `method` parameter
- `method="block"` (new default) → calls `block_register`
- `method="tps"` (legacy) → calls existing `tps_register`
- Same return signature: `(registered_images, displacement_field)`

### Edge Cases

1. **Too few blocks with signal**: `block_phase_correlate` returns < 7 points → `ValueError`, FOV falls back to demons
2. **Degenerate polynomial**: RANSAC finds < 50% inliers → skip RANSAC, use raw least-squares
3. **TPS residual failure**: < 5 non-zero residual shifts → skip pass 2, return polynomial-only field
4. **Very thin volumes** (Z < 8): Set Z-displacement to 0 in polynomial, only fit YX components
5. **Integer quantization**: Phase correlation returns integer shifts; polynomial fit averages across many blocks for sub-pixel coefficient estimation

### FOV Integration

`fov.py:local_registration(method="block")` routes to `register_volume_tps(method="block")` with block-specific kwargs. Fallback to demons on `ValueError` when `fallback=True`.

## Tests

5 new tests in `TestBlockRegistration`:
1. `test_block_correlate_identity` — identical images → near-zero shifts
2. `test_polynomial_field_recovery` — known polynomial from sampled points → field error < 0.5px
3. `test_block_register_polynomial_deformation` — NCC improves after block registration
4. `test_block_register_too_dark` — raises ValueError on dark images
5. `test_register_volume_tps_block_method` — shape/dtype correctness with `method="block"`

## Verification

1. `cd src/python && uv run pytest test/ -v` — all existing + new tests pass
2. Run block registration on polynomial_small benchmark data → NCC > 0.2 (vs current 0.027)
3. Speed: < 50% of demons runtime
4. Memory: no SimpleITK → < 20% of demons memory
