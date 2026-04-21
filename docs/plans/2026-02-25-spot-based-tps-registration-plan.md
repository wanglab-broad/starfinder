# Phase E: Spot-Based TPS Local Registration

## Context

Registration is the dominant bottleneck at 44-60% of per-FOV runtime. After Phases A-D optimized everything around demons (streaming, FFT memory, extraction vectorization), the demons algorithm itself is the next target. This plan adds a **spot-based Thin Plate Spline (TPS)** local registration method that operates directly on matched spot correspondences — replacing the iterative voxel-level optimization of demons with a sparse interpolation approach native to fluorescence microscopy data.

**Why TPS for sparse fluorescence**: Demons wastes 95-99% of its computation on background pixels. TPS operates only on detected spots (the actual signal), fitting a smooth displacement field from ~1000 control points. No SimpleITK dependency required — pure scipy.

**Expected outcome**: ~2x speedup on local registration (tissue-2D: ~280s → ~125s per round), ~45% memory reduction (no iterative optimization buffers).

---

## Architecture

### Current flow
```
FOV.local_registration(method="demons")
  → register_volume_local() [demons.py]
    → demons_register() → displacement field (Z,Y,X,3)
    → SimpleITK warp per channel
```

### Proposed flow (additive — demons unchanged)
```
FOV.local_registration(method="tps")
  → register_volume_tps() [pointset.py]       ← NEW
    → detect_and_match_spots()                 ← NEW (reuses detect_spots from metrics.py)
    → subsample_control_points()               ← NEW (farthest-point sampling)
    → tps_displacement_field()                 ← NEW (RBFInterpolator + coarse grid + zoom)
    → apply_tps_deformation() per channel      ← NEW (scipy map_coordinates, slice-by-slice)
```

### Key design decisions
1. **New file** `registration/pointset.py` — keeps TPS isolated from demons code
2. **Same field format** — TPS produces `(Z,Y,X,3)` displacement field with `(dz,dy,dx)`, same as demons
3. **Scipy-only warping** — `map_coordinates` instead of SimpleITK ResampleImageFilter (no SimpleITK dependency for TPS path)
4. **Slice-by-slice warping** — process one Z-slice at a time to avoid 3.4 GB coordinate array
5. **Fallback** — if TPS has too few spot matches, raise `ValueError`; FOV catches and falls back to demons
6. **Coarse grid + zoom** — evaluate TPS on stride=32 grid (~550K points), zoom to full resolution with cubic interpolation

---

## Implementation Steps

### Step 1: Create `src/python/starfinder/registration/pointset.py`

Core module with 5 functions:

**`detect_and_match_spots(fixed, moving, detection_threshold=3.0, match_distance=10.0, min_matches=50)`**
- Detect spots in both (Z,Y,X) volumes using `detect_spots()` from `metrics.py` (percentile-based CCA centroids)
- Match via `scipy.spatial.cKDTree.query(k=1)` — greedy nearest-neighbor within `match_distance`
- Compute residuals: `matched_fixed - matched_moving` = local displacement at each matched spot
- Raise `ValueError` if `< min_matches` pairs

**`subsample_control_points(positions, displacements, max_points=1000)`**
- Greedy farthest-point sampling for uniform spatial coverage (not random)
- O(N × max_points), instant for N=10K

**`tps_displacement_field(positions, displacements, shape, smoothing=1.0, grid_spacing=32)`**
- Fit `RBFInterpolator(kernel='thin_plate_spline', smoothing=smoothing)` — multi-output (all 3 displacement axes at once)
- Evaluate on coarse grid: Z stride = adaptive (keep ≥15 Z points), YX stride = `grid_spacing`
- Zoom coarse field to full resolution with `scipy.ndimage.zoom(order=3)` per component
- Return `(Z,Y,X,3)` float32 displacement field

**`apply_tps_deformation(volume, displacement_field)`**
- Slice-by-slice `map_coordinates(order=1)` — 113 MB temp per slice vs 3.4 GB for full grid
- Returns same dtype as input

**`register_volume_tps(images, ref_image, mov_image, **kwargs)`**
- Calls `tps_register()` for field, then `apply_tps_deformation()` per channel
- Returns `(registered_images, displacement_field)` — mirrors `register_volume_local()` signature

### Step 2: Update `src/python/starfinder/registration/__init__.py`

Add exports:
```python
from starfinder.registration.pointset import register_volume_tps, tps_register
```

### Step 3: Update `src/python/starfinder/dataset/fov.py` — local_registration routing

Modify `local_registration()` to route `method="tps"` to `register_volume_tps()`:
- Add TPS-specific parameters: `detection_threshold`, `match_distance`, `min_matches`, `max_control_points`, `tps_smoothing`, `grid_spacing`
- Add `fallback: bool = True` parameter — catch `ValueError` from TPS, fall back to demons
- Existing demons path unchanged

### Step 4: Update `src/python/starfinder/dataset/fov.py` — streaming support

Add `local_method: str | None = None` and `local_kwargs: dict | None = None` to `run_streaming()`. In the per-round loop, insert `self.local_registration(layers_to_register=[round_name], method=local_method, **(local_kwargs or {}))` between global registration and extraction.

### Step 5: Add tests in `src/python/test/test_pointset.py`

4 tests:
- `test_identity_no_displacement` — identical images → near-zero field
- `test_known_local_deformation` — apply known polynomial deformation to synthetic volume, verify TPS recovers it (<2px mean error)
- `test_too_few_spots_raises` — `ValueError` when insufficient matches
- `test_register_volume_tps_shape` — output shape matches input, all channels warped

### Step 6: Run existing tests + E2E to verify no regressions

All 155+ existing tests must pass unchanged. TPS is additive — no existing code paths modified (only routing added in FOV).

---

## Performance Estimates (tissue-2D, 3072x3072x30, per round)

| Step | Current (demons) | Proposed (TPS) |
|------|-------------------|----------------|
| Spot detection (2 volumes) | — | ~15s |
| KDTree matching | — | <1s |
| TPS fit + coarse grid eval | — | ~5s |
| Zoom to full resolution | — | ~36s |
| Channel warping (4 ch) | ~60-80s (SimpleITK) | ~68s (map_coordinates) |
| Demons iterations (3-level pyramid) | ~200s | — |
| **Total per round** | **~280s** | **~125s** |

**Note**: Spot detection on ref image is done once and amortized across all non-ref rounds. Per additional round: ~110s.

## Memory Estimates (tissue-2D, streaming mode)

| Component | Demons | TPS |
|-----------|--------|-----|
| Images (ref + 1 round) | 2.2 GB | 2.2 GB |
| Registration overhead | ~6+ GB (pyramid + FFT) | ~3.5 GB (displacement field) |
| Warp temporary | ~2 GB (SimpleITK) | ~113 MB (per-slice) |
| **Peak RSS** | **~11 GB** | **~6 GB** |

---

## Critical Files

| File | Action |
|------|--------|
| `src/python/starfinder/registration/pointset.py` | **Create** — core TPS module |
| `src/python/starfinder/registration/__init__.py` | **Edit** — add TPS exports |
| `src/python/starfinder/dataset/fov.py` | **Edit** — route method="tps", streaming support |
| `src/python/test/test_pointset.py` | **Create** — 4 unit tests |

**Reference (read-only)**:
- `registration/metrics.py` — `detect_spots()`, `spot_matching_accuracy()` for reuse
- `registration/demons.py` — `register_volume_local()` signature to mirror

---

## Verification

1. **Unit tests**: 4 new tests in `test_pointset.py`
2. **Regression**: All 155+ existing tests pass unchanged
3. **E2E smoke test**: Run `run_streaming()` with `local_method="tps"` on small synthetic dataset
4. **Quality benchmark** (manual): Run on tissue-2D (2 FOVs) — compare NCC and Match Rate vs demons baseline. Accept if ≥ 90% of demons quality.
