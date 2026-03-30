# Plan: Mitigate Interpolation Artifacts in Deformation-Based Registration — FINISHED

## Context

CPD (and TPS/demons) registration produces two types of artifacts in warped images:
1. **Blank backgrounds** — out-of-bounds source coordinates filled with 0
2. **Weird noise/distortion shapes** — displacement field folding + cubic zoom ringing

These affect both Python (`map_coordinates(cval=0)`) and MATLAB (`imwarp(DefaultPixelValue=0)`).

The recent CPD correspondence-aware subsampling (`_subsample_with_neighbors`, `_gather_candidates`,
orphan trimming) improves CPD weight quality upstream, which reduces folding severity — but does not
eliminate the downstream artifacts from coarse-grid zoom and OOB boundary handling. The fix adds
opt-in mitigations to the field-generation and warping functions.

## Root Causes

| Artifact | Root Cause | Location |
|----------|-----------|----------|
| Blank bands | `cval=0` for OOB source coords | `apply_tps_deformation` L259, `apply_deformation` L364 |
| Noise shapes | `det(J) < 0` field folding | CPD field generation (sharp kernel transitions) |
| Wavy distortion | Cubic zoom ringing | `cpd_displacement_field` L758, `tps_displacement_field` L216 |

## Changes

### 1. New `sanitize_displacement_field()` in `pointset.py`

```python
def sanitize_displacement_field(
    field: np.ndarray,
    shape: tuple[int, int, int],
    *,
    clamp: bool = True,
    clamp_margin: float = 1.0,
    smooth_sigma: float | None = None,
    detect_folds: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
```

- **Clamp** (default on): Clip `position + displacement` to `[margin, dim-1-margin]`. O(N) — negligible.
- **Smooth**: `gaussian_filter` per displacement component. Removes zoom ringing + prevents folding. ~1s on tissue-size.
- **Fold detection**: Jacobian determinant via finite differences. Returns boolean mask where `det(J) < 0`. Diagnostic only.

### 2. Expose `zoom_order` in field generation

**`tps_displacement_field()`** and **`cpd_displacement_field()`**: Add `zoom_order: int = 3` parameter.
`zoom_order=1` (linear) eliminates ringing entirely. Change only the `zoom(order=...)` call.

Also add `field_smooth_sigma: float | None = None` — if set, call `sanitize_displacement_field()` after zoom.

### 3. Add `boundary_mode` to warping functions

**`apply_tps_deformation()`**: Add `boundary_mode: str = "constant"`.
```python
map_coordinates(volume, coords, order=1, mode=boundary_mode, cval=0)
```
`"nearest"` extends edge pixels instead of filling with black.

**`apply_deformation()` (demons.py)**: Add `boundary_mode: str = "constant"`.
```python
if boundary_mode == "nearest":
    resampler.UseNearestNeighborExtrapolatorOn()
```

### 4. Thread parameters through wrappers

| Function | New params |
|----------|-----------|
| `tps_register()` | `zoom_order`, `field_smooth_sigma` |
| `cpd_register()` | `zoom_order`, `field_smooth_sigma` |
| `register_volume_tps()` | `boundary_mode` |
| `register_volume_cpd()` | `boundary_mode` |
| `register_volume_local()` | `boundary_mode` |
| `FOV.local_registration()` | `boundary_mode` |

All default to current behavior — backward-compatible, no breaking changes.

## Files to modify

1. `src/python/starfinder/registration/pointset.py` — sanitize function, boundary_mode, zoom_order, field_smooth_sigma
2. `src/python/starfinder/registration/demons.py` — boundary_mode in `apply_deformation` + `register_volume_local`
3. `src/python/starfinder/dataset/fov.py` — thread boundary_mode through `local_registration`
4. `src/python/test/test_pointset.py` — tests for sanitize, boundary modes, zoom order

## Implementation order

1. `sanitize_displacement_field()` in pointset.py (standalone, testable)
2. `boundary_mode` in `apply_tps_deformation()`
3. `zoom_order` + `field_smooth_sigma` in `tps_displacement_field()` and `cpd_displacement_field()`
4. Thread through `tps_register()`, `cpd_register()`, `register_volume_tps()`, `register_volume_cpd()`
5. `boundary_mode` in `apply_deformation()` and `register_volume_local()` (demons.py)
6. Thread through `FOV.local_registration()` (fov.py)
7. Tests

## Verification

1. **Unit**: sanitize — clamp prevents OOB, smooth reduces gradient, fold detection finds known folds
2. **Integration**: `boundary_mode="nearest"` eliminates black bands on edge-shifted volume
3. **Regression**: All existing tests pass unchanged
4. **Visual**: Re-run CPD benchmark on artifact case with `boundary_mode="nearest"` + `zoom_order=1` + `field_smooth_sigma=1.0`

## Estimated scope

~210 lines (~100 sanitize function, ~30 param additions, ~80 tests)
