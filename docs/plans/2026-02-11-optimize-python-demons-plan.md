# Plan: Optimize Python Demons Registration to Match MATLAB Quality

## Context

Visual inspection and benchmark metrics show Python SimpleITK demons registration produces lower quality than MATLAB `imregdemons`. On real data (LN dataset), MATLAB dramatically outperforms Python on ALL metrics (+0.313 NCC, +0.467 Match Rate). On synthetic data, MATLAB leads by ~3-5% on NCC/SSIM.

**Root cause analysis** identified two critical differences:

1. **Broken pyramid downsampling**: Python `sitk.Shrink()` does naive subsampling (every Nth voxel) with no anti-aliasing. MATLAB uses a Butterworth low-pass filter before downsampling (`antialiasResize`). This destroys sparse fluorescence signal at coarse pyramid levels, explaining why pyramids are catastrophic for Python but essential for MATLAB.

2. **Algorithm variant mismatch**: MATLAB uses classic Thirion demons (fixed-image gradient only, additive update). Python defaults to diffeomorphic demons which also applies `UpdateFieldStandardDeviations(sigma*0.5)` — extra smoothing that over-smooths fine displacements.

**Solution**: Keep SimpleITK's optimized C++ demons engine for per-level iteration, but wrap it in a custom anti-aliased multi-resolution pyramid matching MATLAB's approach.

---

## Step 1: Create Pyramid Utilities Module

**Create**: `src/python/starfinder/registration/pyramid.py` (~100 lines)

Implement MATLAB-matching anti-aliased downsampling/upsampling:

- `butterworth_3d(shape, cutoff, order=2)` — 3D Butterworth low-pass filter in frequency domain, matching MATLAB's `butterwth()`. Uses `np.fft.fftfreq` + normalized frequency grid.
- `antialias_resize(volume, factor)` — Anti-aliased 3D resize matching MATLAB's `antialiasResize()`:
  - Downsample (`factor < 1`): Apply Butterworth filter with cutoff `0.5 * factor`, then `scipy.ndimage.zoom(order=1)`
  - Upsample (`factor >= 1`): Just `scipy.ndimage.zoom(order=1)` (no filtering needed)
- `pad_for_pyramiding(volume, pyramid_levels)` → `(padded, pad_vec)` — Replicate-border padding so dimensions are divisible by `2^(levels-1)`, matching MATLAB's `padForPyramiding()`
- `crop_padding(volume, pad_vec)` — Remove padding

**Dependencies**: Only `numpy` and `scipy.ndimage.zoom` (already in project).

**Reference**: `imregdemons_ref/MultiResolutionDemons3D.m` lines 100-153 (antialiasResize + butterwth)

---

## Step 2: Add Anti-Aliased Pyramid Mode to `demons_register`

**Modify**: `src/python/starfinder/registration/demons.py`

Add `pyramid_mode` parameter to existing `demons_register()`:

```python
def demons_register(
    fixed, moving,
    iterations=None,
    smoothing_sigma=0.5,
    method="diffeomorphic",
    pyramid_mode="sitk",  # NEW: "sitk" (current) or "antialias" (MATLAB-style)
)
```

When `pyramid_mode="antialias"` and `len(iterations) > 1`:
1. Convert to `float64` (matching MATLAB's `double()` precision — currently uses `float32`)
2. Pad both volumes with `pad_for_pyramiding()`
3. For each pyramid level (coarse → fine):
   a. Downsample fixed/moving with `antialias_resize(vol, 0.5^(levels-level))`
   b. If not first level: upsample displacement field with `antialias_resize(field, 2.0)` and scale magnitudes ×2
   c. Run SimpleITK **single-level** demons with initial field
4. Crop padding from output field

When `pyramid_mode="sitk"` or single-level: existing behavior unchanged.

**Key**: Use `sitk.DemonsRegistrationFilter()` (classic Thirion) when `method="demons"` — this matches MATLAB. The existing code already supports this filter but defaults to diffeomorphic.

---

## Step 3: Add MATLAB-Compatible Convenience Config

**Modify**: `src/python/starfinder/registration/demons.py`

```python
def matlab_compatible_config() -> dict:
    """Return config matching MATLAB imregdemons defaults."""
    return {
        "iterations": [100, 50, 25],    # 3 pyramid levels, coarse-to-fine
        "smoothing_sigma": 1.0,         # AFS default
        "method": "demons",             # Classic Thirion
        "pyramid_mode": "antialias",
    }
```

Usage: `field = demons_register(fixed, moving, **matlab_compatible_config())`

---

## Step 4: Update `register_volume_local` and Module Exports

**Modify**: `src/python/starfinder/registration/demons.py` — pass `pyramid_mode` through `register_volume_local()`

**Modify**: `src/python/starfinder/registration/__init__.py` — add `matlab_compatible_config` to imports and `__all__`

---

## Step 5: Write Tests

**Modify**: `src/python/test/test_demons.py` (~60 new lines)

Minimal focused tests:
- `test_butterworth_filter_shape` — correct shape, peak at center
- `test_antialias_resize_roundtrip` — down then up approximately recovers original
- `test_pad_crop_roundtrip` — pad then crop recovers exact original
- `test_identity_antialias_pyramid` — identical images with `pyramid_mode="antialias"` produce near-zero field
- `test_pyramid_improves_quality` — for known deformation, 3-level antialias pyramid gives better NCC than single-level

---

## Verification

1. Run unit tests: `cd src/python && uv run pytest test/test_demons.py -v`
2. Run benchmark on key datasets to compare against MATLAB:
   ```python
   from starfinder.registration import demons_register, matlab_compatible_config
   field = demons_register(ref, mov, **matlab_compatible_config())
   ```
3. Key datasets to validate: cell_culture_3D (medium size), LN (where MATLAB wins big), medium synthetic (quick iteration)
4. Expected outcome: NCC/SSIM should approach MATLAB levels, especially on LN dataset

---

## Files Changed

| File | Action | Lines |
|------|--------|-------|
| `src/python/starfinder/registration/pyramid.py` | CREATE | ~100 |
| `src/python/starfinder/registration/demons.py` | MODIFY | ~100 new |
| `src/python/starfinder/registration/__init__.py` | MODIFY | +2 lines |
| `src/python/test/test_demons.py` | MODIFY | ~60 new |
