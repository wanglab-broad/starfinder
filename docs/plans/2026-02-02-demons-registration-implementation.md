# Demons Registration Module Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement non-rigid registration using SimpleITK symmetric forces demons to refine alignment after global phase correlation.

**Architecture:** Three public functions (`demons_register`, `apply_deformation`, `register_volume_local`) in a new `demons.py` module, following the existing `phase_correlation.py` pattern. SimpleITK is lazy-imported as an optional dependency.

**Tech Stack:** SimpleITK (optional), NumPy, SciPy (for test deformation generation)

**Design Document:** `docs/plans/2026-02-02-demons-registration-design.md`

---

## Task 1: Create demons.py with Import Guard

**Files:**
- Create: `src/python/starfinder/registration/demons.py`

**Step 1: Create the module file with lazy import helper**

```python
"""Non-rigid registration using demons algorithm (SimpleITK)."""

from __future__ import annotations

import numpy as np


def _import_sitk():
    """Lazy import SimpleITK with helpful error message."""
    try:
        import SimpleITK as sitk
        return sitk
    except ImportError:
        raise ImportError(
            "SimpleITK required for local registration. "
            "Install with: uv add 'starfinder[local-registration]'"
        )
```

**Step 2: Verify file created**

Run: `ls -la src/python/starfinder/registration/demons.py`
Expected: File exists

**Step 3: Commit**

```bash
git add src/python/starfinder/registration/demons.py
git commit -m "feat(registration): add demons.py skeleton with import guard"
```

---

## Task 2: Write Failing Test for demons_register Identity

**Files:**
- Create: `src/python/test/test_demons.py`

**Step 1: Write the test file with skip marker and identity test**

```python
"""Tests for starfinder.registration.demons module."""

import numpy as np
import pytest

# Skip all tests if SimpleITK not installed
sitk = pytest.importorskip("SimpleITK", reason="SimpleITK required for local registration")


class TestDemonsRegister:
    """Tests for demons_register function."""

    def test_identity(self, mini_dataset):
        """Identical images produce near-zero displacement field."""
        from starfinder.io import load_multipage_tiff
        from starfinder.registration.demons import demons_register

        vol = load_multipage_tiff(mini_dataset / "FOV_001" / "round1" / "ch00.tif")
        field = demons_register(vol, vol)

        # Shape should be (Z, Y, X, 3) for displacement vectors
        assert field.shape == (*vol.shape, 3)
        # Displacement should be near zero for identical images
        assert np.abs(field).max() < 1.0, "Displacement field should be near-zero for identical images"
```

**Step 2: Run test to verify it fails**

Run: `cd src/python && uv run pytest test/test_demons.py::TestDemonsRegister::test_identity -v`
Expected: FAIL with "cannot import name 'demons_register'"

**Step 3: Commit**

```bash
git add src/python/test/test_demons.py
git commit -m "test(registration): add failing test for demons_register identity"
```

---

## Task 3: Implement demons_register

**Files:**
- Modify: `src/python/starfinder/registration/demons.py`

**Step 1: Add demons_register function**

Append to `demons.py`:

```python
def demons_register(
    fixed: np.ndarray,
    moving: np.ndarray,
    iterations: list[int] | None = None,
    smoothing_sigma: float = 1.0,
) -> np.ndarray:
    """
    Compute displacement field to align moving to fixed using demons.

    Args:
        fixed: Reference volume with shape (Z, Y, X).
        moving: Volume to align with shape (Z, Y, X).
        iterations: Number of iterations per pyramid level (coarse to fine).
            Defaults to [100, 50, 25] for 3 levels.
        smoothing_sigma: Standard deviation for displacement field smoothing.
            Higher values produce smoother deformations.

    Returns:
        Displacement field with shape (Z, Y, X, 3) where last dimension
        contains (dz, dy, dx) displacement vectors.
    """
    sitk = _import_sitk()

    if iterations is None:
        iterations = [100, 50, 25]

    # Convert numpy (Z, Y, X) to SimpleITK image
    fixed_sitk = sitk.GetImageFromArray(fixed.astype(np.float32))
    moving_sitk = sitk.GetImageFromArray(moving.astype(np.float32))

    # Configure symmetric forces demons filter
    demons = sitk.SymmetricForcesDemonsRegistrationFilter()
    demons.SetNumberOfIterations(iterations[0])
    demons.SetStandardDeviations(smoothing_sigma)
    demons.SetSmoothDisplacementField(True)
    demons.SetSmoothUpdateField(False)

    # Multi-resolution registration
    # Calculate pyramid levels based on Z dimension (matching MATLAB)
    n_levels = len(iterations)
    shrink_factors = [2 ** (n_levels - 1 - i) for i in range(n_levels)]
    smoothing_sigmas = [s * 2.0 for s in shrink_factors]

    # Use multi-resolution framework
    registration = sitk.ImageRegistrationMethod()
    registration.SetInitialTransform(sitk.DisplacementFieldTransform(fixed_sitk.GetDimension()))

    # For symmetric forces demons, we use the demons filter directly with multi-scale
    # SimpleITK's demons filter handles single-scale, so we run coarse-to-fine manually
    displacement_field = None

    for level, (n_iter, shrink) in enumerate(zip(iterations, shrink_factors)):
        # Shrink images for this level
        if shrink > 1:
            fixed_level = sitk.Shrink(fixed_sitk, [shrink] * 3)
            moving_level = sitk.Shrink(moving_sitk, [shrink] * 3)
        else:
            fixed_level = fixed_sitk
            moving_level = moving_sitk

        # Configure demons for this level
        demons_level = sitk.SymmetricForcesDemonsRegistrationFilter()
        demons_level.SetNumberOfIterations(n_iter)
        demons_level.SetStandardDeviations(smoothing_sigma)
        demons_level.SetSmoothDisplacementField(True)
        demons_level.SetSmoothUpdateField(False)

        # If we have a displacement field from previous level, upsample and use it
        if displacement_field is not None:
            # Resample displacement field to current level size
            displacement_field = sitk.Resample(
                displacement_field,
                fixed_level,
                sitk.Transform(),
                sitk.sitkLinear,
                0.0,
                displacement_field.GetPixelID(),
            )
            # Scale displacement vectors by 2 (since we doubled resolution)
            displacement_field = sitk.Compose([
                sitk.VectorIndexSelectionCast(displacement_field, i) * 2
                for i in range(3)
            ])
            demons_level.SetInitialDisplacementField(displacement_field)

        # Run demons at this level
        displacement_field = demons_level.Execute(fixed_level, moving_level)

    # Convert displacement field to numpy array
    # SimpleITK stores as (X, Y, Z) with vector components, we need (Z, Y, X, 3)
    field_array = sitk.GetArrayFromImage(displacement_field)

    return field_array
```

**Step 2: Run test to verify it passes**

Run: `cd src/python && uv run pytest test/test_demons.py::TestDemonsRegister::test_identity -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/python/starfinder/registration/demons.py
git commit -m "feat(registration): implement demons_register with multi-resolution"
```

---

## Task 4: Write Failing Test for apply_deformation

**Files:**
- Modify: `src/python/test/test_demons.py`

**Step 1: Add test class for apply_deformation**

Append to `test_demons.py`:

```python
class TestApplyDeformation:
    """Tests for apply_deformation function."""

    def test_identity_field(self, mini_dataset):
        """Zero displacement field returns original volume."""
        from starfinder.io import load_multipage_tiff
        from starfinder.registration.demons import apply_deformation

        vol = load_multipage_tiff(mini_dataset / "FOV_001" / "round1" / "ch00.tif")

        # Zero displacement field
        field = np.zeros((*vol.shape, 3), dtype=np.float32)
        result = apply_deformation(vol, field)

        assert result.shape == vol.shape
        # Should be nearly identical (interpolation may introduce tiny differences)
        np.testing.assert_allclose(result, vol, rtol=1e-4, atol=1e-4)
```

**Step 2: Run test to verify it fails**

Run: `cd src/python && uv run pytest test/test_demons.py::TestApplyDeformation::test_identity_field -v`
Expected: FAIL with "cannot import name 'apply_deformation'"

**Step 3: Commit**

```bash
git add src/python/test/test_demons.py
git commit -m "test(registration): add failing test for apply_deformation"
```

---

## Task 5: Implement apply_deformation

**Files:**
- Modify: `src/python/starfinder/registration/demons.py`

**Step 1: Add apply_deformation function**

Append to `demons.py`:

```python
def apply_deformation(
    volume: np.ndarray,
    displacement_field: np.ndarray,
) -> np.ndarray:
    """
    Apply displacement field to warp a volume.

    Args:
        volume: Input volume with shape (Z, Y, X).
        displacement_field: Displacement field with shape (Z, Y, X, 3)
            where last dimension contains (dz, dy, dx) vectors.

    Returns:
        Warped volume with same shape as input.
    """
    sitk = _import_sitk()

    # Convert to SimpleITK
    volume_sitk = sitk.GetImageFromArray(volume.astype(np.float32))
    field_sitk = sitk.GetImageFromArray(displacement_field.astype(np.float32), isVector=True)

    # Create displacement field transform
    transform = sitk.DisplacementFieldTransform(field_sitk)

    # Apply transform using resampling
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(volume_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(transform)

    warped_sitk = resampler.Execute(volume_sitk)

    # Convert back to numpy
    result = sitk.GetArrayFromImage(warped_sitk)

    return result.astype(volume.dtype)
```

**Step 2: Run test to verify it passes**

Run: `cd src/python && uv run pytest test/test_demons.py::TestApplyDeformation::test_identity_field -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/python/starfinder/registration/demons.py
git commit -m "feat(registration): implement apply_deformation"
```

---

## Task 6: Write Failing Test for register_volume_local

**Files:**
- Modify: `src/python/test/test_demons.py`

**Step 1: Add test class for register_volume_local**

Append to `test_demons.py`:

```python
class TestRegisterVolumeLocal:
    """Tests for register_volume_local function."""

    def test_multichannel(self, mini_dataset):
        """Registers all channels using computed field."""
        from starfinder.io import load_image_stacks
        from starfinder.registration.demons import register_volume_local

        images, _ = load_image_stacks(
            mini_dataset / "FOV_001" / "round1",
            ["ch00", "ch01", "ch02", "ch03"],
        )

        # Use ch00 max projection as ref/mov (identity case)
        ref_img = images[:, :, :, 0]
        mov_img = images[:, :, :, 0]

        registered, field = register_volume_local(images, ref_img, mov_img)

        assert registered.shape == images.shape
        assert field.shape == (*images.shape[:3], 3)
        # For identity case, registered should be similar to input
        np.testing.assert_allclose(registered, images, rtol=0.1, atol=1.0)
```

**Step 2: Run test to verify it fails**

Run: `cd src/python && uv run pytest test/test_demons.py::TestRegisterVolumeLocal::test_multichannel -v`
Expected: FAIL with "cannot import name 'register_volume_local'"

**Step 3: Commit**

```bash
git add src/python/test/test_demons.py
git commit -m "test(registration): add failing test for register_volume_local"
```

---

## Task 7: Implement register_volume_local

**Files:**
- Modify: `src/python/starfinder/registration/demons.py`

**Step 1: Add register_volume_local function**

Append to `demons.py`:

```python
def register_volume_local(
    images: np.ndarray,
    ref_image: np.ndarray,
    mov_image: np.ndarray,
    iterations: list[int] | None = None,
    smoothing_sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Register multi-channel volume using demons.

    This function computes the displacement field between ref_image and mov_image,
    then applies that field to all channels in the images volume.

    Args:
        images: Multi-channel volume with shape (Z, Y, X, C).
        ref_image: Reference image with shape (Z, Y, X) for field calculation.
        mov_image: Moving image with shape (Z, Y, X) for field calculation.
        iterations: Number of iterations per pyramid level.
            Defaults to [100, 50, 25] for 3 levels.
        smoothing_sigma: Standard deviation for displacement field smoothing.

    Returns:
        Tuple of (registered_images, displacement_field).
        - registered_images: Warped volume with shape (Z, Y, X, C)
        - displacement_field: Computed field with shape (Z, Y, X, 3)
    """
    # Compute displacement field
    displacement_field = demons_register(
        ref_image, mov_image,
        iterations=iterations,
        smoothing_sigma=smoothing_sigma,
    )

    # Apply deformation to each channel
    n_channels = images.shape[-1]
    registered = np.zeros_like(images)

    for c in range(n_channels):
        registered[:, :, :, c] = apply_deformation(images[:, :, :, c], displacement_field)

    return registered, displacement_field
```

**Step 2: Run test to verify it passes**

Run: `cd src/python && uv run pytest test/test_demons.py::TestRegisterVolumeLocal::test_multichannel -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/python/starfinder/registration/demons.py
git commit -m "feat(registration): implement register_volume_local"
```

---

## Task 8: Add Deformation Recovery Test

**Files:**
- Modify: `src/python/test/test_demons.py`

**Step 1: Add test for recovering known deformation**

Add to `TestDemonsRegister` class:

```python
    def test_known_deformation(self, mini_dataset):
        """Recovers direction of synthetic smooth deformation."""
        from scipy.ndimage import map_coordinates, gaussian_filter
        from starfinder.io import load_multipage_tiff
        from starfinder.registration.demons import demons_register

        vol = load_multipage_tiff(mini_dataset / "FOV_001" / "round1" / "ch00.tif")

        # Create smooth synthetic deformation (simulate tissue warping)
        rng = np.random.default_rng(42)
        strength = 3.0

        # Generate smooth random displacement field
        true_field = rng.standard_normal((3, *vol.shape)) * strength
        # Smooth spatially to make it tissue-like
        true_field = gaussian_filter(true_field, sigma=[0, 5, 5, 5])

        # Apply deformation to create "moving" image
        coords = np.meshgrid(*[np.arange(s) for s in vol.shape], indexing='ij')
        warped_coords = [c + f for c, f in zip(coords, true_field)]
        deformed = map_coordinates(vol, warped_coords, order=1, mode='constant', cval=0)

        # Recover displacement field
        estimated_field = demons_register(vol, deformed.astype(np.float32))

        # The estimated field should have similar direction to true field
        # (we check correlation, not exact match, since demons is iterative)
        true_magnitude = np.linalg.norm(true_field.transpose(1, 2, 3, 0), axis=-1)
        est_magnitude = np.linalg.norm(estimated_field, axis=-1)

        # Both should have deformation in similar regions
        mask = true_magnitude > 1.0
        assert est_magnitude[mask].mean() > 0.5, "Should detect deformation in warped regions"
```

**Step 2: Run test to verify it passes**

Run: `cd src/python && uv run pytest test/test_demons.py::TestDemonsRegister::test_known_deformation -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/python/test/test_demons.py
git commit -m "test(registration): add deformation recovery test"
```

---

## Task 9: Update Module Exports

**Files:**
- Modify: `src/python/starfinder/registration/__init__.py`

**Step 1: Add demons exports**

Replace contents of `__init__.py`:

```python
"""Registration module for image alignment."""

from starfinder.registration.phase_correlation import (
    phase_correlate,
    apply_shift,
    register_volume,
)
from starfinder.registration._skimage_backend import phase_correlate_skimage

# Local registration exports (lazy import - SimpleITK optional)
# Import will fail gracefully with helpful message if SimpleITK missing
from starfinder.registration.demons import (
    demons_register,
    apply_deformation,
    register_volume_local,
)

__all__ = [
    # Global (rigid)
    "phase_correlate",
    "apply_shift",
    "register_volume",
    "phase_correlate_skimage",
    # Local (non-rigid)
    "demons_register",
    "apply_deformation",
    "register_volume_local",
]
```

**Step 2: Run all registration tests**

Run: `cd src/python && uv run pytest test/test_registration.py test/test_demons.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/python/starfinder/registration/__init__.py
git commit -m "feat(registration): export demons functions from module"
```

---

## Task 10: Add QC Notebook Section

**Files:**
- Modify: `tests/qc_registration.ipynb`

**Step 1: Add Section 7 cells to notebook**

Add new markdown cell:
```markdown
## 7. Local (Non-Rigid) Registration

This section demonstrates when local registration is needed vs global-only.

### 7.1 Create Synthetic Local Deformation
```

Add code cell:
```python
from scipy.ndimage import map_coordinates, gaussian_filter

def create_local_deformation(volume, strength=5.0, seed=42):
    """Apply smooth spatially-varying deformation (simulates tissue warping)."""
    rng = np.random.default_rng(seed)

    # Generate smooth random displacement field
    field = rng.standard_normal((3, *volume.shape)) * strength
    field = gaussian_filter(field, sigma=[0, 10, 10, 10])  # Smooth spatially

    # Create sampling coordinates
    coords = np.meshgrid(*[np.arange(s) for s in volume.shape], indexing='ij')
    warped_coords = [c + f for c, f in zip(coords, field)]

    deformed = map_coordinates(volume, warped_coords, order=1, mode='constant', cval=0)
    return deformed, field.transpose(1, 2, 3, 0)  # Return as (Z, Y, X, 3)

# Apply local deformation to reference volume
ref_volume = ref_stack[:, :, :, 0]  # Use channel 0
deformed, true_field = create_local_deformation(ref_volume, strength=8.0)

print(f"Reference shape: {ref_volume.shape}")
print(f"Deformed shape: {deformed.shape}")
print(f"True field shape: {true_field.shape}")
print(f"Max deformation: {np.abs(true_field).max():.2f} pixels")
```

Add markdown cell:
```markdown
### 7.2 Global Registration Fails on Local Deformation

Global (phase correlation) registration can only correct bulk translation - it cannot fix spatially-varying deformation.
```

Add code cell:
```python
from starfinder.registration import register_volume

# Try global registration on locally deformed volume
deformed_4d = deformed[..., np.newaxis]  # Add channel dim
global_registered, shifts = register_volume(deformed_4d, ref_volume, deformed)

print(f"Global detected shift: {shifts}")

# Visualize - should show residual misalignment
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Before registration
composite_before = make_composite(ref_volume.max(axis=0), deformed.max(axis=0))
axes[0].imshow(composite_before)
axes[0].set_title('Before Registration\n(Green=Ref, Magenta=Deformed)')

# After global registration
composite_global = make_composite(ref_volume.max(axis=0), global_registered[:,:,:,0].max(axis=0))
axes[1].imshow(composite_global)
axes[1].set_title(f'After GLOBAL Registration\nShift: {shifts}')

# Difference map
diff = np.abs(ref_volume - global_registered[:,:,:,0]).max(axis=0)
im = axes[2].imshow(diff, cmap='hot')
axes[2].set_title('Residual Error (Global)\nNon-uniform = local deformation')
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.show()
```

Add markdown cell:
```markdown
### 7.3 Local Registration Succeeds

Demons registration can correct spatially-varying deformation.
```

Add code cell:
```python
try:
    from starfinder.registration import register_volume_local

    # Local registration
    local_registered, estimated_field = register_volume_local(
        deformed_4d, ref_volume, deformed,
        iterations=[100, 50, 25],
        smoothing_sigma=1.0,
    )

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # After local registration
    composite_local = make_composite(ref_volume.max(axis=0), local_registered[:,:,:,0].max(axis=0))
    axes[0].imshow(composite_local)
    axes[0].set_title('After LOCAL Registration\n(Should be gray/white = aligned)')

    # Difference comparison
    diff_global = np.abs(ref_volume - global_registered[:,:,:,0]).max(axis=0)
    diff_local = np.abs(ref_volume - local_registered[:,:,:,0]).max(axis=0)

    axes[1].imshow(diff_global, cmap='hot', vmin=0, vmax=diff_global.max())
    axes[1].set_title(f'Residual Error (Global)\nMean: {diff_global.mean():.2f}')

    axes[2].imshow(diff_local, cmap='hot', vmin=0, vmax=diff_global.max())
    axes[2].set_title(f'Residual Error (Local)\nMean: {diff_local.mean():.2f}')

    plt.tight_layout()
    plt.show()

    print(f"Global registration mean error: {diff_global.mean():.2f}")
    print(f"Local registration mean error: {diff_local.mean():.2f}")
    print(f"Improvement: {(1 - diff_local.mean()/diff_global.mean())*100:.1f}%")

except ImportError as e:
    print(f"SimpleITK not installed: {e}")
    print("Install with: uv add 'starfinder[local-registration]'")
```

Add markdown cell:
```markdown
### 7.4 Visualize Displacement Field
```

Add code cell:
```python
try:
    # Field magnitude as heatmap (MIP across Z)
    field_magnitude = np.linalg.norm(estimated_field, axis=-1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # True field magnitude
    true_magnitude = np.linalg.norm(true_field, axis=-1)
    im0 = axes[0].imshow(true_magnitude.max(axis=0), cmap='hot')
    axes[0].set_title('True Displacement Magnitude (MIP)')
    plt.colorbar(im0, ax=axes[0], label='pixels')

    # Estimated field magnitude
    im1 = axes[1].imshow(field_magnitude.max(axis=0), cmap='hot')
    axes[1].set_title('Estimated Displacement Magnitude (MIP)')
    plt.colorbar(im1, ax=axes[1], label='pixels')

    plt.tight_layout()
    plt.show()

except NameError:
    print("Run previous cell first to compute estimated_field")
```

**Step 2: Verify notebook is valid JSON**

Run: `python -c "import json; json.load(open('tests/qc_registration.ipynb'))"`
Expected: No error

**Step 3: Commit**

```bash
git add tests/qc_registration.ipynb
git commit -m "docs(qc): add local registration section demonstrating global limitation"
```

---

## Task 11: Final Verification

**Step 1: Run all tests**

Run: `cd src/python && uv run pytest test/ -v`
Expected: All tests PASS

**Step 2: Run ruff linter**

Run: `cd src/python && uv run ruff check starfinder/registration/demons.py`
Expected: No errors

**Step 3: Update notes.md**

Add entry to `docs/notes.md` under Current Progress:

```markdown
### 2026-02-02: Demons Registration Module Implemented

- [x] **Implemented non-rigid registration module** (`starfinder.registration.demons`)
  - `demons_register(fixed, moving)` → displacement field using symmetric forces demons
  - `apply_deformation(volume, field)` → apply displacement field to warp volume
  - `register_volume_local(images, ref, mov)` → multi-channel convenience wrapper
  - SimpleITK as optional dependency (lazy import with helpful error)
  - Multi-resolution pyramid matching MATLAB's `imregdemons` behavior
  - 4 tests in `test/test_demons.py`

- [x] **Added QC notebook section** (Section 7 of `qc_registration.ipynb`)
  - Synthetic local deformation generator
  - Demonstration: global registration fails on local deformation
  - Demonstration: local registration succeeds
  - Displacement field visualization

**Files Created:**
- `src/python/starfinder/registration/demons.py` - Core implementation
- `src/python/test/test_demons.py` - Unit tests

**Files Modified:**
- `src/python/starfinder/registration/__init__.py` - Added exports
- `tests/qc_registration.ipynb` - Added Section 7
- `docs/notes.md` - This entry

**MATLAB Function Mapping:**
| MATLAB | Python |
|--------|--------|
| `RegisterImagesLocal(images, ref, mov, iter, afs)` | `register_volume_local(images, ref, mov, iterations, smoothing_sigma)` |
| `imregdemons(mov, ref, ...)` | `demons_register(fixed, moving, ...)` |
| `imwarp(img, field)` | `apply_deformation(volume, field)` |
```

**Step 4: Commit notes update**

```bash
git add docs/notes.md
git commit -m "docs: update notes with demons registration implementation"
```

---

## Summary

| Task | Description | Tests |
|------|-------------|-------|
| 1 | Create demons.py skeleton | - |
| 2-3 | Implement `demons_register` | `test_identity` |
| 4-5 | Implement `apply_deformation` | `test_identity_field` |
| 6-7 | Implement `register_volume_local` | `test_multichannel` |
| 8 | Add deformation recovery test | `test_known_deformation` |
| 9 | Update module exports | - |
| 10 | Add QC notebook section | - |
| 11 | Final verification | All tests |

**Total commits:** 11
**Total new tests:** 4
**New files:** 2 (`demons.py`, `test_demons.py`)
**Modified files:** 3 (`__init__.py`, `qc_registration.ipynb`, `notes.md`)
