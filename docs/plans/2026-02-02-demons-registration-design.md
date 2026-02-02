# Demons Registration Module Design

**Date:** 2026-02-02
**Status:** Approved

## Overview

Add non-rigid (deformable) registration to the Python backend using SimpleITK's symmetric forces demons algorithm. This module refines alignment after global phase correlation, correcting residual tissue warping between sequencing rounds.

## Scope

- **In scope:** 3D demons registration using SimpleITK, matching MATLAB `RegisterImagesLocal.m` behavior
- **Out of scope:** GPU acceleration, deep learning approaches (documented as future options)

## Module Structure

```
src/python/starfinder/registration/
├── __init__.py              # Add new exports
├── phase_correlation.py     # Existing (global/rigid)
├── _skimage_backend.py      # Existing (comparison)
└── demons.py                # NEW (local/non-rigid)
```

## Algorithm Choice

**Primary:** `SymmetricForcesDemonsRegistrationFilter` from SimpleITK

**Why symmetric forces:**
- Handles intensity variations between sequencing rounds
- Robust for biomedical images
- Matches MATLAB `imregdemons` internal behavior (diffeomorphic symmetric forces)

**Future options (for potential upgrade):**

| Algorithm | When to Consider |
|-----------|------------------|
| `DemonsRegistrationFilter` | Need faster execution, uniform intensity across rounds |
| `FastSymmetricForcesDemonsRegistrationFilter` | Have GPU-enabled ITK build |
| VoxelMorph (DL-based) | Want learned priors, even faster inference |

## Public API

Three functions matching the `phase_correlation.py` pattern:

```python
def demons_register(
    fixed: np.ndarray,           # (Z, Y, X) reference volume
    moving: np.ndarray,          # (Z, Y, X) volume to align
    iterations: list[int] = [100, 50, 25],  # per pyramid level
    smoothing_sigma: float = 1.0,           # regularization
) -> np.ndarray:
    """
    Compute displacement field to align moving to fixed using demons.

    Returns:
        Displacement field with shape (Z, Y, X, 3) where last dim is (dz, dy, dx).
    """

def apply_deformation(
    volume: np.ndarray,          # (Z, Y, X) volume to warp
    displacement_field: np.ndarray,  # (Z, Y, X, 3) from demons_register
) -> np.ndarray:
    """Apply displacement field to volume. Returns warped volume."""

def register_volume_local(
    images: np.ndarray,          # (Z, Y, X, C) multi-channel volume
    ref_image: np.ndarray,       # (Z, Y, X) reference for field calculation
    mov_image: np.ndarray,       # (Z, Y, X) moving for field calculation
    iterations: list[int] = [100, 50, 25],
    smoothing_sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Register all channels using demons.

    Returns:
        Tuple of (registered_images, displacement_field).
    """
```

**Key design decisions:**
- Displacement field exposed for QC visualization and reuse
- Parameters match MATLAB `imregdemons` behavior
- `register_volume_local` mirrors `register_volume` from global module

## Implementation Details

### SimpleITK Usage

```python
def demons_register(fixed, moving, iterations=[100, 50, 25], smoothing_sigma=1.0):
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ImportError(
            "SimpleITK required for local registration. "
            "Install with: uv add 'starfinder[local-registration]'"
        )

    # Convert numpy (Z, Y, X) to SimpleITK image
    fixed_sitk = sitk.GetImageFromArray(fixed.astype(np.float32))
    moving_sitk = sitk.GetImageFromArray(moving.astype(np.float32))

    # Configure symmetric forces demons
    demons = sitk.SymmetricForcesDemonsRegistrationFilter()
    demons.SetNumberOfIterations(iterations[0])  # Coarsest level
    demons.SetStandardDeviations(smoothing_sigma)

    # Multi-resolution (pyramid)
    demons.SetSmoothDisplacementField(True)
    demons.SetSmoothUpdateField(False)

    # Run registration
    displacement_sitk = demons.Execute(fixed_sitk, moving_sitk)

    # Convert back to numpy (Z, Y, X, 3)
    displacement = sitk.GetArrayFromImage(displacement_sitk)

    return displacement
```

### Multi-Resolution Handling

- MATLAB uses `PyramidLevels` auto-calculated from Z dimension: `floor(log2(dimZ))`
- SimpleITK: Use `MultiResolutionPDEDeformableRegistration` wrapper or manual shrink factors
- Match MATLAB's pyramid depth logic for consistency

### Dependency Handling

- SimpleITK remains optional dependency: `uv add 'starfinder[local-registration]'`
- Lazy import inside functions
- Clear error message with install instructions if missing

## Testing Strategy

4 tests in `test/test_demons.py`, mirroring `test_registration.py` pattern:

```python
import pytest

# Skip all tests if SimpleITK not installed
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("SimpleITK", reason="SimpleITK not installed"),
    reason="SimpleITK required"
)

class TestDemonsRegister:
    def test_identity(self, mini_dataset):
        """Identical images produce near-zero displacement field."""
        vol = load_multipage_tiff(mini_dataset / "FOV_001" / "round1" / "ch00.tif")
        field = demons_register(vol, vol)

        assert field.shape == (*vol.shape, 3)
        assert np.allclose(field, 0, atol=0.1)

    def test_known_deformation(self, mini_dataset):
        """Recovers synthetic deformation applied via scipy.ndimage.map_coordinates."""
        # Apply small smooth deformation, verify field direction matches
        ...

class TestApplyDeformation:
    def test_roundtrip(self, mini_dataset):
        """Deform → apply inverse → preserves data."""
        ...

class TestRegisterVolumeLocal:
    def test_multichannel(self, mini_dataset):
        """Registers all channels using computed field."""
        images, _ = load_image_stacks(...)
        registered, field = register_volume_local(images, ref, mov)

        assert registered.shape == images.shape
        assert field.shape == (*images.shape[:3], 3)
```

**Test considerations:**
- Uses existing `mini_dataset` fixture (256×256×5)
- Skip tests if SimpleITK not installed via `pytest.importorskip`
- Tolerances looser than global registration (deformable is approximate)

## Module Integration

### Updated `registration/__init__.py`

```python
"""Registration module for image alignment."""

from starfinder.registration.phase_correlation import (
    phase_correlate,
    apply_shift,
    register_volume,
)
from starfinder.registration._skimage_backend import phase_correlate_skimage

# Local registration (lazy import - SimpleITK optional)
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

## QC Notebook Addition

Add Section 7 to `tests/qc_registration.ipynb`:

### 7.1 Create Synthetic Local Deformation

```python
def create_local_deformation(volume, strength=5.0, seed=42):
    """Apply smooth spatially-varying deformation (simulates tissue warping)."""
    from scipy.ndimage import map_coordinates, gaussian_filter

    rng = np.random.default_rng(seed)

    # Generate smooth random displacement field
    field = rng.standard_normal((3, *volume.shape)) * strength
    field = gaussian_filter(field, sigma=[0, 10, 10, 10])  # Smooth spatially

    # Create sampling coordinates
    coords = np.meshgrid(*[np.arange(s) for s in volume.shape], indexing='ij')
    warped_coords = [c + f for c, f in zip(coords, field)]

    return map_coordinates(volume, warped_coords, order=1), field
```

### 7.2 Demonstrate Global Registration Limitation

```python
# Apply local deformation to reference
deformed, true_field = create_local_deformation(ref_volume, strength=8.0)

# Global registration - can only correct bulk translation
global_registered, shifts = register_volume(deformed[..., None], ref_volume, deformed)

# Visualize: residual misalignment visible (non-uniform across FOV)
# Green/magenta composite shows color fringes in deformed regions
```

### 7.3 Demonstrate Local Registration Success

```python
# Local registration - corrects spatially-varying deformation
local_registered, estimated_field = register_volume_local(
    deformed[..., None], ref_volume, deformed
)

# Visualize: good alignment everywhere
# Green/magenta composite shows white/gray (aligned)
```

### 7.4 Visualize Displacement Field

```python
# Field magnitude as heatmap (MIP across Z)
field_magnitude = np.linalg.norm(estimated_field, axis=-1)
plt.imshow(field_magnitude.max(axis=0), cmap='hot')
plt.colorbar(label='Displacement (pixels)')
plt.title('Estimated Displacement Field Magnitude (MIP)')
```

## MATLAB Function Mapping

| MATLAB | Python | Notes |
|--------|--------|-------|
| `RegisterImagesLocal(input_img, ref_img, mov_img, iterations, afs)` | `register_volume_local(images, ref, mov, iterations, smoothing_sigma)` | Main entry point |
| `imregdemons(mov, ref, ...)` | `demons_register(fixed, moving, ...)` | Core algorithm |
| `imwarp(img, params)` | `apply_deformation(volume, field)` | Apply displacement |

## Files to Create/Modify

**New files:**
- `src/python/starfinder/registration/demons.py` - Core implementation
- `src/python/test/test_demons.py` - Unit tests

**Modified files:**
- `src/python/starfinder/registration/__init__.py` - Add exports
- `tests/qc_registration.ipynb` - Add Section 7

## Verification Commands

```bash
cd src/python

# Install optional dependency
uv add 'starfinder[local-registration]'

# Run tests
uv run pytest test/test_demons.py -v

# Run all registration tests
uv run pytest test/test_registration.py test/test_demons.py -v
```
