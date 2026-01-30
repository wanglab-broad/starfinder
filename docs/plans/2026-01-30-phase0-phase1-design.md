# Phase 0 & Phase 1 Design: Directory Restructure & I/O Module

**Date:** 2026-01-30
**Status:** Approved

## Overview

This document describes the design for:
- **Phase 0:** Restructuring the repository to support dual MATLAB/Python backends
- **Phase 1:** Implementing the Python I/O module for TIFF image loading/saving

---

## Phase 0: Directory Restructure

### Goal
Reorganize the codebase from `code-base/` to `src/` for cleaner dual-backend support.

### Current Structure
```
starfinder/
├── code-base/
│   ├── src/           # 29 MATLAB files
│   └── matlab-addon/  # External MATLAB toolboxes
├── src/
│   └── python/        # Python package
└── workflow/
    └── scripts/       # MATLAB scripts with addpath references
```

### New Structure
```
starfinder/
├── src/
│   ├── matlab/        # Moved from code-base/src/
│   ├── matlab-addon/  # Moved from code-base/matlab-addon/
│   └── python/        # Unchanged
└── workflow/
    └── scripts/       # Updated addpath references
```

### Files to Update

**Workflow scripts** (change `code-base/src/` → `src/matlab/` and `code-base/matlab-addon/` → `src/matlab-addon/`):
- `workflow/scripts/rsf_single_fov.m`
- `workflow/scripts/gr_single_fov_subtile.m`
- `workflow/scripts/lrsf_single_fov_subtile.m`
- `workflow/scripts/deep_create_subtile.m`
- `workflow/scripts/deep_rsf_subtile.m`
- `workflow/scripts/nuclei_registration.m`
- `workflow/scripts/rsf_single_fov_seq.m`

**Documentation:**
- `CLAUDE.md` - Update directory structure section
- `AGENTS.md` - Update module organization section

---

## Phase 1: I/O Module

### Goal
Implement Python equivalents of MATLAB I/O functions using bioio as the primary backend.

### Dependencies

```toml
# pyproject.toml additions
dependencies = [
    "bioio>=1.0",
    "bioio-tifffile>=1.0",
]

[project.optional-dependencies]
ome = ["bioio-ome-tiff>=1.0"]
```

### File Structure

```
src/python/starfinder/
├── io/
│   ├── __init__.py      # Public exports
│   └── tiff.py          # TIFF I/O functions
└── ...
```

### Public API

```python
from pathlib import Path
import numpy as np

def load_multipage_tiff(
    path: Path | str,
    convert_uint8: bool = True,
) -> np.ndarray:
    """
    Load a multi-page TIFF file.

    Args:
        path: Path to TIFF file
        convert_uint8: If True (default), convert to uint8

    Returns:
        Array with shape (Z, Y, X)
    """
    ...

def load_image_stacks(
    round_dir: Path | str,
    channel_order: list[str],
    subdir: str = "",
    convert_uint8: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Load multiple channel TIFFs from a directory.

    Args:
        round_dir: Directory containing channel TIFF files
        channel_order: List of channel patterns, e.g., ["ch00", "ch01", "ch02", "ch03"]
        subdir: Optional subdirectory within round_dir
        convert_uint8: If True (default), convert to uint8

    Returns:
        Tuple of (image array with shape (Z, Y, X, C), metadata dict)

    Notes:
        - If channels have different sizes, crops to minimum and logs warning
        - Metadata includes: shape, dtype, original_shapes, cropped (bool)
    """
    ...

def save_stack(
    image: np.ndarray,
    path: Path | str,
    compress: bool = False,
) -> None:
    """
    Save a 3D or 4D array as a multi-page TIFF.

    Args:
        image: Array with shape (Z, Y, X) or (Z, Y, X, C)
        path: Output path
        compress: If True, use compression

    Notes:
        - Overwrites existing file if present
        - For 4D arrays, saves as ImageJ-compatible hyperstack
    """
    ...
```

### Design Decisions

1. **Axis ordering:** All functions use `(Z, Y, X, C)` convention (volumetric-first, channel-last)
   - Matches ITK/SimpleITK conventions
   - Compatible with scikit-image 3D functions
   - bioio's `get_image_data("ZYX")` handles reordering automatically

2. **Default to uint8:** `convert_uint8=True` by default
   - Matches the common use case in the pipeline
   - Reduces memory usage

3. **Channel discovery:** Explicit `channel_order` list
   - Pass patterns like `["ch00", "ch01", "ch02", "ch03"]`
   - Function finds files containing each pattern
   - Simple and matches MATLAB behavior

4. **Size mismatch handling:** Crop to minimum + log warning
   - Maintains MATLAB compatibility
   - Edge data that differs between channels is often unreliable
   - Warning alerts users to investigate if needed

5. **I/O backend:** bioio with bioio-tifffile
   - Modern, modular architecture
   - Consistent API across formats
   - OME-TIFF support available via optional dependency
   - Lazy loading capability for large images

### Implementation Notes

```python
# Example implementation sketch
from bioio import BioImage
from bioio_tifffile import Reader as TiffReader
import warnings

def load_multipage_tiff(path, convert_uint8=True):
    img = BioImage(path, reader=TiffReader)
    data = img.get_image_data("ZYX")  # bioio handles reordering
    if convert_uint8:
        data = _to_uint8(data)
    return data

def _to_uint8(data):
    """Convert array to uint8 with proper scaling."""
    if data.dtype == np.uint8:
        return data
    # Scale to 0-255 range
    data_min, data_max = data.min(), data.max()
    if data_max > data_min:
        scaled = (data - data_min) / (data_max - data_min) * 255
    else:
        scaled = np.zeros_like(data)
    return scaled.astype(np.uint8)
```

### MATLAB Function Mapping

| MATLAB | Python | Notes |
|--------|--------|-------|
| `LoadMultipageTiff(fname, convert_uint8)` | `load_multipage_tiff(path, convert_uint8)` | Returns (Z,Y,X) not (Y,X,Z) |
| `LoadImageStacks(round_dir, sub_dir, channel_order_dict, convert_uint8)` | `load_image_stacks(round_dir, channel_order, subdir, convert_uint8)` | Simplified channel_order API |
| `SaveSingleStack(input_img, filename)` | `save_stack(image, path, compress)` | Added compression option |

---

## Testing Plan

### Test File Structure
```
src/python/test/
├── test_io.py          # I/O module tests
└── conftest.py         # Shared fixtures
```

### Test Cases

**`load_multipage_tiff()`:**
- Load synthetic test TIFF → verify shape is `(Z, Y, X)`
- Load with `convert_uint8=True` → verify dtype is `uint8`
- Load with `convert_uint8=False` → verify original dtype preserved
- Load non-existent file → verify `FileNotFoundError`

**`load_image_stacks()`:**
- Load 4-channel stack → verify shape is `(Z, Y, X, 4)`
- Load with mismatched channel sizes → verify cropping + warning
- Missing channel file → verify `ValueError`
- Verify channel ordering matches `channel_order` parameter

**`save_stack()`:**
- Save 3D array → reload and verify roundtrip
- Save 4D array → reload and verify roundtrip
- Save with `compress=True` → verify file is smaller
- Overwrite existing file → verify no error

---

## Implementation Order

1. Phase 0: Directory restructure
   - Move directories
   - Update workflow scripts
   - Update documentation
   - Test with `snakemake -n` dry run

2. Phase 1: I/O module
   - Add bioio dependencies to pyproject.toml
   - Implement `starfinder/io/tiff.py`
   - Write tests in `test/test_io.py`
   - Run tests with `uv run pytest test/test_io.py`
