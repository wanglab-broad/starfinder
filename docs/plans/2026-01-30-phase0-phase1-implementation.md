# Phase 0 & Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure repository for dual MATLAB/Python backends and implement Python I/O module with bioio.

**Architecture:** Move MATLAB code from `code-base/` to `src/matlab/`, update all path references in workflow scripts, then implement `starfinder.io` module using bioio for TIFF loading/saving with `(Z, Y, X, C)` axis convention.

**Tech Stack:** Python 3.10+, bioio, bioio-tifffile, tifffile (for writing), pytest

---

## Phase 0: Directory Restructure

### Task 1: Move MATLAB source files

**Files:**
- Move: `code-base/src/*` → `src/matlab/`
- Move: `code-base/matlab-addon/*` → `src/matlab-addon/`
- Delete: `code-base/` (after moves complete)

**Step 1: Create new directories**

```bash
mkdir -p src/matlab
mkdir -p src/matlab-addon
```

**Step 2: Move MATLAB source files**

```bash
mv code-base/src/* src/matlab/
```

**Step 3: Move MATLAB addons**

```bash
mv code-base/matlab-addon/* src/matlab-addon/
```

**Step 4: Remove empty code-base directory**

```bash
rm -rf code-base/
```

**Step 5: Verify move**

```bash
ls src/matlab/*.m | head -5
ls src/matlab-addon/
```

Expected: See MATLAB files in new locations, `code-base/` no longer exists.

---

### Task 2: Update workflow script paths

**Files:**
- Modify: `workflow/scripts/rsf_single_fov.m:12-13`
- Modify: `workflow/scripts/gr_single_fov_subtile.m:12-13`
- Modify: `workflow/scripts/lrsf_single_fov_subtile.m:12-13`
- Modify: `workflow/scripts/deep_create_subtile.m:12-13`
- Modify: `workflow/scripts/deep_rsf_subtile.m:12-13`
- Modify: `workflow/scripts/nuclei_registration.m:12-13`
- Modify: `workflow/scripts/rsf_single_fov_seq.m:12-13`

**Step 1: Update all workflow scripts**

In each file, change:
```matlab
addpath(fullfile(config.starfinder_path, 'code-base/src/'))
addpath(genpath(fullfile(config.starfinder_path, 'code-base/matlab-addon/')))
```

To:
```matlab
addpath(fullfile(config.starfinder_path, 'src/matlab/'))
addpath(genpath(fullfile(config.starfinder_path, 'src/matlab-addon/')))
```

**Step 2: Verify no remaining references**

```bash
grep -r "code-base" workflow/
```

Expected: No matches (or only in test/ files that can be updated separately).

---

### Task 3: Update documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `AGENTS.md`

**Step 1: Update CLAUDE.md directory structure section**

Change:
```markdown
- `code-base/src/` - Core MATLAB image processing scripts (~27 files). Main orchestrator is `STARMapDataset.m`.
```

To:
```markdown
- `src/matlab/` - Core MATLAB image processing scripts (~29 files). Main orchestrator is `STARMapDataset.m`.
- `src/matlab-addon/` - External MATLAB toolboxes (TIFF handling, natural sort, etc.)
- `src/python/` - Python package with I/O, registration, and processing modules.
```

**Step 2: Update AGENTS.md module organization section**

Change:
```markdown
- `code-base/src/`: core MATLAB functions used by image processing steps.
```

To:
```markdown
- `src/matlab/`: core MATLAB functions used by image processing steps.
- `src/matlab-addon/`: external MATLAB toolboxes.
- `src/python/`: Python package (`starfinder`) for I/O and processing.
```

---

### Task 4: Commit Phase 0

**Step 1: Stage all changes**

```bash
git add -A
```

**Step 2: Verify staged changes**

```bash
git status
```

Expected: See moved files, modified workflow scripts, modified docs.

**Step 3: Commit**

```bash
git commit -m "57. restructure repository: move code-base to src/matlab

- Move code-base/src/ to src/matlab/
- Move code-base/matlab-addon/ to src/matlab-addon/
- Update all workflow script addpath references
- Update CLAUDE.md and AGENTS.md documentation

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 1: I/O Module

### Task 5: Add bioio dependencies

**Files:**
- Modify: `src/python/pyproject.toml`

**Step 1: Update pyproject.toml dependencies**

Add to the `dependencies` list:
```toml
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "scikit-image>=0.21",
    "tifffile>=2023.0",
    "pandas>=2.0",
    "h5py>=3.9",
    "matplotlib>=3.7",
    "bioio>=1.0",
    "bioio-tifffile>=1.0",
]
```

Add new optional dependency group:
```toml
[project.optional-dependencies]
ome = ["bioio-ome-tiff>=1.0"]
local-registration = ["SimpleITK>=2.3"]
spatialdata = ["spatialdata>=0.1", "spatialdata-io>=0.1"]
dev = ["pytest>=7.0", "pytest-cov>=4.0", "ruff>=0.1"]
```

**Step 2: Update lock file**

```bash
cd src/python && uv sync
```

Expected: Lock file updated, bioio packages installed.

---

### Task 6: Create I/O module structure

**Files:**
- Create: `src/python/starfinder/io/__init__.py`
- Create: `src/python/starfinder/io/tiff.py`

**Step 1: Create io package directory**

```bash
mkdir -p src/python/starfinder/io
```

**Step 2: Create __init__.py with public exports**

```python
"""I/O utilities for loading and saving image data."""

from starfinder.io.tiff import (
    load_multipage_tiff,
    load_image_stacks,
    save_stack,
)

__all__ = [
    "load_multipage_tiff",
    "load_image_stacks",
    "save_stack",
]
```

**Step 3: Create tiff.py with function stubs**

```python
"""TIFF image I/O using bioio backend."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def load_multipage_tiff(
    path: Path | str,
    convert_uint8: bool = True,
) -> np.ndarray:
    """
    Load a multi-page TIFF file.

    Args:
        path: Path to TIFF file.
        convert_uint8: If True (default), convert to uint8.

    Returns:
        Array with shape (Z, Y, X).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    raise NotImplementedError("TODO: implement")


def load_image_stacks(
    round_dir: Path | str,
    channel_order: list[str],
    subdir: str = "",
    convert_uint8: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Load multiple channel TIFFs from a directory.

    Args:
        round_dir: Directory containing channel TIFF files.
        channel_order: List of channel patterns, e.g., ["ch00", "ch01"].
        subdir: Optional subdirectory within round_dir.
        convert_uint8: If True (default), convert to uint8.

    Returns:
        Tuple of (image array with shape (Z, Y, X, C), metadata dict).

    Raises:
        FileNotFoundError: If directory does not exist.
        ValueError: If no files match a channel pattern.

    Notes:
        If channels have different sizes, crops to minimum and logs warning.
        Metadata includes: shape, dtype, original_shapes, cropped (bool).
    """
    raise NotImplementedError("TODO: implement")


def save_stack(
    image: np.ndarray,
    path: Path | str,
    compress: bool = False,
) -> None:
    """
    Save a 3D or 4D array as a multi-page TIFF.

    Args:
        image: Array with shape (Z, Y, X) or (Z, Y, X, C).
        path: Output path.
        compress: If True, use compression.

    Notes:
        Overwrites existing file if present.
    """
    raise NotImplementedError("TODO: implement")
```

---

### Task 7: Write test for load_multipage_tiff

**Files:**
- Create: `src/python/test/test_io.py`

**Step 1: Write the failing test**

```python
"""Tests for starfinder.io module."""

import numpy as np
import pytest
import tifffile
from pathlib import Path

from starfinder.io import load_multipage_tiff


class TestLoadMultipageTiff:
    """Tests for load_multipage_tiff function."""

    def test_load_returns_zyx_shape(self, tmp_path: Path):
        """Loading a multi-page TIFF returns (Z, Y, X) array."""
        # Create test TIFF: 5 slices, 64x32 pixels
        test_data = np.random.randint(0, 255, (5, 64, 32), dtype=np.uint8)
        tiff_path = tmp_path / "test.tif"
        tifffile.imwrite(tiff_path, test_data)

        result = load_multipage_tiff(tiff_path)

        assert result.shape == (5, 64, 32)

    def test_load_converts_to_uint8_by_default(self, tmp_path: Path):
        """Loading converts to uint8 by default."""
        test_data = np.random.randint(0, 65535, (3, 32, 32), dtype=np.uint16)
        tiff_path = tmp_path / "test16.tif"
        tifffile.imwrite(tiff_path, test_data)

        result = load_multipage_tiff(tiff_path)

        assert result.dtype == np.uint8

    def test_load_preserves_dtype_when_convert_false(self, tmp_path: Path):
        """Loading preserves original dtype when convert_uint8=False."""
        test_data = np.random.randint(0, 65535, (3, 32, 32), dtype=np.uint16)
        tiff_path = tmp_path / "test16.tif"
        tifffile.imwrite(tiff_path, test_data)

        result = load_multipage_tiff(tiff_path, convert_uint8=False)

        assert result.dtype == np.uint16

    def test_load_nonexistent_file_raises(self):
        """Loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_multipage_tiff("/nonexistent/path.tif")
```

**Step 2: Run test to verify it fails**

```bash
cd src/python && uv run pytest test/test_io.py -v
```

Expected: FAIL with `NotImplementedError: TODO: implement`

---

### Task 8: Implement load_multipage_tiff

**Files:**
- Modify: `src/python/starfinder/io/tiff.py`

**Step 1: Implement the function**

Replace the `load_multipage_tiff` function with:

```python
def load_multipage_tiff(
    path: Path | str,
    convert_uint8: bool = True,
) -> np.ndarray:
    """
    Load a multi-page TIFF file.

    Args:
        path: Path to TIFF file.
        convert_uint8: If True (default), convert to uint8.

    Returns:
        Array with shape (Z, Y, X).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    from bioio import BioImage
    from bioio_tifffile import Reader as TiffReader

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TIFF file not found: {path}")

    img = BioImage(path, reader=TiffReader)
    # BioImage returns TCZYX by default, extract ZYX
    data = img.get_image_data("ZYX", T=0, C=0)

    if convert_uint8:
        data = _to_uint8(data)

    return data


def _to_uint8(data: np.ndarray) -> np.ndarray:
    """Convert array to uint8 with proper scaling."""
    if data.dtype == np.uint8:
        return data

    # Get min/max for scaling
    data_min = float(data.min())
    data_max = float(data.max())

    if data_max > data_min:
        # Scale to 0-255 range
        scaled = (data.astype(np.float32) - data_min) / (data_max - data_min) * 255.0
        return scaled.astype(np.uint8)
    else:
        # Constant image
        return np.zeros(data.shape, dtype=np.uint8)
```

**Step 2: Run test to verify it passes**

```bash
cd src/python && uv run pytest test/test_io.py::TestLoadMultipageTiff -v
```

Expected: All 4 tests PASS.

---

### Task 9: Write test for save_stack

**Files:**
- Modify: `src/python/test/test_io.py`

**Step 1: Add test class for save_stack**

```python
from starfinder.io import load_multipage_tiff, save_stack


class TestSaveStack:
    """Tests for save_stack function."""

    def test_save_3d_roundtrip(self, tmp_path: Path):
        """Saving and reloading 3D array preserves data."""
        original = np.random.randint(0, 255, (5, 64, 32), dtype=np.uint8)
        tiff_path = tmp_path / "output.tif"

        save_stack(original, tiff_path)
        result = load_multipage_tiff(tiff_path, convert_uint8=False)

        np.testing.assert_array_equal(result, original)

    def test_save_overwrites_existing(self, tmp_path: Path):
        """Saving overwrites existing file."""
        tiff_path = tmp_path / "output.tif"

        # Write first file
        data1 = np.zeros((3, 32, 32), dtype=np.uint8)
        save_stack(data1, tiff_path)

        # Overwrite with different data
        data2 = np.ones((5, 64, 64), dtype=np.uint8) * 255
        save_stack(data2, tiff_path)

        result = load_multipage_tiff(tiff_path, convert_uint8=False)
        assert result.shape == (5, 64, 64)

    def test_save_with_compression(self, tmp_path: Path):
        """Saving with compression creates smaller file."""
        data = np.random.randint(0, 255, (10, 128, 128), dtype=np.uint8)
        path_uncompressed = tmp_path / "uncompressed.tif"
        path_compressed = tmp_path / "compressed.tif"

        save_stack(data, path_uncompressed, compress=False)
        save_stack(data, path_compressed, compress=True)

        size_uncompressed = path_uncompressed.stat().st_size
        size_compressed = path_compressed.stat().st_size

        # Compressed should be smaller (for random data, at least not larger)
        assert size_compressed <= size_uncompressed
```

**Step 2: Run test to verify it fails**

```bash
cd src/python && uv run pytest test/test_io.py::TestSaveStack -v
```

Expected: FAIL with `NotImplementedError`

---

### Task 10: Implement save_stack

**Files:**
- Modify: `src/python/starfinder/io/tiff.py`

**Step 1: Add import at top of file**

```python
import tifffile
```

**Step 2: Implement the function**

Replace `save_stack` with:

```python
def save_stack(
    image: np.ndarray,
    path: Path | str,
    compress: bool = False,
) -> None:
    """
    Save a 3D or 4D array as a multi-page TIFF.

    Args:
        image: Array with shape (Z, Y, X) or (Z, Y, X, C).
        path: Output path.
        compress: If True, use compression.

    Notes:
        Overwrites existing file if present.
    """
    path = Path(path)

    # Remove existing file if present
    if path.exists():
        path.unlink()

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    compression = "zlib" if compress else None
    tifffile.imwrite(path, image, compression=compression)
```

**Step 3: Run test to verify it passes**

```bash
cd src/python && uv run pytest test/test_io.py::TestSaveStack -v
```

Expected: All 3 tests PASS.

---

### Task 11: Write test for load_image_stacks

**Files:**
- Modify: `src/python/test/test_io.py`

**Step 1: Add test class for load_image_stacks**

```python
from starfinder.io import load_multipage_tiff, load_image_stacks, save_stack


class TestLoadImageStacks:
    """Tests for load_image_stacks function."""

    def test_load_returns_zyxc_shape(self, tmp_path: Path):
        """Loading multiple channels returns (Z, Y, X, C) array."""
        # Create 4 channel files
        for i, ch in enumerate(["ch00", "ch01", "ch02", "ch03"]):
            data = np.full((5, 64, 32), i * 50, dtype=np.uint8)
            tifffile.imwrite(tmp_path / f"img_{ch}.tif", data)

        result, metadata = load_image_stacks(
            tmp_path, ["ch00", "ch01", "ch02", "ch03"]
        )

        assert result.shape == (5, 64, 32, 4)
        assert result.dtype == np.uint8

    def test_load_respects_channel_order(self, tmp_path: Path):
        """Channels are stacked in the order specified."""
        # ch00 = all 0s, ch01 = all 100s
        tifffile.imwrite(tmp_path / "img_ch00.tif", np.zeros((3, 32, 32), dtype=np.uint8))
        tifffile.imwrite(tmp_path / "img_ch01.tif", np.full((3, 32, 32), 100, dtype=np.uint8))

        result, _ = load_image_stacks(tmp_path, ["ch00", "ch01"])

        assert result[0, 0, 0, 0] == 0    # ch00 is first
        assert result[0, 0, 0, 1] == 100  # ch01 is second

    def test_load_with_size_mismatch_crops_and_warns(self, tmp_path: Path):
        """Size mismatch between channels crops to minimum and warns."""
        # ch00: 5x64x32, ch01: 5x60x30
        tifffile.imwrite(tmp_path / "ch00.tif", np.zeros((5, 64, 32), dtype=np.uint8))
        tifffile.imwrite(tmp_path / "ch01.tif", np.zeros((5, 60, 30), dtype=np.uint8))

        with pytest.warns(UserWarning, match="size mismatch"):
            result, metadata = load_image_stacks(tmp_path, ["ch00", "ch01"])

        assert result.shape == (5, 60, 30, 2)  # Cropped to minimum
        assert metadata["cropped"] is True

    def test_load_missing_channel_raises(self, tmp_path: Path):
        """Missing channel file raises ValueError."""
        tifffile.imwrite(tmp_path / "ch00.tif", np.zeros((3, 32, 32), dtype=np.uint8))

        with pytest.raises(ValueError, match="ch01"):
            load_image_stacks(tmp_path, ["ch00", "ch01"])

    def test_load_nonexistent_dir_raises(self):
        """Non-existent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_image_stacks("/nonexistent/dir", ["ch00"])

    def test_load_with_subdir(self, tmp_path: Path):
        """Loading with subdir searches in subdirectory."""
        subdir = tmp_path / "images"
        subdir.mkdir()
        tifffile.imwrite(subdir / "ch00.tif", np.zeros((3, 32, 32), dtype=np.uint8))

        result, _ = load_image_stacks(tmp_path, ["ch00"], subdir="images")

        assert result.shape == (3, 32, 32, 1)
```

**Step 2: Run test to verify it fails**

```bash
cd src/python && uv run pytest test/test_io.py::TestLoadImageStacks -v
```

Expected: FAIL with `NotImplementedError`

---

### Task 12: Implement load_image_stacks

**Files:**
- Modify: `src/python/starfinder/io/tiff.py`

**Step 1: Implement the function**

Replace `load_image_stacks` with:

```python
def load_image_stacks(
    round_dir: Path | str,
    channel_order: list[str],
    subdir: str = "",
    convert_uint8: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Load multiple channel TIFFs from a directory.

    Args:
        round_dir: Directory containing channel TIFF files.
        channel_order: List of channel patterns, e.g., ["ch00", "ch01"].
        subdir: Optional subdirectory within round_dir.
        convert_uint8: If True (default), convert to uint8.

    Returns:
        Tuple of (image array with shape (Z, Y, X, C), metadata dict).

    Raises:
        FileNotFoundError: If directory does not exist.
        ValueError: If no files match a channel pattern.

    Notes:
        If channels have different sizes, crops to minimum and logs warning.
        Metadata includes: shape, dtype, original_shapes, cropped (bool).
    """
    round_dir = Path(round_dir)
    search_dir = round_dir / subdir if subdir else round_dir

    if not search_dir.exists():
        raise FileNotFoundError(f"Directory not found: {search_dir}")

    # Find and load each channel
    channel_images: list[np.ndarray] = []
    original_shapes: list[tuple[int, ...]] = []

    for channel in channel_order:
        # Find file matching channel pattern
        matches = list(search_dir.glob(f"*{channel}*.tif"))
        if not matches:
            raise ValueError(f"No TIFF file found matching channel pattern: {channel}")
        if len(matches) > 1:
            logger.warning(f"Multiple files match '{channel}', using first: {matches[0]}")

        # Load the channel (without uint8 conversion yet)
        img = load_multipage_tiff(matches[0], convert_uint8=False)
        channel_images.append(img)
        original_shapes.append(img.shape)

    # Find minimum dimensions
    min_z = min(img.shape[0] for img in channel_images)
    min_y = min(img.shape[1] for img in channel_images)
    min_x = min(img.shape[2] for img in channel_images)

    # Check for size mismatch
    cropped = False
    for i, (img, shape) in enumerate(zip(channel_images, original_shapes)):
        if shape != (min_z, min_y, min_x):
            cropped = True
            break

    if cropped:
        warnings.warn(
            f"Channel size mismatch detected. Cropping to minimum dimensions "
            f"({min_z}, {min_y}, {min_x}). Original shapes: {original_shapes}",
            UserWarning,
        )

    # Crop all channels to minimum size and stack
    cropped_images = [img[:min_z, :min_y, :min_x] for img in channel_images]
    stacked = np.stack(cropped_images, axis=-1)  # (Z, Y, X, C)

    # Convert to uint8 if requested
    if convert_uint8:
        stacked = _to_uint8(stacked)

    metadata = {
        "shape": stacked.shape,
        "dtype": str(stacked.dtype),
        "original_shapes": original_shapes,
        "cropped": cropped,
    }

    return stacked, metadata
```

**Step 2: Run all tests to verify they pass**

```bash
cd src/python && uv run pytest test/test_io.py -v
```

Expected: All tests PASS.

---

### Task 13: Update starfinder package exports

**Files:**
- Modify: `src/python/starfinder/__init__.py`

**Step 1: Add io module to package exports**

```python
"""STARfinder: Spatial transcriptomics data processing pipeline."""

from starfinder.io import load_multipage_tiff, load_image_stacks, save_stack

__version__ = "0.1.0"

__all__ = [
    "load_multipage_tiff",
    "load_image_stacks",
    "save_stack",
    "__version__",
]
```

**Step 2: Verify import works**

```bash
cd src/python && uv run python -c "from starfinder import load_multipage_tiff; print('OK')"
```

Expected: Prints `OK`.

---

### Task 14: Commit Phase 1

**Step 1: Run all tests**

```bash
cd src/python && uv run pytest test/ -v
```

Expected: All tests pass.

**Step 2: Stage and commit**

```bash
git add -A
git commit -m "58. implement I/O module with bioio backend

- Add bioio and bioio-tifffile dependencies
- Implement load_multipage_tiff with (Z, Y, X) output
- Implement load_image_stacks with (Z, Y, X, C) output
- Implement save_stack for TIFF writing
- Add comprehensive tests for all I/O functions
- Default to uint8 conversion for memory efficiency

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Verification

### Final verification steps

**Step 1: Run full test suite**

```bash
cd src/python && uv run pytest test/ -v --cov=starfinder
```

**Step 2: Verify Snakemake dry run still works**

```bash
snakemake -s workflow/Snakefile --configfile test/tissue_2D_test.yaml -n
```

Expected: Dry run completes without path errors.

**Step 3: Push to remote**

```bash
git push origin dev
```
