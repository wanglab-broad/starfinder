# Registration Module Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement DFT-based phase correlation registration with both NumPy/SciPy and scikit-image backends, including benchmarking.

**Architecture:** Pure functions for registration (`phase_correlate`, `apply_shift`, `register_volume`). Two backends for comparison. Benchmark runner using synthetic test images.

**Tech Stack:** NumPy, SciPy (fft), scikit-image (phase_cross_correlation), tracemalloc (memory profiling)

---

## Task 1: Create registration module structure

**Files:**
- Create: `src/python/starfinder/registration/__init__.py`
- Create: `src/python/starfinder/registration/phase_correlation.py`

**Step 1: Create registration package directory**

```bash
mkdir -p src/python/starfinder/registration
```

**Step 2: Create __init__.py with public exports**

```python
"""Registration module for image alignment."""

from starfinder.registration.phase_correlation import (
    phase_correlate,
    apply_shift,
    register_volume,
)

__all__ = [
    "phase_correlate",
    "apply_shift",
    "register_volume",
]
```

**Step 3: Create phase_correlation.py with stubs**

```python
"""DFT-based phase correlation registration using NumPy/SciPy."""

from __future__ import annotations

import numpy as np


def phase_correlate(
    fixed: np.ndarray,
    moving: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute shift to align moving image to fixed using phase correlation.

    Args:
        fixed: Reference volume with shape (Z, Y, X).
        moving: Volume to align with shape (Z, Y, X).

    Returns:
        Tuple of (dz, dy, dx) shift values.
    """
    raise NotImplementedError("TODO: implement")


def apply_shift(
    volume: np.ndarray,
    shift: tuple[float, float, float],
) -> np.ndarray:
    """
    Apply shift to volume and zero out wrapped regions.

    Args:
        volume: Input volume with shape (Z, Y, X).
        shift: Tuple of (dz, dy, dx) shift values.

    Returns:
        Shifted volume with same shape.
    """
    raise NotImplementedError("TODO: implement")


def register_volume(
    images: np.ndarray,
    ref_image: np.ndarray,
    mov_image: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Register multi-channel volume using phase correlation.

    Args:
        images: Multi-channel volume with shape (Z, Y, X, C).
        ref_image: Reference image with shape (Z, Y, X) for shift calculation.
        mov_image: Moving image with shape (Z, Y, X) for shift calculation.

    Returns:
        Tuple of (registered_images, shifts).
    """
    raise NotImplementedError("TODO: implement")
```

**Step 4: Verify module imports**

```bash
cd src/python && uv run python -c "from starfinder.registration import phase_correlate; print('OK')"
```

Expected: Prints `OK`

---

## Task 2: Write test for phase_correlate (zero shift)

**Files:**
- Create: `src/python/test/test_registration.py`

**Step 1: Write the failing test**

```python
"""Tests for starfinder.registration module."""

import numpy as np
import pytest

from starfinder.registration import phase_correlate


class TestPhaseCorrelate:
    """Tests for phase_correlate function."""

    def test_zero_shift(self, mini_dataset):
        """Identical images return (0, 0, 0)."""
        from starfinder.io import load_multipage_tiff

        vol = load_multipage_tiff(mini_dataset / "FOV_001" / "round1" / "ch00.tif")
        shift = phase_correlate(vol, vol)

        assert np.allclose(shift, (0, 0, 0), atol=0.1)
```

**Step 2: Run test to verify it fails**

```bash
cd src/python && uv run pytest test/test_registration.py::TestPhaseCorrelate::test_zero_shift -v
```

Expected: FAIL with `NotImplementedError: TODO: implement`

---

## Task 3: Implement phase_correlate

**Files:**
- Modify: `src/python/starfinder/registration/phase_correlation.py`

**Step 1: Implement the function**

Replace `phase_correlate` with:

```python
def phase_correlate(
    fixed: np.ndarray,
    moving: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute shift to align moving image to fixed using phase correlation.

    Args:
        fixed: Reference volume with shape (Z, Y, X).
        moving: Volume to align with shape (Z, Y, X).

    Returns:
        Tuple of (dz, dy, dx) shift values.
    """
    from scipy.fft import fftn, ifftn

    nz, ny, nx = moving.shape

    # Cross-correlation in frequency domain
    fixed_fft = fftn(fixed)
    moving_fft = fftn(moving)
    cc = ifftn(fixed_fft * np.conj(moving_fft))

    # Find peak
    peak_idx = np.argmax(np.abs(cc))
    iz, iy, ix = np.unravel_index(peak_idx, cc.shape)

    # Convert to signed shifts (handle wrap-around)
    dz = float(iz if iz < nz // 2 else iz - nz)
    dy = float(iy if iy < ny // 2 else iy - ny)
    dx = float(ix if ix < nx // 2 else ix - nx)

    return (dz, dy, dx)
```

**Step 2: Run test to verify it passes**

```bash
cd src/python && uv run pytest test/test_registration.py::TestPhaseCorrelate::test_zero_shift -v
```

Expected: PASS

---

## Task 4: Write and pass test for known shift

**Files:**
- Modify: `src/python/test/test_registration.py`

**Step 1: Add the test**

Add to `TestPhaseCorrelate` class:

```python
    def test_known_shift(self, mini_dataset):
        """Recovers integer shift applied via np.roll."""
        from starfinder.io import load_multipage_tiff

        vol = load_multipage_tiff(mini_dataset / "FOV_001" / "round1" / "ch00.tif")
        moved = np.roll(vol, (2, -3, 5), axis=(0, 1, 2))
        shift = phase_correlate(vol, moved)

        assert np.allclose(shift, (2, -3, 5), atol=0.5)
```

**Step 2: Run test**

```bash
cd src/python && uv run pytest test/test_registration.py::TestPhaseCorrelate::test_known_shift -v
```

Expected: PASS

---

## Task 5: Write test for apply_shift

**Files:**
- Modify: `src/python/test/test_registration.py`

**Step 1: Add the test class**

```python
from starfinder.registration import phase_correlate, apply_shift


class TestApplyShift:
    """Tests for apply_shift function."""

    def test_roundtrip(self, mini_dataset):
        """shift -> apply -> inverse shift preserves non-zero data."""
        from starfinder.io import load_multipage_tiff

        vol = load_multipage_tiff(mini_dataset / "FOV_001" / "round1" / "ch00.tif")
        original_sum = vol.sum()

        shifted = apply_shift(vol, (3, -2, 4))
        restored = apply_shift(shifted, (-3, 2, -4))

        # Restored should have some data (not all zeroed out)
        assert restored.sum() > 0
        # Shape preserved
        assert restored.shape == vol.shape
```

**Step 2: Run test to verify it fails**

```bash
cd src/python && uv run pytest test/test_registration.py::TestApplyShift::test_roundtrip -v
```

Expected: FAIL with `NotImplementedError`

---

## Task 6: Implement apply_shift

**Files:**
- Modify: `src/python/starfinder/registration/phase_correlation.py`

**Step 1: Implement the function**

Replace `apply_shift` with:

```python
def apply_shift(
    volume: np.ndarray,
    shift: tuple[float, float, float],
) -> np.ndarray:
    """
    Apply shift to volume and zero out wrapped regions.

    Args:
        volume: Input volume with shape (Z, Y, X).
        shift: Tuple of (dz, dy, dx) shift values.

    Returns:
        Shifted volume with same shape.
    """
    from scipy.fft import fftn, ifftn
    from scipy.ndimage import fourier_shift

    # Apply shift in frequency domain
    shifted_fft = fourier_shift(fftn(volume), shift)
    result = np.abs(ifftn(shifted_fft))

    # Zero out wrapped regions
    nz, ny, nx = volume.shape
    dz, dy, dx = shift

    if dz >= 0:
        result[: int(np.ceil(dz)), :, :] = 0
    else:
        result[int(nz + np.floor(dz) + 1) :, :, :] = 0

    if dy >= 0:
        result[:, : int(np.ceil(dy)), :] = 0
    else:
        result[:, int(ny + np.floor(dy) + 1) :, :] = 0

    if dx >= 0:
        result[:, :, : int(np.ceil(dx))] = 0
    else:
        result[:, :, int(nx + np.floor(dx) + 1) :] = 0

    return result.astype(volume.dtype)
```

**Step 2: Run test to verify it passes**

```bash
cd src/python && uv run pytest test/test_registration.py::TestApplyShift::test_roundtrip -v
```

Expected: PASS

---

## Task 7: Write test for register_volume

**Files:**
- Modify: `src/python/test/test_registration.py`

**Step 1: Add the test**

```python
from starfinder.registration import phase_correlate, apply_shift, register_volume


class TestRegisterVolume:
    """Tests for register_volume function."""

    def test_registers_multichannel(self, mini_dataset):
        """Registers all channels and returns shifts."""
        from starfinder.io import load_image_stacks

        images, _ = load_image_stacks(
            mini_dataset / "FOV_001" / "round1",
            ["ch00", "ch01", "ch02", "ch03"],
        )

        # Create shifted version
        shifted = np.roll(images, (2, -3, 5, 0), axis=(0, 1, 2, 3))

        # Use ch00 as ref/mov
        ref_img = images[:, :, :, 0]
        mov_img = shifted[:, :, :, 0]

        registered, shifts = register_volume(shifted, ref_img, mov_img)

        assert registered.shape == images.shape
        assert np.allclose(shifts, (2, -3, 5), atol=0.5)
```

**Step 2: Run test to verify it fails**

```bash
cd src/python && uv run pytest test/test_registration.py::TestRegisterVolume::test_registers_multichannel -v
```

Expected: FAIL with `NotImplementedError`

---

## Task 8: Implement register_volume

**Files:**
- Modify: `src/python/starfinder/registration/phase_correlation.py`

**Step 1: Implement the function**

Replace `register_volume` with:

```python
def register_volume(
    images: np.ndarray,
    ref_image: np.ndarray,
    mov_image: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Register multi-channel volume using phase correlation.

    Args:
        images: Multi-channel volume with shape (Z, Y, X, C).
        ref_image: Reference image with shape (Z, Y, X) for shift calculation.
        mov_image: Moving image with shape (Z, Y, X) for shift calculation.

    Returns:
        Tuple of (registered_images, shifts).
    """
    # Calculate shift
    shifts = phase_correlate(ref_image, mov_image)

    # Apply shift to each channel
    n_channels = images.shape[-1]
    registered = np.zeros_like(images)

    for c in range(n_channels):
        registered[:, :, :, c] = apply_shift(images[:, :, :, c], shifts)

    return registered, shifts
```

**Step 2: Run test to verify it passes**

```bash
cd src/python && uv run pytest test/test_registration.py::TestRegisterVolume::test_registers_multichannel -v
```

Expected: PASS

---

## Task 9: Add scikit-image backend

**Files:**
- Create: `src/python/starfinder/registration/_skimage_backend.py`
- Modify: `src/python/starfinder/registration/__init__.py`

**Step 1: Create _skimage_backend.py**

```python
"""scikit-image based phase correlation (for comparison/benchmarking)."""

from __future__ import annotations

import numpy as np
from skimage.registration import phase_cross_correlation


def phase_correlate_skimage(
    fixed: np.ndarray,
    moving: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute shift using scikit-image phase_cross_correlation.

    Args:
        fixed: Reference volume with shape (Z, Y, X).
        moving: Volume to align with shape (Z, Y, X).

    Returns:
        Tuple of (dz, dy, dx) shift values.
    """
    shift, error, diffphase = phase_cross_correlation(fixed, moving)
    return (float(shift[0]), float(shift[1]), float(shift[2]))
```

**Step 2: Update __init__.py exports**

```python
"""Registration module for image alignment."""

from starfinder.registration.phase_correlation import (
    phase_correlate,
    apply_shift,
    register_volume,
)
from starfinder.registration._skimage_backend import phase_correlate_skimage

__all__ = [
    "phase_correlate",
    "apply_shift",
    "register_volume",
    "phase_correlate_skimage",
]
```

**Step 3: Verify import**

```bash
cd src/python && uv run python -c "from starfinder.registration import phase_correlate_skimage; print('OK')"
```

Expected: Prints `OK`

---

## Task 10: Write test for backend parity

**Files:**
- Modify: `src/python/test/test_registration.py`

**Step 1: Add the test class**

```python
from starfinder.registration import (
    phase_correlate,
    apply_shift,
    register_volume,
    phase_correlate_skimage,
)


class TestBackendParity:
    """NumPy vs scikit-image produce same results."""

    def test_backends_match(self, mini_dataset):
        """Both backends return same shift for same input."""
        from starfinder.io import load_multipage_tiff

        vol = load_multipage_tiff(mini_dataset / "FOV_001" / "round1" / "ch00.tif")
        moved = np.roll(vol, (2, 3, -1), axis=(0, 1, 2))

        shift_np = phase_correlate(vol, moved)
        shift_sk = phase_correlate_skimage(vol, moved)

        assert np.allclose(shift_np, shift_sk, atol=0.5)
```

**Step 2: Run test**

```bash
cd src/python && uv run pytest test/test_registration.py::TestBackendParity::test_backends_match -v
```

Expected: PASS

---

## Task 11: Create benchmark module

**Files:**
- Create: `src/python/starfinder/registration/benchmark.py`

**Step 1: Create benchmark.py**

```python
"""Benchmark utilities for registration methods."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from starfinder.registration import phase_correlate, phase_correlate_skimage


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    method: str
    size: tuple[int, int, int]
    time_sec: float
    memory_peak_mb: float
    shift_error: float


def _measure_registration(
    func: Callable,
    fixed: np.ndarray,
    moving: np.ndarray,
    expected_shift: tuple[float, float, float],
    n_runs: int = 5,
) -> tuple[float, float, float]:
    """
    Measure time, memory, and accuracy for a registration function.

    Returns:
        Tuple of (mean_time_sec, peak_memory_mb, shift_error).
    """
    # Warm-up run
    _ = func(fixed, moving)

    # Timed runs
    times = []
    for _ in range(n_runs):
        tracemalloc.start()
        start = time.perf_counter()

        detected = func(fixed, moving)

        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(elapsed)

    mean_time = np.mean(times)
    peak_mb = peak / (1024 * 1024)

    # Compute shift error (L2 distance)
    error = np.sqrt(sum((d - e) ** 2 for d, e in zip(detected, expected_shift)))

    return mean_time, peak_mb, error


def run_benchmark(
    sizes: list[tuple[int, int, int]] | None = None,
    methods: list[str] | None = None,
    n_runs: int = 5,
    seed: int = 42,
) -> list[BenchmarkResult]:
    """
    Run registration benchmark with synthetic images.

    Args:
        sizes: List of (Z, Y, X) sizes to test. Defaults to standard set.
        methods: List of methods ("numpy", "skimage"). Defaults to both.
        n_runs: Number of runs per measurement.
        seed: Random seed for reproducibility.

    Returns:
        List of BenchmarkResult objects.
    """
    from starfinder.testdata import create_test_volume

    if sizes is None:
        sizes = [
            (5, 128, 128),    # tiny
            (10, 256, 256),   # small
            (30, 512, 512),   # medium
        ]

    if methods is None:
        methods = ["numpy", "skimage"]

    method_funcs = {
        "numpy": phase_correlate,
        "skimage": phase_correlate_skimage,
    }

    rng = np.random.default_rng(seed)
    results = []

    for size in sizes:
        nz, ny, nx = size

        # Generate synthetic volume with spots
        fixed = create_test_volume(
            shape=size,
            n_spots=20,
            spot_intensity=200,
            background=20,
            seed=seed,
        )

        # Apply known shift
        known_shift = (
            int(rng.integers(-5, 6)),
            int(rng.integers(-10, 11)),
            int(rng.integers(-10, 11)),
        )
        moving = np.roll(fixed, known_shift, axis=(0, 1, 2))

        for method in methods:
            func = method_funcs[method]
            mean_time, peak_mb, error = _measure_registration(
                func, fixed, moving, known_shift, n_runs
            )

            results.append(
                BenchmarkResult(
                    method=method,
                    size=size,
                    time_sec=mean_time,
                    memory_peak_mb=peak_mb,
                    shift_error=error,
                )
            )

    return results


def print_benchmark_table(results: list[BenchmarkResult]) -> None:
    """Print benchmark results as formatted table."""
    print()
    print("| Method  | Size           | Time (s) | Memory (MB) | Shift Error |")
    print("|---------|----------------|----------|-------------|-------------|")

    for r in results:
        size_str = f"{r.size[1]}×{r.size[2]}×{r.size[0]}"
        print(
            f"| {r.method:<7} | {size_str:<14} | {r.time_sec:>8.3f} | {r.memory_peak_mb:>11.1f} | {r.shift_error:>11.2f} |"
        )

    print()
```

**Step 2: Verify import**

```bash
cd src/python && uv run python -c "from starfinder.registration.benchmark import run_benchmark; print('OK')"
```

Expected: Error (missing `create_test_volume` in testdata module)

---

## Task 12: Add create_test_volume helper to testdata

**Files:**
- Modify: `src/python/starfinder/testdata/__init__.py`
- Modify: `src/python/starfinder/testdata/synthetic.py`

**Step 1: Add create_test_volume to synthetic.py**

Add at end of file:

```python
def create_test_volume(
    shape: tuple[int, int, int],
    n_spots: int = 20,
    spot_intensity: int = 200,
    background: int = 20,
    noise_std: int = 5,
    seed: int | None = None,
) -> np.ndarray:
    """
    Create a test volume with Gaussian spots for benchmarking.

    Args:
        shape: (Z, Y, X) dimensions.
        n_spots: Number of spots to generate.
        spot_intensity: Peak intensity of spots.
        background: Background intensity level.
        noise_std: Standard deviation of background noise.
        seed: Random seed for reproducibility.

    Returns:
        uint8 array with shape (Z, Y, X).
    """
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape

    # Create background with noise
    volume = rng.normal(background, noise_std, shape).clip(0, 255).astype(np.float32)

    # Add spots at random positions
    for _ in range(n_spots):
        z = rng.integers(2, nz - 2)
        y = rng.integers(10, ny - 10)
        x = rng.integers(10, nx - 10)

        # Add Gaussian spot
        for dz in range(-2, 3):
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    dist = np.sqrt(dz**2 + dy**2 / 4 + dx**2 / 4)
                    if dist < 5:
                        intensity = spot_intensity * np.exp(-(dist**2) / 2)
                        zz, yy, xx = z + dz, y + dy, x + dx
                        if 0 <= zz < nz and 0 <= yy < ny and 0 <= xx < nx:
                            volume[zz, yy, xx] = min(255, volume[zz, yy, xx] + intensity)

    return volume.astype(np.uint8)
```

**Step 2: Update testdata/__init__.py exports**

```python
"""Test data generation utilities."""

from starfinder.testdata.synthetic import (
    generate_synthetic_dataset,
    get_preset_config,
    SyntheticConfig,
    create_test_volume,
)

__all__ = [
    "generate_synthetic_dataset",
    "get_preset_config",
    "SyntheticConfig",
    "create_test_volume",
]
```

**Step 3: Verify benchmark import now works**

```bash
cd src/python && uv run python -c "from starfinder.registration.benchmark import run_benchmark; print('OK')"
```

Expected: Prints `OK`

---

## Task 13: Update package exports and run all tests

**Files:**
- Modify: `src/python/starfinder/__init__.py`

**Step 1: Add registration exports to main package**

```python
"""STARfinder: Spatial transcriptomics data processing pipeline."""

from starfinder.io import load_multipage_tiff, load_image_stacks, save_stack
from starfinder.registration import phase_correlate, apply_shift, register_volume

__version__ = "0.1.0"

__all__ = [
    "load_multipage_tiff",
    "load_image_stacks",
    "save_stack",
    "phase_correlate",
    "apply_shift",
    "register_volume",
    "__version__",
]
```

**Step 2: Run all tests**

```bash
cd src/python && uv run pytest test/ -v
```

Expected: All tests PASS

---

## Task 14: Run benchmark and verify

**Step 1: Run benchmark interactively**

```bash
cd src/python && uv run python -c "
from starfinder.registration.benchmark import run_benchmark, print_benchmark_table
results = run_benchmark(sizes=[(5, 128, 128), (10, 256, 256)], n_runs=3)
print_benchmark_table(results)
"
```

Expected: Benchmark table prints with timing and memory for both methods.

---

## Task 15: Commit all changes

**Step 1: Stage and commit**

```bash
git add -A
git status
```

**Step 2: Commit**

```bash
git commit -m "feat: implement registration module with phase correlation

- Add phase_correlate() using NumPy/SciPy FFT
- Add apply_shift() with frequency-domain shift
- Add register_volume() for multi-channel registration
- Add scikit-image backend for comparison
- Add benchmark module with timing and memory profiling
- Add create_test_volume() helper for benchmarking
- 5 tests passing

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Verification

**Final verification steps:**

```bash
# Run all tests
cd src/python && uv run pytest test/ -v

# Run benchmark with more sizes
cd src/python && uv run python -c "
from starfinder.registration.benchmark import run_benchmark, print_benchmark_table
results = run_benchmark(n_runs=5)
print_benchmark_table(results)
"

# Verify imports from top-level package
cd src/python && uv run python -c "
from starfinder import phase_correlate, apply_shift, register_volume
print('All imports OK')
"
```
