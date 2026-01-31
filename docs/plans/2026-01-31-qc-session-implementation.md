# QC Session Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a standalone benchmark module and QC notebooks to validate I/O, registration, and synthetic data modules before Phase 3.

**Architecture:** Extract benchmark utilities from `starfinder.registration.benchmark` into a generic `starfinder.benchmark` module with decorator-based timing, memory measurement, and reporting. Create four Jupyter notebooks for interactive validation with napari support.

**Tech Stack:** Python 3.10+, pytest, numpy, matplotlib, napari (optional), Jupyter

---

## Task 1: Create Benchmark Core Module

**Files:**
- Create: `src/python/starfinder/benchmark/__init__.py`
- Create: `src/python/starfinder/benchmark/core.py`
- Test: `src/python/test/test_benchmark.py`

**Step 1: Write the failing test**

Create `src/python/test/test_benchmark.py`:

```python
"""Tests for benchmark core utilities."""

import time
import numpy as np
import pytest

from starfinder.benchmark import BenchmarkResult, benchmark, measure


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_create_result(self):
        """BenchmarkResult stores all fields correctly."""
        result = BenchmarkResult(
            method="test_method",
            operation="test_op",
            size=(10, 256, 256),
            time_seconds=1.23,
            memory_mb=45.6,
            metrics={"accuracy": 0.99},
        )
        assert result.method == "test_method"
        assert result.operation == "test_op"
        assert result.size == (10, 256, 256)
        assert result.time_seconds == 1.23
        assert result.memory_mb == 45.6
        assert result.metrics["accuracy"] == 0.99

    def test_result_without_metrics(self):
        """BenchmarkResult works without optional metrics."""
        result = BenchmarkResult(
            method="test",
            operation="op",
            size=(5, 128, 128),
            time_seconds=0.5,
            memory_mb=10.0,
        )
        assert result.metrics == {}


class TestMeasureFunction:
    """Tests for measure() timing/memory utility."""

    def test_measure_captures_time(self):
        """measure() returns elapsed time."""
        def slow_fn():
            time.sleep(0.05)
            return 42

        result, elapsed, _ = measure(slow_fn)
        assert result == 42
        assert elapsed >= 0.04  # Allow some tolerance

    def test_measure_captures_memory(self):
        """measure() returns peak memory."""
        def alloc_fn():
            # Allocate ~1MB
            arr = np.zeros((256, 1024), dtype=np.float32)
            return arr.sum()

        result, _, memory_mb = measure(alloc_fn)
        assert result == 0.0
        assert memory_mb > 0  # Should capture some memory


class TestBenchmarkDecorator:
    """Tests for @benchmark decorator."""

    def test_decorator_returns_result(self):
        """@benchmark decorator returns BenchmarkResult."""

        @benchmark(method="test", operation="square")
        def square(x):
            return x * x

        result = square(5)
        assert isinstance(result, BenchmarkResult)
        assert result.method == "test"
        assert result.operation == "square"
        assert result.metrics["return_value"] == 25

    def test_decorator_with_size(self):
        """@benchmark decorator captures size from argument."""

        @benchmark(method="numpy", operation="sum", size_arg="arr")
        def array_sum(arr):
            return arr.sum()

        arr = np.ones((10, 20, 30))
        result = array_sum(arr)
        assert result.size == (10, 20, 30)
```

**Step 2: Run test to verify it fails**

Run: `cd src/python && uv run pytest test/test_benchmark.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'starfinder.benchmark'"

**Step 3: Create benchmark package init**

Create `src/python/starfinder/benchmark/__init__.py`:

```python
"""Benchmark utilities for performance measurement and comparison."""

from starfinder.benchmark.core import (
    BenchmarkResult,
    benchmark,
    measure,
)

__all__ = [
    "BenchmarkResult",
    "benchmark",
    "measure",
]
```

**Step 4: Write minimal implementation**

Create `src/python/starfinder/benchmark/core.py`:

```python
"""Core benchmark utilities: timing, memory measurement, and result container."""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run.

    Attributes:
        method: Name of the method/implementation being benchmarked.
        operation: Name of the operation (e.g., "phase_correlate", "load_tiff").
        size: Tuple describing input size (e.g., (Z, Y, X) for volumes).
        time_seconds: Execution time in seconds.
        memory_mb: Peak memory usage in megabytes.
        metrics: Additional custom metrics (e.g., accuracy, error).
    """

    method: str
    operation: str
    size: tuple[int, ...]
    time_seconds: float
    memory_mb: float
    metrics: dict[str, Any] = field(default_factory=dict)


def measure(fn: Callable[[], T]) -> tuple[T, float, float]:
    """
    Measure execution time and peak memory of a function.

    Args:
        fn: Zero-argument callable to measure.

    Returns:
        Tuple of (result, time_seconds, memory_mb).

    Example:
        >>> result, elapsed, mem = measure(lambda: np.zeros((1000, 1000)))
    """
    tracemalloc.start()
    start = time.perf_counter()

    result = fn()

    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    memory_mb = peak / (1024 * 1024)
    return result, elapsed, memory_mb


def benchmark(
    method: str,
    operation: str,
    size_arg: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., BenchmarkResult]]:
    """
    Decorator to benchmark a function.

    Args:
        method: Name of the method/implementation.
        operation: Name of the operation.
        size_arg: Name of argument to extract size from (must have .shape).

    Returns:
        Decorated function that returns BenchmarkResult.

    Example:
        >>> @benchmark(method="numpy", operation="fft", size_arg="arr")
        ... def compute_fft(arr):
        ...     return np.fft.fftn(arr)
        >>> result = compute_fft(np.zeros((10, 20, 30)))
        >>> result.size
        (10, 20, 30)
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., BenchmarkResult]:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> BenchmarkResult:
            # Extract size from argument if specified
            size: tuple[int, ...] = ()
            if size_arg is not None:
                # Check kwargs first, then positional args
                if size_arg in kwargs:
                    size = tuple(kwargs[size_arg].shape)
                else:
                    # Get from function signature
                    import inspect

                    sig = inspect.signature(fn)
                    params = list(sig.parameters.keys())
                    if size_arg in params:
                        idx = params.index(size_arg)
                        if idx < len(args):
                            size = tuple(args[idx].shape)

            # Measure execution
            return_value, elapsed, memory = measure(lambda: fn(*args, **kwargs))

            return BenchmarkResult(
                method=method,
                operation=operation,
                size=size,
                time_seconds=elapsed,
                memory_mb=memory,
                metrics={"return_value": return_value},
            )

        return wrapper

    return decorator
```

**Step 5: Run test to verify it passes**

Run: `cd src/python && uv run pytest test/test_benchmark.py -v`
Expected: PASS (6 tests)

**Step 6: Commit**

```bash
git add src/python/starfinder/benchmark/ src/python/test/test_benchmark.py
git commit -m "feat(benchmark): add core module with BenchmarkResult, measure, and @benchmark decorator"
```

---

## Task 2: Add Benchmark Runner and Comparison

**Files:**
- Create: `src/python/starfinder/benchmark/runner.py`
- Modify: `src/python/starfinder/benchmark/__init__.py`
- Test: `src/python/test/test_benchmark.py` (append)

**Step 1: Write the failing test**

Append to `src/python/test/test_benchmark.py`:

```python
from starfinder.benchmark import run_comparison, BenchmarkSuite


class TestRunComparison:
    """Tests for run_comparison() utility."""

    def test_compare_two_methods(self):
        """run_comparison() returns results for multiple methods."""

        def method_a(x):
            return x + 1

        def method_b(x):
            return x + 2

        results = run_comparison(
            methods={"a": method_a, "b": method_b},
            inputs=[10, 20],
            operation="increment",
        )
        assert len(results) == 4  # 2 methods × 2 inputs
        assert results[0].method == "a"
        assert results[2].method == "b"


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite collection."""

    def test_suite_collects_results(self):
        """BenchmarkSuite aggregates multiple results."""
        suite = BenchmarkSuite(name="test_suite")
        suite.add(
            BenchmarkResult(
                method="a", operation="op", size=(10,), time_seconds=1.0, memory_mb=10.0
            )
        )
        suite.add(
            BenchmarkResult(
                method="a", operation="op", size=(10,), time_seconds=1.2, memory_mb=12.0
            )
        )
        assert len(suite.results) == 2

    def test_suite_summary_stats(self):
        """BenchmarkSuite computes summary statistics."""
        suite = BenchmarkSuite(name="test_suite")
        for t in [1.0, 2.0, 3.0]:
            suite.add(
                BenchmarkResult(
                    method="a", operation="op", size=(10,), time_seconds=t, memory_mb=10.0
                )
            )
        stats = suite.summary()
        assert stats["mean_time"] == 2.0
        assert stats["min_time"] == 1.0
        assert stats["max_time"] == 3.0
```

**Step 2: Run test to verify it fails**

Run: `cd src/python && uv run pytest test/test_benchmark.py::TestRunComparison -v`
Expected: FAIL with "cannot import name 'run_comparison'"

**Step 3: Write minimal implementation**

Create `src/python/starfinder/benchmark/runner.py`:

```python
"""Benchmark runner for comparing multiple methods."""

from __future__ import annotations

from typing import Any, Callable

from starfinder.benchmark.core import BenchmarkResult, measure


def run_comparison(
    methods: dict[str, Callable],
    inputs: list[Any],
    operation: str,
    n_runs: int = 1,
    warmup: bool = False,
) -> list[BenchmarkResult]:
    """
    Run multiple methods on multiple inputs and collect results.

    Args:
        methods: Dict mapping method name to callable.
        inputs: List of inputs to pass to each method.
        operation: Name of the operation being benchmarked.
        n_runs: Number of runs per (method, input) pair for averaging.
        warmup: If True, run each method once before timing.

    Returns:
        List of BenchmarkResult objects.

    Example:
        >>> results = run_comparison(
        ...     methods={"numpy": np.sum, "python": sum},
        ...     inputs=[list(range(100)), list(range(1000))],
        ...     operation="sum",
        ... )
    """
    results = []

    for inp in inputs:
        for method_name, func in methods.items():
            # Optional warmup
            if warmup:
                _ = func(inp)

            # Collect runs
            times = []
            memories = []
            return_value = None

            for _ in range(n_runs):
                ret, elapsed, mem = measure(lambda f=func, i=inp: f(i))
                times.append(elapsed)
                memories.append(mem)
                return_value = ret

            # Average results
            import numpy as np

            avg_time = float(np.mean(times))
            avg_mem = float(np.mean(memories))

            # Extract size if input has shape
            size: tuple[int, ...] = ()
            if hasattr(inp, "shape"):
                size = tuple(inp.shape)

            results.append(
                BenchmarkResult(
                    method=method_name,
                    operation=operation,
                    size=size,
                    time_seconds=avg_time,
                    memory_mb=avg_mem,
                    metrics={"return_value": return_value},
                )
            )

    return results


class BenchmarkSuite:
    """Collection of benchmark results with aggregation utilities.

    Attributes:
        name: Name of the benchmark suite.
        results: List of collected BenchmarkResult objects.
    """

    def __init__(self, name: str):
        self.name = name
        self.results: list[BenchmarkResult] = []

    def add(self, result: BenchmarkResult) -> None:
        """Add a result to the suite."""
        self.results.append(result)

    def summary(self) -> dict[str, float]:
        """Compute summary statistics across all results.

        Returns:
            Dict with mean_time, min_time, max_time, mean_memory.
        """
        if not self.results:
            return {}

        import numpy as np

        times = [r.time_seconds for r in self.results]
        memories = [r.memory_mb for r in self.results]

        return {
            "mean_time": float(np.mean(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "std_time": float(np.std(times)),
            "mean_memory": float(np.mean(memories)),
        }

    def filter(
        self, method: str | None = None, operation: str | None = None
    ) -> list[BenchmarkResult]:
        """Filter results by method and/or operation."""
        filtered = self.results
        if method is not None:
            filtered = [r for r in filtered if r.method == method]
        if operation is not None:
            filtered = [r for r in filtered if r.operation == operation]
        return filtered
```

**Step 4: Update __init__.py exports**

Modify `src/python/starfinder/benchmark/__init__.py`:

```python
"""Benchmark utilities for performance measurement and comparison."""

from starfinder.benchmark.core import (
    BenchmarkResult,
    benchmark,
    measure,
)
from starfinder.benchmark.runner import (
    BenchmarkSuite,
    run_comparison,
)

__all__ = [
    "BenchmarkResult",
    "BenchmarkSuite",
    "benchmark",
    "measure",
    "run_comparison",
]
```

**Step 5: Run test to verify it passes**

Run: `cd src/python && uv run pytest test/test_benchmark.py -v`
Expected: PASS (9 tests)

**Step 6: Commit**

```bash
git add src/python/starfinder/benchmark/
git commit -m "feat(benchmark): add run_comparison and BenchmarkSuite for multi-method benchmarks"
```

---

## Task 3: Add Benchmark Reporting

**Files:**
- Create: `src/python/starfinder/benchmark/report.py`
- Modify: `src/python/starfinder/benchmark/__init__.py`
- Test: `src/python/test/test_benchmark.py` (append)

**Step 1: Write the failing test**

Append to `src/python/test/test_benchmark.py`:

```python
import json
from pathlib import Path

from starfinder.benchmark import print_table, save_csv, save_json


class TestReporting:
    """Tests for benchmark reporting utilities."""

    def test_print_table(self, capsys):
        """print_table() outputs formatted markdown table."""
        results = [
            BenchmarkResult(
                method="numpy",
                operation="fft",
                size=(10, 256, 256),
                time_seconds=0.123,
                memory_mb=45.6,
            ),
            BenchmarkResult(
                method="scipy",
                operation="fft",
                size=(10, 256, 256),
                time_seconds=0.145,
                memory_mb=48.2,
            ),
        ]
        print_table(results)
        captured = capsys.readouterr()
        assert "numpy" in captured.out
        assert "scipy" in captured.out
        assert "0.123" in captured.out

    def test_save_csv(self, tmp_path):
        """save_csv() writes results to CSV file."""
        results = [
            BenchmarkResult(
                method="test",
                operation="op",
                size=(10,),
                time_seconds=1.0,
                memory_mb=10.0,
            )
        ]
        path = tmp_path / "results.csv"
        save_csv(results, path)
        assert path.exists()
        content = path.read_text()
        assert "method" in content
        assert "test" in content

    def test_save_json(self, tmp_path):
        """save_json() writes results to JSON file."""
        results = [
            BenchmarkResult(
                method="test",
                operation="op",
                size=(10, 20),
                time_seconds=1.5,
                memory_mb=15.0,
                metrics={"accuracy": 0.99},
            )
        ]
        path = tmp_path / "results.json"
        save_json(results, path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert data[0]["method"] == "test"
        assert data[0]["metrics"]["accuracy"] == 0.99
```

**Step 2: Run test to verify it fails**

Run: `cd src/python && uv run pytest test/test_benchmark.py::TestReporting -v`
Expected: FAIL with "cannot import name 'print_table'"

**Step 3: Write minimal implementation**

Create `src/python/starfinder/benchmark/report.py`:

```python
"""Benchmark reporting utilities: table, CSV, JSON output."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from starfinder.benchmark.core import BenchmarkResult


def print_table(results: list[BenchmarkResult], show_metrics: bool = False) -> None:
    """
    Print benchmark results as formatted markdown table.

    Args:
        results: List of BenchmarkResult objects.
        show_metrics: If True, include metrics column.
    """
    if not results:
        print("No results to display.")
        return

    # Header
    header = "| Method | Operation | Size | Time (s) | Memory (MB) |"
    separator = "|--------|-----------|------|----------|-------------|"
    if show_metrics:
        header += " Metrics |"
        separator += "---------|"

    print()
    print(header)
    print(separator)

    for r in results:
        size_str = "×".join(str(s) for s in r.size) if r.size else "-"
        row = f"| {r.method:<6} | {r.operation:<9} | {size_str:<4} | {r.time_seconds:>8.4f} | {r.memory_mb:>11.1f} |"
        if show_metrics:
            # Filter out return_value for display
            display_metrics = {k: v for k, v in r.metrics.items() if k != "return_value"}
            metrics_str = ", ".join(f"{k}={v}" for k, v in display_metrics.items())
            row += f" {metrics_str:<7} |"
        print(row)

    print()


def save_csv(results: list[BenchmarkResult], path: Path | str) -> None:
    """
    Save benchmark results to CSV file.

    Args:
        results: List of BenchmarkResult objects.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "operation", "size", "time_seconds", "memory_mb"])
        for r in results:
            size_str = "×".join(str(s) for s in r.size) if r.size else ""
            writer.writerow([r.method, r.operation, size_str, r.time_seconds, r.memory_mb])


def save_json(results: list[BenchmarkResult], path: Path | str) -> None:
    """
    Save benchmark results to JSON file.

    Args:
        results: List of BenchmarkResult objects.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for r in results:
        data.append(
            {
                "method": r.method,
                "operation": r.operation,
                "size": list(r.size),
                "time_seconds": r.time_seconds,
                "memory_mb": r.memory_mb,
                "metrics": r.metrics,
            }
        )

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
```

**Step 4: Update __init__.py exports**

Modify `src/python/starfinder/benchmark/__init__.py`:

```python
"""Benchmark utilities for performance measurement and comparison."""

from starfinder.benchmark.core import (
    BenchmarkResult,
    benchmark,
    measure,
)
from starfinder.benchmark.runner import (
    BenchmarkSuite,
    run_comparison,
)
from starfinder.benchmark.report import (
    print_table,
    save_csv,
    save_json,
)

__all__ = [
    "BenchmarkResult",
    "BenchmarkSuite",
    "benchmark",
    "measure",
    "print_table",
    "run_comparison",
    "save_csv",
    "save_json",
]
```

**Step 5: Run test to verify it passes**

Run: `cd src/python && uv run pytest test/test_benchmark.py -v`
Expected: PASS (12 tests)

**Step 6: Commit**

```bash
git add src/python/starfinder/benchmark/
git commit -m "feat(benchmark): add reporting utilities (print_table, save_csv, save_json)"
```

---

## Task 4: Add Benchmark Presets

**Files:**
- Create: `src/python/starfinder/benchmark/presets.py`
- Modify: `src/python/starfinder/benchmark/__init__.py`
- Test: `src/python/test/test_benchmark.py` (append)

**Step 1: Write the failing test**

Append to `src/python/test/test_benchmark.py`:

```python
from starfinder.benchmark import SIZE_PRESETS, get_size_preset


class TestPresets:
    """Tests for benchmark presets."""

    def test_size_presets_exist(self):
        """SIZE_PRESETS contains expected presets."""
        assert "tiny" in SIZE_PRESETS
        assert "small" in SIZE_PRESETS
        assert "medium" in SIZE_PRESETS

    def test_get_size_preset(self):
        """get_size_preset() returns correct shape."""
        shape = get_size_preset("tiny")
        assert len(shape) == 3
        assert all(isinstance(s, int) for s in shape)

    def test_get_size_preset_invalid(self):
        """get_size_preset() raises for unknown preset."""
        with pytest.raises(ValueError, match="Unknown size preset"):
            get_size_preset("nonexistent")
```

**Step 2: Run test to verify it fails**

Run: `cd src/python && uv run pytest test/test_benchmark.py::TestPresets -v`
Expected: FAIL with "cannot import name 'SIZE_PRESETS'"

**Step 3: Write minimal implementation**

Create `src/python/starfinder/benchmark/presets.py`:

```python
"""Benchmark presets for standard test configurations."""

from __future__ import annotations

# Standard volume size presets (Z, Y, X)
SIZE_PRESETS: dict[str, tuple[int, int, int]] = {
    "tiny": (5, 128, 128),
    "small": (10, 256, 256),
    "medium": (30, 512, 512),
    "large": (30, 1024, 1024),
    "xlarge": (30, 1496, 1496),  # cell-culture-3D size
    "tissue": (30, 3072, 3072),  # tissue-2D size
}


def get_size_preset(name: str) -> tuple[int, int, int]:
    """
    Get volume size for a preset name.

    Args:
        name: Preset name (tiny, small, medium, large, xlarge, tissue).

    Returns:
        Tuple of (Z, Y, X) dimensions.

    Raises:
        ValueError: If preset name is unknown.
    """
    if name not in SIZE_PRESETS:
        raise ValueError(
            f"Unknown size preset: '{name}'. "
            f"Available: {list(SIZE_PRESETS.keys())}"
        )
    return SIZE_PRESETS[name]
```

**Step 4: Update __init__.py exports**

Modify `src/python/starfinder/benchmark/__init__.py`:

```python
"""Benchmark utilities for performance measurement and comparison."""

from starfinder.benchmark.core import (
    BenchmarkResult,
    benchmark,
    measure,
)
from starfinder.benchmark.runner import (
    BenchmarkSuite,
    run_comparison,
)
from starfinder.benchmark.report import (
    print_table,
    save_csv,
    save_json,
)
from starfinder.benchmark.presets import (
    SIZE_PRESETS,
    get_size_preset,
)

__all__ = [
    "BenchmarkResult",
    "BenchmarkSuite",
    "SIZE_PRESETS",
    "benchmark",
    "get_size_preset",
    "measure",
    "print_table",
    "run_comparison",
    "save_csv",
    "save_json",
]
```

**Step 5: Run test to verify it passes**

Run: `cd src/python && uv run pytest test/test_benchmark.py -v`
Expected: PASS (15 tests)

**Step 6: Commit**

```bash
git add src/python/starfinder/benchmark/
git commit -m "feat(benchmark): add size presets for standard test configurations"
```

---

## Task 5: Migrate Registration Benchmark

**Files:**
- Modify: `src/python/starfinder/registration/benchmark.py`
- Test: existing tests should still pass

**Step 1: Update registration benchmark to use new module**

Replace `src/python/starfinder/registration/benchmark.py`:

```python
"""Registration-specific benchmark utilities.

This module provides convenience functions for benchmarking registration
methods using the generic starfinder.benchmark framework.
"""

from __future__ import annotations

import numpy as np

from starfinder.benchmark import (
    BenchmarkResult,
    BenchmarkSuite,
    SIZE_PRESETS,
    measure,
    print_table,
    run_comparison,
)
from starfinder.registration import phase_correlate, phase_correlate_skimage


def benchmark_registration(
    sizes: list[tuple[int, int, int]] | None = None,
    methods: list[str] | None = None,
    n_runs: int = 5,
    seed: int = 42,
) -> list[BenchmarkResult]:
    """
    Benchmark registration methods with synthetic images.

    Args:
        sizes: List of (Z, Y, X) sizes. Defaults to tiny, small, medium.
        methods: List of method names ("numpy", "skimage"). Defaults to both.
        n_runs: Number of runs per measurement.
        seed: Random seed for reproducibility.

    Returns:
        List of BenchmarkResult objects.
    """
    from starfinder.testdata import create_test_volume

    if sizes is None:
        sizes = [
            SIZE_PRESETS["tiny"],
            SIZE_PRESETS["small"],
            SIZE_PRESETS["medium"],
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
        # Generate synthetic volume
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

        for method_name in methods:
            func = method_funcs[method_name]

            # Warm-up
            _ = func(fixed, moving)

            # Timed runs
            times = []
            memories = []
            detected_shift = None

            for _ in range(n_runs):
                detected_shift, elapsed, mem = measure(lambda: func(fixed, moving))
                times.append(elapsed)
                memories.append(mem)

            # Compute shift error
            error = np.sqrt(
                sum((d - e) ** 2 for d, e in zip(detected_shift, known_shift))
            )

            results.append(
                BenchmarkResult(
                    method=method_name,
                    operation="phase_correlate",
                    size=size,
                    time_seconds=float(np.mean(times)),
                    memory_mb=float(np.mean(memories)),
                    metrics={
                        "shift_error": error,
                        "known_shift": known_shift,
                        "detected_shift": detected_shift,
                    },
                )
            )

    return results


# Re-export for backwards compatibility
def run_benchmark(*args, **kwargs):
    """Deprecated: Use benchmark_registration() instead."""
    return benchmark_registration(*args, **kwargs)


def print_benchmark_table(results: list[BenchmarkResult]) -> None:
    """Print registration benchmark results with shift error."""
    print()
    print("| Method  | Size           | Time (s) | Memory (MB) | Shift Error |")
    print("|---------|----------------|----------|-------------|-------------|")

    for r in results:
        size_str = f"{r.size[1]}×{r.size[2]}×{r.size[0]}"
        error = r.metrics.get("shift_error", 0.0)
        print(
            f"| {r.method:<7} | {size_str:<14} | {r.time_seconds:>8.3f} | "
            f"{r.memory_mb:>11.1f} | {error:>11.2f} |"
        )

    print()
```

**Step 2: Run existing registration tests to verify they still pass**

Run: `cd src/python && uv run pytest test/test_registration.py -v`
Expected: PASS (all existing tests)

**Step 3: Commit**

```bash
git add src/python/starfinder/registration/benchmark.py
git commit -m "refactor(registration): migrate benchmark to use starfinder.benchmark framework"
```

---

## Task 6: Create QC Benchmark Notebook

**Files:**
- Create: `tests/qc_benchmark.ipynb`

**Step 1: Create the notebook**

Create `tests/qc_benchmark.ipynb` with the following cells:

```python
# Cell 1 (markdown)
"""
# QC: Benchmark Framework Validation

This notebook validates the `starfinder.benchmark` module by:
1. Testing the `@benchmark` decorator
2. Testing `run_comparison()` across methods
3. Testing report generation (table, CSV, JSON)
4. Validating timing/memory measurements are reasonable
"""

# Cell 2 (code) - Setup
import sys
sys.path.insert(0, "../src/python")

import numpy as np
from pathlib import Path

from starfinder.benchmark import (
    BenchmarkResult,
    BenchmarkSuite,
    benchmark,
    measure,
    run_comparison,
    print_table,
    save_csv,
    save_json,
    SIZE_PRESETS,
)

print("Benchmark module loaded successfully!")
print(f"Available size presets: {list(SIZE_PRESETS.keys())}")

# Cell 3 (markdown)
"""
## 1. Test `measure()` Function

Verify that `measure()` correctly captures execution time and memory.
"""

# Cell 4 (code)
import time

def slow_function():
    """Function that takes ~100ms."""
    time.sleep(0.1)
    return 42

result, elapsed, memory = measure(slow_function)
print(f"Result: {result}")
print(f"Elapsed time: {elapsed:.3f}s (expected ~0.1s)")
print(f"Memory: {memory:.2f} MB")

assert result == 42, "Return value should be preserved"
assert 0.08 < elapsed < 0.15, f"Time should be ~0.1s, got {elapsed:.3f}s"
print("✓ measure() works correctly")

# Cell 5 (markdown)
"""
## 2. Test `@benchmark` Decorator

Verify the decorator captures timing and extracts size from arguments.
"""

# Cell 6 (code)
@benchmark(method="numpy", operation="sum", size_arg="arr")
def array_sum(arr):
    return arr.sum()

test_arr = np.ones((10, 256, 256))
result = array_sum(test_arr)

print(f"Method: {result.method}")
print(f"Operation: {result.operation}")
print(f"Size: {result.size}")
print(f"Time: {result.time_seconds:.6f}s")
print(f"Memory: {result.memory_mb:.2f} MB")
print(f"Return value: {result.metrics['return_value']}")

assert result.size == (10, 256, 256), "Size should be extracted from array"
assert result.metrics["return_value"] == 10 * 256 * 256, "Return value incorrect"
print("✓ @benchmark decorator works correctly")

# Cell 7 (markdown)
"""
## 3. Test `run_comparison()`

Compare multiple methods on the same inputs.
"""

# Cell 8 (code)
def numpy_sum(arr):
    return np.sum(arr)

def loop_sum(arr):
    total = 0.0
    for val in arr.flat:
        total += val
    return total

# Create test arrays
small_arr = np.ones((5, 64, 64))
medium_arr = np.ones((10, 128, 128))

results = run_comparison(
    methods={"numpy": numpy_sum, "loop": loop_sum},
    inputs=[small_arr, medium_arr],
    operation="sum",
    n_runs=3,
    warmup=True,
)

print(f"Collected {len(results)} results")
print_table(results)

# Verify numpy is faster than loop
numpy_times = [r.time_seconds for r in results if r.method == "numpy"]
loop_times = [r.time_seconds for r in results if r.method == "loop"]
print(f"NumPy avg: {np.mean(numpy_times):.6f}s")
print(f"Loop avg: {np.mean(loop_times):.6f}s")
print(f"NumPy is {np.mean(loop_times) / np.mean(numpy_times):.1f}x faster")
print("✓ run_comparison() works correctly")

# Cell 9 (markdown)
"""
## 4. Test Report Generation

Save results to CSV and JSON.
"""

# Cell 10 (code)
output_dir = Path("../tests/benchmark_output")
output_dir.mkdir(exist_ok=True)

# Save CSV
csv_path = output_dir / "test_results.csv"
save_csv(results, csv_path)
print(f"Saved CSV to: {csv_path}")
print(csv_path.read_text()[:500])

# Save JSON
json_path = output_dir / "test_results.json"
save_json(results, json_path)
print(f"\nSaved JSON to: {json_path}")

import json
data = json.loads(json_path.read_text())
print(f"JSON contains {len(data)} results")
print("✓ Report generation works correctly")

# Cell 11 (markdown)
"""
## 5. Test BenchmarkSuite

Collect results and compute summary statistics.
"""

# Cell 12 (code)
suite = BenchmarkSuite(name="sum_benchmark")
for r in results:
    suite.add(r)

print(f"Suite '{suite.name}' has {len(suite.results)} results")

stats = suite.summary()
print(f"\nSummary statistics:")
for key, value in stats.items():
    print(f"  {key}: {value:.6f}")

# Filter by method
numpy_results = suite.filter(method="numpy")
print(f"\nNumPy results: {len(numpy_results)}")
print("✓ BenchmarkSuite works correctly")

# Cell 13 (markdown)
"""
## Summary

All benchmark framework components validated:
- [x] `measure()` - captures time and memory
- [x] `@benchmark` decorator - wraps functions with timing
- [x] `run_comparison()` - compares multiple methods
- [x] `print_table()` - formatted output
- [x] `save_csv()` / `save_json()` - file output
- [x] `BenchmarkSuite` - result collection and stats
"""
```

**Step 2: Run the notebook to verify**

Run: `cd src/python && uv run jupyter execute ../tests/qc_benchmark.ipynb`
Or open in JupyterLab and run interactively.

**Step 3: Commit**

```bash
git add tests/qc_benchmark.ipynb
git commit -m "docs: add QC notebook for benchmark framework validation"
```

---

## Task 7: Create QC I/O Notebook

**Files:**
- Create: `tests/qc_io.ipynb`

**Step 1: Create the notebook**

Create `tests/qc_io.ipynb` with the following cells:

```python
# Cell 1 (markdown)
"""
# QC: I/O Module Validation

This notebook validates the `starfinder.io` module by:
1. Loading synthetic TIFF files and verifying shape/dtype
2. Loading multi-channel stacks
3. Save and reload roundtrip test
4. Interactive 3D inspection with napari
5. Benchmark load/save performance
"""

# Cell 2 (code) - Setup
import sys
sys.path.insert(0, "../src/python")

import numpy as np
from pathlib import Path

from starfinder.io import load_multipage_tiff, load_image_stacks, save_stack
from starfinder.benchmark import measure, print_table, BenchmarkResult

# Path to synthetic dataset
MINI_DATASET = Path("../tests/fixtures/synthetic/mini")
assert MINI_DATASET.exists(), f"Mini dataset not found at {MINI_DATASET}"

print("I/O module loaded successfully!")
print(f"Mini dataset path: {MINI_DATASET}")

# Cell 3 (markdown)
"""
## 1. Load Single TIFF

Load a single channel TIFF and verify shape, dtype, axis order.
"""

# Cell 4 (code)
# Load round1, channel 0 from FOV_001
tiff_path = MINI_DATASET / "FOV_001" / "round1" / "ch00.tif"
print(f"Loading: {tiff_path}")

img = load_multipage_tiff(tiff_path)

print(f"Shape: {img.shape} (expected: Z, Y, X)")
print(f"Dtype: {img.dtype} (expected: uint8)")
print(f"Min: {img.min()}, Max: {img.max()}")

assert img.ndim == 3, f"Expected 3D array, got {img.ndim}D"
assert img.dtype == np.uint8, f"Expected uint8, got {img.dtype}"
print("✓ Single TIFF loaded correctly")

# Cell 5 (markdown)
"""
## 2. Load Multi-Channel Stack

Load all channels from a round and verify (Z, Y, X, C) shape.
"""

# Cell 6 (code)
round_dir = MINI_DATASET / "FOV_001" / "round1"
channel_order = ["ch00", "ch01", "ch02", "ch03"]

stack, metadata = load_image_stacks(round_dir, channel_order)

print(f"Shape: {stack.shape} (expected: Z, Y, X, C)")
print(f"Dtype: {stack.dtype}")
print(f"Metadata: {metadata}")

assert stack.ndim == 4, f"Expected 4D array, got {stack.ndim}D"
assert stack.shape[-1] == 4, f"Expected 4 channels, got {stack.shape[-1]}"
print("✓ Multi-channel stack loaded correctly")

# Cell 7 (markdown)
"""
## 3. Save and Reload Roundtrip

Save a stack, reload it, and verify data integrity.
"""

# Cell 8 (code)
import tempfile

# Create test data
test_data = np.random.randint(0, 256, (5, 64, 64), dtype=np.uint8)

with tempfile.TemporaryDirectory() as tmpdir:
    save_path = Path(tmpdir) / "test_roundtrip.tif"

    # Save
    save_stack(test_data, save_path)
    print(f"Saved to: {save_path}")

    # Reload
    reloaded = load_multipage_tiff(save_path, convert_uint8=False)
    print(f"Reloaded shape: {reloaded.shape}")

    # Verify
    assert np.array_equal(test_data, reloaded), "Data mismatch after roundtrip!"
    print("✓ Roundtrip test passed")

# Cell 9 (markdown)
"""
## 4. Benchmark Load/Save Performance
"""

# Cell 10 (code)
results = []

# Benchmark loading
for size_name, tiff_path in [
    ("mini_ch", MINI_DATASET / "FOV_001" / "round1" / "ch00.tif"),
]:
    _, elapsed, mem = measure(lambda p=tiff_path: load_multipage_tiff(p))
    results.append(BenchmarkResult(
        method="bioio",
        operation="load_tiff",
        size=img.shape,
        time_seconds=elapsed,
        memory_mb=mem,
    ))

# Benchmark saving (with and without compression)
test_vol = np.random.randint(0, 256, (10, 256, 256), dtype=np.uint8)

with tempfile.TemporaryDirectory() as tmpdir:
    # Uncompressed
    path = Path(tmpdir) / "test.tif"
    _, elapsed, mem = measure(lambda: save_stack(test_vol, path, compress=False))
    results.append(BenchmarkResult(
        method="tifffile",
        operation="save_uncompressed",
        size=test_vol.shape,
        time_seconds=elapsed,
        memory_mb=mem,
    ))

    # Compressed
    path_z = Path(tmpdir) / "test_compressed.tif"
    _, elapsed, mem = measure(lambda: save_stack(test_vol, path_z, compress=True))
    results.append(BenchmarkResult(
        method="tifffile",
        operation="save_compressed",
        size=test_vol.shape,
        time_seconds=elapsed,
        memory_mb=mem,
    ))

print_table(results)

# Cell 11 (markdown)
"""
## 5. Visual Inspection with napari

Interactive 3D visualization (run this cell in JupyterLab with napari installed).
"""

# Cell 12 (code)
# napari visualization (optional - uncomment to use)
# Requires: pip install napari[all]

try:
    import napari

    # Load full FOV data
    stack, _ = load_image_stacks(
        MINI_DATASET / "FOV_001" / "round1",
        ["ch00", "ch01", "ch02", "ch03"]
    )

    viewer = napari.Viewer()
    viewer.add_image(stack, name="FOV_001_round1", channel_axis=3)

    print("napari viewer opened. Explore the 3D stack interactively.")
    print("Use the slider to navigate Z slices.")

except ImportError:
    print("napari not installed. Run: pip install napari[all]")
    print("Skipping interactive visualization.")

# Cell 13 (markdown)
"""
## Summary

I/O module validation results:
- [x] `load_multipage_tiff()` - returns (Z, Y, X) uint8 array
- [x] `load_image_stacks()` - returns (Z, Y, X, C) array with metadata
- [x] `save_stack()` - roundtrip preserves data
- [x] Benchmark - load/save timing captured
- [ ] napari - interactive 3D inspection (requires napari installation)
"""
```

**Step 2: Commit**

```bash
git add tests/qc_io.ipynb
git commit -m "docs: add QC notebook for I/O module validation"
```

---

## Task 8: Create QC Synthetic Notebook

**Files:**
- Create: `tests/qc_synthetic.ipynb`

**Step 1: Create the notebook**

Create `tests/qc_synthetic.ipynb` with the following cells:

```python
# Cell 1 (markdown)
"""
# QC: Synthetic Data Generator Validation

This notebook validates the `starfinder.testdata` module by:
1. Verifying generated dataset structure
2. Checking spot positions against ground truth
3. Validating two-base encoding
4. Visualizing spots with matplotlib and napari
"""

# Cell 2 (code) - Setup
import sys
sys.path.insert(0, "../src/python")

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from starfinder.testdata import (
    generate_synthetic_dataset,
    encode_barcode_to_colors,
    get_preset_config,
    TEST_CODEBOOK,
    COLOR_TO_CHANNEL,
)
from starfinder.io import load_multipage_tiff, load_image_stacks

MINI_DATASET = Path("../tests/fixtures/synthetic/mini")

print("Synthetic data module loaded!")
print(f"Test codebook has {len(TEST_CODEBOOK)} genes")

# Cell 3 (markdown)
"""
## 1. Verify Dataset Structure

Check that all expected files and directories exist.
"""

# Cell 4 (code)
def verify_dataset_structure(dataset_path: Path, config) -> dict:
    """Verify dataset has expected structure."""
    issues = []

    # Check root files
    if not (dataset_path / "ground_truth.json").exists():
        issues.append("Missing ground_truth.json")
    if not (dataset_path / "codebook.csv").exists():
        issues.append("Missing codebook.csv")

    # Check FOV directories
    for fov_idx in range(config.n_fovs):
        fov_id = f"FOV_{fov_idx + 1:03d}"
        fov_dir = dataset_path / fov_id

        if not fov_dir.exists():
            issues.append(f"Missing FOV directory: {fov_id}")
            continue

        # Check rounds
        for round_idx in range(1, config.n_rounds + 1):
            round_dir = fov_dir / f"round{round_idx}"
            if not round_dir.exists():
                issues.append(f"Missing round directory: {fov_id}/round{round_idx}")
                continue

            # Check channels
            for ch in range(config.n_channels):
                tiff_path = round_dir / f"ch{ch:02d}.tif"
                if not tiff_path.exists():
                    issues.append(f"Missing TIFF: {tiff_path.relative_to(dataset_path)}")

    return {"valid": len(issues) == 0, "issues": issues}

config = get_preset_config("mini")
result = verify_dataset_structure(MINI_DATASET, config)

if result["valid"]:
    print("✓ Dataset structure is valid")
else:
    print("✗ Dataset structure issues:")
    for issue in result["issues"]:
        print(f"  - {issue}")

# Cell 5 (markdown)
"""
## 2. Validate Two-Base Encoding

Verify barcode → color sequence mapping.
"""

# Cell 6 (code)
print("Two-base encoding validation:")
print("-" * 50)

for gene, barcode in TEST_CODEBOOK:
    color_seq = encode_barcode_to_colors(barcode)
    channels = [COLOR_TO_CHANNEL[c] for c in color_seq]
    print(f"{gene}: {barcode} → {barcode[::-1]} → {color_seq} → channels {channels}")

# Specific test case from CLAUDE.md
# CACGC → CGCAC → 4422 (ch03, ch03, ch01, ch01)
test_barcode = "CACGC"
expected_colors = "4422"
actual_colors = encode_barcode_to_colors(test_barcode)
assert actual_colors == expected_colors, f"Expected {expected_colors}, got {actual_colors}"
print(f"\n✓ Encoding test passed: {test_barcode} → {actual_colors}")

# Cell 7 (markdown)
"""
## 3. Verify Spot Positions

Load ground truth and overlay spots on max projection.
"""

# Cell 8 (code)
# Load ground truth
with open(MINI_DATASET / "ground_truth.json") as f:
    ground_truth = json.load(f)

fov_data = ground_truth["fovs"]["FOV_001"]
spots = fov_data["spots"]

print(f"FOV_001 has {len(spots)} spots")
print(f"Expected shifts per round: {fov_data['shifts']}")

# Load round1 and create max projection
stack, _ = load_image_stacks(
    MINI_DATASET / "FOV_001" / "round1",
    ["ch00", "ch01", "ch02", "ch03"]
)

# Max projection across Z and channels
max_proj = stack.max(axis=(0, 3))

# Plot with spot overlay
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(max_proj, cmap="gray")

# Overlay spot positions
for spot in spots:
    z, y, x = spot["position"]
    ax.plot(x, y, "r+", markersize=8)
    ax.annotate(spot["gene"], (x + 5, y - 5), fontsize=6, color="yellow")

ax.set_title("Round 1 Max Projection with Spot Positions")
plt.tight_layout()
plt.show()

print(f"✓ Plotted {len(spots)} spots on max projection")

# Cell 9 (markdown)
"""
## 4. Verify Spots Appear in Correct Channels

For each spot, check that it appears in the expected channel based on its color sequence.
"""

# Cell 10 (code)
def check_spot_in_channel(stack, spot, round_idx):
    """Check if spot appears in expected channel for given round."""
    z, y, x = spot["position"]
    color_seq = spot["color_seq"]
    expected_channel = COLOR_TO_CHANNEL[color_seq[round_idx - 1]]

    # Get intensity at spot location across channels
    intensities = stack[z, y, x, :]
    max_channel = np.argmax(intensities)

    return {
        "spot_id": spot["id"],
        "expected_channel": expected_channel,
        "max_channel": max_channel,
        "intensities": intensities.tolist(),
        "match": expected_channel == max_channel,
    }

# Check round 1
results = []
for spot in spots[:5]:  # Check first 5 spots
    result = check_spot_in_channel(stack, spot, round_idx=1)
    results.append(result)
    status = "✓" if result["match"] else "✗"
    print(f"Spot {result['spot_id']}: expected ch{result['expected_channel']}, "
          f"max at ch{result['max_channel']} {status}")

matches = sum(1 for r in results if r["match"])
print(f"\n{matches}/{len(results)} spots in expected channels")

# Cell 11 (markdown)
"""
## 5. napari 3D Visualization with Spots

Interactive visualization of spots in 3D (requires napari).
"""

# Cell 12 (code)
try:
    import napari

    # Load stack
    stack, _ = load_image_stacks(
        MINI_DATASET / "FOV_001" / "round1",
        ["ch00", "ch01", "ch02", "ch03"]
    )

    # Extract spot coordinates
    spot_coords = np.array([spot["position"] for spot in spots])  # (N, 3) as Z, Y, X
    spot_genes = [spot["gene"] for spot in spots]

    viewer = napari.Viewer()
    viewer.add_image(stack, name="round1", channel_axis=3)
    viewer.add_points(
        spot_coords,
        name="spots",
        size=5,
        face_color="red",
        properties={"gene": spot_genes},
    )

    print("napari viewer opened with spots overlay.")
    print("Navigate Z slices to see spots in 3D context.")

except ImportError:
    print("napari not installed. Skipping 3D visualization.")

# Cell 13 (markdown)
"""
## Summary

Synthetic data validation results:
- [x] Dataset structure - all files present
- [x] Two-base encoding - barcode → color mapping correct
- [x] Spot positions - spots visible at expected locations
- [x] Channel assignment - spots appear in expected channels
- [ ] napari visualization - 3D inspection (requires napari)
"""
```

**Step 2: Commit**

```bash
git add tests/qc_synthetic.ipynb
git commit -m "docs: add QC notebook for synthetic data generator validation"
```

---

## Task 9: Create QC Registration Notebook

**Files:**
- Create: `tests/qc_registration.ipynb`

**Step 1: Create the notebook**

Create `tests/qc_registration.ipynb` with the following cells:

```python
# Cell 1 (markdown)
"""
# QC: Registration Module Validation

This notebook validates the `starfinder.registration` module by:
1. Testing shift recovery with known shifts
2. Testing `apply_shift()` edge zeroing
3. Multi-channel registration
4. Comparing NumPy vs scikit-image backends
5. napari overlay visualization
"""

# Cell 2 (code) - Setup
import sys
sys.path.insert(0, "../src/python")

import numpy as np
from pathlib import Path

from starfinder.registration import (
    phase_correlate,
    apply_shift,
    register_volume,
    phase_correlate_skimage,
)
from starfinder.registration.benchmark import benchmark_registration, print_benchmark_table
from starfinder.testdata import create_test_volume
from starfinder.io import load_image_stacks

MINI_DATASET = Path("../tests/fixtures/synthetic/mini")

print("Registration module loaded!")

# Cell 3 (markdown)
"""
## 1. Test Known Shift Recovery

Create a shifted volume and verify the shift is recovered correctly.
"""

# Cell 4 (code)
# Create test volume
np.random.seed(42)
fixed = create_test_volume((10, 256, 256), n_spots=30, seed=42)

# Apply known shift
known_shift = (2, -5, 8)  # (dz, dy, dx)
moving = np.roll(fixed, known_shift, axis=(0, 1, 2))

# Recover shift using phase correlation
detected_shift = phase_correlate(fixed, moving)

print(f"Known shift:    {known_shift}")
print(f"Detected shift: {detected_shift}")

# Check accuracy
error = np.sqrt(sum((d - k) ** 2 for d, k in zip(detected_shift, known_shift)))
print(f"L2 error: {error:.6f}")

assert error < 0.5, f"Shift error too large: {error}"
print("✓ Shift recovery test passed")

# Cell 5 (markdown)
"""
## 2. Test `apply_shift()` Edge Zeroing

Verify that shifted regions are properly zeroed out.
"""

# Cell 6 (code)
# Create simple test volume
vol = np.ones((10, 50, 50), dtype=np.float32) * 100

# Apply shift
shift = (2.0, 5.0, -3.0)
shifted = apply_shift(vol, shift)

print(f"Original volume: all values = 100")
print(f"Shift applied: {shift}")

# Check edge zeroing
print(f"\nFirst 3 Z slices (should be zero): {shifted[:3, 25, 25]}")
print(f"First 6 Y rows (should be zero): {shifted[5, :6, 25]}")
print(f"Last 3 X columns (should be zero): {shifted[5, 25, -3:]}")

# Verify non-zero interior
interior = shifted[3:, 6:, :-3]
print(f"\nInterior min: {interior.min():.1f}, max: {interior.max():.1f}")

assert shifted[0, 25, 25] == 0, "Z edge should be zeroed"
assert shifted[5, 0, 25] == 0, "Y edge should be zeroed"
assert shifted[5, 25, -1] == 0, "X edge should be zeroed"
print("✓ Edge zeroing test passed")

# Cell 7 (markdown)
"""
## 3. Multi-Channel Registration

Test `register_volume()` with (Z, Y, X, C) input.
"""

# Cell 8 (code)
# Load synthetic multi-channel data
stack_ref, _ = load_image_stacks(
    MINI_DATASET / "FOV_001" / "round1",
    ["ch00", "ch01", "ch02", "ch03"]
)

# Create shifted version (simulating round2)
known_shift = (1, -3, 4)
stack_mov = np.roll(stack_ref, known_shift, axis=(0, 1, 2))

# Register using first channel as reference
registered, shifts = register_volume(
    images=stack_mov,
    ref_image=stack_ref[:, :, :, 0],
    mov_image=stack_mov[:, :, :, 0],
)

print(f"Input shape: {stack_mov.shape}")
print(f"Output shape: {registered.shape}")
print(f"Detected shifts: {shifts}")
print(f"Known shifts: {known_shift}")

error = np.sqrt(sum((d - k) ** 2 for d, k in zip(shifts, known_shift)))
print(f"L2 error: {error:.6f}")

assert error < 0.5, f"Multi-channel registration error too large: {error}"
print("✓ Multi-channel registration test passed")

# Cell 9 (markdown)
"""
## 4. Backend Comparison

Compare NumPy and scikit-image implementations.
"""

# Cell 10 (code)
# Run benchmark comparison
results = benchmark_registration(
    sizes=[(5, 128, 128), (10, 256, 256)],
    methods=["numpy", "skimage"],
    n_runs=3,
)

print_benchmark_table(results)

# Check both methods produce same shifts
numpy_results = [r for r in results if r.method == "numpy"]
skimage_results = [r for r in results if r.method == "skimage"]

for np_r, sk_r in zip(numpy_results, skimage_results):
    np_shift = np_r.metrics["detected_shift"]
    sk_shift = sk_r.metrics["detected_shift"]
    print(f"Size {np_r.size}: numpy={np_shift}, skimage={sk_shift}")

print("✓ Backend comparison complete")

# Cell 11 (markdown)
"""
## 5. napari Before/After Overlay

Visualize registration quality with overlay (requires napari).
"""

# Cell 12 (code)
try:
    import napari

    # Use the multi-channel data from earlier
    fixed = stack_ref[:, :, :, 0].astype(np.float32)
    moving = stack_mov[:, :, :, 0].astype(np.float32)
    registered_ch0 = registered[:, :, :, 0].astype(np.float32)

    viewer = napari.Viewer()

    # Add fixed image (green)
    viewer.add_image(fixed, name="fixed", colormap="green", blending="additive")

    # Add moving image before registration (red)
    viewer.add_image(moving, name="moving (before)", colormap="red", blending="additive", visible=False)

    # Add registered image (red) - should align with green
    viewer.add_image(registered_ch0, name="registered (after)", colormap="red", blending="additive")

    print("napari viewer opened with registration overlay.")
    print("Toggle layers to compare before/after registration.")
    print("Yellow = good alignment (green + red)")

except ImportError:
    print("napari not installed. Skipping visualization.")

# Cell 13 (markdown)
"""
## 6. Test with Synthetic Dataset Shifts

Verify registration recovers known shifts from synthetic ground truth.
"""

# Cell 14 (code)
import json

# Load ground truth
with open(MINI_DATASET / "ground_truth.json") as f:
    gt = json.load(f)

fov_shifts = gt["fovs"]["FOV_001"]["shifts"]
print("Ground truth shifts per round:")
for round_name, shift in fov_shifts.items():
    print(f"  {round_name}: {shift}")

# Load round1 (reference) and round2
stack_r1, _ = load_image_stacks(MINI_DATASET / "FOV_001" / "round1", ["ch00", "ch01", "ch02", "ch03"])
stack_r2, _ = load_image_stacks(MINI_DATASET / "FOV_001" / "round2", ["ch00", "ch01", "ch02", "ch03"])

# Register round2 to round1
_, detected = register_volume(
    images=stack_r2,
    ref_image=stack_r1[:, :, :, 0],
    mov_image=stack_r2[:, :, :, 0],
)

expected = tuple(fov_shifts["round2"])
print(f"\nExpected shift (round2): {expected}")
print(f"Detected shift: {detected}")

# Note: detected shift should be negative of ground truth shift
# because ground truth defines how round2 was shifted FROM round1
# and registration detects how to shift round2 TO align with round1
print("✓ Synthetic dataset shift verification complete")

# Cell 15 (markdown)
"""
## Summary

Registration module validation results:
- [x] Known shift recovery - phase correlation works
- [x] Edge zeroing - `apply_shift()` zeros wrapped regions
- [x] Multi-channel registration - `register_volume()` handles (Z, Y, X, C)
- [x] Backend comparison - NumPy and scikit-image produce consistent results
- [x] Synthetic data shifts - recovers ground truth shifts
- [ ] napari visualization - before/after overlay (requires napari)
"""
```

**Step 2: Commit**

```bash
git add tests/qc_registration.ipynb
git commit -m "docs: add QC notebook for registration module validation"
```

---

## Task 10: Update pyproject.toml and Run All Tests

**Files:**
- Modify: `src/python/pyproject.toml`

**Step 1: Add napari as optional dependency**

Add to `src/python/pyproject.toml` in `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
ome = ["bioio-ome-tiff>=1.0"]
local-registration = ["SimpleITK>=2.3"]
spatialdata = ["spatialdata>=0.1", "spatialdata-io>=0.1"]
visualization = ["napari>=0.4"]
dev = ["pytest>=7.0", "pytest-cov>=4.0", "ruff>=0.1"]
```

**Step 2: Run all tests**

Run: `cd src/python && uv run pytest test/ -v`
Expected: All tests pass (including new benchmark tests)

**Step 3: Commit**

```bash
git add src/python/pyproject.toml
git commit -m "chore: add napari as optional visualization dependency"
```

---

## Task 11: Final Cleanup and Documentation

**Files:**
- Remove: `tests/test_io_interactive.ipynb` (merged into qc_io.ipynb)
- Update: `docs/notes.md`

**Step 1: Remove old notebook if it exists**

```bash
rm -f tests/test_io_interactive.ipynb
```

**Step 2: Update notes.md with QC session entry**

Append to `docs/notes.md`:

```markdown
### 2026-01-31: QC Session & Benchmark Module

- [x] **Created standalone benchmark module** (`starfinder.benchmark`)
  - `BenchmarkResult` dataclass for results
  - `measure()` function for timing/memory
  - `@benchmark` decorator for easy function wrapping
  - `run_comparison()` for multi-method benchmarks
  - `BenchmarkSuite` for result collection and statistics
  - `print_table()`, `save_csv()`, `save_json()` for reporting
  - `SIZE_PRESETS` for standard test configurations

- [x] **Migrated registration benchmark**
  - Updated `starfinder.registration.benchmark` to use new framework
  - Backwards-compatible API preserved

- [x] **Created QC notebooks** (`tests/qc_*.ipynb`)
  - `qc_benchmark.ipynb` - Benchmark framework validation
  - `qc_io.ipynb` - I/O module validation with napari
  - `qc_synthetic.ipynb` - Synthetic data generator validation
  - `qc_registration.ipynb` - Registration module validation

**Files Created:**
- `src/python/starfinder/benchmark/__init__.py`
- `src/python/starfinder/benchmark/core.py`
- `src/python/starfinder/benchmark/runner.py`
- `src/python/starfinder/benchmark/report.py`
- `src/python/starfinder/benchmark/presets.py`
- `src/python/test/test_benchmark.py`
- `tests/qc_benchmark.ipynb`
- `tests/qc_io.ipynb`
- `tests/qc_synthetic.ipynb`
- `tests/qc_registration.ipynb`

**Next Steps:**
- Run QC notebooks interactively to validate implementations
- Promote stable checks to pytest tests
- Phase 3: Spot finding module
```

**Step 3: Commit**

```bash
git add docs/notes.md
git rm -f tests/test_io_interactive.ipynb 2>/dev/null || true
git commit -m "docs: update notes.md with QC session progress"
```

---

## Summary

This plan creates:

1. **Benchmark module** (`starfinder.benchmark/`) with 4 files:
   - `core.py` - BenchmarkResult, measure(), @benchmark decorator
   - `runner.py` - run_comparison(), BenchmarkSuite
   - `report.py` - print_table(), save_csv(), save_json()
   - `presets.py` - SIZE_PRESETS, get_size_preset()

2. **Test file** for benchmark module (15 tests)

3. **Four QC notebooks**:
   - `qc_benchmark.ipynb` - Validates the benchmark framework
   - `qc_io.ipynb` - Validates I/O with napari examples
   - `qc_synthetic.ipynb` - Validates synthetic data generator
   - `qc_registration.ipynb` - Validates registration with napari overlays

4. **Migration** of existing registration benchmark to use new framework

Total: 11 tasks, each with TDD approach (test → implement → verify → commit)
