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
