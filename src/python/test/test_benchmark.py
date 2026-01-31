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
