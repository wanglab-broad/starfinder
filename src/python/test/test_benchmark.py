"""Tests for benchmark core utilities."""

import time

import numpy as np

from starfinder.benchmark import (
    BenchmarkResult,
    BenchmarkSuite,
    benchmark,
    measure,
    run_comparison,
)


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
        # Order is: input[0]-method_a, input[0]-method_b, input[1]-method_a, input[1]-method_b
        assert results[0].method == "a"
        assert results[1].method == "b"
        assert results[2].method == "a"
        assert results[3].method == "b"


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


import json
from pathlib import Path


class TestReporting:
    """Tests for benchmark reporting utilities."""

    def test_print_table(self, capsys):
        """print_table() outputs formatted markdown table."""
        from starfinder.benchmark import print_table

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
        from starfinder.benchmark import save_csv

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
        from starfinder.benchmark import save_json

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
        import pytest
        with pytest.raises(ValueError, match="Unknown size preset"):
            get_size_preset("nonexistent")