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
        size_str = "x".join(str(s) for s in r.size) if r.size else "-"
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
            size_str = "x".join(str(s) for s in r.size) if r.size else ""
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
