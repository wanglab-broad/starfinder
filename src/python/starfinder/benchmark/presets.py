"""Benchmark presets for standard test configurations."""

from __future__ import annotations

from pathlib import Path

# Root benchmark directory and current task
#: Institutional benchmark root Path; override input/output paths on other hosts.
DEFAULT_BENCHMARK_DIR = Path(
    "/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark"
)
#: Default benchmark task subdirectory name.
BENCHMARK_TASK = "registration"
