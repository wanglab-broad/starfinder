"""Registration-specific benchmark utilities.

This module provides convenience functions for benchmarking registration
methods using the generic starfinder.benchmark framework.
"""

from __future__ import annotations

import numpy as np

from starfinder.benchmark.core import BenchmarkResult, measure
from starfinder.synthetic._presets import SIZE_PRESETS
from starfinder.registration import estimate_transform, TranslationConfig
from starfinder.image import ImageMetadata


def run_benchmark(
    sizes: list[tuple[int, int, int]] | None = None,
    configs: tuple[TranslationConfig, ...] = (TranslationConfig(), TranslationConfig(backend="skimage")),
    n_runs: int = 5,
    seed: int = 42,
) -> list[BenchmarkResult]:
    """
    Benchmark registration methods with synthetic images.

    Args:
        sizes: List of (Z, Y, X) sizes. Defaults to tiny, small, medium.
        configs: Translation configs; defaults to scipy_fft and skimage.
        n_runs: Number of runs per measurement.
        seed: Random seed for reproducibility.

    Returns:
        List of BenchmarkResult objects.
    """
    from starfinder.synthetic import generate_volume

    if sizes is None:
        sizes = [
            SIZE_PRESETS["tiny"],
            SIZE_PRESETS["small"],
            SIZE_PRESETS["medium"],
        ]

    rng = np.random.default_rng(seed)
    results = []

    for size in sizes:
        # Generate synthetic volume
        fixed = generate_volume(
            shape=size,
            n_spots=20,
            spot_intensity=200,
            background=20,
            seed=seed,
        )

        # Apply known shift (proportional to volume size to ensure overlap)
        max_z_shift = max(1, size[0] // 4)
        max_yx_shift = max(1, min(size[1], size[2]) // 4)
        known_shift = (
            int(rng.integers(-max_z_shift, max_z_shift + 1)),
            int(rng.integers(-max_yx_shift, max_yx_shift + 1)),
            int(rng.integers(-max_yx_shift, max_yx_shift + 1)),
        )
        moving = np.roll(fixed, known_shift, axis=(0, 1, 2))

        for config in configs:
            method_name = config.backend
            def func(reference, moving):
                return estimate_transform(reference, moving, config=config,
                    reference_metadata=ImageMetadata("benchmark/reference"),
                    moving_metadata=ImageMetadata("benchmark/moving"))

            # Warm-up
            _ = func(fixed, moving)

            # Timed runs
            times = []
            memories = []
            detected_shift = None

            for _ in range(n_runs):
                registration, elapsed, mem = measure(lambda: func(fixed, moving))
                detected_shift = tuple(-s for s in registration.transform.correction_zyx)
                times.append(elapsed)
                memories.append(mem)

            # Compute shift error
            from starfinder.evaluation.registration import evaluate_translation
            metadata = ImageMetadata("benchmark/displacement")
            error = evaluate_translation(
                {"moving": detected_shift}, {"moving": known_shift},
                reference_metadata=metadata, observed_metadata=metadata,
                units="voxel", tolerance=None).values["mean_error_l2"]

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

