"""Benchmark runner for comparing multiple methods."""

from __future__ import annotations

import json
import signal
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tifffile

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


# =============================================================================
# Registration Benchmark Runner
# =============================================================================

# Default benchmark data location
DEFAULT_BENCHMARK_DATA_DIR = Path(
    "/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/data"
)

# Preset order for early stopping (smallest to largest)
PRESET_ORDER = ["tiny", "small", "medium", "large", "xlarge", "tissue", "thick_medium"]


@contextmanager
def timeout_handler(seconds: int):
    """Context manager for timeout handling using SIGALRM.

    Args:
        seconds: Maximum seconds before timeout.

    Raises:
        TimeoutError: If execution exceeds timeout.

    Note:
        Only works on Unix systems. On Windows, this is a no-op.
    """
    def _handler(signum, frame):
        raise TimeoutError(f"Benchmark exceeded {seconds}s timeout")

    # Check if SIGALRM is available (Unix only)
    if hasattr(signal, "SIGALRM"):
        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows: no timeout support, just run
        yield


@dataclass
class BenchmarkPair:
    """Container for a benchmark ref/mov pair with metadata."""

    preset: str
    pair_type: str  # "shift" or deformation name
    ref: np.ndarray
    mov: np.ndarray
    ground_truth: dict
    ref_path: Path
    mov_path: Path


@dataclass
class RegistrationResult:
    """Result of a registration benchmark run."""

    preset: str
    pair_type: str
    method: str
    status: str  # "success", "timeout", "error: ..."
    time_seconds: float | None
    time_std: float | None
    memory_mb: float | None
    n_runs: int
    # Accuracy (for global registration with known ground truth)
    shift_detected: tuple[int, int, int] | None = None
    shift_error_l2: float | None = None
    # Quality metrics
    ncc_before: float | None = None
    ncc_after: float | None = None
    ssim_before: float | None = None
    ssim_after: float | None = None
    spot_iou_before: float | None = None
    spot_iou_after: float | None = None
    spot_match_rate: float | None = None
    # Paths to saved artifacts
    inspection_path: Path | None = None
    registered_path: Path | None = None
    metrics_path: Path | None = None


class RegistrationBenchmarkRunner:
    """Specialized benchmark runner for registration algorithms.

    This class orchestrates registration benchmarks with:
    - Loading benchmark pairs from generated data
    - Running global and local registration methods
    - Computing quality metrics before/after registration
    - Generating inspection images
    - Selective volume saving (failed/best/worst only)
    - Early stopping when methods timeout

    Example:
        >>> runner = RegistrationBenchmarkRunner()
        >>> results = runner.run_global_benchmark(
        ...     methods={"numpy_fft": phase_correlate},
        ...     presets=["tiny", "small", "medium"],
        ... )
    """

    def __init__(
        self,
        data_dir: Path | str = DEFAULT_BENCHMARK_DATA_DIR,
        results_dir: Path | str | None = None,
        timeout_seconds: int = 600,
        n_warmup: int = 1,
        n_repetitions: int = 3,
    ):
        """Initialize the registration benchmark runner.

        Args:
            data_dir: Path to benchmark data directory.
            results_dir: Path to save results. Defaults to data_dir/results.
            timeout_seconds: Maximum seconds per benchmark run.
            n_warmup: Number of warmup runs (discarded).
            n_repetitions: Number of timed runs for averaging.
        """
        self.data_dir = Path(data_dir)
        if results_dir:
            self.results_dir = Path(results_dir)
        else:
            # Default: sibling "results" dir next to data dir
            # e.g., .../starfinder_benchmark/data -> .../starfinder_benchmark/results
            self.results_dir = self.data_dir.parent / "results"
        self.timeout_seconds = timeout_seconds
        self.n_warmup = n_warmup
        self.n_repetitions = n_repetitions

        # Track timeout state for early stopping
        self._method_timeouts: dict[str, str] = {}  # method -> preset where it timed out

    def load_benchmark_pair(
        self,
        preset: str,
        pair_type: str = "shift",
        data_category: Literal["synthetic", "real"] = "synthetic",
    ) -> BenchmarkPair:
        """Load a benchmark ref/mov pair from generated data.

        Args:
            preset: Size preset name (tiny, small, medium, etc.) or dataset name for real.
            pair_type: "shift" or deformation name (polynomial_small, gaussian_large, etc.).
            data_category: "synthetic" or "real".

        Returns:
            BenchmarkPair with loaded images and metadata.

        Raises:
            FileNotFoundError: If benchmark data not found.
        """
        if data_category == "synthetic":
            preset_dir = self.data_dir / "synthetic" / preset
            ref_path = preset_dir / "ref.tif"

            if pair_type == "shift":
                mov_path = preset_dir / "mov_shift.tif"
            else:
                mov_path = preset_dir / f"mov_deform_{pair_type}.tif"

            gt_path = preset_dir / "ground_truth.json"
        else:
            # Real data
            preset_dir = self.data_dir / "real" / preset
            ref_path = preset_dir / "ref.tif"
            mov_path = preset_dir / "mov.tif"
            gt_path = preset_dir / "metadata.json"

        if not ref_path.exists():
            raise FileNotFoundError(f"Reference image not found: {ref_path}")
        if not mov_path.exists():
            raise FileNotFoundError(f"Moving image not found: {mov_path}")

        ref = tifffile.imread(ref_path)
        mov = tifffile.imread(mov_path)

        with open(gt_path) as f:
            full_gt = json.load(f)

        # Extract pair-specific ground truth
        if data_category == "synthetic" and "pairs" in full_gt:
            ground_truth = full_gt.get("pairs", {}).get(pair_type, {})
            ground_truth["preset"] = preset
            ground_truth["shape"] = full_gt.get("shape", list(ref.shape))
        else:
            ground_truth = full_gt

        return BenchmarkPair(
            preset=preset,
            pair_type=pair_type,
            ref=ref,
            mov=mov,
            ground_truth=ground_truth,
            ref_path=ref_path,
            mov_path=mov_path,
        )

    def generate_registration_inspection(
        self,
        ref: np.ndarray,
        mov: np.ndarray,
        registered: np.ndarray,
        metadata: dict,
        output_path: Path,
    ) -> None:
        """Generate a before/after registration inspection image.

        Delegates to evaluate.generate_inspection() for consistent
        inspection image generation across all backends.

        Args:
            ref: Reference volume (Z, Y, X).
            mov: Original moving volume (Z, Y, X).
            registered: Registered volume (Z, Y, X).
            metadata: Dict with preset, method, metrics info.
            output_path: Path to save the inspection image.
        """
        from starfinder.benchmark.evaluate import generate_inspection

        generate_inspection(ref, mov, registered, metadata, output_path)

    def should_save_volume(
        self,
        result: RegistrationResult,
        preset_results: list[RegistrationResult],
    ) -> bool:
        """Determine if registered volume should be saved.

        Saves volumes for:
        - Failed cases (status != "success")
        - Best result per preset (highest spot_iou_after)
        - Worst result per preset (lowest spot_iou_after)

        Args:
            result: Current result to check.
            preset_results: All results for this preset so far.

        Returns:
            True if volume should be saved.
        """
        # Always save failed cases
        if result.status != "success":
            return True

        # Get successful results with spot_iou
        successful = [
            r for r in preset_results
            if r.status == "success" and r.spot_iou_after is not None
        ]

        if not successful:
            return True  # First successful result

        spot_ious = [r.spot_iou_after for r in successful]
        current_iou = result.spot_iou_after

        if current_iou is None:
            return False

        # Save if best or worst
        return current_iou >= max(spot_ious) or current_iou <= min(spot_ious)

    def _compute_quality_metrics(
        self,
        ref: np.ndarray,
        mov: np.ndarray,
        registered: np.ndarray,
    ) -> dict:
        """Compute registration quality metrics.

        Delegates to evaluate.evaluate_registration() for consistent
        metric computation across all backends.

        Args:
            ref: Reference volume.
            mov: Original moving volume.
            registered: Registered volume.

        Returns:
            Dict with flattened before/after metrics.
        """
        from starfinder.benchmark.evaluate import evaluate_registration

        use_mip = ref.size > 100_000_000
        return evaluate_registration(ref, mov, registered, use_mip=use_mip)

    def _run_single_benchmark(
        self,
        pair: BenchmarkPair,
        method_name: str,
        method_fn: Callable,
        apply_fn: Callable,
        operation: str,
    ) -> tuple[RegistrationResult, np.ndarray | None]:
        """Run a single benchmark with timing and metrics.

        Args:
            pair: Benchmark pair to process.
            method_name: Name of the method.
            method_fn: Registration function (ref, mov) -> result.
            apply_fn: Function to apply result: (mov, result) -> registered.
            operation: "global" or "local".

        Returns:
            Tuple of (RegistrationResult, registered_volume or None).
        """
        ref, mov = pair.ref, pair.mov
        is_large = ref.size > 100_000_000  # >100M voxels

        times: list[float] = []
        memories: list[float] = []
        reg_result = None
        registered = None
        status = "success"

        try:
            # Warmup runs
            for _ in range(self.n_warmup):
                with timeout_handler(self.timeout_seconds):
                    _ = method_fn(ref, mov)

            # Timed runs
            n_runs = 1 if is_large else self.n_repetitions
            for _ in range(n_runs):
                with timeout_handler(self.timeout_seconds):
                    reg_result, elapsed, mem = measure(lambda: method_fn(ref, mov))
                    times.append(elapsed)
                    memories.append(mem)

            # Apply registration to get registered volume
            if reg_result is not None:
                registered = apply_fn(mov, reg_result)

        except TimeoutError:
            status = "timeout"
        except Exception as e:
            status = f"error: {type(e).__name__}: {str(e)[:50]}"

        # Compute quality metrics if we have a result
        quality = {}
        if registered is not None:
            try:
                quality = self._compute_quality_metrics(ref, mov, registered)
            except Exception as e:
                quality = {"metrics_error": str(e)}

        # Compute shift error for global registration
        shift_detected = None
        shift_error_l2 = None
        if operation == "global" and reg_result is not None:
            shift_detected = tuple(int(s) for s in reg_result)
            if "shift_zyx" in pair.ground_truth:
                gt_shift = pair.ground_truth["shift_zyx"]
                shift_error_l2 = float(np.sqrt(sum(
                    (d - g) ** 2 for d, g in zip(shift_detected, gt_shift)
                )))

        result = RegistrationResult(
            preset=pair.preset,
            pair_type=pair.pair_type,
            method=method_name,
            status=status,
            time_seconds=float(np.mean(times)) if times else None,
            time_std=float(np.std(times)) if len(times) > 1 else None,
            memory_mb=float(np.max(memories)) if memories else None,
            n_runs=len(times),
            shift_detected=shift_detected,
            shift_error_l2=shift_error_l2,
            ncc_before=quality.get("ncc_before"),
            ncc_after=quality.get("ncc_after"),
            ssim_before=quality.get("ssim_before"),
            ssim_after=quality.get("ssim_after"),
            spot_iou_before=quality.get("spot_iou_before"),
            spot_iou_after=quality.get("spot_iou_after"),
            spot_match_rate=quality.get("match_rate_after"),
        )

        return result, registered

    def run_global_benchmark(
        self,
        methods: dict[str, Callable],
        presets: list[str] | None = None,
        pair_types: list[str] | None = None,
        data_category: Literal["synthetic", "real"] = "synthetic",
        save_volumes: bool = True,
    ) -> list[RegistrationResult]:
        """Run global registration benchmark.

        Args:
            methods: Dict mapping method name to registration function.
                     Each function should take (ref, mov) and return shift tuple.
            presets: List of presets to benchmark. Defaults to PRESET_ORDER.
            pair_types: List of pair types. Defaults to ["shift"] for global.
            data_category: "synthetic" or "real".
            save_volumes: Whether to save registered volumes (selective).

        Returns:
            List of RegistrationResult objects.
        """
        from starfinder.registration import apply_shift

        if presets is None:
            presets = PRESET_ORDER if data_category == "synthetic" else ["cell_culture_3D", "tissue_2D", "LN"]
        if pair_types is None:
            pair_types = ["shift"]

        results: list[RegistrationResult] = []
        preset_results: dict[str, list[RegistrationResult]] = {}

        # Apply function for global registration
        def apply_global(mov, shift):
            # Negate shift to correct alignment
            correction = tuple(-s for s in shift)
            return apply_shift(mov, correction)

        for method_name, method_fn in methods.items():
            print(f"\n=== Method: {method_name} ===")

            for preset in presets:
                # Check early stopping
                if method_name in self._method_timeouts:
                    timeout_preset = self._method_timeouts[method_name]
                    print(f"  Skipping {preset} (timed out at {timeout_preset})")
                    continue

                for pair_type in pair_types:
                    print(f"  {preset}/{pair_type}...", end=" ", flush=True)

                    try:
                        pair = self.load_benchmark_pair(preset, pair_type, data_category)
                    except FileNotFoundError as e:
                        print(f"SKIP ({e})")
                        continue

                    result, registered = self._run_single_benchmark(
                        pair, method_name, method_fn, apply_global, "global"
                    )

                    # Track timeout for early stopping
                    if result.status == "timeout":
                        self._method_timeouts[method_name] = preset
                        print(f"TIMEOUT (will skip larger presets)")
                    elif result.status == "success":
                        print(f"OK ({result.time_seconds:.2f}s, err={result.shift_error_l2})")
                    else:
                        print(f"ERROR: {result.status}")

                    # Save artifacts
                    output_dir = self.results_dir / "global" / preset
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Always save inspection image
                    if registered is not None:
                        inspection_path = output_dir / f"inspection_{method_name}_{pair_type}.png"
                        self.generate_registration_inspection(
                            pair.ref, pair.mov, registered,
                            {
                                "preset": preset,
                                "method": method_name,
                                "pair_type": pair_type,
                                "status": result.status,
                                "time_seconds": result.time_seconds,
                                "ncc_before": result.ncc_before,
                                "ncc_after": result.ncc_after,
                                "ssim_before": result.ssim_before,
                                "ssim_after": result.ssim_after,
                                "spot_iou_before": result.spot_iou_before,
                                "spot_iou_after": result.spot_iou_after,
                            },
                            inspection_path,
                        )
                        result.inspection_path = inspection_path

                    # Save metrics JSON
                    metrics_path = output_dir / f"metrics_{method_name}_{pair_type}.json"
                    with open(metrics_path, "w") as f:
                        json.dump({
                            "preset": result.preset,
                            "pair_type": result.pair_type,
                            "method": result.method,
                            "status": result.status,
                            "time_seconds": result.time_seconds,
                            "time_std": result.time_std,
                            "memory_mb": result.memory_mb,
                            "shift_detected": result.shift_detected,
                            "shift_error_l2": result.shift_error_l2,
                            "ncc_before": result.ncc_before,
                            "ncc_after": result.ncc_after,
                            "ssim_before": result.ssim_before,
                            "ssim_after": result.ssim_after,
                            "spot_iou_before": result.spot_iou_before,
                            "spot_iou_after": result.spot_iou_after,
                            "spot_match_rate": result.spot_match_rate,
                            "ground_truth": pair.ground_truth,
                        }, f, indent=2)
                    result.metrics_path = metrics_path

                    # Selective volume saving
                    if save_volumes and registered is not None:
                        preset_key = f"{preset}_{pair_type}"
                        if preset_key not in preset_results:
                            preset_results[preset_key] = []

                        if self.should_save_volume(result, preset_results[preset_key]):
                            vol_path = output_dir / f"registered_{method_name}_{pair_type}.tif"
                            tifffile.imwrite(vol_path, registered, imagej=True, metadata={"axes": "ZYX"})
                            result.registered_path = vol_path

                        preset_results[preset_key].append(result)

                    results.append(result)

        return results

    def run_local_benchmark(
        self,
        methods: dict[str, Callable],
        presets: list[str] | None = None,
        pair_types: list[str] | None = None,
        data_category: Literal["synthetic", "real"] = "synthetic",
        save_volumes: bool = True,
    ) -> list[RegistrationResult]:
        """Run local (non-rigid) registration benchmark.

        Args:
            methods: Dict mapping method name to registration function.
                     Each function should take (ref, mov) and return displacement field.
            presets: List of presets to benchmark.
            pair_types: List of deformation types. Defaults to all deformations.
            data_category: "synthetic" or "real".
            save_volumes: Whether to save registered volumes (selective).

        Returns:
            List of RegistrationResult objects.
        """
        from starfinder.registration import apply_deformation
        from starfinder.benchmark.data import DEFORMATION_CONFIGS

        if presets is None:
            presets = PRESET_ORDER if data_category == "synthetic" else ["cell_culture_3D", "tissue_2D", "LN"]
        if pair_types is None:
            pair_types = list(DEFORMATION_CONFIGS.keys()) if data_category == "synthetic" else ["real"]

        results: list[RegistrationResult] = []
        preset_results: dict[str, list[RegistrationResult]] = {}

        for method_name, method_fn in methods.items():
            print(f"\n=== Method: {method_name} ===")

            for preset in presets:
                # Check early stopping
                if method_name in self._method_timeouts:
                    timeout_preset = self._method_timeouts[method_name]
                    print(f"  Skipping {preset} (timed out at {timeout_preset})")
                    continue

                for pair_type in pair_types:
                    # For real data, use the single real pair
                    actual_pair_type = "shift" if data_category == "real" else pair_type

                    print(f"  {preset}/{pair_type}...", end=" ", flush=True)

                    try:
                        pair = self.load_benchmark_pair(preset, actual_pair_type, data_category)
                        # Override pair_type for labeling
                        pair.pair_type = pair_type
                    except FileNotFoundError as e:
                        print(f"SKIP ({e})")
                        continue

                    result, registered = self._run_single_benchmark(
                        pair, method_name, method_fn, apply_deformation, "local"
                    )

                    # Track timeout for early stopping
                    if result.status == "timeout":
                        self._method_timeouts[method_name] = preset
                        print(f"TIMEOUT (will skip larger presets)")
                    elif result.status == "success":
                        iou_delta = (result.spot_iou_after or 0) - (result.spot_iou_before or 0)
                        print(f"OK ({result.time_seconds:.2f}s, IoU Δ={iou_delta:+.3f})")
                    else:
                        print(f"ERROR: {result.status}")

                    # Save artifacts
                    output_dir = self.results_dir / "local" / preset
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Always save inspection image
                    if registered is not None:
                        inspection_path = output_dir / f"inspection_{method_name}_{pair_type}.png"
                        self.generate_registration_inspection(
                            pair.ref, pair.mov, registered,
                            {
                                "preset": preset,
                                "method": method_name,
                                "pair_type": pair_type,
                                "status": result.status,
                                "time_seconds": result.time_seconds,
                                "ncc_before": result.ncc_before,
                                "ncc_after": result.ncc_after,
                                "ssim_before": result.ssim_before,
                                "ssim_after": result.ssim_after,
                                "spot_iou_before": result.spot_iou_before,
                                "spot_iou_after": result.spot_iou_after,
                            },
                            inspection_path,
                        )
                        result.inspection_path = inspection_path

                    # Save metrics JSON
                    metrics_path = output_dir / f"metrics_{method_name}_{pair_type}.json"
                    with open(metrics_path, "w") as f:
                        json.dump({
                            "preset": result.preset,
                            "pair_type": result.pair_type,
                            "method": result.method,
                            "status": result.status,
                            "time_seconds": result.time_seconds,
                            "time_std": result.time_std,
                            "memory_mb": result.memory_mb,
                            "ncc_before": result.ncc_before,
                            "ncc_after": result.ncc_after,
                            "ssim_before": result.ssim_before,
                            "ssim_after": result.ssim_after,
                            "spot_iou_before": result.spot_iou_before,
                            "spot_iou_after": result.spot_iou_after,
                            "spot_match_rate": result.spot_match_rate,
                            "ground_truth": pair.ground_truth,
                        }, f, indent=2)
                    result.metrics_path = metrics_path

                    # Selective volume saving
                    if save_volumes and registered is not None:
                        preset_key = f"{preset}_{pair_type}"
                        if preset_key not in preset_results:
                            preset_results[preset_key] = []

                        if self.should_save_volume(result, preset_results[preset_key]):
                            vol_path = output_dir / f"registered_{method_name}_{pair_type}.tif"
                            tifffile.imwrite(vol_path, registered, imagej=True, metadata={"axes": "ZYX"})
                            result.registered_path = vol_path

                        preset_results[preset_key].append(result)

                    results.append(result)

        return results

    def results_to_dataframe(self, results: list[RegistrationResult]):
        """Convert results to pandas DataFrame for analysis.

        Args:
            results: List of RegistrationResult objects.

        Returns:
            pandas DataFrame with all result fields.
        """
        import pandas as pd

        records = []
        for r in results:
            records.append({
                "preset": r.preset,
                "pair_type": r.pair_type,
                "method": r.method,
                "status": r.status,
                "time_seconds": r.time_seconds,
                "time_std": r.time_std,
                "memory_mb": r.memory_mb,
                "n_runs": r.n_runs,
                "shift_error_l2": r.shift_error_l2,
                "ncc_before": r.ncc_before,
                "ncc_after": r.ncc_after,
                "ncc_delta": (r.ncc_after or 0) - (r.ncc_before or 0),
                "ssim_before": r.ssim_before,
                "ssim_after": r.ssim_after,
                "spot_iou_before": r.spot_iou_before,
                "spot_iou_after": r.spot_iou_after,
                "spot_iou_delta": (r.spot_iou_after or 0) - (r.spot_iou_before or 0),
                "spot_match_rate": r.spot_match_rate,
            })

        return pd.DataFrame(records)

    def save_summary(self, results: list[RegistrationResult], output_path: Path | None = None) -> Path:
        """Save benchmark summary to CSV.

        Args:
            results: List of RegistrationResult objects.
            output_path: Path to save CSV. Defaults to results_dir/summary.csv.

        Returns:
            Path to saved file.
        """
        if output_path is None:
            output_path = self.results_dir / "summary.csv"

        df = self.results_to_dataframe(results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved summary: {output_path}")
        return output_path

    def run_parameter_tuning(
        self,
        param_grid: dict[str, list],
        presets: list[str] | None = None,
        pair_types: list[str] | None = None,
        data_category: Literal["synthetic", "real"] = "synthetic",
        n_repetitions: int = 1,
    ) -> tuple[list[RegistrationResult], dict[str, dict]]:
        """Run parameter tuning for demons registration.

        Generates all parameter combinations from the grid and benchmarks
        each configuration. Results are ranked by spot_match_rate.

        The loop is structured as (preset, pair_type, config) so each
        benchmark pair is loaded once and all configs run against it.

        Args:
            param_grid: Dict with keys 'method', 'iterations',
                'smoothing_sigma', each mapping to a list of values.
            presets: Presets to test on. Defaults to ["medium"].
            pair_types: Deformation types to test against. Defaults to
                all synthetic deformations, or ["shift"] for real data.
            data_category: "synthetic" or "real".
            n_repetitions: Timed runs per config (default 1 for speed).

        Returns:
            Tuple of (results, configs) where configs maps config name
            to its parameter dict.

        Example:
            >>> runner = RegistrationBenchmarkRunner()
            >>> results, configs = runner.run_parameter_tuning(
            ...     param_grid={
            ...         'method': ['demons', 'diffeomorphic'],
            ...         'iterations': [[25], [50], [100], [100, 50, 25]],
            ...         'smoothing_sigma': [0.5, 1.0],
            ...     },
            ...     presets=["medium"],
            ... )
        """
        from functools import partial
        from itertools import product

        from starfinder.registration import apply_deformation, demons_register

        if presets is None:
            presets = ["medium"]
        if pair_types is None:
            if data_category == "synthetic":
                from starfinder.benchmark.data import DEFORMATION_CONFIGS

                pair_types = list(DEFORMATION_CONFIGS.keys())
            else:
                pair_types = ["shift"]

        # Generate all parameter combinations
        methods_grid = param_grid.get("method", ["diffeomorphic"])
        iterations_grid = param_grid.get("iterations", [[50]])
        sigma_grid = param_grid.get("smoothing_sigma", [0.5])

        configs: dict[str, dict] = {}
        method_fns: dict[str, Callable] = {}

        for method, iterations, sigma in product(
            methods_grid, iterations_grid, sigma_grid
        ):
            iter_str = "-".join(map(str, iterations))
            name = f"{method}_iter{iter_str}_s{sigma}"
            configs[name] = {
                "method": method,
                "iterations": iterations,
                "smoothing_sigma": sigma,
            }
            method_fns[name] = partial(
                demons_register,
                method=method,
                iterations=iterations,
                smoothing_sigma=sigma,
            )

        total_runs = len(configs) * len(presets) * len(pair_types)
        print(
            f"Parameter tuning: {len(configs)} configs × "
            f"{len(presets)} presets × {len(pair_types)} pairs = "
            f"{total_runs} runs"
        )

        # Override repetitions for speed during tuning
        orig_reps = self.n_repetitions
        orig_warmup = self.n_warmup
        self.n_repetitions = n_repetitions
        self.n_warmup = 0

        results: list[RegistrationResult] = []
        run_count = 0

        try:
            for preset in presets:
                for pair_type in pair_types:
                    # Load each benchmark pair once for all configs
                    actual_pair_type = (
                        pair_type if data_category == "synthetic" else "shift"
                    )
                    try:
                        pair = self.load_benchmark_pair(
                            preset, actual_pair_type, data_category
                        )
                        if data_category == "real":
                            pair.pair_type = pair_type
                    except FileNotFoundError as e:
                        print(f"  SKIP {preset}/{pair_type}: {e}")
                        continue

                    for config_name, method_fn in method_fns.items():
                        run_count += 1
                        print(
                            f"  [{run_count}/{total_runs}] "
                            f"{preset}/{pair_type}/{config_name}...",
                            end=" ",
                            flush=True,
                        )

                        result, registered = self._run_single_benchmark(
                            pair, config_name, method_fn,
                            apply_deformation, "local",
                        )

                        if result.status == "success":
                            iou_delta = (
                                (result.spot_iou_after or 0)
                                - (result.spot_iou_before or 0)
                            )
                            match = result.spot_match_rate or 0
                            print(
                                f"OK ({result.time_seconds:.1f}s, "
                                f"IoU Δ={iou_delta:+.3f}, "
                                f"Match={match:.3f})"
                            )
                        else:
                            print(result.status)

                        # Save artifacts to tuning subdirectory
                        output_dir = self.results_dir / "tuning" / preset
                        output_dir.mkdir(parents=True, exist_ok=True)

                        if registered is not None:
                            inspection_path = (
                                output_dir
                                / f"inspection_{config_name}_{pair_type}.png"
                            )
                            self.generate_registration_inspection(
                                pair.ref,
                                pair.mov,
                                registered,
                                {
                                    "preset": preset,
                                    "method": config_name,
                                    "pair_type": pair_type,
                                    "status": result.status,
                                    "time_seconds": result.time_seconds,
                                    "ncc_before": result.ncc_before,
                                    "ncc_after": result.ncc_after,
                                    "ssim_before": result.ssim_before,
                                    "ssim_after": result.ssim_after,
                                    "spot_iou_before": result.spot_iou_before,
                                    "spot_iou_after": result.spot_iou_after,
                                },
                                inspection_path,
                            )
                            result.inspection_path = inspection_path

                        metrics_path = (
                            output_dir
                            / f"metrics_{config_name}_{pair_type}.json"
                        )
                        with open(metrics_path, "w") as f:
                            json.dump(
                                {
                                    "config": configs[config_name],
                                    "preset": result.preset,
                                    "pair_type": result.pair_type,
                                    "method": result.method,
                                    "status": result.status,
                                    "time_seconds": result.time_seconds,
                                    "memory_mb": result.memory_mb,
                                    "ncc_before": result.ncc_before,
                                    "ncc_after": result.ncc_after,
                                    "ssim_before": result.ssim_before,
                                    "ssim_after": result.ssim_after,
                                    "spot_iou_before": result.spot_iou_before,
                                    "spot_iou_after": result.spot_iou_after,
                                    "spot_match_rate": result.spot_match_rate,
                                },
                                f,
                                indent=2,
                            )
                        result.metrics_path = metrics_path

                        results.append(result)
        finally:
            self.n_repetitions = orig_reps
            self.n_warmup = orig_warmup

        # Save tuning summary with ranking
        tuning_dir = self.results_dir / "tuning"
        tuning_dir.mkdir(parents=True, exist_ok=True)
        self._save_tuning_summary(results, configs, tuning_dir)

        return results, configs

    def _save_tuning_summary(
        self,
        results: list[RegistrationResult],
        configs: dict[str, dict],
        output_dir: Path,
    ) -> None:
        """Save tuning results with aggregate ranking.

        Produces three files:
        - tuning_results.csv: All individual run results
        - tuning_ranking.csv: Configs ranked by mean Spot Match Rate
        - top3_configs.json: Top 3 configs for Phase 2 validation

        Args:
            results: All tuning results.
            configs: Mapping from config name to parameter dict.
            output_dir: Directory to save summary files.
        """
        import pandas as pd

        df = self.results_to_dataframe(results)

        # Save full results
        csv_path = output_dir / "tuning_results.csv"
        df.to_csv(csv_path, index=False)

        # Filter to successful runs for ranking
        successful = df[df["status"] == "success"].copy()

        if len(successful) == 0:
            print("No successful results to rank.")
            return

        # Aggregate: mean across pair_types and presets
        agg = (
            successful.groupby("method")
            .agg(
                mean_match_rate=("spot_match_rate", "mean"),
                mean_spot_iou=("spot_iou_after", "mean"),
                mean_iou_delta=("spot_iou_delta", "mean"),
                mean_ncc=("ncc_after", "mean"),
                mean_time=("time_seconds", "mean"),
                n_success=("status", "count"),
            )
            .reset_index()
        )

        agg = agg.sort_values(
            ["mean_match_rate", "mean_spot_iou"],
            ascending=[False, False],
        )
        agg.insert(0, "rank", range(1, len(agg) + 1))

        ranking_path = output_dir / "tuning_ranking.csv"
        agg.to_csv(ranking_path, index=False)

        # Print ranking table
        print(f"\n{'=' * 85}")
        print("PARAMETER TUNING RANKING (by mean Spot Match Rate)")
        print(f"{'=' * 85}")
        print(
            f"  {'Rank':<5s}  {'Config':<40s}  "
            f"{'Match':>6s}  {'IoU':>6s}  {'IoUΔ':>6s}  {'Time':>6s}"
        )
        print(f"  {'-' * 5}  {'-' * 40}  {'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 6}")
        for _, row in agg.iterrows():
            print(
                f"  #{int(row['rank']):<4d}  {row['method']:<40s}  "
                f"{row['mean_match_rate']:6.3f}  "
                f"{row['mean_spot_iou']:6.3f}  "
                f"{row['mean_iou_delta']:+5.3f}  "
                f"{row['mean_time']:5.1f}s"
            )
        print(f"{'=' * 85}")

        # Save top 3 configs for Phase 2
        top3 = []
        for _, row in agg.head(3).iterrows():
            config = configs.get(row["method"], {})
            top3.append(
                {
                    "name": row["method"],
                    "config": config,
                    "mean_match_rate": float(row["mean_match_rate"]),
                    "mean_spot_iou": float(row["mean_spot_iou"]),
                    "mean_iou_delta": float(row["mean_iou_delta"]),
                    "mean_time": float(row["mean_time"]),
                }
            )

        top3_path = output_dir / "top3_configs.json"
        with open(top3_path, "w") as f:
            json.dump(top3, f, indent=2)

        print(f"\nSaved: {csv_path}")
        print(f"Saved: {ranking_path}")
        print(f"Saved: {top3_path}")
