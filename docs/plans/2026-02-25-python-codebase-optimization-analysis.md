# Python Codebase Optimization Analysis

**Date**: 2026-02-25
**Scope**: `src/python/starfinder/` (~7,800 LOC, 7 modules)

## Context

Analysis of the Python codebase to identify structural improvements for maintainability, consistency, and efficiency. The codebase is well-architected overall — no circular dependencies, clean module layering, good type hint coverage (~90%). These recommendations are ordered by impact and grouped into tiers.

---

## Tier 1: Low-Risk DRY Fixes (4 items)

### 1.1 `fov.py`: Deduplicate `register_volume_local()` calls in `local_registration()`

**File**: `src/python/starfinder/dataset/fov.py` lines 289-376

The `register_volume_local()` call appears identically in both the TPS fallback `except` block (lines 352-359) and the `else` block (lines 363-371). Only difference: the `else` path passes `method=method`.

**Fix**: Extract a `_run_demons()` helper:
```python
def _run_demons(self, images, ref_3d, mov_3d, method="demons", **kwargs):
    from starfinder.registration import register_volume_local
    return register_volume_local(images, ref_3d, mov_3d, method=method, **kwargs)
```
Then both paths call `self._run_demons(...)`, eliminating 9 duplicated lines.

### 1.2 `fov.py`: Use `_apply_to_layers()` in `hist_equalize()`

**File**: `src/python/starfinder/dataset/fov.py` lines 159-177

`hist_equalize()` manually iterates over layers (lines 170-176) instead of using `_apply_to_layers()`. The only complication is that `histogram_match()` needs a `reference` volume — this can be captured in a lambda.

**Fix**:
```python
reference = self.images[self.layers.ref][:, :, :, ref_channel]
self._apply_to_layers(
    lambda v: histogram_match(v, reference, nbins=nbins), layers
)
```

### 1.3 `fov.py`: Add type hint to `_apply_to_layers()` callback

**File**: `src/python/starfinder/dataset/fov.py`

The `func` parameter lacks a type annotation.

**Fix**: `func: Callable[[np.ndarray], np.ndarray]` (import from `collections.abc`)

### 1.4 `pyproject.toml`: Remove deprecated `[project.optional-dependencies].dev`

**File**: `src/python/pyproject.toml`

The `dev` key under `[project.optional-dependencies]` is superseded by `[dependency-groups].dev` (PEP 735). Having both is confusing.

**Fix**: Remove the optional-dependencies `dev` entry, keep only `[dependency-groups]`.

---

## Tier 2: Named Constants & Validation (3 items)

### 2.1 Extract magic numbers into module-level constants

Currently scattered across files:

| Constant | Current location | Value | Meaning |
|----------|-----------------|-------|---------|
| `1.4826` | `spotfinding/local_maxima.py:88` | MAD → σ conversion | Gaussian assumption |
| `radius=3` | `preprocessing/morphology.py:16,50` | Structuring element | Morphology default |
| `99.5` | `registration/metrics.py:148,264` | Percentile threshold | Spot detection |
| `99.0` | `registration/metrics.py:97` | Percentile threshold | SSIM masking |
| `order=2` | `registration/pyramid.py:21` | Butterworth order | Anti-alias filter |

**Fix**: Add named constants at the top of each respective file (not a separate constants file — keeps locality):
```python
# spotfinding/local_maxima.py
MAD_TO_SIGMA = 1.4826  # Converts MAD to σ under Gaussian assumption

# registration/metrics.py
SPOT_THRESHOLD_PERCENTILE = 99.5
SSIM_THRESHOLD_PERCENTILE = 99.0
```

### 2.2 Add validation for `intensity_threshold` by mode

**File**: `src/python/starfinder/spotfinding/local_maxima.py` lines 16-47

`intensity_threshold` means k-sigma (typically 3-5) in `"noise"` mode but a fraction (0-1) in `"adaptive"` mode. Users can silently pass wrong values.

**Fix**: Add a soft warning (not an error, for backward compat):
```python
import warnings
if intensity_estimation in ("adaptive", "adaptive_round", "global") and intensity_threshold > 1.0:
    warnings.warn(
        f"intensity_threshold={intensity_threshold} looks like a k-sigma value, "
        f"but mode '{intensity_estimation}' expects a fraction (0-1).",
        stacklevel=2,
    )
```

### 2.3 Narrow bare `except Exception` in benchmark modules

**Files**: `benchmark/runner.py:473`, `benchmark/data.py:866`

Bare `except Exception` blocks catch too broadly. The benchmark runner intentionally swallows errors for fault tolerance, but could be more specific.

**Fix**: Catch `(TimeoutError, MemoryError, RuntimeError, ValueError)` instead of `Exception` in runner.py. In data.py, catch `(FileNotFoundError, tifffile.TiffFileError)`.

---

## Tier 3: Structural Improvements (3 items)

### 3.1 `benchmark/runner.py`: Extract artifact-saving helper (1149 lines → ~950)

**File**: `src/python/starfinder/benchmark/runner.py`

Inspection + metrics JSON saving code is duplicated in 3 places:
- `run_global_benchmark()` lines 590-633
- `run_local_benchmark()` lines 727-768
- `run_parameter_tuning()` lines 984-1039

**Fix**: Extract `_save_benchmark_artifacts(result, registered, pair, output_dir, metadata)` method (~40 lines) that handles:
- `generate_registration_inspection()` call
- Metrics JSON dump
- Optional volume saving via `should_save_volume()`

Early stopping logic (lines 558-562 and 689-693) can also be extracted. Combined savings: ~200 lines.

### 3.2 Consolidate preset definitions between `benchmark/` and `testdata/`

**Overlap**: `benchmark/presets.py` has `SIZE_PRESETS`, `SPOT_COUNTS`, `SHIFT_RANGES` as flat dicts. `testdata/synthetic.py` has `get_preset_config()` returning `SyntheticConfig` objects with overlapping size/spot definitions.

**Fix**: Make `testdata/synthetic.py` presets reference `benchmark/presets.py` for sizes:
```python
from starfinder.benchmark.presets import SIZE_PRESETS
# Use SIZE_PRESETS["small"] for shape, etc.
```
Or vice versa. Single source of truth for volume dimensions and spot counts.

### 3.3 `benchmark/runner.py`: Consider splitting by responsibility

The file has 10+ distinct responsibilities. If it continues growing, consider splitting into:
- `runner_io.py` — `load_benchmark_pair()`, `BenchmarkPair`, data loading
- `runner_artifacts.py` — inspection generation, metrics saving, volume saving decisions
- `runner.py` — core orchestration (global/local/tuning benchmark loops)

**Note**: This is optional and only worthwhile if the file continues to grow. Current size (1149 lines) is at the threshold but not critical.

---

## Tier 4: Nice-to-Have (3 items, defer)

### 4.1 Add edge-case tests

Missing coverage for:
- Empty DataFrames in `barcode/filtering.py`
- All-zero channels in `preprocessing/normalization.py`
- Volumes smaller than structuring element in `preprocessing/morphology.py`
- Out-of-bounds spot coordinates in `barcode/extraction.py`

### 4.2 Expand `utils.py` or remove it

Currently 31 lines with 1 function (`make_projection`). Either:
- Move `make_projection` to `preprocessing/` and delete `utils.py`
- Or add shared validation helpers (e.g., `validate_image_ndim()`)

### 4.3 Standardize logging across modules

Only `dataset/fov.py` uses `@log_step` for structured logging. Other modules (`io/tiff.py`, `registration/`) use `logging.getLogger(__name__)` directly. Not causing issues, but inconsistent.

---

## What's Already Good (no changes needed)

| Aspect | Assessment |
|--------|-----------|
| Module layering | Clean: io → preprocessing → registration → spotfinding → barcode → dataset |
| Import patterns | No circular deps; lazy imports for optional deps (SimpleITK, bioio) |
| `registration/` API | `tps_register()` vs `register_volume_tps()` is NOT redundant — different abstraction levels (field computation vs multi-channel application). Same pattern as `demons_register()` / `register_volume_local()` |
| `@log_step` decorator | Correct: re-raises exceptions, preserves metadata via `@wraps` |
| Top-level `__init__.py` | Well-organized `__all__` with both module and function exports |
| Error messages | Consistent `f"Description: {context}"` pattern across 28 `ValueError` instances |
| Type hints | 85-95% coverage on public API, excellent for scientific Python |
| `FOV.run_streaming()` | Minimal duplication vs batch path — clean design leveraging fluent API |

---

## Verification

For any implemented changes:
1. Run `cd src/python && uv run pytest test/ -v` — all 155+ tests must pass
2. Run `uv run pytest test/ -v --cov=starfinder` — verify no coverage regression
3. For benchmark/runner.py changes: run a quick benchmark to verify artifact output unchanged
