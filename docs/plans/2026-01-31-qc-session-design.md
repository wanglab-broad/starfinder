# QC Session Design

## Overview

Before proceeding to Phase 3 (Spot Finding), validate all implemented Python modules through a combination of interactive Jupyter notebooks and automated pytest tests.

## Scope

- **I/O module** (`starfinder.io`) - TIFF loading/saving
- **Registration module** (`starfinder.registration`) - Phase correlation, shift application
- **Synthetic data generator** (`starfinder.testdata`) - Dataset generation, encoding
- **Benchmark module** (new) - Refactor from registration to standalone, extensible framework

## Deliverables

1. Standalone benchmark module (`starfinder/benchmark/`)
2. Four QC notebooks (`tests/qc_*.ipynb`)
3. Promoted pytest tests for stable checks

## Data

Synthetic datasets first (known ground truth), real data validation deferred to later.

---

## Benchmark Module Structure

Move and expand benchmark to `starfinder/benchmark/`:

```
src/python/starfinder/benchmark/
├── __init__.py          # Public API exports
├── core.py              # BenchmarkResult, timing/memory utilities
├── runner.py            # run_benchmark(), comparison logic
├── report.py            # print_table(), save_csv(), save_json()
└── presets.py           # Size presets, standard test configurations
```

### Key Components

- **`BenchmarkResult`** - Dataclass with: method, operation, size, time_seconds, memory_mb, metrics (dict for custom values like shift_error)
- **`@benchmark` decorator** - Wraps any function to capture timing and memory
- **`run_comparison()`** - Run multiple methods on same input, return list of results
- **`BenchmarkSuite`** - Collects results across multiple runs/sizes for statistical aggregation

### Design Goals

- Module-agnostic: can benchmark I/O, registration, spot-finding, etc.
- Pipeline support: benchmark full workflows, not just individual ops
- Comparison-focused: Python vs MATLAB, different Python backends
- Extensible: add new metrics without changing core API

---

## QC Notebooks Structure

```
tests/
├── qc_benchmark.ipynb    # Benchmark framework demo + validation
├── qc_io.ipynb           # I/O module validation
├── qc_synthetic.ipynb    # Synthetic data generator validation
├── qc_registration.ipynb # Registration module validation
```

### Notebook Pattern

Each notebook follows:

1. **Setup** - Imports, load synthetic data
2. **Visual inspection** - Display images, overlays, sanity checks
3. **Numerical validation** - Compare against ground truth or expected values
4. **Benchmark** - Time/memory measurements using new framework
5. **Summary** - Pass/fail checklist, notes for promotion to pytest

### Promotion Workflow

When a check is stable, copy the assertion logic to `src/python/test/test_*.py` as a proper pytest test.

---

## QC Checks Per Module

### qc_io.ipynb

- Load synthetic TIFF → verify shape, dtype (uint8), axis order (Z, Y, X)
- Load multi-channel stack → verify (Z, Y, X, C) shape
- Save and reload → roundtrip integrity check
- **napari**: `viewer.add_image(stack, name='loaded')` for 3D inspection
- Benchmark: load/save times for different sizes

### qc_synthetic.ipynb

- Generate dataset → verify folder structure, file existence
- Load ground_truth.json → overlay spots on max projection (matplotlib)
- Verify spot positions match expected coordinates
- Check two-base encoding: barcode → channel sequence mapping
- **napari**: `viewer.add_image()` + `viewer.add_points(spots_coords)` to verify spot placement in 3D
- Visual: ground_truth_annotation.png generation

### qc_registration.ipynb

- Create shifted volume (known shift) → register → verify recovered shift
- Apply shift → verify edge zeroing, data integrity
- Multi-channel registration → verify all channels aligned
- **napari**: before/after overlay with `viewer.add_image(fixed)` + `viewer.add_image(moving, colormap='red', blending='additive')`
- Benchmark: NumPy vs scikit-image timing/memory comparison

### qc_benchmark.ipynb

- Demo `@benchmark` decorator usage
- Demo `run_comparison()` across methods
- Demo report generation (table, CSV, JSON)
- Validate timing/memory measurements are reasonable

---

## Implementation Order

1. **Refactor benchmark module** - Create `starfinder/benchmark/`, migrate registration benchmark code
2. **Create qc_benchmark.ipynb** - Validate benchmark framework
3. **Create qc_io.ipynb** - I/O validation with napari
4. **Create qc_synthetic.ipynb** - Synthetic data validation with napari
5. **Create qc_registration.ipynb** - Registration validation with napari
6. **Promote stable checks** - Move to pytest

---

## Dependencies

Add to `pyproject.toml` (optional/dev):
```toml
napari = ">=0.4"
```

napari usage is for interactive inspection only; automated tests should not depend on it.
