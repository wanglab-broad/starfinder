# Two-Phase Registration Benchmark Workflow

## Context

The current benchmark tightly couples algorithm execution with metric computation in `runner.py`. This means:
- Python benchmarks compute metrics inline using `_compute_quality_metrics()` during the run
- MATLAB benchmarks save only registered images + basic JSON, then metrics are computed post-hoc by separate evaluation scripts
- Different code paths evaluate different backends → unfair comparison
- MATLAB results are incomplete (no inspection images, no per-dataset metrics for most datasets)

**Goal:** Decouple "run" from "evaluate" so all backends are assessed identically by the same Python evaluation code.

## Design

```
Phase 1: RUN (backend-specific)           Phase 2: EVALUATE (unified Python)
┌──────────────────────────────┐          ┌─────────────────────────────────┐
│ Python: runner.py            │          │ evaluate.py                     │
│   run algorithm              │   save   │   load ref.tif + registered.tif │
│   record time + memory       │ ──────→  │   compute NCC, SSIM, IoU, MR   │
│   save registered.tif        │          │   generate inspection.png       │
│   save run_{backend}.json    │          │   save metrics_{backend}.json   │
├──────────────────────────────┤          │   save inspection_{backend}.png │
│ MATLAB: benchmark_*.m        │   save   │                                 │
│   run algorithm              │ ──────→  │   (same code for ALL backends)  │
│   record time + memory       │          └─────────────────────────────────┘
│   save registered.tif        │
│   save run_{backend}.json    │
└──────────────────────────────┘
```

### Output Structure (Option B: Separate Trees)

Each backend gets its own self-contained directory tree with identical structure.
The evaluator is called once per backend tree.

```
results/registration/
├── global_python/                      # Python backend results
│   ├── tiny/
│   │   ├── registered_python.tif       # Phase 1: registered volume
│   │   ├── run_python.json             # Phase 1: time, memory, shift, status
│   │   ├── metrics_python.json         # Phase 2: NCC, SSIM, IoU, Match Rate
│   │   └── inspection_python.png       # Phase 2: green-magenta overlay
│   ├── small/
│   │   └── ... (same 4 files)
│   ├── medium/
│   ├── large/
│   ├── xlarge/
│   ├── tissue/
│   ├── thick_medium/
│   ├── cell_culture_3D/
│   ├── tissue_2D/
│   ├── LN/
│   └── summary.csv                     # Per-backend summary
│
├── global_matlab/                      # MATLAB backend results
│   ├── tiny/
│   │   ├── registered_matlab.tif
│   │   ├── run_matlab.json
│   │   ├── metrics_matlab.json
│   │   └── inspection_matlab.png
│   ├── small/
│   │   └── ... (same 4 files)
│   ├── ...
│   ├── LN/
│   └── summary.csv
│
├── local_python/                       # Same pattern for local registration
│   └── ...
├── local_matlab/
│   └── ...
│
└── comparison/                         # Cross-backend comparison (generated)
    ├── global_comparison.csv           # Side-by-side metrics table
    └── local_comparison.csv
```

**Per dataset:** 4 files (registered TIF + run JSON + metrics JSON + inspection PNG)
**Per backend:** 10 datasets × 4 files + 1 summary = 41 files
**Total for global:** 2 backends × 41 files + comparison CSV = 83 files

## Implementation Steps

### Step 1: Create `evaluate.py` module

**New file:** `src/python/starfinder/benchmark/evaluate.py`

Core functions:
- `evaluate_registration(ref, mov_before, registered, skip_ssim=False)` → flat metrics dict
  - Calls `registration_quality_report()` from `metrics.py` and flattens the result
  - Extracted from `RegistrationBenchmarkRunner._compute_quality_metrics()` (runner.py:499-529)
- `generate_inspection(ref, mov_before, registered, metadata, output_path)` → saves PNG
  - Extracted from `RegistrationBenchmarkRunner.generate_registration_inspection()` (runner.py:331-496)
  - Made into a standalone function (no class instance needed)
- `evaluate_directory(result_dir, data_dir, force=False, skip_ssim_above=100_000_000)` → list of dicts
  - Called once per backend tree (e.g., `global_python/` or `global_matlab/`)
  - Scans `result_dir/{dataset}/` for `registered_*.tif` files
  - For each: resolves ref/mov paths from `data_dir`, computes metrics, saves artifacts
  - Skips files that already have corresponding `metrics_*.json` (unless `force=True`)
  - Generates per-backend `summary.csv` at `result_dir/summary.csv`
- `_resolve_data_paths(registered_path, data_dir)` → (ref_path, mov_path, dataset, pair_type)
  - Infers dataset from parent dir name (e.g., `global_python/small/` → dataset="small")
  - Checks if dataset exists under `data_dir/synthetic/` or `data_dir/real/`
  - Pair type from `run_*.json` if available, else defaults to "shift" for global dirs

Also add `if __name__ == "__main__"` CLI:
```
uv run python -m starfinder.benchmark.evaluate <result_dir> [--data-dir ...] [--force]
```

### Step 2: Refactor `runner.py` — decouple metrics from run

**File:** `src/python/starfinder/benchmark/runner.py`

Changes:
1. **Update path** (line 142): `registration_benchmark` → `starfinder_benchmark`
2. **Add `evaluate=True` parameter** to `_run_single_benchmark()`:
   - When `evaluate=False`: skip `_compute_quality_metrics()`, leave metric fields as `None`
   - Default `True` for backward compatibility
3. **Add `evaluate` parameter** to `run_global_benchmark()` and `run_local_benchmark()`:
   - When `evaluate=False`: run Phase 1 only — save `registered.tif` + `run_{method}.json`
   - When `evaluate=True` (default): current behavior (both phases inline)
4. **Replace inline metric/inspection code** with calls to `evaluate.py` functions:
   - `_compute_quality_metrics()` → delegates to `evaluate.evaluate_registration()`
   - `generate_registration_inspection()` → delegates to `evaluate.generate_inspection()`
   - The runner methods become thin wrappers

### Step 3: Update `__init__.py` exports

**File:** `src/python/starfinder/benchmark/__init__.py`

Add:
```python
from starfinder.benchmark.evaluate import (
    evaluate_registration,
    evaluate_directory,
    generate_inspection,
)
```

### Step 4: Update MATLAB script output format

**Files** (on network mount, use Write tool):
- `starfinder_benchmark/results/registration/scripts/benchmark_global_single.m`
- `starfinder_benchmark/results/registration/scripts/benchmark_local_single.m`

Changes:
- Rename output from `result_matlab.json` → `run_matlab.json`
- Update data path from `registration_benchmark` → `starfinder_benchmark`
- Add `pair_type` and `data_category` fields to JSON output
- **Add memory measurement** using `/proc/self/status` (Linux):
  ```matlab
  function mem_kb = get_vm_rss()
  %GET_VM_RSS Read current RSS from /proc/self/status (Linux only).
      fid = fopen('/proc/self/status', 'r');
      mem_kb = 0;
      while ~feof(fid)
          line = fgetl(fid);
          if startsWith(line, 'VmRSS:')
              tokens = regexp(line, '(\d+)', 'tokens');
              mem_kb = str2double(tokens{1}{1});
              break;
          end
      end
      fclose(fid);
  end
  ```
  Usage pattern in benchmark scripts:
  ```matlab
  rss_before = get_vm_rss();
  tic;
  [params, regImg] = DFTRegister3D(ref_dbl, mov_dbl);
  internal_time = toc;
  rss_after = get_vm_rss();
  memory_mb = (rss_after - rss_before) / 1024;
  ```
  Note: `memory()` is Windows-only. VmRSS delta gives the memory allocated during the algorithm, comparable to Python's `tracemalloc` approach. Include `memory_mb` in the JSON output.

### Step 5: Update Python single-run scripts

**Files** (on network mount):
- `starfinder_benchmark/results/registration/scripts/benchmark_global_single.py`
- `starfinder_benchmark/results/registration/scripts/benchmark_local_single.py`

Changes:
- Rename output from `result_python.json` → `run_python.json`
- Update data path
- Save `registered_python.tif` always (not selectively)

### Step 6: Re-evaluate existing results

Run the evaluator once per backend tree to backfill missing metrics/inspection:
```bash
BASE=.../starfinder_benchmark/results/registration

# Evaluate each backend tree independently
uv run python -m starfinder.benchmark.evaluate $BASE/global_python/ --force
uv run python -m starfinder.benchmark.evaluate $BASE/global_matlab/ --force
uv run python -m starfinder.benchmark.evaluate $BASE/local_python/  --force
uv run python -m starfinder.benchmark.evaluate $BASE/local_matlab/  --force

# Also backfill legacy comparison directories (will be deprecated going forward)
uv run python -m starfinder.benchmark.evaluate $BASE/global_comparison/ --force
uv run python -m starfinder.benchmark.evaluate $BASE/local_comparison/  --force
```

## Key Design Decisions

1. **Separate trees per backend (Option B):** Each backend gets its own directory tree (`global_python/`, `global_matlab/`). Self-contained, easy to add/remove backends, matches current layout. Cross-backend comparison generated separately as a summary CSV.
2. **New `evaluate.py` vs extending `runner.py`:** New module. Runner is already ~1000 lines; evaluation is a different responsibility (disk-based input vs in-memory). Clean separation.
3. **Backward compatible:** `evaluate=True` default preserves existing behavior. Existing callers unaffected.
4. **Extract, don't duplicate:** `_compute_quality_metrics()` and `generate_registration_inspection()` move to `evaluate.py` as standalone functions; runner delegates to them.
5. **SSIM skip for large volumes:** `skip_ssim_above=100_000_000` voxels (tissue_2D is 283M). SSIM takes 20+ min on large float64 volumes.
6. **MATLAB memory via VmRSS:** Read `/proc/self/status` before/after algorithm to get RSS delta, comparable to Python's `tracemalloc` approach.

## Critical Files

| File | Action | Purpose |
|------|--------|---------|
| `src/python/starfinder/benchmark/evaluate.py` | **Create** | Core Phase 2 evaluator |
| `src/python/starfinder/benchmark/runner.py` | Edit | Decouple metrics, update path, delegate to evaluate.py |
| `src/python/starfinder/benchmark/__init__.py` | Edit | Add evaluate exports |
| `src/python/starfinder/registration/metrics.py` | Read only | Reused by evaluate.py (no changes) |
| `.../scripts/benchmark_global_single.{m,py}` | Edit | Update output format + paths |
| `.../scripts/benchmark_local_single.{m,py}` | Edit | Update output format + paths |

## Verification

1. `uv run pytest test/ -v` — existing tests still pass (backward compatibility)
2. Run evaluate CLI on `global_matlab/` — should produce `metrics_matlab.json` + `inspection_matlab.png` for all 3 real datasets
3. Spot-check: metrics from evaluate.py on `global_python/` results should match existing `metrics_*.json` values (same computation path)
