# Plan: Rerun Global Registration Benchmark with Two-Phase System

## Context

The global registration benchmark results currently live in three overlapping directories with inconsistent naming, formats, and coverage:

| Directory | Datasets | Backend names | Has run JSON? | Phase 2 evaluated? |
|-----------|----------|---------------|---------------|---------------------|
| `global_python/` | 10 | `numpy_fft_shift`, `skimage_shift` | No | Old inline metrics |
| `global_matlab/` | 3 (real only) | `dft` | Legacy (`result_dft.json`) | Partial |
| `global_comparison/` | 10 | `python`, `matlab` | Legacy (`result_*.json`), no memory | No |

The two-phase system (Phase 1: `registered_*.tif` + `run_*.json`, Phase 2: `evaluate.py` → `metrics_*.json` + `inspection_*.png`) was designed to solve this. The Phase 1 scripts already exist for synthetic data but the real-data scripts are outdated (wrong paths, old naming, no memory measurement).

**Goal:** Clean rerun producing `global_python/` and `global_matlab/`, each with all 10 datasets in the two-phase format.

## Steps

### Step 1: Back up old global_python/
Rename `global_python/` → `global_python_v1/` to preserve old results for verification.

### Step 2: Update Python Phase 1 script (universal)
**File:** `.../scripts/benchmark_global_single.py` (overwrite existing synthetic-only version)

Changes:
- Add auto-detection of synthetic vs real via `REAL_DATASETS = {"cell_culture_3D", "tissue_2D", "LN"}`
- Use `mov.tif` for real datasets, `mov_shift.tif` for synthetic
- Set `data_category` field in output JSON
- Keep existing: `measure()` for memory, warmup, `run_python.json` naming

### Step 3: Update MATLAB Phase 1 script (universal)
**File:** `.../scripts/benchmark_global_single.m` (overwrite existing synthetic-only version)

Same changes as Python: add real dataset detection, correct mov path, `data_category` field.

### Step 4: Write Python orchestrator
**File:** `.../scripts/run_global_python.py`

Loops over all 10 datasets, calling `benchmark_global_single.py` for each:
```
tiny, small, medium, large, xlarge, tissue, thick_medium, cell_culture_3D, tissue_2D, LN
```

### Step 5: Write MATLAB orchestrator
**File:** `.../scripts/run_global_matlab.sh`

Same loop, calling `benchmark_global_single.m` via `matlab -batch`.

### Step 6: Run Phase 1
```bash
# Python (~5-10 min)
cd /home/unix/jiahao/Github/starfinder/src/python
uv run python .../scripts/run_global_python.py

# MATLAB (~15-20 min)
bash .../scripts/run_global_matlab.sh
```

### Step 7: Verify Phase 1
- Check 10 `registered_python.tif` + 10 `run_python.json` in `global_python/`
- Check 10 `registered_matlab.tif` + 10 `run_matlab.json` in `global_matlab/`
- Compare shifts between backends for synthetic (should match in magnitude)
- Cross-check against old `global_comparison/` results

### Step 8: Run Phase 2 (unified evaluation)
```bash
uv run python -m starfinder.benchmark.evaluate .../global_python/ --force
uv run python -m starfinder.benchmark.evaluate .../global_matlab/ --force
```

Produces per dataset: `metrics_python.json` + `inspection_python.png` (and matlab equivalents), plus `summary.csv` per backend tree.

### Step 9: Generate combined comparison summary
Small script merging both `summary.csv` files into `global_comparison.csv` with speedup and delta columns.

### Step 10: Clean up old directories
After verification:
- Delete `global_python_v1/` (~1.9 GB)
- Delete `global_comparison/` (~1.8 GB)
- Delete outdated scripts: `benchmark_global_real_single.py`, `benchmark_global_real_single.m`, `run_global_comparison.py`, `run_global_comparison_real.py`

## Key files

| File | Role |
|------|------|
| `.../scripts/benchmark_global_single.py` | Phase 1 Python (to update) |
| `.../scripts/benchmark_global_single.m` | Phase 1 MATLAB (to update) |
| `.../scripts/run_global_python.py` | Python orchestrator (new) |
| `.../scripts/run_global_matlab.sh` | MATLAB orchestrator (new) |
| `src/python/starfinder/benchmark/evaluate.py` | Phase 2 evaluator (no changes needed) |
| `src/python/starfinder/benchmark/core.py` | `measure()` function (no changes) |

Scripts location: `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/results/registration/scripts/`

## Verification

1. File counts: 10 `registered_*.tif` + 10 `run_*.json` per backend after Phase 1
2. File counts: 10 `metrics_*.json` + 10 `inspection_*.png` + `summary.csv` per backend after Phase 2
3. Shift consistency: Python vs MATLAB shifts should match on synthetic data (error_l2 = 0)
4. Metric consistency: NCC/SSIM values should be comparable to old `global_comparison/` results
5. Visual: spot-check inspection PNGs for tissue_2D and cell_culture_3D
