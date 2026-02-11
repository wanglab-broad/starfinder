# Plan: Rerun Local Registration Benchmarks (Two-Phase)

## Context

Local registration benchmarks exist in a fragmented state across multiple directories (`local_comparison/`, `local_tuning/`, `local_matlab/`) with inconsistent naming (`result_*.json` vs `run_*.json`), incomplete coverage, and no unified Phase 2 evaluation. The goal is to rerun everything cleanly using the two-phase strategy that works well for global registration:

- **Phase 1**: Run local registration (Python SimpleITK demons + MATLAB imregdemons) and save `registered_*.tif` + `run_*.json`
- **Phase 2**: Run `evaluate_directory()` to compute metrics uniformly for all backends

**Scope**: All 7 synthetic presets × 5 deformation pair types + 3 real datasets = 38 runs per backend, 76 total. Large volumes (tissue, tissue_2D) may timeout — that's expected and handled gracefully.

## Output naming convention

For multiple pair types per dataset, encode the pair type in the backend label:

```
local_python/
├── tiny/
│   ├── registered_python_polynomial_small.tif
│   ├── run_python_polynomial_small.json
│   ├── registered_python_polynomial_large.tif
│   ├── run_python_polynomial_large.json
│   ├── registered_python_gaussian_small.tif
│   ├── ... (5 deformation pairs)
├── cell_culture_3D/
│   ├── registered_python.tif        # real datasets: single pair
│   └── run_python.json
```

This works with `evaluate_directory()` because:
- `glob("*/registered_*.tif")` finds all files
- `_find_run_json()` maps `registered_python_polynomial_small.tif` → `run_python_polynomial_small.json`
- `_resolve_data_paths()` reads `pair_type` from the run JSON to locate the correct `mov_deform_*.tif`

## Implementation

### Step 0: Clean up old local benchmark files

Delete stale results from previous ad-hoc runs so `local_python/` and `local_matlab/` start fresh:

```bash
# 1. Clear local_python/ (currently empty, but ensure clean)
rm -rf .../results/registration/local_python/*

# 2. Clear local_matlab/ old parameter-sweep results (non-standard naming)
#    Contains: cell_culture_3D/, LN/ with mixed configs like
#    registered_matlab_default.tif, registered_single_iter25_afs0.5.tif, field_*.mat
rm -rf .../results/registration/local_matlab/*

# 3. Archive local_comparison/ and local_tuning/ (historical reference)
#    These used old naming (result_*.json) and different methodology.
#    Move to an archive folder rather than delete, in case we need to reference them.
mkdir -p .../results/registration/archive/
mv .../results/registration/local_comparison .../results/registration/archive/
mv .../results/registration/local_tuning .../results/registration/archive/
```

**What's being cleaned and why:**

| Directory | Contents | Action | Reason |
|-----------|----------|--------|--------|
| `local_python/` | Empty | No-op | Already clean |
| `local_matlab/` | cell_culture_3D, LN with 4+ configs each, `.mat` fields, `result_*.json` naming | Delete contents | Non-standard naming, mixed configs, incompatible with evaluate_directory |
| `local_comparison/` | 6 synthetic presets, Python+MATLAB, `result_*.json` | Archive | Useful historical reference but superseded by this rerun |
| `local_tuning/` | 4 datasets, 24-config grid search, `tuning_ranking.csv` | Archive | Parameter tuning results still valuable for reference |

### Step 1: Update `benchmark_local_single.py`

**File:** `scripts/benchmark_local_single.py`

Add `pair_type` parameter (default: `polynomial_small`). Handle real vs synthetic data categories. Use timeout-safe naming.

```python
def run_single(dataset_name: str, pair_type: str = "polynomial_small"):
    # Determine data category
    data_category = "real" if dataset_name in REAL_DATASETS else "synthetic"

    # Resolve paths
    if data_category == "real":
        mov_path = DATA_DIR / "real" / dataset_name / "mov.tif"
        backend_label = "python"
    else:
        mov_path = DATA_DIR / "synthetic" / dataset_name / f"mov_deform_{pair_type}.tif"
        backend_label = f"python_{pair_type}"

    # Output naming
    reg_path = output_dir / f"registered_{backend_label}.tif"
    run_path = output_dir / f"run_{backend_label}.json"

    # Run JSON includes pair_type for evaluate.py resolution
    result = {..., "pair_type": pair_type, ...}
```

Config: `symmetric`, `iterations=[25]`, `sigma=0.5` (best from tuning on Match Rate).

### Step 2: Create `run_local_python.py`

**File:** `scripts/run_local_python.py`

Runner script that loops over all presets × pair types with 600s timeout per run.

```python
SYNTHETIC_PRESETS = ["tiny", "small", "medium", "large", "xlarge", "tissue", "thick_medium"]
DEFORMATION_TYPES = ["polynomial_small", "polynomial_large", "gaussian_small", "gaussian_large", "multi_point"]
REAL_DATASETS = ["cell_culture_3D", "tissue_2D", "LN"]

# Build run list: 7 presets × 5 deformations + 3 real = 38 runs
for preset in SYNTHETIC_PRESETS:
    for deform in DEFORMATION_TYPES:
        run(preset, deform, timeout=600)

for dataset in REAL_DATASETS:
    run(dataset, pair_type="real", timeout=600)
```

Pattern: follows `run_global_python.py` (subprocess calls to `benchmark_local_single.py`).

### Step 3: Update `benchmark_local_single.m`

**File:** `scripts/benchmark_local_single.m`

Add `pair_type` parameter. Use auto pyramid levels (`floor(log2(dimZ))`), `iterations=25`, `AFS=1.3` (MATLAB defaults that work well with pyramids).

```matlab
function benchmark_local_single(preset_name, pair_type, data_dir)
    % Resolve mov path based on pair_type
    if strcmp(data_category, 'real')
        mov_path = fullfile(data_dir, 'real', preset_name, 'mov.tif');
        backend_label = 'matlab';
    else
        mov_path = fullfile(data_dir, 'synthetic', preset_name, ['mov_deform_' pair_type '.tif']);
        backend_label = ['matlab_' pair_type];
    end
```

### Step 4: Create `run_local_matlab.sh`

**File:** `scripts/run_local_matlab.sh`

Shell wrapper that loops MATLAB calls, same structure as `run_global_matlab.sh` but with nested preset × pair_type loop.

### Step 5: Execute Phase 1

```bash
# Python (from src/python/)
cd /home/unix/jiahao/Github/starfinder/src/python
uv run python .../scripts/run_local_python.py

# MATLAB
cd .../scripts
bash run_local_matlab.sh
```

Estimated runtime:
- Python: ~1-3 hours (tiny-xlarge feasible, tissue/tissue_2D may timeout)
- MATLAB: ~2-4 hours (slower per-run but pyramids help on some presets)

### Step 6: Execute Phase 2

```bash
cd /home/unix/jiahao/Github/starfinder/src/python

# Evaluate Python local results
uv run python -m starfinder.benchmark.evaluate \
    .../results/registration/local_python/ --force

# Evaluate MATLAB local results
uv run python -m starfinder.benchmark.evaluate \
    .../results/registration/local_matlab/ --force
```

This generates `metrics_*.json`, `inspection_*.png`, and `summary.csv` for each backend.

## Key files

| File | Action |
|------|--------|
| `scripts/benchmark_local_single.py` | Update: add pair_type param, handle real datasets |
| `scripts/run_local_python.py` | Create: orchestrator looping presets × pair_types |
| `scripts/benchmark_local_single.m` | Update: add pair_type param, handle real datasets |
| `scripts/run_local_matlab.sh` | Create: shell orchestrator for MATLAB |
| `evaluate.py` | No changes needed (already handles this naming) |

## Verification

1. **Phase 1 output**: Check `local_python/` and `local_matlab/` have `registered_*.tif` + `run_*.json` per dataset/pair_type
2. **Phase 2 output**: Check `metrics_*.json` + `inspection_*.png` generated alongside each registered image
3. **Summary CSV**: `local_python/summary.csv` and `local_matlab/summary.csv` have rows for all successful runs
4. **Timeout handling**: tissue/tissue_2D runs that timeout produce `run_*.json` with `"status": "timeout"` and no registered image
5. **Metric sanity**: NCC/match rate values comparable to what we saw in `local_tuning/` and `local_comparison/`
