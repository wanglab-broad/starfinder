# Phase 7: End-to-End Validation Plan

**Date:** 2026-02-13
**Status:** DRAFT
**Prerequisite:** Phases 0–6 complete (143 tests passing)

## Goal

Validate that the Python backend produces correct, MATLAB-compatible outputs across the full pipeline. This implements the contract, integration, and golden test layers from `docs/test_design.md` using both synthetic and real datasets.

## Success Criteria

- [ ] Contract tests enforce output schemas and path conventions
- [ ] Synthetic E2E tests verify pipeline correctness against ground truth
- [ ] Golden data generated from MATLAB for 3 real datasets
- [ ] Golden regression tests confirm Python outputs match MATLAB within tolerance
- [ ] All new tests integrated into `uv run pytest` (golden tests skip gracefully when data unavailable)

---

## Task 1: Contract Tests

**Goal:** Validate that pipeline outputs conform to the schemas expected by downstream tools (reads_assignment, stitch_subtile, Snakemake rules).

**Test file:** `test/contract/test_contracts.py`

**Fixtures:** Run the mini pipeline to produce real outputs, then validate their structure.

### 1.1 CSV Schema Tests

| Test Class | Target File | Validates |
|---|---|---|
| `TestGoodSpotsCsvContract` | `{fov}_goodSpots.csv` | Required columns: `x, y, z, gene`. Coordinates are 1-based integers. All genes exist in codebook. No duplicate (x,y,z) entries. |
| `TestAllSpotsCsvContract` | `{fov}_allSpots.csv` | Required columns: `z, y, x, intensity, channel, color_seq`. Per-round color/score columns present. `color_seq` length = `n_rounds`. |
| `TestShiftLogCsvContract` | `{fov}.txt` (shift log) | Columns: `fov_id, round, row, col, z`. Numeric shift values. One row per registered round (excludes ref). |
| `TestSubtileCoordsCsvContract` | `subtile_coords.csv` | Columns: `t, scoords_x, scoords_y, ecoords_x, ecoords_y`. 1-based coordinates. Sequential `t` index starting at 0. `ecoords >= scoords` for each axis. |

### 1.2 Path Convention Tests

| Test | Validates |
|---|---|
| `test_ref_merged_path` | `FOVPaths.ref_merged_tif` == `{output_root}/images/ref_merged/{fov_id}.tif` |
| `test_signal_csv_path` | `FOVPaths.signal_csv("goodSpots")` == `{output_root}/signal/{fov_id}_goodSpots.csv` |
| `test_shift_log_path` | `FOVPaths.shift_log()` == `{output_root}/log/gr_shifts/{fov_id}.txt` |
| `test_subtile_dir_path` | `FOVPaths.subtile_dir` == `{output_root}/output/subtile/{fov_id}` |

### 1.3 Coordinate Convention Tests

| Test | Validates |
|---|---|
| `test_internal_0based` | Spot DataFrame `z, y, x` values are 0-based (min can be 0) |
| `test_csv_1based_conversion` | After `save_signal()`, CSV coordinates are all >= 1 |
| `test_shift_tuple_order` | `global_shifts` values are `(dz, dy, dx)` matching `(Z, Y, X)` axis order |
| `test_image_axis_order` | Loaded images have shape `(Z, Y, X, C)` — dim 3 matches channel count |

### Checkpoint 1
- [ ] All contract tests pass on mini synthetic dataset
- [ ] Tests run in < 10 seconds

---

## Task 2: Synthetic E2E Validation

**Goal:** Run the full pipeline on synthetic data and compare outputs against known ground truth quantitatively. Goes beyond existing `test_fov.py` which only checks structural correctness (columns exist, shapes match) without comparing actual values.

**Test file:** `test/integration/test_e2e_synthetic.py`

**Datasets:** mini (1 FOV, 256x256x5) and standard (4 FOVs, 512x512x10)

### 2.1 Ground Truth Comparison (Mini Dataset)

The mini dataset `ground_truth.json` contains exact shifts, spot positions, gene labels, and color sequences per FOV.

| Test | What it measures | Tolerance | Rationale |
|---|---|---|---|
| `test_shift_recovery` | Per-axis shift error vs ground truth | < 1.0 px | Small images (256px) limit sub-pixel accuracy |
| `test_spot_detection_recall` | Fraction of GT spots found within 2px radius | > 60% | Preprocessing can shift/merge spots at small scales |
| `test_spot_detection_precision` | Fraction of detected spots near a GT spot | > 40% | Background artifacts can create false positives |
| `test_gene_assignment_accuracy` | Among spatially matched spots, correct gene label | > 50% | Depends on shift recovery + extraction accuracy |
| `test_color_sequence_format` | `color_seq` length matches `n_rounds`, values in {1,2,3,4} | Exact | Structural correctness |

**Note on tolerances:** These are deliberately loose for mini data. The 256x256x5 images have only 20 spots in a tiny volume — boundary effects, preprocessing normalization, and quantization all have outsized impact. The real quality signal comes from golden data on production-sized images.

### 2.2 Multi-FOV Consistency (Standard Dataset)

Run the same pipeline on all 4 FOVs of the standard dataset. Each FOV has independent shifts and spots.

| Test | What it measures | Tolerance |
|---|---|---|
| `test_all_fovs_complete` | Pipeline runs without error on all 4 FOVs | No exceptions |
| `test_shift_recovery_all_fovs` | Shift recovery per FOV | < 1.0 px per axis |
| `test_spot_count_reasonable` | Each FOV detects > 20 spots (GT has 100) | > 20 spots/FOV |
| `test_gene_diversity` | At least 3 distinct genes found per FOV | >= 3 genes |

### 2.3 Pipeline Variant Coverage

Test different pipeline configurations to ensure robustness:

| Variant | Configuration | What it validates |
|---|---|---|
| Direct mode | `load → enhance → global_reg → spot_find → extract → filter` | Standard workflow |
| With tophat | `load → tophat → global_reg → spot_find → extract → filter` | Alternative preprocessing |
| With MIP | `load → enhance → make_projection → global_reg → spot_find → extract → filter` | 2D projection path |
| Subtile mode | `load → enhance → global_reg → create_subtiles → from_subtile → spot_find` | Subtile workflow |

### Checkpoint 2
- [ ] All synthetic E2E tests pass
- [ ] Shift recovery within tolerance on both mini and standard
- [ ] At least one pipeline variant detects > 50% of ground truth spots
- [ ] Tests run in < 60 seconds

---

## Task 3: Golden Data Generation (MATLAB)

**Goal:** Produce reference outputs from MATLAB for 3 real datasets, using the exact same processing steps as the Python pipeline.

**Script:** `tests/golden/generate_golden_data.m`
**Output:** `~/wanglab/jiahao/test/starfinder_benchmark/golden/{dataset}/{fov_id}/`

### 3.1 Datasets

| Dataset | FOV | Dimensions | Rounds | Codebook |
|---|---|---|---|---|
| tissue-2D | Tile 1 (first FOV) | 3072 x 3072 x 30 | 4 | tissue-2D codebook |
| cell-culture-3D | Position001 | 1496 x 1496 x 30 | 4 (of 6) | cell-culture-3D codebook |
| LN | Position001 | 1496 x 1496 x 50 | 4 | LN codebook |

**Input data locations:**
- tissue-2D: `/stanley/WangLab/Data/Processed/sample-dataset/tissue-2D/`
- cell-culture-3D: `/stanley/WangLab/Data/Processed/sample-dataset/cell-culture-3D/`
- LN: `/stanley/WangLab/Data/Processed/sample-dataset/LN/`

### 3.2 MATLAB Processing Steps

For each FOV, run the standard pipeline:

```matlab
% 1. Load raw images (4 rounds, 4 channels)
ds.LoadRawImages('rounds', 1:4, 'channels', 0:3);

% 2. Enhance contrast (min-max normalization)
ds.EnhanceContrast();

% 3. Global registration (phase correlation to round 1)
ds.GlobalRegistration('ref_round', 1);

% 4. Spot finding on reference round
ds.SpotFinding();

% 5. Reads extraction across all rounds
ds.ReadsExtraction();

% 6. Reads filtration against codebook
ds.ReadsFiltration();
```

### 3.3 Saved Outputs Per FOV

| File | Contents | Format |
|---|---|---|
| `gr_shifts.csv` | Global registration shifts | CSV: `round, row, col, z` (1-based naming) |
| `allSpots.csv` | All detected spots with color sequences | CSV: `x, y, z, intensity, channel, color_seq, ...` |
| `goodSpots.csv` | Filtered spots with gene labels | CSV: `x, y, z, gene, ...` |
| `ref_merged.tif` | Reference merged image | TIFF (Z, Y, X) |
| `run_config.json` | Parameters used (rounds, channels, thresholds) | JSON |

### 3.4 MATLAB Script Structure

```matlab
function generate_golden_data(dataset_name, fov_id, input_path, output_path)
    % Process one FOV and save reference outputs
    % Called once per dataset

    ds = STARMapDataset(input_path, output_path);
    ds.LoadRawImages('rounds', 1:4, 'channels', 0:3);
    ds.EnhanceContrast();
    ds.GlobalRegistration('ref_round', 1);
    ds.SpotFinding();
    ds.ReadsExtraction();
    ds.ReadsFiltration();

    % Save outputs
    fov_dir = fullfile(output_path, fov_id);
    mkdir(fov_dir);
    writetable(array2table(ds.gr_shifts, 'VariableNames', {'round','row','col','z'}), ...
               fullfile(fov_dir, 'gr_shifts.csv'));
    writetable(ds.allSpots, fullfile(fov_dir, 'allSpots.csv'));
    writetable(ds.goodSpots, fullfile(fov_dir, 'goodSpots.csv'));
    % ... ref_merged, run_config
end
```

### Checkpoint 3
- [ ] MATLAB script runs successfully on all 3 datasets
- [ ] Output files exist and are non-empty for each FOV
- [ ] Visual inspection of ref_merged.tif looks correct
- [ ] `run_config.json` documents exact parameters used

---

## Task 4: Golden Regression Tests (Python vs MATLAB)

**Goal:** Run the Python pipeline on the same real FOVs and compare outputs against MATLAB reference within defined tolerances.

**Test file:** `test/golden/test_regression.py`
**Marker:** `@pytest.mark.golden`
**Data location:** Set via `STARFINDER_GOLDEN_DATA` env var (defaults to `~/wanglab/jiahao/test/starfinder_benchmark/golden/`)

### 4.1 Test Configuration

```python
# Tolerances for Python vs MATLAB comparison
TOLERANCE = {
    "shift_px": 0.5,           # Per-axis shift difference
    "spot_count_ratio": 0.10,  # Relative spot count difference (10%)
    "spot_position_px": 2.0,   # Spatial matching radius
    "spot_overlap_ratio": 0.80, # Fraction of MATLAB spots matched by Python
    "gene_freq_diff": 0.05,    # Per-gene frequency difference
    "ref_merged_ncc": 0.95,    # NCC between ref_merged images
}
```

### 4.2 Regression Tests

| Test | Metric | Tolerance | Description |
|---|---|---|---|
| `test_shifts_match` | Max per-axis diff | < 0.5 px | Phase correlation should produce near-identical shifts |
| `test_spot_count_similar` | `abs(py - matlab) / matlab` | < 10% | Spot detection can vary due to threshold differences |
| `test_spot_positions_overlap` | Fraction matched within 2px | > 80% | Most spots should be at the same locations |
| `test_gene_distribution_similar` | Max per-gene freq diff | < 5% | Gene assignment distribution should closely match |
| `test_ref_merged_ncc` | NCC of ref_merged images | > 0.95 | Global registration result should be nearly identical |

### 4.3 Per-Dataset Parametrization

Tests are parametrized over all 3 datasets:

```python
@pytest.mark.golden
@pytest.mark.parametrize("dataset_name,fov_id", [
    ("tissue-2D", "<tile_id>"),
    ("cell-culture-3D", "Position001"),
    ("LN", "Position001"),
])
class TestGoldenRegression:
    ...
```

### 4.4 Test Infrastructure

**Fixtures needed:**
- `golden_data_path(dataset_name, fov_id)` — locates MATLAB reference files
- `python_pipeline_result(dataset_name, fov_id)` — runs Python pipeline on same input, caches result for the session
- `golden_config(dataset_name)` — loads `run_config.json` to match MATLAB parameters

**Skip logic:**
- Tests skip with clear message if `STARFINDER_GOLDEN_DATA` path doesn't exist
- Tests skip per-dataset if that dataset's golden data is missing (allows partial runs)

### Checkpoint 4
- [ ] Python pipeline runs without error on all 3 real datasets
- [ ] Registration shifts match MATLAB within 0.5px
- [ ] Spot overlap > 80% for all datasets
- [ ] Gene distribution difference < 5%
- [ ] NCC of ref_merged > 0.95

---

## Implementation Order

```
Task 1 (Contract Tests)          ← No dependencies, start here
    │
Task 2 (Synthetic E2E)           ← Depends on contract test fixtures
    │
Task 3 (Golden Data Generation)  ← Independent of Tasks 1-2, can run in parallel
    │                               Requires MATLAB + access to real data
    │
Task 4 (Golden Regression Tests) ← Depends on Task 3 output
```

**Recommended execution:** Tasks 1 → 2 sequentially (Python-only), Task 3 in parallel (MATLAB). Task 4 after Task 3 completes.

## Test Organization

```
src/python/test/
├── conftest.py                    # Existing + new golden data fixtures
├── contract/
│   ├── __init__.py
│   └── test_contracts.py          # Task 1: schema, path, coordinate tests
├── integration/
│   ├── __init__.py
│   └── test_e2e_synthetic.py      # Task 2: ground truth comparison
├── golden/
│   ├── __init__.py
│   └── test_regression.py         # Task 4: Python vs MATLAB
├── test_io.py                     # Existing unit tests
├── test_registration.py
├── test_fov.py
└── ...
tests/golden/
└── generate_golden_data.m         # Task 3: MATLAB script
```

## Running Tests

```bash
# All fast tests (unit + contract + synthetic E2E)
cd src/python && uv run pytest test/ -v

# Contract tests only
uv run pytest test/contract/ -v

# Synthetic E2E only
uv run pytest test/integration/ -v

# Golden regression tests (requires STARFINDER_GOLDEN_DATA)
STARFINDER_GOLDEN_DATA=~/wanglab/jiahao/test/starfinder_benchmark/golden \
    uv run pytest test/golden/ -v -m golden

# Everything except golden (no external data needed)
uv run pytest test/ -v -m "not golden"
```

## Estimated Effort

| Task | Effort | Dependencies |
|---|---|---|
| Task 1: Contract Tests | ~2 hours | None |
| Task 2: Synthetic E2E | ~3 hours | Task 1 fixtures |
| Task 3: Golden Data Gen | ~2 hours | MATLAB + real data access |
| Task 4: Golden Regression | ~3 hours | Task 3 outputs |
| **Total** | **~10 hours** | |
