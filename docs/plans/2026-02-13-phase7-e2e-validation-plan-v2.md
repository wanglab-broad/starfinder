# Plan: Phase 7 — End-to-End Pipeline Validation (v2)

## Context

Phases 1-6 implemented all Python backend modules and the FOV orchestration layer. Each module has unit tests, and `test_fov.py` has 11 integration tests that exercise individual pipeline steps on the mini synthetic dataset. However, no test validates the **full pipeline output against ground truth** — we don't yet know if the detected spots, decoded genes, and output CSVs are *correct*, only that they're well-formed.

Phase 7 closes this gap with quantitative ground-truth validation and subtile coordinate mapping verification. This is the final validation step before the Python backend can replace MATLAB in production Snakemake rules.

**Outcome**: A test suite (`test_e2e.py`) that proves the Python pipeline produces correct results — measured against synthetic ground truth with quantitative tolerances.

---

## What's Already Tested (Gap Analysis)

| Area | Current Coverage | Gap |
|------|-----------------|-----|
| Individual modules (io, reg, spot, barcode, preproc) | Unit tests per module | ✓ Covered |
| FOV pipeline steps | `test_fov.py` (11 tests, mini dataset) | Steps tested in isolation, not end-to-end |
| Output format (CSV columns, 1-based coords, TIFF) | `test_fov.py` (`test_save_signal`, `test_save_ref_merged`, `test_global_registration`) | ✓ Structural checks covered |
| Subtile create + load | `test_fov.py` (`test_create_and_load_subtiles`) | ✓ Create/load covered; coordinate mapping not tested |
| **Pipeline correctness vs ground truth** | **None** | Spot recall, gene accuracy, shift recovery |
| **Subtile spot coordinate mapping** | **None** | Post-subtile spot finding + offset remapping |
| **Shift serialization round-trip** | **None** | `fov.global_shifts` ↔ shift log CSV value consistency |

---

## File Plan

### New files

| File | Contents | ~Lines |
|------|----------|--------|
| `src/python/test/test_e2e.py` | End-to-end validation tests (8 tests) | ~250 |
| `src/python/starfinder/testdata/validation.py` | Ground truth comparison utilities | ~120 |

### Modified files

| File | Change |
|------|--------|
| `src/python/test/conftest.py` | Add `e2e_result` session fixture |
| `src/python/starfinder/testdata/__init__.py` | Export validation utilities |

---

## Implementation Steps

### Step 1: Ground Truth Comparison Utilities (`testdata/validation.py`)

Utility functions that compare pipeline output against `ground_truth.json`. These are reused by all e2e tests.

```python
# Core comparison functions:

def compare_shifts(detected_shifts: dict[str, Shift3D],
                   ground_truth: dict,
                   fov_id: str,
                   tolerance: float = 1.5) -> dict:
    """Compare detected global shifts against ground truth shifts.

    Returns dict with per-round shift errors and overall pass/fail.
    Ground truth shifts are in [dz, dy, dx] format.
    Tolerance is in pixels (allows sub-pixel registration error).
    """

def compare_spots(detected_spots: pd.DataFrame,
                  ground_truth: dict,
                  fov_id: str,
                  position_tolerance: float = 5.0) -> dict:
    """Match detected spots to ground truth spots by proximity.

    Returns:
        - recall: fraction of GT spots matched to a detected spot
        - precision: fraction of detected spots near a GT spot
        - mean_distance: average position error for matched pairs
        - matched_pairs: list of (gt_id, detected_idx, distance)
    Uses greedy nearest-neighbor matching with exclusion.
    """

def compare_genes(good_spots: pd.DataFrame,
                  ground_truth: dict,
                  fov_id: str,
                  position_tolerance: float = 5.0) -> dict:
    """Compare decoded gene labels against ground truth.

    First matches good_spots to GT spots by position (same as compare_spots).
    Then for matched pairs, compares gene labels and color sequences.

    Color sequence format note: GT stores color_seq as string "4422",
    pipeline stores as int array [4, 4, 2, 2]. This function handles
    the conversion internally.

    Returns:
        - gene_accuracy: fraction of matched spots with correct gene label
        - gene_confusion: dict of {(gt_gene, pred_gene): count}
        - color_seq_accuracy: fraction with correct color_seq
    """

def e2e_summary(shift_result: dict,
                spot_result: dict,
                gene_result: dict) -> dict:
    """Combine comparison results into a single summary dict."""
```

**Key detail**: Ground truth positions are 0-based `(z, y, x)`. The `compare_spots` function operates in 0-based coordinates (matching internal pipeline convention). The shift comparison accounts for the sign convention: ground truth stores the displacement applied during data generation, and `phase_correlate` detects displacement of moving relative to fixed — same sign.

### Step 2: Shared E2E Fixture (`conftest.py` addition)

Add a single cached pipeline fixture that runs the full pipeline once per session and shares results across all e2e tests. The fixture takes the dataset path from the existing `mini_dataset` fixture (default).

```python
@pytest.fixture(scope="session")
def e2e_result(mini_dataset, tmp_path_factory):
    """Run full pipeline on synthetic dataset, return (fov, dataset, ground_truth).

    Runs once per session. Uses the first FOV from the dataset.
    Pipeline: load → enhance → global_reg → spot_find →
    reads_extract → reads_filter → save_signal.

    To test on a different dataset, swap the fixture dependency.
    """
```

The session-scoped fixture ensures the pipeline runs only once even though multiple test functions inspect different aspects of the results.

### Step 3: `test_e2e.py` — Test Classes (8 tests)

#### 3a. `TestE2EShiftRecovery` (2 tests)

Validate that global registration correctly recovers the known inter-round shifts from synthetic data.

- **`test_shift_recovery`**: Compare `fov.global_shifts` against `ground_truth["fovs"][fov_id]["shifts"]`. Assert per-round, per-axis error < 1.5 pixels.
- **`test_shift_log_csv_matches`**: Load the saved shift log CSV, compare values against `fov.global_shifts` dict. Verify that the `row` column maps to `dy` and `col` column maps to `dx` (catches column-swap serialization bugs).

#### 3b. `TestE2ESpotDetection` (2 tests)

Validate spot finding against ground truth spot positions.

- **`test_spot_recall`**: After full pipeline, compare detected spots against ground truth. Assert recall ≥ 0.7 (at least 70% of GT spots found within 5px tolerance). Assert precision ≥ 0.1 (sanity check — at most 10x false positives). Report both metrics.
- **`test_spot_positions_reasonable`**: Assert all detected spot coordinates are within image bounds: `0 <= z < Z, 0 <= y < Y, 0 <= x < X`.

#### 3c. `TestE2EBarcodeDecoding` (2 tests)

Validate the full decoding pipeline against ground truth gene labels.

- **`test_color_seq_accuracy`**: For spatially matched spots, verify extracted `color_seq` matches ground truth. Report per-round accuracy and overall accuracy.
- **`test_gene_accuracy`**: After `reads_filtration`, verify decoded gene labels match ground truth for matched spots. Assert gene accuracy ≥ 0.5 on matched+filtered spots. Verify all gene labels match entries in the codebook and no NaN values in coordinate or gene columns.

#### 3d. `TestE2ESubtileRoundTrip` (1 test)

Validate the only untested part of the subtile workflow: coordinate mapping after spot finding. (Subtile creation, loading, and CSV format are already tested in `test_fov.py::test_create_and_load_subtiles`.)

- **`test_subtile_spot_coordinate_mapping`**: Create 2×2 subtiles from processed FOV. Run spot finding on one subtile. Add subtile offset (`scoords - 1`, converting 1-based CSV coords back to 0-based) to detected spot positions. Verify remapped coordinates fall within the expected global image region.

#### 3e. `TestE2EPipelineSmokeTest` (1 test)

- **`test_pipeline_produces_output`**: Verify the pipeline runs end-to-end without error and produces non-empty `good_spots` with at least 1 gene. This is the most basic "nothing is broken" check.

---

## Key Implementation Details

### Ground Truth Shift Convention

The synthetic dataset stores shifts as `"how much each round was displaced relative to round1"`. For round1 (reference), shifts are `[0, 0, 0]`. For other rounds, the shifts are the actual displacement applied during data generation.

`phase_correlate()` detects the displacement of the moving image relative to the fixed image, then `register_volume()` applies the **negative** shift to correct alignment. So `fov.global_shifts[round]` stores the *detected* displacement (same sign as ground truth shift), not the correction. The comparison should be:

```python
# ground_truth shift = [dz, dy, dx] (how much round was moved)
# fov.global_shifts[round] = (dz, dy, dx) (detected displacement, same sign)
error = abs(detected - ground_truth)
```

### Spot Matching Strategy

Use greedy nearest-neighbor matching with exclusion:
1. Compute pairwise distances between ground truth and detected spots (scipy.spatial.distance.cdist)
2. Sort all pairs by distance
3. Greedily assign closest unmatched pairs
4. A match requires distance < `position_tolerance` (default 5.0 pixels)

This is simpler than Hungarian algorithm and sufficient for our sparse spot distributions (20-100 spots).

### Color Sequence Format Conversion

Ground truth stores `color_seq` as a string (e.g., `"4422"`). The pipeline's `extract_from_location()` returns integer arrays (e.g., `[4, 4, 2, 2]`). The `compare_genes` utility handles this conversion:

```python
gt_colors = [int(c) for c in gt_spot["color_seq"]]  # "4422" → [4, 4, 2, 2]
```

### Realistic Accuracy Expectations

The synthetic data has:
- **Known shifts**: Phase correlation on clean synthetic data should recover shifts within 1-2 pixels
- **Known spots**: LoG detection on bright Gaussian spots (intensity 200-255) should achieve high recall, but registration error and noise can cause misses
- **Known genes**: Color extraction at spot locations depends on registration quality and voxel size. With 4 channels and 4 rounds, a small position error can flip a color assignment

Conservative thresholds:
- Shift recovery: < 1.5 px per axis
- Spot recall: ≥ 0.7
- Spot precision: ≥ 0.1 (sanity floor — prevents catastrophic over-detection)
- Gene accuracy: ≥ 0.5 on matched+filtered spots (barcode decoding is the hardest step)

### Threshold Calibration (Step 4)

After the first passing run:
1. Run tests with `--tb=long` to see printed metrics
2. Record actual values (e.g., actual recall = 0.85, actual gene accuracy = 0.65)
3. Set thresholds at ~70-80% of observed values to allow for noise variance across runs
4. Update thresholds in `test_e2e.py` with a comment noting the observed value

### Session-Scoped Pipeline Execution

The full pipeline runs once per session (~2-5 seconds for mini). Individual test functions inspect different aspects of the same result.

```
e2e_result (session fixture, runs once on mini dataset)
├── test_pipeline_produces_output
├── test_shift_recovery
├── test_shift_log_csv_matches
├── test_spot_recall
├── test_spot_positions_reasonable
├── test_color_seq_accuracy
├── test_gene_accuracy
└── test_subtile_spot_coordinate_mapping
```

---

## Implementation Order

1. **Step 1**: `testdata/validation.py` — Ground truth comparison utilities (~120 lines)
2. **Step 2**: `conftest.py` addition — Session-scoped `e2e_result` fixture
3. **Step 3a-b**: Shift recovery + spot detection tests (highest priority — validates registration)
4. **Step 3c**: Barcode decoding tests (validates full pipeline correctness)
5. **Step 3d**: Subtile coordinate mapping test
6. **Step 3e**: Pipeline smoke test
7. **Step 4**: Calibrate thresholds — Run tests, print actual metrics, adjust thresholds to ~70-80% of observed values

---

## Verification

```bash
# Run only e2e tests
cd src/python && uv run pytest test/test_e2e.py -v

# Run full suite to check no regressions
cd src/python && uv run pytest test/ -v

# Quick smoke test (short traceback)
cd src/python && uv run pytest test/test_e2e.py -v --tb=short
```

---

## Out of Scope (Future)

These items are intentionally deferred:

- **Multi-FOV batch testing**: Running the pipeline on all 4 standard-dataset FOVs is validation of the orchestration layer, not the pipeline correctness. Multi-FOV behavior is exercised manually or via Snakemake, not pytest.
- **Snakemake rule integration**: Wiring Python backend into `workflow/rules/` is a separate task (requires config schema changes and MATLAB fallback logic)
- **Real dataset validation**: Running on cell-culture-3D or tissue-2D requires network mount access and significant runtime; consider as a separate QC notebook
- **MATLAB golden regression tests**: Comparing Python vs MATLAB on real data requires MATLAB golden data generation (see v1 plan Tasks 3-4); deferred until real-data validation is prioritized
- **Performance profiling**: Memory/timing benchmarks are covered by the existing `starfinder.benchmark` infrastructure
- **New data formats (h5, OME-Zarr)**: Explicitly deferred to a later phase per `plan_milestone_2.md`
- **Pipeline variant coverage**: Testing alternative preprocessing paths (tophat, MIP, histogram matching) is deferred; the default pipeline (enhance → global_reg → spot_find → extract → filter) is validated here

---

## Revision History

| Date | Version | Change |
|------|---------|--------|
| 2026-02-13 | v2 | Initial plan — e2e validation with ground truth comparison, multi-FOV, subtile |
| 2026-02-13 | v2.1 | Revised — removed overlap with `test_fov.py` (contracts, subtile create/load), fixed module path (`testdata/` not `testing/`), added shift log round-trip test, precision floor, color_seq format note, threshold calibration procedure, `@slow` marks, prerequisites section |
| 2026-02-13 | v2.2 | Simplified — single `e2e_result` fixture (dataset-agnostic, defaults to mini), removed multi-FOV tests and standard-dataset variants, consolidated output contract into gene accuracy test, added pipeline smoke test. 13 → 8 tests |
