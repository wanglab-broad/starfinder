# Phase 3: Spot Finding & Extraction — Implementation Plan

## Context

STARfinder Phase 2 (Registration) is complete and benchmarked. Phase 3 is the next step in Milestone 2: **porting MATLAB's spot detection and color extraction to Python**. This module transitions the pipeline from image space to coordinate/feature space — detecting fluorescent spots (mRNA molecules) and reading their per-channel intensities across sequencing rounds.

The MATLAB implementation is compact (~100 lines across two files) with well-defined behavior, making this a clean porting target. All required Python dependencies (`scipy`, `scikit-image`, `pandas`) are already in `pyproject.toml`.

---

## Scope

**In scope**: `find_spots_3d()` + `extract_from_location()`
**Out of scope**: Barcode decoding (Phase 4), preprocessing/morphological reconstruction (Phase 5), codebook loading (Phase 4)

---

## Files to Create

```
src/python/starfinder/spotfinding/
├── __init__.py           # Exports: find_spots_3d
└── local_maxima.py       # Core spot detection

src/python/starfinder/barcode/
├── __init__.py           # Exports: extract_from_location
└── extraction.py         # Color vector extraction

src/python/test/
├── test_spotfinding.py   # 7 tests
└── test_extraction.py    # 8 tests
```

**Files to modify**:
- `src/python/starfinder/__init__.py` — add spotfinding + barcode exports

---

## Step 1: Spot Finding Module

### `starfinder/spotfinding/local_maxima.py`

**Reference**: `src/matlab/SpotFindingMax3D.m` (34 lines)

```python
def find_spots_3d(
    image: np.ndarray,                          # (Z, Y, X, C)
    intensity_estimation: str = "adaptive",     # "adaptive" | "global"
    intensity_threshold: float = 0.2,
) -> pd.DataFrame:  # columns: [z, y, x, intensity, channel]
```

**Algorithm** (per channel):
1. Compute threshold: adaptive → `channel.max() * threshold`, global → `dtype_max * threshold`
2. `skimage.feature.peak_local_max(channel, min_distance=1, threshold_abs=threshold)` → (N, 3) coords
3. Extract intensity at each coord
4. Append channel index
5. Concatenate all channels → single DataFrame

**Key decisions**:
- Use `peak_local_max` over manual `maximum_filter` — handles plateaus correctly, single call does maxima + threshold
- `min_distance=1` matches MATLAB's `imregionalmax` 26-connectivity
- Return 0-based coords with columns `[z, y, x, intensity, channel]` (Python ZYX convention, not MATLAB XYZ)
- Empty result → DataFrame with correct schema (0 rows, correct columns)

### Tests (`test_spotfinding.py`)

| Test | What it checks |
|------|---------------|
| `test_finds_known_spots` | Detects spots in mini synthetic dataset, count sanity check |
| `test_returns_correct_schema` | DataFrame columns = [z, y, x, intensity, channel] |
| `test_empty_image` | Blank image → empty DataFrame with correct schema |
| `test_adaptive_threshold` | Adaptive mode: threshold = max * fraction |
| `test_global_threshold` | Global mode: threshold = dtype_max * fraction |
| `test_multichannel` | Spots in different channels detected with correct channel index |
| `test_coordinate_values` | Spot at known (z, y, x) returns correct coordinates |

---

## Step 2: Extraction Module

### `starfinder/barcode/extraction.py`

**Reference**: `src/matlab/ExtractFromLocation.m` (62 lines)

```python
def extract_from_location(
    image: np.ndarray,                          # (Z, Y, X, C)
    spots: pd.DataFrame,                        # must have [z, y, x] columns
    voxel_size: tuple[int, int, int] = (1, 2, 2),  # (dz, dy, dx) half-widths
) -> tuple[np.ndarray, np.ndarray]:  # (color_seq, color_score)
```

**Algorithm** (per spot):
1. Extract voxel neighborhood `[pos-voxel_size : pos+voxel_size+1]`, clipped to image bounds
2. Sum across spatial dims → (C,) vector
3. L2 normalize: `vec / (||vec|| + 1e-6)`
4. Winner-take-all: single max → 1-based channel string, tie → `"M"`, NaN → `"N"`
5. Score: `-log(max_value)`, `inf` for M/N

**Key decisions**:
- `voxel_size` in **(dz, dy, dx)** order (Python ZYX convention). MATLAB uses (dx, dy, dz)=[2,2,1], so Python equivalent is (1, 2, 2).
- Channel strings are **1-based** ("1"–"4") for codebook compatibility with MATLAB encoding scheme
- Vectorize the sum step (extract all voxels at once) but loop for boundary clipping

### Tests (`test_extraction.py`)

| Test | What it checks |
|------|---------------|
| `test_single_spot_extraction` | Extracts correct channel from single-channel signal |
| `test_winner_take_all` | Selects channel with maximum normalized intensity |
| `test_tie_returns_M` | Equal max in multiple channels → "M" |
| `test_zero_signal_handling` | All-zero neighborhood → handled gracefully (epsilon prevents NaN) |
| `test_voxel_neighborhood` | voxel_size controls extraction window size |
| `test_boundary_clipping` | Spots near edges don't crash, extents clipped |
| `test_multiple_spots` | Batch processing of multiple spots |
| `test_score_computation` | Score = -log(max_normalized) ≈ 0 for single-channel signal |

---

## Step 3: Integration & Package Wiring

1. Create `spotfinding/__init__.py` exporting `find_spots_3d`
2. Create `barcode/__init__.py` exporting `extract_from_location`
3. Update `starfinder/__init__.py` to include new modules
4. Run full test suite: `uv run pytest test/ -v`

---

## Step 4: Ground Truth Validation

Validate against the mini synthetic dataset ground truth (`tests/fixtures/synthetic/mini/ground_truth.json`):

```python
# Ground truth format per spot:
# {"position": [z, y, x], "color_seq": "4422", "intensity": 242, "gene": "GeneA"}

# 1. Find spots in round1 → compare count/positions with ground truth
# 2. Extract colors for detected spots → compare color_seq with ground truth
```

This is a lightweight sanity check within tests, not a separate QC notebook.

---

## Verification

```bash
# Run all tests
uv run pytest test/test_spotfinding.py test/test_extraction.py -v

# Run full suite to check nothing broke
uv run pytest test/ -v
```

**Success criteria**:
- All tests pass
- Spot detection on mini dataset finds reasonable count (ground truth has 20 spots)
- Extracted color sequences contain only valid values ("1"–"4", "M", "N")

---

## MATLAB ↔ Python Coordinate Mapping

| Concept | MATLAB | Python |
|---------|--------|--------|
| Image axes | (X, Y, Z, C) | (Z, Y, X, C) |
| DataFrame cols | x, y, z (1-based) | z, y, x (0-based) |
| voxel_size param | [dx, dy, dz] = [2, 2, 1] | (dz, dy, dx) = (1, 2, 2) |
| Color channel strings | "1"–"4" (1-based) | "1"–"4" (1-based, for codebook compat) |
| CSV output | 1-based XYZ | 1-based XYZ (conversion at I/O boundary, Phase 6) |
