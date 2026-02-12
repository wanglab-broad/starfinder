# Phase 5: Preprocessing Module + Utils Implementation Plan

## Context

STARfinder Milestone 2 (Python backend migration) has completed Phases 0-4: I/O, registration, spotfinding, and barcode processing. Phase 5 ports the MATLAB preprocessing functions that sit between image loading and spot detection in the pipeline.

**Note on AdjustSizeAcrossRound**: `load_image_stacks()` already handles dimension normalization *within* a single round (across channels). Cross-round adjustment is deferred to Phase 6 (Dataset class).

## Scope: 5 Functions across 2 Modules

### `starfinder.preprocessing` — Image enhancement (4 functions)

| # | Python Function | MATLAB Source | Status in Workflow |
|---|---|---|---|
| 1 | `min_max_normalize(volume)` | `MinMaxNorm.m` | Used (EnhanceContrast) |
| 2 | `histogram_match(volume, reference)` | `STARMapDataset.m:261-293` | Used (HistEqualize) |
| 3 | `morphological_reconstruction(volume, radius)` | `MorphologicalReconstruction.m` | Used (MorphRecon) |
| 4 | `tophat_filter(volume, radius)` | `STARMapDataset.m:325-361` | Unused but included for completeness |

### `starfinder.utils` — General utilities (1 function)

| # | Python Function | MATLAB Source | Status in Workflow |
|---|---|---|---|
| 5 | `make_projection(volume, method)` | `MakeProjections.m` | Used (all workflows) |

All functions operate on single volumes `(Z, Y, X)` or `(Z, Y, X, C)`. Multi-round coordination deferred to Phase 6.

## Files to Create

```
src/python/starfinder/preprocessing/
    __init__.py           # Preprocessing exports
    normalization.py      # min_max_normalize, histogram_match
    morphology.py         # morphological_reconstruction, tophat_filter
src/python/starfinder/utils.py        # make_projection (single file, can grow later)
src/python/test/test_preprocessing.py  # ~15 preprocessing tests
src/python/test/test_utils.py          # ~4 utils tests
```

## Files to Modify

- `src/python/starfinder/__init__.py` — add preprocessing + utils exports
- `CLAUDE.md` — add modules to Implemented Modules, mark Phase 5 complete
- `docs/notes.md` — add Phase 5 completion entry

## Implementation Steps

### Step 1: `utils.py` — `make_projection`
Simplest function. Pure numpy. General-purpose utility.
```python
def make_projection(volume: np.ndarray, method: str = "max") -> np.ndarray:
```
- `"max"`: `np.max(volume, axis=0)` — preserves dtype
- `"sum"`: `np.sum(volume.astype(np.uint32), axis=0)` → scale to uint8 (matching MATLAB `im2uint8`)
- Handles (Z,Y,X) → (Y,X) and (Z,Y,X,C) → (Y,X,C)

### Step 2: `normalization.py` — `min_max_normalize`
Per-channel [min, max] → [0, 255] contrast stretching.
```python
def min_max_normalize(volume: np.ndarray) -> np.ndarray:
```
- Matches MATLAB `stretchlim(ch, 0)` + `imadjustn(ch, [min, max])`
- Per-channel: compute global min/max across all Z slices, rescale to uint8
- Constant channel (min == max) → zeros
- 3D/4D: add temporary axis if 3D, squeeze back after

### Step 3: `normalization.py` — `histogram_match`
Match histogram of each channel to a reference volume.
```python
def histogram_match(volume: np.ndarray, reference: np.ndarray, nbins: int = 64) -> np.ndarray:
```
- Uses `skimage.exposure.match_histograms(channel, reference)` per channel
- `nbins` accepted for API compat but unused by skimage (exact CDF matching; negligible difference for uint8)
- Preserves input dtype

### Step 4: `morphology.py` — `tophat_filter`
White tophat filtering per Z-slice.
```python
def tophat_filter(volume: np.ndarray, radius: int = 3) -> np.ndarray:
```
- Per Z-slice: `skimage.morphology.white_tophat(slice, disk(radius))`
- Returns uint8

### Step 5: `morphology.py` — `morphological_reconstruction`
Most complex. Per Z-slice multi-step algorithm.
```python
def morphological_reconstruction(volume: np.ndarray, radius: int = 3) -> np.ndarray:
```
Per Z-slice algorithm (matching MATLAB exactly):
1. `marker = erosion(slice, disk(radius))`
2. `obr = reconstruction(marker, slice, method='dilation')`
3. `subtracted = slice - obr`
4. `result = subtracted + white_tophat(subtracted, se) - black_tophat(subtracted, se)`

**Critical**: Use int16 for arithmetic in step 4 to avoid uint8 overflow, then `np.clip(..., 0, 255).astype(np.uint8)`.

### Step 6: Wire up exports
- `preprocessing/__init__.py` — export 4 preprocessing functions with `__all__`
- `starfinder/__init__.py` — add `preprocessing` + `utils` imports and top-level exports

### Step 7: Tests
**`test_utils.py`** (~4 tests):
- **TestMakeProjection**: max 3D, max 4D, sum accumulates, invalid method raises ValueError

**`test_preprocessing.py`** (~15 tests):
- **TestMinMaxNormalize** (5): rescales to uint8, per-channel independence, constant channel, 3D input, 4D input
- **TestHistogramMatch** (3): dtype preserved, shape preserved, histogram shifts toward reference
- **TestMorphologicalReconstruction** (3): uint8 output, shape preserved (3D+4D), background removed
- **TestTophatFilter** (3): uint8 output, shape preserved, removes large structures

### Step 8: Update docs
- `CLAUDE.md`: Add preprocessing + utils module docs, mark Phase 5 complete
- `docs/notes.md`: Add Phase 5 entry

## Dependencies

All already in `pyproject.toml` — no new deps:
- `numpy` — array ops
- `scikit-image>=0.21` — `skimage.exposure.match_histograms`, `skimage.morphology.{erosion, reconstruction, white_tophat, black_tophat, disk}`

## Key Implementation Details

1. **3D/4D handling pattern** (inline in each function):
   ```python
   is_3d = volume.ndim == 3
   if is_3d:
       volume = volume[..., np.newaxis]
   # ... per-channel processing ...
   if is_3d:
       result = result[..., 0]
   ```

2. **uint8 overflow in morphological_reconstruction**: Cast to int16 before `subtracted + top - bot`, clip, cast back.

3. **histogram_match nbins**: skimage does exact CDF matching (no nbins). Accept param for API compat, document as unused.

4. **make_projection sum**: MATLAB `im2uint8(sum(uint32(...)))` scales max value to 255. Python: `(summed / summed.max() * 255).astype(np.uint8)`.

## Verification

1. `cd src/python && uv run pytest test/test_utils.py test/test_preprocessing.py -v`
2. `uv run pytest test/ -v` — full suite, no regressions (expect ~116 tests: 96 existing + ~19 new)
3. Import check: `uv run python -c "from starfinder.preprocessing import min_max_normalize, histogram_match, morphological_reconstruction, tophat_filter; from starfinder.utils import make_projection; print('OK')"`
