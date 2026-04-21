# Plan: Trim Unmatched Fixed Points + Raise Outlier Weight in CPD

## Context

With coordinate-first synthetic data (clean Gaussian spots at deformed positions), CPD still fails on `polynomial_large` — but NOT because of asymmetric false detection. The actual root causes:

1. **Boundary dropout**: 30px polynomial displacement pushes ~31% of moving spots out of bounds. 13/49 fixed spots have no moving candidate within the 15px gather radius.
2. **Orphaned fixed points poison affine EM**: These 13 unmatched points pull the affine solution to garbage (`B-I max = 2.24`), which cascades into divergent non-rigid weights (`W_rms = 169`).

**Fix**: After `_gather_candidates()`, trim fixed points with no nearby moving candidate, and raise the default outlier weight from 0.1 to 0.15 to better tolerate remaining soft mismatches.

Tested results on small/polynomial_large:
- Trim alone (w=0.1): NCC 0.017 → **0.147** (B-I max: 2.24 → 0.40, W_rms: 169 → 0.76)
- Higher w=0.3 alone: NCC 0.017 → **0.115**
- Combined: expected to work at least as well as trim alone

## Changes

### 1. `src/python/starfinder/registration/pointset.py`

**Change default `w`** from 0.1 to 0.15 in `cpd_register()` signature (line 770):
```python
w: float = 0.15,  # was 0.1
```

**Add X trimming** after `_gather_candidates()` call (after line 849, before the `len(Y) < 10` check):
```python
Y = _gather_candidates(X, moving_spots, radius=candidate_radius)

# Trim fixed points with no moving candidate within radius.
# Orphaned fixed points (from boundary dropout under large deformations)
# poison the affine EM with phantom correspondences.
if len(Y) > 0:
    tree_y = cKDTree(Y)
    dists, _ = tree_y.query(X, k=1)
    matched = dists <= candidate_radius
    X = X[matched]
```

**Update docstring** for `w` parameter to note the 0.15 default rationale.

### 2. `src/python/starfinder/dataset/fov.py`

**Change default `cpd_w`** from 0.1 to 0.15 (line 310):
```python
cpd_w: float = 0.15,  # was 0.1
```

### 3. Tests

No new tests needed — existing `test_cpd_register_too_few_spots` tests the error path, and `test_cpd_register_improves_ncc` tests the improvement path. Both use the new code path automatically.

## Verification

1. `cd src/python && uv run pytest test/test_pointset.py -v` — all 9 tests pass
2. `uv run pytest test/ -v` — full 186-test suite passes
3. Quick smoke test: run the diagnostic from the conversation on small/polynomial_large to confirm NCC improvement
