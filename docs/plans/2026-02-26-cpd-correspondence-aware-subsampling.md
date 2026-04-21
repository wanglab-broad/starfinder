# Plan: Correspondence-Aware Subsampling for CPD Registration

## Context

CPD's `cpd_register()` subsamples fixed and moving point clouds **independently** via FPS, destroying cross-cloud correspondences. On tissue (14K→1K), only 22% of subsampled points retain their true nearest neighbor.

Two separate issues:
1. **Independent FPS drops true matches from the moving cloud** — a fixed point's partner may not survive FPS in the other cloud
2. **Single-point anchors lose local context** — FPS picks one representative per region, but CPD needs neighboring spots to disambiguate which moving spot belongs to which fixed spot

**Fix**: FPS for anchors + K nearest fixed neighbors per anchor + radius-based moving candidate gathering. No hard pre-matching — CPD's EM does soft assignment on the enriched clouds.

Results with 300 anchors + 3 neighbors, radius=15px:

| Preset | |X| fixed | |Y| moving | True NN preserved | Kernel mem |
|---|---|---|---|---|
| large | 1105 | 1199 | **100%** (was 82%) | 12 MB |
| tissue | 1200 | 1342 | **98%** (was 22%) | 14 MB |
| thick_medium | 1200 | 1405 | **98%** (was 35%) | 16 MB |

EM solve cost at N=1200: ~5s total (negligible vs 40-330s CPD runtime).

## Changes

### 1. `src/python/starfinder/registration/pointset.py`

**Add helper** `_subsample_with_neighbors()`:
```python
def _subsample_with_neighbors(
    points: np.ndarray,
    max_anchors: int = 300,
    k_neighbors: int = 3,
) -> np.ndarray:
    """FPS for spatial anchors, then expand with K nearest neighbors.

    Preserves local cluster structure so CPD can disambiguate
    nearby spots during soft assignment.
    """
    anchors = _subsample_points(points, max_anchors)
    tree = cKDTree(points)
    k = min(k_neighbors + 1, len(points))  # +1 because nearest is self
    _, indices = tree.query(anchors, k=k)
    all_idx = set(indices.ravel())
    return points[sorted(all_idx)]
```

**Add helper** `_gather_candidates()`:
```python
def _gather_candidates(
    fixed_sub: np.ndarray,
    moving_all: np.ndarray,
    radius: float = 15.0,
) -> np.ndarray:
    """Gather moving points within radius of subsampled fixed points.

    Ensures every fixed control point has candidate correspondences
    for CPD's EM, without hard pre-matching.
    """
    tree = cKDTree(moving_all)
    nearby = tree.query_ball_point(fixed_sub, r=radius)
    idx = set()
    for group in nearby:
        idx.update(group)
    return moving_all[sorted(idx)] if idx else moving_all[:0]
```

**Modify `cpd_register()` signature** — add two new params with defaults:
```python
def cpd_register(
    ...,
    candidate_radius: float = 15.0,   # NEW: moving candidate radius
    k_neighbors: int = 3,             # NEW: fixed neighbors per anchor
) -> np.ndarray:
```

**Replace lines 748-764** (spot detection + subsampling):
```python
# Current: independent FPS
X = _subsample_points(fixed_spots, max_control_points)
Y = _subsample_points(moving_spots, max_control_points)

# New: anchors + neighbors for fixed, radius-based candidates for moving
max_anchors = max_control_points // (1 + k_neighbors)  # 1000 // 4 = 250
X = _subsample_with_neighbors(fixed_spots, max_anchors=max_anchors, k_neighbors=k_neighbors)
Y = _gather_candidates(X, moving_spots, radius=candidate_radius)

if len(Y) < 10:
    raise ValueError(
        f"Too few moving candidates within {candidate_radius}px "
        f"radius: {len(Y)} (need >= 10)."
    )
```

With default max_control_points=1000, k=3: 250 anchors × 4 = ~1000 fixed, ~1200 moving. Matches our tested sweet spot.

**Update docstring**: document `candidate_radius` and `k_neighbors`, describe the new flow.

### 2. `src/python/starfinder/dataset/fov.py`

Lines ~307-311 (CPD params): add `candidate_radius: float = 15.0` and `k_neighbors: int = 3`.
Lines ~342-353 (CPD call): pass `candidate_radius=candidate_radius, k_neighbors=k_neighbors` to `register_volume_cpd()`.

### 3. `src/python/test/test_pointset.py`

No changes needed — error message and test structure unchanged.

## Verification
1. `cd src/python && uv run pytest test/test_pointset.py -v` — all tests pass
2. `uv run pytest test/ -v` — full suite passes
3. Rerun CPD benchmark on medium, large, tissue to compare quality
