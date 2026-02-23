# Plan: Single-FOV Performance Optimization

## Context

Phases 1-8 established a fully functional Python backend with E2E benchmarks across 3 synthetic presets and 3 real datasets. The benchmark QC CSVs provide per-step timing and memory data. This plan identifies concrete optimization targets based on profiling evidence and proposes implementation in priority order.

**Scope**: Single-FOV performance only. Inter-FOV parallelism is handled by Snakemake (each FOV = independent job distributed across UGER nodes). Optimizing per-FOV efficiency benefits all FOVs uniformly.

**Goal**: Reduce per-FOV runtime by 40-65% and peak memory by 60-75% without sacrificing correctness. All optimizations must preserve existing test results (155 tests + 8 E2E tests).

---

## Profiling Summary

### Per-Step Timing Breakdown (average across datasets)

| Step | Large (1024²×30) | Tissue (3072²×30) | Thick (1024²×100) | tissue-2D (real) | LN (real) | cell-culture (real) |
|------|-----|------|------|------|------|------|
| Load | 6.8s (13%) | 56.6s (12%) | 26.3s (15%) | 82.2s (16%) | 39.1s (21%) | 43.1s (17%) |
| Enhance | 5.0s (10%) | 47.2s (10%) | 17.5s (10%) | 47.5s (9%) | 19.3s (10%) | 17.0s (7%) |
| Registration | 30.1s (58%) | 290.6s (60%) | 103.5s (58%) | 289.9s (56%) | 113.3s (60%) | 109.8s (44%) |
| Spot Finding | 8.5s (17%) | 80.1s (17%) | 28.8s (16%) | 30.1s (6%) | 7.6s (4%) | 4.3s (2%) |
| Extraction | 1.1s (2%) | 7.7s (2%) | 2.9s (2%) | 37.1s (7%) | 6.7s (4%) | 71.0s (28%) |
| Filtration | <0.1s | <0.1s | <0.1s | 0.1s | <0.1s | 0.2s |
| Rotate | — | — | — | 28.6s (6%) | 4.4s (2%) | 3.9s (2%) |

### Memory Scaling

Peak RSS scales linearly at ~70 MB per million voxels:
- Large (31M voxels): 2.4 GB
- Tissue (283M voxels): 19.9 GB
- Thick (104M voxels): 7.5 GB

Peak occurs during registration — all N rounds loaded simultaneously + registration temporaries.

### Bottleneck Ranking

1. **Registration**: 44-60% of runtime — dominated by SimpleITK demons iterations on full-resolution 3D volumes
2. **Extraction**: 2-28% (scales with gene count — 998 genes in cell-culture → 28%)
3. **Load**: 12-21% — I/O bound with high variance (network mount caching)
4. **Enhance**: 7-10% — per-round normalization
5. **Spot Finding**: 2-17% — linear with voxel count, already well-optimized

---

## Tier 1: Quick Wins

### 1.1 Vectorize Barcode Extraction

**Problem**: `extract_from_location()` iterates per-spot in a Python loop (`spots.iloc[i]`). For cell-culture-3D (67K spots × 6 rounds), this is the 28% extraction bottleneck.

**Location**: `src/python/starfinder/barcode/extraction.py` lines 53-89

**Fix**: Replace per-spot Python loop with vectorized NumPy indexing:
```python
# Current: per-spot loop with pandas iloc
for i in range(n_points):
    z = int(spots.iloc[i]["z"])
    y = int(spots.iloc[i]["y"])
    x = int(spots.iloc[i]["x"])
    # clip, slice, sum per spot...

# Proposed: vectorized with numpy arrays
z_arr = spots['z'].values.astype(int)
y_arr = spots['y'].values.astype(int)
x_arr = spots['x'].values.astype(int)
# Vectorized boundary clipping + batch extraction
z0 = np.clip(z_arr - vz, 0, Z)
# ... advanced indexing for uniform-size neighborhoods
# ... handle boundary spots separately (small fraction)
```

**Expected impact**: 5-20x speedup on extraction. cell-culture-3D: 71s → ~5-15s.

**Difficulty**: Medium — need to handle variable-size neighborhoods at volume boundaries.

### 1.2 Vectorize Color Sequence Concatenation

**Problem**: `reads_extraction()` uses `.apply(lambda row: ''.join(...))` per-row.

**Location**: `src/python/starfinder/dataset/fov.py` lines 378-379

**Fix**: Vectorize with NumPy string operations.

**Expected impact**: Minor (1-3s), but cleaner code.

### 1.3 Use float32 in Normalization

**Problem**: `min_max_normalize()` converts each channel to float64 for rescaling (line 41: `.astype(np.float64)`).

**Location**: `src/python/starfinder/preprocessing/normalization.py` line 41

**Fix**: Use float32 instead of float64. The output is uint8 anyway — float32 has more than enough precision for [0, 255] rescaling.

**Expected impact**: 50% reduction in normalization temporary memory.

---

## Tier 2: Streaming Pipeline (Memory Optimization)

### Motivation

The current pipeline loads ALL rounds into memory upfront, then processes them:
```
load ALL rounds → enhance ALL → register ALL → spot_find (ref only) → extract ALL → filter → save
```
Peak memory = N_rounds × per_round_size. For tissue-2D (4 rounds × 3.5 GB), that's ~14 GB base before registration temporaries push it to 20 GB.

Freeing images *after* extraction doesn't help — the pipeline is nearly done by then. The real opportunity is to **never load all rounds simultaneously**.

### 2.1 Streaming Workflow

Code analysis confirms this is feasible:
- **Spot finding** uses only the reference round (`fov.py:344`)
- **Extraction** processes one round at a time in a loop (`fov.py:369`)
- **Registration** per-round is independent (all compare against same reference)
- **No cross-round dependencies** exist between non-ref rounds
- **Filtering** only needs the spot DataFrame (`all_spots`), which accumulates per-round `{round}_color` and `{round}_score` columns. The DataFrame is small (~MBs for 67K spots) and stays in memory throughout. No intermediate files needed.

**Proposed streaming flow:**
```
Phase 1: Reference round (kept in memory throughout)
  load ref → enhance ref → spot_find on ref → extract ref colors

Phase 2: Per non-ref round (one at a time)
  for each sequencing round:
    load round → enhance → register against ref → extract colors → discard round

Phase 3: Finalize (no images needed — only spot DataFrame)
  concatenate color_seq from per-round columns → filter against codebook → save
```

**Implementation**: Add `FOV.run_streaming()` as an alternative to the current batch flow:
```python
@log_step
def run_streaming(self, ...) -> FOV:
    """Memory-efficient streaming pipeline. Peak memory = 2 × round_size."""
    # Phase 1: reference
    self.load_raw_images([self.layers.ref])
    self.enhance_contrast()
    self.spot_finding()
    self._extract_round(self.layers.ref, voxel_size)

    # Phase 2: one round at a time
    for round_name in self.layers.to_register:
        self.load_raw_images([round_name])
        self.enhance_contrast(layers=[round_name])
        self.global_registration(layers_to_register=[round_name])
        self._extract_round(round_name, voxel_size)
        del self.images[round_name]  # free immediately

    # Phase 3: finalize (spot DataFrame only, all images released)
    self._build_color_seq()
    self.reads_filtration()
    return self
```

**Peak memory reduction**:
- tissue-2D: 20 GB → ~7 GB (ref 3.5 GB + one round 3.5 GB)
- cell-culture-3D (6 rounds): 5.5 GB → ~2 GB
- LN (4 rounds): 8 GB → ~4 GB

**Tradeoff**: Slightly slower due to per-round I/O overhead (can't batch-load). But since load is I/O-bound and 12-21% of total, the extra overhead is ~5-10%.

**Difficulty**: Medium — restructure `fov.py` while keeping existing batch API intact.

---

## Tier 3: Hybrid Registration (FFT Global + Spot-Based Local)

Registration is 44-60% of runtime. The current approach uses phase correlation for global alignment and demons for local refinement. The strategy is to **keep FFT-based phase correlation for global shifts** (proven, accurate in all 3 axes) and **replace demons with spot-based TPS for local refinement** (fast, sparse-native).

### 3.1 Architecture

```
Current:   phase_correlate (global) → demons (local)       ~290s per round
Proposed:  phase_correlate (global) → spot-based TPS (local) → demons (fallback)
```

**Why this split works**:
- **Global shift** is a rigid translation — FFT phase correlation handles this well and is already proven across all 6 datasets
- **Local deformation** is what demons spends 200+ seconds on. But for sparse fluorescence images (95-99% background), demons wastes computation on uninformative pixels. Spot-based TPS operates directly on the signal.
- **Demons stays as fallback** for difficult cases (low spot count, large deformations)

### 3.2 Spot-Based Local Registration (TPS)

**Key feasibility findings**:
- Spot detection is fast enough to run *before* registration: tissue-2D detects 67K spots in 30s (vs 290s demons)
- The codebase already has `spot_matching_accuracy()` in `registration/metrics.py` with greedy matching + distance thresholds
- Per-round spot counts (2K-67K) are well within range for point-set algorithms

**Step 1: Detect spots in both rounds**
```python
# Use coarse threshold (SNR=3) for more spot candidates
spots_ref = find_spots_3d(ref_image, intensity_estimation="noise", intensity_threshold=3.0)
spots_mov = find_spots_3d(mov_image, intensity_estimation="noise", intensity_threshold=3.0)
```

**Step 2: Match spots after global alignment**
```python
from scipy.spatial import cKDTree

# Apply global shift to moving spots, then match
mov_coords = spots_mov[['z','y','x']].values + global_shift
tree = cKDTree(spots_ref[['z','y','x']].values)
dists, idxs = tree.query(mov_coords, k=1)
mask = dists < max_distance  # e.g., 10 pixels

matched_ref = spots_ref.iloc[idxs[mask]][['z','y','x']].values
matched_mov = mov_coords[mask]
```

**Step 3: Compute local deformation via TPS**
```python
from scipy.interpolate import RBFInterpolator

# Residual displacements at each matched spot
residuals = matched_ref - matched_mov  # local deformation vectors

# Fit TPS interpolator per axis
interp_dz = RBFInterpolator(matched_mov, residuals[:, 0], kernel='thin_plate_spline')
interp_dy = RBFInterpolator(matched_mov, residuals[:, 1], kernel='thin_plate_spline')
interp_dx = RBFInterpolator(matched_mov, residuals[:, 2], kernel='thin_plate_spline')

# Evaluate on grid to produce dense displacement field
grid = np.mgrid[0:Z, 0:Y, 0:X].reshape(3, -1).T
field = np.stack([interp_dz(grid), interp_dy(grid), interp_dx(grid)], axis=-1)
field = field.reshape(Z, Y, X, 3)
```

**Performance**:
- Spot detection: ~30s (tissue-2D) — already happens in pipeline
- KDTree matching: <1s for 67K spots
- TPS fitting + grid evaluation: ~5-40s depending on matched spot count
- **Total: ~40s vs ~290s demons (7x speedup)**

**Quality**:
- TPS produces smooth C¹ deformations
- For sparse fluorescence, local deformations are typically small (1-5 px), well-suited for TPS
- Spot-based alignment directly optimizes what matters — spot positions, not background pixels

**Fallback to demons**:
- If too few spots match (< ~100), TPS becomes unstable → fall back to demons
- If TPS quality (NCC, Match Rate) is below threshold → fall back to demons

### 3.3 Demons Early Stopping (Fallback Optimization)

When demons is used as fallback, it should converge faster since global alignment is already done.

**Problem**: Default [100, 50, 25] iterations may be excessive for post-global-registration images where residual deformations are small.

**Approach**: Monitor displacement field change between iterations. Stop when `mean(|delta_field|) < threshold`.

SimpleITK's `DemonsRegistrationFilter` supports `AddCommand(EventEnum.sitkIterationEvent, callback)` for per-iteration monitoring. Log the metric value and terminate when plateau detected.

**Expected impact**: 20-40% reduction in demons time when used as fallback.

**Difficulty**: Medium — need to implement callback, tune threshold, validate quality isn't degraded.

### 3.4 External Libraries (if TPS is insufficient)

- `probreg` — Coherent Point Drift (CPD): probabilistic, handles outliers, supports rigid/affine/nonrigid. `pip install probreg`
- `pycpd` — Pure NumPy CPD: simpler, no dependencies. `pip install pycpd`

### 3.5 Validation Plan

Run a focused benchmark (all 6 datasets, 2 FOVs each) comparing:
- Current: phase correlation + demons (baseline)
- Proposed: phase correlation + spot-based TPS
- Proposed + fallback: TPS with demons fallback

Metrics: NCC, Match Rate, runtime, peak memory. Accept if ≥ 95% of current quality.

---

## Implementation Roadmap

### Phase A: Quick Wins (1-2 sessions)

| Task | Files | Impact |
|------|-------|--------|
| A1. Vectorize extraction loop | `barcode/extraction.py` | cell-culture: 71s → ~10s |
| A2. float32 normalization | `preprocessing/normalization.py` | 50% norm memory reduction |
| A3. Vectorize color_seq concat | `dataset/fov.py` | Minor cleanup |

**Tests**: All 155 unit + 8 E2E tests must pass unchanged.

### Phase B: Streaming Pipeline (1-2 sessions)

| Task | Files | Impact |
|------|-------|--------|
| B1. Add `FOV.run_streaming()` | `dataset/fov.py` | Peak memory: N×round → 2×round |
| B2. Per-round load/enhance/register/extract | `dataset/fov.py` | Restructured pipeline flow |
| B3. Validate streaming vs batch results | `test/test_e2e.py` | Bit-identical output |

**Tests**: Add E2E test that runs streaming mode and compares output to batch mode.

### Phase C: Hybrid Registration (2-3 sessions)

| Task | Files | Impact |
|------|-------|--------|
| C1. Spot-based TPS local registration | New: `registration/pointset.py` | Replace demons: 290s → ~40s |
| C2. Integration into FOV pipeline | `dataset/fov.py` | Wire TPS into `local_registration()` |
| C3. Demons fallback + early stopping | `registration/demons.py` | Fallback: 20-40% faster |
| C4. Benchmark TPS vs demons | Benchmark scripts | Validate on all 6 datasets |

**Tests**: Registration quality benchmarks on all 6 datasets. NCC/Match Rate must meet or exceed current demons.

---

## Projected Improvements

### tissue-2D Per-FOV (current: ~510s, 20 GB peak)

| After Phase | Runtime | Memory | Key Change |
|-------------|---------|--------|------------|
| Current | 510s | 20 GB | — |
| Phase A (vectorize) | ~445s (-13%) | 18 GB | Extraction fixed |
| Phase B (streaming) | ~460s (+3% I/O) | 7 GB (-65%) | 2 rounds in memory |
| Phase C (spot-based TPS) | ~200s (-61%) | 7 GB | Local reg: 290s → ~45s |

### cell-culture-3D Per-FOV (current: ~249s, 5.5 GB peak)

| After Phase | Runtime | Memory | Key Change |
|-------------|---------|--------|------------|
| Current | 249s | 5.5 GB | Extraction 28% |
| Phase A | ~190s (-24%) | 5 GB | Extraction vectorized |
| Phase B (streaming) | ~200s | 2 GB (-64%) | 2 rounds in memory |
| Phase C (spot-based TPS) | ~100s (-60%) | 2 GB | Local reg: 110s → ~20s |

---

## Validation Strategy

1. **Correctness**: All 155 unit tests + 8 E2E tests must pass unchanged for Tiers 1-2
2. **Registration quality**: For Tier 3 algorithmic changes, benchmark NCC and Match Rate against current demons on all 6 datasets. Accept if ≥ 95% of current quality.
3. **Memory profiling**: Use `/proc/self/status` RSS tracking (already in benchmark scripts)
4. **Regression benchmarks**: Re-run E2E on `small` synthetic dataset after each phase

---

## Dependencies

- **Tiers 1-2**: No new packages (NumPy, SciPy, scikit-image already available)
- **Tier 3 (core)**: No new packages — TPS uses `scipy.interpolate.RBFInterpolator`, spot matching uses `scipy.spatial.cKDTree`
- **Tier 3 (optional)**: `probreg` or `pycpd` for CPD if TPS is insufficient

---

## Research References

- Thin-plate spline interpolation: `scipy.interpolate.RBFInterpolator(kernel='thin_plate_spline')`
- Coherent Point Drift: Myronenko & Song, IEEE TPAMI 2010. Python: `probreg`, `pycpd`
- MatchPoint (fluorescence point-set registration): bioRxiv 2024, 10.1101/2024.06.22.600172
