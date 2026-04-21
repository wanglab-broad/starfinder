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

## Tier 1: Quick Wins (COMPLETED — commit 560f3f8)

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

## Tier 2: Streaming Pipeline (COMPLETED — commit cb520a7)

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

## Tier 3: Registration Memory & Apply Optimization (COMPLETED — commit 9a7ce22)

### Motivation

Profiling after Tiers 1-2 showed that streaming alone only reduced peak RSS by 11-23% because **FFT temporaries during registration dominate peak RSS**, not round image storage. Three targeted fixes in the global registration path reduced peak RSS by ~50%.

### 3.1 `_make_ref_3d` uint16 Sum

**Problem**: `np.sum(uint8_volume, axis=-1)` defaults to int64 (8 bytes/pixel). Max sum of 4 uint8 channels = 1020, which fits in uint16 (2 bytes/pixel).

**Location**: `src/python/starfinder/dataset/fov.py`

**Fix**: `np.sum(vol, axis=-1, dtype=np.uint16)` — saves 6 bytes/pixel, or 3.2 GB for tissue-sized volumes.

### 3.2 `phase_correlate` rfftn

**Problem**: `np.fft.fftn`/`ifftn` computes the full complex spectrum for real-valued input. Since the input is real, the spectrum has conjugate symmetry — half the data is redundant.

**Location**: `src/python/starfinder/registration/phase_correlation.py`

**Fix**: Use `np.fft.rfftn`/`irfftn`. The last axis is halved from X to X//2+1. Produces complex64 arrays at ~50% the size. Mathematically equivalent (same argmax, relative diff ~4×10⁻⁷).

### 3.3 `apply_shift` Integer Fast Path

**Problem**: Phase correlation always returns integer shifts, but `apply_shift` used `fourier_shift` → FFT round-trip, allocating full complex arrays.

**Location**: `src/python/starfinder/registration/phase_correlation.py`

**Fix**: For integer shifts, use `np.roll` + zero-fill (no FFT). Sub-pixel shifts fall back to the FFT path.

**Results (streaming + memory fixes vs batch baseline)**:
- tissue-2D: 19.9 → 11.0 GB (44-45% reduction), 2.2x faster
- cell-culture-3D: 5.5 → 2.8 GB (48-51% reduction), 2.3x faster

---

## Tier 4: Demons Registration Efficiency

### Motivation

Registration remains the dominant bottleneck at 44-60% of runtime. After Tiers 1-3 optimized everything *around* demons (streaming, global-reg FFTs, extraction), the demons algorithm itself is the next target. The current anti-aliased pyramid implementation has several memory and compute inefficiencies that can be fixed **without changing the algorithm**, preserving exact registration quality.

### Profiling: Where Demons Spends Time and Memory

For tissue-2D (3072×3072×30, padded to 3072×3072×32), one round of `_run_antialias_pyramid`:

| Operation | Time | Peak Memory Spike | Location |
|-----------|------|-------------------|----------|
| Pad + convert to float64 | ~1s | 2 × 2.3 GB = **4.6 GB** | `demons.py:141-146` |
| Level 0: `antialias_resize` FFT (0.25×) | ~40-50s | **4.8 GB** (complex128 at full res) | `pyramid.py:87` |
| Level 0: demons (100 iter) | ~80-100s | ~0.1 GB (coarse) | `demons.py:173` |
| Level 1: `antialias_resize` FFT (0.5×) | ~30-40s | ~1.2 GB | `pyramid.py:87` |
| Level 1: field upsample + crop/pad | ~5s | ~1.0 GB (3 tmp arrays) | `demons.py:177-198` |
| Level 1: demons (50 iter) | ~50-70s | ~0.5 GB | `demons.py:205` |
| Level 2: no downsample (factor=1.0) | 0s | 0 | — |
| Level 2: field upsample + crop/pad | ~10s | ~6.9 GB (full-res field × 3 components) | `demons.py:177-198` |
| Level 2: demons (25 iter) | ~60-80s | ~2 GB (SimpleITK internal) | `demons.py:205` |
| Final displacement field | — | **6.9 GB** (Z×Y×X×3 float64) | `demons.py:208` |
| `apply_deformation` × 4 channels | ~60-80s | ~2.3 GB per call (volume + field copies) | `demons.py:327,336` |

**Key insight**: The `antialias_resize` FFT at level 0 creates a **4.8 GB complex128 spike** from a full-resolution volume — larger than the demons iterations themselves. And `apply_deformation` redundantly converts the displacement field to SimpleITK 4 times (once per channel).

### 4.1 Use `rfftn`/`irfftn` in `antialias_resize`

**Problem**: `pyramid.py:87` uses full-spectrum FFT on real-valued input:
```python
vol = np.real(np.fft.ifftn(np.fft.fftn(vol) * filt))
```
`fftn` produces complex128 arrays at the full input size. Since the input is real, conjugate symmetry means half the spectrum is redundant.

**Location**: `src/python/starfinder/registration/pyramid.py` lines 81-87

**Fix**: Use `rfftn`/`irfftn` (real-input FFT). The last axis is halved from X to X//2+1:
```python
# Also update butterworth_3d to accept rfft_shape for last axis
filt = butterworth_3d_rfft(vol.shape, cutoff, order=2)
vol = np.fft.irfftn(np.fft.rfftn(vol) * filt, s=vol.shape)
```

`butterworth_3d` must be updated to produce a half-spectrum filter: the last axis uses `np.fft.rfftfreq(X)` (length X//2+1) instead of `np.fft.fftfreq(X)` (length X). The first N-1 axes still use `fftfreq`.

**Precedent**: Already proven in `phase_correlate` (Tier 3.2). Same pattern — real input, real output, half-spectrum FFT.

**Expected impact** (tissue-2D level 0 downsample):
- Current: complex128 at 3072×3072×32 = **4.8 GB**
- Proposed: complex128 at 3072×3072×17 = **2.5 GB**
- **Saves ~2.3 GB peak spike per FFT call**
- Butterworth filter also halved: 2.3 GB → 1.2 GB

**Difficulty**: Low — same pattern as phase_correlate rfftn fix.

### 4.2 Use float32 Precision in Antialias Pyramid

**Problem**: `demons.py:145-146` converts padded volumes to float64 to "match MATLAB's double precision." But:
- The **sitk pyramid path** (line 67-68) already uses float32 successfully
- Displacement magnitudes are typically 1-5 pixels — float32 has 7 significant digits, more than enough
- SimpleITK internally uses float32 for demons computation regardless of input precision
- The output is applied to uint8 images — float64 warp precision is wasted

**Location**: `src/python/starfinder/registration/demons.py` lines 145-146; `pyramid.py` line 81

**Fix**: Use float32 throughout the antialias pipeline:
```python
fixed_padded = fixed_padded.astype(np.float32)
moving_padded = moving_padded.astype(np.float32)
```
And in `antialias_resize`:
```python
vol = volume.astype(np.float32)
# FFT produces complex64 instead of complex128 → half memory
```

**Expected impact** (tissue-2D):
- Padded volumes: 2 × 2.3 GB → 2 × 1.15 GB (**-2.3 GB**)
- FFT arrays: complex128 → complex64 (**additional 50% reduction**, stacks with rfftn)
- Butterworth filter: float64 → float32 (**-50%**)
- Displacement field: 6.9 GB → 3.45 GB (**-3.45 GB**)
- **Combined with rfftn: level 0 FFT spike drops from 4.8 GB → ~0.6 GB (8x reduction)**

**Risk**: Low — need to verify registration quality is preserved. Run E2E on small synthetic to confirm NCC/Match Rate unchanged.

**Difficulty**: Low — change dtype constants.

### 4.3 Eliminate Redundant `.astype()` Copies

**Problem**: Several `.astype(np.float64)` calls create unnecessary copies when the array is already the target dtype:

1. `demons.py:166-167` — `fixed_level` from `antialias_resize` is already float64; `.astype(np.float64)` copies the entire volume
2. `demons.py:201` — `upsampled` is already float64; `.astype(np.float64)` on the reversed view copies the entire field
3. `demons.py:327` — `volume.astype(np.float64)` in `apply_deformation` always copies even for float64 input
4. `demons.py:336` — `field_sitk_order.astype(np.float64)` always copies the displacement field

**Location**: `src/python/starfinder/registration/demons.py` lines 166-167, 201, 327, 336

**Fix**: Guard with dtype check, or use `np.asarray(arr, dtype=target)` which avoids copying when already correct:
```python
# Before:
fixed_sitk = sitk.GetImageFromArray(fixed_level.astype(np.float64))

# After:
fixed_sitk = sitk.GetImageFromArray(np.ascontiguousarray(fixed_level))
```

**Expected impact** (tissue-2D): Eliminates 4 redundant full-volume copies totaling **~14 GB** of transient allocations across a single registration call.

**Difficulty**: Trivial — single-line changes.

### 4.4 Reuse `DisplacementFieldTransform` Across Channels

**Problem**: `register_volume_local` (line 408-409) calls `apply_deformation` independently for each of 4 channels. Each call:
1. Converts the displacement field from (dz,dy,dx) → (dx,dy,dz) via `.astype()` — **creates a copy of the 6.9 GB field**
2. Creates a `sitk.DisplacementFieldTransform` — wraps the field in SimpleITK
3. Creates a `sitk.ResampleImageFilter` — sets up interpolation

The displacement field is identical for all channels. Steps 1-3 are redundant 3 out of 4 times.

**Location**: `src/python/starfinder/registration/demons.py` lines 357-411

**Fix**: Factor out transform setup, reuse for all channels:
```python
def register_volume_local(images, ref_image, mov_image, **kwargs):
    displacement_field = demons_register(ref_image, mov_image, **kwargs)

    sitk = _import_sitk()
    input_dtype = images.dtype

    # Create transform ONCE
    field_sitk_order = displacement_field[..., ::-1]
    field_sitk = sitk.GetImageFromArray(
        np.ascontiguousarray(field_sitk_order, dtype=np.float64), isVector=True
    )
    transform = sitk.DisplacementFieldTransform(field_sitk)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)

    # Apply to each channel (only Execute per channel)
    n_channels = images.shape[-1]
    registered = np.empty_like(images)
    for c in range(n_channels):
        vol_sitk = sitk.GetImageFromArray(images[:,:,:,c].astype(np.float64))
        resampler.SetReferenceImage(vol_sitk)
        warped = resampler.Execute(vol_sitk)
        registered[:,:,:,c] = sitk.GetArrayFromImage(warped).astype(input_dtype)

    return registered, displacement_field
```

**Expected impact**:
- **Memory**: Eliminates 3 redundant field copies (3 × 6.9 GB = **20.7 GB transient** for tissue-2D)
- **Time**: Saves ~15-20s of setup overhead per round (~40-60s total across all channels)

**Difficulty**: Low — refactor loop in `register_volume_local`.

### 4.5 Pre-allocate Field Upsample Buffer

**Problem**: `demons.py:184-198` creates a `tmp = np.zeros(target_shape)` array 3 times (once per displacement component dz, dy, dx), each the full size of the current pyramid level.

**Location**: `src/python/starfinder/registration/demons.py` lines 184-198

**Fix**: Pre-allocate a single buffer and reuse:
```python
tmp = np.zeros(target_shape, dtype=np.float64)
for d in range(3):
    tmp[:] = 0
    tmp[slices] = upsampled[..., d][src_slices]
    upsampled[..., d] = tmp
```

**Expected impact**: Minor — saves 2 array allocations at full resolution (~4.6 GB transient for tissue-2D).

**Difficulty**: Trivial.

### 4.6 Cache Butterworth Filter Between Fixed/Moving

**Problem**: At each pyramid level, `antialias_resize` is called for both `fixed_padded` and `moving_padded` with the same shape. Each call independently computes `butterworth_3d(vol.shape, cutoff)` — a full-sized float64 array.

**Location**: `src/python/starfinder/registration/pyramid.py` lines 83-86

**Fix**: Accept an optional pre-computed filter, or cache within `_run_antialias_pyramid`:
```python
if factor < 1.0:
    cutoff = 0.5 * factor
    filt = butterworth_3d(fixed_padded.shape, cutoff, order=2)  # compute once
    fixed_level = antialias_resize(fixed_padded, factor, filt=filt)
    moving_level = antialias_resize(moving_padded, factor, filt=filt)
```

**Expected impact**: Minor time savings (~2-5s per level). Also avoids a ~2.3 GB transient allocation per level.

**Difficulty**: Trivial — add optional parameter.

---

## Tier 5: Hybrid Registration (FFT Global + Spot-Based Local)

Registration is 44-60% of runtime. The current approach uses phase correlation for global alignment and demons for local refinement. The strategy is to **keep FFT-based phase correlation for global shifts** (proven, accurate in all 3 axes) and **replace demons with spot-based TPS for local refinement** (fast, sparse-native).

### 5.1 Architecture

```
Current:   phase_correlate (global) → demons (local)       ~290s per round
Proposed:  phase_correlate (global) → spot-based TPS (local) → demons (fallback)
```

**Why this split works**:
- **Global shift** is a rigid translation — FFT phase correlation handles this well and is already proven across all 6 datasets
- **Local deformation** is what demons spends 200+ seconds on. But for sparse fluorescence images (95-99% background), demons wastes computation on uninformative pixels. Spot-based TPS operates directly on the signal.
- **Demons stays as fallback** for difficult cases (low spot count, large deformations)

### 5.2 Spot-Based Local Registration (TPS)

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

### 5.3 Demons Early Stopping (Fallback Optimization)

When demons is used as fallback, it should converge faster since global alignment is already done.

**Problem**: Default [100, 50, 25] iterations may be excessive for post-global-registration images where residual deformations are small.

**Approach**: Monitor displacement field change between iterations. Stop when `mean(|delta_field|) < threshold`.

SimpleITK's `DemonsRegistrationFilter` supports `AddCommand(EventEnum.sitkIterationEvent, callback)` for per-iteration monitoring. Log the metric value and terminate when plateau detected.

**Expected impact**: 20-40% reduction in demons time when used as fallback.

**Difficulty**: Medium — need to implement callback, tune threshold, validate quality isn't degraded.

### 5.4 External Libraries (if TPS is insufficient)

- `probreg` — Coherent Point Drift (CPD): probabilistic, handles outliers, supports rigid/affine/nonrigid. `pip install probreg`
- `pycpd` — Pure NumPy CPD: simpler, no dependencies. `pip install pycpd`

### 5.5 Validation Plan

Run a focused benchmark (all 6 datasets, 2 FOVs each) comparing:
- Current: phase correlation + demons (baseline)
- Proposed: phase correlation + spot-based TPS
- Proposed + fallback: TPS with demons fallback

Metrics: NCC, Match Rate, runtime, peak memory. Accept if ≥ 95% of current quality.

---

## Implementation Roadmap

### Phase A: Quick Wins (COMPLETED — commit 560f3f8)

| Task | Files | Impact |
|------|-------|--------|
| A1. Vectorize extraction loop | `barcode/extraction.py` | cell-culture: 71s → ~10s |
| A2. float32 normalization | `preprocessing/normalization.py` | 50% norm memory reduction |
| A3. Vectorize color_seq concat | `dataset/fov.py` | Minor cleanup |

**Tests**: All 155 unit + 8 E2E tests pass unchanged.

### Phase B: Streaming Pipeline (COMPLETED — commit cb520a7)

| Task | Files | Impact |
|------|-------|--------|
| B1. Add `FOV.run_streaming()` | `dataset/fov.py` | Peak memory: N×round → 2×round |
| B2. Per-round load/enhance/register/extract | `dataset/fov.py` | Restructured pipeline flow |
| B3. Validate streaming vs batch results | `test/test_e2e.py` | Bit-identical output |

**Tests**: E2E test runs streaming mode and confirms identical output to batch mode.

### Phase C: Registration Memory Fixes (COMPLETED — commit 9a7ce22)

| Task | Files | Impact |
|------|-------|--------|
| C1. `_make_ref_3d` uint16 sum | `dataset/fov.py` | -3.2 GB on tissue volumes |
| C2. `phase_correlate` rfftn/irfftn | `registration/phase_correlation.py` | ~50% less FFT memory |
| C3. `apply_shift` integer fast path | `registration/phase_correlation.py` | No FFT for integer shifts |

**Results**: tissue-2D: 19.9 → 11.0 GB (-45%), 2.2x faster.

### Phase D: Demons Registration Efficiency (1-2 sessions)

| Task | Files | Impact |
|------|-------|--------|
| D1. `rfftn`/`irfftn` in `antialias_resize` | `registration/pyramid.py` | -2.3 GB FFT spike (tissue) |
| D2. float32 precision in antialias pyramid | `registration/demons.py`, `registration/pyramid.py` | Halve all array sizes |
| D3. Eliminate redundant `.astype()` copies | `registration/demons.py` | -14 GB transient allocs |
| D4. Reuse `DisplacementFieldTransform` across channels | `registration/demons.py` | -20 GB transient, -40s time |
| D5. Pre-allocate field upsample buffer | `registration/demons.py` | -4.6 GB transient |
| D6. Cache Butterworth filter | `registration/pyramid.py` | -2.3 GB transient, -5s |

**Tests**: All 155 unit + 8 E2E tests must pass unchanged. Registration quality (NCC, Match Rate) identical — no algorithmic change.

### Phase E: Hybrid Registration (2-3 sessions)

| Task | Files | Impact |
|------|-------|--------|
| E1. Spot-based TPS local registration | New: `registration/pointset.py` | Replace demons: 290s → ~40s |
| E2. Integration into FOV pipeline | `dataset/fov.py` | Wire TPS into `local_registration()` |
| E3. Demons fallback + early stopping | `registration/demons.py` | Fallback: 20-40% faster |
| E4. Benchmark TPS vs demons | Benchmark scripts | Validate on all 6 datasets |

**Tests**: Registration quality benchmarks on all 6 datasets. NCC/Match Rate must meet or exceed current demons.

---

## Projected Improvements

### tissue-2D Per-FOV (current: ~510s, 20 GB peak)

| After Phase | Runtime | Memory | Key Change |
|-------------|---------|--------|------------|
| Baseline | 510s | 20 GB | — |
| Phase A (vectorize) | ~445s (-13%) | 18 GB | Extraction fixed |
| Phase B (streaming) | ~460s (+3% I/O) | 7 GB (-65%) | 2 rounds in memory |
| Phase C (global reg memory) | ~230s (-55%) | 11 GB (-45%) | rfftn + integer roll |
| **Phase D (demons efficiency)** | **~190s (-63%)** | **~6 GB (-70%)** | **float32 + reuse transform** |
| Phase E (spot-based TPS) | ~120s (-76%) | ~5 GB (-75%) | Local reg: 290s → ~45s |

### cell-culture-3D Per-FOV (current: ~249s, 5.5 GB peak)

| After Phase | Runtime | Memory | Key Change |
|-------------|---------|--------|------------|
| Baseline | 249s | 5.5 GB | Extraction 28% |
| Phase A (vectorize) | ~190s (-24%) | 5 GB | Extraction vectorized |
| Phase B (streaming) | ~200s | 2 GB (-64%) | 2 rounds in memory |
| Phase C (global reg memory) | ~110s (-56%) | 2.8 GB (-49%) | rfftn + integer roll |
| **Phase D (demons efficiency)** | **~90s (-64%)** | **~1.5 GB (-73%)** | **float32 + reuse transform** |
| Phase E (spot-based TPS) | ~55s (-78%) | ~1.2 GB (-78%) | Local reg: 110s → ~20s |

---

## Validation Strategy

1. **Correctness**: All 155 unit tests + 8 E2E tests must pass unchanged for Phases A-D
2. **Registration quality**: For Phase D, verify NCC and Match Rate are unchanged (no algorithmic change, only precision/memory). For Phase E, benchmark against demons on all 6 datasets. Accept if ≥ 95% of current quality.
3. **Memory profiling**: Use `/proc/self/status` RSS tracking (already in benchmark scripts)
4. **Regression benchmarks**: Re-run E2E on `small` synthetic dataset after each phase

---

## Dependencies

- **Phases A-D**: No new packages (NumPy, SciPy, scikit-image, SimpleITK already available)
- **Phase E (core)**: No new packages — TPS uses `scipy.interpolate.RBFInterpolator`, spot matching uses `scipy.spatial.cKDTree`
- **Phase E (optional)**: `probreg` or `pycpd` for CPD if TPS is insufficient

---

## Research References

- Thin-plate spline interpolation: `scipy.interpolate.RBFInterpolator(kernel='thin_plate_spline')`
- Coherent Point Drift: Myronenko & Song, IEEE TPAMI 2010. Python: `probreg`, `pycpd`
- MatchPoint (fluorescence point-set registration): bioRxiv 2024, 10.1101/2024.06.22.600172
