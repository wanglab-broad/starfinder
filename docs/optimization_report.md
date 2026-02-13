# STARfinder Codebase Optimization Report

**Date:** 2026-02-13
**Scope:** Full Python backend + Snakemake workflow analysis

---

## Executive Summary

This report identifies optimization opportunities across the STARfinder codebase, covering the Python backend (~6,100 lines across 35 files) and the Snakemake workflow (~780 lines across 8 rule files). Findings are organized by priority (high/medium/low) with estimated impact and implementation complexity.

**Key findings:**
- 3 high-priority optimizations in the core computational pipeline
- 5 medium-priority improvements for memory management and parallelism
- 6 low-priority refinements for code quality and minor performance gains
- 4 Snakemake workflow optimizations

---

## 1. High-Priority Optimizations

### 1.1 Vectorize Barcode Extraction Loop

**Location:** `src/python/starfinder/barcode/extraction.py:53-89`

**Problem:** `extract_from_location()` uses a pure Python for-loop over all spots with per-spot `DataFrame.iloc[]` access. For a typical FOV with 100-400 spots across 4 rounds, this creates 400-1600 Python loop iterations with pandas indexing overhead per iteration.

**Current code:**
```python
for i in range(n_points):
    z = int(spots.iloc[i]["z"])
    y = int(spots.iloc[i]["y"])
    x = int(spots.iloc[i]["x"])
    # ... per-spot voxel extraction, normalization, assignment
```

**Proposed fix:** Pre-extract coordinates as NumPy arrays and batch-process:
```python
zs = spots["z"].to_numpy(dtype=int)
ys = spots["y"].to_numpy(dtype=int)
xs = spots["x"].to_numpy(dtype=int)
```

This eliminates repeated `iloc` overhead. Full vectorization of the voxel summation is harder due to variable boundary clipping, but the coordinate extraction alone would reduce overhead noticeably.

**Impact:** ~2-5x speedup for extraction step
**Complexity:** Low

---

### 1.2 Parallelize Per-Channel Registration

**Locations:**
- `src/python/starfinder/registration/phase_correlation.py:129-130` (`register_volume`)
- `src/python/starfinder/registration/demons.py:408-409` (`register_volume_local`)

**Problem:** Both global and local registration apply transformations to each channel sequentially in a for-loop. Each channel's transformation is independent -- the shift/displacement field is computed once from the reference channel, then applied identically to all channels.

**Current code (both functions):**
```python
for c in range(n_channels):
    registered[:, :, :, c] = apply_shift(images[:, :, :, c], correction, workers=workers)
```

**Proposed fix:** Use `concurrent.futures.ThreadPoolExecutor` for channel-parallel application. Since `apply_shift` releases the GIL during FFT computation (via scipy.fft `workers` parameter) and `apply_deformation` spends most time in SimpleITK C++ code, threading would provide real parallelism:

```python
from concurrent.futures import ThreadPoolExecutor

def _apply_channel(c):
    return apply_shift(images[:, :, :, c], correction, workers=workers)

with ThreadPoolExecutor(max_workers=n_channels) as pool:
    results = list(pool.map(_apply_channel, range(n_channels)))
for c, result in enumerate(results):
    registered[:, :, :, c] = result
```

**Impact:** Up to ~4x speedup for 4-channel volumes (typical STARmap has 4 channels)
**Complexity:** Low

---

### 1.3 Optimize Spot Matching for Large Spot Counts

**Location:** `src/python/starfinder/registration/metrics.py:228-234`

**Problem:** `spot_matching_accuracy()` builds a complete list of all valid (within `max_distance`) pairwise matches via a nested Python loop, then sorts them. For N reference spots and M moving spots, this is O(N*M) in the inner loop with Python-level iteration.

**Current code:**
```python
pairs = []
for i in range(len(ref_spots)):
    for j in range(len(mov_spots)):
        if distances[i, j] <= max_distance:
            pairs.append((distances[i, j], i, j))
pairs.sort()
```

**Proposed fix:** Replace with vectorized threshold filtering and `scipy.optimize.linear_sum_assignment` (Hungarian algorithm) for optimal matching, or at minimum use NumPy masking:

```python
# Vectorized pair extraction
valid = distances <= max_distance
ri, mi = np.where(valid)
dists = distances[ri, mi]
order = np.argsort(dists)
# Greedy assignment with numpy arrays
```

**Impact:** O(N*M) Python loop -> vectorized NumPy operations. For 1000 spots: ~10-50x speedup
**Complexity:** Low-Medium

---

## 2. Medium-Priority Optimizations

### 2.1 Add Memory Cleanup to FOV After Subtile Creation

**Location:** `src/python/starfinder/dataset/fov.py:406-461`

**Problem:** `create_subtiles()` extracts cropped regions from all images and saves them to NPZ files, but retains all full-resolution images in `self.images`. For a tissue dataset with 4 rounds of 3072x3072x30x4 volumes, this keeps ~1.4 GB in memory after the subtiles have been saved.

**Proposed fix:** Add an optional `release_images` parameter:

```python
def create_subtiles(self, *, out_dir=None, release_images: bool = False) -> pd.DataFrame:
    # ... existing code ...
    if release_images:
        self.images.clear()
    return coords_df
```

**Impact:** ~50% memory reduction for subtile workflows
**Complexity:** Low

---

### 2.2 Reduce Temporary Allocations in Morphological Reconstruction

**Location:** `src/python/starfinder/preprocessing/morphology.py:78-89`

**Problem:** `morphological_reconstruction()` creates 6 temporary arrays per Z-slice per channel: `marker`, `obr`, `subtracted` (x2 from int16 cast), `top`, `bot`, plus the final `enhanced`. For a volume with 30 Z-slices and 4 channels, that is 720 temporary allocations.

**Current code:**
```python
for c in range(volume.shape[3]):
    for z in range(volume.shape[0]):
        slc = volume[z, :, :, c]
        marker = erosion(slc, se)
        obr = reconstruction(marker, slc, method="dilation")
        subtracted = slc.astype(np.int16) - obr.astype(np.int16)
        subtracted = np.clip(subtracted, 0, 255).astype(np.uint8)
        top = white_tophat(subtracted, se).astype(np.int16)
        bot = black_tophat(subtracted, se).astype(np.int16)
        enhanced = subtracted.astype(np.int16) + top - bot
        result[z, :, :, c] = np.clip(enhanced, 0, 255).astype(np.uint8)
```

**Proposed fix:** Pre-allocate reusable buffers outside the loop and minimize type conversions:

```python
# Pre-allocate buffers
buf_i16 = np.empty((volume.shape[1], volume.shape[2]), dtype=np.int16)
buf_u8 = np.empty((volume.shape[1], volume.shape[2]), dtype=np.uint8)

for c in range(volume.shape[3]):
    for z in range(volume.shape[0]):
        slc = volume[z, :, :, c]
        marker = erosion(slc, se)
        obr = reconstruction(marker, slc, method="dilation")
        np.subtract(slc, obr, out=buf_i16, casting='unsafe')
        np.clip(buf_i16, 0, 255, out=buf_i16)
        buf_u8[:] = buf_i16  # single cast
        top = white_tophat(buf_u8, se)
        bot = black_tophat(buf_u8, se)
        # Compute final in int16 space
        np.add(buf_u8, top, out=buf_i16, casting='unsafe')
        np.subtract(buf_i16, bot, out=buf_i16)
        np.clip(buf_i16, 0, 255, out=buf_i16)
        result[z, :, :, c] = buf_i16
```

**Impact:** ~30% fewer allocations, reduced GC pressure, better cache behavior
**Complexity:** Medium

---

### 2.3 Parallel Channel Loading in I/O

**Location:** `src/python/starfinder/io/tiff.py:126-137`

**Problem:** `load_image_stacks()` loads channels sequentially in a for-loop. Each channel requires disk I/O (potentially from network mount) and decompression. For 4 channels, this serializes ~200ms of I/O.

**Current code:**
```python
for channel in channel_order:
    matches = list(search_dir.glob(f"*{channel}*.tif"))
    img = load_multipage_tiff(matches[0], convert_uint8=False)
    channel_images.append(img)
```

**Proposed fix:** Use `concurrent.futures.ThreadPoolExecutor` since TIFF I/O is I/O-bound (not CPU-bound), so threading is appropriate:

```python
from concurrent.futures import ThreadPoolExecutor

def _load_channel(channel):
    matches = list(search_dir.glob(f"*{channel}*.tif"))
    if not matches:
        raise ValueError(f"No TIFF file found matching channel pattern: {channel}")
    return load_multipage_tiff(matches[0], convert_uint8=False)

with ThreadPoolExecutor(max_workers=len(channel_order)) as pool:
    channel_images = list(pool.map(_load_channel, channel_order))
```

**Impact:** ~2-4x speedup for multi-channel loading (I/O-bound)
**Complexity:** Low

---

### 2.4 Use float32 Instead of float64 in Normalization

**Location:** `src/python/starfinder/preprocessing/normalization.py:33`

**Problem:** `min_max_normalize()` converts each channel to `float64` for min/max computation. Since the output is `uint8`, float32 provides more than sufficient precision and uses half the memory.

**Current code:**
```python
ch = volume[:, :, :, c].astype(np.float64)
```

**Proposed fix:**
```python
ch = volume[:, :, :, c].astype(np.float32)
```

**Impact:** 2x less intermediate memory during normalization
**Complexity:** Trivial

---

### 2.5 Cache Reference Histogram in FOV `hist_equalize()`

**Location:** `src/python/starfinder/dataset/fov.py:116-133`

**Problem:** When `hist_equalize()` is called, `histogram_match()` from scikit-image recomputes the reference CDF for every channel of every round. The reference volume is the same across all calls.

**Current code (fov.py):**
```python
reference = self.images[self.layers.ref][:, :, :, ref_channel]
for name in layers:
    self.images[name] = histogram_match(self.images[name], reference, nbins=nbins)
```

The underlying `match_histograms` (scikit-image) recomputes reference statistics each call.

**Proposed fix:** Pre-compute reference CDF once and pass it to a modified histogram matching function that accepts pre-computed statistics. Alternatively, simply note that scikit-image's `match_histograms` is already fast for uint8 data (only 256 bins), making this optimization low-impact for uint8 volumes.

**Impact:** Minor (~10-20% for histogram matching step)
**Complexity:** Medium

---

## 3. Low-Priority Optimizations

### 3.1 Use `.real` Instead of `np.abs()` After IFFT in `apply_shift`

**Location:** `src/python/starfinder/registration/phase_correlation.py:76`

**Current:** `result = np.abs(ifftn(shifted_fft, workers=workers))`

The output of `ifftn` for a real-valued input shifted in frequency domain is theoretically real (imaginary part is numerical noise). Using `.real` avoids computing the magnitude of the complex array:

```python
result = ifftn(shifted_fft, workers=workers).real
```

Then clip negatives from numerical noise:
```python
np.maximum(result, 0, out=result)
```

**Impact:** ~10-20% faster for IFFT post-processing, avoids sqrt computation
**Complexity:** Trivial

### 3.2 Avoid Redundant 3D/4D Dimension Wrapping

**Location:** `src/python/starfinder/preprocessing/normalization.py:27-29` and `morphology.py:34-36,71-73`

All preprocessing functions add a channel dimension for 3D inputs then remove it at the end. This pattern is repeated in 4 functions. While not expensive (view operation), it adds code complexity.

**Proposed fix:** Create a decorator or context manager that handles dimension wrapping:

```python
def _ensure_4d(func):
    @wraps(func)
    def wrapper(volume, *args, **kwargs):
        is_3d = volume.ndim == 3
        if is_3d:
            volume = volume[..., np.newaxis]
        result = func(volume, *args, **kwargs)
        return result[..., 0] if is_3d else result
    return wrapper
```

**Impact:** Code quality improvement, no performance change
**Complexity:** Low

### 3.3 Use `np.stack` with Pre-Computed Shapes in `load_image_stacks`

**Location:** `src/python/starfinder/io/tiff.py:159-160`

Current approach computes minimum dimensions, then crops, then stacks. Could pre-allocate the output array and copy directly:

```python
stacked = np.empty((min_z, min_y, min_x, len(channel_order)), dtype=channel_images[0].dtype)
for i, img in enumerate(channel_images):
    stacked[:, :, :, i] = img[:min_z, :min_y, :min_x]
```

This avoids the intermediate `cropped_images` list and the `np.stack` allocation.

**Impact:** Minor memory improvement for large volumes
**Complexity:** Low

### 3.4 Type-Annotate Return Values Consistently

Several functions have incomplete type annotations (e.g., metadata dicts). Adding `TypedDict` for structured return values would improve IDE support and catch errors earlier.

**Impact:** Developer experience only
**Complexity:** Low

### 3.5 Add `workers=-1` Default for FFT in `register_volume`

**Location:** `src/python/starfinder/registration/phase_correlation.py:104`

`register_volume()` passes `workers` through to `phase_correlate` and `apply_shift`, but the default is `None` (single-threaded). Since registration is typically the bottleneck in the pipeline, defaulting to multi-threaded FFT would be beneficial.

**Impact:** Depends on hardware; ~1.5-3x for multi-core systems
**Complexity:** Trivial (change default parameter)

### 3.6 Avoid Object Dtype in Extraction Color Sequences

**Location:** `src/python/starfinder/barcode/extraction.py:50`

```python
color_seq = np.empty(n_points, dtype=object)
```

Using `object` dtype for string values ("1"-"4", "M", "N") prevents NumPy vectorization. A fixed-width string dtype `U1` would be more efficient:

```python
color_seq = np.empty(n_points, dtype='U1')
```

**Impact:** Minor memory/performance improvement
**Complexity:** Trivial

---

## 4. Snakemake Workflow Optimizations

### 4.1 Break Hardcoded Sample Dependencies

**Location:** `workflow/rules/reads-assignment.smk`, `workflow/rules/stitching.smk`

**Problem:** `create_tile_config` and `reads_assignment` use `expand(..., sample=SAMPLE)` in their input functions, which forces evaluation of ALL samples before any single sample can proceed to reads assignment. This creates an artificial serialization bottleneck.

**Proposed fix:** Replace hardcoded `expand` with per-sample dynamic input functions that only depend on the current sample's outputs.

**Impact:** Enables per-sample pipelining instead of all-sample synchronization
**Complexity:** Medium

### 4.2 Add Dynamic Retry Backoff to All MATLAB Rules

**Location:** `workflow/rules/registration.smk`

**Problem:** `rsf_single_fov` and `nuclei_registration` use static runtime values. If a job exceeds the configured runtime, it is killed with no retry backoff. Other rules (`gr_single_fov_subtile`, `lrsf_single_fov_subtile`, etc.) correctly use `make_get_runtime()` which multiplies runtime by attempt number.

**Proposed fix:** Apply `make_get_runtime()` to all MATLAB rules:

```python
resources:
    runtime=make_get_runtime('rsf_single_fov')
```

**Impact:** Reduces job failures from timeout on variable-size FOVs
**Complexity:** Trivial

### 4.3 Add Thread Specifications to Python Rules

**Location:** `workflow/rules/spot-finding.smk` (stitch_subtile), `workflow/rules/reads-assignment.smk`

**Problem:** Python rules like `stitch_subtile` and `reads_assignment` don't specify threads, defaulting to 1. These scripts perform pandas operations and H5AD manipulation that could benefit from multiple threads.

**Proposed fix:** Add `threads: 2` or `threads: 4` to Python-heavy rules.

**Impact:** Better resource utilization on cluster
**Complexity:** Trivial

### 4.4 Remove Unused gcloud-backup Rule

**Location:** `workflow/rules/gcloud-backup.smk`

**Problem:** This 74-line rule file contains hardcoded test data paths (`test.zip`, `sample-dataset.zip`) and is never called in the workflow.

**Proposed fix:** Remove the file and its `include` in the Snakefile, or properly parameterize it if cloud backup is needed.

**Impact:** Reduces maintenance burden and confusion
**Complexity:** Trivial

---

## 5. Summary Table

| # | Optimization | Module | Priority | Impact | Complexity |
|---|---|---|---|---|---|
| 1.1 | Vectorize extraction loop | barcode/extraction | High | 2-5x | Low |
| 1.2 | Parallelize per-channel registration | registration | High | ~4x | Low |
| 1.3 | Vectorize spot matching | registration/metrics | High | 10-50x (1000+ spots) | Low-Med |
| 2.1 | FOV memory cleanup after subtiles | dataset/fov | Medium | 50% mem reduction | Low |
| 2.2 | Reduce morphology temporaries | preprocessing | Medium | 30% fewer allocs | Medium |
| 2.3 | Parallel channel loading | io/tiff | Medium | 2-4x I/O | Low |
| 2.4 | float32 in normalization | preprocessing | Medium | 2x less mem | Trivial |
| 2.5 | Cache reference histogram | dataset/fov | Medium | 10-20% | Medium |
| 3.1 | `.real` instead of `abs()` | registration | Low | 10-20% | Trivial |
| 3.2 | Dimension wrapping decorator | preprocessing | Low | Code quality | Low |
| 3.3 | Pre-allocate stacked array | io/tiff | Low | Minor mem | Low |
| 3.4 | Type annotations | All | Low | Dev experience | Low |
| 3.5 | Default `workers=-1` for FFT | registration | Low | 1.5-3x | Trivial |
| 3.6 | Avoid object dtype | barcode/extraction | Low | Minor | Trivial |
| 4.1 | Break sample dependencies | Snakemake | Medium | Pipeline speed | Medium |
| 4.2 | Retry backoff for MATLAB | Snakemake | Medium | Fewer failures | Trivial |
| 4.3 | Thread specs for Python rules | Snakemake | Low | Resource use | Trivial |
| 4.4 | Remove unused gcloud rule | Snakemake | Low | Maintenance | Trivial |

---

## 6. Recommended Implementation Order

1. **Quick wins (trivial complexity, measurable impact):**
   - 2.4: float32 in normalization
   - 3.1: `.real` instead of `abs()`
   - 3.5: Default `workers=-1` for FFT
   - 3.6: Avoid object dtype
   - 4.2: Retry backoff for MATLAB rules

2. **High-impact, low-complexity:**
   - 1.1: Vectorize extraction loop
   - 1.2: Parallelize per-channel registration
   - 2.1: FOV memory cleanup
   - 2.3: Parallel channel loading

3. **Medium effort, good payoff:**
   - 1.3: Vectorize spot matching
   - 2.2: Reduce morphology temporaries
   - 4.1: Break sample dependencies

4. **Polish:**
   - 2.5: Cache reference histogram
   - 3.2-3.4: Code quality improvements
   - 4.3-4.4: Workflow cleanup
