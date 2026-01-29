# STARfinder Python Backend Migration Plan

**Milestone 2: Rewrite the backend with Python & Improve Code Quality**
**Last updated:** 2026-01-29

## Executive Summary

Migrate 28 MATLAB files (~2,500 lines) to a Python package managed with `uv`, enabling modern data formats (HDF5, SpatialData) and 2-5x performance improvements.

---

## User Decisions

- **Directory structure**: Rename `code-base` → `src`, with `src/matlab` and `src/python` backends
- **Package management**: Use `uv` (not conda) for Python dependency management
- **Registration library**:
  - Global (phase correlation): Pure NumPy/SciPy (no extra dependencies)
  - Local (demons): SimpleITK (optional dependency, only needed for `local_registration`)
- **Priority**: Core algorithms first, then modern data formats

---

## Current State

### Already in Python (No Migration Needed)
- `stardist_segmentation.py` - StarDist 2D/3D segmentation
- `reads_assignment.py` - Cell-read assignment, H5AD output
- `create_sample_h5ad.py` - Sample-level H5AD merging
- Preprocessing scripts (DAPI enhancement, overlays, stitching)

### MATLAB Components to Migrate

| Category | Key Files | Priority | Python Equivalent |
|----------|-----------|----------|-------------------|
| **Registration** | `DFTRegister2D/3D.m`, `DFTApply2D/3D.m`, `RegisterImagesGlobal.m`, `RegisterImagesLocal.m` | Critical | scipy.fft (pure implementation) |
| **Spot Finding** | `SpotFindingMax3D.m` | High | scipy.ndimage, skimage |
| **Image I/O** | `LoadImageStacks.m`, `LoadMultipageTiff.m`, `SaveSingleStack.m` | High | tifffile |
| **Barcode** | `EncodeBases.m`, `DecodeCS.m`, `FilterReads.m`, `ExtractFromLocation.m` | Medium | NumPy, pandas |
| **Preprocessing** | `MinMaxNorm.m`, `Tophat.m`, `MorphologicalReconstruction.m` | Medium | skimage, scipy.ndimage |
| **Orchestrator** | `STARMapDataset.m` (1096 lines) | High | Python dataclass |

### Known Bugs to Fix
- `DFTRegister2D.m:18` - Uses 3D indexing `[i,j,k]` for 2D array
- `DFTApply2D.m` - Extra dimension in boundary zeroing
- Inconsistent boundary conditions between 2D/3D versions

---

## Directory Restructure

### Before
```
starfinder/
├── code-base/
│   ├── src/           # MATLAB files
│   └── matlab-addon/
├── workflow/
└── ...
```

### After
```
starfinder/
├── src/
│   ├── matlab/        # Existing MATLAB files (moved from code-base/src)
│   ├── matlab-addon/  # MATLAB addons (moved from code-base/matlab-addon)
│   └── python/        # New Python package
│       └── starfinder/
├── workflow/
└── ...
```

---

## Python Package Architecture

```
src/python/
├── pyproject.toml          # uv-compatible (PEP 517/518)
├── uv.lock                  # Lock file for reproducibility
├── starfinder/
│   ├── __init__.py
│   ├── io/                  # tiff.py, hdf5.py (future)
│   ├── registration/        # phase_correlation.py, demons.py
│   ├── spotfinding/         # local_maxima.py
│   ├── barcode/             # encoding.py, decoding.py, codebook.py
│   ├── preprocessing/       # normalization.py, morphology.py
│   ├── dataset/             # starmap_dataset.py (main class)
│   └── utils/               # metadata.py, visualization.py
├── tests/
└── benchmarks/
```

### pyproject.toml (uv-compatible)

```toml
[project]
name = "starfinder"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "scikit-image>=0.21",
    "tifffile>=2023.0",
    "pandas>=2.0",
    "h5py>=3.9",
]

[project.optional-dependencies]
local-registration = ["SimpleITK>=2.3"]  # Required for demons/local registration
spatialdata = ["spatialdata>=0.1", "spatialdata-io>=0.1"]
dev = ["pytest>=7.0", "pytest-cov>=4.0", "ruff>=0.1"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = ["pytest>=7.0", "pytest-cov>=4.0", "ruff>=0.1"]
```

---

## Implementation Phases

### Phase 0: Directory Restructure
**Goal**: Reorganize codebase for dual backend support

**Tasks**:
1. Create new directory structure
2. Move MATLAB files: `code-base/src/*` → `src/matlab/`
3. Move addons: `code-base/matlab-addon/*` → `src/matlab-addon/`
4. Update all Snakemake paths in `workflow/rules/*.smk`
5. Update `CLAUDE.md` and documentation

**Files to Modify**:
- `workflow/rules/common.smk` - Update `STARFINDER_PATH` references
- `workflow/scripts/*.m` - Update `addpath()` calls
- `CLAUDE.md` - Update directory structure docs

### Cross-Cutting: Backend Dispatch (Boring + Measurable)
**Goal**: Keep Snakemake logic simple while enabling apples-to-apples MATLAB vs Python comparisons.

**Approach**:
- Prefer *parallel scripts per step* (e.g., `workflow/scripts/rsf_single_fov.m` and `workflow/scripts/py_rsf_single_fov.py`) instead of branching deeply inside rule bodies.
- Keep dispatch centralized in `workflow/rules/common.smk` (e.g., `get_backend()` + `get_step_script(step)`), so rule files remain stable as the Python backend grows.
- Enforce identical output paths/filenames between backends for comparable rules to simplify validation and benchmarking.

### Phase 1: Package Setup & I/O Module
**Goal**: Initialize uv project and implement image I/O

**Tasks**:
1. Initialize with `uv init src/python`
2. Create `pyproject.toml` with dependencies
3. Implement `starfinder.io.tiff`:
   - `load_multipage_tiff()` - equivalent to `LoadMultipageTiff.m`
   - `load_image_stacks()` - equivalent to `LoadImageStacks.m`
   - `save_stack()` - equivalent to `SaveSingleStack.m`
4. Set up pytest with fixtures

**Commands**:
```bash
cd src/python
uv init
uv add numpy scipy scikit-image tifffile pandas h5py
uv add --dev pytest pytest-cov ruff
```

**Files to Create**:
- `src/python/pyproject.toml`
- `src/python/starfinder/io/tiff.py`
- `src/python/tests/conftest.py`

### Phase 2: Registration Module (Critical)
**Goal**: Port DFT-based phase correlation (pure NumPy/SciPy)

**Tasks**:
1. Implement `starfinder.registration.phase_correlation`:
   - `phase_correlate_2d()` - fix bug from `DFTRegister2D.m`
   - `phase_correlate_3d()` - equivalent to `DFTRegister3D.m`
   - `apply_shift_2d/3d()` - equivalents to `DFTApply2D/3D.m`
   - `register_global()` - equivalent to `RegisterImagesGlobal.m`
2. Implement `starfinder.registration.demons`:
   - `register_local()` using SimpleITK - equivalent to `RegisterImagesLocal.m`
   - SimpleITK is an optional dependency (only required if local registration is used)
3. Create numerical equivalence tests
4. Benchmark suite

**Reference**: Use `dev/DFT_REGISTRATION_REVIEW.md` Python examples (lines 219-302)

**Files to Create**:
- `src/python/starfinder/registration/phase_correlation.py`
- `src/python/starfinder/registration/demons.py`
- `src/python/tests/test_registration/`
- `src/python/benchmarks/benchmark_registration.py`

### Phase 3: Spot Finding & Extraction
**Goal**: Port 3D local maxima detection

**Tasks**:
1. Implement `starfinder.spotfinding.local_maxima`:
   - `find_spots_3d()` using `scipy.ndimage.maximum_filter`
2. Implement `starfinder.barcode.extraction`:
   - `extract_from_location()` - voxel intensity extraction

**Files to Create**:
- `src/python/starfinder/spotfinding/local_maxima.py`
- `src/python/starfinder/barcode/extraction.py`

### Phase 4: Barcode Processing
**Goal**: Port encoding/decoding and codebook matching

**Tasks**:
1. Implement encoding/decoding (dict-based):
   - `encode_bases()`, `decode_colorspace()`
2. Implement codebook handling:
   - `load_codebook()`, `filter_reads()`, `filter_reads_multi_segment()`

**Files to Create**:
- `src/python/starfinder/barcode/encoding.py`
- `src/python/starfinder/barcode/codebook.py`

### Phase 5: Preprocessing
**Goal**: Port image processing functions

**Tasks**:
1. `min_max_normalize()` using skimage.exposure
2. `morphological_reconstruction()` using scipy.ndimage
3. `tophat_filter()` using skimage.morphology
4. `make_projections()` using numpy

**Files to Create**:
- `src/python/starfinder/preprocessing/normalization.py`
- `src/python/starfinder/preprocessing/morphology.py`

### Phase 6: Dataset Class & Snakemake Integration
**Goal**: Create main orchestrator and integrate with workflow

**Tasks**:
1. Implement `STARMapDataset` Python class
2. Add `backend: python | matlab` config option
3. Create Python rule scripts (e.g., `py_rsf_single_fov.py`)
4. Update `common.smk` with backend dispatch logic
5. Integration tests with test datasets

**Files to Create/Modify**:
- `src/python/starfinder/dataset/starmap_dataset.py`
- `workflow/scripts/py_rsf_single_fov.py`
- `workflow/rules/common.smk` - add `get_backend()` function

### Phase 7: Modern Data Formats (Future)
**Goal**: Add HDF5/SpatialData support (after core is stable)

**Tasks**:
1. Implement `starfinder.io.hdf5` for intermediate storage
2. Implement SpatialData export
3. Add scPortrait export for single-cell crops

---

## Critical Files

| Current Path | New Path | Purpose |
|--------------|----------|---------|
| `code-base/src/STARMapDataset.m` | `src/matlab/STARMapDataset.m` | Main class to replicate |
| `code-base/src/DFTRegister3D.m` | `src/matlab/DFTRegister3D.m` | Core registration |
| `code-base/src/SpotFindingMax3D.m` | `src/matlab/SpotFindingMax3D.m` | Spot detection |
| `dev/DFT_REGISTRATION_REVIEW.md` | (unchanged) | Python examples |
| `workflow/rules/common.smk` | (unchanged) | Backend integration |

---

## Validation Strategy

1. **Numerical Equivalence**: Generate MATLAB reference outputs, compare with rtol=1e-4
2. **Performance Benchmarks**: Track speedup vs MATLAB baseline
3. **End-to-End Tests**: Run on tissue-2D and cell-culture-3D datasets
4. **Backward Compatibility**: Both backends produce identical outputs

### Snakemake-Native Benchmarking (Cluster-Ready)
**Goal**: Collect timing/memory per rule on UGER in a consistent, reproducible way.

**Approach**:
- Add `benchmark:` targets to backend-comparable rules (same rule name, same inputs/outputs; only the invoked script differs).
- Store benchmark files under a predictable directory (e.g., `benchmarks/{backend}/{rule}/{wildcards}.tsv`) so MATLAB vs Python comparisons are scriptable.
- Keep resources comparable (`threads`, `mem_mb`, runtime) and record key runtime knobs (e.g., BLAS/FFT thread env vars) in logs for fair comparisons.

---

## Verification Commands

```bash
# Set up environment
cd src/python
uv sync

# Run tests
uv run pytest tests/ -v --cov=starfinder

# Run benchmarks
uv run python benchmarks/benchmark_registration.py

# Snakemake dry run with Python backend
snakemake -s workflow/Snakefile --configfile test/tissue_2D_test.yaml -n
```
