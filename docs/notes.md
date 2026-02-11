# STARfinder Development Notes

## Testing Datasets

Location: `/home/unix/jiahao/wanglab/Data/Processed/sample-dataset/`

### 1. cell-culture-3D
| Property | Value |
|----------|-------|
| FOVs | 70 (Position351-Position420) |
| Rounds | 6 sequencing (round1-6) + 1 organelle |
| Image size | 1496 × 1496 × 30 (3D) |
| Channels | 5 per FOV (ch00-ch04), ~67MB each |
| FOV pattern | `Position%03d` |
| Reference | round1, DAPI channel |
| Channel mapping | DAPI (488nm), ER (594nm), Flamingo (546nm) |
| Grid | 7×10, column-by-column, 10% overlap |

### 2. tissue-2D
| Property | Value |
|----------|-------|
| Tiles | 56 (tile_1 to tile_56) |
| Rounds | 4 sequencing (round1-4) + 1 protein |
| Image size | 3072 × 3072 × 30 → max projection (2D) |
| Channels | 5 per tile (ch00-ch04), ~283MB each |
| Tile pattern | `tile_%d` |
| Reference | round1, PI channel |
| Channel mapping | plaque (488nm), tau (594nm), PI (546nm), Gfap (647nm) |
| Grid | 7×8, column-by-column, 10% overlap |

### 3. LN (Covid Lymph Node)
| Property | Value |
|----------|-------|
| FOVs | 64 (Position001-Position064) |
| Rounds | 4 sequencing (round1-4) + 1 flamingo |
| Image size | 1496 × 1496 × 50 (3D) |
| Channels | 4 per sequencing FOV (ch00-ch03), ~112MB each |
| Flamingo channels | 3 (ch00-ch02), ~112MB each |
| FOV pattern | `Position%03d` |
| Codebook | 62 genes, 5-char barcodes |
| Source | `20240302_CovidLN_retake` |

## Milestones
1. [x] Modularization & Snakemake Upgrade
2. [] Rewrite the backend with Python & Improve Code Quality
  - [x] Phase 0: Directory restructure (src/matlab, src/python)
  - [x] Phase 1: I/O module (load/save TIFF with axis-aware handling)
  - [x] Phase 2: Registration module (DFT-based phase correlation)
  - [] Phase 3: Spot finding module
  - [] Phase 4: Decoding module
  - [] Phase 5: Segmentation integration
  - [] Phase 6: Dataset/FOV class wrapper
  - [] Adopt new data structure such as h5 and OME-Zarr, but also ensure backward compatibility
  - [] Adopt new 2D/3D image segmentation methods
3. [] Systematically benchmark the performance of the MATLAB backend and the new Python version

## Current Progress
Use this section to track development history.

### 2025-01-21: Snakemake Modularization & Upgrade Project Started
- [x] Created PR #9 to merge dev → main (commits 28-48)
- [x] Created development plan (`dev/current_plan.md`)
- [x] **Phase 1: Modularization** (in progress)
  - [x] Create `common.smk` with shared code
  - [x] Migrate registration rules → `registration.smk`
  - [x] Migrate spot-finding rules → `spot-finding.smk`
  - [x] Migrate segmentation rules → `segmentation.smk`
  - [x] Migrate stitching rules → `stitching.smk`
  - [x] Migrate reads-assignment rules → `reads-assignment.smk`
  - [x] Clean up main Snakefile (reduced from ~566 lines to ~32 lines)
  - [x] Test with dry run (completed successfully on 2026-01-22)
- [x] **Phase 2: Snakemake 9 Upgrade** (mostly completed on 2026-01-23)
  - [x] Create `environment-v9.yaml` (Python 3.11+, Snakemake 9.x)
  - [x] Update `profile/broad-uger/config.yaml` for v9 syntax
  - [x] Update `profile/broad-uger/broad-jobscript.sh` for v9 environment
  - [x] Fix MATLAB PATH inheritance issue in `run_matlab_scripts()`
  - [x] Test environment creation (completed successfully on 2026-01-22)
  - [x] Basic workflow execution test (completed successfully on 2026-01-23)
  - [ ] Full pipeline validation test (waiting for proper testing dataset)
- [x] **Phase 3: Code Quality Improvements** (mostly completed on 2026-01-24)

### 2025-01-22: Workflow Mode System & Config Simplification
- [x] **Researched modern data formats for biomedical imaging**
  - Compared HDF5, OME-Zarr, OME-TIFF for 3D analysis
  - Investigated SpatialData and scPortrait for spatial transcriptomics
  - Decided on hybrid approach: HDF5 (preprocessing) → SpatialData + scPortrait (outputs)
  - Documented strategy in Future Directions section
- [x] **Documented sample dataset structure**
  - Added cell-culture-3D and tissue-2D specifications
  - Included FOV counts, image dimensions, channel mappings, grid layouts
- [x] **Implemented workflow mode system**
  - Added `workflow_mode` config option: 'free', 'direct', 'subtile', 'deep'
  - Created `WORKFLOW_PRESETS` with predefined rule combinations
  - Implemented `is_rule_enabled()` to check rule activation based on mode
  - Updated `get_overall_output()` to use new system
- [x] **Implemented dynamic ruleorder**
  - Fixed rule priority issues (rsf_single_fov vs stitch_subtile conflicts)
  - Differentiated subtile and deep mode priorities
  - Subtile: lrsf_single_fov_subtile > deep_* rules
  - Deep: deep_* rules > subtile rules
  - Fixed N_SUBTILE calculation to check only gr_single_fov_subtile and deep_create_subtile
- [x] **Config file simplification**
  - Added `get_rule_config()` helper with default fallback
  - Added `DEFAULT_RESOURCES` (mem_mb: 8000, runtime: 30)
  - Updated all rule files to use safe config access
  - Users can now omit unused rule sections from config files
- [x] **Testing** (completed successfully on 2026-01-22)
  - Fixing indentation errors in common.smk
  - Need to complete dry run validation

**Files Modified:**
- `workflow/rules/common.smk` - Added workflow mode logic, config helpers
- `workflow/Snakefile` - Implemented dynamic ruleorder
- `workflow/rules/registration.smk` - Updated to use get_rule_config()
- `workflow/rules/spot-finding.smk` - Updated to use get_rule_config()
- `workflow/rules/segmentation.smk` - Updated to use get_rule_config()
- `workflow/rules/stitching.smk` - Updated to use get_rule_config()
- `workflow/rules/reads-assignment.smk` - Updated to use get_rule_config()

### 2026-01-23: Snakemake v9 Upgrade Completed

- [x] **Updated UGER cluster profile for Snakemake v9**
  - Migrated from `--cluster` to executor plugin system
  - Added `executor: cluster-generic` configuration
  - Updated submit/status command syntax: `cluster-generic-submit-cmd`, `cluster-generic-status-cmd`
  - Fixed script paths to be relative to repository root (`profile/broad-uger/...`)
  - Added `software-deployment-method: conda` (replaces `--use-conda`)
  - Changed `restart-times` → `retries` for v9 compatibility

- [x] **Updated job execution environment**
  - Modified `profile/broad-uger/broad-jobscript.sh` to use `starfinder-v9` conda environment
  - Reordered environment loading: activate conda first, then load MATLAB
  - Ensures MATLAB path takes precedence in PATH

- [x] **Fixed MATLAB subprocess execution issue**
  - Root cause: Python 3.12's stricter subprocess environment isolation
  - Subprocess spawns fresh bash shell without jobscript's PATH modifications
  - Solution: Modified `run_matlab_scripts()` in `workflow/rules/common.smk`
  - Now sources MATLAB environment within each subprocess call
  - Command: `source /broad/software/scripts/useuse && use Matlab && matlab ...`
  - More robust than relying on environment inheritance

- [x] **Enabled per-rule conda environments**
  - Added `software-deployment-method: conda` to profile config
  - Allows rules with `conda:` directives to use dedicated environments
  - Example: `stardist_segmentation` rule now activates its own environment

- [x] **Successfully tested basic workflow execution**
  - Dry run validation passed
  - Job submission to UGER cluster working
  - MATLAB execution functional in v9 environment
  - Conda environment activation working for Python rules

**Technical Insights:**
- Python 3.12 has stricter subprocess PATH inheritance compared to earlier versions
- Snakemake v9 requires explicit `software-deployment-method` in profile (not command-line flag)
- Running from repository root (not `workflow/`) is now the standard practice
- Executor plugin system provides better separation of concerns than old cluster flags

**Files Modified:**
- `profile/broad-uger/config.yaml` - v9 executor syntax, conda support
- `profile/broad-uger/broad-jobscript.sh` - Environment order, v9 conda env
- `workflow/rules/common.smk` - MATLAB subprocess PATH fix
- `dev/current_plan.md` - Updated Phase 2 status to "mostly completed"

**Next Steps:**
- Full pipeline validation test on complete dataset
- Update README/documentation with v9 usage instructions

### 2026-01-24: Config Schema Validation Implemented

- [x] **Created JSON Schema for config validation** (`workflow/schemas/config.schema.yaml`)
  - ~300 lines comprehensive schema
  - Validates all 15 required top-level keys
  - Enforces `workflow_mode` enum: 'free', 'direct', 'subtile', 'deep'
  - Conditional validation for subsetting options
  - Reusable `$defs` for resources, parameters, and rule-specific configs
  - Type validation for all fields (integers, strings, booleans, arrays, objects)

- [x] **Added Snakemake validation directive**
  - Modified `workflow/Snakefile` to import `validate` from `snakemake.utils`
  - Added `validate(config, schema="schemas/config.schema.yaml")` before includes
  - Catches config errors at pipeline startup rather than during rule execution

- [x] **Added custom workflow mode validation**
  - Created `validate_workflow_mode_dependencies()` function in `common.smk`
  - Validates that preset modes have required rule configurations defined
  - Raises clear error messages for missing rule sections
  - Called at module load time for early error detection

- [x] **Created minimal config template** (`test/minimal_config.yaml`)
  - Contains all required fields with placeholder values
  - Commented optional fields for easy enabling
  - Serves as starting point for new dataset configurations
  - Documents each field's purpose

**Schema Validation Features:**
- Required keys: `config_path`, `starfinder_path`, `root_input_path`, `root_output_path`, `dataset_id`, `sample_id`, `output_id`, `fov_id_pattern`, `n_fovs`, `n_rounds`, `ref_round`, `rotate_angle`, `img_col`, `img_row`, `rules`
- Conditional requirements: `subset_range: true` requires `subset_start`/`subset_end`
- Each rule requires `run: boolean` flag
- Resource defaults: mem_mb=8000, runtime=30

**Files Created/Modified:**
- `workflow/schemas/config.schema.yaml` - New file (~300 lines)
- `workflow/Snakefile` - Added validate() directive (+7 lines)
- `workflow/rules/common.smk` - Added validate_workflow_mode_dependencies() (+35 lines)
- `test/minimal_config.yaml` - New file, minimal config template
- `dev/current_plan.md` - Updated Phase 3 status
- `dev/notes.md` - Added this entry

**Next Steps:**
- Full validation test once Snakemake v9 conda environment is available
- Consider adding more specific parameter validation per rule type

### 2026-01-29: Python Package Setup & Synthetic Test Dataset

- [x] **Reviewed and fixed design documents**
  - Fixed typos in `main_python_object_design.md` (registeration→registration, filteration→filtration)
  - Changed codebook caching from mutable field to `@lru_cache(maxsize=4)`
  - Updated `split_index` type from `list[int]` to `tuple[int, ...]` for hashability
  - Added `LayerState` invariants, `to_register` property, and `validate()` method
  - Clarified SimpleITK as optional dependency for local registration only
  - Added error handling and benchmark test sections to `test_design.md`
  - Updated `plan_milestone_2.md` with registration library decisions

- [x] **Initialized Python package with uv**
  - Created `src/python/` directory structure
  - Set up `pyproject.toml` with uv-compatible configuration
  - Dependencies: numpy, scipy, scikit-image, tifffile, pandas, h5py
  - Optional: SimpleITK (local-registration), spatialdata
  - Dev: pytest, pytest-cov, ruff

- [x] **Implemented synthetic dataset generator** (`starfinder.testing.synthetic`)
  - Two-base color-space encoding matching MATLAB implementation
  - Generates 3D TIFF stacks with known spot positions
  - Includes inter-round shifts for registration testing
  - Presets: `mini` (1 FOV, 20 spots) and `standard` (4 FOVs, 400 spots)
  - CLI: `python -m starfinder.testing --preset mini --output <path>`

- [x] **Created test codebook with 8 genes**
  - All barcodes start and end with 'C'
  - Verified encoding: barcode reversed first, then two-base encoded
  - Example: CACGC → CGCAC → 4422 (ch03, ch03, ch01, ch01)

- [x] **Generated and committed synthetic fixtures**
  - `tests/fixtures/synthetic/mini/` - 1 FOV, 256×256×5, ~2.5MB
  - `tests/fixtures/synthetic/standard/` - 4 FOVs, 512×512×10, ~40MB
  - Each includes: FOV directories, codebook.csv, ground_truth.json

- [x] **Set up pytest infrastructure**
  - Created `src/python/tests/conftest.py` with session-scoped fixtures
  - Fixtures: `mini_dataset`, `standard_dataset`, `mini_ground_truth`, `mini_codebook`
  - 16 tests passing: encoding validation + fixture verification

**Files Created:**
- `src/python/pyproject.toml` - Package configuration
- `src/python/README.md` - Package documentation
- `src/python/starfinder/__init__.py` - Package root
- `src/python/starfinder/testing/__init__.py` - Testing module exports
- `src/python/starfinder/testing/synthetic.py` - Generator implementation
- `src/python/starfinder/testing/__main__.py` - CLI entry point
- `src/python/tests/conftest.py` - pytest fixtures
- `src/python/tests/test_synthetic.py` - Fixture verification tests
- `src/python/tests/test_encoding.py` - Two-base encoding tests
- `tests/fixtures/synthetic/mini/` - Mini test dataset
- `tests/fixtures/synthetic/standard/` - Standard test dataset
- `docs/plans/2026-01-29-synthetic-dataset-design.md` - Design document

**Commands:**
```bash
cd src/python
uv sync                           # Install dependencies
uv run pytest tests/ -v           # Run tests (16 passed)
uv run python -m starfinder.testing --preset mini --output ../../tests/fixtures/synthetic/mini
```

**Next Steps:**
- Implement Phase 1: I/O module (`starfinder.io.tiff`)
- Implement Phase 2: Registration module (`starfinder.registration`)
- Create numerical equivalence tests against MATLAB outputs

### 2026-01-30: Python I/O Module & Directory Restructure

- [x] **Phase 0: Directory Restructure**
  - Moved MATLAB code from `code-base/src/` to `src/matlab/`
  - Moved MATLAB addons from `code-base/matlab-addon/` to `src/matlab-addon/`
  - Updated 7 workflow scripts to use new paths (`workflow/scripts/*.m`)
  - Consistent `src/` directory structure for all source code

- [x] **Phase 1: I/O Module Implementation** (`starfinder.io`)
  - Implemented `load_multipage_tiff()` - Load multi-page TIFF with auto-detection
  - Implemented `load_image_stacks()` - Load multiple channel TIFFs as (Z, Y, X, C) array
  - Implemented `save_stack()` - Save 3D/4D arrays with optional compression
  - Auto-detects OME-TIFF and ImageJ hyperstacks for correct dimension handling
  - Uses bioio for metadata-aware loading, tifffile for plain TIFFs
  - 15 tests passing

- [x] **Fixed synthetic data Z-axis metadata**
  - Issue: ImageJ TIFFs saved without explicit axis metadata
  - Caused bioio to interpret Z=5 as C=5 (channels instead of slices)
  - Fix: Added `metadata={"axes": "ZYX"}` to `tifffile.imwrite()` call
  - Regenerated mini and standard synthetic datasets

- [x] **Development environment setup**
  - Added bioio and bioio-tifffile dependencies to pyproject.toml
  - Created `.vscode/settings.json` for Cursor/VS Code Python interpreter
  - Registered Jupyter kernel for interactive notebook testing
  - Created `tests/test_io_interactive.ipynb` for manual testing

**Files Created:**
- `src/python/starfinder/io/__init__.py` - Package exports
- `src/python/starfinder/io/tiff.py` - TIFF I/O implementation
- `src/python/test/test_io.py` - 15 unit tests
- `tests/test_io_interactive.ipynb` - Interactive testing notebook
- `.vscode/settings.json` - VS Code/Cursor settings

**Files Modified:**
- `src/python/pyproject.toml` - Added bioio dependencies
- `src/python/starfinder/testdata/synthetic.py` - Fixed Z-axis metadata
- `workflow/scripts/*.m` (7 files) - Updated MATLAB paths

**Dependencies Added:**
```toml
bioio>=1.0
bioio-tifffile>=1.0
# Optional: bioio-ome-tiff>=1.0
```

**Next Steps:**
- Phase 2: Registration module (`starfinder.registration`)
- Phase 3: Spot finding module (`starfinder.spots`)

### 2026-01-30: Phase 2 - Registration Module Implementation

- [x] **DFT-based phase correlation** (`starfinder.registration.phase_correlation`)
  - `phase_correlate(fixed, moving)` → `(dz, dy, dx)` shift tuple
  - `apply_shift(volume, shift)` → shifted volume with edge zeroing
  - `register_volume(images, ref, mov)` → multi-channel registration
  - Uses NumPy/SciPy FFT for cross-correlation in frequency domain
  - Handles wrap-around for signed shift conversion

- [x] **scikit-image backend for comparison** (`starfinder.registration._skimage_backend`)
  - `phase_correlate_skimage()` wrapper around `skimage.registration.phase_cross_correlation`
  - Sign convention normalized to match custom implementation

- [x] **Benchmark utilities** (`starfinder.registration.benchmark`)
  - `BenchmarkResult` dataclass: method, size, time, memory, shift_error
  - `run_benchmark()` - Compare methods across size presets
  - `print_benchmark_table()` - Formatted output
  - Size presets: tiny (128³), small (256³), medium (512³), large (1024³), xlarge (1496³), tissue (3072³)
  - Metrics: execution time, peak memory (tracemalloc), L2 shift error

- [x] **Benchmark results (NumPy vs scikit-image)**
  - NumPy ~25-30% faster across all sizes
  - NumPy ~20% less memory usage
  - Both produce identical shift results

- [x] **Test suite** (`test/test_registration.py`) - 5 tests
  - `test_zero_shift` - Identical images return (0, 0, 0)
  - `test_known_shift` - Recovers integer shift applied via np.roll
  - `test_roundtrip` - shift → apply → inverse preserves data
  - `test_registers_multichannel` - Multi-channel (Z, Y, X, C) registration
  - `test_backends_match` - NumPy vs scikit-image parity

- [x] **Synthetic test data helper** (`starfinder.testdata.create_test_volume`)
  - Creates 3D volumes with Gaussian spots for benchmarking
  - Configurable shape, n_spots, intensity, noise, seed

**Files Created:**
- `src/python/starfinder/registration/__init__.py` - Module exports
- `src/python/starfinder/registration/phase_correlation.py` - Core algorithm
- `src/python/starfinder/registration/_skimage_backend.py` - scikit-image wrapper
- `src/python/starfinder/registration/benchmark.py` - Benchmark utilities
- `src/python/test/test_registration.py` - Unit tests
- `docs/plans/2026-01-30-registration-module-design.md` - Design document
- `docs/plans/2026-01-30-registration-module-implementation.md` - Implementation plan

**Files Modified:**
- `src/python/starfinder/__init__.py` - Export registration module
- `src/python/starfinder/testdata/synthetic.py` - Added create_test_volume()
- `src/python/starfinder/testdata/__init__.py` - Export create_test_volume

**MATLAB Function Mapping:**
| MATLAB | Python |
|--------|--------|
| `DFTRegister3D(fixed, moving)` | `phase_correlate(fixed, moving)` |
| `DFTApply3D(volume, params)` | `apply_shift(volume, shift)` |
| `RegisterImagesGlobal(images, ref, mov)` | `register_volume(images, ref, mov)` |

**Next Steps:**
- Phase 3: Spot finding module (`starfinder.spots`)
- Phase 4: Decoding module (`starfinder.decoding`)

### 2026-01-31: QC Session & Benchmark Module Refactor

- [x] **Refactored benchmark module to standalone package** (`starfinder.benchmark`)
  - Moved from `starfinder.registration.benchmark` to standalone module
  - `BenchmarkResult` dataclass: method, operation, size, time_seconds, memory_mb, metrics
  - `measure(fn)` - Returns (result, time_seconds, memory_mb) using tracemalloc
  - `@benchmark` decorator - Wraps functions to return BenchmarkResult
  - `run_comparison()` - Compare multiple methods on same inputs
  - `BenchmarkSuite` - Collects results with `add()`, `summary()`, `filter()` methods
  - `print_table()`, `save_csv()`, `save_json()` - Reporting utilities
  - `SIZE_PRESETS` - Standard volume sizes (tiny, small, medium, large, xlarge, tissue)
  - 15 tests in `test/test_benchmark.py`

- [x] **Created QC notebooks** (`tests/qc_*.ipynb`)
  - `qc_benchmark.ipynb` - Benchmark framework validation
  - `qc_io.ipynb` - I/O module validation (load/save roundtrip, dtype checks)
  - `qc_synthetic.ipynb` - Synthetic data generator validation (spots overlay, encoding)
  - `qc_registration.ipynb` - Registration module validation (shift recovery, multi-channel)
  - Each notebook includes napari examples (wrapped in try/except for ImportError)

- [x] **Added napari as optional visualization dependency**
  - `pyproject.toml`: `visualization = ["napari>=0.4"]`

- [x] **Removed old interactive notebook**
  - Deleted `tests/test_io_interactive.ipynb` (replaced by `qc_io.ipynb`)

**Files Created:**
- `src/python/starfinder/benchmark/__init__.py` - Module exports
- `src/python/starfinder/benchmark/core.py` - BenchmarkResult, measure, @benchmark
- `src/python/starfinder/benchmark/runner.py` - run_comparison, BenchmarkSuite
- `src/python/starfinder/benchmark/report.py` - print_table, save_csv, save_json
- `src/python/starfinder/benchmark/presets.py` - SIZE_PRESETS, get_size_preset
- `src/python/test/test_benchmark.py` - 15 unit tests
- `tests/qc_benchmark.ipynb` - Benchmark QC notebook
- `tests/qc_io.ipynb` - I/O QC notebook
- `tests/qc_synthetic.ipynb` - Synthetic data QC notebook
- `tests/qc_registration.ipynb` - Registration QC notebook
- `docs/plans/2026-01-31-qc-session-design.md` - QC session design
- `docs/plans/2026-01-31-qc-session-implementation.md` - Implementation plan

**Files Modified:**
- `src/python/starfinder/registration/benchmark.py` - Migrated to use new framework
- `src/python/pyproject.toml` - Added visualization optional dependency

**Test Results:** 51 tests passing (15 benchmark + 11 encoding + 15 I/O + 5 registration + 5 synthetic)

**Next Steps:**
- Run QC notebooks interactively with napari for visual validation
- Promote stable QC checks to automated pytest tests
- Proceed to Phase 3: Spot finding module

### 2026-02-02: Registration Bug Fixes & QC Notebook Improvements

- [x] **Fixed benchmark shift range bug** (`starfinder.registration.benchmark`)
  - Issue: Fixed shift range (-5 to +5 for Z, -10 to +10 for YX) caused failures on small volumes
  - For tiny volumes (5 Z-slices), a Z-shift of ±5 leaves no overlap for phase correlation
  - Fix: Shift ranges now proportional to volume size (±25% of each dimension)
  - `max_z_shift = max(1, size[0] // 4)` ensures realistic test scenarios

- [x] **Fixed critical registration correction bug** (`starfinder.registration.phase_correlation`)
  - Issue: `register_volume()` applied detected shift instead of correcting it (doubled drift!)
  - `phase_correlate()` returns detected shift (how much moving differs from fixed)
  - To align, must apply the **negative** of detected shift
  - Fix: Added `correction = tuple(-s for s in shifts)` before `apply_shift()`
  - Verified with correlation test: interior regions now correlate at 0.9994

- [x] **Replaced napari with matplotlib in QC notebooks** (SSH compatibility)
  - napari requires display server, unusable over SSH
  - Implemented green/magenta composite visualization:
    - Green channel = fixed/reference image
    - Magenta (R+B) = moving/registered image
    - White/gray regions = good alignment
  - Added `make_composite()` helper function

- [x] **Changed visualization to maximum intensity projection (MIP)**
  - Previously used middle Z-slice (could miss misalignments on other slices)
  - MIP captures all spots across entire Z-stack in single 2D image
  - Better for sparse fluorescent data like STARmap spots

- [x] **Improved synthetic dataset registration test** (qc_registration.ipynb Section 6)
  - Changed from single channel (ch00) to max projection across all 4 channels
  - `ref_mip = np.max(ref_stack, axis=-1)` combines channel signals
  - More robust shift detection when individual channels have sparse signals
  - Added before/after visualization grid for rounds 2, 3, 4

**Files Modified:**
- `src/python/starfinder/registration/benchmark.py` - Size-proportional shift ranges
- `src/python/starfinder/registration/phase_correlation.py` - Fixed correction sign in `register_volume()`
- `tests/qc_registration.ipynb` - Matplotlib visualization, MIP, multi-channel reference

**Key Lessons:**
- Always test the actual outcome (alignment quality), not just intermediate values (detected shifts)
- Phase correlation returns "displacement detected", not "correction to apply"
- Shift ranges in benchmarks must be proportional to volume dimensions

**Test Results:** All 51 tests passing

**Next Steps:**
- Phase 3: Spot finding module (`starfinder.spots`)
- Add alignment quality assertions to registration tests (not just shift detection)

### 2026-02-02: Demons Registration Module Implemented

- [x] **Implemented non-rigid registration module** (`starfinder.registration.demons`)
  - `demons_register(fixed, moving)` → displacement field using symmetric forces demons
  - `apply_deformation(volume, field)` → apply displacement field to warp volume
  - `register_volume_local(images, ref, mov)` → multi-channel convenience wrapper
  - SimpleITK as optional dependency (lazy import with helpful error)
  - Multi-resolution pyramid matching MATLAB's `imregdemons` behavior
  - 4 tests in `test/test_demons.py`

- [x] **Added QC notebook section** (Section 7 of `qc_registration.ipynb`)
  - Synthetic local deformation generator
  - Demonstration: global registration fails on local deformation
  - Demonstration: local registration succeeds
  - Displacement field visualization

**Files Created:**
- `src/python/starfinder/registration/demons.py` - Core implementation
- `src/python/test/test_demons.py` - Unit tests

**Files Modified:**
- `src/python/starfinder/registration/__init__.py` - Added exports
- `tests/qc_registration.ipynb` - Added Section 7
- `docs/notes.md` - This entry

**MATLAB Function Mapping:**
| MATLAB | Python |
|--------|--------|
| `RegisterImagesLocal(images, ref, mov, iter, afs)` | `register_volume_local(images, ref, mov, iterations, smoothing_sigma)` |
| `imregdemons(mov, ref, ...)` | `demons_register(fixed, moving, ...)` |
| `imwarp(img, field)` | `apply_deformation(volume, field)` |

### 2026-02-02: Demons Registration Bug Fixes & Quality Metrics Module

- [x] **Fixed critical axis ordering bug** (`starfinder.registration.demons`)
  - Issue: ~90° angular error between estimated and true displacement fields
  - Root cause: SimpleITK returns displacement vectors in (dx, dy, dz) order, but NumPy uses (dz, dy, dx)
  - Fix: Added `field_array = field_array[..., ::-1]` to reverse vector components
  - Also fixed in `apply_deformation()` when converting back to SimpleITK

- [x] **Discovered multi-resolution pyramid degradation** for sparse images
  - Multi-level pyramids ([100, 50, 25]) can *degrade* quality for sparse fluorescence data
  - Upsampling artifacts and intensity interpolation blur sparse spots
  - Single-level registration ([50]) achieved 36% improvement vs multi-level giving -2%
  - Changed default `iterations` from `[100, 50, 25]` to `[50]`

- [x] **Optimized demons defaults** for sparse fluorescence images
  - `method="diffeomorphic"` - more stable, topology-preserving
  - `smoothing_sigma=0.5` - lower value preserves spot sharpness
  - `iterations=[50]` - single-level, no pyramid
  - Added shrink factor limiting to prevent Z=1 NaN issues

- [x] **Implemented spot-based quality metrics module** (`starfinder.registration.metrics`)
  - `normalized_cross_correlation(img1, img2)` → NCC value [-1, 1]
  - `structural_similarity(img1, img2)` → SSIM value [-1, 1] (perceptual quality)
  - `spot_colocalization(ref, img)` → IoU and Dice of bright spots
  - `spot_matching_accuracy(ref_spots, mov_spots)` → match rate, mean distance
  - `detect_spots(volume)` → centroid coordinates via connected components
  - `registration_quality_report(ref, before, after)` → comprehensive metrics dict
  - `print_quality_report(report)` → formatted output with barcode decoding projection

- [x] **Key insight: MAE is misleading for sparse images**
  - Background pixels (99% of image) dominate MAE calculation
  - Spot IoU showed 323% improvement vs MAE showing only 44%
  - Spot matching accuracy is most critical for barcode decoding
  - 90% match/round × 4 rounds = 65% decoded; 99% match/round × 4 rounds = 96% decoded

- [x] **Reorganized qc_registration.ipynb**
  - Section 7.3: Local registration with visualization (removed EPE details)
  - Section 7.4: Quality metrics (NCC, SSIM, Spot IoU, Match Rate)
  - Section 7.5: Parameter sensitivity (methods, pyramid, smoothing comparison)
  - Removed: EPE/angular error sections (misleading for forward/inverse comparison)
  - Removed: Spatial error analysis, improved registration approaches (redundant)

**Files Created:**
- `src/python/starfinder/registration/metrics.py` - Quality metrics module

**Files Modified:**
- `src/python/starfinder/registration/demons.py` - Axis ordering fix, optimized defaults
- `src/python/starfinder/registration/__init__.py` - Added metrics exports
- `tests/qc_registration.ipynb` - Reorganized sections, added spot-based metrics

**Key Lessons:**
- SimpleITK uses (dx, dy, dz) vector ordering, NumPy uses (dz, dy, dx)
- Multi-resolution pyramids hurt sparse fluorescence images
- Spot-based metrics (IoU, match rate) are more meaningful than MAE for registration QC
- SSIM captures perceptual quality that NCC alone may miss

### 2026-02-04: Registration Benchmark Data Generation (Task 1 Complete)

- [x] **Created benchmark data generation module** (`starfinder.benchmark.data`)
  - `create_benchmark_volume()` - Synthetic 3D volume with Gaussian spots
  - `apply_global_shift()` - Zero-padded shifts (no wrap-around)
  - `create_deformation_field()` - Polynomial, Gaussian bump, multi-point deformations
  - `apply_deformation_field()` - Scipy map_coordinates-based warping
  - `generate_inspection_image()` - Green-magenta MIP overlays (G=ref, M=mov)
  - `generate_synthetic_benchmark()` - Full preset generation pipeline
  - `extract_real_benchmark_data()` - Round1/round2 MIP extraction from real datasets

- [x] **Extended benchmark presets** (`starfinder.benchmark.presets`)
  - Added: `thick_medium` (100, 1024, 1024), plus existing tiny through tissue
  - `SPOT_COUNTS`: Density scaling (~50 spots per 10⁶ voxels)
  - `SHIFT_RANGES`: Proportional to volume size (≤25% of each dimension)
  - `DEFORMATION_CONFIGS`: Percentage-based with pixel caps

- [x] **Generated benchmark datasets** (Task 1 of benchmark plan)
  - **Synthetic:** 7 presets × 6 pairs each = 42 ref/mov pairs (~31GB)
    - tiny (8×128×128), small (16×256×256), medium (32×512×512)
    - large (30×1024×1024), xlarge (30×1496×1496), tissue (30×3072×3072)
    - thick_medium (100×1024×1024)
  - **Real:** 3 datasets (cell_culture_3D, tissue_2D, LN) (~0.9GB)
  - **Location:** `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/`

- [x] **Fixed multiple data generation issues**
  - Z-axis shifts always 0 → Fixed with preset-specific seeds + exclude 0 from options
  - Wrap-around in shifted images → Fixed with zero-padding instead of np.roll
  - Large blank margins around spots → Fixed with fixed 5px margin (not percentage)
  - Excessive deformation on large images → Fixed with percentage scaling + pixel caps (15/30px)

- [x] **Removed thick_large preset**
  - (200×2722×2722) required ~112GB RAM for deformation fields
  - OOM killed during generation; decided to skip entirely

- [x] **Updated benchmark plan** (`docs/plans/2026-02-03-registration-benchmark-plan.md`)
  - Task 1 marked complete
  - Visual inspection checkpoint passed
  - Added Section 2.3: Output artifacts per benchmark run (registered images + inspection.png)
  - Added checkpoint after Task 2 for registration results inspection

**Files Created:**
- `src/python/starfinder/benchmark/data.py` - Data generation module (~800 lines)

**Files Modified:**
- `src/python/starfinder/benchmark/presets.py` - Added thick presets, DEFORMATION_CONFIGS
- `src/python/starfinder/benchmark/__init__.py` - Export new functions
- `docs/plans/2026-02-03-registration-benchmark-plan.md` - Updated plan

**Key Lessons:**
- `np.roll()` wraps around; use slicing with zero-fill for realistic shifted images
- Percentage-based spot margins create visible blank bands on large images
- Same random seed across presets → same shifts; use preset-specific seeds
- Deformation field memory: (Z, Y, X, 3) float32 = ~4× volume size × 3
- thick_large (1.5B voxels) exceeded memory limits; keep presets ≤300M voxels

**Next Steps:**
- Task 2: Performance benchmarking with registered image output
- Task 3: Reporting and visualization

### 2026-02-10: Benchmark Folder Reorganization

- [x] **Reorganized benchmark folder structure**
  - Renamed `registration_benchmark/` → `starfinder_benchmark/` (default output folder for all future benchmarks)
  - Created `data/` folder, moved `synthetic/` and `real/` into it
  - Moved all registration results under `results/registration/`
  - Deleted `overview.png` (panels too small, not readable)

- [x] **Renamed result folders for clarity**
  - `global/` → `global_python/` (Python-only global registration results)
  - `matlab_global/` → `global_matlab/` (MATLAB DFTRegister3D results)
  - `tuning/` → `local_tuning/` (demons parameter grid search)
  - `matlab/` → `local_matlab/` (MATLAB imregdemons results)
  - Created `local_python/` (placeholder for future Python local registration benchmarks)
  - Created `scripts/` folder under `results/registration/` for MATLAB/Python comparison scripts

**New structure:**
```
starfinder_benchmark/
├── data/
│   ├── synthetic/          # 7 presets (tiny → tissue), 31 GB
│   └── real/               # 3 datasets (cell_culture_3D, tissue_2D, LN), 886 MB
└── results/
    └── registration/
        ├── global_python/       # Python phase_correlate results (1.9 GB)
        ├── global_matlab/       # MATLAB DFTRegister3D results (446 MB)
        ├── global_comparison/   # MATLAB vs Python head-to-head (1.8 GB)
        ├── local_tuning/        # Demons parameter grid search (107 MB)
        ├── local_matlab/        # MATLAB imregdemons results (14 GB)
        ├── local_python/        # (empty, future Python demons results)
        ├── local_comparison/    # MATLAB vs Python demons comparison (409 MB)
        ├── combined/            # Aggregated results JSON (8 KB)
        ├── figures/             # Publication-style figures (8.5 MB)
        └── scripts/             # MATLAB/Python comparison scripts (136 KB)
```

### 2026-02-10: Two-Phase Benchmark Workflow & Evaluate Module

- [x] **Designed and implemented two-phase benchmark architecture**
  - Phase 1 (Run): Backend-specific — run algorithm, record time + memory, save `registered_{backend}.tif` + `run_{backend}.json`
  - Phase 2 (Evaluate): Unified Python — load saved images, compute all metrics with identical code, save `metrics_{backend}.json` + `inspection_{backend}.png`
  - Ensures fair comparison: all backends evaluated by same metric computation code
  - Plan: `docs/plans/2026-02-10-two-phase-benchmark-plan.md`

- [x] **Created `starfinder.benchmark.evaluate` module** (`src/python/starfinder/benchmark/evaluate.py`)
  - `evaluate_registration(ref, mov_before, registered, skip_ssim=False)` → flat metrics dict
  - `generate_inspection(ref, mov, registered, metadata, output_path)` → 5-panel PNG
  - `evaluate_single(registered_path, data_dir)` → evaluate one registered image from disk
  - `evaluate_directory(result_dir, data_dir)` → batch-evaluate all results in a backend tree
  - CLI: `uv run python -m starfinder.benchmark.evaluate <result_dir> [--data-dir ...] [--force]`
  - Handles legacy JSON naming (`result_*.json` → `run_*.json`)
  - Supports both synthetic and real datasets via `_resolve_data_paths()`

- [x] **Refactored `runner.py` to delegate to evaluate module**
  - `_compute_quality_metrics()` → delegates to `evaluate.evaluate_registration()`
  - `generate_registration_inspection()` → delegates to `evaluate.generate_inspection()`
  - Updated `DEFAULT_BENCHMARK_DATA_DIR` to `starfinder_benchmark/data`
  - Fixed `results_dir` default: `self.data_dir.parent / "results"` (sibling of data/)

- [x] **Updated MATLAB benchmark scripts** (on network mount)
  - `benchmark_global_single.m` — output to `global_matlab/` tree, `run_matlab.json`, VmRSS memory measurement
  - `benchmark_local_single.m` — output to `local_matlab/` tree, VmRSS memory measurement
  - Both use `/proc/self/status` VmRSS delta for memory tracking (since MATLAB `memory()` is Windows-only)

- [x] **Updated Python benchmark scripts** (on network mount)
  - `benchmark_global_single.py` — output to `global_python/` tree, `run_python.json`, uses `measure()` for timing+memory
  - `benchmark_local_single.py` — same pattern for local registration

- [x] **MIP SSIM fallback for large volumes**
  - For volumes >100M voxels (e.g., tissue_2D at 283M), full 3D SSIM takes 20+ minutes
  - Instead of skipping SSIM entirely, computes SSIM on 2D MIP (maximum intensity projection along Z)
  - Output includes `"ssim_method": "mip"` or `"ssim_method": "3d"` to indicate which was used
  - MIP SSIM runs in seconds while still providing meaningful structural similarity

- [x] **Cross-validated metrics**
  - LN and cell_culture_3D metrics from new evaluator match old post-hoc `global_evaluation.json` exactly
  - NCC, match rate, spot IoU all consistent between old and new code paths

**Per-dataset output structure (all backends):**
```
{result_dir}/{dataset}/
  registered_{backend}.tif        # Phase 1: registered volume
  run_{backend}.json              # Phase 1: timing, memory, shifts, status
  metrics_{backend}.json          # Phase 2: all quality metrics + ssim_method
  inspection_{backend}.png        # Phase 2: green-magenta overlay
```

**Files Created:**
- `src/python/starfinder/benchmark/evaluate.py` — Phase 2 evaluator (~600 lines)
- `docs/plans/2026-02-10-two-phase-benchmark-plan.md` — Implementation plan

**Files Modified:**
- `src/python/starfinder/benchmark/runner.py` — Delegated metrics/inspection to evaluate.py
- `src/python/starfinder/benchmark/__init__.py` — Added evaluate exports
- Network mount scripts: `benchmark_{global,local}_single.{m,py}` — Updated output format + memory tracking

**Test Results:** 55 tests passing

**Next Steps:**
- Run Phase 1 on all missing synthetic presets in global_matlab/
- Run Phase 2 evaluator on all backend trees to generate unified metrics
- Task 3 of benchmark plan: Reporting and visualization

## Future Directions

### 1. Replace MATLAB with Python
The ultimate goal is to eliminate all MATLAB implementations and develop equivalent functionality in Python.
- The most challenging aspect will be re-implementing the 3D image registration algorithms. For example, a more efficient implementation with C++ or DL-based tools such as VoxelMorph.

### 2. Adopt Modern Data Format Strategy (Hybrid Approach)

**Workflow:**
```
Raw TIFF → HDF5 (preprocessing) → SpatialData + scPortrait (outputs)
```

**Stage 1: HDF5 for Preprocessing**
- Fast local I/O for iterative processing
- Consolidated multi-round images
- Registration transforms, spot coordinates, segmentation masks
- Compression: Blosc+LZ4 (speed) or Blosc+ZSTD (ratio)

**Stage 2: Dual Output Formats**

| SpatialData (.zarr) | scPortrait (.h5sc) |
|---------------------|---------------------|
| Full FOV images (OME-Zarr) | Single-cell image crops |
| Segmentation masks | Morphological features |
| Spot coordinates | Cell embeddings |
| Cell expression (AnnData) | Ready for DL/ML |
| Spatial graphs | |

| **Use cases** | **Use cases** |
|---------------|---------------|
| napari visualization | Cell type classification |
| squidpy spatial analysis | Representation learning |
| Cloud sharing/publication | Multimodal integration |

**Why not OME-TIFF?**
- Limited scalability for 3D data
- Poor chunking support
- Not cloud-native

**Key References:**
- [SpatialData (Nature Methods 2025)](https://www.nature.com/articles/s41592-024-02212-x)
- [scPortrait (MannLabs)](https://github.com/MannLabs/scPortrait)
- [OME-Zarr specification](https://ngff.openmicroscopy.org/latest/)

**Python Dependencies:**
```
# Preprocessing
h5py, hdf5plugin

# SpatialData output
spatialdata, spatialdata-io, squidpy

# scPortrait output
scportrait

# Visualization
napari, napari-spatialdata
```

### 3. Enable Cloud Compatibility
Make the pipeline compatible with cloud platforms (enabled by OME-Zarr/SpatialData adoption).

### 4. Adopt `uv` for Python Project Management
Replace conda with uv for faster, more reproducible Python dependency management.
- [x] Initialized `src/python/` with uv (2026-01-29)

### 5. Create simple test cases
- [x] Synthetic dataset generator implemented (2026-01-29)
- [x] Mini (1 FOV) and standard (4 FOVs) presets available
- [ ] Add real dataset subset for integration testing
