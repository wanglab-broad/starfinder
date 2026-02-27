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
  - [x] Phase 3: Spot finding & extraction (find_spots_3d, extract_from_location)
  - [x] Phase 4: Barcode processing (encode/decode, codebook, filter_reads)
  - [x] Phase 5: Preprocessing (min_max_normalize, histogram_match, morphological_reconstruction, tophat_filter, make_projection)
  - [x] Phase 6: Dataset/FOV class wrapper (STARMapDataset, FOV, fluent API)
  - [x] Phase 7: E2E validation tests + SNR-gated normalization + noise-floor spot finding threshold
  - [x] Phase 8: Real data E2E benchmark (tissue-2D, LN, cell-culture-3D — all 3 datasets, 2 FOVs each)
  - [x] Phase 9: Performance optimization (streaming pipeline, memory fixes — 50% RSS reduction, 2x speedup)
  - [x] Phase E: Spot-based TPS local registration (6x less memory than demons, competitive quality for small deformations)
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
  - **Location:** `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/registration/`

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

- [x] **Reorganized benchmark folder structure** (updated 2026-02-17)
  - `starfinder_benchmark/` is now a task-scoped organizer with `registration/`, `e2e/`, `spot_finding/` subdirectories
  - Registration data and results moved under `registration/data/` and `registration/results/`

**Current structure:**
```
starfinder_benchmark/
├── registration/
│   ├── data/
│   │   ├── synthetic/          # 7 presets (tiny → tissue), 31 GB
│   │   └── real/               # 3 datasets (cell_culture_3D, tissue_2D, LN), 886 MB
│   └── results/
│       ├── global_python/       # Python phase_correlate results
│       ├── global_matlab/       # MATLAB DFTRegister3D results
│       ├── global_comparison/   # MATLAB vs Python head-to-head
│       ├── local_tuning/        # Demons parameter grid search
│       ├── local_matlab/        # MATLAB imregdemons results
│       ├── local_python/        # Python demons results
│       ├── local_comparison/    # MATLAB vs Python demons comparison
│       ├── figures/             # Publication-style figures
│       └── scripts/             # MATLAB/Python comparison scripts
├── e2e/                         # End-to-end validation results
└── spot_finding/                # Spot finding benchmark results
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
  - Updated `DEFAULT_BENCHMARK_DATA_DIR` to `starfinder_benchmark/registration/data`
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

### 2026-02-10: MIP-based Spot Detection for Large Volumes

Phase 2 evaluation was bottlenecked by 3D spot detection on large datasets like tissue_2D (283M voxels). The existing `skip_ssim` fast path already computed SSIM on 2D MIP, but spot detection (`scipy.ndimage.label` + `center_of_mass`) and colocalization still ran on full 3D — 3 calls on ~283M voxels each.

- [x] **Renamed `skip_ssim` → `use_mip`** throughout the call chain
  - `evaluate_registration(use_mip=)`, `evaluate_single(use_mip=)`, `evaluate_directory(use_mip_above=)`
  - CLI: `--skip-ssim-above` → `--use-mip-above`
- [x] **Extended MIP fast path for spot metrics**: In `use_mip=True` branch, `spot_colocalization()` and `detect_spots()` now run on 2D MIP arrays (already computed for SSIM) instead of full 3D
- [x] **Added `spot_method` field** to metrics JSON output: `"mip"` or `"3d"` for provenance tracking (analogous to existing `ssim_method`)
- [x] **Re-evaluated global_python and global_matlab** with `--force` — all 10 datasets each, ~9 min total

**Files Modified:**
- `src/python/starfinder/benchmark/evaluate.py` — Core MIP fast path extension + rename
- `src/python/starfinder/benchmark/runner.py` — Updated call site

**Validation:**
- 55 tests passing
- tissue_2D correctly uses MIP path (`ssim_method: "mip"`, `spot_method: "mip"`)
- Small datasets correctly use full 3D path (`ssim_method: "3d"`, `spot_method: "3d"`)
- Python vs MATLAB metrics consistent (NCC after: 0.594 vs 0.602, match rate: 0.156 vs 0.155)

### 2026-02-11: Anti-Aliased Pyramid for Demons Registration

- [x] **Implemented MATLAB-matching anti-aliased pyramid** (`starfinder.registration.pyramid`)
  - `butterworth_3d(shape, cutoff, order)` — Separable 3D Butterworth low-pass filter matching MATLAB's `butterwth()`
  - `antialias_resize(volume, factor)` — Anti-aliased 3D resize matching MATLAB's `antialiasResize()`
  - `pad_for_pyramiding(volume, levels)` → `(padded, pad_widths)` — Replicate-border padding for clean 2x downsampling
  - `crop_padding(volume, pad_widths)` — Remove padding after registration
  - Key insight: MATLAB's `imregdemons` uses Butterworth-filtered downsampling internally, while SimpleITK's `Shrink` does naive subsampling (every Nth voxel). This destroys sparse spots at coarse levels.

- [x] **Added `pyramid_mode="antialias"` to `demons_register()`**
  - New parameter: `pyramid_mode` (`"sitk"` default, `"antialias"` for MATLAB-style)
  - When `antialias` + multi-level: pads volumes, downsamples with Butterworth filter, upsamples displacement fields between levels with magnitude scaling
  - Uses float64 precision matching MATLAB's `double()`
  - Single-level or `sitk` mode: existing behavior unchanged

- [x] **Added `matlab_compatible_config()` convenience function**
  - Returns `{iterations=[100,50,25], sigma=1.0, method="demons", pyramid_mode="antialias"}`
  - Usage: `field = demons_register(fixed, moving, **matlab_compatible_config())`

- [x] **Added tests** (`test/test_demons.py`)
  - `TestPyramidUtilities`: butterworth shape, resize roundtrip, pad/crop roundtrip, no-op padding
  - `TestAntialiasedDemonsRegister`: identity test, antialias-outperforms-sitk test
  - `TestMatlabCompatibleConfig`: config keys validation

- [x] **Plan**: `docs/plans/2026-02-11-optimize-python-demons-plan.md` — all 5 steps complete

**Files Created:**
- `src/python/starfinder/registration/pyramid.py` — Anti-aliased pyramid utilities (~150 lines)

**Files Modified:**
- `src/python/starfinder/registration/demons.py` — Added `_run_antialias_pyramid()`, `pyramid_mode` param, `matlab_compatible_config()`
- `src/python/starfinder/registration/__init__.py` — Added `matlab_compatible_config` export
- `src/python/test/test_demons.py` — Added pyramid + antialias tests

### 2026-02-11: Python vs MATLAB Local Registration Comparison Benchmark

- [x] **Ran head-to-head comparison** with matched settings across 3 datasets × 3 configs = 21 total runs
  - **Datasets**: large (31M vox, 5 deformation types), cell_culture_3D (67M vox), LN (112M vox)
  - **Configs**: `py_demons` (Thirion + antialias), `py_diffeo` (diffeomorphic + antialias), `matlab` (imregdemons)
  - All used identical iterations `[100,50,25]` and sigma/AFS=1.0 with 3-level pyramids

- [x] **Created benchmark scripts** (at `.../starfinder_benchmark/registration/results/scripts/`)
  - `benchmark_local_comparison_single.py` — Python worker (117 lines)
  - `benchmark_local_comparison_matlab.m` — MATLAB worker with matched AFS=1.0 (130 lines)
  - `run_local_comparison_v2.py` — Orchestrator with `/usr/bin/time -v` wrapping (309 lines)
  - `generate_local_comparison_v2.py` — Report generator merging quality + timing (203 lines)

- [x] **All 21 runs completed** (0 failures, 39 min total)
  - Phase 2 evaluation computed metrics + inspection images for all runs
  - Results at: `.../results/registration/local_comparison/`

- [x] **Key results**:

  | Metric | py_demons vs MATLAB | py_diffeo vs MATLAB |
  |--------|--------------------|--------------------|
  | Mean NCC delta | +0.042 | +0.115 |
  | Mean Match Rate delta | -0.035 | +0.022 |
  | Mean speedup | 1.60x | 1.12x |

  - **py_diffeo is the quality winner**: beats MATLAB on both NCC and Match Rate
  - **py_demons is the speed winner**: 1.6x faster with better NCC, slight Match Rate tradeoff
  - Anti-aliased pyramid closes the Python-MATLAB gap: on `gaussian_small`, Python NCC 0.939 vs MATLAB 0.853
  - Peak RSS comparable: ~1.0-1.1x MATLAB on synthetic, ~1.04-1.07x on real

- [x] **Plan**: `docs/plans/2026-02-11-local-comparison-benchmark-plan.md`

**Files Created** (on network mount):
- `benchmark_local_comparison_single.py`, `benchmark_local_comparison_matlab.m`
- `run_local_comparison_v2.py`, `generate_local_comparison_v2.py`

**Output artifacts:**
- `local_comparison/summary.csv` — Phase 2 quality metrics (21 rows)
- `local_comparison/comparison.csv` — Merged quality + timing comparison (7 pivot rows)
- `local_comparison/{dataset}/timing_*.json` — GNU time measurements
- `local_comparison/{dataset}/inspection_*.png` — Green-magenta overlays

### 2026-02-11: Phase 3 — Spot Finding & Extraction

- [x] **Implemented spot finding module** (`starfinder.spotfinding`)
  - `find_spots_3d(volume, method, ...)` — 3D spot detection with LoG filtering
  - Adaptive and global thresholding modes
  - Multi-channel support

- [x] **Implemented barcode extraction** (`starfinder.barcode.extraction`)
  - `extract_from_location(image, spots, voxel_size)` — Per-channel intensity extraction
  - L2-normalized, winner-take-all channel assignment → "1"-"4", "M", "N"
  - Voxel neighborhood with boundary clipping

### 2026-02-11: Phase 4 — Barcode Processing

- [x] **Created encoding module** (`starfinder.barcode.encoding`)
  - Moved `BASE_PAIR_TO_COLOR`, `COLOR_TO_CHANNEL` from `testdata/synthetic.py`
  - Added `COLOR_TO_BASE_PAIRS` reverse lookup
  - `encode_bases(sequence)` — Pure 2-base sliding window encoding (no reversal), matches MATLAB `EncodeBases.m`
  - `decode_color_seq(color_seq, start_base)` — Chain-tracking decoder, matches MATLAB `DecodeCS.m`
  - `testdata/synthetic.py` now imports from `barcode.encoding`

- [x] **Created codebook module** (`starfinder.barcode.codebook`)
  - `load_codebook(path, do_reverse=True, split_index=None)` → `(gene_to_seq, seq_to_gene)`
  - Reads CSV, optionally reverses barcodes, encodes to color-space
  - Optional split_index for multi-segment barcodes
  - Matches MATLAB `LoadCodebook.m`

- [x] **Created filtering module** (`starfinder.barcode.filtering`)
  - `filter_reads(spots, seq_to_gene, end_bases=None, start_base="C")` → `(good_spots, stats)`
  - Filters by codebook membership only (matching MATLAB behavior)
  - End-base validation is diagnostic stats, not a filter
  - Matches MATLAB `FilterReads.m`

- [x] **Updated package wiring**
  - `barcode/__init__.py` exports all encoding, codebook, extraction, filtering functions
  - `starfinder/__init__.py` exports `load_codebook`, `filter_reads` at top level
  - `test/test_encoding.py` updated to import from `barcode.encoding`

- [x] **Tests** — 19 new tests in `test/test_barcode.py`
  - Encoding: known sequences, output length, same-base pairs
  - Decoding: known decode, roundtrip, all codebook entries, single color
  - Codebook: load, gene/seq lookup, bidirectional consistency, no_reverse
  - Filtering: basic, invalid exclusion, stats keys, end-base diagnostic, empty input, gene column
  - End-to-end: ground truth pipeline (8 genes + 2 invalid → 8 filtered)

- [x] **Plan**: `docs/plans/2026-02-11-barcode-processing-plan.md`

**Files Created:**
- `src/python/starfinder/barcode/encoding.py` — Encoding/decoding (~95 lines)
- `src/python/starfinder/barcode/codebook.py` — Codebook loading (~65 lines)
- `src/python/starfinder/barcode/filtering.py` — Read filtering (~75 lines)
- `src/python/test/test_barcode.py` — 19 tests

**Files Modified:**
- `src/python/starfinder/barcode/__init__.py` — Added all new exports
- `src/python/starfinder/__init__.py` — Added `load_codebook`, `filter_reads`
- `src/python/starfinder/testdata/synthetic.py` — Imports encoding from `barcode.encoding`
- `src/python/test/test_encoding.py` — Updated imports

**MATLAB Function Mapping:**
| MATLAB | Python |
|--------|--------|
| `EncodeBases(seq)` | `encode_bases(sequence)` |
| `DecodeCS(color_seq, start_base)` | `decode_color_seq(color_seq, start_base)` |
| `LoadCodebook(path, split_index, do_reverse)` | `load_codebook(path, do_reverse, split_index)` |
| `FilterReads(obj, end_base)` | `filter_reads(spots, seq_to_gene, end_bases, start_base)` |

**Test Results:** 96 tests passing (19 barcode + 11 encoding + 8 extraction + ...)

**Next Steps:**
- Phase 5: Preprocessing module
- Phase 6: Dataset class & Snakemake integration

### 2026-02-11: Rerun local_python/ Benchmark with New Defaults

The `local_python/` benchmark folder contained 38 runs using the **old config** (symmetric, iter=25, sigma=0.5, single-level sitk pyramid). After updating `demons_register()` defaults to match MATLAB's `imregdemons`, these results were stale. Reran all 38 with the new config.

- [x] **Updated worker script** (`scripts/benchmark_local_single.py`)
  - `method`: `"symmetric"` → `"demons"` (Thirion)
  - `iterations`: `[25]` → `[100, 50, 25]` (3-level pyramid)
  - `sigma`: `0.5` → `1.0`
  - Added `pyramid_mode="antialias"` to `demons_register()` call
  - Updated metadata: `"config": "demons_iter100-50-25_s1.0_antialias"`

- [x] **Updated orchestrator timeout** (`scripts/run_local_python.py`)
  - `TIMEOUT`: `600` → `900` (tissue_2D peaked at 526s, would have timed out at 600s)

- [x] **Backed up old results**
  - `local_python/` → `local_python_old_symmetric/` (preserved for comparison)

- [x] **Ran all 38 benchmarks** — 38 success, 0 timeout, 0 error, 5302s (88 min) total
  - Timings matched estimates: tiny ~1s, medium ~16s, large ~58s, tissue ~510s, tissue_2D ~505s

- [x] **Phase 2 evaluation** — 38 evaluated, 0 skipped
  - Generated `metrics_*.json`, `inspection_*.png`, `summary.csv`

- [x] **Comparison results** (new config vs old):

  | Metric | Old (symmetric, iter25, σ=0.5) | New (demons, [100,50,25], σ=1.0, antialias) | Delta |
  |--------|-------------------------------|----------------------------------------------|-------|
  | Mean NCC | 0.520 | 0.624 | **+0.103** (35/38 improved) |
  | Mean Match Rate | 0.590 | 0.642 | **+0.052** (17 up, 17 down, 4 same) |
  | Mean Runtime | 100.8s | 133.0s | **1.32x** slower |

  - NCC improved on 92% of datasets; largest gains on real data (LN +0.376, cell_culture_3D +0.161)
  - Match Rate gains concentrated in polynomial deformations (+0.20) and real data (+0.13)
  - Runtime overhead modest for production-size datasets (1.2-1.3x), higher for tiny volumes (1.6-1.8x)
  - Only regression: tissue_2D NCC (-0.165) — low-contrast 2D tissue section

**Files Modified** (on network mount):
- `.../scripts/benchmark_local_single.py` — Config update + antialias pyramid
- `.../scripts/run_local_python.py` — Timeout 600→900

**Output:**
- `local_python/summary.csv` — 38 rows with new config metrics
- `local_python_old_symmetric/` — preserved old results for comparison

### 2026-02-12: Phase 5 — Preprocessing & Utils

Ported MATLAB preprocessing functions to Python. All operate on single `(Z, Y, X)` or `(Z, Y, X, C)` volumes; multi-round coordination deferred to Phase 6 (Dataset class).

**New modules:**
- `starfinder.preprocessing` — 4 image enhancement functions:
  - `min_max_normalize(volume)` — per-channel [min, max] → [0, 255] (ports `MinMaxNorm.m`)
  - `histogram_match(volume, reference)` — CDF-based histogram matching (ports `STARMapDataset.HistEqualize`)
  - `morphological_reconstruction(volume, radius)` — background removal via opening-by-reconstruction (ports `MorphologicalReconstruction.m`)
  - `tophat_filter(volume, radius)` — white tophat per Z-slice (ports `STARMapDataset.Tophat`)
- `starfinder.utils` — `make_projection(volume, method)` (ports `MakeProjections.m`)

**Tests:** 18 new tests (4 utils + 14 preprocessing), full suite 114 passing.

### 2026-02-12: Phase 6 — Dataset & FOV Orchestration Layer

Implemented the `starfinder.dataset` subpackage — the orchestration layer that wraps all Phase 1-5 modules into a stateful, fluent pipeline API.

**New subpackage: `starfinder.dataset`** (6 files):
- `types.py` — Type aliases (`Shift3D`, `ImageArray`, `ChannelOrder`) and dataclasses (`LayerState`, `Codebook`, `CropWindow`, `SubtileConfig`)
- `logging.py` — `log_step` decorator for FOV processing steps (timing + error logging)
- `paths.py` — `FOVPaths` frozen dataclass for consistent output paths
- `dataset.py` — `STARMapDataset` class: `from_config()`, `fov()` factory, `fov_ids()`, `load_codebook()`
- `fov.py` — `FOV` class: 11 `@log_step` pipeline methods + output/subtile operations
- `__init__.py` — Public API exports

**Key design patterns:**
- **Fluent chaining**: Every FOV method returns `self` → `fov.load_raw_images().enhance_contrast().global_registration()`
- **Delegation**: `FOV.layers` and `FOV.codebook` delegate to parent `STARMapDataset` via properties
- **Lazy imports**: Each method imports from Phase 1-5 at call time (avoids heavyweight deps on import)
- **Boundary conversion**: 0-based internally, 1-based only at CSV output boundary in `save_signal()`

**FOV pipeline methods:**
| Method | Delegates to |
|--------|-------------|
| `load_raw_images()` | `io.load_image_stacks()` |
| `enhance_contrast()` | `preprocessing.min_max_normalize()` |
| `hist_equalize()` | `preprocessing.histogram_match()` |
| `morph_recon()` | `preprocessing.morphological_reconstruction()` |
| `tophat()` | `preprocessing.tophat_filter()` |
| `make_projection()` | `utils.make_projection()` |
| `global_registration()` | `registration.register_volume()` |
| `local_registration()` | `registration.register_volume_local()` |
| `spot_finding()` | `spotfinding.find_spots_3d()` |
| `reads_extraction()` | `barcode.extract_from_location()` |
| `reads_filtration()` | `barcode.filter_reads()` |
| `save_ref_merged()` | `io.save_stack()` |
| `save_signal()` | `pd.DataFrame.to_csv()` |
| `create_subtiles()` | `np.savez_compressed()` |
| `from_subtile()` | `np.load()` (classmethod) |

**Tests:** 29 new tests (12 types + 6 dataset + 11 FOV pipeline/subtile), full suite 143 passing.

**Files Created:**
- `src/python/starfinder/dataset/` — 6 files (types.py, logging.py, paths.py, dataset.py, fov.py, __init__.py)
- `src/python/test/test_types.py` — 12 tests
- `src/python/test/test_dataset.py` — 6 tests
- `src/python/test/test_fov.py` — 11 tests

**Files Modified:**
- `src/python/starfinder/__init__.py` — Added `STARMapDataset`, `FOV` exports

**Usage:**
```python
from starfinder import STARMapDataset

dataset = STARMapDataset.from_config(config)
fov = dataset.fov("Position001")
fov.load_raw_images().enhance_contrast().morph_recon()
fov.global_registration()
fov.spot_finding().reads_extraction()
dataset.load_codebook(path)
fov.reads_filtration()
fov.save_signal()
```

### 2026-02-13: Phase 7 — E2E Validation & Spot Finding Improvements

End-to-end validation against synthetic ground truth revealed a false positive problem: 9,044 spots detected for 20 ground truth spots (precision = 0.002). Iterative investigation led to two complementary improvements: SNR-gated normalization and noise-floor-based thresholding.

**Phase 7a: E2E Validation Tests**

- [x] **Created validation utilities** (`starfinder.testdata.validation`)
  - `compare_shifts(shifts, gt, fov_id)` — per-round shift error (L1 per axis)
  - `compare_spots(spots, gt, fov_id)` — greedy nearest-neighbor matching with position tolerance
  - `compare_genes(spots, gt, fov_id)` — gene label accuracy for matched spots

- [x] **Created e2e test suite** (`test/test_e2e.py`) — 8 tests:
  - Pipeline smoke test (non-empty output with valid genes)
  - Shift recovery (per-axis error < 1.5px, CSV matches in-memory)
  - Spot detection (recall ≥ 0.7, precision ≥ 0.5)
  - Barcode decoding (color_seq accuracy, gene accuracy ≥ 0.5)
  - Subtile coordinate round-trip

- [x] **Session-scoped `e2e_result` fixture** (`test/conftest.py`)
  - Runs full pipeline once: load → enhance_contrast → global_reg → spot_finding → reads_extract → reads_filter → save_signal
  - Creates symlinks to restructure mini dataset directory layout

**Phase 7b: SNR-Gated Normalization**

- [x] **Added `snr_threshold` parameter to `min_max_normalize()`** (`preprocessing/normalization.py`)
  - Channels with `max/mean < snr_threshold` keep raw values (cast to uint8)
  - Prevents noise inflation in channels with no real signal
  - Default: `None` (backward compatible)
  - Recommended: `5.0` (noise-only channels have SNR ≈ 3-4, signal channels ≈ 12+)

- [x] **Exposed in FOV layer** (`dataset/fov.py`)
  - `enhance_contrast(snr_threshold=5.0)` passes through to normalization

**Phase 7c: Noise-Floor-Based Spot Finding Threshold**

- [x] **Root cause analysis**: Per-channel `min_max_normalize` inflates noise to [0, 255]. Per-channel adaptive threshold (`channel_max * 0.2`) can't distinguish inflated noise from real spots. The 0.2 fraction is an arbitrary magic number from MATLAB with no principled justification.

- [x] **Added `"adaptive_round"` mode** to `find_spots_3d()` (`spotfinding/local_maxima.py`)
  - Uses `max(all_channels) * intensity_threshold` instead of per-channel max
  - Combined with SNR gating: noise channels keep raw values (low max), signal channels reach 255

- [x] **Added `"noise"` mode** to `find_spots_3d()` — **now the default**
  - `threshold = median + k × MAD × 1.4826`
  - MAD (median absolute deviation) is robust to outliers (spots don't bias noise estimate)
  - `1.4826` converts MAD to σ under Gaussian assumption
  - Default `k=5` (intensity_threshold=5.0)
  - Normalization-independent: gives identical results on raw, SNR-gated, or fully normalized images
  - Principled: threshold adapts to actual noise level, not maximum intensity

- [x] **Changed defaults** in both `find_spots_3d()` and `FOV.spot_finding()`:
  - `intensity_estimation`: `"adaptive"` → `"noise"`
  - `intensity_threshold`: `0.2` → `5.0` (now means k-sigma, not fraction-of-max)

- [x] **Empirical comparison** (mini dataset, round 1, 20 GT spots):

  | Config | Total spots | ch0 (noise) | Recall | Precision |
  |--------|----------:|------------:|-------:|----------:|
  | adaptive + full norm (old) | 9,044 | 7,846 | 1.000 | 0.002 |
  | adaptive_round + SNR-gate | 1,616 | 418 | 1.000 | 0.012 |
  | **noise k=5 (any norm)** | **20** | **0** | **1.000** | **1.000** |

**Tests:** 155 total passing (+4 new: 2 SNR gating, 1 noise threshold, 1 adaptive_round)

**Files Created:**
- `src/python/starfinder/testdata/validation.py` — GT comparison utilities
- `src/python/test/test_e2e.py` — 8 e2e validation tests
- `docs/plans/2026-02-13-snr-gated-spot-finding-plan.md` — Implementation plan

**Files Modified:**
- `src/python/starfinder/testdata/__init__.py` — Added validation exports
- `src/python/starfinder/preprocessing/normalization.py` — Added `snr_threshold` parameter
- `src/python/starfinder/spotfinding/local_maxima.py` — Added `"adaptive_round"` and `"noise"` modes, changed defaults
- `src/python/starfinder/dataset/fov.py` — Exposed `snr_threshold`, updated `spot_finding()` defaults
- `src/python/test/conftest.py` — Added `e2e_result` fixture with `snr_threshold=5.0`
- `src/python/test/test_preprocessing.py` — +2 SNR gating tests
- `src/python/test/test_spotfinding.py` — +2 tests (noise, adaptive_round), fixed existing tests for new defaults

**Available spot finding modes:**
| Mode | Threshold | Use case |
|------|-----------|----------|
| `noise` (default) | `median + k × MAD × 1.4826` | Principled, noise-adaptive |
| `adaptive` | `channel_max × fraction` | MATLAB compatibility |
| `adaptive_round` | `round_max × fraction` | Cross-channel suppression |
| `global` | `dtype_max × fraction` | Fixed hardware threshold |

### 2026-02-17: E2E Benchmark Dataset & Pipeline Validation

Created a large-scale synthetic benchmark dataset and ran the full Python e2e pipeline with comprehensive QC reporting.

- [x] **Added "large" preset to testdata module** (`starfinder.testdata.synthetic`)
  - 1024×1024×30 volume, 2 FOVs, 4 rounds, 4 channels
  - 2000 spots per FOV, 64 genes (all CNNNNC barcodes)
  - Shift range: XY ±30, Z ±5 (seed=123)
  - ~967 MB total dataset

- [x] **Added `generate_codebook(n_genes)` function**
  - Enumerates all CNNNNC barcodes (C + {A,C,G,T}^3 + C = 64 possible)
  - Verifies unique color sequences (all 64 map uniquely)
  - Gene names: Gene001-Gene064

- [x] **Added `codebook` field to `SyntheticConfig`**
  - `None` (default) uses `TEST_CODEBOOK` (8 genes, backward compatible)
  - Large preset sets `codebook=generate_codebook(64)`
  - `generate_synthetic_dataset()` now uses resolved codebook throughout (gene assignment, CSV writing)

- [x] **Updated visualization for large codebooks**
  - `_generate_annotated_visualization()` uses `tab20` colormap instead of hardcoded 8-gene dict
  - Text annotations skipped when >50 spots (avoids clutter)

- [x] **Generated benchmark dataset** at `starfinder_benchmark/e2e/data/large/`
  - 32 TIFF images + codebook.csv + ground_truth.json + 2 annotation PNGs

- [x] **Ran full e2e pipeline** with per-step timing and memory tracking
  - Pipeline: load → enhance_contrast(snr=5.0) → global_registration → spot_finding → reads_extraction → reads_filtration(end_bases="CC")
  - Per-FOV: ~51s, Peak RSS ~2.4 GB
  - Registration dominates runtime (~30s, 58%)

- [x] **E2e benchmark results** (FOV_001 / FOV_002):

  | Metric | FOV_001 | FOV_002 |
  |--------|---------|---------|
  | Shift max error | 0.000 px | 0.000 px |
  | Spot recall | 1.000 | 1.000 |
  | Spot precision | 0.987 | 0.993 |
  | Gene accuracy | 1.000 | 1.000 |
  | Codebook match rate | 86.7% | 87.7% |
  | CNNNNC correct form | 86.7% | 87.7% |
  | Good spots | 1756 | 1765 |
  | Total time | 51.9s | 50.9s |

- [x] **Created inspection images**
  - `signal/{fov}_goodSpots.png` — Red dots on grayscale MIP background
  - `log/{fov}_inspection_registration.png` — Green/magenta composite overlay per round
  - `log/{fov}_qc.csv` — Full QC metrics (detection, extraction, filtering, timing, memory)

- [x] **Updated CLI** (`__main__.py`) to accept `--preset large`

**Files Modified:**
- `src/python/starfinder/testdata/synthetic.py` — Added `generate_codebook()`, `codebook` field, large preset, colormap visualization
- `src/python/starfinder/testdata/__main__.py` — Added "large" choice

**Benchmark output:**
```
starfinder_benchmark/e2e/
├── data/large/                              # 64-gene, 2000 spot/FOV dataset
│   ├── FOV_001/, FOV_002/                   # TIFF images (4 rounds × 4 channels)
│   ├── codebook.csv                         # 64 genes
│   ├── ground_truth.json                    # Shifts, spots, barcodes
│   └── ground_truth_annotation_*.png        # Spot overlay visualization
└── results/large/                           # Pipeline output
    ├── e2e_results.json                     # Full validation metrics
    ├── run_e2e_large.py                     # Reproducible script
    ├── signal/                              # Decoded reads + spot images
    │   ├── FOV_*_goodSpots.csv
    │   └── FOV_*_goodSpots.png
    └── log/                                 # Registration images + QC
        ├── FOV_*_inspection_registration.png
        ├── FOV_*_qc.csv
        └── gr_shifts/FOV_*.txt
```

### 2026-02-17: Phase 8 — Real Data E2E Benchmark (tissue-2D)

First end-to-end Python pipeline run on real microscopy data. Validated against MATLAB outputs (shifts, spot counts, gene overlap).

**Pipeline fixes for real data:**

- [x] **Fixed `load_codebook()` for headerless CSV + BOM** (`barcode/codebook.py`)
  - Real `genes.csv` files lack the `gene,barcode` header; `csv.DictReader` silently corrupts the first row
  - Opens with `encoding="utf-8-sig"` (strips BOM), peeks at first line to detect header
  - Backward compatible with existing header-bearing files
  - +2 tests (headerless, BOM)

- [x] **Added `FOV.rotate()` method with fast path** (`dataset/fov.py`)
  - All real datasets require `rotate_angle: -90` before registration
  - For exact 90° multiples: `np.rot90` (zero-copy, instant) — 390s → 28s (14x speedup)
  - General angles: falls back to `scipy.ndimage.rotate` with bilinear interpolation
  - Pipeline order: `load → rotate → enhance → register → spot_finding → extract → filter`
  - +1 test

- [x] **Discovered MATLAB channel order is wavelength-sorted** (`STARMapDataset.m`)
  - MATLAB default `channel_order_dict`: 488→546→594→647nm → `["ch00", "ch02", "ch01", "ch03"]`
  - ch01 and ch02 are **swapped** compared to filename order
  - All real datasets use this default (`seq_channel_order: []` in YAML config)
  - Confirmed correct via 100% gene overlap with MATLAB (wrong order would scramble gene assignments)

- [x] **Real data spot finding requires `"adaptive"` threshold**
  - `"noise"` mode (k-sigma, default) gave 19.2M spots on 3072×3072×30 — too permissive for autofluorescent tissue
  - MATLAB tissue-2D config uses `intensity_estimation="adaptive"`, `intensity_threshold=0.4` (40% of channel max)
  - With matching params: 67K total → 35.8K good (53.4% codebook match rate)

**tissue-2D tile_1 results:**

| Metric | Python | MATLAB | Agreement |
|--------|--------|--------|-----------|
| Shifts (round2) | (1, 84, -47) | (-1, 84, -48) | dz=2, dy=0, dx=1 |
| Shifts (round3) | (-1, 108, -73) | (1, 108, -73) | dz=2, dy=0, dx=0 |
| Shifts (round4) | (-3, 109, -87) | (2, 108, -92) | dz=5, dy=1, dx=5 |
| Good spots | 35,831 | 46,745 | 0.77 ratio |
| Unique genes | 64/64 | 64/64 | 100% overlap |
| Top-10 genes | — | — | 9/10 match |
| Total time | 487s | — | — |
| Peak RSS | ~4.5 GB | — | — |

**Per-step timing (tile_1, 3072×3072×30×4):**

| Step | Time (s) | Notes |
|------|----------|-------|
| load | 57 | 16 TIFFs from network mount |
| rotate | 28 | np.rot90 fast path |
| enhance | 46 | SNR-gated min-max normalization |
| registration | 290 | Phase correlation, 3 rounds |
| spot_finding | 30 | Adaptive threshold @ 0.4 |
| extraction | 36 | 67K spots × 4 rounds |
| filtration | 0.1 | Codebook lookup |

**Files Created:**
- `starfinder_benchmark/e2e/results/tissue_2D/run_e2e_tissue2D.py` — Self-contained benchmark script

**Files Modified:**
- `src/python/starfinder/barcode/codebook.py` — Header detection + BOM handling
- `src/python/starfinder/dataset/fov.py` — `rotate()` method with `np.rot90` fast path
- `src/python/test/test_barcode.py` — +2 tests
- `src/python/test/test_fov.py` — +1 test

**Test Results:** 158 tests passing (+3 new)

**Next Steps:**
- Scale tissue-2D to multiple FOVs
- Create LN and cell-culture-3D benchmark runners
- Cross-dataset comparison analysis

### 2026-02-18: E2E Benchmark — Tissue & Thick Medium Synthetic Presets

Added two new synthetic presets for e2e benchmarking at production-relevant scales and ran the full pipeline on both.

- [x] **Added `tissue` and `thick_medium` presets to `testdata.synthetic`**
  - `tissue`: 3072×3072×30, 2 FOVs, 14000 spots/FOV, 64 genes, shifts XY ±300 / Z ±7 (seed=456)
  - `thick_medium`: 1024×1024×100, 2 FOVs, 5200 spots/FOV, 64 genes, shifts XY ±100 / Z ±25 (seed=789)
  - Parameters matched `benchmark.presets.SIZE_PRESETS` and `SPOT_COUNTS`

- [x] **Updated CLI** (`__main__.py`) to accept `--preset tissue` and `--preset thick_medium`

- [x] **Generated datasets** at `starfinder_benchmark/e2e/data/{tissue,thick_medium}/`

- [x] **Created and ran e2e benchmark scripts**
  - `starfinder_benchmark/e2e/results/tissue/run_e2e_tissue.py`
  - `starfinder_benchmark/e2e/results/thick_medium/run_e2e_thick_medium.py`

- [x] **Reorganized benchmark output directory structure** (applied to all three scripts: large, tissue, thick_medium)
  - Registration images: `log/` → `log/gr_inspect/`
  - Signal images: `signal/` → `log/signal_inspect/`
  - QC CSV: `{fov}_qc.csv` → `{fov}.csv`
  - Renamed metrics: `spot_recall` → `detection_recall`, `spot_precision` → `detection_precision`, `n_correct_form_CNNNNC` → `n_correct_form`
  - Removed: `spot_mean_distance_px`

- [x] **Results:**

  | Metric | large (1024²×30) | thick_medium (1024²×100) | tissue (3072²×30) |
  |--------|-----------------|------------------------|-------------------|
  | Spots/FOV (GT) | 2,000 | 5,200 | 14,000 |
  | Shift recovery | 0 px | 0 px | 0 px |
  | Recall | 1.000 | 1.000 | 1.000 |
  | Precision | 0.987 | 0.981–0.986 | 0.983–0.986 |
  | Gene accuracy | 1.000 | 1.000 | 1.000 |
  | CNNNNC rate | 86.7% | 67.8–81.5% | 64.3–70.4% |
  | Time/FOV | ~52s | ~179s | ~482s |
  | Peak RSS | 2.4 GB | 7.5 GB | 20 GB |

**Files Modified:**
- `src/python/starfinder/testdata/synthetic.py` — Added tissue and thick_medium presets
- `src/python/starfinder/testdata/__main__.py` — Added preset choices

**Files Created (on network mount):**
- `starfinder_benchmark/e2e/data/{tissue,thick_medium}/` — Synthetic datasets
- `starfinder_benchmark/e2e/results/tissue/run_e2e_tissue.py` — Benchmark script
- `starfinder_benchmark/e2e/results/thick_medium/run_e2e_thick_medium.py` — Benchmark script

**Benchmark output structure (updated for all presets):**
```
starfinder_benchmark/e2e/results/{preset}/
├── e2e_results.json
├── run_e2e_{preset}.py
├── signal/
│   └── FOV_*_goodSpots.csv
└── log/
    ├── FOV_*.csv                              # QC metrics (27 cols)
    ├── gr_inspect/FOV_*_inspection_registration.png
    ├── signal_inspect/FOV_*_goodSpots.png
    └── gr_shifts/FOV_*.txt
```

### 2026-02-18: Phase 8 — Real Data E2E Benchmark (All 3 Datasets)

Standardized the real-data benchmark output format to match synthetic benchmarks, then ran the full Python pipeline on all 3 real datasets (tissue-2D, LN, cell-culture-3D) with 2 FOVs each. Compared against MATLAB outputs for shifts, spot counts, and gene overlap.

**Output format standardization:**

- [x] **Reorganized output directories** to match synthetic benchmark layout:
  - Registration images: `log/gr_inspect/{fov}_inspection_registration.png`
  - Signal images: `log/signal_inspect/{fov}_goodSpots.png`
  - QC CSV: `log/{fov}.csv` (consistent with synthetic format, minus GT fields)
  - MATLAB comparison: separated to `log/matlab_comparison/{fov}.csv`
- [x] **QC CSV format** matches synthetic benchmark minus ground truth fields:
  - Kept: detection metrics (n_all_spots, n_good_spots, codebook_match_rate), per-step timing, memory
  - Added: `gene_coverage`, `mean_color_score` (real-data specific)
  - Removed: GT-dependent fields (detection_recall/precision, gene/color_seq accuracy, shift errors)
- [x] **MATLAB comparison CSV** (`log/matlab_comparison/{fov}.csv`):
  - Per-round shift comparison (Python vs MATLAB with diff)
  - Both backends: n_all_spots, n_good_spots, codebook_match_rate
  - Gene overlap metrics: total overlap, top-10 gene overlap

**Dataset-specific parameters discovered:**

| Parameter | tissue-2D | LN | cell-culture-3D |
|-----------|-----------|----|--------------------|
| Rounds | 4 | 4 | 6 |
| Ref round | round1 | round4 | round1 |
| FOV pattern | `tile_%d` | `Position%03d` | `Position%03d` |
| Image size | 3072×3072×30 | 1496×1496×50 | 1496×1496×30 |
| Genes | 64 | 61 | 998 |
| Voxel size | (1,1,1) | (1,1,1) | (1,2,2) |
| Threshold | adaptive @ 0.4 | adaptive @ 0.2 | adaptive @ 0.2 |
| End bases | CC | AC | CC |
| Start base | C | A | C |

**Results summary (2 FOVs each):**

- [x] **tissue-2D** (tile_1, tile_2): 1085s total, 20 GB peak RSS
  - tile_1: 35.8K good spots (MATLAB: 46.7K, ratio=0.77), 100% gene overlap, 9/10 top-10
  - tile_2: 51.1K good spots (no MATLAB reference for tile_2), 100% gene coverage
  - Shifts match within 0-5px (dz sign differences expected due to Z-axis convention)

- [x] **LN** (Position001, Position002): 402s total, 8.1 GB peak RSS
  - Position001: 10.5K good spots (MATLAB: 8.8K, ratio=1.19), 95.1% gene overlap, 10/10 top-10
  - Position002: 10.1K good spots (MATLAB: 8.4K, ratio=1.21), 95.1% gene overlap, 9/10 top-10
  - **dz sign flip**: Position001 shows systematic dz negation (Python=-16, MATLAB=+16); Position002 matches when dz=0
  - Python detects ~20% more spots than MATLAB (likely from different spot finding internals)

- [x] **cell-culture-3D** (Position351, Position352): 519s total, 5.6 GB peak RSS
  - Position351: 31.6K good spots (MATLAB: 33.5K, ratio=0.95), 100% gene overlap, 10/10 top-10
  - Position352: 32.8K good spots (MATLAB: 33.5K, ratio=0.98), 99.8% gene overlap, 10/10 top-10
  - Best MATLAB agreement of all 3 datasets (0.95-0.98 spot ratio)
  - Most rounds match shifts exactly; dz differs by 0-6px on some rounds

- [x] **MATLAB shift log parsing**: tissue-2D uses structured CSV at `log/gr_shifts/{fov}.txt`; LN and cell-culture-3D embed shifts in text log files at `log/{fov}.txt` — created `parse_matlab_shifts_from_log()` regex parser

**Files Created (on network mount):**
- `starfinder_benchmark/e2e/results/tissue_2D/run_e2e_tissue2D.py` — Updated with standardized output format
- `starfinder_benchmark/e2e/results/LN/run_e2e_LN.py` — New benchmark script
- `starfinder_benchmark/e2e/results/cell_culture_3D/run_e2e_cell_culture_3D.py` — New benchmark script

**Benchmark output structure (real data):**
```
starfinder_benchmark/e2e/results/{dataset}/
├── run_e2e_{dataset}.py                          # Self-contained benchmark script
├── signal/
│   └── {fov}_goodSpots.csv                       # Decoded reads
└── log/
    ├── {fov}.csv                                  # QC metrics (matching synthetic format)
    ├── gr_inspect/{fov}_inspection_registration.png
    ├── signal_inspect/{fov}_goodSpots.png
    ├── gr_shifts/{fov}.txt                        # Shift log
    └── matlab_comparison/{fov}.csv                # MATLAB comparison
```

**Next Steps:**
- Investigate dz sign flip issue (Python vs MATLAB Z-axis convention)
- Scale to more FOVs per dataset for statistical significance
- Cross-dataset comparison analysis

### Performance Optimization (2026-02-23)

Three-tier optimization targeting memory (peak RSS) and runtime for cluster scheduling.

**Phase A: Quick Wins** (commit 560f3f8)
- Vectorized barcode extraction: replaced Python loop over spots with NumPy fancy indexing
- Float32 normalization: `min_max_normalize` uses float32 instead of float64
- String concatenation: `color_seq` built with `np.add.reduce` instead of per-row string ops

**Phase B: Streaming Pipeline** (commit cb520a7)
- `FOV.run_streaming()`: processes one sequencing round at a time, discarding after registration
- Only the reference round remains in memory after completion
- Validated: streaming produces identical spot counts and gene assignments as batch mode
- Tests: `TestE2EStreamingMode` in `test_e2e.py` (2 tests: output match + memory release)

**Phase C: Registration Memory Optimization** (uncommitted)
Three fixes to reduce peak RSS during phase correlation registration:

1. **uint16 merged images** (`fov.py:_make_ref_3d`): `np.sum(uint8, axis=-1)` defaults to int64 (8 bytes/px). Since max sum of 4 uint8 channels = 1020, uint16 (2 bytes/px) suffices. Saves 3.2 GB on tissue-sized volumes.

2. **Real FFT** (`phase_correlation.py:phase_correlate`): Replaced `fftn`/`ifftn` with `rfftn`/`irfftn`. Exploits conjugate symmetry of real-valued input — last axis is half-size. Saves ~2 GB. Verified mathematically equivalent: same argmax, relative cc diff ~4×10⁻⁷.

3. **Integer roll fast path** (`phase_correlation.py:apply_shift`): Phase correlation always returns integer shifts. Old code used FFT round-trip (`fourier_shift` → `fftn` → `ifftn`) even for integer shifts. New code uses `np.roll` + zero-fill — no FFT allocation at all. Saves ~6 GB per channel on tissue-sized volumes.

**Streaming benchmark results (before memory fixes):**

| Dataset | Batch Peak | Stream Peak | Reduction |
|---------|-----------|-------------|-----------|
| large | 2.4 GB | 2.1 GB | 11-15% |
| thick_medium | 7.5 GB | 6.6 GB | 11-13% |
| tissue | 19.8 GB | 17.5 GB | 11-13% |
| tissue_2D | 19.9 GB | 17.5 GB | 11-13% |
| LN | 8.0 GB | 7.0 GB | 11-14% |
| cell_culture_3D | 5.5 GB | 4.3 GB | 19-23% |

Streaming alone only gave 11-23% because FFT temporaries during registration dominate peak RSS (not round image storage).

**After memory fixes (streaming mode):**

| Dataset | Before (batch) | After (stream+fixes) | Reduction | Speedup |
|---------|---------------|---------------------|-----------|---------|
| tissue_2D | 19.9 GB | 11.0 GB | 44-45% | 2.2x |
| tissue | 19.8 GB | 11.0 GB | 44-45% | 2.1x |
| cell_culture_3D | 5.5 GB | 2.8 GB | 48-51% | 2.3x |

Combined streaming + memory fixes achieve ~50% peak RSS reduction and ~2x runtime speedup, primarily from eliminating unnecessary FFT round-trips in `apply_shift`.

**Benchmark script:** `starfinder_benchmark/e2e/results/run_streaming_benchmark.py` — single parametrized script handling all 6 datasets (3 synthetic, 3 real).

### 2026-02-24: Fix dz Sign Flip — Rotation Direction & Shift Comparison

Root-caused and fixed the systematic dz sign flip between Python and MATLAB registration shifts observed in the LN dataset benchmark (and subtly present in all datasets).

**Root cause analysis:**

Two interacting issues created the observed pattern where Y/X shifts agreed but Z was flipped:

1. **Sign convention difference** (`phase_correlate` vs `DFTRegister3D`):
   - Python `phase_correlate()` returns **+d** (physical displacement: "moving is shifted by +d from fixed")
   - MATLAB `DFTRegister3D()` returns **-d** (correction: "apply -d to align moving to fixed")
   - This difference alone would flip **all** axes, not just Z

2. **Rotation direction mismatch** (`fov.py` vs `STARMapDataset.m`):
   - MATLAB: `imrotate(img, -90)` → CW 90° rotation
   - Python (buggy): `np.rot90(vol, k=-k_90)` where `k_90 = angle // 90 = -1`, so `k = -(-1) = 1` → **CCW** 90° rotation
   - Opposite rotations flip Y/X displacement signs in the rotated coordinate system
   - This Y/X flip **cancels** with the sign convention difference on Y/X, making them appear to agree
   - Z is perpendicular to the Y-X rotation plane, so it shows the raw sign convention difference

**The fix (1 character):**

```python
# fov.py line 115
# BEFORE (buggy — CCW rotation):
np.rot90(vol, k=-k_90, axes=yx_axes)

# AFTER (fixed — CW rotation, matches MATLAB):
np.rot90(vol, k=k_90, axes=yx_axes)
```

- Updated comment on line 113 to clarify that `np.rot90` and `imrotate` share sign convention
- All 160 tests pass after the fix

**Benchmark comparison script updates:**

After the rotation fix, Python and MATLAB shifts now have a consistent sign relationship across all axes: `py_val ≈ -ml_val`. Updated all 3 real-data benchmark scripts:

- `run_e2e_tissue2D.py` (line 167-169)
- `run_e2e_LN.py` (line 205-207)
- `run_e2e_cell_culture_3D.py` (line 203-205)

```python
# BEFORE (masked by rotation cancellation on Y/X):
abs(py_dy - ml_dy)  # happened to work for Y/X, failed for Z

# AFTER (correct sign convention: Python=+d, MATLAB=-d):
abs(py_dy + ml_dy)  # sum ≈ 0 for perfect agreement on all axes
```

**Files Modified:**
- `src/python/starfinder/dataset/fov.py` — Fixed rotation direction (`k=-k_90` → `k=k_90`)
- `starfinder_benchmark/e2e/results/tissue_2D/run_e2e_tissue2D.py` — Shift comparison formula
- `starfinder_benchmark/e2e/results/LN/run_e2e_LN.py` — Shift comparison formula
- `starfinder_benchmark/e2e/results/cell_culture_3D/run_e2e_cell_culture_3D.py` — Shift comparison formula

**Test Results:** 160 tests passing (no changes to test count)

**Next Steps:**
- Re-run all 3 real-data benchmarks to verify consistent shift agreement across all axes
- The LN "dz sign flip issue" should be fully resolved

### 2026-02-25: E2E Local Registration Benchmark

Benchmark to evaluate whether adding demons-based local registration after global registration improves gene decoding accuracy. Plan at `docs/plans/2026-02-24-e2e-local-registration-benchmark-plan.md`.

**Implementation:**

1. **Added `"linear"` deformation type** to `starfinder/benchmark/data.py`:
   - Originally: `d = c1*x + c2*y + c3*z` (pure linear, zero mean, 5px cap)
   - Updated (2026-02-25): `d = c0 + c1*x + c2*y + c3*z + c4*x*y` (affine + bilinear cross-term, 10px cap)
   - The `x*y` cross-term creates saddle-shaped non-linearity that only local registration can correct
   - The constant `c0` is absorbable by global registration; the non-linear residual tests demons
   - Config: `"linear_small"` with 10px max displacement (was 5px)

2. **Data generation script** (`starfinder_benchmark/e2e_LR/data/generate_data.py`):
   - Two-step: `generate_synthetic_dataset()` → post-process non-ref rounds with `apply_deformation_field()`
   - 3 presets: large (1024²×30), tissue (3072²×30), thick_medium (1024²×100)
   - Deterministic seeds: `base_seed(200) + fov_idx * 100 + round_idx * 25`

3. **Benchmark script** (`starfinder_benchmark/e2e_LR/results/run_e2e_LR.py`):
   - Streaming-style per-round processing (ref + one moving round in memory at a time)
   - Two modes: `global_only` (baseline) and `global_local` (with demons registration)
   - 6 dataset configs (3 synthetic + 3 real)
   - Per-round registration inspection images generated before discarding round data
   - 29-column QC CSV (27 base + `time_local_reg_s` + `rss_after_local_reg_mb`)
   - Real data mode compares against existing global-only results from `starfinder_benchmark/e2e/results/`

**Synthetic results (10px bilinear deformation, `d = c0 + c1*x + c2*y + c3*z + c4*x*y`):**

| Preset | FOV | Mode | Gene Acc | Match Rate | N Good | Time | Peak RSS |
|--------|-----|------|----------|------------|--------|------|----------|
| large | FOV_001 | global_only | 0.984 | 0.516 | 1046 | 33s | 1.5 GB |
| large | FOV_001 | global_local | 0.983 | 0.534 | 1084 | 196s | 6.9 GB |
| large | FOV_002 | global_only | 0.967 | 0.451 | 924 | 30s | 1.5 GB |
| large | FOV_002 | global_local | 0.907 | 0.433 | 887 | 234s | 6.9 GB |
| tissue | FOV_001 | global_only | 0.985 | 0.464 | 6607 | 288s | 11.2 GB |
| tissue | FOV_001 | global_local | 0.959 | 0.500 | 7114 | 1698s | 59.5 GB |
| tissue | FOV_002 | global_only | 0.947 | 0.500 | 7117 | 342s | 11.2 GB |
| tissue | FOV_002 | global_local | 0.936 | 0.517 | 7352 | 1486s | 59.5 GB |
| thick_medium | FOV_001 | global_only | 0.969 | 0.563 | 2981 | 101s | 4.3 GB |
| thick_medium | FOV_001 | global_local | 0.969 | 0.557 | 2948 | 599s | 20.9 GB |
| thick_medium | FOV_002 | global_only | 0.960 | 0.550 | 2906 | 113s | 4.3 GB |
| thick_medium | FOV_002 | global_local | 0.958 | 0.542 | 2861 | 668s | 20.9 GB |

Even with the stronger 10px bilinear deformation (doubled from 5px, added x*y cross-term), gene accuracy stays >0.93 for both modes. Local registration provides no quality benefit and sometimes actively degrades results (large FOV_002: 0.967→0.907).

**Real data results (global_only → global_local):**

| Dataset | FOV | Good Spots (GO→GL) | Match Rate (GO→GL) | Peak RSS |
|---------|-----|--------------------|--------------------|----------|
| tissue_2D | tile_1 | 35,831 → 33,948 (-5%) | 0.534 → 0.506 | 59.3 GB |
| tissue_2D | tile_2 | 51,095 → 44,411 (-13%) | 0.734 → 0.638 | 59.5 GB |
| LN | Pos001 | 10,471 → 10,182 (-3%) | 0.881 → 0.857 | 23.2 GB |
| LN | Pos002 | 10,123 → 6,961 (-31%) | 0.778 → 0.535 | 23.2 GB |
| cell_culture_3D | Pos351 | 31,623 → 29,175 (-8%) | 0.390 → 0.359 | 14.4 GB |
| cell_culture_3D | Pos352 | 32,842 → 29,567 (-10%) | 0.401 → 0.361 | 14.5 GB |

**Key findings:**

1. **Local registration consistently degrades results on all real datasets.** These datasets don't have significant local deformations — global registration already aligns rounds well.
2. **Demons over-warps sparse fluorescence data.** With only 1-5% of voxels carrying signal, the displacement field optimization is driven by background noise, misaligning the sparse spots. This was confirmed at both 5px and 10px deformation levels — FOV_002 large dropped from 0.967→0.907 with local reg.
3. **Memory cost is prohibitive.** tissue hits 59.5 GB peak RSS with local reg (vs 11.2 GB global-only); thick_medium hits 20.9 GB (vs 4.3 GB). The 3-level anti-aliased pyramid on large volumes creates massive SimpleITK temporaries.
4. **Even 10px bilinear deformation doesn't break global-only.** Synthetic spots are isolated Gaussian blobs — a 10px spatial warp doesn't move spots out of their local neighborhood enough to confuse barcode extraction. Gene accuracy stays >0.93 across all presets without local registration.
5. **Demons' value proposition is for dense-texture real tissue data** (autofluorescence background provides alignment signal), not sparse synthetic spots. The benchmark confirms the known limitation quantitatively.

**Conclusion:** Local registration should only be applied when there is evidence of local deformations (e.g., tissue clearing artifacts, sample warping). For standard STARmap data with rigid body motion, global registration is sufficient. Synthetic sparse-spot benchmarks are inherently poor at evaluating demons because the signal is too sparse for meaningful displacement field optimization.

**Files Created/Modified:**
- `src/python/starfinder/benchmark/data.py` — Added `"linear"` deform type + `"linear_small"` config
- `starfinder_benchmark/e2e_LR/data/generate_data.py` — Data generation script
- `starfinder_benchmark/e2e_LR/results/run_e2e_LR.py` — Benchmark runner script
- `docs/plans/2026-02-24-e2e-local-registration-benchmark-plan.md` — Plan document

**Results location:** `starfinder_benchmark/e2e_LR/results/{large,tissue,thick_medium,tissue_2D,LN,cell_culture_3D}/`

### 2026-02-25: Phase E — Spot-Based TPS Local Registration

Implemented a spot-based Thin Plate Spline (TPS) local registration method as an alternative to demons. TPS operates on matched spot correspondences instead of dense voxel-level optimization — fitting a smooth displacement field from ~1000 control points using scipy's `RBFInterpolator`.

**Implementation:**

1. **New module `starfinder/registration/pointset.py`** with 6 functions:
   - `detect_and_match_spots()` — percentile-based CCA spot detection + `cKDTree` nearest-neighbor matching
   - `subsample_control_points()` — greedy farthest-point sampling for uniform spatial coverage
   - `tps_displacement_field()` — multi-output `RBFInterpolator(kernel='thin_plate_spline')`, coarse grid eval (stride=32), `scipy.ndimage.zoom(order=3)` to full resolution
   - `apply_tps_deformation()` — slice-by-slice `map_coordinates(order=1)` warping (113 MB/slice vs 3.4 GB for full grid)
   - `tps_register()` — end-to-end: detect → match → subsample → fit → dense field
   - `register_volume_tps()` — multi-channel wrapper mirroring `register_volume_local()` signature

2. **FOV integration** — `local_registration(method="tps")` routing with fallback to demons on insufficient spots. `run_streaming()` accepts `local_method`/`local_kwargs` for local registration in streaming mode.

3. **Sign convention**: displacement field stores `p_moving - p_fixed` (backward mapping), so `map_coordinates(moving, pos + disp)` recovers the fixed image. Same convention as SimpleITK internally.

4. **No SimpleITK dependency** — pure numpy/scipy. Dependencies: `RBFInterpolator`, `cKDTree`, `map_coordinates`, `zoom`.

5. **Tests**: 4 new tests (identity, known deformation recovery, too-few-spots ValueError, shape check). All 164 tests pass.

**Benchmark (38 runs: 7 synthetic × 5 deformations + 3 real):**

All 38 runs succeeded (100% success rate), including `tiny` (10 spots) with relaxed config (`min_matches=5`, `smoothing=2.0`).

**Performance — TPS vs Demons (polynomial_small deformation):**

| Dataset | TPS time | Demons time | Speedup | TPS mem | Demons mem | Mem ratio |
|---------|----------|-------------|---------|---------|------------|-----------|
| tiny (128²×8) | 0.3s | 1.1s | 3.4x | 3 MB | 25 MB | 8.6x |
| medium (512²×32) | 15.0s | 15.9s | 1.1x | 173 MB | 993 MB | 5.7x |
| large (1024²×30) | 51.1s | 58.1s | 1.1x | 646 MB | 3.9 GB | 6.1x |
| tissue (3072²×30) | 453.6s | 499.9s | 1.1x | 5.8 GB | 35.4 GB | 6.1x |
| thick_medium (1024²×100) | 169.4s | 159.7s | 0.9x | 2.2 GB | 12.3 GB | 5.7x |
| tissue_2D (real) | 325.7s | 504.7s | 1.5x | 5.7 GB | 35.4 GB | 6.2x |
| LN (real) | 131.5s | 180.1s | 1.4x | 2.2 GB | 13.7 GB | 6.1x |
| cell_culture_3D (real) | 79.3s | 116.4s | 1.5x | 1.3 GB | 8.4 GB | 6.2x |

**Quality — TPS vs Demons (NCC after / Match Rate after):**

| Deformation type | TPS NCC (avg) | Demons NCC (avg) | TPS MR (avg) | Demons MR (avg) |
|------------------|---------------|------------------|--------------|-----------------|
| gaussian_small | 0.86 | 0.93 | 91% | 84% |
| gaussian_large | 0.68 | 0.80 | 71% | 75% |
| multi_point | 0.73 | 0.82 | 76% | 79% |
| polynomial_small | 0.12 | 0.42 | 4% | 56% |
| polynomial_large | 0.04 | 0.13 | 1% | 14% |
| Real datasets | 0.31 | 0.68 | 6% | 37% |

**Key findings:**

1. **Memory: TPS wins decisively (6x less)** across all dataset sizes. No iterative optimization buffers or SimpleITK temporaries.
2. **Speed: 1.0-1.5x faster** — tied on large synthetic, 1.5x faster on real data where spot detection is fast.
3. **Quality: Demons significantly better for large/polynomial deformations.** TPS struggles when spots move beyond the `match_distance` (10px) — it can't recover displacements in regions where no spots matched. Demons' dense iterative optimization handles this by propagating corrections from background gradients.
4. **Quality: TPS competitive for small gaussian deformations.** NCC 0.86 vs 0.93 for gaussian_small. On thick_medium, TPS achieves **better** Match Rate than demons (70-89% vs 54-68%) — more Z-slices provide more spots for matching.
5. **TPS is best suited for small-to-moderate deformations** where spot density is high. For complex tissue deformations, demons remains superior.

**Files Created:**
- `src/python/starfinder/registration/pointset.py` — Core TPS module (6 functions)
- `src/python/test/test_pointset.py` — 4 unit tests
- `starfinder_benchmark/registration/results/scripts/benchmark_tps_single.py` — Phase 1 runner
- `starfinder_benchmark/registration/results/scripts/run_tps_python.py` — Orchestrator (38 runs + Phase 2 eval)

**Files Modified:**
- `src/python/starfinder/registration/__init__.py` — Added TPS exports
- `src/python/starfinder/dataset/fov.py` — TPS routing + streaming support

**Results location:** `starfinder_benchmark/registration/results/local_tps/` (38 datasets, summary.csv)

### 2026-02-25: CPD Local Registration — Implementation & Benchmark (NEGATIVE RESULT)

Implemented Coherent Point Drift (CPD) registration as a second point-set-based alternative to demons. CPD models moving points as GMM centroids and fixed points as observations, solving via EM. Two-stage: affine CPD (global alignment) → non-rigid CPD (local via `T(Y) = Y + G@W` with Gaussian kernel).

**Implementation (7 new functions in `pointset.py`):**

1. `_gaussian_kernel(Y, beta)` — N×N Gaussian kernel matrix `G(i,j) = exp(-||y_i-y_j||²/(2β²))`
2. `_subsample_points(points, max_n)` — Random subsampling for tractable matrix solve
3. `_cpd_e_step(X, T, sigma2, w)` — EM E-step: posterior responsibilities P(m|n)
4. `_cpd_sigma2(X, T, P1, Pt1, PX)` — Updated variance estimate
5. `cpd_affine(X, Y, ...)` — Affine CPD: solves for B (linear) + t (translation) via EM
6. `cpd_nonrigid(X, Y, ...)` — Non-rigid CPD: solves for W weights via `(diag(P1)G + λσ²I)W = PX - diag(P1)Y`
7. `cpd_displacement_field(Y, W, B, t, beta, shape, grid_spacing)` — Dense backward displacement field from CPD parameters

Also added `cpd_register()` (end-to-end) and `register_volume_cpd()` (multi-channel wrapper).

**Benchmark (38 runs: 7 synthetic × 5 deformations + 3 real):**

37/38 success, 1 timeout (tissue/gaussian_large at 600s limit). All completed runs produced **catastrophically wrong** results — NCC dropped in 35/38 cases.

**Results — CPD quality (NCC before → after, representative cases):**

| Dataset/Deformation | NCC before | NCC after | Verdict |
|---------------------|-----------|-----------|---------|
| medium/gaussian_small | 0.923 | 0.004 | Destroyed |
| large/gaussian_small | 0.910 | 0.003 | Destroyed |
| tissue/gaussian_small | 0.913 | 0.001 | Destroyed |
| medium/polynomial_small | 0.029 | 0.026 | No improvement |
| Real: cell_culture_3D | 0.539 | 0.098 | Destroyed |
| Real: tissue_2D | 0.021 | 0.011 | Destroyed |
| Real: LN | 0.346 | 0.029 | Destroyed |

**Bug found and fixed — Per-axis normalization:**

Original `cpd_affine` used `all_pts.std()` (scalar) for coordinate normalization. With Z in [0,31] and YX in [0,511], global std ≈ 210, leaving Z normalized to [-0.07, 0.08] vs YX [-1.2, 1.2]. This made the affine M-step ill-conditioned (B[0,0] = -0.094 instead of ~1.0).

**Fix:** Per-axis std normalization + corrected denormalization formula:
```python
scale = all_pts.std(axis=0)  # was: all_pts.std()
B = (scale[:, None] / scale[None, :]) * Bn  # was: B = Bn
t = center @ (np.eye(D) - B.T) + tn * scale  # was: t = center @ (I - Bn.T) + tn * scale
```

After fix: affine B-I max improved from 1.094 to 0.057. But non-rigid still catastrophically bad.

**Root cause analysis (non-rigid failure):**

Extensive parameter sweep tested all combinations of:
- β ∈ {0.5, 1.0, 2.0, 3.0, 5.0} × median nearest-neighbor distance
- λ ∈ {2, 10, 50}
- n_points ∈ {50, 100, 200, 500, 1000}
- With/without affine pre-alignment
- Percentile thresholds ∈ {95, 97, 99, 99.5, 99.9}
- Even with pre-matched pairs (TPS-style, 500 pairs, ~0px mean displacement)

**ALL configurations failed.** Three fundamental issues:

1. **Gaussian kernel ill-conditioning:** `cond(G)` ranges from 10⁹ to 10¹⁹ depending on β and N. The linear system `(diag(P1)G + λσ²I)W = ...` produces W weights with huge magnitudes (W_rms > 3000) that nearly cancel at control point locations but diverge wildly at interpolation points.

2. **Noisy spot detection:** `detect_spots` with percentile-based thresholding finds 80K-96K "spots" in a 512×512×32 volume. Even at percentile=99.9 (392 spots), the false-positive rate is too high for CPD's soft correspondence model — the EM assigns probability mass to wrong pairs.

3. **Dense field extrapolation failure:** CPD learns `T(Y) = Y + G@W` at control points Y, but evaluating `K(p, Y)@W` at arbitrary grid points p amplifies the ill-conditioned W weights. Values that balance at Y diverge at grid points between and beyond control points.

**Conclusion:** CPD non-rigid registration is fundamentally unsuitable for sparse fluorescence microscopy data. The Gaussian kernel's global support creates severe ill-conditioning that cannot be mitigated by parameter tuning. TPS (local RBF support) and demons (dense iterative optimization) remain the recommended methods.

**Files Created:**
- `starfinder_benchmark/registration/results/scripts/benchmark_cpd_single.py` — Phase 1 runner
- `starfinder_benchmark/registration/results/scripts/run_cpd_python.py` — Orchestrator

**Files Modified:**
- `src/python/starfinder/registration/pointset.py` — Added 7 CPD functions + per-axis normalization fix
- `src/python/starfinder/registration/__init__.py` — Added CPD exports
- `src/python/test/test_pointset.py` — Added CPD unit tests (5 new, 169 total pass)

**Results location:** `starfinder_benchmark/registration/results/local_cpd/` (38 datasets, summary.csv)

### 2026-02-25: CPD Spot Detection Fix — Iterating Toward Working CPD (IN PROGRESS)

Improved spot detection in CPD pipeline from percentile-based to MAD-based noise-floor thresholding. This is a critical prerequisite — the original `detect_spots` found 80K-96K false spots on a 512³ volume, flooding CPD's EM with noise.

**Changes:**

1. **`detect_spots` upgraded** (`metrics.py`): Added `threshold_mode="noise"` using per-channel MAD threshold (`median + k × MAD × 1.4826`) + `peak_local_max` with `min_distance` suppression. Matches `find_spots_3d` workflow. Supports both 3D and 4D (per-channel) input with spatial deduplication via `cKDTree.query_pairs`. Old `threshold_mode="percentile"` preserved for backward compatibility.

2. **CPD default `detection_threshold` raised to 5.0** (was 3.0), matching `FOV.spot_finding()`. On the tiny dataset (128²×8, 10 real spots): k=5.0 detects exactly 10 spots; k=2.0 detects 1342; old percentile mode detected 1090.

3. **Benchmark config updated**: Both `DEFAULT_CONFIG` and `SMALL_CONFIG` now use `detection_threshold=5.0`.

**Tiny dataset results (k=5.0, peak_local_max):**

| Deformation | NCC before | NCC after | Change | Verdict |
|---|---|---|---|---|
| gaussian_small | 0.884 | 0.856 | -0.028 | Near-identity (preserves) |
| gaussian_large | 0.702 | 0.652 | -0.050 | Near-identity (preserves) |
| polynomial_small | 0.323 | **0.471** | **+0.148** | **First CPD improvement!** |
| polynomial_large | 0.067 | 0.063 | -0.004 | No improvement |
| multi_point | — | FAILED | — | 9 moving spots (need ≥10) |

**Key insight — asymmetric spot detection is the remaining bottleneck:**

`polynomial_small` works because both ref and mov have ~10 clean spots → 10×10 kernel (`cond(G) = 8×10⁶`), manageable W weights (`W_rms=2.2`).

`polynomial_large` fails because severe deformation destroys spot morphology in the moving image — the noise-floor detector finds **2574 false spots** in mov (vs 10 in ref). This 257:1 imbalance causes:
- Affine CPD to produce garbage: `B-I max = 15.5` (should be ~0.3)
- Non-rigid kernel to be ill-conditioned: `cond(G) = 2×10¹⁹`, `W_rms = 90`
- Displacement field to blow up: `G@W max = 151px`

**Progression of CPD results across iterations (tiny/gaussian_large):**

| Iteration | NCC after | Field range | Issue |
|---|---|---|---|
| v1: percentile + CCA | 0.036 | [-49, 100] | 80K false spots, kernel blowup |
| v2: MAD k=2.0 + CCA | 0.656 | [-0.02, 0.14] | Still 1400+ spots at k=2 |
| v3: MAD k=5.0 + peak_local_max | 0.652 | [-0.47, 0.41] | Clean 10 spots, stable kernel |

**Next steps:**
- Address asymmetric spot count problem (mov has far more false detections than ref under large deformations)
- Consider pre-matching spots (like TPS) before feeding to CPD, or using CPD only on balanced point clouds
- Test on medium/large datasets where more real spots exist

### 2026-02-26: Unified Synthetic Data Generation & Coordinate-Level Deformation

Merged two independent synthetic data generators (`starfinder.testdata` and `starfinder.benchmark.data`) into a single unified module at `starfinder.benchmark.synthetic`. The key improvement is **coordinate-first rendering**: both global shifts and local deformations are applied to spot *positions* before rendering, so images always contain clean analytical Gaussians — no interpolation blur from warping rendered images.

**What changed:**

1. **New `benchmark/synthetic.py`** — unified generator absorbing `testdata/synthetic.py` + generation code from `benchmark/data.py`
   - `apply_shift_to_spots(spots, shift, shape)` — coordinate-level global shift, drops out-of-bounds spots
   - `apply_deformation_to_spots(spots, field, shape)` — samples displacement field at spot positions, drops out-of-bounds
   - `generate_synthetic_dataset()` — multi-round, multi-channel E2E datasets (replaces `testdata.generate_synthetic_dataset`)
   - `generate_registration_benchmark()` — single-channel ref/mov pairs (replaces `data.generate_synthetic_benchmark`)
   - Per-round spot variation: ~10% intensity jitter, ~5% sigma jitter per (spot, round), deterministic via `seed + spot_id * 100 + round_idx`

2. **New `benchmark/validation.py`** — moved from `testdata/validation.py` (compare_shifts, compare_spots, compare_genes, e2e_summary)

3. **New `benchmark/__main__.py`** — CLI replaces `python -m starfinder.testdata`:
   - `uv run python -m starfinder.benchmark --preset small --output ...` (e2e mode, default)
   - `uv run python -m starfinder.benchmark --mode registration --preset tiny --output ...`

4. **Slimmed `benchmark/data.py`** — only retains `generate_inspection_image`, `generate_overview_grid`, `extract_real_benchmark_data`, `REAL_DATASETS`

5. **Deleted `starfinder/testdata/`** package entirely (4 files)

6. **Deleted `test/test_synthetic.py`**, replaced by `test/test_benchmark_synthetic.py` (19 tests for coordinate transforms, rendering, presets, codebook)

7. **Cleaned up presets** — removed `xlarge` and `thick_large` from `SIZE_PRESETS`, `SPOT_COUNTS`, `SHIFT_RANGES`. 6 canonical presets: tiny, small, medium, large, tissue, thick_medium.

8. **Updated all consumers** — 5 in-repo test/source files, 5 network-mount benchmark scripts, `pyproject.toml`, `CLAUDE.md`

9. **Regenerated small fixture dataset** at `tests/fixtures/synthetic/small/` with new format (ground truth v2.0)

**Spot tuple format change:** `(z, y, x, intensity)` → `(z, y, x, intensity, sigma)` — each spot carries its own Gaussian width, enabling per-round PSF variation.

**Ground truth format:** version 2.0 — adds optional `deformations` key per FOV for rounds with local deformation.

**Tests:** 186 passing (unchanged count — deleted `test_synthetic.py` replaced by 19-test `test_benchmark_synthetic.py`)

### 2026-02-26: CPD Correspondence-Aware Subsampling

Fixed the core subsampling problem in `cpd_register()`: independent FPS on fixed and moving clouds destroyed cross-cloud correspondences. On tissue (14K spots → 1K), only 22% of subsampled points retained their true nearest neighbor.

**Root cause (two issues):**
1. **Independent FPS drops true matches** — a fixed point's partner may not survive FPS in the other cloud
2. **Single-point anchors lose local context** — FPS picks one representative per region, but CPD needs neighboring spots to disambiguate which moving spot belongs to which fixed spot

**Solution — correspondence-aware subsampling (no hard pre-matching):**
1. FPS on fixed cloud for spatial anchors (`max_anchors = max_control_points // (1 + k_neighbors)`)
2. Expand each anchor with K nearest neighbors from full fixed cloud (preserves local cluster structure)
3. Gather all moving points within radius of enriched fixed cloud (ensures true correspondences survive)
4. CPD's EM does soft assignment on the enriched clouds — no explicit matching required

**New functions in `pointset.py`:**
- `_subsample_with_neighbors(points, max_anchors=250, k_neighbors=3)` — FPS + KDTree neighbor expansion
- `_gather_candidates(fixed_sub, moving_all, radius=15.0)` — radius-based moving candidate gathering

**New parameters on `cpd_register()`:**
- `candidate_radius: float = 15.0` — moving candidate search radius
- `k_neighbors: int = 3` — fixed neighbors per FPS anchor

**Updated `FOV.local_registration()`** to pass `candidate_radius` and `k_neighbors` through to CPD.

**Results with 300 anchors + 3 neighbors, radius=15px (from planning analysis):**

| Preset | |X| fixed | |Y| moving | True NN preserved | Kernel mem |
|---|---|---|---|---|
| large | 1105 | 1199 | **100%** (was 82%) | 12 MB |
| tissue | 1200 | 1342 | **98%** (was 22%) | 14 MB |
| thick_medium | 1200 | 1405 | **98%** (was 35%) | 16 MB |

**Budget math:** With default `max_control_points=1000`, `k=3`: 250 anchors × 4 ≈ 1000 fixed, ~1200 moving → kernel 12-16 MB (well within budget).

**Tests:** All 186 tests pass (no new tests needed — existing CPD tests exercise the new code path).

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
