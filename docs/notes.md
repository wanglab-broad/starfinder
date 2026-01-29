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
  - [] Adopt new data structure such as h5 and OME-Zarr, but also ensure backward compatibility 
  - [] Adopt high-performance image registration methods
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
