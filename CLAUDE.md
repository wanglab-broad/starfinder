# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STARfinder is a spatial transcriptomics data processing pipeline for STARmap-related methods. It's a hybrid MATLAB/Python/Snakemake workflow that processes large-scale image datasets from raw microscopy images to cell-by-gene expression matrices.

**Technology Stack:**
- Orchestration: Snakemake 9.x (recently upgraded from v7.32.4)
- Image processing: MATLAB 2023b+ (core algorithms in `src/matlab/`)
- Python backend: Python 3.10+ with uv (I/O, registration, processing modules in `src/python/`)
- Post-processing: Python 3.9+ (reads assignment, segmentation, analysis)
- Cluster execution: Broad UGER with cluster-generic executor plugin

## Common Commands

### Snakemake Workflow
```bash
# Create conda environment
conda env create -f ./config/environment-v9.yaml  # Snakemake 9.x

# Dry run (validate workflow without executing)
snakemake -s workflow/Snakefile --configfile tests/tissue_2D_test.yaml -n

# Run with Broad UGER cluster
snakemake -s workflow/Snakefile --configfile tests/tissue_2D_test.yaml \
  --profile profile/broad-uger --workflow-profile profile/broad-uger

# Check DAG
snakemake -s workflow/Snakefile --configfile tests/tissue_2D_test.yaml --dag | dot -Tpng > dag.png

# Lint check
snakemake -s workflow/Snakefile --configfile tests/tissue_2D_test.yaml --lint
```

### Python Package (uv)
```bash
cd src/python

# Install dependencies
uv sync

# Run tests
uv run pytest test/ -v

# Run tests with coverage
uv run pytest test/ -v --cov=starfinder

# Generate synthetic test dataset
uv run python -m starfinder.testing --preset mini --output ../../tests/fixtures/synthetic/mini
```

## Architecture

### Directory Structure

```
starfinder/
├── src/
│   ├── matlab/            # Core MATLAB scripts (~29 files). Main: STARMapDataset.m
│   ├── matlab-addon/      # External MATLAB toolboxes (TIFF handling, natural sort)
│   └── python/            # Python package (starfinder)
│       ├── starfinder/
│       │   ├── io/        # TIFF I/O (load_multipage_tiff, load_image_stacks, save_stack)
│       │   └── testing/   # Synthetic dataset generator
│       └── test/          # pytest tests
├── workflow/
│   ├── Snakefile          # Main entry point (~58 lines)
│   ├── rules/             # Modular rule files (common, registration, spot-finding, etc.)
│   ├── schemas/           # JSON Schema for config validation
│   └── scripts/           # Python and MATLAB execution scripts
├── tests/
│   ├── fixtures/synthetic/ # Synthetic test datasets (mini, standard)
│   ├── tissue_2D_test.yaml
│   └── minimal_config.yaml
├── config/                # Conda environment definitions
├── profile/broad-uger/    # UGER cluster execution profile
└── docs/                  # Design documents and development notes
```

### Workflow Modes

The pipeline supports 4 workflow modes configured via `workflow_mode` in the config YAML:

1. **`direct`** - Single FOV direct processing via `rsf_single_fov`
2. **`subtile`** - Subtile-based workflow: `gr_single_fov_subtile` → `lrsf_single_fov_subtile` → `stitch_subtile`
3. **`deep`** - Deep-tissue optimized: `deep_create_subtile` → `deep_rsf_subtile` → `stitch_subtile`
4. **`free`** - Mix-and-match rules using individual `run: True/False` flags per rule

### Key Helper Functions (in `workflow/rules/common.smk`)

- `is_rule_enabled(rule_name)` - Check if rule should run based on workflow_mode
- `get_rule_config(rule_name, key, default)` - Safe nested config access
- `make_get_runtime(rule_name)` - Factory for rule-specific runtime functions
- `run_matlab_scripts(param_string, script_name)` - Execute MATLAB in subprocess with proper environment
- `validate_workflow_mode_dependencies()` - Validate preset modes have required rule configs

### Data Flow

```
Raw Images (TIFF) → Registration → Spot Finding → Decoding →
Segmentation (StarDist) → Reads Assignment → Cell Expression Matrix (H5AD)
```

### MATLAB Integration

MATLAB scripts are called via Python subprocess in Snakemake rules. The `run_matlab_scripts()` function sources the Broad environment and MATLAB module before execution. MATLAB addons for large TIFF handling are in `src/matlab-addon/`.

## Python Package (starfinder)

The Python backend is being developed to replace MATLAB components. Uses `(Z, Y, X, C)` axis ordering (volumetric-first, channel-last).

### Implemented Modules

- **`starfinder.io`** - TIFF I/O with bioio backend
  - `load_multipage_tiff(path, convert_uint8=True)` → `(Z, Y, X)` array
  - `load_image_stacks(round_dir, channel_order)` → `(Z, Y, X, C)` array
  - `save_stack(image, path, compress=False)`

- **`starfinder.testing`** - Synthetic dataset generation
  - Two-base color-space encoding matching MATLAB
  - Presets: `mini` (1 FOV, 256×256×5) and `standard` (4 FOVs, 512×512×10)

### Dependencies
```toml
# Core
numpy, scipy, scikit-image, tifffile, pandas, h5py, bioio, bioio-tifffile

# Optional
SimpleITK      # local-registration
spatialdata    # modern output format
```

## Configuration System

Config files are validated against a JSON Schema at pipeline startup (`workflow/schemas/config.schema.yaml`).

**Required top-level keys:**
- Paths: `config_path`, `starfinder_path`, `root_input_path`, `root_output_path`
- Dataset metadata: `dataset_id`, `sample_id`, `output_id`, `fov_id_pattern`, `n_fovs`, `n_rounds`, `ref_round`, `rotate_angle`, `img_col`, `img_row`
- `workflow_mode`: 'free', 'direct', 'subtile', or 'deep'
- `rules`: Per-rule configuration with `run`, `resources`, and `parameters` sections

**Config templates:**
- Full example: `tests/tissue_2D_test.yaml`
- Minimal template: `tests/minimal_config.yaml`

## Test Datasets

### Real Datasets (Zenodo DOI: 10.5281/zenodo.11176779)
1. **cell-culture-3D** - 70 FOVs, 6 sequencing rounds, 3D HeLa cell culture (1496×1496×30)
2. **tissue-2D** - 56 tiles, 4 sequencing rounds, mouse brain tissue section (3072×3072×30)

### Synthetic Datasets (tests/fixtures/synthetic/)
- **mini/** - 1 FOV, 256×256×5, 20 spots (unit tests, CI)
- **standard/** - 4 FOVs, 512×512×10, 100 spots/FOV (integration tests)

## Documentation

Detailed design documents are in `docs/`:

| Document | Description |
|----------|-------------|
| `notes.md` | Development notes, progress tracking, future directions |
| `test_design.md` | Test strategy for Python backend (unit, contract, integration, golden tests) |
| `main_python_object_design.md` | Python STARMapDataset/FOV class design |
| `plan_milestone_1.md` | Snakemake modularization & upgrade plan (COMPLETED) |
| `plan_milestone_2.md` | Python backend migration plan |
| `DFT_REGISTRATION_REVIEW.md` | DFT registration algorithm review with Python examples |
| `plans/` | Implementation plans with step-by-step tasks |

## Current Development Status

### Milestone 1 (Modularization & Snakemake 9): COMPLETED ✓
- Modularized Snakefile from ~566 lines to ~58 lines
- Upgraded to Snakemake 9.x with executor plugin system
- Implemented config schema validation
- Implemented workflow mode system

### Milestone 2 (Python Backend): IN PROGRESS
- [x] Phase 0: Directory restructure (`code-base/` → `src/matlab/`)
- [x] Phase 1: I/O module with bioio backend
- [ ] Phase 2: Registration module (phase correlation, demons)
- [ ] Phase 3: Spot finding & extraction
- [ ] Phase 4: Barcode processing
- [ ] Phase 5: Preprocessing
- [ ] Phase 6: Dataset class & Snakemake integration

## Notes for Claude Code

### Development Philosophy
- Don't tend to over-engineer, be efficient and effective.
- Only write the minimum required tests.

### Tips 
- The `~/wanglab` directory is a network mount. Use `Write` instead of `Edit` tool to avoid false "file modified" errors.
- Development notes are in `docs/notes.md`.
- Array axis convention: `(Z, Y, X, C)` for Python, matches ITK/SimpleITK.
- CSV coordinates: 1-based for MATLAB compatibility (Python uses 0-based internally).
- run python with `uv run python`
- always ask before using `git push`

