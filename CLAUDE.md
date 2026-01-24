# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STARfinder is a spatial transcriptomics data processing pipeline for STARmap-related methods. It's a hybrid MATLAB/Python/Snakemake workflow that processes large-scale image datasets from raw microscopy images to cell-by-gene expression matrices.

**Technology Stack:**
- Orchestration: Snakemake 9.x (recently upgraded from v7.32.4)
- Image processing: MATLAB 2023b+ (core algorithms in `code-base/src/`)
- Post-processing: Python 3.9+ (reads assignment, segmentation, analysis)
- Cluster execution: Broad UGER with cluster-generic executor plugin

## Common Commands

```bash
# Create conda environment
conda env create -f ./config/environment-v9.yaml  # Snakemake 9.x

# Dry run (validate workflow without executing)
snakemake -s workflow/Snakefile --configfile test/tissue_2D_test.yaml -n

# Run with Broad UGER cluster
snakemake -s workflow/Snakefile --configfile test/tissue_2D_test.yaml \
  --profile profile/broad-uger --workflow-profile profile/broad-uger

# Check DAG
snakemake -s workflow/Snakefile --configfile test/tissue_2D_test.yaml --dag | dot -Tpng > dag.png

# Lint check
snakemake -s workflow/Snakefile --configfile test/tissue_2D_test.yaml --lint
```

## Architecture

### Directory Structure

- `code-base/src/` - Core MATLAB image processing scripts (~27 files). Main orchestrator is `STARMapDataset.m`.
- `workflow/` - Snakemake orchestration
  - `Snakefile` - Main entry point (~58 lines), includes modular rule files and config validation
  - `rules/` - Modularized rule files (common.smk, registration.smk, spot-finding.smk, segmentation.smk, stitching.smk, reads-assignment.smk, utils.smk)
  - `schemas/` - JSON Schema for config validation (`config.schema.yaml`)
  - `scripts/` - Python and MATLAB execution scripts called by rules
- `config/` - Conda environment definitions and configuration templates
- `profile/broad-uger/` - UGER cluster execution profile
- `test/` - Test configurations (`tissue_2D_test.yaml`, `minimal_config.yaml`)

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

MATLAB scripts are called via Python subprocess in Snakemake rules. The `run_matlab_scripts()` function sources the Broad environment and MATLAB module before execution. MATLAB addons for large TIFF handling are in `code-base/matlab-addon/`.

## Configuration System

Config files are validated against a JSON Schema at pipeline startup (`workflow/schemas/config.schema.yaml`).

**Required top-level keys:**
- Paths: `config_path`, `starfinder_path`, `root_input_path`, `root_output_path`
- Dataset metadata: `dataset_id`, `sample_id`, `output_id`, `fov_id_pattern`, `n_fovs`, `n_rounds`, `ref_round`, `rotate_angle`, `img_col`, `img_row`
- `workflow_mode`: 'free', 'direct', 'subtile', or 'deep'
- `rules`: Per-rule configuration with `run`, `resources`, and `parameters` sections

**Config templates:**
- Full example: `test/tissue_2D_test.yaml`
- Minimal template: `test/minimal_config.yaml`

## Test Datasets

Located on Zenodo (DOI: 10.5281/zenodo.11176779):
1. **cell-culture-3D** - 70 FOVs, 6 sequencing rounds, 3D HeLa cell culture
2. **tissue-2D** - 56 tiles, 4 sequencing rounds, mouse brain tissue section

## Current Development Status

See `dev/current_plan.md` for detailed status. Summary:
- Phase 1 (Modularization): COMPLETED
- Phase 2 (Snakemake 9 Upgrade): MOSTLY COMPLETED
- Phase 3 (Code Quality): MOSTLY COMPLETED
  - Config schema validation implemented
  - Workflow mode dependency validation added

## Notes for Claude Code

- The `~/wanglab` directory is a network mount. Use `Write` instead of `Edit` tool to avoid false "file modified" errors.
- Development notes are in `dev/notes.md` and `dev/current_plan.md`.
