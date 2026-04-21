# AGENTS.md

This file provides guidance to Coding Agents (i.e., Claude Code, Codex, Gemini) when working with code in this repository.

## 1. Project Overview

STARfinder is a spatial transcriptomics data processing pipeline for STARmap-related methods. It's a hybrid MATLAB/Python/Snakemake workflow that processes large-scale image datasets from raw microscopy images to cell-by-gene expression matrices.

### Technology Stack

- Orchestration: Snakemake 9.x
- Image processing:
  - MATLAB backend: MATLAB 2023b+ (core algorithms in `src/matlab/`)
  - Python backend: Python 3.10+ with uv (I/O, registration, processing in `src/python/`)
- Post-processing: Python 3.9+ (reads assignment, segmentation, analysis)
- Cluster execution: Broad UGER with cluster-generic executor plugin

## 2. Codebase

### Pipeline Order

```
load → rotate → enhance → registration → spot_finding → extraction → filtration
```

### Directory Structure

```
starfinder/
├── src/
│   ├── matlab/            # MATLAB backend (~28 scripts). Main: STARMapDataset.m
│   ├── matlab-addon/      # External MATLAB toolboxes (TIFF handling, natural sort)
│   └── python/            # Python package (starfinder)
│       ├── starfinder/    # Source modules (see Python Backend below)
│       └── test/          # pytest tests
├── workflow/
│   ├── Snakefile          # Main entry point (~58 lines)
│   ├── rules/             # Modular rule files (common, registration, spot-finding, etc.)
│   ├── schemas/           # JSON Schema for config validation
│   └── scripts/           # Python and MATLAB execution scripts
├── tests/
│   ├── fixtures/synthetic/ # Synthetic test datasets (small, medium)
│   ├── qc_*.ipynb         # QC validation notebooks
│   ├── tissue_2D_test.yaml
│   └── minimal_config.yaml
├── config/                # Conda environment definitions
├── profile/broad-uger/    # UGER cluster execution profile
└── docs/                  # Design documents, plans, and development notes
```

### Python Backend

Uses `(Z, Y, X, C)` axis ordering (volumetric-first, channel-last). Replaces MATLAB components.

#### Module Map

| Module | Description |
|--------|-------------|
| `io` | TIFF I/O with bioio backend |
| `registration` | Global (phase correlation, apply_shift) + local (demons, TPS, CPD) |
| `spotfinding` | 3D spot detection with noise/adaptive/global thresholding |
| `barcode` | Encode/decode, codebook, extraction, filtering pipeline |
| `preprocessing` | min_max_normalize, histogram_match, morphological_reconstruction, tophat |
| `dataset` | STARMapDataset + FOV orchestration (fluent API, streaming mode) |
| `benchmark` | Measurement framework, synthetic data generation, evaluation |
| `benchmark.synthetic` | Coordinate-first rendering, presets: tiny/small/medium/large/tissue/thick_medium |

Dependencies: numpy, scipy, scikit-image, tifffile, pandas, h5py, bioio, bioio-tifffile. Optional: SimpleITK (local registration), spatialdata (modern output).

#### Common Commands

```bash
cd src/python

uv sync                                    # Install dependencies
uv run pytest test/ -v                     # Run tests
uv run pytest test/ -v --cov=starfinder    # Run tests with coverage
uv run python -m starfinder.benchmark --preset small --output ../../tests/fixtures/synthetic/small  # Generate synthetic data
```

### MATLAB Backend

28 scripts in `src/matlab/`. Main entry point: `STARMapDataset.m`. Addons in `src/matlab-addon/`.

#### Module Map

| Script | Function |
|--------|----------|
| `STARMapDataset.m` | Main orchestrator (dataset config, pipeline coordination) |
| `LoadImageStacks.m` / `LoadMultipageTiff.m` | TIFF I/O |
| `DFTRegister3D.m` / `DFTApply3D.m` | Global registration (phase correlation) |
| `RegisterImagesGlobal.m` / `RegisterImagesLocal.m` | Registration orchestration |
| `SpotFindingMax3D.m` | 3D spot detection |
| `ExtractFromLocation.m` | Barcode extraction |
| `EncodeBases.m` / `DecodeCS.m` / `Str2Colorseq.m` | Barcode encoding/decoding |
| `LoadCodebook.m` / `FilterReads.m` | Codebook loading and read filtering |
| `MinMaxNorm.m` / `MorphologicalReconstruction.m` | Preprocessing |
| `MakeProjections.m` / `MakeMontage.m` | Visualization |

#### Common Commands

MATLAB scripts are called via Python subprocess in Snakemake rules. The `run_matlab_scripts()` function in `workflow/rules/common.smk` sources the Broad environment and MATLAB module before execution. No standalone CLI — always invoked through Snakemake.

### Snakemake Orchestration

#### Workflow Commands

```bash
conda env create -f ./config/environment-v9.yaml                              # Create environment
snakemake -s workflow/Snakefile --configfile tests/tissue_2D_test.yaml -n      # Dry run
snakemake -s workflow/Snakefile --configfile tests/tissue_2D_test.yaml \
  --profile profile/broad-uger --workflow-profile profile/broad-uger          # Run on UGER
snakemake -s workflow/Snakefile --configfile tests/tissue_2D_test.yaml --dag | dot -Tpng > dag.png  # DAG
snakemake -s workflow/Snakefile --configfile tests/tissue_2D_test.yaml --lint  # Lint
```

#### Configuration System

Config validated against JSON Schema at startup (`workflow/schemas/config.schema.yaml`).

**Required top-level keys:**
- Paths: `config_path`, `starfinder_path`, `root_input_path`, `root_output_path`
- Dataset metadata: `dataset_id`, `sample_id`, `output_id`, `fov_id_pattern`, `n_fovs`, `n_rounds`, `ref_round`, `rotate_angle`, `img_col`, `img_row`
- `workflow_mode`: 'free', 'direct', 'subtile', or 'deep'
- `rules`: Per-rule configuration with `run`, `resources`, and `parameters` sections

**Config templates:** `tests/tissue_2D_test.yaml` (full), `tests/minimal_config.yaml` (minimal)

## 3. Dataset

### Data Flow

```
Raw Images (TIFF) → Registration → Spot Finding → Decoding → Filtering → Spot-level Matrix
Cell Morphology (TIFF) → Segmentation → Reads Assignment → Cell Expression Matrix (H5AD)
```

### Sequencing Benchmark Datasets

#### Real Datasets (Zenodo DOI: 10.5281/zenodo.11176779)

| Dataset | FOVs | Rounds | Ref | Dimensions | Genes | Voxel Size | Params |
|---------|------|--------|-----|------------|-------|------------|--------|
| cell-culture-3D | 70 (Pos351-420) | 6 | round1 | 1496×1496×30 | 998 | (1,2,2) | end="CC", adaptive@0.2 |
| tissue-2D | 56 tiles | 4 | round1 | 3072×3072×30 | 64 | (1,1,1) | end="CC", adaptive@0.4 |
| LN | 64 (Pos001-064) | 4 | round4 | 1496×1496×50 | 61 | (1,1,1) | end="AC", start="A", adaptive@0.2 |
| aging | 848 (Pos400-; 6 in sample) | 9 | round1 | 2048×2048×36 | 2044 | (0.35,0.14,0.14) | 2-seg barcodes, split_index=5, end_base=["CC","TT"] (seg1/seg2), adaptive@0.2 |

#### Synthetic Datasets (`tests/fixtures/synthetic/`)

| Preset | Dimensions | Genes | Spots/FOV | Purpose |
|--------|-----------|-------|-----------|---------|
| small | 256×256×16 | 8 | 50 | Unit tests, CI |
| medium | 512×512×32 | 8 | 100 | Integration tests |
| large | 1024×1024×30 | 64 | 2000 | E2E benchmark |
| tissue | 3072×3072×30 | 64 | 14000 | E2E benchmark |
| thick_medium | 1024×1024×100 | 64 | 5200 | E2E benchmark |

All synthetic presets: 2 FOVs each. Benchmark presets at `starfinder_benchmark/e2e/data/{preset}/`.

### Registration Benchmark Datasets

At `starfinder_benchmark/registration/data/`:
- **Synthetic** (`synthetic/`): 6 presets — tiny, small, medium, large, thick_medium, tissue. Single-channel ref/mov pairs with known shifts + deformations.
- **Real** (`real/`): 3 datasets — cell_culture_3D, tissue_2D, LN. Round1/round2 MIP extractions from real data.

### Benchmark Results Location

All benchmarks at `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/{module}/{data/results}`:

| Module | Contents |
|-----------|----------|
| `registration` | Global/local registration comparisons (Python vs MATLAB), CPD, TPS |
| `e2e` | E2E pipeline results |
| `spot_finding` | Spot finding benchmark results |

## 4. Development

### Current Status

Milestone 1 (Snakemake 9 modularization): COMPLETED. Milestone 2 (Python backend): Phases 0-9 DONE — I/O, registration, spot finding, barcode, preprocessing, dataset/FOV orchestration, E2E validation, real data benchmarks, performance optimization (streaming + 50% RSS reduction).

Detailed docs in `docs/` and `docs/plans/`. Development notes in `docs/notes.md`.

### Notes for Claude Code

#### Development Philosophy
- Don't over-engineer — be efficient and effective.
- Only write the minimum required tests.
- Only change what was explicitly requested — no unsolicited modifications to parameters, counts, or values beyond the task scope.
- Verify diagnosis against actual code before stating root causes — read the relevant code first, don't guess.

#### Environment & Workflow
- Always save the proposed plan in `docs/plans/`, each plan should have **Date:** and **Status:** properties at the beginning.  
- If you finish implementing a plan from `docs/plans/`, mark its **Status** as "FINISHED" in the corresponding plan document.
- After implementing changes, run `uv run pytest test/ -v` and report results before committing.
- Run Python with `uv run python` (from `src/python/`)
- The `~/wanglab` directory is a network mount. Use `Write` instead of `Edit` tool to avoid false "file modified" errors.
- Always ask before using `git push`
- Commit messages: review git history. Use numbered messages for new modules/major changes; otherwise use `prefix(info): message`.

#### Key Conventions
- Array axis convention: `(Z, Y, X, C)` for Python, matches ITK/SimpleITK
- CSV coordinates: 1-based for MATLAB compatibility (Python uses 0-based internally)
- **Registration sign convention**: `phase_correlate()` returns detected displacement. To correct alignment, apply the **negative** shift.
- **MATLAB channel order is wavelength-sorted**: `["ch00", "ch02", "ch01", "ch03"]` — ch01 and ch02 are swapped. All three real datasets use this default.
- **Spot finding modes**: `"noise"` (default, k-sigma MAD) for synthetic data; `"adaptive"` (fraction-of-max) for real data. Always pass `intensity_estimation` and `intensity_threshold` together.
