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
│       │   ├── io/           # TIFF I/O (load_multipage_tiff, load_image_stacks, save_stack)
│       │   ├── registration/ # Phase correlation registration (phase_correlate, apply_shift, demons_register)
│       │   ├── spotfinding/  # 3D spot detection (find_spots_3d)
│       │   ├── barcode/      # Encoding, decoding, codebook, extraction, filtering
│       │   ├── dataset/      # STARMapDataset + FOV orchestration layer
│       │   ├── benchmark/    # Performance measurement framework
│       │   └── testing/      # Synthetic dataset generator
│       └── test/          # pytest tests
├── workflow/
│   ├── Snakefile          # Main entry point (~58 lines)
│   ├── rules/             # Modular rule files (common, registration, spot-finding, etc.)
│   ├── schemas/           # JSON Schema for config validation
│   └── scripts/           # Python and MATLAB execution scripts
├── tests/
│   ├── fixtures/synthetic/ # Synthetic test datasets (mini, standard)
│   ├── qc_*.ipynb         # QC validation notebooks (io, registration, synthetic, benchmark)
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
# To get amplicon/molecular level information
In Situ Sequencing Raw Images (TIFF) → Registration → Spot Finding → Decoding → Filtering → Spot-level Matrix (row: spot/amplicon, col: gene label, x, y, z, other QC metrics ...)
Cell Morphology Raw Images (TIFF) → Segmentation (StarDist/Cellpose/others) → Reads Assignment → Cell Expression Matrix (H5AD)
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

- **`starfinder.registration`** - Image registration (global and local)
  - Global (rigid):
    - `phase_correlate(fixed, moving)` → `(dz, dy, dx)` detected shift
    - `apply_shift(volume, shift)` → shifted volume with edge zeroing
    - `register_volume(images, ref, mov)` → registered multi-channel volume + shifts
    - Note: `phase_correlate` returns detected displacement; apply **negative** to correct alignment
  - Local (non-rigid, requires SimpleITK):
    - `demons_register(fixed, moving, method, iterations, smoothing_sigma, pyramid_mode)` → displacement field (Z, Y, X, 3)
    - `apply_deformation(volume, field)` → warped volume
    - `register_volume_local(images, ref, mov)` → registered volume + displacement field
    - `matlab_compatible_config()` → dict matching MATLAB imregdemons defaults
    - Pyramid modes: `"sitk"` (naive subsampling), `"antialias"` (MATLAB-matching Butterworth)
  - Pyramid utilities (`starfinder.registration.pyramid`):
    - `butterworth_3d(shape, cutoff, order)` → 3D Butterworth low-pass filter
    - `antialias_resize(volume, factor)` → anti-aliased 3D resize
    - `pad_for_pyramiding(volume, levels)` / `crop_padding(volume, pad_widths)`
  - Quality metrics:
    - `normalized_cross_correlation(img1, img2)` → NCC [-1, 1]
    - `structural_similarity(img1, img2)` → SSIM [-1, 1]
    - `spot_colocalization(ref, img)` → IoU/Dice of bright spots
    - `spot_matching_accuracy(ref_spots, mov_spots)` → match rate
    - `registration_quality_report(ref, before, after)` → comprehensive report

- **`starfinder.benchmark`** - Performance measurement, data generation, and evaluation
  - `BenchmarkResult` dataclass, `measure()`, `@benchmark` decorator
  - `run_comparison()`, `BenchmarkSuite` for multi-method comparisons
  - `print_table()`, `save_csv()`, `save_json()` reporting
  - Benchmark data generation (`starfinder.benchmark.data`):
    - `create_benchmark_volume()` - Synthetic spot generation
    - `apply_global_shift()` - Zero-padded shifts (no wrap-around)
    - `create_deformation_field()` - Polynomial, Gaussian, multi-point deformations
    - `apply_deformation_field()` - Scipy-based warping
    - `generate_inspection_image()` - Green-magenta before/after overlays
    - `extract_real_benchmark_data()` - Round1/round2 MIP extraction
  - Benchmark evaluation (`starfinder.benchmark.evaluate`):
    - `evaluate_registration(ref, mov, registered)` - Compute all quality metrics
    - `evaluate_single(registered_path, data_dir)` - Evaluate one registered image from disk
    - `evaluate_directory(result_dir, data_dir)` - Batch-evaluate a backend tree
    - `generate_inspection(ref, mov, registered, metadata, path)` - 5-panel inspection PNG
    - CLI: `uv run python -m starfinder.benchmark.evaluate <result_dir> [--data-dir ...]`
    - Two-phase design: Phase 1 (run) saves registered images, Phase 2 (evaluate) computes metrics uniformly
  - Presets: tiny, small, medium, large, xlarge, tissue, thick_medium

- **`starfinder.spotfinding`** - 3D spot detection
  - `find_spots_3d(volume, method, ...)` → DataFrame with (z, y, x) coordinates
  - LoG filtering with adaptive or global thresholding
  - Multi-channel support

- **`starfinder.barcode`** - Barcode processing pipeline (encoding → codebook → filtering)
  - Encoding/decoding (`barcode.encoding`):
    - `BASE_PAIR_TO_COLOR` / `COLOR_TO_BASE_PAIRS` / `COLOR_TO_CHANNEL` — lookup tables
    - `encode_bases(sequence)` → color sequence (pure 2-base sliding window, no reversal)
    - `decode_color_seq(color_seq, start_base)` → DNA barcode (chain-tracking decoder)
  - Codebook (`barcode.codebook`):
    - `load_codebook(path, do_reverse=True, split_index=None)` → `(gene_to_seq, seq_to_gene)`
  - Extraction (`barcode.extraction`):
    - `extract_from_location(image, spots, voxel_size)` → `(color_seq, color_score)` arrays
  - Filtering (`barcode.filtering`):
    - `filter_reads(spots, seq_to_gene, end_bases=None, start_base="C")` → `(good_spots, stats)`
    - Filters by codebook membership only; end-base validation is diagnostic

- **`starfinder.preprocessing`** - Image enhancement (normalization, morphology)
  - `min_max_normalize(volume)` → per-channel [min, max] → [0, 255] rescaling (uint8)
  - `histogram_match(volume, reference, nbins=64)` → CDF-based histogram matching per channel
  - `morphological_reconstruction(volume, radius=3)` → background removal via opening-by-reconstruction
  - `tophat_filter(volume, radius=3)` → white tophat per Z-slice (removes large structures)
  - All functions handle (Z, Y, X) and (Z, Y, X, C) inputs

- **`starfinder.utils`** - General-purpose utilities
  - `make_projection(volume, method="max")` → Z-axis projection (max or sum with uint8 rescaling)

- **`starfinder.dataset`** - Dataset/FOV orchestration layer (wraps Phases 1-5)
  - Types: `LayerState`, `Codebook`, `CropWindow`, `SubtileConfig`, `Shift3D`, `ImageArray`, `ChannelOrder`
  - `STARMapDataset.from_config(config)` → sample-level config + FOV factory
  - `FOV` — per-FOV stateful processor with fluent API:
    - Loading: `load_raw_images()` → `io.load_image_stacks()`
    - Preprocessing: `enhance_contrast()`, `hist_equalize()`, `morph_recon()`, `tophat()`, `make_projection()`
    - Registration: `global_registration()`, `local_registration()`
    - Spot finding: `spot_finding()`, `reads_extraction()`, `reads_filtration()`
    - Output: `save_ref_merged()`, `save_signal()`, `create_subtiles()`, `from_subtile()`
  - `FOVPaths` — frozen path helper for output locations
  - `log_step` decorator — timing and error logging per method

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
- [x] Phase 2: Registration module (phase correlation, apply_shift, register_volume)
- [x] Phase 3: Spot finding & extraction (find_spots_3d, extract_from_location)
- [x] Phase 4: Barcode processing (encode/decode, codebook, filter_reads)
- [x] Phase 5: Preprocessing (min_max_normalize, histogram_match, morphological_reconstruction, tophat_filter, make_projection)
- [x] Phase 6: Dataset/FOV orchestration layer (STARMapDataset, FOV, fluent pipeline API)

## Notes for Claude Code
Update this file by adding tips whenever you make mistakes to help improve your accuracy.

### Development Philosophy
- Don't tend to over-engineer, be efficient and effective.
- Only write the minimum required tests.

### Tips
- The `~/wanglab` directory is a network mount. Use `Write` instead of `Edit` tool to avoid false "file modified" errors.
- Development notes are in `docs/notes.md`.
- Array axis convention: `(Z, Y, X, C)` for Python, matches ITK/SimpleITK.
- CSV coordinates: 1-based for MATLAB compatibility (Python uses 0-based internally).
- Run Python with `uv run python`
- Always ask before using `git push`
- When creating a commit message, review the previous git history. Use numbered messages for new modules or major changes; otherwise, use a prefix(addtional info if needed): message. 
- **Registration sign convention**: `phase_correlate()` returns detected displacement (how much `moving` differs from `fixed`). To correct alignment, apply the **negative** shift: `correction = tuple(-s for s in detected_shift)`
- **Benchmark shift ranges**: When testing registration, ensure shift ranges are proportional to volume size (≤25% of each dimension) to maintain sufficient image overlap for phase correlation.
- **Demons registration axis ordering**: SimpleITK uses (dx, dy, dz) for displacement vectors, NumPy uses (dz, dy, dx). The `demons.py` module handles this conversion internally.
- **Demons defaults**: `demons_register()` defaults to Thirion demons with 3-level anti-aliased pyramid (`method="demons"`, `iterations=[100,50,25]`, `sigma=1.0`, `pyramid_mode="antialias"`). This matches MATLAB's `imregdemons` quality while being 1.6x faster with identical memory usage. The old single-level defaults (`iterations=[50]`, `sigma=0.5`, `pyramid_mode="sitk"`) are still available for quick tests.
- **Anti-aliased pyramid mode**: `pyramid_mode="antialias"` uses Butterworth-filtered downsampling matching MATLAB's `imregdemons` internal `antialiasResize`. The old `"sitk"` mode does naive subsampling that destroys sparse spots at coarse levels — avoid for multi-level pyramids.
- **Python vs MATLAB local registration benchmark (2026-02-11)**: With matched settings ([100,50,25], sigma/AFS=1.0), Python `py_diffeo` beats MATLAB on NCC (+0.115) and Match Rate (+0.022). Python `py_demons` (Thirion) is 1.6x faster. Results at `local_comparison/`.
- **Registration quality metrics**: For sparse fluorescence images, use spot-based metrics (Spot IoU, Match Rate) instead of MAE. MAE is dominated by background pixels (99% of image) and doesn't reflect spot alignment quality.
- **Benchmark data generation**:
  - Use `apply_global_shift()` with zero-padding, NOT `np.roll()` which wraps around
  - Use fixed pixel margins (5px) for spot placement, not percentage-based (causes blank bands on large images)
  - Use preset-specific random seeds to ensure different shifts for each preset
  - Exclude 0 from Z-shift options to ensure non-zero Z displacements
  - Cap deformation magnitudes at fixed pixels (15/30px) for large images to avoid excessive warping
- **Benchmark data location**: `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/` (~50GB total)
  - Input data: `data/synthetic/` (7 presets) and `data/real/` (3 datasets)
  - Registration results: `results/registration/` (global_python, global_matlab, local_tuning, local_matlab, local_python, local_antialias, global_comparison, local_comparison, figures, scripts)
- **Two-phase benchmark workflow**: Phase 1 saves `registered_{backend}.tif` + `run_{backend}.json`; Phase 2 (`evaluate.py`) computes `metrics_{backend}.json` + `inspection_{backend}.png` uniformly for all backends
- **FOV directory layout**: `FOV.input_dir()` returns `{input_root}/{round}/{fov_id}/`. The synthetic mini dataset uses `{base}/{fov}/{round}/` — tests create symlinks to restructure.
- **FOV fluent API**: All processing methods return `self` for chaining. State lives in `fov.images`, `fov.global_shifts`, `fov.all_spots`, `fov.good_spots`. Config is delegated to `fov.dataset`.
- **SubtileConfig.compute_windows()**: Uses `height // sqrt_pieces` for tile size (MATLAB `dims(1)`). Overlap extends inward only — no overlap on outer edges. 0-based internally; `create_subtiles()` converts to 1-based for `subtile_coords.csv`.
- **MIP fast path for large volumes**: For volumes >100M voxels, `evaluate_registration(use_mip=True)` computes SSIM and spot metrics (detect_spots, spot_colocalization, spot_matching_accuracy) on 2D MIP instead of full 3D. Output includes `"ssim_method"` and `"spot_method"` fields ("mip" or "3d") to indicate which path was used. `evaluate_directory(use_mip_above=)` controls the threshold. CLI flag: `--use-mip-above`.

