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
└── docs/                  # Maintained software documentation
```

### Python Backend

Uses `(Z, Y, X, C)` axis ordering (volumetric-first, channel-last). Replaces MATLAB components.

#### Module Map

| Module | Description |
|--------|-------------|
| `io` | TIFF I/O with bioio backend |
| `registration` | Global (phase correlation, apply_shift) + local (demons, TPS, CPD) |
| `spot_finding` | 3D spot detection with noise/adaptive/global thresholding |
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
uv run starfinder synthetic generate --mode e2e --owner Jiahao --preset small --seed 42 --output ../../tests/fixtures/synthetic/small  # Generate synthetic data
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

### Project coordination

- Use Linear under the `PhD Thesis` initiative. [starfinder: Chapter II evidence and benchmarks](https://linear.app/jiahaoh/project/starfinder-chapter-ii-evidence-and-benchmarks-6f9ba2048496) owns W-92–W-94. [starfinder: review and plan](https://linear.app/jiahaoh/project/starfinder-review-and-plan-71b35aa5e5e3) retains review, coordination, and historical-record migration (W-95). [Chapter II](https://linear.app/jiahaoh/project/thesis-chapter-ii-starfinder-ef5247cb257d) owns scientific scope and acceptance; this repository owns implementation and benchmark execution.
- Follow the living [initiative-wide workflow and handoff conventions](https://linear.app/jiahaoh/document/thesis-and-implementation-workflow-c12f30bffe9f), a PhD Thesis resource for all chapters, development projects, repositories, and hosts. Read the current version before a new handoff. Reuse existing issues, assign them to Jiahao (`assignee: "me"`), and use the relevant project milestone. Record the executing agent/session in a comment.
- Update the canonical workflow in place as agreed practices evolve. Track substantive changes in a matching administrative issue, add dated rationale to its change log, and synchronize affected repository instructions and live issue templates. Preserve its URL; keep project-specific execution details here or in the project, and label unsettled proposals explicitly.
- Before work, read the issue and dependencies, inspect Git state, and verify the repository, starting revision, data, environment, resource limits, and acceptance criteria. Unknown inputs remain explicitly unverified. Preserve unrelated edits; use separate branches/worktrees for concurrent agents.
- Propose scope changes in the linked chapter discussion. Do not turn historical plan checklists into new authorized work automatically.
- Record progress at meaningful decisions, blockers, handoffs, and completion. The completion summary must identify code/PR and commit/push/merge state, exact commands/configuration/data/environment, validation actually performed, artifacts, limitations, and follow-ups. If code is uncommitted, record that and preserve an identifiable patch/snapshot.
- Set an implementation issue to `Done` only after its acceptance checklist and evidence are complete; verify assignee, milestone, status, and milestone progress. Chapter evidence/writing issues remain open until the thesis agent assesses the results and incorporates accepted reasoning and figure provenance. Null results can satisfy a benchmark task.
- Current status and validation evidence live in Linear. [Historical plans and results](https://linear.app/jiahaoh/document/historical-record-index-and-migration-provenance-881cc5edc52b) preserve old claims with their dates and provenance; they do not establish current completion or correctness.
- The authorized autonomous documentation pilot uses one active worker per project and a fresh CLI session per issue, with bounded same-issue repairs. Carry accepted code forward on the dedicated project branch; record session ID, host/worktree, revisions, commands, validation, and stop/resume state in Linear. Sidebar visibility is optional. Successful agent exit alone is not acceptance; preserve blockers and leave unresolved issues open. Follow the linked workflow and the current issue's implementation/validation handoff instructions.

### Plans and artifact storage

- Put short plans, scope, and acceptance criteria in the Linear issue description; progress, decisions, and result summaries in comments; substantial plans/reports in linked Linear project documents.
- Do not commit planning-mode files, prompt histories, development diaries, or run-specific benchmark reports. Planning tools may use ignored `.agent-work/` files or their own temporary directory, but transfer the execution-ready plan to Linear before handoff. If Linear is unavailable, preserve temporary notes and sync them before declaring completion.
- Keep maintained installation/API/usage/architecture documentation, benchmark runners, reusable configurations, evaluation code, and small fixtures in Git. Existing examples and Python package documentation remain the software entry points; `docs/` is for maintained documentation.
- Store new run outputs outside the checkout at `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/<issue-id>/<run-id>/`. Use unique run IDs and retain existing run locations. Put metrics, logs, figures, and a manifest together.
- Each manifest records issue URL, run ID/date/host, code commit and dirty-state/patch identity, dataset source/version/checksums, command/configuration/environment, seeds and hardware/resources, validation/exit status, output paths/checksums, limitations, owner, retention, and backup status. Unknown historical values are explicitly unverified.
- Jiahao owns retention decisions. Retain cited evidence through thesis completion and related publication/release; verify a replacement before deleting it. Institutional backup coverage is unverified unless separately confirmed. Private server paths alone are insufficient for public release reproducibility.
- Return accepted scientific interpretation, figure provenance, and manuscript changes to the thesis repository. Linear is the shared handoff; software Git is not the scientific execution diary.
- Before removing legacy records, preserve exact bytes and Git-state metadata outside the checkout, verify the Linear copy, and repair references. Do not rewrite Git history.

### Guidance for coding agents

#### Development Philosophy
- Don't over-engineer — be efficient and effective.
- Only write the minimum required tests.
- Only change what was explicitly requested — no unsolicited modifications to parameters, counts, or values beyond the task scope.
- Verify diagnosis against actual code before stating root causes — read the relevant code first, don't guess.

#### Environment & Workflow
- Read and update the linked Linear issue as described above; use its checklist and status for completion.
- After implementing changes, run `uv run pytest test/ -v` and report results before committing.
- Run Python with `uv run python` (from `src/python/`)
- The `~/wanglab` directory is a network mount. Use `Write` instead of `Edit` tool to avoid false "file modified" errors.
- Commit messages: review git history. Use numbered messages for new modules/major changes; otherwise use `prefix(info): message`.

#### Key Conventions
- Array axis convention: `(Z, Y, X, C)` for Python, matches ITK/SimpleITK
- CSV coordinates: 1-based for MATLAB compatibility (Python uses 0-based internally)
- **Registration sign convention**: `phase_correlate()` returns detected displacement. To correct alignment, apply the **negative** shift.
- **MATLAB channel order is wavelength-sorted**: `["ch00", "ch02", "ch01", "ch03"]` — ch01 and ch02 are swapped. All three real datasets use this default.
- **Spot finding modes**: `"noise"` (default, k-sigma MAD) for synthetic data; `"adaptive"` (fraction-of-max) for real data. Always pass `intensity_estimation` and `intensity_threshold` together.
