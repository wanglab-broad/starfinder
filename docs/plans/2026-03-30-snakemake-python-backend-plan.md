# Snakemake Python Backend Integration Plan

**Date:** 2026-03-30
**Status:** FINISHED

## Goals

1. **Feature parity**: Python Snakemake rules handle the same tasks as MATLAB (direct, subtile, deep modes)
2. **Side-by-side**: Keep MATLAB rules untouched for A/B comparison
3. **Leverage Python advantages**: Streaming mode, TPS/CPD local registration, noise-floor spot finding, QC logging, NPZ intermediates

## Architecture

### Backend Selector

Add a top-level config key `backend: "python" | "matlab"` (default: `"matlab"` for backward compatibility). The Snakefile conditionally includes either MATLAB or Python rule files — only one set of core rules is active per run. Downstream rules (segmentation, stitching, reads-assignment) are backend-agnostic and remain unchanged.

```
workflow/
├── rules/
│   ├── common.smk                  # Shared (unchanged)
│   ├── registration.smk            # MATLAB rules (unchanged)
│   ├── spot-finding.smk            # MATLAB rules (unchanged)
│   ├── registration-py.smk         # NEW: Python registration rules
│   ├── spot-finding-py.smk         # NEW: Python spot-finding rules
│   ├── segmentation.smk            # Unchanged (already Python)
│   ├── stitching.smk               # Unchanged
│   ├── reads-assignment.smk        # Unchanged
│   └── utils.smk                   # Unchanged
├── scripts/
│   ├── rsf_single_fov.py           # NEW: Python direct-mode script
│   ├── gr_single_fov_subtile.py    # NEW: Python global-reg + subtile script
│   ├── lrsf_single_fov_subtile.py  # NEW: Python local-reg + spot-finding script
│   ├── deep_create_subtile.py      # NEW: Python deep-tissue subtile script
│   ├── deep_rsf_subtile.py         # NEW: Python deep-tissue RSF script
│   ├── rsf_single_fov.m            # Unchanged
│   ├── ...                         # All MATLAB scripts unchanged
```

Note: Python scripts (`.py`) and MATLAB scripts (`.m`) share the same base names — the file extension distinguishes them.

### Output Compatibility

Python rules produce **identical output file paths and formats** as MATLAB rules so downstream rules (stitch_subtile, reads_assignment, etc.) work with either backend:

| Output | MATLAB | Python | Notes |
|--------|--------|--------|-------|
| `signal/{fov}_goodSpots.csv` | 1-based (x,y,z,gene,...) | 1-based (x,y,z,gene,...) | FOV.save_signal already converts |
| `images/ref_merged/{fov}.tif` | 3D/2D TIFF | 3D/2D TIFF | FOV.save_ref_merged matches |
| `log/sf_scores/{fov}.txt` | MATLAB format | Python text log | Format may differ, acceptable |
| `log/gr_shifts/{fov}.txt` | CSV shifts | CSV shifts | FOV already saves this |
| `log/{fov}_rsf.txt` | MATLAB diary | Python log | Format differs, acceptable |
| subtile intermediates | `.mat` files | `.npz` files | Different format (see Phase 3) |

### Conditional Include Strategy

```python
# In Snakefile:
BACKEND = config.get("backend", "matlab")

if BACKEND == "python":
    include: "rules/registration-py.smk"
    include: "rules/spot-finding-py.smk"
elif BACKEND == "matlab":
    include: "rules/registration.smk"
    include: "rules/spot-finding.smk"
```

This is clean: no ruleorder conflicts, no ambiguous rules, no conditionals inside rule files.

---

## Phase 1: Infrastructure

**Goal:** Backend selector, config schema update.

### 1.1 Config Schema Update (`config.schema.yaml`)

Add:
```yaml
backend:
  type: string
  enum: ["python", "matlab"]
  default: "matlab"
  description: "Processing backend for registration and spot finding"
```

Extend Python-specific parameters within existing rule configs (using `additionalProperties: true` already in schema):
```yaml
# Under spot_finding_params — extend for Python noise-floor mode:
intensity_estimation:
  enum: ["local", "global", "noise", "adaptive", "adaptive_round"]  # add Python modes
intensity_threshold:
  # Remove maximum: 1 constraint (noise mode uses k-sigma, typically 3-5)

# Under local_registration_params — add Python method selection:
method:
  enum: ["demons", "tps", "cpd"]
  default: "demons"
```

Add new Python-specific parameters (optional, only used when `backend: "python"`):
```yaml
# Under rule parameters:
streaming:
  type: boolean
  default: false
  description: "Use streaming mode for lower memory (Python only)"

snr_threshold:
  type: number
  description: "SNR threshold for normalization gating (Python only)"
```

### 1.2 Snakefile Update

- Add `BACKEND` variable from config
- Conditional include of rule files based on backend
- Update `get_overall_output()` to handle Python rule names (same outputs, different rule names if needed — but since we use conditional includes, rule names can be identical)

### 1.3 Common Helpers (`common.smk`)

All Python wrapper scripts use Snakemake's built-in `script:` directive (like `stitch_subtile` already does). Each script receives the `snakemake` object with `input`, `output`, `config`, `wildcards` — no subprocess helper needed.

Add `BACKEND` variable and validate it:
```python
BACKEND = config.get("backend", "matlab")
if BACKEND not in ("python", "matlab"):
    raise ValueError(f"Unknown backend '{BACKEND}'. Valid options: 'python', 'matlab'")
```

### 1.4 Files to Create/Modify

| File | Action |
|------|--------|
| `workflow/schemas/config.schema.yaml` | Add `backend`, extend parameter enums |
| `workflow/Snakefile` | Conditional include based on `backend` |
| `workflow/rules/common.smk` | Add `BACKEND` variable |

---

## Phase 2: Direct Mode (`rsf_single_fov`)

**Goal:** Full single-FOV pipeline in Python. This is the most-used workflow mode.

### 2.1 Python Wrapper Script: `workflow/scripts/rsf_single_fov.py`

This script translates Snakemake config into `STARMapDataset` + `FOV` API calls:

```python
"""Python backend: single-FOV registration + spot finding."""
import sys
sys.path.insert(0, snakemake.config["starfinder_path"] + "/src/python")

from pathlib import Path
from starfinder.dataset import STARMapDataset

# Build dataset from Snakemake config
sdata = STARMapDataset.from_config(snakemake.config)
fov_id = snakemake.wildcards.fovID
params = snakemake.config["rules"]["rsf_single_fov"]["parameters"]

# Load codebook
sdata.load_codebook(Path(snakemake.input[1]))  # genes.csv

fov = sdata.fov(fov_id)

streaming = params.get("streaming", False)

if streaming:
    # Streaming mode: one round at a time, ~50% less memory
    fov.run_streaming(
        rotate_angle=snakemake.config.get("rotate_angle"),
        snr_threshold=params.get("snr_threshold"),
        intensity_estimation=params["spot_finding"].get("intensity_estimation", "noise"),
        intensity_threshold=params["spot_finding"]["intensity_threshold"],
        voxel_size=tuple(params["reads_extraction"]["voxel_size"]),
        end_bases=params["reads_filtration"].get("end_base"),
        start_base=params["reads_filtration"].get("start_base", "C"),
        local_method=params.get("local_registration", {}).get("method") if params.get("local_registration", {}).get("run") else None,
    )
else:
    # Batch mode: load all rounds, then process
    fov.load_raw_images()

    if params.get("enhance_contrast", {}).get("run"):
        fov.enhance_contrast(snr_threshold=params.get("snr_threshold"))

    if params.get("hist_equalize", {}).get("run"):
        fov.hist_equalize()

    if params.get("morph_recon", {}).get("run"):
        fov.morph_recon(radius=params["morph_recon"].get("radius", 3))

    if params.get("global_registration", {}).get("run"):
        fov.global_registration(
            ref_img=params["global_registration"].get("ref_img", "merged"),
            mov_img=params["global_registration"].get("mov_img", "merged"),
        )

    if params.get("local_registration", {}).get("run"):
        method = params["local_registration"].get("method", "demons")
        fov.local_registration(method=method)

    if params.get("spot_finding", {}).get("run"):
        fov.spot_finding(
            intensity_estimation=params["spot_finding"].get("intensity_estimation", "noise"),
            intensity_threshold=params["spot_finding"]["intensity_threshold"],
        )

    if params.get("reads_extraction", {}).get("run"):
        fov.reads_extraction(voxel_size=tuple(params["reads_extraction"]["voxel_size"]))

    if params.get("reads_filtration", {}).get("run"):
        fov.reads_filtration(
            end_bases=params["reads_filtration"].get("end_base"),
            start_base=params["reads_filtration"].get("start_base", "C"),
        )

# Save outputs
fov.save_ref_merged()
fov.save_signal(slot="goodSpots")
```

### 2.2 Snakemake Rule: `registration-py.smk`

```python
rule rsf_single_fov:
    input:
        config['config_path'].replace('.yaml', '.json'),
        expand("{input_dir}/genes.csv", input_dir=INPUT_DIR),
        expand("{input_dir}/{rounds}/{{fovID}}", input_dir=INPUT_DIR, rounds=ROUND),
    output:
        expand("{output_dir}/log/{{fovID}}_rsf.txt", output_dir=OUTPUT_DIR),
        expand("{output_dir}/log/sf_scores/{{fovID}}.txt", output_dir=OUTPUT_DIR),
        expand("{output_dir}/images/ref_merged/{{fovID}}.tif", output_dir=OUTPUT_DIR),
        expand("{output_dir}/signal/{{fovID}}_goodSpots.csv", output_dir=OUTPUT_DIR),
    threads: 4
    resources:
        mem_mb=get_rule_config('rsf_single_fov', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb']),
        runtime=get_rule_config('rsf_single_fov', 'resources.runtime', DEFAULT_RESOURCES['runtime'])
    benchmark:
        f"{OUTPUT_DIR}/log/benchmark/rsf_single_fov/{{fovID}}.txt"
    script:
        "../scripts/rsf_single_fov.py"
```

**Key:** Same rule name, same inputs/outputs, same config keys. Only the `run:` block becomes `script:`. The conditional include in Snakefile ensures no conflict with the MATLAB rule.

### 2.3 `STARMapDataset.from_config()` Enhancement

The `from_config()` classmethod needs to handle the full Snakemake config dict. Currently it expects a subset of keys. Verify and extend to handle:
- `seq_channel_order` → `channel_order`
- `maximum_projection`
- `fov_id_pattern` → `fov_pattern`
- `rotate_angle`
- `split_index` for codebook

### 2.4 FOV Output Adjustments

Verify `FOV.save_signal()` CSV columns match MATLAB output:
- MATLAB: `x, y, z, gene, color_seq, score, ...` (1-based)
- Python: Currently saves `x, y, z, gene` (1-based) — may need to add `color_seq`, `score` columns

Verify `FOV.save_ref_merged()` handles `maximum_projection` correctly.

Add log file output: FOV currently doesn't write a `_rsf.txt` log or `sf_scores/{fov}.txt`. Need to add logging that writes to the expected output paths.

### 2.5 Files to Create/Modify

| File | Action |
|------|--------|
| `workflow/scripts/rsf_single_fov.py` | **Create**: Python direct-mode script |
| `workflow/rules/registration-py.smk` | **Create**: Python registration rules |
| `src/python/starfinder/dataset/dataset.py` | Extend `from_config()` |
| `src/python/starfinder/dataset/fov.py` | Add log file outputs (rsf.txt, sf_scores) |

---

## Phase 3: Subtile Mode

**Goal:** Python subtile workflow (gr → lrsf → stitch) with NPZ intermediates and streaming support.

### 3.1 Key Difference: NPZ vs MAT

MATLAB saves subtile data as `.mat` files. Python uses `.npz` (NumPy compressed). The `stitch_subtile` rule only reads the per-subtile `_goodSpots.csv` files (not the `.mat`/`.npz`), so the **stitch step is already compatible**.

The intermediate format change only affects:
- `gr_single_fov_subtile` output → `.npz` instead of `.mat`
- `lrsf_single_fov_subtile` input → reads `.npz` instead of `.mat`
- `deep_create_subtile` output → `.npz` instead of `.mat`
- `deep_rsf_subtile` input → reads `.npz` instead of `.mat`

Since Python and MATLAB rules are mutually exclusive via conditional includes, this is fine — the file extension just changes in the Python rule files.

### 3.2 Python Script: `workflow/scripts/gr_single_fov_subtile.py`

Pipeline: load → preprocess → global register → create subtiles (NPZ)

**Batch mode:**
```
1. sdata = STARMapDataset.from_config(config)
2. fov = sdata.fov(fov_id)
3. fov.load_raw_images()
4. fov.enhance_contrast() / hist_equalize() / morph_recon()  (conditional)
5. fov.global_registration()
6. fov.save_ref_merged()
7. fov.create_subtiles()  → saves .npz files + subtile_coords.csv
```

**Streaming mode** (`streaming: true`):

The streaming variant processes rounds one at a time for global registration, keeping only the reference round in memory, then creates subtiles from the globally-registered data. This is particularly valuable for the subtile workflow since it handles large FOVs (e.g., 3072x3072x30) where loading all rounds simultaneously can exceed available memory.

```
1. sdata = STARMapDataset.from_config(config)
2. fov = sdata.fov(fov_id)
3. fov.run_streaming_gr()  → global registration only, streaming
   - Load ref round, preprocess, keep in memory
   - For each non-ref round: load, preprocess, global register, store shift, discard
   - Reload all rounds with shifts applied (or apply shifts during subtile creation)
4. fov.save_ref_merged()
5. fov.create_subtiles()  → saves .npz files + subtile_coords.csv
```

Implementation note: This requires a new `run_streaming_gr()` method on FOV (or a `streaming_stop_after="global_registration"` parameter to `run_streaming()`) that performs only load → preprocess → global registration in streaming fashion, then creates subtiles. The local registration + spot finding happens later in `lrsf_single_fov_subtile`.

### 3.3 Python Script: `workflow/scripts/lrsf_single_fov_subtile.py`

Pipeline: load subtile → local register → spot find → extract → filter

```
1. sdata = STARMapDataset.from_config(config)
2. fov = FOV.from_subtile(subtile_path, sdata, fov_id)
3. fov.local_registration(method=...)  (conditional)
4. fov.morph_recon()  (conditional)
5. fov.spot_finding()
6. fov.reads_extraction()
7. fov.reads_filtration()
8. fov.save_signal(slot="goodSpots")  → subtile_goodSpots_{n}.csv
```

### 3.4 Python Rules

**`registration-py.smk`** (add `gr_single_fov_subtile`):

```python
rule gr_single_fov_subtile:
    input:
        config['config_path'].replace('.yaml', '.json'),
        expand("{input_dir}/genes.csv", input_dir=INPUT_DIR),
        expand("{input_dir}/{rounds}/{{fovID}}", input_dir=INPUT_DIR, rounds=ROUND),
    output:
        expand("{output_dir}/log/{{fovID}}_gr.txt", output_dir=OUTPUT_DIR),
        expand("{output_dir}/images/ref_merged/{{fovID}}.tif", output_dir=OUTPUT_DIR),
        temp(expand("{output_dir}/output/subtile/{{fovID}}/subtile_coords.csv", output_dir=OUTPUT_DIR)),
        temp(expand("{output_dir}/output/subtile/{{fovID}}/subtile_data_{n_subtile}.npz",
                    output_dir=OUTPUT_DIR, n_subtile=N_SUBTILE)),
    threads: 4
    resources:
        mem_mb=get_rule_config('gr_single_fov_subtile', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb']),
        runtime=make_get_runtime('gr_single_fov_subtile')
    benchmark:
        f"{OUTPUT_DIR}/log/benchmark/gr_single_fov_subtile/{{fovID}}.txt"
    script:
        "../scripts/gr_single_fov_subtile.py"
```

**`spot-finding-py.smk`** (add `lrsf_single_fov_subtile`):

```python
rule lrsf_single_fov_subtile:
    input:
        config['config_path'].replace('.yaml', '.json'),
        expand("{input_dir}/genes.csv", input_dir=INPUT_DIR),
        expand("{output_dir}/output/subtile/{{fovID}}/subtile_data_{{n_subtile}}.npz", output_dir=OUTPUT_DIR),
    output:
        expand("{output_dir}/log/sf_scores/{{fovID}}_{{n_subtile}}.txt", output_dir=OUTPUT_DIR),
        temp(expand("{output_dir}/output/subtile/{{fovID}}/subtile_goodSpots_{{n_subtile}}.csv", output_dir=OUTPUT_DIR)),
    threads: 4
    resources:
        mem_mb=get_rule_config('lrsf_single_fov_subtile', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb']),
        runtime=make_get_runtime('lrsf_single_fov_subtile')
    benchmark:
        f"{OUTPUT_DIR}/log/benchmark/lrsf_single_fov_subtile/{{fovID}}_{{n_subtile}}.txt"
    script:
        "../scripts/lrsf_single_fov_subtile.py"
```

The `stitch_subtile` rule is **reused as-is** (reads only CSVs) — include it in `spot-finding-py.smk` or keep it backend-agnostic.

### 3.5 `FOV.create_subtiles()` Verification

Currently saves `.npz` files. Need to verify:
- Output paths match Snakemake expectations: `subtile_data_{1..N}.npz`
- `subtile_coords.csv` format is compatible with `stitch_subtile.py`
- `FOV.from_subtile()` properly restores state for downstream processing

### 3.6 Streaming Support for `gr_single_fov_subtile`

Need to add a streaming-capable global-registration-only path to FOV. Options:

**Option A**: New `FOV.run_streaming_gr()` method — does load → preprocess → global registration in streaming, stops before spot finding. After streaming GR, the FOV has `global_shifts` populated and ref round in memory, ready for `create_subtiles()`.

**Option B**: Extend `FOV.run_streaming()` with a `stop_after="global_registration"` parameter.

Recommend **Option A** for clarity — it's a distinct operation (global-reg only, then subtile creation) vs the full pipeline in `run_streaming()`.

### 3.7 Files to Create/Modify

| File | Action |
|------|--------|
| `workflow/scripts/gr_single_fov_subtile.py` | **Create** |
| `workflow/scripts/lrsf_single_fov_subtile.py` | **Create** |
| `workflow/rules/registration-py.smk` | Add `gr_single_fov_subtile` rule |
| `workflow/rules/spot-finding-py.smk` | **Create**: `lrsf_single_fov_subtile`, `stitch_subtile` |
| `src/python/starfinder/dataset/fov.py` | Add `run_streaming_gr()` method; verify subtile I/O paths |

---

## Phase 4: Deep Mode + Streaming

### 4.1 Deep Mode

Deep-tissue workflow differs from subtile in preprocessing (no enhance_contrast, different hist_equalize). The Python scripts are similar to subtile but with different parameter handling.

**Scripts:**
- `workflow/scripts/deep_create_subtile.py` — Like `gr_single_fov_subtile.py` but uses deep-tissue preprocessing config; also supports streaming mode
- `workflow/scripts/deep_rsf_subtile.py` — Like `lrsf_single_fov_subtile.py`

**Rules:** Add to `registration-py.smk` and `spot-finding-py.smk`.

### 4.2 Streaming Mode (Python-only)

Streaming mode is a Python-exclusive feature that doesn't exist in MATLAB. It processes one round at a time, achieving ~50% lower peak memory.

Streaming is enabled via `parameters.streaming: true` within rule config. Both `rsf_single_fov` and `gr_single_fov_subtile` (and `deep_create_subtile`) support it. The Python script checks this flag internally.

**Config example:**
```yaml
backend: "python"
workflow_mode: "direct"
rules:
  rsf_single_fov:
    run: True
    parameters:
      streaming: true     # Python streaming mode
      snr_threshold: 5.0  # Python SNR gating
      ...
```

**Config example (subtile + streaming):**
```yaml
backend: "python"
workflow_mode: "subtile"
rules:
  gr_single_fov_subtile:
    run: True
    parameters:
      streaming: true     # Stream rounds during global registration
      ...
  lrsf_single_fov_subtile:
    run: True
    parameters:
      ...
```

### 4.3 Files to Create/Modify

| File | Action |
|------|--------|
| `workflow/scripts/deep_create_subtile.py` | **Create** |
| `workflow/scripts/deep_rsf_subtile.py` | **Create** |
| `workflow/rules/registration-py.smk` | Add deep rules |
| `workflow/rules/spot-finding-py.smk` | Add deep rules |

---

## Phase 5: Testing & Validation

### 5.1 Unit Test: Python Scripts

Add tests in `src/python/test/` that verify each wrapper script works correctly on synthetic data:
- `test_snakemake_rsf.py` — Direct mode on small dataset
- `test_snakemake_subtile.py` — Subtile mode on small dataset

These tests create a mock `snakemake` object and call the script logic directly.

### 5.2 A/B Comparison Framework

Create a comparison config that runs both backends on the same data:
```bash
# Run MATLAB backend
snakemake --configfile config_matlab.yaml  # backend: "matlab"

# Run Python backend (different output_id to avoid overwrite)
snakemake --configfile config_python.yaml  # backend: "python"

# Compare outputs
python compare_backends.py matlab_output/ python_output/
```

The comparison script (`workflow/scripts/compare_backends.py`) would:
1. Load both `_goodSpots.csv` files
2. Compare spot counts, gene distributions, spatial overlap
3. Compare `ref_merged` images (NCC, SSIM)
4. Report registration shift differences from `gr_shifts/` logs

### 5.3 Validation on Real Datasets

Run on the 3 real datasets with both backends:
- tissue-2D (subtile mode)
- LN (direct mode)
- cell-culture-3D (direct mode)

Compare:
- Spot count ratio (Python/MATLAB)
- Gene overlap percentage
- Spatial colocalization of spots
- Runtime and memory (from Snakemake benchmark files)

### 5.4 Files to Create

| File | Action |
|------|--------|
| `src/python/test/test_snakemake_rsf.py` | **Create** |
| `src/python/test/test_snakemake_subtile.py` | **Create** |
| `workflow/scripts/compare_backends.py` | **Create** |
| `tests/tissue_2D_test_python.yaml` | **Create**: Python-backend test config |

---

## Phase 6: Config & Documentation

### 6.1 Update Test Configs

Create Python-backend variants of existing test configs:
```yaml
# tests/tissue_2D_test_python.yaml
backend: "python"
output_id: "tissue-2D-test-python"
# ... same parameters but with Python-specific additions:
rules:
  rsf_single_fov:
    parameters:
      streaming: true
      snr_threshold: 5.0
      spot_finding:
        intensity_estimation: "noise"  # Python noise-floor mode
        intensity_threshold: 5.0
      local_registration:
        run: true
        method: "tps"  # Python-only TPS method
```

### 6.2 Update CLAUDE.md

Add Python backend section to common commands:
```bash
# Python backend (direct mode)
snakemake -s workflow/Snakefile --configfile tests/tissue_2D_test_python.yaml -n

# Compare backends
python workflow/scripts/compare_backends.py output_matlab/ output_python/
```

### 6.3 Update Config Schema Documentation

Document new fields: `backend`, `streaming`, `snr_threshold`, `method` for local registration.

---

## Implementation Order

| Phase | Effort | Dependencies | Priority |
|-------|--------|-------------|----------|
| **Phase 1**: Infrastructure | Small | None | Highest |
| **Phase 2**: Direct mode | Medium | Phase 1 | Highest |
| **Phase 5.1**: Unit tests for Phase 2 | Small | Phase 2 | High |
| **Phase 3**: Subtile mode | Medium | Phase 1 | High |
| **Phase 4.1**: Deep mode | Small | Phase 3 | Medium |
| **Phase 4.2**: Streaming mode | Small | Phase 2 | Medium |
| **Phase 5.2-5.3**: A/B comparison | Medium | Phase 2+3 | High |
| **Phase 6**: Config & docs | Small | All | Low |

**Estimated total: ~7 implementation sessions**

Phase 1+2 together give a working Python direct-mode pipeline — the most impactful milestone. Phase 3 follows naturally (subtile reuses most of Phase 2's code). Phase 4-6 are incremental improvements.

---

## Key Design Decisions

1. **Conditional include (not ruleorder)**: Avoids ambiguous rule resolution. Only one backend's rules are loaded per run.

2. **Same rule names**: Python rules use identical names to MATLAB rules. This means `get_overall_output()` works without modification — it checks rule names, not backend.

3. **`script:` directive**: Python wrapper scripts use Snakemake's `script:` directive (not `subprocess`). This gives access to `snakemake.input`, `snakemake.output`, `snakemake.config`, `snakemake.wildcards` natively. No subprocess helper is needed — this is consistent with existing Python rules like `stitch_subtile`.

4. **NPZ not MAT for subtiles**: Python subtile intermediates use `.npz`. Since Python and MATLAB rules never coexist in the same run, this is clean.

5. **Streaming as a parameter, not a mode**: Streaming is enabled via `parameters.streaming: true` within rule config, not a separate workflow mode. This keeps the mode system simple. Both `rsf_single_fov` (direct) and `gr_single_fov_subtile` (subtile) support streaming.

6. **Log format divergence is acceptable**: MATLAB diary logs and Python logs will differ in format. The important outputs (CSVs, TIFFs) must match.

7. **Suffix naming convention**: Rule files use `-py` suffix (`registration-py.smk`), scripts share base names with MATLAB (`.py` vs `.m` extension distinguishes).

## File Naming Summary

| New File | Purpose |
|----------|---------|
| `workflow/rules/registration-py.smk` | Python rules: rsf_single_fov, gr_single_fov_subtile, deep_create_subtile, nuclei_registration, rotate_nuclei |
| `workflow/rules/spot-finding-py.smk` | Python rules: lrsf_single_fov_subtile, deep_rsf_subtile, stitch_subtile |
| `workflow/scripts/rsf_single_fov.py` | Direct-mode pipeline |
| `workflow/scripts/gr_single_fov_subtile.py` | Global registration + subtile creation (batch or streaming) |
| `workflow/scripts/lrsf_single_fov_subtile.py` | Local registration + spot finding on subtile |
| `workflow/scripts/deep_create_subtile.py` | Deep-tissue subtile creation (batch or streaming) |
| `workflow/scripts/deep_rsf_subtile.py` | Deep-tissue RSF on subtile |
| `workflow/scripts/compare_backends.py` | A/B comparison tool |

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| CSV column mismatch | Phase 2: Verify FOV.save_signal() columns match MATLAB output exactly |
| `from_config()` missing keys | Phase 2: Audit all config keys used by MATLAB scripts, ensure Python handles them |
| Subtile coordinate system | Phase 3: Verify create_subtiles() 1-based coords match MATLAB subtile_coords.csv |
| stitch_subtile.py compatibility | Phase 3: Test with Python-generated subtile CSVs before real data |
| Performance regression | Phase 5: Snakemake benchmark files capture time/memory automatically |
| UGER cluster compatibility | Phase 2: Test `uv run` works within UGER job environment |
| Streaming GR + subtile interaction | Phase 3: Verify run_streaming_gr() correctly populates state for create_subtiles() |
