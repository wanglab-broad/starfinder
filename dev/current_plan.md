# STARfinder Development Plan

## User Decisions
- **Target Snakemake version**: 9.x (latest)
- **Implementation order**: Modularization first, then upgrade
- **Rule naming**: Keep existing names (backward compatibility)

## Overview
This plan addresses two main goals and identifies additional improvements:
1. Modularize Snakemake rules from a monolithic Snakefile to separate .smk files
2. Upgrade Snakemake from v7.32.4 to v9.x and leverage new features
3. Code quality improvements identified during review

---

## Part 1: Modularize Snakemake Rules

### Current State
- **Main Snakefile**: ~566 lines with all 20+ rules embedded
- **Existing rule files** (mostly empty scaffolds):
  - `utils.smk` - Has `create_sample_maf` rule, already included
  - `registration.smk` - Skeleton with different config keys
  - `spot-finding.smk` - Skeleton with different config keys
  - `gcloud-backup.smk` - Standalone backup logic

### Proposed Rule File Structure

```
workflow/rules/
├── common.smk          # Shared imports, helper functions, parameters
├── registration.smk    # Global/local registration, nuclei registration
├── spot-finding.smk    # RSF variants, deep RSF, spot filtering
├── segmentation.smk    # StarDist, DAPI enhancement
├── stitching.smk       # BigStitcher, tile config
├── reads-assignment.smk # Reads assignment, sample h5ad creation
├── utils.smk           # Utility rules (MAF creation, etc.)
└── gcloud-backup.smk   # Cloud backup (keep separate)
```

### Rule Assignments

| New File | Rules to Move |
|----------|---------------|
| `common.smk` | Helper functions (`yaml_to_json`, `run_matlab_scripts`, `run_fiji_macros`), shared parameters (INPUT_DIR, OUTPUT_DIR, FOVS, etc.) |
| `registration.smk` | `rsf_single_fov`, `gr_single_fov_subtile`, `nuclei_registration`, `rotate_nuclei` |
| `spot-finding.smk` | `lrsf_single_fov_subtile`, `deep_create_subtile`, `deep_rsf_subtile`, `rsf_single_fov_seq`, `stitch_subtile` |
| `segmentation.smk` | `stardist_segmentation`, `enhance_dapi_with_flamingo`, `create_nuclei_amplicon_overlay` |
| `stitching.smk` | `stitching_preparation`, `create_BigStitcher_macro`, `run_BigStitcher_macro`, `create_tile_config` |
| `reads-assignment.smk` | `reads_assignment`, `create_sample_h5ad`, `create_sample_reads_assignment` |

### Refactored Main Snakefile Structure

```python
# Snakefile (~50 lines)
include: "rules/common.smk"
include: "rules/registration.smk"
include: "rules/spot-finding.smk"
include: "rules/segmentation.smk"
include: "rules/stitching.smk"
include: "rules/reads-assignment.smk"
include: "rules/utils.smk"

localrules: rsf_preparation, create_sample_maf

rule all:
    input:
        get_overall_output
```

### Implementation Steps

1. **Create `common.smk`** with shared code:
   - All imports (os, glob, random, pandas, Path)
   - Helper functions
   - Parameter definitions (INPUT_DIR, OUTPUT_DIR, FOVS, SAMPLE, etc.)
   - `get_overall_output()` function
   - `rsf_preparation` rule

2. **Migrate rules to respective files** (one file at a time):
   - Each file starts with `# Uses variables from common.smk`
   - Move rules with their `get_runtime` functions
   - Test after each migration

3. **Update main Snakefile**:
   - Remove migrated code
   - Add include statements
   - Keep only `rule all` and localrules

4. **Clean up existing rule files**:
   - Remove duplicate boilerplate from `registration.smk`, `spot-finding.smk`
   - Update to use shared variables from `common.smk`

---

## Part 2: Upgrade Snakemake Version

### Current Version: 7.32.4
### Target Version: 9.x (latest)

### Breaking Changes to Address

| Old (v7) | New (v8+) |
|----------|-----------|
| `--use-conda` | `--software-deployment-method conda` |
| `--cluster CMD` | `--executor cluster-generic --cluster-generic-submit-cmd CMD` |
| `--default-remote-provider` | `--default-storage-provider` |
| `--greediness` | `--scheduler-greediness` |

### Files to Update

1. **`config/environment.yaml`**:
   ```yaml
   # Change from
   - snakemake=7.32.4
   # To
   - snakemake>=9.0
   - snakemake-executor-plugin-cluster-generic  # For UGER support
   - snakemake-storage-plugin-gcs  # For Google Cloud (if needed)
   ```

2. **`profile/broad-uger/config.yaml`**:
   - Update command-line flags to v8 syntax
   - Add executor plugin configuration

3. **Documentation/README**:
   - Update installation instructions
   - Document new command syntax

### New Features Available in v9

| Feature | Benefit | Priority |
|---------|---------|----------|
| **Pathvars** | Configure output paths at workflow/module level - great for modular rules | High |
| **Storage plugins** | Better cloud storage integration (GCS, S3) | Medium |
| **Profile versioning** | Different configs for different Snakemake versions | Medium |
| **Enhanced resource handling** | Better memory defaults, improved scheduling | Medium |
| **Improved module system** | Better isolation for modular workflows | High |

---

## Part 3: Additional Improvements

### High Priority

1. **Remove commented config paths** (lines 14-31 in Snakefile)
   - These are stale references to old test configurations
   - Create a `config/examples/` directory if examples are needed

2. **Fix duplicate `get_runtime` function definitions**
   - Currently defined 4 times with different config keys
   - Create a factory function or use lambda

3. **Keep existing rule names** (per user preference)
   - Maintains backward compatibility with existing configs and scripts
   - No changes to `rsf_single_fov`, `gr_single_fov_subtile`, etc.

### Medium Priority

4. **Add config schema validation**
   - Use Snakemake's built-in schema validation
   - Create `workflow/schemas/config.schema.yaml`

5. **Add snakefmt for consistent formatting**
   - Install: `pip install snakefmt`
   - Run: `snakefmt workflow/`

6. **Improve error handling in helper functions**
   - `run_matlab_scripts()` doesn't check return codes
   - `run_fiji_macros()` has potential issues

### Low Priority

7. **Consider Snakemake wrappers**
   - For common tools like StarDist segmentation
   - Promotes reusability across projects

8. **Add workflow documentation**
   - DAG visualization in README
   - Rule dependency documentation

---

## Implementation Order (User-selected: Modularization First)

### Phase 1: Modularization
1. Create `common.smk` with shared code (imports, helpers, parameters)
2. Migrate registration rules → `registration.smk`
3. Migrate spot-finding rules → `spot-finding.smk`
4. Migrate segmentation rules → `segmentation.smk`
5. Migrate stitching rules → `stitching.smk`
6. Migrate reads-assignment rules → `reads-assignment.smk`
7. Clean up main Snakefile (keep only includes + rule all)
8. Test with dry run: `snakemake -n`

### Phase 2: Snakemake 9 Upgrade
1. Create new conda environment with Snakemake 9.x
2. Install required executor plugins
3. Update `profile/broad-uger/config.yaml` command flags
4. Test on single FOV with new syntax
5. Update documentation/README
6. Full pipeline test

### Phase 3: Code Quality (Optional)
1. Remove commented configfile paths (lines 14-31)
2. Refactor duplicate `get_runtime` functions
3. Add config schema validation (optional)

---

## Verification Plan

1. **After modularization** (still on Snakemake 7):
   ```bash
   snakemake -s workflow/Snakefile --configfile <test_config> -n  # Dry run
   snakemake -s workflow/Snakefile --configfile <test_config> --dag | dot -Tpng > dag.png  # Check DAG
   snakemake -s workflow/Snakefile --configfile <test_config> --lint  # Lint check
   ```

2. **After Snakemake 9 upgrade**:
   ```bash
   # Test with new executor syntax
   snakemake --executor cluster-generic --cluster-generic-submit-cmd "qsub" -n
   # Test with new software deployment syntax
   snakemake --software-deployment-method conda -n
   ```

3. **Full validation**:
   - Run on small test dataset (tissue-2D or cell-culture-3D from Zenodo)
   - Compare outputs with previous version
   - Verify all rules execute correctly on UGER cluster

---

## References
- [Snakemake 9 Changelog](https://snakemake.readthedocs.io/en/stable/project_info/history.html)
- [Snakemake Migration Guide](https://snakemake.readthedocs.io/en/stable/getting_started/migration.html)
- [Snakemake Modularization](https://snakemake.readthedocs.io/en/stable/snakefiles/modularization.html)

---

## Critical Files to Modify

- `workflow/Snakefile` - Major refactor
- `workflow/rules/common.smk` - New file
- `workflow/rules/registration.smk` - Rewrite
- `workflow/rules/spot-finding.smk` - Rewrite
- `workflow/rules/segmentation.smk` - New file
- `workflow/rules/stitching.smk` - New file
- `workflow/rules/reads-assignment.smk` - New file
- `config/environment.yaml` - Version update
- `profile/broad-uger/config.yaml` - Flag updates
