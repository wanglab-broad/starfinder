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

## Current Progress / Milestones
Use this section to track development milestones.

### 2025-01-21: Snakemake Modularization & Upgrade Project Started
- [x] Created PR #9 to merge dev → main (commits 28-48)
- [x] Created development plan (`dev/current_plan.md`)
- [ ] **Phase 1: Modularization** (in progress)
  - [x] Create `common.smk` with shared code
  - [x] Migrate registration rules → `registration.smk`
  - [x] Migrate spot-finding rules → `spot-finding.smk`
  - [x] Migrate segmentation rules → `segmentation.smk`
  - [x] Migrate stitching rules → `stitching.smk`
  - [x] Migrate reads-assignment rules → `reads-assignment.smk`
  - [x] Clean up main Snakefile (reduced from ~566 lines to ~32 lines)
  - [ ] Test with dry run
- [ ] **Phase 2: Snakemake 9 Upgrade** (in progress)
  - [x] Create `environment-v9.yaml` (Python 3.11+, Snakemake 9.x)
  - [ ] Update `profile/broad-uger/config.yaml` for v9 syntax
  - [ ] Test environment creation
  - [ ] Test pipeline with new environment
- [ ] **Phase 3: Code Quality Improvements**

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
- [ ] **Testing** (in progress)
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

## Future Directions

### 1. Replace MATLAB with Python
The ultimate goal is to eliminate all MATLAB implementations and develop equivalent functionality in Python.
- The most challenging aspect will be re-implementing the 3D image registration algorithms.

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
