# STARfinder Development Notes

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

## Future directions
* Replace MATLAB with Python: The ultimate goal is to eliminate all MATLAB implementations and develop equivalent functionality in Python.
    * The most challenging aspect will be re-implementing the 3D image registration algorithms.
* Adopt modern data standards: Implement state-of-the-art data structures for biomedical imaging analysis, such as OME-TIFF with high compression ratios, to improve pipeline computational efficiency.
* Enable cloud compatibility: Make the pipeline compatible with cloud platforms.
