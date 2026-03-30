# Aging Dataset E2E Benchmark Plan

**Date:** 2026-03-30
**Status:** FINISHED

## Goal

Run an E2E benchmark on the aging dataset (mPFC brain tissue, 2044 genes, 9 sequencing rounds) using the Python backend, following the established benchmark pattern from tissue-2D, LN, and cell-culture-3D.

## Dataset Summary

| Property | Value |
|----------|-------|
| Path | `/home/unix/jiahao/wanglab/Data/Processed/sample-dataset/aging/` |
| Config | `2025-04-26-Jiakun-mPFC-aging.yaml` |
| FOVs in sample | 6 (Position400–405), 848 total |
| Sequencing rounds | 9 (round1–round9) |
| Reference round | round1 |
| Image dimensions | 2048×2048×36, uint8 |
| Channels | 5 in round1 (ch00-ch04, incl. extra DAPI), 4 in others (ch00-ch03) |
| Voxel size (physical) | 0.14 µm XY, 0.35 µm Z |
| Voxel size (extraction) | (2, 2, 1) per YAML config |
| Codebook | 14,242 entries, 2,044 unique genes, 11-nt barcodes, ~7 pads/gene |
| Barcode structure | 2 segments, split_index=4 (Python 0-based) = MATLAB split_index=5 |
| End bases | seg1: "CC" (all probes), seg2: "AT" (STAR probes) / "TT" (RIBO probes) |
| Rotation | -90° |
| Spot finding | adaptive@0.2 |

## Barcode Structure (RESOLVED)

Each 11-nt barcode has the structure:

```
GCGATCATCTT → G (randomized) + CGATC (seg1, CC) + ATCTT (seg2, AT/TT)
              ^                 |---5 nt--------|  |---5 nt--------|
              differentiates    starts C, ends C   STAR: starts A, ends T
              probes (pads)                        RIBO: starts T, ends T
```

**Encoding pipeline:**
1. Reverse barcode: `GCGATCATCTT` → `TTCTACTAGCG`
2. Encode to color-space: 11 bases → 10 colors `1334234344`
3. `split_index=4` removes the boundary color between reversed segments:
   - `1334` (seg2) + **2** (boundary, removed) + `3434` (seg1) + **4** (random)
4. Swap halves: `34344` + `1334` = `343441334` (9 colors)

**Result:** 9-color codebook entries match 9 colors extracted from 9 sequencing rounds.

**Index convention:** MATLAB `split_index=5` (1-based) = Python `split_index=4` (0-based). Verified to produce identical codebook entries.

## Plan

### Phase 1: Full E2E Pipeline

Since the barcode length mismatch is resolved, we can run the full pipeline directly.

**Parameters:**
```python
DATA_ROOT = Path("/home/unix/jiahao/wanglab/Data/Processed/sample-dataset/aging")
RESULTS_DIR = BENCHMARK_ROOT / "e2e" / "results" / "aging"
N_ROUNDS = 9
REF_ROUND = "round1"
CHANNEL_ORDER = ["ch00", "ch02", "ch01", "ch03"]  # 4 sequencing channels only
FOV_PATTERN = "Position%03d"
FOV_IDS = [400, 401]  # First 2 of 6 sample FOVs
ROTATE_ANGLE = -90
INTENSITY_ESTIMATION = "adaptive"
INTENSITY_THRESHOLD = 0.2
VOXEL_SIZE = (2, 2, 1)  # Per YAML config
SPLIT_INDEX = 4          # Python 0-based (= MATLAB 5)
END_BASES = "CC"         # For diagnostic stats (first segment)
START_BASE = "C"         # For diagnostic decoding
```

**Channel handling:** Use the standard 4-channel order `["ch00", "ch02", "ch01", "ch03"]`. Round1's extra ch04 (DAPI) is automatically excluded — `load_image_stacks` only loads files matching the specified channel patterns.

**Steps:**
1. Create benchmark script at `starfinder_benchmark/e2e/results/aging/run_e2e_aging.py`
2. Follow existing pattern from `run_e2e_LN.py` (closest in structure: Position-based FOVs, non-default start_base)
3. Run pipeline: load → rotate → enhance → global_registration → spot_finding → extraction → filtration
4. Load codebook with `split_index=4`
5. Run filtration with `end_bases="CC"`, `start_base="C"` (single-segment diagnostic — Python doesn't support multi-segment end_base checking yet, but actual filtering is codebook-membership-only)
6. Compute pad-balance QC: for each gene, count reads per pad variant, report CV and min/max ratio across pads
7. Save: goodSpots CSV, QC metrics (incl. pad balance), registration + signal inspection images

**Expected output:**
```
aging/
├── signal/
│   ├── Position400_goodSpots.csv
│   └── Position401_goodSpots.csv
├── log/
│   ├── Position400.csv            # QC metrics
│   ├── Position401.csv
│   ├── gr_inspect/                # Registration overlays
│   ├── gr_shifts/                 # Shift values per round
│   └── signal_inspect/            # GoodSpots visualizations
└── e2e_results.json               # Summary
```

**Memory estimate:**
- Per channel: 2048×2048×36 × 1 byte = ~150 MB
- 9 rounds × 4 channels = 36 channels × 150 MB = ~5.4 GB (batch mode)
- Peak RSS (with FFT overhead): ~10–14 GB estimated
- Streaming mode would reduce to ~2–3 GB

### Phase 2: Streaming Mode (Optional)

If batch-mode RSS is too high (>15 GB):
- Switch to `fov.run_streaming()` for memory efficiency
- Expected ~50% RSS reduction based on prior benchmarks

### Phase 3: MATLAB Comparison (If Available)

If MATLAB results become available at `Analyzed/sample-dataset/aging/`:
1. Add MATLAB comparison logic to benchmark script
2. Generate comparison CSV at `log/matlab_comparison/`
3. Compare: spot count ratio, gene overlap, shift differences

## Notes

- **Multi-segment `end_bases` diagnostic:** The Python `filter_reads()` takes a single `end_bases: str`. The MATLAB `FilterReadsMultiSegment` checks each segment independently with different end_bases: seg1 = "CC" (all probes), seg2 = "AT" (STAR probes) or "TT" (RIBO probes). For now we use "CC" (first segment) for diagnostic. The actual filtering (codebook membership) is unaffected.
- **STAR vs RIBO probes:** Gene names in codebook contain `_STAR_pad_` or `_RIBO_pad_` suffix. Both probe types are present in the codebook. The distinction only matters for end_base diagnostic stats, not for codebook matching.
- **No MATLAB baseline:** No analyzed output exists at `Analyzed/sample-dataset/aging/` for comparison.
- **FOV naming:** The 6 sample FOVs are Position400–405. Full dataset has 848 FOVs.

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `starfinder_benchmark/e2e/results/aging/run_e2e_aging.py` | CREATE | Benchmark script |
| `CLAUDE.md` | DONE | Added aging to real datasets table |
