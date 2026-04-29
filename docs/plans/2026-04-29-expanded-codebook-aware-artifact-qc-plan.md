# Expanded Codebook-Aware Artifact QC Plan

**Date:** 2026-04-29
**Status:** FINISHED

## Scope

Extend the existing codebook-aware rescued-read artifact QC to the newly added
`cell_culture_3D` and `tissue_2D` benchmark datasets.

This plan does not change decoder thresholds. It checks whether `balanced`
rescued reads look plausible across the expanded real datasets, with emphasis
on `tissue_2D tile_1` because the 2026-04-29 benchmark showed a top-10 gene
composition shift there.

## Implementation Tasks

1. Teach diagnostic scripts to infer codebook paths from the codebook-aware
   raw-E2E input metadata for `cell_culture_3D` and `tissue_2D`.
2. Add a registered-stack cache path for raw-E2E datasets so montage scripts can
   render image evidence even when backend-comparison TIFF stacks do not exist.
3. Run rescued-read artifact diagnostics for:
   - `tissue_2D tile_1` and `tile_2`;
   - `cell_culture_3D Position351` and `Position352`.
4. Generate all-round decoding example montages for `tissue_2D tile_1`.
5. Summarize whether rescued reads show single-gene concentration, high-abundance
   bias, correction-round bias, spatial hotspot/edge artifacts, or weak image
   evidence.

## Outputs

Expected outputs are under:

```text
/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/decoding/codebook_aware/results/
```

Diagnostics:

```text
{dataset}/diagnostics/{fov}_balanced/
```

Montage evidence:

```text
tissue_2D/qc/tile_1_balanced_decoding_examples/
```

## Results

Implemented support for raw-E2E datasets in the artifact QC scripts:

- `scripts/diagnose_codebook_aware_rescues.py` can infer codebooks from
  codebook-aware raw input metadata for `cell_culture_3D` and `tissue_2D`.
- `scripts/qc_codebook_aware_rescues.py` and
  `scripts/generate_decoding_example_montages.py` can generate a registered
  stack cache for raw-E2E datasets when backend-comparison TIFF stacks are not
  available.

Diagnostics were generated for:

```text
results/tissue_2D/diagnostics/tile_1_balanced/
results/tissue_2D/diagnostics/tile_2_balanced/
results/cell_culture_3D/diagnostics/Position351_balanced/
results/cell_culture_3D/diagnostics/Position352_balanced/
```

Summary:

| Dataset/FOV | Rescued | Top rescued gene | Top rescued fraction | Top10 rescued fraction | Top10 Jaccard | Dominant round | Spatial hotspot | Warnings |
|-------------|---------|------------------|----------------------|------------------------|---------------|----------------|-----------------|----------|
| tissue `tile_1` | 10,275 | Cst3 | 12.49% | 58.58% | 0.8182 | round4, 57.28% | 1.23x | single gene, top10, one round, lower probability, top10 changed |
| tissue `tile_2` | 6,730 | Cst3 | 10.74% | 61.49% | 1.0000 | round4, 41.61% | 1.44x | single gene, top10, one round, lower probability |
| cell culture `Position351` | 8,746 | RPL37 | 1.90% | 10.50% | 1.0000 | round6, 27.78% | 1.51x | none |
| cell culture `Position352` | 8,213 | TSHZ1 | 2.56% | 12.58% | 1.0000 | round6, 32.65% | 1.59x | none |

Generated tissue `tile_1` image evidence:

```text
results/tissue_2D/qc/tile_1_balanced_decoding_examples/
results/tissue_2D/qc/tile_1_balanced_gene_evidence/
results/tissue_2D/registered_stacks/tile_1/
```

The generic montage set contains 24 all-round PNGs. The gene-level evidence set
contains 72 local round montages and 12 all-round montages focused on
`Cst3`, `Cplx1`, `Calm1`, `Gfap`, and `Ctsb`.

Tissue `tile_1` h1 tensor evidence for top rescued genes:

| Gene | h1 rescued | Median target/WTA intensity | Fraction >=0.75 | Fraction >=0.90 | Median target prob | Median WTA prob |
|------|------------|-----------------------------|-----------------|-----------------|--------------------|-----------------|
| Cst3 | 1,037 | 0.775 | 0.564 | 0.218 | 0.353 | 0.451 |
| Cplx1 | 770 | 0.776 | 0.579 | 0.218 | 0.340 | 0.431 |
| Calm1 | 669 | 0.774 | 0.562 | 0.224 | 0.346 | 0.449 |
| Gfap | 546 | 0.743 | 0.487 | 0.158 | 0.343 | 0.459 |
| Ctsb | 424 | 0.770 | 0.535 | 0.217 | 0.339 | 0.442 |

Interpretation:

- Cell culture rescued reads do not show broad artifact flags in these checks.
- Tissue rescued reads are not spatially edge-enriched and do not form a strong
  XY hotspot, but they are heavily concentrated in highly abundant genes and
  round4 corrections.
- Tissue `tile_1` top10 instability is therefore a real QC concern. Before
  treating `balanced` rescued reads as production calls for tissue, run threshold
  sensitivity with `rescued_unknown` disabled and stricter h1 confidence gates.
