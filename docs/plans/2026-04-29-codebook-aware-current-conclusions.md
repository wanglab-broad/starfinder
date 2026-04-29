# Codebook-Aware Decoder Current Conclusions

**Date:** 2026-04-29
**Status:** FINISHED

## Scope

Record the current conclusions from the codebook-aware decoder experiments.
This document does not propose or implement any output-logic change.

## Current Decision

Do not change the current output logic yet.

The codebook-aware decoder remains an experimental/diagnostic path. The current
benchmark and QC outputs are useful for evaluating rescued reads, but production
matrix generation should not be changed until a more complete overall test is
designed and reviewed.

## Dataset-Level Conclusions

### Synthetic

Synthetic datasets remain the main ground-truth sanity check. `balanced`
improves useful yield and accuracy in the tested presets without showing wrong
rescues in the current benchmark.

### LN

LN is a useful real-data control. `balanced` improves match rate in both tested
FOVs and keeps the top-10 gene set stable.

### Aging

Aging remains the most informative real dataset for this work because it has a
large codebook and enough reads to expose abundance and rescue artifacts.

Current aging evidence does not support a broad rescued-read artifact: rescued
reads are not dominated by a single gene, a spatial hotspot, edge enrichment,
low intensity, repeated barcode patterns, or codebook-neighbor bias. However,
the `Flt3`/`Mbp` top-10 swap and weaker `Flt3` image evidence remain important
gene-level follow-ups.

### Cell Culture 3D

Cell culture is a useful additional real-data control. The expanded diagnostics
did not raise artifact warnings in the two tested FOVs:

- top rescued gene fraction stayed low;
- top-10 gene set stayed stable;
- rescued reads were not concentrated in a small set of genes.

### Tissue 2D

Tissue 2D should be recorded as an expanded benchmark result, but it should not
drive threshold optimization.

Reasoning:

- the codebook has only 64 genes;
- WTA exact decoding is already relatively high;
- rescued reads are strongly biased toward high-abundance genes;
- `tile_1` shows top-10 instability;
- both tested tiles show strong round4 rescue bias.

This makes tissue useful as a cautionary example, not as the main dataset for
tuning the decoder.

## Current Policy Position

For now:

- keep all current decoder outputs and diagnostic columns unchanged;
- continue saving `call_type`, including `exact`, `rescued_h1`,
  `rescued_unknown`, and `no_call`;
- do not silently merge all rescued reads into production matrices;
- keep `rescued_unknown` separate in interpretation because wildcard uniqueness
  can produce high `score_delta` without strong image evidence.

A future production policy can consider exact plus `rescued_h1` as a candidate
default, but that should wait for a more comprehensive test plan.

## Future Overall Testing Needed

A later full test should evaluate:

1. full-FOV and multi-FOV behavior across aging, LN, cell culture, tissue, and
   synthetic datasets;
2. per-read accuracy on synthetic ground truth;
3. gene abundance stability on real data, including Spearman correlation and
   top-k Jaccard;
4. rescued-read artifact checks: gene concentration, correction-round bias,
   channel-transition bias, spatial hotspots, edge enrichment, intensity, and
   codebook-neighbor bias;
5. confidence stratification using `geomean_prob`, `score_delta`,
   `corrected_round_margin`, and target/WTA intensity evidence;
6. runtime, peak RSS, and cache size;
7. downstream matrix impact, ideally comparing exact-only, exact-plus-h1, and
   all-balanced outputs side by side.

No output logic should be changed before that overall test is specified.
