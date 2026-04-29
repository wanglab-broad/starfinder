# Decoding Accuracy Improvement Plan

**Date:** 2026-04-28
**Status:** PROPOSED

## Context

The 2026-04-27/2026-04-28 LR backend comparison shows that dense local
registration is not a reliable decoding-accuracy improvement. It changes color
sequences at fixed reference-round spot coordinates, often reduces
spot-neighborhood intensity, and can create multi-channel ambiguity. The next
round should therefore treat LR as one optional correction source, not as the
main decoding strategy.

## Main Hypothesis

Read loss is driven by fragile per-round winner-take-all color calls after
image warping. Accuracy should improve more from confidence-aware decoding,
per-round selective correction, and barcode-level error correction than from
applying dense LR to every round.

## Priority Experiments

### 1. Add a no-regret decoder baseline before more LR tuning

Use the existing raw intensity tensor path and evaluate codebook-aware decoders
that do not require full Postcode inference:

- nearest-codebook decoding from per-round channel probabilities;
- Hamming-distance rescue for reads one color away from a valid codebook entry;
- margin-gated rescue only when the corrected codebook sequence has high
  intensity support;
- confidence thresholding to separate accuracy from yield.

Primary outputs should include yield, synthetic gene/color accuracy, and
low-confidence/error buckets. On real data without ground truth, report
distribution stability, top-gene overlap, and agreement with the current
global-only decoder.

### 2. Make LR selective, not all-or-nothing

Evaluate LR per round and optionally per spatial block. Decode from the
global-only stack unless LR passes local acceptance gates:

- merged-image NCC does not decrease beyond a small threshold;
- spot-neighborhood intensity median is preserved;
- fraction of spots losing more than 25% intensity stays low;
- color-call disagreement against global-only is not excessive;
- nonzero voxel fraction near boundaries is preserved.

This should first be tested on LN `Position002` and aging `Position400`, since
they show opposite LR backend behavior and clear failure signatures.

### 3. Replace dense demons with spot-preserving correction candidates

Dense demons should be compared against methods that constrain the deformation
at the places decoding actually uses:

- TPS/CPD from matched bright spots, with displacement outlier rejection;
- low-degree block-polynomial or B-spline fields with displacement clipping;
- local translation-only block correction instead of dense free-form warping;
- displacement-field smoothing plus Jacobian/fold and boundary masks.

The evaluation unit should be spot-level color stability and codebook recovery,
not only image-level registration metrics.

### 4. Improve extraction before changing registration further

Run small ablations on the current extraction step:

- test neighborhood radius per dataset and per z/xy anisotropy;
- compare sum pooling, max pooling, weighted Gaussian pooling, and small
  subpixel recentering around the reference spot;
- preserve raw channel intensities and add channel/round normalization before
  argmax;
- use channel margin and total intensity as explicit quality scores.

These are cheaper than full LR and directly target the observed intensity-loss
and multi-max failure mode.

### 5. Continue Postcode only after making it computationally practical

Postcode improves synthetic top-1 accuracy and high-confidence accuracy, but
the aging smoke run is too memory-heavy. Before full aging runs, implement a
chunked/top-k probability path or a simpler local probabilistic decoder that
uses the same codebook likelihood idea without storing an `N x K` matrix.

## Suggested Order

1. Build the lightweight nearest-codebook and one-error rescue decoder on top
   of `extract_intensity_tensor()`.
2. Run it on synthetic medium/large, LN `Position002`, and aging `Position400`
   using global-only stacks.
3. Add confidence/yield curves and compare against current deterministic
   winner-take-all filtering.
4. Add per-round LR acceptance gates and test hybrid decoding that chooses
   global-only or LR per round.
5. Only then tune or replace the LR algorithm.

## Success Criteria

For synthetic data, a method is promising only if it increases recall without
lowering gene accuracy beyond an agreed threshold. For real data, require both
higher codebook match rate and stable gene-distribution metrics; extra reads
that strongly shift the abundance distribution should be treated as suspect
until validated.
