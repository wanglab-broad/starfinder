# Project Status and Plan

**Date:** 2026-04-27
**Status:** FINISHED

## Current Status

The core STARfinder Python backend is in a mature state relative to the
Milestone 2 plan:

- Snakemake 9 modularization is finished.
- Python backend phases 0-9 are finished: I/O, registration, spot finding,
  barcode processing, preprocessing, dataset/FOV orchestration, synthetic E2E
  tests, real-data E2E benchmarks, and memory/performance optimization.
- Recent completed work added the Snakemake Python backend wrappers and aging
  E2E support.
- The current active line of work is decoder evaluation, specifically whether
  Postcode is useful enough to justify a future pipeline integration.

Working tree at analysis time:

- Modified:
  - `src/python/starfinder/barcode/extraction.py`
  - `src/python/starfinder/barcode/__init__.py`
  - `src/python/test/test_extraction.py`
  - `docs/temp.md`
- Untracked:
  - `docs/plans/2026-04-21-postcode-decoding-benchmark-plan.md`

The uncommitted source changes implement and export
`extract_intensity_tensor()`, which returns raw `(N, C, R)` neighborhood-summed
intensities for decoder benchmarking. The focused extraction test passes:

```bash
cd src/python
uv run pytest test/test_extraction.py -v
# 9 passed in 0.32s
```

## Postcode Benchmark Status

Benchmark work exists under:

```text
/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/decoding/postcode/
```

Implemented there:

- isolated Postcode environment files;
- synthetic and aging benchmark drivers;
- shared helpers for codebook one-hot conversion, STARfinder tensor decoding,
  Postcode output conversion, aging pad/base-gene aggregation, and comparison
  metrics;
- synthetic medium benchmark outputs;
- aging `Position400` 5k-spot smoke output;
- a generated HTML report.

Synthetic medium result summary:

- FOV_001: STARfinder accuracy 0.675, Postcode top1 accuracy 0.715,
  Postcode p80 accuracy 0.996.
- FOV_002: STARfinder accuracy 0.748, Postcode top1 accuracy 0.908,
  Postcode p80 accuracy 1.000.
- Postcode increases assigned calls and high-confidence calls look very clean
  on synthetic data.

Aging smoke result summary:

- `Position400` had 682,811 detected spots.
- Smoke input was limited to 5,000 spots.
- STARfinder assigned 3,015 / 5,000.
- Postcode assigned 4,945 / 5,000.
- Agreement rate was 0.578.
- Base-gene distribution Spearman correlation was 0.749.
- Top-10 gene Jaccard was 0.667.
- Runtime was 445.6 seconds.
- Peak RSS was 40,960 MB.

This is the main blocker: a full aging FOV is not practical with the current
Postcode execution strategy. The 50k aging smoke and 2-FOV production run
should stay blocked until memory is reduced. GPU aging attempts produced input
artifacts but no completed comparison CSV, so they are not yet a successful
result.

## Risks and Decisions

1. Do not run full aging Postcode decoding with the current strategy. The 5k
   smoke already reached about 40 GB peak RSS, so scaling to hundreds of
   thousands of spots is unsafe.
2. Treat synthetic success as evidence that tensor ordering and codebook
   encoding are probably correct, not as proof that Postcode is production
   ready for aging.
3. The next useful experiment is not another blind full run. It is a memory
   strategy test: chunked spot batches, compact top-k output, or a modification
   that avoids materializing the full `N x K` posterior matrix.
4. Keep `extract_intensity_tensor()` in the STARfinder package if decoder
   benchmarking continues. It is small, tested, and does not alter the existing
   pipeline path.

## Today's Plan

Scope update after review: no additional Postcode benchmark runs today. The
near-term goal is to commit the repo-side helper and document the current
insights. Chunked aging Postcode decoding remains the next experiment, but it
is deferred.

### 1. Close the repo-side helper change

- Review `extract_intensity_tensor()` for API details and edge cases:
  empty `round_order`, empty spots, missing rounds, non-4D inputs, channel
  mismatch.
- Decide whether to add one more unit test for the empty/missing-round cases.
  Keep this minimal.
- Run:

```bash
cd src/python
uv run pytest test/test_extraction.py -v
```

- If the helper is accepted, stage the source/test files and the April 21 plan.
  Keep `docs/temp.md` separate unless the notes are intentionally part of the
  commit.

### 2. Make the benchmark status explicit

- Update the Postcode benchmark README or report with the interpretation:
  synthetic looks promising, aging full-FOV is blocked by memory.
- Record that GPU synthetic is not clearly beneficial at this scale because
  overhead dominates small inputs.
- Record that aging GPU runs are incomplete until a comparison file exists.

### 3. Design the next aging feasibility experiment

Primary experiment:

- Reuse saved `Position400` tensor/spots/starfinder inputs.
- Run Postcode on spot chunks, starting with 1,000 and 5,000 spot chunks.
- Save compact per-spot output only:
  `spot_id`, `top_gene`, `top_prob`, `class_label`, and optionally top-k genes.
- Avoid saving full probability matrices for aging.
- Measure wall time, peak RSS, and whether chunked calls match the existing
  5k smoke output on the same first 5k spots.

Exit criteria:

- Peak RSS under 16 GB for a 5k chunk, or clear evidence that memory is dominated
  by model size rather than batch size.
- Agreement with the existing 5k run close enough to confirm chunking does not
  change calls materially.
- A scaling estimate for one full aging FOV.

### 4. Only then decide on full aging scope

Proceed to a larger run only if chunking makes memory predictable:

- 50k `Position400` run first.
- Full `Position400` run only after the 50k run is acceptable.
- `Position400` plus `Position401` only after the single-FOV run has a clear
  runtime and memory budget.

## End-of-Day Target

End state for today:

- repo helper change tested and committed;
- current synthetic and aging Postcode conclusions written down;
- no additional Postcode test run started;
- chunked-aging feasibility work deferred until the next decoder session.
