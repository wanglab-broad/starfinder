# Codebook-Aware Decoder Implementation And Test Plan

**Date:** 2026-04-28
**Status:** FINISHED

## Scope

Implement and benchmark a lightweight codebook-aware decoder that consumes the
same per-spot raw intensity tensor already used by the Postcode benchmark. This
plan intentionally does not modify local registration. All first-pass
experiments should use global-only registered stacks.

The goal is to improve decoded-read accuracy and useful yield beyond the
current winner-take-all exact codebook lookup while avoiding Postcode's aging
memory cost.

### 2026-04-29 Dataset Expansion

Add `cell_culture_3D` and `tissue_2D` to the same benchmark runner. These
datasets do not currently have backend-comparison all-spots/tensor artifacts,
so the runner should fall back to the existing real-data Python E2E recipe:

1. load raw round images with the dataset-specific channel order;
2. rotate, enhance, and run global registration;
3. detect all reference-round spots with the same adaptive threshold used by
   the E2E benchmark;
4. extract all-read raw intensity tensors and cache them under the codebook
   aware result directory;
5. run `wta_exact` and `balanced` on the full read set, recording decode time,
   input preparation time, current RSS, peak RSS, rescued counts, match-rate
   gain, abundance correlation, and top-10 Jaccard.

The expansion should not change the LR strategy or the decoder thresholds. Its
purpose is to check whether rescued reads remain plausible across two additional
real datasets with different round counts, codebook sizes, tissue types, and
spot densities.

## Current Baseline

The current STARfinder decoder does:

1. sum per-channel intensities in a fixed voxel neighborhood;
2. L2-normalize each spot's channel vector per round;
3. choose the argmax channel per round;
4. concatenate color digits into `color_seq`;
5. keep only exact codebook matches.

This is fast and MATLAB-compatible, but it discards any read whose color
sequence has one uncertain or wrong round. The new decoder should preserve this
baseline as `wta_exact` and add confidence-gated rescue modes.

## Design Principles

- Keep the implementation deterministic and NumPy/Pandas-only.
- Do not allocate an `N x K` probability matrix for large real datasets.
- Decode unique observed color sequences when possible, then map results back
  to spots.
- Use synthetic ground truth to set thresholds; do not tune thresholds only by
  increasing real-data match rate.
- Prefer no-call over aggressive rescue when confidence is weak.
- Preserve exact STARfinder parity as a testable baseline.

## Files

### Package Code

Create:

```text
src/python/starfinder/barcode/codebook_aware.py
```

Export from:

```text
src/python/starfinder/barcode/__init__.py
```

Core public functions:

```python
def decode_codebook_aware(
    intensity_tensor: np.ndarray,
    seq_to_gene: dict[str, str],
    *,
    spot_ids: np.ndarray | list | None = None,
    max_hamming: int = 1,
    unknown_chars: str = "MN",
    min_corrected_round_margin: float | None = 0.20,
    min_score_delta: float = 0.25,
    min_geomean_prob: float = 0.45,
    max_correction_penalty: float = 1.50,
    allow_exact: bool = True,
    allow_rescue: bool = True,
) -> pd.DataFrame:
    """Decode `(N, C, R)` intensities using exact and confidence-gated
    codebook-aware rescue."""
```

Supporting functions should stay small and directly unit-tested:

```python
def channel_probabilities(intensity_tensor, eps=1e-6) -> np.ndarray
def wta_color_sequences(probs) -> tuple[np.ndarray, pd.DataFrame]
def build_one_error_index(seq_to_gene, n_channels, n_rounds) -> dict[str, list[str]]
def candidate_sequences(wta_seq, one_error_index, seq_to_gene, unknown_chars="MN") -> list[str]
def score_candidates(probs_for_spot, candidates) -> pd.DataFrame
```

### Tests

Create:

```text
src/python/test/test_codebook_aware_decoder.py
```

### Benchmark Scripts

Prefer a repo-local benchmark driver first:

```text
src/python/test/test_codebook_aware_decoder.py          # unit/integration smoke
```

Then create an external benchmark harness when ready:

```text
/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/decoding/codebook_aware/
├── README.md
├── scripts/
│   ├── common.py
│   ├── run_decoding_synthetic.py
│   └── run_decoding_real.py
└── results/
```

The external harness should reuse code from the existing Postcode benchmark
where possible: intensity tensor extraction, `strip_aging_pad_suffix`, synthetic
`compare_genes()`, and real-data distribution metrics.

## Decoder Algorithm

### 1. Convert Intensities To Per-Round Channel Probabilities

Input shape is `(N, C, R)`.

For each spot and round:

1. cast to `float64`;
2. clip negative values to zero;
3. subtract the per-spot-round minimum only if this improves zero-background
   robustness in tests;
4. compute `p[c] = (v[c] + eps) / sum(v + eps)`;
5. if total signal is zero or non-finite, mark the round as unknown.

Store per-round diagnostics:

- `round_top_channel`;
- `round_top_prob`;
- `round_second_prob`;
- `round_margin = top_prob - second_prob`;
- `round_total_intensity`.

### 2. Preserve Exact Baseline

Construct the WTA color sequence. If it is an exact key in `seq_to_gene`, return
that gene with:

- `call_type = "exact"`;
- `hamming_to_wta = 0`;
- `confidence = geometric_mean(prob_of_called_digits)`;
- `score = sum(-log(prob_of_called_digits))`.

This should match current STARfinder output exactly except for extra diagnostic
columns.

### 3. Candidate Generation Without Full `N x K`

Use an index over codebook sequences:

- exact index: `seq -> [seq]`;
- one-error index: for each codebook sequence, generate all sequences that
  differ at exactly one round and map them back to the valid codebook sequence.

Example:

```text
codebook seq 4422, C=4
one-error keys include 1422, 2422, 3422, 4122, ...
```

For an invalid WTA sequence:

1. if it contains only digits, fetch candidates from the one-error index;
2. if it contains `M` or `N`, treat those positions as wildcards only when the
   number of unknown positions is `<= max_hamming`;
3. reject immediately if no candidates are found;
4. reject if too many candidates remain after scoring and the score gap is too
   small.

The first implementation should support `max_hamming=1` for real data.
`max_hamming=2` can be enabled only for synthetic upper-bound experiments.

### 4. Candidate Scoring

For each candidate codebook sequence `s`, compute:

```text
score(s) = sum_r -log(p[channel(s_r), r] + eps)
geomean_prob(s) = exp(-score(s) / R)
```

Pick the lowest-score candidate. Also compute:

```text
score_delta = second_best_score - best_score
correction_penalty = score(best_candidate) - score(wta_digits_with_unknowns_skipped)
```

For one corrected round, also record:

- corrected round index;
- WTA digit;
- candidate digit;
- WTA margin at corrected round;
- candidate digit probability at corrected round.

### 5. Rescue Gates

Only emit a rescued gene when all selected gates pass:

- candidate is within `max_hamming` of the observed WTA sequence;
- `score_delta >= min_score_delta`;
- `geomean_prob >= min_geomean_prob`;
- `correction_penalty <= max_correction_penalty`;
- if a digit is changed, the corrected round's WTA margin is not too strong
  (`round_margin <= min_corrected_round_margin`);
- if multiple candidates map to different genes with similar scores, return
  no-call.

Recommended initial thresholds for grid search:

| Parameter | Values |
|-----------|--------|
| `max_hamming` | `0`, `1` |
| `min_score_delta` | `0.10`, `0.25`, `0.50`, `1.00` |
| `min_geomean_prob` | `0.35`, `0.45`, `0.55`, `0.65` |
| `max_correction_penalty` | `0.75`, `1.50`, `2.50` |
| `min_corrected_round_margin` | `0.10`, `0.20`, `0.30`, `None` |

### 6. Output Schema

Return one row per input spot:

| Column | Meaning |
|--------|---------|
| `spot_id` | Stable spot id from input order or caller |
| `color_seq_wta` | Raw WTA sequence before rescue |
| `gene_wta` | Exact STARfinder gene if WTA is in codebook |
| `decoded_seq` | Final codebook sequence after exact/rescue/no-call |
| `gene` | Final assigned gene or null |
| `call_type` | `exact`, `rescued_h1`, `rescued_unknown`, `no_call` |
| `reject_reason` | Empty for assigned reads, otherwise diagnostic reason |
| `hamming_to_wta` | Number of changed known WTA digits |
| `corrected_rounds` | Comma-separated 0-based round indices |
| `score` | Best codebook negative log score |
| `score_delta` | Second-best minus best score |
| `geomean_prob` | Geometric mean probability of called digits |
| `min_round_margin` | Minimum WTA margin across rounds |
| `corrected_round_margin` | WTA margin on corrected round if any |
| `mean_total_intensity` | Mean raw neighborhood intensity across rounds |

For exact parity, `gene_wta` should match current `filter_reads()` output.

## Codebook Collision Audit

Before decoding, group `gene_to_seq` or `seq_to_gene` by color sequence and
report:

- number of unique color sequences;
- number of duplicate color sequences;
- duplicate genes per sequence;
- aging base-gene duplicates after stripping pad suffixes.

The first implementation can preserve current `seq_to_gene` behavior for exact
pipeline parity, but benchmark reports should flag collisions because a rescued
sequence with multiple possible genes is not a high-confidence unique call.

## Unit Test Plan

Run from `src/python`:

```bash
uv run pytest test/test_codebook_aware_decoder.py -v
```

Tests:

1. `test_channel_probabilities_sum_to_one`
   - simple `(N, C, R)` tensor;
   - probabilities sum to one per spot-round;
   - zero-signal round is handled deterministically.

2. `test_wta_exact_matches_filter_reads`
   - use a small codebook and clean one-hot-like intensities;
   - `decode_codebook_aware(..., allow_rescue=False)` matches
     `run_starfinder_decode()` or `filter_reads()`.

3. `test_one_error_index_returns_expected_candidate`
   - codebook sequence `4422`;
   - observed `4322` returns `4422`;
   - observed sequence two edits away is not returned when `max_hamming=1`.

4. `test_low_margin_one_error_is_rescued`
   - WTA picks wrong channel in one round by a tiny margin;
   - true codebook candidate has strong total likelihood;
   - decoder emits `call_type="rescued_h1"`.

5. `test_high_margin_wrong_sequence_is_rejected`
   - WTA wrong sequence is confidently wrong in the changed round;
   - rescue candidate exists but `corrected_round_margin` gate rejects it.

6. `test_ambiguous_candidates_are_rejected`
   - two codebook candidates have near-identical scores;
   - decoder returns no-call with `reject_reason="ambiguous_candidate"`.

7. `test_unknown_round_wildcard_rescue`
   - one round has tie/zero signal and WTA digit is `M` or `N`;
   - exactly one codebook candidate fits known rounds;
   - decoder rescues with `call_type="rescued_unknown"`.

8. `test_unknown_round_multiple_candidates_rejected`
   - wildcard produces multiple plausible genes;
   - decoder returns no-call.

9. `test_output_schema_for_empty_input`
   - empty tensor returns empty DataFrame with expected columns.

10. `test_invalid_shapes_raise`
    - non-3D tensor, mismatched channel count, and malformed sequence lengths
      raise clear `ValueError`s.

## Integration Test Plan

### Synthetic Smoke

Use the synthetic medium data already generated for Postcode:

```bash
cd /home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/decoding/codebook_aware
uv run python scripts/run_decoding_synthetic.py --preset medium --smoke
```

Compare:

- `wta_exact`;
- `h1_rescue_conservative`;
- `h1_rescue_balanced`;
- optional `h2_rescue_synthetic_only`.

Expected:

- `wta_exact` exactly matches current STARfinder metrics;
- conservative rescue should not reduce synthetic gene accuracy;
- balanced rescue may increase recall, but wrong-rescue rate must be reported;
- all runs finish without GPU or special env.

### Synthetic Full

Run:

```bash
uv run python scripts/run_decoding_synthetic.py --preset medium
uv run python scripts/run_decoding_synthetic.py --preset large
```

Primary metrics:

| Metric | Meaning |
|--------|---------|
| `n_input_spots` | all detected spots |
| `wta_n_assigned` | exact STARfinder assigned reads |
| `cba_n_assigned` | codebook-aware assigned reads |
| `n_rescued` | assigned by rescue only |
| `gene_accuracy` | on matched synthetic spots |
| `color_seq_accuracy` | on matched synthetic spots |
| `recall` | correct genes / GT spots |
| `wrong_rescue_rate` | rescued matched spots with wrong gene |
| `confidence_auc` | area under confidence-yield curve if implemented |

Decision rule:

- Accept a threshold set only if it increases recall while keeping gene
  accuracy within 0.5 percentage points of `wta_exact`, or if it provides a
  useful high-confidence rescued subset with gene accuracy >= 99%.

### Real Global-Only Smoke

Use existing global-only artifacts first:

- LN `Position001`;
- LN `Position002`;
- aging `Position400`, starting with a 5k or 50k spot subset.

Run:

```bash
uv run python scripts/run_decoding_real.py --dataset LN --fovs Position001 Position002 --variants global_only
uv run python scripts/run_decoding_real.py --dataset aging --fovs Position400 --spot-limit 5000
```

Real-data metrics:

| Metric | Meaning |
|--------|---------|
| `wta_match_rate` | current exact codebook assignment rate |
| `cba_match_rate` | final codebook-aware assignment rate |
| `n_rescued` | additional assigned spots |
| `rescue_rate` | rescued / input |
| `exact_agreement_rate` | exact calls unchanged |
| `rescued_top_gene_fraction` | concentration of rescued calls in top genes |
| `base_gene_distribution_corr` | Spearman count correlation vs WTA |
| `top10_gene_jaccard` | top-10 overlap vs WTA |
| `per_round_correction_counts` | which rounds are being corrected |
| `rescue_margin_histogram` | whether rescues come from low-margin rounds |

Decision rule:

- Real data must show stable abundance distribution, not just higher match
  rate.
- If rescued calls are dominated by a small number of genes or a single
  corrected round, inspect those reads before expanding.

## Benchmark Output Layout

```text
results/{dataset}/
├── intensity_tensors/{fov}.npy
├── spots/{fov}.csv
├── starfinder/{fov}.csv
├── codebook_aware/{config_name}/{fov}.csv
├── log/{fov}_{config_name}.csv
├── comparison.csv
└── comparison_summary.json
```

`comparison.csv` should have one row per FOV and decoder config. Save the exact
thresholds used in every row.

## Implementation Phases

### Phase 1: Core Decoder And Unit Tests

1. Add `starfinder.barcode.codebook_aware`.
2. Implement probability conversion, WTA parity, one-error index, candidate
   scoring, and rescue gates.
3. Add unit tests listed above.
4. Run:

   ```bash
   cd src/python
   uv run pytest test/test_codebook_aware_decoder.py -v
   uv run pytest test/test_extraction.py test/test_barcode.py -v
   ```

### Phase 2: Synthetic Benchmark

1. Build synthetic runner using existing generated Postcode synthetic medium
   data.
2. Reuse `extract_intensity_tensor()` and `compare_genes()`.
3. Run threshold grid on medium, then large.
4. Save confidence-yield curves and rescued-read examples.

### Phase 3: Real Global-Only Benchmark

1. Run LN `Position001` and `Position002`.
2. Run aging `Position400` with a subset first.
3. Expand aging only if runtime and distribution metrics look sane.
4. Compare against Postcode smoke results where available, but do not require
   Postcode to run.

### Phase 4: Integrate As Optional Decoder

Only after benchmark review:

1. Add a dataset/FOV-level option to choose `decoder: wta_exact|codebook_aware`.
2. Keep `wta_exact` as default.
3. Write both exact and rescued outputs so downstream users can filter by
   `call_type` and confidence.

## Risks And Controls

- Risk: higher match rate from false-positive rescue.
  Control: require synthetic wrong-rescue metrics and real abundance stability.

- Risk: aging pad variants make apparent gene rescue ambiguous.
  Control: report both pad-level and base-gene-level metrics.

- Risk: thresholds overfit synthetic medium.
  Control: validate on synthetic large and at least two real datasets before
  considering pipeline integration.

- Risk: one-error rescue misses multi-round weak reads.
  Control: first measure conservative gains; use `max_hamming=2` only as a
  synthetic upper bound before considering real data.

## Completion Criteria

This plan is complete when:

1. `wta_exact` parity tests pass;
2. unit tests cover exact, rescue, ambiguous, unknown, and invalid inputs;
3. synthetic medium and large benchmark results are saved;
4. LN and aging smoke benchmark results are saved;
5. a short results document recommends either:
   - keep `wta_exact`;
   - add `codebook_aware` as an optional high-confidence decoder;
   - continue with a different probabilistic decoder design.

## Progress Log

### 2026-04-28 Phase 1

Implemented the core decoder in `starfinder.barcode.codebook_aware` and added
unit tests covering exact parity, one-error rescue, high-margin rejection,
ambiguous candidates, unknown-round wildcard rescue, empty input, and invalid
input handling.

Verification:

```bash
cd src/python
uv run pytest test/test_codebook_aware_decoder.py test/test_extraction.py test/test_barcode.py -v
uv run pytest test/ -v
```

Results:

- targeted barcode/extraction tests: 40 passed;
- full Python test suite: 207 passed, 11 warnings.

Read-only smoke checks using existing Postcode benchmark intensity tensors:

Synthetic medium:

| FOV | Config | Exact | Rescued | Assigned | WTA accuracy | CBA accuracy | WTA recall | CBA recall | Rescued accuracy |
|-----|--------|-------|---------|----------|--------------|--------------|------------|------------|------------------|
| FOV_001 | conservative | 306 | 0 | 306 | 0.6750 | 0.6750 | 0.6750 | 0.6750 | n/a |
| FOV_001 | balanced | 306 | 28 | 334 | 0.6750 | 0.7450 | 0.6750 | 0.7450 | 1.0000 |
| FOV_001 | permissive | 306 | 37 | 343 | 0.6750 | 0.7650 | 0.6750 | 0.7650 | 0.9730 |
| FOV_002 | conservative | 348 | 0 | 348 | 0.7475 | 0.7475 | 0.7475 | 0.7475 | n/a |
| FOV_002 | balanced | 348 | 85 | 433 | 0.7475 | 0.9425 | 0.7475 | 0.9425 | 1.0000 |
| FOV_002 | permissive | 348 | 88 | 436 | 0.7475 | 0.9500 | 0.7475 | 0.9500 | 1.0000 |

Aging `Position400`, 5,000-spot smoke:

| Config | Exact | Rescued | Assigned | Match rate | Base-gene Spearman vs WTA | Top10 Jaccard | Rescue top-gene fraction |
|--------|-------|---------|----------|------------|----------------------------|---------------|--------------------------|
| conservative | 3,015 | 301 | 3,316 | 0.6632 | 0.9618 | 1.0000 | 0.0399 |
| balanced | 3,015 | 724 | 3,739 | 0.7478 | 0.9108 | 1.0000 | 0.0359 |
| permissive | 3,015 | 1,071 | 4,086 | 0.8172 | 0.8784 | 0.8182 | 0.0308 |

Initial interpretation: the balanced setting is the best next benchmark
candidate. It improves synthetic recall without observed wrong rescues in this
smoke run and increases aging match rate while keeping top-10 base genes stable.
The permissive setting may be useful as an upper bound, but its aging
distribution drift is already visible.

### 2026-04-28 Phase 2/3 Saved Benchmark

Added a reusable benchmark runner:

```text
src/python/scripts/run_codebook_aware_benchmark.py
```

Command:

```bash
cd src/python
uv run python scripts/run_codebook_aware_benchmark.py
```

Outputs:

```text
/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/decoding/codebook_aware/results/comparison_all.csv
/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/decoding/codebook_aware/results/synthetic_medium/comparison.csv
/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/decoding/codebook_aware/results/aging/comparison.csv
```

Synthetic medium benchmark:

| FOV | Config | Assigned | Rescued | Match rate | Gene accuracy | Recall | Rescued accuracy |
|-----|--------|----------|---------|------------|---------------|--------|------------------|
| FOV_001 | wta_exact | 306 | 0 | 0.6815 | 0.6750 | 0.6750 | n/a |
| FOV_001 | balanced | 334 | 28 | 0.7439 | 0.7450 | 0.7450 | 1.0000 |
| FOV_001 | permissive | 343 | 37 | 0.7639 | 0.7650 | 0.7650 | 0.9730 |
| FOV_002 | wta_exact | 348 | 0 | 0.7615 | 0.7475 | 0.7475 | n/a |
| FOV_002 | balanced | 433 | 85 | 0.9475 | 0.9425 | 0.9425 | 1.0000 |
| FOV_002 | permissive | 436 | 88 | 0.9540 | 0.9500 | 0.9500 | 1.0000 |

Aging `Position400`, 5,000-spot benchmark:

| Config | Assigned | Rescued | Match rate | Base-gene Spearman vs WTA | Top10 Jaccard | Rescued top-gene fraction |
|--------|----------|---------|------------|----------------------------|---------------|--------------------------|
| wta_exact | 3,015 | 0 | 0.6030 | 1.0000 | 1.0000 | 0.0000 |
| conservative | 3,316 | 301 | 0.6632 | 0.9618 | 1.0000 | 0.0399 |
| balanced | 3,739 | 724 | 0.7478 | 0.9108 | 1.0000 | 0.0359 |
| permissive | 4,086 | 1,071 | 0.8172 | 0.8784 | 0.8182 | 0.0308 |

Current recommendation: carry `balanced` forward for larger benchmarks. It gives
the strongest synthetic gain without observed wrong rescues in this dataset and
keeps the aging top-10 base-gene set stable. Keep `permissive` as an upper-bound
diagnostic only because aging abundance distribution starts to drift.

### 2026-04-28 Expanded Full-Read Benchmark

Updated `scripts/run_codebook_aware_benchmark.py` to:

- generate and cache full-read tensors for synthetic presets from raw images;
- generate and cache full-read tensors for real datasets from Python
  global-only backend-comparison registered stacks;
- record decoder wall time and process RSS/peak RSS metrics in every row.

Command set:

```bash
cd src/python
uv run python scripts/run_codebook_aware_benchmark.py --datasets synthetic_large --configs wta_exact balanced
uv run python scripts/run_codebook_aware_benchmark.py --datasets LN --configs wta_exact balanced
uv run python scripts/run_codebook_aware_benchmark.py --datasets aging --configs wta_exact balanced
```

Combined output:

```text
/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/decoding/codebook_aware/results/comparison_all.csv
```

All expanded runs used all detected reads available from the selected input
source, not a 5,000-read subset.

| Dataset/FOV | Config | Input reads | Assigned | Rescued | Match rate | Accuracy/Distribution check | Decode time | Peak RSS |
|-------------|--------|-------------|----------|---------|------------|-----------------------------|-------------|----------|
| synthetic_large `FOV_001` | wta_exact | 2,026 | 1,756 | 0 | 0.8667 | gene acc 0.8660 | 0.085s | 1.82 GB |
| synthetic_large `FOV_001` | balanced | 2,026 | 1,799 | 43 | 0.8880 | gene acc 0.8875, rescued acc 1.0000 | 0.130s | 1.82 GB |
| synthetic_large `FOV_002` | wta_exact | 2,013 | 1,765 | 0 | 0.8768 | gene acc 0.8774 | 0.081s | 1.82 GB |
| synthetic_large `FOV_002` | balanced | 2,013 | 2,001 | 236 | 0.9940 | gene acc 0.9950, rescued acc 1.0000 | 0.264s | 1.82 GB |
| LN `Position001` | wta_exact | 11,887 | 10,466 | 0 | 0.8805 | base-gene rho 1.0000 | 0.479s | 2.32 GB |
| LN `Position001` | balanced | 11,887 | 11,263 | 797 | 0.9475 | base-gene rho 0.9959, top10 1.0000 | 1.533s | 2.32 GB |
| LN `Position002` | wta_exact | 13,005 | 10,114 | 0 | 0.7777 | base-gene rho 1.0000 | 0.513s | 2.34 GB |
| LN `Position002` | balanced | 13,005 | 11,047 | 933 | 0.8494 | base-gene rho 0.9968, top10 1.0000 | 2.900s | 2.34 GB |
| aging `Position400` | wta_exact | 682,811 | 182,264 | 0 | 0.2669 | base-gene rho 1.0000 | 33.495s | 7.75 GB |
| aging `Position400` | balanced | 682,811 | 289,118 | 106,854 | 0.4234 | base-gene rho 0.9773, top10 0.8182 | 529.250s | 7.75 GB |

Tensor cache sizes:

- synthetic_large: ~0.25 MB per FOV;
- LN: ~1.5-1.6 MB per FOV;
- aging `Position400`: 188 MB.

Expanded interpretation:

- `balanced` remains strong on synthetic and LN. It improves synthetic accuracy
  and recall, and on LN improves match rate while keeping abundance structure
  very close to WTA exact.
- Full aging is computationally feasible but slow: balanced decode is about
  8.8 minutes for 682,811 reads, with peak RSS about 7.75 GB including tensor
  preparation. It rescues many reads and keeps high overall base-gene Spearman,
  but top-10 Jaccard drops to 0.8182. This needs gene-level inspection before
  treating full-aging rescued reads as production-quality calls.

### 2026-04-28 Rescued-Read Artifact Diagnostics

Added a reusable diagnostic script:

```text
src/python/scripts/diagnose_codebook_aware_rescues.py
```

Commands:

```bash
cd src/python
uv run python scripts/diagnose_codebook_aware_rescues.py --dataset aging --config balanced --baseline-config wta_exact --fovs Position400
uv run python scripts/diagnose_codebook_aware_rescues.py --dataset LN --config balanced --baseline-config wta_exact --fovs Position001 Position002
uv run python scripts/diagnose_codebook_aware_rescues.py --dataset synthetic_large --config balanced --baseline-config wta_exact --fovs FOV_001 FOV_002 --min-bin-reads 20
```

Primary aging output:

```text
/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/decoding/codebook_aware/results/aging/diagnostics/Position400_balanced/
```

The diagnostic output includes:

- `summary.json`
- `gene_rescue_enrichment.csv`
- `top10_shift.csv`
- `round_corrections.csv`
- `correction_transitions.csv`
- `rescue_sequence_patterns.csv`
- `confidence_by_call_type.csv`
- `gate_sensitivity.csv`
- `spatial_bins.csv`
- `z_slices.csv`
- `edge_summary.csv`
- `codebook_neighbor_bias.csv`
- `rescued_examples.csv`

Aging `Position400` balanced artifact checks:

| Check | Result | Interpretation |
|-------|--------|----------------|
| Single-gene concentration | top rescued gene `Penk`, 1,657 reads, 1.55% of rescued | no single-gene dominance |
| Top-10 rescued concentration | 10.52% | not concentrated in a small gene set |
| Dominant correction round | round 1, 15.71% | no single-round collapse |
| Dominant channel transition | round1 `4->3`, 2.99% | no transition-level artifact |
| Repeated WTA-to-decoded sequence | top pattern 0.27% | no repeated barcode artifact |
| XY edge enrichment | edge 13.38% vs interior 15.88% | rescued reads are not edge-enriched |
| Spatial hotspot | max 8x8 bin 18.56%, 1.19x global | no strong local hotspot |
| Intensity | rescued median 2,338 vs exact 2,349 | rescued reads are not low-intensity artifacts |
| Probability | rescued median geomean 0.558 vs exact 0.706 | expected lower confidence than exact calls |
| Codebook neighbor bias | Spearman rescued vs neighbor slots 0.147 | rescues do not track codebook catchment size |

The only abundance warning is a one-gene top-10 swap: `Flt3` enters and `Mbp`
exits. This is sensitive because baseline WTA exact counts are close
(`Mbp` rank 10 with 1,559 reads; `Flt3` rank 11 with 1,537 reads). The swap
persists under stricter post-hoc gates in `gate_sensitivity.csv`, so it should
be treated as a gene-level follow-up rather than a broad rescued-read artifact.
`Flt3` does not have an unusually large codebook one-error neighborhood
(`neighbor_slots=150`, below the 192 maximum tier), so the current evidence does
not support codebook-neighborhood bias as the cause.

LN and synthetic controls show why these flags should be interpreted jointly,
not independently. LN has much stronger concentration in `Cd74` and specific
rounds, but its top-10 gene set remains unchanged in both FOVs. Synthetic large
can trigger round/spatial/top10 warnings despite 100% rescued accuracy in the
benchmark, because rescued counts are small and ground-truth abundance ranks are
sensitive. For aging, the stronger evidence is therefore the absence of
single-gene, spatial, round, transition, repeated-sequence, intensity, and
codebook-catchment artifacts.

Current recommendation: keep `balanced` as a diagnostic decoder and allow users
to filter on `call_type`. Before making rescued reads part of the default
production output, inspect `Flt3` examples and consider reporting exact-only and
exact-plus-rescued matrices side by side.

Verification after adding diagnostics:

```bash
cd src/python
uv run ruff check scripts/diagnose_codebook_aware_rescues.py starfinder/barcode/codebook_aware.py starfinder/barcode/__init__.py test/test_codebook_aware_decoder.py
XDG_CACHE_HOME=/tmp/xdg-cache uv run pytest test/ -v
```

Result: targeted ruff passed; pytest passed with 207 tests and 11 existing
warnings. A repository-wide ruff run still fails on pre-existing lint in
benchmark/test modules outside this change.

### 2026-04-28 Gene-Level Image Evidence QC

Added a reusable QC script for rescued-read image evidence:

```text
src/python/scripts/qc_codebook_aware_rescues.py
```

Main command:

```bash
cd src/python
MPLCONFIGDIR=/tmp/matplotlib-cache UV_CACHE_DIR=/tmp/uv-cache \
  uv run python scripts/qc_codebook_aware_rescues.py \
  --dataset aging \
  --fov Position400 \
  --config balanced \
  --genes Flt3 Mbp \
  --control-genes Penk Camk2a \
  --examples-per-group 3 \
  --all-round-limit 10
```

Output:

```text
/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/decoding/codebook_aware/results/aging/qc/Position400_balanced_gene_evidence/
```

The QC output includes:

- `qc_examples.csv`
- `qc_group_metrics.csv`
- `qc_summary.json`
- `all_rescued_h1_tensor_metrics.csv`
- `all_rescued_h1_tensor_summary.csv`
- `round_montages/*.png`
- `all_round_montages/*.png`

The script samples `rescued_h1`, `rescued_unknown`, exact, and near-no-call
examples separately. This separation matters because `rescued_unknown` calls can
have `score_delta=inf` simply because a wildcard `M/N` leaves only one codebook
candidate. That is codebook uniqueness, not strong image evidence.

Full aging `Position400` call-type split:

| Call type | Count |
|-----------|-------|
| exact | 182,264 |
| rescued_h1 | 105,137 |
| rescued_unknown | 1,717 |
| no_call | 393,693 |

`rescued_unknown` is small and does not explain the top-10 swap:

| Gene | rescued_h1 | rescued_unknown |
|------|------------|-----------------|
| Flt3 | 1,549 | 81 |
| Mbp | 508 | 14 |
| Penk | 1,639 | 18 |
| Camk2a | 1,298 | 17 |

All-rescued_h1 tensor-level corrected-round evidence:

| Gene | h1 rescues | Median target/WTA intensity | Fraction >=0.75 | Fraction >=0.90 | Median target prob | Median WTA prob |
|------|------------|-----------------------------|-----------------|-----------------|-------------------|-----------------|
| Penk | 1,639 | 0.844 | 0.727 | 0.339 | 0.365 | 0.442 |
| Flt3 | 1,549 | 0.762 | 0.527 | 0.201 | 0.328 | 0.435 |
| Kalrn | 1,423 | 0.764 | 0.533 | 0.212 | 0.310 | 0.410 |
| Camk2a | 1,298 | 0.849 | 0.757 | 0.351 | 0.381 | 0.460 |
| Mbp | 508 | 0.820 | 0.663 | 0.297 | 0.386 | 0.471 |

Patch-level sampled QC:

- `Flt3` h1 rescued examples have mixed local image support. High-score h1
  examples have median target center/background 6.34 vs WTA 17.8, with target
  center rank 1 in 1/3 sampled examples. Low-score h1 examples have median
  target center/background 2.42 vs WTA 2.19, also target rank 1 in 1/3.
- `Mbp` h1 high-score examples look stronger in local patch evidence: median
  target center/background 23.96 vs WTA 10.2, target center rank 1 in 2/3.
- `rescued_unknown` examples often have weak local center evidence despite
  `score_delta=inf`. They should be reported separately or disabled for
  production use.

Post-hoc h1 target/WTA intensity-ratio gates do not remove the `Flt3`/`Mbp`
top-10 swap:

| h1 ratio gate | Assigned | h1 rescued kept | Top10 Jaccard | Flt3 count | Mbp count |
|---------------|----------|-----------------|---------------|------------|-----------|
| h1 only, no ratio gate | 287,401 | 105,137 | 0.8182 | 3,086 | 2,067 |
| ratio >= 0.75 | 245,817 | 63,553 | 0.8182 | 2,353 | 1,896 |
| ratio >= 0.90 | 208,876 | 26,612 | 0.8182 | 1,848 | 1,710 |

Interpretation:

- The QC still does not support a broad rescued-read artifact: h1 rescued reads
  are usually ambiguous one-error cases where the corrected channel is often
  close to WTA rather than random background.
- `Flt3` h1 rescued evidence is weaker than `Mbp`, `Penk`, and `Camk2a`, and
  similar to `Kalrn`. This makes `Flt3` the main gene-level follow-up, but not
  a clear artifact from the current evidence.
- `rescued_unknown` should not be mixed into a single "high-confidence" rescue
  bucket. A safer production policy is to default to exact + `rescued_h1`, keep
  `rescued_unknown` optional/off by default, and always preserve `call_type` so
  downstream analyses can filter.

### 2026-04-28 Decoding Example Montages

Added a presentation-oriented montage generator:

```text
src/python/scripts/generate_decoding_example_montages.py
```

Command:

```bash
cd src/python
MPLCONFIGDIR=/tmp/matplotlib-cache UV_CACHE_DIR=/tmp/uv-cache \
  uv run python scripts/generate_decoding_example_montages.py \
  --dataset aging \
  --fov Position400 \
  --config balanced \
  --examples-per-category 6
```

Output:

```text
/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/decoding/codebook_aware/results/aging/qc/Position400_balanced_decoding_examples/
```

Generated 24 all-round montage PNGs, six examples per category:

1. `01_high_confidence_WTA_exact`
2. `02_high_confidence_rescued_h1`
3. `03_low_confidence_rescued_h1`
4. `04_cannot_rescue_potential_noise`

Each all-round montage title includes `call_type`, gene name, spot id, WTA color
sequence, decoded/target color sequence, and reject reason for no-call reads.
The green channel border marks the decoded/target color for each round; the red
border marks the WTA color when it differs from the target. For no-call/noise
examples there is no decoded target, so only the WTA/no-call sequence is shown.

Selection summary:

| Category | N | Median geomean | Median min margin | Median total intensity | Median target/WTA intensity ratio |
|----------|---|----------------|-------------------|------------------------|-----------------------------------|
| High-confidence WTA exact | 6 | 0.9842 | 0.9058 | 3,944.6 | n/a |
| High-confidence rescued h1 | 6 | 0.8435 | 0.0256 | 3,257.9 | 0.9486 |
| Low-confidence rescued h1 | 6 | 0.5512 | 0.0247 | 3,374.6 | 0.7928 |
| Cannot rescue / potential noise | 6 | n/a | 0.0000 | 79.6 | n/a |

Primary files:

```text
decoding_examples.csv
decoding_example_summary.csv
decoding_example_summary.json
all_round_montages/*.png
```

### 2026-04-29 Expanded Real-Dataset Benchmark

Updated the benchmark runner so `cell_culture_3D` and `tissue_2D` can be run
through the same codebook-aware benchmark. These datasets do not have saved
backend-comparison all-spots tensors, so the runner now has a raw-E2E fallback
that:

- loads raw images with the dataset-specific MATLAB channel order;
- runs rotate, contrast enhancement, global registration, and spot finding;
- extracts full all-read raw intensity tensors;
- caches tensors/spots and per-FOV input metadata for repeat benchmark runs.

Main aggregate command:

```bash
cd src/python
uv run python -u scripts/run_codebook_aware_benchmark.py \
  --datasets synthetic_medium synthetic_large LN aging cell_culture_3D tissue_2D \
  --configs wta_exact balanced
```

Aggregate output:

```text
/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/decoding/codebook_aware/results/comparison_all.csv
```

New raw input metadata:

```text
results/cell_culture_3D/input_metadata/Position351.json
results/cell_culture_3D/input_metadata/Position352.json
results/tissue_2D/input_metadata/tile_1.json
results/tissue_2D/input_metadata/tile_2.json
```

Expanded real-data results, full reads:

| Dataset/FOV | WTA assigned | Balanced assigned | Rescued | Match-rate gain | Spearman | Top10 Jaccard | Decode time | Raw input prep peak |
|-------------|--------------|-------------------|---------|-----------------|----------|---------------|-------------|---------------------|
| LN `Position001` | 10,466 | 11,263 | 797 | +0.0670 | 0.9959 | 1.0000 | 1.54s | cached backend |
| LN `Position002` | 10,114 | 11,047 | 933 | +0.0717 | 0.9968 | 1.0000 | 2.91s | cached backend |
| aging `Position400` | 182,264 | 289,118 | 106,854 | +0.1565 | 0.9773 | 0.8182 | 529.51s | cached backend |
| cell culture `Position351` | 31,650 | 40,396 | 8,746 | +0.1077 | 0.9836 | 1.0000 | 48.42s | 3.81 GB |
| cell culture `Position352` | 32,817 | 41,030 | 8,213 | +0.1002 | 0.9863 | 1.0000 | 48.08s | 3.82 GB |
| tissue `tile_1` | 35,820 | 46,095 | 10,275 | +0.1533 | 0.9942 | 0.8182 | 27.26s | 13.21 GB |
| tissue `tile_2` | 51,081 | 57,811 | 6,730 | +0.0966 | 0.9970 | 1.0000 | 16.60s | 13.22 GB |

Raw input prep times for the newly added datasets:

| Dataset/FOV | Prep time | Tensor shape |
|-------------|-----------|--------------|
| cell culture `Position351` | 73.02s | `(81182, 4, 6)` |
| cell culture `Position352` | 73.06s | `(81981, 4, 6)` |
| tissue `tile_1` | 199.22s | `(67038, 4, 4)` |
| tissue `tile_2` | 223.97s | `(69663, 4, 4)` |

Interpretation:

- The new datasets support the same broad conclusion as LN: `balanced`
  increases assigned reads substantially while keeping gene abundance
  correlation high.
- cell culture looks stable by top-10 composition in both FOVs.
- tissue `tile_1` has a top-10 swap despite high Spearman correlation; this is
  similar in kind to aging and should be included in the rescued-read artifact
  QC pass before production use.
- The raw-E2E fallback is practical for repeated benchmarking because the
  tensor cache is small relative to image data: cell culture tensors are about
  15-16 MB per FOV and tissue tensors are about 8-9 MB per FOV.
