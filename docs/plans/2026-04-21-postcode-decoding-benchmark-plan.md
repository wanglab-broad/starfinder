# Postcode Decoding Benchmark Plan

**Date:** 2026-04-21
**Status:** FINISHED

## Context

Starfinder's current decoder is a deterministic codebook-membership lookup: `extract_from_location()` reduces each spot's per-channel intensities to a single winner-take-all label per round, then `filter_reads()` checks whether the concatenated color sequence is present in the codebook. Spots whose color sequence is not in the codebook are dropped.

[Postcode](https://github.com/milana-gataric/postcode) (Gataric 2021) takes a fundamentally different approach: it fits a re-parameterised matrix-variate Gaussian mixture model over raw per-(spot, round, channel) intensities, producing **posterior probabilities** over genes plus explicit "background" and "infeasible" classes. This can (in theory) rescue spots that starfinder discards as noise and flag low-confidence calls.

We want to quantify this tradeoff on two datasets where we already have reliable starfinder outputs, to decide whether Postcode is worth integrating as an alternative decoder.

## Decisions (Locked with User)

1. **Datasets:** synthetic `medium` preset (full ground truth) + `aging` real dataset (6 in-sample FOVs, but run on the 2-FOV subset already benchmarked in e2e).
2. **Comparison mode:** pure decoder comparison — both decoders consume the *same* spot list and intensity tensor. Spot detection variance is factored out.
3. **Raw-intensity helper:** add `extract_intensity_tensor()` to `src/python/starfinder/barcode/extraction.py` (sibling of `extract_from_location`).
4. **Aging remains required:** production relevance must be tested on aging, but the real-data run is staged behind synthetic/smoke-run time and memory measurements so we can choose a feasible full-FOV strategy.
5. **Synthetic primary input:** use detected `all_spots` as the shared decoder input, then use ground truth only for evaluation. A GT-spot-only mode is retained as a clean control.

## Non-trivial Design Notes

### Shape convention bridge
- **Postcode expects** `spots: (N, C, R)` raw intensities, `barcodes_01: (K, C, R)` one-hot. See `decoding_function()` in Postcode's `source-code/postcode/decoding_functions.py`.
- **Starfinder uses** `(Z, Y, X, C)` images and iterates per-round. Per round, `extract_from_location` computes a pre-normalisation `(N, C)` vector (line 83). We sum-pool the same neighbourhood across all rounds and stack to `(N, C, R)`.

### Aging dataset segmentation
Aging uses **2-segment barcodes** (split_index=4 Python / 5 MATLAB): each gene has ~7 pad variants → 14,242 codebook entries over 2,044 unique genes, producing 9 color rounds after segment-boundary stripping (per `docs/plans/2026-03-30-aging-e2e-benchmark-plan.md:29-47`). Postcode has no segment concept.

**Strategy:** use starfinder's already-computed per-gene `seq_to_gene` dict (the post-`load_codebook(split_index=4)` output, length 14,242). For Postcode, build `barcodes_01` of shape `(K=14242, C=4, R=9)` where `barcodes_01[k, c, r] = 1` iff channel `c+1` is the `r`-th digit of the color sequence for entry `k`. Postcode's SVI treats K=14,242 classes as independent mixture components — may require `batch_size=5000` and extended `num_iter` (120-200). This is a hyperparameter that the benchmark script exposes.

Postcode's default "remaining/infeasible barcode" path enumerates all `4^R` possible sequences. That is fine for synthetic medium (`4^4 = 256`) but potentially expensive for aging (`4^9 = 262,144`) and can make a full `(N, K + infeasible + background)` probability matrix infeasible for hundreds of thousands of spots. Therefore:
- synthetic runs save the full probability matrix first, because `N` and `K` are small;
- aging runs first execute smoke-size subsets and record wall time plus peak RSS;
- aging full-FOV runs save compact top-call summaries by default, and only save full probabilities after the synthetic/smoke results show that it is practical.

### Aging pad variants
Aging gene names include pad/probe variants (for example `_STAR_pad_6` and `_RIBO_pad_1`). Postcode returns probabilities over the 14,242 codebook entries, not directly over the 2,044 base genes.

For aging, save both levels:
- **pad-level call:** top codebook entry, mapped through `seq_to_gene`;
- **base-gene call:** sum Postcode probabilities across all pad/probe variants that strip to the same base gene, then take the top base gene.

Primary aging agreement/distribution metrics use the base-gene calls. Pad-level outputs are retained for troubleshooting and pad-balance analysis.

### Codebook fidelity
Starfinder's `codebook.load_codebook()` yields `seq_to_gene: dict[str, str]` where keys are color sequences like `"4422"` (chars `"1"`-`"4"`). To build Postcode's one-hot: for each `seq` key, `one_hot[k, :, r] = np.eye(4)[int(seq[r]) - 1]`. Simple loop, ~10 LOC. Saved as `codebook_onehot.npy` so Postcode and starfinder share exactly the same K.

## Core Code Changes

### 1. `src/python/starfinder/barcode/extraction.py`

Add (sibling, not replacement):

```python
def extract_intensity_tensor(
    images: dict[str, np.ndarray],        # {round_name: (Z, Y, X, C)}
    spots: pd.DataFrame,                  # z, y, x (0-based)
    round_order: list[str],               # e.g. ["round1", "round2", ...]
    voxel_size: tuple[int, int, int] = (1, 2, 2),
) -> np.ndarray:
    """Return raw per-(spot, channel, round) intensities, shape (N, C, R).

    Same voxel-neighbourhood sum as `extract_from_location`, but without
    L2-normalisation or winner-take-all. Output feeds probabilistic
    decoders (e.g. Postcode).
    """
```
- Reuses the exact padding/index-grid construction already in `extract_from_location` (lines 52-83).
- Output ordering `(N, C, R)` matches Postcode directly; we do not reshape inside Postcode's `torch_format`.
- Add 1 pytest in `src/python/test/test_extraction.py`: a 2-round synthetic image where the argmax of `extract_intensity_tensor` per round equals the integer form of `extract_from_location`'s `color_seq`. Cross-validates both functions.
- Export the helper from `src/python/starfinder/barcode/__init__.py` so benchmark scripts can import it through `starfinder.barcode`.

No other starfinder code is modified. The existing pipeline keeps using `extract_from_location`.

## Benchmark Directory Layout

All paths under `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/decoding/postcode/`.

```
postcode/
├── README.md                         # Quickstart + results summary
├── env/
│   ├── environment.yml               # Conda env: py3.9, pyro-ppl, torch, scikit-image, numpy, pandas
│   └── install_postcode.sh           # git clone postcode + pip install -e
├── scripts/
│   ├── common.py                     # Shared helpers: codebook → one-hot, postcode_to_starfinder_df, metrics
│   ├── run_decoding_synthetic.py     # Synthetic medium driver
│   └── run_decoding_aging.py         # Aging 2-FOV driver
├── data/
│   └── synthetic_medium/             # Generated once via `uv run python -m starfinder.benchmark --preset medium`
│       ├── ground_truth.json
│       ├── codebook.csv
│       └── FOV_001/, FOV_002/
└── results/
    ├── synthetic_medium/
    │   ├── intensity_tensors/FOV_*.npy     # (N, C, R) raw intensities (shared input)
    │   ├── spots/FOV_*.csv                 # Detected input spot list (z, y, x, spot_id)
    │   ├── starfinder/FOV_*.csv            # Cols: spot_id, color_seq, gene (or NaN if not in codebook)
    │   ├── postcode/FOV_*.csv              # Cols: spot_id, top_gene, top_prob, class_label, full_probs_path
    │   ├── postcode/FOV_*_probs.npy        # Synthetic only: full class probabilities
    │   ├── log/FOV_*.csv                   # Per-FOV QC row (see schema below)
    │   ├── comparison.csv                  # One row per FOV, decoder-level metrics
    │   └── comparison_summary.json         # Aggregated metrics + config used
    └── aging/
        ├── intensity_tensors/Position*.npy # Common input, generated after fresh spot detection
        ├── spots/Position*.csv             # Full detected spot list, not only goodSpots
        ├── starfinder/Position*.csv
        ├── postcode/Position*.csv          # Compact top-call outputs by default
        ├── log/Position*.csv
        ├── comparison.csv
        └── comparison_summary.json
```

This mirrors the `e2e/results/<dataset>/` structure (signal/, log/, comparison.csv) described in MEMORY.md, so future readers can navigate both without re-learning.

## Comparison CSV Schema (`results/<dataset>/comparison.csv`)

One row per FOV, side-by-side metrics. Both decoders consume identical inputs.

| Column | Source | Meaning |
|--------|--------|---------|
| `fov_id` | input | `FOV_001`, `FOV_002` |
| `n_input_spots` | input | Spots fed to both decoders |
| `n_genes_codebook` | codebook | K (before segment dedup for aging) |
| `sf_n_assigned` | starfinder | Spots with non-null gene (= in_codebook) |
| `sf_codebook_match_rate` | starfinder | `n_in_codebook / n_input_spots` |
| `pc_n_assigned_top1` | postcode | Spots whose top-1 class is a gene (not bkg/inf/nan) |
| `pc_n_assigned_p80` | postcode | Top-1 class is a gene AND prob ≥ 0.8 |
| `pc_n_background` | postcode | Top-1 class = background |
| `pc_n_infeasible` | postcode | Top-1 class = infeasible |
| `agreement_rate` | both | Fraction where both assigned same gene |
| `sf_only_rate` | both | starfinder assigned a gene, Postcode said bkg/inf |
| `pc_only_rate` | both | Postcode assigned (top-1) a gene, starfinder said not-in-codebook |
| **Synthetic only:** | | |
| `sf_gene_accuracy` | vs GT | starfinder gene correctness on matched spots |
| `pc_gene_accuracy_top1` | vs GT | Postcode top-1 correctness on matched spots |
| `pc_gene_accuracy_p80` | vs GT | Postcode correctness when top-1 prob ≥ 0.8 |
| `pc_recall_top1` | vs GT | Fraction of GT spots Postcode gets right |
| `sf_recall` | vs GT | Fraction of GT spots starfinder gets right |
| **Aging only (no GT):** | | |
| `gene_distribution_corr` | both | Spearman ρ between per-gene counts (n=2044) |
| `top10_gene_jaccard` | both | Jaccard of top-10 most-abundant genes |
| `pad_gene_distribution_corr` | both | Optional Spearman ρ across 14,242 pad/probe entries |
| **Perf:** | | |
| `sf_time_s`, `pc_time_s` | both | Decoder wall-clock |
| `pc_peak_rss_mb` | postcode | Peak RSS during Postcode run |
| `pc_num_iter`, `pc_batch_size`, `pc_spot_limit` | config | Postcode hyperparams actually used |
| `pc_full_probs_saved` | config | Whether the full probability matrix was saved |

Metrics reuse `starfinder.benchmark.validation.compare_genes()` for synthetic GT (already handles gene comparison on position-matched spots, per `src/python/starfinder/benchmark/validation.py:158-226`). For aging (no spot-level GT) we use distribution-level metrics only.

## Script Responsibilities

### `scripts/common.py`
- `seq_to_onehot(seq_to_gene: dict, n_channels: int, n_rounds: int) -> tuple[np.ndarray, list[str]]`: build `(K, C, R)` + gene name list (index-aligned with the first axis).
- `postcode_classes_to_df(out, gene_names, spot_ids) -> pd.DataFrame`: top-1 + prob + class label (gene/background/infeasible/nan).
- `collapse_aging_pad_probabilities(probs, gene_names) -> tuple[np.ndarray, list[str]]`: sum pad/probe probabilities to base-gene probabilities for aging.
- `run_starfinder_decode(intensity_tensor, seq_to_gene) -> pd.DataFrame`: wraps argmax-per-round + `filter_reads` directly on the tensor to keep both decoders starting from the same (N, C, R) input.
- `build_comparison_row(sf_df, pc_df, gt=None) -> dict`: produces one row of `comparison.csv`.

### `scripts/run_decoding_synthetic.py`
1. If missing, generate synthetic medium: `uv run python -m starfinder.benchmark --mode e2e --preset medium --output data/synthetic_medium --seed 42`.
2. For each FOV: load images (4 rounds × 4 channels), align all rounds to the reference frame before extraction, run the same spot detector used by the Python backend, load ground truth for evaluation, and load the codebook.
3. Call **new** `extract_intensity_tensor()` using detected `all_spots` positions → saves `intensity_tensors/FOV_*.npy` and `spots/FOV_*.csv`.
4. Run starfinder decode path (per-round argmax → `filter_reads`) → saves `starfinder/FOV_*.csv`.
5. Activate Postcode conda env via subprocess; call `decoding_function(spots=tensor, barcodes_01=codebook_onehot)` → saves `postcode/FOV_*.csv` + probs.
6. Build comparison row with `compare_genes()` against `ground_truth.json`.
7. Optional control: `--gt-spots` skips spot detection and extracts intensities only at ground-truth positions. This validates tensor ordering/codebook encoding but is not the primary rescue-rate benchmark.

### `scripts/run_decoding_aging.py`
Same structure as synthetic but:
- Reruns the relevant Python backend steps for each requested FOV: load → rotate → enhance → global registration → spot finding → intensity-tensor extraction.
- Saves the full detected spot list under `results/aging/spots/Position*.csv`; do not use existing `goodSpots` outputs as input because that would pre-filter away the reads Postcode might rescue.
- Calls `load_codebook(path, split_index=4)` to get the 14,242-entry `seq_to_gene` mapping.
- Supports `--fovs Position400 Position401`, `--spot-limit`, `--num-iter`, `--batch-size`, and `--save-full-probs`.
- Passes `num_iter=150, batch_size=5000` by default for large K, but smoke runs use smaller `--spot-limit`/`--num-iter` combinations to estimate feasibility.
- Skips GT-dependent metrics; adds base-gene distribution-level metrics plus optional pad-level diagnostics.

## Environment Isolation

Postcode's README says it was tested with Python 3.6.12. Starfinder runs on Python 3.10+ with modern scikit-image (per `pyproject.toml` in `src/python/`). We cannot assume the environments are compatible, so Postcode gets an isolated conda env plus an import preflight.

`env/environment.yml` (new conda env `postcode-bench`):
```yaml
name: postcode-bench
channels: [pytorch, conda-forge, defaults]
dependencies:
  - python=3.9          # Compromise: 3.6 is EOL, pyro-ppl wheels available for 3.9
  - pytorch
  - pyro-ppl
  - numpy, pandas, scipy, scikit-image, tifffile, tqdm
  - pip
  - pip:
      - git+https://github.com/milana-gataric/postcode@master#egg=postcode
```

`env/install_postcode.sh` clones the repo and runs `pip install -e .` from the repo root if the git-URL install fails. The install script ends with an import preflight:

```bash
python - <<'PY'
from postcode.decoding_functions import decoding_function
print("Postcode import OK:", decoding_function.__name__)
PY
```

## Verification Plan

1. **Unit test** for the new `extract_intensity_tensor`:
   ```bash
   cd src/python && uv run pytest test/test_extraction.py -v
   ```
   Passes if argmax-per-round on the new helper's output equals `extract_from_location`'s `color_seq` on the same inputs.

2. **Synthetic dry-run** (fastest signal of correctness):
   ```bash
   cd starfinder_benchmark/decoding/postcode
   conda activate postcode-bench
   python scripts/run_decoding_synthetic.py --smoke --save-full-probs  # FOV_001 only, num_iter=30
   ```
   Expected: `sf_gene_accuracy ≈ 1.0`, `pc_gene_accuracy_top1 ≥ 0.95`, `agreement_rate ≥ 0.95`. Large divergence → bug in tensor ordering, alignment, or codebook encoding. Record `pc_time_s`, `pc_peak_rss_mb`, and saved probability matrix size.

3. **Synthetic full run:**
   ```bash
   python scripts/run_decoding_synthetic.py --output-dir results/synthetic_medium
   ```
   Inspect `comparison.csv` — expected 2 rows, matching metrics. Sanity-check `pc_only_rate`: Postcode should "rescue" some spots starfinder drops as noise on synthetic data (low — synthetic is clean), but agreement should dominate.

4. **Aging full run:**
   First run staged feasibility checks:
   ```bash
   python scripts/run_decoding_aging.py --fovs Position400 --spot-limit 5000 --num-iter 30
   python scripts/run_decoding_aging.py --fovs Position400 --spot-limit 50000 --num-iter 60
   ```
   Use the observed time/RSS scaling to decide whether a full FOV is practical and whether full probabilities are safe to save.

   Then run the production-relevance benchmark:
   ```bash
   python scripts/run_decoding_aging.py --fovs Position400 Position401 --num-iter 150 --batch-size 5000
   ```
   Watch for: (a) Postcode converges (loss plateau in log), (b) memory stays within the selected budget, (c) `gene_distribution_corr > 0.9` between the two decoders on the top-100 most abundant base genes, and (d) Postcode's extra gene assignments are not dominated by low-probability calls.

5. **Document results** in `README.md` in the benchmark dir (1 table per dataset, 3-5 sentences of interpretation).

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/python/starfinder/barcode/extraction.py` | Add `extract_intensity_tensor()` (~30 LOC) |
| `src/python/starfinder/barcode/__init__.py` | Export `extract_intensity_tensor()` |
| `src/python/test/test_extraction.py` | Add 1 cross-validation test |
| `starfinder_benchmark/decoding/postcode/env/environment.yml` | New (conda env spec) |
| `starfinder_benchmark/decoding/postcode/env/install_postcode.sh` | New (fallback install) |
| `starfinder_benchmark/decoding/postcode/scripts/common.py` | New (helpers) |
| `starfinder_benchmark/decoding/postcode/scripts/run_decoding_synthetic.py` | New (synthetic driver) |
| `starfinder_benchmark/decoding/postcode/scripts/run_decoding_aging.py` | New (aging driver) |
| `starfinder_benchmark/decoding/postcode/README.md` | New (quickstart + results) |

## Out of Scope (Future)

- Integrating Postcode as an alternative decoder in the main Snakemake pipeline.
- Multi-segment native handling in Postcode (would require forking — not worth it unless results justify).
- Comparison against other decoders (ISTDECO, starfish's TrackpyLocalMaxPeakFinder).
- Full aging (all 6 in-sample FOVs or 848 FOVs) — 2-FOV subset is sufficient for a decoder comparison.
