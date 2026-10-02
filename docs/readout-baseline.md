# Readout baseline: extraction, decoding and read QC

Status: Proposed

This page records how intensity extraction, barcode decoding and read filtering
behave at revision `141c093` (branch `runner/s28-spec-20261001`, on `dev` after
the §2.7 work), before the Chapter II §2.8 work changes them. It is the reference
that the golden test `src/python/test/test_readout_golden.py` pins. The proposed
replacement is described in {doc}`readout-contract` and {doc}`readout-algorithms`;
neither is accepted. Paths are relative to `src/python/starfinder/` unless they
start with `src/matlab/`, `workflow/` or `docs/`, and line numbers are at
`141c093`.

## Extraction

`extract_intensities(rounds, spots, *, config)` in `starfinder.barcode`
({doc}`api/barcode`) sums a box around each candidate in every round.

| Field | Behavior at `141c093` |
| --- | --- |
| Inputs | An ordered mapping of round label to `ImageLoadResult` (ZYXC), whose insertion order defines the round axis, and a `SpotFindingResult`. Every round must have the same shape, the spot result's metadata and the same channel labels, else `ValueError` (`barcode/extraction.py:105-138`). Coordinates must lie in `[0, size − 1]` on every axis before rounding (`:141-142`). |
| Sampling | Nearest voxel: the box centre is `floor(coord + 0.5)` (`:143`). `NeighborhoodSumConfig.sampling` accepts only `"nearest"` (`:36-37`). |
| Box and units | `neighborhood_radius_zyx` (default `(1, 2, 2)`) are half-widths in voxel counts, never physical spacing: the box is 3×5×5 voxels by default (`:22`). The value is the float64 sum of the image's grey levels over the box, per channel (`:146-152`); it is not divided by the voxel count, and nothing is subtracted. |
| Boundary | `boundary="zero"` only (`:36-37`). Near a face the box is clipped to the image (`:148-149`), which equals zero padding: a clipped box sums fewer voxels, and nothing records how many. |
| Result | `IntensityExtractionResult`: `values` (N×C×R float64, finite), `spot_ids` and `spot_namespace` from the spot table, `channel_labels`, `round_labels`, `metadata`, `config`, `valid` and `diagnostics` (`source_shape_zyx`, `calculation_dtype`, `sampling`) (`:40-96`, `:153-167`). |
| The `valid` mask | Boolean N×R, documented as "false marks an unavailable measurement, not zero signal" (`:44`). `extract_intensities` always sets it to all true (`:161`), including for a clipped box at a face and for a round that is zero at the candidate. Nothing in the package sets it to false. It can only be false in a result a caller builds directly or in a `candidates` checkpoint whose `valid_<round>` column was edited. Decoding turns any false round into `unmatched` with reason `invalid_measurement` (`barcode/decoding.py:195`, `:267-276`). |
| `FOV` extraction | `FOV.extract_intensities(config, rounds)` and `FOV.run` extract one round at a time (`_extract_round`, `dataset/fov.py:887-900`) from the round's image or the recipe's `extraction_source` snapshot, then concatenate the rounds (`_assemble_intensities`, `:902-915`). Every sequencing round is extracted at every candidate. |
| Cost | W-278 measured the per-spot loop on `medium` (32×512×512, 587 candidates, 4 rounds, 4 channels): median 0.0586 s, 25 µs per spot and round, 685,180 KiB peak RSS for the process (W-278 `extraction-cost.csv`). |
| MATLAB counterpart | `STARMapDataset.ReadsExtraction` (`src/matlab/STARMapDataset.m:762-811`) calls `ExtractFromLocation` (`src/matlab/ExtractFromLocation.m:1-48`) per round on the channels whose name contains `seq`. It clips the box to the image (`GetExtents`, `:51-67`), L2-normalizes the channel sums, and stores only the round's WTA color (`M` for a tie, `N` for NaN) and its score `-log(max)`. The color strings of the rounds are concatenated into `color_seq` (`STARMapDataset.m:796-805`); the sums themselves are not kept. Its `voxel_size` is `(row, column, Z)`, Python's is `(z, y, x)` ({doc}`conventions`). |

## Codebook and `EncodingConfig`

| Field | Behavior at `141c093` |
| --- | --- |
| Encoding | Two-base color space only (`barcode/_encoding.py:10-30`): each pair of adjacent bases maps to one of four colors (`AA, CC, GG, TT` → 1; `AC, CA, GT, TG` → 2; `AG, CT, GA, TC` → 3; `AT, CG, GC, TA` → 4), so `n` bases give `n − 1` colors. `decode_color_sequence(colors, start_base)` inverts it from a known first base (`:70-119`). There is no other encoding. |
| `EncodingConfig.reverse_bases` | Default `True`: the barcode is reversed before encoding (`barcode/codebook.py:31`, `:48`), as MATLAB `do_reverse` (`src/matlab/LoadCodebook.m:16-18`). |
| `EncodingConfig.split_index` | `None` or a positive integer `i` (`codebook.py:32-42`). After encoding, the color at **zero-based** position `i` is removed and the colors after it are moved before the colors before it: `seq[i+1:] + seq[:i]` (`:49-53`). Both parts must be nonempty. This is the only representation of two segments: there is no segment count, no segment length in bases and no per-segment end bases. |
| `Codebook` | A table with `gene_id` and `color_sequence` (and an optional `base_sequence`, checked against the encoding), `round_labels`, `channel_labels` (exactly four) and `color_to_channel` (a bijection of colors `1`–`4` onto the channel indices) (`codebook.py:57-127`). Each `color_sequence` must have one color per round. A repeated `gene_id` is an error (`:100-101`), and so is a repeated color sequence (`:108-109`). A row is therefore a gene: `n_genes` counts rows, `gene_to_seq` maps each gene to one sequence, and `seq_to_gene` maps each sequence to its gene (`:135-153`). |
| `load_codebook` | Reads a headerless `gene,barcode` CSV (the MATLAB `genes.csv`), a `gene,barcode` header, or a canonical `gene_id,color_sequence[,base_sequence]` header; barcodes are encoded with the given `EncodingConfig` (`codebook.py:156-218`). Errors name the source row. `Dataset.load_codebook(path, split_index=None, reverse_bases=True)` builds the `EncodingConfig` and uses the sequencing rounds and `channel_order` as labels (`dataset/dataset.py:139-165`). |
| MATLAB counterpart | `LoadCodebook.m` reads `genes.csv`, reverses (`:16-18`), encodes (`:20-22`) and, with a nonempty `split_index`, erases the color at **one-based** position `split_index` and concatenates the part after it with the part before it (`:24-30`). It builds `seqToGene` and `geneToSeq` dictionaries (`:34-35`). MATLAB `split_index` `s` therefore equals Python `split_index` `s − 1`. |

## Decoders

`decode_barcodes(intensity_result, codebook, *, config)` takes `WtaDecoderConfig`
or `CodebookAwareDecoderConfig` (`barcode/decoding.py:152-380`). Both:

* require the intensity and codebook channel and round labels to match exactly
  (`:175-179`);
* reject negative sums unless `negative_policy="clip_negative"` (`:180-188`), then
  reorder channels into color order `1`–`4` through `color_to_channel` (`:190`);
* mark a read `no_signal` with reason `zero_signal_round` when any round's channel
  sum is 0, and `unmatched` with reason `invalid_measurement` when any round is not
  `valid`; these override the decoder's own status, set `call_type` to `no_call` and
  clear `gene_id` and `decoded_color_sequence`, but keep the decoder's score columns
  (`:194-195`, `:267-276`). In the golden fixture the zero-round read keeps its
  codebook-aware `probability_nll` of 2.209;
* return one row per candidate with `spot_id`, `spot_namespace`,
  `observed_color_sequence`, `decoded_color_sequence`, `gene_id`, `call_status`
  (`assigned`, `unmatched`, `ambiguous`, `no_signal`), `failure_reason`,
  `call_type` and the decoder's named scores (`BarcodeDecodingResult`, `:96-149`).

| | `WtaDecoderConfig` | `CodebookAwareDecoderConfig` |
| --- | --- | --- |
| Per-round call | The channel with the largest sum; an exact tie of the maximum gives `M` (`:236-241`). | The channel with the largest probability `p = (v + 1e-6) / Σ(v + 1e-6)`; a tie within absolute 1e-12 gives `M` (`_codebook_aware.py:58-159`). |
| Assignment | The observed sequence is looked up in the codebook; a miss is `unmatched` / `not_in_codebook` (`decoding.py:245-259`). Any `M` round is `ambiguous` / `tied_channels` (`:260-263`). No rescue. | An exact match is `exact` (`_codebook_aware.py:486-500`). Otherwise, with `allow_rescue`, codebook sequences within `max_hamming` (known substitutions plus `M`/`N` rounds) compete: the best is assigned only if it passes every gate (`:502-592`); otherwise `unmatched` with the gate's reason, or `ambiguous` for `ambiguous_candidate` or an unrescued `M` (`decoding.py:230-234`). |
| Scores | `wta_l2_nll` = Σ over rounds of −log(max / (L2 + 1e-6)), infinite for a tie (`:236-256`). | `probability_nll` = Σ −log max(p, 1e-12) of the decoded sequence, `geomean_probability` = exp(−`probability_nll` / R), `score_delta` (runner-up minus best; infinite for an exact or sole candidate), `min_round_margin`, `corrected_round_margin`, `hamming_to_wta`, `mean_total_intensity`, `gene_wta`, `corrected_rounds` (`_codebook_aware.py:17-33`). |
| Gates and defaults | — | `max_hamming=1`; `max_corrected_round_margin=0.20` (an upper bound on the top-minus-second probability at a changed round); `min_score_delta=0.25`; `min_geomean_probability=0.45`; `max_correction_penalty=1.50`; `allow_exact=True`; `allow_rescue=True` (`decoding.py:56-62`). |
| Call types | `exact`, `no_call` | `exact`, `rescued_unknown` (an `M`/`N` round filled), `rescued_h<k>` (k substitutions), `no_call` |
| Failure reasons | `not_in_codebook`, `tied_channels`, `zero_signal_round`, `invalid_measurement` | `rescue_disabled`, `no_candidate`, `ambiguous_candidate`, `too_many_edits`, `corrected_round_margin_too_high`, `correction_penalty_too_high`, `geomean_prob_too_low`, `zero_signal_round`, `invalid_measurement` |

With `diagnostics=True` the result also holds the per-round probabilities, a
per-round margin table and a candidate table (`decoding.py:285-372`); they are not
saved in checkpoints. Exact calls are identical in both decoders; they differ in
scores and in rescue. Both treat each codebook row as one gene. The decoders are
hard-coded in `decoding.py:169`, `:207`, `dataset/config.py:226` and
`io/_checkpoint.py:537` ({doc}`method-registry`, "Decoding").

## Read filtering

`filter_reads(decoding_result, *, config)` (`barcode/filtering.py:109-168`)
annotates every row and keeps it; `ReadFilteringResult.accepted` is a view.

| Field | Behavior at `141c093` |
| --- | --- |
| Status predicate | `call_statuses`, default `("assigned",)` (`filtering.py:20`). Rejection reason `call_status`. |
| Score predicates | `score_bounds` maps a score column to an inclusive `(lower, upper)`, either `None`; NaN fails (`:21`, `:127-136`). The accepted names are a fixed list of eight: `wta_l2_nll`, `probability_nll`, `score_delta`, `geomean_probability`, `min_round_margin`, `corrected_round_margin`, `mean_total_intensity`, `hamming_to_wta` (`:37-47`). Any other name raises at construction, and a listed name the decoder did not produce raises at filtering. Rejection reason `score:<name>`. |
| End bases | `end_bases` is one two-base pair or `None`, with one `start_base` (default `C`) (`:22-23`, `:57-61`). The observed color sequence is decoded from `start_base` and `endpoint_valid` is true when the decoded first and last bases equal the pair (`:138-147`). The check is diagnostic; it rejects (reason `endpoint`) only with `exclude_invalid_endpoints=True` (`:148-149`). It decodes the whole sequence as one segment. |
| Output | The decoding table plus `accepted`, `rejection_reasons` (`;`-joined) and `endpoint_valid` when checked; `counts` (`total`, `accepted`, `rejected`) and `fractions` (`accepted`, `endpoint_valid`; `None` with a reason for an empty table) (`:150-168`). |
| MATLAB counterpart | `STARMapDataset.ReadsFiltration` (`STARMapDataset.m:848-953`) first drops reads whose `color_seq` contains `N` or `M` (`:880-887`), then calls `FilterReads` for `n_barcode_segments == 1` and `FilterReadsMultiSegment` otherwise (`:900-904`), and writes the counts to `log/sf_scores`. `FilterReads.m` decodes every read from the first base of the first `end_base` pair (`:19`), selects reads with `contains(color_seq, codebook keys)` (`:23-24`), which for equal-length strings is exact membership, and reports the fraction with each end-base pair (`:35-59`) as statistics only (`:11-12`). |

## The MATLAB two-segment path

`FilterReadsMultiSegment.m` is the MATLAB route for two segments, with `end_base`
holding one pair per segment and `split_index` the shared key (aging: `5`,
`["CC", "TT"]`, {doc}`datasets`):

* **Membership uses the complete concatenated color string.**
  `barcodes_in_codebook = contains(color_seq, codebook_barcodes)` compares the whole
  observed `color_seq` with the codebook keys, which `LoadCodebook.m:24-30` built
  with the junction color removed and the segments swapped
  (`src/matlab/FilterReadsMultiSegment.m:48-49`). The selected reads and their genes
  come only from this test (`:82-84`).
* **The per-segment end-base checks are diagnostic.** The file says so (`:7-8`,
  `:11-12`); each segment is decoded with `DecodeCS(segment, end_base(i)(1))`
  (`:37-43`), the fraction of reads whose segment starts and ends with
  `end_base(i)` is printed and appended to `scores` (`:61-76`), and nothing is
  filtered by it.
* **The diagnostic cuts the observed sequence one color early.** With one
  `split_index` `s`, segment 1 is columns `1:s−1` and segment 2 is columns `s:end`
  of the observed string (`:21-33`, the `n==1` branch). The codebook string is
  `extractAfter(f, s−1) + extractBefore(f, s)` of the string with color `s` erased
  (`LoadCodebook.m:25-30`): for 10 encoded colors and `s = 5`, that is colors 6–10
  (5 colors) followed by colors 1–4 (4 colors), the 5 + 4 order. The diagnostic
  instead cuts after 4 colors (4 + 5), so its segment 1 holds the first 4 of the
  5 colors of codebook segment 1, and its segment 2 holds the last color of
  segment 1 followed by the 4 colors of segment 2. On the 11-base example below, the
  read equal to its codebook entry is cut into `2423 | 24242`; decoded from `C`
  and `T`, the cut segments end in `CA` and `TG`, while the 5 + 4 cut
  `24232 | 4242` gives `CC` and `TT`. The membership result is unaffected.

## The `split_index` index base: an 11-base example

The barcode `CAGTACTGCAT` is segment A `CAGTAC` (6 bases) followed by segment B
`TGCAT` (5 bases). Reversed it is `TACGTCATGAC`, which encodes to the 10 colors
`4242324232`; color 5 (`3`, the pair `TC`) spans the junction. Computed on current
source by `scripts/w279_split_index_demo.py` in the W-279 run directory
(output `split-index-demo.txt`):

| Route | `split_index` used by Python | Color sequence | Segments |
| --- | --- | --- | --- |
| MATLAB `LoadCodebook` with the shared value `5` (transcribed) | — | `242324242` | 5 + 4: `24232` (A, rounds 1–5), `4242` (B, rounds 6–9) |
| `EncodingConfig(split_index=4)` (expected) | 4 | `242324242` | 5 + 4, equal to MATLAB |
| `from_workflow_config` with `load_codebook.split_index: [5]`, then `Dataset.load_codebook` (obtained) | 5 | `423242423` | 4 + 5: colors 7–10, then 1–5 |

`from_workflow_config` takes the one-element list and passes `5` unchanged
(`dataset/workflow.py:407-411`); `_run_workflow` and the benchmark pipeline pass it
to `Dataset.load_codebook` (`:426`, `benchmark/_pipeline.py:67-68`), which uses it as
the zero-based Python index. The obtained codebook drops color 6 instead of the
junction color, so no correctly read two-segment barcode matches it. The research
scripts already use `4` for aging (`scripts/run_codebook_aware_benchmark.py:75`).

## Order in `FOV.run`

`FOV.run` (`dataset/fov.py:954-1157`) runs, for each round in reference-first
order, loading, rotation, the preprocessing steps, registration and the
post-registration steps (`:1072-1103`); detection on the reference round
(`:1107-1108`); and extraction of each sequencing round right after its processing,
so streaming mode can drop the image (`:1111-1112`). With a detection plan that names
rounds, the rounds' candidates are combined after the loop and every sequencing
round is extracted at every candidate (`:1128-1136`). Then the rounds are assembled
(`:1137-1138`), the `candidates` checkpoint is written (`:1139-1141`), the reads are
decoded and the `pre_qc` checkpoint written (`:1142-1146`), and the reads are
filtered (`:1147-1148`). There is no stage between decoding and filtering: no score
stage and no deduplication. `run.json` records the call-status and filtering
counts (`dataset/_run_record.py:119-129`).

## Checkpoints

| Stage | Content at `141c093` |
| --- | --- |
| `candidates` | `candidates.<csv|parquet>`: identity, spot columns, then `sig_<round>_<channel>` and `valid_<round>` (`io/_checkpoint.py:196-220`); `candidates.json` with the detection plan, the extraction config, labels and metadata (`:458-484`). Reloading rebuilds the `SpotFindingResult` and `IntensityExtractionResult` (`:496-518`). |
| `pre_qc` | `pre_qc.<csv|parquet>`: the decoding table unchanged; `pre_qc.json` with the decoder config, labels and the JSON-representable diagnostics (`:523-531`). It holds the decoding table only: no extracted values, no background, no score and no filtering. Reloading rebuilds a `BarcodeDecodingResult` without its array diagnostics (`:534-540`). |
| Version | `FORMAT_VERSION = 2`; readers accept 1 and 2 (`:20-22`). |
| Rerun | From `candidates`: decode and filter without images; from `pre_qc`: filter only ({doc}`checkpoints`, "Reload and continue"). There is no filtering checkpoint. |

## YAML keys and their translation

`from_workflow_config` (`dataset/workflow.py:301-415`) reads these keys of a Python
rule:

| Key | Translation |
| --- | --- |
| `load_codebook.split_index` | Integer list; empty or missing gives `None`; one element is taken as is; more raise (`:352`, `:407-411`). Passed unchanged to `Dataset.load_codebook` (`:426`): the one-based MATLAB position is used as a zero-based Python index (previous section). `load_codebook.run` is not read; the codebook is loaded whenever decoding runs (`:425-426`). |
| `reads_extraction.voxel_size` | `NeighborhoodSumConfig(tuple(voxel_size))`, default `(1, 2, 2)`, read as `(z, y, x)` (`:350`, `:390`). |
| `reads_filtration` | `run` enables both decoding and filtering. Decoding is always `WtaDecoderConfig(diagnostics=True)` (`:391`); the codebook-aware decoder cannot be selected from YAML. The filter gets `end_base` (one pair; a list such as the MATLAB default `["CC"]` raises `invalid endpoint bases`), `start_base` (default `C`), `exclude_invalid_endpoints` and `score_bounds` (`:351`, `:392`). |
| `reads_filtration.n_barcode_segments`, `reads_filtration.split_index` | Accepted keys, but `n_barcode_segments` other than 1, or a nonempty `split_index`, raises `ValueError("segmented endpoint filtering is not supported by ReadFilterConfig")` (`:380-381`). The schema allows both (`workflow/schemas/config.schema.yaml:968-986`). |

## The multi-round candidate set left by §2.7

{doc}`spot-finding-contract` ("Detection in several rounds", option A) added a
`round` column to candidates of a `SpotFindingPlan` with `rounds`, one table and
namespace for all listed rounds, and never merges coincident candidates. Extraction
reads every sequencing round at every candidate (`dataset/fov.py:1131-1135`).
Decoding such a set raises
`ValueError("decoding candidates from several detection rounds needs a readout mode (§2.8)")`
in `FOV.run` (`:1018-1020`) and in `FOV.decode_barcodes` for any spot table with a
`round` column (`:940-941`, message `:33`). There is no direct readout: no gene
mapping per round and channel, and no extraction limited to a candidate's own round.

## Evaluation

`starfinder.evaluation.evaluate_decoding(decoded, truth, *, matches)`
(`evaluation/barcode.py:8-51`) gives gene and color-sequence accuracy over
`match_points` pairs. There is no ranking measure for a score (AUROC, error at
fixed retention) and no duplicate measure; W-278 computed both in its own scripts.

## Discrepancies with the agreed §2.8 scope

| # | Discrepancy | Proposed resolution |
| --- | --- | --- |
| 1 | Two segments exist only as `EncodingConfig.split_index`: no segment count, lengths in bases and colors, acquisition order, junction rule or per-segment end bases. | A typed `BarcodeLayout` on the codebook ({doc}`readout-contract`, "Segment layout"); `split_index` stays the shared MATLAB-facing key and is translated at the workflow boundary. |
| 2 | `dataset/workflow.py` passes the shared one-based `split_index` to Python without converting its index base (`:407-411`, `:426`). On the 11-base example the expected sequence is `242324242` and the obtained one `423242423`. | Subtract one at the workflow boundary and state the index base in the contract; a test with this example. The fix is drafted as its own criterion of the task-group-2 issue, not made here. |
| 3 | The workflow adapter rejects `n_barcode_segments` other than 1 and a nonempty `reads_filtration.split_index` (`:380-381`). | Translate them into the segment layout (`n_barcode_segments` must equal the number of segments the layout's split produces) and accept the MATLAB two-segment configuration. |
| 4 | Segment end bases are checked for one pair with one start base over the whole sequence (`filtering.py:22-23`, `:138-147`); YAML lists of pairs raise. | Per-segment sets of allowed (first, last) pairs, each segment decoded separately; the check stays diagnostic unless exclusion is requested. |
| 5 | A codebook row is treated as a gene: repeated `gene_id` raises (`codebook.py:100-101`), `n_genes` counts rows, `gene_to_seq` maps a gene to one sequence. The aging codebook has 14,242 entries for 2,044 genes. | An entry identifier separate from the gene identifier; several entries may map to one gene; collisions are checked on entries ({doc}`readout-contract`, "Codebook entries and genes"). |
| 6 | `valid` is always true (`extraction.py:161`), so `invalid_measurement` never occurs in a pipeline run. | `valid` false for rounds not extracted (the direct mode's other rounds) and kept as the decoders' required-round rule; the semantics are qualified in the contract. |
| 7 | No background or noise measurement exists per (spot, channel, round); the §2.7 image median and MAD are per round and channel only. | A local ring estimate kept with the traces, as W-278 recommends ({doc}`readout-algorithms`, "Background and noise"). |
| 8 | The filter's score names are a fixed list of eight (`filtering.py:37-47`). | Bounds on any numeric score column the input result declares, including the shared score; unknown names still raise. |
| 9 | `FOV.run` has no shared score stage and no deduplication between decoding and filtering (`fov.py:1142-1148`). | The order extract → decode or assign → score → deduplicate → filter, with deduplication off by default. |
| 10 | `pre_qc` holds the decoding table only (`_checkpoint.py:523-531`). | `pre_qc` holds the read table after scoring and deduplication, before filtering, with each stage's config. |
| 11 | Multi-round candidates raise on decoding (`fov.py:33`, `:940-941`, `:1018-1020`). | A `direct` readout mode assigns them by round and channel; `multiplexed` mode keeps the error. |
| 12 | `CodebookAwareDecoderConfig` defaults to `allow_rescue=True` (`decoding.py:62`), while the agreed default is rescue off. | The pipeline default decoder stays WTA, which never rescues, so the agreed default holds without a change. Whether the codebook-aware config itself should default to `allow_rescue=False`, which would change its golden digest, is an open choice ({doc}`readout-contract`, "Runtime order and defaults"). |
| 13 | The YAML route always decodes with WTA (`workflow.py:391`). | A Python-only `decoding` block selects the decoder and its settings. |
| 14 | `no_signal` and `invalid_measurement` rows keep the codebook-aware scores (`decoding.py:267-276`; golden spot 5). | Scores of reads without a call are NaN in the shared score; the decoder's own columns are kept as diagnostics and documented. |
| 15 | MATLAB drops `M`/`N` reads before membership (`STARMapDataset.m:880-887`); Python keeps every row with a status. | Recorded; Python keeps every row, which the agreed scope requires. |
| 16 | `starfinder.evaluation` has no ranking or duplicate measure. | Add `ranking_quality` and `evaluate_deduplication` ({doc}`readout-algorithms`, "Engineering validation design"). |

## Golden test

`src/python/test/test_readout_golden.py` pins the current behavior on its own
seeded fixture (seed 20261002): four uint16 rounds of 12×48×48 voxels with four
channels, a background of 20 grey levels with noise sd 3, and ten hand-placed
candidates, each a Gaussian of amplitude 800 in its color's channel with 5 % in the
next channel. The codebook holds eight 5-base barcodes (two-base, reversed, one
segment) written as a `gene,barcode` CSV. The fixture contains:

* a round whose two brightest channels tie exactly (spot 4, round 2): WTA
  `ambiguous` / `tied_channels`; the codebook-aware decoder rescues it as
  `rescued_unknown`;
* a round with zero signal in every channel (spot 5, round 4): `no_signal` /
  `zero_signal_round` in both decoders;
* a read one substitution from a codeword (spot 6, observed `1244` for `1234`):
  WTA `unmatched` / `not_in_codebook`; codebook-aware `rescued_h1` with a corrected
  margin of 0.068;
* a read two substitutions from every codeword (spot 7, `2424`): `unmatched`
  (`not_in_codebook`; codebook-aware `no_candidate`);
* a spot at y = 1 whose box is clipped by the y = 0 face (spot 3), still `valid`.

It pins with exact SHA-256 digests (a) the extracted tensor and `valid` for two
radii; (b) the decoding table of both decoders at their defaults; (c) the filtering
table with the default filter, with one upper score bound (`wta_l2_nll` ≤ 0.069,
`probability_nll` ≤ 1.105) and with the end-base check `CC`; (d) the `pre_qc` table
after `FOV.run`, written as a CSV checkpoint and reloaded, with the digests of
`candidates.csv` and `pre_qc.csv`; (e) the same digests when the pipeline and
codebook come from the legacy keys through `from_workflow_config` (WTA only, the
only decoder those keys select). Three tests show that changing the neighborhood
radius, a codebook-aware gate (`allow_rescue`, `max_corrected_round_margin`,
`min_geomean_probability`) or a score bound changes a digest. Two tests document
legacy behavior that §2.8 changes: `valid` is all true, and a candidate table with a
`round` column raises on decoding. Every configuration is built by one helper,
`readout_config`, which holds the only imports of extraction, decoding, filtering,
encoding, detection, pipeline and checkpoint config types.

Three separate single-thread processes (`taskset -c 0`, every thread variable 1)
recomputed every pinned value with byte-identical output
(`scripts/w279_compute_pins.py` in the W-279 run directory; each took 3.4 s and
188 MB peak RSS), so the test uses exact equality.
