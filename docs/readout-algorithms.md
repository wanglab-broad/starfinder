# Readout algorithm specification

Status: Accepted (W-280, 2026-10-02, at b1c7261)

This page specifies the numerical methods of §2.8: the `two_base` and `one_base`
encodings, the WTA and codebook-aware decoders, direct assignment, the local
background and noise estimator, the shared read-QC score and the deduplication rule.
It gives worked examples and the engineering validation design of task group 6. The
registries, configs, modes and records they plug into are in {doc}`readout-contract`;
the current behavior is in {doc}`readout-baseline`. Nothing here is implemented,
and nothing here sets a score cutoff, a calibrated probability or a default filter.

## Evidence

Every measured number comes from W-278, run directory
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-278/20261002T011718Z-0cfc7de7`
(found with `find … -maxdepth 2 -name readout-manifest.json`; handoff
`worker-notes.md`). It is design selection for a development contract on synthetic
data, not a method comparison. W-279 repeats no measurement; it reads the tables
with `scripts/w279_cite_w278.py` in its own run directory. The tables cited are:

* `estimators.csv`: background and noise estimators per condition and stratum;
* `scores.csv`: score components and designs per condition, decoder and call class;
* `direct.csv`: the same on each sequencing round as a four-gene panel;
* `duplicates.csv`: cross-channel pair distances and the duplicate rules;
* `extraction-cost.csv`: the extraction loop on `medium`.

A row is cited as table, `split`, `condition`, and the row's key columns. Unless
noted, the cited rows are `split=heldout` (seeds 103, 104 and 105, pooled
`pooled:103,104,105`), `condition=all_calibrated`. The scenes are
`calibrated_scene_preset` (calibrated-development-v1, a balanced 16-gene codebook),
8×64×64 voxels, four rounds and four channels, uint8, 80 amplicons (2.44 per 1000
voxels); `dense` has 160 (4.88 per 1000). The pipeline is `LocalMaximaConfig()` on
round 1, `NeighborhoodSumConfig((1, 2, 2))` and both decoders at their defaults.
Matching is `match_points` with policy `greedy`, threshold 3.0, boundary
`inclusive`, units `voxel`, then `evaluate_decoding`; the reference is the formed
amplicon centres with `center_in_bounds` in the detection round. A call is
incorrect when its matched amplicon has a different gene; assigned calls without a
match are a separate population, unmatched detections.

## What each element addresses

| Element | Problem addressed | Cause |
| --- | --- | --- |
| `two_base` | STARmap barcodes are read as color transitions of adjacent bases. | Sequencing by ligation reads each pair of adjacent bases as one of four colors. |
| `one_base` | Assays whose color directly encodes one base per round. | One fluorophore per base, through a known base-to-color assignment. |
| Segment layout | Two separately read barcodes stored joined in one codebook string. | The junction color has no meaning, segments are read in their own rounds, and their end bases differ by probe type. |
| WTA | A fast call from the brightest channel per round. | Each round's amplicon signal is strongest in its color's channel. |
| Codebook-aware | Reads one color away from a codeword because of a weak or tied round. | Channel mixing, round weakening and noise flip or tie a round's brightest channel. |
| Direct assignment | One gene per (round, channel) panels. | Each round images up to four genes; a candidate's channel is its identity. |
| Local background | A background that varies over the field and across channels and rounds. | Broad and regional background, gain and trend differ per channel and round; a global median misses them. |
| Shared score | Ranking calls by reliability across decoders and modes. | A read's correctness depends on how clearly its assigned channel stands above the others once background is removed. |
| Deduplication | One amplicon read twice, as a candidate in its own channel and as a crosstalk copy in the next. | 5 % crosstalk into the next channel makes a weaker copy that local maxima detects correctly (W-218 re-measurement, {doc}`spot-finding-baseline`). |

## Encodings

| Field | `two_base` | `one_base` |
| --- | --- | --- |
| Algorithm | Reverse the segment's bases when `reverse_bases`, then map each adjacent base pair to a color (`AA, CC, GG, TT` → 1; `AC, CA, GT, TG` → 2; `AG, CT, GA, TC` → 3; `AT, CG, GC, TA` → 4). Decoding walks the colors from a known first base: each color and the previous base fix the next base. | Reverse when `reverse_bases`, then map each base through `base_to_color`. Decoding applies the inverse mapping. |
| Parameters | `reverse_bases` (bool, default True). | `base_to_color` (mapping of `A, C, G, T` to distinct colors `1`–`4`, required, no default); `reverse_bases` (bool, default False). |
| Colors per segment | bases − 1; one junction color between segments is dropped. | bases; no junction color. |
| Failure | A base outside `ACGT`, a barcode shorter than 2 bases, or a color outside `1234` raises `ValueError` naming the codebook row. Decoding needs a first base; with several allowed first bases, each is tried (end-base check below). | A base outside the mapping, a mapping that is not one-to-one, or a missing base raises `ValueError`. |
| Resources | Linear in barcode length; the aging codebook (14,242 entries) encodes in well under a second. | Linear. |

**End-base check of one segment.** For `two_base`, the segment's colors are decoded
from each allowed first base `f`; the segment is valid when for some allowed pair
`(f, l)` the decoded bases end in `l`. For `one_base`, the decoded first and last
bases must be an allowed pair. The check is diagnostic; `endpoint_valid` is true
when every segment with declared ends is valid.

## Decoders

The algorithms are unchanged from {doc}`readout-baseline`; §2.8 changes what they
receive and report (entries, layout, modes).

| Field | WTA | Codebook-aware |
| --- | --- | --- |
| Parameters (units, defaults) | `negative_policy` (`reject`), `diagnostics` (False). | `max_hamming` (1 color), `max_corrected_round_margin` (0.20, probability, upper bound), `min_score_delta` (0.25, natural-log units), `min_geomean_probability` (0.45), `max_correction_penalty` (1.50, natural-log units), `allow_exact` (True), `allow_rescue` (True; the workflow adapter passes False), `negative_policy`, `diagnostics`. |
| Entries | Exact lookup of the observed sequence among entries; reports the entry and its gene. | Candidates are entries; two entries of one gene compete as two candidates. |
| Encodings | Any `"color"` encoding; the decoder never reads bases. | The same. |
| Failure | Statuses and reasons as in the baseline; invalid labels or negative sums raise. | The same, plus the gate reasons. |
| Resources | Vectorized, O(N·C·R). Not measured by W-278. | The one-error index costs O(entries × R × C) to build and each read a dictionary lookup; a read with `M`/`N` rounds scans the entries. Not measured by W-278; the aging codebook gives an index of 14,242 × 9 × 3 ≈ 384,000 keys. |

## Direct assignment

| Field | Specification |
| --- | --- |
| Algorithm | For each candidate, read its `round` and `channel`; look up the gene in the panel; read the own round's extracted sums; set `no_signal` when they are all 0 and `invalid_measurement` when the own round is not valid. Record `own_channel_rank` and `own_channel_fraction` (own channel sum over the round's total). |
| Parameters | None besides the panel. `DirectAssignmentConfig()` has `method="direct"` only. |
| Failure | A panel with a repeated gene or a repeated (round, channel) raises at load; a candidate table without `round` raises; a round or channel label outside the dataset raises. |
| Resources | O(N); extraction reads one round per candidate, so it costs 1/R of the multiplexed extraction. |

W-278 measured the direct-readout ranking with each sequencing round of a
calibrated scene as one four-gene panel, detecting and extracting that round only
(`direct.csv`). Its calls came from the decoders on a one-round panel, not from the
detection channel, so its counts describe the prototype, not this assignment: 7616
assigned exact calls, 411 incorrect and 458 unmatched detections (`direct.csv`,
`call_class=exact`, `decoder=wta`).

## Background and noise

| Field | Specification |
| --- | --- |
| Problem | A per-spot background and noise next to the extracted sums, so that the score sees signal above the local background. |
| Algorithm | For each candidate and round, take the voxels of the outer box `outer_radius_zyx=(1, 6, 6)` that lie outside the inner box `inner_radius_zyx=(1, 3, 3)` around the rounded centre, clipped to the image; per channel, `background` is their median and `noise` is 1.4826 × their median absolute deviation. Unclipped, that is the box's own three z-planes at lateral Chebyshev distance 4 to 6: 360 voxels. Per round and channel, the image median and 1.4826 × MAD are kept beside it. |
| Parameters (units, defaults) | `inner_radius_zyx` (1, 3, 3) and `outer_radius_zyx` (1, 6, 6) voxels; `min_voxels` 16 (**provisional**: W-278 measured no ring below 99 voxels, so the failure path is unmeasured). |
| Units | Grey levels per voxel of the extraction image; × `box_voxels` for sum units. |
| Failure | Fewer than `min_voxels` ring voxels: NaN background and noise for that spot and round, and reason `background_unavailable` in the score. |
| Resources | 24.6 µs per spot, channel and round in the W-278 prototype (`estimators.csv`, `estimator=local_ring`, `stratum=all`, `cost_us_per_estimate`, which divides the time by N×C×R estimates), so about 98.6 µs per spot and round across four channels: about four times the extraction loop itself (25 µs per spot and round on `medium`, `extraction-cost.csv`). |

**Why this estimator.** W-278 chose it on the design seeds by the smallest median
absolute error against the realized box background (`selection.json`: 2.22 grey
levels against 2.41 for the 3D shell and 3.39 for the image estimate). Held-out rows
(`estimators.csv`, `estimator=local_ring`):

| Row | Value |
| --- | --- |
| `stratum=all` (38,400 estimates) | realized-error median 2.28 grey levels (p90 5.87), against 2.45 for `local_shell3d` and 3.32 for `image`; latent bias median +1.0; noise relative error median −12.4 %. |
| `stratum=all` per condition | realized-error median 2.16 to 2.48 in all nine conditions; latent bias median +1.0 in seven, +0.87 to +0.90 in `background_only` and `combined`, +2.0 in `dense`. |
| `stratum=neighbor_in_shell` (8,949) | contamination median +2 grey levels (p90 +5); noise +33 % (`contamination_noise_rel_median` 0.33). `condition=dense`: +3 (p90 +6), noise +50 %. |
| `stratum=neighbor_in_center_only` (3,186) | contamination +1 (p90 +3). |
| `stratum=no_neighbor` (26,265) | contamination 0 (p90 +2). |
| `stratum=at_border_box_clipped` (7,488) | realized error 2.37, region median 207 of 360 voxels (minimum 99), noise −13.8 %. |
| `stratum=near_border_region_clipped` (7,216) | realized error 2.21, region median 282 voxels. |

Known limits from W-278: the ring follows the local correlated noise, so it tracks
the realized background, not the latent one; the image estimate is closer to the
latent background for uniform backgrounds (median error 1.0 against 2.83). The ring's
MAD underestimates the analytic noise by about 12 % on clipped uint8 data (and the
image MAD overestimates it by 9.5 %); both are estimates for ranking, not calibrated
noise. On uint8 data a median moves in half-grey-level steps. About 19 % of the
calibrated voxels are clipped at 0.

## Shared read-QC score

| Field | Specification |
| --- | --- |
| Problem | One ranking of call reliability for both decoders, exact and rescued calls, and both modes. |
| Algorithm | For each used round r and channel c: `v'_c = max(v_c − box_voxels_r × background_{c,r}, 0)`; `p_r = (v'_a + 1e-6) / Σ_c (v'_c + 1e-6)` for the assigned channel `a` of round r; `qc_score = Σ_r −log max(p_r, 1e-12)`. Components: `qc_ambiguity_max = max_r max(v'_{other}, 0) / v'_a` (strongest other channel), `qc_signal_to_background = mean_r (v_a − box_voxels_r × background_{a,r}) / (box_voxels_r × background_{a,r})` with the signed, unclipped numerator, `qc_rounds`. Every formula is the form W-278 measured (`scripts/w278_lib.py`, `components`: `bgcorr_probability_nll__local_ring`, `ambiguity_max__local_ring`, `sbr_mean__local_ring`). |
| Parameters | None tunable; the constants are the decoder's 1e-6 and 1e-12. |
| Orientation and meaning | Lower `qc_score` ranks as more reliable. It is a ranking, not an error probability, and is not calibrated. |
| Failure | NaN with `no_assignment` for reads without an assigned identity, and with `background_unavailable` when a used round has no background. |
| Resources | Vectorized, O(N·C·R); negligible beside extraction. |

**Why this score.** W-278 chose design D1 (`D1_bgcorr_probability`) on the design
seeds by the highest mean exact-call AUROC over conditions and decoders
(`selection.json`: 0.874 against 0.714 for the decoder score D0). Held-out rows
(`scores.csv`, `score_kind=design`, pooled):

| Row | D1 | Reference D0 (decoder score) |
| --- | --- | --- |
| `call_class=exact`, `decoder=wta` (1709 assigned: 1543 correct, 46 incorrect, 120 unmatched) | AUROC 0.897 ± 0.015; error at 90 % retention 0.013, at 80 % 0.006 | 0.782 ± 0.027; 0.021, 0.016 |
| `call_class=exact`, `decoder=codebook_aware` (same calls) | 0.897 ± 0.015 | 0.743 ± 0.030; 0.023, 0.020 |
| `call_class=rescued`, `decoder=codebook_aware` (102 assigned, 12 incorrect) | 0.851 ± 0.046; error at 90 % 0.091 | 0.670 ± 0.075; 0.114 |
| Per condition, exact | D1 ≥ D0 in all 18 condition × decoder cells; smallest margin +0.041 (`combined`, codebook-aware) | — |
| `direct.csv`, exact (7616 assigned, 411 incorrect) | 0.930 ± 0.004 | 0.878 (WTA), 0.849 (codebook-aware) |

`ambiguity_max__local_ring` matches D1 on exact calls (0.900) and direct readout
(0.944), but is weaker on rescued calls (0.697), so it is kept as a component, not
as the score (W-278 choice 3). The signal-strength components rank weakly on exact
calls (`snr_mean__local_ring` 0.505, `sbr_mean__local_ring` 0.533,
`weakest_round_snr__local_ring` 0.576). In direct readout the weakest-round and
codeword-support components do not apply (`direct.csv`, outcome `not_applicable`).

Known limits from W-278: incorrect calls are few (46 exact and 12 rescued held-out
calls pooled), so per-condition cells (2 to 18 incorrect, SE 0.016 to 0.160) support
no per-condition tolerance; rescued calls are not uniform per condition
(`weakening` 0.60 against 0.90 on one incorrect call); no score separates unmatched
detections from correct calls (`auroc_correct_vs_unmatched` 0.44 to 0.50), because
all 120 held-out unmatched calls are same-channel second maxima of amplicons that
already have a candidate (the W-218 within-channel case, whose merge was off); the
z-components scale noise as white despite correlated noise; the repair run
re-selected after the run-1 held-out results had been read (protocol caveat).

## Deduplication

| Field | Specification |
| --- | --- |
| Problem | Remove the second read of one amplicon detected in two channels. |
| Algorithm | Among candidates of one detection round in different channels, link two when their Euclidean distance in voxel index space is at most `distance_voxels` (inclusive) and their WTA observed sequences are identical (W-278's measured rule) and contain no `M` or `N` (a proposed exclusion W-278 did not measure). Groups are the connected components; a group with assigned members of different entries is not merged (`conflicting_calls`); otherwise one original candidate represents the group: among the members with an `assigned` call (all members if none is assigned), the one with the largest extracted sum in its own detection channel in the detection round, ties going to the earliest row of the spot table. The others are marked duplicates of it. This is the rule of {doc}`readout-contract`. The pairwise link is what W-278 measured; the grouping, the representative and the conflict rule are additions it did not measure. |
| Parameters (units, defaults) | `distance_voxels` 1.0 voxel (index space; 0.094 µm laterally or 0.35 µm along Z at the repository example voxel size); `compatibility` `"same_sequence"`. Off by default. |
| Failure | `ValueError` in `direct` mode; a result without decoding raises. Nothing iterates. |
| Resources | A KD-tree pair query, O(N log N) per round; negligible at the calibrated density. |

**Why this rule.** W-278 chose it on the design seeds by the fewest missed
duplicates plus false merges (`selection.json`); the design seeds hold no detectable
crosstalk copy, so the false merges decided alone: 6 at d = 1 against 26 at d = 2
(`duplicates.csv`, `split=design`, `rule=same_sequence`). Held-out rows
(`duplicates.csv`, `row_type=rule`):

| Row | Value |
| --- | --- |
| `rule=same_sequence`, `distance_voxels=1` | 5 false merges among 763 distinct pairs within 5 voxels (0.66 %); every rule gives 5 at d = 1. The 5 pair two matched amplicons of different genes that decoded to one sequence, so one call of each pair was already wrong. |
| `rule=same_sequence`, `distance_voxels=2` | 13 (1.7 %), against 31 for `distance_only`, 21 for `trace_cosine` and 17 for `channel_consistency`. |
| Missed duplicates | Undefined: `n_true_duplicate_pairs` 0; the held-out scenes hold no detectable crosstalk copy. |

Outside the seed split (not the basis of the choice): on the W-218 case
(`split=reference_case`, `condition=w218_small`, benchmark `small`, seed 42) there
are 22 crosstalk copies, all within 2.0 voxels (|dz| ≤ 2, |dy|, |dx| ≤ 1), all with
the same sequence; every rule misses 9 of 22 at d = 1 and none at d = 2 or 3, with
no false merge. The `mixing` development fixtures (`split=fixture`) each hold one
copy at distance 0, merged by every rule.

**Measured against proposed.** W-278 measured the pairwise rule: each cross-channel
pair is linked or not (`scripts/w278_lib.py`, `rule_merges`), and missed duplicates
and false merges are counted over pairs. It did not measure connected-component
grouping, representative selection or the conflicting-call rule, and its sequence
equality did not exclude `M` or `N` sequences (`duplicate_pairs`, `rule_merges`); that
exclusion is a proposal it did not measure. The exclusion can only remove links, so the
proposed rule links a subset of the pairs the measured rule links. Where a group is a
single pair the grouped result equals the pairwise one; where links chain, it can
merge pairs the pairwise rule did not link. Checks R13 to R15 therefore gate the
grouped behavior as provisional, and R14 gates the pairwise links separately: the
measured metric, for links that the `M`/`N` exclusion can only make fewer.

Known limits from W-278: no held-out missed-duplicate evidence; the 5 % crosstalk at
the calibrated noise produced no detected copy in any calibrated scene; the choice
of d = 1 against d = 2 is open (W-278 choice 1).

## Worked examples

Each example is checked by `src/python/test/test_readout_examples.py`. The
`two_base` examples run against current code. For `one_base` and direct readout the
current code checks what exists today (codebook validation and WTA decoding of
explicit color sequences, multi-round detection with `round` and `channel`,
single-round extraction); the rest is expected behavior, written in the test as small
reference functions that the implementation issues replace with the real ones.

**1. `two_base`, one segment.** Gene `Slc17a7`, barcode `CTGACC`, 5 rounds.

1. Reverse: `CCAGTC`.
2. Pairs `CC, CA, AG, GT, TC` → colors `1, 2, 3, 2, 3`: `12323`, one per round.
3. Observed `12323` → WTA finds the entry → `Slc17a7`, `assigned`.
4. Back to bases: decode from the first base `C`: `CCAGTC`; reverse: `CTGACC`.
5. End bases in read orientation: `C…C`, so the check `CC` passes.

**2. `two_base`, two segments (aging shape).** Entry `Gfap_probe1`, barcode
`CAGTACTGCAT` = segment A `CAGTAC` (6 bases) + segment B `TGCAT` (5 bases), read as
5 + 4 colors over 9 rounds.

1. Reverse the whole barcode: `TACGTCATGAC`; encode: `4242324232` (colors c1–c10).
2. c5 = `3` is the pair `TC` = (first base of B, last base of A), the junction: it is
   dropped.
3. Segment A in read orientation is `CATGAC` → `24232` (c6–c10, rounds 1–5);
   segment B is `TACGT` → `4242` (c1–c4, rounds 6–9).
4. Color sequence `242324242`. Current code: `EncodingConfig(split_index=4)`, the
   zero-based index of MATLAB's `split_index` 5.
5. Back: cut `24232 | 4242`; decode A from `C`: `CATGAC`, reversed `CAGTAC`; decode B
   from `T`: `TACGT`, reversed `TGCAT`; joined: `CAGTACTGCAT`. The end bases in read
   orientation are `CC` (A) and `TT` (B).

The same test checks the contract's `split_index` translation for every valid split
of this barcode, with and without `reverse_bases`.

**3. `one_base`.** Gene `Mbp`, barcode `GATC`, mapping `A→1, C→2, G→3, T→4`,
`reverse_bases=False`, 4 rounds.

1. `G, A, T, C` → `3, 1, 4, 2`: `3142` (expected `one_base` behavior).
2. Current code: a codebook entry `Mbp` with `color_sequence` `3142` validates, and
   WTA assigns it from observed `3142` (`exact`).
3. Back: inverse mapping `3142` → `GATC`.
4. The `two_base` encoding of the same 4 bases gives 3 colors (`343`), which is why
   `one_base` is a separate entry.

**4. Direct readout, two rounds.** Panel:

| Round | ch00 | ch01 | ch02 | ch03 |
| --- | --- | --- | --- | --- |
| round1 | Gfap | Slc17a7 | Gad1 | Mbp |
| round2 | Pvalb | Sst | Vip | Aqp4 |

Spots planted at (z, y, x) = (3, 8, 8) in round1 ch00, (3, 16, 16) in round1 ch02,
(3, 8, 16) in round2 ch01 and (3, 8, 8) in round2 ch03 (6×24×24, background 100,
noise sd 3, seed 100). Current code: `FOV.find_spots` with a plan for both rounds
gives four candidates, with `round` and `channel`, the two at (3, 8, 8) as separate
rows. Expected: the panel gives `Gfap`, `Gad1`, `Sst`, `Aqp4`, one read per
candidate; replacing `Vip` with `Gfap` makes the panel invalid. Current code:
extracting each candidate's own round only gives its detection channel as the
brightest channel.

## Engineering validation design (task group 6)

Task group 6 is engineering validation only (W-152 §2.14 decision, 2026-09-29):
known-answer fixtures with pass/fail tolerances fixed before the run, in
default-tier pytest modules, and extended-tier modules (`-m extended`, one thread,
CPU) for the checks on calibrated scenes. It has no comparison matrix, no parameter
sweep, no cutoff selection, no benefit flag and no default chosen from comparative
data; comparisons and real-data cutoffs belong to E03. The checks follow the §2.7
S-table ({doc}`spot-finding-algorithms`).

Rules:

* Every image is at most 32×64×64 voxels, four channels and four rounds, generated in
  the test. Hand-built fixtures use seeds 100, 101 and 102. The checks whose
  tolerance is derived from W-278 held-out rows run on the same scenes, seeds 103,
  104 and 105, with W-278's conditions, pipeline and matching policy ("Evidence").
* Each tolerance either cites the W-278 row it is derived from or is marked
  **provisional** with a one-line reason. A tolerance is not adjusted in the run; a
  correct implementation that cannot meet one goes to Jiahao.
* An ordering check ("A above B") compares the shared score with the decoder score on
  the same calls, as W-278's rows support; it is a pass/fail engineering check, not a
  benefit claim, and its margin is the one the rows support.

Fixtures:

| Fixture | Construction | Density |
| --- | --- | --- |
| `golden` | The golden fixture of `test_readout_golden.py` (12×48×48, 4 rounds, 10 candidates). | 10 candidates |
| `cal` | `calibrated_scene_preset` in the W-278 conditions `noise`, `mixing`, `weakening`, `gain`, `trend`, `round_effect_only`, `background_only`, `combined` (8×64×64, 4 rounds, 80 amplicons), seeds 103–105, built as in the W-278 manifest (`conditions`). | 2.44 per 1000 voxels |
| `dense` | `cal` with `count=160` (W-278 `dense`). | 4.88 per 1000 |
| `two_seg` | 16×64×64, 4 rounds: 24 amplicons of a codebook of 6-base barcodes in two 3-base segments (`load_codebook.split_index: [3]`, 2 + 2 colors), with allowed ends `CC` and `TT`; 4 reads planted one color from an entry, and 4 reads off the codebook whose segment A colors equal an entry's and whose segment B colors decode to ends outside `TT`. | 0.37 per 1000 |
| `one_base` | The `golden` geometry with a `one_base` codebook of 4-base barcodes (mapping `A→1, C→2, G→3, T→4`). | 10 candidates |
| `entries` | `cal` seed 103 with its 16-entry codebook relabeled to 6 genes (entries `e1`–`e16`, genes `g1`–`g6`, 2 to 3 entries each). | as `cal` |
| `dropout` | `golden` with (i) round 3 set to 0 in a 9×20×20 region covering 3 candidates, (ii) `valid` set false for round 2 of 2 candidates (as a caller mask), (iii) a 4-voxel-wide zero band at the x=47 face in round 4 (registration fill). | 10 candidates |
| `direct2` | The direct-readout worked example (2 rounds, 6×24×24). | 4 candidates |
| `cal_direct` | `cal` with each sequencing round as a four-gene panel (genes `color-1` to `color-4`), detected in that round only, as W-278 `direct.csv`. | as `cal` |
| `bg_const` | 12×48×48, 4 channels, constant 37 in every voxel except one spot; candidates at the centre, on each face and in a corner. | — |
| `bg_neighbor` | `bg_const` with a second spot of known amplitude centred in the ring (lateral distance 5) and one in the excluded centre (distance 3). | — |
| `crosstalk` | 16×64×64, 4 rounds: 20 amplicons, each with a 5 % copy in the next channel at a known offset: 8 at 0, 4 at 1, 4 at √2 and 4 at 2 voxels; 6 pairs of different genes 1 voxel apart in different channels; one copy whose read is rescued to a different entry than its source (a conflicting group). | 0.31 amplicons per 1000 |

Checks:

| # | Check | Fixture | Metric (source) | Pass/fail tolerance |
| --- | --- | --- | --- | --- |
| R1 | Encodings | 1,000 random 6- to 11-base barcodes per seed (100–102); `one_base` | `decode(encode(b))`; the 16 two-base pairs against `src/matlab/EncodeBases.m` | Round trip exact for both entries; the pair table equals MATLAB's; a non-bijective `base_to_color` raises. **Provisional**: contract rules; no W-278 row. |
| R2 | One and two segments | worked example 2; every valid `split_index` of an 11-base barcode with and without `reverse_bases`; `two_seg` | Codebook equality; `decode_barcodes` table; per-segment `endpoint_valid` | The layout codebook equals `EncodingConfig(split_index=s − 1)` for every case; the workflow adapter with `load_codebook.split_index: [5]` gives `242324242` (the `split_index` fix); on `two_seg`, gene calls equal those of the same colors in a one-segment codebook, every read equal to an entry passes both segments' end checks, and the 4 planted wrong-end reads pass segment A and fail segment B. **Provisional**: known answers of the layout rules. |
| R3 | `one_base` decoding | `one_base` | `evaluate_decoding` (gene accuracy) with the golden spot positions as truth | Every clean read assigned to its gene (accuracy 1.0) and statuses equal to the `golden` two-base run of the same color sequences. **Provisional**: no W-278 row; the decoders do not read the encoding. |
| R4 | Entries sharing a gene | `entries` | Decoding table; `summarize_reads` per gene | Per-entry rows (status, entry, scores) equal the run with unique genes; each gene's count equals the sum of its entries' counts; repeated `color_sequence`, `entry_id` or `base_sequence` raise naming both rows. **Provisional**: bookkeeping. |
| R5 | Required rounds and acquisition-local failures | `dropout`, both decoders | Statuses and reasons; table equality elsewhere | (i) exactly the 3 covered candidates `no_signal` / `zero_signal_round`; (ii) the 2 masked candidates `unmatched` / `invalid_measurement`, never rescued; (iii) candidates whose round-4 box lies in the band `no_signal`; every other row equal to the unperturbed run; the score NaN with `no_assignment` for all of them. **Provisional**: the golden test pins today's zero-round status. |
| R6 | Multiplexed regression | `golden`; `cal` seed 103 | Table digests after dropping the new columns | Equal to the pinned golden digests and to the `141c093` output on `cal`. **Provisional**: no W-278 row covers it; the bound is today's behavior, pinned by the W-279 golden test. |
| R7 | Direct assignment, known answer | `direct2` | Read table | Genes `Gfap`, `Gad1`, `Sst`, `Aqp4`; other rounds `valid=False` with values 0; an unmapped (round, channel) gives `unmatched` / `unmapped_channel`; a zero own round gives `no_signal`; a repeated gene or (round, channel) raises; deduplication raises. **Provisional**: new interface. |
| R8 | Direct assignment on calibrated scenes (extended) | `cal_direct` | `ranking_quality` (new) of `qc_score` and of the decoder `probability_nll` on the same direct calls | `qc_score` AUROC above the decoder score's, pooled over conditions and seeds. **Provisional**: W-278 `direct.csv` (D1 0.930 ± 0.004 against 0.878 and 0.849) measured calls made by the decoders on a one-round panel, while this assignment takes the detection channel, so the row motivates the ordering but does not establish it for these calls. |
| R9 | Background analytic | `bg_const`, `bg_neighbor` | `background`, `noise`, `background_voxels`, `box_voxels` | On `bg_const`: background exactly 37 and noise exactly 0 away from the spot; 360 ring voxels in the interior and the exact clipped counts on faces and in the corner; NaN and `background_unavailable` below `min_voxels`. On `bg_neighbor`: the ring neighbor raises the background, the centre neighbor does not. **Provisional**: analytic expectations. |
| R10 | Background on calibrated scenes (extended) | `cal`, `dense`, at truth positions | Median absolute error against the realized background (the spot-free twin, as W-278); latent bias; noise relative error; contamination (W-278 strata) | Realized-error median ≤ 2.5 grey levels in every condition (`estimators.csv`, `local_ring`, `stratum=all`: 2.16 to 2.48); latent bias median in [0, +1] in every condition except `dense`, [0, +2] there (+0.87 to +1.0; dense +2.0); noise relative error median in [−15 %, +10 %] in the strata without a shell neighbor and at the border (−12.4 % to −13.8 %); contamination median ≤ +2 (p90 ≤ +5) with a shell neighbor, ≤ +3 (p90 ≤ +6) in `dense` (`stratum=neighbor_in_shell`). |
| R11 | Score definition | `golden`; `cal` seed 103 | `qc_score` and components against the W-278 formula computed in the test; identity columns | Maximum absolute difference ≤ 1e-12; NaN with the stated reasons for reads without identity; `gene_id`, `entry_id`, `call_status` and `call_type` equal before and after scoring. **Provisional**: a formula identity, exact up to float rounding. |
| R12 | Score ranking (extended) | `cal`, both decoders | `ranking_quality` (new): AUROC with Hanley–McNeil SE and error at 50, 80, 90 and 100 % retention, per `call_type` | Exact calls, pooled: `qc_score` AUROC ≥ decoder-score AUROC + 0.05 for each decoder (`scores.csv`, `call_class=exact`: 0.897 against 0.782 and 0.743, SE 0.015 to 0.030). Rescued calls (codebook-aware): `qc_score` AUROC above `probability_nll`'s (`call_class=rescued`: 0.851 against 0.670, 12 incorrect; ordering only). Per condition: reported, not gated (2 to 18 incorrect per cell). Unmatched detections: reported, not gated. |
| R13 | Deduplication, known answer | `crosstalk` | `evaluate_deduplication` (new): missed duplicates, false merges, groups; representatives | With d = 1: the 12 copies at ≤ 1 voxel merged (missed 0); the 8 copies at √2 and 2 voxels not merged (missed 8, as planted); the 6 different-gene pairs not merged (false merges 0); the conflicting group kept with `conflicting_calls`; every representative is the source amplicon's own-channel candidate. **Provisional**: known answers; the W-218 case (9 of 22 copies beyond 1 voxel, `duplicates.csv`, `reference_case`) motivates the √2 and 2 offsets. |
| R14 | Deduplication on calibrated scenes (extended) | `cal`, `dense` | `evaluate_deduplication` | (a) Pairwise links, the metric W-278 measured (its rule had no `M`/`N` exclusion; the proposed exclusion only removes links, so W-278's rate bounds them from above): falsely linked distinct cross-channel pairs within 5 voxels ≤ 0.7 %, pooled (`duplicates.csv`, `heldout`, `same_sequence`, d = 1: 5 of 763, 0.66 %). (b) After grouping, representative selection and the conflict rule: false merges ≤ 0.7 % of the same pairs, **provisional**, because W-278 did not measure grouping. Missed duplicates: reported, not gated (no held-out true duplicates; W-278 limitation). |
| R15 | Deduplication defaults and direct mode | `golden`; `direct2` | Table equality; raised error | Without a `deduplication` config the tables equal a run without the stage; `direct` mode raises `ValueError`. **Provisional**: contract rules. |
| R16 | Checkpoint round trips | `golden`, `two_seg` and `direct2` runs; CSV and Parquet | `pd.testing.assert_frame_equal(check_exact=True)`; configs and header keys | Reloaded `candidates` (with background) and `pre_qc` (with score and deduplication) equal the originals; a `141c093` checkpoint written by the golden helper loads with no background and no score. **Provisional**: the golden test shows today's CSV round trip is exact. |
| R17 | Rerun without images | `golden`, `crosstalk` | Table equality | From `candidates`: decode, score, deduplicate and filter equal the full run; from `candidates` and `pre_qc`: rescoring and deduplication equal; from `pre_qc`: filtering equals. **Provisional**: a contract rule. |
| R18 | Filtering | `golden`, `two_seg`, `crosstalk` | Filter table | Bounds on `qc_score` and on any declared score column; `exclude_duplicates`; per-segment end bases; the default keeps every `assigned` read with no score bound. **Provisional**: contract rules. |
| R19 | Determinism | `golden`; `cal` seed 103 | SHA-256 of every table, three single-thread processes | Identical. **Provisional**: no W-278 row covers it; the W-279 golden test gave byte-identical pins in three processes. |

Metrics that must be added to `starfinder.evaluation` (task group 2, because task
groups 4 and 5 use them already):

* `ranking_quality(score, correct, *, orientation, retention=(0.5, 0.8, 0.9, 1.0))`:
  AUROC (correct ranked above incorrect), its Hanley–McNeil standard error, and the
  error at each retention level, with an undefined value and a reason when a class
  is empty (the W-278 `scores.csv` columns);
* `evaluate_deduplication(groups, source, *, pairs)`: missed duplicates (pairs of one
  source not merged), false merges (merged pairs of different sources) and their
  rates over the stated pair population (the W-278 `duplicates.csv` columns).

`match_points` and `evaluate_decoding` are used as they are.

Resource plan: every fixture is at most 32×64×64 voxels. W-278 ran 27 calibrated
scenes per split (design or held-out) in about 45 s at one thread (worker notes,
"Budget"); the extended checks run 24 to 27 scenes, so about a minute, and the
default-tier checks seconds. Each run records wall time and maximum RSS with
`/usr/bin/time -v` against the 4 GiB stop target.

### Implemented checks (W-296)

The checks are in the modules of `src/python/test/` listed below, each marked
`validation` (subsystem `barcode`). The task groups added most of them with the code
they check; `test_readout_validation.py` adds R3, R5, R8 and R19, R6 on `cal`, and the
`two_seg` parts of R2 and R16. Hand-built fixtures run with seeds 100 to 102 (`dropout`
and `one_base` are the golden geometry with each of these seeds; R6 and R19 use the
pinned golden fixture itself); the W-278-derived checks use seeds 103 to 105 with
W-278's conditions, pipeline and matching (`readout_scenes.py`). R8, R10, R12 and R14
are in the extended tier (`-m extended`; also `slow`), and R19 is `slow` in the
default tier. Every tolerance is the one in the table above, unchanged. No check is
skipped or marked as an expected failure. The values were measured at one thread on
one CPU.

| # | Module and tests | Measured value | Tolerance |
| --- | --- | --- | --- |
| R1 | `test_readout_encodings.py`, `test_r1_*` | 3,000 barcodes × 5 encoding configs round-trip exactly; the 16 pairs equal `EncodeBases.m`; 5 invalid mappings raise | exact (provisional) |
| R2 | `test_readout_layout.py`, `test_r2_*` (16 splits × `reverse_bases`, worked example 2); `test_the_shared_split_index_is_one_based`; `test_readout_validation.py`, `test_r2_two_seg_*` | layout codebooks equal the zero-based split in all 16 cases; `[5]` gives `242324242`; on `two_seg` (both decoders) the decoding table equals the one-segment run, the 24 entry reads pass both segments, the 4 wrong-end reads pass A and fail B | exact (provisional) |
| R3 | `test_readout_validation.py`, `test_r3_*` | gene accuracy 1.0 on the 6 clean reads; every column but `entry_id` equals the two-base run; statuses equal the golden pins (both decoders) | exact (provisional) |
| R4 | `test_readout_entries.py`, `test_r4_*` | per-entry rows equal the unique-gene run; each of the 6 genes' counts equals its entries' sum; repeated identifiers raise naming both rows | exact (provisional) |
| R5 | `test_readout_validation.py`, `test_r5_*` | (i) candidates 0, 3 and 9 `no_signal` / `zero_signal_round`; (ii) 1 and 8 `unmatched` / `invalid_measurement`, `no_call`; (iii) no golden box reaches the band, so it changes no read, and an added candidate in the band is `no_signal`; the other 5 rows equal the unperturbed run; the score is NaN / `no_assignment` | exact (provisional) |
| R6 | `test_readout_validation.py`, `test_r6_*`; `test_readout_golden.py` | golden extraction, decoding and filtering digests equal the pins; on `cal` seed 103, the spots, sums, `valid` and both decoding tables (without `entry_id` and score columns) equal the `141c093` digests in all 8 conditions | equal (provisional) |
| R7 | `test_readout_direct.py`, `test_r7_*`; `test_readout_deduplication.py`, `test_r15_deduplication_raises_in_direct_mode` | `Gfap`, `Gad1`, `Sst`, `Aqp4`; other round 0 and `valid=False`; `unmapped_channel`; `no_signal`; repeated gene and (round, channel) raise; deduplication raises | exact (provisional) |
| R8 | `test_readout_validation.py`, `test_r8_*` (extended) | `qc_score` AUROC 0.790 (SE 0.029) against `probability_nll` 0.731 over the nine conditions (7,158 calls, 34 incorrect); 0.730 against 0.651 over the eight `cal` conditions (5,847, 13) | above (provisional) |
| R9 | `test_readout_background.py`, `test_r9_*` | background 37 and noise 0; ring voxels 360 (interior), 240, 189 (faces), 66 (corner); NaN / `background_unavailable` below `min_voxels`; ring neighbor background 87, centre neighbor 37 | exact (provisional) |
| R10 | `test_readout_background.py`, `test_r10_*` (extended) | realized-error median 2.16 to 2.48; latent bias 0.87 to 1.0, `dense` 2.0; noise relative error −13.05 % (no neighbor), −13.80 % (border); shell contamination median ≤ 2 (p90 ≤ 5), `dense` 3 (p90 6) | W-278 `estimators.csv` |
| R11 | `test_readout_scoring.py`, `test_r11_*` | maximum absolute difference 0 on golden and the 8 `cal` conditions of seed 103, both decoders; NaN / `no_assignment`; identity columns equal | ≤ 1e-12 (provisional) |
| R12 | `test_readout_scoring.py`, `test_r12_*` (extended) | exact, nine conditions: 0.897 against 0.782 (WTA) and 0.743 (codebook-aware), 1,589 calls, 46 incorrect; rescued 0.851 against 0.670 (97, 12); the eight `cal` conditions 0.897 against 0.795 and 0.758, rescued 0.866 against 0.749 | W-278 `scores.csv` |
| R13 | `test_readout_deduplication.py`, `test_r13_*` | 12 copies at ≤ 1 voxel merged; 8 at √2 and 2 voxels not merged; 0 false merges of 6 pairs; the conflicting group kept; every representative is the source | exact (provisional) |
| R14 | `test_readout_deduplication.py`, `test_r14_*` (extended) | (a) 5 of 763 distinct pairs linked (0.655 %); (b) 5 of 763 false merges for each decoder (0.655 %); 0 true duplicate pairs, so missed duplicates are undefined | (a) W-278 `duplicates.csv`; (b) provisional |
| R15 | `test_readout_deduplication.py`, `test_r15_*` | tables equal the golden pins without the stage; direct mode raises `ValueError` | exact (provisional) |
| R16 | `test_readout_scoring.py`, `test_r16_*`; `test_readout_direct.py`, `test_r16_*`; `test_readout_validation.py`, `test_r16_two_seg_*` | golden, `direct2` and `two_seg` (with deduplication and two segments) round-trip exactly in CSV and Parquet; the emulated `141c093` checkpoint loads without background or score | exact (provisional) |
| R17 | `test_readout_scoring.py`, `test_r17_*`; `test_readout_deduplication.py`, `test_r17_*` | every rerun from `candidates`, `candidates` + `pre_qc` and `pre_qc` equals the full run (golden, `crosstalk`) | exact (provisional) |
| R18 | `test_readout_deduplication.py`, `test_r18_*` | score bounds on `qc_score` and every declared column; `exclude_duplicates` rejects the 12 copies; per-segment ends on `two_seg`; the default keeps every assigned read | exact (provisional) |
| R19 | `test_readout_validation.py`, `test_r19_*` | 164 table digests (golden, both decoders; 8 `cal` conditions of seed 103 through deduplication and filtering) identical in three single-thread processes | identical (provisional) |

The W-278 limitations bound these values: synthetic data only, one 8×64×64 uint8
field of view per seed; few incorrect calls (46 exact and 12 rescued held-out calls,
and 34 direct calls in R8); no held-out crosstalk copy, so missed duplicates on the
calibrated scenes stay unmeasured; fixed pipeline settings; and every score is an
uncalibrated ranking with no cutoff. R8 measures this assignment's direct calls
(detection channel), not the decoder calls of `direct.csv`, so its numbers are not
comparable with that row. The checks are engineering validation only: they compare no
methods and set no cutoff or default.

## Limitations

These come from W-278 and apply to every number above: synthetic data only, one
8×64×64 uint8 field of view per seed with about 19 % of voxels clipped at 0; one
thread, one CPU, no GPU; no real data and no E03 evidence; no cutoff and no default
filter, so error at fixed retention describes a ranking only; every score is a
ranking, none is calibrated; few incorrect calls (46 exact and 12 rescued held-out);
fixed pipeline settings (detection 5 MAD, box (1, 2, 2), matching at 3 voxels); no
held-out crosstalk copy, so no held-out missed-duplicate evidence; extraction cost
measured on `medium` only; z-components that treat correlated noise as white; and
the repair re-selection after the run-1 held-out results had been read.
