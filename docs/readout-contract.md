# Readout contract: extraction, decoding and read QC

Status: Accepted (W-280, 2026-10-02, at b1c7261)

This page proposes the §2.8 readout contract: two readout modes (multiplexed
sequencing and direct readout) as one explicit setting, registered barcode
encodings, a typed segment layout for one- and two-segment codebooks, codebook
entries separate from genes, local background and noise measurements kept with the
traces, one shared read-QC score, optional cross-channel deduplication, complete
retained outputs with three kinds of diagnostics, and the checkpoint and workflow
rules that go with them. It builds on {doc}`method-registry`, whose registry fields,
lookup, dependency and provenance rules it uses unchanged, and on the option-A
multi-round candidate set of {doc}`spot-finding-contract`. The current behavior is
recorded in {doc}`readout-baseline`; the numerical methods, worked examples and the
engineering validation design are in {doc}`readout-algorithms`. Nothing here is
implemented, and nothing here renames anything: the existing names
(`starfinder.barcode`, `BarcodeDecodingResult`, `PipelineConfig.decoding`) stay.

The measured evidence comes from W-278, run directory
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-278/20261002T011718Z-0cfc7de7`
(handoff `worker-notes.md`; tables `estimators.csv`, `scores.csv`, `direct.csv`,
`duplicates.csv`, `extraction-cost.csv`; `readout-manifest.json`,
`selection.json`). The algorithm page cites the table rows behind each number.

## Settled decisions this page follows

Decided by Jiahao in the §2.8 planning session (W-152 comment "§2.8 planning
decisions (2026-10-01)"):

1. **D1** Two readout modes at the top level, `multiplexed` (sequencing with a
   codebook) and `direct` (a round and channel identify a gene), in the existing
   package. One- and two-segment barcodes are layouts within `multiplexed`. This page
   proposes the names of results and configuration keys; W-280 decides any rename.
2. **D2** Barcode encodings are registered on the W-240 mechanism: `two_base` (the
   current behavior) and `one_base`, which §2.8 implements and validates on
   hand-built known answers, with no real one-base data.
3. **D3** The segment layout is a typed description, not a registry: the number of
   segments, each segment's length, the acquisition order, the junction treatment and
   the allowed end bases, with several first/last pairs per segment. `split_index`
   stays the shared MATLAB-facing key and is translated at the workflow boundary.
4. **D4** The `split_index` index-base defect is fixed by an implementation issue;
   W-279 demonstrates it and drafts the fix.
5. **D5** A codebook entry is not a gene: the entry identifier is separate from the
   gene identifier (aging: 14,242 entries for 2,044 genes).
6. **D6** MATLAB's two-segment path matches the complete concatenated color string;
   its per-segment end-base checks are diagnostic ({doc}`readout-baseline`).
7. **D7** Direct readout: each round is detected (option A of the §2.7 multi-round
   candidate set), each candidate is extracted in its own round, the other rounds are
   unavailable in `valid`, and the barcode rules (codeword competition, rescue,
   required rounds) do not apply.
8. **D8** A binary on/off code is not a segment layout; §2.8 states the extension
   point and delivers no binary entry, decoder or example.
9. **D9** Every encoding and layout has a worked example checked by a test, the
   two-segment one on the 11-base 5 + 4 design, and direct readout a mapping example
   ({doc}`readout-algorithms`, "Worked examples").
10. **D10** Left to this specification, with evidence, and decided at W-280: the
    background and noise estimator; the score components and ranking design; the
    duplicate distance, signal compatibility, grouping and representative ranking;
    the `pre_qc` content and any `FORMAT_VERSION` change; the golden-test scope; and
    whether extraction must be vectorized. This page proposes each of them.

The W-279 issue and the W-152 scope comments of 2026-09-27 add the rest of the
agreed scope: complete retained outputs with three kinds of diagnostics (read
inspection, population summaries, decision inspection); the runtime order extract →
decode or assign → score → deduplicate → filter, with the defaults WTA, rescue off,
deduplication off, and assigned calls kept with no score cutoff; and bounded
engineering validation only, with no score cutoff, calibrated probability or
recommended default filter (W-152 §2.14 decision, 2026-09-29).

Facts about the two-segment assay (Jiahao, 2026-10-01): the two segments are
separate barcodes in the experiment and are stored joined in the codebook; the
color across the junction has no meaning; the aging design reads 5 + 4 colors over
9 rounds; segment end bases differ between segments and between probe types (`CC`,
`TT`), and a segment may start and end with different bases (for example `CA`); the
aging codebook has 14,242 entries for 2,044 genes by design. Python `split_index` 4
corresponds to MATLAB 5.

## Terms

* A **candidate** is one row of the spot table (`(spot_namespace, spot_id)`).
* A **read** is the readout of one candidate: one row of the read table.
* A **readout mode** says how a read gets its identity: from a color sequence over
  all sequencing rounds (`multiplexed`) or from the candidate's own round and channel
  (`direct`).
* An **encoding** maps a barcode's bases to a color sequence and back.
* A **segment layout** describes how one codebook barcode is cut into separately
  read segments and in which order their colors are acquired.
* A **codebook entry** is one barcode of the codebook with its color sequence; a
  **gene** is what the entry measures. Several entries may share a gene.
* A **call** is a read's identity decision (`call_status`, `call_type`); the **score**
  ranks calls and never changes them.

## Names

| Option | Names | Effect on the golden digests | Effect on the checkpoints | Effect on the MATLAB-facing keys |
| --- | --- | --- | --- | --- |
| **N1. Extend in place (recommended)** | Keep `starfinder.barcode`, `BarcodeDecodingResult`, `decode_barcodes`, `filter_reads`, `EncodingConfig`, `PipelineConfig.decoding`. Add the names in the table below. Direct readout returns a `BarcodeDecodingResult` with `readout_mode="direct"`. | None from the names themselves; the helper `readout_config` keeps its imports. Column additions are covered under "Checkpoints and reruns". | Header keys stay (`decoding_config`, `extraction_config`); new keys are added. | None; every new key is Python-only. |
| N2. Neutral new names | A module `starfinder.readout` holding the stage, `ReadCallResult` replacing `BarcodeDecodingResult`, `PipelineConfig.calling` replacing `decoding`, YAML `read_calling` replacing the `decoding` block; old names as aliases for one release. | Digests unchanged, but the helper and every test importing the old names need edits, and the aliases must be tested. | `pre_qc.json` would rename `decoding_config`; a reader alias or `FORMAT_VERSION` 3 is needed, and `test_registration_recipe.py:520-525` and `test_registration_validation.py:433` pin version 2. | None. |

**Recommendation: N1.** The batch keeps the existing names, the direct mode fits the
existing read table (its identity columns are the same), and N2 costs a format
version and alias tests for a rename alone. "Barcode" in a direct-mode result is
the price of N1.

New names under N1:

| Concept | Name | Kind |
| --- | --- | --- |
| Readout mode | `Dataset.readout_mode`, `"multiplexed"` (default) or `"direct"`; YAML top-level `readout_mode` | Dataset field |
| Encoding registry | `ENCODINGS: dict[type, EncodingSpec]` in `starfinder.barcode`; entries `two_base` (`EncodingConfig`, which gains `method="two_base"`) and `one_base` (`OneBaseEncodingConfig`) | registry |
| Decoding registry | `DECODING_METHODS: dict[type, DecodingSpec]` in `starfinder.barcode`; entries `wta`, `codebook_aware`, `direct` | registry (the "Decoding" row of {doc}`method-registry` names it `DECODERS`; this page uses the `*_METHODS` pattern of decision 2 there) |
| Segment layout | `BarcodeLayout`, `Segment`; `Codebook.layout` | frozen dataclasses |
| Codebook entries | `Codebook.table` column `entry_id`; `n_entries`, `seq_to_entry`, `entry_to_seq` | codebook |
| Direct mapping | `DirectPanel`, `load_direct_panel`; `Dataset.direct_panel`, `Dataset.load_direct_panel` | dataset |
| Direct assignment | `DirectAssignmentConfig`, `assign_direct` | config, function |
| Background | `LocalBackgroundConfig`; `NeighborhoodSumConfig.background`; result fields `background`, `noise`, `background_voxels`, `box_voxels`, `image_background`, `image_noise` | config, result fields |
| Score | `ReadScoreConfig`, `score_reads`, `ReadScoringResult`; `PipelineConfig.scoring` | config, function, result |
| Deduplication | `DeduplicationConfig`, `deduplicate_reads`, `ReadDeduplicationResult`; `PipelineConfig.deduplication` | config, function, result |
| Diagnostics | `inspect_read`, `summarize_reads`, `explain_read` | functions |
| Evaluation | `ranking_quality`, `evaluate_deduplication` in `starfinder.evaluation` | functions |

Where the mode lives:

| Option | Effect on the golden digests | Effect on the checkpoints | Effect on the MATLAB-facing keys |
| --- | --- | --- | --- |
| **`Dataset.readout_mode` (recommended)** | None (default `multiplexed`). | `candidates.json` and `pre_qc.json` record it; a checkpoint without it loads as `multiplexed`. | A new top-level YAML key, next to `n_rounds` and `seq_channel_order`; MATLAB scripts never read it, and the schema rejects `direct` unless `backend: python`. |
| `PipelineConfig.readout_mode` | None. | The same records. | A rule parameter key repeated in every Python rule. |

The mode describes the assay, like the codebook and the panel that the dataset
already holds, so it is a dataset setting. `FOV.run` reads it from `fov.dataset`.

## Readout modes

| | `multiplexed` (default) | `direct` |
| --- | --- | --- |
| Reference | `Dataset.codebook`: entries with color sequences over all sequencing rounds | `Dataset.direct_panel`: a gene per (round, channel) |
| Candidates | One table without a `round` column (reference-round detection, or a plan without `rounds`) | One table with a `round` column (a plan with explicit `rounds`, option A) |
| Extraction | Every sequencing round at every candidate (today) | Each candidate's own round only; the other rounds are `valid=False` |
| Identity | Decoder over the color sequence (`wta`, `codebook_aware`) | `direct`: the panel gene of the candidate's own (round, channel) |
| Score | All rounds | The candidate's own round |
| Deduplication | Optional | Not available |

A decoder declares the modes it supports in `DecodingSpec.modes`: `wta` and
`codebook_aware` support `multiplexed`, `direct` supports `direct`. `FOV.run` and
`decode_barcodes` raise `TypeError` naming the mode and the decoder when they do not
match, and `ValueError` when the mode's reference (codebook or panel) is not loaded.
In `multiplexed` mode a candidate table with a `round` column still raises
`ValueError`; its message keeps the words "needs a readout mode (§2.8)" and adds
"set readout_mode='direct'".

## Encoding registry

`ENCODINGS` follows {doc}`method-registry`: the shared fields (stable `name`, exact
config type as the key, `requires`, provenance entry) and the lookup helpers
(`spec_for`, `config_type_for`, `names`) are those of "Fields of the shared spec"
and "Shared and stage-specific capabilities". Each encoding config has a `method`
discriminator equal to its spec name. `EncodingSpec` adds:

| Field | Meaning |
| --- | --- |
| `encode(bases, config) -> str` | The color sequence of one segment's bases, in acquisition order. |
| `decode(colors, config, first_base) -> str` | The bases of one segment from its colors; `first_base` is `None` for encodings that do not need it. |
| `colors_for(n_bases) -> int` | Colors produced by a segment of `n_bases` bases. |
| `needs_first_base` | `True` when decoding needs the segment's first base in read orientation. |
| `junction_colors` | Colors that span two adjacent segments of the barcode and are never acquired. |
| `alphabet` | The color symbols, `"1234"` for both entries; colors map to channels through `Codebook.color_to_channel`. |
| `symbols` | `"color"`: one color per round. A decoder that supports `"color"` encodings needs nothing else from the encoding. |

| Name | Config (parameters, defaults) | `encode` | `decode` | `colors_for(n)` | `needs_first_base` | `junction_colors` |
| --- | --- | --- | --- | --- | --- | --- |
| `two_base` | `EncodingConfig(reverse_bases=True)`; the legacy `split_index` stays as a field, translated into a two-segment layout and mutually exclusive with `Codebook.layout` | `encode_bases(bases[::-1] if reverse_bases else bases)` | `decode_color_sequence(colors, first_base)`, reversed back when `reverse_bases` | `n − 1` | yes | 1 |
| `one_base` | `OneBaseEncodingConfig(base_to_color, reverse_bases=False)`: `base_to_color` maps `A`, `C`, `G`, `T` to distinct colors `1`–`4` and has no default | `"".join(base_to_color[b] for b in bases)`, reversed first when `reverse_bases` | the inverse mapping | `n` | no | 0 |

**What a decoder declares.** `DecodingSpec` (registry `DECODING_METHODS`, same
shared fields) adds `modes` (above), `encodings` (the encoding `symbols` kinds it
decodes), `rescue` (whether it can change an observed color) and `score_columns`
(the numeric columns it writes, which `filter_reads` may bound). `wta` and
`codebook_aware` declare `encodings={"color"}`: they decode one color per round
from the four-color alphabet and use nothing else from the encoding, so they
support `two_base` and `one_base` unchanged. `direct` declares no encoding.
`decode_barcodes` raises `TypeError` when the codebook's encoding kind is not
among the decoder's.

**Adding a different code type.** A binary on/off code (one bit per channel per
round, as in MERFISH) is not a color sequence. Adding one needs: an `EncodingSpec`
with a new `symbols` kind (for example `"binary"`) and its own codeword
representation and validation in `Codebook`; at least one decoder registered with
that kind in `encodings`; its own score components; and its own validation checks.
`wta` and `codebook_aware` do not declare it, so pairing them with such a codebook
raises. This page registers no such entry.

## Segment layout

A segment layout is a typed description on the codebook, not a registry:

```python
@dataclass(frozen=True)
class Segment:
    name: str                                   # e.g. "A"
    bases: int                                  # length in bases, barcode order
    ends: tuple[tuple[str, str], ...] = ()      # allowed (first, last) bases, read orientation; () = unchecked

@dataclass(frozen=True)
class BarcodeLayout:
    segments: tuple[Segment, ...]               # in barcode order, as written in the codebook file
    acquisition_order: tuple[str, ...] | None = None   # segment names in round order; None = barcode order
```

| Element | Rule |
| --- | --- |
| Number of segments | `len(segments)`, at least 1. The default layout is one segment covering the whole barcode, which is today's behavior. |
| Length in bases | `Segment.bases`; the lengths must sum to every entry's barcode length. |
| Length in colors | `encoding.colors_for(bases)`: `bases − 1` for `two_base`, `bases` for `one_base`. The colors of all segments must sum to the number of sequencing rounds. |
| Acquisition order | `acquisition_order` lists the segments in the order their colors are acquired; segment k's colors occupy consecutive rounds. Each segment is encoded on its own bases (with `reverse_bases` applied within the segment), and the color sequence is the segments' colors concatenated in acquisition order. |
| Junction | For `two_base` the color of the base pair that spans two adjacent segments in barcode order belongs to no segment: it is never acquired, never decoded and never compared. `one_base` has no junction color. |
| Allowed ends | `Segment.ends` is a set of allowed (first, last) base pairs of the segment in read orientation (after `reverse_bases`). First and last may differ (for example `("C", "A")`), and a segment may allow several pairs (for example `(("C", "C"), ("T", "T"))` for two probe types). A codebook entry whose segment ends are not among them raises `ValueError` naming the entry and segment. |
| Membership | A read matches an entry on the whole concatenated color sequence, as in MATLAB. The end-base check of a read is per segment (decoding each segment from each allowed first base and requiring the matching last base); it is diagnostic and rejects only with `ReadFilterConfig.exclude_invalid_endpoints=True`. |

**`split_index` translation.** `split_index` stays the shared MATLAB-facing key
(`load_codebook.split_index`, `reads_filtration.split_index`). Its value `s` is
MATLAB's **one-based** position, in the encoded color string, of the junction color
that is removed. For a `two_base` barcode of `n` bases with `reverse_bases=True`,
the workflow boundary translates `s` into `BarcodeLayout((Segment("A", n − s),
Segment("B", s)), acquisition_order=("A", "B"))`, which equals today's
`EncodingConfig(split_index=s − 1)` (zero-based). With `reverse_bases=False` the
same `s` gives `BarcodeLayout((Segment("A", s), Segment("B", n − s)),
acquisition_order=("B", "A"))`, again equal to `EncodingConfig(split_index=s − 1)`:
the legacy split always acquires first the colors that follow the removed color.
The translation lives only in `dataset/workflow.py`; Python callers state the layout
or the zero-based `EncodingConfig.split_index`. For aging (`s = 5`, 11 bases) the
layout is A = 6 bases (5 colors, rounds 1–5) and B = 5 bases (4 colors, rounds 6–9),
the 5 + 4 order of {doc}`readout-baseline`.

| Option | Representation | Effect on the golden digests | Effect on the checkpoints | Effect on the MATLAB-facing keys |
| --- | --- | --- | --- | --- |
| **S1. Typed `BarcodeLayout` on the codebook (recommended)** | As above; `EncodingConfig.split_index` kept as a legacy field and translated; per-segment `ends` on the layout; `ReadFilterConfig.end_bases` kept as the one-segment shortcut. | None: the golden codebook has one segment, the default. | `pre_qc.json` records the layout; a checkpoint without it loads with the one-segment layout. | Unchanged keys (`split_index`, `end_base`, `n_barcode_segments`), translated at the workflow boundary with the index base stated. |
| S2. MATLAB-style flat keys in Python | `EncodingConfig.split_index` with its index base fixed, `ReadFilterConfig.end_bases` as one pair per segment, and a segment count. | None. | Config fields only. | A one-to-one mirror of the MATLAB keys. Lengths, acquisition order and several pairs per segment cannot be stated; the order stays "the part after the split first", which depends on `reverse_bases`. |
| S3. Segment columns in the codebook file | Each `genes.csv` row carries its segments. | None for one segment. | A wider codebook record. | Changes `genes.csv`, which MATLAB reads; rejected. |

**Recommendation: S1.** It states every element the assay needs (lengths, order,
junction, several end pairs), keeps the MATLAB keys, and reproduces today's codebook
exactly through the translation.

## Codebook entries and genes

| Rule | Specification |
| --- | --- |
| Columns | `entry_id` (string, unique), `gene_id` (string, may repeat), `color_sequence` (unique), optional `base_sequence`. |
| Legacy files | A headerless or `gene,barcode` `genes.csv` row gives `gene_id` = the first column and `entry_id` = the barcode as written, which is unique by the collision rule. |
| Canonical files | Header `entry_id,gene_id,color_sequence[,base_sequence]`; without `entry_id`, `entry_id` = `color_sequence`. |
| Collisions | Repeated `entry_id`, repeated `color_sequence` (even for one gene) and repeated `base_sequence` raise `ValueError` naming both source rows. A repeated `gene_id` is allowed. |
| Lookups | `n_entries` (rows), `n_genes` (distinct genes), `genes` (distinct, in first-appearance order), `seq_to_entry`, `entry_to_seq`, `seq_to_gene`. `gene_to_seq` stays for codebooks whose genes are unique and raises `ValueError` otherwise. |
| Decoding table | Reports both `entry_id` and `gene_id` of the decoded entry (`gene_wta` keeps the gene of the observed sequence). Codeword competition and rescue are between entries: two entries of one gene are different candidates, and a tie between them is `ambiguous`, as today. |
| Filters | Act on reads; `call_statuses` and score bounds are unchanged. |
| Outputs | The exported spot CSV keeps `gene` (= `gene_id`) and adds `entry` only when asked through `columns`; population summaries count per gene and per entry. |

| Option | Effect on the golden digests | Effect on the checkpoints | Effect on the MATLAB-facing keys |
| --- | --- | --- | --- |
| **E1. `entry_id` column, always present (recommended)** | The codebook, decoding, filtering and `pre_qc` tables gain `entry_id`. Named edit: each digest is computed on the table without `entry_id` and equals today's pin, and one new pin covers the column. | `pre_qc.<format>` gains a string column; `FORMAT_VERSION` stays 2; an older reader keeps it as an ordinary column. | `genes.csv` unchanged; the exported CSV unchanged by default. |
| E2. `entry_id` only when declared | Golden unchanged (no declared entries). | The column appears for some codebooks only. | Unchanged. Tables change schema with the codebook, which every consumer must handle. |
| E3. The color sequence is the entry | Golden unchanged (no new column). | Unchanged. | Unchanged. Entries have no name that survives a change of encoding or layout; the `split_index` fix itself would rename every aging entry. |

**Recommendation: E1.** A stable entry name independent of the encoding is what the
aging codebook needs, and the golden edit is mechanical.

## Direct readout

| Element | Specification |
| --- | --- |
| Mapping | `DirectPanel(table)` with columns `round`, `channel` (channel label) and `gene_id`. Each (round, channel) appears once; each gene appears once, and a repeated gene raises `ValueError` naming it. Rounds must be sequencing rounds and channels must be labels of the dataset. A (round, channel) may be absent (for example a stain channel). `load_direct_panel(path, *, round_labels, channel_labels)` reads a CSV with that header. |
| One read per candidate | The read table has one row per candidate, with `round` and `channel`; `entry_id` is `"<round>/<channel>"`. |
| Identity | From the candidate's own `round` and `channel` columns only. The brightest channel does not change it: no reassignment, no merge across channels or rounds. |
| Extraction | Only each candidate's own round is extracted. The values of the other rounds are `0.0` with `valid=False`. |
| Statuses | `assigned` (`call_type="direct"`); `unmatched` with `unmapped_channel` (no gene for that (round, channel)); `no_signal` with `zero_signal_round` (every channel of the own round sums to 0); `unmatched` with `invalid_measurement` (the own round is not valid). No `ambiguous`: ties do not decide the identity. Diagnostic columns `own_channel_rank` (1 = brightest in its round) and `own_channel_fraction` record when another channel is brighter. |
| Barcode rules that do not apply | Competition between codewords, rescue (a codebook-aware config raises in `direct` mode), required rounds (only the own round is read), encodings, segment layout, end-base checks and the WTA color sequence (`observed_color_sequence` and `decoded_color_sequence` are missing values). |
| Score | The shared score over the own round only (below). |

**Amendments to the option-A interface of {doc}`spot-finding-contract`.** Option A
stays as specified: one table, a `round` column, `spot_id` over the combined table,
and coincident candidates never merged. This page amends three rules:

1. "Extraction accepts a multi-round candidate set unchanged: it reads every
   sequencing round at every candidate" holds in `multiplexed` mode only. In
   `direct` mode each candidate is extracted in its own round, and the other rounds
   are `valid=False`.
2. The decoding guard becomes a mode check: a `round` column requires
   `readout_mode="direct"`, and `direct` mode requires a `round` column (a plan
   with explicit `rounds`; listing only the reference round is allowed).
3. Candidates detected in a round that the panel does not name are `unmatched` with
   `unmapped_channel`, not an error.

## Extraction

**Retained and qualified unchanged.** Neighborhood sums (`values`, N×C×R float64),
nearest sampling `floor(coord + 0.5)`, the zero boundary (a clipped box sums fewer
voxels), half-widths in voxel counts, identities, labels and the default radius
`(1, 2, 2)` stay exactly as in {doc}`readout-baseline`. `valid` keeps its meaning,
"false marks an unavailable measurement, not zero signal", and is now set false for
the rounds that `direct` mode does not extract; in `multiplexed` mode it stays all
true. A new `box_voxels` (N×R int64) records how many voxels each box summed, so
clipping is visible.

**Added per (spot, channel, round)**, from the W-278 recommendation (estimator
`local_ring`):

| Field | Units | Definition |
| --- | --- | --- |
| `background` | grey levels per voxel of the extraction image | Median over the ring: the voxels within the outer box `outer_radius_zyx=(1, 6, 6)` and outside the inner box `inner_radius_zyx=(1, 3, 3)` around the rounded centre (the box's own z-planes at lateral Chebyshev distance 4 to 6; 360 voxels unclipped). |
| `noise` | grey levels per voxel | 1.4826 × the median absolute deviation over the same ring. |
| `background_voxels` (per spot and round) | count | Ring voxels inside the image. |
| `image_background`, `image_noise` (per round and channel) | grey levels per voxel | Median and 1.4826 × MAD of the whole round and channel, the §2.7 noise record, kept beside the local estimate. |

A box's background in sum units is `background × box_voxels`. The measurement is
configured by `NeighborhoodSumConfig.background = LocalBackgroundConfig(
inner_radius_zyx=(1, 3, 3), outer_radius_zyx=(1, 6, 6), min_voxels=16)` and is on by
default; `background=None` turns it off. The inner box must contain the extraction
box.

* **Next to other spots.** The ring is not masked. A neighbor whose centre lies in
  the ring raises the background median by about 2 grey levels (p90 +5) at the
  calibrated density and 3 (p90 +6) in the dense scene, and the noise estimate by
  33 % (50 % dense); a neighbor in the excluded centre raises it by about 1 (W-278
  `estimators.csv`; {doc}`readout-algorithms`). This bias is recorded, not removed:
  masking neighbors would be a new extraction algorithm.
* **At the border.** The ring is clipped to the image and the estimate uses the
  remaining voxels: a median of 207 of 360 voxels when the box itself is clipped,
  282 when only the ring is (W-278).
* **Evidence and limits.** W-278 chose the ring on its design seeds (`selection.json`:
  realized-error median 2.22 grey levels against 2.41 and 3.39); held out, it is 2.28
  (p90 5.87), 2.16 to 2.48 per condition (`estimators.csv`, `local_ring`,
  `stratum=all`). W-278 lists these limits: the ring tracks the realized, not the
  latent background; its MAD underestimates the analytic noise by about 12 % on
  clipped uint8 data; synthetic data only. The row-level citations are in
  {doc}`readout-algorithms` ("Background and noise").
* **Vectorization (D10).** Not required. W-278 measured the current per-spot loop on
  `medium` (32×512×512, 587 candidates) at a median of 0.0586 s, 25 µs per spot and
  round, while a full-image box-filter prototype took 3.26 s for the same sums
  (`extraction-cost.csv`). The `large` and `tissue` tiers were not measured.
* **Failure values.** With fewer than `min_voxels` ring voxels inside the image,
  `background` and `noise` are NaN for that (spot, round), and the score of a read
  that uses that round is NaN with reason `background_unavailable`. `values` stay
  finite and `valid` is not changed by it.

## Shared read-QC score

`score_reads(decoding_result, intensity_result, *, reference, config=ReadScoreConfig())`
returns a `ReadScoringResult` whose table is the read table with score columns
added; every row is kept and no identity column changes.

| Element | Specification |
| --- | --- |
| Inputs | The read table (any mode and decoder); the extraction result with `values`, `background`, `box_voxels` and `valid`; the reference (codebook or panel) for the assigned channels. |
| Ranking score `qc_score` | W-278 design D1: the decoder's probability NLL of the assigned entry recomputed on background-subtracted sums. Per used round r: `v'_c = max(v_c − box_voxels × background_c, 0)`, `p = (v'_a + 1e-6) / Σ_c (v'_c + 1e-6)` for the assigned channel `a`; `qc_score = Σ_r −log max(p, 1e-12)`. Lower ranks as more reliable. |
| Components kept | `qc_ambiguity_max` (the largest, over used rounds, of the strongest other channel's background-subtracted sum over the assigned channel's, both clipped at 0), `qc_signal_to_background` (mean over used rounds of `(v_a − box_voxels × background_a) / (box_voxels × background_a)`, with the **signed**, unclipped numerator, as W-278 measured `sbr_mean__local_ring`), `qc_rounds` (rounds used). The ambiguity component and `qc_score` use the clipped `v'`, also as W-278 measured them. |
| Used rounds | `multiplexed`: every sequencing round. `direct`: the own round only. The weakest-round and codeword-support components do not apply in `direct` mode and are not computed (W-278 `direct.csv`, `not_applicable`). |
| Decoders and call types | One score for `wta`, `codebook_aware` and `direct`, and for exact and rescued calls. It is computed on the assigned entry's channels, so a rescued round is scored at the codeword's channel. Exact and rescued calls are distinguished by `call_type` (`exact`, `rescued_unknown`, `rescued_h<k>`), which the score keeps; summaries report scores per `call_type`. |
| Reads without an identity | `qc_score` and the components are NaN with `qc_reason` `no_assignment` (`unmatched`, `ambiguous`, `no_signal`), or `background_unavailable`. |
| What it is not | A ranking of calls, not an error probability: it is not calibrated, sets no cutoff, recommends no filter and never changes or rescues an identity (`gene_id`, `entry_id`, `call_status`, `call_type`). |

`ReadScoreConfig` has `method="bgcorr_probability"` and no tunable parameter; the
constants 1e-6 and 1e-12 are the decoder's.

Evidence and limits: W-278 chose D1 on its design seeds (`selection.json`, mean
exact-call AUROC 0.874 against 0.714 for the decoder score). On its held-out seeds
D1 ranks exact calls at AUROC 0.897 against 0.782 (WTA) and 0.743 (codebook-aware)
and rescued calls at 0.851 against 0.670 (`scores.csv`, `heldout`,
`all_calibrated`, pooled), and direct-readout calls at 0.930 (`direct.csv`). W-278
lists these limits: synthetic data only; few incorrect calls (46 exact, 12 rescued),
so no per-condition tolerance; no score separates unmatched detections from correct
calls; scores are rankings, not calibrated probabilities; no cutoff. The row-level
citations are in {doc}`readout-algorithms` ("Shared read-QC score").

## Optional deduplication

`deduplicate_reads(result, spots, intensity_result, *, config)` returns a
`ReadDeduplicationResult`; `PipelineConfig.deduplication` is `None` by default
(off).

| Element | Specification |
| --- | --- |
| Pairs | Candidates of the same detection round in different channels (`channel` column). Same-channel pairs are never grouped here (that is the §2.7 within-channel `merge_radius_zyx`). |
| Distance | Euclidean distance between candidate coordinates in zero-based voxel index space, inclusive: `≤ distance_voxels`, default 1.0 voxel (W-278 design choice). At the repository example voxel size (0.094 µm in XY, 0.35 µm in Z) 1 voxel is 0.094 µm laterally or 0.35 µm along Z; the rule uses index space, not microns. |
| Signal compatibility | `compatibility="same_sequence"`: both reads have identical WTA observed color sequences (the W-278 rule, which tested sequence equality and distance only) and neither sequence contains `M` or `N`. The `M`/`N` exclusion is a proposal **W-278 did not measure**: its `duplicate_pairs` and `rule_merges` compared sequences without excluding them. The exclusion only removes links, so the proposed rule links a subset of the measured rule's pairs. |
| Grouping | Connected components of the graph of compatible pairs. **Not measured by W-278**, which scored pairs only (below). |
| Representative | One original candidate per group, never a new or averaged one: among the members with an `assigned` call (all members if none is assigned), the one with the largest extracted sum in its own detection channel in the detection round; ties go to the earliest row of the spot table. Its coordinates, values and identity are unchanged. {doc}`readout-algorithms` states the same rule. **Not measured by W-278.** |
| Conflicting or ambiguous calls | A group whose assigned members do not all share one `entry_id` is not merged: every member stays a representative, with reason `conflicting_calls`. Reads with `M` or `N` in their sequence are never compatible (the unmeasured exclusion above). Unassigned members of a merged group are marked duplicates of its assigned representative. **Not measured by W-278.** |
| Direct mode | Not available: `deduplicate_reads` raises `ValueError` in `direct` mode, because different channels and rounds are different genes there and no direct-mode read is merged or reassigned automatically. The population summary still counts cross-channel candidate pairs within `distance_voxels` in one round. |
| Retained | Every row, with `duplicate_group` (the representative's `spot_id`, or missing when not grouped), `duplicate_of` (the representative's `spot_id` for a merged member), `is_representative` (true for representatives and for ungrouped reads) and `duplicate_reason` (`""`, `same_sequence_within_distance`, `conflicting_calls`); `counts` holds groups, merged reads and conflicting groups. |
| Filter | `ReadFilterConfig.exclude_duplicates=True` (default) rejects reads with `is_representative` false, with reason `duplicate`; it has no effect on a result that was not deduplicated. |

Evidence and limits: W-278 chose `same_sequence` at d = 1 on its design seeds
(`selection.json`; 6 false merges against 26 at d = 2). On its held-out seeds it
makes 5 false merges among 763 distinct cross-channel pairs within 5 voxels (0.66 %;
`duplicates.csv`, `heldout`, `all_calibrated`, `row_type=rule`). W-278 lists these
limits: the held-out scenes hold no detectable crosstalk copy, so there is no
held-out missed-duplicate evidence; outside the split, d = 1 misses 9 of the 22
W-218 copies and d = 2 none (`reference_case`). The distance is an open choice
recorded with W-279. Row-level citations are in {doc}`readout-algorithms`
("Deduplication").

What W-278 measured is the pairwise rule only: for each pair of candidates in
different channels, whether `same_sequence` within d links them (W-278
`scripts/w278_lib.py`, `rule_merges`), with missed duplicates and false merges
counted over pairs. The grouping into connected components, the representative
choice and the conflicting-call rule above are additions that W-278 did not measure;
on scenes where every group is a single pair they give the same merges as the pairwise
rule, and the validation design marks the checks that depend on them as provisional.

## Runtime order and defaults

`FOV.run` runs extract → decode (`multiplexed`) or assign (`direct`) → score →
deduplicate → filter, each stage optional and each reading only the previous
stages' retained results.

| Setting | Python API (`PipelineConfig`) | Workflow adapter (YAML) |
| --- | --- | --- |
| Readout mode | `Dataset.readout_mode="multiplexed"` | top-level `readout_mode`, default `multiplexed` |
| Decoder | `decoding=None` (opt-in, as every operation) | `WtaDecoderConfig(diagnostics=True)` when `reads_filtration.run`; the Python-only `decoding` block may name `codebook_aware` or `direct` |
| Rescue | `CodebookAwareDecoderConfig` keeps `allow_rescue=True` (open choice below) | off: the adapter passes `allow_rescue=False` unless the `decoding` block sets it |
| Background | `NeighborhoodSumConfig.background=LocalBackgroundConfig()` | on |
| Score | `scoring=None` | `ReadScoreConfig()` whenever decoding runs; the Python-only `scoring: {run: false}` turns it off |
| Deduplication | `deduplication=None` | off; the Python-only `deduplication` block turns it on |
| Filter | `ReadFilterConfig()`: `assigned` reads, no score bound | the same; no score cutoff is recommended |

Whether `CodebookAwareDecoderConfig` itself should default to `allow_rescue=False`
is open: it would match the agreed default for direct Python callers but change the
codebook-aware golden digests (b) and (d).

## Checkpoints and reruns

| Option | Layout | Effect on the golden digests | Effect on the checkpoints | Effect on the MATLAB-facing keys |
| --- | --- | --- | --- | --- |
| **C1. Extend the existing stages (recommended)** | `candidates` adds `bg_<round>_<channel>`, `noise_<round>_<channel>`, `bgvox_<round>`, `boxvox_<round>` and `candidates.json` adds `background_config`, `image_background`, `image_noise` and `readout_mode`. `pre_qc` holds the read table after scoring and deduplication, before filtering, and `pre_qc.json` adds `scoring_config`, `deduplication_config`, `layout`, `readout_mode` and `stages_applied`. | `candidates.csv` and `pre_qc.csv` gain columns. Named edit: the tables without the new columns, written by the same writer, have today's digests; new pins cover the new files. | Same stages and files; `FORMAT_VERSION` stays 2, under the wire layout below. A reader at `141c093` then loads a new `multiplexed` checkpoint: the new `candidates` with the background columns dropped (they are not in `spot_columns`) and the new `pre_qc` with the extra columns kept. It cannot load a `direct` `pre_qc`: its reader maps only `wta` and `codebook_aware` (`io/_checkpoint.py:537`) and raises on `direct`. A checkpoint written at `141c093` loads with the new reader, with `background=None` and no score. | None. |
| C2. New stages, version 3 | New stages `measurements` (background) and `scored` (score and deduplication) beside the unchanged `candidates` and `pre_qc`; `FORMAT_VERSION` 3. | None for the existing files. | Two more stages for `CheckpointConfig.stages`, `clear_stages` and `load_checkpoint`; version 3 makes `test_registration_recipe.py:520-525` and `test_registration_validation.py:433` need named edits. | None. |
| C3. A separate `qc` stage, version 2 | `pre_qc` unchanged (decoding only); a new `qc` stage holds the score and deduplication columns, joined by identity. | None for `pre_qc`; `candidates.csv` still gains the background columns unless they also move to `qc`. | A fourth stage name; the read table is split across two files. | None. |

**Recommendation: C1.** `pre_qc` then means what its name says, everything before
the QC filter, and no format version changes. An older reader loses nothing it could
use from a `multiplexed` checkpoint; `direct` checkpoints need the new reader.

**Wire layout of C1.** Today's loader rebuilds a config by passing every serialized
field to its constructor (`_config`, `io/_checkpoint.py:104-109`), so a field added to
a saved config would make a `141c093` reader raise `TypeError`. C1 therefore keeps
every saved config at its `141c093` fields and stores the new settings as separate
header keys:

* `candidates.json`: `signals.extraction_config` holds exactly
  `neighborhood_radius_zyx`, `sampling` and `boundary` (the writer leaves out
  `NeighborhoodSumConfig.background`); the new top-level keys are `background_config`
  (the `LocalBackgroundConfig` fields, or `null` when off), `image_background` and
  `image_noise` (per round and channel) and `readout_mode`. The table adds
  `bg_<round>_<channel>`, `noise_<round>_<channel>` (float64), `bgvox_<round>` and
  `boxvox_<round>` (int64) after the `valid_<round>` columns.
* `pre_qc.json`: `decoding_config` holds the decoder's fields as today (`direct`
  configs carry only `method`); the new top-level keys are `scoring_config`,
  `deduplication_config` (`null` when the stage did not run), `layout`,
  `readout_mode` and `stages_applied`. The table adds `entry_id`, the score columns
  and, after deduplication, `duplicate_group`, `duplicate_of` (string),
  `is_representative` (bool) and `duplicate_reason` (string).
* The new reader rebuilds the decoder config through `DECODING_METHODS` and treats a
  missing key as its `141c093` meaning: no `background_config` is `background=None`,
  no `layout` is one segment, no `readout_mode` is `multiplexed`, and no
  `scoring_config` or `deduplication_config` means the stage did not run.

**Reruns without images.**

* `load_checkpoint("candidates")`, then `run(PipelineConfig(decoding=..., scoring=...,
  deduplication=..., filtering=...))`: decode or assign, score, deduplicate and filter
  from the retained values and background.
* `load_checkpoint("candidates")` and `load_checkpoint("pre_qc")`, then `run` with
  `scoring`, `deduplication` and `filtering`: rescore or deduplicate again without
  decoding (`run` accepts a stage whose inputs are resident).
* `load_checkpoint("pre_qc")`, then `run(PipelineConfig(filtering=...))`: filter only.
* Scoring a checkpoint without background raises `ValueError` naming the stage to
  rerun (extraction).

`run.json` keeps `format_version` 1 and records the mode, each stage's config and
the population summary under `counts`.

## Diagnostics

| Component | What it shows | Where it is produced |
| --- | --- | --- |
| Read inspection | For one read: per round and channel the sum, the background times `box_voxels`, the background-subtracted sum, the channel probability and the noise; the observed and assigned color per round; for `two_base` the decoded bases per segment. | `inspect_read(intensity_result, reads, spot_id, *, reference)` returns a table; `plot_read` draws it with matplotlib. On demand; never called by `FOV.run`. |
| Population summaries | Counts by `call_status`, `failure_reason` and `call_type`; per gene and per entry; `qc_score` quantiles per `call_type`; per round and channel the medians of sums, background and noise; deduplication and filtering counts; `valid` and `background_unavailable` counts. | `summarize_reads(...)`; `FOV.run` stores it in `run.json` (`counts`) and `FOV.save_diagnostics` writes it beside the existing counts. |
| Decision inspection | For one read: the ordered decisions and their inputs: the decoder status and reason, the codebook-aware candidates with their scores and every gate value against its limit, the per-segment end-base checks, the score components, the deduplication group and representative, and each filter predicate with its reason. | `explain_read(reads, spot_id, *, intensity_result=None, reference=None)` returns a table. On demand; the codebook-aware candidates need `diagnostics=True`. |

## Workflow configuration

| Key | Rule |
| --- | --- |
| `readout_mode` (top level) | Python-only, `multiplexed` (default) or `direct`; the schema accepts `direct` only with `backend: python`. |
| `load_codebook.split_index` | Shared; MATLAB's one-based position `s`; translated into the two-segment layout (equal to zero-based `EncodingConfig.split_index = s − 1`). |
| `reads_filtration.n_barcode_segments`, `reads_filtration.split_index` | Shared; accepted. `n_barcode_segments` must equal the number of segments of the layout and `reads_filtration.split_index` must equal `load_codebook.split_index`, else `ValueError`. |
| `reads_filtration.end_base` | Shared; a string or a list. With one segment, every listed pair is an allowed end of that segment. With two segments, item k gives the allowed ends of segment k in acquisition order, as `FilterReadsMultiSegment.m` reads it. `start_base` is no longer needed when the ends are given and is kept for one-segment compatibility. |
| `reads_extraction.background` | Python-only; `false` or a mapping of `LocalBackgroundConfig` fields. |
| `decoding` | Python-only; `method` (`wta`, `codebook_aware`, `direct`) and that config's fields. |
| `scoring`, `deduplication` | Python-only; `run` and the config fields. |
| Codebook input | In `direct` mode the rule's codebook input file is the panel CSV (`round,channel,gene_id`). |

The schema lists of `decoding.method` and of the encodings are kept equal to the
registries by a default-tier test, as for the other stages ({doc}`method-registry`,
"YAML naming per stage").

## `docs/migration.md` entries

The implementation adds these entries:

1. **Readout mode.** `Dataset.readout_mode`, `DirectPanel`, `load_direct_panel`,
   `DirectAssignmentConfig`, `assign_direct`; the YAML key `readout_mode`; decoding a
   multi-round candidate set needs `readout_mode="direct"`.
2. **Encodings and decoders.** `ENCODINGS` with `two_base` (`EncodingConfig`, now
   with `method`) and `one_base` (`OneBaseEncodingConfig`); `DECODING_METHODS`;
   exact-type lookup.
3. **Segment layout.** `BarcodeLayout`, `Segment`, `Codebook.layout`; per-segment
   end bases. **Intentional change:** the workflow adapter converts the shared
   `split_index` from MATLAB's one-based position; a configuration that worked
   around the defect by giving the Python value (for example 4 for aging) must give
   the MATLAB value (5).
4. **Codebook entries.** `entry_id`; repeated genes allowed (D5); `n_entries`;
   `gene_to_seq` raises for codebooks with repeated genes. **Intentional change:**
   two barcodes for one gene, which raised before, now load as two entries, and the
   decoding table gains `entry_id`.
5. **Background measurements.** `LocalBackgroundConfig`,
   `NeighborhoodSumConfig.background` (on by default), the new result fields and the
   candidate columns.
6. **Shared score.** `ReadScoreConfig`, `score_reads`, `PipelineConfig.scoring`;
   **intentional change:** the workflow adapter scores whenever it decodes.
7. **Deduplication.** `DeduplicationConfig`, `deduplicate_reads`,
   `PipelineConfig.deduplication` (off); `ReadFilterConfig.exclude_duplicates`.
8. **Filtering.** Score bounds on any declared score column; per-segment end bases.
9. **Checkpoints.** The new `candidates` columns and `pre_qc` content;
   `FORMAT_VERSION` stays 2.
10. **Diagnostics and evaluation.** `inspect_read`, `summarize_reads`, `explain_read`;
    `ranking_quality`, `evaluate_deduplication`.

## Tests the implementation changes

* `test/test_readout_golden.py`: every pinned value stays, with these named edits
  only: the body of `readout_config` (registry configs); the digests of the codebook,
  decoding, filtering and `pre_qc` tables computed after dropping `entry_id` and the
  score columns, plus new pins for the extended tables; `candidates.csv` re-pinned,
  with an assertion that the table without the background columns, written by the
  same writer, has today's digest. `test_valid_is_always_true_at_the_border_and_in_a_zero_round`
  and `test_candidates_with_a_round_column_raise_on_decoding` stay unchanged, because
  they run in `multiplexed` mode. If Jiahao changes the codebook-aware rescue default,
  its digests (b) and (d) are re-pinned with a comment naming the decision.
* `test/test_readout_examples.py`: the reference functions of the expected `one_base`
  and direct-readout behavior are replaced by calls to `ENCODINGS` and
  `assign_direct`; every asserted value stays.
* `test/test_barcode.py`: one named edit. The case `"A,CACGC\nA,CATGC\n"` of
  `test_invalid_csv_has_row_context` (`test_barcode.py:59-74`) requires two different
  barcodes for one gene to raise, which D5 reverses. The task-group-2 issue removes
  that case and adds a test that the same rows load as two entries of gene `A`. The
  other cases of that test still raise (the identical-row case as a repeated
  `entry_id`), and the rest of the file is unchanged.
* `test/test_spot_finding_rounds.py:196-208` stays unchanged: the `multiplexed`
  message keeps "readout mode (§2.8)".
* Every other existing test passes unchanged, in particular `test_encoding.py`, `test_extraction.py`, `test_codebook_aware_decoder.py`,
  `test_checkpoints.py`, `test_fov.py`, `test_e2e.py`,
  `test_coordination_contract.py`, `test_spot_finding_golden.py` and the
  registration checkpoint-version tests. Each draft implementation issue names its
  set.

## Exclusions

No binary or MERFISH-style code, decoder or example; no new extraction algorithm,
background-subtracted extraction, spot fitting, trace summing, consolidated-position
re-extraction or incomplete-barcode inference; no external decoder; no automatic
direct-mode reassignment or merging; no score cutoff, calibrated probability or
recommended default filter; no real data. Comparisons and real-data cutoffs belong
to E03.
