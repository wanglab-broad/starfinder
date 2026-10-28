# Readout reference

The decoders, encodings, options, columns, files and workflow keys of readout. The
method and option tables are read from the code when the site is built. The
{doc}`readout guide <readout-contract>` explains how to use them; the
{doc}`readout API <api/readout>` documents every class.

## Decoders

```{method-table} starfinder.readout.DECODING_METHODS
```

Each decoder's name is its `method` value, used by the workflow's `decoding` block.
`wta` and `codebook_aware` decode readout mode `multiplexed`, and `direct` assigns the
reads of readout mode `direct`.

### Winner-take-all

Takes the brightest channel of each round as its color and looks the color sequence up
in the codebook; its calls are exact matches. See
{py:class}`~starfinder.readout.WtaDecoderConfig`.

```{config-options} starfinder.readout.WtaDecoderConfig
```

### Codebook-aware

Scores the codebook entries near the observed sequence with the channel probabilities.
With `allow_exact` it calls exact matches, and with `allow_rescue` it can rescue a read
to a nearby entry when its rescue gates pass.
See {py:class}`~starfinder.readout.CodebookAwareDecoderConfig`.

```{config-options} starfinder.readout.CodebookAwareDecoderConfig
```

### Direct

Assigns each candidate the panel gene of its own round and channel. It has no tunable
option.
See {py:class}`~starfinder.readout.DirectDecoderConfig` and
{py:class}`~starfinder.readout.DirectPanel`.

```{config-options} starfinder.readout.DirectDecoderConfig
```

## Encodings

```{method-table} starfinder.readout.ENCODINGS
```

The encoding is the codebook's `encoding`, given to `Dataset.load_codebook`. Its name is
the `method` of the workflow's `load_codebook.encoding` block.

```{config-options} starfinder.readout.EncodingConfig
```

```{config-options} starfinder.readout.OneBaseEncodingConfig
```

See {py:class}`~starfinder.readout.EncodingConfig` and
{py:class}`~starfinder.readout.OneBaseEncodingConfig`.

### Segment layout

A {py:class}`~starfinder.readout.BarcodeLayout` lists the barcode's
{py:class}`~starfinder.readout.Segment`s and the order in which their colors are
acquired.

```{config-options} starfinder.readout.BarcodeLayout
```

```{config-options} starfinder.readout.Segment
```

## Extraction options

{py:class}`~starfinder.readout.NeighborhoodSumConfig` sets the summed box, and
{py:class}`~starfinder.readout.LocalBackgroundConfig` the ring of the local background
and noise.

```{config-options} starfinder.readout.NeighborhoodSumConfig
```

```{config-options} starfinder.readout.LocalBackgroundConfig
```

## Score, deduplication and filter options

{py:class}`~starfinder.readout.ReadScoreConfig` has no tunable option.
{py:class}`~starfinder.readout.DeduplicationConfig` sets the duplicate rule, and
{py:class}`~starfinder.readout.ReadFilterConfig` the predicates of the filter.

```{config-options} starfinder.readout.ReadScoreConfig
```

```{config-options} starfinder.readout.DeduplicationConfig
```

```{config-options} starfinder.readout.ReadFilterConfig
```

## Read table

The read table has one row per candidate. Each stage adds its columns to the table of
the stage before:

| Column | Written by | Contents |
| --- | --- | --- |
| `spot_id`, `spot_namespace` | every decoder | The candidate's identity |
| `round`, `channel` | `direct` | The candidate's detection round and channel label |
| `observed_color_sequence` | every decoder (missing for `direct`) | The color of the brightest channel of each round; `M` marks tied channels |
| `gene_wta` | `codebook_aware` | The gene of the observed sequence, if it is an entry's |
| `decoded_color_sequence` | every decoder (missing for `direct`) | The color sequence of the called entry; missing without a call |
| `gene_id`, `entry_id` | every decoder | The called gene and codebook entry (`<round>/<channel>` for `direct`) |
| `call_status` | every decoder | `assigned`, `unmatched`, `ambiguous` or `no_signal` |
| `failure_reason` | every decoder | Why the read has no call, such as `not_in_codebook` or `zero_signal_round` |
| `call_type` | every decoder | `exact`, `rescued_h<k>`, `rescued_unknown`, `direct` or `no_call` |
| `wta_l2_nll` | `wta` | The decoder's score |
| `probability_nll`, `score_delta`, `geomean_probability`, `min_round_margin`, `corrected_round_margin`, `mean_total_intensity`, `hamming_to_wta`, `corrected_rounds` | `codebook_aware` | The decoder's scores and the rounds a rescue changed |
| `own_channel_rank`, `own_channel_fraction` | `direct` | The rank of the candidate's channel in its round (1 is the brightest) and its share of the round's total |
| `qc_score`, `qc_ambiguity_max`, `qc_signal_to_background`, `qc_rounds` | scoring | The shared read-QC score (lower ranks as more reliable) and its components |
| `qc_reason` | scoring | Empty when scored, else `no_assignment` or `background_unavailable` |
| `duplicate_group`, `duplicate_of` | deduplication | The `spot_id` that represents the read's group (missing outside a group), and, for a merged read, of its representative |
| `is_representative`, `duplicate_reason` | deduplication | Whether the read represents itself or its group, and why it was grouped |
| `endpoint_valid`, `endpoint_valid_<segment>` | filtering, with an end-base check | Whether the read's end bases, or each segment's, are allowed |
| `accepted`, `rejection_reasons` | filtering | Whether the read passed, and the failed predicates joined by `;` |

`filter_reads` and `FOV.run` keep every row; `ReadFilteringResult.accepted` gives the
accepted ones.

## Molecule table

After the filter, `FOV.export_spots()` writes the accepted reads as `molecules.csv`, a
typed CSV:

| Column | Type | Contents |
| --- | --- | --- |
| `spot_namespace`, `spot_id` | string | The candidate's identity |
| `z`, `y`, `x` | float64 | Zero-based voxel coordinates of the candidate |
| `gene_id` | string | The called gene |
| `entry_id` | string | The called codebook entry |
| `qc_score` | float64 | The shared read-QC score; `<NA>` for a read without a score |

After decoding, `FOV.export_spots("reads")` writes `reads.csv`: every decoded read
joined to its candidate, with `spot_namespace`, `spot_id`, `z`, `y` and `x` first.

## Diagnostics

| Function | Returns |
| --- | --- |
| `inspect_read` | One read's sums, background, noise and colors, one row per round and channel |
| `plot_read` | A matplotlib figure of `inspect_read`'s table |
| `summarize_reads` | A summary of the reads: counts by status, reason, call type, gene and entry, score quantiles and intensity medians |
| `explain_read` | One read's decisions in order, from decoding to the filter, with their values and limits |

With checkpoints, `FOV.run` stores `summarize_reads` in `run.json` under
`counts.summary` whenever it decoded reads; the others run on demand. See
{py:func}`~starfinder.readout.summarize_reads`.

## Files

| File | Written by | Contents |
| --- | --- | --- |
| `checkpoints/candidates.csv` or `.parquet`, with `candidates.json` | `FOV.run` with the `candidates` stage | The candidates; after extraction, their sums and `valid`, and the box sizes and local background when measured |
| `checkpoints/pre_qc.csv` or `.parquet`, with `pre_qc.json` | `FOV.run` with the `pre_qc` stage, when it decodes, scores or deduplicates | The read table before the filter, with the score and duplicate columns of the stages that ran |
| `run.json` | `FOV.run` with `checkpoints` | The readout mode, the configs, the codebook's encoding (`multiplexed`) and, after decoding, the read summary |
| `molecules.csv` | `FOV.export_spots()`, after the filter | The accepted reads (above) |
| `reads.csv` | `FOV.export_spots("reads")`, after decoding | Every decoded read |

Paths are inside the FOV's folder, `<output_root>/fovs/<fov_id>/` by default;
{doc}`checkpoint reference <checkpoints-reference>` describes them.

## Workflow keys

With `backend: python`:

| Key | What it is for | Fields |
| --- | --- | --- |
| `readout_mode` (top level) | How reads get their identity | |
| `load_codebook` | The codebook's segment layout and encoding | `split_index`, `encoding` |
| `reads_extraction` | Extraction and the local background | `run`, `voxel_size`, `background` |
| `reads_filtration` | Decoding and the read filter | `run`, `end_base`, `start_base`, `exclude_invalid_endpoints`, `score_bounds`, `n_barcode_segments`, `split_index` |
| `decoding` | The decoder | `method` and the decoder config's fields |
| `scoring` | The shared read-QC score | `run`, `method` |
| `deduplication` | Optional deduplication | `run` and the `DeduplicationConfig` fields |

The keys go under `rules.<rule>.parameters`, except `readout_mode`.
{doc}`workflow-configuration` gives each key's values and rules ("Readout mode",
"Codebook segment layout", "Codebook encoding table").
