# Readout

Readout turns the candidates of {doc}`spot finding <spot-finding-contract>` into reads.
It sums a small box around each candidate, decodes or assigns each read, ranks the calls
with one shared quality score, can mark duplicates seen in two channels, and filters the
reads. After the filter, `FOV.export_spots` writes the accepted reads as
`molecules.csv`, which {doc}`assignment <assignment-contract>` can read.

The {doc}`readout reference <readout-algorithms>` lists every method, option, column
and file; the {doc}`readout API <api/readout>` documents the classes.

## Quick start

```python
from dataclasses import replace

from starfinder.dataset import CheckpointConfig, Dataset, PipelineConfig, RoundState
from starfinder.io import ImageLoadConfig
from starfinder.readout import NeighborhoodSumConfig, ReadFilterConfig, ReadScoreConfig, WtaDecoderConfig
from starfinder.registration import RegistrationRecipe, RegistrationStep, TranslationConfig
from starfinder.spot_finding import LocalMaximaConfig
from starfinder.synthetic import benchmark_scene_preset, generate_amplicon_dataset, save_amplicon_dataset

# A synthetic raw dataset with its codebook.
codebook, scene = benchmark_scene_preset("tiny")
generated = generate_amplicon_dataset(codebook, replace(scene, seed=42), fov_ids=("FOV_001",), preset="tiny")
save_amplicon_dataset(generated, "input")
rounds, channels = list(generated.codebook.round_labels), list(generated.channel_labels)

dataset = Dataset(input_root="input", output_root="results", dataset_id="tiny", sample_id="synthetic",
                  output_id="demo", rounds=RoundState(sequencing_rounds=rounds, reference_round="round1"),
                  channel_order=channels)
dataset.load_codebook("input/codebook.csv")
pipeline = PipelineConfig(
    load=ImageLoadConfig(channel_labels=tuple(channels)),
    registration=RegistrationRecipe((RegistrationStep(TranslationConfig()),)),
    spot_finding=LocalMaximaConfig(),
    extraction=NeighborhoodSumConfig(),
    decoding=WtaDecoderConfig(),
    scoring=ReadScoreConfig(),
    filtering=ReadFilterConfig(),
)
fov = dataset.fov("FOV_001")
fov.run(pipeline, checkpoints=CheckpointConfig(stages=("candidates", "pre_qc")))

print(fov.decoding_result)                      # reads by call status
print(fov.filtering_result)                     # accepted and rejected
print(fov.filtering_result.accepted[["spot_id", "gene_id", "call_status", "qc_score"]].head())
print(fov.export_spots())                       # results/fovs/FOV_001/molecules.csv
```

The run writes `checkpoints/candidates.csv` and `checkpoints/pre_qc.csv`, with their
`.json` headers, and `run.json` in `results/fovs/FOV_001/`.

## What it does

1. **Extract.** Sum a box around each candidate in each channel and sequencing round
   (in readout mode `direct`, in the candidate's own round), and by default measure a
   local background and noise beside each sum.
2. **Decode or assign.** In readout mode `multiplexed`, a decoder turns the sums into a
   color sequence and looks it up in the codebook. In readout mode `direct`, each
   candidate is assigned from its own round and channel.
3. **Score.** The shared read-QC score ranks the assigned reads. It never changes a
   call.
4. **Deduplicate**, if asked: mark cross-channel duplicates.
5. **Filter.** Mark each read accepted or rejected, with its reasons.

No stage removes a read.

## Methods

```{method-table} starfinder.readout.DECODING_METHODS
```

`wta` and `codebook_aware` decode readout mode `multiplexed`; `codebook_aware` can also
rescue a read whose colors are close to an entry's. `direct` assigns the reads of
readout mode `direct`.

## What you get

* **In memory**, for each stage that ran: `fov.intensity_result` holds the sums (`values`, candidates ×
  channels × rounds), `valid` and, when recorded, `box_voxels` and the local
  `background` and `noise`. `fov.decoding_result` holds one read per candidate, with
  `gene_id`, `entry_id`, `call_status`, `failure_reason`, `call_type` and the decoder's
  scores; `scoring_result` and `deduplication_result` add their columns, and
  `filtering_result` adds `accepted` and `rejection_reasons`. `fov.results` maps each
  stage that ran to its result.
* **On disk:** the `candidates` and `pre_qc` checkpoints (below), and `run.json` with
  each stage's config and, after decoding, a summary of the reads under `counts`.
  After the filter, `fov.export_spots()` writes `molecules.csv`, the accepted reads
  with their positions, genes and `qc_score`; after decoding, `fov.export_spots("reads")`
  writes every decoded read.

The reference lists every column.

## Main options

* `Dataset.readout_mode` and the codebook's `encoding` and `layout` (below).
* `NeighborhoodSumConfig.neighborhood_radius_zyx`: the summed box.
* `CodebookAwareDecoderConfig.allow_rescue`: turns rescue on or off.
* `DeduplicationConfig.distance_voxels`: how close duplicates lie.
* `ReadFilterConfig`: `call_statuses` (the statuses to accept) and `score_bounds`.

The reference gives every option and its default.

## Readout modes

`Dataset.readout_mode` says how a read gets its identity:

* `multiplexed`, the default: from its color sequence over all sequencing rounds,
  against the codebook. Detect in the reference round, the spot-finding default.
* `direct`: each round and channel images at most one gene, named by a panel CSV with
  the header `round,channel,gene_id` (`Dataset.load_direct_panel`). Detect in each
  round with a `SpotFindingPlan` whose `rounds` lists them, and decode with
  `DirectDecoderConfig()`. A candidate is assigned the gene of its own round and
  channel, with `entry_id` `<round>/<channel>`; `call_status` and `failure_reason` say
  when it is not, for example `unmatched` for a round and channel without a gene. A
  brighter channel never changes the identity: `own_channel_rank` and
  `own_channel_fraction` record it.

## Codebook and encoding

A codebook row is an entry: a color sequence, one color per sequencing round, and its
gene. Entries may share a gene; reads report `entry_id` and `gene_id`.
`Dataset.load_codebook` reads `gene,barcode` rows, with or without that header, or the
columns `entry_id,gene_id,color_sequence[,base_sequence]`, where `entry_id` is
optional.

The encoding turns bases into colors:

* `two_base`, the default (`EncodingConfig`): each adjacent pair of bases gives one
  color, as in STARmap. `dataset.encoding_table()` shows its configurable
  `pair_to_color` table.
* `one_base` (`OneBaseEncodingConfig`): each base gives one color through
  `base_to_color`, which you always give.

## Segment layout

Some assays read a barcode as several segments, each in its own rounds. A
`BarcodeLayout` lists the `Segment`s (name, length in bases and allowed end bases) and
the order in which their colors are acquired; pass it to `load_codebook` as `layout`.
Without it, or the legacy `EncodingConfig.split_index`, the barcode is one segment. A
read matches an entry on its whole color sequence; the end-base check of each segment
with allowed end bases is a diagnostic, and
`ReadFilterConfig(exclude_invalid_endpoints=True)` rejects the reads that fail it. In
the workflow, `load_codebook.split_index` gives a two-segment layout
([workflow segment layout](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/workflow-configuration.md#codebook-segment-layout)).

## Extraction

`NeighborhoodSumConfig` sums a box (`neighborhood_radius_zyx`) around each candidate;
`box_voxels` counts the voxels summed. By default it also measures a local
`background`, the median of a ring of voxels around the box, and `noise`, the ring's
robust spread, per channel and round. `LocalBackgroundConfig` sets the ring;
`background=None` turns it off. `valid` is false where a measurement is unavailable,
such as the rounds a direct-readout candidate was not detected in.

## Decoding

* **`wta`** takes the brightest channel of each round as its color and looks the color
  sequence up in the codebook. Its calls are exact matches (`call_type` `exact`).
* **`codebook_aware`** scores the entries near the observed sequence with the channel
  probabilities; with `allow_rescue`, it can rescue a read to a nearby entry when
  its rescue gates pass (`call_type` `rescued_h<k>` or `rescued_unknown`).

Each read gets a `call_status` (`assigned`, `unmatched`, `ambiguous` or `no_signal`)
and a `failure_reason`.

## Shared read-QC score

`ReadScoreConfig` adds `qc_score` and its components. For an assigned read, `qc_score`
is a probability score of the assigned channels on background-subtracted sums, over
every sequencing round (`multiplexed`) or the candidate's own round (`direct`). Lower
ranks as more reliable. It is one score for every decoder and call type. A read without
a score has NaN, and `qc_reason` says why (`no_assignment` for a read that is not
assigned). The score is a ranking, not a calibrated probability, and no cutoff is
recommended; `ReadFilterConfig(score_bounds={"qc_score": (None, upper)})` applies one
you choose.

## Optional deduplication

An amplicon can also be detected in the next channel.
`DeduplicationConfig` groups candidates of one detection round in different channels
that lie within `distance_voxels` and read alike; the
[specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/readout.md#optional-deduplication)
gives the rule. A read it merges keeps its row, with `is_representative` false and
`duplicate_of` naming the read that represents its group; with `exclude_duplicates`,
on by default, the filter rejects it.

## Runtime order and defaults

`FOV.run` runs extract, decode or assign, score, deduplicate, then filter. Each stage
has its `PipelineConfig` field; `None` turns it off.

| Stage | Python (`PipelineConfig`) | Workflow |
| --- | --- | --- |
| Extraction | `extraction=None`; `NeighborhoodSumConfig()` measures the background | on with `reads_extraction.run` |
| Decoding | `decoding=None` | on with `reads_filtration.run`: `wta` or `direct`, or the `decoding` block's method |
| Rescue | `CodebookAwareDecoderConfig()` has rescue on | off, unless the `decoding` block sets `allow_rescue` |
| Score | `scoring=None` | on whenever the rule decodes, unless `scoring: {run: false}` |
| Deduplication | `deduplication=None` | off, unless `deduplication: {run: true}` |
| Filter | `filtering=None`; `ReadFilterConfig()` accepts `assigned` reads, with no score bound, and rejects duplicates | on with `reads_filtration.run` |

## Checkpoints and reruns

With the `candidates` stage, `FOV.run` saves the candidates and, when extraction ran,
their sums and `valid`, with the background when measured. With `pre_qc`, it saves the
read table before the filter. After loading a `candidates` checkpoint with sums,
`fov.run` can decode, deduplicate and filter again without images, and score when the
background was measured.
{doc}`checkpoints` gives the files and calls ("Reload and continue").

## Workflow keys

On the Python backend, readout uses the top-level `readout_mode` and these rule
parameters:

```yaml
backend: python
rules:
  rsf_single_fov:
    parameters:
      spot_finding: {run: true, method: local_maxima}
      reads_extraction: {run: true, voxel_size: [1, 2, 2]}
      reads_filtration: {run: true}
      decoding: {method: codebook_aware, allow_rescue: true}
      deduplication: {run: true, distance_voxels: 1.0}
```

The reference lists the keys, and the [workflow specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/workflow-configuration.md) their rules.

## Limits

* Color codes only: no binary, MERFISH-style codes.
* No calibrated probability, score cutoff or recommended filter.
* No deduplication, merging or reassignment in readout mode `direct`.
* No spot fitting and no background-subtracted sums.
* The background ring is not masked for other spots, so a close neighbour can raise it.

## How it is checked

A golden test pins the extraction, decoding and filtering tables.
Worked examples and known-answer fixtures check each encoding, the layouts, direct
readout, the background, the score, deduplication, checkpoints and reruns; extended
checks run on calibrated synthetic scenes. The validation design is in the
[algorithms specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/readout-algorithms.md#engineering-validation-design-task-group-6).

The full specification is in
[`design/specs/readout.md`](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/readout.md).

```{toctree}
:hidden:

Readout reference <readout-algorithms>
```
