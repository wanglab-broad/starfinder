# Checkpoints and run records

`FOV.run` can save three per-FOV checkpoints and a small run record. They let you
rerun decoding or read filtering without images, or rerun detection and
extraction without loading and registering again. Checkpoints use the existing
TIFF, CSV and typed-result formats. They are development aids, not a final
output format: the workflow's molecule CSVs and images stay as they are.

## Turn checkpoints on

```python
from starfinder.dataset import CheckpointConfig

fov = dataset.fov("Position001")
fov.run(pipeline, execution=execution, checkpoints=CheckpointConfig())
```

`checkpoints=None`, the default, writes nothing and leaves `run` unchanged.
{py:class}`~starfinder.dataset.CheckpointConfig` fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `stages` | `("registered", "candidates", "pre_qc")` | Stages to write. |
| `directory` | `None` | The parent of the FOV folders; `None` means `<output_root>/fovs`. |
| `table_format` | `"csv"` | `"csv"` or `"parquet"` (requires the `checkpoint` extra, `pyarrow>=15`). |
| `hash_inputs` | `True` | Record a streamed SHA-256 of every loaded TIFF. |
| `overwrite` | `False` | Replace the run's own files: the FOV folder's `run.json` and `checkpoints/`. |

`run` checks everything before it loads or processes any image. It raises
`FileExistsError` if the FOV folder already holds a `run.json` or a `checkpoints/`
folder and `overwrite` is false, and `ImportError` if Parquet is requested without
pyarrow. The other contents of the FOV folder (the folders and files below that other
calls and workflow rules write) never stop a run. With `overwrite=True`, `run` first
removes the existing checkpoint files of that FOV's `checkpoints/` folder (stage headers
and tables, registered TIFFs under either name, registered snapshot TIFFs and dense
fields; other files are left alone), so a stage the new run does not reach can never be
loaded from an earlier run, and then replaces `run.json`.

`run` writes a stage only when it computes it. `registered` is written for each
round inside the round loop, `candidates` after detection or extraction, and
`pre_qc` after decoding, scoring and deduplication, before filtering.

## Layout

Each FOV has one folder for its durable results, the FOV folder. Its `checkpoints/`
folder holds the restart material of `FOV.run`, the three stages.

```text
<output_root>/fovs/<fov_id>/                 # the FOV folder; or <directory>/<fov_id>/
    run.json                                  # the run record of FOV.run
    molecules.csv | reads.csv                 # FOV.export_spots
    nuclei_registration.json                  # FOV.save_registration_record
    reference.tif                             # FOV.save_reference_image (reference_image, the default)
    checkpoints/                              # the restart material of FOV.run (CheckpointConfig.stages)
        registered/
            <round>.ome.tif                   # ZYXC OME-TIFF, dtype preserved, ImageMetadata
            <snapshot>/<round>.ome.tif        # the recipe's extraction source, when set
            <round>_field.npz                 # dense transforms only
            transforms.json
        candidates.csv | candidates.parquet
        candidates.json
        pre_qc.csv | pre_qc.parquet
        pre_qc.json
    other_rounds/<name>/                      # FOV.prepare_morphology / FOV.register(checkpoints=…)
        image.ome.tif                         # the prepared image, ZYXC, every channel, ImageMetadata
        registration.json
        field.npz                             # only for a chain with a dense or B-spline transform
    segmentation/<run>/                       # FOV.segment(checkpoints=…), one folder per run
        labels.tif
        input.ome.tif                         # computed runs only
        segmentation.json
    assignment/<name>/                        # FOV.assign_molecules(checkpoints=…), one folder per assignment
        molecules.csv | molecules.parquet
        cells.csv | cells.parquet
        counts.csv | counts.parquet
        nuclei.csv | nuclei.parquet           # with nuclei
        territories.tif                       # with an expansion
        cell_labels.tif                       # only when the cell run is not saved under its run
        nucleus_labels.tif                    # only when the nucleus run is not saved under its run
        assignment.json
```

An FOV loaded from a subtile writes its `run.json` and `checkpoints/` to `subtile_<n>/`
inside its FOV folder, so subtiles of one FOV never share files. Round labels must be
plain file names. `molecules.csv`, `reads.csv`, `nuclei_registration.json` and
`reference.tif` are written by the calls named beside them into
`<output_root>/fovs/<fov_id>/`; the reference image goes wherever the dataset's
`reference_image` names (`fovs/{fov_id}/reference.tif` by default), and a dataset that
declares `reference_image_import` reads an imported image instead and writes none
("The reference image" below). `CheckpointConfig.directory` moves the FOV folders of
`run.json`, `checkpoints/`, `other_rounds/`, `segmentation/` and `assignment/`;
the readers of the sample stages (stitching, export, the report) read the default
`<output_root>/fovs/`.

The `other_rounds/`, `segmentation/` and `assignment/` folders are written by
`FOV.prepare_morphology` (or `FOV.register`), `FOV.segment` and `FOV.assign_molecules`, not
by `FOV.run`: they are not checkpoint stages, `CheckpointConfig.stages` does not name
them, and they do not change the stage format version or `run.json` (see "Prepared
morphology images", "Segmentation runs" and "Assignments" below). `FOV.run` with
checkpoints refuses only an existing `run.json` or `checkpoints/` folder unless
`overwrite=True`, so it may run after `FOV.prepare_morphology` in another process; with
`overwrite=True` it removes only the stage files of `checkpoints/` and leaves
`other_rounds/` as it is, so the two calls give the same files in either order.

### registered

`registered/<round>.ome.tif` is each round exactly as it enters spot finding and
extraction: after preprocessing, registration and any post-registration
reconstruction. The checkpoint holds the reference round and the sequencing rounds
only, the rounds `FOV.run` processes (also when written by `save_checkpoint`); an other
round is saved under `other_rounds/` ("Prepared morphology images" below), and its
transforms and attempts are not written here. It is written with {py:func}`~starfinder.io.save_volume` as
OME-TIFF and read with {py:func}`~starfinder.io.load_volume_zyxc`, which keeps
singleton Z and C, the dtype (float64 included) and the stored
{py:class}`~starfinder.image.ImageMetadata`. Checkpoints written before the
OME-TIFF change named this file `<round>.tif` and stored it in tifffile's own
ZYXC layout; loading still accepts that name and layout when no `.ome.tif` file
is present.

When the preprocessing recipe names an `extraction_source`, that snapshot of
each round is stored as `registered/<snapshot>/<round>.ome.tif`, in the same
format and with the round's `ImageMetadata`. It has been through the same
single registration resampling as the round image, so the two stay aligned.
`load_checkpoint("registered")` restores it in `FOV.snapshots`, and a later
`run` without a recipe extracts from it, as the restored preprocessing record
names it. A recipe without an extraction source stores one image per round.

To look at a registered round in Fiji, use **File › Import › Bio-Formats** and
choose **Hyperstack** under *View stack with*. The file stores one YX plane per
page, and its OME-XML declares SizeZ, SizeC, the pixel type and the dimension
order `XYCZT`, so Bio-Formats opens it as a Z×C hyperstack in its stored dtype.

`transforms.json` records the FOV identity, the rounds and the channel order,
the rounds written, the channel information of each written round (`channels`:
round name to its `ChannelInfo` entries `channel`, `name` and `wavelength`, as in
`run.json`; added on 2026-10-05), the stored snapshot names (`snapshots`), the preprocessing
record (as in `run.json`, below), the registration attempts, the registration
recipe summary (`registration_recipe`: step method names, signals, warp,
reference round and QC config), `registration_semantics` (`recipe`) and, per
moving round, its step results in step order and one `applications` entry with
the `WarpConfig` of the round's one final resampling. Each result (`transforms`)
has its `step` index, its transform and its diagnostics:

% W-337

| Kind | In `transform` | In `<round>_field.npz` |
| --- | --- | --- |
| `translation` | `displacement_zyx` | none |
| `affine` | `matrix_zyx` (4×4 index-space matrix) and `physical` (the backend's physical parameters, or null) | none |
| `bspline` | `bspline` (grid size, origin, spacing and direction in ITK XYZ order, `order`, `spacing_zyx`) and `coefficients`, the file name | `result_<i>`: the coefficients, float64 |
| `dense` | `field`, the file name | `result_<i>`: the displacement field, float32 or float64 |

`<i>` is the result's index in the round's list. JSON floats round-trip exactly,
so the reloaded transforms equal the saved ones. `dataset.load_checkpoint` and
`FOV.load_checkpoint("registered")` rebuild a
{py:class}`~starfinder.registration.TransformChain` per round
(`registration_chains`) and the round's `WarpConfig`
(`registration_record["application"]`, also each result's
`application_config`); applying the chain with it to the pre-registration
images gives the registered images bit for bit.

`load_checkpoint("registered")` reads each round with the patterns of its
`channels` entry and falls back to `channel_labels` for a header without the key
(written before 2026-10-05); the key is additive, so the stage's `format_version` stays 2.

% W-337

The header `format_version` is 3 for the three stages (since 2026-10-24). A header of
version 3 holds `files`: every payload file it relies on, `path` relative to the
header's folder and `sha256` of its bytes. For `candidates.json` and `pre_qc.json`
that is the table (`candidates.csv`, `pre_qc.parquet`, ...); for
`registered/transforms.json` the round images, their snapshots
(`<snapshot>/<round>.ome.tif`) and the field files (`<round>_field.npz`), each
hashed as it is written (in streaming mode, round by round). The readers
({py:func}`~starfinder.dataset.load_checkpoint`,
{py:meth}`~starfinder.dataset.FOV.load_checkpoint`) check every listed file before
they read anything: a missing file or another SHA-256 raises `ValueError` naming
the file. Headers of versions 1 and 2 hold no hashes and load without the check
(`READ_VERSIONS` is 1 to 3); version 2 has the layout of version 3 without
`files`. A version-1 registered checkpoint (before the registration recipe) holds
translation and dense results, each with its own `application_config`, which
were applied one after another; it loads with those configs, without chains,
and with `registration_record["semantics"] == "sequential"`. Nothing converts
it to a recipe. Its translation entries store the correction (the negated
displacement), which the reader loads as `displacement_zyx`; see the
[migration guide](migration.md).

% W-565

### candidates

This is a wide table with one row per spot, in the spot table's order:

| Columns | Type | Content |
| --- | --- | --- |
| `spot_namespace`, `spot_id` | string | Identity; joins use both, never row position. |
| `z`, `y`, `x` | float64 | Zero-based voxel coordinates. |
| `channel`, `peak_intensity`, ... | int64 or float64 | Detector columns, when present: `radius` (Starfish LoG) and `probability` (Spotiflow) are float64. |
| `round` | string | The detection round of each row, present only when the detection plan names `rounds`. |
| `sig_<round>_<channel>` | float64 | Extracted sum for each round, then each channel in channel order. |
| `valid_<round>` | bool | False marks an unavailable measurement, not zero signal. |
| `bg_<round>_<channel>` | float64 | Since 2026-10-02, when the background was measured: the local background, grey levels per voxel (ring median); NaN below `min_voxels` or in a round that was not read. |
| `noise_<round>_<channel>` | float64 | The local noise, 1.4826 × the ring's median absolute deviation; NaN as `bg_`. |
| `bgvox_<round>` | int64 | Ring voxels inside the image (0 in a round that was not read). |
| `boxvox_<round>` | int64 | Voxels the extraction box summed (`IntensityExtractionResult.box_voxels`). |

% §2.8, W-294

The rounds follow the extraction round labels, which are the `RoundState`
sequencing rounds in a pipeline run. The table is built and split with array
reshapes; there is no per-row work. Labels that would produce duplicate column
names are rejected. When detection ran without extraction, the table has no
signal columns and reloads with `intensity_result=None`.

`candidates.json` holds the detector and extraction configurations, the image
metadata, the JSON-representable diagnostics, the column dtype map and (since
header version 3) `files`, the table with its SHA-256. Since the spot-finding stage's detection plan was added it
also records the detection plan in four keys:

% §2.7

| Key | Content |
| --- | --- |
| `detection_rounds` | `null` (the reference round only) or the plan's list of rounds. |
| `detection_plan` | The channel overrides, as `{channel, config}` entries (empty without overrides). |
| `execution` | The execution entry: device, framework and thread settings. |
| `weights` | The provenance `artifacts` entries of the loaded pretrained weights (empty for methods without weights). |

Since the readout modes of 2026-10-02 `candidates.json` also records `readout_mode`, the dataset's
readout mode (`multiplexed` or `direct`, {doc}`readout-contract`). A header without
it is `multiplexed`.

% §2.8, W-293

Since 2026-10-02 extraction measures the local background and noise by default
({doc}`readout-contract`, "Extraction"; option C1 of the [readout specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/readout.md#checkpoints-and-reruns), "Checkpoints and reruns"). The
saved extraction configuration `signals.extraction_config` keeps exactly the fields
`neighborhood_radius_zyx`, `sampling` and `boundary`, so a reader at `141c093`
still rebuilds it; the background settings are new top-level keys:

% W-294

| Key | Content |
| --- | --- |
| `background_config` | The `LocalBackgroundConfig` fields (`inner_radius_zyx`, `outer_radius_zyx`, `min_voxels`), or `null` when the background was off. |
| `image_background`, `image_noise` | Per round and channel (`{round: {channel: value}}`), the median and 1.4826 × MAD of the whole extraction image; `null` when not measured. |

The table then adds the four background column groups after the `valid_<round>`
columns. A reader at `141c093` loads such a checkpoint and drops those columns,
because they are not spot columns. A checkpoint written without the keys (before
the local background measurement of 2026-10-02, or with `background=None`) loads with `background=None` in the reloaded
`NeighborhoodSumConfig`, no background measurements and `box_voxels=None`.

% W-294

`detection_config` stays the plan's base config. Reloading gives a
`SpotFindingResult` whose table equals the written one exactly (CSV and Parquet) and
whose `config` and `plan` equal the original. A checkpoint written before these keys
existed loads as before, with no overrides and `rounds` `None`. `format_version`
stays 2: a reader without the keys still loads every row, with `round` as an ordinary
spot column.

With a plan that names several rounds (see {doc}`spot-finding-contract`, "Detection in
several rounds"), the table holds every round's candidates, reference round first,
with `spot_id` running over the whole table and the round in `round`; coincident
candidates of different rounds stay separate rows. In readout mode `multiplexed` the
signal columns are read at every candidate in every sequencing round. In readout mode
`direct` each candidate is read in its own round only: the signals of its other
rounds are `0.0` and their `valid_<round>` is false, and their background and
noise are NaN with 0 ring voxels. `IntensityExtractionResult.box_voxels` is stored in
the `boxvox_<round>` columns, so it is reloaded only from a checkpoint with background
measurements; otherwise a reloaded result has `box_voxels=None`.

### pre_qc

`pre_qc.<format>` is `ReadDecodingResult.table`, unchanged. Since 2026-10-02
the table holds the string column `entry_id`, the codebook entry of the decoded
color sequence, after `gene_id`; `FORMAT_VERSION` stays 2, an older reader keeps
it as an ordinary column, and a `pre_qc` written without it still loads.
`pre_qc.json` holds the decoder configuration, labels, diagnostics and (since
header version 3) `files`, the table with its SHA-256; the reader
rebuilds the configuration through `DECODING_METHODS` from its `method` (a `direct`
configuration holds only its `method`). Since the readout modes of 2026-10-02 it also records `readout_mode`,
which becomes `ReadDecodingResult.readout_mode` on reloading; a header without it
loads as `multiplexed`. A `direct` table adds `round`, `channel`,
`own_channel_rank` and `own_channel_fraction`, and its color sequences are missing
values. Array
and table diagnostics are not saved: decoder probabilities, per-round and
candidate tables, and WTA per-round scores. A reloaded result therefore lacks
those keys.

% §2.8, W-292, W-293

Since the scoring of 2026-10-02 `pre_qc` holds the read table after scoring, before filtering (option
C1): when the run scored the reads, the table is `ReadScoringResult.table`, the
decoding table followed by the score columns `qc_score`, `qc_ambiguity_max`,
`qc_signal_to_background`, `qc_rounds` (float64) and `qc_reason` (string). The
decoder configuration stays in `decoding_config`, and `pre_qc.json` adds:

% W-294

| Key | Content |
| --- | --- |
| `scoring_config` | The `ReadScoreConfig` (`{"method": "bgcorr_probability"}`), or `null` when the reads were not scored. |
| `deduplication_config` | The `DeduplicationConfig` (`distance_voxels`, `compatibility`) since 2026-10-02, or `null` when deduplication did not run. |
| `layout` | The codebook's segment layout (`segments` with `name`, `bases` and `ends`, and `acquisition_order`); `null` in readout mode `direct`. A header without it is one segment. |
| `encoding` | Since 2026-10-04, the codebook's encoding that decoded the reads: `method`, `reverse_bases` and the table, `pair_to_color` (`two_base`, the 16 ordered pairs) or `base_to_color` (`one_base`); `split_index` is left out because `layout` records it. `null` in readout mode `direct`. A header without it (written before 2026-10-04) is not checked. |
| `stages_applied` | The stages that made the table, in order: `"decoding"`, then `"scoring"` and `"deduplication"` when they ran. |

% W-295, W-304

On reloading, a scored checkpoint gives `decoding_result` (the table without the
score columns, which come last) and `scoring_result` (the whole table); an unscored
one, or one written before the scoring of 2026-10-02, gives `scoring_result=None`. A reader at `141c093`
loads a scored `multiplexed` `pre_qc` with the score columns as ordinary columns,
and it ignores the `encoding` key, as the reader at `9aeb220` does (checked by
loading checkpoints with the encoding key of 2026-10-04 with both readers).

% W-294, W-304

Since 2026-10-02 a run with `PipelineConfig.deduplication` writes the reads after
deduplication: the table ends with `duplicate_group`, `duplicate_of` (string,
missing when not set), `is_representative` (bool) and `duplicate_reason` (string),
after the score columns when the reads were scored. On reloading, a deduplicated
checkpoint also gives `deduplication_result` (the whole table, its counts
recomputed from these columns, without the pair diagnostics), and `scoring_result`
and `decoding_result` are the table without the columns of the later stages.
Without `deduplication_config` it is `None`.

% W-295

All JSON files are strict JSON: a non-finite diagnostic or configuration value
(NaN or infinity) is written as `null`.

## Prepared morphology images

`fov.prepare_morphology(config, checkpoints=CheckpointConfig(…))` writes each prepared
image (the reference stain `reference_stain` and each other round) to
`other_rounds/<name>/`; `fov.register(recipe, rounds=…, checkpoints=…)` writes each
round it registers the same way ({doc}`coordination`, "Sequencing rounds and other
rounds"). Only `directory` and `overwrite` are used. An existing folder raises
`FileExistsError` before anything is loaded or estimated unless `overwrite=True`, which
replaces its files.

| File | Content |
| --- | --- |
| `image.ome.tif` | The prepared image in the reference frame: ZYXC OME-TIFF by `save_volume`, every channel in the round's C order, with its `ImageMetadata` (the reference round's). |
| `registration.json` | The record, `format_version` 1 and `stage` `prepared_image`: the name; `dataset_id`, `sample_id`, `fov_id` and `subtile_id`; `channels` (the `ChannelInfo` entries); `rotation` (the rotation diagnostics, or null); `registration` (the `registration_record["rounds"]` entry: for a round the recipe summary, the reference label and `reference_sha256`, with `transforms` (the step results, in the form of `transforms.json`), `application_config` and `attempts`; for `reference_stain` the `relation` `same acquisition as the reference round` and no transform or attempt); `image` (path, shape, dtype, metadata, `sha256` of the image's C-order bytes and `file_sha256` of the file); `software` (as in `run.json`). |
| `field.npz` | Only for a chain with a dense or B-spline transform: `result_<i>` as in `<round>_field.npz`, with its `file_sha256` in the record. |

`fov.load_registered_round(name, checkpoints=CheckpointConfig(directory=…))` reads the
folder on any `FOV` object of the same FOV. It checks the record's identity against the
FOV and its channels against the dataset, each file's SHA-256 and the image's SHA-256
(`ValueError` naming the path and both hashes), the stored metadata against the record
and, when the reference round is resident, that the metadata equals the reference
grid's. It restores the image, its metadata, the rotation diagnostics and
`registration_record["rounds"][name]`, with `saved` (the image file's path relative to
the FOV folder and its SHA-256); a registered round also gets its results, chain,
attempts and `WarpConfig` back. `FOV.segment` links that saved file in the record of a
run that reads the image. The Python `nuclei_registration` rule writes these folders when
`rules.nuclei_registration.parameters.checkpoints` is set ({doc}`workflows`).

**The registration record.** Without saving images, `fov.save_registration_record(path=None)`
writes `<output_root>/fovs/<fov_id>/nuclei_registration.json`, which the Python
`nuclei_registration` rule declares: `fov_id`; `rounds`, the round roles; `registered_rounds`,
the `registration_record["rounds"]` entries of the other rounds that have registration
attempts (the reference stain, which has none, is left out); and `registration`, those
rounds' attempts, unchanged. Each estimation attempt holds its translation as
`qc.details.transform.displacement_zyx`. The file is not a checkpoint and nothing reads it
back; `load_registered_round` reads the folders above (since 2026-10-23).

% W-563

**Segmenting in another process.** These folders and the reference image ("The reference
image" below; written by `FOV.save_reference_image`, not a checkpoint)
are what a process that only segments reads: `fov.load_reference_image()`, then
`fov.load_registered_round(name)` for each image its plan names, then
`fov.segment(plan, checkpoints=…)`, with no sequencing round resident and no partial load
of `registered` ({doc}`segmentation-contract`, "Coordination per FOV"). `fov.assign_molecules` there
takes the molecules from the `candidates` and `pre_qc` stages, which restore no image,
and re-applies the filter with `FOV.run` ("Reload and continue").

## The reference image

`FOV.save_reference_image` writes the reference round's merged (or single-channel) image to
the file of the dataset's `reference_image` setting (the workflow key and `Dataset` field of
that name): a pattern with `{fov_id}`, absolute or relative to the output root, by default
`fovs/{fov_id}/reference.tif`. `FOV.load_reference_image`, the workflow's rule declarations
and every reader of the sample stages (`Dataset.stitch`, `Dataset.export`, the report and
the workflow adapters) resolve the same file through one function. A reference image that
another tool or rule produced is declared with `reference_image_import`, a pattern of the
same form that excludes `reference_image`: every reader reads that file,
`FOV.save_reference_image` raises instead of writing it, and the Python processing rules
write no reference image. The TIFF content, calibration and projection metadata are as
before. A segmentation run that reads the saved image records it by its path relative to
the output root, or by its absolute path when it lies outside, with its SHA-256 (since
2026-10-24; {doc}`migration`).

% W-564

## Segmentation runs

`fov.segment(plan, checkpoints=CheckpointConfig(…))` writes each run of the plan to
`segmentation/<run>/` (option F1 of {doc}`segmentation-contract`, "Saved format"). Only
`directory` and `overwrite` are used. A run folder that already holds these files raises
`FileExistsError` before any run when `overwrite` is false; with `overwrite=True` the
run's files are replaced. Other runs and the `FOV.run` files are never touched.

| File | Content |
| --- | --- |
| `labels.tif` | The label image: ZYX `uint32` (a plane is 1×Y×X), zlib, written by `save_volume` with the grid's `ImageMetadata` in its description. |
| `input.ome.tif` | The segmentation input the labels were computed from: ZYXC OME-TIFF by `save_volume`, with its grid's metadata (the projected grid for a projected run). An imported mask has no segmentation input and no such file. |
| `segmentation.json` | The run record of {doc}`segmentation-contract` ("Run record"), with `labels.path` and `labels.file_sha256`, and for a computed run `input.path` and `input.file_sha256`. Each `input.channels` entry has the channel's `name` and `wavelength` beside `channel` (since 2026-10-05; additive, so a record without them still loads). `format_version` is 1. |

% W-336

`fov.load_segmentation(name, checkpoints=CheckpointConfig(directory=…))` reads the folder
back into a `SegmentationResult`, stores it in `fov.segmentation_results[name]` and
returns it. It raises `ValueError` naming the path and both hashes when a file is missing
or its SHA-256, or the label array's SHA-256, differs from the record, and when the
record names another run or FOV. The record of the returned result equals the record of
the result `FOV.segment` stored.

## Assignments

`fov.assign_molecules(…, name="default", checkpoints=CheckpointConfig(…))` writes
`assignment/<name>/` (option L1 of {doc}`assignment-contract`, "Persistence"). `directory`,
`table_format` and `overwrite` are used, with the same `FileExistsError` and `ImportError`
rules as `run`, checked before the assignment runs.

* `molecules`, `cells` (every cell, kept and excluded), `counts` and, with nuclei, `nuclei`
  are written in `table_format` with the table writer below; `assignment.json` records
  each file's path, SHA-256 and dtype map under `files`.
* Label images follow the one rule of {doc}`assignment-contract` ("Label images of a
  checkpointed assignment"): a cell or nucleus run saved under its run in the same
  checkpoint root (written by `FOV.segment(checkpoints=…)` or read by
  `FOV.load_segmentation` there, with an unchanged `labels.tif`) is linked by its relative
  path, `../../segmentation/<run>/labels.tif`, and never copied; any other input, among them
  a run saved under another root, an `import_labels` result or a run kept in memory, is
  written as `cell_labels.tif` or `nucleus_labels.tif`. With an expansion the expanded
  territories are `territories.tif`. Label images are ZYX `uint32` with the grid's
  `ImageMetadata`.
* `assignment.json` is the run record of {doc}`assignment-contract` with every `file`
  entry filled in: `inputs.cells.file` and `inputs.nuclei.file` (path and file SHA-256 of
  the label file that holds each run, with `saved_under_run`), `expansion.original` and
  `expansion.expanded` (their paths and `file_sha256`, beside the label arrays'
  `sha256`), and `files`. `inputs.molecules.genes` holds the gene order of the counts.

`fov.load_assignment(name, checkpoints=CheckpointConfig(directory=…))` reads the folder
and the linked label files, checks every recorded SHA-256, rebuilds the tables with their
recorded dtypes, stores the `AssignmentResult` in `fov.assignment_results[name]` and
returns it. A written or linked file that is missing or changed raises `ValueError` naming
the path and both hashes; a linked `segmentation/<run>/labels.tif` is therefore a
prerequisite of the reload.

## Table formats

CSV is written with floats as `%.17g` and missing values as `<NA>`. It is read
back with an explicit dtype map from the stage header: pandas `string`, nullable
`Int64`, `UInt32` (the assignment tables' cell and nucleus identifiers) or `boolean`, and
float64. It is then cast to the recorded in-memory
dtypes. String columns are read verbatim, without pandas' default missing-value
parsing. A literal string equal to `<NA>`, or one that starts with a backslash,
is written with one extra leading backslash, which is removed on reading.
Printable text round-trips exactly in CSV and Parquet, including `""`, `"NA"`,
`"NaN"`, `"null"`, `"N/A"` and `"<NA>"`, and stays distinct from a missing
value. CSV raises a clear `ValueError`, naming the column, for any string that
contains a control character (U+0000–U+001F or U+007F, which includes tab and
newline); nothing is written with a changed value. Use Parquet for such data.
Infinite scores survive, and floats round-trip exactly. Parquet stores the same
columns and is cast through the same dtype map, so CSV and Parquet reload to
identical typed results. Empty tables keep their columns and dtypes.

## Reload and continue

{py:meth}`~starfinder.dataset.FOV.load_checkpoint` restores one stage into an FOV
that has no results at or after that stage. It checks the FOV id (and subtile
id), the round labels and the channel order against the dataset, and, for
`candidates` and `pre_qc`, the readout mode (`multiplexed` when the header has
none) against `Dataset.readout_mode`, and, for `pre_qc`, the recorded `encoding`
against the loaded codebook's (since 2026-10-04; not checked when the header has none
or no codebook is loaded); it raises `ValueError` on a mismatch, naming both values. Then call `run` with a `PipelineConfig` that starts
after the loaded stage:

% W-304

```python
from starfinder.readout import DeduplicationConfig, ReadScoreConfig
from starfinder.dataset import PipelineConfig

# Decode and filter again, without images.
fov = dataset.fov("Position001").load_checkpoint("candidates")
fov.run(PipelineConfig(decoding=decoder, filtering=read_filter))

# Decode, score and filter from the retained values and background.
fov = dataset.fov("Position001").load_checkpoint("candidates")
fov.run(PipelineConfig(decoding=decoder, scoring=ReadScoreConfig(), filtering=read_filter))

# Score again without decoding: load candidates (values, background) and pre_qc (reads).
fov = dataset.fov("Position001").load_checkpoint("candidates").load_checkpoint("pre_qc")
fov.run(PipelineConfig(scoring=ReadScoreConfig(), filtering=read_filter))

# Deduplicate again: load candidates (coordinates, channels, sums) and pre_qc (reads).
fov = dataset.fov("Position001").load_checkpoint("candidates").load_checkpoint("pre_qc")
fov.run(PipelineConfig(deduplication=DeduplicationConfig(), filtering=read_filter))

# Only filter again, with a different predicate.
fov = dataset.fov("Position001").load_checkpoint("pre_qc")
fov.run(PipelineConfig(filtering=other_filter))

# Detect and extract again from registered images.
fov = dataset.fov("Position001").load_checkpoint("registered")
fov.run(PipelineConfig(spot_finding=detector, extraction=extraction,
                       decoding=decoder, filtering=read_filter))
```

Scoring needs the background of the loaded `candidates`: scoring a checkpoint
without it (written before the local background measurement of 2026-10-02, or with `background=None`) raises `ValueError`
naming extraction, the stage to rerun.

% W-294

When no image operation is configured and no images are resident, `run` skips
the round loop. Resuming is always an explicit call; there is no scheduler.
Nothing verifies a resume against `run.json` today: verified resume, scheduling
and job completion belong to workflow orchestration. `run.json` version 2 provides what verified
resume needs (each stage record with its SHA-256, each payload file hashed by
its record, every input and source hashed, and paths that resolve in a copied
tree); version 1, which named the stage files only and recorded paths as given,
does not.
{py:meth}`~starfinder.dataset.FOV.save_checkpoint` writes one stage from the
current results outside `run` into the FOV folder's `checkpoints/` and returns that folder.
It uses `CheckpointConfig.directory`, `table_format` and `overwrite`.
{py:func}`~starfinder.dataset.load_checkpoint` reads a stage directly from a checkpoint folder
(`<FOV folder>/checkpoints`) and returns a dict keyed by the FOV attributes it restores.

% §2.13

## run.json

`run.json` is replaced atomically (a temporary file, then `os.replace`) when the
run starts, after each completed step and when the run ends, and again when a
writer records an output in `outputs`. Its `format_version` is 2 (since 2026-10-24;
{doc}`migration`, "run.json version 2"). The record is portable: no path in it
depends on where the output tree or the input tree lies.

% W-565

* **Paths.** Output and stage paths are relative to the FOV folder, which holds
  `run.json`. A file the run reads inside the input root (the image inputs, a
  codebook kept there) is recorded relative to the input root. A file outside it
  (a codebook or panel elsewhere, the workflow configuration, supplied
  statistics, pretrained weights) is recorded by its absolute path. Every file
  the run reads has its SHA-256 (the image inputs unless `hash_inputs=False`).
  `input_root` and `output_root` record the roots of the run as provenance.
* **Two meanings, kept apart.** `status` is the outcome of `FOV.run`'s processing:
  `succeeded` is written when that processing succeeded. `outputs` records the
  files written after it (the molecule and read tables, the reference image, the
  subtile files), each added as its writer writes it. A `succeeded` record
  without an `outputs` entry says that the processing succeeded and that file
  was not written.

| Field | Content |
| --- | --- |
| `format_version` | `2`. |
| `dataset_id`, `sample_id`, `fov_id`, `subtile_id` | Identity. |
| `input_root`, `output_root` | The `Dataset`'s input root and output root, absolute (`null` input root for a dataset without one): provenance. The other paths do not depend on them; a reader of a version-2 record resolves the input paths against the input root it is given ({doc}`report-contract`, "The replay"), and against `input_root` only when it is given none. |
| `status`, `started_at`, `ended_at` | `running`, `succeeded`, `failed` or `interrupted`, with UTC times: the outcome of `FOV.run`'s processing. |
| `error` | `null`, or the failing `step` and `round`, the exception `type`, `message` and `traceback`. |
| `code` | Package `version`, `git_commit` and `git_dirty`; each is `null` when unknown. The commit is recorded only when the package runs from a starfinder checkout, never from an enclosing repository. |
| `environment` | Python, platform and package versions (`null` when not installed). |
| `config` | `pipeline`, `execution` (including the execution `device`) and `checkpoints` configurations, `readout_mode`, the dataset's readout mode (since 2026-10-02), and `encoding`, the codebook's encoding as in `pre_qc.json` (`null` without a codebook or in readout mode `direct`; since 2026-10-04). `checkpoints.directory`, the parent of the FOV folders when the configuration names one, is relative to the FOV folder (`..`; `../..` for a subtile), else `null`. The other configurations are recorded as given. |
| `workflow_config` | `{path, sha256}` of the workflow configuration file a workflow rule ran with (the top-level `config_path`: absolute, or relative to the input root when it lies there; `sha256` of its bytes, `null` when no file is there), kept on the `Dataset` by `from_workflow_config`; `null` for a run outside the workflow (since 2026-10-12). |
| `channels` | The channels of each configured round, and of `reference_stain` when `Dataset.reference_stains` is set: per round, in C order, one entry with `channel` (the file pattern), `name` (`null` when not given) and `wavelength` (nm, or `"unavailable"`), from `Dataset.channel_info` ({doc}`coordination`, "Channels"; since 2026-10-05). An additive key: a `run.json` without it still loads. |
| `rounds` | The round roles of the run, as `RoundState` holds them: `sequencing_rounds`, `other_rounds` and `reference_round` (since 2026-10-23). |
| `inputs` | Each loaded TIFF: `path`, relative to the input root (POSIX), and its streamed `sha256` (`null` with `hash_inputs=False`). |
| `codebook` | The codebook of readout mode `multiplexed` or the direct panel of readout mode `direct`: `kind` (`codebook` or `direct_panel`), `path` (the file `Dataset.load_codebook` or `Dataset.load_direct_panel` read: absolute, or relative to the input root when it lies there) and `sha256` (of the file's bytes), both `null` for one set on the `Dataset` without a file, `entries` and `genes` (distinct genes); `null` when the dataset has neither (since 2026-10-12). |
| `steps` | `name`, `round`, `seconds` and `status` of each completed or failed step. A preprocessing step is named `preprocess:<step name>`. The `find_spots` record also has `methods`, a list with the detection's provenance entry ({doc}`method-registry`, "Provenance in run.json"): `stage` (`spot_finding`), `method`, `config_type`, `implementation`, `config`, `requires` (installed versions of the optional dependencies), `artifacts` (pretrained weights files, each by absolute path and SHA-256; empty for methods without weights) and `execution` (device, framework and thread settings). For a plan that names rounds, `run` records one `find_round_spots` step per detected round, each with its own entry, and `FOV.find_spots` called inside a recorded step lists one entry per round, each with its `round`. |
| `preprocessing` | `null` without a preprocessing recipe. Otherwise `recipe` (the step names of `steps` and `post_registration`, `extraction_source` and `registration_source`), `rounds`: per round, one record per step with `index`, `stage` (`steps` or `post_registration`), `step`, `config`, `fitted`, `diagnostics`, `input_dtype`, `output_dtype` and `save_as`; `transforms`: per round and image (`detection` and each snapshot), the transforms composed in order, each with `result` (its index in the round's registration results in `transforms.json`), `method` and `kind` (`translation`, `affine`, `bspline` or `dense`), empty for the reference round; and `supplied_statistics`, `{path, sha256}` of the supplied-statistics file (absolute, or relative to the input root when it lies there; both `null` without one). |
| `registration` | Ordered registration attempts per round: the estimation entries and one application entry per moving round (see {doc}`coordination`). |
| `transforms` | Per moving round that `run` registered, in processing order: `reference` (the reference round), `results` (the round's results in order, each `{step, transform, diagnostics}` as in `registered/transforms.json`) and `application_config` (the round's `WarpConfig`, as `applications.<round>` holds it). A dense field's `field` or a B-spline's `coefficients` is `<round>_field.npz` (a name in `checkpoints/registered/`, hashed by that record's `files`) when the stage `registered` wrote that file, else `null`; the arrays are never in `run.json`. `{}` without registration; the reference round and a round whose registration failed have no entry (since 2026-10-12). |
| `counts` | Spots, intensities, decoding call statuses, scoring counts (`scoring`: total, scored, `no_assignment` and `background_unavailable`, since 2026-10-02, when the reads were scored), deduplication counts (`deduplication`: total, groups, `merged_reads` and `conflicting_groups`, since 2026-10-02, when the reads were deduplicated) and filtering counts (`filtering`: total, accepted and rejected; when the filter checked endpoints, with `end_bases` or a codebook layout's segment ends, also `endpoint_valid`, the reads of the correct form, and with segment ends `endpoint_valid_segments`, that count per segment; since 2026-10-12; and `fractions`, the filter result's `accepted` and, when it checked endpoints, `endpoint_valid` over all reads, `null` without reads; since 2026-10-23). Whenever reads were decoded, `summary` holds the population summary of {py:func}`~starfinder.readout.summarize_reads` of the latest reads (since 2026-10-02 for scored or deduplicated reads; since 2026-10-23 also without scoring or deduplication). |
| `checkpoint_directory` | `checkpoints`, the FOV folder's checkpoint folder, relative to the FOV folder. |
| `checkpoints` | Per stage written, the stage record the run relies on: `path` (`checkpoints/candidates.json`, `checkpoints/pre_qc.json` or `checkpoints/registered/transforms.json`, relative to the FOV folder) and `sha256` (of the record's bytes). The record names and hashes its payload files (`files`, see "registered" under "Layout"); a stage whose record was not written (a run that failed before it) has no entry. |
| `outputs` | `{}` when the processing ends; then, per slot, each file written after the processing, added by its writer when the FOV's latest `run` had checkpoints and succeeded, the record rewritten atomically (a later `run` writes a new record without them). Every entry has `path` (relative to the FOV folder, POSIX) and `sha256` (of the file's bytes). `molecules` and `reads` ({py:meth}`~starfinder.dataset.FOV.export_spots`, since 2026-10-23) also have `molecules_sha256` (the keyed hash of the six molecule columns, `MoleculeTable.sha256`; `null` for `reads` or a file without those columns), `dtypes` (the column dtypes) and `rows`; `reference_image` ({py:meth}`~starfinder.dataset.FOV.save_reference_image`); `subtile_coords` (with `rows`) and `subtile_data` (per subtile ID, the NPZ's entry) ({py:meth}`~starfinder.dataset.FOV.save_subtiles`); and, in a resident subtile's record, `subtile_spots` (the workflow's `subtile_goodSpots_<n>.csv`). The declared outputs of the Python processing rules are all recorded here (since 2026-10-24). |

% W-293, W-304, W-451, W-336, W-563, W-294, W-295, W-562, W-565

If a step raises, `run` records `failed` (or `interrupted` for
`KeyboardInterrupt` and other non-`Exception` errors), with the innermost failing
step and its round, and re-raises the original exception. If that final write
fails, the failure is logged and the original exception still propagates.
Stages written before the failure stay usable. The codebook file is not recorded
in `inputs`, which lists the loaded images; `codebook` names it.

`transforms`, `codebook`, `workflow_config`, the endpoint counts, `outputs`, `rounds` and the
filter's `fractions` were added to version 1 as additive keys, and a `run.json` written without
them still loads. Version 2 changes the recorded paths and `checkpoints`, and adds `input_root`
and `output_root` ({doc}`migration`, "run.json version 2"). The readers of `run.json` (the
report's collector and replay) accept versions 1 and 2: a version-1 record is read as recorded,
its paths opened as written, and it is not eligible for verified resume; a version-2 record's
relative input paths are resolved against an input root, never against the current folder. The workflow's processing rules write `run.json` ({doc}`workflow-configuration`,
"Run records of the processing rules"), and it replaces the Python backend's former per-FOV logs
({doc}`migration`, "Sequencing logs and the morphology record"). A direct caller gets the record
without stage files with `FOV.run(..., checkpoints=CheckpointConfig(stages=()))`.
