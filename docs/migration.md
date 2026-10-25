# Python migration guide

This is a coordinated breaking Python cleanup: update maintained imports and
callers together. Replaced interfaces have no compatibility aliases. **FOV stays
FOV.** MATLAB APIs, and the files, keys and coordinate conventions that MATLAB
code reads or writes, remain unchanged. Since 2026-10-19 the Python package
otherwise follows its own design (W-526): a Python name, key, layout or format may
change even where it once matched the MATLAB pipeline, and each such change is
listed here. Historical scripts and outputs retain their original provenance; they
are not examples of the current API.

## Mechanical names and ownership

| Before | Current owner/interface |
| --- | --- |
| `dataset.STARMapDataset`, `LayerState`, `layers` | `dataset.Dataset`, `RoundState`, `rounds` |
| `dataset.Codebook` | `readout.Codebook` |
| `io.load_multipage_tiff`, `load_image_stacks`, `save_stack` | `io.load_volume`, `load_round`, `save_volume` |
| `min_max_normalize`, `histogram_match`, `morphological_reconstruction`, `tophat_filter` | `preprocessing.normalize_min_max`, `match_histogram`, `correct_background`, `filter_tophat` |
| `utils.make_projection` | `preprocessing.project_image` |
| `spotfinding.find_spots_3d` | `spot_finding.find_spots` |
| `decode_color_seq` | `readout.colors_to_bases` |
| `benchmark.synthetic` | `synthetic` |
| `registration.metrics`, benchmark truth comparisons | Pure domain functions in `evaluation` |
| Separate benchmark drivers and `starfinder-generate` | `starfinder synthetic generate`, `starfinder benchmark run/evaluate/report` |
| `io.read_checkpoint` | `dataset.load_checkpoint` (no alias) |
| `io.export_spots` | `readout.export_spots` (no alias) |
| `sample_output.SampleSegmentation` | `segmentation.SampleSegmentation` (no alias) |
| `dataset.RegistrationRecipe`, `RegistrationStep`, `RecoveryConfig`, `ExternalReference` | `registration.RegistrationRecipe`, `RegistrationStep`, `RecoveryConfig`, `ExternalReference` (no alias) |
| `starfinder.barcode` (the module, its submodules and the root export `starfinder.barcode`) | `starfinder.readout` (no alias) |
| `starfinder.sample_output` (the module and its submodules) | `starfinder.export` (no alias) |
| `starfinder.evaluation.barcode` | `starfinder.evaluation.readout` (no alias) |
| pytest marker `barcode` (`-m barcode`) | pytest marker `readout` (`-m readout`) |
| `readout.decode_barcodes` and the root export `starfinder.decode_barcodes` | `readout.decode_reads`, root `starfinder.decode_reads` (no alias) |
| `FOV.decode_barcodes` | `FOV.decode_reads` (no alias) |
| `readout.encode_bases` | `readout.bases_to_colors` (no alias) |
| `readout.decode_color_sequence` | `readout.colors_to_bases` (no alias) |
| `readout.BarcodeDecodingResult` | `readout.ReadDecodingResult` (no alias) |
| `readout.assign_direct` | `readout.decode_direct` (no alias) |
| `readout.DirectAssignmentConfig` | `readout.DirectDecoderConfig` (no alias) |
| `spot_finding.resolve_weights` | `spot_finding.resolve_model` (no alias) |
| `spot_finding.fetch_weights` | `spot_finding.fetch_model` (no alias) |
| `spot_finding.KNOWN_WEIGHTS` | `spot_finding.KNOWN_MODELS` (no alias) |
| `spot_finding.KnownWeights` | `spot_finding.KnownModel` (no alias) |
| `spot_finding.WeightsFile` | `spot_finding.ModelFile` (no alias) |
| `spot_finding.MissingWeightsError` | `spot_finding.MissingModelError` (no alias) |
| `spot_finding.WeightsHashMismatchError` | `spot_finding.ModelHashMismatchError` (no alias) |
| `preprocessing.normalize_intensity`, the root export `starfinder.normalize_intensity` and `FOV.normalize_intensity` | `preprocessing.normalize_min_max`, root `starfinder.normalize_min_max` and `FOV.normalize_min_max` (no alias) |
| `preprocessing.reconstruct_background`, the root export `starfinder.reconstruct_background` and `FOV.reconstruct_background` | `preprocessing.correct_background`, root `starfinder.correct_background` and `FOV.correct_background` (no alias) |
| `preprocessing.summary_stage` | `preprocessing.summary_recipe` (no alias) |
| `preprocessing.supplied_section` | `preprocessing.fit_supplied_section` (no alias) |
| `dataset.read_checkpoint` | `dataset.load_checkpoint` (no alias; the method `FOV.load_checkpoint` keeps its name) |
| `report.read_report` | `report.load_report` (no alias) |
| `preprocessing.read_histograms` | `preprocessing.load_histograms` (no alias) |
| `preprocessing.write_histograms` | `preprocessing.save_histograms` (no alias) |
| `preprocessing.read_supplied_statistics` | `preprocessing.load_supplied_statistics` (no alias) |
| `preprocessing.write_supplied_statistics` | `preprocessing.save_supplied_statistics` (no alias) |
| `segmentation.labels_to_grid` | `segmentation.resample_labels` (no alias) |
| `segmentation.normalize_percentiles` (the function) | `segmentation.normalize_percentile_unclipped` (no alias) |
| `assignment.molecule_table_from_csv` | `assignment.import_legacy_molecules_csv` (no alias) |
| `assignment.sample_labels` (the function) | `assignment.labels_at_positions` (no alias) |
| `synthetic.FormedSceneConfig` | `synthetic.AmpliconSceneConfig` (no alias) |
| `synthetic.FormedScene` | `synthetic.AmpliconScene` (no alias; its field `formed`, written as `formed.csv`, keeps its name) |
| `synthetic.generate_formed_scene` | `synthetic.generate_amplicon_scene` (no alias) |
| `synthetic.formed_scene_preset` | `synthetic.amplicon_scene_preset` (no alias; the preset IDs `formed-small-v1` and `formed-z1-v1` stay) |
| `synthetic.save_formed_scene` | `synthetic.save_amplicon_scene` (no alias) |
| `synthetic.CellularSceneConfig` (the sample scene's cells, `SampleSceneConfig.cells`) | `synthetic.CellsConfig` (no alias). The name `CellularSceneConfig` now names the cellular scene's configuration (the former `ContextSceneConfig`), so a call written for the old class fails on its field names |
| `synthetic.ContextSceneConfig` | `synthetic.CellularSceneConfig` (no alias) |
| `synthetic.ContextScene` | `synthetic.CellularScene` (no alias) |
| `synthetic.generate_context_scene` | `synthetic.generate_cellular_scene` (no alias) |
| `synthetic.save_context_scene` | `synthetic.save_cellular_scene` (no alias) |
| `synthetic.save_context_dataset` | `synthetic.save_cellular_dataset` (no alias; the `format` value of `dataset.json` stays `"starfinder.synthetic.save_context_dataset"`) |
| `synthetic.context_scene_preset` | `synthetic.cellular_scene_preset` (no alias; the preset IDs `context-*-v1` stay) |
| `synthetic.CONTEXT_PRESETS` | `synthetic.CELLULAR_PRESETS` (no alias) |
| `synthetic.CONTEXT_PRESET_VERSION` | `synthetic.CELLULAR_PRESET_VERSION` (no alias; its value `"context-presets-v1"` stays) |
| `synthetic.CONTEXT_TRUTH_VERSION` | `synthetic.CELLULAR_TRUTH_VERSION` (no alias; its value `"1"` stays) |
| `synthetic.generate_dataset` | `synthetic.generate_amplicon_dataset` (no alias) |
| `synthetic.closed_form_bank` | `synthetic.generate_closed_form_bank` (no alias) |
| `assembly.sample_anndata` | `assembly.write_sample_h5ad` (no alias; the file `sample.h5ad` keeps its name) |
| `stitching.bigstitcher_macro` | `stitching.export_bigstitcher` (no alias; the method ID `bigstitcher`, the macro text and the Snakemake rule `bigstitcher_macro_sample` stay) |
| `export.fuse` | `export.fuse_images` (no alias; the export part `fuse` keeps its name: `ExportConfig.fuse`, the YAML key `export.fuse` and the `parts` of `export.json`) |
| `FOV.assign` | `FOV.assign_molecules` (no alias) |
| `FOV.save_spots` | `FOV.export_spots` (no alias; the slots `goodSpots` and `allSpots` and their file names stay; W-562 later replaced them: "Molecule and read tables") |
| `FOV.create_subtiles` | `FOV.save_subtiles` (no alias; the workflow key `create_subtiles` stays) |
| `Dataset.channel_record` | private `Dataset._channel_record`, which only the run record calls (no public replacement; the `channels` key of `run.json` stays, and `Dataset.channel_info` and `Dataset.channel_labels` stay public) |
| `FOV.register_rounds(recipe, *, rounds, reference=None, checkpoints=None)` | `FOV.register(recipe, *, rounds, reference=None, checkpoints=None)`, which routes other rounds to the former `register_rounds` (no alias; see the routing below) |

These relocations do not make the old positional arguments or return values valid.
Use typed operation configs and structured results. See the [API
reference](api/python.rst) and [coordination member map](coordination.md).
Registration internals, benchmark helpers and barcode lookup constants are private.
`io` holds image file I/O only: the checkpoint serializers are private to the package
and `dataset.load_checkpoint` is public, beside the method `FOV.load_checkpoint`.

The module renames of W-556 change the module paths only: every member keeps its name,
and the root exports (`decode_barcodes`, `extract_intensities`, `filter_reads`,
`load_codebook`) keep theirs. The OME annotation namespace `"starfinder.sample_output"`
that the package's OME-TIFF files carry stays as it is: it is a format identifier, which
`check_export` reads back.

The member renames of W-557 (W-526, NC-01, NC-02, NM-16, NM-17 and NM-18) change these
recorded strings with the names:

* the `run.json` step name `decode_barcodes` is `decode_reads`, the name of the FOV method
  that `FOV.run` calls to decode. The report reads either name: one mapping of the report,
  `RECORDED_STEP_NAMES` (`report/_values.py`), maps the old step name to the new one, so a
  report of a run written before the rename still shows its decoding method and card;
  `steps.csv` keeps the recorded name. No other reader maps an old name.
* the `reads` value of a molecule table's `source` (`type(reads).__name__`, kept in the
  assignment records) is `ReadDecodingResult` for unfiltered reads;
* the repr `BarcodeDecodingResult: n reads — …` is `ReadDecodingResult: n reads — …`, and
  the `TypeError` and `ValueError` messages that named a renamed function or class (in
  `decode_reads`, `deduplicate_reads`, the read diagnostics, `filter_reads`,
  `score_reads` and the spot-finding model check, "a model of KNOWN_MODELS") name the new
  one;
* an error record's `type` (`type(error).__name__`) is `MissingModelError` or
  `ModelHashMismatchError` for a spot-finding model, the names segmentation records too;
* the description of the workflow schema's spot-finding `model` says `KNOWN_MODELS`.

`spot_finding` and `segmentation` now both define `resolve_model`, `KNOWN_MODELS`,
`KnownModel`, `ModelFile`, `MissingModelError` and `ModelHashMismatchError`. They stay
separate types with separate tables (their schemas and resolver signatures differ), so
import them through their module where both are used. The CLI group `weights`,
`STARFINDER_WEIGHTS_DIR`, `starfinder-weights.json`, the method IDs of
`DECODING_METHODS` and `ENCODINGS`, `PipelineConfig.decoding`, the checkpoint header keys
and `readout_mode` keep their names.

The member renames of W-558 (W-526, NM-02, NM-03, NM-06, NM-07, NM-10, NM-19, NM-20,
NM-26, NC-07, SM-03 and SM-23) make the package's own files read and written with one
verb pair, `load_`/`save_`. They change these recorded strings with the names:

* the `"function"` of the record that `segmentation.resample_labels` returns is
  `resample_labels`, and that of `segmentation.normalize_percentile_unclipped` is
  `normalize_percentile_unclipped`. Neither record is written to a file by the package;
* the `TypeError` messages "summary_recipe requires a PreprocessingRecipe",
  "fit_supplied_section requires a merged HistogramSummary" and "save_histograms
  requires a HistogramSummary" name the new functions.

No `run.json` step name changes: `FOV.run` preprocesses through its recipe (step names
`preprocess:<stage>`), not through `FOV.normalize_min_max` or `FOV.correct_background`.
These keep their names: the registry IDs `min_max_normalization` and `reconstruction`;
the config fields `StarDistConfig.normalize_percentiles` and
`CellposeConfig.normalize_percentiles`, their validation messages and their run-record
keys; the field `SampleScene.sample_labels`; the method `FOV.load_checkpoint`, which
restores state onto a FOV (`dataset.load_checkpoint` returns a dict);
`preprocessing.normalize_percentile`, `supplied_statistics` and `assignment.write_h5ad`;
the schema `starfinder.preprocessing.supplied/1` and its `summarized_after` field; and
the recorded `"coordinates": "one-based"` of the legacy CSV reader.
`segmentation.normalize_percentile_unclipped` differs from
`preprocessing.normalize_percentile`: it returns float32, does not clip, takes linear
percentiles over the given axes and returns a record.

The synthetic renames of W-559 (W-526, NC-06, NM-01, NM-15, NM-22 and SM-20) give the
four synthetic scene kinds names a user can guess: the formed scene is the **amplicon**
scene and the context scene the **cellular** scene; the sample and appearance scenes
keep their names. **`CellularSceneConfig` changes meaning.** Before W-559 it named the
sample scene's cell configuration (`SampleSceneConfig.cells`), which is now `CellsConfig`;
it now names the cellular scene's configuration, the former `ContextSceneConfig`. A call
written for the old class fails on its field names. The renames change these recorded
labels, which no reader dispatches on:

* the `"type"` of a recorded configuration: `"FormedSceneConfig"` is
  `"AmpliconSceneConfig"` (the `requested_config` and `effective_config` of every amplicon
  scene's provenance and the `fov` template of a sample scene's record) and
  `"ContextSceneConfig"` is `"CellularSceneConfig"` (the cellular scene's record and
  requested configuration, also inside an appearance scene's record); the nested
  `type(x).__name__` values follow, so the sample scene's `cells` record `"CellsConfig"`;
* the `generator` paths `starfinder.synthetic.generate_formed_scene`,
  `generate_context_scene`, `generate_dataset` (the `generation.json` of
  `starfinder synthetic generate --mode e2e`) and `closed_form_bank` (the record of a
  closed-form appearance bank) are `generate_amplicon_scene`, `generate_cellular_scene`,
  `generate_amplicon_dataset` and `generate_closed_form_bank`.

These keep their values, because files and random draws carry them: the version values
`"context-presets-v1"` and `"1"`; the preset IDs `formed-small-v1`, `formed-z1-v1` and
`context-{dense3d,tissue2d,culture}[-slow]-v1`; the default `scene_key="formed-v1"` and
the stream components `context.*` and `context.appearance.*`; the namespace `"formed"` and
the column `formed_index`; the files `formed.csv`, `context.json` and `cell_context.csv`;
the record keys `context_truth_version`, `context_preset_version` and
`formed_generator_version`; the `context` key of each cellular preset entry; the `format`
value `"starfinder.synthetic.save_context_dataset"` of `dataset.json`; and the fields
`AmpliconScene.formed`, `SyntheticDataset.formed`, `AppearanceSceneConfig.context` and
`AppearanceScene.context`. The private modules (`_formed.py`, `_context.py`,
`_context_presets.py`) keep their names. The examples `docs/examples/formed_scene.py`,
`formed_geometry.py` and `context_scene.py` are `amplicon_scene.py`,
`amplicon_geometry.py` and `cellular_scene.py`.

`synthetic.save_amplicon_dataset(dataset, directory)` is new: it writes an in-memory
dataset from `generate_amplicon_dataset` in the round-first layout that `Dataset` reads
(`<round>/<fov>/<channel>.tif`), with `codebook.csv` (`entry_id`, `gene_id`,
`color_sequence`) and the truth files `formed.csv`, `round_truth.csv`, `scene_truth.csv`,
`ground_truth.json` and `generation.json`, and no `dataset.json`. The streaming writer of
`starfinder synthetic generate` stays private and unchanged apart from its `generator`
path. **The quickstart's layout changes:** `docs/examples/quickstart.py` now builds its
data with `benchmark_scene_preset("tiny")` and `generate_amplicon_dataset`, and writes it
with `save_amplicon_dataset` into `input/`. The images, the codebook (as
`entry_id,gene_id,color_sequence`) and the truth files are in `input/`; there is no
`synthetic/` folder with FOV-first copies, no `gene,barcode` codebook and no annotation
PNGs. Its results are unchanged.

The renames of W-560 (W-526, SM-17, NM-04, NM-05, NM-08, NM-09, NM-21, NC-03, NC-04 and
NC-05) name what a sample-level function or a FOV method writes or runs.
**`FOV.register_rounds` is merged into `FOV.register`**, which now has the signature
`register(recipe, *, rounds=None, source=None, reference=None, checkpoints=None)` and
routes by the rounds' category before any estimator runs:

* `rounds=None` registers the sequencing rounds other than the reference round, as before;
* when every named round is a sequencing round, the **sequencing route** is the former
  `register`: `source` selects the snapshot, one recipe is kept per FOV in
  `registration_record["recipe"]`, and `reference` and `checkpoints` must be `None`;
* when no named round is a sequencing round (other rounds, and loaded rounds the dataset
  does not configure), the **other-round route** is the former `register_rounds`: its
  reference rules, its checks of labels, grids and folders before estimation, the
  per-round `registration_record["rounds"][round]` and the saved form `other_rounds/<round>/`
  of `checkpoints`; `source` must be `None`;
* sequencing and other rounds in one call, an empty sequence, or a parameter of the other
  route raise `ValueError` naming the rounds or the parameter.

Two behaviours change on purpose. `register(rounds=["morph"])` with the other round
`morph` registers it on the other-round route, where it raised `ValueError` before. A
sequencing round can no longer be registered on the other-round route:
`register_rounds(rounds=["morph", "round2"])` accepted the sequencing round `round2`,
while `register(rounds=["morph", "round2"])` raises the mixing error. `FOV.run` still
calls `register(recipe, rounds=[name], source=…)`, so `run.json` and the report are
unchanged; `FOV.prepare_morphology` and the `nuclei_registration` adapter call `register`.
A registration of other rounds logs `register` as its step, where it logged
`register_rounds`.

The renames change these recorded strings and messages with the names:

* the report's per-FOV step row of an assignment is named `assign_molecules`, where it was
  `assign`, in `steps.csv` and the time figure (the producer in `report/_collect.py`), with
  its reason "no record holds the time of assign_molecules"; the page's assignment card
  reads that row (`report/_page.py`). A report of earlier records gets the new name, since
  the row is made by the report, not read from a record;
* the `ValueError` messages of `FOV.assign_molecules` name it, as do the error of the
  workflow's `reads_assignment` rule when an `assignment` block is present and the
  description of the schema's `assignment` block;
* the error of a missing `anndata` names `write_sample_h5ad`; the geometry and dtype
  errors of `fuse_images` name it ("fuse_images needs known spacing…", "fuse_images
  accepts…"); the `TypeError` of a recipe of another type on the other-round route says
  "register requires a RegistrationRecipe"; and the segmentation error for a morphology
  round without a registration entry names `FOV.register`.

These keep their names: the export part `fuse` (`ExportConfig.fuse`, `PARTS`, the
`export.json` sections and parts, the YAML key `export.fuse` and the warning "export part
fuse could not run"); the workflow key `create_subtiles` and the file names of the
subtiles; the Snakemake rule `bigstitcher_macro_sample`, its script and its adapter; the
method ID `bigstitcher` and the macro text; the file `sample.h5ad`;
`assignment.write_h5ad`, `readout.export_spots`, `Dataset.channel_info` and
`Dataset.channel_labels`; and the `run.json` step name `register`.

## Before and after Python calls

The **before** snippets below are historical text, not runnable imports.

### Load and preprocess

Before:

```text
image = load_multipage_tiff(path)
image = min_max_normalize(image)
```

After:

```python
from starfinder.io import load_volume
from starfinder.preprocessing import normalize_min_max, MinMaxNormalizationConfig

loaded = load_volume(path)
image = normalize_min_max(loaded.image, config=MinMaxNormalizationConfig(
    output_dtype="uint8", output_range=(0, 255)))
metadata = loaded.metadata
```

Loaders preserve dtype by default and return `ImageLoadResult`. Conversion is
explicit; physical calibration is never inferred from shape. See [contracts](api/contracts.md).

### Estimate, then apply registration

Before:

```text
shift = phase_correlate(reference, moving)
registered = apply_shift(moving, -shift)
```

After (given same-grid ZYX arrays):

```python
from starfinder.image import ImageMetadata
from starfinder.registration import TranslationConfig, estimate_transform, apply_transform

result = estimate_transform(reference, moving, config=TranslationConfig(),
    reference_metadata=ImageMetadata("reference"),
    moving_metadata=ImageMetadata("moving"))
registered = apply_transform(moving, result.transform, config=result.application_config)
```

`apply_transform` applies the transform as it is. Do not negate it. Every
transform, a translation included, is a pull map from reference to moving
coordinates (see the translation displacement below). Dense
transforms are pull fields, not forward scene perturbations. Estimation errors
are explicit; optional recovery belongs to [coordination](coordination.md).

### Keep extraction, decoding and filtering separate

Before:

```text
calls, scores = extract_from_location(images, locations, ...)
good_reads = filter_reads(calls, ...)
```

After (given labeled rounds, a SpotFindingResult and a validated Codebook):

```python
from starfinder.readout import (NeighborhoodSumConfig, WtaDecoderConfig,
    ReadFilterConfig, extract_intensities, decode_reads, filter_reads)

intensities = extract_intensities(rounds, spots,
    config=NeighborhoodSumConfig(neighborhood_radius_zyx=(1, 2, 2)))
decoded = decode_reads(intensities, codebook, config=WtaDecoderConfig())
filtered = filter_reads(decoded, config=ReadFilterConfig())
accepted = filtered.accepted
```

All rows retain `(spot_namespace, spot_id)`, including ambiguous/unmatched calls.
Filtering keeps rejection reasons. Export joins by identity rather than row order.
Rerun filtering without decoding, or decoding without repeating extraction.

### Coordinate a pipeline

Before: `dataset.fov(id).run_streaming(...)` and separate batch stage calls.
After: `dataset.fov(id).run(pipeline, execution=ExecutionConfig("streaming"))`.
`PipelineConfig` specifies scientific stages; `ExecutionConfig` specifies residency.
Use `from_workflow_config(config, rule)` at the shared YAML boundary. Full
construction and recovery examples are in [coordination](coordination.md).

### Preprocessing recipe

`PipelineConfig` no longer has one fixed slot per preprocessing operation. An
ordered {py:class}`~starfinder.preprocessing.PreprocessingRecipe` in
`PipelineConfig.preprocessing` replaces them, as specified in the
[preprocessing contract](preprocessing-contract.md). There are no aliases.

| Before | After |
| --- | --- |
| `PipelineConfig.normalization` | `PreprocessingStep(MinMaxNormalizationConfig(...))` in `preprocessing.steps` |
| `PipelineConfig.histogram` | `PreprocessingStep(HistogramMatchingConfig(...))` in `preprocessing.steps` |
| `PipelineConfig.histogram_reference_channel` | `HistogramMatchingConfig.reference_channel` (default 0) |
| `PipelineConfig.reconstruction` | `PreprocessingStep(ReconstructionConfig(...))` in `preprocessing.steps` |
| `PipelineConfig.reconstruction_after_registration` | The same step in `preprocessing.post_registration`, which accepts only `ReconstructionConfig` |
| `PipelineConfig.tophat` | `PreprocessingStep(TophatConfig(...))` in `preprocessing.steps` |
| `PipelineConfig.projection` | Removed. `FOV.run` never projects; projection is an output view, and 2D data are volumes with Z = 1 |
| `FOV.match_histogram(reference_channel=...)` | `FOV.match_histogram(config=HistogramMatchingConfig(reference_channel=...))` |

Steps run in the declared order. The fixed order of the removed slots
(normalization, histogram matching, reconstruction, top-hat) is recipe 1, which
reproduces the earlier outputs exactly:

```python
from starfinder.dataset import PipelineConfig
from starfinder.preprocessing import (HistogramMatchingConfig, MinMaxNormalizationConfig,
    PreprocessingRecipe, PreprocessingStep, ReconstructionConfig)

pipeline = PipelineConfig(preprocessing=PreprocessingRecipe((
    PreprocessingStep(MinMaxNormalizationConfig("uint8", (0, 255))),
    PreprocessingStep(HistogramMatchingConfig(reference_channel=0)),
    PreprocessingStep(ReconstructionConfig(radius_yx=3)))))
```

Every step runs through one wrapper that checks shape, dtype, finiteness and
metadata. A histogram-matching step therefore keeps the input dtype; a
`HistogramMatchingConfig.output_dtype` that differs from it now raises in
`FOV.run`. The workflow keys `enhance_contrast`, `hist_equalize`, `morph_recon`
and `tophat` are unchanged and translate to recipe 1. `run.json` and the
`registered` checkpoint gain a `preprocessing` entry with the recipe and one
record per round and step.

Recipes can also keep named snapshots (`PreprocessingStep.save_as`) and name an
`extraction_source` and a `registration_source`; the new Python-only workflow
key `preprocessing` declares such a recipe and cannot be combined with the
legacy keys. These additions are optional: a recipe without them gives the same
results as before. The preprocessing record gains the two sources, `save_as` per
step and the transforms applied per round and snapshot, and the `registered`
checkpoint stores the extraction source under `registered/<snapshot>/`.
Checkpoints written before these additions still load, with no snapshots.

### Preprocessing registry names

Three public preprocessing names follow the shared
[method registry](method-registry.md) terms. There are no aliases; replace the
identifiers where you import or use them.

| Before | After |
| --- | --- |
| `starfinder.preprocessing.STEPS` | `starfinder.preprocessing.PREPROCESSING_METHODS` |
| `starfinder.preprocessing.StepSpec` | `starfinder.preprocessing.PreprocessingSpec` |
| `starfinder.preprocessing.RecipeStep` | `starfinder.preprocessing.PreprocessingStep` |

`PreprocessingSpec` keeps the positional fields of `StepSpec` and gains the
keyword-only fields `requires`, `min_shape_zyx`, `post_registration` and
`supplied`, all with defaults, so existing constructions stay valid.
`PreprocessingRecipe`, `StepContext`, `StepResult`, `run_step`, `step_spec`,
`step_config_type`, the recipe field `steps` and every YAML key and saved field
keep their names. The only changed message is
`requires PreprocessingStep entries` (was `requires RecipeStep entries`).

### Reference merged image

`FOV.save_reference_image` used to write the reference round's full ZYXC image
as OME-TIFF, with channels not merged, to `images/ref_merged/{fovID}.tif`.
The MATLAB backend writes the channel-merged image to the same file. Python now
writes the same content as MATLAB: the reference round's detection image,
reduced to its channel maximum (`reference_image="merged"`, the default) or to
one channel (`reference_image="single-channel"` with `reference_channel`), as a
ZYX TIFF. With `projection=ProjectionConfig()` (top-level
`maximum_projection`) it is also reduced along Z and saved as YX. The filename
and dtype are unchanged. Python outputs written before this change hold ZYXC;
rerun the rule to replace them. `ProjectionConfig` gains `axis`, `"z"` (the
default and earlier behavior) or `"channel"`; see
[projection views](workflows.md#projection-views-and-the-reference-merged-image).
Since W-564 the Python file is `fovs/{fovID}/reference.tif` by default; see "The FOV folder, the reference image and channel folders by label".

### Generate, evaluate and report

`synthetic.generate_amplicon_dataset(codebook, config, fov_ids=...)` returns arrays and
truth records, or streams each round to a callback; `synthetic.save_amplicon_dataset`
writes an in-memory dataset. The CLI persists generated inputs; benchmark cases
explicitly select them.
`evaluation` accepts supplied results/truth and explicit matching/units; it does
not rerun algorithms. [Benchmark](benchmark.md) documents immutable processing
runs, checksum validation, and separate saved-output evaluations/reports.

### Generate synthetic data with one generator

The historical generator (`SyntheticConfig`, integer-center `render_spots`,
hash-seeded registration pairs) is removed. Benchmark data and test fixtures
come from the amplicon-scene generator, with benchmark presets that keep the
historical shapes, amplicon counts, seeds and shift ranges. There are no aliases.

| Before | After |
| --- | --- |
| `SyntheticConfig`, `get_preset_config(name)` | `benchmark_scene_preset(name, dtype=...)` returns `(Codebook, AmpliconSceneConfig)`; adjust with `dataclasses.replace`; `SCENE_PRESETS` lists every preset |
| `generate_dataset(config, preset=...)` | `generate_amplicon_dataset(codebook, config, fov_ids=..., preset=..., on_round=...)` |
| `generate_registration_pairs([preset], seed=...)` | `generate_registration_pair(preset, deformation=..., seed=...)`, one call per pair |
| `generate_displacement_field`, `DEFORMATION_CONFIGS` | `deformation_geometry(name, shape)` with `GeometryConfig` affine, polynomial and RBF terms; `forward_displacement(transform, shape)`. Magnitudes are reduced where needed for invertibility; notably `multi_point` uses 1.3–10.5 px instead of 5–20 px below `tissue` (see the synthetic API) |
| `render_spots`, `generate_volume` | `generate_amplicon_scene` (single scenes) |
| `generate_codebook(n)` list of `(gene, barcode)` | `generate_codebook(n)` returns a `Codebook` with `base_sequence` |
| `SyntheticDataset.spot_truth`, `perturbations`, `molecular_truth`, `config` | `formed`, `round_truth`, `spot_truth`, per-FOV `provenance`; `historical_truth` keeps the ground_truth.json keys |
| `--dtype uint8` default; registration uint8 only | uint16 default in both modes; `--dtype uint8` scales intensities by 1/16 |

Images change: uint16 with a camera offset, a spatially varying background,
Poisson plus read noise, lognormal brightness, crosstalk and a round trend
replace the constant background 20, Gaussian σ 10 and uint8 200–255 spots.
Positions and shifts are continuous, spots are no longer dropped at the border,
and the 8-gene test codebook gains GeneI–GeneL so every channel is used in
every round. Outputs add `formed.csv` and `round_truth.csv`; registration
pairs add continuous forward fields for every deformation.

### Name color sequences and gene IDs the same way

The Python API names the color sequence `color_sequence` (decoded reads:
`observed_color_sequence` and `decoded_color_sequence`) and the gene
identifier `gene_id`. There are no aliases. MATLAB code and the
MATLAB-compatible files keep their names: `scene_truth.csv`,
`ground_truth.json` (and `historical_truth`), `codebook.csv` and the exported
spot CSVs still use `gene` and `color_seq`.

| Before | After |
| --- | --- |
| `FormedScene.formed.codeword`, `formed.csv` column `codeword` | `color_sequence` |
| `SyntheticDataset.spot_truth` columns `gene`, `color_seq` | `gene_id`, `color_sequence` |
| `evaluate_decoding` columns `gene`, `color_seq` | truth `gene_id`, `color_sequence`; decoded `gene_id` and `sequence_column` (default `observed_color_sequence`) |
| `evaluate_decoding` values `gene_accuracy`, `color_seq_accuracy`; counts `*_gene`, `*_color_seq` | `gene_id_accuracy`, `color_sequence_accuracy`; `*_gene_id`, `*_color_sequence` |
| `decode_color_sequence(color_seq=...)` | `colors_to_bases(color_sequence=...)` |

`formed.csv` files written before this change carry `codeword` and are not
supported; regenerate them.

### Registration recipe

`PipelineConfig.registration` is a
{py:class}`~starfinder.registration.RegistrationRecipe` or `None` (no registration),
not a tuple of steps, as specified in the
[registration contract](registration-contract.md). `FOV.register` takes a
recipe too. There are no aliases.

| Before | After |
| --- | --- |
| `PipelineConfig(registration=(RegistrationStep(...), ...))` | `PipelineConfig(registration=RegistrationRecipe(steps=(RegistrationStep(...), ...)))` |
| `PipelineConfig(registration=())` | `PipelineConfig(registration=None)` (the default) |
| `fov.register(RegistrationStep(TranslationConfig()))` | `fov.register(RegistrationRecipe((RegistrationStep(TranslationConfig()),)))` |
| `RegistrationStep(config, reference_image, moving_image, reference_channel, recovery, warp)` | `RegistrationStep(config, recovery, signal)` |
| `reference_image`, `moving_image` = `"merged"` | `RegistrationSignalConfig(mode="max")` on the recipe (see the signal change below); `mode="sum"` keeps the earlier sum |
| `reference_image`, `moving_image` = `"single-channel"`, `reference_channel=c` | `RegistrationSignalConfig(mode="channel", reference_channel=c)`; a channel label is accepted too |
| `RegistrationStep.warp` | `RegistrationRecipe.warp`, one final resampling per moving round (`None` derives it) |
| per-step `RegistrationResult.application_config` in `FOV` | the round's one `WarpConfig`, also in `FOV.registration_record["application"]` |

A recipe is zero or more global steps (`translation`, `rigid`, `affine`) followed
by at most one local step (`demons`, `bspline`, `tps`, `cpd`); a recovery
alternative must have its step's kind, so a TPS step can no longer fall back to
translation. `FOV.registration_chains` holds each moving round's
{py:class}`~starfinder.registration.TransformChain`, and
`FOV.registration_attempts[round]` gains the estimation fields (`record`,
`step`, `attempt`, `fallback`, `backend`, `backend_versions`, `reference`,
`qc`) and one `application` entry per round. `run.json` keeps
`format_version` 1 (note (W-565): now 2, see "run.json version 2"); its `config.pipeline.registration` holds the recipe's
fields. In the workflow, the Python-only `registration` key declares a recipe;
see [workflow configuration](workflow-configuration.md).

### Save and reload pipeline checkpoints

An earlier development branch had an artifact and provenance contract with
contract IDs, reference chains, per-file manifests, immutable directories and
HTML reports. It was never merged, and it has been replaced by three plain
checkpoints and one run record. There are no aliases.

| Development branch | Current interface |
| --- | --- |
| `starfinder.provenance` run recorder and event snapshots | `run.json`, written by `FOV.run(..., checkpoints=CheckpointConfig())` |
| `io.checkpoints`, `FOV.load_image_checkpoint` | `FOV.save_checkpoint("registered")`, `FOV.load_checkpoint("registered")` |
| `io.candidates` combined candidates and signals | `candidates.csv` or `.parquet`, reloaded as `SpotFindingResult` and `IntensityExtractionResult` |
| `io.molecules`, final checkpoints and `MoleculeIndex` | `pre_qc` checkpoint for decoding; final outputs stay the workflow CSVs |
| `reporting` HTML summaries | None; read `run.json` or the checkpoint tables directly |

Images are OME-TIFF (`<round>.ome.tif`) through `save_volume`, tables are CSV unless Parquet is
requested, and the location is `<output_root>/checkpoints/<fov_id>/`. Spot
identity, input SHA-256 hashes, atomic writes and reruns of decoding or
filtering without images are kept. See [checkpoints](checkpoints.md).
Since W-564 the location is `<output_root>/fovs/<fov_id>/checkpoints/` beside `run.json`; see "The FOV folder, the reference image and channel folders by label".

### Registered checkpoints version 2

Note (W-565): the headers are now of version 3, which adds the payload hashes (`files`);
see "run.json version 2".

The checkpoint `FORMAT_VERSION` is 2 for the three stages. A registered
checkpoint stores the new kinds `affine` (the 4×4 index matrix and the physical
parameters, inline) and `bspline` (the grid inline, the coefficients in
`<round>_field.npz`), the step index of each result, one `applications` entry
per round with its `WarpConfig` instead of a per-result `application_config`,
and the recipe summary. Reloading rebuilds a `TransformChain` per round;
applying it to the pre-registration images reproduces the registered images
bit for bit. Version-1 checkpoints still load: their translation and dense
results keep their per-result `application_config`, no chain is built, and
`registration_record["semantics"]` is `"sequential"`. Nothing converts them to
a recipe; only their translation corrections load as displacements (below).
See [checkpoints](checkpoints.md).

### New registration methods and Z=1 demons

`RigidConfig`, `AffineConfig` and `BSplineConfig` are registered in
`REGISTRATION_METHODS` as `rigid`, `affine` and `bspline`. They need the optional
extra `registration-elastix` (`itk-elastix` 0.25.4 and `itk` 5.4.7), imported
only when one of them runs, and return the new `AffineTransform` or
`BSplineTransform`. `DemonsConfig` now accepts Z=1 input and estimates it as 2D
(its registry entry declares `dimensions={2, 3}`); calls that relied on demons
rejecting Z=1 with `IncompatibleGeometryError` now get a field of shape
(1, Y, X, 3) whose Z component is 0. 3D demons results are unchanged.
`RegistrationDiagnostics` gains optional fields (`final_metric_value`,
`stop_condition`, `elapsed_iterations`, `final_rms_change`, `backend_versions`,
`backend_parameters`, `spacing_source`), all `None` unless a method records them.
`REGISTRATION_METHODS` maps each exact config type to its method; subclasses of
a registered config are not accepted anywhere. A step that fails a configured
routine QC criterion (`RegistrationRecipe.qc`, nothing by default) raises
`RegistrationRejectedError`, a `RegistrationEstimationError` that recovery may
allow.

### Other rounds

`FOV.register(recipe, *, rounds, reference=None)` on other rounds and
{py:class}`~starfinder.registration.ExternalReference` are new: other rounds, such
as morphology rounds, register to a reference round or an external reference
image through a shared stain, and the transform is applied to every channel of
the round. A channel label in `RegistrationSignalConfig(mode="channel")` is
looked up in the round's own labels: `Dataset.other_channel_order` (new) gives
an other round its channels, and `FOV.load_images` checks each round against
`Dataset.channel_labels(round)`. Estimation attempts gain `reference_sha256`
(`None` for a round reference). `FOV.save_processing_log` accepts `"nr"` (W-563 later
removed it: "Sequencing logs and the morphology record").

With `backend: python`, the `nuclei_registration` rule now runs the Python
script `workflow/scripts/nuclei_registration.py` instead of MATLAB, with the
same inputs and outputs; see [workflows](workflows.md). The MATLAB backend's
rule and script are unchanged.

### Translation displacement

`TranslationTransform` stores the detected displacement `d` of the moving
content as `displacement_zyx` and pulls from `p + d`, with
`direction="reference_to_moving"`, like every other transform. It replaces
`correction_zyx`, the correction `c = −d` of the pull `p − c` labelled
`moving_to_reference`; there is no alias.

| Before | After |
| --- | --- |
| `transform.correction_zyx` | `transform.displacement_zyx`, the negated value |
| `tuple(-v for v in transform.correction_zyx)` (the detected displacement) | `transform.displacement_zyx` |
| `TranslationTransform(c, ...)`, also positionally | `TranslationTransform(d, ...)` with `d = −c` |
| `TransformChain.translation()`: the sum of the corrections | the sum of the displacements |
| QC transform summary and `run.json` records `{"kind": "translation", "correction_zyx": c}` | `{"kind": "translation", "displacement_zyx": d}` |
| benchmark `transform.json` field `correction_zyx` | `displacement_zyx` |
| benchmark truth `{"correction": "correction.json"}` holding `c` | `{"displacement": "displacement.json"}` holding `d` |
| `RegistrationResult` summary `correction_zyx (…)` | `displacement_zyx (…)` |

Version-2 registered checkpoints write `displacement_zyx`. Version-1 registered
checkpoints store `correction_zyx`; the reader loads it as
`displacement_zyx = −correction_zyx`, and the result re-applies identically.
Registered images, pull fields and the `log/gr_shifts` rows (which already held
the detected displacement) are unchanged.

### Spot-finding registry

`SPOT_FINDING_METHODS` in `starfinder.spot_finding` maps each exact config type
(`LocalMaximaConfig`, `NoiseLandmarkConfig`, `PercentileCentroidConfig`) to its
`SpotFindingSpec` (`local_maxima`, `noise_landmark`, `percentile_centroid`).
`find_spots`, `SpotFindingResult`, `PipelineConfig.spot_finding`, `FOV.find_spots`,
the workflow adapter and the `candidates` checkpoint reader look methods up
through it by exact type, so a subclass of a detection config, which
`isinstance` accepted before, now raises `TypeError` (`unsupported detection
config` from `find_spots`). A missing optional dependency of a method raises
`SpotFindingBackendUnavailableError`, an `ImportError` naming the module and
the extra. Detection results are unchanged; see {doc}`spot-finding-contract`.

### Spot-finding workflow keys

The `spot_finding` block of a Python rule accepts the Python-only key `method`
(a pipeline method of `SPOT_FINDING_METHODS`, default `local_maxima`), every
init field of the selected config (for example `exclude_border` and
`measure_peak_intensity`, which YAML could not set before) and
`channel_overrides`, a mapping from channel label to config fields. A
rule-level Python-only `device` key sets `ExecutionConfig.device`. The legacy
keys `intensity_estimation`, `intensity_threshold` and `min_distance` stay
aliases of `threshold_mode`, `threshold_value` and `min_distance_voxels` for
`local_maxima` only; for another method `min_distance` is that config's own
field. The schema drops `local` from `intensity_estimation`, which neither
Python nor MATLAB accepts.

| Before | After |
| --- | --- |
| `min_distance` and `min_distance_voxels` together: `min_distance_voxels` silently wins | `ValueError` |
| `intensity_threshold` with `threshold_value`, or `intensity_estimation` with `threshold_mode` | `ValueError` (the field names were unknown keys before) |
| `exclude_border` in YAML: unknown key | the `LocalMaximaConfig` field |

### Detection plan and execution device

`find_spots`, `FOV.find_spots` and `PipelineConfig.spot_finding` accept a
`SpotFindingPlan(config, channel_overrides=())`: one method for every channel,
and `ChannelOverride(channel, config)` entries that replace the whole config of
one channel (same exact config type; `channel_labels` `None` or the plan's).
An override may change any setting except one that changes the output columns,
such as `measure_peak_intensity`, which raises `ValueError` at plan validation.
A bare config means a plan without overrides, and `PipelineConfig` keeps what it
is given. `ExecutionConfig` gains `device="cpu"`, the only accepted value
(`ValueError` otherwise), and `find_spots` takes the same `device` keyword.
`run.json` records `config.execution.device`, and the `find_spots` step record
gains `methods`, a list with the detection's provenance entry (`stage`,
`method`, `config_type`, `implementation`, `config`, `requires`, `artifacts`,
`execution`); see {doc}`checkpoints`.

`SpotFindingPlan` also takes `rounds`, the round labels to detect in (the workflow
`spot_finding` block's Python-only `rounds` key). `None`, the default, is the reference
round only, and every result, identity, column, checkpoint and golden digest stays as
before. With `rounds`, `FOV.run` detects each listed round after its registration and
post-registration steps and `FOV.find_spots` detects the listed rounds' resident images;
a listed round whose metadata or shape differs from the reference round's raises
`IncompatibleGeometryError`. The result is one table with a `round` column (pandas
string), the rounds in `FOV.run` order (reference first) and `spot_id` running
`"0"`…`"N-1"` over the combined table, so the reference round's identities are
unchanged; coincident candidates are never merged. The new `SpotFindingResult.plan`
holds the plan (`None` on construction means `SpotFindingPlan(config)`). Direct
`find_spots` detects one image and raises `ValueError` for a plan with `rounds`.
**Intentional change:** decoding a candidate set of several rounds raises
`ValueError("decoding candidates from several detection rounds needs a readout mode
(§2.8)")`, from `FOV.run` before any processing and from `FOV.decode_reads` for a
spot table with a `round` column. Extraction reads every candidate in every sequencing
round, as before; in `FOV.run` it then runs after the last detected round, which needs
batch mode or `retain_images=True`. Since W-293 this holds in readout mode
`multiplexed`; readout mode `direct` reads such a set ([Readout mode](#readout-mode)).

### Spot-finding diagnostics

`SpotFindingResult.diagnostics` keeps its keys and adds `effective_settings`
(the config of every channel, serialized as `run.json` serializes dataclasses),
`warnings` and `execution` (device, torch build when a method uses it, and the
thread settings in effect). Local maxima adds `noise`: per channel the zero
fraction, median, MAD and threshold, in every threshold mode. Every method with a
`channel` column adds `counts` (candidates per channel), `outcomes` (`ok`, `empty`, or
`constant`: the pipeline methods never run on a constant channel, which yields no
candidates; its thresholds and local-maxima noise record are still recorded, and
Spotiflow and Piscis still verify their weights), `native` (per channel the minimum,
median and maximum of `radius` or `probability`, when the method has them); every
method adds `software` (the versions of starfinder, NumPy, SciPy, scikit-image and the
method's optional dependencies). A result of several rounds keeps `thresholds`,
`counts`, `outcomes`, `noise` and `merged` per round under `rounds`. A method that
finds nothing returns its declared columns with their dtypes.
`plot_detections(image, result, *, channel, z, yx_window=None, round=None, ax=None)`
draws one channel's detections on one slice or crop and returns the matplotlib axes;
`FOV.run` never plots.

### Candidates checkpoint plan keys

`candidates.json` adds the optional keys `detection_rounds` (`null` or the plan's
rounds), `detection_plan` (the channel overrides as `{channel, config}` entries),
`execution` and `weights` (the provenance `artifacts` entries of the loaded weights),
and the table may carry the spot columns `round` (string), `radius` and `probability`
(float64). `FORMAT_VERSION` stays 2. `dataset.load_checkpoint(..., "candidates")` rebuilds the
plan, so the reloaded `spot_result.plan` equals the original; a checkpoint written
before these keys loads unchanged, with no overrides and `rounds` `None`. See
{doc}`checkpoints`.

### Spot evaluation metrics

`starfinder.evaluation.spot_finding` gains `localization_errors` (per-axis,
lateral and 3D error maxima and 95th percentiles of the matched pairs) and
`classify_detections` (matched, duplicate and spurious counts, with duplicates
split by group, such as the channel). Both read an `evaluate_spots` result and
return an `EvaluationResult`.

### Pretrained weights

`spot_finding.KNOWN_MODELS` lists the pretrained weights Starfinder can fetch and verify
(two Spotiflow 3D, two Spotiflow 2D and two Piscis models, with full SHA-256
values) and sets no default. `starfinder weights fetch <method> <model>`
(`fetch_model`) downloads, verifies and installs a model under
`STARFINDER_WEIGHTS_DIR` (else `$XDG_CACHE_HOME/starfinder/weights`);
`starfinder weights list` and `starfinder weights verify` show and re-hash the
local copies. `resolve_model` raises `MissingModelError` or
`ModelHashMismatchError`. Detection never downloads.

### Starfish LoG

`StarfishLogConfig` selects the pipeline method `starfish_log`, a native
reimplementation of starfish `BlobDetector` (`blob_log`, `is_volume=True`, no
reference image, one round and channel at a time; starfish `1fb00cbc`) that needs
no extra and does not depend on starfish. Its tables equal starfish's exactly
(the W-266 parity tables in `test/data/starfish_blob_parity`). `min_sigma`,
`max_sigma` (σ in voxels, a number or a ZYX 3-tuple), `num_sigma` and `threshold`
are required, because starfish has no defaults for them; `overlap` (0.5) and
`exclude_border` (False) keep starfish's defaults. Integer images are scaled by
their dtype maximum to float32 [0, 1] (`img_as_float32`) and a float image outside
[0, 1] raises `ValueError`; the threshold applies to that scaled image. The table
holds the truncated integer coordinates as float64, `channel`, `peak_intensity`
(the original pixel value) and `radius` (`round(σ·√ndim)`). A Z=1 image is
detected as a plane (z=0), where a 3-tuple σ raises `IncompatibleGeometryError`.
There is no tiling and no memory guard: `diagnostics["geometry"]` records the
estimate 10.4 bytes × `num_sigma` × voxels. In YAML, `method: starfish_log`
requires the four settings.

### Spotiflow and Piscis

`SpotiflowConfig` and `PiscisConfig` select the pipeline methods `spotiflow`
(spotiflow 0.6.5) and `piscis` (piscis 1.1.0, the `Piscis` class), which need the
extras `spotiflow` and `piscis` (`pip install starfinder[spotiflow]`; CPU torch
2.7.1; not available on Python 3.14 and later). Without the extra a run raises
`SpotFindingBackendUnavailableError` naming the module and the extra. Every config
names its weights (`model`, a row of `KNOWN_MODELS`; there is no default), and an
unknown model or one of the other method raises `ValueError` listing the known
names. Detection loads only the copy fetched with `starfinder weights fetch`: each
call re-hashes every file `KNOWN_MODELS` lists for the model before the model is
built, passes the libraries absolute paths only (a relative
`STARFINDER_WEIGHTS_DIR` is resolved), never downloads and never reads
`~/.spotiflow`, `~/.piscis/models` or the Hugging Face cache; a loaded model is
reused within the process. A 3D Spotiflow model (`synth_3d`,
`smfish_3d`) detects Z>1 images in 3D and a 2D model (`general`, `hybiss`) detects
Z=1 images as a plane; another pairing, or a shape below the model's minimum,
raises `IncompatibleGeometryError` where Spotiflow would return nothing. Piscis
uses plane mode for Z=1 and stack mode for Z>1, where `z` is an integer and
vertically aligned spots merge. `scale` must be 1. The tables hold fractional
coordinates, `channel`, `peak_intensity` (the original pixel value at the rounded
coordinate) and, for Spotiflow, `probability`. `diagnostics["effective_settings"]`
records the resolved native defaults (Spotiflow's stored `prob_thresh`, `subpix`
and `n_tiles`; Piscis's tile size), `diagnostics["model"]` and the `run.json`
provenance `artifacts` record the path, SHA-256, source and revision of every
verified file, and `diagnostics["geometry"]` records Spotiflow's `n_tiles` or Piscis's
tile size and keep-boundaries, near which Piscis can repeat a spot (not merged).
`KNOWN_MODELS` entries gain `extracted`, every file of a Spotiflow archive with
its SHA-256; `resolve_model(..., extracted=...)` also checks the named files, and
detection and `starfinder weights verify` re-hash all of them. The weights root is
always an absolute path (`~` expanded, relative paths resolved).

### Local-maxima W-218 merge

`LocalMaximaConfig.merge_radius_zyx` (default `None`) is the opt-in W-218
within-channel merge: after border exclusion, a channel's maxima are visited by
decreasing pixel value, then increasing z, y, x, and a maximum is dropped when an
earlier kept maximum of the same channel lies within the ellipsoid
Σ (Δᵢ / rᵢ)² ≤ 1 (radii in voxels; the Z radius is unused for Z=1). It removes
tied maxima of a plateau and split maxima of one amplicon; kept maxima are
unchanged, identities are assigned after the merge, and `diagnostics["merged"]`
records the number removed per channel. W-268 kept `None` as the default, so
detections are unchanged unless the option is set; channels are never merged with
each other (cross-channel duplicates are §2.8's). `exclude_border` is settable in
YAML for the border misses W-218 reported.

### Spot-finding pipeline field

The spot-finding field of {py:class}`~starfinder.dataset.PipelineConfig` is
`spot_finding`, the stage name that the module, the YAML block, the registry and
the provenance `stage` already use; it was `detection`. There is no alias, so
`PipelineConfig(detection=...)` raises `TypeError`. The accepted types, the
validation and the results are unchanged, and the error message for a wrong type
names the new field. YAML already used the `spot_finding` block and is unchanged.

| Before | After |
| --- | --- |
| `PipelineConfig(detection=LocalMaximaConfig())` | `PipelineConfig(spot_finding=LocalMaximaConfig())` |
| `config.detection`, `replace(config, detection=None)` | `config.spot_finding`, `replace(config, spot_finding=None)` |
| `run.json`: `config.pipeline.detection` | `run.json`: `config.pipeline.spot_finding` |
| `TypeError("detection requires its typed operation config")` | `TypeError("spot_finding requires its typed operation config")` |

`run.json` keeps `format_version` 1 (note (W-565): now 2, see "run.json version 2"). Starfinder does not read `config.pipeline`
back from `run.json`, so records written before the rename stay valid as they
are; a script that reads them should accept either key. The `candidates.json`
keys `detection_config`, `detection_rounds`, `detection_plan` and
`detection_diagnostics` and the preprocessing `detection` image keep their names.

### Readout mode

§2.8 adds two readout modes as one dataset setting ({doc}`readout-contract`,
"Readout modes" and "Direct readout"). `Dataset.readout_mode` is `"multiplexed"`
(the default: every earlier result, identity and digest is unchanged) or
`"direct"`, where a round and channel identify a gene.
{py:class}`~starfinder.readout.DirectPanel` holds the gene of each (round, channel),
{py:func}`~starfinder.readout.load_direct_panel` reads it from a
`round,channel,gene_id` CSV, and `Dataset.load_direct_panel` stores it as
`Dataset.direct_panel`; a repeated gene or (round, channel) raises `ValueError`
naming it. {py:class}`~starfinder.readout.DirectDecoderConfig` (`direct` in
`DECODING_METHODS`, no parameters) and
{py:func}`~starfinder.readout.decode_direct` assign each candidate the gene of its
own round and channel: one read per candidate with `round`, `channel`, `entry_id`
`"<round>/<channel>"`, `call_type` `direct`, and the diagnostic columns
`own_channel_rank` and `own_channel_fraction`; a (round, channel) without a gene is
`unmatched` with `unmapped_channel`, and a zero own round `no_signal`. Nothing is
merged or reassigned. `ReadDecodingResult` gains `readout_mode`.

In `direct` mode, `extract_intensities(..., readout_mode="direct")` (which
`FOV.run` and `FOV.extract_intensities` pass from the dataset) reads each candidate
in its own round only; the other rounds are `0.0` with `valid=False`.
`IntensityExtractionResult` gains `box_voxels` (N×R int64, the voxels each box
summed, 0 for a round that was not read; since W-294 a candidates checkpoint
stores it with the background measurements, and a checkpoint without them reloads
it as `None`). `FOV.run` and `FOV.decode_reads`
assign with `DirectDecoderConfig` and `Dataset.direct_panel`.

Decoding a multi-round candidate set (a `round` column) needs
`readout_mode="direct"`: in `multiplexed` mode it still raises `ValueError`, whose
message keeps "needs a readout mode (§2.8)" and adds "set readout_mode='direct'".
A decoder that does not support the dataset's mode (`wta` or `codebook_aware` in
`direct` mode, `direct` in `multiplexed` mode) raises `TypeError` naming the mode
and the decoder, in `FOV.run`, `FOV.decode_reads`, `decode_reads` and
`decode_direct`; `direct` mode without a `round` column, or without a loaded panel,
raises `ValueError`.

The YAML top-level key `readout_mode` (Python only; the schema accepts `direct`
only with `backend: python`) sets the mode. In `direct` mode the Python-only
`decoding` block defaults to and may only name `method: direct`, the rule's
codebook input is the panel CSV, and the barcode keys `load_codebook.split_index`
and `encoding` and `reads_filtration.end_base`, `split_index`, `n_barcode_segments`
and `exclude_invalid_endpoints` raise. In `multiplexed` mode `decoding: {method:
direct}` now raises `TypeError` instead of "unknown decoding method". The
`candidates.json` and `pre_qc.json` headers and the `config` record of `run.json`
gain `readout_mode`; a checkpoint without it loads as `multiplexed`, and
`FOV.load_checkpoint` raises when it differs from the dataset's mode
({doc}`checkpoints`).

### Readout encodings and decoders

§2.8 registers the barcode encodings and the decoders on the shared registry
mechanism ({doc}`method-registry`, {doc}`readout-contract`, "Encoding registry").
`starfinder.readout.ENCODINGS` maps each encoding config type to its
`EncodingSpec`: `two_base` is {py:class}`~starfinder.readout.EncodingConfig`, which
gains the discriminator `method="two_base"` (its positional constructor is
unchanged), and `one_base` is the new
{py:class}`~starfinder.readout.OneBaseEncodingConfig`, whose `base_to_color` maps A,
C, G and T one-to-one onto the colors 1–4 and has no default.
`starfinder.readout.DECODING_METHODS` maps `WtaDecoderConfig` (`wta`) and
`CodebookAwareDecoderConfig` (`codebook_aware`) to their `DecodingSpec`, with the
readout modes, encoding kinds, rescue and score columns each declares. Lookups use
the exact config type, so `decode_reads` rejects a subclass of a decoder config
with `TypeError("unsupported decoder config")`, and it raises `TypeError` when the
codebook's encoding kind is not one the decoder declares. The decoders' numerical
behavior is unchanged. `Codebook(encoding=...)` accepts any registered encoding
config; `Dataset.load_codebook` gains keyword-only `encoding` and `layout`.

### Encoding table

W-304 makes the `two_base` pair-to-color table visible, configurable and recorded
({doc}`readout-contract`, "Encoding registry"). {py:class}`~starfinder.readout.EncodingConfig`
gains `pair_to_color`, the 16 ordered base pairs mapped to the colors `"1"` to `"4"`,
whose default is the active table of `src/matlab/EncodeBases.m`; `EncodingConfig()`
and every default result are unchanged, the positional constructor is unchanged,
and the module functions `bases_to_colors` and `colors_to_bases` keep the default
table. A table whose keys are not the 16 pairs, whose values are not `"1"` to `"4"`,
or in which the four pairs of one first base do not have four different colors
raises `ValueError`. Every encode and decode path follows the codebook's table,
including the `end_bases` shortcut of `filter_reads` when the codebook is given.
`EncodingSpec` gains the required keyword field `table`;
{py:meth}`~starfinder.readout.Codebook.encoding_table`,
{py:meth}`~starfinder.dataset.Dataset.encoding_table` and
{py:meth}`~starfinder.dataset.FOV.encoding_table` show the table with the channel of
each color. `repr(codebook)` ends with the encoding method and segment layout, and
`repr(dataset)` gains an `encoding:` line after `codebook:` when a codebook is
loaded. The Python-only YAML key `load_codebook.encoding.pair_to_color` sets the
table (the schema rejects it unless `backend: python`; MATLAB is unchanged).
`pre_qc.json` and `run.json` (`config.encoding`) record the encoding (`method`,
`reverse_bases`, and `pair_to_color` or `base_to_color`) under a new key beside
`layout`, with `FORMAT_VERSION` 2; loading a `pre_qc` checkpoint into a dataset
whose codebook has another encoding raises `ValueError` naming both, and a
checkpoint without the key loads as before.

### Segment layout

{py:class}`~starfinder.readout.BarcodeLayout` and
{py:class}`~starfinder.readout.Segment` describe how a barcode is cut into
separately read segments: lengths in bases, acquisition order and allowed
(first, last) end bases per segment, several pairs allowed. `Codebook.layout` holds
the effective layout: one segment by default, or the two segments that a legacy
`EncodingConfig.split_index` describes (the two are mutually exclusive). A codebook
entry whose segment ends are not declared raises `ValueError` naming the entry and
segment. `filter_reads(..., codebook=...)` checks the observed colors per segment
(`endpoint_valid_<segment>` and `endpoint_valid`); `FOV.filter_reads` passes the
dataset codebook. `ReadFilterConfig.end_bases` stays the one-segment shortcut and
cannot be combined with layout ends, and `exclude_invalid_endpoints=True` without
`end_bases` now raises when the reads are filtered without layout ends, instead of
when the config is built.

**Intentional change:** the workflow adapter converts the shared
`load_codebook.split_index` from MATLAB's one-based position; `WorkflowConfig.split_index`
holds the zero-based Python value and `WorkflowConfig.layout` the translated layout.
A configuration that worked around the defect by giving the Python value (for
example 4 for aging) must give the MATLAB value (5). On the 11-base barcode
`CAGTACTGCAT`, `[5]` now gives `242324242`; before, it gave `423242423`
({doc}`readout-baseline`). The adapter also accepts the MATLAB two-segment keys:
`reads_filtration.n_barcode_segments` and `reads_filtration.split_index`, when given,
must agree with the layout, and a list `end_base` gives the allowed segment ends
({doc}`workflow-configuration`). The benchmark pipeline profiles record the
zero-based split and pass the shared one-based value.

### Codebook entries

A codebook row is an entry, not a gene (D5). `Codebook.table` gains `entry_id`
(unique) before `gene_id`, which may now repeat; `color_sequence` and
`base_sequence` stay unique. A `gene,barcode` row has `entry_id` equal to the
barcode as written; a canonical file may state `entry_id` and otherwise gets the
color sequence. Repeated `entry_id`, `base_sequence` or `color_sequence` raise
naming both source rows. `n_entries`, `seq_to_entry` and `entry_to_seq` are new;
`n_genes` counts distinct genes and `genes` lists them once each; `gene_to_seq`
raises `ValueError` for a codebook with repeated genes.

**Intentional change:** two barcodes for one gene, which raised before, now load
as two entries of that gene, and the decoding table (and so `pre_qc`) gains
`entry_id`, the entry of the decoded color sequence, next to `gene_id`.

### Background measurements

Extraction now measures a local background and noise next to the sums, on by
default ({doc}`readout-contract`, "Extraction"; {doc}`readout-algorithms`,
"Background and noise"). {py:class}`~starfinder.readout.LocalBackgroundConfig`
(`inner_radius_zyx=(1, 3, 3)`, `outer_radius_zyx=(1, 6, 6)`, `min_voxels=16`,
provisional) is the new field `NeighborhoodSumConfig.background`; `background=None`
turns it off, and the inner box must contain the extraction box, so a radius
beyond `(1, 3, 3)` (for example `NeighborhoodSumConfig((2, 2, 2))`) now raises
`ValueError` unless it states a wider ring or `background=None`. For each candidate,
channel and extracted round, `IntensityExtractionResult.background` is the median and
`noise` 1.4826 × the median absolute deviation of the ring (the voxels of the outer box
outside the inner box, clipped to the image; 360 unclipped), in grey levels per voxel;
`background_voxels` counts the ring voxels, and `image_background` and `image_noise`
(per channel and round) are the median and 1.4826 × MAD of the whole image. Below
`min_voxels` ring voxels, and in rounds that readout mode `direct` does not read, the
background and noise are NaN. The ring is not masked for neighboring spots. The sums,
`valid` and `box_voxels` are unchanged. The `candidates` checkpoint gains the
`bg_<round>_<channel>`, `noise_<round>_<channel>`, `bgvox_<round>` and
`boxvox_<round>` columns (see {ref}`readout-checkpoints`). In the workflow
adapter the background is on; the Python-only `reads_extraction.background` is
`false` or a mapping of `LocalBackgroundConfig` fields, and without it the default
ring grows along the axes where `voxel_size` exceeds its inner box
({doc}`workflow-configuration`).

### Shared read-QC score

{py:func}`~starfinder.readout.score_reads` (`reference` is the codebook, or the
direct panel in readout mode `direct`) returns a
{py:class}`~starfinder.readout.ReadScoringResult`: the read table with the columns
`qc_score`, `qc_ambiguity_max`, `qc_signal_to_background`, `qc_rounds` and
`qc_reason` appended ({doc}`readout-contract`, "Shared read-QC score"). `qc_score`
(W-278 design D1) is the probability NLL of the assigned entry recomputed on
background-subtracted sums; lower ranks as more reliable. It is a ranking, not a
calibrated probability; it sets no cutoff and never changes `gene_id`, `entry_id`,
`call_status` or `call_type`. Reads without an assignment have NaN with
`no_assignment`, and reads with a round without background NaN with
`background_unavailable`; scoring intensities that have no background raises
`ValueError` naming extraction. {py:class}`~starfinder.readout.ReadScoreConfig`
(`method="bgcorr_probability"`, no parameter) is `PipelineConfig.scoring`, `None` by
default in the Python API; `FOV.run` scores after decoding or assignment and before
filtering, `FOV.score_reads` scores the stored reads, `FOV.scoring_result` holds the
result and `FOV.results` lists it as `scoring`. `filter_reads` accepts a
`ReadScoringResult` and keeps its score columns; `FOV.filter_reads` filters the scored
reads when scoring ran.

**Intentional change:** the workflow adapter scores whenever it decodes; the
Python-only `scoring: {run: false}` turns it off. The filtering table and the
exported reads of a workflow run therefore carry the score columns; the shared
spot CSV columns are unchanged.

### Optional deduplication

{py:func}`~starfinder.readout.deduplicate_reads` (`DeduplicationConfig`,
`ReadDeduplicationResult`; {doc}`readout-contract`, "Optional deduplication") groups
the reads of one amplicon that was detected in two channels. Candidates of one
detection round in different detection channels are linked when they lie within
`distance_voxels` (default 1.0, inclusive, voxel index space) and have identical WTA
observed sequences without `M` or `N`; a group keeps one original read as its
representative, and a group whose assigned reads name different entries keeps every
read (`conflicting_calls`). The read table gains `duplicate_group`, `duplicate_of`,
`is_representative` and `duplicate_reason`; nothing is removed or changed. It is off
by default: `PipelineConfig.deduplication` is `None`, and the workflow adapter runs
it only with the Python-only `deduplication: {run: true}` block. `FOV.run` runs it
after scoring and before filtering, `FOV.deduplicate_reads` runs it on the stored
reads, and `FOV.deduplication_result` and `FOV.results["deduplication"]` hold the
result. It raises `ValueError` in readout mode `direct`.

### Read filter

`ReadFilterConfig.score_bounds` accepts any declared score column: the
`DecodingSpec.score_columns` of every decoder (`own_channel_rank` and
`own_channel_fraction` of `direct` are new) and the shared score's `qc_score`,
`qc_ambiguity_max`, `qc_signal_to_background` and `qc_rounds`. The new
`ReadFilterConfig.exclude_duplicates` (default true) rejects the reads that
deduplication made duplicates, with reason `duplicate`; reads that were not
deduplicated are unaffected, so every earlier filter result is unchanged.
`filter_reads` also accepts a `ReadDeduplicationResult`. End bases are checked per
segment from the codebook's segment layout (see "Segment layout" above);
`end_bases` stays the one-segment shortcut. No score cutoff is set by default.

(readout-checkpoints)=
### Checkpoints

The checkpoint stages and files stay, and `FORMAT_VERSION` stays 2 (option C1 of
{doc}`readout-contract`, "Checkpoints and reruns"; {doc}`checkpoints`). `candidates`
adds the background columns after `valid_<round>` and `candidates.json` the top-level
keys `background_config`, `image_background` and `image_noise`; the saved
`signals.extraction_config` keeps only its earlier fields, so a reader at `141c093`
still loads the stage and drops the new columns. `pre_qc` holds the read table after
scoring and deduplication, before filtering, and `pre_qc.json` adds `scoring_config`,
`deduplication_config` (`null` when deduplication did not run), `layout` and
`stages_applied`. Loading `pre_qc` sets `FOV.scoring_result` and
`FOV.deduplication_result` as well as `decoding_result`. A checkpoint written before these
keys loads with `background=None` and no score; `load_checkpoint("candidates")` then
`run` with decoding, scoring and filtering reruns the readout without images when the
background was stored, and raises `ValueError` naming extraction when it was not.

### Readout evaluation metrics

`starfinder.evaluation.readout` adds `ranking_quality(score, correct, *,
orientation, retention=(0.5, 0.8, 0.9, 1.0))`: the AUROC of a score with its
Hanley–McNeil standard error and the error at fixed retention, and
`evaluate_deduplication(groups, source, *, pairs)`: missed duplicates, false
merges and their rates over a stated pair population. Both report an undefined
value with a reason when a class is empty. They rank and count; they set no
cutoff.

### Read diagnostics

`starfinder.readout` adds the three diagnostics of {doc}`readout-contract`
("Diagnostics"), which read retained results only.
{py:func}`~starfinder.readout.inspect_read` returns one read's sums, background,
background-subtracted sums, channel probabilities, noise, observed and assigned
colors and per-segment bases, one row per round and channel, and
{py:func}`~starfinder.readout.plot_read` draws it.
{py:func}`~starfinder.readout.summarize_reads` returns the population summary (counts
by status, reason and call type, per gene and per entry, `qc_score` quantiles per call
type, deduplication and filtering counts, per round and channel medians, valid and
background-unavailable counts, cross-channel pairs).
{py:func}`~starfinder.readout.explain_read` returns a read's ordered decisions with
their values and limits. They accept a read result or `FOV.results`. `FOV.run` never
calls `inspect_read`, `plot_read` or `explain_read`; when it scores or deduplicates,
`run.json` records the summary under `counts["summary"]`, and
`FOV.save_diagnostics` now writes it as `summary` beside the filtering counts (W-563 later
removed `save_diagnostics` and records the summary whenever reads were decoded).

### Segmentation label contract and mask import

The new module `starfinder.segmentation` ({doc}`segmentation-contract`) starts with the
label contract that every later segmentation result follows.
{py:class}`~starfinder.segmentation.SegmentationResult` holds a `uint32` ZYX label
image (a plane is 1×Y×X) on a {py:class}`~starfinder.segmentation.ReferenceGrid`, with
its target, geometry, label namespace and run record.
{py:func}`~starfinder.segmentation.to_label_dtype` converts int32, uint16 or other integer
labels to `uint32` and raises on a negative value or one above 2³²−1, where the legacy
`stardist_segmentation.py` cast to `uint16` and wrapped label 65,536 to 0.
`FOV.reference_grid()` returns the grid of the resident reference round after
`FOV.run`, and {py:func}`~starfinder.segmentation.reference_grid_from_file` reads one
from a TIFF such as `images/ref_merged/{fovID}.tif`
(since W-564 Python's reference image is `fovs/{fovID}/reference.tif` by default; see "The FOV folder, the reference image and channel folders by label").

Masks made elsewhere, such as CellProfiler outputs, the legacy
`images/stardist_segmentation` files or the culture references, enter through
{py:func}`~starfinder.segmentation.import_labels` (with
{py:class}`~starfinder.segmentation.LabelImportConfig` for a plan run) instead of a
plain `imread`. Big-endian files are read with native values, float and boolean masks
are rejected, and the shape and any stored metadata are checked against the grid.
{py:func}`~starfinder.segmentation.resample_labels` replaces the legacy round trip
`rescale(labels, [1, 2, 2], order=0)`, which turned a 61×63 grid into 60×64, by an
exact map onto the target shape.
{py:func}`~starfinder.segmentation.extend_labels_through_z` with
{py:class}`~starfinder.segmentation.ZExtensionConfig` is the per-FOV Python form of
`create_3d_segmentation.m`: sizes are in µm instead of pixels, a numeric threshold is on
the [0, 1] scale of the stain's dtype range, and `Cyto = Cell − Nuclei` is not
reproduced. MATLAB was not run, so there is no parity with the example.

### Segmentation input functions

The morphology preprocessing of the workflow scripts is now a set of plain functions in
`starfinder.segmentation` that return their result and a record mapping (the config,
the values reached and the SHA-256 of every input and of the output) and never change
their inputs. {py:func}`~starfinder.segmentation.composite_nuclei_amplicon` with
{py:class}`~starfinder.segmentation.CompositeConfig` is the composite of
`create_nuclei_amplicon_overlay.py`, and
{py:func}`~starfinder.segmentation.enhance_with_flamingo` with
{py:class}`~starfinder.segmentation.FlamingoEnhancementConfig` the enhancement of
`enhance_dapi_with_flamingo.py`; both give the scripts' output bit for bit (the W-307
golden digests), including on constant and all-zero images. Two edge cases change: two
images of different shapes raise `IncompatibleGeometryError` instead of a broadcasting
error, and a plane is accepted as 1×Y×X where the scripts raised on YX inputs. The
composite's `maximum_projection` is no longer part of the composite; it is the z
maximum of the result (`project_image` with `ProjectionConfig()`), which gives the same
image. The inputs come from the reference frame: a morphology round's DAPI after
`FOV.register` and the reference round's channel maximum, the image
`FOV.save_reference_image` writes. The workflow scripts call these functions (see
"Segmentation and assignment workflow rules" below).

{py:func}`~starfinder.segmentation.normalize_percentile_unclipped` is csbdeep's `normalize` as
`stardist_segmentation.py` calls it (percentiles 1 and 99.8, float32, unclipped), with
its values recorded. {py:func}`~starfinder.segmentation.rescale_input` performs the
script's `rescale(image, [1, .5, .5])` shrink for any factors and also returns
metadata whose spacing is divided by the factors and whose `frame_id` records the
rescale; {py:func}`~starfinder.segmentation.resample_labels` maps labels detected on the
shrunk image back onto the exact input grid. None of these is registered in
`PREPROCESSING_METHODS`.

### The segment entry and the seeded watershed

Segmentation is its own entry, separate from `FOV.run` (decision D1 of
{doc}`segmentation-contract`): `PipelineConfig` and `ExecutionConfig` gain no field.
{py:func}`~starfinder.segmentation.segment` runs one method of
{py:data}`~starfinder.segmentation.SEGMENTATION_METHODS` on a
{py:class}`~starfinder.segmentation.SegmentationInput` (a ZYXC image on its grid, with one
role per channel) behind the stage checks, and `FOV.segment(plan)` runs a
{py:class}`~starfinder.segmentation.SegmentationPlan` of named runs on the FOV's resident
reference-frame images, keeping the results in `FOV.segmentation_results`. With
`checkpoints` it also saves each run (see "Saved segmentation runs and assignments" below).

`segment` and `FOV.segment` take a `device` keyword, `"cpu"` (default) or `"cuda"`, which
each method accepts only for its own devices. `ExecutionConfig.device` and the §2.7
`device="cpu"` rule of `FOV.run` do not change. The legacy scripts have no device setting.

The first registered method is `seeded_watershed`
({py:class}`~starfinder.segmentation.SeededWatershedConfig`), the nucleus-seeded,
stain-guided watershed of the W-306 prototype. It grows each cell from a nucleus of an
earlier run on a stain (amplicon, cytoplasm, membrane or composite), so cell k carries
nucleus k's value; the legacy workflow has no such step (`reads_assignment.py` expands the
nuclei instead). There is no foreground gate: an image without objects gives an empty
label image with outcome `empty` instead of an error.

### The assign entry

Assignment is the third call per FOV, after `FOV.run` and `FOV.segment` (decision D1 of
{doc}`assignment-contract`): the new module `starfinder.assignment` holds
{py:func}`~starfinder.assignment.assign_molecules`, which places the molecules of a
{py:class}`~starfinder.assignment.MoleculeTable` (from
{py:func}`~starfinder.assignment.molecule_table` or a legacy goodSpots CSV through
{py:func}`~starfinder.assignment.import_legacy_molecules_csv`) in the territories of a cell
{py:class}`~starfinder.segmentation.SegmentationResult` and returns an
{py:class}`~starfinder.assignment.AssignmentResult` with the molecule, cell, count and
nucleus tables. `FOV.assign_molecules(config, cells=…, nuclei=…)` runs it on the FOV's results and
keeps the result in `FOV.assignment_results`; `PipelineConfig` gains no field, and with
`checkpoints` it also saves the assignment (see the next section). The workflow's `reads_assignment`
rule calls it too (see "Segmentation and assignment workflow rules" below).

Compared with the per-FOV steps of `reads_assignment.py`, the package changes these on
purpose:

* every molecule keeps a row and one status: `assigned`, `unassigned`, `excluded_cell` or
  `outside_grid`. A position is sampled at `floor(c + 0.5)` per axis, so float coordinates
  (`8.0`, as `export_spots` writes them), which raised `IndexError`, are read; one-based 0
  no longer reads the far edge and one beyond the grid no longer raises: both are
  `outside_grid`;
* the cells are the territories of the label image, also when no molecule lands in them
  (the script dropped every cell of such a FOV);
* a gene outside the gene list raises `ValueError` naming it instead of being dropped from
  the counts silently;
* the expansion is applied once, by assign, with
  {py:func}`~starfinder.segmentation.expand_labels`
  ({py:class}`~starfinder.segmentation.ExpandLabelsConfig`; `planar` and `pixel` give the
  script's `expand_labels` per plane), and both the original and the expanded territories
  are kept. A cell run whose record lists `expand_labels` (an expansion in segmentation)
  is refused, because its original mask was not kept. On a calibrated grid the distance
  is in µm; a pixel distance needs `AssignmentConfig.legacy_pixel_expansion=True`;
* with nuclei, each nucleus is matched to the cell holding more than half of it, every
  doubtful correspondence is flagged, nuclear and cytoplasmic counts exist only where it
  allows, and cells without a matched nucleus are excluded by default with the reason
  `no_matched_nucleus`, their molecules `excluded_cell`. None of this existed before;
* cell sizes and centroids are computed on the original territories, with the expanded
  ones beside them (the script's `volume` and `fov_*` are the expanded size and the
  truncated expanded centroid).

A µm expansion compares physical distances in floating point, so a distance that is an
exact multiple of the spacing (0.3 µm at 0.1 µm) may leave out the outermost ring of
voxels; the legacy pixel distance has no such rounding.

### Saved segmentation runs and assignments

`FOV.segment(plan, checkpoints=CheckpointConfig(…))` and `FOV.assign_molecules(…, checkpoints=…)`
save their results beside the `FOV.run` checkpoints, in two new folders of the per-FOV
checkpoint directory ({doc}`checkpoints`, "Segmentation runs" and "Assignments"):
`segmentation/<run>/` holds `labels.tif` (ZYX `uint32`, zlib, with the grid's
`ImageMetadata`), `input.ome.tif` (the segmentation input; not for an imported mask) and
`segmentation.json` (the run record); `assignment/<name>/` holds the `molecules`, `cells`
(kept and excluded), `counts` and `nuclei` tables as CSV or Parquet, `assignment.json`,
and the label images the assignment used that are not already saved under their run.
`FOV.load_segmentation(name)` and `FOV.load_assignment(name)` read them back and check
every recorded SHA-256. The workflow rules keep writing
`images/stardist_segmentation/{fovID}.tif`, `expr/{fovID}/raw.h5ad` and
`expr/{fovID}/reads_assignment.csv`; nothing reads the new folders yet.

Nothing changes for `FOV.run`: its stages (`registered`, `candidates`, `pre_qc`), its
checkpoint `FORMAT_VERSION` 2 and `run.json` (`format_version` 1) stay as they are (note (W-565):
now 3 and 2, see "run.json version 2"), and
`FOV.assign_molecules` never rewrites a file of `FOV.run` or of segmentation (a saved label image is
linked by its relative path). The new records carry their own `format_version` 1.
`checkpoints` other than `None` or a `CheckpointConfig` now raises `TypeError` in
`FOV.segment` and `FOV.assign_molecules`, as in `FOV.run`, where these two calls raised
`ValueError` for any value other than `None` before. The checkpoint tables also accept
`UInt32` columns (the cell and nucleus identifiers).

### Segmentation and assignment workflow rules

The scripts of `enhance_dapi_with_flamingo`, `create_nuclei_amplicon_overlay`,
`stardist_segmentation` and `reads_assignment` are now adapter calls into the package
(`dataset/workflow.py`), like `nuclei_registration.py`. The rules keep their names,
inputs, outputs and file names, and the composite and the Flamingo enhancement keep the
scripts' output bit for bit.

**New requirement of MATLAB-backend runs.** `rules/segmentation.smk` and
`rules/reads-assignment.smk` are included for both backends, so a run with
`backend: matlab` that enables these rules now needs the Starfinder package in the
environment that runs Snakemake: with the `stardist` extra for `stardist_segmentation`
(StarDist, CSBDeep and TensorFlow; Python 3.11 to 3.13), and the `anndata` extra for
`reads_assignment`. `stardist_segmentation` no longer runs in the conda environment
`{envs_path}/stardist`, and no rule reads `envs_path`; the key stays in the schema until
§2.13. The legacy keys are translated the same way under both backends, and the
Python-only keys below are rejected without `backend: python`, so a MATLAB-backend run
has the legacy target and the CPU.

New configuration: the Python-only top-level `segmentation` and `assignment` blocks
({doc}`workflow-configuration`, "Segmentation and assignment"), which the §2.13 rule
runs with `FOV.segment` and `FOV.assign_molecules` (the legacy rules raise when a block is
present), and the Python-only `target` and `device` keys of
`rules.stardist_segmentation.parameters`. The `stardist` and `cellpose` extras supply
the learned methods.

Intentional changes of the outputs of `stardist_segmentation`:

* `images/stardist_segmentation/{fovID}.tif` holds `uint32` labels instead of `uint16`,
  so label 65,536 no longer wraps to 0; the run record is written beside it as
  `{fovID}.json`.
* `rescale: true` is StarDist's `scale` 0.5 in Y and X, and the labels come back on the
  input grid; the legacy shrink, prediction and nearest-neighbour restore turned an odd
  grid such as 61×63 into 60×64.
* There is no foreground gate: an image without objects gives an empty label image
  (outcome `empty`) instead of an error.
* Thresholds equal to the model's `thresholds.json` are recorded as its stored
  thresholds; a known model in the weights cache is checked against `KNOWN_MODELS`.
* Unknown parameter keys raise; `rotate_nuclei` raises unless exactly one DAPI file
  matches, naming the matches.

Intentional changes of the outputs of `reads_assignment`:

* Cells are kept when no molecule lands in them: they have zero rows in `X`, where the
  script wrote a FOV without any cell.
* Float goodSpots coordinates (`8.0`, as `export_spots` writes them) are accepted.
* Molecules outside the label grid are `outside_grid` instead of reading the far edge
  (one-based 0) or raising `IndexError` (beyond the grid); they are counted in the
  record, and `reads_assignment.csv` keeps each of them as a row with status
  `outside_grid`, `seg_label` 0 and no cell, after the rows the legacy overlap filter
  keeps (the filter itself is unchanged for the molecules on the grid).
* A goodSpots gene outside the codebook raises `ValueError` naming it before
  assignment, where the script silently left it out of the counts; `documents/genes.csv`
  must list the codebook's genes.
  (W-567 no longer reads `documents/genes.csv`: the genes and their order are the codebook's.)
* A label file expanded by `stardist_segmentation` (`expand_labels: true` there) raises,
  naming the key: give the distance as `reads_assignment.parameters.dilation_distance`
  with `expand_labels: true`, so assign expands once and keeps both masks.
* `raw.h5ad` gains the obs columns `size_voxels`, `expanded_size_voxels`,
  `size_physical`, `centroid_z/y/x`, `n_molecules`, `n_nuclei`, `correspondence`,
  `correspondence_flags` and `compartments`, the record in `uns["assignment"]` (JSON
  text), and, with nuclei, the `nucleus` and `cytoplasm` layers (NaN where compartments
  are not available). `reads_assignment.csv` gains `spot_id`, `assignment_status`,
  `cell_id`, `in_expansion`, `original_cell_id`, `nucleus_id` and `compartment`.
* `assignment.png` (`plot_assignment`, the Z-maximum view), with
  `assignment_single_layer.png` (the middle plane) for a ZYX label image, and a new
  `log.txt` with the assignment ratio replace the four diagnostic PNGs and the coverage
  log ("Assignment diagnostics" below).

The legacy `obs` columns keep their meaning, computed on the territories assign samples,
and the tile configuration, global coordinates and overlap filter are applied outside the
package as before (§2.10).

### Default correspondence tolerance

The default `CorrespondenceConfig.outside_tolerance` is 0.1 instead of 0.0 (option C2 of
{doc}`assignment-contract`, chosen on 2026-10-05 in the W-321 review); `match_fraction`
stays 0.5. A matched nucleus is flagged `outside`, and its cell's compartments withheld
with `nucleus_outside_cell`, only when its share outside its cell is strictly greater than
0.1, so a nucleus with up to a tenth of its voxels outside its cell no longer withholds the
nuclear and cytoplasmic counts of its cell. A part of a nucleus inside another cell still
raises `foreign_nucleus` on that cell. The default applies to `assign_molecules`,
`match_nuclei`, `FOV.assign_molecules` and the `assignment` block when `correspondence` or its
`outside_tolerance` is omitted, and the record's `config.correspondence` shows it. The
value is provisional, from one culture crop of W-320 (largest outside share of a matched
nucleus 0.073). To keep exact containment, pass
`CorrespondenceConfig(outside_tolerance=0.0)`, or `correspondence: {outside_tolerance: 0.0}`
in the block. The golden digests and the outputs of `reads_assignment`, which has no
nuclei, do not change.

### Assignment diagnostics

`plot_assignment` draws one row of four panels instead of the territory panel and three
histograms (W-321 review, 2026-10-05; the centre colours and the legend entry `excluded`
from the checkpoint after it, the same day): the stain in grey scale with green territory
outlines and a dot at each cell's centroid, coloured by the cell's status (`kept` blue,
excluded orange; legend entries `kept` and `excluded`, the latter only when a cell is
excluded); the outlines are lines 0.8 points wide and the centre dots 30 points² with a
thin black edge, both of a fixed size, so they stay visible when a large image is shown
small (2026-10-06); the same with the molecules (`assigned` blue, `unassigned` red, `excluded_cell`
orange with the legend entry `excluded`, only when present; `outside_grid` not drawn);
voxels per cell; molecules per cell. The "nuclei per cell" histogram is gone. A
new `view` argument selects the Z-maximum view (`"z_max"`) or one plane
(`"single_layer"`, default plane `Z // 2`); `z` alone still selects that plane, and
`view="single_layer"` on a plane or Z=1 result, or `z` with `view="z_max"`, raises
`ValueError`. The `reads_assignment` rule writes `assignment.png` for every FOV and
`assignment_single_layer.png` for a ZYX label image, and `log.txt` gains a second line
with the assignment ratio (assigned over all molecules, both counts; `none` without a
molecule). `summarize_assignment(result)["exclusion"]["before"]` gains `nucleus` and
`cytoplasm`, equal to those after the exclusion, as its docstring promised. No table,
count or digest changes: `raw.h5ad` and `reads_assignment.csv` are as before.

### Channel name and wavelength per round

{py:class}`~starfinder.dataset.ChannelInfo` (`channel`, `name`, `wavelength`; MATLAB's
`channel_order_dict` fields) is new (W-336; {doc}`coordination`, "Channels").
`Dataset.channel_order` and each `Dataset.other_channel_order[round]` accept patterns as
before, `ChannelInfo` values or mappings, and still hold the patterns after
construction, so `Dataset.channel_labels(round)` returns what it returned.
`Dataset.reference_stains` lists the stain files of the reference round that are not
sequencing colours (image name `reference_stain`, which a configured round may no longer
take), and `Dataset.channel_info(round)` returns the `ChannelInfo` tuple of a round.
`Dataset.channel_index(round, key)` is the one channel lookup: an index, an exact
pattern, else a name that occurs once. `RegistrationSignalConfig` channel labels and
`InputChannel.channel` and `prepare_channel` may therefore name a channel by its
content, for example `"DAPI"`; an unknown key now raises the same `ValueError` from all
three, naming the round and its channels. Repeated patterns raise as before; a
non-string or empty pattern in `channel_order` (accepted before) and a non-positive or
non-finite wavelength now raise. `dataclasses.replace(dataset, ...)` reads the channels
the call passes as given (patterns carry no name or wavelength), except that the
dataset's own tuple of a round (`channel_order`, or `other_channel_order[round]` inside
any mapping), which `replace` passes for an unchanged field, keeps that round's names
and wavelengths; it lost them before (W-350).

In the workflow translation the shared keys keep their names: `seq_channel_order` also
accepts MATLAB's list of `wavelength`/`channel`/`name` objects; every rule names the
other rounds by `additional_round[i].round_name` (the sequencing rules raised
`TypeError` for a non-empty list before) and takes their `channel_order`; `dapi_round`
with `ref_channel` gives `reference_stains`, and a `dapi_round` other than `ref_round`
now raises `ValueError`. The schema declares both forms of `seq_channel_order` and
`channel_order` under `additional_round` items. `run.json` gains `channels` and each
`input.channels` entry of a segmentation record gains `name` and `wavelength`; both
are additive. No image, table or digest changes.

### Sequencing rounds and other rounds handled separately

Two breaking changes (W-337), following the MATLAB workflow, where every
`STARMapDataset` method works on `layers.seq` and other rounds are prepared by their own
rules:

* **`FOV.run` no longer loads or registers other rounds.** It loads, rotates,
  preprocesses, registers and checkpoints the reference round and the sequencing rounds
  only; a configured other round is neither loaded nor changed (before, batch mode loaded
  it with the sequencing load config, which raised for a round with its own channels, and
  registered it with the sequencing recipe). A `SpotFindingPlan` whose rounds name an
  other round now raises `ValueError` in `run`. The `registered` checkpoint, also from
  `save_checkpoint`, holds the reference and sequencing rounds only.
* **`FOV.register` no longer registers other rounds by default.** Its default rounds are
  the sequencing rounds other than the reference (before: every moving round, other rounds
  included). An other round named in `rounds` raised `ValueError`; since W-560 it takes
  the other-round route of `FOV.register` ("Mechanical names and ownership").

Prepare other rounds and the reference stain with the new entry beside `run`:

```python
from starfinder.dataset import CheckpointConfig, MorphologyConfig

fov.run(pipeline)                                  # reference and sequencing rounds
fov.prepare_morphology(MorphologyConfig(rotation_degrees=pipeline.rotation_degrees),
                       checkpoints=CheckpointConfig())
fov.segment(plan)                                  # reads "reference_stain" and the other rounds
```

`FOV.load_images(..., rotation_degrees=…)` rotates as `run` does, and a round is now
rotated once: a second rotation of a round (by `rotate`, `load_images` or `run`) raises
`ValueError` where it rotated again before. `FOV.register` on other rounds gains `checkpoints`,
and `FOV.load_registered_round` reads a saved prepared image ({doc}`checkpoints`,
"Prepared morphology images"). The `registered` header gains `channels` (additive;
`FORMAT_VERSION` stays 2). Datasets without other rounds give the same images, tables
and digests as before.

### Segmenting from saved reference-frame images

Additions (W-338), following the MATLAB segmentation rules, which read saved
reference-frame images and never load a sequencing round. Nothing that worked before
changes its result:

* `FOV.load_reference_image()` restores `images/ref_merged/{fovID}.tif` (since W-564 the file
  of the dataset's `reference_image` or `reference_image_import`; see "The FOV folder, the reference image and channel folders by label"), written by
  `FOV.save_reference_image`. Without the reference round, `FOV.reference_grid()`
  returns that file's grid (source `file:<path>`) and `InputChannel(reference_merged=True)`
  (also the amplicon of a composite) reads it; the input channel's record gains
  `reference_image` (path and SHA-256), an additive key. A process can therefore segment
  with `load_reference_image`, `load_registered_round` and `segment` only:

  ```python
  fov = dataset.fov(fov_id).load_reference_image()
  for name in ("reference_stain", "morph"):
      fov.load_registered_round(name, checkpoints=checkpoints)
  fov.segment(plan, checkpoints=checkpoints)        # no sequencing round resident
  ```

* The grid rule is stated in {doc}`assignment-contract` ("The grid rule") and implemented
  once: labels and molecules are on one grid when shape and `ImageMetadata` are equal;
  `source` and `sha256` are recorded, not compared. This is the rule assign already
  applied. `FOV.assign_molecules` refuses the Z=1 grid of a projected saved reference image without
  the reference round, a case that did not exist before.
* The Python `nuclei_registration` rule calls `FOV.prepare_morphology`; its declared files
  are byte-identical to before. A new Python-only key,
  `rules.nuclei_registration.parameters.checkpoints` (`CheckpointConfig` fields, `{}`
  for the defaults), also writes the saved form under `checkpoints/{fovID}/other_rounds/`
  (since W-564 `fovs/{fovID}/other_rounds/`; see "The FOV folder, the reference image and channel folders by label").
  `save_processing_log("nr")` lists only registered rounds, so the reference stain, which
  has no registration, is not in `log/{fovID}_nr.txt` (W-563 later replaced both files with
  `fovs/{fovID}/nuclei_registration.json`, which leaves the reference stain out the same way).

### Multi-FOV synthetic sample scene

New, opt-in names in `starfinder.synthetic`; nothing existing changes. No
generator function, config, preset, registry, version or stream descriptor is
modified, so every existing preset, digest and saved file stays as it was.

| Need | New name |
| --- | --- |
| One sample scene acquired as a grid of FOVs at true, fractional offsets, with placement, cell and per-FOV truth | `generate_sample_scene(codebook, config=SampleSceneConfig(), on_fov=None)` returns `SampleScene` |
| The grid, jitter, cellular scene and nuclear stain | `AcquisitionConfig`, `CellsConfig`, `StainConfig`, composed in `SampleSceneConfig` with an `AmpliconSceneConfig` template |
| Files: `sample.json`, truth tables, label TIFFs, per-FOV amplicon scenes, stains, labels and legacy molecule CSVs | `save_sample_scene(scene, directory)` |
| Presets, separate from `SCENE_PRESETS` | `sample_scene_preset(name)`, `SAMPLE_PRESETS` (`sample-tiny-v1`, `sample-small-v1`), `SAMPLE_PRESET_VERSION` (`sample-presets-v1`) |
| Version of the sample-level truth files | `SAMPLE_TRUTH_VERSION` (`"1"`) |

Every FOV is rendered by `generate_amplicon_scene` and records generator version 6;
the sample-level draws use new `sample.*` and `stain.noise.*` stream components.
The specification is {doc}`synthetic-sample-scene`.

### Stitching: tile positions, the placement record and the stitch entry

New names in a new module, `starfinder.stitching`, and one new method,
`Dataset.stitch`; nothing existing changes. The four legacy stitching rules,
`workflow/scripts/` and `tile_config_{sample}.csv` stay as they are, and no
workflow key is added yet.

| Need | New name |
| --- | --- |
| Nominal FOV positions from a Leica stage file, a regular grid or a supplied table | `positions_from_leica(path, stage_axes=StageAxisMapping(...), fov_ids=..., snap_to_grid=GridSnap(...) or None)`, `positions_from_grid(...)`, `positions_from_table(...)` return `TilePositionTable` |
| Pairs of FOVs whose nominal boxes overlap | `neighbor_pairs(table, grids, tolerance_um=0.0)` returns `NeighborPair` |
| One placement record for every backend | `PlacementResult` (`fovs`, `links`, `record`); `placed_metadata(result, fov_id)` |
| Place a sample with a registered method | `stitch(inputs, table, config=...)` with `StitchingInput`; `STITCHING_METHODS` (the first method, `multiview_stitcher`, is below), `StitchingSpec`, `MethodPlacement` |
| A known placement, such as a synthetic scene's truth | `placement_from_table(table, grids, sample=..., origins=None)` |
| Review before sample assembly | `ReviewPolicy`, `review_placement`, `approve_placement`, `PlacementReviewRequiredError` |
| Save and read `stitching/<sample>/` | `save_placement(result, directory)`, `load_placement(directory)` |
| Place a dataset's sample from its saved per-FOV images | `Dataset.stitch(sample, fov_ids, table=..., config=..., input="reference_merged", output_dir=None)` |
| A stitching method's missing extra | `StitchingBackendUnavailableError` |

Origins are float64 in µm in the sample frame `sample:<sample>`, never truncated to
pixels as `tile_config_{sample}.csv` is; the gauge is zero mean correction over the
`image_supported` FOVs. The specification is {doc}`stitching-contract` and
{doc}`stitching-algorithms`.

### Stitching: the BigStitcher macro and the import of `dataset.xml`

New names in `starfinder.stitching`; nothing existing changes. The legacy rules
`create_BigStitcher_macro`, `run_BigStitcher_macro` and `create_tile_config` and their
scripts stay as they are, and Fiji is still run only by the workflow.

| Legacy step | New name |
| --- | --- |
| `create_BigStitcher_macro_1.py` writes the macro from `grid.csv`; `stitching_preparation.py` copies the tiles and blanks | `export_bigstitcher(table, inputs, config=BigStitcherConfig(...), tile_dir=..., save_dir=...)` writes `tile_{j}.tif` and returns the macro text and the view map; `BigStitcherConfig` holds today's parameters as defaults, and `pad_missing=True` (off by default) turns an acquired FOV without an image into a placeholder view |
| `create_tile_config.py` reads `dataset.xml` (lines 19-58) | `import_bigstitcher(dataset_xml, table=..., view_map=..., grids=..., config=..., fallback=None)` returns a `PlacementResult` |
| `create_tile_config.py` casts, shifts and draws the rectangles (lines 43-130) | `legacy_tile_config_view(result, img_col=..., img_row=...)`, whose `to_csv()` is today's `tile_config_{sample}.csv` |
| `dataset.xml` beside the tiles | `save_placement` of an import also writes `view_map.csv` and records the SHA-256 of the macro and of `dataset.xml` |

The macro text equals today's except for the two folders, which are arguments, and the
voxel size, which is the reference grids' spacing instead of `voxel_size_xy` and
`voxel_size_z`. The import keeps BigStitcher's raw translations unrounded
(`method_translation_*`, world units) and gives origins in µm, z divided by the
calibration's z scale; a view without a link into the reference component is `failed`,
or `metadata_only` at its nominal position with `fallback="nominal"`, and review is then
required. The specification is {doc}`stitching-contract` ("BigStitcher") and
{doc}`stitching-algorithms` ("Import arithmetic").

### Sample assembly and the shared AnnData writer

New names in a new module, `starfinder.assembly`, and one new function in
`starfinder.assignment`; nothing existing changes. The legacy overlap filter of the
workflow adapter, `create_sample_h5ad.py`, `create_sample_reads_assignment.py` and
`tile_config_{sample}.csv` stay as they are, and no workflow key is added yet: the
`assembly` block and `Dataset.export` belong to the export entry.

| Legacy step | New name |
| --- | --- |
| `reads_assignment` adds integer `global_x/y/z` from `tile_config` offsets | `assemble(placement, inputs, config=AssemblyConfig())` with one `AssemblyInput(fov_id, grid, assignment=… or molecules=…)` per FOV returns an `AssemblyResult` whose `sample_*` columns are float64 µm in the sample frame |
| The overlap filter keeps cells and background molecules inside the `start/end_*_norm` rectangles | Ownership from the placement, `ownership_map(placement, grids, config=…, candidates=None)` returning `OwnershipMap`: nearest footprint centre among the owning FOVs, ties by table order; placeholders own no point |
| Dropped cells and molecules are not written | Every cell and molecule keeps a row with `assembly_status`, `assembly_reason` and `assembly_flags` (`ASSEMBLY_STATUSES`, `CELL_ASSEMBLY_REASONS`, `MOLECULE_ASSEMBLY_REASONS`, `CELL_ASSEMBLY_FLAGS`, `MOLECULE_ASSEMBLY_FLAGS`) |
| `create_sample_h5ad.py` concatenates the per-FOV `raw.h5ad` | `write_sample_h5ad(result, path, annotation=None)` writes `sample.h5ad` from the sample tables, one row per counted cell |
| `create_sample_reads_assignment.py` concatenates the per-FOV CSVs | `save_assembly(result, folder, table_format="csv")` writes `tables/fovs`, `tables/cells`, `tables/molecules`, `tables/counts`, `sample.h5ad` and the `assembly` section of `export.json`; `load_assembly(folder)` reads it back |
| The adapter's `_write_raw_h5ad` builds the AnnData itself | `write_h5ad(path, obs, keys, counts, genes, layers=…, uns=…, var=None, obsm=None)` in `starfinder.assignment`, which the adapter now calls; `raw.h5ad` has the same rows, columns, values and `uns` |

Once a configuration adopts the `assembly` block (a later issue), its per-FOV `raw.h5ad`
and `reads_assignment.csv` are not box-filtered and have no `global_*` columns, the
sample outputs move to `export/<sample>/`, cells and molecules are retained by the
ownership regions of the placement instead of the `tile_config` rectangles, and every
excluded observation is kept with its reason. The specification is
{doc}`assembly-contract` and {doc}`assembly-algorithms`.

### Stitching: the `multiview_stitcher` placement method

One new name in `starfinder.stitching`, `MultiviewStitcherConfig`, the first method of
`STITCHING_METHODS`; nothing existing changes. BigStitcher stays the default, the legacy
rules stay as they are, and no workflow key is added yet. The method needs the optional
extra `starfinder[multiview-stitcher]` (multiview-stitcher 0.1.62, Python 3.11 to 3.13);
without it, `stitch` raises `StitchingBackendUnavailableError` naming the extra, and
importing `starfinder.stitching` never imports the library.

| Need | New name |
| --- | --- |
| Place a sample in process, without Fiji | `stitch(inputs, table, config=MultiviewStitcherConfig())` returns a `PlacementResult` |
| The contract's starting configuration | the defaults: `binning_zyx=(4, 4, 4)`, `pairwise="phase_correlation"`, `upsample_factor=None` (the library's), `quality_threshold=0.7`, `drop_nan_quality=True`, `pruning="alternating_pattern"`, `overlap_tolerance_um=0.0`, `groupwise="global_optimization"`, `abs_tol_um`, `rel_tol` and `max_iter` `None` (the library's) |

A Z=1 tile is registered as a plane, so its z translation is exactly 0. Links follow rule
L1: a link below 0.7 or with NaN quality (which the library itself keeps) is not kept. A
neighbour pair the library does not register is `pruned`; a link its optimizer removes is
`optimizer_removed` and requires review. When no link is left, every FOV is `failed`
(the library's `NotEnoughOverlapError`), and the result is still returned and can be
saved. The specification is {doc}`stitching-contract` ("Method registry") and
{doc}`stitching-algorithms` ("`multiview_stitcher`").

### Sample outputs: the export entry and the fused image

New names in a new module, `starfinder.export`, one new method, `Dataset.export`,
and one new folder, `<OUTPUT>/export/<sample>/`; nothing existing changes.
`Dataset.stitch` and `stitching/<sample>/` stay as they are, the legacy rules and
`images/fused/{sample}/` (BigStitcher's inputs and `dataset.xml`) are untouched, and no
workflow key or rule is added yet: the `export` block and the `export_sample` rule come
with the workflow issue. The label export (`fuse_labels`) and the SpatialData export
(`export_spatialdata`) are described in the next sections.

| Need | New name |
| --- | --- |
| One sample-level export: the sample tables, `sample.h5ad` and the fused image | `Dataset.export(sample, placement=…, assignment="default", config=ExportConfig(), output_dir=<OUTPUT>/export/<sample>/)` returns an `ExportRecord`; it calls `assemble` (saved with `save_assembly`) and `fuse_images` in order, each turned off by `ExportConfig(assemble=False)` or `ExportConfig(fuse=False)` |
| The record of the folder | `export.json`: `parts` (`written`, `reused`, `off`, `empty`, `not_run`, `failed`), `complete`, one section per part (`assembly`, `fusion`, `labels`, `spatialdata`) with the SHA-256 of every file; `check_export(folder)` checks the folder against it |
| A part that cannot run | `ExportIncompleteWarning`; the part is `not_run` with `extra_missing` (`multiview-stitcher` not installed) or `source_image_missing`, the earlier files are written and `complete` is false |
| The fused image of a sample, from either backend's placement | `fuse_images(placement, inputs, assembly=None, config=FusionConfig(), output_dir)` with one `FusionInput(fov_id, grid, images)` per included FOV, returning a `FusionRecord`; multiview-stitcher's `fusion.fuse` with `weighted_average_fusion` at order 1 |
| Today's `MAX_*_05.tif` overview | `<image>_projection.ome.tif`: the 2D maximum projection at `scale_yx` 0.5 (the default), OME-TIFF (BigTIFF, 512 × 512 zlib tiles, calibration and origin) in the input dtype |
| A 3D fused stack, such as LN sample1's | `FusionConfig(dimensionality="stack")` (the stack alone) or `"both"`: `<image>_stack.ome.tif`, Y and X reduced by `scale_yx`, Z as acquired; the middle plane `n_z // 2` reads alone with `tifffile.imread(path, key=i)` |
| Other options | `FusionConfig(images=(…), projection="sum", scale_yx=1/k, ome_zarr=True, ome_zarr_version="0.5", pyramid_levels=L)`; `fused_grid(placement, grids, config=…)` gives the output grid (`FusedGrid`) |

`fuse_images` and `Dataset.export`'s fuse part need the optional extra
`starfinder[multiview-stitcher]`; the assemble part needs `starfinder[anndata]`.
Importing `starfinder.export` imports neither multiview-stitcher, zarr nor
spatialdata. With `assemble` on, `assemble` now keeps at most two territory images in
memory: a saved assignment's territories are read again from its folder, with their
SHA-256 checked, when the flags need them; its results are unchanged. The specification
is {doc}`sample-output-contract`.

### Sample outputs: the assembled label export

One new function, `starfinder.export.fuse_labels`, with `LabelRecord`, and one new
file, `export/<sample>/cell_labels.ome.tif`; nothing existing changes, and the part is
off by default (`ExportConfig(fuse_labels=True)` turns it on).

| Need | New name |
| --- | --- |
| A label image of the assembled sample, for inspection | `fuse_labels(placement, assembly, config=LabelConfig(), output_dir)` returns a `LabelRecord`; `Dataset.export` calls it third, after `assemble` and `fuse_images` |
| What the image holds | The retained cells of the sample cell table (`assembly_status == "retained"`), each drawn whole as its FOV segmented it, with its `instance_id` (`1..N` in cell table order); 0 elsewhere. A cell retained by two FOVs shows two IDs, a cell retained by no FOV is absent; nothing is matched or merged across FOVs |
| How it is placed | multiview-stitcher's `fusion.fuse` at interpolation order 0 with the first positive view in placement table order (`_first_positive`, a small function passed to the library); labels are never blended |
| Its scale | `LabelConfig(scale_yx=1/k)`, 0.25 by default, requested with `output_spacing`; the grid starts at the smallest FOV origin, so it does not nest with the fused image's |
| The `labels` section of `export.json` | The territory files read and their SHA-256, the library's grid and the formula grid, the counts of `ids_input` and `ids_output`, `absent_ids` (cells too small for the output scale) and the conflict count; `check_export` checks L1 to L5 |

`fuse_labels` needs `starfinder[multiview-stitcher]`; without it, or when a territory file
of a FOV with a retained cell is missing, the part is `not_run` with
`ExportIncompleteWarning`. A sample with more than 16,777,216 cell rows is refused before
anything is written. `load_assembly` now also checks the SHA-256 of each territory file the
`assembly` section records (assemblies made with `conflicts=True` from saved assignment
folders), as {doc}`assembly-contract` ("Persistence") states; an export whose territory
files were changed after assembly now raises `ValueError` naming the file.

### Sample outputs: the SpatialData export

Two new functions, `starfinder.export.export_spatialdata` and `lookup_source`, and
one new store, `export/<sample>/sample.zarr/`; nothing existing changes, and the part is
off by default (`ExportConfig(export_spatialdata=True)` turns it on). The reference
branch's `sample_export` API is not adopted, and no code is brought over from it.

| Need | New name |
| --- | --- |
| A SpatialData view of the assembled sample | `export_spatialdata(load_assembly(folder), fused=None, labels=None, config=SpatialDataExportConfig(), output_dir=folder)` returns the store path; `Dataset.export` calls it fourth, after `assemble`, `fuse_images` and `fuse_labels` |
| What the store holds | One coordinate system named after the sample, in µm. Points `molecules`, shapes `cell_centroids` (circles at the ownership centroid, equivalent radius), tables `cells` and `expression` (annotating `cell_centroids` by `region` and `instance_id`), `cells_unplaced` and `molecules_unplaced`; one image element per fused OME-TIFF (`<image>_projection`, `<image>_stack`) and `cell_labels`, each read from its file and written once with one level |
| Values | Taken from `tables/`, `sample.h5ad` and the image files unchanged: nothing is filtered, reassigned or recounted; `instance_id` is `1..N` over the sample cell table, `cell_instance_id` links each molecule to its cell |
| Empty populations | An element without a row, or whose image does not exist, is left out and recorded `omitted` with `empty`, `no_cells`, `no_included_fov`, `not_requested` or `part_not_run`; with every element omitted no store is written (`store: null`) |
| Format | Zarr format 3, spatialdata's default container (`SpatialDataExportConfig(zarr_format=3)`), or Zarr format 2 (`zarr_format=2`); no standalone OME-Zarr beside the store |
| The `spatialdata` section of `export.json` | Every element with its status, reason or path, rows or shape, transform, `spatialdata_attrs` and column dtypes after the round trip; the counts, the column renames, the formats and the SHA-256 of every store file; `check_export` reads the store back and checks S1 to S8 |
| Where a row came from | `lookup_source(folder, spot=(spot_namespace, spot_id))` or `cell=(cell_namespace, cell_id)` returns its FOV, the §2.9 assignment folder, its `assignment.json` SHA-256 and the row's position in the §2.9 table |

`export_spatialdata` needs `starfinder[spatialdata]`; without it the part is `not_run`
(`extra_missing`) with `ExportIncompleteWarning`, and the earlier files are written. A
sample name outside spatialdata's element naming rule (letters, digits, `_`, `-`, `.`), a
table key that breaks it or equals another ignoring case, a cell of zero voxels and a
`sample.h5ad` that already has `uns["spatialdata_attrs"]` are refused, and no store is
written. The store is written into a staging folder and renamed into place, so an
interrupted export leaves none.

### Stitching, assembly and export workflow rules

Three Python-only YAML blocks, `stitching`, `assembly` and `export`
({doc}`workflow-configuration`, "Stitching, assembly and export"), new rule files
`workflow/rules/stitching-py.smk` and `workflow/rules/export.smk`, and the folders
`<OUTPUT>/stitching/<sample>/` and `<OUTPUT>/export/<sample>/` written by the workflow. A
configuration without the blocks runs the four legacy stitching rules, `reads_assignment`
with its tile filter, `create_sample_h5ad` and `create_sample_reads_assignment`
unchanged; its outputs, including `images/fused/{sample}/`, do not change.

| Legacy rule | With the blocks |
| --- | --- |
| `stitching_preparation`, `create_BigStitcher_macro`, `run_BigStitcher_macro` | With `method: bigstitcher` (the default): `bigstitcher_macro_sample` (`export_bigstitcher`, the tiles in `<bigstitcher_save_dir>/<sample>/tiles/`) and `bigstitcher_engine` (Fiji on the macro; `fiji_path` read when it runs; fails unless `dataset.xml` holds a `Stitching Transform` for every view). With `method: multiview_stitcher`: none |
| `create_tile_config` | `stitch_sample`: `Dataset.stitch`, or `import_bigstitcher` of the engine's `dataset.xml`, saved to `stitching/<sample>/`; with `bigstitcher` also `output/tile_config_{sample}.csv` from the record (`legacy_tile_config_view`), byte for byte today's file |
| `reads_assignment` (with an `assembly` block) | The same rule without the tile filter and without `global_*`, on the grid of `images/ref_merged/{fovID}.tif`, also saving `checkpoints/{fovID}/assignment/<assignment>/` |
| `create_sample_h5ad`, `create_sample_reads_assignment` | `export_sample`: `Dataset.export` with the `assembly` and `export` blocks into `export/<sample>/` (declared output `export.json`) |

Since W-564 the reference grid is that of `fovs/{fovID}/reference.tif` by default and the folder is `fovs/{fovID}/assignment/<assignment>/`; see "The FOV folder, the reference image and channel folders by label".

Since W-568 the route of a saved assignment writes that folder alone and no tile configuration; see "The route of a saved assignment writes the saved folder alone; opt-in assignment previews".

The adapter's translation of a legacy configuration is `StageAxisMapping(x="+x", y="+y",
source="legacy")` with the `rotate_angle` it ran with; a configuration with a
`stitching` block and a stage file names `stage_axes` itself.

### The ownership region on the assignment figures

`plot_assignment` gains the keyword `ownership=None` (W-330): a `FOVOwnership` of
`starfinder.assignment` (a boolean Y, X region mask, the excluded `cell_id` and `spot_id`),
built for one FOV of an assembled sample by the new `starfinder.assembly.fov_ownership(result,
fov_id, *, placement)`. With it, panels 1 and 2 draw the FOV's ownership region as a yellow
line and the cells and molecules the assembly excluded grey, with the legend entries
`ownership region` and `excluded from sample` and the retained counts as x-axis labels. It
gives back the view of the legacy tile box of `reads_assignment.py`, from the ownership
regions instead of the tile configuration; no tile box is drawn. Without
`ownership` the figure is unchanged. With an `assembly` block and an assignment name,
`export_sample` rewrites `expr/{fovID}/assignment.png` and `assignment_single_layer.png`
with ownership after the sample is assembled; `reads_assignment` writes them first, as
before. No table, record, count or `export/<sample>/` file changes.

Since W-568 these figures are opt-in (`assembly.previews`) and the two kinds have their own folders; see "The route of a saved assignment writes the saved folder alone; opt-in assignment previews".

### A deterministic UUID in the package's OME-TIFF files

`save_volume` (a ZYXC OME-TIFF, and a ZYX volume on a `*.ome.*` name, which tifffile
writes as OME-TIFF) and the OME-TIFF writer of `starfinder.export` (the fused
images and `cell_labels.ome.tif`) now write a name-based (version 5) `urn:uuid:` in the
OME-XML header, derived from the pixel data and the metadata written, instead of
tifffile's time-based UUID. The same array and metadata therefore give the same bytes in
any process, and a recorded file SHA-256 (for example in `sample.json` or a
`registration.json`) repeats between two runs of the same input; another voxel or
another metadata value gives another UUID. Nothing else in the files changes: pixel data,
compression, tiling, calibration and annotations are as before, readers ignore the UUID
as before, and existing files are not rewritten. No signature changes.

### The BigStitcher import refuses a non-finite affine

`import_bigstitcher` raises `ValueError` for a `dataset.xml` in which a view's
`ViewTransform` affine (the `Stitching Transform`, the `Translation to Regular Grid`, the
calibration or any other) holds a non-finite number (`NaN`, `inf`), naming the file, the
setup and the transform. The file is refused while it is parsed, before any link or
placement is built, with `fallback=None` and with `fallback="nominal"`; before, such a
view could become `metadata_only` under `fallback="nominal"`. Files with finite affines
import as before. No signature changes.

### Sample segmentation: the route after fusion and the route setting

A second route to a sample's cells beside the route per FOV: the fused Z maximum of a
projection sample is segmented once and the sample's molecules are assigned to those
cells ({doc}`sample-segmentation-contract`; for users, {doc}`workflow-downstream`, "Two
routes to a sample's cells"). New names in `starfinder.export`, one keyword of
`assemble`, three fields of `ExportConfig`, three keys of the workflow `export` block and
one column of the sample molecule table; no existing name, module or format version
changes. The route per FOV keeps its rules, the duplicate rule of W-400 included.

| Need | New name |
| --- | --- |
| Choose the route of a sample | `ExportConfig(route=…)`: a value of `ROUTES = ("per_fov", "after_fusion")`, or None (the default) for the route by `dimensionality`: `projection` gives `after_fusion`, `stack` and `both` give `per_fov`; the `assembly` section of `export.json` records `route` and `route_source` (`set` or `default`) |
| The settings of the route after fusion | `ExportConfig(segmentation=SampleSegmentationConfig(runs, cells, nuclei=None, assignment=AssignmentConfig(), device="cpu"))` with one `SampleSegmentationRun(name, target, inputs, method, seeds=None)` per run: `inputs` are `(role, image)` pairs of image names of `FusionConfig.images`, `method` a `SEGMENTATION_METHODS` config or a `LabelImportConfig`, its pixel lengths given for the acquired grid |
| Segment the fused projection | The new part `segment` of `Dataset.export`, after `fuse`; alone, `segment_fused(output_dir, *, config, sample, identity=None)` on a folder whose `fuse` ran returns a `SampleSegmentation(grid, runs, record)` and writes `segmentation_<run>.ome.tif` per run and the `segmentation` section; `load_sample_segmentation(output_dir)` reads them back; `ExportConfig(segment=False)` reuses them |
| Assign the sample's molecules to the sample's cells | `assemble(placement, inputs, config=…, segmentation=…)` with molecule-table inputs: the `molecules_only` assembly, then one `assign_molecules` call on the segmentation grid; `territories.ome.tif` when the assignment expands the cells |
| Compare the two routes of one sample | `compare_routes(per_fov, after_fusion, *, threshold=0.5, output=None)` returns a `RouteComparison`; it reads two export folders, writes nothing unless `output` names a folder outside both, and decides nothing |
| Link a molecule to its cell | The new column `cell_namespace` of `tables/molecules` on both routes (the cell's namespace for a molecule with a `cell_id`, null otherwise); `check_identities`, `cell_links` and `check_export` link by `(cell_namespace, cell_id)`. A folder saved before the column existed still loads: the column is filled from the cell table |
| The workflow | The `export` block's keys `route`, `segment` and `segmentation` ({doc}`workflow-configuration`, "Stitching, assembly and export"); on the route after fusion `export_sample` needs the molecule tables and the source images instead of the outputs of `reads_assignment`, and draws no assignment figure again |

`export.json` gains the part `segment` in `parts` on both routes (`off`, reason
`route_per_fov`, on the route per FOV; `no_cells` without an assignment) and the
`segmentation` section; on the route after fusion the `assembly` and `labels` sections
record `segmentation_files`, and `lookup_source(cell=…)` returns `assignment_folder` null,
the cell's row in the sample cell table, `label_file` and `label_value`. The SHA-256 of
`tables/molecules` changes on the route per FOV because of the new column.

### Records added to `run.json`

The records the processing report reads ({doc}`report-contract`, "Records added to
run.json"):

* **`run.json` gains `transforms`, `codebook`, `workflow_config` and
  `counts.filtering.endpoint_valid`** (with `endpoint_valid_segments`); `format_version`
  stays 1 (note (W-565): now 2, see "run.json version 2"), and a `run.json` without them still loads. `transforms` holds, per registered
  moving round, the entries of `registered/transforms.json` and the round's
  `WarpConfig`; a dense field or B-spline coefficients are named only when the stage
  `registered` wrote them, and no array is written into `run.json`.
* **The workflow writes `checkpoints/<fov>/run.json`** for the processing rules, alone by
  default, with `overwrite` on; `rules.<rule>.parameters.checkpoints` adds stages
  ({doc}`workflow-configuration`, "Run records of the processing rules"). The logs the
  rules write do not change (W-563 later removed the Python logs; W-564 moved the record to
  `fovs/<fov>/run.json`, see "The FOV folder, the reference image and channel folders by label").
* **`Dataset.load_codebook` and `load_direct_panel` keep the file's path and SHA-256**,
  which `run.json` records as `codebook`.

### Molecule and read tables: `molecules.csv`, `reads.csv` and zero-based coordinates

W-562 (OM-04, OM-18) replaces the one-based, identity-free `goodSpots` and `allSpots`
exports of the Python route with typed, zero-based tables that keep each read's identity.
This is a breaking change of the files, their columns and their coordinate origin, with no
alias, no dual write and no dual search.

* **`FOV.export_spots` slots and files.** `FOV.export_spots(slot="molecules", columns=None, *,
  path=None)` takes the slots `molecules` (the accepted reads of the filtering result) and
  `reads` (on request, every read of the decoding result joined to its detection) in place of
  `goodSpots` and `allSpots`; there is no `export_molecules` or `export_reads` (NM-09). The
  default paths are `<output_root>/fovs/<fov_id>/molecules.csv` and
  `<output_root>/fovs/<fov_id>/reads.csv`, in place of `signal/<fov_id>_goodSpots.csv` and
  `signal/<fov_id>_allSpots.csv`; `FOV.paths.spot_table(slot)` names them.
* **`readout.export_spots(detection, reads, path, *, accepted_only=False, columns=None)`**
  writes the new formats: with `accepted_only` the molecule table, otherwise the read table
  (without reads, the detection table). `columns` takes the new column names (`gene_id`,
  `observed_color_sequence`); the renames to `gene` and `color_seq` are gone. The file is a
  typed CSV written atomically by `_records._write_table` (`<NA>` for a missing value, floats
  with 17 significant digits); the path must end in `.csv`.
* **Coordinates are zero-based ZYX**, equal to the in-memory detection coordinates: a new
  `x`, `y` or `z` is the old value minus 1.
* **New: `assignment.load_molecules(path, *, genes, spot_namespace=None)`** reads a
  `molecules.csv` with its identities as a `MoleculeTable` of population `final`; its source
  records the namespace, the path and the file's SHA-256. The ordered gene list is a
  keyword, as for `import_legacy_molecules_csv`, which stays for the one-based legacy files.
* **`run.json` gains `outputs.<slot>`** when the FOV's latest `FOV.run` had checkpoints and
  succeeded: `export_spots` adds `{path, sha256, molecules_sha256, dtypes, rows}` (the path
  relative to the record's folder; `molecules_sha256` the keyed hash of the six molecule
  columns, null for `reads`) and rewrites the record atomically. `format_version` stays 1,
  and a `run.json` without `outputs` still loads. Note (W-565): version 2 holds `outputs`
  from the start and records more outputs, see "run.json version 2".
* **Removed: `FOV.paths.signal_png`**, the unused helper of `signal/<fov_id>_<slot>.png`
  (OM-18; no replacement).

`molecules.csv` against the old `goodSpots` (same rows, in the same order):

| Old column (`goodSpots`) | New column (`molecules.csv`) | dtype | Origin |
| --- | --- | --- | --- |
| — | `spot_namespace` | string | the FOV's identity list |
| — | `spot_id` | string | the detection's spot ID (the legacy reader made up `csv:<row>`) |
| `z` | `z` | float64 | zero-based: old `z` − 1 |
| `y` | `y` | float64 | zero-based: old `y` − 1 |
| `x` | `x` | float64 | zero-based: old `x` − 1 |
| `gene` | `gene_id` | string | the same gene |
| — | `entry_id` | string | the codebook entry (`<NA>` where none) |
| — | `qc_score` | float64 | the shared read score; `<NA>` for unscored reads |

`reads.csv` against the old `allSpots` (same rows, in the same order): the columns are
`spot_namespace, spot_id, z, y, x`, then the other detection columns, then the decoding
columns, each name once; the old default `x, y, z, gene` is gone (`columns` selects).

| Old column (`allSpots`) | New column (`reads.csv`) | Origin |
| --- | --- | --- |
| `x`, `y`, `z` | `x`, `y`, `z` | zero-based: old value − 1 |
| `gene` | `gene_id` | unchanged values |
| `color_seq` | `observed_color_sequence` | unchanged values |
| `round_x`, `round_y`; `channel_x`, `channel_y` (direct readout) | `round`; `channel` | one copy, the reads' (the channel label), after a check that the detection agrees (its channel index read through the channel labels); a disagreement raises |
| every other column | the same name | unchanged values |

Consumers on the Python processing route read `molecules.csv` by key with
`load_molecules`, and their outputs keep the molecules' `spot_id`s: the `reads_assignment`
adapter, `Dataset.export(assignment=None)`, the inputs of `export_sample` and of
`reads_assignment`, and the Python `rsf_single_fov`'s declared output
`fovs/{fovID}/molecules.csv`. On this route `expr/{fovID}/reads_assignment.csv` has the
`molecules.csv` columns, then `global_x, global_y, global_z` (without an assembly block)
and `seg_label`, then the assignment columns, joined by `(spot_namespace, spot_id)` rather
than by row position.

The legacy route keeps its files and reader: MATLAB's `rsf_single_fov` and `stitch_subtile`
(the subtile and deep modes, either backend) write the one-based
`signal/{fovID}_goodSpots.csv`, the resident subtile rules write
`output/subtile/{fovID}/subtile_goodSpots_<n>.csv` through a private legacy writer, and
the consumers read them with `import_legacy_molecules_csv`. Each consumer chooses its file
from the configuration ({doc}`workflows`, "Molecule tables"): the backend and the producer
the `Snakefile`'s `ruleorder` selects; it never looks for whichever file exists.

### Sequencing logs and the morphology record

W-563 (OM-05, OM-17) folds the Python backend's per-FOV sequencing logs into `run.json` and
replaces the morphology log and its shift table with one JSON record. The MATLAB backend
writes its logs under the same names as before; only the Python files are removed, with no
alias and no dual write. Existing output trees are not rewritten.

* **`run.json` gains three fields** (`format_version` stays 1, now 2 since W-565; a record
  without them still loads): `counts.filtering.fractions` (`accepted` and, when the filter checked endpoints,
  `endpoint_valid`, as `ReadFilteringResult.fractions` computes them: over all reads, `null`
  without reads); `counts.summary` (`summarize_reads` of the latest reads) whenever reads were
  decoded, also without scoring or deduplication; and `rounds`, the run's `RoundState`
  roles (`sequencing_rounds`, `other_rounds`, `reference_round`).
* **Removed: `FOV.save_processing_log`, `FOV.save_diagnostics` and `FOV._save_shift_log`**,
  and the path helpers `FOV.paths.rsf_log`, `gr_log`, `nr_log`, `shift_log` and `score_log`.
  A direct caller obtains the record with `FOV.run(..., checkpoints=CheckpointConfig(stages=()))`,
  which writes `checkpoints/<fov_id>/run.json` and no stage file (since W-564 `fovs/<fov_id>/run.json`; see
  "The FOV folder, the reference image and channel folders by label").
* **New: `FOV.save_registration_record(path=None)`** writes
  `<output_root>/fovs/<fov_id>/nuclei_registration.json`: `fov_id`, `rounds` (the round roles),
  `registered_rounds` (the `registration_record["rounds"]` entries of the other rounds with
  registration attempts; the reference stain, which has none, is left out) and
  `registration` (those rounds' attempts, unchanged). Prepared-image persistence
  (`other_rounds/<name>/`) stays opt-in and unchanged.

Each removed file and field, and its place now:

| Removed file | Field | Now |
| --- | --- | --- |
| `log/<fov>_rsf.txt`, `log/<fov>_gr.txt` (JSON) | `fov_id` | `run.json` `fov_id` |
| | `backend` (always `python`) | none: only the Python package writes `run.json` (`code` names its version) |
| | `rounds` | `run.json` `rounds` |
| | `registration_attempts` | `run.json` `registration` |
| | `detected` | `run.json` `counts.spots` |
| | `filtering` (the filter counts) | `run.json` `counts.filtering` (`total`, `accepted`, `rejected`, beside the endpoint counts and `fractions`) |
| `log/sf_scores/<fov>.txt` (JSON; `<fov>_<n>.txt` for subtile `n`, now in `checkpoints/<fov>/subtile_<n>/run.json`) | `detected` | `run.json` `counts.spots` |
| | `counts` | `run.json` `counts.filtering` (`total`, `accepted`, `rejected`) |
| | `fractions` | `run.json` `counts.filtering.fractions` |
| | `summary` | `run.json` `counts.summary` |
| | `registration_attempts` | `run.json` `registration` |
| `log/gr_shifts/<fov>.txt` (CSV) | `fov_id` | `run.json` `fov_id` |
| | `round` | the round's key in `run.json` `transforms` |
| | `row`, `col`, `z` (one row per translation result) | `dy`, `dx`, `dz` of `transforms.<round>.results[i].transform.displacement_zyx`, which is `[dz, dy, dx]` |
| `log/<fov>_nr.txt` (JSON) | `fov_id`, `rounds` | `nuclei_registration.json` `fov_id`, `rounds` |
| | `backend` (always `python`) | none |
| | `registration_attempts` (the registered other rounds) | `nuclei_registration.json` `registration`, unchanged; new beside it: `registered_rounds` |
| | `detected`, `filtering` (always `null` there) | none |
| `log/gr_shifts/<fov>_nr.txt` (CSV) | `fov_id`, `round` | `nuclei_registration.json` `fov_id` and the round's key in `registration` |
| | `row`, `col`, `z` | `dy`, `dx`, `dz` of the translation estimation attempts' `qc.details.transform.displacement_zyx` in `registration.<round>`, one attempt per translation step |

The workflow's declared outputs change on the Python backend only; the final targets of
`common.smk` branch on `backend`, and `registration.smk` and `spot-finding.smk` are unchanged:

| Rule | Python before | Python now | MATLAB |
| --- | --- | --- | --- |
| `rsf_single_fov` | `log/{fovID}_rsf.txt`, `log/sf_scores/{fovID}.txt` | `checkpoints/{fovID}/run.json` | unchanged |
| `gr_single_fov_subtile` | `log/{fovID}_gr.txt` | `checkpoints/{fovID}/run.json` | unchanged |
| `deep_create_subtile` | none (it wrote `log/{fovID}_gr.txt` and the shift table undeclared) | `checkpoints/{fovID}/run.json` | unchanged |
| `lrsf_single_fov_subtile`, `deep_rsf_subtile` | `log/sf_scores/{fovID}_{n}.txt` | `checkpoints/{fovID}/subtile_{n}/run.json` | unchanged |
| `nuclei_registration` | `log/{fovID}_nr.txt`, `log/gr_shifts/{fovID}_nr.txt` | `fovs/{fovID}/nuclei_registration.json` | unchanged |

The other outputs of these rules do not change. `log/gr_shifts/{fovID}.txt`, which the
Python processing rules wrote without declaring it, is no longer written. Without an
`assembly` or `stitching` block, the declared input of `report_sample` is every FOV's
`checkpoints/{fovID}/run.json` instead of `log/sf_scores/{fovID}.txt`, so it now has a
producer in the subtile modes too. No new human log file is written (deferred to §2.13).
Since W-564 the run records are `fovs/{fovID}/run.json` and `fovs/{fovID}/subtile_{n}/run.json`; see "The FOV folder, the reference image and channel folders by label".

### The FOV folder, the reference image and channel folders by label

W-564 (OM-08, OM-06, OM-01) collects a FOV's durable Python results in one folder,
writes and reads the reference image through one resolver, and names the Python channel
folders by the unique channel label. New runs write the new layout. Existing output trees
are not moved, and no reader looks in the old places (no alias, no dual search, no
fallback). File contents are unchanged, except the recorded paths that name a moved file.
The sample-stage roots (`stitching/`, `export/`, `report/`), `log/benchmark/`, the shared
`output/subtile/` files, the legacy segmentation views under `images/`, `expr/` and the
MATLAB backend's tree do not change.

* **The FOV folder.** `<output_root>/fovs/<fov_id>/` (a subtile: `fovs/<fov_id>/subtile_<n>/`)
  holds `run.json`, `molecules.csv` and `reads.csv`, `nuclei_registration.json`,
  `reference.tif`, `other_rounds/<name>/`, `segmentation/<run>/` and `assignment/<name>/`.
  The restart material of `FOV.run` goes in its `checkpoints/` folder: `registered/`,
  `candidates.*` and `pre_qc.*` ({doc}`checkpoints`, "Layout").
* **`CheckpointConfig.directory`** names the parent of the FOV folders, default
  `<output_root>/fovs` (was `<output_root>/checkpoints`). A FOV's folder is still
  `<directory>/<fov_id>/`, and its stage files move into `checkpoints/` below it. A call
  that passed `directory=<output_root>/"checkpoints"` to mean the default drops the argument.
  `stages`, `table_format` and `hash_inputs` keep their meaning.
* **`CheckpointConfig.overwrite`** guards the run's own files: without it, an existing
  `run.json` or `checkpoints/` of the FOV folder raises `FileExistsError` before any
  processing, and the folder's other contents, which other calls and rules write, no longer
  do. With it, `FOV.run` clears only `checkpoints/` (`clear_stages`) and replaces `run.json`.
* **Recorded paths in `run.json`** (`format_version` stays 1; path portability is unchanged;
  note (W-565): version 2 makes the paths portable, see "run.json version 2"):
  `checkpoints.<stage>` lists each file relative to the FOV folder, which holds `run.json`
  (`checkpoints/registered/round1.ome.tif`, was `registered/round1.ome.tif`);
  `checkpoint_directory` names `<FOV folder>/checkpoints` (was the FOV directory); and
  `outputs.<slot>.path` is relative to the FOV folder (`molecules.csv`, was
  `../../fovs/<fov_id>/molecules.csv`). A saved assignment's link to
  `../../segmentation/<run>/labels.tif` is unchanged: the two folders stay siblings.
* **One reference-image resolver.** `FOV.save_reference_image`, `FOV.load_reference_image`,
  the workflow's rule declarations and every reader (stitching, export and assembly, the
  report, the workflow adapters and `stitch_subtile`) resolve the file through one function
  of the dataset's settings. The default of `reference_image` (workflow key and `Dataset`
  field) is `fovs/{fov_id}/reference.tif` (was `images/ref_merged/{fov_id}.tif`); the
  configuration schema has no default value for it, and absent or null means the default.
  An imported reference image, a file another tool or rule produced, is declared with the
  new `reference_image_import` (workflow key and `Dataset` field), a pattern of the same
  form that excludes `reference_image`: every reader reads it, `FOV.save_reference_image`
  raises `ValueError` instead of writing it, and the Python processing rules declare and
  write no reference image. The TIFF content, calibration, projection metadata and grid
  identity do not change. A segmentation run that reads the saved reference image records
  its `reference_image.path` relative to the output root, or, for a file outside it, by
  its absolute path, with its SHA-256 (it raised `ValueError` before).
* **`FOV.paths`** loses `ref_merged_tif` (the reference image's file depends on the
  dataset's settings; `FOV.save_reference_image` returns it) and gains `fov_dir` and
  `run_record`; `checkpoint_dir` is `fovs/<fov_id>/checkpoints`.
* **Channel folders by label.** The Python `nuclei_registration` rule writes
  `images/<round>/<channel label>/<fov>.tif`, the label being the unique
  `ChannelInfo.channel` (the configuration's `channel`), where it wrote the descriptive
  name's folder, which two channels of the same name shared. The Python backend's
  `enhance_dapi_with_flamingo` reads the folders of the labels of the `flamingo` round's
  channels named `DAPI` and `Flamingo`. A channel pattern that is not a safe single path
  component (`[A-Za-z0-9][A-Za-z0-9_.-]*`) now raises `ValueError` when the `Dataset` is
  built, and so in the workflow's translation. MATLAB keeps its name folders and rules.
* **The report** names the new places: `report.json` `assignment.folders` holds
  `fovs/<fov>/assignment/<name>` (was `checkpoints/<fov>/assignment/<name>`), the page's
  output row "FOV folders" shows `fovs/<fov_id>/` (was "Checkpoints",
  `checkpoints/<fov_id>/`), and the reasons of unavailable parts name `fovs/<fov>/run.json`
  and `fovs/<fov>/checkpoints/registered/transforms.json`.

Each moved path, old and new (`<fov>` is the FOV ID; the Python backend only):

| Before | After |
| --- | --- |
| `checkpoints/<fov>/run.json` | `fovs/<fov>/run.json` |
| `checkpoints/<fov>/subtile_<n>/run.json` | `fovs/<fov>/subtile_<n>/run.json` |
| `checkpoints/<fov>/registered/` (`<round>.ome.tif`, `<snapshot>/<round>.ome.tif`, `<round>_field.npz`, `transforms.json`) | `fovs/<fov>/checkpoints/registered/` |
| `checkpoints/<fov>/candidates.csv`, `.parquet`, `.json` | `fovs/<fov>/checkpoints/candidates.csv`, `.parquet`, `.json` |
| `checkpoints/<fov>/pre_qc.csv`, `.parquet`, `.json` | `fovs/<fov>/checkpoints/pre_qc.csv`, `.parquet`, `.json` |
| `checkpoints/<fov>/subtile_<n>/<stage files>` | `fovs/<fov>/subtile_<n>/checkpoints/<stage files>` |
| `checkpoints/<fov>/other_rounds/<name>/` | `fovs/<fov>/other_rounds/<name>/` |
| `checkpoints/<fov>/segmentation/<run>/` | `fovs/<fov>/segmentation/<run>/` |
| `checkpoints/<fov>/assignment/<name>/` | `fovs/<fov>/assignment/<name>/` |
| `<directory>/<fov>/<stage files>` (`CheckpointConfig(directory=…)`) | `<directory>/<fov>/checkpoints/<stage files>` |
| `images/ref_merged/<fov>.tif` | `fovs/<fov>/reference.tif` (the default `reference_image`) |
| `images/<round>/<channel name>/<fov>.tif` (`nuclei_registration`) | `images/<round>/<channel label>/<fov>.tif` |

`fovs/<fov>/molecules.csv`, `reads.csv` and `nuclei_registration.json` were already in the
FOV folder (W-562, W-563).

The configuration entries:

| Key or field | Before | After |
| --- | --- | --- |
| `CheckpointConfig.directory` | parent of the FOV checkpoint folders; `None`: `<output_root>/checkpoints` | parent of the FOV folders; `None`: `<output_root>/fovs` |
| `CheckpointConfig.overwrite` | guarded the existing FOV directory and cleared its stage files | guards `run.json` and `checkpoints/`; clears `checkpoints/` only |
| `reference_image` (workflow key, `Dataset` field) | default `images/ref_merged/{fov_id}.tif` (also the schema's default) | default `fovs/{fov_id}/reference.tif`; absent or null is the default; excludes `reference_image_import` |
| `reference_image_import` (workflow key, `Dataset` field) | none | new: the pattern of an imported reference image, read and never written |
| `rules.<rule>.parameters.checkpoints` | stage files beside `checkpoints/<fov>/run.json` | stage files in `fovs/<fov>/checkpoints/` beside `fovs/<fov>/run.json` |
| `ChannelInfo.channel` (`channel` of a channel entry) | any nonempty string | a safe single path component, checked when the `Dataset` is built |

The workflow's declarations change on the Python backend only; `registration.smk` and the
MATLAB branch of `segmentation.smk` are unchanged:

| Rule | Python before | Python now |
| --- | --- | --- |
| `rsf_single_fov` | `checkpoints/{fovID}/run.json`, `images/ref_merged/{fovID}.tif`, `fovs/{fovID}/molecules.csv` | `fovs/{fovID}/run.json`, the reference image (none with `reference_image_import`), `fovs/{fovID}/molecules.csv` |
| `gr_single_fov_subtile`, `deep_create_subtile` | `checkpoints/{fovID}/run.json`, `images/ref_merged/{fovID}.tif` and the subtile files | `fovs/{fovID}/run.json`, the reference image (none with `reference_image_import`) and the subtile files |
| `lrsf_single_fov_subtile`, `deep_rsf_subtile` | `checkpoints/{fovID}/subtile_{n}/run.json` | `fovs/{fovID}/subtile_{n}/run.json` |
| `stitch_subtile` | read `images/ref_merged/{fovID}.tif` without declaring it | declares and reads the reference image (its input `reference`) |
| `create_nuclei_amplicon_overlay` | input `images/ref_merged/{fovID}.tif` | input: the reference image |
| `enhance_dapi_with_flamingo` | inputs `images/flamingo/DAPI/{fovID}.tif`, `images/flamingo/Flamingo/{fovID}.tif` | inputs `images/flamingo/<label>/{fovID}.tif` of the channels named `DAPI` and `Flamingo` |
| `reads_assignment` with an `assembly` block | output `checkpoints/{fovID}/assignment/<name>/assignment.json` | output `fovs/{fovID}/assignment/<name>/assignment.json` |
| `export_sample` (route per FOV) | input `checkpoints/{fovID}/assignment/<name>/assignment.json` | input `fovs/{fovID}/assignment/<name>/assignment.json` |
| `report_sample` without `assembly` or `stitching` | input `checkpoints/{fovID}/run.json` | input `fovs/{fovID}/run.json` |

### run.json version 2: portable paths, stage records that own their hashes, and recorded outputs

W-565 (OM-09) makes `run.json` the record a later verified resume can rely on. New runs write
`format_version` 2; existing records are not rewritten. The readers of `run.json` (the report's
collector and replay) read versions 1 and 2. Verified resume, scheduling and job completion
are not implemented (§2.13); version 2 provides what they need, and version 1 does not
({doc}`checkpoints`, "Reload and continue").

* **Portable paths.** Output and stage paths are relative to the FOV folder, the image inputs
  relative to the input root. A file the run reads outside the input root (a codebook or panel
  elsewhere, the workflow configuration, supplied statistics, pretrained weights) is recorded by
  its absolute path and SHA-256; such a file inside the input root is recorded relative to it.
  `input_root` and `output_root` keep the roots of the run, absolute, as provenance.
* **Stage records.** `checkpoints.<stage>` names the stage record with that record's SHA-256
  (`{path, sha256}`, was the list of the stage's files). The stage headers `candidates.json`,
  `pre_qc.json` and `registered/transforms.json` are of version 3 (checkpoint `FORMAT_VERSION`
  3, `READ_VERSIONS` 1 to 3) and gain `files`: every payload file they name (the table; the
  round images, snapshots and field files), relative to the header's folder, with its SHA-256.
  `load_checkpoint` (the function and `FOV.load_checkpoint`) checks them before reading and
  raises `ValueError` naming a missing or changed file; headers of versions 1 and 2 load
  without the check.
* **Outputs.** `outputs` is `{}` when the processing ends (it was absent until
  `export_spots`). Besides `molecules` and `reads`, `FOV.save_reference_image` adds
  `reference_image`, `FOV.save_subtiles` adds `subtile_coords` and `subtile_data`, and a
  resident subtile rule adds `subtile_spots` (`subtile_goodSpots_<n>.csv`), each with `path`
  (relative to the FOV folder) and `sha256`, as it is written, after a checkpointed run that
  succeeded. `status` `succeeded` keeps its meaning, that `FOV.run`'s processing succeeded;
  `outputs` records what was written afterwards.
* **Readers.** The report's check 3 accepts `format_version` 1 or 2. A version-1 record is
  read as recorded (its paths opened as written), and `report.json` `stages` gains
  `version_1_run_records`, the FOVs whose record is of version 1 and so not eligible for
  verified resume. A version-2 record's relative input paths are resolved against the input
  root given at read time: `Dataset.report` passes `Dataset.input_root`, and
  `collect_report`, `report_examples`, `report_measurements` and the replay's `replay_fov`
  take `input_root` (None: the record's `input_root`). No reader opens a version-2 relative
  path against the current folder. The replay accepts a saved `transforms.json` of version 2
  or 3, checks it against `checkpoints.registered.sha256`, and checks the field files against
  its `files`.

Each key of version 1 and its place in version 2 (`<IN>` and `<OUT>` are the input and
output roots as the `Dataset` holds them, `<FOV>` the FOV folder):

| Version 1 | Version 2 | Change |
| --- | --- | --- |
| `format_version` `1` | `format_version` `2` | the version |
| none | `input_root`, `output_root` | new: the roots, absolute (provenance) |
| `dataset_id`, `sample_id`, `fov_id`, `subtile_id` | the same | unchanged |
| `status`, `started_at`, `ended_at`, `error` | the same | unchanged; `status` is the outcome of the processing |
| `code`, `environment` | the same | unchanged |
| `config.pipeline`, `config.execution`, `config.readout_mode`, `config.encoding` | the same | unchanged (recorded as given) |
| `config.checkpoints.stages`, `.table_format`, `.hash_inputs`, `.overwrite` | the same | unchanged |
| `config.checkpoints.directory` | the same | as given → relative to `<FOV>` (`..`; `../..` for a subtile); `null` stays `null` |
| `workflow_config.path` | the same | absolute → absolute, or relative to `<IN>` when it lies there |
| `workflow_config.sha256` | the same | unchanged |
| `channels`, `rounds` | the same | unchanged |
| `inputs[].path` | the same | `<IN>/<round>/<fov>/<file>` as given → `<round>/<fov>/<file>`, relative to `<IN>` |
| `inputs[].sha256` | the same | unchanged |
| `codebook.path` | the same | absolute → absolute, or relative to `<IN>` when it lies there |
| `codebook.kind`, `.sha256`, `.entries`, `.genes` | the same | unchanged |
| `steps` (with `methods[].artifacts[]`: weights by absolute path and SHA-256) | the same | unchanged |
| `preprocessing.supplied_statistics.path` | the same | as given → absolute, or relative to `<IN>` when it lies there |
| `preprocessing` (the other keys), `registration`, `transforms`, `counts` | the same | unchanged |
| `checkpoint_directory` `<OUT>/fovs/<fov>/checkpoints` (as the root is given) | `checkpoint_directory` `checkpoints` | relative to `<FOV>` |
| `checkpoints.<stage>`: the stage's files, relative to `<FOV>` | `checkpoints.<stage>`: `{path, sha256}` of the stage record | the files move to the record's `files`, each with its SHA-256 |
| `outputs.<slot>` (`molecules`, `reads`), absent before `export_spots` | `outputs.<slot>`, `{}` from the start | adds `reference_image`, `subtile_coords`, `subtile_data`, `subtile_spots` |

The stage headers:

| Header key | Version 2 | Version 3 |
| --- | --- | --- |
| `format_version` | `2` | `3` |
| `files` | none | `[{path, sha256}]`: the table (`candidates`, `pre_qc`); the round images, snapshots and field files (`registered`), relative to the header's folder |
| every other key | as before | unchanged |

No field of version 1 is dropped: each value appears in version 2 equal, with its path
mapped, or (the stage file lists) in the stage record it names. A saved base-versus-result
comparison of the quickstart's records checks this key by key.

### FOV membership in `fovs.csv`, `{i}` FOV patterns and explicit stage files

W-566 (OM-07, OM-03, OM-14) makes the Python workflow's FOVs explicit, gives FOV names one
pattern syntax and reads stage positions from a named file. The MATLAB backend keeps its
sample annotation, its keys and its scripts; where the two backends share a rule or a
target, the change is a `BACKEND` branch. Existing output trees are not rewritten. Each
change has no alias, no dual read and no dual search.

* **Membership: `INPUT/fovs.csv`** (`<root_input_path>/<dataset_id>/<sample_id>/fovs.csv`),
  for the Python backend, replaces `OUTPUT/documents/sample-annotation.csv` and its
  `sample_id,fov_start,fov_end` ranges. Columns `sample,fov_id` and an optional
  `position_id` (the stage file's integer `PositionID`), read in row order. A repeated row,
  a FOV in two rows, a `position_id` of two FOVs, an empty value or another column raises
  `ValueError` when the rule files are parsed. The `sample` column groups FOVs; the
  acquisition `sample_id` and every namespace are unchanged.
* **`FOVS`, `SAMPLE` and `SAMPLE_TO_FOV`** of `common.smk` come, for the Python backend,
  from `fovs.csv` in row order (`FOVS` was the sorted names of `1..n_fovs`, `SAMPLE` a set
  under `subset_list`); for the MATLAB backend they are computed as before.
* **Configuration keys refused for the Python backend:** `fov_id_pattern`, `n_fovs`,
  `subset_list`, `subset_range`, `subset_start`, `subset_end`, `subset_random` and
  `n_random_tests`. They are MATLAB-only; the schema requires `fov_id_pattern` and `n_fovs`
  for the MATLAB backend only and refuses all eight with `backend: python`. Python has no
  random FOV selection.
* **The defaults of `subset_range` and `subset_random` (`false`)** moved from the schema to
  `common.smk`'s MATLAB branch (`config.setdefault`), because the Snakefile's
  `snakemake.utils.validate` inserts schema defaults into every configuration, the Python
  ones included. A MATLAB configuration gets the same values. Omitting a switch does not
  remove its companion keys: the schema's conditional clauses still require `subset_start`
  and `subset_end`, and `n_random_tests`, so `tests/minimal_config.yaml`, which sets no
  subset key, still fails validation, as it already did before this change.
* **New key `selected_fovs`** (Python only; absent or null: every FOV): a nonempty list of
  unique FOV IDs of `fovs.csv`. It restricts every rule, the per-sample stitching, export
  and report rules included (the subset keys did not restrict the per-sample rules). An
  unknown or repeated ID raises. The Python backend constrains the `fovID` wildcard of every
  rule to the run's FOVs (`wildcard_constraints` in the Snakefile), so an explicit target of
  another FOV resolves to no rule. Under a selection a supplied position table keeps its
  rows of the selected FOVs, and a regular grid is laid out over the sample's FOVs of
  `fovs.csv`, so each selected FOV keeps its cell.
* **The legacy sample rules run with `backend: matlab` only:** `create_sample_maf`, the
  four rules of `stitching.smk`, `create_sample_h5ad` and `create_sample_reads_assignment`.
  A Python configuration that enables one of them without the block that replaces it
  (`stitching`; `assembly` for the two sample rules), or `reads_assignment` without either
  block, raises `ValueError` when the rule files are parsed, naming the blocks to add.
* **`reads_assignment`'s first input** is the run's membership file (`INPUT/fovs.csv` for
  the Python backend, the annotation for MATLAB). For the Python backend without an
  assembly block, the FOV's row of `tile_config_<sample>.csv` is the one whose `id` is its
  `position_id`, or without that column its 1-based place among the sample's FOVs.
* **`report_dataset`'s samples** are in the order of `fovs.csv` (was the annotation's).
* **`Dataset.fov_pattern` is a `{i}` pattern:** `str.format` with the field `i`, default
  `"Position{i:03d}"` (was the percent pattern `"Position%03d"`). `Dataset.fov_ids(n_fovs,
  start=0)` keeps its name and its starting index (direct calls default to 0; the workflow
  numbers FOVs from 1) and formats `fov_pattern.format(i=i)`. A pattern that is not a `{i}`
  pattern, a percent pattern included, raises `ValueError` when the `Dataset` is built.
  A workflow-built `Dataset` holds the default pattern: the Python workflow names its FOVs
  by `fovs.csv` and no longer reads `fov_id_pattern`.
* **Stage files are named explicitly.** `stitching.positions.stage_file` (and the
  `stage_file` key of `Dataset.stitch`'s positions mapping) is required with
  `source: stage_file`; the default `documents/maf/<sample>.maf` is removed, and the Python
  route writes no per-sample copy. `stitching.positions.table` is required with
  `source: table`, now when the rule files are parsed. With a stage file, `fovs.csv` must
  have `position_id`.
* **Selection before conversion.** `positions_from_leica` gains `position_ids` (the
  `PositionID` values to read; None reads every position): the positions are selected
  before the axis mapping, the shift to the smallest y and x, the snap step, the cells and
  the clash check, and the record's `fov_ids` and `stage_positions` list the positions
  read. `Dataset.stitch` gains `position_ids` (each FOV's `PositionID`; None: `PositionID`
  `p` names the FOV `fov_pattern.format(i=p)`) and selects the positions of its `fov_ids`,
  so a whole-acquisition stage file gives the placement its per-sample copy gave. A FOV
  whose `PositionID` the file does not hold raises `ValueError`; a position of another FOV
  is no longer an error. For a file that holds only the sample's positions the placement
  values, links and units are unchanged.
* **Examples.** `workflow-minimal.yaml`, `workflow-full.yaml` and `workflow-recipe-2.yaml`
  drop the eight keys; `prepare_workflow.py` writes `INPUT/fovs.csv` instead of the
  annotation; `quickstart.py` uses `fov_pattern="FOV_{i:03d}"` and writes `input/fovs.csv`.

### One gene authority, the reference-file keys and `image_dir_template`

W-567 (OM-13, NM-11) takes the count matrix's genes from the codebook alone and lets Python
read FOV-first image folders. The MATLAB code, its codebook input `INPUT/genes.csv` and its
round-first layout are unchanged. Existing output trees are not rewritten, no input file is
renamed, and each change has no alias, no dual read and no dual search.

* **`OUTPUT/documents/genes.csv` is no longer read.** `reads_assignment`'s fifth input is
  the codebook the sequencing rules read (`CODEBOOK` of `common.smk`), on both backends,
  since the rule and its adapter are shared; it was `documents/genes.csv`, checked against
  the codebook by gene set. The adapter loads the codebook as the sequencing rules do (the
  encoding and segment layout of the first Python rule the configuration names; a
  configuration without `seq_channel_order` labels its four colour channels `1`–`4`, on
  which the genes do not depend), and its genes are the gene universe and the matrix
  columns. The private helpers `_assignment_genes` and `_codebook_genes` of
  `starfinder._snakemake` are removed; `_reference_genes` of `starfinder.dataset.workflow`
  replaces them. The report's private reader of the same name, which reads a recorded
  codebook's gene names for the marker check, is renamed `_reference_gene_names` and is
  otherwise unchanged.
* **The column order of `raw.h5ad` may change.** The columns are the codebook's genes in
  its order: `Codebook.genes` (first appearance) or, with `readout_mode: direct`,
  `DirectPanel.genes` (table order). They were in the order of `documents/genes.csv`. Where
  the two orders differed, the columns are permuted and each gene's counts are unchanged:
  compare matrices by gene name (`adata[:, genes]`), not by column position. A codebook gene
  that no molecule carries is a zero column, as before when both lists held it; a gene of
  `documents/genes.csv` absent from the codebook no longer exists to raise.
* **New keys `codebook` and `direct_panel`** (Python only; absent or null:
  `<input root>/genes.csv`, as before): the codebook of readout_mode `multiplexed` or the
  direct panel of readout_mode `direct`, absolute or relative to the input root. The key of
  the other mode raises `ValueError`. The Python sequencing rules declare the file as their
  codebook input, and `_run_workflow`, `_export_dataset` (the codebook of `export_sample`)
  and `reads_assignment` load it. The MATLAB rules keep `INPUT/genes.csv`, and `common.smk`
  refuses the two keys (and `image_dir_template`) with `backend: matlab`.
* **New `image_dir_template`** (the workflow key, Python only, and the `Dataset` field;
  default `"{round}/{fov}"`): the layout of one round of one FOV under the input root,
  `<round>/<fov>` or `<fov>/<round>`. Any other value raises `ValueError` when the
  `Dataset` is built, and the schema refuses it. `FOV.input_dir` follows it, and through
  it `FOV.load_images` (the reference stains included) and the benchmark pipeline
  adapter's input links; so do the Python rules' image inputs (`_image_dir_inputs` of
  `starfinder._snakemake`: `rsf_single_fov`, `gr_single_fov_subtile`,
  `deep_create_subtile`, `nuclei_registration`) and `get_dapi_input` of
  `registration-py.smk`. Channel matching and `source_paths` are unchanged; `fovs.csv`
  stays the membership. The MATLAB rules and loader stay round-first.
* **Tests.** The tests that linked the FOV-first synthetic e2e data into the round-first
  layout (`conftest.py`'s `e2e_result`, `test_fov.py`'s `small_pipeline_dataset`, the two
  streaming tests of `test_e2e.py`) read it in place with `image_dir_template="{fov}/{round}"`.

### The route of a saved assignment writes the saved folder alone; opt-in assignment previews

W-568 (OM-10, OM-16 choice A) removes the per-FOV files that nothing reads on the route of
a saved assignment (an `assembly` block whose `assignment` names a folder) and makes the
assignment figures opt-in. Existing output trees are not rewritten, and nothing is written
twice. Configurations without an `assembly` block (the legacy route, both backends) and
with `assignment: null` (molecules only) write the same files as before.

| Output | Before | After |
| --- | --- | --- |
| `expr/{fovID}/raw.h5ad`, `expr/{fovID}/reads_assignment.csv` | written by `reads_assignment` on every route; inputs of `export_sample` on the route per FOV | not written on the route of a saved assignment; `export_sample`'s per-FOV inputs are the saved folders' `assignment.json` alone, and the final targets list the saved folders |
| `expr/{fovID}/log.txt` | written on every route | not written on the route of a saved assignment: `assignment.json`'s `counts` holds the same numbers |
| `fovs/{fovID}/assignment/<name>/` | written and declared beside the files above | the only output `reads_assignment` writes and declares on that route |
| `output/tile_config_{sample}.csv` | written and declared by the BigStitcher `stitch_sample`, also with an `assembly` block, where nothing reads it | written and declared only without an `assembly` block |
| `expr/{fovID}/assignment.png`, `assignment_single_layer.png` | written by `reads_assignment` by default, then overwritten with the ownership region by `export_sample`, declared by neither | on the route of a saved assignment, written only with `previews: true`: the local figures in `fovs/{fovID}/previews/` (by `reads_assignment`), the ownership figures in `export/<sample>/previews/{fovID}/` (by `export_sample`); each folder a declared directory output of its one writer (the export folder on the route per FOV with `assemble` on, created empty when the assemble part could not run) |

* **New key `assembly.previews`** (Boolean, default `false`; Python only, in the schema and
  `_assembly_block`): turns on both preview kinds on the route of a saved assignment. It does
  not change molecules only, whose figures stay in `expr/{fovID}/`, or the legacy route.
* **`check_export` skips `previews/`** in its R2 listing check: the ownership previews are
  drawn after the export and `export.json` does not list them. No export file, record or
  table changes.
* **Removed: `workflow/scripts/sample_preview.py`**, a stale notebook export that no rule,
  test or doc referenced and that could not run as written: it imports nothing it uses,
  reads `expr/sample4_raw.h5ad` with a hard-coded sample and parses the percent and coverage
  lines of the legacy `log.txt`, a format that no longer exists.
* **New private helpers** in `starfinder._snakemake`: `_reads_assignment_outputs` (the
  outputs `reads_assignment` declares with an `assembly` block, and the final targets) and
  `_export_preview_outputs` (the preview folder `export_sample` declares); and in
  `starfinder._layout`: `PREVIEWS`, `fov_previews_dir` and `export_previews_dir`.
* **Tests.** The tests that listed the route's files follow: the export inputs
  (`test_stitching_workflow.py`, `test_sample_segmentation_workflow.py`'s
  `per_fov_inputs`), the figures (`test_workflow_assembly_figures.py`,
  `test_sample_segmentation_workflow.py`'s `test_wf2_the_assignment_figures`, now with
  `previews: true`, and `test_declared_grid_header.py`, which turns the previews on to keep
  comparing the figures), the declared-grid workflow comparison
  (`test_workflow_declared_grids.py` compares the saved `molecules.csv` instead of
  `reads_assignment.csv`), the assembly schema keys, and the BigStitcher end-to-end test,
  which runs with and without an `assembly` block.

### One Python configuration dialect

W-569 (OM-12, SM-16, OM-02, OM-15, OM-11) makes the Python backend read one configuration
dialect and only the inputs its rules use. The MATLAB scripts, the MATLAB rules, their JSON
copy and MATLAB's schema definitions are unchanged. Each change has no alias and no dual read.

* **The MATLAB keys of the processing rules raise under `backend: python`.** Each key below,
  in `rules.<rule>.parameters` of `rsf_single_fov`, `gr_single_fov_subtile`,
  `lrsf_single_fov_subtile`, `deep_create_subtile` or `deep_rsf_subtile`, raises `ValueError`
  in `from_workflow_config` and when `common.smk` is parsed, naming its native form, and the
  schema's backend guard refuses it (also `tophat`, which the guard did not cover). The check
  does not depend on a `run` flag (a disabled block, or a rule that does not run) and covers the
  nested positions: each `spot_finding.channel_overrides` entry; the MATLAB parameter and block
  names and the demons variant names in each `registration.steps` entry and recovery
  alternative; the MATLAB image representations as a signal `mode`; and MATLAB's `radius` in a
  `reconstruction` or `white_tophat` step (`_check_python_dialect`). Migrate each key to its
  native form; every default is written out, so the effective `PipelineConfig` is the same:

  | Legacy key or value | Native form (`backend: python`) |
  | --- | --- |
  | `enhance_contrast: {run: true, snr_threshold: s}` (or the rule-level `snr_threshold: s`) | `preprocessing.steps` entry `{method: min_max_normalization, output_dtype: uint8, output_range: [0, 255], rounding: truncate, snr_threshold: s}` (`snr_threshold` omitted when unset) |
  | `hist_equalize: {run: true, reference_channel: c}` (default 0) | `{method: histogram_matching, reference_channel: c}` |
  | `morph_recon: {run: true, radius: r}` (default 3) in `rsf_single_fov`, `gr_single_fov_subtile`, `deep_create_subtile` | `{method: reconstruction, radius_yx: r}` in `preprocessing.steps` |
  | `morph_recon` in `lrsf_single_fov_subtile`, `deep_rsf_subtile` | the same entry in the new `preprocessing.post_registration` list (`steps: []` when nothing runs before registration) |
  | `tophat: {run: true, radius: r}` (default 3) | `{method: white_tophat, radius_yx: r}` |
  | the fixed order normalize, histogram, reconstruction, top-hat | the order of `preprocessing.steps` |
  | `global_registration: {run: true, method: m}` (default `translation`) | a `registration.steps` entry `{method: m}` |
  | `local_registration: {run: true, method: m}` (default `demons`) | a `registration.steps` entry after the global ones |
  | method `diffeomorphic`, `symmetric`, `fast_symmetric` | `{method: demons, variant: <name>}` |
  | `ref_img`/`mov_img: merged-image` (or `merged`) | `registration.signal: {mode: max}` (the default) |
  | `ref_img`/`mov_img: single-channel` with block `ref_channel: c` (default 0) | `registration.signal: {mode: channel, reference_channel: c}`; a second block with another signal becomes that step's own `signal` |
  | block `boundary_mode: nearest` | `registration.warp: {backend: scipy, boundary_mode: nearest}`; `constant` (or none) is no `warp` |
  | block `ref_round` | `registration.reference_round`, or nothing (both are checked against `ref_round`) |
  | block `recovery` with alternatives without `method` | step `recovery`, each alternative naming its `method` |
  | `detection_threshold`, `match_distance`, `tps_smoothing`, `grid_spacing`, `beta`, `lmbda`, `cpd_w`, `candidate_radius`, `k_neighbors` | `detection_noise_sigma`, `match_distance_voxels`, `smoothing`, `grid_spacing_voxels`, `kernel_width_voxels`, `regularization_weight`, `outlier_fraction`, `candidate_radius_voxels`, `neighbors_per_anchor` |
  | a `cpd` block without `detection_threshold` or `grid_spacing` (hidden defaults 3.0 and 32) | `detection_noise_sigma: 3.0`, `grid_spacing_voxels: 32` written out (the native defaults are 5.0 and 16) |
  | `ref_img: single-channel` in `rsf_single_fov` with `global_registration` alone (the single-channel reference image) | the new rule parameter `reference_image_view: {mode: channel, channel: c}` (an index or a channel label; default `{mode: max}`, the channel maximum) |
  | local_maxima's `intensity_estimation`, `intensity_threshold`, `min_distance` (also in `channel_overrides`) | `threshold_mode`, `threshold_value`, `min_distance_voxels` (`min_distance` stays the native field of `spotiflow` and `piscis`) |
  | top-level `dapi_round` (the stain `*ch04.tif`) | the new top-level `reference_stains: [{channel: ch04, name: DAPI}]` |

  `spot_finding.ref_round`, `load_codebook.split_index` and the `reads_filtration` keys are
  shared keys, not aliases, and stay (the zero-based `split_index` is a separate follow-up).
* **The MATLAB-only translation moves to `starfinder.dataset._legacy_config`.** The private
  `_LEGACY_PREPROCESSING`, `_DEMONS_VARIANTS`, `_SPOT_ALIASES`, `_SIGNAL_MODES`,
  `_registration`, `_legacy_registration` and `_legacy_recipe` leave `dataset/workflow.py`,
  as does `_NUCLEI_REFERENCE_CHANNEL` (now `_legacy_config._MATLAB_STAIN_CHANNEL`).
  `from_workflow_config` imports the module only for a MATLAB-backend configuration (backend
  absent or `matlab`), which the adapters of the shared rules read
  (`reads_assignment` loads its codebook with the first rule's layout); such a configuration
  translates as before. The Python backend's refusal table is `_NATIVE_FORMS`.
* **New native forms.** `preprocessing.post_registration` (a list of `reconstruction` steps
  without `save_as`, carried to `PreprocessingRecipe.post_registration`) and the rule
  parameter `reference_image_view` (`{mode: max | channel, channel}`; `WorkflowConfig`'s
  `reference_image` and `reference_channel` follow it; the resident subtile rules refuse it).
* **No JSON copy for Python (OM-02).** The Python rules `rsf_single_fov`,
  `gr_single_fov_subtile`, `deep_create_subtile`, `nuclei_registration`,
  `lrsf_single_fov_subtile` and `deep_rsf_subtile` no longer take
  `<config_path with .json>`, and their inputs are named: `codebook`, `images` and, in the
  resident subtile rules, `subtile`. `_run_workflow` reads `snakemake.input.codebook` and
  `snakemake.input.subtile` (they were `input[1]` and `input[2]`). `rsf_preparation`, which
  writes the copy, is defined and a local rule for the MATLAB backend only. `run.json`
  records the source file as before (`workflow_config`). The unused committed copy
  `tests/tissue_2D_test.json` is removed.
* **Explicit reference stains (OM-11).** The Python backend declares the reference round's
  stain files with the new top-level `reference_stains` (`{channel, name}` entries, optional
  `wavelength`; `Dataset.reference_stains`). `dapi_round` raises for it, naming the new key;
  there is no hard-coded `ch04` and no fallback to it. The Python `rotate_nuclei`
  (`get_dapi_input`) reads `<ref_round>`'s file of the stain, and `nuclei_registration`
  registers against it (label `<ref_round>:<channel>`), the one entry or, with several, the
  one whose name contains `ref_channel` (`_nuclear_stain`). MATLAB's `dapi_round`,
  `*ch04.tif` and rules are unchanged.
* **Conditional image dimensions and Fiji (OM-11, OM-15).** `img_row` and `img_col` are
  required for the MATLAB backend as before, and for the Python backend only by an active
  `create_subtiles` block (the subtile windows) and by stitching method `bigstitcher` without
  an `assembly` block (the legacy tile configuration writer); missing there, the schema and
  the adapters raise naming them. `fiji_path` is read at parse time only by the legacy
  `run_BigStitcher_macro` (`stitching.smk`, included for the MATLAB backend only). The
  minimal example drops `envs_path` and `fiji_path` (and `prepare_workflow.py` sets them only
  where a template has them). `common.smk`'s MATLAB branch reads the subset keys with `.get`.
  The unused `run_fiji_macros` (`common.smk`) and the private `_legacy_stitching`
  (`dataset/workflow.py`, used only by a test) are removed.
* **Migrated configurations.** `docs/examples/workflow-full.yaml`, `workflow-minimal.yaml`
  and `workflow-recipe-2.yaml`; `benchmarks/recipes.py` (the pipeline case's workflow is a
  `backend: python` configuration of native keys; its unused `fov_id_pattern` is dropped) and
  the registration steps of `benchmarks/configs/pipeline.json` (native step entries with
  `signal`). `tests/minimal_config.yaml` and `tests/tissue_2D_test.yaml` are MATLAB
  configurations and keep the MATLAB keys.
* **Tests.** The Python-backend test configurations use the native keys and named rule
  inputs; tests whose configurations have no `backend` (MATLAB ones) keep testing the
  translation, now `_legacy_config`'s, and several also assert the native form gives the same
  result. The tests of the removed `_legacy_stitching` and of the Python backend's acceptance
  of `tophat` are replaced by the refusal tests.

### The processing report

* **New: the processing report.** `Dataset.report(sample, fov_ids, …)` writes
  `<output_root>/report/<sample>/` (`report.html`, `report.json`, `tables/`) and returns a
  `ReportRecord` ({doc}`report-contract`, "The sample entry"); `starfinder.report.load_report`
  reads a report back. The entry reads the saved records only and calls no processing stage.
  Nothing existing changes.
* **New: the comparison page.** `Dataset.compare_samples(samples, *, report_root=None)` writes
  `<output_root>/report/index.html` and `report/samples.csv` from the finished sample reports
  only and returns a `ComparisonRecord` ({doc}`report-contract`, "The comparison entry"); a
  sample without a report is a row marked `missing`. Nothing existing changes.
* **New: the workflow's `report` block and rules.** A Python-only top-level `report` block
  whose keys are the `ReportConfig` fields ({doc}`workflow-configuration`, "Processing
  report"), and with it the new rule file `workflow/rules/report.smk`: `report_sample` per
  sample (`Dataset.report`, declared output `report/<sample>/report.json`, after the sample's
  `export_sample`, else its `stitch_sample`, else its FOVs' processing rule) and
  `report_dataset` (`Dataset.compare_samples` over every sample, declared output
  `report/samples.csv`). A configuration without the block gains no rule and no output.

## Intentional behavior changes — not mechanical equivalence

| Area | Change and consequence |
| --- | --- |
| I/O / preprocessing | Preserve loaded dtype; conversion, cropping and channel selection are explicit. Constant normalization groups map to the lower endpoint even with SNR gating. Float64 computation can change quantization boundaries. Slice morphology avoids uint16 signed overflow; projection preserves singleton Z and uses wider sums without display scaling. |
| Detection | Singleton-Z local maxima operate in YX. Empty tables are typed, identities/geometry explicit. Distinct landmark and pipeline detector policies remain distinct. Local maxima now emits a `SpotFindingWarning` when a channel's noise MAD is 0 or more than half of its voxels are zero, where nothing was reported before; thresholds and detections are unchanged. Subclasses of the detection configs are rejected (exact-type lookup), and for `local_maxima` a legacy YAML key together with its field raises instead of one silently winning. |
| Translation | Singleton axes return zero; odd-length peak wrapping is corrected. Signed fractional Fourier output uses the real inverse FFT rather than magnitude. Even half-period backend signs and Nyquist behavior are documented rather than hidden. |
| Transform application | Integer output rounds once with nearest-even ties and saturation; floating output retains signed interpolation/overshoot. No silent method fallback; unsupported geometry/backend/dimensions fail explicitly. |
| Barcodes | Validate codebook collisions and label alignment; neighborhoods use explicit ZYX radii and subpixel/boundary policy. Preserve ambiguous/unmatched/rejected identities instead of dropping them. Scores and endpoint filtering have explicit meanings. |
| Coordination | Rectangular subtiles cover both axes and remainders. Batch/streaming honor the same stage flags, unlike legacy forced/omitted stages. |
| Registration signal | `merged` and `merged-image` mean the channel maximum, as in MATLAB, not the float64 channel sum; the default `RegistrationSignalConfig` is `mode="max"`, and the workflow's local `ref_img`/`mov_img` default is `merged-image`, not `single-channel`. Use `mode="sum"` for the earlier Python behavior. Different `ref_img` and `mov_img` in one block are rejected. |
| Registration resampling | A multi-step recipe composes its steps into one pull map and resamples each image of a moving round once from its pre-registration array (linear SciPy by default). Results differ from the earlier per-step resampling at the boundary, where an intermediate image had been sampled outside its grid, and by integer rounding; an integer translation no longer rounds a later step. The registration golden test re-pins the translation → demons image for this reason. |
| Synthetic | One amplicon-scene generator with keyed SHA-256/PCG64 streams: byte-repeatable across processes for a pinned NumPy build on the same CPU, but every image and truth record differs from the historical generator. Appearance defaults are uncalibrated and do not establish molecular truth. |
| Evaluation | Centered NCC has no epsilon bias. Missing/failed shifts, zero denominators and constant images are undefined rather than zero/passing. Shift errors preserve floats; matching thresholds/policies are explicit. |
| Benchmark / recipes | Failures retain requested/actual method identity. Evaluation/reporting reuse saved artifacts. Optional legacy experiments remain recipes with prerequisites, not validated research results. |
| Segmentation / assignment workflow | The shared rules call the package under both backends: `uint32` labels, `rescale` on the input grid, no foreground gate, cells kept without molecules, `outside_grid` molecules, unknown genes and a double expansion raise, new `raw.h5ad` and `reads_assignment.csv` columns; MATLAB-backend runs need the package and its `stardist` and `anndata` extras. |
| Sample assembly workflow (configurations with an `assembly` block only) | Option W1 of {doc}`assembly-contract`: the per-FOV `raw.h5ad` and `reads_assignment.csv` are not box-filtered and have no `global_*` columns (every kept cell, and every molecule in input order); the label file is imported on the grid of `images/ref_merged/{fovID}.tif` instead of one declared from `voxel_size_xy` and `voxel_size_z`; the sample outputs are in `export/<sample>/` instead of `expr/{sample}_raw.h5ad` and `expr/{sample}_reads_assignment.csv`; cells and molecules are retained by the ownership regions of the placement instead of the `tile_config` rectangles, and every excluded observation is kept with its reason. Configurations without the block are unchanged. |
| FOV folder and reference image (W-564) | A FOV's durable results move to `fovs/<fov>/` and its stage files to `fovs/<fov>/checkpoints/`; earlier output roots are not read in the old places. `overwrite=False` no longer refuses a FOV folder that holds only other calls' results, and `overwrite=True` clears only `checkpoints/`. The Python reference image moves to `fovs/<fov>/reference.tif`, and a dataset with `reference_image_import` writes none. A channel pattern that is not a safe path component now raises when the `Dataset` is built. See "The FOV folder, the reference image and channel folders by label". |
| Sample segmentation route (configurations and calls that export a projection with an assignment) | Rule RD1 of {doc}`sample-segmentation-contract`: without `route`, a sample exported as a projection (the default `dimensionality`) takes the route after fusion. Every existing configuration with `stitching` and `assembly` blocks whose `assembly.assignment` is a name (the default `default`) and whose `export` block has no `route` and no `segmentation` while `dimensionality` is `projection` (also when the block is absent) stops at once: the workflow refuses it when the rules are parsed, before any job is planned, and `Dataset.export` with the same `ExportConfig` refuses it before anything is written, with `ValueError: the route after fusion (route: after_fusion, the default for dimensionality 'projection') segments the fused projection and needs export.segmentation, the settings of that segmentation; set export.segmentation, or route: per_fov to assemble the per-FOV assignments`. The one line `route: per_fov` in the `export` block (`ExportConfig(route="per_fov")` in Python) restores the earlier behaviour: the same jobs and the route per FOV. The route after fusion also refuses `dimensionality: stack` and `projection: sum`. Exports with `route: per_fov` or `dimensionality` `stack` or `both`, and samples without segmentation (`assignment` null), keep their route; their outputs change only by the column `cell_namespace`, the part `segment` recorded `off` with its `segmentation` section, and the keys `route` and `route_source` of the `assembly` section. |

Detailed numerical policies, tolerances and edge cases remain canonical in
[contracts](api/contracts.md), [evaluation](api/evaluation.registration.rst),
[synthetic](api/synthetic.rst) and [benchmark recipes](benchmark-recipes.md).
This guide does not claim bitwise equivalence or MATLAB runtime validation.
Scientific qualification remains separate; notably, molecular truth and
calibration of synthetic appearance remain unresolved.
