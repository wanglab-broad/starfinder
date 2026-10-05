# Python migration guide

This is a coordinated breaking Python cleanup: update maintained imports and
callers together. Replaced interfaces have no compatibility aliases. **FOV stays
FOV.** MATLAB APIs, shared MATLAB-facing workflow keys, CSV coordinate bases and
filenames remain unchanged. Historical scripts and outputs retain their original
provenance; they are not examples of the current API.

## Mechanical names and ownership

| Before | Current owner/interface |
| --- | --- |
| `dataset.STARMapDataset`, `LayerState`, `layers` | `dataset.Dataset`, `RoundState`, `rounds` |
| `dataset.Codebook` | `barcode.Codebook` |
| `io.load_multipage_tiff`, `load_image_stacks`, `save_stack` | `io.load_volume`, `load_round`, `save_volume` |
| `min_max_normalize`, `histogram_match`, `morphological_reconstruction`, `tophat_filter` | `preprocessing.normalize_intensity`, `match_histogram`, `reconstruct_background`, `filter_tophat` |
| `utils.make_projection` | `preprocessing.project_image` |
| `spotfinding.find_spots_3d` | `spot_finding.find_spots` |
| `decode_color_seq` | `barcode.decode_color_sequence` |
| `benchmark.synthetic` | `synthetic` |
| `registration.metrics`, benchmark truth comparisons | Pure domain functions in `evaluation` |
| Separate benchmark drivers and `starfinder-generate` | `starfinder synthetic generate`, `starfinder benchmark run/evaluate/report` |

These relocations do not make the old positional arguments or return values valid.
Use typed operation configs and structured results. See the [full export
inventory](api/inventory.rst) and [coordination member map](coordination.md).
Registration internals, benchmark helpers and barcode lookup constants are private.

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
from starfinder.preprocessing import normalize_intensity, MinMaxNormalizationConfig

loaded = load_volume(path)
image = normalize_intensity(loaded.image, config=MinMaxNormalizationConfig(
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
from starfinder.barcode import (NeighborhoodSumConfig, WtaDecoderConfig,
    ReadFilterConfig, extract_intensities, decode_barcodes, filter_reads)

intensities = extract_intensities(rounds, spots,
    config=NeighborhoodSumConfig(neighborhood_radius_zyx=(1, 2, 2)))
decoded = decode_barcodes(intensities, codebook, config=WtaDecoderConfig())
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

### Generate, evaluate and report

`synthetic.generate_dataset(codebook, config, fov_ids=...)` returns arrays and
truth records, or streams each round to a callback. The CLI persists generated
inputs; benchmark cases explicitly select them.
`evaluation` accepts supplied results/truth and explicit matching/units; it does
not rerun algorithms. [Benchmark](benchmark.md) documents immutable processing
runs, checksum validation, and separate saved-output evaluations/reports.

### Generate synthetic data with one generator

The historical generator (`SyntheticConfig`, integer-center `render_spots`,
hash-seeded registration pairs) is removed. Benchmark data and test fixtures
come from the formed-scene generator, with benchmark presets that keep the
historical shapes, amplicon counts, seeds and shift ranges. There are no aliases.

| Before | After |
| --- | --- |
| `SyntheticConfig`, `get_preset_config(name)` | `benchmark_scene_preset(name, dtype=...)` returns `(Codebook, FormedSceneConfig)`; adjust with `dataclasses.replace`; `SCENE_PRESETS` lists every preset |
| `generate_dataset(config, preset=...)` | `generate_dataset(codebook, config, fov_ids=..., preset=..., on_round=...)` |
| `generate_registration_pairs([preset], seed=...)` | `generate_registration_pair(preset, deformation=..., seed=...)`, one call per pair |
| `generate_displacement_field`, `DEFORMATION_CONFIGS` | `deformation_geometry(name, shape)` with `GeometryConfig` affine, polynomial and RBF terms; `forward_displacement(transform, shape)`. Magnitudes are reduced where needed for invertibility; notably `multi_point` uses 1.3–10.5 px instead of 5–20 px below `tissue` (see the synthetic API) |
| `render_spots`, `generate_volume` | `generate_formed_scene` (single scenes) |
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
| `decode_color_sequence(color_seq=...)` | `decode_color_sequence(color_sequence=...)` |

`formed.csv` files written before this change carry `codeword` and are not
supported; regenerate them.

### Registration recipe

`PipelineConfig.registration` is a
{py:class}`~starfinder.dataset.RegistrationRecipe` or `None` (no registration),
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
`format_version` 1; its `config.pipeline.registration` holds the recipe's
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

### Registered checkpoints version 2

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

`FOV.register_rounds(recipe, *, rounds, reference=None)` and
{py:class}`~starfinder.dataset.ExternalReference` are new: other rounds, such
as morphology rounds, register to a reference round or an external reference
image through a shared stain, and the transform is applied to every channel of
the round. A channel label in `RegistrationSignalConfig(mode="channel")` is
looked up in the round's own labels: `Dataset.other_channel_order` (new) gives
an other round its channels, and `FOV.load_images` checks each round against
`Dataset.channel_labels(round)`. Estimation attempts gain `reference_sha256`
(`None` for a round reference). `FOV.save_processing_log` accepts `"nr"`.

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
(§2.8)")`, from `FOV.run` before any processing and from `FOV.decode_barcodes` for a
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
(float64). `FORMAT_VERSION` stays 2. `read_checkpoint(..., "candidates")` rebuilds the
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

`KNOWN_WEIGHTS` lists the pretrained weights Starfinder can fetch and verify
(two Spotiflow 3D, two Spotiflow 2D and two Piscis models, with full SHA-256
values) and sets no default. `starfinder weights fetch <method> <model>`
(`fetch_weights`) downloads, verifies and installs a model under
`STARFINDER_WEIGHTS_DIR` (else `$XDG_CACHE_HOME/starfinder/weights`);
`starfinder weights list` and `starfinder weights verify` show and re-hash the
local copies. `resolve_weights` raises `MissingWeightsError` or
`WeightsHashMismatchError`. Detection never downloads.

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
names its weights (`model`, a row of `KNOWN_WEIGHTS`; there is no default), and an
unknown model or one of the other method raises `ValueError` listing the known
names. Detection loads only the copy fetched with `starfinder weights fetch`: each
call re-hashes every file `KNOWN_WEIGHTS` lists for the model before the model is
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
`KNOWN_WEIGHTS` entries gain `extracted`, every file of a Spotiflow archive with
its SHA-256; `resolve_weights(..., extracted=...)` also checks the named files, and
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

`run.json` keeps `format_version` 1. Starfinder does not read `config.pipeline`
back from `run.json`, so records written before the rename stay valid as they
are; a script that reads them should accept either key. The `candidates.json`
keys `detection_config`, `detection_rounds`, `detection_plan` and
`detection_diagnostics` and the preprocessing `detection` image keep their names.

### Readout mode

§2.8 adds two readout modes as one dataset setting ({doc}`readout-contract`,
"Readout modes" and "Direct readout"). `Dataset.readout_mode` is `"multiplexed"`
(the default: every earlier result, identity and digest is unchanged) or
`"direct"`, where a round and channel identify a gene.
{py:class}`~starfinder.barcode.DirectPanel` holds the gene of each (round, channel),
{py:func}`~starfinder.barcode.load_direct_panel` reads it from a
`round,channel,gene_id` CSV, and `Dataset.load_direct_panel` stores it as
`Dataset.direct_panel`; a repeated gene or (round, channel) raises `ValueError`
naming it. {py:class}`~starfinder.barcode.DirectAssignmentConfig` (`direct` in
`DECODING_METHODS`, no parameters) and
{py:func}`~starfinder.barcode.assign_direct` assign each candidate the gene of its
own round and channel: one read per candidate with `round`, `channel`, `entry_id`
`"<round>/<channel>"`, `call_type` `direct`, and the diagnostic columns
`own_channel_rank` and `own_channel_fraction`; a (round, channel) without a gene is
`unmatched` with `unmapped_channel`, and a zero own round `no_signal`. Nothing is
merged or reassigned. `BarcodeDecodingResult` gains `readout_mode`.

In `direct` mode, `extract_intensities(..., readout_mode="direct")` (which
`FOV.run` and `FOV.extract_intensities` pass from the dataset) reads each candidate
in its own round only; the other rounds are `0.0` with `valid=False`.
`IntensityExtractionResult` gains `box_voxels` (N×R int64, the voxels each box
summed, 0 for a round that was not read; since W-294 a candidates checkpoint
stores it with the background measurements, and a checkpoint without them reloads
it as `None`). `FOV.run` and `FOV.decode_barcodes`
assign with `DirectAssignmentConfig` and `Dataset.direct_panel`.

Decoding a multi-round candidate set (a `round` column) needs
`readout_mode="direct"`: in `multiplexed` mode it still raises `ValueError`, whose
message keeps "needs a readout mode (§2.8)" and adds "set readout_mode='direct'".
A decoder that does not support the dataset's mode (`wta` or `codebook_aware` in
`direct` mode, `direct` in `multiplexed` mode) raises `TypeError` naming the mode
and the decoder, in `FOV.run`, `FOV.decode_barcodes`, `decode_barcodes` and
`assign_direct`; `direct` mode without a `round` column, or without a loaded panel,
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
`starfinder.barcode.ENCODINGS` maps each encoding config type to its
`EncodingSpec`: `two_base` is {py:class}`~starfinder.barcode.EncodingConfig`, which
gains the discriminator `method="two_base"` (its positional constructor is
unchanged), and `one_base` is the new
{py:class}`~starfinder.barcode.OneBaseEncodingConfig`, whose `base_to_color` maps A,
C, G and T one-to-one onto the colors 1–4 and has no default.
`starfinder.barcode.DECODING_METHODS` maps `WtaDecoderConfig` (`wta`) and
`CodebookAwareDecoderConfig` (`codebook_aware`) to their `DecodingSpec`, with the
readout modes, encoding kinds, rescue and score columns each declares. Lookups use
the exact config type, so `decode_barcodes` rejects a subclass of a decoder config
with `TypeError("unsupported decoder config")`, and it raises `TypeError` when the
codebook's encoding kind is not one the decoder declares. The decoders' numerical
behavior is unchanged. `Codebook(encoding=...)` accepts any registered encoding
config; `Dataset.load_codebook` gains keyword-only `encoding` and `layout`.

### Encoding table

W-304 makes the `two_base` pair-to-color table visible, configurable and recorded
({doc}`readout-contract`, "Encoding registry"). {py:class}`~starfinder.barcode.EncodingConfig`
gains `pair_to_color`, the 16 ordered base pairs mapped to the colors `"1"` to `"4"`,
whose default is the active table of `src/matlab/EncodeBases.m`; `EncodingConfig()`
and every default result are unchanged, the positional constructor is unchanged,
and the module functions `encode_bases` and `decode_color_sequence` keep the default
table. A table whose keys are not the 16 pairs, whose values are not `"1"` to `"4"`,
or in which the four pairs of one first base do not have four different colors
raises `ValueError`. Every encode and decode path follows the codebook's table,
including the `end_bases` shortcut of `filter_reads` when the codebook is given.
`EncodingSpec` gains the required keyword field `table`;
{py:meth}`~starfinder.barcode.Codebook.encoding_table`,
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

{py:class}`~starfinder.barcode.BarcodeLayout` and
{py:class}`~starfinder.barcode.Segment` describe how a barcode is cut into
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
"Background and noise"). {py:class}`~starfinder.barcode.LocalBackgroundConfig`
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

{py:func}`~starfinder.barcode.score_reads` (`reference` is the codebook, or the
direct panel in readout mode `direct`) returns a
{py:class}`~starfinder.barcode.ReadScoringResult`: the read table with the columns
`qc_score`, `qc_ambiguity_max`, `qc_signal_to_background`, `qc_rounds` and
`qc_reason` appended ({doc}`readout-contract`, "Shared read-QC score"). `qc_score`
(W-278 design D1) is the probability NLL of the assigned entry recomputed on
background-subtracted sums; lower ranks as more reliable. It is a ranking, not a
calibrated probability; it sets no cutoff and never changes `gene_id`, `entry_id`,
`call_status` or `call_type`. Reads without an assignment have NaN with
`no_assignment`, and reads with a round without background NaN with
`background_unavailable`; scoring intensities that have no background raises
`ValueError` naming extraction. {py:class}`~starfinder.barcode.ReadScoreConfig`
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

{py:func}`~starfinder.barcode.deduplicate_reads` (`DeduplicationConfig`,
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

`starfinder.evaluation.barcode` adds `ranking_quality(score, correct, *,
orientation, retention=(0.5, 0.8, 0.9, 1.0))`: the AUROC of a score with its
Hanley–McNeil standard error and the error at fixed retention, and
`evaluate_deduplication(groups, source, *, pairs)`: missed duplicates, false
merges and their rates over a stated pair population. Both report an undefined
value with a reason when a class is empty. They rank and count; they set no
cutoff.

### Read diagnostics

`starfinder.barcode` adds the three diagnostics of {doc}`readout-contract`
("Diagnostics"), which read retained results only.
{py:func}`~starfinder.barcode.inspect_read` returns one read's sums, background,
background-subtracted sums, channel probabilities, noise, observed and assigned
colors and per-segment bases, one row per round and channel, and
{py:func}`~starfinder.barcode.plot_read` draws it.
{py:func}`~starfinder.barcode.summarize_reads` returns the population summary (counts
by status, reason and call type, per gene and per entry, `qc_score` quantiles per call
type, deduplication and filtering counts, per round and channel medians, valid and
background-unavailable counts, cross-channel pairs).
{py:func}`~starfinder.barcode.explain_read` returns a read's ordered decisions with
their values and limits. They accept a read result or `FOV.results`. `FOV.run` never
calls `inspect_read`, `plot_read` or `explain_read`; when it scores or deduplicates,
`run.json` records the summary under `counts["summary"]`, and
`FOV.save_diagnostics` now writes it as `summary` beside the filtering counts.

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
from a TIFF such as `images/ref_merged/{fovID}.tif`.

Masks made elsewhere, such as CellProfiler outputs, the legacy
`images/stardist_segmentation` files or the culture references, enter through
{py:func}`~starfinder.segmentation.import_labels` (with
{py:class}`~starfinder.segmentation.LabelImportConfig` for a plan run) instead of a
plain `imread`. Big-endian files are read with native values, float and boolean masks
are rejected, and the shape and any stored metadata are checked against the grid.
{py:func}`~starfinder.segmentation.labels_to_grid` replaces the legacy round trip
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
`FOV.register_rounds` and the reference round's channel maximum, the image
`FOV.save_reference_image` writes. The workflow scripts call these functions (see
"Segmentation and assignment workflow rules" below).

{py:func}`~starfinder.segmentation.normalize_percentiles` is csbdeep's `normalize` as
`stardist_segmentation.py` calls it (percentiles 1 and 99.8, float32, unclipped), with
its values recorded. {py:func}`~starfinder.segmentation.rescale_input` performs the
script's `rescale(image, [1, .5, .5])` shrink for any factors and also returns
metadata whose spacing is divided by the factors and whose `frame_id` records the
rescale; {py:func}`~starfinder.segmentation.labels_to_grid` maps labels detected on the
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
{py:func}`~starfinder.assignment.molecule_table_from_csv`) in the territories of a cell
{py:class}`~starfinder.segmentation.SegmentationResult` and returns an
{py:class}`~starfinder.assignment.AssignmentResult` with the molecule, cell, count and
nucleus tables. `FOV.assign(config, cells=…, nuclei=…)` runs it on the FOV's results and
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

`FOV.segment(plan, checkpoints=CheckpointConfig(…))` and `FOV.assign(…, checkpoints=…)`
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
checkpoint `FORMAT_VERSION` 2 and `run.json` (`format_version` 1) stay as they are, and
`FOV.assign` never rewrites a file of `FOV.run` or of segmentation (a saved label image is
linked by its relative path). The new records carry their own `format_version` 1.
`checkpoints` other than `None` or a `CheckpointConfig` now raises `TypeError` in
`FOV.segment` and `FOV.assign`, as in `FOV.run`, where these two calls raised
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
runs with `FOV.segment` and `FOV.assign` (the legacy rules raise when a block is
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
  record and, lying outside the tile box, not written to `reads_assignment.csv`.
* A goodSpots gene outside the codebook raises `ValueError` naming it before
  assignment, where the script silently left it out of the counts; `documents/genes.csv`
  must list the codebook's genes.
* A label file expanded by `stardist_segmentation` (`expand_labels: true` there) raises,
  naming the key: give the distance as `reads_assignment.parameters.dilation_distance`
  with `expand_labels: true`, so assign expands once and keeps both masks.
* `raw.h5ad` gains the obs columns `size_voxels`, `expanded_size_voxels`,
  `size_physical`, `centroid_z/y/x`, `n_molecules`, `n_nuclei`, `correspondence`,
  `correspondence_flags` and `compartments`, the record in `uns["assignment"]` (JSON
  text), and, with nuclei, the `nucleus` and `cytoplasm` layers (NaN where compartments
  are not available). `reads_assignment.csv` gains `spot_id`, `assignment_status`,
  `cell_id`, `in_expansion`, `original_cell_id`, `nucleus_id` and `compartment`.
* `assignment.png` (`plot_assignment`) and a new `log.txt` replace the four diagnostic
  PNGs and the coverage log.

The legacy `obs` columns keep their meaning, computed on the territories assign samples,
and the tile configuration, global coordinates and overlap filter are applied outside the
package as before (§2.10).

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
| Synthetic | One formed-scene generator with keyed SHA-256/PCG64 streams: byte-repeatable across processes for a pinned NumPy build on the same CPU, but every image and truth record differs from the historical generator. Appearance defaults are uncalibrated and do not establish molecular truth. |
| Evaluation | Centered NCC has no epsilon bias. Missing/failed shifts, zero denominators and constant images are undefined rather than zero/passing. Shift errors preserve floats; matching thresholds/policies are explicit. |
| Benchmark / recipes | Failures retain requested/actual method identity. Evaluation/reporting reuse saved artifacts. Optional legacy experiments remain recipes with prerequisites, not validated research results. |
| Segmentation / assignment workflow | The shared rules call the package under both backends: `uint32` labels, `rescale` on the input grid, no foreground gate, cells kept without molecules, `outside_grid` molecules, unknown genes and a double expansion raise, new `raw.h5ad` and `reads_assignment.csv` columns; MATLAB-backend runs need the package and its `stardist` and `anndata` extras. |

Detailed numerical policies, tolerances and edge cases remain canonical in
[contracts](api/contracts.md), [evaluation](api/evaluation.registration.rst),
[synthetic](api/synthetic.rst) and [benchmark recipes](benchmark-recipes.md).
This guide does not claim bitwise equivalence or MATLAB runtime validation.
Scientific qualification remains separate; notably, molecular truth and
calibration of synthetic appearance remain unresolved.
