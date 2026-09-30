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

## Intentional behavior changes — not mechanical equivalence

| Area | Change and consequence |
| --- | --- |
| I/O / preprocessing | Preserve loaded dtype; conversion, cropping and channel selection are explicit. Constant normalization groups map to the lower endpoint even with SNR gating. Float64 computation can change quantization boundaries. Slice morphology avoids uint16 signed overflow; projection preserves singleton Z and uses wider sums without display scaling. |
| Detection | Singleton-Z local maxima operate in YX. Empty tables are typed, identities/geometry explicit. Distinct landmark and pipeline detector policies remain distinct. |
| Translation | Singleton axes return zero; odd-length peak wrapping is corrected. Signed fractional Fourier output uses the real inverse FFT rather than magnitude. Even half-period backend signs and Nyquist behavior are documented rather than hidden. |
| Transform application | Integer output rounds once with nearest-even ties and saturation; floating output retains signed interpolation/overshoot. No silent method fallback; unsupported geometry/backend/dimensions fail explicitly. |
| Barcodes | Validate codebook collisions and label alignment; neighborhoods use explicit ZYX radii and subpixel/boundary policy. Preserve ambiguous/unmatched/rejected identities instead of dropping them. Scores and endpoint filtering have explicit meanings. |
| Coordination | Rectangular subtiles cover both axes and remainders. Batch/streaming honor the same stage flags, unlike legacy forced/omitted stages. |
| Registration signal | `merged` and `merged-image` mean the channel maximum, as in MATLAB, not the float64 channel sum; the default `RegistrationSignalConfig` is `mode="max"`, and the workflow's local `ref_img`/`mov_img` default is `merged-image`, not `single-channel`. Use `mode="sum"` for the earlier Python behavior. Different `ref_img` and `mov_img` in one block are rejected. |
| Registration resampling | A multi-step recipe composes its steps into one pull map and resamples each image of a moving round once from its pre-registration array (linear SciPy by default). Results differ from the earlier per-step resampling at the boundary, where an intermediate image had been sampled outside its grid, and by integer rounding; an integer translation no longer rounds a later step. The registration golden test re-pins the translation → demons image for this reason. |
| Synthetic | One formed-scene generator with keyed SHA-256/PCG64 streams: byte-repeatable across processes for a pinned NumPy build on the same CPU, but every image and truth record differs from the historical generator. Appearance defaults are uncalibrated and do not establish molecular truth. |
| Evaluation | Centered NCC has no epsilon bias. Missing/failed shifts, zero denominators and constant images are undefined rather than zero/passing. Shift errors preserve floats; matching thresholds/policies are explicit. |
| Benchmark / recipes | Failures retain requested/actual method identity. Evaluation/reporting reuse saved artifacts. Optional legacy experiments remain recipes with prerequisites, not validated research results. |

Detailed numerical policies, tolerances and edge cases remain canonical in
[contracts](api/contracts.md), [evaluation](api/evaluation.registration.rst),
[synthetic](api/synthetic.rst) and [benchmark recipes](benchmark-recipes.md).
This guide does not claim bitwise equivalence or MATLAB runtime validation.
Scientific qualification remains separate; notably, molecular truth and
calibration of synthetic appearance remain unresolved.
