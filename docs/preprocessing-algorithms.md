# Preprocessing reference

The methods, options, records and files of preprocessing. The method and option
tables are read from the code when the site is built. The
{doc}`preprocessing guide <preprocessing-contract>` explains how to use them; the
{doc}`preprocessing API <api/preprocessing>` documents every class.

## Methods

```{method-table} starfinder.preprocessing.PREPROCESSING_METHODS
```

Each method's name is its `method` value, used by the workflow key.

## Method options

### Min–max normalization

The legacy pipeline's min–max normalization. See
{py:class}`~starfinder.preprocessing.MinMaxNormalizationConfig`.

```{config-options} starfinder.preprocessing.MinMaxNormalizationConfig
```

### Histogram matching

Matches each channel's intensity distribution to one channel of the reference round.
See {py:class}`~starfinder.preprocessing.HistogramMatchingConfig`.

```{config-options} starfinder.preprocessing.HistogramMatchingConfig
```

### Reconstruction and white top-hat

Both remove background in XY, on each Z plane and channel separately. See
{py:class}`~starfinder.preprocessing.ReconstructionConfig` and
{py:class}`~starfinder.preprocessing.TophatConfig`.

```{config-options} starfinder.preprocessing.ReconstructionConfig
```

```{config-options} starfinder.preprocessing.TophatConfig
```

### Scalar background

Subtracts one background level per channel and round, estimated from the channel's
intensities or read from a supplied-statistics file. See
{py:class}`~starfinder.preprocessing.ScalarBackgroundConfig`.

```{config-options} starfinder.preprocessing.ScalarBackgroundConfig
```

### 3D background

Subtracts a volumetric grey opening, for background that also varies along Z.
"Choose the 3D background radius" in {doc}`recipes` helps choose its size. See
{py:class}`~starfinder.preprocessing.Background3DConfig`.

```{config-options} starfinder.preprocessing.Background3DConfig
```

### Percentile normalization

Maps each channel's range between two percentiles linearly onto the output range,
keeping the input dtype. See
{py:class}`~starfinder.preprocessing.PercentileNormalizationConfig`.

```{config-options} starfinder.preprocessing.PercentileNormalizationConfig
```

## Recipe options

```{config-options} starfinder.preprocessing.PreprocessingRecipe
```

```{config-options} starfinder.preprocessing.PreprocessingStep
```

See {py:class}`~starfinder.preprocessing.PreprocessingRecipe` and
{py:class}`~starfinder.preprocessing.PreprocessingStep`.

## Records

With a recipe, `fov.preprocessing_record` and the `preprocessing` entry of `run.json`
hold:

| Key | Contents |
| --- | --- |
| `recipe` | The method names of `steps` and `post_registration`, `extraction_source` and `registration_source` |
| `rounds` | Per round, one record per step, in order |
| `transforms` | Per round and image (`detection` and each snapshot), the registration transforms applied to it |
| `supplied_statistics` | `path` and `sha256` of the supplied-statistics file; both `null` without one |

Each step record holds:

| Key | Contents |
| --- | --- |
| `index`, `stage` | The step's position, and `steps` or `post_registration` |
| `step`, `config` | The method name and its config |
| `fitted` | The values the step fitted or read (below) |
| `diagnostics` | Per-channel measurements of the output (below) |
| `input_dtype`, `output_dtype` | The image dtype before and after the step |
| `save_as` | The snapshot name, or `null` |

| Method | `fitted` |
| --- | --- |
| `min_max_normalization` | `groups`: per channel (or once for the whole volume), `channel`, `min`, `max` and `mode` (`rescale` or `clip`) |
| `histogram_matching` | `reference_round`, `reference_channel` |
| `reconstruction`, `white_tophat` | Empty |
| `scalar_background` | `background`: one level per channel |
| `background_3d` | `radius_voxels_zyx` |
| `percentile_normalization` | `low`, `high`: one value per channel |

`scalar_background`, `background_3d` and `percentile_normalization` record these
diagnostics of their output; all but `footprint_voxels` are lists with one entry per
channel. The other methods record none.

| Diagnostic | Meaning |
| --- | --- |
| `zero_fraction` | Fraction of voxels equal to 0 |
| `median`, `mad` | Median and median absolute deviation |
| `noise_threshold` | `median + 5 × 1.4826 × mad`, the local-maxima `noise` threshold at its default `threshold_value` |
| `mad_zero` | Whether `mad` is 0 |
| `constant_channel` | Whether the input channel was constant |
| `degenerate_range` | `percentile_normalization` only: whether `low` equals `high` or the channel was constant |
| `footprint_voxels` | `background_3d` only: the number of voxels of the footprint |

## Sample-level statistics

A step with `fit="supplied"` reads its values from the recipe's `supplied_statistics`
file. These functions make the file:

| Function | Does |
| --- | --- |
| `summary_recipe(recipe, step)` | Returns the recipe of the steps before `step`, to run on each FOV, and the record of those steps |
| `summarize_histograms(images, channel_labels=..., fov_id=..., summarized_after=...)` | Records the per-channel histograms of one FOV's rounds |
| `save_histograms`, `load_histograms` | Write and read one summary as `.npz` |
| `merge_histograms(summaries, exclude=...)` | Sums the summaries of the FOVs, leaving out those named in `exclude` |
| `scalar_background_histograms(summary, config, supplied=...)` | Derives the histograms after a scalar background step without another pass |
| `fit_supplied_section(config, merged, reference_round=...)` | Fits one step's section from the merged histograms |
| `supplied_statistics(merged, sections)` | Builds the file's document from the merged histograms and the sections |
| `save_supplied_statistics`, `load_supplied_statistics` | Validate and write, or read and validate, the file |

The file is JSON with the schema `starfinder.preprocessing.supplied/1`:

| Key | Contents |
| --- | --- |
| `schema`, `dtype`, `channel_labels` | The schema identifier, and the image dtype and channel labels it was made from |
| `fovs_used`, `fovs_excluded` | The FOVs merged and those left out |
| `steps` | One section per method name, each with `summarized_after` (the steps run before it), `params` (the fitting parameters) and `fitted` (per round, the method's fitted values; for `histogram_matching`, the reference round's merged `values` and `counts`) |

The guide's [two-pass example](preprocessing-contract.md#fitting-modes) uses them in order.

## Files

| File | Written by | Contents |
| --- | --- | --- |
| `run.json` | `FOV.run` with `checkpoints` | The `preprocessing` record above |
| `checkpoints/registered/<round>.ome.tif` | `FOV.run` with the `registered` stage | The round after its preprocessing steps, any registration and its post-registration steps |
| `checkpoints/registered/<snapshot>/<round>.ome.tif` | the same, when the recipe names an `extraction_source` | That snapshot, registered with the round when registration is set |
| `checkpoints/registered/transforms.json` | the same | Includes the stored snapshot names and the `preprocessing` record |
| Supplied-statistics file | `save_supplied_statistics` | The JSON file above |
| Histogram summary | `save_histograms` | `counts` (rounds, channels, bins), labels, FOVs and the steps run before |

Paths under `checkpoints/` are inside the FOV's folder, `<output_root>/fovs/<fov_id>/` by
default; {doc}`checkpoint reference <checkpoints-reference>` describes them.

## Workflow keys

With `backend: python`, the `preprocessing` parameter of a Python processing rule holds
the recipe:

| Key, under `rules.<rule>.parameters.preprocessing` | Meaning |
| --- | --- |
| `steps` | A list of steps: `method` (a name in the method table), that config's fields, and optional `save_as` |
| `post_registration` | Steps run after registration (after `steps` when no registration is set), in the same form |
| `extraction_source`, `registration_source`, `supplied_statistics` | The recipe fields of the same names |

On the MATLAB backend, the legacy keys are translated to recipe 1, in this order:

| MATLAB key | Step |
| --- | --- |
| `enhance_contrast` (with `snr_threshold`) | `min_max_normalization` with `output_dtype="uint8"`, `output_range=(0, 255)` and `rounding="truncate"` |
| `hist_equalize` (with `reference_channel`) | `histogram_matching` with that `reference_channel` |
| `morph_recon` (with `radius`) | `reconstruction` with `radius_yx=radius`; in `post_registration` for the resident subtile rules |
| `tophat` (with `radius`) | `white_tophat` with `radius_yx=radius` |

`maximum_projection` sets the projection of the saved reference image and is not a
step. {doc}`migration` describes the move from the MATLAB keys.
