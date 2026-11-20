# Segmentation reference

The methods, options, functions, records, files and workflow keys of segmentation. The
method and option tables are read from the code when the site is built. The
{doc}`segmentation guide <segmentation-contract>` explains how to use them; the
{doc}`segmentation API <api/segmentation>` documents every class.

## Methods

```{method-table} starfinder.segmentation.SEGMENTATION_METHODS
```

Each method's name is its `method` value in the workflow's `segmentation` block.

## Method options

### StarDist

StarDist predicts star-convex objects, nuclei above all, with a pretrained or
user-trained model; the model's dimensionality decides between 2D and 3D. See
{py:class}`~starfinder.segmentation.StarDistConfig`.

```{config-options} starfinder.segmentation.StarDistConfig
```

### Cellpose

Cellpose segments nuclei or whole cells from one or two stains in 2D or 3D. See
{py:class}`~starfinder.segmentation.CellposeConfig`.

```{config-options} starfinder.segmentation.CellposeConfig
```

### Seeded watershed

The seeded watershed grows one cell from each nucleus of the seed run over the
smoothed stain, and each cell keeps its nucleus's value. See
{py:class}`~starfinder.segmentation.SeededWatershedConfig`.

```{config-options} starfinder.segmentation.SeededWatershedConfig
```

## Plans

A plan is an ordered tuple of runs; each run names its source channels with
`InputChannel` and how they become the method's input with an `InputRecipe`. See
{py:class}`~starfinder.segmentation.SegmentationPlan`,
{py:class}`~starfinder.segmentation.SegmentationRun` and
{py:class}`~starfinder.segmentation.InputChannel`.

```{config-options} starfinder.segmentation.SegmentationRun
```

```{config-options} starfinder.segmentation.InputChannel
```

## Input recipes

An input recipe turns a run's source channels into the image its method receives, in five
fixed stages that run in this order. A stage that is absent changes nothing, and a run
without a recipe has the empty recipe: its input is the source channels stacked as ZYXC in
their dtype. `apply_input_recipe` is the one function that applies a recipe, for
`FOV.segment`, for the route after fusion and for `starfinder segment preview`. See
{py:class}`~starfinder.segmentation.InputRecipe` and
{py:func}`~starfinder.segmentation.apply_input_recipe`.

| Stage | Takes | Gives |
| --- | --- | --- |
| `channels` | Each source channel, ZYX | The same shape, float32 once a step has run |
| `combine` | The source channels, by role | The model channels, with their roles; `single` (the default) keeps each source channel |
| `projection` | Each model channel | Its Z projection, on the grid's projection |
| `after_projection` | Each model channel | The same shape |
| `model_input` | The run grid's spacing | The scale factor ZYX at which the method's model sees the input |

* **Lengths.** Every length is in micrometres and is converted with the run grid's spacing
  before any projection; an integer length rounds as `floor(x + ½)`. A length in
  micrometres on a grid without spacing raises `ValueError` naming the field and the keys
  `voxel_size_xy` and `voxel_size_z`.
* **Working type.** A channel is converted with `img_as_float32` before its first step (an
  unsigned image to [0, 1] of its dtype range), so every later step works in float32. A
  channel without a step keeps its dtype.
* **Roles.** A step's `roles` name source roles in `channels` and model roles in
  `after_projection`; without `roles` it applies to every channel of its stage.
* **Identity.** The recipe's SHA-256 is that of its canonical JSON (sorted keys, the configs as
  given). Its `name` is recorded beside it and is not part of it.

```{config-options} starfinder.segmentation.InputRecipe
```

```{config-options} starfinder.segmentation.RecipeStep
```

### Steps

A step is a registered preprocessing step of scope `per_channel` that is not fitted from
supplied statistics and whose lengths are in micrometres. Any other step is refused,
naming it and this rule.

| Step | Config | Value used on the run grid | Fitted |
| --- | --- | --- | --- |
| `gaussian` | `GaussianConfig(sigma_um)` | `sigma_um / spacing` per axis, 0 on an axis of length 1 | none |
| `clahe` | `ClaheConfig(kernel_um, clip_limit, nbins)` | `max(1, floor(kernel_um / spacing + ½))` per axis | `in_range`, over the fit region |
| `white_tophat` | `TophatConfig(radius_um=…)` | `floor(radius_um / spacing_yx + ½)` pixels | none |
| `reconstruction` | `ReconstructionConfig(radius_um=…)` | as `white_tophat` | none |
| `percentile_normalization` | `PercentileNormalizationConfig(p_low, p_high)` | unchanged | `low` and `high`, over the fit region |

The fit region is the whole image per FOV, and the included FOVs' footprints after fusion.
The {doc}`preprocessing reference <preprocessing-algorithms>` describes each step.

### Combine methods

`COMBINE_METHODS` maps each combine config to its method.

| Method | Config | Source roles | Model channel |
| --- | --- | --- | --- |
| `single` | none | any, each at most once | one per source channel, same role |
| `composite` | `CompositeConfig` | `nuclear`, `amplicon` | one, role `composite`, by `composite_nuclei_amplicon` |
| `flamingo_enhance` | `FlamingoEnhancementConfig` with `median_radius_um` | `nuclear`, `cytoplasm` (the Flamingo stain) | one, role `nuclear`, by `enhance_with_flamingo` |

The source roles must be exactly the method's. The composite's amplicon is a source channel
of its own, `InputChannel("amplicon", reference_merged=True)`. See
{py:class}`~starfinder.segmentation.CombineSpec`.

### Projection

`max` keeps the dtype. `sum` and `mean` add up in float64 and round the result once to
the output dtype, float32 unless `output_dtype` is `float64`. A value outside the output
dtype's finite range raises `ValueError` naming `output_dtype: float64`, and nothing is
clipped. The record's `exact` is true when an integer sum is held exactly (at most 2²⁴ in
magnitude for float32, 2⁵³ for float64), false otherwise, and null for `max`, `mean` and
float input. `conversion` is refused.

### Model input

`model_input` gives the pixel size the model sees, `pixel_size_um` (a number for Y and X, or
ZYX), or the factor itself, `factor`, on a grid without spacing. On a grid of spacing `s`
the factor is `s / pixel_size_um`, rounded to six significant digits; `factor` is used as
given per FOV and multiplied by the fusion factor `k` in Y and X after fusion. A method
with a model needs `model_input`, and `seeded_watershed` refuses it.

* StarDist receives the factor as its `scale`: ZYX for a 3D model, Y and X for a 2D model,
  and none when every factor is 1. A 2D model refuses a Z factor other than 1.
* Cellpose receives `diameter = 30 / f_yx`, and needs equal Y and X factors. A 3D run scales
  Z by `anisotropy × f_yx`; a model input that sets a Z factor must equal it, or the run is
  refused naming both factors.

```{config-options} starfinder.segmentation.ModelInputConfig
```

## Import

A run imports a mask with `LabelImportConfig`; `import_labels` imports one onto a grid
without a FOV. See {py:class}`~starfinder.segmentation.LabelImportConfig` and
{py:func}`~starfinder.segmentation.import_labels`.

```{config-options} starfinder.segmentation.LabelImportConfig
```

## Input and label functions

Each function can be called on its own and returns its result with a record. The first
two are the combine methods of an input recipe; `expand_labels` and
`extend_labels_through_z` are a run's `operations`.

| Function | Config | What it does |
| --- | --- | --- |
| `composite_nuclei_amplicon` | `CompositeConfig` | One 8-bit image in which both the nuclei and the amplicon signal are bright |
| `enhance_with_flamingo` | `FlamingoEnhancementConfig` | An 8-bit DAPI image dimmed where the Flamingo stain is bright |
| `normalize_percentile_unclipped` | | Percentile normalization to float32 without clipping |
| `rescale_input` | | Rescales an image by per-axis factors, with its metadata |
| `resample_labels` | | Maps a label image onto another shape by nearest neighbour |
| `expand_labels` | `ExpandLabelsConfig` | Grows each label into the background, in each plane or in 3D |
| `extend_labels_through_z` | `ZExtensionConfig` | Extends plane labels through Z where a stain is foreground |

### Composite input

Stretch settings for the nuclei–amplicon composite. See
{py:class}`~starfinder.segmentation.CompositeConfig`.

```{config-options} starfinder.segmentation.CompositeConfig
```

### Flamingo enhancement

Settings for enhancing the nuclear stain with Flamingo. See
{py:class}`~starfinder.segmentation.FlamingoEnhancementConfig`.

```{config-options} starfinder.segmentation.FlamingoEnhancementConfig
```

### Label expansion

Settings for growing label territories. See
{py:class}`~starfinder.segmentation.ExpandLabelsConfig`.

```{config-options} starfinder.segmentation.ExpandLabelsConfig
```

### Z extension

Settings for extending plane labels through a stain volume. See
{py:class}`~starfinder.segmentation.ZExtensionConfig`.

```{config-options} starfinder.segmentation.ZExtensionConfig
```

## Known models

`KNOWN_MODELS` lists the pretrained models that Starfinder can fetch and verify;
`starfinder weights list` prints them with their local state.

| Method | Model | Dimensionality | Channels |
| --- | --- | --- | --- |
| `stardist` | `2D_versatile_fluo` | 2D | 1 |
| `cellpose` | `cpsam_v2` | 2D network; 3D by planes with `do_3d` | 1 to 3 |

See {py:class}`~starfinder.segmentation.KnownModel` and
{py:func}`~starfinder.segmentation.resolve_model`.

## Label images

Every label image, computed or imported, is a C-contiguous `uint32` ZYX array: 0 is
background and each positive value is one object; a plane is 1×Y×X.
{py:func}`~starfinder.segmentation.to_label_dtype` converts a method's or a file's
integer labels to `uint32` without wrapping. Label values stay as the method returned
them.

| Geometry | Labels |
| --- | --- |
| `volume` | A 3D result on Z>1 |
| `plane` | A result on one plane, Z=1 |
| `extended` | Plane labels extended through Z by `extend_labels_through_z` |

## Records

`SegmentationResult.record`, written as `segmentation.json`, holds:

| Key | Contents |
| --- | --- |
| `format_version`, `stage` | `2` and `"segmentation"`; a record of version 1 still loads |
| `dataset_id`, `sample_id`, `fov_id`, `subtile_id`, `run` | The identities of the label namespace |
| `target`, `geometry`, `label_namespace` | As on the result |
| `grid` | The grid's `shape_zyx`, `metadata`, `source` and `sha256` |
| `input` | A computed run's model input: its SHA-256, shape, dtype and metadata, one entry per source channel (`channels`), the recipe record (`recipe`), with the file's path and SHA-256 when saved; `null` for an import |
| `seeds` | The seed run, its label namespace and labels' SHA-256, and the method's label rule; or `null` |
| `methods` | The method's provenance entry, with its model files (`artifacts`), `execution` and `effective` parameters; empty for an import |
| `import` | An import only: the file, its SHA-256, the source dtype and shape, where the metadata came from, and the relabel map |
| `operations` | The label operations applied, each with its config and record |
| `outcome` | `ok`, or `empty` when no label remains |
| `labels` | The labels' dtype, SHA-256, `n_labels` and `max_label`, with the file's path and SHA-256 when saved |
| `software` | The software versions |
| `wall_s` | Runs of `FOV.segment`: the wall time of the run in seconds, from its input to its last label operation; the report's time of the step `segment` |
| `upstream` | Runs of `FOV.segment`: the SHA-256 of the FOV's preprocessing and registration records |


The recipe record, `input.recipe`, holds:

| Key | Contents |
| --- | --- |
| `name`, `source` | The recipe's name in `segmentation.input_recipes` (null inline), and `given` or `legacy` |
| `sha256`, `config` | The recipe's identity and the recipe as given |
| `spacing_zyx`, `spatial_unit`, `spacing_source` | The spacing its lengths are converted with and its unit: the run grid's, before any projection |
| `fit_region`, `fit_pixels` | `image` per FOV or `footprints` after fusion, and the number of its pixels |
| `stages` | One entry per step (its config, the value `used` on the grid, `fitted`, `diagnostics`, the input's and output's SHA-256 and dtype) and per stage (`combine` with its record and roles; `projection` with `applied`, `exact`, `rounding`, `max_abs` and the grid) |
| `model_input` | The pixel size or factor given, `k`, the Y and X spacing, the factor ZYX before and after rounding, and `method_value`, the scale the method applied |

## Files

| File | Written by | Contents |
| --- | --- | --- |
| `segmentation/<run>/labels.tif` | `FOV.segment` with `checkpoints` | The labels, ZYX `uint32`, with the grid's metadata |
| `segmentation/<run>/input.ome.tif` | the same, for a computed run | The model input, the array the method received after every stage of the recipe, ZYXC |
| `segmentation/<run>/segmentation.json` | the same | The run record |
| `images/stardist_segmentation/{fovID}.tif` | the workflow rule `stardist_segmentation` | The labels, YX or ZYX as the input file |
| `images/stardist_segmentation/{fovID}.json` | the same | The run record |

The `segmentation/` folders are inside the FOV's folder, `<output_root>/fovs/<fov_id>/`
by default ({doc}`checkpoint reference <checkpoints-reference>`); `FOV.load_segmentation(name)` reads them. The
`images/` paths are under the workflow's output folder.

## Workflow keys

### The segmentation block

With `backend: python`, the top-level `segmentation` block holds a plan, which the
workflow's `segment` tasks run, and the named input recipes of both routes.

| Key | Meaning |
| --- | --- |
| `device` | The device of the plan's runs |
| `runs` | The runs, in order |
| `runs[].name`, `target`, `seeds` | As in `SegmentationRun` |
| `input_recipes` | Named input recipes, each a mapping of its stages; the block may hold recipes without `runs` |
| `runs[].inputs` | A list of `InputChannel` fields, the source channels |
| `runs[].input_recipe` | A name of `input_recipes`, or a recipe inline |
| `runs[].operations` | A list of label operations, each `expand_labels` or `extend_labels_through_z` with its config's fields |
| `runs[].method` | A method of the table above, or `import` |
| The config's fields | The fields of that method's config, or of `LabelImportConfig` |

The {doc}`workflow reference <workflow/reference>` describes the block.

### Legacy StarDist keys

The rule turns `rules.stardist_segmentation.parameters` into one StarDist run:

| Legacy key | Becomes |
| --- | --- |
| `stardist_base_path`, `stardist_model_name` | The model: the folder `<base>/<name>`, or the known model of that name when the base path is the weights cache's `stardist` folder |
| `prob_thresh`, `nms_thresh` | The StarDist thresholds |
| `rescale` | The implicit input recipe's `model_input` factor 0.5 when true, 1.0 when false |
| `maximum_projection` (the rule's, or the top-level key) | The implicit recipe's `projection: max`, recorded as not applied: the input file is the projection |
| `expand_labels`, `distance` | One planar expansion by `distance` pixels |
| `segmentation_input_folder` | The input channel's role: `composite` for `overlay`, the default, and `nuclear` for any other folder |
| `target` (`backend: python`) | The run's target; without it, `cell` for `overlay` and `nucleus` otherwise |
| `device` (`backend: python`) | The device |

The run record keeps the legacy keys under `workflow.legacy`, and the implicit recipe under
`input.recipe` with `name` null and `source` `legacy`.

## The preview command

`starfinder segment preview` segments a few FOVs with a few recipes and draws them, so that
a dataset's recipe can be chosen before a workflow run; {py:func}`~starfinder.segmentation.preview_recipes`
is the same in Python. It reads the files the workflow would read and writes only under
`--output`: `overlay.png` (one row of model inputs with label outlines and one row at the
model's pixel size per FOV, one column per recipe), `sizes.png` (each recipe's distribution
of the equivalent diameter `2·√(area/π)` in µm), `sizes.csv` (one row per label),
`summary.csv` (one row per FOV and recipe) and `preview.json` (the records). It takes at most
8 FOVs and 8 recipes, and chooses or ranks none.

```{segment-commands}
```
