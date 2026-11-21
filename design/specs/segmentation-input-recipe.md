# Segmentation input recipe: dataset-tunable input preparation on both routes

% Status: Approved (Jiahao, 2026-11-18, W-885), with amendments A and B1–B3; written against 76cd058. Implemented by W-891 (2026-11-20) and W-907 (2026-11-20); their implementation notes are under "Implementation (W-891)" at the end. Its tissue-2D description was corrected on 2026-11-21 (W-919: the tiles have Z = 30).

This page specifies the **input recipe**: a named, dataset-level description of how a
segmentation run turns its source channels into the image the model receives. One recipe
serves the route per FOV and the route after fusion, and has the same meaning in µm on both
grids.

* **What it is.** Five fixed stages run in order: `channels` (steps on each source
  channel), `combine` (source channels to model channels), `projection` (Z), `after_projection`
  (steps on the projected model channels) and `model_input` (the pixel size the model sees)
  ([The recipe](#the-recipe), [The steps](#the-steps)).
* **Where it goes.** `segmentation.input_recipes` names recipes. A run of
  `segmentation.runs` (per FOV) or of `export.segmentation.runs` (after fusion) refers to
  one with `input_recipe`. In Python, `InputRecipe` is a field of `SegmentationRun` and
  `SampleSegmentationRun`. `apply_input_recipe` is the one function both routes and the
  preview call ([Names](#names), [Configuration](#configuration)).
* **What it replaces.** `InputChannel.prepare` and `prepare_channel`, the run's
  `projection`, `StarDistConfig.scale` and `CellposeConfig.diameter`. They are replaced and
  not shimmed, and each has a migration row
  ([Replaced keys and their migration](#replaced-keys-and-their-migration)).
* **What it records.** Every step's configuration as given, the value used on the run's
  grid and what the step fitted. The saved `input.ome.tif` per FOV, and on the route after
  fusion the fused file or `model_inputs/<run>.ome.tif`, is the array passed to `segment`
  ([Records](#records)).
* **How a dataset's recipe is chosen.** `starfinder segment preview` draws a few FOVs ×
  a few recipes as an overlay grid on the model input with size distributions, without a
  workflow run ([The preview tool](#the-preview-tool)).

The validation design is [segmentation-input-recipe validation](../validation/segmentation-input-recipe.md).
The pages this one builds on are [segmentation](segmentation.md) (the plan, `InputChannel`,
the run record, the input functions, the model scale),
[segmentation-algorithms](segmentation-algorithms.md) (the composite, the Flamingo
enhancement), [sample-segmentation](sample-segmentation.md) (the route after fusion, LS2,
check 7, the `segmentation` section), [workflow-configuration](workflow-configuration.md),
[sample-output](sample-output.md) (the export folder) and [report](report.md) (the
segmentation preview). The sections of those pages that change when this one is
implemented are listed under [Pages that change](#pages-that-change).

This is a software design. It makes no data claim and no segmentation-quality claim. The
recipes named on this page are examples of the layout and are not recommended settings.

## Why

Filed from Jiahao's review of the tissue-2D shakedown's segmentation (W-870): input
preparation sometimes has to be tuned per dataset. Sometimes the image needs smoothing,
sometimes contrast enhancement, sometimes a sum projection works better, sometimes a new
composite is needed (such as the Flamingo-enhanced DAPI), and sometimes the input has to be
rescaled for the model.

What the package cannot express at `76cd058`:

* Per FOV, `InputChannel.prepare` takes one `CompositeConfig` or one
  `FlamingoEnhancementConfig` (`src/python/starfinder/segmentation/_plan.py:85`). There is
  no smoothing, contrast or background step and no chaining.
* After fusion, `SampleSegmentationRun.inputs` are `(role, image)` pairs
  (`src/python/starfinder/export/_config.py:266`) with no preparation.
* The model scale is a factor of the grid. One intent therefore needs different numbers per
  route. W-870 gave StarDist `scale` 0.25 on both routes, and LS2 turned it into 0.5 on the
  fused grid (`LENGTH_FIELDS`, `src/python/starfinder/export/_segment.py:39`). The per-FOV
  grid is 0.094635 µm per pixel and the fused grid at `k = 2` is 0.18927 µm.
* The projection is a field of the run (`SegmentationRun.projection`), apart from the rest
  of the preparation.

% W-870 run directory runs/W-870/20261118T0309Z-tissue2d: configs/tissue2d.yaml (scale 0.25, projection max), scripts/make_configs.py:27-32 (after_fusion copies the run's scale), results/stage1-look/stats.json (diameter quantiles per FOV)

## Decisions (Jiahao, 2026-11-18)

Jiahao approved this page on 2026-11-18 and accepted all eight recommendations ("Accept
all eight", W-885). The options of each choice are further down.

| # | Choice | Decision | Section |
| --- | --- | --- | --- |
| 1 | Where the steps come from | ST1: the registered preprocessing steps, with `gaussian` and `clahe` added to `PREPROCESSING_METHODS` | [The steps](#the-steps) |
| 2 | The working type of the steps | WT1: float32 from the first step on, on the `img_as_float32` scale | [The working type](#the-working-type) |
| 3 | How the route after fusion gets its channels | AF1: every source channel the recipe needs is an image of `export.images`, fused as today; the recipe runs on the fused projection | [The route after fusion](#the-route-after-fusion) |
| 4 | How the model scale is given | MI1: `model_input.pixel_size_um`, the pixel size the model sees, with `factor` only for a grid without spacing | [The model input](#the-model-input) |
| 5 | Where the image is rescaled for the model | RS1: inside the method, as today (StarDist `scale`, Cellpose `diameter`); the labels come back on the run's grid | [The model input](#the-model-input) |
| 6 | Where recipes live | RL1: named recipes in `segmentation.input_recipes`, referred to by name; an inline mapping is also accepted | [Configuration](#configuration) |
| 7 | The model input file after fusion | SI1: the fused file itself when the recipe leaves it unchanged; otherwise `model_inputs/<run>.ome.tif`, in a subfolder only model inputs use | [Records](#records) |
| 8 | The reusable-functions decision (D4) | Amend its sentence "there is no general step chain" to "the input recipe is the one chain: fixed stages of registered steps" | [Relation to the settled decisions](#relation-to-the-settled-decisions) |

**Amendments approved with the page** (Jiahao, 2026-11-18: "批准 A(a) 和 B"):

* **A.** A 3D Cellpose run refuses a model input whose Z factor differs from the one
  Cellpose applies ([The model input](#the-model-input)).
* **B1.** Input recipes are not presets
  ([Relation to the settled decisions](#relation-to-the-settled-decisions)). W-887 joins
  W-872 and W-876 among the issues whose keys these must fit ([Configuration](#configuration)).
* **B2.** The implementation starts after W-887 (which removes `processing.recipe`) and
  W-873 (a configuration per dataset) are merged. It migrates W-873's configurations to
  `input_recipe` and `model_input`.
* **B3.** The first use on W-870's stage 1 (FU1–FU3) is a separate evidence issue, which
  the coordinating session runs on UGER under W-889's rule for real-data evidence
  ([validation](../validation/segmentation-input-recipe.md)).

## Relation to the settled decisions

The settled decisions of [segmentation](segmentation.md) and
[sample-segmentation](sample-segmentation.md) hold, with one amendment, which Jiahao
confirmed on 2026-11-18 (choice 8):

* **The reusable-functions decision (D4)** says that input preparation reuses the
  preprocessing mechanism where it fits and that there is no second recipe engine. The
  segmentation page adds: "A plan is not a recipe engine … at most one input function per
  channel … There is no general step chain." The input recipe keeps the first part: its
  steps are `PREPROCESSING_METHODS` configs run through `run_step`, so the registry, the
  step contract and the fitted-value records are the preprocessing ones (ST1). It changes
  the second part: a run may have more than one step per channel. The chain is bounded. It
  has five fixed stages in a fixed order, steps only where a stage takes them, no
  branching, no snapshots, no supplied statistics and no reference images. `FOV.run`'s
  `PreprocessingRecipe` is not used and does not change.
* **The input scale follows the model's training data and is explicit and recorded, with
  no default.** It holds: a run whose method names a model needs `model_input`, and no
  value is derived from another model.
* **Cell outlines are never computed on a shrunk grid.** It holds: the recipe never
  shrinks the run's grid. The method scales internally (RS1), and `seeded_watershed` refuses
  a `model_input`.
* **The route after fusion segments the Z maximum, and `projection: sum` is refused on
  it (PS1).** It holds for the export's own projection. For a recipe after fusion, see
  [The route after fusion](#the-route-after-fusion).
* **Lengths in pixels (LS2).** It holds for the method fields that remain in pixels
  (Cellpose `min_size`, StarDist `block_size`, `min_overlap`, `context`). `scale` and
  `diameter` leave `LENGTH_FIELDS` because `model_input` replaces them.
* **Input recipes are not presets (amendment B1).** W-887 removes `processing.recipe`
  (`recipe: deep`). That was a preset defined in package code; its name did not describe
  its content, and run records did not keep it. An input recipe differs on each point:
  * it is written in the user's configuration;
  * its records keep its content and SHA-256;
  * the package ships no named input recipe;
  * a run gives a name or a full inline recipe, never a name plus overrides.

## Terms

* A **source channel** is one image a run reads: a round's channel, the reference merged
  image, or a fused image name on the route after fusion. Each has a source role
  (`nuclear`, `cytoplasm`, `membrane`, `amplicon`).
* A **model channel** is one channel of the `SegmentationInput` the method receives. Each
  has a role, as today (`nuclear`, `cytoplasm`, `membrane`, `amplicon`, `composite`).
* The **run grid** is the grid of the source channels: the FOV's reference grid per FOV, or
  the segmentation grid after fusion ([sample-segmentation](sample-segmentation.md),
  "Terms").
* A **step** is a registered preprocessing step applied to one channel: same shape in, same
  shape out.
* The **fit region** is the set of pixels from which a step's global statistics are taken.
  Per FOV it is the whole image. After fusion it is the union of the included FOVs'
  footprints.

## The recipe

```text
source channels ──channels──▶ prepared sources ──combine──▶ model channels (ZYXC)
     ──projection──▶ (1, Y, X, C) ──after_projection──▶ model input ──model_input──▶ method scale
```

| Stage | Takes | Gives | Default |
| --- | --- | --- | --- |
| `channels` | Each source channel, ZYX | The same shape, float32 once a step has run | No step |
| `combine` | The prepared source channels, by role | The model channels, ZYXC, with their roles | `single`: each source channel is one model channel, in the order of `inputs` |
| `projection` | The model channels | `(1, Y, X, C)` on `grid.projected(method=…)` | None: no projection |
| `after_projection` | Each model channel, on the projected grid or the run grid | The same shape | No step |
| `model_input` | The run grid's Y and X spacing | The scale factor the method uses, ZYX | None; required for a run whose method names a model |

* The stages run in this order and in no other. A stage that is absent is the identity.
  A recipe with every stage absent is the **empty recipe**: the model input equals the
  source channels stacked as ZYXC, in their dtype, as `FOV.segment` assembles a run today
  without `prepare` and without `projection`.
* `after_projection` also applies when `projection` is absent. Its steps then run on the
  run grid, and their name says only where they sit in the order.
* A step may name `roles`. In `channels` these are source roles, and in `after_projection`
  model roles. Without `roles`, a step applies to every channel. A role that the run's
  channels do not have is refused when the run is built (`ValueError` naming the step and
  the role).
* Every value the recipe converts from µm uses the run grid's spacing before any
  projection. `ImageMetadata.projected` gives a frame of unknown physical calibration
  (`src/python/starfinder/image.py:208`), but a Z projection does not change Y and X, so
  the unprojected Y and X spacing is exact for every stage. The record names the spacing
  used.
* A recipe with a field in µm on a grid without spacing raises `ValueError`, naming the
  field and the keys `voxel_size_xy` and `voxel_size_z`.

```python
@dataclass(frozen=True)
class RecipeStep:
    config: object                         # a PREPROCESSING_METHODS config admitted by the recipe ("The steps")
    roles: tuple[str, ...] | None = None   # None: every channel of the stage

@dataclass(frozen=True)
class ModelInputConfig:
    pixel_size_um: float | tuple[float, float, float] | None = None  # YX (Z factor 1), or ZYX
    factor: float | tuple[float, float, float] | None = None         # exactly one of the two

@dataclass(frozen=True, kw_only=True)
class InputRecipe:
    channels: tuple[RecipeStep, ...] = ()
    combine: object | None = None          # a COMBINE_METHODS config; None: single
    projection: ProjectionConfig | None = None   # axis z; method max, sum or mean
    after_projection: tuple[RecipeStep, ...] = ()
    model_input: ModelInputConfig | None = None
```

A bare step config in `channels` or `after_projection` is taken as `RecipeStep(config)`.
Every field is validated in `__post_init__`. The recipe's identity is the SHA-256 of its
canonical JSON (sorted keys, the configs as given). The record keeps it, so two runs that
name one recipe can be shown to have used it.

### The steps

**Options for where the steps come from.**

| Option | Rule | What it changes | Consequence |
| --- | --- | --- | --- |
| **ST1. The registered preprocessing steps (recommended)** | A step is a `PREPROCESSING_METHODS` config run with `run_step`. `gaussian` and `clahe` are registered there as new steps of category `contrast` and scope `per_channel`. | Two new registered steps, available to `PreprocessingRecipe` too. `TophatConfig`, `ReconstructionConfig` and `FlamingoEnhancementConfig` gain a µm field. | One registry, one step contract and one record layout. The preprocessing reference lists the two new steps. |
| ST2. A separate registry of input steps | `INPUT_STEPS` in `starfinder.segmentation`, with its own µm configs. | A new registry and a second contract wrapper. | `white_tophat`, `reconstruction` and `percentile_normalization` exist twice. This is the second recipe engine that D4 rules out. |

**Recommendation: ST1.**

The recipe admits a registered step when its scope is `per_channel`, it is not fitted from
supplied statistics, and its lengths are in µm. At implementation, these five steps are
admitted. Any other registered step is refused (`ValueError` naming the step and this
rule). A later issue can admit another step by the same rule.

| Step (`method`) | Config | Fields in the recipe | Value used on the run grid | Fitted (recorded) | Function |
| --- | --- | --- | --- | --- | --- |
| `gaussian` (new) | `GaussianConfig(sigma_um)` | `sigma_um`: a number (every axis) or ZYX, > 0 | `sigma_px_zyx = sigma_um / spacing` per axis; 0 on an axis of length 1 | none | `skimage.filters.gaussian(image, sigma=sigma_px_zyx)` with its defaults (mode `nearest`, truncate 4.0), the conversion and call that `seeded_watershed` makes (`src/python/starfinder/segmentation/_methods.py:216`) |
| `clahe` (new) | `ClaheConfig(kernel_um, clip_limit=0.01, nbins=256)` | `kernel_um` (required, number or ZYX), `clip_limit` in (0, 1], `nbins` ≥ 2 | `kernel_px_zyx = max(1, floor(kernel_um / spacing + ½))` per axis; 1 on an axis of length 1 | `in_range`: the low and high values mapped to 0 and 1 | Map the fit region's minimum and maximum linearly to [0, 1] and clip there (a constant fit region gives zeros and records `degenerate_range` true), then `skimage.exposure.equalize_adapthist(image, kernel_size=kernel_px_zyx, clip_limit, nbins)`, cast to float32. The library rescales its input between the image's minimum and maximum and works on 2¹⁴ grey levels (skimage 0.26). After the clip these are 0 and 1, so its rescale is the identity and `in_range` is the whole mapping. |
| `white_tophat` | `TophatConfig(radius_um=…)` | `radius_um` ≥ 0; `radius_yx` (pixels) is refused in a recipe | `radius_yx = floor(radius_um / spacing_yx + ½)` | none | `filter_tophat`, unchanged |
| `reconstruction` | `ReconstructionConfig(radius_um=…)` | as `white_tophat` | as `white_tophat` | none | `correct_background`, unchanged |
| `percentile_normalization` | `PercentileNormalizationConfig(p_low, p_high, fit="fov")` | `p_low`, `p_high`; `fit: supplied` is refused | unchanged (no length) | `low`, `high` per channel, over the fit region | `normalize_percentile`, unchanged; on float32 it clips to [0, 1] (`src/python/starfinder/preprocessing/normalization.py:268`) |

* `kernel_um` has no default. skimage's default kernel is one eighth of the image's size,
  which is 384 × 384 pixels (36 µm) on a 3072-pixel FOV and 1,411 × 1,231 pixels (267 ×
  233 µm) on the fused tissue-2D projection. One recipe would then mean two different
  kernels on the two routes.
* `TophatConfig` and `ReconstructionConfig` take exactly one of `radius_yx` (pixels, as
  today, for `FOV.run`) and `radius_um`. The step converts `radius_um` with
  `StepContext.metadata`. This follows `Background3DConfig`, which already takes
  `radius_um_zyx` or `radius_voxels_zyx` (`src/python/starfinder/preprocessing/background.py:277`).
  `radius_um` needs equal Y and X spacing (relative difference at most 1e-9), else
  `ValueError`.
* `floor(x + ½)` is the rounding of every integer length on this page, so that 2.5 gives 3
  on both routes. Python's `round` would give 2.
* `StepContext` gains `fit_mask`, a boolean ZYX array of the fit region, or None for the
  whole image. The steps that fit a global statistic (`percentile_normalization`, `clahe`)
  take it from the masked pixels, and every other step ignores it. `run_step` checks that
  its shape is the image's ZYX shape. `FOV.run` passes None, so its results do not change.
* Inside `FOV.run`, `gaussian` and `clahe` behave as in a recipe: per channel, with
  `StepContext.metadata` for the µm values and the whole image as the fit region. Their
  dtype policy is `preserve`, as for every preprocessing step. A float input stays in its
  dtype. On an integer input, `gaussian` smooths with `preserve_range=True` and rounds
  with `np.rint` back to the dtype. `clahe` scales its [0, 1] output to the dtype's full
  range and rounds with `np.rint`, as `percentile_normalization` does
  (`src/python/starfinder/preprocessing/normalization.py:291`). The recipe always passes
  float32 ("The working type"), so the rounding never applies there.

### The working type

| Option | Rule | Consequence |
| --- | --- | --- |
| **WT1. float32 from the first step (recommended)** | Before a channel's first step, its image is converted with `img_as_float32` (an unsigned integer image to [0, 1] of its dtype range, a float image unchanged). Every later step and stage works in float32, and the model input is float32. A channel without a step keeps its dtype. | Smoothing and CLAHE are not quantized. A numeric `seeded_watershed` threshold keeps its meaning, because it is already on the `img_as_float` scale. |
| WT2. The input dtype at every step | Each step returns its input's dtype, so a uint8 PI stain stays uint8. | Smaller files. A Gaussian of 3 pixels on uint8 and CLAHE's float output are rounded back to 256 levels. |

**Recommendation: WT1.** With the empty recipe, or with `combine` alone, the dtypes are
those of today. The composite and the Flamingo enhancement still return uint8, and their
golden digests do not change (see [The combine stage](#the-combine-stage)).

### The combine stage

`combine` maps prepared source channels to model channels. Its methods form a registry on
the shared mechanism ([method-registry](../../docs/method-registry.md)):
`COMBINE_METHODS: dict[type, CombineSpec]`, where `CombineSpec(name, run, source_roles,
output_roles)`. A new composite is a new method here.

| `method` | Config | Source roles (exactly) | Model channels | Function |
| --- | --- | --- | --- | --- |
| `single` (default) | `None` | any, each at most once | one per source channel, same role, same order | none |
| `composite` | `CompositeConfig(nuclear_quantile, amplicon_quantile)` | `nuclear`, `amplicon` | one, role `composite` | `composite_nuclei_amplicon(nuclear, amplicon, config=…)`, unchanged |
| `flamingo_enhance` | `FlamingoEnhancementConfig(flamingo_quantile, nuclear_quantile, median_radius_um)` | `nuclear`, `cytoplasm` (the Flamingo channel) | one, role `nuclear` | `enhance_with_flamingo(nuclear, flamingo, config=…)` with `median_radius_px = floor(median_radius_um / spacing_yx + ½)` |

* The amplicon of `composite` is no longer implicit. It is a source channel of the run,
  `{role: amplicon, reference_merged: true}` per FOV and `{role: amplicon, image:
  reference_merged}` after fusion. The Flamingo channel is a source channel of role
  `cytoplasm` (per FOV the same round's Flamingo channel, which `prepare_channel` used to
  name).
* `FlamingoEnhancementConfig` takes exactly one of `median_radius_px` (pixels, for a
  direct call to `enhance_with_flamingo`) and `median_radius_um`. A recipe refuses
  `median_radius_px`, because a recipe's steps take µm.
* The source roles of a run must be exactly those of its combine method (`single`: any),
  else `ValueError` naming the run, the method and both role lists. The model channels'
  roles must then satisfy the method's `roles` and `required_roles`, as checked today.
* Both functions stretch between quantiles and return uint8. After float32 channel steps,
  their inputs lie in [0, 1], and the stretch maps them as it maps the uint8 inputs of
  today (`rescale_intensity` with a nonnegative input range gives [0, 1]). The validation
  checks this ([validation](../validation/segmentation-input-recipe.md), row RC3).

### The projection stage

* `ProjectionConfig` gains the method `mean`. `ImageMetadata.projected` and
  `ReferenceGrid.projected` accept `mean` too, recorded in the frame as
  `/projection:mean`.
* In a recipe, `max` keeps the dtype. `sum` and `mean` accumulate in float64 and convert
  the result once to the output dtype: float32 by default, or `output_dtype: float64`.
  `conversion` is refused in a recipe, because a display rescale is not a projection.
* **The overflow check.** After accumulating in float64 and before converting, the stage
  checks that every value lies in the finite range of the output dtype (float32: magnitude
  at most 3.4028235 × 10³⁸). A value outside it raises `ValueError`, naming the run, the
  recipe's projection, the largest magnitude reached and `output_dtype: float64` as the
  remedy. Nothing is clipped or converted silently. The record keeps the largest magnitude
  as `max_abs`.
  * Only a `sum` of float input can reach the limit. Two float32 planes near 3 × 10³⁸,
    for example, sum beyond it. A `mean` lies between the smallest and largest input value,
    and float64 holds the sum of any float32 stack the package can load.
  * Integer inputs of the sizes this package allows cannot reach it. Integer inputs are
    at most 32 bits, since `ProjectionConfig` refuses 64-bit integer sums. Even 2³² − 1 in
    every one of 10⁹ planes sums to about 4.3 × 10¹⁸. For a uint16 stack of 50 planes, the
    largest sum is 3,276,750.
* **The `exact` flag** applies to a `sum` of an integer input only, signed or unsigned
  (`_validate_image` accepts signed integer images). It is `true` when every sum has an
  absolute value of at most 2²⁴ = 16,777,216, the bound up to which float32 holds every
  integer of either sign (the uint16 example above is), and `false` otherwise. An int32
  sum of −16,777,217 is therefore `false`, because float32 rounds it to −16,777,216. With
  `output_dtype: float64` the bound is |sum| ≤ 2⁵³. For `max` it is null, because the input dtype is kept. For `mean`,
  and for any float input, it is null: the mean of integers need not be representable (the
  mean of 0, 1 and 1 is 2/3). For every `sum` and `mean`, the record states `rounding:
  "float64 result rounded once to <output dtype>"`, so a reader knows that a null `exact`
  means a value rounded once, not an error.
* A run whose operations include `extend_labels_through_z` needs a recipe with a projection
  and one model channel. The stain it extends through is that model channel before the
  projection (after `channels` and `combine`). This is the channel used today, the prepared
  channel before projection, and with the empty recipe and no combine it is the same array.

### The model input

**Options for how the scale is given.**

| Option | The setting | What it changes | Consequence |
| --- | --- | --- | --- |
| **MI1. `pixel_size_um`, the pixel size the model sees (recommended)** | `model_input: {pixel_size_um: p}`. The factor on a grid of spacing `s` is `s / p` per axis. A number sets Y and X with Z factor 1; a ZYX 3-tuple sets each axis. `factor: f` instead is accepted for a grid without spacing. | `StarDistConfig.scale` and `CellposeConfig.diameter` leave the configs ([Replaced keys](#replaced-keys-and-their-migration)). | One number means the same on both routes: 0.37854 µm gives 0.25 per FOV and 0.5 after fusion at `k = 2`, the two values W-870 used. |
| MI2. `object_diameter_um` per method | The expected object size in µm, with the method's nominal size in pixels (Cellpose 30). | A nominal size per model. | StarDist's `2D_versatile_fluo` records no nominal size or training pixel size ([segmentation](segmentation.md), "Known-models table"). |
| MI3. Factors per route (LS2 as today) | `scale` and `diameter` stay, given for the acquired grid and multiplied or divided by `k`. | Nothing. | The factor still depends on the grid. The preview cannot show a pixel size, and a dataset with another acquisition spacing needs new numbers. |

**Recommendation: MI1.**

* **The factor.** `factor_zyx = s_zyx / p_zyx`, Z factor 1 for a number, rounded to six
  significant digits. Both the unrounded and the rounded value are recorded. The rounding
  removes float noise, so 0.094635 / 0.37854 gives exactly 0.25. `factor` given directly
  is used as given per FOV and multiplied by `k` in Y and X after fusion, which is the LS2
  rule for the old `scale`.
* **How it reaches the method (RS1).** `SegmentationInput` gains `model_scale_zyx:
  tuple[float, float, float] | None`. A spec gains `scales: bool`, which is true for
  `stardist` and `cellpose`. For a method that scales, the stage wrapper requires
  `model_scale_zyx` (`ValueError` naming `model_input`). A method that does not scale
  refuses it. StarDist passes it to `predict_instances(scale=…)` as `_learned.py` passes
  `scale` today (`library_scale`, `src/python/starfinder/segmentation/_learned.py:429`). A
  3D model receives the ZYX factor, such as `(1, 0.25, 0.25)`. A 2D model receives only Y
  and X, such as `(0.25, 0.25)`, because StarDist refuses a scale whose length differs from
  the image's axes, and a Z factor other than 1 is refused for it as today. A factor of 1
  on every axis is passed as None. Cellpose passes `diameter = 30 / f_yx`, since its image is rescaled
  by 30 / `diameter`. `f_yx = 1` therefore gives diameter 30, which is the old
  `diameter: None` (no rescale). Cellpose needs equal Y and X factors.
* **A 3D Cellpose run (amendment A).** For a run with Z > 1, Cellpose scales Z by
  `anisotropy` × 30 / `diameter`, that is `anisotropy` × `f_yx`. `anisotropy` is the run's
  configured value, and the recipe does not change it.
  * A `model_input` that sets a Z factor (a ZYX `pixel_size_um` or `factor`) must equal
    that value within the float64 tolerance.
  * Otherwise the run is refused at configuration, with `ValueError` naming the run, the
    requested Z factor and the one Cellpose applies.
  * A number sets Y and X only, and Cellpose's own Z factor applies.
* **The record.** The method's `effective` entry records `scale_zyx`, with the Z factor
  actually applied, and `diameter` as used. The preview states the same factors.
* **`seeded_watershed`** refuses a recipe with `model_input` (`ValueError`: cell
  outlines are never computed on a shrunk grid).
* **Where the image is rescaled.**

| Option | Rule | Consequence |
| --- | --- | --- |
| **RS1. Inside the method, as today (recommended)** | The recipe's output is on the run grid. The method resamples internally and returns labels on that grid. | Labels and seeds stay on the run grid. The saved model input is the array `segment` receives, and the method's own normalization and resampling are recorded in `effective`. The W-870 labels can be reproduced exactly (validation row FU1). |
| RS2. The recipe resamples | A last stage calls `rescale_input`, the method runs at factor 1 and `resample_labels` maps the labels back by nearest neighbour. | The saved input shows what the network sees, but the labels change against today (nearest-neighbour polygons in place of StarDist's own rendering), and seeds and the watershed would need the shrunk grid. |

**Recommendation: RS1.** The preview shows the model input at the pixel size the model
sees, so what the network sees can still be inspected
([The preview tool](#the-preview-tool)).

### One function for both routes

```python
def apply_input_recipe(sources: Sequence[tuple[str, np.ndarray, Mapping]], grid: ReferenceGrid, *,
                       recipe: InputRecipe | None, fit_mask: np.ndarray | None = None,
                       k: int = 1) -> tuple[SegmentationInput, np.ndarray | None, dict]: ...
```

* `sources` are `(source role, ZYX array, source record)` in the order of the run's
  `inputs`. `grid` is the run grid. `fit_mask` is a boolean ZYX mask of the fit region
  (None: the whole image). `k` is the fusion factor that a `factor` model input is
  multiplied by.
* It returns the `SegmentationInput` (image, grid or projected grid, model roles, sources
  and `model_scale_zyx`), the unprojected model channels (for `extend_labels_through_z`;
  None without a projection) and the recipe record ([Records](#records)).
* Each step runs through `run_step` with a `StepContext` whose `metadata` is the run
  grid's metadata (unprojected, for the µm values), with `round_name` set to the source or
  model role and `reference_round` to the run's name.
* `FOV.segment` (through `assemble_input`), `segment_fused` and the preview call it, and
  nothing else applies a recipe. It reads no file and writes no file.

## The route after fusion

Only fused images exist after fusion. The two ways to give a recipe what it needs:

| Option | Rule | Cost on tissue-2D (56 tiles of 30 × 3072 × 3072 uint8 at 0.094635 µm in Y and X; fused projection 11,291 × 9,848 at 0.18927 µm) | Consequence |
| --- | --- | --- | --- |
| **AF1. Fused channels (recommended)** | Every source channel of a recipe is an image of `export.images` (checked by check 7 before anything is written). `segment_fused` reads the fused projections and runs the recipe on them, on the segmentation grid, with the fit region of the footprints. | No new fused file for the recipes of W-870. `reference_stain` (PI) and `reference_merged` are already fused, so a PI recipe and the PI–amplicon composite need nothing more. Each further channel adds one fused projection, 111.2 MB as uint8 (11,291 × 9,848 = 111,193,768 pixels). A recipe that changes the image writes `model_inputs/<run>.ome.tif`, 444.8 MB as uncompressed float32. One float32 copy of the fused plane is 444.8 MB in memory, so a Gaussian (input and output) holds about 0.9 GB. These figures are arithmetic and not measured. | One fit over the whole sample: one percentile range and one CLAHE mapping, so no step makes a seam. Steps run after fusion's own resampling to `k`. The recipe's `channels` stage runs on a plane that is already the Z maximum (see below). |
| AF2. Fused prepared inputs | Each FOV's prepared input (stages `channels` to `after_projection`) is written per FOV and fused as a new image. | 56 prepared files, 56 × 37.7 MB = 2.11 GB as float32, plus their fused projection, 444.8 MB as float32, plus a preparation task per FOV on a route that has none today. | Steps fitted per FOV (percentiles, CLAHE) differ between neighbouring tiles. The blend then mixes two different mappings in every overlap (10 % of each tile's side), and the seams show in the model input. It is the only way to run 3D steps before the projection on a fused 3D sample. |

**Recommendation: AF1.** On tissue-2D it needs no new fused image, and it fits each
statistic once over the sample.

**Z.** The fused images are Z maxima of the tiles. When the tiles have one plane (Z = 1),
the projection of a plane is the plane for `max`, `sum` and `mean` alike, and the recipe
means what it means per FOV. When the tiles have more than one plane (tissue-2D's have
30), a step before the Z maximum cannot be applied after it, and `sum` or `mean` cannot be
recovered from a maximum. Check 7 therefore refuses, on the route after fusion with tiles of Z > 1, a recipe
with a `channels` step or with a `projection` other than `max` (`ValueError` naming the
recipe, the stage and `after_projection` as the stage for 2D steps). There is either an
exact meaning or a refusal, never a silent difference. A projection of `max`, or no
projection, is the fused maximum itself, recorded `applied: false, reason: fused_maximum`.
The tiles' Z is known before any fusion from their reference grids, the `grids` that
`fused_grid` takes. With Z = 1, the recipe's projection is recorded `applied: false,
reason: single_plane`, and its output dtype rule still applies.

**Corrected 2026-11-21 (W-919).** This section first named tissue-2D as the case of tiles
of one plane. Its tiles are 30 × 3072 × 3072 (W-870's `reference.tif`; W-892's runs); it is
2D only after the maximum projection. On tissue-2D's route after fusion, check 7 therefore
refuses a recipe with a `channels` step or a `projection` other than `max`, and 2D steps go in
`after_projection`, as W-892 ran them (`r1a`, `r2a`). The rule and the recommendation are
unchanged; the cost column's figures are those of the fused projection and stand.

**Filter scale on the fused grid.** Steps take µm, and the fused grid's spacing is `k·s`.
`sigma_um` 0.3 is therefore 3.17 pixels per FOV and 1.59 pixels on the fused grid at
`k = 2`, and `model_input.pixel_size_um` gives each route its own factor (MI1). What still
differs between the routes, stated so that `compare_routes` and the preview are read
correctly:

* the fusion's resampling to `k`, which runs before the recipe after fusion and is not part
  of it;
* integer lengths (`radius_um`, `kernel_um`, `median_radius_um`) round to whole pixels on
  each grid (both values recorded);
* global statistics (`percentile_normalization`, CLAHE's `in_range`) are fitted per FOV on
  one route and over the sample on the other;
* image borders: per FOV a filter meets the tile's edge, and after fusion the neighbouring
  tile's pixels.

**The fit region after fusion** is the union of the included FOVs' footprints on the
segmentation grid. These are the pixels that `ownership_map` gives an owner, so the empty
canvas outside the tiles does not pull a percentile towards zero. The record names
`fit_region: footprints` and its pixel count. Per FOV it is `fit_region: image`.

**Check 7 gains**, on the route after fusion with an `assignment`: each run's
`input_recipe` names a recipe of `segmentation.input_recipes` (or is inline); each source
role is one of the recipe's combine source roles; each source image is a name of
`export.images` (as today); the Z rule above; `model_input` present for a method with a
model and absent for `seeded_watershed`.

## Records

### Per FOV: `segmentation.json`

`format_version` becomes 2. In `input`, `projection` is replaced by `recipe`. Each entry
of `channels` is a source channel and loses `prepare` and `prepare_channel`:

```json
"input": {
  "path": "input.ome.tif", "sha256": "...", "file_sha256": "...",
  "shape_zyxc": [1, 3072, 3072, 1], "dtype": "float32", "metadata": {"frame_id": ".../projection:sum"},
  "channels": [{"role": "nuclear", "round": "reference_stain", "channel": "PI", "name": "PI",
                "wavelength": "unavailable", "reference_merged": false, "registration": {}, "sha256": "..."}],
  "recipe": {
    "name": "pi_smooth", "sha256": "<canonical JSON of the recipe as given>", "config": {},
    "spacing_zyx": [1.0, 0.094635, 0.094635], "spacing_source": "run grid before projection",
    "fit_region": "image", "fit_pixels": 9437184,
    "stages": [
      {"stage": "channels", "index": 0, "role": "nuclear", "method": "gaussian",
       "config": {"sigma_um": 0.3}, "used": {"sigma_px_zyx": [0.0, 3.17, 3.17]},
       "fitted": {}, "diagnostics": {}, "input": "<sha256>", "output": "<sha256>", "dtype": "float32"},
      {"stage": "combine", "method": "single", "config": null, "record": null,
       "roles_in": ["nuclear"], "roles_out": ["nuclear"]},
      {"stage": "projection", "method": "sum", "config": {"method": "sum"}, "applied": true, "exact": null,
       "rounding": "float64 result rounded once to float32", "max_abs": 1.0, "dtype": "float32",
       "grid": {"shape_zyx": [1, 3072, 3072], "frame_id": "..."}},
      {"stage": "after_projection", "index": 0, "role": "nuclear", "method": "clahe",
       "config": {"kernel_um": 6.0, "clip_limit": 0.01, "nbins": 256}, "used": {"kernel_px_zyx": [1, 63, 63]},
       "fitted": {"in_range": [0.0, 0.83]}, "diagnostics": {}, "input": "<sha256>", "output": "<sha256>",
       "dtype": "float32"}
    ],
    "model_input": {"pixel_size_um": 0.37854, "factor": null, "spacing_yx": [0.094635, 0.094635],
                    "factor_zyx_exact": [1.0, 0.25, 0.25], "factor_zyx": [1.0, 0.25, 0.25],
                    "method_value": {"scale_zyx": [1.0, 0.25, 0.25], "library_scale": [0.25, 0.25]}}
  }
}
```

* `config` is the recipe as given (µm), and each stage's `used` holds the pixel values on
  this grid. `fitted` and `diagnostics` are the `StepResult` values. `input` and `output`
  are array SHA-256 values computed as the run record computes them. The `combine`
  stage's `record` is the composite or Flamingo record of today (config, quantiles reached,
  input hashes).
* `input.ome.tif` is the model input: the array that `segment` receives, after every
  stage. It was already the prepared and projected input; it now also holds the steps. The
  method's own normalization (`normalize_percentiles`) and its internal resampling are not
  saved. They are recorded in `methods[0].effective`. The report's segmentation preview
  (W-884) shows this file, and can caption it with `recipe.name` and
  `model_input.pixel_size_um`.
* A record of `format_version` 1 still loads. `FOV.load_segmentation` does not read
  `prepare` or `projection` from it, and a reader that shows the recipe shows none for it.
  Reading an old record is data compatibility, not an API shim.
* Size: with WT1, a FOV's float32 `input.ome.tif` is 37.7 MB at 3072 × 3072 where the
  uint8 PI input is 9.4 MB. For 56 FOVs and one run that is 2.11 GB against 0.53 GB,
  before compression.

### After fusion: `export.json` and the model input file

| Option | When the recipe changes the fused image | When it does not (the empty recipe, or `combine: single` with one fused channel and no step) |
| --- | --- | --- |
| **SI1 (recommended)** | `model_inputs/<run>.ome.tif` (float32, or the combine's uint8; the segmentation grid's geometry; the package's OME-TIFF writer) | No new file. The run's `input` names the fused projection file and its SHA-256, as today. |
| SI2 | The same file | The same file, a copy of the fused projection |

**Recommendation: SI1.**

**Where the file goes.** A model input is written as `model_inputs/<run>.ome.tif`, in a
subfolder of `export/<sample>/` that only model inputs use. A file name prefix at the top
of the folder is not used, because it can collide. An image of `export.images` may be
named `model_input_cell` from a file source (`export/_config.py`, `_check_images` allows
letters, digits, `_`, `-` and `.`). Its fused projection, `model_input_cell_projection.ome.tif`,
would then be the model input of a run named `cell_projection` under a prefix
`model_input_<run>.ome.tif`. The subfolder cannot collide:

* A fused image is a top-level entry whose name ends in `_projection.ome.tif`,
  `_stack.ome.tif`, `_projection.ome.zarr` or `_stack.ome.zarr`. An image name holds no
  `/`, so no image can name `model_inputs` or anything inside it.
* The other top-level entries have fixed names (`export.json`, `tables/`,
  `sample.h5ad`, `sample.zarr/`, `cell_labels.ome.tif`, `territories.ome.tif`,
  `segmentation_<run>.ome.tif`, the workflow's `previews/`), and none is `model_inputs`.
* Inside the subfolder, a file is named by its run alone. Run names are lowercase
  snake_case and differ within a configuration, so two runs cannot share a file, also on a
  file system that ignores case.

The subfolder exists only while a run has a model input. Its files are listed among the
`segmentation` section's outputs, so the export removes them with the section's other
listed files when the part runs again (`remove_listed`; [sample-output](sample-output.md),
"How the folder is kept equal to its record"), and it removes the subfolder when the
subfolder is empty. R2 of `check_export` lists the files under `model_inputs/` and fails on
an unlisted file there, as for any other unlisted file outside `previews/`. T1 gains: each
listed `model_inputs/<run>.ome.tif` exists with its recorded SHA-256 and the segmentation
grid's shape and geometry, and names a run of the section.

The `segmentation` section's run entry gains:

* `recipe`: the layout above, with `spacing_zyx` of the segmentation grid, `fit_region:
  footprints` and its pixel count, and the `factor` rule's `k`;
* `input`: `{path, sha256, dtype, shape_zyxc, written}`, where `written` is false when
  `path` is the fused file;
* `lengths`: as today, without `scale` and `diameter`.

`load_sample_segmentation` and the reuse rule (`segment: false`) compare the recipe's
SHA-256 with the current configuration's. A different recipe refuses the reuse before
anything is written, naming the run and both hashes, as a changed placement does today.

## The preview tool

`starfinder segment preview` lets a dataset's recipe be chosen before a workflow run: a few
FOVs × a few recipes, segmented with one run's method, drawn on the model input with their
size distributions.

```text
starfinder segment preview CONFIG --run NAME --fovs F1,F2,… --recipes R1,R2,… --output DIR
                           [--recipe-file FILE] [--factor-yx K] [--crop-um Y,X,H,W]
                           [--device cpu|cuda] [--save-inputs]
```

* `CONFIG` is the workflow configuration. `--run` names a run of `segmentation.runs` (or,
  with `--factor-yx`, of `export.segmentation.runs`): its inputs, method, model, seeds and
  operations are used as configured, and only its recipe is replaced by each of
  `--recipes`. The recipes are names of `segmentation.input_recipes` or of `--recipe-file`
  (a YAML mapping in the same layout, so a recipe can be tried before it is written into
  the configuration). A run with seeds also computes its seed run, with that run's own
  recipe.
* **Inputs.** Each source channel is read from a file the workflow would read: the saved
  `other_rounds/<name>/image.ome.tif` and the saved reference image when present. Without
  them, `reference_stain` is prepared in memory from the raw files by
  `FOV.prepare_morphology` (the morphology task's own call; nothing is written). A source
  that needs `FOV.run` or a registration that has not been saved is refused before
  anything runs, naming the workflow step that produces it. No preprocessing,
  registration or detection runs. Tissue-2D's PI recipes therefore need no earlier run.
* **`--factor-yx K`** previews at the pixel size of the route after fusion. Each FOV's
  source channels are resampled to spacing `K·s` with the call `fuse` makes for one view
  (one FOV, no blend), and the recipe runs with `k = K`. The record names this grid
  `fused_view`. It is not the fused sample: neighbours and seams are absent.
* **Outputs**, all under `--output`, which must be absent or empty. Nothing is written
  under the dataset's output root.
  * `overlay.png`: rows are FOVs and columns are recipes. Each panel is the model input (or
    the `--crop-um` window of it), shown linearly between its 1st and 99.8th percentiles,
    with the label outlines on top. The title gives the recipe, the number of labels and
    the median equivalent diameter in µm. A second row of panels per FOV shows the model
    input resampled by `model_scale_zyx`, at the pixel size the model sees.
  * `sizes.png`: per recipe, the empirical distribution of the equivalent diameter
    `2·√(area/π)` in µm, pooled over the FOVs, with one thin line per FOV.
  * `sizes.csv`: one row per label (`fov`, `recipe`, `label`, `area_um2`,
    `equivalent_diameter_um`, `touches_border`). `summary.csv`: one row per FOV and recipe
    (`n_labels`, the 5th, 25th, 50th, 75th and 95th percentiles of the diameter, the
    fraction of the image covered by labels, the number of labels touching the border).
  * `preview.json`: the configuration's path and SHA-256, the run, the method record
    (model artifacts, `effective`), and per FOV and recipe the recipe record as
    `segmentation.json` holds it, the model input's SHA-256, the label SHA-256, the counts,
    wall time and peak RSS, and the software. With `--save-inputs` it also writes
    `inputs/<fov>/<recipe>.ome.tif`.
* **What it does not do.** It chooses no recipe, ranks none and applies no threshold to
  the sizes. It is not a parameter sweep: at most 8 FOVs and 8 recipes per call
  (provisional guard, `ValueError` beyond). It does not run assignment.
* **Where it lives.** `starfinder.segmentation.preview_recipes(config, *, run, fovs,
  recipes, output, …)` is the function. The CLI group `segment` with the command `preview`
  is added in `src/python/starfinder/__main__.py`, beside `weights`.

## Names

All names are in `starfinder.segmentation` unless stated.

| Concept | Name | Kind |
| --- | --- | --- |
| Recipe | `InputRecipe`, `RecipeStep`, `ModelInputConfig` | frozen dataclasses |
| Applying it | `apply_input_recipe(sources, grid, *, recipe, fit_mask=None, k=1)` | function |
| Combine registry | `COMBINE_METHODS`, `CombineSpec` | registry, frozen dataclass |
| New steps | `GaussianConfig` (`gaussian`), `ClaheConfig` (`clahe`) in `starfinder.preprocessing`, registered in `PREPROCESSING_METHODS` | configs |
| µm fields | `TophatConfig.radius_um`, `ReconstructionConfig.radius_um`, `FlamingoEnhancementConfig.median_radius_um` | fields |
| Projection | `ProjectionConfig(method="mean")`; `ImageMetadata.projected(method="mean")` | value |
| Run fields | `SegmentationRun.input_recipe`, `SampleSegmentationRun.input_recipe` | fields |
| Import grid | `LabelImportConfig.projection`: `max`, `sum`, `mean` or None | field |
| Model scale | `SegmentationInput.model_scale_zyx`; `SegmentationSpec.scales` | field |
| Preview | `preview_recipes(...)`; CLI `starfinder segment preview` | function, command |
| After fusion | `model_inputs/<run>.ome.tif` | file in a subfolder of `export/<sample>/` |

The new exports go into `__all__` and the autosummary lists of `docs/api/segmentation.rst`
and `docs/api/preprocessing.rst`, which `docs/check_reference.py` requires.

## Configuration

**Options for where recipes live.**

| Option | Rule | Consequence |
| --- | --- | --- |
| **RL1. Named recipes, referred to by name (recommended)** | `segmentation.input_recipes: {<name>: <recipe>}`. A run's `input_recipe` is a name, or a mapping with the recipe inline. | One recipe serves runs of both routes and the preview by its name. Its SHA-256 shows that they used the same recipe. |
| RL2. Inline only | Each run carries its recipe. | The two routes repeat the same block. |

**Recommendation: RL1.**

```yaml
segmentation:
  device: cpu
  input_recipes:
    pi_max:                                   # W-870's run, as a recipe
      projection: {method: max}
      model_input: {pixel_size_um: 0.37854}
    pi_smooth:
      channels: [{method: gaussian, sigma_um: 0.3}]
      projection: {method: sum}
      after_projection: [{method: clahe, kernel_um: 6.0, clip_limit: 0.01}]
      model_input: {pixel_size_um: 0.37854}
    pi_composite:
      combine: {method: composite, nuclear_quantile: 0.005, amplicon_quantile: 0.001}
      projection: {method: max}
  runs:
    - name: nucleus
      target: nucleus
      method: stardist
      model: 2D_versatile_fluo
      inputs: [{role: nuclear, round: reference_stain, channel: PI}]
      input_recipe: pi_smooth
export:
  route: after_fusion
  images: [reference_merged, reference_stain]
  segmentation:
    cells: nucleus
    nuclei: nucleus
    runs:
      - name: nucleus
        target: nucleus
        method: stardist
        model: 2D_versatile_fluo
        inputs: [{role: nuclear, image: reference_stain}]
        input_recipe: pi_smooth               # the same recipe: 0.25 per FOV, 0.5 at k = 2
```

The values illustrate the layout and are not recommended settings. On the route after
fusion, `pi_smooth` runs only when the tiles have one plane: on tiles of more than one
plane, tissue-2D's included, check 7 refuses its `channels` step and its `sum` projection
("The route after fusion", "Z"; corrected 2026-11-21, W-919).

* A step is a mapping with `method` (the registered step name), its config fields and an
  optional `roles`. `combine` is a mapping with `method` and its config fields.
  `projection` is a mapping of `method` and an optional `output_dtype`. `model_input` holds
  exactly one of `pixel_size_um` and `factor`.
* `segmentation.input_recipes` may be given without `segmentation.runs`, for a dataset
  that segments after fusion only. The per-FOV segment rule is planned only when `runs` is
  given. The schema's `runs` loses `minItems: 1` for this case only, and an empty
  `segmentation` block is still refused.
* A run without `input_recipe` has the empty recipe. Its method must therefore be one
  that does not scale (`seeded_watershed`) or an import, because a model needs
  `model_input`.
* An unknown key in a recipe, a step or a stage is refused, naming it.
* `workflow/_plan.py`'s `_reference_inputs` (which runs read the reference merged image)
  reads the source channels with `reference_merged: true`. The amplicon of a composite is
  such a source now.
* These keys are written against `76cd058`. Three other issues also change the
  configuration schema:
  * W-872 (the FOV list in the configuration);
  * W-876;
  * W-887, which removes `processing.recipe` (amendment B1).

  When all three have landed, the implementation issue checks that
  `segmentation.input_recipes`, `input_recipe` and
  `export.segmentation.runs[*].input_recipe` fit their keys, and restates any conflict
  for Jiahao rather than resolving it.

**The legacy rule.** The translation of `rules.stardist_segmentation.parameters` gives an
implicit recipe, recorded with `name: null` and `source: legacy`:

| Legacy key | Recipe |
| --- | --- |
| `rescale: true` / `false` | `model_input: {factor: 0.5}` / `{factor: 1.0}` (the input file may lack spacing) |
| `maximum_projection` (rule or top level) | `projection: {method: max}` |
| `segmentation_input_folder` | unchanged: the file's one channel, role as today, combine `single` |

## Replaced keys and their migration

Replaced, not shimmed (AGENTS.md). A replaced key is refused with a `ValueError` (Python:
`TypeError` for an unknown field) that names the new key and this table's row. Rows for
`docs/migration.md`:

(segmentation-input-recipe-migration)=

| Before | After |
| --- | --- |
| `InputChannel(role="composite", round=R, channel=C, prepare=CompositeConfig(...))` | Two source channels, `InputChannel(role="nuclear", round=R, channel=C)` and `InputChannel(role="amplicon", reference_merged=True)`, with `InputRecipe(combine=CompositeConfig(...))` |
| `InputChannel(role="nuclear", round=R, channel=C, prepare=FlamingoEnhancementConfig(median_radius_px=m, ...), prepare_channel=F)` | `InputChannel(role="nuclear", round=R, channel=C)` and `InputChannel(role="cytoplasm", round=R, channel=F)`, with `InputRecipe(combine=FlamingoEnhancementConfig(median_radius_um=m × spacing_yx, ...))` |
| `SegmentationRun(projection=ProjectionConfig(method=M))` for a method | `SegmentationRun(input_recipe=InputRecipe(projection=ProjectionConfig(method=M)))` |
| `SegmentationRun(projection=…)` for an import | `LabelImportConfig(projection=M)` |
| `StarDistConfig(scale=f)` | `InputRecipe(model_input=ModelInputConfig(pixel_size_um=spacing_yx / f))`, or `ModelInputConfig(factor=f)` on a grid without spacing |
| `CellposeConfig(diameter=d)` | `ModelInputConfig(pixel_size_um=spacing_yx × d / 30)`; `diameter=None` becomes `ModelInputConfig(factor=1.0)` |
| YAML `segmentation.runs[*].inputs[*].prepare`, `prepare_channel` | `input_recipe` with `combine`, and the extra source channel in `inputs` |
| YAML `segmentation.runs[*].projection` | `input_recipe` with `projection`; for `method: import`, the import's `projection` |
| YAML `scale` (StarDist), `diameter` (Cellpose) in `segmentation.runs` and `export.segmentation.runs` | `input_recipe` with `model_input` |
| `LENGTH_FIELDS["stardist"]["scale"]`, `LENGTH_FIELDS["cellpose"]["diameter"]` | Removed; `model_input` derives each route's factor |
| `segmentation.json` `format_version` 1 (`input.projection`, `channels[*].prepare`, `prepare_channel`) | `format_version` 2 (`input.recipe`); version 1 records still load |

`enhance_with_flamingo` and `composite_nuclei_amplicon` keep their signatures and their
golden digests. `FlamingoEnhancementConfig` keeps `median_radius_px` for direct calls.

## Pages that change

On implementation, the implementing issue updates these and states the change in each:

* [segmentation](segmentation.md): "Coordination per FOV" (`InputChannel`, `SegmentationRun`,
  and the sentence of D4), "Run record", "Method input", "Registered methods"
  (`StarDistConfig.scale`, `CellposeConfig.diameter`, `scales`), "Input preparation and
  label functions", "Input scale", "Workflow configuration".
* [sample-segmentation](sample-segmentation.md): "Lengths in pixels", "Where the settings
  live", check 7, "The record", T1.
* [segmentation-algorithms](segmentation-algorithms.md): the composite's and the Flamingo
  enhancement's place in a recipe. Their algorithms do not change.
* [preprocessing](preprocessing.md) and [preprocessing-algorithms](preprocessing-algorithms.md):
  `gaussian`, `clahe`, the µm radii and `mean`.
* [workflow-configuration](workflow-configuration.md): the new keys.
* [report](report.md): the caption of the segmentation input (W-884).
* `docs/`: the segmentation guide and reference, `docs/migration.md` (the rows above) and
  the CLI reference for `starfinder segment preview`.

## Exclusions

No new segmentation method, model or default recipe. No recommended sigma, kernel or pixel
size. No segmentation-quality claim and no ranking of recipes. No change to `FOV.run`'s
preprocessing or its records beyond the two registered steps. No 3D steps before the Z
maximum on the route after fusion (refused, see AF1). No block-wise driver. No change to
Cellpose's `anisotropy`, which stays a ratio of the method. No change to the MATLAB rules
or keys.

## Implementation (W-891)

Implemented on 2026-11-20 by W-891, on the branch of the §2.14 phase 3 batch. What the
implementation decided within the rules above, stated so that a reader of this page knows
it:

* **`apply_input_recipe` takes three more keyword arguments**: `run` (the run's name, for the
  steps' `reference_round` and the messages), `projected` (None per FOV; `fused_maximum` or
  `single_plane` after fusion, `input_file` for the legacy rule's projected file: the
  projection stage is then recorded `applied: false` with that reason and keeps the grid) and
  `source` (`given`, or `legacy` for the implicit recipe). The recipe record names `source`.
* **`InputRecipe.name`** holds the recipe's name in `segmentation.input_recipes` (None
  inline). It is recorded and is not part of the recipe's SHA-256.
* **The projection stage projects each model channel in its own dtype**, so each stage's input
  hash is the previous stage's output hash; the model input is then stacked as ZYXC with
  NumPy's promotion, as the empty recipe stacks the sources.
* **`check_model_input`** applies the model input's rules at configuration: a method with a
  model needs it, `seeded_watershed` refuses it, Cellpose needs equal Y and X factors and, in
  3D, a Z factor equal to `anisotropy × f_yx` when the model input sets one (amendment A),
  and StarDist's block-wise prediction needs the factor 1. It runs when a run is built (a
  `factor` needs no grid), per FOV before any step runs, and in check 7 after fusion.
* **The export task reads the `segmentation` block** (`TASK_BLOCKS`, amended in
  [orchestration](orchestration.md), "The blocks of a task kind"), because the runs of
  `export.segmentation` name the recipes of `segmentation.input_recipes`.
* **The preview** refuses `--factor-yx` for a run whose seeds are imported: an imported mask
  lies on the FOV's grid, not on a fused view. A run of `segmentation.runs` previewed with
  `--factor-yx` uses its source channels' one-view fusion; a run of `export.segmentation.runs`
  reads the files of `export.images`.
* **A model input of several model channels** after fusion is written as one CYX OME-TIFF.
* **`TophatConfig()`, `ReconstructionConfig()` and `FlamingoEnhancementConfig()`** keep their
  pixel defaults (3, 3 and 1 pixels when neither length is given), and a record of a length in
  pixels keeps its form without the µm field (`config_record`).
* **The recipe record names its spacing's unit** (`spatial_unit` beside `spacing_zyx`, null
  without a spacing), as every persisted grid of the workflow does (the declared-grid check of
  `test_workflow_engines_e2e.py`).
* **The golden comparison** names the record changes above as its exception IR
  ([orchestration](orchestration.md), "The exception of the golden comparison"): the cell run's
  record of format version 2 and the export run's `recipe` and `input`. Everything else stays
  compared.

### Follow-up (W-907)

W-907 (2026-11-20) finished what W-891's review found missing, within the rules above:

* **Benchmark trials build their model input as a run does.** A segmentation trial
  (`python -m starfinder.benchmark._trial`) takes an optional `input_recipe`, a mapping in the
  layout of `segmentation.input_recipes`; without one it has the empty recipe. Its image
  channels are the source channels of its roles, `apply_input_recipe` makes the model input,
  `check_model_input` runs after the weights check and before any input is loaded, and the
  record holds the recipe record as `input.recipe`. `scale`, `diameter` and `projection` in a
  trial's `config` are refused with the migration row, not shimmed.
* **The FOV's grid of an export run's preview** is the one the export's fusion reads, that of
  the saved reference image. Without that file it is the grid of the reference stain the
  preview read (`fov:reference_stain`, from its saved form or prepared in memory), the same
  acquisition as the reference round. `preview.json` records each FOV's grid as `fov_grids`
  (source, SHA-256, shape and spacing). An export run that reads no reference stain on a FOV
  without a saved reference image is refused before anything runs.
* **The reference stain prepared in memory** is the first step of `FOV.prepare_morphology`
  alone: the reference round's stain files loaded with the morphology task's rotation and
  recorded as the same acquisition. `prepare_morphology` itself always prepares the other
  rounds too, and `MorphologyConfig.rounds` cannot name none.
* **The label operations in the preview.** Each result of `preview.json` records the run's
  `operations` as `segmentation.json` does. With `--factor-yx`, a run of `segmentation.runs`
  has its `expand_labels` applied on the fused view's grid, as `FOV.segment` applies it; a run
  that extends its labels through Z is refused, because the view has one plane. A run of
  `export.segmentation.runs` has no operations.
