# Preprocessing step and recipe contract

**Status: Proposed (W-226, 2026-09-27; revised after the W-227 review notes).
Not accepted.** Human review in W-227 accepts, amends or rejects this page.
Until then it does not change any behavior or authorize implementation.

This contract replaces the fixed preprocessing slots of `PipelineConfig` with an
ordered recipe of steps. It supports the two agreed recipes: min–max → histogram
matching → optional morphology, and background correction → percentile
normalization. It also supports a pre-normalization extraction source and a
declared registration source. Current behavior is recorded in
{doc}`preprocessing-baseline`; the new methods are specified in
{doc}`preprocessing-algorithms`.

## Step interface

Every step is a pure function of one round's image:

```python
def run(volume: np.ndarray, config: StepConfig, context: StepContext) -> StepResult: ...

@dataclass(frozen=True)
class StepContext:
    round_name: str
    reference_round: str
    metadata: ImageMetadata
    reference: np.ndarray | None = None      # only for needs_reference steps
    supplied: Mapping[str, Any] | None = None  # this step's supplied section, fit="supplied" only

@dataclass(frozen=True)
class StepResult:
    image: np.ndarray                  # same shape and axes as the input
    fitted: Mapping[str, Any]          # JSON-serializable fitted values
    diagnostics: Mapping[str, Any]     # JSON-serializable, recorded in provenance
```

A step never mutates its input and never reads other rounds except through
`context.reference`.

## Registration of step implementations

Implementations are registered in one explicit mapping, keyed by the exact
frozen config type:

```python
@dataclass(frozen=True)
class StepSpec:
    name: str            # stable identifier, e.g. "percentile_normalization"
    run: Callable[..., StepResult]
    category: str        # "background", "intensity" or "contrast"
    scope: str           # "per_channel", "per_round" or "needs_reference"
    dtype_policy: str    # "preserve" (default) or "declared"

STEPS: dict[type, StepSpec]  # populated in starfinder.preprocessing
```

* `name` is the single identifier of a step. The workflow adapter, provenance and
  the supplied-statistics file all use it. Names are unique lowercase
  snake_case; a lookup from name to config type is derived from `STEPS`, never
  maintained separately.
* `category` follows the chapter's categories. It documents intent and does not
  impose an order.
* `scope="per_channel"` computes statistics per channel within the round.
  `per_round` uses the whole round. `needs_reference` also receives the
  reference round's input to the same step (histogram matching).
* `dtype_policy="preserve"` requires the output dtype to equal the input dtype.
  `declared` allows a dtype set in the config; it is reserved for legacy min–max.
* Lookup uses `type(config)` exactly; subclasses are not matched. The existing
  frozen-config validation (`__post_init__`) is kept.

| Step name | Config | Category | Scope | dtype policy |
| --- | --- | --- | --- | --- |
| `min_max_normalization` | `MinMaxNormalizationConfig` | intensity | per_channel | declared |
| `histogram_matching` | `HistogramMatchingConfig` | intensity | needs_reference | preserve |
| `reconstruction` | `ReconstructionConfig` | background | per_channel | preserve |
| `white_tophat` | `TophatConfig` | background | per_channel | preserve |
| `scalar_background` | `ScalarBackgroundConfig` | background | per_channel | preserve |
| `background_3d` | `Background3DConfig` | background | per_channel | preserve |
| `percentile_normalization` | `PercentileNormalizationConfig` | intensity | per_channel | preserve |

### Histogram reference channel

`HistogramMatchingConfig` gains `reference_channel: int = 0` and replaces
`PipelineConfig.histogram_reference_channel`. The reference round is
`context.reference_round`. For `fit="fov"`, the wrapper passes that round's
input to the histogram-matching step, restricted to `reference_channel`, as
`context.reference`. With `fit="supplied"`, the step's supplied section records
the round and channel that were summarized ({doc}`preprocessing-algorithms`).
The default reproduces the legacy behavior.

## Recipe

```python
@dataclass(frozen=True)
class RecipeStep:
    config: StepConfig
    save_as: str | None = None     # snapshot name for this step's output

@dataclass(frozen=True)
class PreprocessingRecipe:
    steps: tuple[RecipeStep, ...]
    post_registration: tuple[RecipeStep, ...] = ()
    extraction_source: str | None = None     # snapshot name; None = detection image
    registration_source: str | None = None   # snapshot name; None = detection image
    supplied_statistics: Path | None = None  # file for fit="supplied" steps
```

* **Detection image.** The output of the last step in `steps` is the detection
  image. With no steps it is the loaded image.
* **Snapshots.** `save_as` keeps a named copy of that step's output. Names are
  unique within a recipe and must not be `"detection"`. Snapshots cost memory
  only when declared.
* **Extraction source.** By default extraction reads the detection image. If
  `extraction_source` names a snapshot, extraction reads that snapshot. The
  agreed alternative is the background-corrected image before normalization.
* **Registration source.** `registration_source` names the snapshot from which
  registration signals are built, for both the reference round and each moving
  round. By default it is the detection image, which is today's behavior. The
  §2.6 signal modes (maximum, sum or selected channel) apply to this snapshot.
* **Post-registration steps.** `post_registration` may contain only
  `ReconstructionConfig`, the legacy reconstruction-after-registration path for
  resident subtiles. Any statistic a post-registration step computes is
  restricted to the valid-overlap region, excluding fill introduced by
  resampling.
* **Order.** Steps run in the declared order, before registration. No branching
  or DAG: snapshots are taps on one linear sequence.
* **Supplied statistics.** A recipe whose steps include the same step name twice
  with `fit="supplied"` is rejected at validation.

`PipelineConfig` gains `preprocessing: PreprocessingRecipe | None` and loses the
slots `normalization`, `histogram`, `histogram_reference_channel`,
`reconstruction`, `reconstruction_after_registration`, `tophat` and
`projection`. The migration guide records the replaced Python fields.

### Projection is not part of preprocessing

Projection is an output view, not a correction of the image, so it is neither a
step nor a pipeline slot. The pipeline always processes ZYX(C) volumes; 2D data
are volumes with Z = 1. If a dataset ever needs processing on projected images,
it is projected when loaded, before the recipe. Projection is used for two
views, both with maximum projection by default:

* **Visualization:** projection along Z for 2D figures such as registration
  overlaps and spot-finding results.
* **Inspection:** projection along channels for each FOV's reference merged
  image (`images/ref_merged/{fovID}.tif`): the reference round's detection
  image, merged over channels, and also projected along Z when the workflow's
  `maximum_projection` is true. It shows preprocessing before registration and
  is the stitching input.

The views and the alignment of the Python reference merged image with MATLAB
are specified separately (W-235).

## Registration and snapshots

For each moving round, registration estimates the transform from the
`registration_source` snapshot. It then applies the same transform to every
snapshot used downstream: the detection image and, if different, the extraction
source.

* **Single registration step.** Each snapshot is resampled once from its
  pre-registration version.
* **Several registration steps** (for example global then local). Until §2.6
  task group 4 composes the stages into one resampling, each step's transform is
  applied to every snapshot in turn, as it is to the single image today. A
  snapshot is then resampled once per registration step.
* **Consistency.** Every downstream snapshot goes through the same sequence of
  resamplings, so all snapshots of a round stay aligned with each other and with
  the detection coordinates.
* The reference round is not transformed.
* Provenance records, per round and snapshot, the transforms applied.
* The `registered` checkpoint stores every snapshot used downstream under its
  name. A recipe without an extraction source stores one image, as today.

## Enforcement wrapper

One wrapper calls every step and checks the result:

* shape and axis order equal the input;
* dtype equals the input dtype (`preserve`) or the declared dtype (`declared`);
* values are finite;
* metadata is unchanged (steps never modify `ImageMetadata`);
* `fitted` and `diagnostics` are JSON-serializable.

A violation raises and is recorded, with the step name, as the failing step in
the run record.

## Provenance

The existing run record (`run.json`) gains a `preprocessing` entry:

```json
{"recipe": {"steps": ["..."], "extraction_source": null, "registration_source": null},
 "rounds": {"round1": [{"index": 0, "step": "percentile_normalization",
                        "config": {}, "fitted": {}, "diagnostics": {},
                        "input_dtype": "uint8", "output_dtype": "uint8",
                        "save_as": null}]},
 "supplied_statistics": {"path": null, "sha256": null}}
```

Checkpoints that store images also store the recipe and the per-round step
records, so a reloaded run can report how each image was produced.

## Streaming

The reference round is processed first, as today. For each `needs_reference`
step, the reference round's input to that step is retained, restricted to the
channel the config names (one channel volume for histogram matching), until all
moving rounds have passed the step. The reference round's `registration_source`
snapshot is retained until registration of all moving rounds is complete. No
other cross-round image is kept in streaming mode.

## Fitting modes

Steps that fit statistics accept `fit="fov"` (fit on the current round of the
current FOV) or `fit="supplied"` (read from the step's section of
`supplied_statistics`, keyed by step name). This covers percentile
normalization, scalar background estimation and the histogram-matching
reference. Min–max keeps its per-FOV behavior. The recipe stage at which
statistics are summarized, the histogram summary and merge, and the file format
are in {doc}`preprocessing-algorithms`.

## Workflow configuration

### Legacy mapping

Python workflow rules keep their existing keys. The adapter maps them to recipe 1,
reproducing the golden outputs exactly:

| Legacy key | Recipe step |
| --- | --- |
| `enhance_contrast.run` (with `snr_threshold`) | `MinMaxNormalizationConfig("uint8", (0, 255), snr_threshold=..., rounding="truncate")`, `dtype_policy="declared"` |
| `hist_equalize.run` (with `reference_channel`) | `HistogramMatchingConfig(reference_channel=reference_channel)`, scope `needs_reference` |
| `morph_recon.run` (with `radius`) | `ReconstructionConfig(radius_yx=radius)`, in `post_registration` for resident subtile rules |
| `tophat.run` (with `radius`) | `TophatConfig(radius_yx=radius)` |

`maximum_projection` keeps its meaning for the saved reference merged image and
does not enter the recipe.

### Explicit step list

A new optional Python-rule key, `preprocessing`, takes an explicit recipe and is
mutually exclusive with the legacy keys:

```yaml
preprocessing:
  steps:
    - method: scalar_background
      percentile: 10.0
      save_as: bg_corrected
    - method: percentile_normalization
  extraction_source: bg_corrected
```

* `method` is a step name from `STEPS`; an unknown name raises. The remaining keys
  of a step, except `save_as`, are the fields of its config dataclass.
* `extraction_source`, `registration_source` and `supplied_statistics` are the
  recipe fields above.
* `workflow/schemas/config.schema.yaml` declares the `preprocessing` key as a
  static schema. A default-tier test checks that its step names and parameters
  match `STEPS` and the config dataclass fields, so a registered step cannot be
  missing from the schema.

The key is available only on the Python backend. MATLAB APIs and shared MATLAB
keys are unchanged. The pipeline default remains recipe 1 until evaluation
supports a change.

## Exclusions

* No DAG or branching recipes; one linear sequence with named snapshots.
* No entry-point or plugin discovery; the step mapping is explicit in the package.
* No metaclass-based logging; provenance is written by the recipe runner.
* No projection inside the recipe or pipeline.
* No change to MATLAB behavior or shared MATLAB-facing keys.
* No registration or detection registries; §2.6 and §2.7 own those.
