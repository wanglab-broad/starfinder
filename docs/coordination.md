# Dataset and FOV coordination

`Dataset` owns paths, ordered rounds/channel labels, a codebook and FOV creation.
`FOV` stores images, metadata and structured results. Processing remains in public
functions. `PipelineConfig` describes the scientific sequence; `ExecutionConfig`
controls when images are loaded and released. Unknown config arguments raise.

```python
from starfinder.dataset import PipelineConfig, ExecutionConfig, RegistrationRecipe, RegistrationStep
from starfinder.io import ImageLoadConfig
from starfinder.registration import TranslationConfig
from starfinder.spot_finding import LocalMaximaConfig
from starfinder.barcode import NeighborhoodSumConfig, WtaDecoderConfig, ReadFilterConfig

config = PipelineConfig(
    load=ImageLoadConfig(channel_labels=dataset.channel_order),
    registration=RegistrationRecipe((RegistrationStep(TranslationConfig()),)),
    detection=LocalMaximaConfig(threshold_mode="noise", threshold_value=5),
    extraction=NeighborhoodSumConfig(neighborhood_radius_zyx=(1, 2, 2)),
    decoding=WtaDecoderConfig(),
    filtering=ReadFilterConfig(),
)
fov = dataset.fov("FOV_001").run(config, execution=ExecutionConfig("streaming"))
fov.save_spots()  # unchanged goodSpots filename, 1-based XYZ
```

Load the dataset codebook before decoding. None disables a stage. The sequence is
load, rotation, the steps of the preprocessing recipe (`preprocessing`, a
{py:class}`~starfinder.preprocessing.PreprocessingRecipe`) in their declared
order, the registration recipe (`registration`, a
{py:class}`~starfinder.dataset.RegistrationRecipe`), the preprocessing recipe's
`post_registration` steps, detection,
extraction, decoding and filtering. `post_registration` accepts only
reconstruction, which legacy subtile workflows place after registration. The
pipeline never projects; projection is an output view. Histogram matching uses
a copy of the reference round's configured channel taken as it enters that step,
retained until every moving round has passed it. Steps with `save_as` keep named
snapshots (`FOV.snapshots`). Registration builds its signals from the
preprocessing recipe's `registration_source` snapshot, or the detection image by
default; see "Registration recipe" below. Extraction reads the
`extraction_source` snapshot, or the detection image by default. Both execution
modes use this
sequence and the same operation configs; see {doc}`preprocessing-contract`.
Batch preloads and retains rounds. Streaming releases moving images after their
last use unless `retain_images=True`; subtile creation requires retention.
Streaming is a residency policy, not a claim that transforms or retained outputs
consume constant memory.

Inputs must have explicit round/channel labels. Registration replaces moving
metadata with reference metadata. Without registration, extraction requires
verified common frame/grid metadata; it does not assume that matching shapes
mean alignment. Crops retain physical geometry and source mappings. Subtile files
carry dataset/sample/FOV, round/channel labels and subtile IDs; incompatible
reload identities are rejected. Each rectangular axis is partitioned separately,
including remainder pixels. These Python changes do not alter MATLAB tiling.

## Registration recipe

A {py:class}`~starfinder.dataset.RegistrationRecipe` is zero or more global steps
(`translation`, `rigid`, `affine`) followed by at most one local step (`demons`,
`bspline`, `tps`, `cpd`), as specified in {doc}`registration-contract`. For
each moving round, `FOV.register` (which `run` calls):

1. builds a float64 ZYX signal per round with the recipe's
   {py:class}`~starfinder.registration.RegistrationSignalConfig` (default: the
   channel maximum; `sum`; or one `channel` per round, by index or label), or
   with a step's own `signal`;
2. estimates step k on the moving signal resampled in float64 through steps 1
   to k−1 (the reference signal is not resampled), runs
   `registration_qc` on the result and raises `RegistrationRejectedError` when
   a criterion of `recipe.qc` fails (none is set by default);
3. composes the step transforms into one
   {py:class}`~starfinder.registration.TransformChain`
   (`Φ(p) = T₁(T₂(…Tₙ(p)))`) and resamples the round's detection image and every
   one of its snapshots once, each from its pre-registration array. A chain of
   translations uses the exact translation path; any other chain is sampled
   linearly with SciPy, plane by plane, unless `recipe.warp` selects another
   policy. Integer outputs are rounded once.

The reference round is not transformed. `registration_results[round]` keeps the
step results (their `application_config` is the round's one `WarpConfig`),
`registration_chains[round]` the chain, and `registration_record` the recipe
summary, the `WarpConfig` applied per round and the semantics (`recipe`; a
loaded version-1 checkpoint is `sequential`). A round is registered once.

```python
from starfinder.dataset import RegistrationRecipe, RegistrationStep
from starfinder.registration import DemonsConfig, RegistrationSignalConfig, TranslationConfig

recipe = RegistrationRecipe(
    (RegistrationStep(TranslationConfig()), RegistrationStep(DemonsConfig())),
    signal=RegistrationSignalConfig("channel", reference_channel="ch00"),
)
fov.register(recipe)
fov.registration_chains["round2"].pull_field()  # composite pull displacement, float64
```

### Other rounds and external references

`FOV.register_rounds(recipe, *, rounds, reference=None)` registers loaded
rounds, such as morphology rounds, through a shared stain, as specified in
{doc}`registration-contract` ("Other-round and external-reference
registration"). The recipe's signal is usually `mode="channel"` with
`reference_channel` naming the stain in the reference and `moving_channel` in
each moving round, by index or by label. Sequencing rounds use
`Dataset.channel_order`; an other round with its own channels lists them in
`Dataset.other_channel_order`, and `Dataset.channel_labels(round)` returns
either. `reference=None` uses `recipe.reference_round` (default: the dataset
reference round); any loaded round may be named. An
{py:class}`~starfinder.dataset.ExternalReference` supplies a ZYX reference
signal directly; it must have the moving rounds' grid.

Unknown labels, channel indices outside a round and grid mismatches raise
before any estimator runs (`ValueError`, `IncompatibleGeometryError`). Each
round is then registered as by `register`: the chain estimated on the stain is
applied once to every channel and snapshot of the round. Each estimation
attempt records `reference` (the round name or the external label) and
`reference_sha256` (the SHA-256 of the external image's C-order bytes, `None`
for a round); `registration_record["rounds"][round]` keeps the recipe summary
and the reference. `save_processing_log("nr")` writes `log/<fov>_nr.txt` and
`log/gr_shifts/<fov>_nr.txt` for these rounds.

```python
from starfinder.dataset import ExternalReference

stain = RegistrationSignalConfig("channel", reference_channel="ch04", moving_channel="ch00")
recipe = RegistrationRecipe((RegistrationStep(TranslationConfig()),), signal=stain)
fov.register_rounds(recipe, rounds=["morphology"])            # to the dataset reference round
atlas = ExternalReference(image, metadata, label="round1:ch04")
fov.register_rounds(recipe, rounds=["morphology2"], reference=atlas)
```

(inspecting-results)=
## Summaries and results by stage

`repr(dataset)` and `repr(fov)` are plain-text summaries of names and
structure. They never print array values or table rows, whatever the image
size. The dataset summary shows its IDs, sequencing rounds (reference marked
`*`), other rounds, channel order, codebook size and input/output roots; it
does not count FOVs. A fully run FOV looks like this:

```text
FOV 'FOV_001' of Dataset 'test' (sample 'small')
    images:   round1*, round2, round3, round4 — (16, 256, 256, 4) uint16 ZYXC   (* reference)
    channels: ch00, ch01, ch02, ch03
    results:  registration, spot_finding, extraction, decoding, filtering
      registration  3 moving rounds, translation
      spot_finding  68 spots × [spot_id, z, y, x, ...]
      extraction    68 spots × 4 channels × 4 rounds
      decoding      68 reads — assigned 66, no_signal 1, unmatched 1
      filtering     66 accepted / 68 (97.1%), rejected 2 — call_status 2
```

Stages that have not run are omitted, and rounds whose images are not
resident are listed as not loaded. `Codebook`, `SpotFindingResult`,
`IntensityExtractionResult`, `BarcodeDecodingResult`, `ReadFilteringResult`,
`RegistrationResult` and `EvaluationResult` each have a one-line summary with
the same counts. The filtering summary shows the accepted fraction and rejection
reasons only; precision and accuracy need truth and come from `evaluation`.
`EvaluationResult` shows its status and up to six metrics.

`fov.results` is a read-only mapping, in pipeline order, of the stages that have
run: `registration`, `spot_finding`, `extraction`, `decoding` and `filtering`.
Its values are the objects stored in `registration_results`, `spot_result`,
`intensity_result`, `decoding_result` and `filtering_result`; `registration`
maps each round label to that round's ordered `RegistrationResult` list. It
reflects the current state, including stages restored by `load_checkpoint`.

```python
fov.results["decoding"] is fov.decoding_result  # True
list(fov.results)  # stages that have run, in pipeline order
```

## Breaking Python migration

| Before | After |
| --- | --- |
| `STARMapDataset`, `LayerState` | `Dataset`, `RoundState` |
| `layers.seq`, `layers.other`, `layers.ref` | `rounds.sequencing_rounds`, `rounds.other_rounds`, `rounds.reference_round` |
| `all_layers`, `to_register` | `all_rounds`, `moving_rounds` |
| `STARMapDataset.from_config(config)` | `from_workflow_config(config, rule).dataset` |
| `load_raw_images`, `enhance_contrast`, `hist_equalize`, `morph_recon`, `tophat` | `load_images`, `normalize_intensity`, `match_histogram`, `reconstruct_background`, `filter_tophat` |
| `global_registration()`, `local_registration(method=...)` | `register(RegistrationRecipe((RegistrationStep(TranslationConfig()),)))`, `register(RegistrationRecipe((RegistrationStep(TpsConfig()),), signal=RegistrationSignalConfig("channel", 0)))` |
| `run_streaming(...)`, `run_streaming_gr(...)` | `run(config, execution=ExecutionConfig("streaming"))` |
| `all_spots`, `good_spots` | `spot_result`, `intensity_result`, `decoding_result`, `filtering_result.accepted` |
| `global_shifts`, `local_registered` | `registration_results`, `registration_chains`, `registration_attempts`, keyed by round |
| `save_signal`, `save_ref_merged`, `save_log`, `save_score_log` | `save_spots`, `save_reference_image`, `save_processing_log`, `save_diagnostics` |

FOV remains FOV. Path construction and logging helpers are private. No replaced
Python aliases remain. Rotation and output projection are explicit operation
configs rather than Dataset fields. Shared MATLAB config keys and filenames stay
unchanged; `from_workflow_config` is their single Python translation boundary.
The adapter rejects unknown fields in the selected rule, validates reference
round consistency and preserves workflow CPD's threshold 3/grid spacing 32.
Unsupported segmented endpoint filtering fails explicitly; it is not silently
reinterpreted as a single-segment predicate.

## Recovery and output boundaries

Recovery is disabled by default. For an explicitly recoverable landmark failure:

```python
from starfinder.dataset import RecoveryConfig
from starfinder.registration import DemonsConfig, TpsConfig, InsufficientLandmarksError
step = RegistrationStep(
    TpsConfig(),
    recovery=RecoveryConfig((InsufficientLandmarksError,), (DemonsConfig(),)),
)
```

An alternative must have its step's kind (global or local). Every estimation
attempt records `record="estimation"`, the step index, the attempt number,
requested and actual method, `fallback`, the backend that ran and its versions,
the reference round, the effective config, the outcome (`failed`, `rejected` or
`succeeded`), the failure (with the QC criterion for `rejected`) and, once
estimated, the step's `qc`. After the last step the round gets one
`record="application"` entry with its outcome, `application_config`, failure
and the whole chain's `qc`. Only listed estimation errors trigger ordered
alternatives; `RegistrationRejectedError` is an estimation error. Invalid
parameters, incompatible geometry, unavailable dependencies and application
errors propagate. A recovered result is labeled with its actual method.

`io.export_spots(detection, reads, path, accepted_only=True)` joins complete
filtering results to detections one-to-one on `(spot_namespace, spot_id)`, then
selects accepted rows. Decoding results also work with `accepted_only=False`.
Duplicate, missing or foreign keys fail; row order never defines identity.
Coordinates convert from zero-based ZYX to one-based XYZ exactly at export.
Empty detections and all-rejected results produce header-only CSVs. The default
columns and shared filenames remain compatible with downstream consumers.

`run(config, checkpoints=CheckpointConfig())` also saves registered images,
candidates with signals and pre-QC decoding per FOV, plus a `run.json` record.
`load_checkpoint(stage)` restores a stage so that a later `run` can continue
without earlier steps. See [checkpoints](checkpoints.md).

Two intentional corrections accompany coordination: registration signals are
float64 (the channel maximum by default, the sum with `mode="sum"`), which
preserves signed/high-range values, and rectangular subtiles cover both axes
and remainder pixels. Stage flags/parameters now apply equally
to batch and streaming; this can change results from legacy streaming recipes
that silently forced or omitted operations. Tests establish software behavior,
not scientific validation. MATLAB execution and historical notebook reruns are
excluded.
