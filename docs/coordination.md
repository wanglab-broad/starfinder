# Dataset and FOV coordination

`Dataset` owns paths, ordered rounds/channel labels, a codebook and FOV creation.
`FOV` stores images, metadata and structured results. Processing remains in public
functions. `PipelineConfig` describes the scientific sequence; `ExecutionConfig`
controls when images are loaded and released. Unknown config arguments raise.

```python
from starfinder.dataset import PipelineConfig, ExecutionConfig, RegistrationStep
from starfinder.io import ImageLoadConfig
from starfinder.registration import TranslationConfig
from starfinder.spot_finding import LocalMaximaConfig
from starfinder.barcode import NeighborhoodSumConfig, WtaDecoderConfig, ReadFilterConfig

config = PipelineConfig(
    load=ImageLoadConfig(channel_labels=dataset.channel_order),
    registration=(RegistrationStep(TranslationConfig()),),
    detection=LocalMaximaConfig(threshold_mode="noise", threshold_value=5),
    extraction=NeighborhoodSumConfig(neighborhood_radius_zyx=(1, 2, 2)),
    decoding=WtaDecoderConfig(),
    filtering=ReadFilterConfig(),
)
fov = dataset.fov("FOV_001").run(config, execution=ExecutionConfig("streaming"))
fov.save_spots()  # unchanged goodSpots filename, 1-based XYZ
```

Load the dataset codebook before decoding. None disables a stage. The sequence is
load, rotation, normalization, histogram matching, reconstruction, tophat,
projection, ordered registration, detection, extraction, decoding and filtering.
Legacy subtile workflows explicitly place reconstruction after registration.
Histogram matching uses a copy of the reference channel before downstream
processing. Both execution modes use this sequence and the same operation configs.
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
| `global_registration()`, `local_registration(method=...)` | `register(RegistrationStep(TranslationConfig()))`, `register(RegistrationStep(TpsConfig(), "single-channel", "single-channel"))` |
| `run_streaming(...)`, `run_streaming_gr(...)` | `run(config, execution=ExecutionConfig("streaming"))` |
| `all_spots`, `good_spots` | `spot_result`, `intensity_result`, `decoding_result`, `filtering_result.accepted` |
| `global_shifts`, `local_registered` | `registration_results`, `registration_attempts`, keyed by round |
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
from starfinder.registration import TpsConfig, InsufficientLandmarksError
step = RegistrationStep(
    TpsConfig(),
    recovery=RecoveryConfig((InsufficientLandmarksError,), (TranslationConfig(),)),
)
```

Every attempt records requested method, actual method, effective config, outcome
and failure. Only listed estimation errors trigger ordered alternatives. Invalid
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

Two intentional corrections accompany coordination: merged registration images
sum in float64 to preserve signed/high-range values, and rectangular subtiles
cover both axes and remainder pixels. Stage flags/parameters now apply equally
to batch and streaming; this can change results from legacy streaming recipes
that silently forced or omitted operations. Tests establish software behavior,
not scientific validation. MATLAB execution and historical notebook reruns are
excluded.
