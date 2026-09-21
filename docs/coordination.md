# Dataset and FOV coordination

`Dataset` owns paths, ordered rounds/channel labels, a codebook and FOV creation.
`FOV` stores images, metadata and structured results. Processing remains in public
functions. `PipelineConfig` describes the scientific sequence; `ExecutionConfig`
controls when images are loaded and released. Unknown config arguments raise.

An empty FOV can explicitly load [prepared or registered HDF5 image checkpoints](image-checkpoints.md)
with `load_image_checkpoint`. Loading restores saved values/state without rerunning
processing; continue registered data through the downstream methods directly.

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

Two intentional corrections accompany coordination: merged registration images
sum in float64 to preserve signed/high-range values, and rectangular subtiles
cover both axes and remainder pixels. Stage flags/parameters now apply equally
to batch and streaming; this can change results from legacy streaming recipes
that silently forced or omitted operations. Tests establish software behavior,
not scientific validation. MATLAB execution and historical notebook reruns are
excluded.

## Persistent run attribution

`starfinder.provenance.RunRecorder` observes the existing `FOV.run` sequence.
Use a fresh directory outside the checkout for each attempt:

```python
from starfinder.provenance import RunRecorder, read_run

record = RunRecorder(output_directory, dataset_id=dataset.dataset_id,
                     sample_id=dataset.sample_id, owner="analysis owner",
                     retention="retain through analysis handoff")
fov.run(pipeline, execution=execution, provenance=record)
run = read_run(record.path)
assert run["status"] == "succeeded"
```

`run.json` follows `starfinder.artifacts/1`. It holds requested pipeline/residency
config, per-operation effective parameters, ordered started/terminal events,
registration attempts and actual transforms, warnings, failures and recovery
links. `extensions["starfinder.provenance"]` contains initial/final state,
per-stage status, codebook identity and round registration state. Event snapshots
preserve each completed operation's geometry and counts before streaming releases
images. Successful empty detection has a succeeded event and zero count;
unassembled per-round signals retain a partial stage and `incomplete_stage`
omission rather than claiming a complete combined result.
Unavailable values remain null with reasons. Unknown physical calibration remains
null. Existing processing/shift log filenames and MATLAB contracts are unchanged.

Supply `code` with `commit`, `dirty`, `patch_sha256`, `snapshot_sha256`,
`package_version` and `source_location` for attributable work. Dirty code requires
a patch or source-snapshot SHA-256. Unknown revision/dirty values require
`unknown_reason`; the default explicitly marks them unknown and never guesses
from a neighboring checkout. `seed` carries caller-supplied seed/stream identity.
The environment records dependency/Python versions, host, backend and a fixed
thread/device allowlist; it does not dump environment variables or credentials.
Resource allocation remains explicitly unknown unless documented by the external
execution manifest. Explicit caller metadata and exception messages should not
contain secrets.

Optional `sources` entries carry `source_id`, `catalog`, `uri`, `sha256`,
`unverified_reason` and a `selection` object. Null hashes need a nonempty reason.
Resident arrays receive C-order byte hashes with shape/dtype/geometry in their
selection. Loaded TIFFs receive file hashes and per-channel selection/original
geometry, separately from derived output geometry. A source path or content hash
does not establish acquisition lineage or public accessibility.

Compact transforms stay in typed JSON metadata. Dense transforms and numerical
array diagnostics use lossless, checksummed NPY components with explicit dtype
and shape, loaded without pickle. Tagged nonfinite diagnostic floats distinguish
infinity/NaN from missing values. `read_run(path, sha256=...)` can also pin the
exact manifest bytes. It checks schema, references, geometry, component paths,
size and checksum before returning records; it does not execute processing or
instantiate arbitrary serialized configuration types.

The recorder saves provenance and, by default, the complete
[candidate/signal checkpoint](candidate-checkpoints.md) after extraction and
before decoding/QC. This requires the optional `checkpoint` extra. Explicitly
set `save_candidates_signals=False` for provenance-only operation. Other unlinked
payloads have omitted records, and unsuccessful stages have failed records;
these records alone are not reloadable checkpoints. A checkpoint writer can call
`record.record_artifact(record_dict)` while the run is active, after validating
its format-specific payload; components must already exist under the run root.
The recorder verifies those component hashes and links their immutable IDs.
Image and candidate/signal payloads remain separate APIs owned by their format
implementations. Decoded/final table persistence is a separate delivery.

Writes publish `run.json` last by atomic replacement. The same recorder/directory
cannot be reused. A caught interruption is recorded as interrupted; a hard kill
may leave running plus a started event, which the reader never promotes to
success. Serialization/I/O failures propagate, even if recording that failure is
itself impossible. The last published state and orphan temporary/components are
inspectable, never silently completed or regenerated. No resume scheduler or
automatic cleanup is added. The run's own checksum belongs in an external
handoff manifest; benchmark/controller logs, commands, resource measurements and
retention evidence remain outside the software artifact schema.

Run the bounded Z=1/3 example from `src/python` with a new output path:

```bash
uv run python ../../docs/examples/provenance.py /external/new-run/provenance
```

It uses two rounds/four channels, literal uint16 `(Z,7,9,4)` arrays and no random
seed or historical TIFFs. This checks attribution and software behavior, not
scientific validity or image/table reload equivalence.
