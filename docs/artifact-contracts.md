# Artifact, checkpoint and provenance contracts

**Contract ID: `starfinder.artifacts/1`.** This is the W-154 specification for
processing persistence. W-156 implements run/event/failure metadata and diagnostic
components through `starfinder.provenance`; see [usage](coordination.md#persistent-run-attribution).
W-158 implements opt-in per-FOV HDF5 prepared inputs and registered image
checkpoints; see [image checkpoint usage](image-checkpoints.md). W-159 implements
the pre-rejection [Parquet candidate/signal checkpoint](candidate-checkpoints.md).
Decoded/final table payload writers remain a separate delivery.
[Array contracts](api/contracts.md) remain authoritative for numerical APIs.
The acceptance cases below freeze independent expectations before W-156,
W-158 and W-159 implement persistence. A producer records this contract ID and
the exact source revision (or reviewed patch hash); controller acceptance of
W-154 establishes the implementation baseline. Scientific acceptance is separate.

## Scope and existing types

Persistence adapts existing values, without introducing a scheduler, restart
engine, monolithic sample store or parallel processing API:

| Payload | Existing Python structure to reuse |
| --- | --- |
| Image and per-layer geometry | `io.ImageLoadResult`, `image.ImageMetadata` |
| Acquisition roles and order | `dataset.RoundState` and explicit channel labels |
| Registration and application | `RegistrationResult`, `TranslationTransform`, `DenseDisplacementTransform`, `WarpConfig` |
| Candidates | `spot_finding.SpotFindingResult` |
| Signals and availability | `barcode.IntensityExtractionResult`, `NeighborhoodSumConfig` |
| Codebook and encoding | `barcode.Codebook`, `EncodingConfig` |
| Decoded pre-QC calls | `barcode.BarcodeDecodingResult` |
| QC accounting and accepted view | `barcode.ReadFilteringResult`, `ReadFilterConfig` |

The existing benchmark manifest is a separate schema. Reuse its checksum,
immutable-run and explicit-failure principles, not its trial/evaluation model
as a processing dependency. Existing CSV, TIFF and subtile NPZ readers retain
their current contracts; they are not implicitly upgraded into v1 checkpoints.
MATLAB-facing filenames, keys and one-based CSV coordinates remain unchanged.

## Versioned records and identity

The following are normative v1 record fields. JSON records are UTF-8 objects;
required fields must be present even when their documented value is null.
Unknown facts use null plus a reason, never a fabricated value. JSON numbers
must be finite; nullable diagnostics represent undefined values with null and
an accompanying reason. Existing nonfinite score diagnostics (for example WTA
infinity for a tied round) use explicit tagged values in JSON: an object with
`float_special` equal to `nan`, `+inf` or `-inf`. This is distinct from missing
data and must round-trip to the original numerical value. Parquet score columns
retain IEEE special values separately from null masks. Images/signals remain
finite as required by their numerical APIs. Arrays and tables hold numerical payloads, not JSON
lists of image pixels. Typed configurations serialize a type discriminator and
all effective fields (including defaults); no pickle, import path execution or
arbitrary constructor loading is permitted.

Each record has `schema_name` (one of `starfinder.run`, `starfinder.artifact`,
`starfinder.event`, `starfinder.failure`), integer `schema_version=1`, and
`contract_id="starfinder.artifacts/1"`. Readers reject unsupported versions or
names, missing required fields, wrong types and unknown enum values before
returning payloads. Optional extensions live under `extensions`, keyed by
producer namespace, and cannot override core meaning. No heuristic migration
of legacy files. A future migration must explicitly name both versions and
preserve the source bytes. New required fields or changed semantics require a
new schema version; adding optional namespaced extensions does not.

| Record | Required content beyond the common header |
| --- | --- |
| Run | `run_id`, UTC `created_at`, `status`, dataset/sample identity, `code`, `environment`, requested/effective `config`, `sources`, ordered `artifacts`, `events`, `failures`, `saving_policy`, `owner`, `retention`, `backup_status` |
| Artifact | `artifact_id`, `run_id`, `stage`, `status`, dataset/sample/FOV/subtile identity, `parents`, `config_ref`, `source_refs`, `components`, `payload`, `omission_reason`, `failure_id` or null |
| Event | `event_id`, `run_id`, `sequence`, UTC `timestamp`, `stage`, FOV/round or null, `attempt`, `operation`, `outcome`, `requested_method`, `actual_method`, `config_ref`, input/output artifact IDs, `diagnostics`, `failure_id` or null |
| Failure | `failure_id`, `run_id`, `event_id`, `stage`, FOV/round or null, `category`, exception `type`, `message`, `traceback_ref` or null, requested/actual method, `recovery_action`, `recovery_event_id` or null |

Run status is `running`, `succeeded`, `failed` or `interrupted`. Artifact status
is `complete`, `omitted` or `failed`; only complete artifacts are loadable.
Event outcomes are `started`, `succeeded`, `failed`, `skipped` or `recovered`.
A successful empty result is `complete`, not a failure or an omitted artifact.
An omitted artifact has empty components and a nonempty omission reason
(e.g. `images_disabled`); a failed artifact points to its failure record.
Complete artifacts have null omission reason and failure ID. Run `artifacts`,
`events` and `failures` are ordered record references (embedded or checksummed
relative files); IDs must resolve uniquely within the run. Event `sequence` is
a nonnegative, strictly increasing integer and `attempt` is a positive integer
within the stage/FOV/round scope. Timestamps are ISO 8601 UTC strings; order is
defined by sequence, not clock resolution. Operations are nonempty names of
the actual invoked stage functions; requested/actual methods may be null with
reason for non-method operations. Diagnostics are named typed values, not prose
substitutes for status or failure fields.
A recovered estimation failure remains a failure record even when the run
succeeds using an explicitly configured fallback. Actual method never silently
inherits the requested method. Failure categories are `invalid_input`,
`missing_input`, `estimation`, `application`, `serialization`, `integrity`, or
`interrupted`; they do not automatically authorize retries.

`code` records commit, dirty flag, patch/snapshot hashes if dirty, package version
and source location. `environment` records Python/dependency versions, backend,
host, thread/device configuration and available resource/seed information;
unobserved values remain null with reasons. `config_ref` and other references
identify an embedded record or a relative file with SHA-256. Hash stored bytes,
not reserialized JSON. The run manifest is not self-hashed: its checksum belongs
to the external handoff or the record that references it.

Source records carry a stable `source_id`, dataset version/catalog locator,
URI/path, SHA-256 or an explicit unverified reason, and source selection
(series/time/channel/dataset key, axes, crop or other conversion). Preserve the
supplied selection and original geometry separately from derived geometry.
A path alone is not source identity or proof of public accessibility.

Run IDs are unique across attempts; artifact/event/failure IDs are unique within
the run and externally addressed by `(run_id, id)`. Dataset/sample/FOV identifiers
are nonempty strings; subtile is the existing one-based integer or null.
Candidate keys are **(spot_namespace, spot_id)**. Preserve the existing FOV JSON
array namespace; record the creating detection artifact as well. Independent
redetection must use a distinct namespace, with the detection run/version in its
scope. Do not regenerate IDs after filtering, sorting, batching or reload.
Synthetic truth identities have their own namespace and are not detection IDs.
No equality or join between independently detected spots is assumed.

## Logical stages and saving policy

| Stage value | Required payload and invariant | Default retention |
| --- | --- | --- |
| `prepared_input` | Supplied image layers, each with role, round/channel selection, dtype, geometry and source mapping; no implicit preprocessing or registration | Images opt-in |
| `registered_images` | Saved per-round arrays with actual operation history, reference identity, transforms/application config and output geometry; retain reference and failed/skipped round state | Images opt-in |
| `candidates_signals` | Complete pre-decoding candidate table plus N×C×R signals and N×R validity; detection/extraction configs, geometry and ordered labels | Provisionally on; explicit disable |
| `decoded_pre_qc` | One call row for every input candidate, codebook snapshot, decoder settings, statuses/reasons and method-specific scores/diagnostics | On when decoding runs |
| `final_accepted` | Accepted view plus complete QC decision/accounting table, filter config, counts and undefined-fraction reasons; link pre-QC and candidates | On when filtering runs |

Configuration, provenance, diagnostics and final outputs are retained by
default. Disabling candidate/signals persistence must record the omission and
that source traces cannot be reloaded; it does not disable computation or final
identity retention. Images are independently opt-in, not controlled by a size
threshold. Candidate-only output is optional and is not a combined checkpoint.
These are provisional software defaults pending W-171 storage qualification;
no automatic deletion policy or empirical storage claim is implied.

Prepared and registered containers must have distinct stage declarations.
Loading a registered checkpoint never reruns preprocessing/registration. Store
for each round the ordered completed operations and effective configs, source
and output metadata, registration attempts, actual applied transform (or null
with reason for an unchanged reference/skipped operation), and terminal state.
An unregistered round cannot masquerade as registered merely by adopting the
reference frame ID. Failed/partial runs remain inspectable; a downstream
operation requiring a complete set must reject missing/failed rounds.

Decoded tables retain existing `assigned`, `unmatched`, `ambiguous`, `no_signal`
semantics. QC retains all input identities, acceptance flags and rejection
reasons, including all-rejected and empty populations. Persist existing fields
and their dtypes rather than inventing a universal score. Cellular assignment,
assembled sample populations, overlap reconciliation, exporter-specific payloads
and new scientific QC fields remain W-152/§§2.8–2.10/W-168 decisions. They gate
only those future payloads, not the image or candidates/signals checkpoints.

## Geometry, transforms and physical components

Every image layer records axes (`ZYX` or `ZYXC`), shape, NumPy dtype descriptor,
ordered channel labels and all five `ImageMetadata` fields: `frame_id`,
`spacing_zyx`, `origin_zyx`, `direction_zyx`, `spatial_unit`. Null remains null.
Z=1 is a volume; do not squeeze it. Stains, sequencing images and registration
references can have different shapes/frames: prepared storage does not align
them. Consumers explicitly validate compatible geometry before computation.
Preserve supplied integer width/sign and float width; loading cannot normalize,
rescale, crop, project, reorder channels or coerce to uint8.

Transforms retain their kind, reference/moving shapes and metadata, direction,
units, exact correction or field dtype/values, application config and diagnostics.
Translation is content correction with pull `moving[p - correction_zyx]`;
dense displacement is pull `moving[p + displacement_zyx[p]]`. No implicit
negation, inverse or composition. Reference geometry labels the applied output.
Unknown calibration permits index-space operations but forbids physical
conversion; an identity direction/zero origin must not replace unknown fields.

A component descriptor contains `component_id`, relative `path`, `format`, byte
`size`, `sha256`, and format-specific `key`/schema/shape/dtype. Paths cannot escape
the artifact root; absolute/traversal paths and duplicate component IDs error.
References to external original sources are allowed only in source records.
Parent references include artifact identity and manifest checksum, binding a
checkpoint to the exact configuration, codebook, source selection and geometry.

### Image layout (W-158)

Use per-FOV HDF5, with one dataset per layer at `/layers/<opaque-layer-id>/image`
and an ordered layer inventory in the artifact payload. Layer IDs are safe opaque
keys, not raw user labels interpreted as paths. Dataset attributes bind artifact
ID and layer ID; the manifest carries authoritative geometry/role/order and the
reader checks consistency with shape/dtype. Roles include `sequencing`, `stain`
and `registration_reference`; one layer may list multiple roles. RoundState
still records sequencing/other/reference membership separately.

Prepared layers may use ZYX or ZYXC; an extraction consumer requires explicit
ZYXC construction and compatible labels. Dense fields use separate datasets,
not image channels. Initial lossless image chunks are axiswise
`min(shape, (8,64,64))`, plus channel chunk 1 for ZYXC, gzip level 4 with shuffle.
Dense fields use spatial chunks bounded the same way and component chunk 3.
Record actual chunks/codec settings. These are bounded initial engineering
choices, not measured optimal defaults. No lossy filter or dtype conversion.

### Candidate and signal layout (W-159)

One logical per-FOV checkpoint consists of a manifest, `candidates.parquet`,
`signals.parquet`, `validity.parquet` and a codebook snapshot when available.
A codebook is not required to extract; decoding requires the exact supplied
codebook and records its identity in the decoded artifact. Existing extraction
requires C,R > 0 and float64 finite values. Coordinates are float64 ZYX.

| Component | Required physical columns and types | Key / cardinality |
| --- | --- | --- |
| Candidates | `spot_namespace`, `spot_id`: nonnull UTF-8; `candidate_index`: int64; `z,y,x`: float64; optional detector columns with declared dtypes | Identity unique; index exactly 0..N−1 |
| Signals | namespace/ID, `channel_index`, `round_index`: int64, `value`: float64 | Exactly N×C×R unique identity/channel/round keys |
| Validity | namespace/ID, `round_index`: int64, `valid`: nonnull Boolean | Exactly N×R unique identity/round keys |
| Codebook | Ordered `gene_id`, `color_sequence`, optional `base_sequence`: UTF-8; `codebook_index`: int64 | Unique genes/sequences; explicit row order |

Namespace/ID in signals and validity are the same UTF-8 types as candidates.
The manifest holds ordered unique `round_labels` and `channel_labels`, tensor
axis names `NCR`/`NR`, N/C/R, one namespace, geometry, detector/extractor configs
and diagnostics, column dtype/nullability schema, and codebook encoding plus
`color_to_channel`. Color symbols are not channel indices; the current Codebook
requires four channels and a bijection for symbols `1`–`4`. Preserve a nontrivial
mapping and nonlexical acquisition order. Table schemas restore pandas string,
nullable integer/Boolean and numeric widths explicitly, including empty tables.
Unknown optional columns must be declared and preserved, not silently discarded.

This long representation avoids engine-specific nested-array reshaping. On
load, allocate N×C×R and N×R from manifest dimensions, join by identity, and
assign by declared indices. Physical row order is irrelevant. Missing or
duplicate rows, out-of-range indices, null core fields, conflicting namespace,
extra identities and nonfinite signals error. `valid=False` is unavailable,
not zero: preserve its finite stored signal too. Missing rows cannot be filled
with zero/false. Empty N=0 still has typed tables and nonempty label axes.

Initial Parquet settings: lossless Zstandard, row groups at most 65,536 rows,
dictionary encoding permitted for string IDs. Record actual engine/version,
codec and row-group size. No pandas index column is authoritative. An engine
must be an explicit scoped dependency before W-159 execution; no automatic
installation or fallback to CSV. HDF5 and Parquet qualification belong to their
implementation issues, not this specification.

### FOV, sample batches and trace lookup

Group artifacts by dataset/sample/FOV/stage using manifest IDs; opaque directory
keys map to original labels in the index (never parse labels from filenames).
No partition per spot/channel/round is required. Whole-FOV reads reconstruct the
existing result structures; a sample index is an ordered list of per-FOV artifact
references, not a mandatory consolidated file. Candidate row order is restored
by `candidate_index`. Batches enumerate FOV index order then candidate index,
without splitting a candidate's signals or validity across returned batches.
Concatenated batches equal whole-FOV concatenation exactly, including empty
FOVs, namespaces, dtypes and labels. Different acquisition schemas cannot be
silently stacked into one tensor; return FOV-scoped batches or reject that request.

A source-trace key carries `(run_id, candidate_artifact_id, spot_namespace,
spot_id)`. Resolve the pinned artifact, validate checksums, find the candidate,
and return its C×R signal, R validity, coordinates/geometry, acquisition mappings
and source/config/codebook references. Never choose the latest matching FOV or
join by gene/coordinates alone. Final/sample tables carry this locator without
duplicating all traces. If persistence was disabled, return an explicit
unavailable reason; an unknown identity in a complete checkpoint is an error.
Sample assembly semantics and cell links are deferred; this is only access and
source identity, not an assembly algorithm.

## Integrity and failure behavior

Writers validate payloads before publishing a complete manifest. Write components
under a fresh artifact identity, close and hash them, then publish the complete
manifest last. An interrupted write without a complete manifest is not loadable.
Do not overwrite a completed artifact; changed config creates a new artifact/run
with explicit parent links. A later reader verifies component existence, size,
checksum, schema, dimensions and cross-component invariants before returning a
complete result. It must not return partially reconstructed success on error.

| Condition | Required behavior |
| --- | --- |
| Unsupported version/stage or malformed metadata | ValueError with artifact/component and field context; no guessing |
| Required path missing | FileNotFoundError with artifact/component identity; no regeneration |
| Size/checksum mismatch, truncated data, inconsistent IDs/shapes/labels | ValueError identifying integrity failure; preserve original bytes |
| Optional backend missing | ImportError naming the scoped dependency; no sync/install or silent format switch |
| Complete empty table | Valid typed result with N=0; counts 0 and undefined fractions null/reason |
| Failed estimation/application | Preserve attempt/failure and requested/actual backend; only explicit recovery config permits fallback |
| Unavailable optional checkpoint | Report omitted status/reason; do not claim source-trace/downstream reload equivalence |

I/O exceptions may be chained as causes. Run-wide diagnostics record the
category and location without converting invalid input into a successful empty
result. Log persistence failure must remain visible to the caller even when a
failure record itself cannot be written.

## Independent acceptance examples and numerical rules

The executable example below builds tiny deterministic inputs, not simulated
molecular truth. It exercises existing numerical APIs; it does **not** emulate a
checkpoint writer and cannot pass future storage round-trip tests by itself.
No random seeds or historical TIFFs are used. Maximum image shape is (3,4,5,4),
R=2; singleton-Z repeats the same model. Literal expectations are fixed here:

* Geometry: spacing (2,3,4), origin (10,20,30), direction diag(1,−1,−1)
  maps index (1,2,3) to world (12,14,18); inverse recovers the index exactly.
  Crop start (1,1,2) gives origin (12,17,22). Unknown calibration stays null.
* Integer correction (0,1,−1) moves a value 7 from (z,1,2) to (z,2,1), with
  zero-filled edges, unchanged uint16. Save/load preserves that already-applied
  result, never applies the correction twice.
* Dense linear pull on `[0,1,2,3,4]` with dx=0.5 gives interior
  `[0.5,1.5,2.5,3.5]`; uint16 output is `[0,2,2,4]` (nearest-even), and
  floating output retains the fractions. This tests sampling/rounding, not
  estimation accuracy. Boundary comparison is separate from interior values.
* Radius-zero extraction for candidate A at (z,1,2) has channel-by-round matrix
  `[[0,9],[7,0],[0,0],[0,0]]`; candidate B at (z,2,3) has all zeros.
  Labels are `("round10","round2")` and `("ch02","ch00","ch03","ch01")`.
  Mapping `1→1, 2→0, 3→3, 4→2` gives A code `12` and gene `gene-A`;
  B is `no_signal`. Default filtering accepts A and retains B's rejection.
  Invalidating A's second round prevents assignment without dropping its ID.

Lossless persistence requires exact equality of identity/order, labels, strings,
null masks, Boolean validity, configs, geometry, integer values and floating
values/dtypes (including signed zero for numerical arrays); no tolerance is
allowed to excuse serialization changes. File hashes identify particular bytes;
two independently written files may differ in container metadata yet reconstruct
exactly equal payloads. Nullable score columns preserve missingness; no NaN-to-zero
conversion. The current computational API rejects nonfinite images/signals.

Rerunning deterministic downstream decoding/filtering in the same pinned
backend/environment on an exactly reloaded checkpoint must equal uninterrupted
results exactly after identity alignment, including statuses, scores and QC
counts. A comparison across numerical backends is a separate W-171 protocol,
not an automatic relaxation of this requirement. Independent computation uses
`rtol=0`: exact equality for the representable integer/half-integer cases above;
`atol=1e-12` for the small float64 WTA log score (elementary-function roundoff);
existing signed Fourier fixtures use `atol=2e-6` for float32 or `1e-12` for
float64 per [resampling contracts](api/contracts.md#translation-edge-cases-and-resampling-precision).
These bound numerical rounding in specific small examples, not scientific
accuracy or full-volume registration. A new computation needs its own analytic
expectation and justified tolerance before execution.

### Required downstream acceptance matrix

Every implementation cites `starfinder.artifacts/1`, this document's accepted
Git revision (or reviewed patch hash), and the applicable case IDs in its run
manifest and Linear evidence. Unreviewed W-154 edits are not an approved baseline.

| Case | Owner | Required validation beyond the executable example |
| --- | --- | --- |
| P1 | W-156 | Successful, empty, failed, recovered and interrupted run/event/failure records; immutable identity and checksum links; serialization failure visible; batch/streaming retain the same scientific stage order |
| I1 | W-158 | Prepared image round trip for 3D/Z=1, uint8/uint16/signed float, nonlexical channels, unequal stain/reference geometry and unknown calibration; exact bytes/metadata values |
| I2 | W-158 | Registered multi-round round trip with compact/dense transforms and actual state; extraction/decode/filter after reload exactly equals uninterrupted processing; no double application |
| C1 | W-159 | Combined candidates/signals round trip of the literal example, empty/all-invalid/signed-signal cases and optional columns; preserve dtype, IDs, labels, config, coordinates, validity and codebook mapping |
| C2 | W-159 | Deliberately shuffle all physical table rows; restore original candidate order and tensors; reject missing/duplicate/extra keys or altered labels; same-ID different-namespace FOVs stay distinct |
| C3 | W-159 | Decode/filter reloaded versus uninterrupted example exactly, including failures for negative-policy mismatch; whole-FOV loading and source-trace identity checks |
| D1 | W-164 (future) | Separate pre-QC and final records, complete rejected population/accounting, all-rejected/empty schemas, rerun filter without images; no invented scientific fields |
| S1 | W-164/W-168 (future) | Sample batches concatenate exactly to full reads; source trace resolves pinned candidate artifact; omitted traces explicit; assembly/cell policy requires its own approved specification |
| X1 | Each writer/reader owner | Missing/truncated/corrupt component; unsupported version; mismatched stage/config/geometry; incomplete write; no partial-success return |
| E1 | W-155/W-157/W-160 | Synthetic specification/truth links use distinct namespaces and v1 provenance; new fixture catalog entry; saved 3D/Z=1 example and independent assertions in standalone review packet |

The future rows are specifications, not authorization to execute later batches.
No storage round trip, sample reader, Parquet backend, viewer or MATLAB execution
is certified by this document. W-173 human review and W-57/W-93 scientific
qualification remain independent gates.

### Run the bounded specification example

From `src/python` in the prepared environment:

```bash
uv run python ../../docs/examples/artifact_contracts.py
```

The script writes no image files and ends with `Artifact contract examples passed.`
The controller's normal test/docs/reference gates remain required. The example
is also exercised by a focused pytest entry point.

```{literalinclude} examples/artifact_contracts.py
:language: python
```
