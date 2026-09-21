# Architecture

STARfinder separates numerical functions from workflow state. Direct callers and
Dataset/FOV coordination use the same processing implementations. Batch and
streaming select image residency, not different scientific pipelines.

| Namespace | Responsibility | Boundary |
| --- | --- | --- |
| `barcode` | Codebook validation, extraction, decoding, filtering | Does not load images or select benchmarks |
| `benchmark` | Cases, timing, manifests, saved-output evaluation/reporting | Does not implement algorithms or select hidden scientific defaults |
| `dataset` | Dataset/FOV inputs, rounds, typed pipeline and execution policy | Calls public processing functions |
| `evaluation` | Pure metrics and explicit matching policies | Does not detect, register, rerun or write reports |
| `image` | ImageMetadata, spatial validation/conversion | Does not own intensities or acquisition state |
| `io` | Image/table persistence and explicit conversion | Does not process experiments |
| `preprocessing` | Typed finite-array operations | Does not own FOV state |
| `registration` | Estimate/apply transforms and diagnostics | Does not evaluate truth or time experiments |
| `spot_finding` | Detection and stable SpotFindingResult identities | Does not perform registration evaluation |
| `synthetic` | Processed-image scenes, rendering and truth records | Does not orchestrate benchmarks or invent molecular truth |

## Processing flow

Load → preprocess → estimate/apply registration → find spots → extract
intensities → decode barcodes → filter reads → export molecule tables.
Segmentation and read-to-cell assignment are separate downstream operations.
See [Workflow](workflows.md) for Snakemake and [coordination](coordination.md)
for typed Python stage configuration and residency.

Images remain NumPy arrays with separate spatial metadata. Config dataclasses
select methods and validate parameters; structured results retain identity,
geometry and diagnostics. Failed estimation is distinct from invalid input,
empty detection and undefined evaluation. Recovery is explicit in coordination.

## Where to find the contract

[Artifact contracts](artifact-contracts.md) specify versioned run records,
image and candidates/signals checkpoints, source lookup and downstream
acceptance cases. They reuse the existing typed results; persistence delivery
is tracked separately from specification and scientific qualification.

[Python contracts](api/contracts.md) define shapes, dtype, physical geometry,
transforms, spot identities and barcode stages. [Conventions](conventions.md)
explain boundary conversions; [API](api/index.md) lists supported interfaces.
[Migration](migration.md) distinguishes mechanical changes from corrected
numerical behavior. MATLAB remains a separately documented backend; no parity
or runtime validation is implied by corresponding names.

At the repository root, CONTEXT.md gives concise orientation and AGENTS.md gives
operational instructions. Linear holds live scope, decisions and run evidence;
the site holds maintained software contracts rather than execution diaries.
