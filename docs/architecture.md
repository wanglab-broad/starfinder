# Architecture

STARfinder separates numerical functions from workflow state. Direct callers and
Dataset/FOV coordination use the same processing implementations. Batch and
streaming select image residency, not different scientific pipelines.

| Namespace | Responsibility | Boundary |
| --- | --- | --- |
| `assembly` | Sample coordinates, overlap ownership, retention, accounting and the sample tables | Does not place FOVs, fuse images or write the export |
| `assignment` | Molecules to cells, nucleus–cell correspondence, compartments and counts on one FOV's grid | Does not segment, stitch or run inside `FOV.run` |
| `benchmark` | Cases, timing, manifests, saved-output evaluation/reporting | Does not implement algorithms or select hidden scientific defaults |
| `dataset` | Dataset/FOV inputs, rounds, typed pipeline and execution policy | Calls public processing functions |
| `evaluation` | Pure metrics and explicit matching policies | Does not detect, register, rerun or write reports |
| `export` | Sample export: fusion, labels, SpatialData, the export check and the route comparison | Does not place FOVs or assemble molecule tables |
| `image` | ImageMetadata, spatial validation/conversion | Does not own intensities or acquisition state |
| `io` | Image/table persistence and explicit conversion | Does not process experiments |
| `preprocessing` | Typed finite-array operations | Does not own FOV state |
| `readout` | Codebook validation, extraction, decoding, filtering | Does not load images or select benchmarks |
| `registration` | Estimate/apply transforms and diagnostics | Does not evaluate truth or time experiments |
| `report` | The processing report of one sample and the comparison page of a dataset, read from saved records | Judges nothing: no verdict, filtering or correction, and does not rerun stages |
| `segmentation` | Label images on a reference grid, mask import and label functions | Does not assign molecules or run inside `FOV.run` |
| `spot_finding` | Detection and stable SpotFindingResult identities | Does not perform registration evaluation |
| `stitching` | Tile positions, neighbour pairs, the placement record and the stitch entry | Does not assemble tables or fuse images |
| `synthetic` | Processed-image scenes, rendering and truth records | Does not orchestrate benchmarks or invent molecular truth |

## Processing flow

Per FOV: load → preprocess → estimate/apply registration → find spots → extract
intensities → decode reads → filter reads → molecule table; then segment and
assign molecules to cells, which are separate entries outside `FOV.run`. Per
sample: stitch the FOVs into one placement → assemble the sample tables → export
the fused image, labels and SpatialData → write the processing report.
See [Workflow](workflows.md) for Snakemake and [coordination](coordination.md)
for typed Python stage configuration and residency.

Images remain NumPy arrays with separate spatial metadata. Config dataclasses
select methods and validate parameters; structured results retain identity,
geometry and diagnostics. The methods of preprocessing, registration,
spot finding, decoding, segmentation and stitching, and the readout encodings, are kept
in registries built on the shared helper `starfinder._registry`; the
[method registry](method-registry.md) describes how they are named in YAML and
recorded in provenance. Failed estimation is distinct from invalid input,
empty detection and undefined evaluation. Recovery is explicit in coordination.

## Where to find the contract

[Python contracts](api/contracts.md) define shapes, dtype, physical geometry,
transforms, spot identities and the readout stages; their
[MATLAB boundary](api/contracts.md#matlab-boundary) section explains the
conversions at the Python↔MATLAB boundary; [API](api/index.md) lists supported interfaces.
[Migration](migration.md) distinguishes mechanical changes from corrected
numerical behavior. MATLAB remains a separately documented backend; no parity
or runtime validation is implied by corresponding names.

At the repository root, CONTEXT.md gives concise orientation and AGENTS.md gives
operational instructions. The site holds maintained software contracts rather
than execution diaries.
