# Architecture

STARfinder processes microscopy images into molecules, cell assignments and sample
outputs. Numerical functions do the processing; `Dataset` and `FOV` coordinate
sample inputs and results. The {doc}`coordination guide <coordination>` shows how to
run a typed Python pipeline.

| Namespace | Responsibility |
| --- | --- |
| `assembly` | Sample coordinates, overlap ownership, retention, accounting and the sample tables |
| `assignment` | Molecules to cells, nucleus–cell correspondence, compartments and counts on one FOV's grid |
| `benchmark` | Cases, timing, manifests, saved-output evaluation/reporting |
| `dataset` | Dataset/FOV inputs, rounds, typed pipeline and execution policy |
| `evaluation` | Pure metrics and explicit matching policies |
| `export` | Sample export: fusion, labels, SpatialData, the export check and the route comparison |
| `image` | ImageMetadata, spatial validation/conversion |
| `io` | Image/table persistence and explicit conversion |
| `preprocessing` | Typed finite-array operations |
| `readout` | Codebook validation, extraction, decoding, filtering |
| `registration` | Estimate/apply transforms and diagnostics |
| `report` | The processing report of one sample and the comparison page of a dataset, read from saved records |
| `segmentation` | Label images on a reference grid, mask import and label functions |
| `spot_finding` | Detection and stable SpotFindingResult identities |
| `stitching` | Tile positions, neighbour pairs, the placement record and the stitch entry |
| `synthetic` | Processed-image scenes, rendering and truth records |

## Processing flow

Per FOV: load → preprocess → register → find spots → extract intensities → decode
reads → filter reads. Segmentation and assignment are separate calls after
`FOV.run`. The pipeline also offers scoring and deduplication before filtering.

Per sample: stitch FOVs into a placement → assemble tables → export images, labels
and SpatialData → write the processing report. Sample segmentation offers a route
that segments the fused image. See the {doc}`pipeline-stage index <wiki/pipeline-stages>`
for the guide and reference of a stage, and {doc}`workflows` for Snakemake.

## Shared representations

Images are NumPy arrays with separate `ImageMetadata`. Config dataclasses describe
processing choices; results retain coordinates, identities and diagnostics.
The {doc}`array and coordinate reference <api/contracts>` gives their conventions,
including the [MATLAB boundary](api/contracts.md#matlab-boundary).

The {doc}`API index <api/index>` lists public interfaces. Detailed specifications
are kept in [design/](https://github.com/wanglab-broad/starfinder/blob/dev/design/README.md).
