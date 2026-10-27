# STARfinder context

STARfinder processes STARmap-related microscopy into molecule tables, cell
segmentations, cell-by-gene counts and sample-level outputs.
Python functions implement processing; Dataset and FOV coordinate them;
Snakemake coordinates file-based workflows with Python or MATLAB backends.
Segmentation and read-to-cell assignment are separate entries after `FOV.run`.

## Vocabulary

A **round** is an acquisition cycle; a **channel** is a spectral image within it.
Python images are ZYX or ZYXC; 2D uses singleton Z. Spots are zero-based ZYX,
with subpixel coordinates and identity `(namespace, spot_id)`. ImageMetadata
carries spatial interpretation separately from intensities. Unknown calibration
stays unknown. Python names use snake_case and PascalCase, with **FOV** retained.
MATLAB API names, and the files and keys MATLAB code reads or writes, are excluded
from renaming.

## Responsibilities and data flow

`io` → `preprocessing` → `registration` → `spot_finding` → `readout` is the
per-FOV flow, followed by `segmentation` and `assignment`; `stitching` →
`assembly` → `export` → `report` is the per-sample flow. `image` owns metadata;
`dataset` owns coordination.
`synthetic` produces processed images/truth; `evaluation` computes pure metrics;
`benchmark` runs explicit cases, measures resources and writes reports.
Algorithms do not depend on benchmark orchestration. Registration estimates a
correction transform, then applies it directly. Intensity extraction, read
decoding and read filtering are independent readout stages that retain spot identity.

Use the [architecture](docs/architecture.md), the
[contracts](docs/api/contracts.md) (with their MATLAB boundary section) and the
[migration guide](docs/migration.md) as maintained detail; the specifications, validation
designs and baselines behind each stage are design records in [design/](design/README.md).
[AGENTS.md](AGENTS.md) owns operational instructions;
[Linear](https://linear.app/jiahaoh/document/thesis-and-implementation-workflow-c12f30bffe9f)
owns current plans, decisions and evidence. This file is orientation, not a run log.
