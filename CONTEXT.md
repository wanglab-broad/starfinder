# STARfinder context

STARfinder processes STARmap-related microscopy into molecule-level tables.
Python functions implement processing; Dataset and FOV coordinate them;
Snakemake coordinates file-based workflows with Python or MATLAB backends.
Cell segmentation, read assignment and cell matrices are downstream work.

## Vocabulary

A **round** is an acquisition cycle; a **channel** is a spectral image within it.
Python images are ZYX or ZYXC; 2D uses singleton Z. Spots are zero-based ZYX,
with subpixel coordinates and identity `(namespace, spot_id)`. ImageMetadata
carries spatial interpretation separately from intensities. Unknown calibration
stays unknown. Python names use snake_case and PascalCase, with **FOV** retained.
MATLAB API names and shared workflow/CSV contracts are excluded from renaming.

## Responsibilities and data flow

`io` → `preprocessing` → `registration` → `spot_finding` → `barcode`
is the processing flow. `image` owns metadata; `dataset` owns coordination.
`synthetic` produces processed images/truth; `evaluation` computes pure metrics;
`benchmark` runs explicit cases, measures resources and writes reports.
Algorithms do not depend on benchmark orchestration. Registration estimates a
correction transform, then applies it directly. Barcode extraction, decoding
and filtering are independent stages that retain spot identity.

Use the [architecture](docs/architecture.md), [contracts](docs/api/contracts.md),
[conventions](docs/conventions.md), and [migration guide](docs/migration.md) as
maintained detail. [AGENTS.md](AGENTS.md) owns operational instructions;
[Linear](https://linear.app/jiahaoh/document/thesis-and-implementation-workflow-c12f30bffe9f)
owns current plans, decisions and evidence. This file is orientation, not a run log.
