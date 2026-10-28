# Pipeline stages

:::{container} sf-lead
Each stage of the pipeline has a guide: its contract page, which states the stage's
entries, configurations, records, limits and expected failures. Beneath each guide are
the stage's other pages. The stages are listed in processing order.
:::

## Per-FOV stages

:::{container} sf-stage-table

| Stage | Covers | Algorithms | Validation | Baseline |
| --- | --- | --- | --- | --- |
| [Preprocessing](../../docs/preprocessing-contract.md) | The ordered recipe of preprocessing steps | [reference](../../docs/preprocessing-algorithms.md) | | |
| [Registration](../../docs/registration-contract.md) | Aligning each round with the reference round | [reference](../../docs/registration-algorithms.md) | | |
| [Spot finding](../../docs/spot-finding-contract.md) | Detection and stable spot identities | [reference](../../docs/spot-finding-algorithms.md) | | |
| [Readout](../../docs/readout-contract.md) | Extraction, decoding and read QC | [reference](../../docs/readout-algorithms.md) | | |
| [Segmentation](../../docs/segmentation-contract.md) | Label images, methods and imported masks | [reference](../../docs/segmentation-algorithms.md) | | |
| [Assignment](../../docs/assignment-contract.md) | Molecules to cells, compartments and counts | [reference](../../docs/assignment-algorithms.md) | | |

:::

## Sample stages

:::{container} sf-stage-table

| Stage | Covers | Algorithms | Validation | Baseline |
| --- | --- | --- | --- | --- |
| [Stitching](../../docs/stitching-contract.md) | Tile positions and the placement of each FOV | [reference](../../docs/stitching-algorithms.md) | | |
| [Assembly](../../docs/assembly-contract.md) | Sample coordinates, ownership and the sample tables | [reference](../../docs/assembly-algorithms.md) | |  |
| [Sample output](../../docs/sample-output-contract.md) | The export: fused image, labels, SpatialData and profiles | [reference](../../docs/sample-output-reference.md) | | |
| [Sample segmentation](../../docs/sample-segmentation-contract.md) | Segmentation after fusion and the check of the two routes | [reference](../../docs/sample-segmentation-reference.md) | [validation](https://github.com/wanglab-broad/starfinder/blob/dev/design/validation/sample-segmentation.md) | |
| [Processing report](../../docs/report-contract.md) | One sample's report and a dataset's comparison page | [reference](../../docs/report-algorithms.md) | | |

:::

## The pages beneath a guide

* **Algorithms:** the specification of each method. For assignment, the validation design is part of this page. Preprocessing, registration, spot
  finding, readout, stitching and the processing report have a reference instead: their methods, options and records,
  with the specification in `design/`.
* **Validation:** a separate validation design, where the stage has one. The design that
  checks placement, assembly and the sample outputs sits under the sample-output guide,
  the last stage it checks.
* **Baseline:** how the existing code behaved before the stage's contract, pinned to the
  revision it describes. A baseline is history, not a description of the current code.
