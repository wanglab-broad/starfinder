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
| [Preprocessing](../preprocessing-contract.md) | The ordered recipe of preprocessing steps | [reference](../preprocessing-algorithms.md) | | |
| [Registration](../registration-contract.md) | Aligning each round with the reference round | [reference](../registration-algorithms.md) | | |
| [Spot finding](../spot-finding-contract.md) | Detection and stable spot identities | [reference](../spot-finding-algorithms.md) | | |
| [Readout](../readout-contract.md) | Extraction, decoding and read QC | [algorithms](../readout-algorithms.md) | in algorithms | [baseline](../readout-baseline.md) |
| [Segmentation](../segmentation-contract.md) | Label images, methods and imported masks | [algorithms](../segmentation-algorithms.md) | | [baseline](../segmentation-baseline.md) |
| [Assignment](../assignment-contract.md) | Molecules to cells, compartments and counts | [algorithms](../assignment-algorithms.md) | in algorithms | [baseline](../assignment-baseline.md) |

:::

## Sample stages

:::{container} sf-stage-table

| Stage | Covers | Algorithms | Validation | Baseline |
| --- | --- | --- | --- | --- |
| [Stitching](../stitching-contract.md) | Tile positions and the placement of each FOV | [algorithms](../stitching-algorithms.md) | | [baseline](../stitching-baseline.md) |
| [Assembly](../assembly-contract.md) | Sample coordinates, ownership and the sample tables | [algorithms](../assembly-algorithms.md) | | [baseline](../assembly-baseline.md) |
| [Sample output](../sample-output-contract.md) | The export: fused image, labels, SpatialData and profiles | | [validation](../stitching-assembly-validation.md) | [baseline](../sample-output-baseline.md) |
| [Sample segmentation](../sample-segmentation-contract.md) | Segmentation after fusion and the check of the two routes | | [validation](../sample-segmentation-validation.md) | |
| [Processing report](../report-contract.md) | One sample's report and a dataset's comparison page | [algorithms](../report-algorithms.md) | in algorithms | [baseline](../report-baseline.md) |

:::

## The pages beneath a guide

* **Algorithms:** the specification of each method. For readout, assignment and the
  processing report, the validation design is part of this page.
  Preprocessing, registration and spot finding have a reference instead: their methods,
  options and records, with the specification in `design/`.
* **Validation:** a separate validation design, where the stage has one. The design that
  checks placement, assembly and the sample outputs sits under the sample-output guide,
  the last stage it checks.
* **Baseline:** how the existing code behaved before the stage's contract, pinned to the
  revision it describes. A baseline is history, not a description of the current code.

```{toctree}
:hidden:
:maxdepth: 2
:titlesonly:
:includehidden:

Preprocessing <../preprocessing-contract>
Registration <../registration-contract>
Spot finding <../spot-finding-contract>
Readout <../readout-contract>
Segmentation <../segmentation-contract>
Assignment <../assignment-contract>
Stitching <../stitching-contract>
Assembly <../assembly-contract>
Sample output <../sample-output-contract>
Sample segmentation <../sample-segmentation-contract>
Processing report <../report-contract>
```
