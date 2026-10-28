# Pipeline stages

:::{container} sf-lead
The stages below follow processing order, with a guide for running each stage and a
reference for looking up its methods, options and outputs; detailed specifications
and validation designs live in [design/](https://github.com/wanglab-broad/starfinder/blob/dev/design/README.md).
:::

## Per-FOV stages

:::{container} sf-stage-table

| Stage | Guide | Reference |
| --- | --- | --- |
| Preprocessing | [Prepare images](../preprocessing-contract.md) | [Methods and options](../preprocessing-algorithms.md) |
| Registration | [Align rounds](../registration-contract.md) | [Methods and transforms](../registration-algorithms.md) |
| Spot finding | [Detect spots](../spot-finding-contract.md) | [Methods and results](../spot-finding-algorithms.md) |
| Readout | [Extract and decode](../readout-contract.md) | [Methods and records](../readout-algorithms.md) |
| Segmentation | [Segment images](../segmentation-contract.md) | [Methods and labels](../segmentation-algorithms.md) |
| Assignment | [Assign molecules to cells](../assignment-contract.md) | [Methods and counts](../assignment-algorithms.md) |

:::

## Sample stages

:::{container} sf-stage-table

| Stage | Guide | Reference |
| --- | --- | --- |
| Stitching | [Place FOVs](../stitching-contract.md) | [Methods and placement](../stitching-algorithms.md) |
| Assembly | [Assemble sample tables](../assembly-contract.md) | [Options and tables](../assembly-algorithms.md) |
| Sample output | [Export the sample](../sample-output-contract.md) | [Options and files](../sample-output-reference.md) |
| Sample segmentation | [Segment the fused image](../sample-segmentation-contract.md) | [Routes and records](../sample-segmentation-reference.md) |
| Processing report | [Inspect saved processing](../report-contract.md) | [Sections and records](../report-algorithms.md) |

:::

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
