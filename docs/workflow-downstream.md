# Segmentation and assignment in the workflow

(downstream-segmentation-and-assignment)=

Sequencing produces molecule positions and gene calls. Segmentation supplies cell
labels, and assignment counts the molecules in those cells. Use this guide after
the {doc}`Snakemake guide <workflows>` to prepare cell-level outputs.

## Quick start

Start with the {download}`full workflow template <examples/workflow-full.yaml>`
and edit the paths, morphology inputs and model settings for your experiment.
Enable `nuclei_registration` or `rotate_nuclei` to prepare the nuclear stain,
`create_nuclei_amplicon_overlay` for a composite image,
`stardist_segmentation` for labels and `reads_assignment` for counts.
Dry-run `all` with the command in the
[Snakemake guide](workflows.md#small-reproducible-dry-run) before executing it.

For the Python sample workflow, configure `stitching`, `assembly` and `export`.
The [configuration reference](workflow-configuration.md#stitching-assembly-and-export)
links to the settings and route examples.

## What it does

(morphology-and-labels)=
### Morphology and labels

`nuclei_registration` aligns additional rounds to the reference stain;
`rotate_nuclei` prepares the nuclear image. `create_nuclei_amplicon_overlay`
combines that image with the sequencing reference image.
`stardist_segmentation` calls the package's StarDist method on its configured
image. Install the package's `stardist` extra in the environment running the rule
and provide the model locally.

The Python `segmentation` block describes named per-FOV runs for the API; it is
not connected to a workflow rule. Use the existing `stardist_segmentation` rule
for this workflow's per-FOV label preparation. The
{doc}`segmentation guide <segmentation-contract>` shows the API route.

(tile-geometry-and-sample-aggregation)=
### Assignment and sample outputs

`reads_assignment` reads molecules, labels and the codebook and calls the
package's assignment functions. `stitching` supplies the sample placement;
`assembly` and `export` configure the Python sample output. The MATLAB sample
workflow uses its annotation, MAF and BigStitcher rules.

(two-routes-to-a-samples-cells)=
### Two routes to a sample's cells

Choose `export.route: per_fov` to assemble assignments made within FOVs. Choose
`after_fusion` to segment a fused sample projection, especially when cells cross
FOV seams. The {doc}`sample segmentation guide <sample-segmentation-contract>`
explains the choice; the
[route specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/sample-segmentation.md)
gives the supported settings and complete examples are preserved in the
[workflow configuration record](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/workflow-configuration.md#stitching-assembly-and-export).

## What you get

| Job | Main output under OUTPUT |
| --- | --- |
| `rotate_nuclei` | `images/DAPI/<fov>.tif` |
| `create_nuclei_amplicon_overlay` | `images/overlay/<fov>.tif` |
| `stardist_segmentation` | `images/stardist_segmentation/<fov>.tif` and its JSON record |
| `reads_assignment` with a named `assembly.assignment` | `fovs/<fov>/assignment/<name>/` |
| Legacy `reads_assignment` output | `expr/<fov>/raw.h5ad` and `reads_assignment.csv` |
| Python `export_sample` | `export/<sample>/export.json` and the files it lists |

See the {doc}`assignment reference <assignment-algorithms>` and
{doc}`sample-output reference <sample-output-reference>` for saved tables and
records. `assembly.previews` requests assignment figures on the saved-assignment
route.

## Main options

* `reference_stains` and `additional_round`: identify morphology images for the
  Python backend.
* `rules.stardist_segmentation.parameters`: choose the input image and model.
* `rules.reads_assignment.parameters`: configure per-FOV assignment.
* `assignment`: configure assignment after fusion.
* `assembly.assignment`: choose the saved assignment's name.
* `export.segmentation`: configure segmentation of the fused projection.

## Limits

A count matrix does not establish segmentation quality or biological identity.
The full template's disabled downstream sections are examples, not an executed
image-to-cell analysis. Use the
[downstream specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/workflow-downstream.md)
for legacy file layouts and coordinate conversions.

## How it is checked

Inspect a dry run for morphology, label and assignment inputs. Before a full run,
inspect the labels and assignment figures on a small sample with your own model
and acquisition metadata.

[Downstream workflow specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/workflow-downstream.md).
