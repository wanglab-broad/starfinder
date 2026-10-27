# Synthetic

:::{container} sf-lead
`starfinder.synthetic` generates processed images together with their truth: where
every molecule, nucleus and cell is, and where every FOV really sits. Use it to test
each stage of the pipeline, or the whole chain, against known answers. The four scenes
build on each other; pick the smallest one that has what your test needs.
:::

:::{container} sf-stage-table

| Scene | What it renders | Use it to test |
| --- | --- | --- |
| [Amplicon](../synthetic-specification.md) | One FOV of amplicons read out round by round | Registration, spot finding, decoding |
| [Sample](../synthetic-sample-scene.md) | Overlapping FOVs of one sample, with simple cells and a nuclear stain | Stitching, assignment, assembly |
| [Cellular](../synthetic-cellular-scene.md) | Lobed, touching nuclei, cell types, stains and a morphology round, for dense tissue, 2D tissue and culture | Segmentation and the chain after it |
| [Real appearance](../synthetic-real-appearance.md) | The cellular scene with nuclei and cells taken from real images | Segmentation on realistic images |

:::

The {doc}`synthetic reference <../synthetic-reference>` lists every preset and every
file the scenes write; the {doc}`synthetic API <../api/synthetic>` documents the
functions.

```{toctree}
:hidden:
:maxdepth: 2
:titlesonly:
:includehidden:

Amplicon scene <../synthetic-specification>
Sample scene <../synthetic-sample-scene>
Cellular scene <../synthetic-cellular-scene>
Real-appearance scene <../synthetic-real-appearance>
Synthetic reference <../synthetic-reference>
```
