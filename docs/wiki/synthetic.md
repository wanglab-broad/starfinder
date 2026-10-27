# Synthetic

:::{container} sf-lead
The synthetic generator of `starfinder.synthetic` renders processed-image scenes and
their truth records. Each scene has its design page, and the cellular scenes also a
validation design. The synthetic guide, the amplicon scene's page, has the sample scene,
the cellular scene and its validation design beneath it; the real-appearance guide has
its validation design beneath it. The presets in use are listed in the
[dataset catalog](../datasets.md#current-synthetic-presets), and the functions in the
[synthetic API](../api/synthetic.rst).
:::

:::{container} sf-stage-table

| Scene | Renders | Design | Validation |
| --- | --- | --- | --- |
| Amplicon | Formed amplicons read out round by round | [Amplicon scene](../synthetic-specification.md) | |
| Sample | A multi-FOV sample with a minimal cellular scene | [Sample scene](../synthetic-sample-scene.md) | |
| Cellular | Nuclei, cells, stains and the morphology round | [Cellular scene](../synthetic-cellular-scene.md) | [validation](../synthetic-cellular-scene-validation.md) |
| Real appearance | The cellular scene with appearance sampled from real images | [Real-appearance scene](../synthetic-real-appearance.md) | [validation](../synthetic-real-appearance-validation.md) |

:::

```{toctree}
:hidden:
:maxdepth: 2
:titlesonly:
:includehidden:

Synthetic guide <../synthetic-specification>
Synthetic real-appearance guide <../synthetic-real-appearance>
```
