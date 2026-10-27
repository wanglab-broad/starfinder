# Synthetic

The synthetic generator of `starfinder.synthetic` renders processed-image scenes and
their truth records: the amplicon scene of formed amplicons, the multi-FOV sample scene,
and the cellular scenes, with or without appearance sampled from real images. Each scene
has its design page, and the cellular scenes also a validation design. The synthetic
guide, the amplicon scene's page, has the sample scene, the cellular scene and its
validation design beneath it. The presets in
use are listed in the
[dataset catalog](../datasets.md#current-synthetic-presets), and the functions in the
[synthetic API](../api/synthetic.rst).

```{toctree}
:maxdepth: 2
:titlesonly:
:includehidden:

Synthetic guide <../synthetic-specification>
Real-appearance scene <../synthetic-real-appearance>
Real-appearance scene validation <../synthetic-real-appearance-validation>
```
