# Synthetic

The synthetic generator of `starfinder.synthetic` renders processed-image scenes and
their truth records: the amplicon scene of formed amplicons, the multi-FOV sample scene,
and the cellular scenes, with or without appearance sampled from real images. Each scene
has its design page, and the cellular scenes also a validation design. The presets in
use are listed in the
[dataset catalog](../datasets.md#current-synthetic-presets), and the functions in the
[synthetic API](../api/synthetic.rst).

```{toctree}
:maxdepth: 1

Amplicon scene <../synthetic-specification>
Sample scene <../synthetic-sample-scene>
Cellular scene <../synthetic-cellular-scene>
Cellular scene validation <../synthetic-cellular-scene-validation>
Real-appearance scene <../synthetic-real-appearance>
Real-appearance scene validation <../synthetic-real-appearance-validation>
```
