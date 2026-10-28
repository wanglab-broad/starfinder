# Get Started

:::{container} sf-lead
Install the Python package, then run two small demos on a local CPU. The demos are
executed notebooks, shown here with their outputs.
:::

::::{grid} 1 1 3 3
:gutter: 3
:class-container: sf-steps

:::{grid-item-card} Installation
:link: installation
:link-type: doc
:class-card: sf-card
Set up the Python package and its optional extras.
:::

:::{grid-item-card} Single FOV demo
:link: single-fov-demo
:link-type: doc
:class-card: sf-card
One synthetic field of view to a table of molecules.
:::

:::{grid-item-card} Multi-FOV / sample demo
:link: multi-fov-demo
:link-type: doc
:class-card: sf-card
A small synthetic sample to cells, exports and a report.
:::

::::

Both demos use classical methods only and need no microscopy data, MATLAB, Snakemake,
GPU, model or download; the sample demo needs the `multiview-stitcher` and `anndata`
extras. The same single-FOV route as a script is the [quickstart](../recipes.md#quick-start).

```{toctree}
:hidden:
:maxdepth: 1

Installation <installation>
Single FOV demo <single-fov-demo>
Multi-FOV / sample demo <multi-fov-demo>
```
