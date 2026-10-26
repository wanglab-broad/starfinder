<img class="sf-hero__logo dark-light" src="_static/logo.png" alt="">

# STARfinder

<p class="sf-lead">STARfinder turns raw microscopy images from STARmap and related
spatial transcriptomics methods into molecules, cells and gene counts. It processes each
field of view, stitches the fields of view into a sample, and exports each sample with a
processing report. Use it from Python or MATLAB, or run whole datasets with a scalable
workflow.</p>

<div class="sf-actions">

```{button-ref} get-started/index
:color: primary
:class: sf-btn
Get Started
```

```{button-ref} api/python
:color: secondary
:outline:
:class: sf-btn
Python API
```

```{button-ref} api/matlab
:color: secondary
:outline:
:class: sf-btn
MATLAB API
```

</div>

## Explore the documentation

::::{grid} 1 1 2 2
:gutter: 3
:class-container: sf-sections

:::{grid-item-card} {fas}`rocket` [Get Started](get-started/index.md)
:columns: 12
:class-card: sf-card sf-card-wide
Install the package and run two small demos on a local CPU.
+++
[Installation](get-started/installation.md)
[Single FOV demo](get-started/single-fov-demo.md)
[Multi-FOV / sample demo](get-started/multi-fov-demo.md)
:::

:::{grid-item-card} {fas}`diagram-project` [Workflow](workflow/index.md)
:class-card: sf-card
Run every FOV and sample of a dataset with a scalable workflow framework.
+++
[Snakemake](workflow/snakemake.md)
[Nextflow](workflow/nextflow.md)
:::

:::{grid-item-card} {fas}`code` [API](api/index.md)
:class-card: sf-card
Python and MATLAB API, and script examples.
+++
[Python](api/python.rst)
[MATLAB](api/matlab.md)
[Examples](api/examples.md)
:::

:::{grid-item-card} {fas}`book` [Wiki](wiki.md)
:class-card: sf-card
How STARfinder is built, its pipeline stages, data and development.
+++
[Starfinder Design](wiki/design.md)
[Dataset](wiki/dataset.md)
[Method registry](method-registry.md)
[Development](wiki/development.md)
:::

:::{grid-item-card} {fas}`flask` [Benchmark](benchmark/index.md)
:class-card: sf-card
Benchmark data, recipes and results.
+++
[Data](benchmark/data.md)
[Recipe / Experiments](benchmark/experiments.md)
[Results](benchmark/results.md)
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

Get Started <get-started/index>
Workflow <workflow/index>
API <api/index>
Wiki <wiki>
Benchmark <benchmark/index>
```

## Existing entry points

<div class="sf-entry-points">

The repository's [project README](https://github.com/wanglab-broad/starfinder/blob/dev/README.md),
[Python README](https://github.com/wanglab-broad/starfinder/blob/dev/src/python/README.md),
[workflow examples](https://github.com/wanglab-broad/starfinder/blob/dev/example/README.md),
and [downstream examples](https://github.com/wanglab-broad/starfinder/blob/dev/example/downstream/README.md)
remain available. These links follow the development branch; use your checkout's
copies when working with a different revision.

</div>
