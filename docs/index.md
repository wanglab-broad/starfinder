<img class="sf-hero__logo dark-light" src="_static/logo.png" alt="">

# STARfinder

<p class="sf-lead">Image processing for STARmap-related spatial transcriptomics.
STARfinder takes raw microscopy volumes to molecule-level tables with Python and
MATLAB backends coordinated by Snakemake.</p>

<div class="sf-actions">

```{button-ref} getting-started
:color: primary
:class: sf-btn
Get started
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

<div class="sf-pipeline" aria-label="Pipeline stages">
<span class="sf-pipeline__label">Pipeline</span>
<span>load</span><span>rotate</span><span>enhance</span><span>registration</span><span>spot finding</span><span>extraction</span><span>filtration</span>
</div>

## Explore the documentation

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fas}`rocket` Get started
:link: getting-started
:link-type: doc
:class-card: sf-card
Generate tiny synthetic images and process them to molecule-level CSVs on a local CPU.
:::

:::{grid-item-card} {fas}`diagram-project` Workflow
:link: workflows
:link-type: doc
:class-card: sf-card
Configure Snakemake, coordinate Dataset/FOV processing, and understand downstream boundaries.
:::

:::{grid-item-card} {fas}`flask` Benchmark
:link: benchmark
:link-type: doc
:class-card: sf-card
Generate inputs, run explicit cases, evaluate saved outputs and report results with provenance.
:::

:::{grid-item-card} {fas}`code` API
:link: api/index
:link-type: doc
:class-card: sf-card
Alphabetical Python and MATLAB references, supported exports and source links.
:::

:::{grid-item-card} {fas}`ruler-combined` Wiki / Convention
:link: wiki
:link-type: doc
:class-card: sf-card
Architecture, spatial contracts, recipes, migration and contributor guidance.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

Get started <getting-started>
Workflow <workflows>
Benchmark <benchmark>
API <api/index>
Wiki / Convention <wiki>
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
