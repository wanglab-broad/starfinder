<img class="sf-hero__logo dark-light" src="_static/logo.png" alt="">

# STARfinder

<p class="sf-lead">Image processing for STARmap-related spatial transcriptomics.
STARfinder takes raw microscopy volumes to per-FOV molecule tables, cells and counts,
then stitches, assembles and exports each sample and writes its processing report.
Snakemake coordinates the Python and MATLAB backends.</p>

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

<div class="sf-pipeline" aria-label="Pipeline stages">
<span class="sf-pipeline__label">Pipeline</span>
<span>preprocessing</span><span>registration</span><span>spot finding</span><span>readout</span><span>segmentation</span><span>assignment</span><span>stitching</span><span>assembly</span><span>export</span><span>report</span>
</div>

## Explore the documentation

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fas}`rocket` Get Started
:link: get-started/index
:link-type: doc
:class-card: sf-card
Install the package, then process tiny synthetic images to molecule-level CSVs on a local CPU.
:::

:::{grid-item-card} {fas}`diagram-project` Workflow
:link: workflow/index
:link-type: doc
:class-card: sf-card
Run the Snakemake workflow: its relationship to the Python API, the config scheme and the Broad UGER profile.
:::

:::{grid-item-card} {fas}`code` API
:link: api/index
:link-type: doc
:class-card: sf-card
Alphabetical Python and MATLAB references, supported exports, source links and small examples.
:::

:::{grid-item-card} {fas}`ruler-combined` Wiki
:link: wiki
:link-type: doc
:class-card: sf-card
Starfinder design and the stage guides, the synthetic and real datasets, the method registry and development.
:::

:::{grid-item-card} {fas}`flask` Benchmark
:link: benchmark/index
:link-type: doc
:class-card: sf-card
Benchmark data, the lifecycle with its recipes and research scripts, and where results are kept.
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
