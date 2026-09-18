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

:::{grid-item-card} {fas}`rocket` Getting started
:link: getting-started
:link-type: doc
:class-card: sf-card
Generate tiny synthetic images and process them to molecule-level CSVs on a
local CPU. A tested quickstart with no microscopy data or MATLAB license.
:::

:::{grid-item-card} {fas}`diagram-project` Workflows
:link: workflows
:link-type: doc
:class-card: sf-card
Snakemake rules and configuration mapped to source, with bounded DAG examples
and explicit downstream and cluster limitations.
:::

:::{grid-item-card} {fab}`python` Python API
:link: api/python
:link-type: doc
:class-card: sf-card
Public reference with an export inventory, array and coordinate contracts,
backend behavior, and small executable examples.
:::

:::{grid-item-card} {fas}`square-root-variable` MATLAB API
:link: api/matlab
:link-type: doc
:class-card: sf-card
Reference parsed from project-owned source, linking workflow stages to Python
operations. It does not certify MATLAB runtime behavior.
:::

:::{grid-item-card} {fas}`flask` Recipes
:link: recipes
:link-type: doc
:class-card: sf-card
Development recipes that exercise small Python inputs with fixed seeds and
documented resource bounds.
:::

:::{grid-item-card} {fas}`ruler-combined` Conventions
:link: conventions
:link-type: doc
:class-card: sf-card
Axis order, channel order, coordinate bases, registration sign, and threshold
modes shared by both backends.
:::

:::{grid-item-card} {fas}`life-ring` Troubleshooting
:link: troubleshooting
:link-type: doc
:class-card: sf-card
Backend requirements, environment problems, and workflow diagnostics.
:::

:::{grid-item-card} {fas}`code-pull-request` Contributing
:link: contributing
:link-type: doc
:class-card: sf-card
Build and preview this site locally, run the documentation checks, and publish
a verified release.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2

Getting started <getting-started>
Workflows <workflows>
Benchmark <benchmark>
Python API <api/python>
MATLAB API <api/matlab>
Recipes <recipes>
Conventions <conventions>
Troubleshooting <troubleshooting>
Contributing <contributing>
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
