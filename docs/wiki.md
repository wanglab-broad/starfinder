# Wiki

:::{container} sf-lead
Guides to how STARfinder is built: its vocabulary, package responsibilities and
detailed contracts.
:::

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Starfinder Design
:link: wiki/design
:link-type: doc
:class-card: sf-card
The architecture, the shared contracts and the guide of each pipeline stage.
:::

:::{grid-item-card} Dataset
:link: wiki/dataset
:link-type: doc
:class-card: sf-card
The synthetic scenes and the real data.
:::

:::{grid-item-card} Method registry
:link: method-registry
:link-type: doc
:class-card: sf-card
How each stage selects its methods, and how to add one.
:::

:::{grid-item-card} Development
:link: wiki/development
:link-type: doc
:class-card: sf-card
Contributing, recipes, troubleshooting and migration.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

Starfinder Design <wiki/design>
Dataset <wiki/dataset>
Method registry <method-registry>
Development <wiki/development>
```
