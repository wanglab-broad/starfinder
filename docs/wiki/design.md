# Starfinder Design

:::{container} sf-lead
How STARfinder is built, and the guide of each pipeline stage.
:::

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} Architecture
:link: ../architecture
:link-type: doc
:class-card: sf-card
What each namespace does, and where it stops.
:::

:::{grid-item-card} Array and coordinate contracts
:link: ../api/contracts
:link-type: doc
:class-card: sf-card
Shapes, axes, channels and coordinates that every stage shares.
:::

:::{grid-item-card} Dataset and FOV coordination
:link: ../coordination
:link-type: doc
:class-card: sf-card
How a Dataset and its FOVs run the stages.
:::

:::{grid-item-card} Checkpoints and run records
:link: ../checkpoints
:link-type: doc
:class-card: sf-card
Per-FOV checkpoints and the run record.
:::

:::{grid-item-card} Dependencies and backends
:link: ../api/backends
:link-type: doc
:class-card: sf-card
Optional packages, and how a missing one fails.
:::

:::{grid-item-card} Learned detectors
:link: ../learned-detectors
:link-type: doc
:class-card: sf-card
Running the Spotiflow and Piscis spot detectors.
:::

:::{grid-item-card} Pipeline stages
:link: pipeline-stages
:link-type: doc
:columns: 12
:class-card: sf-card sf-card-wide
The guide of each stage, with its algorithms, validation and baseline.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

Architecture <../architecture>
Array and coordinate contracts <../api/contracts>
Dataset and FOV coordination <../coordination>
Checkpoints and run records <../checkpoints>
Dependencies and backends <../api/backends>
Learned detectors <../learned-detectors>
Pipeline stages <pipeline-stages>
```
