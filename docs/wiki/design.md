# Starfinder Design

:::{container} sf-lead
Architecture, shared conventions, coordination and saved processing, with guides
and references for the pipeline stages.
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

:::{grid-item-card} Coordination reference
:link: ../coordination-reference
:link-type: doc
:class-card: sf-card
Configuration fields and result attributes.
:::

:::{grid-item-card} Checkpoints and run records
:link: ../checkpoints
:link-type: doc
:class-card: sf-card
Per-FOV checkpoints and the run record.
:::

:::{grid-item-card} Checkpoint reference
:link: ../checkpoints-reference
:link-type: doc
:class-card: sf-card
Saved files, table columns and run-record fields.
:::

:::{grid-item-card} Dependencies and backends
:link: ../api/backends
:link-type: doc
:class-card: sf-card
Packages used by the public interfaces.
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
A guide and reference for each stage.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

Architecture <../architecture>
Array and coordinate contracts <../api/contracts>
Dataset and FOV coordination <../coordination>
Coordination reference <../coordination-reference>
Checkpoints and run records <../checkpoints>
Checkpoint reference <../checkpoints-reference>
Dependencies and backends <../api/backends>
Learned detectors <../learned-detectors>
Pipeline stages <pipeline-stages>
```
