# Workflow configuration reference

(configuration-reference)=

Look up workflow YAML keys here. Tables are generated from
`workflow/schemas/config.schema.yaml` in this checkout. The
{doc}`Snakemake guide <workflows>` prepares a runnable dry-run configuration;
the {doc}`downstream guide <workflow-downstream>` covers cell-level outputs.

(channels)=
## Top-level fields

Types, enumerated values and descriptions come from the schema. A dash in
“Schema default” means the schema declares no default; the consuming API or rule
may supply one. Named types refer to the generated definitions below. These
tables describe fields, not conditional validation or backend compatibility.

```{workflow-config}
```

## FOV membership

The [input guide](workflows.md#paths-and-inputs) introduces the membership
files. The
[membership specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/workflow-configuration.md#fov-membership)
gives their complete layout.

## Reference file and image layout

The generated top-level table lists the path keys. See the
[image-layout specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/workflow-configuration.md#reference-file-and-image-layout)
for path interpretation.

## MATLAB launcher

The top-level table lists the launcher settings. See the
[launcher specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/workflow-configuration.md#matlab-launcher)
for invocation details.

## Rule resources and parameters

The generated definitions list the rule configurations, resource fields and
processing parameters. Dotted names are fields of inline objects; `[]` denotes
an array item. A referenced object has its own definition table.

(rule-parameter-defaults)=
(python-backend-keys-and-wrapper-defaults)=
(python-subtile-parameter-sources)=
(run-records-of-the-processing-rules)=
## Processing settings

The stage references explain the processing methods:
{doc}`preprocessing <preprocessing-algorithms>`,
{doc}`registration <registration-algorithms>`,
{doc}`spot finding <spot-finding-algorithms>` and
{doc}`readout <readout-algorithms>`.

(codebook-segment-layout)=
(codebook-encoding-table)=
(readout-mode)=
(explicit-preprocessing-recipe)=
(explicit-registration-recipe)=
(spot-finding-method-key)=
For workflow-specific translation and saved records, use the
[configuration specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/workflow-configuration.md#python-backend-keys-and-wrapper-defaults).

## Segmentation and assignment

See the {doc}`downstream guide <workflow-downstream>` and the
{doc}`segmentation <segmentation-algorithms>` and
{doc}`assignment <assignment-algorithms>` references. The generated definitions
include the top-level blocks and the existing rule parameters.

(stitching-assembly-and-export)=
## Stitching, assembly and export

The {doc}`stitching <stitching-algorithms>`, {doc}`assembly <assembly-algorithms>`
and {doc}`sample-output <sample-output-reference>` references describe these
sample operations. The
[route examples](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/workflow-configuration.md#stitching-assembly-and-export)
show configurations for per-FOV assignment and segmentation after fusion.

## Processing report

The {doc}`report guide <report-contract>` introduces the sample report and dataset
comparison page; its {doc}`reference <report-algorithms>` lists saved tables.
The generated `report_block` definition lists its YAML fields.

(downstream-parameter-blocks)=
## Schema definitions

These tables include the reusable types referenced above, with their schema
field descriptions and defaults.

```{workflow-config} #/$defs
```

## Overrides and validation boundaries

See the
[override specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/workflow-configuration.md#overrides-and-validation-boundaries)
before combining a file with command-line overrides.

## Full configuration example

Download the {download}`full template <examples/workflow-full.yaml>` or the
{download}`explicit preprocessing example <examples/workflow-recipe-2.yaml>`.
The full template includes disabled downstream jobs; model and acquisition paths
need editing for your data.

[Configuration specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/workflow-configuration.md).
