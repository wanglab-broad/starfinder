# Dataset catalog

Look up input names and historical version identities here. Current synthetic
presets are listed in the {doc}`synthetic reference <synthetic-reference>`.
The catalog does not establish calibration, accuracy, or biological truth.

## Identity, versions and verification

The historical entries below are names recovered from earlier input records,
not a check of current availability. Their local files, archive lineage and
scientific suitability remain unverified here. Sampled header measurements and
past availability observations are preserved in the
[historical inventory](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/datasets.md#identity-versions-and-verification).

| To identify a stored version | Record |
| --- | --- |
| Source | Accession/version or parent input; generator revision for synthetic data |
| Configuration | Effective settings, seed and codebook identity |
| Images | FOV, round and channel membership; axes, shape and dtype |
| Geometry | Spacing, units and frame, with unknown values left unknown |
| Verification | What was checked, when, and file checksums |
| Use | Intended use and known limitations |

## Real sequencing inputs

These names identify historical STARmap-related sequencing inputs. The precise
assay, upstream processing and association of local files with an archive remain
unverified. See the
[historical real-input inventory](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/datasets.md#real-sequencing-inputs)
for the recorded shapes, codebook observations and unresolved provenance.

| Input name | Code and recipe spelling | Historical input description |
| --- | --- | --- |
| `tissue-2D` | `tissue_2D` in research scripts | Tissue sequencing stacks; the name does not establish a singleton Z axis |
| `cell-culture-3D` | `cell_culture_3D` in research scripts | Cell-culture sequencing stacks |
| `LN` | `LN` | Lymph-node sequencing stacks |
| `aging` | `aging` | Sequencing stacks with a segmented-codebook history |

The {doc}`array and coordinate reference <api/contracts>` describes Python's
ZYX/ZYXC conventions and explicit channel ordering.

## Current synthetic presets

Use the {doc}`synthetic overview <wiki/synthetic>` to choose a scene and the
{doc}`synthetic reference <synthetic-reference>` to look up presets, version
constants and saved files. Synthetic images are generated on demand; a current
preset name does not identify the bytes of a historical dataset.

## Historical synthetic sequencing inputs

These version keys refer to the removed Gaussian processed-image generator.
They do not identify the current scene generator's outputs. Stored bytes and
full effective configurations remain unverified here.

| Historical version key | Description |
| --- | --- |
| `sequencing/tiny` | Earlier tiny sequencing input |
| `sequencing/small/truth-v2.0` | Earlier small regression fixture |
| `sequencing/medium/truth-v1.0` | Earlier medium sequencing input |
| `sequencing/large/truth-v1.0` | Earlier large sequencing input |
| `sequencing/tissue/historical` | Earlier tissue-sized sequencing input |
| `sequencing/thick_medium/truth-v1.0` | Earlier thick-volume sequencing input |

The [historical version inventory](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/datasets.md#historical-synthetic-sequencing-inputs)
keeps reported dimensions, seeds, count disagreements and generator limitations.

## Historical registration derivatives

The historical inventory also describes real round1–round2 pairs from tissue,
cell culture and LN, and synthetic registration versions named `tiny`, `small`,
`medium`, `large`, `tissue` and `thick_medium`. Their extraction lineage and
complete file sets remain unverified here.

| Input family | Identity details |
| --- | --- |
| Real registration pairs | [Recorded parent inputs and pair names](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/datasets.md#historical-registration-derivatives) |
| Synthetic registration pairs | [Historical versions and recorded transforms](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/datasets.md#historical-registration-derivatives) |
