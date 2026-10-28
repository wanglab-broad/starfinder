# Sample-segmentation reference

Settings and outputs for {doc}`sample segmentation <sample-segmentation-contract>`.
The {doc}`export API <api/export>` documents the functions and configuration classes.

## Settings

### Sample segmentation

{py:class}`~starfinder.export.SampleSegmentationConfig` holds the runs and the
assignment settings.

```{config-options} starfinder.export.SampleSegmentationConfig
```

### A named run

{py:class}`~starfinder.export.SampleSegmentationRun` describes a segmentation or a
label import.

```{config-options} starfinder.export.SampleSegmentationRun
```

### Export

{py:class}`~starfinder.export.ExportConfig` selects the route and the parts to run.
The {doc}`sample-output reference <sample-output-reference>` lists fusion, label and
SpatialData settings. The {doc}`assignment guide <assignment-contract>` explains
molecule assignment.

```{config-options} starfinder.export.ExportConfig
```

## Files

These files live in the sample's export folder; the
{doc}`sample-output reference <sample-output-reference>` lists the other export products.

| File | Contents |
| --- | --- |
| `segmentation_<run>.ome.tif` | The named run's labels on the segmentation grid |
| `territories.ome.tif` | The cell territories written when assignment expands labels |
| `tables/cells.<fmt>` | Sample cells, including `cell_namespace`, `cell_id` and `obs_name` |
| `tables/molecules.<fmt>` | Molecules with source spot identity, assigned cell key and sample coordinates |
| `tables/counts.<fmt>` | Cell-by-gene counts in long form |
| `tables/fovs.<fmt>` | FOV summaries |
| `sample.h5ad` | The sample's expression matrix and annotations |
| `cell_labels.ome.tif` | Optional inspection label image |
| `export.json` | The export record, including its `segmentation` section |

`<fmt>` is the export's table format. For the table columns, see the
{doc}`assembly reference <assembly-algorithms>`.

## Segmentation record

The `segmentation` section of `export.json` describes the saved segmentation.

| Field | Contents |
| --- | --- |
| `status`, `reason` | Outcome and its explanation |
| `route` | Export route |
| `inputs` | Grid, fused input files and placement hashes |
| `runs` | Named run records, method provenance, pixel lengths given and used, label paths and hashes |
| `cells`, `nuclei` | Runs used for assignment |
| `assignment` | Assignment configuration |
| `device` | Segmentation device |
| `outputs` | Written files |
| `software`, `wall_s`, `cpu_s`, `peak_rss_kib` | Software versions and resource use |

## Route comparison

{py:func}`~starfinder.export.compare_routes` returns a
{py:class}`~starfinder.export.RouteComparison`. Its `output` folder receives these files.

| File | Result field | Contents |
| --- | --- | --- |
| `summary.json` | `summary`, `record` | Population and matching summaries, configuration and input provenance |
| `pairs.csv` | `pairs` | Overlapping cell pairs and matching results |
| `counts.csv` | `counts` | Counts and differences for matched cells |
| `molecules.csv` | `molecules` | Molecule assignments and comparison categories |

The [specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/sample-segmentation.md#the-consistency-check-of-the-two-routes)
defines the comparison measures.
