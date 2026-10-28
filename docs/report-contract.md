# Processing report

The processing report describes the saved processing of one sample: its run, FOVs,
readout, cells and sample measurements. Use it to inspect a run and find FOVs worth
looking at. It gives no pass or fail verdict and does not filter or correct results.

`Dataset.report` writes the sample page. `Dataset.compare_samples` puts finished
sample reports beside one another. The {doc}`reference <report-algorithms>` lists
the sections, tables, columns and figures; the {doc}`API <api/report>` documents the
configuration and returned records.

## Quick start

After processing a sample, use its `Dataset`, sample name and FOV list:

```python
from starfinder.report import ReportConfig, load_report

report = dataset.report(sample, fov_ids, config=ReportConfig())
record = load_report(report.path)
comparison = dataset.compare_samples([sample])
```

The {doc}`multi-FOV demo <get-started/multi-fov-demo>` builds a sample, exports it
and makes a report. It provides the processing steps before this example.

In a Python workflow configuration, add:

```yaml
backend: python
report: {}
```

This is a configuration excerpt: add it to a workflow with its dataset inputs
already configured. `report_sample` writes the sample reports and `report_dataset`
writes the comparison page. The block uses the `ReportConfig` field names;
`markers` holds mappings with `name` and `genes`, `marker_pairs` holds gene pairs,
and `limits` holds the `ReportLimits` fields. See
{doc}`workflow configuration <workflow-configuration>` for the complete workflow.

## What it does

1. **Read saved outputs.** It collects per-FOV run records, placement, assignment,
   segmentation and export records from the dataset's output root.
2. **Summarize the run.** It collects counts and timings, computes report
   measurements, and lists problematic FOVs with the reason for each entry.
3. **Show examples.** It chooses example FOVs for the image panels. The main
   example replays recorded preprocessing and applies saved registration transforms
   to show the images; it does not estimate registration again.
4. **Write the report.** It writes the HTML page, CSV tables and a JSON record.
   The comparison entry reads the finished sample reports to make its page.

### Inputs and unavailable parts

The usual inputs are under `fovs/<FOV>/`, `stitching/<sample>/` and
`export/<sample>/`. The main example also reads its raw input images. Keep the
dataset's input root available for replay. `run.json` records the pipeline,
input files, preprocessing, registration, transforms, read counts, codebook or
direct panel, and workflow configuration provenance.

A report can describe an unfinished sample. Within a report, a part can be
unavailable because its input is missing or changed. The page shows “Unavailable”
and a reason; `tables/unavailable.csv` records the details.

The report reads exports from the route per FOV and the route after fusion.
The export records which route produced its cells and molecules. For input
locations and route-specific interpretation, see the
[input and route rules](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/report.md#inputs).

### Example FOVs and replay

The automatic choice uses read counts for a representative main example.
`example_fovs` lets you choose the examples instead. `report.json` records their
selection and the replay outcome.
If a replay cannot be reproduced, its unavailable reason helps identify the
missing input or saved state. The
[example and replay rules](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/report.md#the-example-fovs)
give the detailed selection and replay procedure.

## What you get

`Dataset.report` returns a {py:class}`~starfinder.report.ReportRecord` with `path`
and `record`. Its default folder is `<output_root>/report/<sample>/`:

| Output | Contents |
| --- | --- |
| `report.html` | The sample page, with embedded figures and data, viewable offline |
| `report.json` | Configuration, input and output hashes, summary, unavailable reasons, examples and replay |
| `tables/` | CSV tables listed in the reference |

The comparison returns a {py:class}`~starfinder.report.ComparisonRecord` and writes
`report/index.html` and `report/samples.csv`. It shows spots per FOV, matched
fraction and cells per FOV by sample. A sample without a report appears as
“missing”.

## Main options

* `markers`: gene groups for the marker figures.
* `marker_pairs`: gene pairs for mutually exclusive co-expression rate (MECR).
* `positive_threshold`: the count used to call a cell positive for a gene.
* `example_fovs`: choose the image examples.
* `limits`: tune which FOVs the problematic-FOV list draws attention to.
* `figure_width_px` and `jpeg_quality`: adjust whole-sample figure size and quality.

The reference generates the option names and defaults from the configuration
classes. The API gives their full meaning.

## Limits

The report does not recover information absent from saved records, rerun the
processing pipeline, or establish biological accuracy. Its problematic-FOV list
is a set of pointers for inspection, not a quality grade. Replay depends on the
raw inputs and recorded processing state being available.

## How it is checked

Checks compare collected tables with small saved-record fixtures and exercise
missing inputs, replay and report rendering. Chain checks cover the path from
processing through export to the report. The
[validation design](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/report-algorithms.md#validation-design)
describes these checks.

Full [processing-report specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/report.md).

```{toctree}
:hidden:

Processing-report reference <report-algorithms>
```
