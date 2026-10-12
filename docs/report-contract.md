# Processing-report contract: the sample report, the comparison page, their inputs, tables and record, and the records added to `run.json`

Status: Proposed (W-385, 2026-10-12, at dc5fc3c; corrected by W-447, 2026-10-12)

This page specifies the §2.11 processing report so that it can be implemented after Jiahao
approves it: the entry that writes one sample's report, the entry that writes the comparison
page of a dataset, the configuration, where every input is looked for and what the report
holds at each stage of a sample's processing, the closed list of reasons a part can be
unavailable, the output folder with the schema of every table and of the record
`report.json`, the two routes to a sample's cells, the example FOVs, the replay of the
example images, and the fields the earlier stages add to `run.json`. The algorithms, the
layout and the validation design are the algorithms page, `docs/report-algorithms.md` (W-386); where this page needs
one of them it names the open point ("Open points for the algorithms page"). What exists
today, the source of every value and every gap are in {doc}`report-baseline`. Nothing here
is implemented, and no accepted page is changed.

**The report describes a run and judges nothing.** It shows what the records of one
sample's processing hold. Its table "Problematic FOVs" lists each FOV whose value lies
outside a stated limit, with the measured value, the sample median and the limit; that is a
description of where to look, not a verdict. The report gives no pass or fail verdict for a
FOV, a step or a sample, computes no composite score, and filters, corrects, normalizes or
rewrites nothing: it only reads the sample's files and writes its own folder. A value whose
input is missing is shown as unavailable with its reason, never as zero and never as
passing.

It builds on these pages and redefines none of them: {doc}`checkpoints` (`run.json`, the
checkpoint stages), {doc}`registration-contract` (transforms and their convention, the
registration QC), {doc}`readout-contract` (call statuses, the read filter, deduplication),
{doc}`stitching-contract` (the placement record), {doc}`assembly-contract` (the sample
tables, ownership, population accounting), {doc}`sample-output-contract` (the export entry,
`export/<sample>/`, `export.json`, `check_export`), {doc}`sample-segmentation-contract` (the
route setting and every output column on the route after fusion) and
{doc}`workflow-downstream` ("Two routes to a sample's cells").

Evidence, cited by path under the benchmark root
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/`:

* W-384, `W-384/20261012T065105Z-18876aac/`: the source inventory
  (`results/inventory/source_inventory.csv`, looked up in the files of two synthetic runs and
  two W-398 exports), the three known gaps (`worker-notes.md`, section 4), the replay
  (`results/replay/replay.csv`: the largest absolute difference 0 in every round), the
  candidate limits (`results/distributions/candidate_limits.csv`), the figure sizes
  (`results/figures/figure_sizes.csv`) and the sizes of the records of a real FOV
  (`results/tables/real_fovs.csv`).
* The agreed layout mockup `W-152/20261007T1939Z-s211-report-scope/processing-report-layout.html`
  and the W-383 inputs `W-383/20261008T0240Z-inputs/` (`markers.yaml`; the survey that
  defines MECR).

Every statement about an existing record names the file and function that writes it at
`dc5fc3c`; paths are relative to `src/python/starfinder/`.

## What Jiahao decides

The open choices of this page, each with its options further down and one recommendation.
Statements not in this table follow the settled decisions or the accepted pages.

| # | Choice | Recommendation | Section |
| --- | --- | --- | --- |
| 1 | The names of the two entries and of the package module | N1: `Dataset.report` and `Dataset.compare_samples`; module `starfinder.report` | "Names" |
| 2 | What a second report of the same sample does to the first | RP1: the new report replaces the earlier one as a whole once it is complete; a file of the folder that the earlier `report.json` does not list stops the entry before anything is written | "The output folder" |
| 3 | The roles of the example FOVs when the user names them | EX1: the roles `example_1` (main), `example_2`, `example_3` in the given order | "The example FOVs" |
| 4 | The registration example when a round's dense or B-spline field was not saved | RE1: the example is unavailable with reason `field_not_saved` | "The replay" |
| 5 | The form of the global transforms in `run.json` (R1) | T1: a new top-level key `transforms`, one entry per moving round, in the form of `transforms.json` | "Records added to run.json" |
| 6 | The form of the codebook record (R2) | C1: a new top-level key `codebook` with path, SHA-256 and kind | "Records added to run.json" |
| 7 | The form of the correct-form count (R3) | F1: `counts.filtering.endpoint_valid` and, for a layout with segment ends, `endpoint_valid_segments` | "Records added to run.json" |
| 8 | The workflow configuration file in `run.json` (R4) | W1: a new top-level key `workflow_config` with path and SHA-256 | "Records added to run.json" |
| 9 | Whether the workflow writes `run.json` (R5) | K1: yes, `run.json` alone by default, through `rules.<rule>.parameters.checkpoints` of the processing rules | "Records added to run.json" |
| 10 | The format version of `run.json` | V1: stays `1`; the five additions are additive keys | "Format version" |
| 11 | The shifts of a `run.json` written before R1 | OF1: read `registered/transforms.json` when it exists, otherwise unavailable; the QC records' transform summaries are not read | "Older files" |
| 12 | The area a FOV owns (cell density) | A1 of the baseline (G6): the report computes it from the placement, the grids and the assembly configuration | "The two routes" |
| 13 | The times of stitching, assembly and the per-FOV segmentation and assignment | Unavailable with reason `not_recorded` (baseline, G7); adding `wall_s` to those records is a later choice | "Parts, their inputs and their reasons" |
| 14 | The provisional limits of the problematic-FOV rules | The defaults of `ReportLimits` below, from W-384's candidate limits; Jiahao sets the numbers | "Configuration" |
| 15 | The pixel width and format of the whole-sample figures | 1600 px, JPEG at quality 85 for figures that draw the fused image; PNG for plots (W-384 `figure_sizes.csv`) | "Configuration" |
| 16 | The workflow rule of the report | WR1: rules `report_sample` and `report_dataset`, run when the configuration has a `report` block | "Workflow configuration" |
| 17 | The existing test that pins the exact key set of `run.json` | TC1: change it to "at least these keys" with the additions listed; it is the one existing test the additions change | "Tests the implementation keeps and adds" |
| 18 | The reason of a value to which several reasons apply | P1: one reason, the first that applies in the order of the closed list | "The reasons" |
| 19 | The form of the record of unavailable values | U1: one row per unavailable value, key (`part`, `table`, `column`, `row_key`) | "Where a reason is shown and recorded" |
| 20 | The route of an export whose assembly records none (W-447) | RT1: read as `per_fov`, as `sample_output/_compare.py`: `_route_of` does | "The two routes" |
| 21 | The Preprocessing outputs after a moving round's `post_registration` entries when its transforms cannot be rebuilt (W-447) | PR1: unavailable with the reason; the input and the `steps` outputs are shown | "The replay" |

Choices 14 and 15 are numbers: this page proposes them and states their source; they stay
provisional until Jiahao sets them, and the algorithms page may refine how they are applied.

## Settled decisions this page follows

Decided by Jiahao on 2026-10-07 and 2026-10-08 (W-152, the §2.11 scope and decisions D1 to
D8) and amended on 2026-10-12 (A1 to A3). They are stated as settled and not reopened:

* **The report's sections**, in this order: Summary; Workflow overview; Preprocessing;
  Registration; Readout; Segmentation and assembly; Sample measurements; Files.
* **D1. Entry, input, output.** One sample-level entry on `Dataset`, beside `Dataset.stitch`
  and `Dataset.export` and of the same form, with the FOV list given explicitly. It finds the
  records at their fixed places under the output root. Its output is its own folder
  `<output_root>/report/<sample>/` with `report.html`, `tables/` and the record
  `report.json`; not inside `export/<sample>/`, where `check_export` fails on an unlisted
  file. The report can be made at any stage of a sample's processing and gains content as
  later steps finish. A part whose input is missing is unavailable with a reason, never zero
  or passing. The example images are replayed: the recorded preprocessing recipe is rerun on
  the example FOV from the raw input and the saved registration transforms are applied; a
  registration is never estimated again.
* **D2. Several samples.** A second, separate entry writes a comparison page for a dataset
  (`report/index.html` and `samples.csv`) from the finished sample reports only: a table of
  one row per sample and three box plots by sample (spots per FOV, fraction matched to the
  codebook, cells per FOV). A sample without a report is a row marked missing.
* **D3. How the page is built.** No new dependency and no lock change. A template shipped
  with the package, CSS and JavaScript inline, filled with the standard library; the tables
  embedded as one JSON block; the FOV maps as plain SVG with a small script; statistical and
  image figures static, drawn with matplotlib; a light theme; no external resource.
* **D4. What the user supplies.** One small typed configuration with a `report` block of the
  workflow configuration. Markers as groups of at most four genes, each group one figure;
  `marker_pairs` for MECR. One set of three example FOVs for the whole report, chosen
  automatically unless named: the main example drawn by a fixed seed among the FOVs near the
  median read count, the FOV with the fewest cells and the FOV with the most cells (the
  fewest and the most reads before a segmentation exists). The zoom-in regions of the
  marker patterns are the footprints of these three FOVs. Without markers the patterns show
  the three most abundant genes and say so, and MECR is unavailable. A gene name that is not
  in the codebook raises before anything is written.
* **D5. The quantities** (spots detected, correct barcode form, matched to the codebook,
  passed the read QC filter, deduplication, alignment score, shift, shift residual, cell
  size), each read level as a count and a fraction, the sample totals ending with the reads
  assigned to cells, which equal the sum of `sample.h5ad`'s whole counts and which the
  report checks; problematic FOVs judged on fractions; one rule form (a deviation from the
  sample median by more than a multiple of a robust spread, with an absolute floor, or an
  absolute limit the user sets); MECR with numerator and denominator, a zero denominator
  undefined; no verdict and no composite score.
* **D6. Records the earlier stages add.** The missing records are added to `run.json`
  within §2.11, not recomputed when the report is made. The global transforms are always
  saved; a dense field is not, and stays with the stage `registered`. One implementation
  issue adds all of them and comes first.
* **A1.** The report serves both routes to a sample's cells. Cell density is, on both
  routes, the retained cells attributed to a FOV divided by the area of the region that FOV
  owns. The segmentation preview shows the example FOV's labels on the route per FOV, and
  the fused label image cut to the example FOV's footprint on the route after fusion. The
  population accounting follows the assembly record of the route.
* **A2.** A sample with an export and no `run.json` gets a report. At least one of the
  per-FOV `run.json` records and the export folder must exist.
* **A3.** The comparison of the two routes (`compare_routes`) is not part of the report.

Excluded from the first delivery: neighbour-associated incompatible-marker burden, control
or background summaries, breakdowns by cell type or region, cell-level review flags,
filtering, correction, normalization or batch correction, cell typing, vertical-overlap
detection, biological interpretation.

## Terms

* The **report** is one sample's folder `<output_root>/report/<sample>/`; the
  **comparison page** is the dataset's `<output_root>/report/index.html` with
  `samples.csv`.
* The **FOV list** is the entry's `fov_ids`: the FOVs expected, in order.
* A **run record** is a FOV's `<output_root>/checkpoints/<fov>/run.json`.
* A **part** is one block of the page with its tables ("Parts, their inputs and their
  reasons"); it is **unavailable** for the sample or for one FOV, with one reason of the
  closed list.
* The **route** is the export's `assembly.route` (`per_fov` or `after_fusion`), written by
  `sample_output/_export.py`: `export_sample`.
* The **example FOVs** are the three FOVs whose images the page shows; the **main example**
  is the first.
* The **replay** reruns a FOV's recorded preprocessing on its raw input and applies its saved
  transforms.
* A **global** transform is a step result of kind `translation` or `affine`
  (`registration/_chain.py`: `transform_kind`); a **dense** or **B-spline** one is of kind
  `dense` or `bspline`.

## Names

All names are additions; no existing public name, module or format version changes.

| Concept | Name | Kind | Module |
| --- | --- | --- | --- |
| The sample entry | `Dataset.report(sample, fov_ids, *, assignment="default", segmentation=None, config=None, output_dir=None)` returning `ReportRecord` | method | `starfinder.dataset` (implemented in `starfinder.report`) |
| The comparison entry | `Dataset.compare_samples(samples, *, report_root=None)` returning `ComparisonRecord` | method | `starfinder.dataset` (implemented in `starfinder.report`) |
| Configuration | `ReportConfig`, `MarkerGroup`, `RuleLimit`, `ReportLimits` | frozen dataclasses | `starfinder.report` |
| Results | `ReportRecord(path, record)`, `ComparisonRecord(path, record)` | frozen dataclasses | `starfinder.report` |
| Reader | `read_report(folder)` returning the checked `report.json` | function | `starfinder.report` |
| Closed lists | `UNAVAILABLE_REASONS`, `REPORT_PARTS`, `REPORT_TABLES`, `PROBLEM_RULES` | tuples | `starfinder.report` |
| Template | `report.html` and `index.html` templates | package data | `starfinder/report/templates/` |
| Records | `report.json` (`stage` `report`, `format_version` 1); in `run.json` the keys `transforms`, `codebook`, `workflow_config` and `counts.filtering.endpoint_valid` | JSON | — |

`docs/api/` gains a page for `starfinder.report`, and `docs/api/inventory.rst` and
`python-index.rst` list its exports, which `docs/check_reference.py` requires.

### Options for the names

| Option | Entries | Module | Note |
| --- | --- | --- | --- |
| **N1 (recommended)** | `Dataset.report`, `Dataset.compare_samples` | `starfinder.report` | Verbs of the same form as `stitch` and `export`; "report" names what is written, not a judgement. |
| N2 | `Dataset.processing_report`, `Dataset.dataset_report` | `starfinder.reporting` | `starfinder.reporting` is the module name of the reference branch's summary writer ({doc}`report-baseline`), with another meaning. |
| N3 | `Dataset.qc_report`, `Dataset.qc_summary` | `starfinder.qc` | "QC" suggests a verdict, which the report does not give. |

## The sample entry

```python
Dataset.report(sample: str, fov_ids: Sequence[str], *, assignment: str | None = "default",
               segmentation: str | None = None, config: ReportConfig | None = None,
               output_dir: str | Path | None = None) -> ReportRecord
```

A sample-level reader, not a runner: it calls no processing stage, reads only saved files
(and the raw input of the main example for the replay) and writes only its output folder.

| Parameter | Meaning |
| --- | --- |
| `sample` | The sample, as `Dataset.stitch` and `Dataset.export` name it: the folders `stitching/<sample>/`, `export/<sample>/` and `report/<sample>/`. |
| `fov_ids` | The FOV list: nonempty, unique, in the order the report shows them. The workflow passes the sample's FOVs from the sample annotation (`dataset/workflow.py`: `_sample_fovs`), as `stitch_sample` does. |
| `assignment` | The §2.9 assignment name of the route per FOV, read at `checkpoints/<fov>/assignment/<assignment>/` (written by `assignment/_persist.py`: `write_assignment`); the default of `Dataset.export`. `None`: a sample without cells. |
| `segmentation` | The segmentation run whose outlines the examples show. On the route per FOV, the name of a run under `checkpoints/<fov>/segmentation/<run>/` (written by `segmentation/_persist.py`: `write_run`); `None` is the cell run the assignment read (`assignment.json` `inputs.cells`), with the nucleus run when `inputs.nuclei` names one. On the route after fusion, a run of the export's `segmentation` section (written by `sample_output/_segment.py`: `segment_fused`); `None` is its `cells` run, with its `nuclei` run. |
| `config` | A `ReportConfig`; `None` is `ReportConfig()`. |
| `output_dir` | `None` (the default) is `<output_root>/report/<sample>/`, the place the comparison entry reads. Another folder is allowed (for a test or a copy). |

### Checks before anything is written

In this order; each raises with the output folder unchanged:

1. **Types and values.** `config` is a `ReportConfig` (`TypeError`); `fov_ids` is a
   nonempty sequence of unique strings and `sample` a nonempty string (`ValueError`).
2. **Something to report.** At least one FOV of the list has a run record, or
   `export/<sample>/export.json` exists (A2). Otherwise `FileNotFoundError`, naming both
   places.
3. **The run records.** Each run record found parses as JSON with `format_version` 1, and
   its `dataset_id` and `sample_id` equal the `Dataset`'s and its `fov_id` the FOV of its
   folder (`ValueError` naming the file, the key and both values). Both are written by
   `dataset/_run_record.py`: `_RunRecord.__init__`.
4. **The export.** When `export.json` exists: it passes
   `sample_output/_record.py`: `verify_section_hashes`, its `sample` equals `sample`, the
   FOVs of its `assembly.fovs` equal the FOV list as a set, and on the route per FOV its
   assignment name equals `assignment` (`ValueError` naming both). When
   `stitching/<sample>/placement.json` exists, its FOVs (`inputs`, written by
   `stitching/_persist.py`: `save_placement`) equal the FOV list as a set.
5. **The markers.** Every gene of `config.markers` and `config.marker_pairs` is in the gene
   list: the genes of the codebook file the run records name (`codebook`, R2), checked
   against its recorded SHA-256, or, without one, the export's `assembly.genes`. A gene not
   in it raises `ValueError` naming the gene and the file the list came from. Markers given
   with no gene list to check them against raise `ValueError` naming the two places.
6. **The example FOVs.** `config.example_fovs`, when given, are three distinct FOVs of the
   list (`ValueError`).
7. **The output folder.** The report's own files are `report.json` and the files its `files` list names. An existing output folder is replaced only when it holds no file other than the report's own; a file that is not one of them, which includes every file of a nonempty folder without `report.json`, stops the entry with `FileExistsError` naming that file before anything is written, and an empty folder is used as it is.

Anything that goes wrong after these checks makes a part unavailable with its reason; the
entry does not raise for a missing or changed input file.

### Result

`ReportRecord(path, record)`: the folder and the `report.json` content. The entry also
emits no warning for an unavailable part: the record lists them.

## The comparison entry

```python
Dataset.compare_samples(samples: Sequence[str], *, report_root: str | Path | None = None) -> ComparisonRecord
```

It reads only finished sample reports: `report_root` (default `<output_root>/report/`)
`/<sample>/report.json` and the tables that record lists, each checked against its
recorded SHA-256 (`ValueError` naming the file). A sample whose folder or `report.json` is
missing is a row with `report` `missing`. It writes `report_root/index.html` and
`report_root/samples.csv`, replacing earlier ones; it reads no run record, placement or
export, and recomputes nothing a sample report holds. `samples` is nonempty and unique
(`ValueError`).

## Configuration

```python
@dataclass(frozen=True)
class MarkerGroup:
    name: str                      # the figure's title
    genes: tuple[str, ...]         # 1 to 4 distinct gene names

@dataclass(frozen=True)
class RuleLimit:
    k: float | None = None         # multiple of the robust spread, 1.4826 × MAD of the sample
    floor: float | None = None     # absolute floor of the deviation, in the rule's unit
    absolute: float | None = None  # an absolute limit used instead of k and floor

@dataclass(frozen=True)
class ReportLimits:
    shift_residual: RuleLimit = RuleLimit(k=5.0, floor=5.0)            # voxels, per round and axis
    alignment_score: RuleLimit = RuleLimit()                           # not evaluated by default
    correct_form_fraction: RuleLimit = RuleLimit(k=3.0, floor=0.05)    # low side, fraction
    matched_fraction: RuleLimit = RuleLimit(k=3.0, floor=0.05)         # low side, fraction
    spots_near_zero: float = 0.05                                      # fraction of the sample median

@dataclass(frozen=True)
class ReportConfig:
    markers: tuple[MarkerGroup, ...] = ()
    marker_pairs: tuple[tuple[str, str], ...] = ()
    positive_threshold: int = 1
    example_fovs: tuple[str, str, str] | None = None
    limits: ReportLimits = ReportLimits()
    figure_width_px: int = 1600
    jpeg_quality: int = 85
```

| Field | Default | Meaning and checks |
| --- | --- | --- |
| `markers` | `()` | Groups of one to four distinct genes, each one figure of the marker patterns; names unique. Empty: the patterns show the three most abundant genes of the sample (by retained molecules) and say so (D4). |
| `marker_pairs` | `()` | Pairs of two distinct genes for MECR, in the order the table shows them. Empty: MECR is unavailable with reason `no_markers`. |
| `positive_threshold` | `1` | A cell is positive for a gene when its whole count is at least this (an integer ≥ 1). |
| `example_fovs` | `None` | Three distinct FOVs of the list, or `None` for the automatic choice ("The example FOVs"). |
| `limits` | `ReportLimits()` | The limits of the problematic-FOV rules. A `RuleLimit` with `absolute` set uses that limit alone; with `k` and `floor` set it uses a deviation from the sample median larger than `max(k × 1.4826 × MAD, floor)`; with all three `None` the rule is not evaluated and `report.json` says so. `k`, `floor` and `absolute` are ≥ 0. `spots_near_zero` is in (0, 1): a FOV whose spots are below this fraction of the sample median. |
| `figure_width_px` | `1600` | The pixel width of a figure that draws the whole-sample image; a positive integer. |
| `jpeg_quality` | `85` | The JPEG quality (1 to 95) of those figures; plots are PNG. |

**The provisional limits (choice 14).** From W-384's `candidate_limits.csv` on the logs of
the earlier pipeline for three samples, never pooled: shift residual `k = 5`, floor 5
voxels (marks 1 of 64 LN sample1 FOVs, the edge FOV Position001, and none in
cell-culture-3D or tissue-2D; an absolute limit of 10 voxels marks 35 of 56 tissue-2D FOVs,
whose shifts drift smoothly across the grid); correct-form and matched fractions, low side,
`k = 3`, floor 0.05 (1 LN, 0 culture, 2 tissue-2D FOVs); spots below 0.05 × the sample
median (none in the three samples; the lowest LN FOV is at 0.07). The alignment score has no
default because no earlier log holds a registration quality value; W-384 recommends deciding
it from the `ncc_after` distribution of the first Python run of a real sample. The earlier
pipeline defines its quantities differently ({doc}`report-baseline`, discrepancy 6), so
these numbers are a starting point for Jiahao, not a measurement of the Python pipeline.
How a rule is computed (which FOVs enter the median, the high and low sides, the rounds and
axes) is on the algorithms page.

| Option | Provisional limits (choice 14) | Note |
| --- | --- | --- |
| **L1 (recommended)** | The defaults above, from W-384's candidates | Each marks at most a few FOVs per sample in the earlier logs. |
| L2 | No default: every rule off until Jiahao sets its numbers | The table "Problematic FOVs" stays empty for every sample until then. |
| L3 | Other candidates of W-384's table, for example shift residual `k = 3`, floor 2 voxels, or an absolute 10 voxels | `k = 3`, floor 2 marks 13 of 70 cell-culture-3D FOVs; 10 voxels marks 35 of 56 tissue-2D FOVs. |

**The figure width (choice 15).** W-384 measured a whole-sample projection at the LN pixel
size: 1.1 MB as JPEG and 4.2 MB as PNG at 1600 px, 1.4 MB and 4.7 MB with three marker
genes; base64 adds a third. 1000 px halves the JPEG bytes; 2400 px costs 1.9 to 2.4 MB.

| Option | Whole-sample figures (choice 15) | Note |
| --- | --- | --- |
| **FW1 (recommended)** | 1600 px, JPEG at quality 85 | About 1.5 to 1.9 MB per figure in the page. |
| FW2 | 1000 px, JPEG at quality 85 | Half the bytes; less detail in the zoom-in regions. |
| FW3 | 2400 px, JPEG at quality 85 | 1.9 to 2.4 MB per figure before base64. |
| FW4 | PNG at 1600 px | 4.2 to 4.7 MB per figure; lossless. |

## Workflow configuration

A Python-only top-level `report` block (rejected unless `backend: python`), read by a new
adapter in `dataset/workflow.py`, each key a field of `ReportConfig`; lists become tuples,
and an unknown key raises `ValueError` naming it, as for the other typed blocks:

```yaml
report:
  markers:
    - name: lymph node compartments
      genes: [Cd3e, Cd79a, Lyz2, Pecam1]
    - name: T cell subsets
      genes: [Cd4, Cd8a, Foxp3]
  marker_pairs:
    - [Cd3e, Cd79a]
    - [Cd4, Cd8a]
  positive_threshold: 1
  example_fovs: null              # or three FOV IDs
  limits:
    shift_residual: {k: 5, floor: 5, absolute: null}
    alignment_score: {k: null, floor: null, absolute: null}
    correct_form_fraction: {k: 3, floor: 0.05, absolute: null}
    matched_fraction: {k: 3, floor: 0.05, absolute: null}
    spots_near_zero: 0.05
  figure_width_px: 1600
  jpeg_quality: 85
```

The example is W-383's `markers.yaml` for LN sample1, shortened. `markers` entries are
`MarkerGroup(name, genes)`; `marker_pairs` entries are pairs; `limits` is `ReportLimits`
with each rule a `RuleLimit`.

### Options for the rules (choice 16)

| Option | What runs | Note |
| --- | --- | --- |
| **WR1 (recommended)** | With a `report` block: `report_sample` per sample, calling `Dataset.report(sample, <the sample's FOVs>, assignment=<assembly.assignment>, config=…)`, declared output `report/{sample}/report.json`, declared input the last per-sample output the configuration has (`export/{sample}/export.json` with an `export` block; else `stitching/{sample}/placement.json`; else the FOVs' `log/sf_scores/{fovID}.txt`); and `report_dataset`, calling `Dataset.compare_samples` on every sample, with output `report/samples.csv`. `report: {}` gives the defaults. | A configuration without the block runs as today. |
| WR2 | Both rules always run with the Python backend. | Every existing configuration gains two rules and their outputs. |
| WR3 | No rule; the user calls the entries. | The workflow writes no report. |

A static schema test keeps the block equal to the fields of `ReportConfig`, as for the other
blocks.

## Inputs

### Where each record is looked for

Every place is fixed under the output root (D1); nothing is searched for.

| Input | Place | Written by, at `dc5fc3c` | Needed for |
| --- | --- | --- | --- |
| Run record | `checkpoints/<fov>/run.json` | `dataset/_run_record.py`: `_RunRecord.write`, `_RunRecord.finish` (`FOV.run` with a `CheckpointConfig`) | Every part up to the readout; the identity, version and times |
| Saved transforms | `checkpoints/<fov>/registered/transforms.json`, `<round>_field.npz` | `io/_checkpoint.py`: `write_registered_header` (stage `registered`) | The shifts of a run record without `transforms`; the dense and B-spline fields of the replay |
| Reads before the filter | `checkpoints/<fov>/pre_qc.json` | `io/_checkpoint.py`: `write_pre_qc` (stage `pre_qc`) | Only the codebook layout's segment ends, for R3 of an older run record |
| Raw input | the files `run.json` `inputs[*].path` names | the acquisition | The replay of the main example |
| Supplied statistics | `run.json` `preprocessing.supplied_statistics.path` | the user | The replay of a recipe with `fit="supplied"` |
| Codebook | `run.json` `codebook.path` (R2) | the user | The gene list of the marker check |
| Per-FOV segmentation and assignment | `checkpoints/<fov>/segmentation/<run>/`, `checkpoints/<fov>/assignment/<assignment>/` | `segmentation/_persist.py`: `write_run`; `assignment/_persist.py`: `write_assignment` | The per-FOV lane on the route per FOV; the segmentation examples on that route |
| Reference image | `images/ref_merged/<fov>.tif`, or the `reference_image` pattern | `dataset/fov.py`: `FOV.save_reference_image` | The background of the segmentation examples on the route per FOV; the Z size when no export exists |
| Placement | `stitching/<sample>/placement.json`, `fovs.csv` | `stitching/_persist.py`: `save_placement` | The FOV maps, the correction from the stage position, the placement status, the owned area |
| Export record | `export/<sample>/export.json` | `sample_output/_record.py`: `write_export`, `update_section` (through `sample_output/_export.py`: `export_sample`) | The route, the parts, the population accounting, the fused image and label files, the export's software |
| Sample tables | `export/<sample>/tables/fovs.csv`, `cells.csv`, `molecules.csv`, `sample.h5ad` | `assembly/_persist.py`: `save_assembly`; `assembly/_anndata.py`: `sample_anndata` | Cells per FOV, reads in cells, per-cell distributions, MECR, marker patterns, reads along Z |
| Fused image | the `fusion.outputs` of `export.json` (`<image>_projection.ome.tif`, `<image>_stack.ome.tif`) | `sample_output/_fuse.py`: `fuse` | The marker patterns' image; the XZ zoom (a stack) |
| Sample label image | `export/<sample>/segmentation_<run>.ome.tif` | `sample_output/_segment.py`: `segment_fused` | The segmentation examples on the route after fusion |

Each file the report reads is checked against the SHA-256 a record holds for it, where one
does (`run.json` `inputs`; `codebook.sha256`; `assignment.json` `files` and `inputs.cells.file`;
`export.json` `assembly.files` and the sections' file lists), and is listed with its
SHA-256 in `report.json` `inputs`. A missing file makes its parts unavailable with
`input_missing`, a changed one with `input_changed`.

### What must exist

At least one of the FOVs' run records and the export folder (A2, check 2). Everything else
is optional: each missing input makes the parts that need it unavailable, with the reason
"Parts, their inputs and their reasons" names.

### A FOV without a run record

It is a row of every per-FOV table with `run_record` false. In the sections up to the
readout (Workflow overview, Preprocessing, Registration, Readout) its values are null, and
`unavailable.csv` names each of those null cells on its own with reason `no_run_record`
("Where a reason is shown and recorded"); the workflow
lane and the FOV status show "no run record". The later sections are made from the
export, which holds the FOV when it was exported. The rule "expected and not run" of the
problematic FOVs marks it only when at least one FOV of the list has a run record; a sample
with no run record at all (A2) is not marked FOV by FOV, and `report.json` records the rule
as not evaluated with reason `no_run_record`.

### The report at each stage of a sample's processing

| What exists | What the report holds | Unavailable, with reason |
| --- | --- | --- |
| Run records of some or all FOVs, nothing else | Summary (identity, configuration check, version, times, the key numbers up to the matched fraction), the per-FOV lane, FOV status and times, preprocessing and registration examples (replayed), the registration heatmap, the readout totals and per-FOV values, their distributions and table; the problematic FOVs of the workflow, registration and readout rules | The FOV maps (`no_placement`); Segmentation and assembly, Sample measurements, the reads assigned to cells (`no_export`); a running FOV's later values (`run_not_finished`) |
| … and the placement | The FOV maps of registration and readout, the placement block (correction and status), the rules "not placed or fell back" | The owned area and density (`no_export`: the ownership configuration is in the export) |
| … and the per-FOV assignments (route per FOV) | The segmentation examples and the per-FOV lane's segmentation and assignment | Cells per FOV, accounting, per-cell values (`no_export`) |
| … and the export, route per FOV | Everything | Only what its parts did not write (`part_not_written`) or the configuration lacks (`no_markers`, `no_stack`) |
| … and the export, route after fusion | Everything, with the route's meanings ("The two routes") | As above |
| The export, assembled `molecules_only` (no assignment) | As above without cells | Cells, density, reads in cells, the segmentation examples, per-cell values, MECR, "reads but no cell" (`no_segmentation`) |
| **The export and no run record** (A2; for example an export made from saved molecule tables, as W-398 made for LN sample1 and cell-culture-3D) | Summary identity (dataset from the `Dataset`, sample, version and the Z size from the export), the placement block and maps (from the placement folder when it exists, otherwise the export's footprints and status), the reads that passed the earlier filter per FOV (`tables/fovs.csv` `n_molecules`), the population accounting, the marker patterns, the reads along Z, and, with cells, every cell value | Everything up to the readout (`no_run_record`: steps, methods, times, status, inputs, preprocessing, registration and its QC, spots detected, correct form, matched, deduplication); the correction from the stage position without a placement folder (`no_placement`); the rounds and channels (`no_run_record`) |

## Unavailable parts

### The reasons

The list is closed (`UNAVAILABLE_REASONS`); a reason outside it is an error of the
implementation, which a test checks. A value gets one reason: when several apply, the first
of them in the order of this table (a FOV without a run record and without an export is
`no_run_record` for its spots and `no_export` for its cells). Choice 18: **P1
(recommended)**, this rule; P2, every reason that applies, `;`-joined in `reason`, which
makes the column a list and its values harder to count; P3, leave it unstated, so that two
implementations may give one value different reasons.

| Reason | When |
| --- | --- |
| `no_run_record` | The FOV has no run record at its place. |
| `run_not_finished` | The run record's `status` is `running`: the run is going on, or it stopped without recording its end. |
| `step_failed` | The run's `status` is `failed` or `interrupted`, and the failing step (`error.step`) comes at or before the step that makes the value. |
| `not_configured` | The step or check that makes the value was not part of the run's configuration (its configuration is null, or the filter checked no endpoints), or a rule has no limit. |
| `not_recorded` | The step ran, but no record holds the value: a run record written before a field of "Records added to run.json", or a time that no record holds (stitching, assembly, the per-FOV segmentation and assignment). |
| `no_placement` | `stitching/<sample>/placement.json` does not exist. |
| `fov_not_placed` | The FOV's placement status is `missing` or `failed`, so it has no position on a map. |
| `no_export` | `export/<sample>/export.json` does not exist. |
| `part_not_written` | The export part that writes the input is `off`, `not_run`, `failed` or pending in `export.json` `parts`; the detail names the part and its status. |
| `no_segmentation` | The sample has no cells: the export is `molecules_only`, or `assignment` is `None`. |
| `no_markers` | MECR without `marker_pairs`. |
| `no_stack` | The XZ zoom without a fused stack of more than one plane (`dimensionality` `projection`, or a sample with `Z=1`). |
| `input_missing` | A file a record names does not exist (a raw input, a field `.npz`, a label image, the codebook, the supplied statistics). |
| `input_changed` | A file's SHA-256 differs from the one its record holds. |
| `field_not_saved` | The round's chain has a dense or B-spline step whose field was not saved (stage `registered` off). |
| `replay_differs` | The replayed preprocessing's fitted values differ from the recorded ones, so the replayed image is not the run's. |
| `undefined` | The value has a zero denominator: a fraction of zero spots, MECR with no cell positive for either gene, a median over no FOV. |

### Parts, their inputs and their reasons

`REPORT_PARTS`, in page order. "Per FOV" marks a part that can be unavailable for one FOV
while the others show; the other parts are unavailable for the whole sample. Each part's
inputs are in "Where each record is looked for".

| Part | Section | Inputs | Per FOV | Reasons it can be unavailable |
| --- | --- | --- | --- | --- |
| `summary.identity` | Summary | The `Dataset`'s IDs; `run.json` `channels`, `config`, `code`, `started_at`, `ended_at`; `export.json` `assembly.software` and the fused grid's Z size (`fusion.fovs[*].grid.shape_zyx`), else the reference image's shape | | `no_run_record` (rounds, channels, configuration check), `not_recorded` |
| `summary.key_numbers` | Summary | `run.json` `status`, `counts.spots`, `counts.call_status.assigned`; `sample.h5ad` `X` | | `no_run_record`, `run_not_finished`, `step_failed`, `not_configured` (no FOV has the count), `no_export`, `part_not_written`, `no_segmentation`, `undefined` |
| `summary.problematic_fovs` | Summary | The values each rule reads (`PROBLEM_RULES`); `config.limits` | yes | A rule whose value is unavailable for a FOV is not evaluated for it, with that value's reason; `not_configured` for a rule without limit |
| `workflow.steps` | Workflow overview | `run.json` `config.pipeline`, `steps`, `preprocessing.recipe`, `registration`; on the route per FOV `segmentation.json` `methods` (a computed run `FOV.segment` saved with checkpoints; `segmentation/_persist.py`: `write_run`) and `assignment.json` `inputs.cells`; `placement.json` `method.method`; `export.json` `parts` and `segmentation.runs` | yes | `no_run_record`, `no_placement`, `no_export` |
| `workflow.fov_status` | Workflow overview | `run.json` `status`, `error` | yes | `no_run_record` |
| `workflow.time` | Workflow overview | `run.json` `steps[*].seconds`; `export.json` `fusion.wall_s`, `labels.wall_s`, `spatialdata.wall_s`, `segmentation.wall_s` | yes | `no_run_record`, `run_not_finished`; `not_recorded` for stitching, assembly and the per-FOV segmentation and assignment |
| `preprocessing.example` | Preprocessing | The main example's `run.json` `inputs`, `config.pipeline.load`, `config.pipeline.rotation_degrees`, `preprocessing`; the raw input; the supplied statistics; for the outputs of a moving round's `post_registration` entries, also the round's transforms as "The replay" step 4 reads them: `run.json` `transforms` (R1), else `registered/transforms.json`, and a dense or B-spline field file `registered/<round>_field.npz` | | `no_run_record`, `step_failed`, `not_configured` (no recipe), `input_missing`, `input_changed`, `replay_differs`; for the outputs of a moving round's `post_registration` entries, per round: `step_failed` (the round's registration failed), `not_recorded` (neither record holds the round's transforms), `field_not_saved` (a dense or B-spline field was not saved), `input_missing` or `input_changed` (the field file); when no example could be chosen, the reason of that choice (`no_run_record`, `run_not_finished`, `step_failed` or `not_configured`; "The example FOVs") |
| `preprocessing.intensity` | Preprocessing | The replay of `preprocessing.example`; for the distributions after a moving round's `post_registration` entries, also the round's transforms: `run.json` `transforms` (R1), else `registered/transforms.json`, and a dense or B-spline field file `registered/<round>_field.npz` | | as `preprocessing.example`, with the same per-round reasons for the distributions after a moving round's `post_registration` entries |
| `registration.maps` | Registration | `run.json` `transforms` (R1), else `registered/transforms.json`; `registration.<round>` (outcome, fallback, the application entry's `qc.values.ncc_after`); the placement | yes | `no_run_record`, `step_failed`, `not_configured` (no registration), `not_recorded`, `no_placement`, `fov_not_placed`, `undefined` |
| `registration.heatmap` | Registration | As `registration.maps`, without the placement | yes | as `registration.maps` but `no_placement` and `fov_not_placed` |
| `registration.example` | Registration | The replay of the main example; its transforms (R1 or `transforms.json`) and `application_config`; a dense or B-spline field file | | as `preprocessing.example`, and `not_configured`, `not_recorded`, `field_not_saved` |
| `readout.totals` | Readout | `run.json` `counts.spots`, `counts.filtering.endpoint_valid` (R3), `counts.call_status.assigned`, `counts.deduplication.merged_reads`, `counts.filtering.accepted`; `sample.h5ad` `X`; `tables/molecules.csv` | | `no_run_record`, `run_not_finished`, `step_failed`, `not_configured`, `not_recorded`, `no_export`, `no_segmentation` (the reads assigned to cells) |
| `readout.per_fov` | Readout | As `readout.totals`, per FOV; the placement for the map | yes | as `readout.totals`, and `no_placement`, `fov_not_placed` for the map, `undefined` for a fraction |
| `readout.distributions` | Readout | `tables/fovs.csv` of the report | | `no_run_record` (no FOV has one) |
| `readout.table` | Readout | `tables/fovs.csv` of the report | | `no_run_record` (no FOV has one) |
| `assembly.cells_per_fov` | Segmentation and assembly | `export/…/tables/fovs.csv` `n_cells`, `n_cells_retained`, `n_molecules_retained`; `tables/molecules.csv`; the owned area (placement, grids, `export.json` `assembly.config`) | yes | `no_export`, `part_not_written` (`assemble`), `no_segmentation`, `no_placement` (density), `undefined` |
| `assembly.placement` | Segmentation and assembly | `stitching/<sample>/fovs.csv` `status`, `correction_z/y/x`; `export/…/tables/fovs.csv` `placement_status` | yes | `no_placement`, `fov_not_placed` |
| `assembly.examples` | Segmentation and assembly | Route per FOV: the assignment folder's label images (`cell_labels.tif`, `nucleus_labels.tif`, or the linked `segmentation/<run>/labels.tif`) and the reference image; route after fusion: `segmentation_<run>.ome.tif`, the fused image and the FOV's footprint | yes | `no_segmentation`, `no_export` (route after fusion), `part_not_written` (`segment`, `fuse`), `input_missing`, `input_changed`; when no example could be chosen, the reason of that choice (`no_run_record`, `run_not_finished`, `step_failed` or `not_configured`; "The example FOVs") |
| `assembly.accounting` | Segmentation and assembly | `export.json` `assembly.counts` | | `no_export`, `part_not_written`, `no_segmentation` (the cells, and the counted molecules, of an export without cells) |
| `sample.per_cell` | Sample measurements | `sample.h5ad` `X`, `obs.size_physical`; the size unit of the assignment record | | `no_export`, `part_not_written`, `no_segmentation` |
| `sample.markers` | Sample measurements | `tables/molecules.csv` (`gene_id`, `sample_x`, `sample_y`, `assembly_status`); the fused image of `export.json` `fusion.outputs` | | `no_export`, `part_not_written` (`fuse`, `assemble`) |
| `sample.marker_zoom` | Sample measurements | As `sample.markers`, and the example FOVs' footprints; for XZ a fused stack | | as `sample.markers`, and `no_stack` (XZ only); when no example could be chosen, the reason of that choice (`no_run_record`, `run_not_finished`, `step_failed` or `not_configured`; "The example FOVs") |
| `sample.mecr` | Sample measurements | `sample.h5ad` `X`, `var`; `marker_pairs`; `positive_threshold` | | `no_markers`, `no_export`, `no_segmentation`, `undefined` (per pair) |
| `sample.reads_along_z` | Sample measurements | `tables/molecules.csv` `fov_id`, `z` | | `no_export`, `part_not_written` |
| `files.input` | Files | `run.json` `inputs`, `codebook` (R2); `placement.json` `table` | | `no_run_record`, `not_recorded`, `no_placement` |
| `files.configuration` | Files | `run.json` `workflow_config` (R4); the report's `config` | | `no_run_record`, `not_recorded` |
| `files.output` | Files | The fixed places that exist | | none |

### Where a reason is shown and recorded

* **On the page**, a part unavailable for the whole sample shows, in place of its figure or
  table, the words "Unavailable:" and the reason's text; a per-FOV value shows a dash, with
  the reason in the tooltip of the map or of the table cell. Nothing unavailable is drawn as
  zero or with a colour of the scale.
* **In `tables/unavailable.csv`** and in `report.json` `unavailable` (the same rows), each
  unavailable value on its own, with its reason and a detail (the file, the key, the step or
  the part). A null cell of a report table, or a null key of `report.json` `summary`, is of
  one of two kinds, and its column's null rule says which:
  * **not applicable** ("n/a: <case>" in the null rule): the value does not exist in that
    case (a FOV that is not an example has no role; a reason the route does not have has no
    count). No row of `unavailable.csv` names it.
  * **unavailable** ("unavailable" in the null rule): exactly one row of `unavailable.csv`
    names it, by its `table`, its `column` and the `row_key` of its row. A value computed from
    other values (a fraction, a residual, a density, a time, a sum over FOVs) gets its own row.
    Its reason is the first, in the order of the reasons, of the reasons of its own missing
    inputs and of the values it is computed from that are unavailable; `undefined` only when
    none is missing and the denominator is 0. A value computed from one other value is
    unavailable whenever that value is; a sum or median over FOVs only when no FOV has the
    value. The same order holds for every computed column.

  A part with no table cell for what it lacks (an example image, a figure of a table that has
  no row, a Files block, a rule not evaluated for the whole sample) has one row with `table`
  null, for the whole sample or for the FOV, round, example role, phase or view it is drawn for. A
  figure drawn from table cells (a map, a distribution) needs no row of its own: the rows of
  its null cells explain it. A rule of the problematic FOVs that cannot be evaluated for a FOV
  because one of its inputs is unavailable needs no row either: the input's row explains it.

  Choice 19: **U1 (recommended)**, this record; U2, one row per part, FOV and round with the
  unavailable columns `;`-joined, which has fewer rows but cannot give two values of one row
  different reasons (an older record lacks the correct-form count and the run failed before
  deduplication); U3, the earlier key (`part`, `fov_id`, `round`), which cannot tell two
  values of one part, FOV and round apart (leave it as it is).

## The two routes

The route is read from `export.json` `assembly.route` (written by
`sample_output/_export.py`: `export_sample`); an assembly saved before the route existed
records none and is read as `per_fov`, as `sample_output/_compare.py`: `_route_of` reads it
(the W-398 exports record none); without an export it is null, and the per-FOV
assignment folders, when they exist, are read as the route per FOV for the examples and the
per-FOV lane. The comparison of the two routes (`compare_routes`) is not read (A3).
Choice 20: **RT1 (recommended)**, an export without `assembly.route` is read as `per_fov`,
as `_route_of` does, so that `report.json` `route` and `samples.csv` `route` are null only
without an export; RT2, the route stays null for such an export, which makes `route` null
for a sample that has one and needs its own null rule in `samples.csv`.

| Report element | Route per FOV | Route after fusion |
| --- | --- | --- |
| `fovs.csv` `n_cells` | The FOV's own cells (`tables/fovs.csv` `n_cells`) | The sample's cells attributed to the FOV by their ownership centroid ({doc}`sample-segmentation-contract`, "The FOVs"); equal to `n_cells_retained` |
| `fovs.csv` `n_cells_retained` | The FOV's cells retained by ownership and the duplicate rule | The same as `n_cells` |
| Cell density (`cell_density_per_mm2`) | `n_cells_retained` / the area the FOV owns | The same formula (A1) |
| Reads in cells (`reads_in_cells`, its fraction) | The FOV's retained molecules whose cell is counted, over its retained molecules | The same columns; a molecule's FOV is the owner of its position |
| The region a FOV owns | From the placement, the FOVs' grids and the export's ownership rule (`assembly.config`, `assembly.ownership`), on both routes; the geometry is on the algorithms page | The same |
| Segmentation examples | The FOV's own label images from its assignment folder, on its own grid, over its reference image | The fused label image of the run cut to the FOV's footprint, over the fused image (A1) |
| Population accounting (`population.csv`) | Cells and molecules: input, retained, excluded `not_owner`, `fov_not_placed`, `duplicate_observation` (with the rule on), molecules also `cell_not_owner`, `cell_duplicate_observation`, `outside_grid` | Cells: input (the run's labels), retained, `not_owner`, `fov_not_placed`; molecules: `not_owner`, `outside_grid`, `fov_not_placed`, `cell_not_owner`; no duplicate rows (null, not applicable) and the flags null (`export.json` `assembly.counts`) |
| Workflow lanes | The per-FOV lane has segmentation (or the imported mask) and assignment | The per-sample lane has the export's `segment` part with its runs' methods |
| Time of the segmentation | `not_recorded` | `export.json` `segmentation.wall_s` |
| Rule "reads but no cell" | `n_molecules_retained > 0` and `n_cells_retained = 0` | The same columns |

The page names the route in the Summary. Every other table and figure has the same
columns on both routes.

## The example FOVs

* **Roles.** `example_1` is the main example: the Preprocessing and Registration examples
  show it. All three are shown in the segmentation examples and are the zoom-in regions of
  the marker patterns.
* **Automatic choice** (`example_fovs` `None`). `example_1` is drawn with the fixed seed 0
  among the FOVs near the median read count; `example_2` is the FOV with the fewest counted
  cells and `example_3` the one with the most (`n_cells_retained` of `tables/fovs.csv`),
  or, before a segmentation exists, the fewest and the most reads. The read count is
  `counts.filtering.accepted`. Only FOVs with a value enter; when no FOV has one, the automatic choice is unavailable with the first reason, in the order of the reasons, of the FOVs' `fovs.csv` `passed_filter` cells (`no_run_record` for a sample without run records, A2; `not_configured` for a run without a filter, or without detection), and the user can name the three FOVs. What "near the median" is,
  the ties, and what happens when two roles fall on one FOV are open points of
  the algorithms page.
* **Named** (`example_fovs` given, choice 3). The three FOVs in the given order are
  `example_1`, `example_2` and `example_3`.

| Option | Roles of named FOVs | Note |
| --- | --- | --- |
| **EX1 (recommended)** | `example_1`, `example_2`, `example_3` in the given order, on both choices | One set of names; the record says how each was chosen. |
| EX2 | `main`, `fewest`, `most`, also when named | A named FOV labelled "fewest" may not have the fewest cells. |

* **Recorded.** `report.json` `examples`: one entry per role with `fov_id`, `chosen_by`
  (`automatic` or `configuration`), and for the automatic choice the `rule`
  (`median_reads`, `fewest_cells`, `most_cells`, `fewest_reads`, `most_reads`), the `value`
  and the `seed`; `tables/fovs.csv` `example` holds the role.

## The replay

What the examples of Preprocessing and Registration show, rebuilt from the run record
(D1); W-384 replayed a synthetic FOV this way and got every round bit-identical
(`results/replay/replay.csv`).

1. **The raw input.** The main example's files `run.json` `inputs` lists for the rounds the
   examples show (which rounds is an open point of the algorithms page). Each file's
   SHA-256 is checked against the recorded one (written by `_RunRecord.add_inputs`; null
   with `hash_inputs=False`, in which case the file is read unverified and
   `report.json` `replay.verified` is false).
2. **Loading and rotation**, as `FOV.run` does first for each round (`dataset/fov.py` lines
   1312–1313 in batch mode, 1323–1324 in streaming mode, then 1327–1328, at `dc5fc3c`): the
   round is loaded with `config.pipeline.load` through `io/tiff.py`: `load_round`, which
   applies the configuration's `conversion` when it has one (`io/tiff.py` line 300), and
   rotated by `config.pipeline.rotation_degrees` with the code of `FOV._rotate_round`
   (`_rotated`).
3. **The `steps` phase.** The entries of `preprocessing.rounds[<round>]` whose `stage` is
   `steps`, in `index` order. `FOV._preprocess` records each entry with `stage` set to its
   phase (`steps` or `post_registration`) and `step` set to the step's name (`dataset/fov.py`
   lines 521–524), and `FOV.run` runs the `steps` entries of a round right after its rotation
   (lines 1329–1332); `stage` is the field that tells the two phases apart. Each entry is
   rebuilt from its `step` and `config` with `preprocessing.step_config_type` and run with
   `preprocessing.run_step`; an entry with `save_as` keeps its output as that snapshot, as
   `FOV._preprocess` does (lines 526–528). A step with `fit="supplied"` reads the file
   `preprocessing.supplied_statistics` names, checked against its SHA-256. A step whose scope is
   `needs_reference` (histogram matching with `fit="fov"`) takes the reference round's input to
   that same step, channel `reference_channel` (lines 507–517): to replay it on a moving
   round, the replay first replays the reference round's `steps` entries up to that step. The
   replayed fitted values are compared with each entry's recorded `fitted`; a difference makes
   the examples unavailable with `replay_differs`.
4. **Registration, a moving round only.** `FOV.run` registers a moving round after its `steps`
   phase and before its `post_registration` phase (`dataset/fov.py` lines 1336–1344); the
   reference round is not registered and not transformed, and has no entry in R1. The replay
   applies the round's transforms to the image the `steps` phase left, and to each of the
   round's snapshots, as `FOV._register_round` resamples them once (line 813); it estimates
   nothing, so it needs neither the registration signals nor the reference image that the
   estimation reads (lines 1337–1339, 549–566). Each result of the round is rebuilt with `io/_checkpoint.py`:
   `_restore_result`, which reads an entry of the form `_transform_json` writes (`step`,
   `transform`, `diagnostics`) and, for a dense or B-spline result, one array of a field
   file. The entries come from `run.json` `transforms.<round>.results` (R1), or, for a run
   record without R1, from `registered/transforms.json` `transforms.<round>` with the round's
   `applications.<round>.application_config` (written by `write_registered_header`, stage
   `registered`). The results are composed into the round's chain in order and resampled
   once with the round's `application_config`, as `FOV._register_round` does. No
   registration is estimated. The table below names, for each kind, every file and field
   read, the function that writes it, and what the Registration example shows when it was
   not saved. The Preprocessing example and the intensity distributions need the same
   transforms for the outputs of a moving round's `post_registration` entries (step 5), with
   the same reasons when they cannot be rebuilt ("A moving round whose transforms cannot be
   rebuilt", below); they need none of them for the loaded input, for the `steps` entries, for
   the reference round, or for a round without `post_registration` entries.
   | Kind | Fields read, in `transform` of the entry | Files read | Written by, at `dc5fc3c` (R1: proposed) | Not saved |
   | --- | --- | --- | --- | --- |
   | every kind | `kind`, `reference_shape_zyx`, `moving_shape_zyx`, `reference_metadata`, `moving_metadata`, `direction`, `units`; the entry's `diagnostics`; the round's `application_config` | `run.json` (R1), else `registered/transforms.json` | `_transform_json` through `write_registered_header`; R1 through `_RunRecord.write` | Neither record: the example is unavailable, `not_recorded`. A `transforms.json` of `format_version` 1 (sequential semantics, no chain; {doc}`checkpoints`) is not replayed, `not_recorded`. |
   | `translation` | `displacement_zyx` | as above | as above | as above |
   | `affine` | `matrix_zyx`, `physical` | as above | as above | as above |
   | `bspline` | `bspline` (`grid_size_xyz`, `grid_origin_xyz`, `grid_spacing_xyz`, `grid_direction_xyz`, `order`, `spacing_zyx`: the control grid, inline in the entry) and `coefficients` (the name of the field file) | the entry's record, and the array `result_<i>` (`i` the result's index) of `registered/<round>_field.npz`, which holds only the coefficients | the entry as above; the NPZ by `write_registered_header` | No field file, or `coefficients` null in R1: unavailable, `field_not_saved` (choice 4). |
   | `dense` | `field` (the name of the field file) | the entry's record, and the array `result_<i>` of `registered/<round>_field.npz`, the displacement field | as for `bspline` | as for `bspline` |

5. **The `post_registration` phase.** The entries whose `stage` is `post_registration`, in
   `index` order, run on the registered image, as `FOV.run` runs them after the registration
   (lines 1345–1352); the reference round runs them too, after its `steps` phase. On a moving
   round they therefore need the round's transforms (step 4). Without a registration
   (`config.pipeline.registration` null) `FOV.run` registers nothing (line 1341) and runs them
   on the round as its `steps` phase left it, so the replay applies no transform to it.
6. **The registered images** (`registered/<round>.ome.tif`) are not read. `FOV.run` writes
   a round's image after its `post_registration` phase (lines 1353–1355, through
   `FOV._write_registered_round` and `io/_checkpoint.py`: `write_registered_round`), with
   `io/tiff.py`: `save_volume` called without `conversion`, so the saved image keeps the
   dtype and the values of the round at that point (`io/tiff.py` lines 350–352). The image
   the replay ends with after step 5 is therefore the image the checkpoint saves; the
   validation compares the two where the stage was written (W-384 did:
   `results/replay/replay.csv`), and the report itself needs neither the 447 MB per round of
   an LN FOV that the stage writes (W-384 `real_fovs.csv`) nor the stage.

Which state of the round each thing the replay produces shows:

| What the replay produces | State of the round's image |
| --- | --- |
| The intensity distribution and the image of `intensity.csv` `step_index` 0 | Loaded and rotated (after step 2) |
| The same for each `steps` entry | After that entry, in the `steps` phase (step 3) |
| The image the transforms are applied to; the "before" of the Registration example | After the whole `steps` phase, before registration (end of step 3) |
| The "after" of the Registration example | After the transforms (step 4), before the `post_registration` phase |
| The same for each `post_registration` entry (`intensity.csv` `phase` `post_registration`) | After that entry (step 5); on a moving round, after the transforms |
| The image that equals the saved registered image | After the whole `post_registration` phase (end of step 5); without `post_registration` entries, the end of step 4 (a moving round) or of step 3 (the reference round) |

The reference image of the Registration example (green) is the reference round after its
`steps` phase, the state against which `FOV.run` registered the moving round when the recipe
has `post_registration` entries (lines 1337–1339 and 1348–1349).

| Option | Registration example without a saved field | Note |
| --- | --- | --- |
| **RE1 (recommended)** | Unavailable, `field_not_saved` | What is shown is always the run's image. |
| RE2 | The global steps only, labelled so | A partial registration shown as "after" can be read as the run's. |

**A moving round whose transforms cannot be rebuilt.** A moving round of the main example
with `post_registration` entries, in a run with a registration, whose transforms step 4
cannot rebuild: the round's registration failed (`step_failed`: `FOV.run` stopped at the
round, line 1342, and the record holds none of its `post_registration` entries); neither
R1 nor `registered/transforms.json` holds the round, or that file is of `format_version` 1
(`not_recorded`); a dense or B-spline step's field was not saved (`field_not_saved`); or
the field file is missing or changed (`input_missing`, `input_changed`). When several
apply, the first in the order of the reasons is recorded. For that round:

* **Shown:** the loaded and rotated input (`step_index` 0) and the output of each `steps`
  entry, as images in the Preprocessing example and as intensity distributions, since they
  are made before the registration (steps 2 and 3) and need no transform.
* **Unavailable, with that reason:** the output of each `post_registration` entry, its
  image in the Preprocessing example and its intensity distribution. No partial transform
  is applied to them (for a chain whose field was not saved, not its global steps alone),
  for the reason of RE1. `unavailable.csv` names them by two rows without a table, one of
  the part `preprocessing.example` and one of `preprocessing.intensity`, each with the row
  key `fov_id=<main example>;role=example_1;round=<round>;phase=post_registration`, and
  `intensity.csv` has no row of those entries.

A round without `post_registration` entries needs no transform for the Preprocessing
example or the intensity distributions, nor does the reference round, nor a moving round
of a run without a registration: all of their outputs are shown, also when the round's
transforms cannot be rebuilt (the Registration example then follows the table above).
Choice 21: **PR1 (recommended)**, this rule; PR2, apply the global steps of the chain alone
when a field was not saved, labelled so, which shows an image the run never had; PR3, run
the `post_registration` entries on the round before its transforms, labelled so, which
shows the entries on an unregistered image.

`report.json` `replay` records the FOV, the rounds, each input with its SHA-256 and whether
it was verified, the steps rerun, where the transforms came from (`run.json` or
`transforms.json`), and whether the fitted values were equal. How the images are drawn
(the signal, the channels, the planes) is on the algorithms page.

## The output folder

```text
<output_root>/report/
    <sample>/
        report.html
        report.json
        tables/
            fovs.csv
            registration.csv
            steps.csv
            readout_totals.csv
            population.csv
            problematic_fovs.csv
            cells.csv
            mecr.csv
            reads_along_z.csv
            intensity.csv
            unavailable.csv
    index.html                 # the comparison page (Dataset.compare_samples)
    samples.csv
```

Every table is written, also when all its values are unavailable (then with its header and
the rows its row rule gives, or no row). Tables are CSV with the writer of
{doc}`checkpoints` ("Table formats": floats as `%.17g`, a missing value as `<NA>`), and
`report.json` lists each with its dtypes, so that they read back exactly. Parquet is not
offered: the report's tables are small, and Parquet needs the `checkpoint` extra.

**Replacing a report (choice 2).** The report's own files are `report.json` and the files its `files` list names. An existing output folder is replaced only when it holds no file other than the report's own; a file that is not one of them, which includes every file of a nonempty folder without `report.json`, stops the entry with `FileExistsError` naming that file before anything is written, and an empty folder is used as it is. The entry writes the new report into a
staging folder beside the output folder. When it is complete it removes the earlier report's
own files and moves the new files in; an error before that leaves the earlier report as it
was (check 7).

| Option | A second report of the sample | Note |
| --- | --- | --- |
| **RP1 (recommended)** | Replaces the earlier one when complete; a file that is not the report's own stops it | The report gains content as the processing goes on (D1), and a user's file is never removed. |
| RP2 | Refused unless `overwrite=True` | The workflow would always pass `overwrite=True`. |
| RP3 | A new dated folder each time | The comparison page would need to choose one. |

### The tables

Units: `reads` and `molecules` count decoded reads; a fraction is a float in [0, 1];
voxels are of the reference grid of the FOV; µm of the sample frame. Every table's key
identifies one row; rows are in the stated order. In the column "Null when", "unavailable"
means that `unavailable.csv` names the cell on its own (table, column and row key), with the
reasons the part can have; "n/a: <case>" means that the value does not exist in that case and
no row names it; "never" means that the cell is never null. A row key is written
`name=value` for each key column, in key order, joined by `;` (`fov_id=Position001;round=round2`).

**`fovs.csv`**: one row per FOV of the list, in list order; key `fov_id`.

| Column | Type | Unit | Content | Null when |
| --- | --- | --- | --- | --- |
| `fov_id` | string | — | The FOV. | never |
| `list_index` | int64 | — | Its zero-based position in the list. | never |
| `example` | string | — | The FOV's example roles, `;`-joined in role order (`example_1`, `example_2`, `example_3`; a FOV holds several only when two roles fall on one FOV, as in a sample of fewer than three FOVs). | n/a: not an example FOV, also when no example could be chosen (the example parts then have rows of their own) |
| `run_record` | boolean | — | Whether its run record exists. | never |
| `status` | string | — | `run.json` `status`: `running`, `succeeded`, `failed`, `interrupted`. | unavailable (`no_run_record`) |
| `failed_step` | string | — | `run.json` `error.step`, `error.round`; `error` is null unless the run failed or was interrupted (`dataset/_run_record.py`: `_RunRecord.finish`). | unavailable (`no_run_record`); n/a: the run did not fail |
| `failed_round` | string | — | `run.json` `error.step`, `error.round`; `error` is null unless the run failed or was interrupted (`dataset/_run_record.py`: `_RunRecord.finish`). | unavailable (`no_run_record`); n/a: the run did not fail, or failed outside a round |
| `started_at` | string | ISO 8601, UTC | From `run.json`. | unavailable (`no_run_record`) |
| `ended_at` | string | ISO 8601, UTC | From `run.json`. | unavailable (`no_run_record`); n/a: while the status is `running` |
| `run_seconds` | float64 | s | `ended_at − started_at`. | unavailable: the first reason, in the order of the reasons, of `started_at` and `ended_at`; n/a: while the status is `running` |
| `placement_status` | string | — | `stitching/<sample>/fovs.csv` `status` (`image_supported`, `metadata_only`, `placeholder`, `missing`, `failed`), else the export's `placement_status`. | unavailable (`no_placement`, when there is no export either) |
| `correction_z`, `correction_y`, `correction_x` | float64 | µm | `stitching/<sample>/fovs.csv` `correction_*`: origin minus the stage position. | unavailable (`no_placement`; `fov_not_placed` for the status `missing` or `failed`) |
| `footprint_y0`, `footprint_y1`, `footprint_x0`, `footprint_x1`, `centre_y`, `centre_x` | float64 | µm | The export's `tables/fovs.csv`, else from the placement's origin and the grid `placement.json` `inputs.<fov>.grid` records. | unavailable (`no_placement` when neither the placement nor the export exists; `fov_not_placed`) |
| `spots` | int64 | reads | `counts.spots`. | unavailable (`no_run_record`; `run_not_finished` or `step_failed` when the run did not get to detection; `not_configured` without detection: `spot_finding` null and no detections held, so that `_RunRecord.write` writes no `counts.spots`) |
| `correct_form` | int64 | reads | `counts.filtering.endpoint_valid` (R3). | unavailable (`no_run_record`; `run_not_finished` or `step_failed` when the run did not get to the filter; `not_configured` without a filter or without an endpoint check; `not_recorded` in a run record without R3) |
| `matched` | int64 | reads | `counts.call_status.assigned` (0 when `call_status` exists without the key). | unavailable (`no_run_record`; `run_not_finished` or `step_failed` when the run did not get to decoding; `not_configured` without decoding) |
| `merged_by_deduplication` | int64 | reads | `counts.deduplication.merged_reads`, present when the run deduplicated (`PipelineConfig.deduplication` set; `_RunRecord.write`). | unavailable (`no_run_record`; `run_not_finished` or `step_failed` when the run did not get to deduplication; `not_configured` without deduplication) |
| `passed_filter` | int64 | reads | `counts.filtering.accepted`. | unavailable (`no_run_record`; `run_not_finished` or `step_failed` when the run did not get to the filter; `not_configured` without a filter) |
| `correct_form_fraction`, `matched_fraction`, `merged_fraction`, `passed_filter_fraction` | float64 | fraction of `spots` | The count over `spots`, each fraction on its own. | unavailable: the first reason, in the order of the reasons, of its count and of `spots`; `undefined` when `spots` is 0 |
| `n_cells`, `n_cells_retained` | int64 | cells | The export's `tables/fovs.csv`, with the route's meaning. | unavailable (`no_export`; `part_not_written` when the assemble part wrote nothing; `no_segmentation` for an export without cells) |
| `n_molecules`, `n_molecules_retained` | int64 | molecules | The export's `tables/fovs.csv`. | unavailable (`no_export`; `part_not_written` when the assemble part wrote nothing) |
| `reads_in_cells` | int64 | molecules | The FOV's retained molecules whose cell is counted (`tables/molecules.csv`, `tables/cells.csv`). | unavailable (`no_export`; `part_not_written` when the assemble part wrote nothing; `no_segmentation` for an export without cells) |
| `reads_in_cells_fraction` | float64 | fraction of `n_molecules_retained` | `reads_in_cells / n_molecules_retained`. | unavailable: the first reason, in the order of the reasons, of `reads_in_cells` and `n_molecules_retained`; `undefined` when `n_molecules_retained` is 0 |
| `owned_area_um2` | float64 | µm² | The area of the region the FOV owns. | unavailable (`no_placement`; `fov_not_placed`; `no_export`, whose record holds the ownership rule); n/a: not an owning FOV (a `placeholder`) |
| `cell_density_per_mm2` | float64 | cells per mm² | `n_cells_retained / owned_area_um2 × 10⁶`. | unavailable: the first reason, in the order of the reasons, of `n_cells_retained` and `owned_area_um2`; `undefined` when the area is 0; n/a: not an owning FOV |

**`registration.csv`**: one row per FOV of the list and moving round, in list order, then
the rounds in the order of the run records' `registration` keys (the union over the run
records, in first-seen order); key (`fov_id`, `round`).

| Column | Type | Unit | Content | Null when |
| --- | --- | --- | --- | --- |
| `fov_id`, `round` | string | — | The FOV and the moving round. | never |
| `outcome` | string | — | The outcome of the round's last attempt in `run.json` `registration.<round>` (`succeeded`, `rejected`, `failed`, `application_failed`). | unavailable (`no_run_record`; `run_not_finished` or `step_failed` when the run did not get to the round; `not_configured` without registration) |
| `fallback` | boolean | — | Whether a succeeded estimation attempt has `fallback` true. | unavailable (`no_run_record`; `run_not_finished` or `step_failed` when the run did not get to the round; `not_configured` without registration) |
| `methods` | string | — | The `actual_method` of the succeeded estimation attempts, in step order, `;`-joined. | unavailable (`no_run_record`; `run_not_finished` or `step_failed` when the run did not get to the round; `not_configured` without registration); n/a: no estimation attempt of the round succeeded |
| `transform_source` | string | — | `run.json` or `transforms.json`. | unavailable (`no_run_record`; `run_not_finished` or `step_failed` when the run did not get to the round; `not_configured` without registration; `step_failed` when the round's registration failed; `not_recorded` when neither R1 nor `registered/transforms.json` holds the round) |
| `shift_z`, `shift_y`, `shift_x` | float64 | voxels | The displacement of the round's composed global steps, in the convention of `displacement_zyx` ({doc}`registration-contract`); for an affine step the definition is on the algorithms page. | unavailable (`no_run_record`; `run_not_finished` or `step_failed` when the run did not get to the round; `not_configured` without registration; `step_failed` when the round's registration failed; `not_recorded` when neither R1 nor `registered/transforms.json` holds the round; n/a: the round's chain has no global step |
| `residual_z`, `residual_y`, `residual_x` | float64 | voxels | The shift minus its median over the FOVs whose round succeeded. | unavailable: the reason of the shift on the same axis; `undefined` when no FOV's round succeeded; n/a: as the shift |
| `alignment_score` | float64 | dimensionless | The application entry's `qc.values.ncc_after` (masked NCC of the reference and the registered moving round, the whole chain; `evaluation/registration.py`: `registration_qc`). | unavailable (`no_run_record`; `run_not_finished` or `step_failed` when the run did not get to the round; `not_configured` without registration; `step_failed` when the round's registration failed); `undefined` when the application entry's QC holds null (its reason in the detail). It needs no transform record |

**`steps.csv`**: one row per step; key (`scope`, `fov_id`, `index`). Order: the FOVs in list
order, each with its run record's `steps` in record order, then its segmentation and
assignment (route per FOV); then the sample steps (`stitch`, then the export's parts in the
order of `export.json` `parts`).

| Column | Type | Unit | Content | Null when |
| --- | --- | --- | --- | --- |
| `scope` | string | — | `fov` or `sample`. | never |
| `fov_id` | string | — | The FOV. | n/a: a sample step |
| `index` | int64 | — | Zero-based position within its FOV or within the sample. | never |
| `name` | string | — | `run.json` `steps[*].name` (`load_images`, `preprocess:<step>`, `register`, …), `segment`, `assign`, `stitch`, or an export part. | never |
| `round` | string | — | `steps[*].round`. | n/a: not a per-round step |
| `method` | string | — | The step's method: from `steps[*].methods` (detection), the preprocessing step name, the succeeded registration methods, the decoder, `segmentation.json` or `assignment.json` `inputs.cells`, `placement.json` `method.method`. | n/a: the step has none |
| `status` | string | — | `succeeded`, `failed`, or the export part's status (`written`, `reused`, `off`, `not_run`, `failed`). | never |
| `seconds` | float64 | s | `steps[*].seconds`; an export part's `wall_s`. | unavailable (`not_recorded`: stitching, an export part without `wall_s`, the per-FOV segmentation and assignment); n/a: an export part that did not run (`off`, `not_run`) |

**`readout_totals.csv`**: one row per level, in the order `spots`, `correct_form`, `matched`,
`merged_by_deduplication`, `passed_filter`, `reads_in_cells`; key `level`.

| Column | Type | Unit | Content | Null when |
| --- | --- | --- | --- | --- |
| `level` | string | — | The level. | never |
| `count` | int64 | reads | The sum over the FOVs with a value; for `reads_in_cells` the retained molecules with a counted cell, which the report checks against the sum of `sample.h5ad`'s whole counts. | unavailable when no FOV has the value: the first reason, in the order of the reasons, of the FOVs' cells of that level |
| `fraction_of_spots` | float64 | fraction | `count` over the `spots` of the same FOVs. | unavailable: the first reason, in the order of the reasons, of `count` and of the `count` of the level `spots`; `undefined` when the spots are 0 |
| `fovs` | int64 | FOVs | How many FOVs the sum covers. | never |
| `overlaps_counted_per_fov` | boolean | — | True for every level but `reads_in_cells`: a read in an overlap counts once per FOV that saw it (D5). | never |

The levels are not nested: a read can be matched without the correct form, and the
reverse ({doc}`report-baseline`, discrepancy 1). The order follows the pipeline:
deduplication runs before the filter (discrepancy 2).

**`population.csv`**: one row per population, in this order: `input`, `retained`,
`excluded_not_owner`, `excluded_fov_not_placed`, `excluded_duplicate_observation`,
`excluded_cell_not_owner`, `excluded_cell_duplicate_observation`, `excluded_outside_grid`,
`counted`; key `population`.

| Column | Type | Unit | Content | Null when |
| --- | --- | --- | --- | --- |
| `population` | string | — | The population. | never |
| `cells` | int64 | cells | `export.json` `assembly.counts.cells` (`input`, `retained`, `excluded.<reason>`); `counted` is the rows of `sample.h5ad`. | unavailable (`no_export`; `part_not_written` when the assemble part wrote nothing; `no_segmentation` for an export without cells); n/a: a population of molecules only (`excluded_cell_not_owner`, `excluded_cell_duplicate_observation`, `excluded_outside_grid`), or a reason the export's `assembly.counts.cells.excluded` does not hold (`duplicate_observation` on the route after fusion, with the rule off, or in a `molecules_only` export) |
| `molecules` | int64 | molecules | `assembly.counts.molecules`; `counted` is the retained molecules with a counted cell. | unavailable (`no_export`; `part_not_written` when the assemble part wrote nothing; `no_segmentation` for `counted` in an export without cells); n/a: `excluded_duplicate_observation` (a reason of cells only), or a reason the export's `assembly.counts.molecules.excluded` does not hold (`cell_duplicate_observation` on the route after fusion, with the rule off, or in a `molecules_only` export) |

**`problematic_fovs.csv`**: one row per FOV, rule, round and axis that the rule marks, in
list order, then the rule order of `PROBLEM_RULES`, then rounds and axes; key (`fov_id`,
`rule`, `round`, `axis`). The rules (D5): `fov_failed`, `fov_not_run`, `shift_residual`,
`alignment_score`, `registration_rejected`, `registration_fallback`,
`correct_form_fraction`, `matched_fraction`, `spots_near_zero`, `not_placed`,
`stage_position_fallback`, `reads_no_cell`.

| Column | Type | Unit | Content | Null when |
| --- | --- | --- | --- | --- |
| `fov_id` | string | — | The FOV. | never |
| `section` | string | — | The section the rule belongs to. | never |
| `rule` | string | — | One of `PROBLEM_RULES`. | never |
| `round` | string | — | The round. | n/a: a rule that is not per round |
| `axis` | string | — | `z`, `y` or `x`. | n/a: a rule that is not per axis |
| `measured` | float64 | `unit` | The FOV's value. | n/a: a categorical rule |
| `sample_median` | float64 | `unit` | The median the rule compares with. | n/a: a categorical rule, or an absolute limit |
| `limit` | float64 | `unit` | The limit applied: `max(k × 1.4826 × MAD, floor)` or `absolute`. | n/a: a categorical rule |
| `unit` | string | — | `voxels`, `fraction`, `reads` or `dimensionless`. | n/a: a categorical rule |
| `reason` | string | — | The text the table shows, for example "residual above the limit" or "fell back to the stage position". | never |

**`cells.csv`**: one row per counted cell, in `sample.h5ad` `obs` order; key `obs_name`.
Not embedded in the page.

| Column | Type | Unit | Content | Null when |
| --- | --- | --- | --- | --- |
| `obs_name` | string | — | The cell. | never |
| `fov_id` | string | — | Its FOV (`obs`). | never |
| `reads` | int64 | molecules | The row sum of `X`. | never |
| `genes` | int64 | genes | The nonzero entries of its row. | never |
| `size` | float64 | `size_unit` | `obs.size_physical`, or the size in voxels when the calibration is unknown (D5). | never |
| `size_unit` | string | — | `µm³`, `µm²` for a plane, or `voxels` when the calibration is unknown (D5). | never |

**`mecr.csv`**: one row per pair of `marker_pairs`, in configuration order; key
(`gene_a`, `gene_b`).

| Column | Type | Unit | Content | Null when |
| --- | --- | --- | --- | --- |
| `gene_a`, `gene_b` | string | — | The pair. | never |
| `threshold` | int64 | molecules | `positive_threshold`. | never |
| `cells_a`, `cells_b`, `both`, `either` | int64 | cells | Counted cells positive for A, for B, for both, for either. | unavailable (`no_export`; `part_not_written` when the assemble part wrote nothing; `no_segmentation` for an export without cells), each cell of each pair on its own |
| `mecr` | float64 | fraction | `both / either`. | unavailable: the first reason, in the order of the reasons, of `both` and `either`; `undefined` when `either` is 0, for that pair only |

**`reads_along_z.csv`**: one row per FOV of the export and plane, in list order, then `z`
ascending; key (`fov_id`, `z`).

| Column | Type | Unit | Content | Null when |
| --- | --- | --- | --- | --- |
| `fov_id` | string | — | The FOV. | never |
| `z` | int64 | voxel plane | The plane of the FOV's grid (`tables/molecules.csv` `z`, rounded as the algorithms page states). | never |
| `reads` | int64 | molecules | The FOV's molecules of the sample table on that plane, retained or not. | never |

**`intensity.csv`**: the intensity distribution of the main example after each step of the
recipe, in the order `FOV.run` runs them ("The replay"); one row per round, step, channel and
bin, in that order, for each step whose output the replay rebuilds; key (`round`,
`step_index`, `channel`, `bin`). Not embedded in the page.

| Column | Type | Unit | Content | Null when |
| --- | --- | --- | --- | --- |
| `round` | string | — | The round. | never |
| `step_index` | int64 | — | 0 for the loaded and rotated input; then 1, 2, … for the round's recorded entries in the order `FOV.run` runs them: the `steps` entries, then the `post_registration` entries. | never |
| `phase` | string | — | `input`, `steps` or `post_registration` (the entry's `stage`). | never |
| `step` | string | — | `input` or the entry's `step`. | never |
| `channel` | string | — | The channel label. | never |
| `bin` | int64 | — | Zero-based bin index. | never |
| `bin_low`, `bin_high` | float64 | grey levels of the step's output dtype | The bin's edges (their rule is on the algorithms page). | never |
| `count` | int64 | voxels | The voxels in the bin. | never |

**The null rules of the rows of the phase `post_registration`.** No cell of these rows is
null: a distribution the replay cannot rebuild has no row, not a row of null cells. On a
moving round whose transforms cannot be rebuilt ("The replay", "A moving round whose
transforms cannot be rebuilt"), the rows of its `post_registration` entries are not
written, and one row of `unavailable.csv` without a table names them: part
`preprocessing.intensity`, row key
`fov_id=<main example>;role=example_1;round=<round>;phase=post_registration`, with the
reason (`step_failed`, `not_recorded`, `field_not_saved`, `input_missing` or
`input_changed`); the round's rows of the phases `input` and `steps` are written. A round
without `post_registration` entries has no row of that phase and needs no transform; the
reference round, and a moving round of a run without a registration, have their rows of
that phase without a transform.

**`unavailable.csv`**: one row per unavailable value ("Where a reason is shown and
recorded"): each unavailable cell of a report table, each unavailable key of `report.json`
`summary`, and each part, or part of a part, with no cell for what it lacks. Order:
`REPORT_PARTS` order, then the tables in the order of this section with `summary` last and
rows without a table first, then the column order of the table, then its row order; key
(`part`, `table`, `column`, `row_key`). A cell is named by one row at most, so two values of
one part, FOV and round are told apart by their column, and two values of one column by
their row key.

| Column | Type | Unit | Content | Null when |
| --- | --- | --- | --- | --- |
| `part` | string | — | One of `REPORT_PARTS`, or `rule:<rule>` for a rule of `PROBLEM_RULES` not evaluated for the whole sample. | never |
| `table` | string | — | The table of the cell: `fovs`, `registration`, `steps`, `readout_totals`, `population`, `mecr`, or `summary` for a key of `report.json` `summary`. | n/a: a row about a part with no cell |
| `column` | string | — | The cell's column, or the `summary` key. | n/a: as `table` |
| `row_key` | string | — | The cell's row key (`fov_id=Position001`; `fov_id=Position001;round=round2`; `scope=fov;fov_id=Position001;index=4`; `level=correct_form`; `population=excluded_not_owner`; `gene_a=Cd3e;gene_b=Cd79a`); empty for a `summary` key. For a row without a table, the FOV, round, example `role`, `phase`, `view` or Files `entry` the part is drawn for (`fov_id=Position031;role=example_1`, `fov_id=Position031;role=example_1;round=round2;phase=post_registration`, `view=xz`, `entry=positions`), or empty for the whole part. | never |
| `reason` | string | — | One of `UNAVAILABLE_REASONS`. | never |
| `detail` | string | — | The file, key, step or part. | n/a: nothing to add |

**`samples.csv`** (the comparison page): one row per sample of `samples`, in that order;
key `sample`.

| Column | Type | Unit | Content | Null when |
| --- | --- | --- | --- | --- |
| `sample` | string | — | The sample. | never |
| `report` | string | — | `present` or `missing`. | never |
| `report_sha256` | string | — | The SHA-256 of its `report.json`. | n/a: `report` is `missing` |
| `generated_at` | string | ISO 8601, UTC | `report.json` `generated_at`. | n/a: `report` is `missing` |
| `route` | string | — | `report.json` `route`. | n/a: `report` is `missing`, or the sample has no export |
| `fovs_expected` | int64 | FOVs | `report.json` `summary`. | n/a: `report` is `missing` |
| `fovs_succeeded` | int64 | FOVs | `report.json` `summary`. | n/a: `report` is `missing`; unavailable in the sample, named by the sample's own `unavailable.csv` (`table` `summary`), and `samples.csv` repeats no row |
| `spots_median_per_fov` | float64 | reads | `summary.spots_median_per_fov`. | n/a: `report` is `missing`; unavailable in the sample, named by the sample's own `unavailable.csv` (`table` `summary`), and `samples.csv` repeats no row |
| `matched_fraction` | float64 | fraction | `summary.matched_fraction`. | n/a: `report` is `missing`; unavailable in the sample, named by the sample's own `unavailable.csv` (`table` `summary`), and `samples.csv` repeats no row |
| `counted_cells` | int64 | cells | `summary.counted_cells`. | n/a: `report` is `missing`; unavailable in the sample, named by the sample's own `unavailable.csv` (`table` `summary`), and `samples.csv` repeats no row |
| `cells_median_per_fov` | float64 | cells | `summary.cells_median_per_fov`. | n/a: `report` is `missing`; unavailable in the sample, named by the sample's own `unavailable.csv` (`table` `summary`), and `samples.csv` repeats no row |

The three box plots read each present sample's `tables/fovs.csv` (`spots`,
`matched_fraction`, `n_cells_retained`), checked against the SHA-256 its `report.json`
lists.

### `report.json`

Strict JSON (non-finite numbers as `null`), written last.

| Key | Content |
| --- | --- |
| `format_version` | `1`. |
| `stage` | `report`. |
| `sample`, `dataset_id`, `sample_id` | The sample; the `Dataset`'s IDs. |
| `fov_ids` | The FOV list. |
| `route` | `per_fov`, `after_fusion`, or null without an export; `per_fov` for an export whose assembly records no route ("The two routes", choice 20). |
| `assignment`, `segmentation` | The entry's arguments, with the runs they resolved to. |
| `generated_at` | UTC time the report was written. |
| `software` | `code` and `environment`, as `run.json` records them (`dataset/_run_record.py`: `_code`, `_environment`). |
| `config` | The `ReportConfig`, as JSON. |
| `stages` | What existed: the FOVs with a run record, `placement` (boolean), `export` (boolean) and the export's `parts`. |
| `inputs` | Every file read: `path` (relative to the output root, absolute outside it), `kind` (`run_record`, `transforms`, `field`, `raw_image`, `supplied_statistics`, `codebook`, `segmentation_record`, `assignment_record`, `label_image`, `reference_image`, `placement`, `placement_table`, `export_record`, `export_table`, `anndata`, `fused_image`), `sha256`, `bytes`, and `verified` (whether a record's SHA-256 was compared). |
| `files` | Every file of the folder but `report.json`: `path` (relative to the folder), `sha256`, `bytes`, and for a table its `dtypes`. The report's own files are `report.json` and the files its `files` list names. An existing output folder is replaced only when it holds no file other than the report's own; a file that is not one of them, which includes every file of a nonempty folder without `report.json`, stops the entry with `FileExistsError` naming that file before anything is written, and an empty folder is used as it is. |
| `unavailable` | The rows of `tables/unavailable.csv`, each an object of its columns (`part`, `table`, `column`, `row_key`, `reason`, `detail`). |
| `examples` | "The example FOVs". |
| `replay` | "The replay". |
| `rules` | Per rule of `PROBLEM_RULES`: the limit applied (`k`, `floor`, `absolute`), the unit, whether it was evaluated, and the number of FOVs it marks. |
| `summary` | The key numbers: `fovs_expected`, `fovs_with_run_record`, `fovs_succeeded`, `spots_median_per_fov`, `matched_fraction` (the sum of `matched` over the sum of `spots`), `counted_cells`, `reads_per_cell_median`, `genes_per_cell_median`, `cells_median_per_fov`, `reads_in_cells`, `h5ad_whole_count_sum` and `reads_in_cells_equal` (whether the two are equal); each key null when unavailable, and then named by a row of `unavailable` with `table` `summary` and `column` the key (`reads_in_cells_equal` with the reason of the value it lacks). |
| `wall_s`, `cpu_s`, `peak_rss_kib` | The entry's own cost, as `export.json` records it for a part. |

A `report.json` holds no verdict, no score and no key that grades the sample.

### The data embedded in the page

`report.html` holds one `<script type="application/json" id="report-data">` block: an object
whose keys are the tables `fovs`, `registration`, `steps`, `readout_totals`, `population`,
`problematic_fovs`, `mecr`, `reads_along_z` and `unavailable`, each a list of rows in the
table's order, each row an object of its columns. Read back with the table's dtypes, every
embedded table equals its saved CSV, cell by cell, with null for a missing value; the page's
maps, tooltips and paged tables read only this block. `cells.csv` and `intensity.csv` are
saved but not embedded: their figures are static images. The comparison page embeds
`samples.csv` and the per-FOV values of its box plots the same way. The page loads no
external resource.

## Records added to `run.json`

One implementation issue, the first of the implementation batch (D6), adds them to
`dataset/_run_record.py` (`_RunRecord.__init__` and `_RunRecord.write`) and to the
workflow adapter. Each is an additive key: an older `run.json` without it is still read by
the report, which treats the value as unavailable with reason `not_recorded` (R1 first
reads `registered/transforms.json`, "Older files"). No package code reads `run.json` at
`dc5fc3c` (searched in `src/python/starfinder/`), so no other reader changes. The golden
test `src/python/test/test_report_records_golden.py` (this issue) pins every key and value
a small synthetic FOV writes today; the additions pass it, and a removed key or a changed
value fails it.

### R1. The transforms of every round (`transforms`)

Today `run.json` has no transforms. The estimation attempts' QC records hold a summary of
each step's transform (`registration.<round>[i].qc.details.transform`, written by
`evaluation/registration.py`: `_transform_summary` through `dataset/fov.py`: `_qc_record`):
a translation's `displacement_zyx`, an affine's 3 × 3 `matrix`, `offset` and `det`, a dense
or B-spline step's displacement statistics ({doc}`report-baseline`, G2). The full
transforms are written only by the stage `registered` (`registered/transforms.json`,
`io/_checkpoint.py`: `write_registered_header`).

```json
"transforms": {
  "round2": {
    "reference": "round1",
    "results": [
      {"step": 0,
       "transform": {"reference_shape_zyx": [5, 36, 36], "moving_shape_zyx": [5, 36, 36],
                     "reference_metadata": {"…": "…"}, "moving_metadata": {"…": "…"},
                     "direction": "…", "units": "…", "kind": "translation",
                     "displacement_zyx": [0.0, 0.0, 1.0]},
       "diagnostics": {"method": "translation", "backend": "scipy_fft", "…": "…"}}
    ],
    "application_config": {"backend": "translation", "boundary_mode": "constant", "fill_value": 0,
                           "output_dtype": "input", "integer_rounding": "nearest_even",
                           "clip_to_dtype": true, "fft_workers": 1}
  }
}
```

Each entry of `results` is the entry `registered/transforms.json` holds for the same result:
`io/_checkpoint.py`: `_transform_json(result, i, field_name)`, which writes `step` (the
result's index `i` in the round's results; the one place that index is called `result` is
`run.json` `preprocessing.transforms`, written by `FOV._record_transforms`), `transform`
and `diagnostics`, and no `result` key. The only difference is the field file's name:
`write_registered_header` always passes `<round>_field.npz`, which it writes; R1 passes it
only when the stage `registered` is on, and null otherwise, so that R1 never names a file
that does not exist.

| Field | Type | Unit | Content |
| --- | --- | --- | --- |
| `transforms` | object | — | One entry per moving round that `FOV.run` registered, keyed by round, in processing order; `{}` when nothing was registered. The reference round has no entry. A round whose registration failed has no entry (its attempts stay in `registration`). |
| `.reference` | string | — | The reference round. |
| `.results` | array | — | The round's results in order, each `{step, transform, diagnostics}` as `_transform_json` writes it. |
| `.results[*].step` | int | — | The result's index in the round's results (`_transform_json`'s `step`). |
| `.results[*].transform.kind` | string | — | `translation`, `affine`, `bspline` or `dense` (`registration/_chain.py`: `transform_kind`). `translation` and `affine` are the global kinds, whatever the method that estimated them. |
| `.results[*].transform.reference_shape_zyx`, `.moving_shape_zyx` | array of 3 ints | voxels | The grids' shapes. |
| `.results[*].transform.reference_metadata`, `.moving_metadata`, `.direction`, `.units` | object; string | — | As `transforms.json` holds them. |
| `.results[*].transform.displacement_zyx` | array of 3 floats | voxels of the reference grid | A translation: `TranslationTransform.displacement_zyx`, the convention of {doc}`registration-contract`. |
| `.results[*].transform.matrix_zyx`, `.physical` | 4 × 4 array of floats; object or null | index space (voxels); the backend's physical parameters | An affine. |
| `.results[*].transform.bspline` | object | ITK XYZ order; `spacing_zyx` µm | A B-spline's control grid: `grid_size_xyz`, `grid_origin_xyz`, `grid_spacing_xyz`, `grid_direction_xyz`, `order`, `spacing_zyx` (`_BSPLINE_FIELDS`). Always written: it is small, and the coefficients alone cannot be applied without it. |
| `.results[*].transform.coefficients` | string or null | — | A B-spline: `<round>_field.npz` when the stage `registered` wrote it (array `result_<i>`), else null. The coefficients themselves are not in `run.json` (D6). |
| `.results[*].transform.field` | string or null | — | A dense result: `<round>_field.npz` when the stage `registered` wrote it (array `result_<i>`), else null. The field is not in `run.json` (D6). |
| `.results[*].diagnostics` | object | — | The result's `RegistrationDiagnostics`, which `_restore_result` needs. |
| `.application_config` | object | — | The round's `WarpConfig` of its one resampling, as `transforms.json` `applications.<round>.application_config` holds it. |

Written by `_RunRecord.write` from `FOV.registration_results` and
`FOV.registration_record["application"]`, so the record is rewritten after each registered
round, as the attempts are. JSON floats round-trip exactly, so the replay rebuilds the
same chain as `read_checkpoint(…, "registered")` does (W-384: bit-identical).

| Option | Form | Note |
| --- | --- | --- |
| **T1 (recommended)** | The top-level key above | One place, the entries of `transforms.json`, independent of the QC. |
| T2 | A `transform` key in each succeeded estimation attempt of `registration` | The attempts are a log of tries; the chain would be rebuilt by picking attempts. |
| T3 | The stage `registered` writes `transforms.json` always, without the images | Changes what a stage writes and the files `CheckpointConfig.stages` names. |

### R2. The codebook file (`codebook`)

Today no record names it: `Dataset.load_codebook` (`dataset/dataset.py`) passes the path
to `barcode.load_codebook` and keeps only the `Codebook`; `run.json` `config.encoding`
holds the encoding ({doc}`checkpoints`: "The codebook file is not recorded in `inputs`").

| Field | Type | Unit | Content |
| --- | --- | --- | --- |
| `codebook` | object or null | — | Null when the dataset has neither a codebook nor a direct panel. |
| `.kind` | string | — | `codebook` (readout mode `multiplexed`) or `direct_panel` (`direct`). |
| `.path` | string or null | — | The absolute path `Dataset.load_codebook` or `Dataset.load_direct_panel` read; null for a codebook set on the `Dataset` without a file. |
| `.sha256` | string or null | — | The file's SHA-256 when it was read; null with `path`. |
| `.entries`, `.genes` | int | entries; genes | The codebook's entries and distinct genes. |

`Dataset.load_codebook` and `load_direct_panel` keep the path and SHA-256 in a private
attribute that `_RunRecord.__init__` reads.

| Option | Form | Note |
| --- | --- | --- |
| **C1 (recommended)** | The top-level key above | Beside `inputs`, with the same path and SHA-256 form. |
| C2 | `config.codebook` | `config` holds configurations, not files read. |
| C3 | An entry of `inputs` | `inputs` lists the loaded images; the golden test pins its length. |

### R3. The correct-form count (`counts.filtering.endpoint_valid`)

Today `counts.filtering` holds `total`, `accepted`, `rejected` (`ReadFilteringResult.counts`,
copied by `_RunRecord.write`); the count is the filter's
`diagnostics["endpoint_valid_count"]` (`barcode/filtering.py`: `filter_reads`), not saved.

| Field | Type | Unit | Content |
| --- | --- | --- | --- |
| `counts.filtering.endpoint_valid` | int | reads | The reads whose every segment with ends starts and ends with a declared pair: `diagnostics["endpoint_valid_count"]`. Present exactly when the filter checked endpoints (`end_bases`, or a codebook layout with segment ends). |
| `counts.filtering.endpoint_valid_segments` | object of int | reads | Per segment with ends, `diagnostics["endpoint_valid_segment_counts"]`. Present with a layout's segment ends. |

The denominator is `counts.spots` on the page (D5: every read level as a fraction of the
spots detected).

| Option | Form | Note |
| --- | --- | --- |
| **F1 (recommended)** | The two keys in `counts.filtering` | Beside the counts of the same filter. |
| F2 | A new `counts.endpoint` section | A second section for one filter. |
| F3 | The filter's whole `diagnostics` and `fractions` in `counts.filtering` | Copies values the report does not use and whose form is not fixed as a record. |

A record without the key: `not_configured` when `config.pipeline.filtering` shows no
endpoint check (no `end_bases`, and `pre_qc.json` `layout` or the codebook record shows no
segment ends), else `not_recorded`.

### R4. The workflow configuration file (`workflow_config`)

| Field | Type | Unit | Content |
| --- | --- | --- | --- |
| `workflow_config` | object or null | — | `{"path", "sha256"}` of the workflow configuration file a workflow rule ran with (the top-level `config_path` key, {doc}`workflow-configuration`); null for a run outside the workflow. |

`from_workflow_config` (`dataset/workflow.py`) keeps the path on the `Dataset` it builds,
in a private attribute that `_RunRecord.__init__` reads. Options: **W1 (recommended)**, this
key; W2, the report takes the file from its own configuration (it would show the file the
user names, not the one the run used); W3, unavailable.

### R5. The workflow writes `run.json`

Today `_run_workflow` (`dataset/workflow.py`) calls `fov.run(adapted.pipeline,
execution=adapted.execution)` without a `CheckpointConfig`, so a workflow run has no run
record ({doc}`report-baseline`, G1). With K1 the processing rules that call
`_run_workflow` pass `CheckpointConfig(stages=<stages>, overwrite=True)`, with the stages
from a Python-only key `rules.<rule>.parameters.checkpoints` of the form of
`rules.nuclei_registration.parameters.checkpoints` (`_nuclei_checkpoints`), whose default
is `{stages: []}`: `run.json` alone, about 43 kB for an LN FOV with its registration
attempts (W-384 `real_fovs.csv`). `overwrite=True` because a rerun of the rule replaces its
record; it removes only the stage files and leaves `other_rounds/`, `segmentation/` and
`assignment/` ({doc}`checkpoints`, "Layout"). `hash_inputs` stays true, so that the replay
can verify the raw input; it reads each input once more. The logs the rules write stay as
they are. `run.json` is not a declared output of the rules. Options: **K1 (recommended)**;
K2, the report reads the three logs as a second source (baseline G1, option a); K3, both.

### Format version

| Option | `run.json` `format_version` | Note |
| --- | --- | --- |
| **V1 (recommended)** | Stays `1` | The five additions are keys; the earlier additive key `channels` (W-336) kept version 1 too, and the golden test pins `format_version` 1. |
| V2 | `2` | Tells a reader that the fields exist, but no reader checks the version, and the golden test pins 1 and must then change. |

### Older files

A `run.json` without one of the keys is read as it is. R1: the report reads
`registered/transforms.json` `transforms.<round>` and `applications.<round>` when the FOV has
it (written by `write_registered_header`; the same entries, `_transform_json`; choice 11),
otherwise the shifts and the registration example are unavailable with `not_recorded`. R2:
the codebook file and the marker check fall back to the export's gene list. R3: as stated
above. R4: `not_recorded`.

| Option | Shifts of a run record without R1 | Note |
| --- | --- | --- |
| **OF1 (recommended)** | `registered/transforms.json` when it exists, else unavailable (`not_recorded`) | The same entries as R1; nothing is read that is not a record of the transforms. |
| OF2 | As OF1, and else the translation or affine summary of the estimation attempts' QC records (`registration.<round>[i].qc.details.transform`, `evaluation/registration.py`: `_transform_summary`) | Shows shifts for more older runs, but the summary is a QC value whose form no contract fixes as a record, it is null for an attempt that failed before its QC, and an affine's 3 × 3 `matrix` and `offset` are not the 4 × 4 index matrix the chain is built from. |
| OF3 | Unavailable for every run record without R1 (leave it as it is) | Simplest; costs the shifts of every run saved with the stage `registered` before R1. |

## `docs/migration.md` entries

* **New: the processing report.** `Dataset.report(sample, fov_ids, …)` writes
  `<output_root>/report/<sample>/` (`report.html`, `report.json`, `tables/`), and
  `Dataset.compare_samples(samples)` writes `report/index.html` and `report/samples.csv`;
  the workflow's `report` block adds the rules `report_sample` and `report_dataset`.
  Nothing existing changes.
* **`run.json` gains `transforms`, `codebook`, `workflow_config` and
  `counts.filtering.endpoint_valid`** (with `endpoint_valid_segments`); `format_version`
  stays 1, and a `run.json` without them still loads.
* **The workflow writes `checkpoints/<fov>/run.json`** for the processing rules, alone by
  default, with `overwrite` on; `rules.<rule>.parameters.checkpoints` adds stages.
* **`Dataset.load_codebook` and `load_direct_panel` keep the file's path and SHA-256.**
* **`workflow/notebook/create_rsf_qc_report.py.ipynb` is superseded** by the report; it is
  left in place.

## Tests the implementation keeps and adds

Kept unchanged: `src/python/test/test_report_records_golden.py` (this issue; every pinned
key and value of `run.json` and every checkpoint file stays, so the additions pass it);
the checkpoint, workflow, stitching and export tests. One existing test changes (choice
17): `test_checkpoints.py::test_run_record_fields_inputs_and_hashes` asserts the exact set
of top-level keys of `run.json`.

| Option | The exact-key-set test (choice 17) | Note |
| --- | --- | --- |
| **TC1 (recommended)** | It becomes "at least these keys", with `transforms`, `codebook` and `workflow_config` listed | A later additive key passes it, as it passes the golden test. |
| TC2 | It stays an exact set, with the three keys added to it | Every later additive key edits it again. |
| TC3 | Leave it as it is | The implementation of R1, R2 and R4 then fails the routine gate. |

Added (the numbered validation table with known answers is on the algorithms page):

* the records R1 to R4 on a synthetic FOV: a translation and an affine step with and
  without the stage `registered`, a dense step with its `field` reference and without;
  `endpoint_valid` equal to the filter's diagnostic; the codebook's path and SHA-256;
  `workflow_config` from the workflow adapter, null outside it; R5: a workflow run writes
  `run.json` and a rerun replaces it;
* an older `run.json` (the golden test's form) gives `not_recorded` for each addition, and
  the shifts from `registered/transforms.json` when it exists;
* the entry's checks 1 to 7, each raising with the folder unchanged, among them a marker
  gene not in the codebook;
* every null cell of a report table is either n/a by its column's null rule or named by
  exactly one row of `unavailable.csv`, and no row names two values (the cases the W-385
  repair walked: an older `run.json`, an undefined MECR pair beside a defined one, a FOV
  without a run record, a failed round, a run without `registered`, an export without run
  records, the stages before and after segmentation on both routes, no placement, no
  markers, one FOV, `Z=1`);
* the closed lists: every reason in `unavailable.csv` and `report.json` is one of
  `UNAVAILABLE_REASONS`, every part one of `REPORT_PARTS`;
* every table's columns, dtypes, row order and key uniqueness; the data embedded in the
  page equal to the saved tables; the page with no external resource;
* the stages of "The report at each stage of a sample's processing", each with the parts
  it states present and unavailable; both routes; an export without run records (A2);
  a `molecules_only` export;
* the replay bit-identical to the saved registered images of a synthetic FOV;
  `field_not_saved`; `input_changed`; `replay_differs`;
* the reads assigned to cells equal to the sum of `sample.h5ad`'s whole counts;
* the comparison page with a missing sample;
* no key of `report.json` or column of a table is a verdict or a score.

## Open points for the algorithms page

The algorithms page (W-386) states: which FOVs are "near the median read count", the
draw with seed 0, the ties and the coincidence of two example roles; the median, MAD and
sides of each problematic-FOV rule and which FOVs enter them; the shift of an affine step;
the geometry of the region a FOV owns and its area; the rounds and signals the examples
show; the bins of the intensity distributions; how a molecule's `z` becomes a plane; the
layout of every section and figure; the figure sizes; and the numbered validation table.

## Exclusions

No implementation, template or change of `run.json` or of a checkpoint in this issue; no
algorithm, layout or validation design (W-386); no new dependency and no lock change; no
real data; nothing here marks a page as accepted. Not read by the report: the comparison of
the two routes, the earlier pipeline's logs, the registered images. Not in the first
delivery: the items listed under "Settled decisions this page follows".
