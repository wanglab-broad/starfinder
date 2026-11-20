# Processing-report reference

Use the {doc}`guide <report-contract>` to make a sample report and a comparison
page, and the {doc}`API <api/report>` for configuration details.

## Configuration

These tables read the field names, defaults and descriptions from the configuration classes.

### Report

{py:class}`~starfinder.report.ReportConfig`

```{config-options} starfinder.report.ReportConfig
```

### Marker group

{py:class}`~starfinder.report.MarkerGroup`

```{config-options} starfinder.report.MarkerGroup
```

### Problematic-FOV limits

{py:class}`~starfinder.report.ReportLimits`

```{config-options} starfinder.report.ReportLimits
```

{py:class}`~starfinder.report.RuleLimit`

```{config-options} starfinder.report.RuleLimit
```

## Page sections

The sample page groups its displays into these sections.

| Section | What it shows |
| --- | --- |
| Summary | Sample identity, acquisition information, key numbers and problematic FOVs |
| Workflow overview | A card for every step the run executed, in execution order, each opening on its parameters; FOV status and time per step |
| Preprocessing | Main example before and after preprocessing, and intensity distributions |
| Registration | Shift maps, residual heatmap and the main example's alignment |
| Readout | Totals, counts and fractions per FOV, distributions and a FOV table |
| Segmentation and assembly | Cell counts and density, placement (with the link-quality threshold and the links the optimizer removed, when the placement records them), segmentation examples and population accounting |
| Sample measurements | Reads, genes and size per cell, marker patterns, MECR and reads along Z |
| Files | Input, configuration (the workflow configuration file with its hash) and output records |

The Summary's acquisition counts the rounds of the run records' `channels`. A reference stain taken in the reference
round's acquisition (`reference_stain` with no registration entry in the run record, as `prepare_morphology` records it)
counts with that round and adds its channels to the round's; a stain registered as an image of its own stays a round.
Time per step shows every step the records hold, `segment`, `assign_molecules`, `stitch` and `assemble` included.

The Workflow overview has a card for every stage the run records, in the order the steps run: on the "Per FOV" lane
Preprocessing, Registration, Post-registration preprocessing, Spot finding, Read extraction, Decoding, Scoring,
Deduplication and Read QC filter, then Segmentation and Assignment on the route per FOV; on the "Per sample" lane
Stitching, Segmentation (route after fusion), Assembly and Export. Post-registration preprocessing lists the methods
of `preprocessing.post_registration` and reads "not run" when the configuration has none; a stage the configuration
leaves out reads "not run", and a stage with no record yet "not yet run".

Clicking a card opens its parameters, read from `workflow/resolved.json`, the file whose hash the run records hold:

* the method, then the values that differ from the defaults of the configuration dataclass the block names (the `type`
  of each block in `configuration.api`), which are the only values listed; a step whose values are all defaults says
  so;
* the fitted values of the example FOV, where the step has them, each with a link to the section and the table that
  show them: the minimum and maximum per channel of a normalization (Preprocessing), the shifts of each round
  (Registration, `registration.csv`) and the number and quality of the links (Stitching, `links.csv` of the
  placement);
* the whole block behind "all parameters", and the file the values come from. When the file's hash is not the one the
  FOVs wrote (a task other than a FOV's changed it), the panel says so and shows the file's values.

The Files section keeps the path and hash of `resolved.json`. A run outside the workflow, or a `resolved.json` that is
missing or is not a resolved configuration, has cards without parameters; the Files table's Parameters row says why.

The comparison page has a sample table and box plots of spots per FOV, fraction
matched to the codebook, and cells per FOV.

## Tables

The sample's CSV files live in `report/<sample>/tables/`. `samples.csv` lives
beside the comparison page in `report/`.

| Table | Contents |
| --- | --- |
| `fovs.csv` | FOV identity, run status, placement, read counts and fractions, cells and density |
| `registration.csv` | Registration outcome, fallback, methods, shifts, residuals and alignment score |
| `steps.csv` | Per-FOV and sample steps, methods, status and duration; the durations of `segment`, `assign_molecules`, `stitch` and the export parts are the `wall_s` of their records |
| `readout_totals.csv` | Counts and fractions at the readout levels |
| `population.csv` | Input, retained, excluded and counted cells and molecules |
| `problematic_fovs.csv` | FOVs to inspect, with rule, measurement, limit and reason |
| `cells.csv` | Reads, genes and size of counted cells |
| `mecr.csv` | Positive-cell counts and MECR for marker pairs |
| `reads_along_z.csv` | Read counts by FOV and Z plane |
| `intensity.csv` | Binned channel intensities from the main example's replay |
| `unavailable.csv` | The part or table value affected, its reason and details |
| `samples.csv` | Sample report presence, route, summary values and record hash |

### Columns

Column names below come from the saved-table schemas in the code.
`report.json` records the sample tables' dtypes.

```{report-tables}
```

## Problematic FOVs

The list draws attention to these conditions. It gives no composite score.
The [rule definitions](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/report-algorithms.md#problematic-fovs)
give the calculations and eligibility details.

| Rule | What it draws attention to |
| --- | --- |
| `fov_failed` | Failed or interrupted processing |
| `fov_not_run` | An expected FOV without a run record |
| `shift_residual` | A shift far from the sample's usual shift |
| `alignment_score` | A low alignment score (off by default) |
| `registration_rejected` | A rejected registration attempt |
| `registration_fallback` | Successful registration that used fallback |
| `correct_form_fraction` | A low fraction of correct-form reads |
| `matched_fraction` | A low fraction matched to the codebook |
| `spots_near_zero` | A spot count close to zero relative to the sample |
| `not_placed` | A FOV missing from placement or whose placement failed |
| `stage_position_fallback` | Placement that fell back to the stage position |
| `reads_no_cell` | An owning FOV with retained molecules but no retained cell |

## Figures

The page embeds its figures, so viewing them needs no image server.

| Display | What to look at |
| --- | --- |
| FOV maps | Status, shifts, read counts and fractions, cell counts and density in sample space |
| Time per step | The contributions of FOV and sample processing steps |
| Preprocessing example | Channels and signal before and after the recorded steps |
| Intensity distributions | Channel intensities at stages of the replay |
| Registration example | Reference and moving signal before and after alignment, with a zoom |
| Shift residual heatmap | FOVs and rounds with shifts different from the sample median |
| Readout distributions | Counts and fractions across FOVs |
| Segmentation examples | Image panels with segmentation outlines |
| Per-cell histograms | Reads, genes and cell size |
| Marker patterns | Whole-sample patterns and example regions for the marker groups |
| Reads along Z | Counts by depth and a FOV-by-depth heatmap |
| Comparison box plots | Distributions of the three comparison measurements by sample |

Full [layout and measurement specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/report-algorithms.md).
