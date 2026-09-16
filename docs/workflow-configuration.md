# Configuration reference

The authoritative validation source is
[config.schema.yaml](https://github.com/wanglab-broad/starfinder/blob/dev/workflow/schemas/config.schema.yaml).
The Snakefile calls `snakemake.utils.validate`, which inserts schema defaults.
A plain JSON Schema validator checks constraints but does not insert defaults.
The tables distinguish schema requirements from values the rule code needs.

## Top-level fields

An em dash in the default column means the schema defines no default.

| Field(s) | Type / constraint | Required by schema | Default / runtime meaning |
| --- | --- | --- | --- |
| `config_path` | string | Yes | —; absolute path to the same `.yaml` passed to `--configfile`; conversion writes a sibling JSON |
| `starfinder_path` | string | Yes | —; repository root used to locate Python/MATLAB code |
| `root_input_path`, `root_output_path` | strings | Yes | —; roots before dataset/sample or dataset/output components |
| `envs_path`, `fiji_path` | strings | No | —; **needed at parse time**, because included StarDist/Fiji rules index them even if disabled |
| `dataset_id`, `sample_id`, `output_id` | strings | Yes | —; directory identifiers |
| `fov_id_pattern` | string | Yes | —; Python format string with `{i}`, e.g. `tile_{i}`, `Position{i:03}` |
| `n_fovs`, `n_rounds` | integers >=1 | Yes | —; FOV indices start at 1, round names are exactly `round1` … `roundN` |
| `ref_round` | string | Yes | —; must identify an existing sequencing round; membership is not schema-checked |
| `ref_channel`, `dapi_round` | strings | No | —; nuclei registration / DAPI input selection, not Python sequencing-channel selection |
| `rotate_angle` | number | Yes | —; degrees, 0 for no rotation |
| `img_col`, `img_row` | integers >=1 | Yes | —; width X and height Y; subtile windows use these values |
| `img_z` | integer >=1 | No | —; downstream 3-D metadata/preview uses it |
| `voxel_size_xy`, `voxel_size_z` | numbers >0 | No | —; physical microns for stitching; **not** extraction window radii |
| `maximum_projection` | boolean | No | `false`; reference-image/DAPI output projection; does not turn sequencing into a 2-D algorithm |
| `seq_channel_order` | array of strings | No | —; set explicit patterns for Python, e.g. `[ch00, ch02, ch01, ch03]` |
| `additional_round` | array of objects with string `round_name` | No | —; supply `[]` when unused; accessed at parse time |
| `backend` | `python` or `matlab` | No | `matlab` |
| `workflow_mode` | `free`, `direct`, `subtile`, `deep` | No | `free`; preset selection described in [workflows](workflows.md#modes-and-backend-selection) |
| `subset_list` | array of integers >=1 | No | —; supply `[]` when unused; code indexes it directly |
| `subset_range` | boolean | No | `false` |
| `subset_start`, `subset_end` | integers >=1 | Conditional | —; see caveat below |
| `subset_random` | boolean | No | `false` |
| `n_random_tests` | integer >=1 | Conditional | —; see caveat below |
| `rules` | object of recognized rule names | Yes | —; each provided section requires boolean `run` |

The schema's `dependentRequired` checks **presence**, not truth: if
`subset_range` is present (even false), `subset_start` and `subset_end` must be
present; likewise `subset_random` requires `n_random_tests`. The `if` clauses
also lack an explicit presence test. Supply all five fields as in the examples
rather than relying on omission/defaults. The original minimal template fails
these conditional requirements.

Subsetting precedence is nonempty `subset_list`, then true `subset_range`
(inclusive), then true `subset_random`, else all `1..n_fovs`. The list path
updates the sample list through annotation lookup; range/random paths do not.
Random selection calls Python's unseeded `random.sample`; use explicit indices
for a reproducible subset. Bounds, duplicate indices, reversed ranges and
annotation coverage are not validated by the schema. Sample aggregation still
expects every FOV in each annotation range, even when sequencing is subsetted.

Python arrays are `(Z,Y,X,C)`. Explicit channel patterns must match the files
and codebook ordering. The Python factory leaves an empty channel list empty;
it does not supply MATLAB's default. MATLAB's default order is
`ch00,ch02,ch01,ch03`, but its nonempty custom channel setting is passed to a
loader expecting a struct array with `channel`/`name`, incompatible with this
schema's string array. Use `seq_channel_order: []` for default MATLAB loading;
do not assume a Python custom list is portable to MATLAB.

## Rule resources and parameters

Each `rules.<name>` requires `run: true|false`. `resources` is optional; if
present, it accepts only `mem_mb` (integer >=1000, default 8000) and `runtime`
(integer >=1 minute, default 30). `get_rule_config` also supplies these fallbacks
when the resource object is absent. There is no YAML `threads` resource: core
FOV/subtile rules declare 4 threads, StarDist and sequential MATLAB declare 2,
BigStitcher declares 4, and others default to 1. Snakemake can scale threads down
to local `--cores`; use `--set-threads` / `--set-resources` for deliberate CLI
resource overrides. These do not alter algorithm settings.

GR/subtile creation and per-subtile RSF rules multiply configured runtime by
attempt number. Direct RSF, stitch_subtile, StarDist, sequential RSF and
BigStitcher use fixed configured runtime. Several auxiliary rules declare only
memory, so adding a YAML runtime to those sections has no effect;
`create_sample_maf` fixes memory/runtime to 8000 MB / 2 minutes in code.

The following keys sit under `rules.<rule>.parameters`. Unless stated otherwise,
there is no schema default. Optional blocks are not a promise that every script
can omit them: MATLAB scripts often index fields directly. The minimal example
is specifically for the Python batch/direct wrapper.

| Block | Schema fields and constraints | Wrapper behavior / defaults |
| --- | --- | --- |
| `enhance_contrast` | boolean `run` | Python direct/GR batch calls enhancement only when true; optional top-level parameter `snr_threshold` is forwarded; MATLAB uses min-max enhancement |
| `hist_equalize` | boolean `run`, integer `reference_channel` | Python direct/GR/deep uses channel index 0 by default; MATLAB deep passes the configured 1-based index, direct uses its method default |
| `morph_recon` | boolean `run`, integer `radius` >=1 | Python default radius 3; used in direct/GR and subtile processing, not deep creation |
| `global_registration` | boolean `run`, string `ref_round`, `ref_img`/`mov_img` in `merged-image`,`single-channel` | Python default image modes are merged; Python uses dataset top-level `ref_round`, not this block's reference; MATLAB wrappers pass the block reference; MATLAB deep uses scale 0.25 |
| `create_subtiles` | boolean `run`, integer `sqrt_pieces` >=1 | Grid default 4; only GR/deep creation rules produce subtile files; Python creation scripts call splitting unconditionally |
| `local_registration` | boolean `run`, string `ref_round`, method `demons`,`tps`,`cpd` (schema default `demons`) | Python direct/local-subtile wrappers forward method only; demons needs optional SimpleITK; MATLAB wrappers do not forward `method`; deep subtile does not perform local registration |
| `spot_finding` | boolean `run`, string `ref_round`, nonnegative numeric `intensity_threshold`, mode `local`,`global`,`noise`,`adaptive`,`adaptive_round` | Python wrappers default mode to `noise` but require a threshold when used; pass both explicitly. `local` is schema-accepted but unsupported by Python detector; MATLAB supports adaptive/global only |
| `load_codebook` | boolean `run`, integer-array `split_index` | Python wrappers load unconditionally, turn missing/empty split into None; MATLAB respects `run` |
| `reads_extraction` | boolean `run`, exactly three integers >=1 in `voxel_size` | Pixel half-widths, Python `(z,y,x)` versus MATLAB `(row,column,z)`; e.g. `[1,2,2]` versus `[2,2,1]`, not physical microns |
| `reads_filtration` | boolean `run`, string or string-array `end_base`, integer `n_barcode_segments` >=1, integer-array `split_index` | Python wrappers forward `end_base` and extra `start_base` (default `C`), not `n_barcode_segments` or this block's split; MATLAB forwards segment count and split |

Always pair `intensity_estimation` with `intensity_threshold`. Python `noise`
uses a k-sigma threshold (e.g. 5), whereas `adaptive` is a fraction of channel
maximum (e.g. 0.2). These are different quantities. MATLAB direct and local
subtile wrappers pass only the threshold and retain the method's adaptive mode;
only the deep wrapper forwards `intensity_estimation`. Putting `noise` in a
MATLAB configuration does not enable the Python detector.

`streaming: true` is an extra Python parameter supported in direct, GR and deep
creation scripts, default false. It selects a different fixed processing path;
it does not honor all batch `run` flags. For example direct streaming always
runs enhancement, registration, detection, extraction and filtering and does
not forward histogram-equalization/morphological-reconstruction flags. Do not
use it as a drop-in switch for an arbitrary batch recipe. Python registration
kwargs such as TPS control settings are not forwarded by these wrappers.

For subtiles, Python `STARMapDataset.from_config` inspects GR settings before
deep settings, irrespective of enabled mode, when configuring windows. Keep
only the relevant creation section in production configs or make the grid
settings consistent (the full example uses 2 for both). Dimensions must match
the arrays **after rotation**; the schema cannot verify that.

## Downstream parameter blocks

| Rule | Parameters | Constraints / effective defaults |
| --- | --- | --- |
| `create_nuclei_amplicon_overlay` | `maximum_projection` | boolean; script indexes it directly; separate from top-level projection |
| `stardist_segmentation` | `stardist_base_path`, `stardist_model_name`, `segmentation_input_folder` | strings; input folder defaults in rule code to `overlay`; supply actual model path/name for execution |
| `stardist_segmentation` | `prob_thresh`, `nms_thresh` | numbers in [0,1]; required by script, no schema defaults |
| `stardist_segmentation` | `rescale`, `expand_labels`, `distance` | booleans and integer >=0; no script defaults; rescale halves XY and restores labels, expansion of 3-D labels is per slice |
| `reads_assignment` | `expand_labels`, `dilation_distance` | boolean and integer >=0; required by script, no defaults; expansion of 3-D labels is per slice |

See [downstream inputs and limitations](workflow-downstream.md) before enabling
these flags. Disabled sections in the full example do not validate model paths
or exercise those scripts.

## Overrides and validation boundaries

Top-level unknown fields and most extra parameter fields are allowed. Unknown
**rule names** and extra **resource keys** are rejected. A spelling error in an
otherwise permissive parameter object can silently leave a script default in
use. The schema does not check path existence, reference membership, barcode
length, channel count, stage prerequisites or backend feature compatibility.

Snakemake combines `--configfile` and CLI `--config` values for Python scripts
and DAG selection. However, `rsf_preparation` serializes the original file on
disk for MATLAB; CLI mode/parameter/path overrides are not written back to
that file. A `.yml` extension is also unsafe because conversion literally
replaces `.yaml`. Use one fully resolved `.yaml` file and matching
`config_path`, keep it writable for the sibling JSON, and edit that file when
changing scientific parameters. Avoid overriding `config_path` independently.

Unsupported or unverified combinations to avoid:

- Python `rsf_single_fov_seq`: no rule is defined for that backend.
- Python spot mode `local`, MATLAB `noise`/`adaptive_round`, and a nonempty
  Python-style channel string list passed to MATLAB's loader.
- Treating all schema-accepted per-stage references, registration methods,
  segmented-barcode or streaming settings as cross-backend equivalents.
- Disabling detection/extraction/filtering in direct mode while still requesting
  its full declared output contract, without checking how the wrapper saves
  missing state. Flags do not automatically repair downstream dependencies.
- Mixing MATLAB `.mat` and Python `.npz` subtile archives, or expecting a
  successful DAG check to certify subtile overlap reconciliation.

## Full configuration example

This file covers both optional core paths and disabled downstream sections.
It selects Python direct mode by default and uses small illustrative dimensions;
it is not a validated biological parameter set. For MATLAB, adapt channel order,
extraction axes, spot mode and script-required fields first.

```{literalinclude} examples/workflow-full.yaml
:language: yaml
```

{download}`Download the full YAML <examples/workflow-full.yaml>`.
