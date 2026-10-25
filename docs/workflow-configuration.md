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
| `config_path` | string | Yes | —; absolute path to the same `.yaml` passed to `--configfile`; for the MATLAB backend `rsf_preparation` writes a sibling JSON, which the MATLAB scripts read (the Python rules read `snakemake.config` and take no JSON) |
| `starfinder_path` | string | Yes | —; repository root used to locate Python/MATLAB code |
| `root_input_path`, `root_output_path` | strings | Yes | —; roots before dataset/sample or dataset/output components |
| `fiji_path` | string | No | —; read at parse time only by the legacy BigStitcher rule `run_BigStitcher_macro` (`stitching.smk`, included for the MATLAB backend without a `stitching` block); the Python BigStitcher engine rule reads it when it runs |
| `envs_path` | string | No | —; no rule reads it: `stardist_segmentation` runs in the environment that runs Snakemake (see [segmentation and assignment](#segmentation-and-assignment)); kept in the schema until §2.13 |
| `dataset_id`, `sample_id`, `output_id` | strings | Yes | —; directory identifiers |
| `fov_id_pattern` | string | MATLAB backend | —; MATLAB backend only (the Python backend refuses it, see [FOV membership](#fov-membership)): Python format string with `{i}`, e.g. `tile_{i}`, `Position{i:03}` |
| `n_fovs` | integer >=1 | MATLAB backend | —; MATLAB backend only (refused by the Python backend): FOV indices start at 1 |
| `n_rounds` | integer >=1 | Yes | —; round names are exactly `round1` … `roundN` |
| `selected_fovs` | nonempty array of unique FOV IDs, or null | No | null (every FOV); Python only: the FOV IDs of `fovs.csv` to run; it restricts every rule, per-sample rules included, see [FOV membership](#fov-membership) |
| `ref_round` | string | Yes | —; must identify an existing sequencing round; membership is not schema-checked |
| `ref_channel` | string | No | —; nuclei registration: the shared stain's name in each other round (and, with several `reference_stains`, the one of the reference round), not Python sequencing-channel selection |
| `dapi_round` | string | No | —; MATLAB backend only: the round whose `*ch04.tif` `nuclei_registration.m` and `rotate_nuclei` read. The Python backend refuses it, naming `reference_stains` |
| `reference_stains` | nonempty array of `channel_info` objects (`channel`, `name`, optional `wavelength`), or null | No | None; Python only: the reference round's stain files that are not sequencing colours (`Dataset.reference_stains`), for example `[{channel: ch04, name: DAPI}]`; the Python `rotate_nuclei` and `nuclei_registration` read its one entry, or with several the one whose name contains `ref_channel`. See [channels](#channels) |
| `rotate_angle` | number | Yes | —; degrees, 0 for no rotation |
| `img_col`, `img_row` | integers >=1 | MATLAB backend; Python where read | —; width X and height Y. The Python backend requires them only for the subtile windows (an active `create_subtiles` block of `gr_single_fov_subtile` or `deep_create_subtile`) and the legacy tile configuration writer (stitching method `bigstitcher` without an `assembly` block); elsewhere they are not read |
| `img_z` | integer >=1 | No | —; downstream 3-D metadata/preview uses it; `reads_assignment` needs it for a YX label image (the Z size of the molecule grid) |
| `voxel_size_xy`, `voxel_size_z` | numbers >0 | No | —; physical microns for stitching and the declared calibration of the files the segmentation and assignment rules read, and of a FOV's reference image (`reference_image`) that stores no metadata when the stitching rules, `reads_assignment` (with an `assembly` block) and `export_sample` read its grid; **not** extraction window radii |
| `reference_image` | string with `{fov_id}` | No | `fovs/{fov_id}/reference.tif`; Python only: the file pattern of a FOV's reference image, absolute or relative to the output root, whose grid the stitching rules, `reads_assignment` (with an `assembly` block) and `export_sample` read and which `reference_merged` names (`Dataset.reference_image`); a pattern without `{fov_id}` raises `ValueError`, a missing file `FileNotFoundError`, each naming the key. Absent or null is the default; the Python processing rules write the file there, and `FOV.load_reference_image` and every reader read it. Excludes `reference_image_import` |
| `reference_image_import` | string with `{fov_id}` | No | None; Python only: the file pattern of an imported reference image, one another tool or rule produced, absolute or relative to the output root (`Dataset.reference_image_import`). The processing rules then write no reference image, and every reader reads this file; nothing writes to it. Excludes `reference_image` |
| `codebook`, `direct_panel` | strings, or null | No | None (`<input root>/genes.csv`); Python only: the codebook (`codebook`, readout_mode `multiplexed`) or direct panel (`direct_panel`, readout_mode `direct`) the sequencing rules, `reads_assignment` and `export_sample` read, absolute or relative to the input root; the key of the other mode raises. See [reference file and image layout](#reference-file-and-image-layout) |
| `image_dir_template` | `"{round}/{fov}"` or `"{fov}/{round}"`, or null | No | None (`"{round}/{fov}"`); Python only: the layout of the acquisition images under the input root (`Dataset.image_dir_template`); any other value raises. See [reference file and image layout](#reference-file-and-image-layout) |
| `maximum_projection` | boolean | No | `false`; reference-image/DAPI output projection; does not turn sequencing into a 2-D algorithm |
| `seq_channel_order` | array of strings, or of objects with `wavelength`, `channel`, `name` | No | —; set explicit patterns for Python, e.g. `[ch00, ch02, ch01, ch03]`, or MATLAB's `channel_order_dict` entries; see [channels](#channels) |
| `additional_round` | array of objects with string `round_name` and optional `channel_order` | No | —; supply `[]` when unused; accessed at parse time; `channel_order` lists the round's `wavelength`/`channel`/`name` entries, see [channels](#channels) |
| `backend` | `python` or `matlab` | No | `matlab` |
| `segmentation`, `assignment` | objects | No | —; Python only (`backend: python`): a segmentation plan and an assign call for the §2.13 rule, see [segmentation and assignment](#segmentation-and-assignment) |
| `report` | object | No | —; Python only (`backend: python`): the `ReportConfig` of the processing report, and with it the rules `report_sample` and `report_dataset`, see [processing report](#processing-report) |
| `readout_mode` | `multiplexed` or `direct` | No | `multiplexed`; Python only (`Dataset.readout_mode`), and `direct` only with `backend: python`; see [readout mode](#readout-mode) |
| `matlab_launcher` | `path` or `broad` | No | `path`; how MATLAB rules start MATLAB, see [MATLAB launcher](#matlab-launcher) |
| `matlab_single_thread` | boolean | No | `false`; `true` adds `-singleCompThread` to every MATLAB call |
| `workflow_mode` | `free`, `direct`, `subtile`, `deep` | No | `free`; preset selection described in [workflows](workflows.md#modes-and-backend-selection) |
| `subset_list` | array of integers >=1 | No | —; MATLAB backend only; supply `[]` when unused; code indexes it directly |
| `subset_range` | boolean | No | `false`, applied by `common.smk`'s MATLAB branch (the schema holds no default); MATLAB backend only |
| `subset_start`, `subset_end` | integers >=1 | Conditional | —; MATLAB backend only; see caveat below |
| `subset_random` | boolean | No | `false`, applied by `common.smk`'s MATLAB branch (the schema holds no default); MATLAB backend only |
| `n_random_tests` | integer >=1 | Conditional | —; MATLAB backend only; see caveat below |
| `rules` | object of recognized rule names | Yes | —; each provided section requires boolean `run` |

### Reference file and image layout

Three Python-only keys name the inputs that the Python rules read from the input root
`INPUT = root_input_path/dataset_id/sample_id`; the MATLAB backend reads
`INPUT/genes.csv` and the round-first layout only, and `common.smk` refuses the three
keys for it.

* `codebook` (readout_mode `multiplexed`) or `direct_panel` (readout_mode `direct`)
  names the reference file, absolute or relative to INPUT; absent or null, it is
  `INPUT/genes.csv`. The other mode's key raises `ValueError` naming both. The
  sequencing rules (`rsf_single_fov`, `gr_single_fov_subtile`,
  `lrsf_single_fov_subtile`, `deep_rsf_subtile`) declare it as their codebook input,
  `export_sample` loads it for the genes of the molecule tables, and `reads_assignment`
  loads it as they do: its genes, in its order (`Codebook.genes`, first appearance, or
  `DirectPanel.genes`, table order), are the gene universe and the columns of the count
  matrix. No input file is renamed.
* `image_dir_template` lays out the images of one round of one FOV: `"{round}/{fov}"`
  (the default), `INPUT/<round>/<fov>/`, or `"{fov}/{round}"`,
  `INPUT/<fov>/<round>/`. Any other value raises. `FOV.input_dir` builds the folder
  with it, so do the image inputs of the Python rules and the stain file of
  `rotate_nuclei` (`get_dapi_input`); the channel files inside the folder are matched
  as before, by substring or `source_paths`. `fovs.csv` stays the FOV membership.

```yaml
backend: python
codebook: panels/codebook-v2.csv      # INPUT/panels/codebook-v2.csv
image_dir_template: "{fov}/{round}"   # INPUT/<fov>/<round>/*ch00.tif ...
```

### FOV membership

The Python backend reads its FOVs from `fovs.csv` in the input root
(`<root_input_path>/<dataset_id>/<sample_id>/fovs.csv`): the columns
`sample,fov_id`, and an optional `position_id` (the stage file's integer
`PositionID` of the FOV, required when the stitching block reads a stage file),
read in row order. `FOVS` is the FOVs in row order; `SAMPLE` the samples in the
order of their first row; each sample's FOVs keep their row order. A repeated
row, a FOV in two rows (another sample or another `position_id`), a
`position_id` of two FOVs, an empty value or another column raises
`ValueError` when the rule files are parsed. The `sample` column groups FOVs for
the per-sample rules; it is not the acquisition `sample_id`, which stays part of
every path and namespace. The optional `selected_fovs` list restricts a run to
those FOV IDs of the file, for every rule (the per-FOV rules and the per-sample
stitching, export and report rules alike); a sample with none of them is left
out. The Python rules' `fovID` wildcard matches the run's FOVs only, so an
explicit target of another FOV resolves to no rule. A stage file or a supplied
position table may hold other FOVs: the sample's selected FOVs are taken from
it, and a regular grid is laid out over the sample's FOVs of `fovs.csv`, so a
selected FOV keeps its cell. Python has no random selection. The Python backend refuses
`fov_id_pattern`, `n_fovs` and the six subset keys, and needs no sample
annotation.

The MATLAB backend keeps its membership: `documents/sample-annotation.csv`
under the output root, `fov_id_pattern`, `n_fovs` and the subset keys below.

A subset switch requires its companion keys only when it is `true`: `subset_range: true`
requires `subset_start` and `subset_end`, and `subset_random: true` requires
`n_random_tests`, and the validation error names each missing key. A switch that is
absent or `false` requires nothing, so `tests/minimal_config.yaml`, which sets no subset
key, passes. The schema gives no subset key a default, because the Snakefile's
`snakemake.utils.validate` inserts schema defaults into every configuration and the
Python backend refuses these keys; an omitted `subset_range` or `subset_random` is set to
`false` by `common.smk`'s MATLAB branch, after validation, and the companion keys are read
only when their switch is true.

For the MATLAB backend, subsetting precedence is nonempty `subset_list`, then true `subset_range`
(inclusive), then true `subset_random`, else all `1..n_fovs`. The list path
updates the sample list through annotation lookup; range/random paths do not.
Random selection calls Python's unseeded `random.sample`; use explicit indices
for a reproducible subset. Bounds, duplicate indices, reversed ranges and
annotation coverage are not validated by the schema. Sample aggregation still
expects every FOV in each annotation range, even when sequencing is subsetted.

Python arrays are `(Z,Y,X,C)`. Explicit channel patterns must match the files
and codebook ordering. The Python factory leaves an empty channel list empty;
it does not supply MATLAB's default. MATLAB's default order is
`ch00,ch02,ch01,ch03`, and its nonempty custom channel setting is passed to a
loader expecting a struct array with `channel`/`name`: the object form of
`seq_channel_order` ([channels](#channels)) has those fields, the list of
patterns does not. Use `seq_channel_order: []` for default MATLAB loading;
do not assume a Python custom list is portable to MATLAB.

### Channels

The shared keys keep their names; Python reads them into the channel groups of
`Dataset` ({doc}`coordination`, "Channels"):

* `seq_channel_order` is either a list of file patterns or a list of objects
  with `wavelength` (nm, optional), `channel` (the file pattern) and `name`
  (MATLAB's `channel_order_dict`). Both give the same patterns
  (`Dataset.channel_order`); the object form adds the names and wavelengths
  (`Dataset.channel_info`). The object form is the one MATLAB loads.
* Every rule's `Dataset` names the other rounds by each `additional_round`
  entry's `round_name` and takes the round's channels from its `channel_order`
  (`Dataset.other_channel_order`); an entry without `channel_order` takes the
  sequencing channels.
* The Python-only `reference_stains` declares the reference round's stain
  files that are not sequencing colours (`Dataset.reference_stains`), as
  `{channel, name}` entries with an optional `wavelength`; absent or null,
  there is none. The Python `rotate_nuclei` reads the reference round's file of
  its one entry (with several, the one whose name contains `ref_channel`), and
  `nuclei_registration` registers each other round against it, labelled
  `<ref_round>:<channel>`. No channel is assumed: there is no hard-coded
  `ch04` on the Python backend.
* `dapi_round` is the MATLAB backend's key: `nuclei_registration.m` and the
  MATLAB `rotate_nuclei` read `dapi_round`'s `*ch04.tif`. The Python backend
  refuses it, naming `reference_stains`; for a MATLAB-backend configuration,
  which the adapters of the shared rules read, it still gives
  `ChannelInfo("ch04", ref_channel)` and must equal `ref_round`.

`nuclei_registration` still selects each other round's shared stain by
MATLAB's substring match of `ref_channel` in the channel names.

```yaml
ref_round: round1
reference_stains:
  - {wavelength: 405, channel: ch04, name: DAPI}
ref_channel: DAPI
seq_channel_order:
  - {wavelength: 488, channel: ch00, name: seq}
  - {wavelength: 546, channel: ch02, name: seq}
  - {wavelength: 594, channel: ch01, name: seq}
  - {wavelength: 647, channel: ch03, name: seq}
additional_round:
  - round_name: round5
    channel_order:
      - {wavelength: 488, channel: ch00, name: Flamingo}
      - {channel: ch01, name: RBD}
      - {wavelength: 405, channel: ch02, name: DAPI}
```

## MATLAB launcher

MATLAB rules (the MATLAB core backend, including its `nuclei_registration`;
the Python backend runs a Python `nuclei_registration`) call
[`run_matlab_scripts`](https://github.com/wanglab-broad/starfinder/blob/dev/workflow/rules/common.smk),
which delegates to
[`matlab_launcher.py`](https://github.com/wanglab-broad/starfinder/blob/dev/workflow/rules/matlab_launcher.py).
It adds `workflow/scripts` to the MATLAB path and runs the entry point with
`-batch`, so a MATLAB error ends the job with a non-zero exit status. The full
command line is printed to the Snakemake log before MATLAB starts.

| `matlab_launcher` | Command | Use |
| --- | --- | --- |
| `path` (default) | The executable named by the `STARFINDER_MATLAB_EXECUTABLE` environment variable if set, otherwise `matlab` on `PATH` | Local runs and hosts where MATLAB is already on `PATH` |
| `broad` | `source /broad/software/scripts/useuse && use Matlab && exec matlab ...` in one Bash shell | Broad cluster jobs; `STARFINDER_MATLAB_EXECUTABLE` is ignored |

An unknown launcher, a non-boolean `matlab_single_thread`, an entry-point name
that is not a plain MATLAB function name, a missing executable or, for `broad`,
a missing `useuse` script raises an error before MATLAB starts. Configurations
without these keys use `path` and `false`; set `matlab_launcher: broad` in
Broad cluster configurations.

`matlab_single_thread: false` keeps MATLAB's default multithreading, so rules
that declare `threads: 4` can use those cores. Set it to `true` only when MATLAB
must stay on one computational thread, for example on a shared host.

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
`create_sample_maf` (MATLAB backend only) fixes memory/runtime to 8000 MB / 2 minutes in code.

The following keys sit under `rules.<rule>.parameters`. Unless stated otherwise,
there is no schema default. Optional blocks are not a promise that every script
can omit them: MATLAB scripts often index fields directly. The minimal example
is specifically for the Python batch/direct wrapper.

The Python backend accepts one dialect: the native keys `preprocessing`,
`registration`, `reference_image_view` and the `spot_finding` fields. The
MATLAB keys of the processing rules (`enhance_contrast`, `hist_equalize`,
`morph_recon`, `tophat`, `snr_threshold`, `global_registration`,
`local_registration`, and local_maxima's `intensity_estimation`,
`intensity_threshold` and `min_distance`) raise `ValueError` under
`backend: python`, naming the native form, and the schema's backend guard
refuses them, whatever a block's or a rule's `run` flag and also inside
`channel_overrides` entries; inside the native keys, the MATLAB names of a
registration step or recovery alternative, a MATLAB image representation as a
signal `mode` and MATLAB's `radius` of a preprocessing step raise the same way.
The table of {doc}`migration` ("One Python configuration
dialect") gives each key's native form. Their definitions stay for the MATLAB
backend, whose scripts read them, and for the adapters of the shared rules,
which read MATLAB-backend configurations (`dataset/_legacy_config.py`).

| Block | Schema fields and constraints | Wrapper behavior / defaults |
| --- | --- | --- |
| `enhance_contrast` | boolean `run` | MATLAB backend: min-max enhancement. The Python backend refuses it (`preprocessing` step `min_max_normalization`) |
| `hist_equalize` | boolean `run`, integer `reference_channel` | MATLAB backend: deep passes the configured 1-based index, direct uses its method default. The Python backend refuses it (step `histogram_matching`) |
| `morph_recon` | boolean `run`, integer `radius` >=1 | MATLAB backend: reconstruction, after registration in the resident subtile scripts. The Python backend refuses it (step `reconstruction`, in `post_registration` for the resident subtile rules) |
| `tophat` | boolean `run`, integer `radius` >=1 | No MATLAB script reads it; the Python backend refuses it (step `white_tophat`); kept for MATLAB-backend configurations read by the shared rules' adapters |
| `preprocessing` | `steps` (list; each with `method`, a step name, its config fields and optional `save_as`), optional `post_registration` (a list of `reconstruction` steps without `save_as`), `extraction_source`, `registration_source`, `supplied_statistics` | Python rules only (the schema rejects it unless `backend: python`): an explicit [preprocessing recipe](#explicit-preprocessing-recipe); `post_registration` lists the steps run after registration (`PreprocessingRecipe.post_registration`) |
| `registration` | `steps` (list; each with `method`, a registered method name, its config fields and optional `recovery` and `signal`), optional `signal`, `warp`, `qc`, `reference_round` | Python rules only (the schema rejects it unless `backend: python`): an explicit [registration recipe](#explicit-registration-recipe). Every default is written out: `method` is required, and a CPD step migrated from the MATLAB keys states `detection_noise_sigma: 3.0` and `grid_spacing_voxels: 32` (the native defaults are 5.0 and 16) |
| `reference_image_view` | `mode` `max` (default) or `channel`, with `channel` (a zero-based index or a channel label of the reference round) for `channel` | Python rules only: the reference image the rule writes (`FOV.save_reference_image`), the channel maximum or one channel; `rsf_single_fov`, `gr_single_fov_subtile` and `deep_create_subtile` (the resident subtile rules write none and refuse it). It replaces the single-channel reference image that `ref_img: single-channel` selected in `rsf_single_fov` |
| `global_registration` | boolean `run`, string `ref_round`, `ref_img`/`mov_img` in `merged-image`,`single-channel` | MATLAB backend: the wrappers pass the block reference; MATLAB deep uses scale 0.25. The Python backend refuses it (a global `registration.steps` entry, the recipe `signal`) |
| `create_subtiles` | boolean `run`, integer `sqrt_pieces` >=1 | Grid default 4; only GR/deep creation rules produce subtile files; Python creation scripts call splitting unconditionally; an active block (`run: true` or a window setting) needs `img_row` and `img_col` |
| `local_registration` | boolean `run`, string `ref_round`, method `demons`,`bspline`,`tps`,`cpd` or the demons variants `diffeomorphic`,`symmetric`,`fast_symmetric` (schema default `demons`); `ref_img`/`mov_img` | MATLAB backend: the wrappers do not forward `method`; deep subtile does not perform local registration. The Python backend refuses it (a local `registration.steps` entry) |
| `spot_finding` | boolean `run`, string `ref_round`; MATLAB keys nonnegative numeric `intensity_threshold` and mode `intensity_estimation` (`global`,`noise`,`adaptive`,`adaptive_round`); Python-only `method`, the selected config's fields, `channel_overrides` and `rounds` | Python rules: see [the method key](#spot-finding-method-key); for local_maxima the fields are `threshold_mode`, `threshold_value` and `min_distance_voxels`, and the MATLAB keys raise naming them. MATLAB supports adaptive/global only |
| `load_codebook` | boolean `run`, integer-array `split_index`; Python-only `encoding` (`method` `two_base` or `one_base` and that config's fields; `two_base`: `reverse_bases` and `pair_to_color`, see [encoding table](#codebook-encoding-table)) | `split_index` is **one-based**: MATLAB's position in the encoded color string of the junction color that is removed (aging `[5]`). Python wrappers load unconditionally, turn missing/empty split into None and translate `[s]` into the two-segment layout, equal to the zero-based `EncodingConfig(split_index=s - 1)` ([segment layout](#codebook-segment-layout)); MATLAB respects `run` |
| `reads_extraction` | boolean `run`, exactly three integers >=1 in `voxel_size`; Python-only `background` (`false` or `inner_radius_zyx`, `outer_radius_zyx`, `min_voxels`) | Pixel half-widths, Python `(z,y,x)` versus MATLAB `(row,column,z)`; e.g. `[1,2,2]` versus `[2,2,1]`, not physical microns. Python rules measure the local background and noise next to the sums (`LocalBackgroundConfig`, on by default; {doc}`readout-contract`, "Extraction"); `background: false` turns it off. Without the key, the default ring (inner box `(1,3,3)`, outer box `(1,6,6)`, `min_voxels` 16) grows by the same number of voxels in both boxes along any axis where `voxel_size` exceeds the inner box, so the inner box contains the extraction box |
| `reads_filtration` | boolean `run`, string or string-array `end_base`, integer `n_barcode_segments` >=1, integer-array `split_index` | Python wrappers forward a string `end_base` and extra `start_base` (default `C`) to `ReadFilterConfig`; a list `end_base` becomes the allowed ends of the layout's segments; `n_barcode_segments` and this block's `split_index`, when given, must agree with the layout ([segment layout](#codebook-segment-layout)); MATLAB forwards segment count and split |
| `decoding` | Python-only `method` (`wta`, `codebook_aware` or `direct`) and that config's fields | Python rules only: the decoder used with `reads_filtration.run`; default `WtaDecoderConfig(diagnostics=True)`, or `DirectDecoderConfig()` with `readout_mode: direct`; the adapter sets `diagnostics` true and `allow_rescue` false unless given; a method that does not support the readout mode raises `TypeError` |
| `scoring` | Python-only boolean `run` (default true) and `method` (`bgcorr_probability`) | Python rules only (the schema rejects it unless `backend: python`): the adapter adds the shared read-QC score (`ReadScoreConfig`, {doc}`readout-contract`, "Shared read-QC score") whenever it decodes; `run: false` turns it off. It ranks calls, sets no cutoff and never changes a call. Giving the block without `reads_filtration.run`, or scoring with `reads_extraction.background: false`, raises `ValueError` |
| `deduplication` | Python-only boolean `run` (default false), `distance_voxels` (default 1.0, voxel index space) and `compatibility` (`same_sequence`) | Python rules only (the schema rejects it unless `backend: python`): with `run: true` the adapter sets `PipelineConfig.deduplication` ({doc}`readout-contract`, "Optional deduplication"), which marks cross-channel reads of one amplicon within the distance with the same WTA sequence as duplicates of one representative; the read filter then rejects them (`ReadFilterConfig.exclude_duplicates`, default true). Off without the block. It needs `reads_filtration.run` and raises `ValueError` with `readout_mode: direct` |
| `checkpoints` | Python-only mapping with the one key `stages` (a list of `registered`, `candidates`, `pre_qc`), or null | Python rules only: the checkpoint stages the five processing rules (`rsf_single_fov`, `gr_single_fov_subtile`, `lrsf_single_fov_subtile`, `deep_create_subtile`, `deep_rsf_subtile`) write besides `run.json`; absent or null is `{stages: []}`, `run.json` alone. See [run records of the processing rules](#run-records-of-the-processing-rules) |

In a MATLAB configuration, always pair `intensity_estimation` with
`intensity_threshold`; in a Python one, `threshold_mode` with `threshold_value`.
`noise` uses a k-sigma threshold (e.g. 5), whereas `adaptive` is a fraction of
channel maximum (e.g. 0.2). These are different quantities. MATLAB direct and
local subtile wrappers pass only the threshold and retain the method's adaptive
mode; only the deep wrapper forwards `intensity_estimation`. Putting `noise` in
a MATLAB configuration does not enable the Python detector.

`streaming: true` selects round-at-a-time loading in the same Python
`FOV.run(PipelineConfig, execution=ExecutionConfig(...))` sequence as batch.
Every enabled/disabled stage and its supported settings follow the same path.
Creation jobs explicitly retain processed images for subtile export, so streaming
creation does not imply bounded total resident image memory. The
`registration` key's steps become one `RegistrationRecipe` (global steps, then
at most one local step); a recovery alternative of another kind is rejected.
Unknown Python rule parameters and conflicting reference rounds raise early;
shared MATLAB keys remain unchanged. See [coordination](coordination.md).

`from_workflow_config` chooses the relevant subtile creation settings; image
sizes must match arrays after rotation. Rectangular and remainder dimensions
are partitioned independently with complete edge coverage.

### Run records of the processing rules

Each Python processing rule runs its FOV (or subtile) with
`CheckpointConfig(stages=<stages>, overwrite=True)`, so every run writes
`<output_root>/fovs/<fov>/run.json` (`subtile_<n>/run.json` below it for a
subtile rule; {doc}`checkpoints`, "run.json"). The stages come from
`rules.<rule>.parameters.checkpoints`, a mapping with the one key `stages`; absent
or null it is `{stages: []}`, which writes `run.json` alone. For example, the
stages that let the processing report read the candidates and the reads:

```yaml
rules:
  rsf_single_fov:
    parameters:
      checkpoints:
        stages: [candidates, pre_qc]
```

`overwrite` is on because a rerun of the rule replaces its record: it removes the
earlier stage files of the FOV folder's `checkpoints/` and leaves `other_rounds/`,
`segmentation/` and `assignment/`. The inputs are hashed (`hash_inputs`), so each
raw image is read once more. `run.json` records the configuration file the rule ran
with (`workflow_config`: the top-level `config_path` and the SHA-256 of its bytes)
and the codebook or panel file the rule loaded (`codebook`), each absolute (or
relative to the input root when it lies there) with its SHA-256; the raw images are
recorded relative to the input root (`run.json` version 2). `run.json` is a
declared output of each Python processing rule and replaces the logs those rules
wrote before ({doc}`migration`, "Sequencing logs and the morphology record"); the
stage files are not declared. Its `status` `succeeded` says that the rule's
`FOV.run` processing succeeded; each other declared output the rule writes after it
(`molecules.csv`, the reference image, the subtile files, a resident subtile's
`subtile_goodSpots_<n>.csv`) is then added to its `outputs` with its path and SHA-256. The key is allowed with `backend: python` only: the schema refuses it with `backend: matlab` or without `backend` (whose default is `matlab`), and the adapter raises `ValueError`.

## Codebook segment layout

The shared `split_index` is MATLAB's one-based position `s` (`LoadCodebook.m`
erases color `s` and puts the colors after it first). For a `two_base` barcode of
`n = n_rounds + 2` bases, the adapter translates `load_codebook.split_index: [s]`
into `BarcodeLayout((Segment("A", n - s), Segment("B", s)), ("A", "B"))`, or with
`load_codebook.encoding.reverse_bases: false` into
`BarcodeLayout((Segment("A", s), Segment("B", n - s)), ("B", "A"))`; both equal the
zero-based `EncodingConfig(split_index=s - 1)`, which `WorkflowConfig.split_index`
holds. For aging (`s = 5`, 11 bases, 9 rounds) segment A has 6 bases (5 colors,
rounds 1–5) and segment B 5 bases (4 colors, rounds 6–9). Before W-292 the adapter
passed `s` unchanged as the zero-based index, so `[5]` dropped the wrong color; a
configuration that gave the Python value (4) must give the MATLAB value (5).

`reads_filtration.n_barcode_segments` must equal the number of segments (2 with a
split, 1 without) and `reads_filtration.split_index` must equal
`load_codebook.split_index`, when they are given; otherwise the adapter raises
`ValueError`. A list `end_base` gives the allowed (first, last) bases of the
segments in read orientation: with one segment every listed pair, with two
segments item k (a pair or a list of pairs) for segment k in acquisition order, as
`FilterReadsMultiSegment.m` reads it. A codebook entry whose ends are not listed
raises when the codebook is loaded; reads are checked per segment by
`filter_reads` and rejected only with `exclude_invalid_endpoints: true`. A string
`end_base` keeps its one-segment meaning with `start_base`.

```yaml
load_codebook: {run: true, split_index: [5]}
reads_filtration: {run: true, n_barcode_segments: 2, split_index: [5], end_base: ["CC", "TT"]}
```

## Codebook encoding table

The Python-only `load_codebook.encoding.pair_to_color` sets the `two_base` table of
`EncodingConfig.pair_to_color` ({doc}`readout-contract`, "Encoding registry"): the
16 ordered base pairs, in read orientation, each mapped to a color 1–4 (integers
or strings), where the four pairs that start with one base have four different
colors. The adapter gives the same codebook as the Python call and raises the same
`ValueError` for an invalid table; without the key the table is the default, the
active table of `EncodeBases.m`. The schema accepts the key only with
`backend: python`: MATLAB keeps its own table. For example, the second table kept
as comments in `EncodeBases.m`:

```yaml
backend: python
# under rules.<rule>.parameters:
load_codebook:
  run: true
  encoding:
    method: two_base
    pair_to_color: {AT: 1, TA: 1, GC: 1, CG: 1, AC: 2, CA: 2, GT: 2, TG: 2,
                    AA: 3, TT: 3, GG: 3, CC: 3, AG: 4, GA: 4, CT: 4, TC: 4}
```

`Dataset.encoding_table()` shows the loaded table with the channel of each color,
and `pre_qc.json` and `run.json` record it ({doc}`checkpoints`).

## Readout mode

The Python-only top-level `readout_mode` ({doc}`readout-contract`, "Readout modes")
is `multiplexed` (the default: reads are decoded from color sequences with the
codebook) or `direct`: each candidate is assigned the gene of its own round and
channel. The schema accepts `direct` only with `backend: python`. In `direct` mode:

* the rule's codebook input is the panel CSV, with the header `round,channel,gene_id`
  (one gene per (round, channel); a repeated gene or (round, channel) is an error):
  `INPUT/genes.csv`, or the file `direct_panel` names;
* `spot_finding.rounds` must list the rounds to detect in, so that every candidate
  has its round (listing only the reference round is allowed);
* each candidate is extracted in its own round only, and the `decoding` block
  defaults to, and may only name, `method: direct`;
* the barcode keys `load_codebook.split_index` and `encoding` and
  `reads_filtration.end_base`, `split_index`, `n_barcode_segments` and
  `exclude_invalid_endpoints` have no meaning and raise.

The Python `rsf_single_fov` writes the same molecule table as in multiplexed mode,
`fovs/{fovID}/molecules.csv`; its `entry_id` is `<round>/<channel>`.

```yaml
backend: python
readout_mode: direct
rules:
  rsf_single_fov:
    run: true
    parameters:
      spot_finding: {run: true, threshold_mode: noise, threshold_value: 10, rounds: [round1, round2]}
      reads_extraction: {run: true}
      reads_filtration: {run: true}
```

## Explicit preprocessing recipe

The Python-only `preprocessing` parameter of the five Python rules declares a
preprocessing recipe explicitly, as specified in the
[preprocessing contract](preprocessing-contract.md#explicit-step-list). Each
entry of `steps` names a step by its registered name (`method`, one of
`min_max_normalization`, `histogram_matching`, `reconstruction`,
`white_tophat`, `scalar_background`, `background_3d` and
`percentile_normalization`). Its other keys are the fields of that step's
config class, and `save_as` keeps a named snapshot of the step's output.
`extraction_source` and `registration_source` name a snapshot for extraction
and for registration signals; each defaults to the detection image, the last
step's output. `supplied_statistics` is the file read by steps with
`fit: supplied`. An unknown method, field or snapshot name raises. The steps
run before registration; `post_registration` (absent: none) lists the steps run
after it, in the same form, which may only be `reconstruction` steps without
`save_as` (`PreprocessingRecipe.post_registration`; the resident subtile rules'
form of MATLAB's `morph_recon`). Without the key no preprocessing runs. A legacy
preprocessing key raises under `backend: python`, naming its step; MATLAB keys
and behaviour are unchanged.

```yaml
rules:
  lrsf_single_fov_subtile:
    run: true
    parameters:
      preprocessing:
        steps: []
        post_registration:
          - {method: reconstruction, radius_yx: 3}
```

This recipe-2 configuration subtracts a scalar background, keeps that image as
the snapshot `bg_corrected`, then applies percentile normalization. Detection
and registration use the normalized image; extraction reads `bg_corrected`.

```{literalinclude} examples/workflow-recipe-2.yaml
:language: yaml
```

{download}`Download the recipe-2 YAML <examples/workflow-recipe-2.yaml>`.

## Explicit registration recipe

The Python-only `registration` parameter of the five Python rules declares a
registration recipe explicitly, as specified in the
[registration contract](registration-contract.md#python-only-key). Each entry
of `steps` names a method by its registered name (`translation`, `rigid`,
`affine`, `bspline`, `demons`, `tps` or `cpd`; the demons variants are the
`variant` field of `demons`). Its other keys are the fields of that method's
config class, plus an optional `recovery` (`allowed_errors`, and
`alternatives` naming methods of the step's kind) and an optional per-step
`signal`. YAML lists become tuples. `signal` (`mode`: `max`, the default,
`sum` or `channel` with `reference_channel` and `moving_channel`, an index or
a channel label), `warp` (the `WarpConfig` of the one final resampling;
omitted, it is derived from the transforms) and `qc` (the
`RegistrationQcConfig` rejection criteria; none by default) configure the
recipe. The steps must be zero or more global steps followed by at most one
local step. An unknown method, field or sequence raises; `global_registration`
and `local_registration` raise under `backend: python`, naming this key.
MATLAB keys and behaviour are unchanged. Each step states its settings: the
MATLAB keys' hidden defaults are written out when a configuration is migrated,
for example `detection_noise_sigma: 3.0` and `grid_spacing_voxels: 32` for a
`cpd` step (the native defaults are 5.0 and 16).

The rule parameter `reference_image_view` selects the reference image the rule
writes: `{mode: max}` (the default) is the channel maximum, `{mode: channel,
channel: <index or label>}` one channel of the reference round. The resident
subtile rules write no reference image and refuse the key.

```yaml
registration:
  signal: {mode: max}
  steps:
    - method: translation
    - method: affine
    - method: bspline
      recovery: {allowed_errors: [RegistrationEstimationError], alternatives: [{method: demons}]}
```

## Spot-finding method key

The `spot_finding` block of the five Python rules accepts the Python-only key
`method`, a `SPOT_FINDING_METHODS` method that the pipeline accepts
(`local_maxima`, the default, `starfish_log`, `spotiflow` or `piscis`), and every
other key is an init field of that method's config (YAML lists become tuples), for
example `exclude_border`, `measure_peak_intensity` or the opt-in W-218 merge
`merge_radius_zyx: [2, 2, 2]`. A config field without a default must be given:
`starfish_log` requires `min_sigma`, `max_sigma`, `num_sigma` and `threshold`, and
`spotiflow` and `piscis` require `model`, a pretrained model fetched beforehand with
`starfinder weights fetch <method> <model>` (detection never downloads).
For `local_maxima` the MATLAB keys `intensity_estimation`,
`intensity_threshold` and `min_distance` raise under `backend: python`, in the
block and in `channel_overrides`, naming the fields `threshold_mode`,
`threshold_value` and `min_distance_voxels`. For another method, `min_distance`
is that config's own field and the other MATLAB keys are errors.
`channel_overrides` maps a channel label of `seq_channel_order` to fields that
replace the block's for that channel (a `SpotFindingPlan`), and `rounds` lists the
round labels to detect in (omitted: the reference round only); a candidate set of several
rounds is decoded only with `readout_mode: direct` ([readout mode](#readout-mode)). The rule-level
Python-only key `device` sets `ExecutionConfig.device`; `cpu` is the only value.
The method-aware rules are in the
[spot-finding contract](spot-finding-contract.md#workflow-configuration).

```yaml
rsf_single_fov:
  parameters:
    device: cpu
    spot_finding:
      run: true
      ref_round: round1
      method: local_maxima
      threshold_mode: noise
      threshold_value: 5.0
      exclude_border: false
      channel_overrides:
        ch03: {threshold_value: 6.0}
```

## Segmentation and assignment

The rules `enhance_dapi_with_flamingo`, `create_nuclei_amplicon_overlay`,
`stardist_segmentation` and `reads_assignment` call the Starfinder package under both
backends (`_snakemake.py`), keeping their names, inputs and outputs. The
environment that runs Snakemake therefore needs the package, with the `stardist` extra
for `stardist_segmentation` and the `anndata` extra for `reads_assignment`; no rule reads
`envs_path` any more. The legacy keys are translated as the
[segmentation contract](segmentation-contract.md#workflow-configuration) and the
[assignment contract](assignment-contract.md#workflow-configuration) state; the
[downstream parameter blocks](#downstream-parameter-blocks) summarize them.

Two Python-only top-level blocks (rejected unless `backend: python`) describe the same
steps for the §2.13 rule that loads a FOV and runs `FOV.segment` and `FOV.assign_molecules`; the
legacy rules translate their own keys and raise when a block is present. In
`segmentation`, `device` is `cpu` (default) or `cuda`, and each run has a `name`, a
`target`, `inputs` (`InputChannel` fields; `prepare` names `composite_nuclei_amplicon` or
`enhance_with_flamingo` with its config fields), optional `seeds`, `projection` (`true`
or `ProjectionConfig` fields) and `operations` (`expand_labels` or
`extend_labels_through_z` with their config fields), and a `method`: a
`SEGMENTATION_METHODS` name whose config fields are the run's other keys, or `import`
with the `LabelImportConfig` fields. The `assignment` block holds the `AssignmentConfig`
fields and the `FOV.assign_molecules` arguments (`cells`, `nuclei`, `name`, `population`,
`checkpoints`); `correspondence` holds the `CorrespondenceConfig` fields (by default
`match_fraction` 0.5 and `outside_tolerance` 0.1, a provisional value). Default-tier
tests keep the schema's method list equal to the registry and the assignment keys equal
to those fields and arguments.

```yaml
segmentation:
  device: cpu
  runs:
    - name: nucleus
      target: nucleus
      inputs:
        - {role: nuclear, round: reference_stain, channel: DAPI}
      method: stardist
      model_path: /absolute/stardist_models/3D_spleen
      scale: 1.0
    - name: cell
      target: cell
      seeds: nucleus
      inputs:
        - {role: amplicon, reference_merged: true}
      method: seeded_watershed
      sigma_um: 1.5
assignment:
  cells: cell
  nuclei: nucleus
  population: final
  expansion: {distance: 0.78, unit: um, mode: planar}
  correspondence: {match_fraction: 0.5, outside_tolerance: 0.1}
  exclude_cells_without_nucleus: null
```

## Stitching, assembly and export

Three Python-only top-level blocks (rejected unless `backend: python`) replace the legacy
sample-level rules for a configuration that has them. The specification is the
"Workflow configuration" section of the
[stitching contract](stitching-contract.md#workflow-configuration), the
[assembly contract](assembly-contract.md#workflow-configuration) and the
[sample-output contract](sample-output-contract.md#workflow-configuration).

* **`stitching`.** With the block, `rules/stitching-py.smk` is included instead of the
  four legacy stitching rules, which then do not run whatever their `run` flags. One rule
  per sample, `stitch_sample`, writes `<OUTPUT>/stitching/<sample>/` (`placement.json`,
  `fovs.csv`, `links.csv`). `method` is a `STITCHING_METHODS` name, run in process by
  `Dataset.stitch` (`multiview_stitcher`), or `bigstitcher`, the default: then
  `bigstitcher_macro_sample` writes the tiles, the macro and the view map into
  `<bigstitcher_save_dir>/<sample>/` (a local folder, required with `bigstitcher`),
  `bigstitcher_engine` runs Fiji on the macro, and `stitch_sample` imports BigStitcher's
  `dataset.xml`, saves the placement and, without an `assembly` block, writes
  `output/tile_config_{sample}.csv` with the legacy writer from the record's raw
  translations and view map (with the block nothing reads that file, so it is neither
  written nor declared). `fiji_path` is read by
  `bigstitcher_engine` when it runs, not when the rules are parsed, and the rule fails
  unless `dataset.xml` holds a `Stitching Transform` for every view, whatever Fiji's exit
  status. `positions` is the positions mapping of `Dataset.stitch`; `stage_axes` is
  required with `source: stage_file` and is never inferred from `rotate_angle`, and so is
  `stage_file`, the stage file's path (no default; no rule writes a per-sample copy). The
  file may hold every sample's positions: each FOV's `PositionID` is the `position_id`
  column of `fovs.csv`, required then, and the sample's positions are selected before the
  conversion. The four legacy stitching rules, `create_sample_maf`, `create_sample_h5ad`
  and `create_sample_reads_assignment` run with `backend: matlab` only: a Python
  configuration that enables one without the block that replaces it (`stitching`, or
  `assembly` for the two sample rules), or `reads_assignment` without either block, raises
  when the rule files are parsed, naming the blocks to add. `config`
  holds the fields of `MultiviewStitcherConfig` or `BigStitcherConfig`, `review` the
  `ReviewPolicy` fields, and `fallback` is null or `nominal`. With `multiview_stitcher`
  no `tile_config_{sample}.csv` is written, so the legacy `reads_assignment` rule needs an
  `assembly` block there.
* **`assembly`** (needs `stitching`). The keys are the `AssemblyConfig` fields,
  `assignment` (the assignment name read per FOV, default `default`; null for molecules
  only), `table_format` (`csv` or `parquet`) and `previews` (default `false`; below).
  `duplicate_threshold` (default 0.5; null
  turns it off) is the duplicate rule of the route per FOV: two retained cells of different
  FOVs whose territories share at least that share of the smaller one are one cell, the
  larger observation stays and the other is excluded `duplicate_observation` with its
  molecules; it removes a second observation and does not complete a cell cut by a FOV
  edge ({doc}`assembly-contract`, "Retention", rule 8). With the block, `reads_assignment`
  assigns on the grid of the FOV's reference image (`reference_image`, default
  `fovs/{fov_id}/reference.tif`; its stored metadata, or for a file that stores none the
  grid `voxel_size_xy` and `voxel_size_z` declare, as `export_sample` reads it) and reads no
  `tile_config_{sample}.csv`. With an `assignment` name it writes and declares the FOV's
  assignment folder `fovs/{fovID}/assignment/<assignment>/` alone, whose record holds the
  counts: no `expr/{fovID}/raw.h5ad`, `reads_assignment.csv` or `log.txt` (W-568). With
  `assignment: null` (molecules only) it writes the per-FOV `raw.h5ad` and
  `reads_assignment.csv` without the tile filter and without `global_*` columns, and
  `log.txt` and the figures beside them, as before. `create_sample_h5ad` and
  `create_sample_reads_assignment` do not run. One rule per sample, `export_sample`, runs
  every part of `Dataset.export` after the placement and, on the route per FOV, every
  FOV's `reads_assignment`; on the route after fusion it needs no output of
  `reads_assignment`, but the FOVs' molecule tables (`fovs/{fovID}/molecules.csv`, or on the
  legacy route of {doc}`workflows`, "Molecule tables", `signal/{fovID}_goodSpots.csv`), their
  reference images (the files `reference_image` names) and, with `fuse` on, the source
  images of `images` that are files of the workflow (`reference_merged` and each
  `<name>=file:<pattern>`; the saved form of `reference_stain` is read but is no declared
  output of any rule, so it is not declared, as for `stitch_sample`); `reads_assignment`
  still runs while its `run` flag is on, and can be turned off on that route. Its declared output
  is `export/<sample>/export.json`, which lists every other file of the folder, so a part
  that cannot run (an extra not installed) is recorded `not_run` and does not fail the
  rule. **Previews** (opt-in, W-568). With an `assignment` name, `previews: true` turns on
  the two assignment figures, `assignment.png` (and `assignment_single_layer.png` for a
  ZYX label image), each kind with one writer and its own folder, declared as a directory
  output: `reads_assignment` draws each FOV's figures into `fovs/{fovID}/previews/`, and
  on the route per FOV `export_sample` draws them with the FOV's ownership region into
  `export/<sample>/previews/{fovID}/`, which `export.json` does not list; that folder is
  declared only on the route per FOV with the export block's `assemble` on, and created
  (empty) when the assemble part could not run. Neither rule overwrites the other's files. With `previews: false`, the default, no figure is drawn on
  the route of a saved assignment. On the route after fusion `export_sample` draws no
  figure: the sample's cells come from no per-FOV assignment. `previews` does not change
  molecules only (`assignment: null`), whose figures stay in `expr/{fovID}/`.
* **`export`** (needs `assembly`). The parts of `Dataset.export` and their settings: the
  flags `assemble`, `fuse`, `fuse_labels` and `export_spatialdata`, the `FusionConfig`
  fields (`images`, `dimensionality`, `projection`, `scale_yx`, `ome_zarr`,
  `ome_zarr_version`, `pyramid_levels`), `labels_scale_yx` (`LabelConfig.scale_yx`) and
  `spatialdata_zarr_format` (`SpatialDataExportConfig.zarr_format`), and the route
  ({doc}`sample-segmentation-contract`): `route` (`per_fov`, `after_fusion`, or null for the
  default by `dimensionality`: `projection` gives `after_fusion`, `stack` and `both` give
  `per_fov`), `segment` (default true; false reuses the saved sample segmentation) and
  `segmentation`, the settings of the route after fusion (`SampleSegmentationConfig`):
  `device`, `cells`, `nuclei` and `runs`, each run with `name`, `target`, `inputs` (a list
  of `{role, image}`, each `image` a name of `images`), optional `seeds` and a `method`, a
  `SEGMENTATION_METHODS` name whose config fields are the run's other keys, or `import`
  with the `LabelImportConfig` fields. The pixel settings of a method are given for the
  acquired grid, as for the route per FOV, and the route converts them to the segmentation
  grid at the factor `k` of `scale_yx = 1/k` by the table of
  {doc}`sample-segmentation-contract` ("Lengths in pixels"), recording both values:
  Cellpose's `diameter` becomes `diameter / k` and its area `min_size` becomes
  `max(1, round(min_size / k²))`; StarDist's `scale` is multiplied by `k` in Y and X, and its
  `block_size`, `min_overlap` and `context` are divided by `k` and rounded up, so block runs
  are possible at `k = 1` only; `n_tiles` and lengths in µm (`sigma_um`) do not change. The assignment settings are those of the top-level `assignment`
  block (its `AssignmentConfig` fields; the defaults without one). Without the block the
  defaults apply, so a configuration with an assignment name that exports the default
  projection takes the route after fusion and is refused until it sets
  `export.segmentation` or `route: per_fov`; the route after fusion also refuses
  `dimensionality: stack` and `projection: sum`.
* **The device of the route after fusion.** On that route the part `segment` runs inside
  `export_sample`, so that rule needs the device and memory of the segmentation methods:
  with a learned model on `cuda` (`segmentation.device: cuda`) the rule must run on a GPU
  node (choice E1 of
  {doc}`sample-segmentation-contract`, whose stated cost this is; the GPU resource of the
  rule belongs to §2.13 and is not set by this block). The whole fused projection is
  segmented in one call, so its size, set by `scale_yx`, must fit that device.

Default-tier tests keep the method list equal to `STITCHING_METHODS` plus `bigstitcher`
and the `assembly` and `export` keys equal to those fields, and the methods of a
`segmentation` run equal to `SEGMENTATION_METHODS`. The blocks raise naming the key when
the rule files are parsed; so does a route the configuration alone refuses (check 7 of
`Dataset.export` without the conditions of a reused segmentation and without the models,
which `export_sample` checks before anything is written).

```yaml
stitching:
  method: bigstitcher            # or multiview_stitcher
  input: reference_merged        # or reference_stain, or file:<pattern>
  positions:
    source: stage_file           # stage_file | regular_grid | table
    stage_axes: {x: "+x", y: "+y"}
    snap_to_grid: {overlap: 0.1}
    stage_file: /data/acquisition/positions.maf   # required; fovs.csv gives each FOV's position_id
  review: {max_correction_fraction: 0.5, max_residual_um: 1.0}
  fallback: null
  bigstitcher_save_dir: /local/scratch/bigstitcher
assembly:
  assignment: default
  ownership: nearest_center
  duplicate_threshold: 0.5       # null: the duplicate rule off
  table_format: csv
export:
  dimensionality: stack          # dense 3D tissue; the default is projection
```

One example configuration for each route ({doc}`workflow-downstream`, "Two routes to a
sample's cells", says which to choose). The route per FOV, for a projection sample that
keeps the per-FOV segmentation and assignment of `reads_assignment`:

```yaml
assembly:
  assignment: default
export:
  route: per_fov                 # the default for dimensionality stack and both
  dimensionality: projection
```

The route after fusion, for a projection sample (the culture's two stains, segmented at
1/4 with Cellpose on a GPU; the values illustrate the layout and are not recommended
settings):

```yaml
assembly:
  assignment: default
export:
  route: after_fusion            # the default for dimensionality projection
  images:
    - reference_stain
    - "dapi=file:images/organelle/DAPI/{fov_id}.tif"
    - "flamingo=file:images/organelle/Flamingo/{fov_id}.tif"
  scale_yx: 0.25
  segmentation:
    device: cuda
    cells: cell
    nuclei: nucleus
    runs:
      - name: nucleus
        target: nucleus
        inputs: [{role: nuclear, image: dapi}]
        method: cellpose
        model: cpsam_v2
        diameter: 60             # pixels of the acquired grid: 15 at 1/4
      - name: cell
        target: cell
        inputs: [{role: cytoplasm, image: flamingo}, {role: nuclear, image: dapi}]
        method: cellpose
        model: cpsam_v2
        diameter: 240            # 60 at 1/4
```

## Processing report

A Python-only top-level `report` block (rejected unless `backend: python`) makes the workflow
write the processing reports; the specification is the "Workflow configuration" section of
the [report contract](report-contract.md#workflow-configuration) (option WR1). A
configuration without the block runs as before: no report rule and no report output.

* **The keys** are the `ReportConfig` fields: `markers`, a list of `{name, genes}`
  (`MarkerGroup`, one to four genes, each group one figure); `marker_pairs`, gene pairs for
  MECR; `positive_threshold`; `example_fovs`, three FOV IDs or null for the automatic
  choice; `limits`, the `ReportLimits` fields, each rule (`shift_residual`,
  `alignment_score`, `correct_form_fraction`, `matched_fraction`) a mapping of the
  `RuleLimit` fields `k`, `floor` and `absolute`, and `spots_near_zero`; `figure_width_px`
  and `jpeg_quality`. Lists become tuples. An omitted key keeps its default, so `report: {}`
  gives `ReportConfig()`; an omitted rule of `limits` keeps its default limit, and an omitted
  key of a given rule is null. An unknown key, at any level, raises `ValueError` naming it
  when the rule files are parsed.
* **`report_sample`**, one rule per sample, calls `Dataset.report(sample, <the sample's
  FOVs from the sample annotation>, assignment=<assembly.assignment>, config=…)` and writes
  `<OUTPUT>/report/<sample>/`. Its declared output is `report/<sample>/report.json`, which
  lists every other file of the folder. Its declared input is the sample's last per-sample
  output: `export/<sample>/export.json` with an `assembly` block (the `export_sample` rule,
  which an `export` block configures); else `stitching/<sample>/placement.json` with a
  `stitching` block; else every FOV's `fovs/{fovID}/run.json`. Without an `assembly`
  block no rule saves a per-FOV assignment, so the report is made without cells
  (`assignment=None`).
* **`report_dataset`** calls `Dataset.compare_samples` on every sample of the run, in the
  order of the sample annotation, and writes `<OUTPUT>/report/index.html` and
  `report/samples.csv`, its declared output, after every sample's `report_sample`.

A default-tier test keeps the block's keys equal to the fields of `ReportConfig`,
`MarkerGroup`, `ReportLimits` and `RuleLimit`. The example is W-383's markers of the lymph
node sample, shortened; the other values are the defaults:

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

## Downstream parameter blocks

| Rule | Parameters | Constraints / effective defaults |
| --- | --- | --- |
| `create_nuclei_amplicon_overlay` | `maximum_projection` | boolean, default false; the Z maximum of the composite; separate from top-level projection |
| `stardist_segmentation` | `stardist_base_path`, `stardist_model_name`, `segmentation_input_folder` | strings; the model is `<base>/<name>` (the known model of that name when the base is the weights cache's `stardist` folder); the input folder defaults in rule code to `overlay` and gives the channel role (`overlay`: composite, any other folder: nuclear) |
| `stardist_segmentation` | `prob_thresh`, `nms_thresh` | numbers in [0,1]; equal to the model's `thresholds.json`, they are its stored thresholds, otherwise overrides; omitted, the stored ones |
| `stardist_segmentation` | `rescale`, `expand_labels`, `distance` | booleans and integer >=0, defaults false; `rescale: true` is StarDist `scale` 0.5 in Y and X, rendered on the input grid; `expand_labels` is a planar pixel expansion by `distance`, which `reads_assignment` refuses (expand there instead) |
| `stardist_segmentation` | `target`, `device` | Python only: `nucleus` or `cell` (default `cell` for the `overlay` folder, `nucleus` otherwise) and `cpu` (default) or `cuda` |
| `reads_assignment` | `expand_labels`, `dilation_distance` | boolean, default false, and integer >=0; with `expand_labels`, assign expands the territories once by `dilation_distance` pixels per plane and keeps both masks |

See [downstream inputs and limitations](workflow-downstream.md) before enabling
these flags. Disabled sections in the full example do not validate model paths
or exercise those scripts. Unknown keys of the `stardist_segmentation` and
`reads_assignment` parameters raise when the rule runs.

## Overrides and validation boundaries

Top-level unknown fields and most extra parameter fields are allowed. Unknown
**rule names** and extra **resource keys** are rejected. A spelling error in an
otherwise permissive parameter object can silently leave a script default in
use. The schema does not check path existence, reference membership, barcode
length, channel count, stage prerequisites or backend feature compatibility.

Snakemake combines `--configfile` and CLI `--config` values for Python scripts
and DAG selection; the Python rules read that combined `snakemake.config` and
take no JSON copy. However, for the MATLAB backend `rsf_preparation` serializes
the original file on disk for MATLAB; CLI mode/parameter/path overrides are not
written back to that file. A `.yml` extension is also unsafe because conversion literally
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
