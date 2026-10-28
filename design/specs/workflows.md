# Workflows and configuration

The sequencing workflow turns round/FOV TIFF directories and a codebook into
per-FOV molecule CSVs, and its later rules segment cells and assign molecules to
them, stitch each sample's FOVs, assemble and export the sample and write its
processing report. Snakemake schedules whole FOVs or subtiles; load, rotate,
enhance, register, detect, extract and filter are operations **inside** those
jobs, not seven independent Snakemake rules. Segmentation and assignment are
separate jobs after the molecule tables ([workflow-downstream](workflow-downstream.md)); the
stitching, assembly, export and report blocks are in [workflow-configuration](workflow-configuration.md).
How the workflow relates to the Python API is in [workflow/index](../notes/workflow-api.md); installing
Snakemake is in [get-started/installation](../../docs/get-started/installation.md).

This guide follows `workflow/Snakefile`, `workflow/rules/*.smk`,
`workflow/scripts/*` and `workflow/schemas/config.schema.yaml` in this checkout.
The [Python API](../../docs/api/python.rst) and [MATLAB API](../../docs/api/matlab.md) describe the
underlying operations. Source links on this page follow the development branch;
use the checkout's files when working at another revision.

## Paths and input contract

The [common rules](https://github.com/wanglab-broad/starfinder/blob/dev/workflow/rules/common.smk)
construct these two roots:

```text
INPUT  = root_input_path / dataset_id / sample_id
OUTPUT = root_output_path / dataset_id / output_id

INPUT/genes.csv                      # headerless gene,barcode rows (Python: or the file `codebook` names)
INPUT/fovs.csv                       # Python backend: the FOV membership
INPUT/round1/tile_1/*ch00.tif          # per-channel Z stacks
INPUT/round1/tile_1/*ch02.tif
INPUT/round1/tile_1/*ch01.tif
INPUT/round1/tile_1/*ch03.tif
INPUT/round2/tile_1/...               # through round<n_rounds>
OUTPUT/documents/sample-annotation.csv  # MATLAB backend: the FOV membership
```

The membership file must exist **before parsing the workflow**, even for a dry
run or sequencing-only job; the workflow reads it during startup and cannot
produce it as an upstream job. The Python backend reads `INPUT/fovs.csv`, with
the columns `sample,fov_id` and an optional `position_id` (the stage file's
`PositionID`, required when the stitching block reads a stage file), in row
order; the optional `selected_fovs` list restricts every rule to those FOVs
([configuration](workflow-configuration.md#fov-membership)). For one FOV:

```text
sample,fov_id
sample,tile_1
```

The MATLAB backend reads `OUTPUT/documents/sample-annotation.csv`, whose
minimum columns are `sample_id,fov_start,fov_end` (inclusive integer FOV
indices, named by `fov_id_pattern`), and selects with the subset keys:

```text
sample_id,fov_start,fov_end
sample,1,1
```

Sample names group FOVs for stitching and aggregation; they need not be the
input directory's `sample_id`, which stays part of every path and namespace. A
`fovs.csv` that repeats a row or puts a FOV in two rows raises; for the MATLAB
annotation, keep groups non-overlapping and cover each selected FOV. Prepare the
codebook and acquisition metadata from the actual experiment before real
execution.

The Python backend reads two more input settings
([configuration](workflow-configuration.md#reference-file-and-image-layout)). The
codebook (the direct panel with `readout_mode: direct`) is `INPUT/genes.csv` unless
the key `codebook` (`direct_panel`) names another file, absolute or relative to
INPUT; the sequencing rules and `reads_assignment` read that file, and its genes, in
its order, are the columns of the count matrix. The images are laid out round-first,
`INPUT/<round>/<fov>/`, unless `image_dir_template: "{fov}/{round}"` reads them
FOV-first, `INPUT/<fov>/<round>/`; the channel files inside are found as above. The
MATLAB backend reads `INPUT/genes.csv` and the round-first layout only, and refuses
the three keys.

## Stages and files

For direct mode, the rule is `rsf_single_fov` in
[registration-py.smk](https://github.com/wanglab-broad/starfinder/blob/dev/workflow/rules/registration-py.smk)
or [registration.smk](https://github.com/wanglab-broad/starfinder/blob/dev/workflow/rules/registration.smk).
It calls `workflow/scripts/rsf_single_fov.py` or `.m` respectively.

| Stage | Python script operation | MATLAB script operation | Data transition |
| --- | --- | --- | --- |
| load | `from_workflow_config`, `load_codebook`, `FOV.load_images` | `STARMapDataset`, `LoadRawImages` | Round/FOV channel stacks → in-memory images; Python loads the codebook before processing |
| rotate | `FOV.rotate` if angle is nonzero | `LoadRawImages(..., 'rotate_angle', ...)` | Rotated image arrays; no standalone stage file |
| enhance | Optional `normalize_min_max`, `match_histogram`, `correct_background` | `EnhanceContrast`, `HistEqualize`, `MorphRecon` | Preprocessed arrays |
| registration | `register(RegistrationRecipe(...))`: global then local step, composed and resampled once | `GlobalRegistration`, `LocalRegistration` | Aligned arrays and reference image |
| spot_finding | `find_spots` | `SpotFinding` | Candidate coordinates |
| extraction | `extract_intensities` / `decode_reads` | `ReadsExtraction` | Per-round intensities/color calls |
| filtration | `filter_reads` using loaded codebook | `LoadCodebook`, `ReadsFiltration` | Decoded, filtered molecules |

The direct rule declares these outputs, relative to OUTPUT:

- the processing record: on the Python backend `fovs/{fovID}/run.json`, the FOV's
  run record ([checkpoints](../../docs/checkpoints.md), "run.json"); on the MATLAB backend its logs
  `log/{fovID}_rsf.txt` and `log/sf_scores/{fovID}.txt`;
- the reference image: on the Python backend the file `reference_image` names
  (`fovs/{fovID}/reference.tif` by default; none with `reference_image_import`, whose file
  the rules never write), on the MATLAB backend `images/ref_merged/{fovID}.tif`;
- the FOV's molecule table: on the Python backend `fovs/{fovID}/molecules.csv`, the
  typed, zero-based table of the accepted reads with their identities
  (`spot_namespace, spot_id, z, y, x, gene_id, entry_id, qc_score`); on the MATLAB
  backend `signal/{fovID}_goodSpots.csv`, containing 1-based `x,y,z` and `gene`.

The Python script writes them using `FOV.run` with checkpoints, `save_reference_image` and
`export_spots`; `export_spots` also records the table as `outputs.molecules` of the
FOV's `fovs/{fovID}/run.json`. MATLAB also writes preview images. Benchmarked rules write
`log/benchmark/{rule}/{fovID}[_{n_subtile}].txt`. A reference projection is not a
segmentation label image, and a molecule table is not a cell-by-gene matrix.

### Molecule tables

Each FOV has one molecule table that the downstream rules read (`reads_assignment`, and
`export_sample` with `assembly.assignment: null` or on the route after fusion). Its file
follows from the configuration, the backend and the producer the `ruleorder` of the
`Snakefile` selects, never from whichever file exists (`MOLECULE_TABLE` in `common.smk`,
`_molecule_table` in the package):

| Producer | Backend | File | Reader |
| --- | --- | --- | --- |
| `rsf_single_fov` | Python | `fovs/{fovID}/molecules.csv` (zero-based, identities kept) | `assignment.load_molecules` |
| `rsf_single_fov` | MATLAB | `signal/{fovID}_goodSpots.csv` (one-based `x,y,z,gene`) | `assignment.import_legacy_molecules_csv` |
| `stitch_subtile` (the `subtile` and `deep` modes, or free mode with `gr_single_fov_subtile` or `deep_create_subtile` on) | either | `signal/{fovID}_goodSpots.csv` (one-based) | `assignment.import_legacy_molecules_csv` |

The subtile and deep route stays legacy: subtile spot IDs repeat across subtiles and their
namespaces hold the subtile, so the resident rules write the one-based
`subtile_goodSpots_{n}.csv` without identities, and `stitch_subtile` merges them.

### Projection views and the reference merged image

Projection is an output view, not a preprocessing step: `FOV.run` never
projects, and 2D data are volumes with Z = 1. `ProjectionConfig(axis=...)`
selects one of two views, each a maximum by default:

| View | `axis` | Result | Consumers |
| --- | --- | --- | --- |
| Visualization | `"z"` (default) | ZYX(C) → singleton-Z (1, Y, X[, C]) | 2D figures such as registration overlaps and spot-finding results |
| Inspection | `"channel"` | ZYXC → ZYX | Each FOV's reference merged image, below |

The reference image (Python: `fovs/{fovID}/reference.tif` by default, the file the
dataset's `reference_image` names; MATLAB: `images/ref_merged/{fovID}.tif`) holds the
reference round's detection image, the preprocessing output before registration, reduced
to one channel:

- the channel maximum in merged-image mode, or the selected channel in
  single-channel mode, as ZYX;
- then its Z maximum, as YX, when top-level `maximum_projection` is true.

The dtype is that of the detection image. Both backends write this content, one
YX page per Z plane. MATLAB saves `sdata.registration{ref}`, which the last
registration that ran sets. Only the direct rule's global registration follows
its `ref_img`; its local registration, the subtile rule and the deep rule use
the channel maximum, and so does the direct rule without registration. The
Python rules write the view their rule parameter `reference_image_view` names:
`{mode: max}` (the default) or `{mode: channel, channel: <index or label>}`
(for a MATLAB-backend configuration the adapter picks MATLAB's view, selecting
the rule's zero-based `ref_channel` in single-channel mode); the MATLAB script
passes no channel, so `GlobalRegistration` selects its default channel name,
`DAPI`. The file shows preprocessing before
registration, and is read by `create_nuclei_amplicon_overlay`, by the stitch
preview and, later, as stitching input. MATLAB also writes per-round,
per-channel Z-maximum preview montages; Python does not.

`rsf_preparation`, defined for the MATLAB backend only, is a local rule that
reads `config_path` and writes the same name with `.yaml` replaced by `.json`;
the MATLAB core rules depend on that JSON. It reads the YAML file from disk,
**not** Snakemake's merged config dictionary. The Python rules take no JSON copy:
their inputs are named (`codebook`, `images`, `subtile`), the adapters read them
by name and read the configuration from `snakemake.config`; MATLAB scripts read
the JSON. See the
[override caveat](workflow-configuration.md#overrides-and-validation-boundaries).

## Modes and backend selection

`backend: matlab` is the default. `backend: python` selects the Python core
registration and spot-finding rules and the Python `nuclei_registration` rule
(below). It does not replace all downstream code. `matlab_launcher: path` (the
default) runs a local MATLAB executable; `matlab_launcher: broad` sources
`/broad/software/scripts/useuse` and runs `use Matlab` first. See the
[MATLAB launcher](workflow-configuration.md#matlab-launcher) settings.

### Nuclei registration

`nuclei_registration` registers the `additional_round` rounds (for example
morphology rounds) to the reference round's DAPI stain, one job per FOV. The
MATLAB backend runs `workflow/scripts/nuclei_registration.m`. The Python backend
runs `workflow/scripts/nuclei_registration.py`, which calls
`FOV.prepare_morphology` ([coordination](../../docs/coordination.md)) and keeps the MATLAB inputs and
image file names:

- it reads the reference round's stain file that `reference_stains` declares
  (for example `{channel: ch04, name: DAPI}`; with several entries the one whose
  name contains `ref_channel`), rotated by `rotate_angle`, and registers on it as
  an `ExternalReference` labelled `<ref_round>:<channel>`; the MATLAB script reads
  the reference round's `*ch04.tif`;
- it loads each `additional_round` entry with its `channel_order` (each
  entry's `channel` is the filename pattern and the unique label, which names
  the output folder; MATLAB names the folder by `name`), rotated by
  `rotate_angle`; the stain is the one channel whose
  `name` contains the top-level `ref_channel`, as MATLAB matches it;
- it registers each round by one translation step on its stain and applies the
  transform to all of the round's channels;
- it writes `fovs/<fov>/nuclei_registration.json` (`FOV.save_registration_record`:
  the round roles, the registered rounds and their registration attempts, whose
  estimation entries hold each translation; MATLAB writes its console log and a
  shift table instead) and `images/<round>/<channel label>/<fov>.tif` (ZYX, or the
  Z maximum as YX with top-level `maximum_projection`), one file per channel, also for
  channels that share a `name`. A label that is not a safe single path component
  (`[A-Za-z0-9][A-Za-z0-9_.-]*`) is rejected. On the Python backend the shared rule
  `enhance_dapi_with_flamingo` reads the label folders of the `flamingo` round's channels
  named `DAPI` and `Flamingo`.

Unlike MATLAB, the Python rule does not min–max stretch the other rounds before
registering and saving them: the images keep their loaded dtype and values.
With the Python-only `rules.nuclei_registration.parameters.checkpoints` (a mapping
of `CheckpointConfig` fields, `{}` for the defaults) it also writes the saved form
of each prepared image under `fovs/<fov>/other_rounds/<name>/`, which
`FOV.load_registered_round` reads ([checkpoints](../../docs/checkpoints.md)); the declared files do not
change.

| Mode | Selected core rules | Intermediate files and execution |
| --- | --- | --- |
| `free` (default) | Each configured rule's `run` flag | Use for a deliberate combination or rerun; output requests can still pull in prerequisites whose flags are false |
| `direct` | `rsf_single_fov` | All sequencing stages in one job per FOV |
| `subtile` | `gr_single_fov_subtile` → `lrsf_single_fov_subtile` → `stitch_subtile` | Load/preprocess/global registration, then local registration/detection/extraction/filtering per subtile, then recombine molecules |
| `deep` | `deep_create_subtile` → `deep_rsf_subtile` → `stitch_subtile` | Load/histogram equalization/global registration and split, then reconstruction/detection/extraction/filtering; no local registration step in the deep subtile script |

Preset modes require all named rule sections to exist and ignore their core
`run` flags. Schema validation still requires a boolean `run` in each section.
Preparation, nuclei/rotation, overlay, StarDist, stitching/BigStitcher,
assignment and sample aggregation use their individual flags in all modes.
`enhance_dapi_with_flamingo` and the MATLAB-only `rsf_single_fov_seq` are outside
that always-available list: their flags select aggregate targets only in `free`.
`create_sample_maf` and `rsf_preparation` (MATLAB backend only) are
dependency-driven utilities and are not accepted keys under `rules` by the schema.
The legacy sample rules (the four stitching rules, `create_sample_maf`,
`create_sample_h5ad` and `create_sample_reads_assignment`) read the MATLAB sample
annotation and run with `backend: matlab` only; a Python configuration uses the
stitching and assembly blocks instead, and enabling one of them without its block
raises when the rule files are parsed, naming the block to add.

Flags control the aggregate target list, not whether a rule definition exists.
Use the explicit target **`all`** in commands: the preparation rule is included
before `all`, so relying on the first/default rule can stop at config conversion.
Rules producing the same output are prioritized by `ruleorder`: subtile wins
in subtile mode (or free with GR enabled), deep wins in deep mode (or free with
deep creation enabled), otherwise direct wins. Avoid enabling multiple core
paths to the same FOV output in free mode.

Subtile rule implementations live in `registration[-py].smk` and
`spot-finding[-py].smk`; scripts have the rule's name and `.py`/`.m` extension.
`stitch_subtile.py` is shared by both backends.

| Rule | Additional inputs | Declared outputs relative to OUTPUT |
| --- | --- | --- |
| `gr_single_fov_subtile` | Config JSON, input codebook and all round/FOV directories | `fovs/{fovID}/run.json` and the reference image (Python; none with `reference_image_import`) or `log/{fovID}_gr.txt` and `images/ref_merged/{fovID}.tif` (MATLAB), `output/subtile/{fovID}/subtile_coords.csv`, `subtile_data_{n}.npz` (Python) or `.mat` (MATLAB) in that subtile directory |
| `deep_create_subtile` | Config JSON and all round/FOV directories | Same reference and subtile files, and on the Python backend `fovs/{fovID}/run.json`; MATLAB declares no log |
| `lrsf_single_fov_subtile`, `deep_rsf_subtile` | Config JSON, input codebook, one subtile data file | `fovs/{fovID}/subtile_{n}/run.json` (Python) or `log/sf_scores/{fovID}_{n}.txt` (MATLAB), `output/subtile/{fovID}/subtile_goodSpots_{n}.csv` |
| `stitch_subtile` | Coordinates and all subtile molecule CSVs | `signal/{fovID}_goodSpots.csv` and `.png` |

The grid contains `sqrt_pieces ** 2` subtiles numbered from 1. Coordinates, data
archives and subtile molecule CSVs are marked temporary. Local execution may
remove them after consumption; the UGER profile sets `notemp: true` to retain
them. Stitching also reads the FOV's reference image for its preview (its Y and X
axes, after a maximum over any other axis): on the Python backend the file
`reference_image` or `reference_image_import` names, declared as the rule's input
`reference`; on the MATLAB backend `images/ref_merged/{fovID}.tif`, which is not
declared as an input to the stitch rule. Check it exists when rerunning stitching
from saved CSVs alone.

## Small, reproducible dry run

The downloadable [minimal YAML](#minimal-configuration) and
[full YAML](workflow-configuration.md#full-configuration-example) provide
portable examples. The preparation helper validates a template, rewrites paths
to a new external directory, and creates a one-FOV `fovs.csv`, one illustrative
codebook row and empty round/FOV directories. **It creates no TIFFs. Keep `-n`
on every command using these placeholder inputs.** This checks configuration
and DAG construction; use the [Python quickstart](../../docs/getting-started.md) for actual
small image processing.

Run from `src/python`, using absolute paths for `RUN` and `REPO`:

```bash
export PYTHONDONTWRITEBYTECODE=1
export UV_PYTHON=/path/to/python3.12
REPO=$(cd ../.. && pwd)
RUN=/absolute/external/new-workflow-check
uv run --with 'snakemake>=9,<10' python ../../docs/examples/prepare_workflow.py "$RUN"
export XDG_CACHE_HOME="$RUN/cache"
uv run --with 'snakemake>=9,<10' python -m snakemake \
  -s "$REPO/workflow/Snakefile" --configfile "$RUN/config.yaml" \
  --directory "$RUN" --cores 1 -n all
```

For `subtile` or `deep`, prepare a **new** run directory with
`--template full --mode subtile` or `--template full --mode deep`, then use the
same dry-run command. `--mode free` checks the full template's direct-only flags.
The helper rejects existing output directories. `--directory` keeps Snakemake's
working files outside the checkout; `XDG_CACHE_HOME` also redirects its source
cache. An installed Snakemake 9 environment can
use `python -m snakemake` directly instead of the transient `--with` dependency.
Snakemake is separate from the `docs` dependency group.

For real inputs, copy/edit the YAML, point `config_path` at that exact `.yaml`
file, supply real images and the membership file (`fovs.csv` for the Python backend, the annotation CSV for
MATLAB), then dry-run `all` before
removing `-n`. Inspect the job list: a direct one-FOV example schedules config
preparation, `rsf_single_fov`, and `all`; a 2×2 subtile grid schedules one
creation job, four processing jobs, a stitch job, preparation and `all`.
DAG construction only checks that round/FOV directories exist, not whether
their TIFFs have the right channels, dimensions or barcode semantics.

### Minimal configuration

```{literalinclude} ../../docs/examples/workflow-minimal.yaml
:language: yaml
```

[Download the minimal YAML](../../docs/examples/workflow-minimal.yaml) or
[the preparation helper](../../docs/examples/prepare_workflow.py).

The older `tests/minimal_config.yaml` and `tests/tissue_2D_test.yaml` remain
historical starting templates, not ready-to-run portable examples. The former
holds placeholder paths and needs a sample annotation at startup; the latter contains site paths and
commented variants (including a trailing space in `intensity_threshold `).
Use the explicit examples here and check the selected backend's parameters.

## Broad UGER execution

Running the workflow on the Broad UGER cluster with `profile/broad-uger/`, its
environment, commands, settings, submission and logs, is described on the profile page
[workflow/broad-uger](broad-uger.md).

