# Workflows and configuration

The sequencing workflow turns round/FOV TIFF directories and a codebook into
per-FOV molecule CSVs. Snakemake schedules whole FOVs or subtiles; load, rotate,
enhance, register, detect, extract and filter are operations **inside** those
jobs, not seven independent Snakemake rules. Cell segmentation and assignment
are separate downstream jobs.

```{toctree}
:maxdepth: 2

workflow-configuration
workflow-downstream
```

This guide follows `workflow/Snakefile`, `workflow/rules/*.smk`,
`workflow/scripts/*` and `workflow/schemas/config.schema.yaml` in this checkout.
The [Python API](api/python.rst) and [MATLAB API](api/matlab.md) describe the
underlying operations. Source links on this page follow the development branch;
use the checkout's files when working at another revision.

## Paths and input contract

The [common rules](https://github.com/wanglab-broad/starfinder/blob/dev/workflow/rules/common.smk)
construct these two roots:

```text
INPUT  = root_input_path / dataset_id / sample_id
OUTPUT = root_output_path / dataset_id / output_id

INPUT/genes.csv                      # headerless gene,barcode rows
INPUT/round1/tile_1/*ch00.tif          # per-channel Z stacks
INPUT/round1/tile_1/*ch02.tif
INPUT/round1/tile_1/*ch01.tif
INPUT/round1/tile_1/*ch03.tif
INPUT/round2/tile_1/...               # through round<n_rounds>
OUTPUT/documents/sample-annotation.csv
```

The annotation CSV must exist **before parsing the workflow**, even for a dry
run or sequencing-only job. Its minimum columns are `sample_id,fov_start,fov_end`;
start/end are inclusive integer FOV indices. For one FOV:

```text
sample_id,fov_start,fov_end
sample,1,1
```

Annotation sample names group FOVs for stitching and aggregation; they need not
be the input directory's `sample_id`. Keep groups non-overlapping and cover each
selected FOV. The workflow reads this CSV directly during startup; it cannot
produce it as an upstream job. Prepare the codebook and acquisition metadata
from the actual experiment before real execution.

## Stages and files

For direct mode, the rule is `rsf_single_fov` in
[registration-py.smk](https://github.com/wanglab-broad/starfinder/blob/dev/workflow/rules/registration-py.smk)
or [registration.smk](https://github.com/wanglab-broad/starfinder/blob/dev/workflow/rules/registration.smk).
It calls `workflow/scripts/rsf_single_fov.py` or `.m` respectively.

| Stage | Python script operation | MATLAB script operation | Data transition |
| --- | --- | --- | --- |
| load | `STARMapDataset.from_config`, `load_codebook`, `FOV.load_raw_images` | `STARMapDataset`, `LoadRawImages` | Round/FOV channel stacks → in-memory images; Python loads the codebook before processing |
| rotate | `FOV.rotate` if angle is nonzero | `LoadRawImages(..., 'rotate_angle', ...)` | Rotated image arrays; no standalone stage file |
| enhance | Optional `enhance_contrast`, `hist_equalize`, `morph_recon` | `EnhanceContrast`, `HistEqualize`, `MorphRecon` | Preprocessed arrays |
| registration | Optional `global_registration`, `local_registration` | `GlobalRegistration`, `LocalRegistration` | Aligned arrays and reference image |
| spot_finding | `find_spots` | `SpotFinding` | Candidate coordinates |
| extraction | `extract_intensities` / `decode_barcodes` | `ReadsExtraction` | Per-round intensities/color calls |
| filtration | `filter_reads` using loaded codebook | `LoadCodebook`, `ReadsFiltration` | Decoded, filtered molecules |

The direct rule declares these outputs, relative to OUTPUT:

- `log/{fovID}_rsf.txt` and `log/sf_scores/{fovID}.txt`;
- `images/ref_merged/{fovID}.tif`;
- `signal/{fovID}_goodSpots.csv`, containing 1-based `x,y,z` and `gene`.

The Python script writes them using `save_ref_merged`, `save_signal`, `save_log`
and `save_score_log`. MATLAB also writes preview images. Benchmarked rules write
`log/benchmark/{rule}/{fovID}[_{n_subtile}].txt`. A reference projection is not a
segmentation label image, and a goodSpots CSV is not a cell-by-gene matrix.

`rsf_preparation` is a local rule that reads `config_path` and writes the same
name with `.yaml` replaced by `.json`; core rules depend on that JSON. It reads
the YAML file from disk, **not** Snakemake's merged config dictionary. Python
scripts use `snakemake.config`; MATLAB scripts read this JSON. See the
[override caveat](workflow-configuration.md#overrides-and-validation-boundaries).

## Modes and backend selection

`backend: matlab` is the default. `backend: python` selects only the Python core
registration and spot-finding rules. It does not replace all downstream code:
`nuclei_registration` still invokes MATLAB. The MATLAB launcher sources
`/broad/software/scripts/useuse`, runs `use Matlab`, then starts MATLAB; a local
MATLAB executable alone does not satisfy that Broad-specific launcher.

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
`create_sample_maf` and `rsf_preparation` are dependency-driven utilities and
are not accepted keys under `rules` by the schema.

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
| `gr_single_fov_subtile` | Config JSON, input codebook and all round/FOV directories | `log/{fovID}_gr.txt`, `images/ref_merged/{fovID}.tif`, `output/subtile/{fovID}/subtile_coords.csv`, `subtile_data_{n}.npz` (Python) or `.mat` (MATLAB) in that subtile directory |
| `deep_create_subtile` | Config JSON and all round/FOV directories | Same reference and subtile files; GR log is not a declared output |
| `lrsf_single_fov_subtile`, `deep_rsf_subtile` | Config JSON, input codebook, one subtile data file | `log/sf_scores/{fovID}_{n}.txt`, `output/subtile/{fovID}/subtile_goodSpots_{n}.csv` |
| `stitch_subtile` | Coordinates and all subtile molecule CSVs | `signal/{fovID}_goodSpots.csv` and `.png` |

The grid contains `sqrt_pieces ** 2` subtiles numbered from 1. Coordinates, data
archives and subtile molecule CSVs are marked temporary. Local execution may
remove them after consumption; the UGER profile sets `notemp: true` to retain
them. Stitching also reads `images/ref_merged/{fovID}.tif` internally for its
preview, although that file is not declared as an input to the stitch rule.
Check it exists when rerunning stitching from saved CSVs alone.

## Small, reproducible dry run

The downloadable [minimal YAML](#minimal-configuration) and
[full YAML](workflow-configuration.md#full-configuration-example) provide
portable examples. The preparation helper validates a template, rewrites paths
to a new external directory, and creates one annotation row, one illustrative
codebook row and empty round/FOV directories. **It creates no TIFFs. Keep `-n`
on every command using these placeholder inputs.** This checks configuration
and DAG construction; use the [Python quickstart](getting-started.md) for actual
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
file, supply real images and the annotation CSV, then dry-run `all` before
removing `-n`. Inspect the job list: a direct one-FOV example schedules config
preparation, `rsf_single_fov`, and `all`; a 2×2 subtile grid schedules one
creation job, four processing jobs, a stitch job, preparation and `all`.
DAG construction only checks that round/FOV directories exist, not whether
their TIFFs have the right channels, dimensions or barcode semantics.

### Minimal configuration

```{literalinclude} examples/workflow-minimal.yaml
:language: yaml
```

{download}`Download the minimal YAML <examples/workflow-minimal.yaml>` or
{download}`the preparation helper <examples/prepare_workflow.py>`.

The older `tests/minimal_config.yaml` and `tests/tissue_2D_test.yaml` remain
historical starting templates, not ready-to-run portable examples. The former
omits fields needed by schema/startup; the latter contains site paths and
commented variants (including a trailing space in `intensity_threshold `).
Use the explicit examples here and check the selected backend's parameters.

## Broad UGER execution

The repository's `config/environment-v9.yaml` describes a Snakemake 9 environment
(Python >=3.11,<3.13) including the cluster-generic executor. It is separate from
the Python package's uv environment; install the package/dependencies into the
actual compute environment when using the Python backend. StarDist uses its
own environment at `envs_path/stardist`.

On a suitably configured Broad host, after editing paths and preparing real data:

```bash
conda env create -f config/environment-v9.yaml
conda activate starfinder-v9
# From the repository root, with a real, fully edited config:
snakemake -s workflow/Snakefile --configfile /absolute/config.yaml --cores 1 -n all
snakemake -s workflow/Snakefile --configfile /absolute/config.yaml \
  --profile profile/broad-uger --workflow-profile profile/broad-uger \
  --jobscript profile/broad-uger/broad-jobscript.sh all
```

The explicit `--jobscript` is intentional: the checked-in profile puts a
`jobscript` string under `default-resources`, which does not select Snakemake's
jobscript template. Review a site-local copy of the template before submission:
it activates `/stanley/WangLab/envs/starfinder-v9` and loads UGER, Anaconda3 and
MATLAB through Broad `useuse`. Profile submit/status paths are relative to the
working directory; running from another directory requires absolute paths in
a local profile copy. This guide does not certify submission on other clusters.

The [profile](https://github.com/wanglab-broad/starfinder/tree/dev/profile/broad-uger)
sets `executor: cluster-generic`, conda deployment, `jobs: 1000`, one local core,
60-second latency wait, three retries, keep-going, rerun-incomplete and retention
of temporary files. Choose a smaller `--jobs` limit appropriate to your account.
`rsf_preparation` and `create_sample_maf` always run locally.

`broad-submit.py` reads job properties, submits via `qsub -P broad`, uses an SMP
parallel environment for multithreaded jobs, divides total `mem_mb` by threads
for `h_vmem`, and converts runtime minutes to `h_rt`. It requests reservation
above 15 GiB or at four or more threads. Logs use a rule log, `params.uger_log`,
or `~/snakemake-logs/{rule}/`. `broad-status.py` consults `qstat` then `qacct`;
missing accounting information is treated as still running. Scheduler access,
account/project allocation, tool modules, environment paths, model files and
licenses must be checked on the submission host. A local dry run does not test
them or validate cluster resource sufficiency.
