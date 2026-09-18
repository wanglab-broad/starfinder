# Troubleshooting

Start with the failing call or first traceback, the selected backend, input
shape/dtype, channel patterns, rounds and effective configuration. The
[development recipes](recipes.md) isolate small Python operations; the
[quickstart](getting-started.md) checks the complete molecule-output path.
The diagnoses below follow the current source. They do not certify real data,
MATLAB runtime or cluster execution.

## Missing dependencies or optional backends

| Symptom | Check and supported action |
| --- | --- |
| `ImportError` mentioning SimpleITK from demons or `apply_deformation` | From `src/python`, install the declared extra with `uv sync --extra local-registration`, then use the same uv environment to run. Installation requires registry/cache access. Global registration does not need this extra. |
| TPS/CPD cannot estimate a transform | Inspect `InsufficientLandmarksError` or `RegistrationEstimationError`. FOV and direct estimation propagate errors; there is no default fallback. |
| Installing SpatialData or napari does not change output files | These packaging extras do not enable a public SpatialData writer or automatic viewer. FOV outputs are TIFF/CSV/text/NPZ. Inspect CSVs with the [output recipe](recipes.md#inspect-molecule-outputs). |
| MATLAB launcher fails before image processing | Check `/broad/software/scripts/useuse`, `use Matlab`, license and required toolboxes on the execution host. The workflow launcher depends on Broad environment setup; finding a `matlab` executable alone is insufficient. See [MATLAB requirements](api/matlab.md#runtime-requirements-and-validation-limits). |

See [backend behavior](api/backends.md) for direct versus runner dependencies.
In particular, the registration benchmark runner's warp step uses SimpleITK
even for TPS/CPD, unlike their direct warp functions. A strict docs build only
imports/parses APIs; it does not execute these optional algorithms.

## Image shape, channel and dtype errors

| Symptom | Diagnosis and next check |
| --- | --- |
| `Directory not found` or `No TIFF file found matching channel pattern` | Inspect the resolved `input_root/round/FOV` and actual filenames. {py:func}`~starfinder.io.load_round` matches `*{pattern}*.tif`; `.tiff` is also supported, but a wrong suffix/layout will not match. The synthetic generator's FOV/round layout needs the quickstart's explicit layout conversion. |
| `min() arg is an empty sequence` while loading | An empty `channel_order` reaches the channel loader without a default. Supply a nonempty, acquisition-correct list; check `seq_channel_order` when using {py:meth}`~starfinder.dataset.STARMapDataset.from_config`. |
| Ambiguous-file error or unexpected channel intensities | Narrow each pattern to exactly one file, or provide explicit source paths. Check [channel order](conventions.md#channel-order) against the codebook. |
| Channel-size mismatch warning or smaller-than-expected image | The loader crops to minimum Z/Y/X dimensions. Inspect `metadata['original_shapes']` and `metadata['cropped']`; verify acquisition/export alignment before accepting the crop. Cropping is not registration. |
| `Expected 4D (Z, Y, X, C) image` in detection | Inspect `image.shape`. For a known single-channel ZYX volume, append a singleton channel axis with `volume[..., None]`. A YX projection or unknown axis order cannot be repaired by blindly appending dimensions. |
| Shape unpacking/broadcast error during global registration | The single-channel registration call expects equal-shaped 3D ZYX inputs. Check both shapes and the intended channel/merge; the function does not provide full input-shape validation. |
| `Global mode requires uint8/uint16` | The global threshold uses dtype maximum. Check whether your input is floating point and choose an explicitly justified threshold mode or scaling step; casting arbitrary floats to uint8 can discard signal. |
| TIFF loaded with wrong dimensions or intensity range | Plain TIFFs default to ZYX; ambiguous OME/ImageJ axes require explicit selection. Inspect `ImageLoadConfig` and result diagnostics. Dtype is preserved unless an explicit conversion is configured. |

The [I/O recipe](recipes.md#read-and-write-image-stacks) checks a lossless uint16
round-trip. It does not validate arbitrary multi-channel acquisition formats.

## No spots, no genes or unexpected coordinates

An empty detection table is a possible outcome, not necessarily an exception.
Check the reference image's range, all-zero channels, the mode/threshold pair,
and excluded border (`min_distance`). Do not mix noise k-sigma values with
adaptive fractions. Python rejects the schema-accepted `local` spot mode;
see [threshold conventions](conventions.md#spot-finding-thresholds).

If filtering raises `Codebook not loaded`, call
{py:meth}`~starfinder.dataset.STARMapDataset.load_codebook` first. If candidates
exist but none survive, inspect `all_spots['color_seq']` alongside
`dataset.codebook.seq_to_gene`: check channel ordering, sequencing-round order,
barcode orientation (`EncodingConfig.reverse_bases`), sequence lengths and optional suffix
filtering. `M` is a tied maximum and `N` a NaN maximum; they are not valid
four-color gene calls. A codebook parser `KeyError` can indicate an unexpected
header or unsupported nucleotide pair. Use the actual two-column `gene,barcode`
format described by {py:func}`~starfinder.barcode.load_codebook`.

`save_signal` raises `No spots in '...' to save` for an empty or absent table.
Inspect counts **before saving** and retain diagnostic state/logs; do not invent
molecules to satisfy the output contract. Its default columns omit extraction
scores/colors; pass `columns=list(fov.all_spots.columns)` to save candidate
diagnostics, as in the recipe. Calling extraction before detection or omitting
a configured round leaves required columns/images unavailable; follow the
[one-FOV call order](recipes.md#decode-one-fov).

A one-voxel overlay offset often warrants checking origins: signal CSVs are
1-based XYZ, Python indexing is 0-based ZYX. Verify that conversion exactly
once and check bounds against the loaded image. A subtile coordinate also needs
its crop offset before full-FOV comparison; a FOV coordinate is not a stitched
sample coordinate. See [conventions](conventions.md#axes-and-coordinates).
These are diagnostic possibilities, not a diagnosis of every shifted overlay.

## Configuration and workflow startup

Use the [minimal/full documentation templates](workflows.md#small-reproducible-dry-run)
and their bounded **dry-run** helper. Keep `-n` for the helper's empty placeholder
image directories. A successful DAG does not inspect TIFF content.

| Symptom | Check |
| --- | --- |
| Schema validation error | Read the failing field path. Each provided rule needs boolean `run`; unknown rule/resource keys are rejected. Subset conditionals require companion fields even for false-valued flags. See [top-level fields](workflow-configuration.md#top-level-fields). |
| Startup `KeyError`, even with downstream flags off | Rule parsing indexes fields including `envs_path`, `fiji_path`, `additional_round` and `subset_list`. Schema-optional does not mean every script can omit a key; the minimal documentation template supplies these. |
| Only config preparation is scheduled | Request explicit target `all`; the preparation rule occurs before it in the Snakefile. Use the [documented dry-run command](workflows.md#small-reproducible-dry-run). |
| Missing direct/subtile/deep rule sections | Preset modes need their named sections and select them regardless of their `run` flags. See [mode selection](workflows.md#modes-and-backend-selection). |
| Reference-round `KeyError` | Check `ref_round` belongs to the loaded rounds, and their names match exactly. In the direct API, call {py:meth}`~starfinder.dataset.LayerState.validate` explicitly; it checks membership/overlap but does not check that TIFFs exist or require a non-null reference. |
| A YAML edit or CLI override seems ineffective | Check exact key spelling and the selected wrapper. Unknown parameter fields can pass validation. Python wrappers do not forward every API option; MATLAB receives JSON serialized from the original YAML, not all merged CLI values. See [overrides](workflow-configuration.md#overrides-and-validation-boundaries). |
| Input directories exist but the job fails | DAG inputs are round/FOV directories, not validated TIFF inventories. Check files, channels, codebook and backend compatibility. Schema validation does not check reference membership or scientific parameter suitability. |

Keep one resolved `.yaml` and matching `config_path`; preparation writes its
sibling JSON. Do not hand-edit generated JSON as a durable configuration repair.
Before rerunning, identify the failed stage and preserve its diagnostics.

## Locate workflow logs

For workflow runs, `OUTPUT` is
`root_output_path/dataset_id/output_id`; direct Python recipes use their explicit
`output_root`. These paths have different purposes:

| Location | What it records |
| --- | --- |
| Console / Snakemake working directory `.snakemake/log/` | Scheduling and execution diagnostics; start here for the failed job and traceback |
| `OUTPUT/log/{FOV}_rsf.txt` or `{FOV}_gr.txt` | FOV processing summaries, when the relevant stage writes them; may be absent on early failure |
| `OUTPUT/log/gr_shifts/{FOV}.txt` | Detected global shifts, CSV content despite `.txt` suffix |
| `OUTPUT/log/sf_scores/{FOV}_{subtile}.txt` | Per-subtile spot/filter diagnostics declared by subtile processing rules |
| `OUTPUT/log/benchmark/{rule}/...txt` | Snakemake timing/resource records; these are not the Python exception log |
| UGER submit log | `broad-submit.py` chooses the first declared rule log with `-qsub.log`, else `params.uger_log` with timestamp/job ID, else `/home/unix/{username}/snakemake-logs/{rule}/` |

UGER submission combines stdout/stderr with `qsub -j y -o <logfile>`. A rule's
output named `log/...` is not necessarily a Snakemake `log:` directive, so
check the actual job properties and fallback directory. See the checked-in
[submit helper](https://github.com/wanglab-broad/starfinder/blob/dev/profile/broad-uger/broad-submit.py)
and [UGER guide](workflows.md#broad-uger-execution). Log routing is source-checked;
submission, accounting and cluster recovery have not been exercised here.

## Documentation builds and local environment

- **`sphinx-build` is unavailable:** run `uv sync --locked --no-default-groups --group docs`
  from `src/python`, and include `--group docs` in the build command.
- **`starfinder` cannot be imported:** build from `src/python` with `uv run` so the
  checkout and its runtime dependencies are installed. Autodoc uses real imports.
- **uv cache is read-only:** set `UV_CACHE_DIR` to a writable external directory
  before running uv. A new cache may need package-registry access; it does not
  repair missing runtime libraries.
- **System MKL linkage error:** the recorded host's default Python 3.14 environment
  failed at native-library loading. The validated examples use `UV_PYTHON=python3.12`
  (or an absolute interpreter path), with actual version 3.12.12. Check
  `uv run python --version` and the traceback. This host-specific workaround
  changes no dependency pin and does not establish a general Python 3.14 defect.
- **A warning makes the build fail:** read the file and line in the build output.
  Fix the reference or markup, then rebuild into a fresh external output directory.
- **Preview port is in use:** choose a different port in the `http.server` command
  and open the matching URL. Stop the server with Ctrl-C when finished.

See [contributing](contributing.md) for the exact strict build and preview commands.
