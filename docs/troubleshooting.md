# Troubleshooting

Start with the first traceback, the backend, input shape and dtype, channel
patterns, round names and effective configuration. The [Python recipes](recipes.md)
isolate small operations; the [single-FOV demo](get-started/single-fov-demo.md)
shows the complete molecule-output route.

## Missing dependencies or optional backends

| Symptom | What to do |
| --- | --- |
| `ImportError` naming SimpleITK | Install the `local-registration` extra in the environment used to run the call; see the [backend reference](api/backends.md). |
| `InsufficientLandmarksError` or `RegistrationEstimationError` | Inspect the registration inputs and landmark matches; use the [registration guide](registration-contract.md) to choose a method. |
| Installing SpatialData does not create a store | Request the SpatialData export through the [sample-output guide](sample-output-contract.md); installing an extra alone does not run an export. |
| MATLAB fails before processing | Inspect the printed launcher command, executable path, license and toolboxes on the execution host; see the [launcher settings](workflow-configuration.md#matlab-launcher). |

## Image shape, channel and dtype errors

| Symptom | What to do |
| --- | --- |
| Directory missing or no TIFF matches a channel | Inspect `input_root`, round and FOV names, channel patterns and `image_dir_template`; the default layout is round/FOV. |
| Empty or repeated channel labels | Supply the acquisition's channel patterns through `ImageLoadConfig.channel_labels` or the dataset's `channel_order`. |
| Several files match one channel | Narrow the pattern or supply explicit paths. |
| Cropped image warning | Inspect the load result's `diagnostics`, especially `original_shapes` and `cropped`, and check acquisition alignment. |
| Detection expects a ZYXC image | For a known single-channel ZYX volume, use `volume[..., None]`; first verify the original axes. |
| Global threshold rejects the dtype | Inspect scaling and choose the intended threshold mode in `LocalMaximaConfig`; avoid an arbitrary cast. |
| TIFF has unexpected axes or intensity range | Inspect `ImageLoadConfig` and conversion settings in the [I/O API](api/io.rst). |

The [I/O recipe](recipes.md#read-and-write-image-stacks) provides a small TIFF
round trip for comparison.

## No spots, no genes or unexpected coordinates

Inspect the reference image's intensity range and the spot-finding threshold
mode and value. Compare an isolated spot with the
[detection recipe](recipes.md#detect-spots-in-a-volume).

If no reads survive filtering, compare
`fov.decoding_result.table['observed_color_sequence']` with the codebook.
Check sequencing-round order, channel order, encoding orientation and the
input codebook format. The {doc}`readout guide <readout-contract>` shows the
separate extraction, decoding and filtering results for inspection.

For a one-voxel overlay offset, check the file's coordinate origin.
Python `molecules.csv` and `reads.csv` use zero-based ZYX coordinates;
legacy molecule CSVs use one-based XYZ. Load a legacy file through
{py:func}`~starfinder.assignment.import_legacy_molecules_csv`.
A FOV coordinate also needs placement before comparison on a sample image;
see the {doc}`coordinate reference <api/contracts>`.

## Configuration and workflow startup

Use the [bounded dry run](workflows.md#small-reproducible-dry-run) to inspect the
DAG before processing. A dry run does not inspect image content.

| Symptom | What to do |
| --- | --- |
| Schema validation error | Read the field path in the error and compare with the [workflow reference](workflow-configuration.md). |
| Missing `fovs.csv` for a Python workflow | Create the membership file under the input root with `sample,fov_id` columns; use `selected_fovs` to select members. |
| A Python rule rejects a MATLAB key | Use native `preprocessing`, `registration` and `spot_finding` settings from the [migration guide](migration.md#one-python-configuration-dialect). |
| Only config preparation is scheduled | Request the target `all` in the documented dry-run command. |
| Reference round is not found | Compare `ref_round` with configured round names and actual inputs. |
| A YAML edit seems ineffective | Check the selected rule and resolved configuration; compare with its [parameter reference](workflow-configuration.md). |
| Projection export asks for segmentation settings | Configure segmentation after fusion or choose `route: per_fov` for existing per-FOV assignments; see [sample segmentation](sample-segmentation-contract.md). |

Preserve the failed stage's diagnostics before rerunning. Edit the source YAML,
rather than treating a generated JSON copy as the durable configuration.

## Locate workflow logs

| Location | Use |
| --- | --- |
| Console and `.snakemake/log/` | Find the failed job and first traceback. |
| `OUTPUT/fovs/<FOV>/run.json` | Inspect Python status, steps, counts and registration attempts; a subtile record is below `subtile_<n>/`. |
| `OUTPUT/fovs/<FOV>/nuclei_registration.json` | Inspect the saved morphology registration record. |
| `OUTPUT/log/` | Inspect MATLAB processing logs and Snakemake benchmark records. |

The {doc}`UGER guide <workflow/broad-uger>` explains scheduler logs. The
{doc}`checkpoint reference <checkpoints-reference>` lists Python saved records.

## Documentation builds and local environment

| Symptom | What to do |
| --- | --- |
| `sphinx-build` unavailable | Install the locked `docs` dependency group and run with `uv run --group docs`. |
| `starfinder` cannot be imported | Run from `src/python` in the installed environment; autodoc imports the package. |
| uv cache is read-only | Set `UV_CACHE_DIR` to a writable directory before running uv. |
| Native-library linkage error | Inspect the traceback and interpreter selected by `UV_PYTHON`; compare with the [installation guide](get-started/installation.md). |
| Build warning | Fix the named reference or markup and repeat the strict build. |
| Preview port in use | Choose another port and open the matching localhost URL. |

See [contributing](contributing.md) for build commands. These diagnoses are
source-based help; they do not verify MATLAB or cluster execution.

[Earlier troubleshooting notes](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/troubleshooting.md)
preserve the longer environment and workflow discussion.
