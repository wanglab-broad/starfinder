# Checkpoints and run records

`FOV.run` can save three per-FOV checkpoints and a small run record. They let you
rerun decoding or read filtering without images, or rerun detection and
extraction without loading and registering again. Checkpoints use the existing
TIFF, CSV and typed-result formats. They are development aids, not a final
output format: the workflow's molecule CSVs and images stay as they are.

## Turn checkpoints on

```python
from starfinder.dataset import CheckpointConfig

fov = dataset.fov("Position001")
fov.run(pipeline, execution=execution, checkpoints=CheckpointConfig())
```

`checkpoints=None`, the default, writes nothing and leaves `run` unchanged.
{py:class}`~starfinder.dataset.CheckpointConfig` fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `stages` | `("registered", "candidates", "pre_qc")` | Stages to write. |
| `directory` | `None` | Checkpoint root; `None` means `<output_root>/checkpoints`. |
| `table_format` | `"csv"` | `"csv"` or `"parquet"` (requires the `checkpoint` extra, `pyarrow>=15`). |
| `hash_inputs` | `True` | Record a streamed SHA-256 of every loaded TIFF. |
| `overwrite` | `False` | Allow an existing FOV checkpoint directory. |

`run` checks everything before it loads or processes any image. It raises
`FileExistsError` if the FOV directory exists and `overwrite` is false, and
`ImportError` if Parquet is requested without pyarrow. With `overwrite=True`,
files are replaced as they are written. Nothing is deleted, so only the files
listed in `run.json` belong to that run.

`run` writes a stage only when it computes it. `registered` is written for each
round inside the round loop, `candidates` after detection or extraction, and
`pre_qc` after decoding.

## Layout

```text
<output_root>/checkpoints/<fov_id>/          # or <directory>/<fov_id>/
    run.json
    registered/
        <round>.tif                           # ZYXC, dtype preserved, ImageMetadata
        <round>_field.npz                     # dense transforms only
        transforms.json
    candidates.csv | candidates.parquet
    candidates.json
    pre_qc.csv | pre_qc.parquet
    pre_qc.json
```

An FOV loaded from a subtile writes to `subtile_<n>/` inside its FOV directory,
so subtiles of one FOV never share files. Round labels must be plain file names.

### registered

`registered/<round>.tif` is each round exactly as it enters spot finding and
extraction: after preprocessing, registration and any post-registration
reconstruction. It is written with {py:func}`~starfinder.io.save_volume` and read
with {py:func}`~starfinder.io.load_volume_zyxc`, which keeps singleton Z and C,
the dtype and the stored {py:class}`~starfinder.image.ImageMetadata`.

`transforms.json` records the FOV identity, the rounds and the channel order,
the rounds written, the registration attempts and every `RegistrationResult`:
its transform, diagnostics and warp configuration. Translations are stored
inline. Dense displacement fields go to `<round>_field.npz`, one array per
registration result of that round (`result_0`, `result_1`, ...), with their
float32 or float64 dtype.

### candidates

This is a wide table with one row per spot, in the spot table's order:

| Columns | Type | Content |
| --- | --- | --- |
| `spot_namespace`, `spot_id` | string | Identity; joins use both, never row position. |
| `z`, `y`, `x` | float64 | Zero-based voxel coordinates. |
| `channel`, `peak_intensity`, ... | int64 or float64 | Detector columns, when present. |
| `sig_<round>_<channel>` | float64 | Extracted sum for each round, then each channel in channel order. |
| `valid_<round>` | bool | False marks an unavailable measurement, not zero signal. |

The rounds follow the extraction round labels, which are the `RoundState`
sequencing rounds in a pipeline run. The table is built and split with array
reshapes; there is no per-row work. Labels that would produce duplicate column
names are rejected. When detection ran without extraction, the table has no
signal columns and reloads with `intensity_result=None`.

`candidates.json` holds the detector and extraction configurations, the image
metadata, the JSON-representable diagnostics and the column dtype map.

### pre_qc

`pre_qc.<format>` is `BarcodeDecodingResult.table`, unchanged. `pre_qc.json`
holds the decoder configuration, labels and diagnostics. Array and table
diagnostics are not saved: decoder probabilities, per-round and candidate tables,
and WTA per-round scores. A reloaded result therefore lacks those keys.

## Table formats

CSV is written with floats as `%.17g` and missing values as `<NA>`. It is read
back with an explicit dtype map from the stage header: pandas `string`, nullable
`Int64` or `boolean`, and float64. It is then cast to the recorded in-memory
dtypes. Empty strings, missing values and infinite scores survive, and floats
round-trip exactly. A string value equal to `<NA>` cannot be stored in CSV, and
writing one raises `ValueError`. Parquet stores the same columns and is cast
through the same dtype map, so CSV and Parquet reload to identical typed
results. Empty tables keep their columns and dtypes.

## Reload and continue

{py:meth}`~starfinder.dataset.FOV.load_checkpoint` restores one stage into an FOV
that has no results at or after that stage. It checks the FOV id (and subtile
id), the round labels and the channel order against the dataset, and raises
`ValueError` on a mismatch. Then call `run` with a `PipelineConfig` that starts
after the loaded stage:

```python
from starfinder.dataset import PipelineConfig

# Decode and filter again, without images.
fov = dataset.fov("Position001").load_checkpoint("candidates")
fov.run(PipelineConfig(decoding=decoder, filtering=read_filter))

# Only filter again, with a different predicate.
fov = dataset.fov("Position001").load_checkpoint("pre_qc")
fov.run(PipelineConfig(filtering=other_filter))

# Detect and extract again from registered images.
fov = dataset.fov("Position001").load_checkpoint("registered")
fov.run(PipelineConfig(detection=detector, extraction=extraction,
                       decoding=decoder, filtering=read_filter))
```

When no image operation is configured and no images are resident, `run` skips
the round loop. Resuming is always an explicit call; there is no scheduler.
{py:meth}`~starfinder.dataset.FOV.save_checkpoint` writes one stage from the
current results outside `run`. It uses `CheckpointConfig.directory`,
`table_format` and `overwrite`. {py:func}`~starfinder.io.read_checkpoint` reads a
stage directly and returns a dict keyed by the FOV attributes it restores.

## run.json

`run.json` is replaced atomically (a temporary file, then `os.replace`) when the
run starts, after each completed step and when the run ends. It contains:

| Field | Content |
| --- | --- |
| `format_version` | `1`. |
| `dataset_id`, `sample_id`, `fov_id`, `subtile_id` | Identity. |
| `status`, `started_at`, `ended_at` | `running`, `succeeded`, `failed` or `interrupted`, with UTC times. |
| `error` | `null`, or the failing `step` and `round`, the exception `type`, `message` and `traceback`. |
| `code` | Package `version`, `git_commit` and `git_dirty`; each is `null` when unknown. |
| `environment` | Python, platform and package versions (`null` when not installed). |
| `config` | `pipeline`, `execution` and `checkpoints` configurations. |
| `inputs` | Loaded TIFF `path` and streamed `sha256` (`null` with `hash_inputs=False`). |
| `steps` | `name`, `round`, `seconds` and `status` of each completed or failed step. |
| `registration` | Ordered registration attempts per round. |
| `counts` | Spots, intensities, decoding call statuses and filtering counts. |
| `checkpoint_directory`, `checkpoints` | The FOV directory and the files written for each stage. |

If a step raises, `run` records `failed` (or `interrupted` for
`KeyboardInterrupt` and other non-`Exception` errors), with the innermost failing
step and its round, and re-raises the original exception. If that final write
fails, the failure is logged and the original exception still propagates.
Stages written before the failure stay usable. The codebook file is not recorded
in `inputs`, because `Dataset.load_codebook` does not keep its path.
