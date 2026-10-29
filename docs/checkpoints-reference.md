# Checkpoint reference

Files and fields written by the checkpoint APIs. Start with the
{doc}`checkpoint guide <checkpoints>` for a runnable example.

## Configuration

Checkpoint options come from {py:class}`~starfinder.dataset.CheckpointConfig`.

```{config-options} starfinder.dataset.CheckpointConfig
```

## Layout

Paths below are relative to a FOV folder, normally
`<output_root>/fovs/<fov_id>/`. A subtile uses its `subtile_<n>/` folder.

| Path | Written by / contents |
| --- | --- |
| `run.json` | `FOV.run` with checkpoints; processing record |
| `molecules.csv`, `reads.csv` | `FOV.export_spots`; molecule or read table |
| `reference.tif` | `FOV.save_reference_image`; default reference-image path |
| `nuclei_registration.json` | `FOV.save_registration_record`; other-round registration attempts |
| `checkpoints/registered/` | Processed images and transforms |
| `checkpoints/candidates.csv` or `.parquet`, `candidates.json` | Spots, signals and header |
| `checkpoints/pre_qc.csv` or `.parquet`, `pre_qc.json` | Reads and header |
| `other_rounds/<name>/image.ome.tif`, `registration.json`, optional `field.npz` | Prepared morphology image and its record |
| `segmentation/<run>/labels.tif`, `segmentation.json` | Saved segmentation; computed runs also have `input.ome.tif` |
| `assignment/<name>/` | Assignment tables, label files and `assignment.json`; see the {doc}`assignment reference <assignment-algorithms>` |

## registered

| File / field | Contents |
| --- | --- |
| `<round>.ome.tif` | Processed round, ZYXC with metadata |
| `<snapshot>/<round>.ome.tif` | Saved extraction-source snapshot |
| `<round>_field.npz` | Dense displacement or B-spline coefficient arrays, when present |
| `transforms.json` | Header with grids, configs, registration results, chains, applications and payload hashes |

## candidates

| Column / file | Contents |
| --- | --- |
| `spot_namespace`, `spot_id` | Spot identity |
| `z`, `y`, `x` | Zero-based coordinates |
| Detector columns | Measurements retained from the spot table |
| `sig_<round>_<channel>` | Extracted sums, when extraction ran |
| `valid_<round>` | Extraction validity |
| `bg_<round>_<channel>`, `noise_<round>_<channel>` | Local background and noise, when measured |
| `bgvox_<round>`, `boxvox_<round>` | Background-ring and extraction-box voxel counts, when background is measured |
| `candidates.json` | Identity, labels, metadata, config, diagnostics, dtypes, extraction settings and payload hashes |

(checkpoint-pre-qc)=
## pre_qc

| Column / file | Contents |
| --- | --- |
| `spot_namespace`, `spot_id` | Spot identity |
| Read columns | Decoding, scoring and deduplication columns; see the {doc}`readout reference <readout-algorithms>` |
| `pre_qc.json` | Identity, readout mode, labels, encoding, decoder settings, diagnostics, dtypes and payload hashes |

## Table formats

| Format | Stored representation |
| --- | --- |
| CSV | Header row, `%.17g` floats, `<NA>` for missing values, dtype map in the record |
| CSV strings | Literal missing-value markers and leading backslashes escaped by a leading backslash |
| Parquet | Typed columns; requires pyarrow |

The [format specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/checkpoints.md#table-formats)
describes string handling and exact round trips.

(checkpoint-run-record)=
## run.json

The run record's format version is 2. Fields are grouped below for lookup.

| Key | Contents |
| --- | --- |
| `format_version` | Run-record format version |
| `dataset_id`, `sample_id`, `fov_id`, `subtile_id` | Processing identity |
| `input_root`, `output_root` | Roots recorded for provenance |
| `status` | Processing outcome: `running`, `succeeded`, `failed` or `interrupted` |
| `started_at`, `ended_at` | UTC timestamps |
| `error` | Failure information, or null |
| `code` | Package and checkout identity |
| `environment` | Python, platform and package versions |
| `config` | Pipeline, execution, checkpoints, readout mode and encoding |
| `workflow_config` | Workflow configuration path and hash, or null |
| `channels` | Channel pattern, name and wavelength per round |
| `rounds` | Sequencing rounds, other rounds and reference round |
| `inputs` | Loaded image paths and optional hashes |
| `codebook` | Codebook or direct-panel source, counts and hash, or null |
| `steps` | Step names, rounds, elapsed seconds, status and method provenance |
| `preprocessing` | Recipe, per-round steps, snapshots, transforms and supplied statistics |
| `registration` | Registration attempts and applications |
| `transforms` | Per-round transforms and application settings |
| `counts` | Spot/read counts, read summaries and filter fractions |
| `checkpoint_directory` | Checkpoint folder path |
| `checkpoints` | Stage record paths and hashes |
| `outputs` | Files added by output writers, with paths, hashes and table details where applicable |

The [run-record specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/checkpoints.md#runjson)
gives nested fields, path resolution and version compatibility.
