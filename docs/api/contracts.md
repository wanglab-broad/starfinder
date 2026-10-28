# Array and coordinate contracts

Look up Python array, coordinate, channel and output conventions here. The
{doc}`API <python>` gives function signatures; stage-specific methods and options
are in the {doc}`stage references <../wiki/pipeline-stages>`.

## Arrays

| Object | Shape | Interpretation |
| --- | --- | --- |
| Single-channel image | `(Z, Y, X)` | Intensity volume; a plane has singleton Z |
| Multi-channel round | `(Z, Y, X, C)` | Channel-last intensity volume |
| Z projection | `(1, Y, X)` or `(1, Y, X, C)` | Singleton Z is retained |
| Channel projection | `(Z, Y, X)` | Channel axis reduced |
| Spot coordinates | `(N, 3)` or named `z, y, x` columns | Zero-based voxel indices |
| Extracted intensities | `(N, C, R)` | Float64 sums; C and R follow their labels |
| Extraction validity | `(N, R)` | Boolean flags |
| Dense displacement | `(Z, Y, X, 3)` | Components `(dz, dy, dx)` |

## Registration transforms

The {doc}`registration reference <../registration-algorithms>` lists the transform
classes and the {doc}`registration guide <../registration-contract>` shows an
estimate-and-apply example.

| Stored value | Meaning |
| --- | --- |
| `TranslationTransform.displacement_zyx` | Detected displacement of moving content in ZYX voxel indices |
| `DenseDisplacementTransform.displacement_zyx` | Pull displacement at reference-grid points, in ZYX voxel indices |
| `AffineTransform.matrix_zyx` | Homogeneous index-space pull matrix |
| `reference_metadata`, `moving_metadata` | The two grids described by a transform |
| `RegistrationResult.application_config` | Application settings returned with an estimate |

See the [transform specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/registration.md#transform-types)
for composition, interpolation and boundary rules.

### Translation edge cases and resampling precision

See the [application specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/array-coordinate-contracts.md#translation-edge-cases-and-resampling-precision).

## Intensities and channels

| Interface | Result / convention |
| --- | --- |
| `io.load_volume` | `ImageLoadResult` with ZYX `image`, `metadata`, `channel_labels`, `source_paths` and `diagnostics` |
| `io.load_round` | `ImageLoadResult` with ZYXC `image`, channels in the supplied pattern order |
| `io.convert_image` | Explicit intensity conversion configured with `ImageConversionConfig` |
| `preprocessing.normalize_min_max` | Intensity normalization; see the {doc}`preprocessing reference <../preprocessing-algorithms>` |

## Geometry and derived frames

{py:class}`~starfinder.image.ImageMetadata` describes space separately from image
intensities. Unknown calibration is represented by `None`.

| Field | Meaning |
| --- | --- |
| `frame_id` | Coordinate-frame identifier |
| `spacing_zyx` | Voxel spacing in ZYX order |
| `origin_zyx` | Physical origin |
| `direction_zyx` | Direction matrix |
| `spatial_unit` | Physical unit |

The {doc}`image API <image>` provides index/physical conversion. Crop, rotation and
projection geometry are described in the
[full conventions](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/array-coordinate-contracts.md#geometry-and-derived-frames).

### Before and after

See the {doc}`migration guide <../migration>` for replaced Python calls.

## Spots, reads and output

| Object / file | Fields to look up |
| --- | --- |
| `SpotFindingResult` | `spots`, `metadata`, `spot_namespace`, `config`, `diagnostics` |
| Spot table | `spot_id`, `z`, `y`, `x`; method-dependent measurements in the {doc}`spot-finding reference <../spot-finding-algorithms>` |
| Spot identity | `(spot_namespace, spot_id)` |
| `IntensityExtractionResult` | `values`, `valid`, spot identities, round/channel labels and background measurements |
| `molecules.csv` from `FOV.export_spots()` | `spot_namespace, spot_id, z, y, x, gene_id, entry_id, qc_score` |
| `reads.csv` from `FOV.export_spots("reads")` | Reads joined to their detections, including rejected reads |
| Legacy `goodSpots` CSV | One-based `x, y, z, gene` |

### Independent readout stages

| Stage | Public function | Result |
| --- | --- | --- |
| Extraction | `extract_intensities` | `IntensityExtractionResult` |
| Decoding | `decode_reads` | `ReadDecodingResult` |
| Direct readout | `decode_direct` | `ReadDecodingResult` |
| Scoring | `score_reads` | `ReadScoringResult` |
| Deduplication | `deduplicate_reads` | `ReadDeduplicationResult` |
| Filtering | `filter_reads` | `ReadFilteringResult` |

The {doc}`readout reference <../readout-algorithms>` lists records and configuration
classes; the {doc}`readout guide <../readout-contract>` shows the workflow.

## MATLAB boundary

The {doc}`MATLAB reference <matlab>` describes the MATLAB interfaces.

### Axes and coordinates

| Quantity | Python | MATLAB / legacy CSV |
| --- | --- | --- |
| Image axes | ZYXC; ZYX for one channel | YXZC; YXZ for one channel |
| Spot coordinates | Zero-based `z, y, x` | One-based `x, y, z` |
| Channel index | Zero-based | One-based |
| Four-channel color labels | Strings `1`–`4` | Strings `1`–`4` |
| Extraction half-widths | `neighborhood_radius_zyx=(dz, dy, dx)` | `voxel_size=(row, column, Z)` |

`assignment.import_legacy_molecules_csv` converts the legacy molecule coordinates.
The {doc}`output recipe <../recipes>` shows how to inspect Python molecule tables.

### Channel order

| Context | Channel order |
| --- | --- |
| Python | Supplied channel patterns, in C order |
| Wavelength-sorted MATLAB default | `ch00, ch02, ch01, ch03` |
| Tiny synthetic quick start | `ch00, ch01, ch02, ch03` |

### Registration signs

| Interface | Translation convention |
| --- | --- |
| Python transform and `run.json` | Detected `(dz, dy, dx)`; pull sampling uses `moving[p + displacement_zyx]` |
| MATLAB `DFTRegister3D` → `DFTApply3D` | Correction parameters in `(row, column, Z)` order |

### Spot-finding thresholds

| Interface | Configuration |
| --- | --- |
| Python API / Python workflow | `LocalMaximaConfig.threshold_mode` and `threshold_value` / the `spot_finding` block |
| MATLAB workflow | `intensity_estimation` and `intensity_threshold` |

The {doc}`spot-finding reference <../spot-finding-algorithms>` links the Python
configuration classes. See the
[boundary specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/array-coordinate-contracts.md#matlab-boundary)
for conversion details and MATLAB forwarding behavior.
