# Array and coordinate contracts

These conventions describe the current Python implementation. Individual
[generated signatures](python.rst) give defaults and more specific requirements.
Arrays are accompanied by optional `ImageMetadata`; distances below are in voxel-index
units unless a function explicitly says otherwise. Unknown calibration stays unknown.

| Object | Shape / columns | Convention |
| --- | --- | --- |
| Single-channel volume | `(Z, Y, X)` | Numeric intensity array |
| Multi-channel round | `(Z, Y, X, C)` | Channel-last, usually uint8 or uint16 |
| Projection | `(1, Y, X)` or `(1, Y, X, C)` | Singleton Z is retained; check each registration backend’s dimensional support |
| Spot coordinates | `(N, 3)` or DataFrame `z, y, x` | Zero-based voxel indices |
| Raw extracted intensities | `(N, C, R)` | Float64 neighborhood sums, rounds in caller-supplied order |
| Dense displacement field | `(Z, Y, X, 3)` | Last axis `(dz, dy, dx)`, backward sampling |
| Signal CSV | `x, y, z[, gene]` by default | One-based coordinates, written by `FOV.save_signal` |

## Displacement and correction

{func}`starfinder.registration.phase_correlate` and
{func}`starfinder.registration.phase_correlate_skimage` return the **detected
displacement** of moving relative to fixed, `(dz, dy, dx)`. For a moving image
translated by `(1, -2, 3)`, correct it with
`apply_shift(moving, (-1, 2, -3))`. Positive values passed to
{func}`starfinder.registration.apply_shift` move content toward larger indices.
Wrapped edges are zero-filled; content lost at an edge cannot be recovered.
{func}`starfinder.registration.register_volume` applies that negation internally
but returns the detected displacement. `FOV.global_shifts` and its `row, col, z`
CSV log likewise store detected `(dy, dx, dz)` components, with no origin offset.

Dense fields from demons, TPS and CPD use a different operation:
`registered[p] = moving[p + field[p]]`. Supply these fields directly to their
warp functions, without negating them. TPS/CPD fields are float32; demons fields
are float64. Warp functions preserve image dtype, so interpolation can be
quantized for integer images. SimpleITK component reversal is handled internally.

## Intensities and channels

{func}`starfinder.io.load_volume` returns an `ImageLoadResult` with a ZYX
`image`, `metadata`, ordered `channel_labels`, `source_paths` and `diagnostics`.
{func}`starfinder.io.load_round` returns ZYXC and requires `ImageLoadConfig`.
Both preserve dtype by default. TIFF axes select OME/ImageJ dimensions; multiple
series/time points/channels require explicit indices. Plain TIFFs are ZYX (YX
is expanded); `source_axes` explicitly describes other supported layouts.

Channel patterns determine output order. To match wavelength-sorted MATLAB data,
use `("ch00", "ch02", "ch01", "ch03")` explicitly. Missing/ambiguous matches,
repeated source files, mixed dtypes or inconsistent geometry raise `ValueError`.
Unequal shapes error unless `crop_policy="minimum"` requests a low-index crop;
its source shapes and start are recorded in diagnostics. No silent rescale.

`ImageConversionConfig` specifies `output_dtype`, mode `cast`, `clip` or `rescale`,
ranges, `global`/`per_channel` scope and `nearest_even`/`truncate` rounding. A cast
rejects range loss. Clip needs an output range. Rescale needs a declared input
range or `range_policy="data"`, plus output range. Declared input-range violations
error. Constant data maps to the lower output endpoint. This is intensity
conversion, not physical-unit conversion. Float64 work is allocated per active
channel/group, plus output. Inputs are never mutated.

{func}`starfinder.preprocessing.normalize_intensity` requires
`MinMaxNormalizationConfig(output_dtype="uint8", output_range=(0, 255))` to
reproduce the historical output choice. Its default scope is per-channel and
integer rounding truncates; float64 calculation can differ at quantization
boundaries from the old float32 calculation. The SNR gate retains nonconstant
low-SNR raw values, clipped to the declared range. Constant groups always map
to the lower endpoint before SNR gating, including a nonzero lower endpoint
when explicitly requested. This applies to global and per-channel scope and
intentionally corrects the historical constant-with-enabled-gate behavior.

{func}`starfinder.preprocessing.match_histogram` matches each channel to one
ZYX reference with an exact CDF. `HistogramMatchingConfig` preserves input dtype
by default and rejects unrepresentable results; the unused `nbins` is removed.
{func}`starfinder.preprocessing.reconstruct_background` and
{func}`starfinder.preprocessing.filter_tophat` use a disk of `radius_yx` pixels,
independently per XY slice/channel, with reflect boundaries. Float64 slice work
avoids signed uint16 overflow; results retain source dtype and saturate to its
representable range, never wrap or force uint8. Float reconstruction can be
negative. Integers wider than 32 bits are unsupported for morphology.

{func}`starfinder.preprocessing.project_image` retains singleton Z. Max retains
dtype; sum uses uint64/int64 for integers up to 32 bits and float64 for floats.
64-bit integer sums error; explicitly convert first. Requested integer output
errors on overflow unless an explicit clipping conversion is supplied. There is
no implicit display rescaling. Projection allocates only a reduced result;
normalization and histogram matching allocate output plus active-channel work;
morphology allocates output plus slice work. All processing rejects empty,
nonfinite, complex and unsupported-dimensional arrays without mutating inputs.

## Geometry and derived frames

{class}`starfinder.image.ImageMetadata` stores `frame_id`, `spacing_zyx`,
`origin_zyx`, orthonormal `direction_zyx`, and `spatial_unit`. Spacing is a finite,
strictly positive triple (repeated positive values are valid), or `None`.
Physical conversion requires every field:
`world_zyx = origin_zyx + direction_zyx @ (index_zyx * spacing_zyx)`.
TIFF resolution tags do not supply an invented origin, direction or unit.
`save_volume(metadata=...)` persists these fields for round-trip loading.

Crop/subtile output indices map to source indices by adding the crop start;
known origin is translated accordingly. FOV NPZ subtiles retain geometry and
source metadata/start mappings, while shared filenames and one-based MATLAB
coordinate tables remain unchanged. Legacy NPZs without geometry load with an
explicit unknown frame.

XY rotation uses the pull map `source = R @ (output - output_center) + source_center`,
where the YX block of R is `[[cos, sin], [-sin, cos]]`. Right angles use rot90 and
swap XY shape/spacing as needed; other angles retain shape and interpolate.
Known anisotropic XY geometry at non-right angles would require shear, which
this orthonormal geometry model rejects explicitly. Unknown calibration remains
unknown. Projection creates a derived frame: `(0,y,x)` represents all source
`(z,y,x)` along the collapsed column. It has no unique 3D physical coordinate;
physical fields remain unknown and FOV diagnostics retain the source metadata.

### Before and after

Old array-returning I/O and positional preprocessing arguments are removed:

```python
from starfinder.io import ImageLoadConfig, load_round
from starfinder.preprocessing import MinMaxNormalizationConfig, normalize_intensity

loaded = load_round("round1", config=ImageLoadConfig(channel_labels=("ch00", "ch01")))
image = normalize_intensity(loaded.image, config=MinMaxNormalizationConfig(
    output_dtype="uint8", output_range=(0, 255), snr_threshold=5.0))
```

Access `.image` instead of treating the loader result as an array or tuple.
The old I/O names, preprocessing names and `starfinder.utils` projection import
have no aliases. FOV/Dataset's remaining coordination renames belong to the
separate orchestration migration. MATLAB APIs and shared configuration keys are
unchanged; Python numeric corrections are not claims of MATLAB equivalence.

## Spots, barcodes and output

{func}`starfinder.spotfinding.find_spots_3d` returns `z, y, x, intensity, channel`.
Coordinates and channel indices are zero-based. Noise mode uses
`median + k * MAD * 1.4826`; adaptive mode uses each channel maximum,
adaptive_round uses the maximum across all channels, and global mode uses the
uint8/uint16 dtype maximum. Always pass `intensity_estimation` and
`intensity_threshold` together when changing modes. `min_distance` also controls
the default excluded image border. Distances are not adjusted for anisotropy.

Extraction's `voxel_size=(1, 2, 2)` means integer **half-widths** `(dz, dy, dx)`
of a `3 × 5 × 5` neighborhood, not micrometre spacing. Coordinate columns are
cast to integers and must index the input volume. Edge neighborhoods use zero
padding. {func}`starfinder.barcode.extract_from_location` uses L2-normalized
intensities; codebook-aware decoding instead normalizes raw tensors across C
into probabilities. Those scores are not interchangeable.

Color strings use one-based channel labels (`'1'` through `'4'` for four channels).
The extractor returns `M` for a tied maximum (including an all-zero multi-channel
neighborhood) and `N` for a NaN maximum. The probability decoder sanitizes
nonfinite/negative input to zero, adds a pseudocount, and therefore turns a
signal-free round into uniform probabilities and an `M` tie.

{meth}`starfinder.dataset.FOV.save_signal` copies selected columns, increments
`x`, `y`, and `z` by one, and leaves the in-memory table unchanged. Subtile tables
use one-based inclusive start/end coordinates and one-based tile IDs;
{class}`starfinder.dataset.CropWindow` uses zero-based, end-exclusive slices.
Benchmark comparison helpers expect zero-based `z, y, x`; convert exported
signal CSVs before using them.
