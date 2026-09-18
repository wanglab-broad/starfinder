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
are float64. SimpleITK component reversal is handled internally.

### Translation edge cases and resampling precision

The FFT estimator still returns integer-valued displacements; these corrections
do not add subpixel estimation. Singleton axes return zero. For odd length `n`,
a correlation peak at `n//2` is not wrapped. An exact even half-period cannot
distinguish positive from negative motion: `phase_correlate` reports `+n/2`,
whereas `phase_correlate_skimage` retains its native `-n/2` convention. Both
align the periodic interior; their zero-filled boundaries can differ.

`apply_shift` uses exact integer rolling (within its existing `1e-6` voxel
integer tolerance), otherwise Fourier shifting followed by the **real part**
of the inverse FFT. Signed intensities and negative ringing are retained;
the previous magnitude operation lost signs. For even axes, taking the real
part also projects away the fractional-shift Nyquist imaginary component.
This is periodic Fourier interpolation followed by zeroing wrapped edges,
not spatial interpolation of a zero-padded volume. Shifts spanning an entire
axis return zeros. Lost edge content is not recoverable.

All existing registration application functions and multichannel wrappers
accept `output_dtype="input"` (default), `"float32"` or `"float64"`. For example:

```python
shifted = apply_shift(image, (0, 0, 0.5), output_dtype="float32")
```

Source dtype is preserved by default. Integer output is interpolated in floating
point, rounded **once with nearest-even ties**, saturated to the source dtype's
range, then cast. Explicit floating output skips rounding and clipping; it does
not rescale intensities or select calculation precision. This intentionally
changes legacy truncation (Fourier/SimpleITK), SciPy's implicit integer rounding,
and overflow/wrap behavior. For example, interpolated `0.5, 1.5, 2.5, 3.5` now
becomes `0, 2, 2, 4`, while uint8 Fourier overshoot saturates to `[0, 255]`.

Fractional Fourier application calculates in float32, except float64 source
images use float64. Native SimpleITK linear resampling uses float64; SciPy
linear sampling produces a float64 slice before final conversion. Integer
rolling preserves exact stored values; interpolation of wide integers remains
subject to floating precision. Tests use `2e-6` absolute tolerance for small
float32 signed Fourier fixtures and `1e-12` for float64, not a universal
full-image accuracy guarantee. No estimator tuning or calibration is implied.
Translation stays compact, SimpleITK reuses the prepared transform across
channels, and TPS/CPD allocate coordinates and sampled values per Z slice.

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

{func}`starfinder.spot_finding.find_spots` requires `config`, `metadata` and
`spot_namespace`, and returns {class}`starfinder.spot_finding.SpotFindingResult`.
Its `spots` table has `spot_id` (pandas string) and `z/y/x` (float64), including
empty results. Coordinates are zero-based voxel indices; physical conversion
requires complete `ImageMetadata`. Detection never performs landmark matching.

{class}`starfinder.spot_finding.LocalMaximaConfig` preserves the pipeline's
per-channel peaks: noise uses `median + threshold_value * MAD * 1.4826` (sigma
units); adaptive uses channel maximum, adaptive_round the image maximum, and
global the uint8/uint16 maximum (fractions in [0,1]). `min_distance_voxels` is a
positive integer, with `exclude_border=True` by default. Singleton Z is processed
as a YX plane with Z=0 and only the YX border excluded. Other volumes retain the
original 3D peak policy. No spacing correction is applied to distances.
Optional `channel` is int64; `peak_intensity` is float64 in original input units
and can be omitted with `measure_peak_intensity=False`. No integrated intensity
or detection score is invented. Optional `channel_labels` must uniquely label
all channels, and their tuple is recorded in diagnostics.

{class}`starfinder.spot_finding.NoiseLandmarkConfig` retains registration's
per-channel MAD peaks followed by its original radius deduplication: for each
pair within `min_distance_voxels`, drop the higher concatenated index.
{class}`starfinder.spot_finding.PercentileCentroidConfig` instead sums channels,
thresholds strictly above `threshold_percentile` in [0,100], labels face-connected
components and computes intensity-weighted centroids. These policies return
coordinates without inventing channel or peak measurements. All accept finite
ZYX/ZYXC arrays; 2D callers must explicitly add singleton Z. The benchmark MIP
caller now does this, correcting the old accidental collapse of a 2D image's X
axis by the former detector. This is an intentional dimensional correction.

Namespaces must identify dataset/sample/FOV and subtile when applicable. FOV
uses an unambiguous JSON array of those identifiers (subtile ID is one-based or
null for a full FOV), exposes `spot_result`, and retains `spot_id` and
`spot_namespace` in its downstream tables. IDs survive filtering, reordering
and joins on **(spot_namespace, spot_id)**; never regenerate IDs for a subset or
join by row position. Results reject duplicate IDs within a namespace. Callers
must choose distinct namespaces for independent detection results; identities
are not promised invariant across changed images or detector configurations.
The structured barcode/export redesign remains a subsequent migration.

Before: `find_spots_3d(image, intensity_estimation="adaptive", intensity_threshold=0.2)`.
After:

```python
from starfinder.image import ImageMetadata
from starfinder.spot_finding import find_spots, LocalMaximaConfig
result = find_spots(image, config=LocalMaximaConfig("adaptive", 0.2),
                    metadata=ImageMetadata("dataset/sample/FOV/round1"),
                    spot_namespace="dataset/sample/FOV")
spots = result.spots
```

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
