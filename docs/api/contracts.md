# Array and coordinate contracts

These conventions describe the current Python implementation. Individual
[generated signatures](python.rst) give defaults and more specific requirements.
Arrays carry no physical-spacing metadata; distances below are in voxel-index
units unless a function explicitly says otherwise.

| Object | Shape / columns | Convention |
| --- | --- | --- |
| Single-channel volume | `(Z, Y, X)` | Numeric intensity array |
| Multi-channel round | `(Z, Y, X, C)` | Channel-last, usually uint8 or uint16 |
| Projection | `(Y, X)` or `(Y, X, C)` | Z is removed; do not pass directly to a 3D registration function |
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

{func}`starfinder.io.load_multipage_tiff` uses bioio for OME/ImageJ metadata and
selects the first time point and channel (`T=0, C=0`). Plain TIFFs retain their
raw axis order, with a singleton Z inserted for a 2D image. Supply plain files
already arranged as ZYX. `convert_uint8=True` preserves uint8 inputs, otherwise
min-max scales to 0..255; constant inputs become zero. False preserves dtype.
The multi-channel loader crops to the minimum common shape from the low-index
corner and scales across the whole stacked array, not separately per channel.
Its metadata reports shapes/dtype/cropping, not physical spacing.

Channel patterns determine output order. To match wavelength-sorted MATLAB data,
use `['ch00', 'ch02', 'ch01', 'ch03']` explicitly; the Python loader does not infer
that ordering. Multiple filename matches use the first glob result, so provide
unambiguous channel patterns.

{func}`starfinder.preprocessing.min_max_normalize` scales each channel to uint8;
its optional SNR gate clips low-SNR raw values to uint8 instead. Histogram
matching and morphology preserve input shape/dtype. `histogram_match(nbins=64)`
accepts `nbins` for compatibility but does not use it. Maximum projection
preserves dtype; sum projection casts to uint32, sums Z, then rescales to uint8.

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
