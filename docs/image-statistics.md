# Image statistics and calibration targets

`benchmarks/image_statistics.py` measures intensity, puncta and noise statistics
of one channel volume. The same function measures a real ZYX TIFF and a channel
of a `starfinder.synthetic` scene, so real and synthetic images are compared
with identical definitions. The real-data run below sets development target
ranges for the calibrated synthetic preset version (W-241) and the §2.5 rerun
(W-239). It is development evidence (W-238): benchmark calibration and D04 stay
with the Chapter II benchmark register (W-93 Stage 2), and the measured FOVs are
not qualified (W-92).

## Usage

The command line measures one TIFF and prints its JSON record, or runs the
planned real-data measurement into a directory outside Git and outside the data
root. From `src/python`:

```bash
uv run python ../../benchmarks/image_statistics.py measure /external/volume_ch00.tif
uv run python ../../benchmarks/image_statistics.py run --root /external/sample-dataset --output /external/run/measurement-pilot --pilot
uv run python ../../benchmarks/image_statistics.py run --root /external/sample-dataset --output /external/run/measurement --pilot-manifest /external/run/measurement-pilot/manifest.json
uv run python ../../benchmarks/image_statistics.py attach-time --output /external/run/measurement --time-log /external/run/time.txt
```

In Python, `measure_volume` takes a NumPy ZYX array or a `TiffVolume`, which
reads one TIFF page at a time:

```python
import sys
from dataclasses import replace

sys.path.insert(0, "../../benchmarks")
from image_statistics import TiffVolume, measure_volume, synthetic_channel
from starfinder.synthetic import development_scene_preset, generate_formed_scene

book, config = development_scene_preset("combined")
scene = generate_formed_scene(book, config=replace(config, dtype="uint8"))
synthetic = measure_volume(synthetic_channel(scene, "round1", "ch00"))
with TiffVolume("/external/volume_ch00.tif") as volume:
    real = measure_volume(volume)
real.record["p99_9"], real.record["adaptive"]["snr_clutter_p50"], len(real.puncta)
```

`record` holds the volume statistics and per-selection summaries; `puncta` has
one row per measured punctum. Volumes must be `uint8` or `uint16`.

## Definitions

### Volume histogram

The full volume is read plane by plane into one integer histogram; no volume is
converted to a wider dtype.

| Statistic | Definition |
| --- | --- |
| Zero fraction | Fraction of voxels equal to 0. |
| Saturated fraction | Fraction of voxels at the dtype maximum (255 or 65535). |
| p50, p90, p99, p99.9, p99.99 | Inverted-CDF percentile: the smallest grey level whose cumulative count reaches ⌈q·n⌉ of the n voxels. |
| Max | Largest grey level present. |
| MAD | Median absolute deviation from the p50, unscaled, with the same inverted-CDF median. |
| Depth profile | The p99.9 of each z plane, by the same definition. |
| Depth attenuation | Mean depth-profile value of the first depth quarter divided by that of the last (undefined with fewer than four planes or a zero last quarter). |
| Depth quarter | Plane z of Z planes is in quarter ⌊4z/Z⌋ (0 to 3), in stored z order. |

### Puncta

Puncta are measured on the central 512×512 YX crop over all Z planes (the full
extent when smaller).

| Statistic | Definition |
| --- | --- |
| Local maximum | A voxel equal to the maximum of its 3×5×5 (ZYX) footprint, with zero outside the crop, and above 0. Each 26-connected plateau of equal maxima counts once, at its first voxel in raster order. Maxima within 10 voxels of the crop's Y or X edge are dropped, so every annulus lies in the crop. |
| Adaptive selection | Maxima at or above 0.2 × the volume's maximum (the historical real-data setting). |
| Lenient selection | Maxima at or above 10 grey levels. |
| Subsample | At most 1000 maxima per selection and volume, drawn without replacement by `numpy.random.default_rng(0)`, fresh for each volume (adaptive, then lenient). |
| Annulus | The same-z 21×21 window centred on the maximum, minus its central 7×7 (392 pixels). |
| Local background | Median of the annulus. |
| Clutter σ | Standard deviation of the annulus: noise plus background structure and neighbouring puncta. |
| Pixel σ | Standard deviation of the horizontal and vertical differences between neighbouring annulus pixels (both in the annulus, 728 pairs), divided by √2: the white-noise σ, insensitive to smooth structure. |
| Peak | The maximum's grey level. |
| Amplitude | Peak − local background. |
| SNR (clutter), SNR (pixel) | Amplitude / clutter σ and amplitude / pixel σ; undefined when the σ is 0. |
| Background fraction | Local background / amplitude, for positive amplitudes. |
| Clutter / pixel ratio | Clutter σ / pixel σ; 1 for white noise on a flat background. |

### Summaries

Target statistics pool the measured puncta of the development FOVs of one
dataset; held-out FOVs are summarized separately. Each volume contributes at most
1000 puncta per selection, so volumes are weighted nearly equally rather than by
punctum count. Pooled p10/p50/p90 use linear interpolation (`numpy.percentile`).

| Statistic | Definition |
| --- | --- |
| Zero fraction p50 | Median over volumes. |
| Channel gain spread | Maximum / minimum over channels of the pooled median amplitude per channel. |
| Round trend | Maximum / minimum over rounds of the pooled median amplitude per round. |
| Depth attenuation p50 | Median over volumes of the volume depth attenuation. |
| Amplitude by depth quarter | Pooled median amplitude per depth quarter; the first/last ratio is reported. |
| σ_log (truncated fit) | Per volume, the maximum-likelihood σ of a lognormal fitted to the peaks, left-truncated at the selection threshold (continuity-corrected to the first included grey level − 0.5); the median over volumes. A fit needs at least 20 puncta and is reported as unidentified when σ > 3 or when less than 0.1 % of the fitted distribution lies above the truncation (only a far tail is observed). |
| σ_log (within volume) | Standard deviation of ln amplitude after subtracting each volume's mean ln amplitude; channel and round gains are removed, but the truncation is not. |
| σ_log (pooled) | Standard deviation of ln amplitude over all pooled puncta; includes gain spread and is truncated. |

The two selections truncate the brightness distribution differently: the
adaptive selection keeps peaks at or above 0.2 × the volume maximum, and the
lenient selection keeps peaks at or above 10 grey levels. Only the truncated fit
corrects for this, and only under the lognormal assumption.

## Real-data run

Source run: `runs/W-238/20260928T194004Z-1da19f6c/measurement/`, outside the
repository (host GP099-29C, 2026-09-28, Starfinder `7d08eee` with the
uncommitted W-238 changes). Its manifest records the plan, every input's path,
size and modification time, the tool's SHA-256, the environment and the
`/usr/bin/time -v` record. The data root was read only; the inputs' sizes and
modification times were unchanged after the run.

For each dataset, the FOV directory names in `round1` are sorted as strings and
the FOVs at indices round(i·(n − 1)/3), i = 0…3, are chosen; i = 2 is held out.
Every round and channels ch00–ch03 are measured; ch04 is excluded.

| Dataset | round1 FOVs | Development FOVs | Held out | Rounds | Volumes |
| --- | --- | --- | --- | --- | --- |
| D01 `tissue-2D` | 56 | tile_1, tile_26, tile_9 | tile_43 | 4 | 64 |
| D02 `cell-culture-3D` | 70 | Position351, Position374, Position420 | Position397 | 6 | 96 |
| D03 `LN` | 64 | Position001, Position022, Position064 | Position043 | 4 | 64 |
| `aging` (no benchmark ID) | 6 | Position400, Position402, Position405 | Position403 | 9 | 144 |

The pilot (one FOV and round1 per dataset) projected 982 s for the full run,
within the 35-minute budget, so no reduction was applied. The full run measured
368 volumes on one CPU in 1258 s wall time with a maximum RSS of 806,916 KiB and
wrote 46 MB of tables, records and overlays.

## Development targets

The tables pool the development FOVs of each dataset. The adaptive selection is
the primary target: it matches the historical detection setting. The lenient
selection shows how the statistics change when dimmer maxima are included. Peak,
amplitude and σ values are in uint8 grey levels.

Adaptive selection (≥ 0.2 × volume maximum), development FOVs:

| Statistic | D01 `tissue-2D` | D02 `cell-culture-3D` | D03 `LN` | `aging` |
| --- | --- | --- | --- | --- |
| Volumes / measured puncta | 48 / 32797 | 72 / 72000 | 48 / 23989 | 108 / 107623 |
| Threshold p50 (grey levels) | 43.8 | 47.5 | 49.2 | 47.8 |
| Peak p10 / p50 / p90 | 50 / 81 / 146 | 53 / 83 / 142 | 50 / 89 / 182 | 52 / 84 / 136 |
| Amplitude p10 / p50 / p90 | 48 / 79 / 143 | 49 / 77 / 135 | 33 / 82 / 174 | 45 / 72 / 130 |
| SNR (clutter) p10 / p50 / p90 | 3.85 / 11.1 / 45 | 2.95 / 5.97 / 15 | 2.95 / 7.31 / 32.7 | 2.26 / 6.25 / 23.8 |
| SNR (pixel) p10 / p50 / p90 | 14.5 / 37.7 / 112 | 6.59 / 12.9 / 30.6 | 6.94 / 18.6 / 70.8 | 7.49 / 18.1 / 57.2 |
| Background / amplitude p10 / p50 / p90 | 0 / 0.0141 / 0.0833 | 0 / 0.0439 / 0.175 | 0.00518 / 0.0952 / 0.531 | 0 / 0.0426 / 0.452 |
| Clutter σ p50 | 7.41 | 13.8 | 9.28 | 12.8 |
| Pixel σ p50 | 2.21 | 6.42 | 3.92 | 4.41 |
| Clutter / pixel σ p50 | 3.26 | 2.12 | 2.35 | 2.82 |
| Zero fraction p50 (min–max) | 0.784 (0.671–0.862) | 0.919 (0.876–0.943) | 0.766 (0.236–0.932) | 0.583 (0.208–0.908) |
| Channel gain spread | 1.45 | 1.26 | 3.15 | 1.31 |
| Round trend | 1.04 | 1.18 | 2.04 | 1.12 |
| Depth attenuation p50 (min–max) | 1.12 (0.751–1.72) | 2.86 (0.312–32.2) | 6.68 (0.109–38.2) | 0.806 (0.508–1.11) |
| Amplitude first / last depth quarter | 1.11 | 1.2 | 0.593 | 1.24 |
| σ_log truncated fit p50 (p10–p90) | 0.495 (0.426–0.566) | 0.434 (0.406–0.474) | 0.492 (0.43–0.674) | 0.42 (0.37–0.49) |
| Identified fits / volumes with puncta | 48/48 | 72/72 | 21/44 | 108/108 |
| σ_log within volume | 0.382 | 0.364 | 0.5 | 0.426 |
| σ_log pooled | 0.408 | 0.379 | 0.641 | 0.446 |

Lenient selection (≥ 10 grey levels), development FOVs:

| Statistic | D01 `tissue-2D` | D02 `cell-culture-3D` | D03 `LN` | `aging` |
| --- | --- | --- | --- | --- |
| Volumes / measured puncta | 48 / 47541 | 72 / 72000 | 48 / 39153 | 108 / 108000 |
| Peak p10 / p50 / p90 | 11 / 30 / 108 | 15 / 52 / 126 | 10 / 19 / 44 | 11 / 31 / 108 |
| Amplitude p10 / p50 / p90 | 10 / 29 / 104 | 14 / 48 / 118 | 9 / 14 / 31 | 10 / 27 / 96 |
| SNR (clutter) p10 / p50 / p90 | 1.72 / 7.66 / 37.2 | 1.56 / 4.61 / 13.9 | 1.06 / 3.41 / 32.8 | 1.11 / 4.21 / 17.3 |
| SNR (pixel) p10 / p50 / p90 | 6.2 / 25.6 / 89.4 | 3.34 / 9.87 / 26.9 | 2.48 / 7.36 / 40.9 | 3.24 / 11.2 / 39.1 |
| Background / amplitude p50 | 0.01 | 0.0417 | 0.238 | 0.0312 |
| Clutter σ / pixel σ p50 | 4.33 / 1.3 | 11.4 / 5.38 | 5.07 / 2.32 | 7.78 / 2.88 |
| Channel gain spread | 1.46 | 1.41 | 1.08 | 1.43 |
| Round trend | 1.28 | 1.4 | 1.25 | 1.76 |
| σ_log truncated fit p50 (identified fits) | 1.04 (48/48) | 0.792 (72/72) | 0.778 (32/48) | 1.09 (108/108) |
| σ_log within volume | 0.845 | 0.763 | 0.617 | 0.837 |

### Target ranges

The range over the four datasets' development values, from
`tables/target_ranges.csv`:

| Statistic | Adaptive | Lenient |
| --- | --- | --- |
| Peak p50 | 81–89 | 19–52 |
| Amplitude p10 / p50 / p90 | 33–49 / 72–82 / 130–174 | 9–14 / 14–48 / 31–118 |
| SNR (clutter) p10 / p50 / p90 | 2.26–3.85 / 5.97–11.1 / 15–45 | 1.06–1.72 / 3.41–7.66 / 13.9–37.2 |
| SNR (pixel) p10 / p50 / p90 | 6.59–14.5 / 12.9–37.7 / 30.6–112 | 2.48–6.2 / 7.36–25.6 / 26.9–89.4 |
| Background / amplitude p50 | 0.014–0.095 | 0.010–0.238 |
| Clutter σ p50 | 7.41–13.8 | 4.33–11.4 |
| Pixel σ p50 | 2.21–6.42 | 1.3–5.38 |
| Clutter / pixel σ p50 | 2.12–3.26 | 2.02–3.21 |
| Zero fraction p50 | 0.583–0.919 | (same volumes) |
| Channel gain spread | 1.26–3.15 | 1.08–1.46 |
| Round trend | 1.04–2.04 | 1.25–1.76 |
| Depth attenuation p50 | 0.806–6.68 | (same volumes) |
| σ_log within volume | 0.364–0.5 | 0.617–0.845 |
| σ_log truncated fit p50 | 0.42–0.495 | 0.778–1.09 |

Reading the targets:

* The adaptive peak and amplitude medians are nearly equal across datasets
  because the threshold scales with each volume's maximum. They describe the
  bright puncta of an 8-bit export, not the full brightness distribution.
* The clutter σ is 2–3.3 times the pixel σ everywhere: the local background is
  spatially structured, not white noise. A white-noise-only model cannot match
  both SNRs at once.
* The median voxel is at or near 0 (p50 at most 18; MAD 0 in 77 % of volumes,
  including every D01 and D02 volume), and the zero fraction is high. Nothing
  is saturated (at most 1.5 × 10⁻⁸ of a volume).
* The adaptive σ_log of 0.42–0.50 is the lognormal width of the bright puncta.
  In the identified fits, the median fraction of the fitted lognormal above the
  adaptive threshold is 0.81–0.93 per dataset (p10 0.59–0.75), so the truncation
  is moderate. In LN, the threshold often sits in the far tail of the fitted
  distribution, and about half the fits are unidentified. The lenient σ_log
  (0.78–1.09) mixes dim maxima with puncta and is not a single-population width.
* LN has the largest channel gain spread (3.15: ch00 at 41 and ch03 at 129 grey
  levels), the largest round trend (2.04: round2 at 53 and round4 at 108) and
  strong depth attenuation. Depth attenuation varies widely between volumes in
  D02 and LN, so its median is a weak target.

### Held-out FOVs

Held-out FOVs are summarized separately in `tables/targets.csv` (split
`held_out`) and were not used for the ranges above. Adaptive selection:

| Statistic | D01 tile_43 | D02 Position397 | D03 Position043 | aging Position403 |
| --- | --- | --- | --- | --- |
| Peak p10 / p50 / p90 | 47 / 75 / 133 | 54 / 88 / 144 | 51 / 64 / 169 | 52 / 82 / 137 |
| Amplitude p50 | 73 | 80 | 47 | 74 |
| SNR (clutter) p10 / p50 / p90 | 4.03 / 11.3 / 47.2 | 2.59 / 5.06 / 12.8 | 2.3 / 4.73 / 20.8 | 2.62 / 7.26 / 24.2 |
| SNR (pixel) p10 / p50 / p90 | 14.9 / 37.9 / 116 | 6.02 / 11.5 / 26.8 | 5.25 / 11.1 / 43.7 | 8.5 / 20.4 / 58.1 |
| Background / amplitude p50 | 0.0122 | 0.0769 | 0.316 | 0.028 |
| Clutter / pixel σ p50 | 3.23 | 2.21 | 2.24 | 2.76 |
| Zero fraction p50 | 0.79 | 0.903 | 0.141 | 0.5 |
| Channel gain spread / round trend | 1.46 / 1.1 | 1.29 / 1.19 | 2.56 / 2.51 | 1.29 / 1.13 |
| σ_log truncated fit p50 | 0.48 | 0.413 | 0.564 (6/16 fits) | 0.441 |

D01, D02 and aging held-out values fall within or close to their development
values. LN Position043 differs: few zero voxels, a higher local background
relative to the amplitude, and lower SNRs. The LN targets from three FOVs do not
cover the dataset's FOV-to-FOV variation.

## Visual check of detected maxima

`overlays/<dataset>.png` in the run directory shows, for the first development
FOV and round1, each channel's plane with the most adaptive maxima: a central
256×256 window of the crop, with adaptive maxima in red and lenient-only maxima
in cyan. Inspection on 2026-09-28:

* **D01 `tissue-2D`** (tile_1): the adaptive maxima sit on compact, isolated
  bright spots and look like puncta. The lenient-only maxima are dimmer spots,
  with a few on faint haze.
* **D02 `cell-culture-3D`** (Position351): the maxima fill the cell region
  densely. The adaptive maxima sit on compact bright spots and look like puncta,
  but crowding is high, so neighbouring puncta enter the annulus.
* **D03 `LN`** (Position001): mixed. In ch01 and ch03 most adaptive maxima are
  compact puncta. In ch00 and ch02, many adaptive maxima and most lenient-only
  maxima lie inside diffuse, cell-sized bright bodies and are texture maxima, not
  isolated puncta. The plane shown is z = 0, the edge plane.
* **`aging`** (Position400): mixed. Many adaptive maxima lie in dense bright bands
  where puncta are crowded or merged into continuous signal, especially in ch03.
  Isolated lenient-only maxima between the bands often look like dim puncta.

## Limitations

1. **Post-deconvolution uint8 exports only.** Every measured volume is a Huygens
   CMLE result exported to 8 bits with one scale factor for all channels. The
   targets describe these exports, not raw acquisitions, and export modelling
   is out of scope.
2. **The uint16 and raw-acquisition scale is unverified.** No uint16 or raw
   input was measured. Multiplying uint8 targets by a fixed factor (such as the
   16× used by the §2.5 presets) is an assumption, not a measurement.
3. **The detection selection is biased towards bright puncta.** The adaptive
   selection keeps maxima at or above 0.2 × the volume maximum and the lenient
   selection at or above 10 grey levels, so both describe the brighter part of
   the puncta population; dim puncta are under-represented and the brightness
   distributions are truncated. In dense regions the annulus also contains
   neighbouring puncta, which raises the clutter σ.
4. **Development targets, not D04 calibration.** The FOVs are unqualified (no
   independent split exists, W-92), one FOV per dataset is held out only
   informally, and the targets serve W-241 and W-239. Benchmark calibration and
   D04 remain with W-93 Stage 2.

Two further caveats apply. Depth quarters follow the stored z order; which end
is the sample surface is not verified. The first and last planes have only
one neighbouring plane in the 3×5×5 footprint, so more maxima pass there.
