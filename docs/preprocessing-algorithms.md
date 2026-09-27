# Preprocessing algorithm specification

**Status: Proposed (W-226, 2026-09-27). Not accepted.** Human review in W-227
accepts, amends or rejects this page. Defaults marked *provisional* are
development choices to be evaluated; they are not recommendations.

This page specifies the three agreed additions for Chapter II §2.5: scalar
background subtraction, 3D background subtraction and percentile normalization.
It also specifies the shared numerical policy, the fitting modes and the
histogram summary behind sample-level statistics. The step interface is in
{doc}`preprocessing-contract`; current behavior is in
{doc}`preprocessing-baseline`.

## Shared numerical policy

* **Input.** Finite, nonempty ZYX or ZYXC arrays. Empty or nonfinite input
  raises. Laboratory catalog data are uint8 exports, and some data are uint16, so
  the policy is designed for uint8 first and must also hold for uint16. Float input
  is supported where stated.
* **Working precision.** float64.
* **Output dtype.** Equal to the input dtype (`dtype_policy="preserve"`).
* **Integer output.** Round half to even (`np.rint`), then clip to the dtype's
  range. Values below zero become zero. The integer cast never wraps.
* **Float output.** No rounding. Values below zero become zero where a method
  subtracts a background. Percentile normalization of float input maps to [0, 1].
* **Constant channel.** Each method states its result and records a
  `constant_channel` diagnostic.
* **Diagnostics.** Every method records, per channel, the fitted values, the
  output zero fraction, median and MAD, and the resulting local-maxima `noise`
  threshold (`median + 5 × 1.4826 × MAD`). It records `mad_zero: true` when MAD is
  0. These values expose the interaction with detection described below; they do
  not change detection.

## Scalar background subtraction

One background level per channel and round, subtracted uniformly.

| Item | Specification |
| --- | --- |
| Config | `ScalarBackgroundConfig(estimator="percentile", percentile=10.0, fit="fov")` |
| Estimators | `"mode"`: the most frequent value, excluding zero and the dtype maximum; ties take the lowest value. `"percentile"`: the `percentile`-th value by the inverted-CDF definition over all voxels. Both are computed from the channel's integer histogram |
| Parameters | `percentile` in [0, 100), intensity percent (*provisional* default 10); `fit` is `"fov"` or `"supplied"` |
| Output | `max(x − b, 0)` in the input dtype |
| Fitted values | `{"background": [b_0, ..., b_C-1]}` |
| Float input | `"percentile"` uses `np.percentile(method="inverted_cdf")`. `"mode"` is rejected for float input |
| Constant channel | `b` equals the constant; the output is zero |
| Failure | Invalid percentile, float input with `"mode"`, or `fit="supplied"` without a matching entry raises |

Sources: the inverted-CDF quantile is definition 1 of Hyndman and Fan (1996),
*Sample quantiles in statistical packages*, The American Statistician 50(4),
361–365, as implemented by `numpy.percentile(method="inverted_cdf")`. Percentile
background subtraction follows the precedent of starfish `ClipPercentileToZero`
(revision `1fb00cbc`), which subtracts its `p_min` percentile after clipping.
The mode estimator is the peak of a background-dominated histogram, computed with
`numpy.bincount`; no external algorithm is claimed. A median estimator is not
offered, because subtracting the median makes at least half of the voxels zero,
so the noise threshold becomes zero.

## 3D background subtraction

A volumetric white top-hat: grey opening with an anisotropic ellipsoidal
footprint, subtracted from the image. It is the 3D counterpart of the existing
XY top-hat.

| Item | Specification |
| --- | --- |
| Config | `Background3DConfig(radius_um_zyx=None, radius_voxels_zyx=None)`; exactly one must be set |
| Footprint | Ellipsoid with semi-axes `r_z, r_y, r_x` voxels: voxels with `(dz/r_z)² + (dy/r_y)² + (dx/r_x)² ≤ 1`. An axis with radius 0 is not filtered |
| Radii | From `radius_um_zyx` and `ImageMetadata.spacing_zyx` (µm): `r = round(radius_um / spacing)`. If spacing is unknown, `radius_um_zyx` raises and `radius_voxels_zyx` must be given. There is no default radius |
| Background | `scipy.ndimage.grey_opening(x_c, footprint=ellipsoid)` per channel, with reflect boundaries |
| Output | `max(x − background, 0)` in the input dtype |
| Fitted values | `{"radius_voxels_zyx": [r_z, r_y, r_x]}` |
| Constant channel | Background equals the constant; the output is zero |
| Failure | Both or neither radius given, a negative radius, µm radii without spacing, or a footprint larger than the volume along an axis raises |

Choose the radius larger than the puncta: the opening removes bright structures
smaller than the footprint. For synthetic evaluation, use
`r = ceil(3σ) + 1` voxels per axis from the preset's documented puncta widths σ.

Sources: grey opening and the white top-hat are defined in Serra (1982), *Image
Analysis and Mathematical Morphology*, and Soille (2003), *Morphological Image
Analysis*, 2nd ed. The implementation is `scipy.ndimage.grey_opening`. starfish
`Filter.WhiteTophat(masking_radius, is_volume=True)` (revision `1fb00cbc`) is
the precedent for a volumetric top-hat in spot-based transcriptomics. It uses an
isotropic ball; this specification uses an anisotropic ellipsoid, because Z
sampling differs from XY.

## Percentile normalization

A linear map of a per-channel percentile range onto the full output range.

| Item | Specification |
| --- | --- |
| Config | `PercentileNormalizationConfig(p_low=1.0, p_high=99.9, fit="fov")` |
| Range | `low`, `high`: the `p_low`-th and `p_high`-th values by the inverted-CDF definition, per channel of the round (`fit="fov"`), or read from the supplied file (`fit="supplied"`) |
| Parameters | `0 ≤ p_low < p_high ≤ 100`, intensity percent (*provisional* defaults 1 and 99.9) |
| Output | `clip((x − low) / (high − low), 0, 1) × dtype_max`, rounded half to even, for unsigned integers; values in [0, 1] for float input |
| Fitted values | `{"p_low": ..., "p_high": ..., "low": [...], "high": [...]}` per round, reusable as supplied values |
| Constant channel or `high == low` | Output zero; `degenerate_range` diagnostic |
| Failure | Invalid percentiles, or supplied values missing for the round or channel, raises |

Values above `high` saturate at the dtype maximum. With `p_high=99.9`, about
0.1 % of voxels saturate by design.

Sources: the inverted-CDF quantile as above. Per-bit percentile normalization is
documented by split-FISH (Goh et al. 2020, *Nature Methods* 17, 689–693; the
`split-fish` repository parameters).

## Fitting modes and sample-level statistics

`fit="fov"` fits on the current round of the current FOV. `fit="supplied"` reads
values from the recipe's `supplied_statistics` file. Sample-level statistics come
from two passes; wiring them into Snakemake belongs to §2.13.

1. **Summary pass.** For each FOV and round, `summarize_histograms` records the
   per-channel histogram. For uint8 and uint16 input this is
   `numpy.bincount(values, minlength=dtype_max + 1)`, with all bins. Float input
   requires explicit bin edges and is not used for exact statistics.
2. **Merge.** Histograms from the selected FOVs are summed elementwise. They must
   share dtype, round, channel labels and bin layout. FOVs used and excluded,
   such as near-empty tissue-edge fields, are recorded.
3. **Statistics from merged counts.**
   * The inverted-CDF percentile is the smallest value whose cumulative count
     reaches `p/100 × N`. For `p = 0` it is the smallest value with a nonzero
     count. This equals `numpy.percentile(concatenated, p, method="inverted_cdf")`
     exactly.
   * The mode is the value with the largest count, excluding zero and the dtype
     maximum.
   * The histogram-matching reference is the merged count vector of the
     reference round's selected channel. For unsigned input, scikit-image
     `match_histograms` uses the template only through `numpy.bincount`.
     Matching against the merged counts therefore equals matching against the
     concatenated reference volumes exactly.
4. **Supplied file.** JSON with schema identifier
   `starfinder.preprocessing.supplied/1`:

```json
{"schema": "starfinder.preprocessing.supplied/1",
 "dtype": "uint8", "channel_labels": ["ch00", "ch01", "ch02", "ch03"],
 "fovs_used": ["Position001"], "fovs_excluded": [],
 "rounds": {"round1": {"percentile_range": {"p_low": 1.0, "p_high": 99.9,
                                            "low": [3, 2, 2, 3], "high": [201, 180, 190, 222]},
                       "scalar_background": {"estimator": "percentile", "percentile": 10.0,
                                             "values": [4, 3, 3, 5]}}},
 "histogram_reference": {"round": "round1", "channel": 0,
                         "values": [0, 1, 2], "counts": [10, 250, 40]}}
```

The per-FOV histograms are saved as `.npz` arrays of shape
(rounds, channels, bins) with the channel and round labels.

## Interaction with noise-mode detection

The local-maxima `noise` threshold uses the median and MAD over all voxels of a
channel.

* **Min–max and percentile normalization.** For the unclipped part of the range,
  a linear map scales the median and MAD equally, so detections change little.
  Quantization to few grey levels and saturation above `high` create plateaus.
* **Histogram matching.** This is a monotonic but nonlinear map, so the effective
  cutoff changes with the reference distribution.
* **Scalar background.** The zero fraction is the fraction of voxels at or below
  `b`. With `"mode"` on a background-dominated channel, this approaches one half,
  and MAD can collapse to zero. With `"percentile"` at 10, it is about 10 %, and
  MAD shrinks less.
* **3D background.** Like XY reconstruction, it is expected to produce many zero
  voxels. The baseline fixture shows XY reconstruction reaching MAD = 0. Task
  group 2 measures the 3D method.

§2.7 records these quantities during detection and warns when MAD is 0 or more
than half of the voxels are zero. It does not change the default threshold.

## Evaluation settings for task group 5

* `threshold_value` sweep: {2, 3, 4, 5, 6, 8, 10, 12, 15}, with default 5.
* Development seeds {0, 1, 2}; held-out evaluation seeds {100, 101, 102}.
* Each recipe is reported at the max-F1 threshold chosen on development seeds and
  at the default threshold.
