# Preprocessing algorithm specification

**Status: Proposed (W-226, 2026-09-27; revised after the W-227 review notes).
Not accepted.** Human review in W-227 accepts, amends or rejects this page.
Defaults and thresholds marked *provisional* are development choices to be
evaluated; they are not recommendations.

This page specifies the three agreed additions for Chapter II §2.5: scalar
background subtraction, 3D background subtraction and percentile normalization.
It also specifies two recipe modes, sample-level fitting and the
pre-normalization extraction source, together with the shared numerical policy,
the histogram summary behind sample-level statistics, and the before/after
evaluation design. Each method and mode starts with the problem it addresses,
its cause, the synthetic condition that isolates it, and why its benefit may be
small. The step interface is in {doc}`preprocessing-contract`; current behavior
is in {doc}`preprocessing-baseline`.

## What preprocessing corrects

A simplified image-formation model separates the effects the methods target:

```text
observed(x, c, r) ≈ gain(c, r) × [signal(x, c, r) + background(x, c, r)] + baseline(c, r) + noise
```

| Term | Typical cause | §2.12 development condition |
| --- | --- | --- |
| Channel and round gain | Dye brightness, laser, filter and detector settings, per-round chemistry and bleaching | `gain`, `trend` |
| Baseline | Camera offset, export offset, near-uniform nonspecific background | `baseline` |
| Spatial background | Tissue autofluorescence, texture, out-of-focus light | `gradient`, `regions`, `texture` (each can vary along Z) |
| Bright outliers | Aggregates, debris, hot pixels | No dedicated condition; a few bright `texture` blobs approximate it |
| FOV content | Tissue density; near-empty fields at tissue edges | FOVs differing in `FormedSceneConfig.density` |

The conditions are those of `starfinder.synthetic.development_scene_preset`.
They isolate one effect each but are not calibrated. Catalog real data are
post-Huygens uint8 exports; deconvolution may already remove out-of-focus light
and part of the background, so real-data benefits may be smaller than synthetic
ones. E13 answers that question on real data.

### Existing methods as baselines

| Method | Problem addressed | Known limitation ({doc}`preprocessing-baseline`) |
| --- | --- | --- |
| Min–max | Channel and round gain | The range is set by the single brightest voxel; output forced to uint8; background compressed into few grey levels |
| Histogram matching | Channel distributions differing in shape | Assumes every channel should share one intensity distribution, which holds only when barcode bases are balanced across channels; nonlinear |
| XY reconstruction | Spatial background in XY | Zeroes 70–79 % of voxels on the golden fixture, so the noise threshold is 0 |
| XY white top-hat | Spatial background in XY | Slice-wise; radii in pixels, ignoring Z sampling |

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

* **Problem.** A constant additive offset per channel and round. It biases the
  channel ratios used for colour calling: a channel with a higher offset looks
  partly "on" where it should be off.
* **Cause.** The baseline term.
* **Targeted condition.** `baseline`.
* **Why the benefit may be small.** Post-Huygens exports may already have their
  background subtracted, and the `low` of percentile normalization already
  subtracts a low percentile. In recipe 2 its main role is an offset-free, still
  linear extraction source.

One background level per channel and round, subtracted uniformly.

| Item | Specification |
| --- | --- |
| Config | `ScalarBackgroundConfig(percentile=10.0, fit="fov")` |
| Estimator | The `percentile`-th value by the inverted-CDF definition over all voxels of the channel, computed from its integer histogram for uint8 and uint16, or with `np.percentile(method="inverted_cdf")` for float input |
| Parameters | `percentile` in [0, 100), intensity percent (*provisional* default 10); `fit` is `"fov"` or `"supplied"` |
| Output | `max(x − b, 0)` in the input dtype |
| Fitted values | `{"background": [b_0, ..., b_C-1]}` |
| Constant channel | `b` equals the constant; the output is zero |
| Failure | An invalid percentile, or `fit="supplied"` without a matching section, round or channel, raises |

Only a percentile estimator is offered. After subtraction the zero fraction is
the fraction of voxels at or below `b`: about `percentile / 100` for continuous
data, and more for uint8 input, where many voxels share each grey level.
`percentile = 50` is the median; it zeroes at least half of the voxels, so MAD
and the noise threshold become zero. Any estimator near the centre of the
background distribution behaves the same way, including the mode of a symmetric
background, so neither a median nor a mode estimator is offered. Low percentiles
keep the zero fraction predictable, with `percentile` as the single control.

Sources: the inverted-CDF quantile is definition 1 of Hyndman and Fan (1996),
*Sample quantiles in statistical packages*, The American Statistician 50(4),
361–365, as implemented by `numpy.percentile(method="inverted_cdf")`. Percentile
background subtraction follows the precedent of starfish `ClipPercentileToZero`
(revision `1fb00cbc`), which subtracts its `p_min` percentile after clipping.

## 3D background subtraction

* **Problem.** Spatially varying background that also changes along Z. The
  existing XY methods estimate each slice independently, so the estimate can jump
  between adjacent slices and distort the 3D profile of puncta; their radii are in
  pixels, so Z sampling is ignored.
* **Cause.** The spatial background term: thick-tissue autofluorescence and
  out-of-focus light from neighbouring planes.
* **Targeted condition.** `regions` and `texture` with extent along Z, and
  `gradient` with a Z slope.
* **Why the benefit may be small.** Removing out-of-focus light is the main
  purpose of deconvolution, which catalog data have had. If the background varies
  mainly in XY, the 3D and XY methods may give similar results. The cost is also
  high (below).

A volumetric white top-hat: grey opening with an anisotropic ellipsoidal
footprint, subtracted from the image. It is the 3D counterpart of the existing
XY top-hat.

| Item | Specification |
| --- | --- |
| Config | `Background3DConfig(radius_um_zyx=None, radius_voxels_zyx=None)`; exactly one must be set |
| Footprint | Ellipsoid with semi-axes `r_z, r_y, r_x` voxels: voxels with `(dz/r_z)² + (dy/r_y)² + (dx/r_x)² ≤ 1`. An axis with radius 0 is not filtered (footprint length 1 along it) |
| Radii | From `radius_um_zyx` and `ImageMetadata.spacing_zyx` (µm): `r = round(radius_um / spacing)`. If spacing is unknown, `radius_um_zyx` raises and `radius_voxels_zyx` must be given. There is no default radius |
| Background | `scipy.ndimage.grey_opening(x_c, footprint=ellipsoid)` per channel, with reflect boundaries |
| Output | `max(x − background, 0)` in the input dtype |
| Fitted values | `{"radius_voxels_zyx": [r_z, r_y, r_x]}` |
| Constant channel | Background equals the constant; the output is zero |
| Cost | Grey erosion and dilation with an arbitrary footprint each take about (voxels × footprint voxels) comparisons, single-threaded. There is no fast path and no downsampling |
| Failure | Both or neither radius given, a negative radius, µm radii without spacing, or a footprint larger than the volume along an axis raises |

Choose the radius larger than the puncta: the opening removes bright structures
smaller than the footprint. For synthetic evaluation, use
`r = ceil(3σ) + 1` voxels per axis from the preset's documented puncta widths σ.

Task group 2 measures the cost for several footprint sizes within the shared
image bounds and extrapolates it, labelled as an extrapolation, to the real FOV
size and radii planned for E13. If the extrapolated cost per FOV exceeds the
budget set when E13 is planned, a faster approximation (a separable cuboid
footprint or downsampling) is a separate, bounded follow-up with an agreement
criterion against this exact method.

Sources: grey opening and the white top-hat are defined in Serra (1982), *Image
Analysis and Mathematical Morphology*, and Soille (2003), *Morphological Image
Analysis*, 2nd ed. The implementation is `scipy.ndimage.grey_opening`. starfish
`Filter.WhiteTophat(masking_radius, is_volume=True)` (revision `1fb00cbc`) is
the precedent for a volumetric top-hat in spot-based transcriptomics. It uses an
isotropic ball; this specification uses an anisotropic ellipsoid, because Z
sampling differs from XY.

## Percentile normalization

* **Problem.** Channel and round gain differences, as for min–max, but robust to
  bright outliers. Min–max takes its upper bound from the single brightest voxel,
  so one aggregate or hot pixel sets the scale and compresses the background into
  few grey levels (MAD of 2–3 on the golden fixture). The `p_high` percentile
  ignores the brightest `100 − p_high` percent. It also preserves the input dtype,
  where legacy min–max forces uint8.
* **Cause.** The gain term, plus bright outliers.
* **Targeted condition.** `gain` and `trend`; bright outliers approximated by a
  few bright `texture` blobs.
* **Why the benefit may be small.** Without bright outliers it differs from
  min–max mainly in dtype and in the `p_low` offset.

A linear map of a per-channel percentile range onto the full output range.

| Item | Specification |
| --- | --- |
| Config | `PercentileNormalizationConfig(p_low=1.0, p_high=99.9, fit="fov")` |
| Range | `low`, `high`: the `p_low`-th and `p_high`-th values by the inverted-CDF definition, per channel of the round (`fit="fov"`), or read from the supplied file (`fit="supplied"`) |
| Parameters | `0 ≤ p_low < p_high ≤ 100`, intensity percent (*provisional* defaults 1 and 99.9) |
| Output | `clip((x − low) / (high − low), 0, 1) × dtype_max`, rounded half to even, for unsigned integers; values in [0, 1] for float input |
| Fitted values | `{"low": [...], "high": [...]}` per round, with `p_low` and `p_high` recorded in the config; reusable as supplied values |
| Constant channel or `high == low` | Output zero; `degenerate_range` diagnostic |
| Failure | Invalid percentiles, or supplied values missing for the round or channel, raises |

Values above `high` saturate at the dtype maximum. With `p_high=99.9`, about
0.1 % of voxels saturate by design.

Sources: the inverted-CDF quantile as above. Per-bit percentile normalization is
documented by split-FISH (Goh et al. 2020, *Nature Methods* 17, 689–693; the
`split-fish` repository parameters).

## Pre-normalization extraction source

* **Problem.** Normalization changes the intensities used for colour calling:
  histogram matching is nonlinear, and percentile normalization saturates the
  brightest voxels. Extracting from the background-corrected image before
  normalization keeps the linear channel ratios.
* **Cause.** Nonlinearity and saturation introduced by the intensity step.
* **Targeted condition.** `gain` together with `baseline` or `texture`,
  compared with extraction from the detection image.
* **Why the benefit may be small.** Intensities extracted before normalization
  are not corrected for channel gain; whether that helps depends on how the
  decoder handles channel scale. Only the comparison can show it.

The mechanism is `extraction_source` in {doc}`preprocessing-contract`.

## Fitting modes and sample-level statistics

* **Problem.** Statistics fitted per FOV depend on that FOV's content. In a
  near-empty FOV, the `p_high` percentile or the maximum falls in the noise, so
  normalization stretches noise over the full range and creates false
  detections; a dense FOV is compressed instead.
* **Cause.** FOV content: tissue density and near-empty fields at tissue edges.
* **Targeted condition.** At least three FOVs differing in
  `FormedSceneConfig.density` (dense, sparse, near-empty), with between-FOV
  gain drift as a secondary case.
* **Why the benefit may be small.** When FOVs have similar content, per-FOV and
  sample-level statistics coincide.

`fit="fov"` fits on the current round of the current FOV. `fit="supplied"` reads
values from the step's section of the recipe's `supplied_statistics` file.
Sample-level statistics come from summary passes over the FOVs followed by an
application pass; wiring them into Snakemake belongs to §2.13.

### Recipe stage of a summary

Statistics for a fitted step are summarized **at that step's input**: after all
preceding steps of the recipe, each applied in its own fit mode. Recipe 2's
percentile range therefore comes from background-corrected data, and recipe 1's
histogram reference comes from the reference round after min–max. Each supplied
section records the preceding steps it was summarized after, and the reader
rejects a section that does not match the recipe.

The number of passes follows from this rule:

* Each step with `fit="supplied"` needs one summary pass at its input, and a
  later supplied step can be summarized only after the earlier ones are known.
  With `k` supplied steps, the recipe needs `k` summary passes and one
  application pass.
* Steps without supplied statistics, including every `fit="fov"` step and the
  3D background, are simply executed within each pass.
* **Scalar-background shortcut.** The histogram of `max(x − b, 0)` follows
  exactly from the histogram of `x`: bins above `b` shift down by `b`, and bins
  at or below `b` accumulate in bin 0. When only scalar background steps lie
  between two supplied steps, the later summary is derived from the earlier one
  without another pass. The shortcut applies per FOV before merging (`fit="fov"`)
  and to the merged counts (`fit="supplied"`). A test must show it equals
  summarizing the subtracted volume.

| Recipe | Passes |
| --- | --- |
| Min–max (per FOV) → histogram matching (`supplied` reference) | 2 |
| Scalar background (`fov`) → percentile normalization (`supplied`) | 2 |
| Scalar background (`supplied`) → percentile normalization (`supplied`) | 2 with the shortcut, otherwise 3 |
| 3D background → percentile normalization (`supplied`) | 2; the summary pass runs the 3D background, so its cost is paid twice unless the background-corrected image is kept between passes |

### Summary, merge and statistics

1. **Summary pass.** For each FOV and round, `summarize_histograms` records the
   per-channel histogram at the fitted step's input. For uint8 and uint16 input
   this is `numpy.bincount(values, minlength=dtype_max + 1)`, with all bins.
   Float input requires explicit bin edges and is not used for exact statistics.
2. **Merge.** Histograms from the selected FOVs are summed elementwise. They must
   share dtype, round, channel labels, bin layout and preceding steps. FOVs used
   and excluded, such as near-empty tissue-edge fields, are recorded.
3. **Statistics from merged counts.**
   * The inverted-CDF percentile is the smallest value whose cumulative count
     reaches `p/100 × N`. For `p = 0` it is the smallest value with a nonzero
     count. This equals `numpy.percentile(concatenated, p, method="inverted_cdf")`
     exactly.
   * The histogram-matching reference is the merged count vector of the
     reference round's `reference_channel`. For unsigned input, scikit-image
     `match_histograms` uses the template only through `numpy.bincount`.
     Matching against the merged counts therefore equals matching against the
     concatenated reference volumes exactly. The implementation reproduces
     scikit-image's CDF mapping from counts and is tested against it.

### Supplied file

The file is JSON with schema identifier `starfinder.preprocessing.supplied/1`.
A generic envelope is validated once; each fitted step has one section, keyed by
its step name ({doc}`preprocessing-contract`), and validates it itself.

```json
{"schema": "starfinder.preprocessing.supplied/1",
 "dtype": "uint8", "channel_labels": ["ch00", "ch01", "ch02", "ch03"],
 "fovs_used": ["Position001", "Position002", "Position003"], "fovs_excluded": [],
 "steps": {
   "scalar_background": {
     "summarized_after": [],
     "params": {"percentile": 10.0},
     "fitted": {"round1": {"background": [4, 3, 3, 5]}}},
   "percentile_normalization": {
     "summarized_after": [{"step": "scalar_background",
                           "config": {"percentile": 10.0, "fit": "supplied"}}],
     "params": {"p_low": 1.0, "p_high": 99.9},
     "fitted": {"round1": {"low": [0, 0, 0, 0], "high": [197, 177, 187, 217]}}}}}
```

* **Envelope.** Schema identifier, dtype, channel labels (which must equal the
  run's), and the FOVs used and excluded.
* **Step sections.** `summarized_after` lists the preceding steps (name and
  config) at whose output the statistics were summarized. `params` holds the
  fitting parameters. `fitted` is keyed by round; each entry has the structure of
  that step's `StepResult.fitted`, so values fitted with `fit="fov"` can be
  written as supplied values without conversion.
* **Histogram matching.** Its section's `params` holds the reference round and
  `reference_channel`; `fitted` holds the merged count vector as `values` and
  `counts`.
* **Failure.** A wrong schema, dtype or channel labels, a missing section, round
  or channel, or a `summarized_after` that differs from the recipe raises. A
  recipe that uses the same step name twice with `fit="supplied"` is rejected
  ({doc}`preprocessing-contract`).

The per-FOV histograms are saved as `.npz` arrays of shape
(rounds, channels, bins) with the channel and round labels and the preceding
steps.

## Interaction with noise-mode detection

The local-maxima `noise` threshold uses the median and MAD over all voxels of a
channel.

* **Min–max and percentile normalization.** For the unclipped part of the range,
  a linear map scales the median and MAD equally, so detections change little.
  Quantization to few grey levels and saturation above `high` create plateaus.
* **Histogram matching.** This is a monotonic but nonlinear map, so the effective
  cutoff changes with the reference distribution.
* **Scalar background.** The zero fraction is the fraction of voxels at or below
  `b`: about `percentile / 100`, and more for uint8 input. At the provisional 10
  it is about 10 %, and MAD shrinks far less than after subtracting the median.
* **3D background.** Like XY reconstruction, it is expected to produce many zero
  voxels. The baseline fixture shows XY reconstruction reaching MAD = 0. Task
  group 2 measures the 3D method.

§2.7 records these quantities during detection and warns when MAD is 0 or more
than half of the voxels are zero. It does not change the default threshold.

## Evaluation design for task groups 5 and 6

### Detection operating points

* `threshold_value` sweep: {2, 3, 4, 5, 6, 8, 10, 12, 15}, with default 5.
* Development seeds {0, 1, 2}; held-out evaluation seeds {100, 101, 102}.
* Each recipe is reported at the max-F1 threshold chosen on development seeds and
  at the default threshold.

### Before/after comparison for each method

Each method and mode above, and each existing method as a baseline, gets two
comparisons:

* **Isolated:** no preprocessing versus the method alone.
* **Ablation:** the full recipe versus the recipe without that step. For the
  modes, sample-level versus per-FOV fitting, and the extraction snapshot versus
  the detection image.

Each comparison runs on the targeted condition, on the clean condition as a harm
check, and on the combined condition, in uint8 and uint16 variants:

| Method or mode | Targeted condition |
| --- | --- |
| Scalar background | `baseline` |
| 3D background | `regions`, `texture` and `gradient`, with extent along Z |
| Percentile normalization | `gain`, `trend`; bright `texture` blobs as outliers |
| Sample-level fitting | Multi-FOV density set (dense, sparse, near-empty); gain drift as a secondary case |
| Pre-normalization extraction source | `gain` with `baseline` or `texture` |
| Min–max, histogram matching (baselines) | `gain`, `trend` |
| XY reconstruction, XY white top-hat (baselines) | `regions`, `texture`, `gradient` |

Each comparison reports:

* **Direct metrics for the targeted problem.** Background methods: residual
  error against the background truth, and puncta contrast
  `(peak − local background) / noise`. Intensity methods: the spread of
  true-spot intensities across channels and rounds, and the saturated fraction.
  Sample-level fitting: the spread of per-FOV results. Extraction source:
  per-read colour-call agreement with the true codeword.
* **Downstream metrics.** Detection precision, recall, max-F1 and AUPRC; correct,
  wrong-gene and false-detection reads at both operating points.
* **Diagnostics.** Per-channel zero fraction, median, MAD, noise threshold and
  `mad_zero`.
* **Variation.** The mean and range across held-out seeds.

The task group 6 report is organized by method: problem, condition,
before/after panels, metrics and flag. The panels show the same Z slice and XZ
view before and after, with the display range stated and the ground truth
overlaid; per-channel histograms with the noise threshold marked; and one line
profile through a dim punctum.

If no condition can supply bright outliers, or the density variation for
sample-level fitting, the evaluation records a fixture gap and does not extend
the generator.

### Low-benefit flag (*provisional*)

A method or mode is flagged **low benefit** when either holds:

* on its targeted condition, the primary endpoint (max-F1 or the fraction of
  correct decodes) improves by less than the held-out seed range, or by less than
  2 percentage points;
* on the clean condition, it worsens by more than the seed range.

The 2-point threshold is provisional and may be revised once results are seen.
The flag informs review; it does not remove a method, and the decision to stop
discussing one stays with the chapter author. The same per-method layout is used
for real data in E13, where only images and proxy metrics are available.
