# Preprocessing algorithm specification

**Status: Accepted (W-227, 2026-09-27, at `0d13c26`).**
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

## Evaluation design amendment for the calibrated rerun (Accepted, W-243)

**Status: Accepted (W-243, 2026-09-28), approved with amendments at `57656ae`.**
Drafted in W-242. The approved resolutions of the open choices, including the
amendment to the multi-FOV precondition (C5), are listed under *Approved
resolutions (W-243)* and take precedence over the recommendations in *Open
choices*. This section amends *Evaluation design for task groups 5 and 6* for
the calibrated rerun (W-239) only. The page's *Accepted* status (W-227) still
applies to every other section, which this amendment does not change. It
recommends no preprocessing default.

Where this section and the accepted design differ, the rerun follows this
section. Everything it does not mention stays as accepted and as implemented for
W-233 in `benchmarks/preprocessing_synthetic.py`: the arms and their page
defaults, the before/after comparisons, the direct metrics, the diagnostics,
development seeds {0, 1, 2}, held-out seeds {100, 101, 102}, and the truth
matching.

### Findings this amendment answers

The W-237 review of the W-233 results (evaluation run
`runs/W-233/20260927T234158Z-cf84b743/evaluation/`, outside the repository) and
the W-238 measurement ({doc}`image-statistics`) found:

| ID | Source | Finding |
| --- | --- | --- |
| F1 | W-237 | The development presets were uncalibrated. `combined` at ×12 had an SNR of about 2.3 (W-238 preliminary run), and the single-factor conditions carried no noise. |
| F2 | W-237 | Several targeted conditions did not degrade `none`. In uint16, `none` reached the same max-F1 on `baseline`, `gain`, `trend` and `gain_baseline` as on `clean` on every held-out seed (0.904, 0.912, 0.920); in uint8, `baseline` and `trend` matched `clean`. No method could show a benefit there, so those flags carried no information. |
| F3 | W-237 | Only the noise mode was evaluated. The operating point of `none` sat at the grid minimum 2 in 9 of 13 uint8 conditions. The historical real-data settings, `adaptive` 0.2 and 0.4, were not evaluated. |
| F4 | W-237 | The flag counted a gain in one endpoint while the other collapsed. 3D background on `gradient` (uint8, isolated) raised the correct-decode fraction by 0.53 while max-F1 fell by 0.87, and counted as a benefit. |
| F5 | W-237 | Fixture gaps. The texture blobs were dimmer than the puncta, so no bright-outlier fixture existed (W-233 fixture gap). No condition saturated. W-233's `gain` scaled all channels by 0.5, which leaves channel ratios unchanged. |
| F6 | W-237 | Histogram matching lowered the `clean` correct-decode fraction by 0.84 in both dtypes. The codebook's channel balance was not controlled, so the method's balance assumption could not be tested. |
| F7 | W-237 | The report led with audit tables. The review set a human-summary draft format and asked for nine visualization changes. |
| T1 | W-238 | Real clutter SNR p50 5.97–11.1 and pixel SNR p50 12.9–37.7; the clutter σ is 2–3.3 times the pixel σ. |
| T2 | W-238 | MAD is 0 in 77 % of real volumes, so the noise-mode cutoff equals the median whatever the threshold value. The adaptive cutoff (0.2 × maximum) has a p50 of 43.8–49.2 grey levels. |
| T3 | W-238 | Nothing is saturated (at most 1.5 × 10⁻⁸ of a volume). The channel gain spread is 1.26–3.15 and the round trend 1.04–2.04. |

### 1. Presets

*Answers F1 and F2 (W-237) and T1 (W-238).*

* **Scenes.** Every condition, including `clean`, uses
  `calibrated_scene_preset(condition, dtype, seed=seed, codebook="balanced")`
  (`calibrated-development-v1`, W-241; {doc}`datasets`). The scenes are
  8×64×64 ZYX with four rounds, four channels, 80 amplicons and the balanced
  16-gene `development_codebook`. The preset is used as returned: no intensity
  scale or field change as in W-233. Only the multi-FOV sets and the added
  conditions of item 5 change fields.
* **Conditions.** The 13 names of `CALIBRATED_CONDITIONS`, the same names as in
  W-233, plus the conditions of item 5. `clean` carries the calibrated baseline:
  a uniform pedestal, Poisson, white and spatially correlated noise, mild
  channel gains and a mild round trend. Each single-factor condition adds its
  factor on top of that baseline, so no evaluation scene is noise free.
  Noise-free scenes remain unit-test fixtures only.
* **Multi-FOV sets.** As in W-233: `mf_density` (counts 80, 20 and 2 on
  `combined`), `mf_gain_drift` (readout gains × 1, 0.75 and 0.5 on `combined`)
  and `mf_density_clean` (counts 80, 20 and 2 on `clean`). Each FOV is the
  calibrated configuration with `count`, `FOV_id`, `scene_key` and the readout
  `gains` replaced, and with `coordinates`, `amplicon_ids` and `gene_ids`
  cleared.
* **Dtypes.** uint8, the measured scale, and uint16 at ×16, which is unverified
  against real data (see choice C2 and reductions R2 and R3).
* **Truth.** Reference round `round1`. The background and signal truths are
  generated as in W-233; the background truth includes the pedestal.
* **3D background radius.** The accepted rule `r = ceil(3σ) + 1` uses the
  preset's median widths: axial 1.5 and lateral 1.3 voxels, which are the
  exponentials of the lognormal locations, not the locations themselves. It gives
  (6, 5, 5) ZYX. A Z footprint of 13 voxels is longer than the 8 planes, so
  `background_3d` raises. The rerun uses (3, 5, 5):
  `r_z = floor((8 − 1) / 2) = 3` is the largest Z radius whose footprint fits
  (choice C4).

### 2. Threshold modes

*Answers F3 (W-237) and T2 (W-238).*

| Mode | Cutoff per channel | Grid | Fixed operating points |
| --- | --- | --- | --- |
| `noise` | `median + v × 1.4826 × MAD` over the channel | {2, 3, 4, 5, 6, 8, 10, 12, 15} | 5 |
| `adaptive` | `v ×` the channel maximum | {0.1, 0.15, 0.2, 0.25, 0.3, 0.4} | 0.2 and 0.4 |

* **Detection.** `LocalMaximaConfig(threshold_mode=mode, threshold_value=v)`
  with its other fields at their defaults, applied to the reference round after
  the recipe, as `FOV.run` detects. A maximum is kept when its peak is strictly
  above the cutoff. The adaptive grid brackets the historical real-data settings
  0.2 and 0.4.
* **Sweep.** For each mode, as in W-233's `sweep`: one `FOV.run` at the mode's
  lowest grid value; every higher value is the subset of spots whose
  `peak_intensity` is above that value's cutoff. For every arm on seeds 0 and
  100, a direct run at the mode's first fixed point (5 or 0.2) must give the same
  spots and reads, or the run stops.
* **Per seed and mode.** Precision, recall and F1 at every grid value (F1 is 0
  when nothing is detected), reads, the correct-decode fraction (correct accepted
  reads divided by the number of truth amplicons) and AUPRC over the mode's own
  grid (W-233 formula).
* **Operating point.** For each mode, condition, dtype and arm (multi-FOV: each
  mode, set, dtype and arm, with F1 pooled over the set's FOVs as in W-233): the
  mean F1 over development seeds 0, 1 and 2 at each grid value, in float64 with
  seeds in ascending order. The highest mean wins; exact ties go to the smallest
  grid value. A selection at a grid end (2 or 15; 0.1 or 0.4) is marked
  `at_grid_edge` in the operating-point table and on the method card. The grid
  is not extended.
* **Endpoints.** Per held-out seed and mode: max-F1 is the maximum F1 over the
  mode's grid on that seed; the correct-decode fraction is taken at the mode's
  development-selected value. Both are also reported at the fixed operating
  points.
* **Side by side.** Every table carries a `threshold_mode` column. The
  precondition, the flag and every figure are computed per mode and shown next
  to each other. Nothing is pooled across modes, and neither mode is preferred.

### 3. Precondition check

*Answers F2 (W-237).*

The check is a property of the fixture. It is computed once for each targeted
condition `c`, dtype and mode, and applies to every method that targets `c`.

* **Inputs.** Arm `none` on `c` and on `clean` (for multi-FOV sets:
  `mf_density_clean`, with endpoints pooled over the FOVs); held-out seeds
  `s` ∈ {100, 101, 102}; endpoints `E` ∈ {max-F1, correct-decode fraction} as
  defined in item 2. Seeds pair because the conditions share the scene key: one
  seed has the same amplicons and noise draws in every condition (W-241).
* **Degradation.** `g_E = mean_s [E(none, clean, s) − E(none, c, s)]`.
* **Seed range.** `R_E = max(range_s E(none, clean, s), range_s E(none, c, s))`,
  where a range is the maximum minus the minimum over the three seeds (W-233's
  seed range).
* **Rule.** `c` degrades `E` when `g_E > R_E`, strictly; equality does not
  count. `c` is a valid target for this dtype and mode when it degrades at least
  one endpoint. An undefined endpoint does not degrade.
* **Failure.** A fixture-gap row records the condition, dtype, mode, both
  degradations and both ranges. The condition is left out of the flag of every
  method that targets it, for that dtype and mode. Its comparison rows are still
  reported and marked `fixture_gap`.
* **Scope.** The check is not applied to `clean`, to `combined` (reported
  only) or to `clean_unbalanced` (a harm test, item 5).
* **Multi-FOV sets: a second, set-specific check (W-243, C5).** Both checks run
  for each multi-FOV set, dtype and mode, and both are reported.
  * The check above uses `none` on the set against `none` on
    `mf_density_clean`. It tests the calibrated `combined` base of the set.
  * A second check tests the variation the set was built for. Its arm is the
    per-FOV-fitted recipe arm of the sample-level-fitting comparison (`r1`,
    `r2_scalar` or `r2_3d` with `fit="fov"`). Its reference FOV is the dense
    FOV of `mf_density` or the gain-1.00 FOV of `mf_gain_drift`. Its degraded
    FOV is the near-empty FOV or the gain-0.50 FOV.
  * The second check uses the same formulas with the reference FOV in place
    of `clean` and the degraded FOV in place of `c`: the endpoints of each
    FOV on each held-out seed, `g_E`, `R_E`, the strict rule, undefined
    endpoints and rounding. It is evaluated separately for each recipe arm.
  * A multi-FOV set is a valid target for the sample-level-fitting flag of a
    recipe only if the second check passes for that recipe's per-FOV-fitted
    arm. When it fails, a fixture-gap row names the set, recipe, dtype and
    mode, and the set is left out of that flag.
* **Rounding.** Every mean, range and delta in items 3 and 4 is rounded to 10
  decimal places before it is compared, so that values on the `1/n` lattice of
  the fractions compare exactly.

### 4. Revised low-benefit rule (*provisional*)

*Answers F4 and F2 (W-237).*

For each comparison (method, comparison, before arm `b`, after arm `a`), dtype
and mode:

* **Inputs.** Held-out seeds 100, 101 and 102; the endpoints of item 2; for
  each condition `c`, the paired delta
  `Δ_E(c) = mean_s [E(a, c, s) − E(b, c, s)]` and the seed range
  `R_E(c) = max(range_s E(b, c, s), range_s E(a, c, s))`.
* **Eligible targets.** `T*` is the set of the comparison's targeted
  conditions that pass item 3 for this dtype and mode.
* **Benefit on `c` in `T*`.** Some endpoint `E` has
  `Δ_E(c) ≥ max(R_E(c), 0.02)`, and the other endpoint `E′` has
  `Δ_E′(c) ≥ −R_E′(c)`.
* **Clean holds.** Both endpoints have `Δ_E(clean) ≥ −R_E(clean)`.
* **Result.**
  * *Not flagged* when some `c` in `T*` shows a benefit and `clean` holds.
  * *Not assessable* when `T*` is empty. No flag is given; the reason lists the
    fixture gaps, and a `clean` failure is still reported.
  * *Low benefit* otherwise.
* **Ties and gaps.** The benefit bound is inclusive (`≥`); a worsening exactly
  equal to the seed range does not count as harm. When a delta or range that a
  clause needs is undefined, that clause is not met, and the case is listed as an
  anomaly.
* **Multi-FOV sets.** Endpoints are pooled over the set's FOVs per seed, and
  `clean` is `mf_density_clean`, as in W-233.
* **Reporting.** Each targeted condition gets a row stating which clause
  failed. `combined` is reported and does not enter the flag. The 0.02 threshold
  stays provisional, and the flag still only informs review.

For the rerun, this rule replaces the two bullets of *Low-benefit flag*.

### 5. Added conditions

*Answers F5 and F6 (W-237) and T3 (W-238).*

Each is defined on top of the calibrated `clean` of the same dtype and seed, and
each targeted one is subject to item 3.

| Condition | Definition | Role |
| --- | --- | --- |
| `bright_outliers` | Texture enabled with `TextureConfig(count=4)`, fixed axial width 1.5 and lateral width 2.0 voxels, and lognormal brightness with median 4 × 88 = 352 grey levels (× 16 in uint16) and log SD 0.1; `tissue_weights` 1 for every round and channel. The median is above the puncta's p99 amplitude (about 256). In uint8 the blob cores clip at 255 | Targeted for percentile normalization, replacing bright `texture` blobs as the outlier proxy. Min–max and histogram matching are reported on it, outside their flags |
| `saturation` | Every intensity parameter that the uint16 variant multiplies by 16 (brightness median, pedestal, Poisson α, white σ and correlated σ) is multiplied by `k`. `k` is the smallest value of `2^(j/4)`, `j` = 0…32, for which the mean clipped fraction over development seeds 0–2 reaches `f`. The clipped fraction is the generator's `above` clipping count summed over rounds and channels, divided by rounds × channels × voxels. `k` is set per dtype before any held-out scene is generated; `k` and each seed's fraction are recorded. `f` = 10⁻³ is proposed (choice C1) | Targeted for the extraction-source mode, with `gain_baseline` and `gain_texture` |
| `gain` | W-241's calibrated `gain`: channel factors 1.08, 1.02, 0.98 and 0.93 on the clean gains 1, 0.94, 0.88 and 0.83 (measured spread 1.68). It replaces W-233's uniform gain | Targeted for min–max, histogram matching and percentile normalization, as before, and part of `gain_baseline` and `gain_texture` |
| `clean_unbalanced` | `codebook="unbalanced"`: every round uses the four colours 7, 5, 3 and 1 times | Histogram matching's targeted harm test |

**Harm test.** For each histogram-matching comparison, dtype and mode, the test
shows harm when either endpoint has
`Δ_E(clean_unbalanced) < −R_E(clean_unbalanced)` (item 4 notation). It is
reported on the method card beside the balanced `clean` deltas, so that the
codebook's share is visible. It does not enter the flag (choice C7).

### 6. Report

*Answers F7 (W-237).*

**Human summary first**, in the W-237 draft format, before any audit table:

1. *Setup.* The recipes and their steps (with page defaults); the conditions,
   each with its noise level as the clutter and pixel SNR p50 measured with the
   {doc}`image-statistics` tool on development seeds, as in W-241's
   side-by-side run; the comparison types (isolated, ablation, sample-level
   against per-FOV fitting, extraction source against detection image, the
   harm test); what each metric measures; the precondition results per
   condition, dtype and mode; the seeds, dtypes, modes and grids; the reductions
   applied and the wall time; and the qualification (development evidence, no
   defaults, provisional flag).
2. *One card per method or mode*, in the order of this page, each giving:
   * what the method does and the problem it addresses;
   * the targeted conditions and their precondition status;
   * the key before/after numbers, with seed ranges, in both threshold modes:
     the direct metrics for its problem (background RMSE and bias, puncta
     contrast, intensity spread across channels and rounds, clipped and
     saturated fractions, as they apply), the downstream metrics (max-F1,
     AUPRC, correct-decode fraction, wrong-gene and false-detection reads), and
     the `clean` harm check;
   * the flag verdict per dtype and mode with the failing clause, and for
     histogram matching the harm test;
   * caveats, such as fixture gaps, grid-edge selections or a deviation from this
     section.
3. *Cross-cutting findings and anomalies.* Each is labelled *finding* (shown by
   the tables) or *hypothesis* (an explanation not tested by this run) and names
   the table it comes from. Anomalies include grid-edge selections, fixture gaps,
   undefined endpoints, MAD-zero channels and unexpected clipping.
4. *Reading order*, with anchors to the cards and figures.

**Appendix.** Identity, revision and checksums (the manifest), the full
per-dtype tables, per-seed values, operating points, diagnostics, and the
reductions with their projections.

**Visualization**, following the W-237 visualization comment of 2026-09-28:

1. One shared linear display range for the before and after panels, stated on
   the figure, plus a difference panel (after − before) or a residual-against-truth
   panel.
2. For each method, a grid with rows for the targeted condition, `clean` and
   `combined`, and columns for before, after and difference.
3. Overlays of true positives, false positives and misses at the
   development-selected threshold of each mode.
4. A colour-calling view: for a few truth puncta, including the dim punctum of
   change 9, the channel × round intensity vector before and after, as a heatmap
   or bars, with the true colour sequence marked.
5. For background methods, the estimated background against the truth
   background on the same slice, and the residual.
6. Histograms of the signal-carrying channels only, marking both the noise
   cutoff and the detection threshold actually used at the development-selected
   operating point.
7. A per-method dot plot of Δ max-F1 and Δ correct-decode fraction, with
   seed-range error bars, for the targeted conditions, `clean` and `combined` in
   both dtypes and both modes, with the provisional flag band (±`max(range,
   0.02)`) shaded.
8. PR curves as small multiples that show only each method's before/after pair,
   one panel per condition and mode. Curves that collapse to a point (MAD 0) are
   marked or dropped, and the choice is stated.
9. Larger panels cropped around the dim punctum, so that the XZ view is legible.
   The dim punctum is the emitting reference-round truth punctum with the
   lowest realized peak whose centre is in bounds, on held-out seed 100; the
   line profile of the accepted design passes through it.

### Resource projection

Scaled from W-233's measured full run, without running anything.

* **A1, source.** W-233 took 2006 s wall time and 1969 s of compute: 1155 s for
  13 conditions with 14 single-FOV arms and 814 s for three multi-FOV sets of
  three FOVs with 7 arms (`multi_fov_recipes`), each over two dtypes and six
  seeds, at 16×64×64 ZYX with three rounds, in noise mode only.
  The remaining 37 s is fixed overhead.
* **A2, scene size.** Cost is linear in rounds × voxels: (4 × 8) / (3 × 16) =
  2/3. This is conservative: halving Z in the W-233 pilot cut the cost per unit
  to 0.43, not 0.5 (34.6 s to 14.8 s), and the capped Z radius shrinks the 3D
  footprint.
* **A3, condition cost.** Every calibrated condition carries noise, so each
  condition and dtype is costed at W-233's most expensive condition
  (`combined_geometry`, 70.5 s per dtype), giving 47.0 s. Each multi-FOV set
  and dtype is costed at W-233's `mf_density` (147.8 s per dtype), giving
  98.5 s.
* **A4, second mode.** Adding the adaptive mode multiplies the cost by `m`. The
  projection uses `m = 2`, as if everything were repeated. `m = 1`, a lower
  bound, treats detection as free. Generation, preprocessing, registration and
  the direct metrics are shared, so the true factor lies between.
* **A5, matrix.** 16 conditions (13 plus `bright_outliers`, `saturation` and
  `clean_unbalanced`) with W-233's 14 single-FOV arms, and 3 multi-FOV sets
  with its 7 multi-FOV arms, each in 2 dtypes with 6 seeds, and 60 s for the
  saturation `k` search (about 80 scene generations). Each part is scaled from
  the measured cost of the same arms, so no arm count is extrapolated.
* **A6, exclusions.** Report rendering is not included, because W-233's run did
  not measure it. W-239's pilot times it.
* **Budget.** 2700 s, W-233's 45 minutes (choice C3).

Proposed pre-authorized reductions, applied in order after W-239's own pilot
projection, stopping as soon as the projection fits:

* **R1.** Multi-FOV sets in uint8 only.
* **R2.** uint16 only for `clean`, `combined` and `bright_outliers` (the
  outlier test needs a scale without clipping).
* **R3.** No uint16 at all.

If the projection still exceeds the budget, W-239 returns blocked. Seeds,
uint8 conditions, arms, threshold modes and grid values are never dropped.

| Matrix | Projected wall time, `m = 2` | Lower bound, `m = 1` |
| --- | --- | --- |
| Full, no reduction | 4285 s (71 min) | 2191 s |
| R1 alone | 3694 s | 1896 s |
| R2 alone | 3064 s | 1580 s |
| R3 alone | 2191 s | 1144 s |
| R1 then R2 | 2473 s (41 min) | 1285 s |
| R1, R2 and R3 | 2191 s (37 min) | 1144 s |

At `m = 2` the full matrix exceeds 2700 s, and R1 then R2 brings it within
budget. Adding `gain_strong` (choice C6) adds 188 s without reductions and
94 s after R2 (2567 s after R1 and R2).

### Open choices

| ID | Choice | Options | Recommendation |
| --- | --- | --- | --- |
| C1 | Saturation fraction `f` | 10⁻⁴; 10⁻³; 10⁻² | 10⁻³. Calibrated `clean` already clips about 8 × 10⁻⁵ in uint8 (W-241), so 10⁻⁴ is barely different; 10⁻³ equals the fraction percentile normalization saturates by design at `p_high` 99.9; 10⁻² clips the cores of most puncta, far beyond real data (T3) |
| C2 | uint16 in the matrix | Every condition; only `clean`, `combined` and `bright_outliers`; none | Only the three. The uint16 scale is unverified (W-238), but the numerical policy must hold for uint16, and the outlier test needs a scale without clipping. This equals reduction R2 |
| C3 | W-239 time budget | 2700 s, as W-233; 3600 s; the full matrix at about 4300 s | 2700 s with R1–R3 pre-authorized; the projection after R1 and R2 is 2473 s |
| C4 | 3D background radius on 8-plane scenes | Cap `r_z` at 3; generate 16 planes for every condition; `r_z = 0` (XY-only opening) | Cap at 3. Sixteen planes would change the calibrated crowding (W-241) and double the cost, and `r_z = 0` is no longer a 3D method. The cap is a stated deviation from `ceil(3σ) + 1` for this rerun |
| C5 | Precondition reference for multi-FOV sets | `none` on the set against `none` on `mf_density_clean`, as drafted; the per-FOV-fitted recipe on the near-empty FOV against the dense FOV | As drafted, to keep one rule for every condition. It tests the `combined` base rather than the density variation, and the method card says so |
| C6 | A stronger gain condition | None; `gain_strong` with the LN spread of 3.15 (T3) | Add `gain_strong`. The calibrated `gain` spread of 1.68 may not degrade `none`, and 3.15 is the largest measured spread |
| C7 | Histogram matching's harm test in its flag | Reported beside the flag; counted as `clean` harm | Beside the flag. The unbalanced codebook violates the method's stated assumption; folding it into the flag would mix two questions |
| C8 | Bright-outlier brightness | 2.5 × the brightness median (no uint8 clipping, but not above the brightest puncta); 4 ×; 8 × | 4 ×: above the puncta's p99 amplitude, with the uint16 variant as the test without clipping |

### Approved resolutions (W-243)

Jiahao approved this amendment with amendments on 2026-09-28 (W-243), at
`57656ae`. These resolutions replace the recommendations above wherever they
differ.

| ID | Resolution |
| --- | --- |
| C1 | Saturation fraction `f = 10⁻³`. |
| C2 | uint16 only for `clean`, `combined` and `bright_outliers` (equal to reduction R2). |
| C3 | A 2700 s budget for W-239's full run, with R1, R2 and R3 pre-authorized in that order. |
| C4 | `r_z` capped at 3 on the 8-plane scenes. This is a stated deviation from `ceil(3σ) + 1` for this rerun only. |
| C5 | **Amended:** both multi-FOV precondition checks run and are reported. A set counts for a recipe's sample-level-fitting flag only if the set-specific check passes (item 3, *Multi-FOV sets*). |
| C6 | `gain_strong` is added, with the LN channel gain spread of 3.15. |
| C7 | Histogram matching's unbalanced-codebook harm test is reported beside its flag, not inside it. |
| C8 | Bright outliers at 4 × the brightness median, with uint16 as the test without clipping. |

The projection with these resolutions (`m = 2`, R1 then R2, `gain_strong`
added) is 2567 s, within the 2700 s budget. The second multi-FOV check reuses
the per-FOV endpoints that the run already computes, so it adds no generation or
detection.
