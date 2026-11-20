# Validation design: the segmentation input recipe

% Status: Approved with its specification (Jiahao, 2026-11-18, W-885), with amendments A and B3; written against 76cd058

This is the validation plan for implementing
[segmentation-input-recipe](../specs/segmentation-input-recipe.md). It covers unit tests
per step and per stage, both routes on the synthetic fixtures, the records, the replaced
keys, the preview tool and a first use on the four FOVs of W-870's stage 1. The first use
is a development check and not a scientific sweep. It makes no segmentation-quality claim,
and no row on this page accepts or rejects a recipe by a quality figure.

The expected values below are computed from the specification and from the library calls
and formulas named in each row. They do not come from running the new code.

**Tolerances** follow `docs/contributing.md` ("Expected values and tolerances"):

* A computed float32 array or value compares with `rtol=1e-6, atol=1e-6`, and a computed
  float64 one with `rtol=1e-9, atol=1e-12` (`|actual − expected| ≤ atol + rtol·|expected|`).
  This holds even when the expected array comes from the same library call as the code
  under test.
* A physical length or coordinate with a declared grid precision compares at that
  precision (1e-9 µm).
* "Exact" is used only for integers, counts, labels, integer images, strings, statuses and
  booleans; for floats that hold integer values; for hashes and file round trips; and for
  the A19 parity hashes. Every exact cell names which of these it is.
* A row that compares more than one quantity gives each its own tolerance.

## Fixtures

Every routine fixture is at most 32 × 64 × 64 voxels with at most four channels, as the
project guidance bounds them. The fixtures marked `slow` are existing ones.

| Fixture | Construction | Needs |
| --- | --- | --- |
| `delta` | A float32 image of 1 × 63 × 63, value 1 at (0, 31, 31) and 0 elsewhere; spacing (1.0, 0.1, 0.1) µm. | hand-built |
| `blob_pair` | One continuous Gaussian blob of σ₀ = 0.5 µm at the centre of a 6.4 µm square, sampled on two grids: 1 × 64 × 64 at 0.1 µm and 1 × 32 × 32 at 0.2 µm, float32. | hand-built |
| `stacks` | A uint16 stack of 32 × 4 × 4 with every value 65,535 (sum 2,097,120). Two uint32 stacks of 2 × 4 × 4: planes 8,388,608 and 8,388,608, which sum to 2²⁴ = 16,777,216 (float32 holds it), and planes 8,388,608 and 8,388,609, which sum to 2²⁴ + 1 (float32 does not hold it). Two int32 stacks of 2 × 4 × 4: planes −8,388,608 and −8,388,608, which sum to −2²⁴ = −16,777,216 (float32 holds it), and planes −8,388,608 and −8,388,609, which sum to −2²⁴ − 1 = −16,777,217 (float32 rounds it to −16,777,216). A uint8 stack of 3 × 4 × 4 with planes 0, 1 and 1 (mean 2/3). A float32 stack of 2 × 4 × 4 with both planes 2 × 10³⁸ (sum 4 × 10³⁸, beyond float32's largest finite value 3.4028235 × 10³⁸). | hand-built |
| `seg_golden` | The fixture of `test_segmentation_golden.py` (`fixture()`: DAPI, amplicon and Flamingo, uint8 of 16 × 64 × 64), with its pinned composite and Flamingo digests. | existing |
| `morphology_fov` | The FOV of `test_segmentation_inputs.py`: a sequencing round and a morphology round registered by `FOV.register`, and the reference merged image. | existing |
| `scene_plane_small` | The single-plane sample of [sample-segmentation validation](sample-segmentation.md) ("Fixtures"): 3 × 3 FOVs of 1 × 96 × 96, written by `write_scene_plane`, with the stain as `dapi` and `flamingo`. This is the tissue-2D case (Z = 1). | existing; `multiview-stitcher` extra; `slow` |
| `scene_plane` | The same on `sample-tiny-v1` (FOVs of 1 × 36 × 36), for the routine tier. | existing; `multiview-stitcher` extra |
| `fused_pair` | Two FOVs of 3 × 24 × 24 at spacing (1.0, 0.5, 0.5) µm, fused at `k` 1 and 2 (Z > 1). | existing; `multiview-stitcher` extra |
| `recipes` | The recipes of the specification's example (`pi_max`, `pi_smooth`, `pi_composite`) with lengths chosen for the fixture's spacing, and one recipe per refusal of the rows below. | hand-built |
| `model_stub` | Model folders written by the test (as `length_configs` does), with a StarDist `config.json` of `n_dim` 2 and one of `n_dim` 3, so that `resolve_model` passes. The library call is replaced by a stub that records the `scale` or `diameter` it receives and returns the `truth_import` labels. No model is loaded. | hand-built |

(segmentation-input-recipe-checks)=

## Checks

"Needs": **H** hand-built fixtures only; **MV** the `multiview-stitcher` extra; **SD** the
`stardist` extra; **S** `slow`; **L** `learned`. A test that needs an extra skips without
it, naming the extra, so the CI environment (extras `dev`, `local-registration`,
`checkpoint`) passes or skips every row. In the Tolerance column, "f32" means `rtol=1e-6,
atol=1e-6` and "f64" means `rtol=1e-9, atol=1e-12`.

### Steps (ST)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| ST1 | `gaussian` on a delta | `delta`, `sigma_um` 0.3 | `used.sigma_px_zyx` = (0, 3, 3), computed as 0.3 / 0.1 per axis. The output equals `skimage.filters.gaussian(delta, sigma=(0, 3, 3))` and sums to 1. Its second moment about the centre on Y and on X equals the variance of the truncated sampled kernel, `Σ k² w_k / Σ w_k` with `w_k = exp(−k²/18)` for k = −12 … 12 (radius `int(4·3 + 0.5)`). A scratch computation gave 8.995245 for both. | sigma: f64; output array: f32; sum: f32; moment: f32 | H |
| ST2 | `gaussian` means the same in µm on two grids | `blob_pair`, `sigma_um` 0.3 | On each grid, the output's spatial variance in µm² equals the sampled blob's variance plus the truncated kernel's variance (the ST1 formula at σ = 3 and 1.5 pixels) × s². The variances of a convolution add. Both formulas are near σ₀² + 0.3² = 0.34: a scratch computation gave 0.339952 (0.1 µm) and 0.339981 (0.2 µm), and the outputs agreed with them to about 1e-6 relative. The two grids differ only by the kernel's truncation, which removes at most 0.12 % of its variance (the tail beyond 4σ of a Gaussian holds 0.11 %). | each grid against its formula: f32; the two grids: within the stated truncation bound, 0.12 % of 0.09 µm² | H |
| ST3 | `white_tophat` and `reconstruction` in µm | `seg_golden` DAPI plane (uint8), spacing 0.1 µm; `radius_um` 0.25 and 0.24 | `radius_yx` 3 for 0.25 (`floor(2.5 + ½)`; Python's `round` would give 2) and 2 for 0.24. Through `run_step` on the uint8 plane, the output equals `filter_tophat` / `correct_background` with that `radius_yx`. Unequal Y and X spacing raises `ValueError`. | radius: exact (integers); output: exact (integer image); refusal: exact (strings) | H |
| ST4 | `percentile_normalization` with a fit region | A float32 plane of 1 × 64 × 64 with a zero border outside a mask | `fitted.low` and `fitted.high` equal `np.percentile(x[mask], p, method="inverted_cdf")`. The output equals `clip((x − low) / (high − low), 0, 1)` evaluated by the test. Without a mask the values are those over the whole image. | fitted values: f32; output array: f32 | H |
| ST5 | `clahe` | A float32 ramp plane of 1 × 64 × 64, spacing 0.1 µm, `kernel_um` 1.6, `clip_limit` 0.01 | `used.kernel_px_zyx` = (1, 16, 16). The output equals `equalize_adapthist(clip((x − lo)/(hi − lo), 0, 1), kernel_size=(1, 16, 16), clip_limit=0.01, nbins=256).astype(float32)`, and `fitted.in_range` = [lo, hi] of the fit region. A constant plane gives zeros with `degenerate_range` true. A missing `kernel_um` is refused. | kernel: exact (integers); output array: f32; `in_range`: f32; constant plane: exact (floats holding integer values, all 0); flag and refusal: exact (booleans, strings) | H |
| ST6 | `gaussian` and `clahe` inside `FOV.run` | A uint16 plane of 1 × 64 × 64 in a `PreprocessingRecipe` | `run_step` accepts both. The dtype is uint16, and the values are `np.rint` of the float result (Gaussian with `preserve_range=True`; CLAHE scaled to 65,535). | exact (integer image; dtype string). A test that computes the float result separately before rounding may differ by 1 where the float lies within f64 of a half-integer, and allows that difference there only. | H |
| ST7 | Refusals | `recipes` | Each refusal raises `ValueError` naming the step and the rule: `min_max_normalization`, `histogram_matching`, `scalar_background` and `background_3d` in a recipe; `fit: supplied`; `radius_yx` or `median_radius_px` in a recipe; both `radius_yx` and `radius_um`; a role not among the run's channels; a µm field on a grid without spacing (naming `voxel_size_xy`). | exact (exception types and strings) | H |
| ST8 | Registry | — | `PREPROCESSING_METHODS` holds `gaussian` and `clahe` with category `contrast`, scope `per_channel` and dtype policy `preserve`. The preprocessing reference's generated method table lists them. | exact (strings) | H |

### Stages (RC)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| RC1 | The empty recipe is today's input | `morphology_fov`; `seg_golden` | `apply_input_recipe` with `recipe=None` returns the sources stacked as ZYXC in their dtype. Its array SHA-256 equals that of `assemble_input` at `76cd058` for the same run without `prepare` and `projection`. The hashes are taken from a test at the base revision and written into this test as constants. | exact (hashes; dtype strings) | H |
| RC2 | The order of the stages and `roles` | `recipes` on two channels | A step restricted to `nuclear` leaves the other channel's array SHA-256 unchanged. A trace of the `run_step` calls shows `channels`, `combine`, `projection`, `after_projection` in that order. Each stage's `input` hash equals the previous stage's `output` hash. | exact (hashes; strings) | H |
| RC3 | `combine` | `seg_golden` | `composite` with no channel step gives the pinned `COMPOSITE_DIGESTS`, and `flamingo_enhance` with `median_radius_um` = 1 × spacing gives `FLAMINGO_DIGEST`, both unchanged. After a float32 `gaussian` on both sources, the output is uint8 and equals `composite_nuclei_amplicon` of the two step outputs, and each stretched input lies in [0, 1]. Wrong source roles raise `ValueError` naming both role lists. | digests: exact (hashes); uint8 output: exact (integer image); range: exact (booleans); refusal: exact (strings) | H |
| RC4 | `projection`: max, sum and mean of integers | `stacks` (uint16) | `max` keeps uint16, every value 65,535. `sum` gives float32 2,097,120 (32 × 65,535) with `exact` true. `mean` gives 65,535.0 with `exact` null and `rounding` stated. `output_dtype: float64` gives float64 with the same values. `conversion` is refused. The frame ends in `/projection:mean` for `mean`. | values: exact (integer image, and floats holding integer values); flags and frame: exact (booleans, strings) | H |
| RC5 | The `exact` flag at and above 2²⁴ in absolute value | `stacks` (the two uint32 and the two int32 stacks) | The sum 16,777,216 = 2²⁴ records `exact` true and the value 16,777,216.0. The sum 16,777,217 records `exact` false, and its float32 value is 16,777,216.0, the one rounding of the float64 result. The signed sum −16,777,216 = −2²⁴ records `exact` true and the value −16,777,216.0. The signed sum −16,777,217 = −2²⁴ − 1 records `exact` false, and its float32 value is −16,777,216.0. With `output_dtype: float64`, all four are `exact` true and hold their integer sums. | exact (booleans; floats holding integer values) | H |
| RC6 | A mean that is not an integer | `stacks` (uint8 planes 0, 1, 1) | `mean` gives float32 values equal to 2/3, with `exact` null and `rounding: "float64 result rounded once to float32"`. With `output_dtype: float64` the value equals 2/3 in float64. | value: f32 (float64 output: f64); flag and record: exact (strings) | H |
| RC7 | The overflow check | `stacks` (float32 planes 2 × 10³⁸) | `sum` with the default float32 output raises `ValueError` naming the run, the projection, the largest magnitude reached and `output_dtype: float64`, and no stage output is produced. With `output_dtype: float64` the sum equals the float64 sum of the two float32 values (about 4 × 10³⁸), recorded as `max_abs`. `mean` with float32 output passes and equals the float32 input value. | refusal: exact (exception type and strings); float64 sum: f64; mean: f32 | H |
| RC8 | The stain of `extend_labels_through_z` | `morphology_fov`, a recipe with `gaussian`, `projection: max` and the extension | The stain passed to `extend_labels_through_z` is the `gaussian` output before the projection: its array SHA-256 equals the `channels` stage's `output`. A recipe without a projection, or with two model channels, is refused. | exact (hashes; strings) | H |
| RC9 | Recipe identity | `recipes` | One recipe given by name and inline has one `sha256`. Changing any field changes it. | exact (hashes) | H |

### Model input (MI)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| MI1 | The factor from a pixel size | Grids of spacing 0.094635 and 0.18927 µm, `pixel_size_um` 0.37854 | `factor_zyx` = (1, 0.25, 0.25) and (1, 0.5, 0.5) after rounding to six significant digits. `factor_zyx_exact` holds the unrounded quotient `s / p`. | f64 for both (computed float64 values) | H |
| MI2 | `factor` per route | `factor` 0.25 at `k` 1 and 2 | 0.25 per FOV; 0.5 after fusion (LS2's rule for the old `scale`). | f64 | H |
| MI3a | StarDist, a 2D model | `model_stub` (StarDist, `n_dim` 2), `factor_zyx` (1, 0.25, 0.25) | The stub receives `scale=(0.25, 0.25)`, Y and X only, as `library_scale` passes it to a 2D model today (`_learned.py:429`). `effective.scale_zyx` records (1, 0.25, 0.25). A factor of 1 on every axis passes `scale=None`. A 2D model with a Z factor other than 1 is refused as today. | tuple length: exact (integers); values: f64; None and refusal: exact (strings) | H |
| MI3b | StarDist, a 3D model | `model_stub` (StarDist, `n_dim` 3), `factor_zyx` (1, 0.25, 0.25) | The stub receives `scale=(1, 0.25, 0.25)`, all three axes. `effective.scale_zyx` records the same. | tuple length: exact (integers); values: f64 | H |
| MI4b | Cellpose, a 3D run (amendment A) | `model_stub` (Cellpose, Z > 1), grid spacing (1.0, 0.25, 0.25) µm, `anisotropy` 4 | A number `pixel_size_um` 1.0 (`f_yx` 0.25) passes `diameter` 120 and `anisotropy` 4 unchanged. `effective.scale_zyx` records (1.0, 0.25, 0.25), where 1.0 = 4 × 0.25 is Cellpose's Z factor. A ZYX `pixel_size_um` (1.0, 1.0, 1.0) (Z factor 1.0) is accepted. A ZYX `pixel_size_um` (2.0, 1.0, 1.0) (Z factor 0.5) is refused with `ValueError` naming the run, 0.5 and 1.0. | diameter, anisotropy and factors: f64; refusal: exact (strings) | H |
| MI4 | Cellpose receives the diameter | `model_stub` (Cellpose) | Factor 0.25 gives `diameter` 120. Factor 1 gives `diameter` 30, which Cellpose 4.2's `eval` turns into `rescale` 1.0, as for `diameter=None` (`rescale = 30. / diameter`). Unequal Y and X factors are refused. | diameter: f64; refusal: exact (strings) | H |
| MI5 | Refusals | `recipes` | A run of a method with a model and no `model_input`; `seeded_watershed` with `model_input`; both or neither of `pixel_size_um` and `factor` in a `model_input`; `pixel_size_um` on a grid without spacing. Each raises `ValueError` naming the key. | exact (exception types and strings) | H |
| MI6 | `LENGTH_FIELDS` | — | The table no longer holds `stardist.scale` or `cellpose.diameter`. The test that keeps it equal to the registered methods' pixel fields still passes. | exact (strings) | H |

### The route per FOV (PF)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| PF1 | `FOV.segment` with a recipe | `morphology_fov`, `pi_smooth`-like recipe, `seeded_watershed` cell run with seeds from an import | `input.ome.tif` read back equals the captured `SegmentationInput.image`, and the record's `input.sha256` equals its hash. The labels equal those of `segment` called directly on that array. | file: exact (file round trip); hash: exact (hashes); labels: exact (labels) | H |
| PF2 | The composite run in its new form | `morphology_fov` | The run with sources `nuclear` and `amplicon` (`reference_merged`) and `combine: composite` gives the same uint8 input and the same labels as the `76cd058` run with `InputChannel(role="composite", prepare=CompositeConfig())`. Their array hashes are written into the test as constants from the base revision. | exact (hashes of an integer image and of labels) | H |
| PF3 | A19 parity | The A19 configurations rewritten with recipes | The controller's `a19-parity` check: the A19 hash lists equal the base revision's byte for byte. Rewriting a configuration is allowed; changing an expected hash is not. | exact (A19 parity hashes) | H |

### The route after fusion (AF)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| AF1 | The recipe runs on the fused projection | `scene_plane` at `k` 1 and 2; `seeded_runs` with a recipe (`gaussian`, `percentile_normalization`) on the cell run | The `SegmentationInput` passed to `segment` equals `apply_input_recipe` applied by the test to the fused projection files, with the segmentation grid, the footprint `fit_mask` (`ownership_map`'s covered pixels) and `k`. `model_inputs/cell.ome.tif` read back equals the image `segment` received, and has the grid's shape, spacing and origin. | image against the test's call: f32; file against the received image: exact (file round trip); shape: exact (integers); spacing and origin: 1e-9 µm | H, MV |
| AF2 | One recipe, two routes, the same µm values | `scene_plane` per FOV and after fusion at `k` 2, one recipe | The `used` pixel values per stage are the µm values divided by each grid's spacing (`sigma_px` halves after fusion). `model_input.factor_zyx` doubles after fusion. The recipe's `sha256` is equal on both routes. | pixel values and factors: f64; integer lengths: exact (integers); `sha256`: exact (hashes) | H, MV |
| AF3 | The fit region | `fused_quad_notch` (an L-shaped covered region) | `fit_pixels` equals the number of covered pixels. The percentile `low` equals `np.percentile` of the covered pixels with `method="inverted_cdf"`, not of the canvas. | `fit_pixels`: exact (counts); `low`: f32 | H, MV |
| AF4 | Z > 1 is refused, Z = 1 is allowed | `fused_pair` (Z = 3), `scene_plane` (Z = 1) | On `fused_pair`, a recipe with a `channels` step or `projection: sum` is refused by check 7 before anything is written: the folder's files keep their SHA-256 and no file is added. `after_projection` steps with `projection: max` pass and record `applied: false, reason: fused_maximum`. On `scene_plane`, `projection: sum` passes and is recorded `applied: false, reason: single_plane`. | exact (hashes; file lists; strings, booleans) | H, MV |
| AF5 | Check 7 | `route_configs` with recipes | Each refuses before anything is written, naming the key: an unknown recipe name; a source image not in `export.images`; source roles that differ from the combine's; a missing `model_input` for a method with a model. A valid configuration passes. | exact (exception types and strings; hashes of the folder) | H |
| AF6 | Reuse with another recipe | `scene_plane`, `segment: false` after a changed recipe | Refused before anything is written, naming the run and both recipe SHA-256 values. With the same recipe, the reuse is accepted. | exact (hashes; strings) | H, MV |
| AF7 | SI1: no copy of an unchanged input | `scene_plane`, the empty recipe | No `model_inputs/` subfolder and no file in it is written. The run's `input.path` is the fused file, `written` false. | exact (file lists; strings, booleans) | H, MV |
| AF8 | A model input cannot collide with a fused image | `scene_plane`, `export.images` with `model_input_cell=file:<the stain's pattern>` and `cell_projection=file:<the same pattern>`, and two `seeded_watershed` runs seeded by `seeded_runs`' nucleus import, each with a `gaussian` recipe that changes the image: `cell_projection` (source image `cell_projection`) and `cell` (source image `model_input_cell`). Under a prefix layout, the model input of `cell_projection` would be named `model_input_cell_projection.ome.tif`, the fused projection of the image `model_input_cell` | The export writes `model_input_cell_projection.ome.tif` (the fused image) and `model_inputs/cell_projection.ome.tif` and `model_inputs/cell.ome.tif` (the model inputs). These are three files with three SHA-256 values: the fused file's hash equals that recorded in the `fusion` section, and each model input's hash equals that recorded in its run's `input`. A second export run with the same settings removes and rewrites the same three files, and no file is left unlisted. `check_export` passes R2 and T1. With the recipe of `cell_projection` changed to the empty recipe, the export removes `model_inputs/cell_projection.ome.tif`, and with both recipes empty it removes the `model_inputs/` subfolder. | exact (file lists; hashes; statuses) | H, MV |

### Records (RE)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| RE1 | `segmentation.json` version 2 | PF1 | `format_version` 2. Every key of the specification's layout is present. `input.projection`, `channels[*].prepare` and `prepare_channel` are absent. The `docs_pages.design()` reader parses the page's record layout, if the implementation labels it. | exact (integers; strings) | H |
| RE2 | A version 1 record loads | A record written at `76cd058` (a fixture file) | `FOV.load_segmentation` restores the run. The recipe is shown as none. | exact (old-format round trip; labels) | H |
| RE3 | `export.json` | AF1 | The run's `recipe`, `input` and `lengths` keys as specified. T1 and R2 of `check_export` cover `model_inputs/<run>.ome.tif`: a changed byte of it fails T1 and names the file. An unlisted file added under `model_inputs/` fails R2 and names the file. A listed file whose run is not in the section fails T1. | exact (strings; hashes) | H, MV |

### Replaced keys (MG)

| # | Check | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- |
| MG1 | Python | `InputChannel(prepare=…)`, `InputChannel(prepare_channel=…)`, `SegmentationRun(projection=…)`, `StarDistConfig(scale=…)` and `CellposeConfig(diameter=…)` raise `TypeError` (unknown field). | exact (exception types) | H |
| MG2 | YAML | `prepare`, `prepare_channel`, a run's `projection`, and `scale` or `diameter` in `segmentation.runs` and `export.segmentation.runs` raise `ValueError` naming `input_recipe` and the field of the recipe that replaces them. | exact (exception types and strings) | H |
| MG3 | The legacy rule | `rules.stardist_segmentation.parameters` with `rescale` true and false and with `maximum_projection` gives recipes with `factor` 0.5 or 1.0 and `projection: max`, recorded `source: legacy`, and the same labels as at `76cd058` on the existing legacy test. | factor: f64; strings: exact (strings); labels: exact (labels) | H |
| MG4 | `docs/migration.md` | Every row of the specification's migration table is in the guide. | exact (strings) | H |
| MG5 | Schema | The workflow schema's recipe, step, combine and projection keys equal the configs' init fields, and the step names equal the admitted subset of `PREPROCESSING_METHODS`, kept equal by a default-tier test as for the methods. | exact (strings) | H |

### Workflow (WF)

| # | Check | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- |
| WF1 | Planning | A configuration with `segmentation.input_recipes` and no `runs` plans no per-FOV segment task and passes check 7 on the route after fusion. One with `runs` and recipes plans the tasks it planned before. `_reference_inputs` names a run whose `amplicon` source is `reference_merged`. | exact (task lists; strings) | H |
| WF2 | The existing workflow test configurations | Rewritten with recipes, they plan the same jobs and give the same outputs as before. A rewrite that changes a result is reported, not absorbed. | the tolerances the existing tests already state, unchanged | H |

### The preview tool (PV)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| PV1 | Outputs | `morphology_fov` saved to disk (`FOV.save_reference_image` and the morphology round's saved form under `other_rounds/`, the voxel sizes declared) and a copy of it as a second FOV; run `cell` (`seeded_watershed`, seeds from an import run); two FOVs × two recipes | `overlay.png` (2 × 2 panels and the model-scale rows), `sizes.png`, `sizes.csv`, `summary.csv` and `preview.json` exist. Each `sizes.csv` area equals the label's pixel count × the grid's pixel area. The equivalent diameter equals `2·√(area/π)`. Each `summary.csv` percentile equals `np.percentile` of the run's diameters. | files and counts: exact (file lists; counts); areas, diameters and percentiles: f64 | H |
| PV2 | Same as the routes | PV1 | The labels for each FOV and recipe equal those of `FOV.segment` with that recipe. The model input equals that of `FOV.segment`. With `--factor-yx 2`, the model input equals `apply_input_recipe` applied by the test to the one-view fused channels at `k` 2. | labels: exact (labels); model inputs: f32 (or exact for an integer image) | H, MV |
| PV3 | Nothing written outside `--output` | PV1 | The dataset's output tree is byte-identical before and after (a listing with SHA-256). A non-empty `--output`, more than 8 FOVs or 8 recipes, and a source that needs `FOV.run` without its saved file are refused before anything runs. | exact (hashes; exception strings) | H |
| PV4 | The CLI | — | `python -m starfinder segment preview --help` lists the options of the specification. The CLI reference page is generated from the parser. | exact (strings) | H |

### What the default tier covers and what it does not

The rows marked H run in the default tier. The rows marked MV run where the
`multiview-stitcher` extra is installed and skip in the CI environment. `scene_plane_small`
appears only as an existing `slow` fixture and is not needed by a new row. No row loads a
model: `model_stub` replaces the backend. The real model is used only in the first use
below.

## First use on W-870's stage 1 (development check)

This is the implementation issue's first use of the recipe on real data. It shows that the
code works on a real dataset and that the preview can compare recipes. It chooses no
recipe and makes no quality claim.

**Where it runs (amendment B3).**

* FU1–FU3 are a separate evidence issue, under W-889's rule for real-data evidence.
* The coordinating session runs them on UGER, on the implementation branch's commit, before
  merge.
* The implementation batch runs every other check of this page.

* **Data.** W-870's stage 1: the four FOVs tile_28, tile_29, tile_36 and tile_37 of the
  tissue-2D dataset (single-plane tiles of 3072 × 3072 uint8, 0.094635 µm per pixel), the
  PI stain as `reference_stain`, configurations `configs/stage1-per_fov.yaml` and
  `configs/stage1-after_fusion.yaml` of
  `runs/W-870/20261118T0309Z-tissue2d/`. The inputs are read only.
* **Recipes**, a few and no sweep:
  * R0 `pi_max`: projection `max`, `pixel_size_um` 0.37854, W-870's run as a recipe;
  * R1: R0 with `channels: [gaussian, sigma_um 0.3]`;
  * R2: R1 with `after_projection: [clahe, kernel_um 6.0]`;
  * R3: R0 with `pixel_size_um` 0.30.
  The values are chosen to exercise the stages, not as candidates for a default.
* **FU1. Reproduction (exact).** R0 per FOV gives labels whose array SHA-256 equals that
  of W-870's stage-1 nucleus labels on each of the four FOVs
  (`output/sample-dataset/smoke/fovs/<tile>/segmentation/nucleus/labels.tif`, the
  `labels.sha256` of their `segmentation.json`). R0 after fusion gives labels whose array
  SHA-256 equals that of W-870's stage-1 route after fusion
  (`output/sample-dataset/smoke/export/tissue-2D/segmentation_nucleus.ome.tif`, 2928 ×
  2918, 1,055 labels in `results/stage1-look/stats.json`). Array hashes are compared, not
  file hashes, because the files also record software versions. The comparison holds
  because rounding to six significant digits gives exactly the factors 0.25 and 0.5 that
  W-870 passed (MI1). A difference stops the
  first use and is reported, with both hashes, as a defect of the implementation.
  Tolerance: exact (hashes of labels).
* **FU2. Preview.** `starfinder segment preview` with `--run nucleus`, the four FOVs and
  R0 to R3 per FOV, and again with `--factor-yx 2`. The deliverables are the overlay grid,
  the size figure and tables, and `preview.json`. R0's per-FOV diameter percentiles are
  compared with W-870's `stats.json` for the same FOVs, as a check of the size code and not
  of the recipe. `stats.json` gives them rounded to 0.01 µm, so they agree within 0.005 µm
  (half the rounding step) when the size definitions agree. If they do not, the difference
  is explained, for example by border labels. The overlay and size figures are looked at,
  not compared.
* **FU3. Both routes.** R1 through the workflow on stage 1's per-FOV and after-fusion
  configurations with `input_recipe: R1`. The checks are the records (the recipe and its
  `sha256` equal on both routes, exact as hashes; `used.sigma_px` equal to 0.3 / 0.094635
  and 0.3 / 0.18927, about 3.17 and 1.59, and `factor_zyx` 0.25 and 0.5, both at f64) and
  `check_export` passing. `compare_routes` runs and its report is kept, as a
  report only.
* **Records.** Records as the project guidance defines them ("Records of a run with
  steps"), with the resource table per step. Resource bounds come from the evidence
  issue's UGER requests. Stage 1's segment tasks ran with 32,000 MB and the export with
  64,000 MB in W-870.
* **What it is not.** Not a sweep, not a choice of a tissue-2D recipe, and not a
  comparison of recipes by quality. The owner looks at the overlay and the sizes and
  decides what to try next.

## Implemented checks

Filled in by the implementation issue (W-891, 2026-11-20): one row per check of this page,
with the test file and test name. The rows marked MV skip without the
`multiview-stitcher` extra (and AF4, AF6, AF8 and RE3 without `anndata`); the stubs of MI3
and MI4 replace the libraries' calls, so those rows need no extra. RC1 and PF2 hold the base
revision's (12a8f14) hashes as constants, computed there before the change
(`src/python/test/data/segmentation_record_v1/README.md`). FU1 to FU3 are W-892's
(amendment B3).

(segmentation-input-recipe-implemented-checks)=

| Check | File (under `src/python/test/`) | Test |
| --- | --- | --- |
| ST1 | `test_segmentation_recipe_steps.py` | `test_st1_gaussian_on_a_delta` |
| ST2 | `test_segmentation_recipe_steps.py` | `test_st2_gaussian_means_the_same_in_um_on_two_grids` |
| ST3 | `test_segmentation_recipe_steps.py` | `test_st3_radii_in_um`, `test_st3_a_record_keeps_its_form_without_radius_um` |
| ST4 | `test_segmentation_recipe_steps.py` | `test_st4_percentile_normalization_with_a_fit_region` |
| ST5 | `test_segmentation_recipe_steps.py` | `test_st5_clahe`, `test_st5_clahe_fits_its_range_on_the_fit_region` |
| ST6 | `test_segmentation_recipe_steps.py` | `test_st6_the_steps_keep_an_integer_dtype_inside_a_preprocessing_recipe`, `test_st6_gaussian_and_clahe_run_in_fov_run` |
| ST7 | `test_segmentation_recipe_steps.py` | `test_st7_refusals`, `test_st7_a_role_the_channels_do_not_have_and_a_grid_without_spacing` |
| ST8 | `test_segmentation_recipe_steps.py` | `test_st8_the_registry_holds_gaussian_and_clahe` |
| RC1 | `test_segmentation_recipe.py` | `test_rc1_the_empty_recipe_is_the_input_of_the_base_revision` |
| RC2 | `test_segmentation_recipe.py` | `test_rc2_the_order_of_the_stages_and_the_roles` |
| RC3 | `test_segmentation_recipe.py` | `test_rc3_combine` |
| RC4 | `test_segmentation_recipe.py` | `test_rc4_max_sum_and_mean_of_integers` |
| RC5 | `test_segmentation_recipe.py` | `test_rc5_the_exact_flag_at_and_above_two_to_the_24` |
| RC6 | `test_segmentation_recipe.py` | `test_rc6_a_mean_that_is_not_an_integer` |
| RC7 | `test_segmentation_recipe.py` | `test_rc7_the_overflow_check` |
| RC8 | `test_segmentation_recipe.py` | `test_rc8_the_extension_reads_the_channel_before_the_projection` |
| RC9 | `test_segmentation_recipe.py` | `test_rc9_one_recipe_by_name_and_inline_has_one_sha256` |
| MI1 | `test_segmentation_recipe.py` | `test_mi1_the_factor_from_a_pixel_size` |
| MI2 | `test_segmentation_recipe.py` | `test_mi2_a_factor_per_route` |
| MI3a | `test_segmentation_recipe.py` | `test_mi3a_stardist_with_a_2d_model` |
| MI3b | `test_segmentation_recipe.py` | `test_mi3b_stardist_with_a_3d_model` |
| MI4b | `test_segmentation_recipe.py` | `test_mi4b_a_3d_cellpose_run`, `test_mi4b_through_fov_segment` |
| MI4 | `test_segmentation_recipe.py` | `test_mi4_cellpose_receives_the_diameter` |
| MI5 | `test_segmentation_recipe.py` | `test_mi5_refusals` |
| MI6 | `test_segmentation_recipe.py`; `test_sample_segmentation_segment.py` | `test_mi6_the_length_table_has_no_model_scale`; `test_sg2_the_table_of_pixel_fields_equals_the_registered_methods` |
| PF1 | `test_segmentation_recipe.py` | `test_pf1_fov_segment_with_a_recipe` |
| PF2 | `test_segmentation_recipe.py` | `test_pf2_the_composite_run_in_its_new_form` |
| PF3 | the controller's check `a19-parity` | the A19 hash lists of `test_assignment.py` (`HASH_SCRIPT`), compared with the base revision's |
| AF1 | `test_sample_segmentation_recipe.py` | `test_af1_the_recipe_runs_on_the_fused_projection` |
| AF2 | `test_sample_segmentation_recipe.py` | `test_af2_one_recipe_means_the_same_um_on_both_routes` |
| AF3 | `test_sample_segmentation_recipe.py` | `test_af3_the_fit_region_is_the_footprints` |
| AF4 | `test_sample_segmentation_recipe.py` | `test_af4_tiles_of_more_than_one_plane_refuse_before_anything_is_written`, `test_af4_after_projection_steps_pass_and_a_plane_projects_nothing` |
| AF5 | `test_sample_segmentation_recipe.py` | `test_af5_check_7_refuses_naming_the_key` |
| AF6 | `test_sample_segmentation_recipe.py` | `test_af6_the_reuse_refuses_another_recipe` |
| AF7 | `test_sample_segmentation_recipe.py` | `test_af7_the_empty_recipe_writes_no_model_input` |
| AF8 | `test_sample_segmentation_recipe.py` | `test_af8_a_model_input_cannot_collide_with_a_fused_image` |
| RE1 | `test_segmentation_recipe.py` | `test_re1_the_record_of_format_version_2` |
| RE2 | `test_segmentation_recipe.py` | `test_re2_a_version_1_record_still_loads` |
| RE3 | `test_sample_segmentation_recipe.py` | `test_re3_the_export_record_and_check_export_cover_the_model_inputs` |
| MG1 | `test_segmentation_recipe.py` | `test_mg1_the_replaced_python_fields_are_unknown` |
| MG2 | `test_segmentation_recipe_workflow.py`; `test_benchmark_trial.py` | `test_mg2_the_replaced_keys_of_a_segmentation_run_are_refused`, `test_mg2_the_replaced_keys_of_an_export_run_are_refused`, `test_mg2_an_import_keeps_its_own_projection`; `test_learned_segmentation_trial_refuses_the_replaced_keys`, `test_learned_segmentation_trial_reaches_the_model_through_its_recipe` |
| MG3 | `test_segmentation_workflow.py` | `test_rescale_is_the_implicit_recipes_model_factor`, `test_the_projection_that_made_the_input_is_recorded`, `test_the_rule_writes_uint32_labels_equal_to_the_legacy_pins` |
| MG4 | `test_segmentation_recipe_docs_contract.py` | `test_mg4_every_migration_row_is_in_the_guide` |
| MG5 | `test_segmentation_recipe_workflow.py` | `test_mg5_the_schema_keys_equal_the_configs`, `test_mg5_the_schema_validates_the_recipe_layout` |
| WF1 | `test_segmentation_recipe_workflow.py` | `test_wf1_recipes_without_runs_plan_no_segment_task`, `test_wf1_runs_with_recipes_plan_the_tasks_they_planned_before` |
| WF2 | `test_workflow_plan.py`; `test_workflow_morphology_tasks.py` | `test_the_plan_writes_the_expected_manifest_of_each_variant`, `test_the_tasks_respect_their_needs_and_the_contracts_counts`; `test_the_segment_command_writes_uint32_labels_equal_to_the_legacy_pins` |
| PV1 | `test_segmentation_preview.py` | `test_pv1_the_outputs` |
| PV2 | `test_segmentation_preview.py` | `test_pv2_the_labels_and_the_model_input_are_those_of_fov_segment`, `test_pv2_the_fused_view_is_the_recipe_on_one_view_at_k`, `test_pv2_an_export_run_on_the_raw_reference_stain_needs_no_saved_reference_image`, `test_pv2_a_fused_view_applies_and_records_the_label_operations` |
| PV3 | `test_segmentation_preview.py` | `test_pv3_nothing_is_written_outside_the_output`, `test_pv3_a_recipe_file_tries_a_recipe_before_it_is_written`, `test_pv3_a_fused_view_refuses_extension_through_z` |
| PV4 | `test_segmentation_preview.py` | `test_pv4_the_command_lists_the_options_and_runs`, `test_pv4_the_cli_reference_is_generated_from_the_parser` |

W-907 (2026-11-20, from W-891's review) added cases to MG2, PV2 and PV3 without changing a
check or a tolerance. MG2: a benchmark segmentation trial builds its model input with its
`input_recipe` (none is the empty recipe), so StarDist and Cellpose trials reach the model,
and it refuses `scale`, `diameter` and `projection` with the migration row. PV2: a run of
`export.segmentation.runs` whose source is the raw reference stain, without a saved
reference image, takes the grid of the reference stain prepared in memory and records its
source (`fov_grids`); with `--factor-yx` a run's `expand_labels` is applied and recorded, and
at `K` 1 its labels equal those of the preview without `--factor-yx`. PV3: with
`--factor-yx`, a run that extends its labels through Z is refused before anything runs.
