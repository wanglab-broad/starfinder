# Validation design: the segmentation input recipe

% Status: Proposed (W-885, 2026-11-18, written against 76cd058); awaits Jiahao's approval with its specification

This is the validation plan for implementing
[segmentation-input-recipe](../specs/segmentation-input-recipe.md). It covers unit tests
per step and per stage, both routes on the synthetic fixtures, the records, the replaced
keys, the preview tool and a first use on the four FOVs of W-870's stage 1. The first use
is a development check and not a scientific sweep. It makes no segmentation-quality claim,
and no row on this page accepts or rejects a recipe by a quality figure.

The expected values below are computed from the specification and from library calls named
in each row. They do not come from running the new code. Floating-point comparisons use the
tolerances of `docs/contributing.md` ("Expected values and tolerances"), or the tolerance
given in the row. "Exact" means equal integers, strings, hashes or arrays.

## Fixtures

Every routine fixture is at most 32 × 64 × 64 voxels with at most four channels, as the
project guidance bounds them. The fixtures marked `slow` are existing ones.

| Fixture | Construction | Needs |
| --- | --- | --- |
| `delta` | A float32 image of 1 × 63 × 63, value 1 at (0, 31, 31) and 0 elsewhere; spacing (1.0, 0.1, 0.1) µm. | hand-built |
| `blob_pair` | One continuous Gaussian blob of σ₀ = 0.5 µm at the centre of a 6.4 µm square, sampled on two grids: 1 × 64 × 64 at 0.1 µm and 1 × 32 × 32 at 0.2 µm, float32. | hand-built |
| `stack32` | A uint16 stack of 32 × 4 × 4 with every value 65,535; a uint32 stack of 2 × 4 × 4 with planes 8,388,608 and 8,388,609, which sum to 16,777,217 = 2²⁴ + 1, a value float32 cannot hold; and a float32 copy of the uint16 stack. | hand-built |
| `seg_golden` | The fixture of `test_segmentation_golden.py` (`fixture()`: DAPI, amplicon, Flamingo), with its pinned composite and Flamingo digests. | existing |
| `morphology_fov` | The FOV of `test_segmentation_inputs.py`: a sequencing round and a morphology round registered by `FOV.register`, and the reference merged image. | existing |
| `scene_plane_small` | The single-plane sample of [sample-segmentation validation](sample-segmentation.md) ("Fixtures"): 3 × 3 FOVs of 1 × 96 × 96, written by `write_scene_plane`, with the stain as `dapi` and `flamingo`. This is the tissue-2D case (Z = 1). | existing; `multiview-stitcher` extra; `slow` |
| `scene_plane` | The same on `sample-tiny-v1` (FOVs of 1 × 36 × 36), for the routine tier. | existing; `multiview-stitcher` extra |
| `fused_pair` | Two FOVs of 3 × 24 × 24 at spacing (1.0, 0.5, 0.5) µm, fused at `k` 1 and 2 (Z > 1). | existing; `multiview-stitcher` extra |
| `recipes` | The recipes of the specification's example (`pi_max`, `pi_smooth`, `pi_composite`) with lengths chosen for the fixture's spacing, and one recipe per refusal of the rows below. | hand-built |
| `model_stub` | Model folders written by the test (as `length_configs` does), so that `resolve_model` passes. The backend's `run` is replaced by a stub that records the `scale` or `diameter` it receives and returns the `truth_import` labels. No model is loaded. | hand-built |

## Checks

"Needs": **H** hand-built fixtures only; **MV** the `multiview-stitcher` extra; **SD** the
`stardist` extra; **S** `slow`; **L** `learned`. A test that needs an extra skips without
it, naming the extra, so the CI environment (extras `dev`, `local-registration`,
`checkpoint`) passes or skips every row.

### Steps (ST)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| ST1 | `gaussian` on a delta | `delta`, `sigma_um` 0.3 | `used.sigma_px_zyx` = (0, 3, 3). The output equals `skimage.filters.gaussian(delta, sigma=(0, 3, 3))`. The output sums to 1. The second moment of the output about the centre on Y and on X is 9 px², less the truncation at 4σ (8.995 in a scratch computation). | Array exact; sum 1e-6 absolute; moment 0.5 % relative | H |
| ST2 | `gaussian` means the same in µm on two grids | `blob_pair`, `sigma_um` 0.3 | On both grids, the spatial variance of the output in µm² is σ₀² + 0.3² = 0.34, the analytic result for two Gaussians (a scratch computation gave 0.33995 on both grids). | 0.5 % relative on each grid | H |
| ST3 | `white_tophat` and `reconstruction` in µm | `seg_golden` DAPI plane, spacing 0.1 µm; `radius_um` 0.25 and 0.24 | `radius_yx` 3 for 0.25 (`floor(2.5 + ½)`, where Python's `round` would give 2) and 2 for 0.24. The output equals `filter_tophat` / `correct_background` with that `radius_yx`. Unequal Y and X spacing raises `ValueError`. | Exact | H |
| ST4 | `percentile_normalization` with a fit region | A float32 plane with a zero border outside a mask | `fitted.low` and `fitted.high` equal `np.percentile(x[mask], p, method="inverted_cdf")`. The output equals the step's formula with those values, clipped to [0, 1]. Without a mask the values are those over the whole image. | Exact | H |
| ST5 | `clahe` | A float32 ramp plane of 1 × 64 × 64, spacing 0.1 µm, `kernel_um` 1.6, `clip_limit` 0.01 | `used.kernel_px_zyx` = (1, 16, 16). The output equals `equalize_adapthist(clip((x − lo)/(hi − lo), 0, 1), kernel_size=(1, 16, 16), clip_limit=0.01, nbins=256).astype(float32)`, and `fitted.in_range` = [lo, hi] of the fit region. A constant plane gives zeros with `degenerate_range` true. A missing `kernel_um` is refused. | Exact | H |
| ST6 | `gaussian` and `clahe` inside `FOV.run` | A uint16 plane in a `PreprocessingRecipe` | `run_step` accepts both. The dtype is uint16, and the values are `np.rint` of the float result (Gaussian with `preserve_range=True`; CLAHE scaled to 65,535). | Exact | H |
| ST7 | Refusals | `recipes` | Each refusal raises `ValueError` naming the step and the rule: `min_max_normalization`, `histogram_matching`, `scalar_background` and `background_3d` in a recipe; `fit: supplied`; `radius_yx` or `median_radius_px` in a recipe; both `radius_yx` and `radius_um`; a role not among the run's channels; a µm field on a grid without spacing (naming `voxel_size_xy`). | Exact | H |
| ST8 | Registry | — | `PREPROCESSING_METHODS` holds `gaussian` and `clahe` with category `contrast`, scope `per_channel`, dtype policy `preserve`. The preprocessing reference's generated method table lists them. | Exact | H |

### Stages (RC)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| RC1 | The empty recipe is today's input | `morphology_fov`; `seg_golden` | `apply_input_recipe` with `recipe=None` returns the sources stacked as ZYXC in their dtype. Its array SHA-256 equals that of `assemble_input` at `76cd058` for the same run without `prepare` and `projection`. The values are taken from a test at the base revision and written into this test as constants. | Exact | H |
| RC2 | The order of the stages and `roles` | `recipes` on two channels | A step restricted to `nuclear` changes only that channel. A trace of the `run_step` calls shows `channels`, `combine`, `projection`, `after_projection` in that order. Each stage's `input` hash equals the previous stage's `output` hash. | Exact | H |
| RC3 | `combine` | `seg_golden` | `composite` with no channel step gives the pinned `COMPOSITE_DIGESTS`, and `flamingo_enhance` with `median_radius_um` = 1 × spacing gives `FLAMINGO_DIGEST`, both unchanged. After a float32 `gaussian` on both sources, the output equals `composite_nuclei_amplicon` of the two step outputs, is uint8, and each stretched input lies in [0, 1]. Wrong source roles raise `ValueError` naming both role lists. | Exact | H |
| RC4 | `projection` | `stack32` | `max` keeps uint16. `sum` gives float32 2,097,120 (32 × 65,535) with `exact` true. `mean` gives 65,535.0. The uint32 sum of 16,777,217 records `exact` false. The float32 stack records `exact` null. `output_dtype: float64` gives float64. `conversion` is refused. The frame ends in `/projection:mean` for `mean`. | Exact | H |
| RC5 | The stain of `extend_labels_through_z` | `morphology_fov`, a recipe with `gaussian`, `projection: max` and the extension | The stain passed to `extend_labels_through_z` equals the `gaussian` output before the projection (its SHA-256 equals the `channels` stage's `output`). A recipe without a projection, or with two model channels, is refused. | Exact | H |
| RC6 | Recipe identity | `recipes` | One recipe given by name and inline has one `sha256`. Changing any field changes it. | Exact | H |

### Model input (MI)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| MI1 | The factor from a pixel size | Grids of spacing 0.094635 and 0.18927 µm, `pixel_size_um` 0.37854 | `factor_zyx` = (1, 0.25, 0.25) and (1, 0.5, 0.5) after rounding to six significant digits. `factor_zyx_exact` holds the unrounded quotient. | Rounded value exact; unrounded value 1e-12 relative | H |
| MI2 | `factor` per route | `factor` 0.25 at `k` 1 and 2 | 0.25 per FOV; 0.5 after fusion (LS2's rule for the old `scale`). | Exact | H |
| MI3 | StarDist receives the scale | `model_stub` (StarDist) | The stub receives `scale` = (1, 0.25, 0.25) as `predict_instances(scale=…)`. `effective.scale_zyx` records it. A 2D model with a Z factor other than 1 is refused as today. | Exact | H |
| MI4 | Cellpose receives the diameter | `model_stub` (Cellpose) | Factor 0.25 gives `diameter` 120. Factor 1 gives `diameter` 30, which Cellpose 4.2's `eval` turns into `rescale` 1.0, as for `diameter=None` (`rescale = 30. / diameter`). Unequal Y and X factors are refused. | Exact | H |
| MI5 | Refusals | `recipes` | A run of a method with a model and no `model_input`; `seeded_watershed` with `model_input`; both or neither of `pixel_size_um` and `factor` in a `model_input`; `pixel_size_um` on a grid without spacing. Each raises `ValueError` naming the key. | Exact | H |
| MI6 | `LENGTH_FIELDS` | — | The table no longer holds `stardist.scale` or `cellpose.diameter`. The test that keeps it equal to the registered methods' pixel fields still passes. | Exact | H |

### The route per FOV (PF)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| PF1 | `FOV.segment` with a recipe | `morphology_fov`, `pi_smooth`-like recipe, `seeded_watershed` cell run with seeds from an import | `input.ome.tif` holds the array `segment` received (its file read back equals the captured `SegmentationInput.image`, and the record's `input.sha256` equals its hash). The labels equal those of `segment` called directly on that array. | Exact | H |
| PF2 | The composite run in its new form | `morphology_fov` | The run with sources `nuclear` and `amplicon` (`reference_merged`) and `combine: composite` gives the same input array and labels as the `76cd058` run with `InputChannel(role="composite", prepare=CompositeConfig())`, written into the test as constants from the base revision. | Exact | H |
| PF3 | A19 parity | The A19 configurations rewritten with recipes | The controller's `a19-parity` check: the A19 hash lists equal the base revision's byte for byte. Rewriting a configuration is allowed; changing an expected hash is not. | Exact | H |

### The route after fusion (AF)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| AF1 | The recipe runs on the fused projection | `scene_plane` at `k` 1 and 2; `seeded_runs` with a recipe (`gaussian`, `percentile_normalization`) on the cell run | The `SegmentationInput` passed to `segment` equals `apply_input_recipe` applied to the fused projection files with the segmentation grid, the footprint `fit_mask` (`ownership_map`'s covered pixels) and `k`. `model_input_cell.ome.tif` exists, equals that image and has the grid's geometry. | Exact | H, MV |
| AF2 | One recipe, two routes, the same µm values | `scene_plane` per FOV and after fusion at `k` 2, one recipe | The `used` pixel values per stage are the µm values divided by each grid's spacing (`sigma_px` halves after fusion). `model_input.factor_zyx` doubles after fusion. The recipe's `sha256` is equal on both routes. | Exact; sigma 1e-12 relative | H, MV |
| AF3 | The fit region | `fused_quad_notch` (an L-shaped covered region) | `fit_pixels` equals the number of covered pixels. The percentile `low` equals the percentile over the covered pixels and not over the canvas. | Exact | H, MV |
| AF4 | Z > 1 is refused, Z = 1 is allowed | `fused_pair` (Z = 3), `scene_plane` (Z = 1) | On `fused_pair`, a recipe with a `channels` step or `projection: sum` is refused by check 7 before anything is written: the folder's files keep their SHA-256 and no file is added. `after_projection` steps with `projection: max` pass. On `scene_plane`, `projection: sum` passes and is recorded `applied: false, reason: single_plane`. | Exact | H, MV |
| AF5 | Check 7 | `route_configs` with recipes | Each refuses before anything is written, naming the key: an unknown recipe name; a source image not in `export.images`; source roles that differ from the combine's; a missing `model_input` for a method with a model. A valid configuration passes. | Exact | H |
| AF6 | Reuse with another recipe | `scene_plane`, `segment: false` after a changed recipe | Refused before anything is written, naming the run and both recipe SHA-256 values. With the same recipe, the reuse is accepted. | Exact | H, MV |
| AF7 | SI1: no copy of an unchanged input | `scene_plane`, the empty recipe | No `model_input_<run>.ome.tif` is written. The run's `input.path` is the fused file, `written` false. | Exact | H, MV |

### Records (RE)

| # | Check | Fixture | Expected | Tolerance | Needs |
| --- | --- | --- | --- | --- | --- |
| RE1 | `segmentation.json` version 2 | PF1 | `format_version` 2. Every key of the specification's layout is present. `input.projection`, `channels[*].prepare` and `prepare_channel` are absent. The `docs_pages.design()` reader parses the page's record layout, if the implementation labels it. | Exact | H |
| RE2 | A version 1 record loads | A record written at `76cd058` (a fixture file) | `FOV.load_segmentation` restores the run. The recipe is shown as none. | Exact | H |
| RE3 | `export.json` | AF1 | The run's `recipe`, `input` and `lengths` keys as specified. T1 and R2 of `check_export` cover `model_input_<run>.ome.tif`: a changed byte of it fails T1 and names the file. | Exact | H, MV |

### Replaced keys (MG)

| # | Check | Expected | Needs |
| --- | --- | --- | --- |
| MG1 | Python | `InputChannel(prepare=…)`, `InputChannel(prepare_channel=…)`, `SegmentationRun(projection=…)`, `StarDistConfig(scale=…)` and `CellposeConfig(diameter=…)` raise `TypeError` (unknown field). | H |
| MG2 | YAML | `prepare`, `prepare_channel`, a run's `projection`, and `scale` or `diameter` in `segmentation.runs` and `export.segmentation.runs` raise `ValueError` naming `input_recipe` and the field of the recipe that replaces them. | H |
| MG3 | The legacy rule | `rules.stardist_segmentation.parameters` with `rescale` true and false and with `maximum_projection` gives recipes with `factor` 0.5 or 1.0 and `projection: max`, recorded `source: legacy`, and the same labels as at `76cd058` on the existing legacy test. | H |
| MG4 | `docs/migration.md` | Every row of the specification's migration table is in the guide. | H |
| MG5 | Schema | The workflow schema's recipe, step, combine and projection keys equal the configs' init fields, and the step names equal the admitted subset of `PREPROCESSING_METHODS`, kept equal by a default-tier test as for the methods. | H |

### Workflow (WF)

| # | Check | Expected | Needs |
| --- | --- | --- | --- |
| WF1 | Planning | A configuration with `segmentation.input_recipes` and no `runs` plans no per-FOV segment task and passes check 7 on the route after fusion. One with `runs` and recipes plans the tasks it planned before. `_reference_inputs` names a run whose `amplicon` source is `reference_merged`. | H |
| WF2 | The existing workflow test configurations | Rewritten with recipes, they plan the same jobs and give the same outputs as before. A rewrite that changes a result is reported, not absorbed. | H |

### The preview tool (PV)

| # | Check | Fixture | Expected | Needs |
| --- | --- | --- | --- | --- |
| PV1 | Outputs | `morphology_fov` saved to disk (`FOV.save_reference_image` and the morphology round's saved form under `other_rounds/`, the voxel sizes declared) and a copy of it as a second FOV; run `cell` (`seeded_watershed`, seeds from an import run); two FOVs × two recipes | `overlay.png` (2 × 2 panels and the model-scale rows), `sizes.png`, `sizes.csv`, `summary.csv` and `preview.json` exist. Each `sizes.csv` area equals the label's pixel count × the grid's pixel area. The equivalent diameter equals `2·√(area/π)`. Each `summary.csv` percentile equals `np.percentile` of the run's diameters. | H |
| PV2 | Same as the routes | PV1 | The model input and labels for each FOV and recipe equal those of `FOV.segment` with that recipe (array SHA-256). With `--factor-yx 2`, the model input equals `apply_input_recipe` on the one-view fused channels at `k` 2. | H, MV |
| PV3 | Nothing written outside `--output` | PV1 | The dataset's output tree is byte-identical before and after (a listing with SHA-256). A non-empty `--output`, more than 8 FOVs or 8 recipes, and a source that needs `FOV.run` without its saved file are refused before anything runs. | H |
| PV4 | The CLI | — | `python -m starfinder segment preview --help` lists the options of the specification. The CLI reference page is generated from the parser. | H |

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
  because the factors 0.25 and 0.5 are reproduced exactly (MI1). A difference stops the
  first use and is reported, with both hashes, as a defect of the implementation.
* **FU2. Preview.** `starfinder segment preview` with `--run nucleus`, the four FOVs and
  R0 to R3 per FOV, and again with `--factor-yx 2`. The deliverables are the overlay grid,
  the size figure and tables, and `preview.json`. R0's per-FOV diameter percentiles are
  compared with W-870's `stats.json` for the same FOVs, as a check of the size code and not
  of the recipe. They are equal when the size definitions agree. If they do not, the
  difference is explained, for example by border labels.
* **FU3. Both routes.** R1 through the workflow on stage 1's per-FOV and after-fusion
  configurations with `input_recipe: R1`. The checks are the records (the recipe and its
  `sha256` equal on both routes, `used.sigma_px` 3.17 and 1.59, `factor_zyx` 0.25 and
  0.5) and `check_export` passing. `compare_routes` runs and its report is kept, as a
  report only.
* **Records.** Records as the project guidance defines them ("Records of a run with
  steps"), with the resource table per step. Resource bounds come from the implementation
  batch's guidance. Stage 1's segment tasks ran with 32,000 MB and the export with
  64,000 MB in W-870, so a StarDist call on one 3072 × 3072 FOV needs more than the 4 GiB
  worker default, and the guidance has to name the exception.
* **What it is not.** Not a sweep, not a choice of a tissue-2D recipe, and not a
  comparison of recipes by quality. The owner looks at the overlay and the sizes and
  decides what to try next.

## Implemented checks

Filled in by the implementation issue: one row per check of this page, with the test file
and test name.
