# Validation design of the real-appearance cellular scene: target definitions, generator checks and chain checks

Status: Accepted (W-515, 2026-10-16, at 71ef8a1)

This page is the validation design of the appearance model of
{doc}`synthetic-real-appearance` (W-513): nuclei and culture cells sampled from real
images, augmented, placed by a synthetic layout, with the placed masks as truth. It amends
the accepted validation design {doc}`synthetic-cellular-scene-validation` (W-395, amended
by W-482, Accepted at W-396) for the scenes of that appearance and edits no accepted page.
It has five parts:

* the **target definitions** of the appearance presets ("Target definitions");
* the **generator checks**, GA1 to GA24 and GA26 to GA27, and how the cellular checks also
  apply to appearance ("Generator checks");
* the **consequences for the chain checks** CD, CT and CU of the accepted design
  ("Consequences for the chain checks");
* the **execution conditions**, with where the banks live and the measured cost of each
  tier ("Execution conditions");
* the **accepted statements this page changes**, with the old and the proposed wording.

The draft implementation issues are files of the W-514 run directory, not created in
Linear ("Implementation order").

It is a page beside {doc}`synthetic-real-appearance` rather than a section of it, for the
reason the accepted design gives for being a page beside {doc}`synthetic-cellular-scene`.
The gate reviews the appearance and its validation as two decisions. A threshold amended
after acceptance is then recorded here without touching the appearance page. It also keeps
the appearance page's checks beside the accepted checks they amend, in the same form.

**Sources.** Statements about existing behaviour cite the file and function under
`src/python/starfinder/` at `eb333bc`, as {doc}`synthetic-real-appearance` does.
Measurements are cited as follows, each under the artifact root
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/`:

* **W-513**: a file under `results/` of `W-513/20261016T204608Z-f983442a/` and its row, as
  {doc}`synthetic-real-appearance` cites them (for example W-513
  `tables/visible_share_ln.csv` row `ln/Position048/slab16_z17/DAPI-3D/nuclei-3D`);
* **the prototype**: `W-485/20261016T1714Z-operator-prototype/round10/logs/<context>.log.json`
  (keys `proto` and `real_crop`) and `<context>.run.log`, `<context>` being `ln`,
  `tissue2d` or `culture`;
* **W-393**: as {doc}`synthetic-cellular-scene-validation` cites it ("Sources");
* **W-485**: `W-485/20261015T230509Z-d0c681ff/results/tables/` (`gn23_targets.csv`,
  `gn24_resources.csv`).

Where this page needs a number that none of them measured, it says so and gives none.

## The claim boundary

The claim boundary of {doc}`synthetic-cellular-scene-validation` holds as written, with
the following points for the appearance presets.

* **An engineering check of the chain (D1).** The chain checks establish that the stain,
  the segmentation call, the assignment, the assembly and the report are wired to each
  other and to the scene's truth. A threshold is a bound on that wiring, not a measure of a
  method.
* **Development targets, not a D04 calibration (D6).** The appearance presets' density and
  background targets come from W-513's measurements of the visible stain and of the
  background at label rims. The other targets come from W-393's measurements of method
  labels. Both are development targets for a fixture. Passing them does not make the scene
  a calibrated tissue model.
* **Truth masks that come from a method's labels.** In the appearance presets every truth
  nucleus and every culture cell is a mask a method made on a real image: StarDist
  `3D_spleen` in dense 3D, StarDist through the legacy pipeline (`PI_label.tif`) in 2D
  tissue, and Cellpose `cpsam_v2` extended through z in the culture
  ({doc}`synthetic-real-appearance`, "What this scene is not"). A chain check reads them
  as follows.
  1. **The same method on its own labels.** In dense 3D StarDist `3D_spleen` segments a
     scene whose truth is StarDist `3D_spleen` labels. In the culture Cellpose `cpsam_v2`
     segments a scene whose truth is Cellpose `cpsam_v2` labels. Agreement there shows
     that the chain hands the method a scene and carries its output to the report so that
     the method recovers its own kind of labels after augmentation, packing and
     composition. It is a check of the chain and of the method's consistency with
     itself. It is not evidence of the method's accuracy: the masks inherit the method's
     errors, and a method agrees with its own labels also where they are wrong. StarDist's
     labels sit inside the visible nucleus: in the LN FOVs 0.93 to 0.97 of label voxels
     are visible stain, and 0.75 to 0.85 of the visible stain lies outside every label
     (limit L4; W-513 `tables/visible_share_ln.csv`, rows `ln/<FOV>/whole/DAPI-3D/nuclei-3D`,
     columns `label_voxels_visible`, `visible_outside_labels`).
  2. **A related method.** In 2D tissue the masks are StarDist labels of the legacy
     pipeline and the method is StarDist `2D_versatile_fluo` with the seeded watershed.
     They are separate runs of one family, and point 1 applies to them.
  3. **What the scene leaves out.** The scene holds stain only where an entry was placed
     (its mask, its fall-off and their out-of-focus light). A dim nucleus that the source
     method missed is in no entry, so it is not in the scene (limit L1). A high matched
     fraction therefore says nothing about how many real nuclei a method misses. The one
     unlabelled stain the scene does hold is the DAPI of a second nucleus inside a
     culture source cell ({doc}`synthetic-real-appearance`, "Kinds of cells"), which can
     show up as a false positive and is reported (TH-FP).
  4. **A miss** is read first as a finding about the scene or the chain (density,
     out-of-focus light, fall-off length, 8-bit plateau, grid, route), as "What a failure
     means" of {doc}`synthetic-cellular-scene-validation` states, never as a property of
     the method.
  5. **No comparison.** Each context's truth comes from its own method, so no number of a
     chain check compares two methods.
* **No accuracy of any segmentation method on real tissue (D7).** Nothing on this page
  says how well any method segments LN, tissue-2D or culture images. Comparisons belong to
  the evidence project (§2.14).
* **No annotation.** The source labels are method output, and nothing here is or replaces
  an annotation. Annotation stays with W-123.
* **W-93 keeps synthetic truth and calibration.** This design adds checks of one
  generator's truth. What a synthetic truth must be and how a synthetic model is
  calibrated stay with W-93.

## Conventions

The conventions of {doc}`synthetic-cellular-scene-validation` ("Conventions") hold. The
additions are:

* **Tiers and markers.** As {doc}`contributing` defines them ("Test markers"). Every
  generator check carries the subsystem marker `synthetic` and one kind marker:
  `validation` for a row of the table below, `contract` for names, fields, refusals and
  the truth schema, `golden` for pinned outputs. A check that takes 5 s or more on one CPU
  and one thread is `slow`. No new marker is registered.
* **Banks.** The **routine tier and the CI environment read no real data.** Every routine
  check uses a closed-form bank (CF1 of {doc}`synthetic-real-appearance`), built in the test
  session from an accepted context preset, or hand-made arrays written in the test (the
  `bank_cases` and `app_cases` fixtures). A check that needs a bank built from real images
  is `slow`. It reads the bank file through `STARFINDER_APPEARANCE_BANKS` (ST1) and skips,
  naming the variable or the file, where either is missing. Such a check is called a
  **real-bank check** below. The real-bank generator checks GA23, GA24's `slow` part,
  GA26 and GA27 run in the Chapter II evidence project as
  `benchmarks/appearance_evidence.py` since W-536 ("Implemented checks").
* **Test files.** The generator checks go in `test_synthetic_appearance_<topic>.py`. Their
  helpers are `cellular_appearance_oracle.py` (the test oracle) and
  `cellular_appearance_scenes.py` (the fixtures). The `cellular_` prefix puts the helpers
  among the inputs of the controller's `pytest-slow-synthetic` selection
  (`src/python/test/cellular_*.py`), as the accepted helpers are.
* **The test oracle** is written from the formulas of {doc}`synthetic-real-appearance`, the
  truth tables and the bank file alone. It imports nothing from `starfinder.synthetic`
  beyond the table and bank readers. It uses SciPy and scikit-image for the library steps
  the page names (the distance transform, the Gaussian filter, linear interpolation), as the
  generator does. The generator's **point evaluator** (a private function) evaluates `N`,
  `C`, each stain and `δ_k` at supplied sample positions, and the two are compared in GA8
  and GA12.
* **Tolerances.** The three kinds of the accepted design (float rounding of an exact
  identity, derived from a formula, sampling tolerance at 4 SE) apply. Where a row says
  "boundary voxels", these are the voxels of {doc}`synthetic-real-appearance` ("Identities
  a test can check"). At such a voxel the interpolated `D_{e_k}(T_k⁻¹ σ)` lies within 1e-9
  µm of 0, or two objects' distances `δ_k` tie under CT1 within 1e-9 µm. They are counted
  and reported, and the expected count is 0.
* **Thresholds and target ranges are provisional.** Each cites the file and row it comes
  from. The gate W-515 approves each one before an implementation issue asserts it. None
  was measured on an appearance scene, because none exists yet. The first run of GA23 and
  of the chain checks is their first measurement, and a miss is a finding ("What a failure
  means" of {doc}`synthetic-cellular-scene-validation`).
* **Seeds.** Every scene uses the preset seed 42. The one exception is the culture chain
  check, which runs at the seeds 42 and 43, as accepted ("Consequences for the chain
  checks"). No check is repeated with another seed to pass.
* **Objects counted.** The rule V12 of the accepted design holds: a target quantity
  measured on fewer than 100 objects of its kind is reported, not compared.

## Target definitions

**Which TR rows stay and which change.** The accepted rows TR1 to TR12 stay as written for
the closed-form presets. For the appearance presets they are read as the table below
states. The truth labels of an appearance scene are method labels placed at the density of
the visible stain. A quantity that W-393 measured on a real image's method labels, which
miss nuclei, therefore does not compare with the same quantity on the scene's truth labels.
Density and background are measured instead on the stain, the same way on real and
synthetic images (TG2 of {doc}`synthetic-real-appearance`).

**The measures** are those of W-513 `scripts/measure.py` (its docstring) and W-393
`notes.md` §2:

* **visible-stain share**: the share of an image whose stain, smoothed by a Gaussian of one
  voxel, lies above that smoothed image's Otsu threshold (`skimage.filters.threshold_otsu`).
  On a scene it is measured on the stain of the sample grid, which DN1 also renders. The
  culture uses the Z maximum, as W-513 does, and the 2D tissue uses the Z maximum, which
  equals every plane under TZ1;
* **median outside the visible stain**: the median stain outside the visible stain, over
  the plateau;
* **background at label rims**: per label not at the border, the 20th percentile of the
  stain over the unlabelled voxels of the label's box with a margin of 1.5 µm, over the
  plateau. The value of an image is the median over its labels. This is the prototype's
  local source background (`proto.py::extract_sprites`, `b`);
* **plateau**: the median over the labels of each label's median stain;
* the other measures as W-393 defines them, rebuilt in the test helper as GN23 rebuilds
  them (`cellular_measures.py`).

**Projection or volume.** A scene is measured the way the real rows it is compared with
were measured, so each row names its form:

* **Dense 3D** is measured on the volume, with the 3D labels.
* **2D tissue** is measured on the Z maximum, with the projected labels `P`.
* **The culture** uses two forms:
  * a quantity of W-393's culture rows, TR2v and TR8v are measured on the Z maximum, with
    the projected truth labels and the plateau on those projected labels. This follows
    W-393's convention and W-513's `DAPI-2D` and `Flamingo-2D` rows (W-513
    `scripts/measure.py`, docstring);
  * TR8r is measured on the volume, around the truth **cells**, with the plateau on the
    volume. W-513 measured the culture's rim background only that way: its
    `rim_background_culture.csv` rows come from the cell entries' boxes, which span every
    plane, with the plateaus of the labels extended through z (`run_culture`,
    `rim_rows(..., [("DAPI", p_d3), ("Flamingo", p_f3)])`). It is also the level that
    `κ` matches. No projected rim background was measured, and this exception says so.

**Ranges.** W-513 measured three or four images per context, too few for a p5–p95. A
range below is the smallest to the largest per-image value of the named rows. Each range
is **provisional**.

| Row | For the appearance presets | Quantity and measure | Range (dense 3D; 2D tissue; culture), provisional | Source (file and row) | In GA23 |
| --- | --- | --- | --- | --- | --- |
| TR1 | **Changed: reported on the truth labels; compared like for like in the chain checks** | Share of touching nuclei, per image | Truth labels: none (reported). Method labels in step 5 of the chain checks: the accepted 0.44–0.57; 0.76–0.82; 0.33–0.72 | W-393 `geometry.csv` "share of touching nuclei, per image" (the method's labels of the real image). Round 10 gave 0.99 on its truth against 0.56 on the held-out LN crop's labels (`round10/logs/ln.log.json`, `proto` and `real_crop`, `touching_share`) | reported |
| TR2 | **Changed: replaced by TR2v; the label share reported beside it** | Nucleus volume (area) fraction of the truth labels, per image | Truth labels: none (reported). Method labels in step 5 of the chain checks: the accepted 0.07–0.16; 0.30–0.44; 0.17–0.45 | W-393 `geometry.csv` "nucleus volume fraction, per image"; the real label shares beside the visible ones: W-513 `tables/visible_share_<context>.csv`, column `label_share` of the rows of TR2v (the culture `DAPI-2D` rows of Position407 and Position408 carry the 2D-label artefact named under TR8v) | reported |
| TR2v | **New** | Visible-stain share of the scene, per scene | 0.47–0.68; 0.29–0.44; DAPI 0.12–0.14 and Flamingo 0.41–0.53 | W-513 `tables/visible_share_ln.csv`, rows `ln/<FOV>/slab16_z17/DAPI-3D/nuclei-3D` for Position048, 049, 057, 058 (the 16 planes of the preset; 0.583, 0.681, 0.539, 0.472); `tables/visible_share_tissue2d.csv`, rows `tissue2d/<crop>/whole/PI-2D/nuclei-2D` (0.443, 0.361, 0.288); `tables/visible_share_culture.csv`, rows `culture/<FOV>/whole/DAPI-2D/nuclei-2D` (0.118, 0.139, 0.140, 0.134) and `culture/<FOV>/whole/Flamingo-2D/cells-2D` (0.410, 0.455, 0.533, 0.506) for Position397, 398, 407, 408 | asserted where V12 allows (2D tissue); dense 3D reported under the amendment of 2026-10-18 below; culture reported |
| TR3 | **Changed: reported, beside the source entries and the real range** | Share not described by one ellipsoid (IoU < 0.8), on the truth labels and on the entries they come from | None asserted. The placed share, the entries' share and their difference are reported; a bound on the difference is **missing**: neither W-393 nor W-513 measured a placed share, and no derivation bounds it (below). Real range, reported: the accepted 0.064–0.074; 0.027–0.060; 0.12–0.43 | W-393 `geometry.csv` "share of nuclei not described by one ellipsoid (IoU < 0.8), per image". The miss against the real range is limit L2: round 10 gave 0.009 against 0.047 on the held-out LN crop (`proto` and `real_crop`, `non_ellipsoid_share`). It is reported with the squash threshold of rule 4, its control | reported |
| TR4 | **Stays: compared with the real ranges** | Edge width 10–90 % (µm) | 0.20–2.20; 0.31–1.57; 0.33–1.63 | W-393 `stain.csv` "edge width 10-90 %". The edge is made by the fall-off length `λ`, the blur and the out-of-focus light, not by the entry alone, so only the real range describes it. The held-out crop's value is reported beside it (limit L3: round 10 0.82 µm against 1.54 µm in LN, `edge_width_p5_med_p95`) | asserted (dense 3D); 2D tissue and culture on `app_slow_acquired`, reported |
| TR5 | **Stays: compared with the real ranges** | Bright regions per nucleus; size; peak over the nucleus median | as accepted | W-393 `stain.csv` (the accepted rows). The texture is the entries' own; the prefilter of IN2 at a coarser grid smooths it, so the real ranges, not the entries, are the reference | asserted (dense 3D); 2D tissue and culture on `app_slow_acquired`, reported |
| TR6 | **Stays** | Nucleus median over the image's median of nucleus medians | as accepted | W-393 `stain.csv` (the accepted row) | asserted where V12 allows |
| TR7 | **Stays: compared with the real ranges; the source entries reported** | Within-nucleus CV of interior voxels | as accepted (0.25–0.41; 0.20–0.38; 0.11–1.35) | W-393 `stain.csv` "variation within a nucleus (CV of interior voxels)". Round 10 gave 0.317, 0.285 and 0.218 against 0.348, 0.266 and 0.343 on the held-out crops (`within_cv_p5_med_p95`). Reported beside it: the median over placed objects of their CV over their entry's CV | asserted where V12 allows (dense 3D, 2D tissue); culture reported |
| TR8 | **Changed: the per-image background replaced by TR8r and TR8v; the noise row stays** | Background over plateau (W-393's, more than 2 µm from every label); noise over plateau per 128-px block | Background: none (reported). Noise: 0–0.063; 0–0.033; 0–0.24 | W-393 `stain.csv` (the accepted rows). W-393's background contains the unlabelled nuclei (0.34 in Position048; {doc}`synthetic-real-appearance`, "Composition") | background reported; noise asserted where V12 allows |
| TR8r | **New** | Background at label rims, median over the truth labels (the truth cells in the culture), over the plateau; dense 3D on the volume, 2D tissue on the Z maximum, the culture on the volume (the exception above) | 0.33–0.40; 0.125–0.130; 0 | W-513 `tables/rim_background_ln.csv` rows `ln/<FOV>/DAPI` (0.400, 0.333, 0.385, 0.360; 3D labels of the whole FOVs), `tables/rim_background_tissue2d.csv` rows `tissue2d/<crop>/PI` (0.125, 0.125, 0.130), `tables/rim_background_culture.csv` rows `culture/<FOV>/DAPI` and `culture/<FOV>/Flamingo` for Position397, 398, 407, 408 (0 in all eight; volume, cell boxes, plateaus on the extended labels), column `local_bg_over_plateau_median` | reported: dense 3D and 2D tissue under the amendment of 2026-10-18 below; culture reported |
| TR8v | **New** | Median stain outside the visible stain, over the plateau, per scene; dense 3D on the volume, 2D tissue and the culture on the Z maximum (the same form as TR2v) | 0–0.54; 0.022–0.047; DAPI 0.036–0.038 and Flamingo 0.022–0.074 | W-513 `tables/visible_share_ln.csv` rows `ln/<FOV>/slab16_z17/DAPI-3D/nuclei-3D` (0.486, 0.111, 0.538, 0; over the whole FOV's plateau); `tables/visible_share_tissue2d.csv` rows `tissue2d/<crop>/whole/PI-2D/nuclei-2D` (0.047, 0.038, 0.022); `tables/visible_share_culture.csv` rows `culture/Position397/whole/DAPI-2D/nuclei-2D` and `culture/Position398/whole/DAPI-2D/nuclei-2D` (0.038, 0.036), and rows `culture/<FOV>/whole/Flamingo-2D/cells-2D` for Position397, 398, 407, 408 (0.039, 0.022, 0.074, 0.0625), column `not_visible_over_plateau`. The DAPI rows of Position407 and Position408 (0.308, 0.158) are left out: their plateaus of 13 and 19 come from 2D nucleus labels covering 0.47 and 0.33 of the projection, against a visible DAPI of 0.14 and 0.13 (the same rows, `label_share`, `visible_share`, `plateau`), the artefact {doc}`synthetic-real-appearance` reports (SRC2) | reported: the LN value is 0 where a tile holds empty regions, so the range is wide; a block-wise form is **not measured** by W-513 |
| TR9 | **Stays** | 8-bit plateau (grey levels), measured on the scene as W-393 measured it (the culture on the Z maximum with the projected nucleus labels) | 35–50; 64–93; 13–83 | W-393 `notes.md` §9. O8 sets `g` at 45, 86 and 46 (W-513 `tables/bank_selection_<context>.csv`, `plateau`). In dense 3D and 2D tissue `g` is the same quantity as the measured plateau. In the culture `g` is the plateau on the volume with the extended labels, whereas TR9 is measured on the projection, so the two differ | asserted where V12 allows |
| TR10 | **Stays** | Reads per cell; nuclear fraction of reads per cell | as accepted | W-393 `molecules.csv` (the accepted rows); the molecules are the accepted ones | asserted where V12 allows |
| TR11 | **Changed: reported on the truth; compared like for like in the chain check** | Cell volume (dense 3D) | Truth: none (reported). Method labels in step 5 of CD (the method's nuclei expanded by 0.7776 µm): the accepted 90–185 µm³ | W-393 `geometry.csv` "cell volume" (StarDist labels of the real image, expanded). The truth's territories grow by CT1 among nuclei placed at the visible stain's density, so they are smaller than expansions of sparser labels | reported |
| TR12 | **Stays** | Composite signal share per cell (2D tissue) | 0.52–0.96 | W-393 `molecules.csv` (the accepted row) | asserted where V12 allows |

**Amendment of 2026-10-18 (W-520; operator decision, confirmed by Jiahao on 2026-10-18).** The
first run of GA23 (W-520, seed 42) measured three rows outside their ranges:

* TR2v of `appearance-dense3d-slow-v1`: 0.385 against 0.47–0.68;
* TR8r of `appearance-dense3d-slow-v1`: 0.238 against 0.33–0.40;
* TR8r of `appearance-tissue2d-slow-v1`: 0.034 against 0.125–0.130.

The causes, from W-520's `results/tables/ga23_diagnostics.csv`, are the following. DN1
places with a fixed threshold, 0.567 of the plateau in dense 3D, and reached 0.542 against
its target 0.539. TR2v measures the same scene with its own Otsu threshold, which is 0.826
of the plateau there, so its share is 0.385. In 2D tissue the two rules agree (0.348 and
0.340). `κ` matches the masks' inner rims to the placed entries' mean local background,
0.379 of the plateau in dense 3D and 0.120 in 2D tissue, close to the real ranges. TR8r
instead takes the 20th percentile of the free voxels within 1.5 µm of each label, and the
scene is darker there. The real rim level includes unlabelled stain beside each label, which
no entry holds (limits L1 and L4). These three rows are **reported beside their range, not
asserted**, until a scene control reaches them or Jiahao decides otherwise. TR2v of dense 3D
stays reported until DN1 and TR2v use one threshold rule. No range, source, tolerance or
seed changes, and every other row is read as written.

**The choice for TR3, TR4, TR5 and TR7.** Shape and texture now come from bank entries,
so each of these rows can be read in two ways. It can compare the placed objects with
their source entries, which shows that augmentation and rendering keep what the entry
carries. It can also compare them with the real ranges, which shows how the scene stands
against real images. The rows above choose one comparison for each, or report both:

* **TR3 is reported, against both.** Rotation about Z, flips and an isotropic scale keep a
  continuous shape, so the placed share should follow the entries' share. Rendering on the
  sample grid changes this, because each mask is evaluated from `D_e` at rotated and scaled
  positions. That changes the voxelized mask and its moment ellipsoid by an amount that
  depends on the mask's size in voxels and on the IoU's distance from 0.8. No measurement
  bounds this change and no derivation from the augmentation rules does, so the difference
  is reported and its bound is named as missing. The implementation's first run of GA23
  measures it, and a bound can be proposed at a later gate. The real range is missed by the
  selection (limit L2), not by placement, and is reported with its control.
* **TR4 and TR5 are compared with the real ranges.** The edge is made by composition
  (`λ`, the blur, `κ`), and the bright regions are smoothed by the prefilter at the
  preset's grid. An entry on its own grid is not their reference.
* **TR7 is compared with the real range.** This is the quantity the W-485 amendment was
  about, and the appearance presets are its control (below). The ratio to the entries is
  reported, because interpolation lowers a CV a little and no measurement bounds by how
  much.

**The W-485 amendment (TR7 of dense 3D).** AM1 of {doc}`synthetic-real-appearance` is
adopted. The amendment stands for the closed-form presets: GN23 keeps reporting TR7 of
`context-dense3d-slow-v1` and does not assert it. It does not apply to the appearance
presets. GA23 asserts TR7 of `appearance-dense3d-slow-v1` against 0.25–0.41 (about 738
nuclei, so V12 allows it), and of `appearance-tissue2d-slow-v1` against 0.20–0.38. The
appearance presets are the "texture control that reaches it" which the amendment names as
a candidate for a later issue. Round 10 reached 0.317 and 0.285 (`within_cv_p5_med_p95`).

**What the targets cannot show.** The visible-stain share matches density as the eye and
the methods see it. It does not say whether the nuclei behind it are separable. That is
what step 5 of the chain checks reports, on the method's own labels.

## Fixtures

The accepted fixtures stay. These are added:

| Fixture | Construction | Seeds | Needs |
| --- | --- | --- | --- |
| `bank_cases` | Hand-made label and stain arrays written in the test, with a known outcome for each selection rule: a label touching the Y or X border, a 3D label touching a Z end, labels below the 5th and above the 95th size percentile, a culture cell holding no nucleus label of 50 voxels, a culture cell holding two nucleus labels of 80 and 60 voxels with distinct DAPI levels, a label with rim share 0.5 against a neighbour, and one label that passes every rule. Each image has a known plateau and local background, and a background region with a known share of foreground for the patch rule. | none | bank build step (A1) |
| `fall_off_cases` | Hand-made arrays for OUT1, each with a known answer. (a) **Cap:** one label alone in its box with a constant excess `r` inside and a constant excess `2r` around it, so that `G_0.3 * X ≥ r` outside, the cap binds everywhere and the weight `v` is 1: the field before the outside blur is `r · max(0, 1 − d / λ)`. (b) **Neighbour:** case (a) with a second label sharing a face with the first, its stain set to `10r` in one copy and `20r` in another. Every value is at least `r`, so the cap binds in both copies and step 1 cannot see the neighbour's level. (c) **Rising profile:** case (b) with a free-side surround that rises with distance between 0.2 and 0.5 µm. | none | bank build step (A1) |
| `cf_banks` | `closed_form_bank(preset, n)` for each routine context preset `context-dense3d-v1`, `context-tissue2d-v1` and `context-culture-v1`, noise off, in memory (CF1), with `n` = 4 source scenes: the preset rendered at the seeds 42, 43, 44 and 45, each scene's sample grid one source image, the seed 45 image held out. ({doc}`synthetic-real-appearance` leaves the meaning of `n` open; A1 fixes it.) | 42 to 45 | A1 |
| `cf_bank_files` | The same four scenes per context written by `save_context_scene` under the test's `tmp_path`, configured as file sources of `build_appearance_bank` (their stain files and truth label files), built twice into two files. A copy with one source stain multiplied by 2, so that its plateau lies outside BR4's factor 1.5, and a copy with one source's labels rolled by half a FOV, so that its plateau and its median over the visible stain disagree. | 42 to 45 | A1 |
| `app_tiny` | `AppearanceSceneConfig` on `sample-tiny-v1` (2 × 2 FOVs of 5 × 36 × 36, fractional J1 jitter in Z, Y and X), contact 1.0, DN3 with the count that fills the domain at the first refusal run, in three copies: dense 3D (the dense-3D `cf_banks` bank), culture (the culture bank: cell entries with DAPI and Flamingo, CU1), and 2D tissue (the 2D tissue bank, five planes, TZ1). Output cast off and float64; `app_tiny_uint8` is the same with the 8-bit cast. | 42 | A1 to A3 |
| `app_plane` | Each `app_tiny` copy on the single-plane configuration `scene_plane` (FOVs of 1 × 36 × 36, jitter bound (0, 1.5, 1.5) voxels). | 42 | A1 to A3 |
| `app_morph` | The culture `app_tiny` with a morphology round `organelle` (`DAPI`, `Flamingo`): once with a supplied whole-voxel translation (0, 2, −3) voxels for every FOV and once with MT1's drawn fractional translation; a copy whose supplied jitter differs between two FOVs by whole voxels. | 42 | A1 to A3 |
| `app_cases` | Placement tables written by hand on a two-entry bank built from hand-made arrays: the two unit spheres 1.9 µm apart of {doc}`synthetic-real-appearance` ("One ownership rule for nuclei and cells"); a culture cell whose nucleus lies in an earlier cell's overlap; two objects tying under CT1; each at a FOV at a fractional offset in Z, Y and X and at `Z=1`. | none | point evaluator (A2) |
| `app_crowded` | `app_tiny` (dense 3D) with a DN3 count larger than the domain holds, at `floor` 0.9 (raises) and 0.1 (stops and records); with `max_refusals` 50; and with `max_proposals` 30. | 42 | A2 |
| `app_dn1` | The dense-3D and culture `app_tiny` copies with DN1 instead of DN3: the target and the threshold from the closed-form bank's source images (the medians over them on the region of the scene's Z extent), a density check after every placement (`m` = 1). A copy with `max_proposals` too small to reach the target, at `floor` 0.1 (stops and records) and 0.99 (raises). | 42 | A2, A3 |
| `app_coarse` | The dense-3D `app_tiny` on a grid twice as coarse in Y and X as the bank's source spacing, so that IN2's prefilter is active on every object. | 42 | A2, A3 |
| `app_cu_offset` | The culture `app_tiny` configuration (DN3, 2 × 2 FOVs, 7 planes) on a bank built by the build step from hand-made culture source arrays of 7 planes. Every cell spans all planes, so its box centre lies on plane 3, while its nucleus fills planes 3 to 5, with its centroid on plane 4, one plane (0.346 µm) above the box centre. Every source nucleus centroid lies on plane 4, so the median of any subset of them is plane 4. A check of the box centre `centre_z` against the source nucleus centroid plus the shift would be off by 0.346 µm and fails on this fixture. | none | A1, A2 |
| `app_bg1` | The dense-3D and culture `app_tiny` copies with BG1, the background assembled from the closed-form bank's patches, instead of BG2. | 42 | A1 to A3 |
| `app_streams` | `app_tiny` (dense 3D) and copies: with the gain range of AU1 changed; with type 2's abundance row changed; with type 2's `size_scale` 1.5; with the culture bank's Flamingo channel dropped (one stain less). | 42 | A1 to A3 |
| `app_routine` | The three routine appearance presets `appearance-dense3d-v1`, `appearance-tissue2d-v1`, `appearance-culture-v1` (closed-form banks, DN3) as generated. | 42 | A4 |
| `real_banks` | The three bank files `ln-sample1-v1`, `tissue2d-v1` and `culture3d-v1` with their build records and build configurations, read through `STARFINDER_APPEARANCE_BANKS` on the lab machine. | none | A4; real-bank checks only |
| `app_slow` | The three `slow` appearance presets `appearance-dense3d-slow-v1`, `appearance-tissue2d-slow-v1` (0.18927 µm) and `appearance-culture-slow-v1` (0.388796 µm) on `real_banks`, as generated: about 738, 337 and 32 objects ({doc}`synthetic-real-appearance`, "Presets"; estimates). | 42 | A4; `slow`; real-bank; RG1 |
| `app_slow_acquired` | `appearance-tissue2d-slow-v1` and `appearance-culture-slow-v1` with `spacing_zyx` replaced by the acquired (0.346255, 0.094635, 0.094635) and (0.346255, 0.194398, 0.194398) µm, every other value as the preset, as `context_slow_acquired` is built. Not presets. | 42 | A4; `slow`; real-bank; RG1 |

## Generator checks

**How the cellular checks apply.** The rows GN1 to GN24 stay as written for the closed-form scene and its
six presets, which are unchanged (E1, PR1). For a scene of real appearance each is read as
below. The rows on labels, packing, territories, types and molecules depend on the layout
option. Under PL2, which {doc}`synthetic-real-appearance` recommends, nothing is packed
by K1 and no lobe is drawn, so GN5 and GN7 have nothing to check. Under PL1 they would
apply to the fitted sites.

| GN row | For a scene of real appearance | Replaced or applied by |
| --- | --- | --- |
| GN1 | **Stays**, extended: the six context presets are pinned too | GA1 |
| GN2 | **Changed**: the appearance names and presets added; the accepted fields unchanged | GA2 |
| GN3 | **Changed**: the appearance refusals added | GA3 |
| GN4 | **Changed**: labels recomputed from `placements.csv` and the bank (identity 2) | GA8 |
| GN5 | **Retired** for appearance scenes (no lobes and no harmonics; the shape columns are null) | GA21 (the nulls); GA6 (the entry mask) |
| GN6 | **Changed**: ownership by placement order and anchors (identity 3) | GA9 |
| GN7 | **Retired** for appearance scenes (no K1 packing): replaced by the placement and stopping rules of PL2 | GA10 |
| GN8 | **Stays**, with `C` and `N` of the appearance (identity 8) | GA17 |
| GN9 | **Stays**, as GN8 | GA17 |
| GN10 | **Stays**, as GN8 | GA17 |
| GN11 | **Stays**, as GN8 | GA17 |
| GN12 | **Stays**, as GN8 | GA17 |
| GN13 | **Retired** for appearance scenes (no closed-form stain): replaced by the composition identity (identity 4) | GA12, GA13 |
| GN14 | **Retired** (the boundary source is not used; O2) | none |
| GN15 | **Retired** (no bright regions are drawn; `bright_regions.csv` is empty) | GA21 (the empty table); TR5 in GA23 |
| GN16 | **Retired** (no field; `unevenness.csv` is empty) | GA21 (the empty table) |
| GN17 | **Changed**: the cast by `g` instead of `k` (identity 5) | GA15 |
| GN18 | **Changed**: both rounds from the entry's channels (identity 6, MO1) | GA16 |
| GN19 | **Changed**: TZ1 columns; reported for the culture (identity 9) | GA18 |
| GN20 | **Stays** on `app_plane` | GA19 |
| GN21 | **Changed**: the `context.appearance.` components | GA20 |
| GN22 | **Changed**: the AT2 layout | GA21 |
| GN23 | **Changed**: the target definitions above, on real banks | GA23 |
| GN24 | **Stays**, extended to the appearance presets and the bank build | GA24 |

Routine tier unless the row says otherwise. "Source" names the section of
{doc}`synthetic-real-appearance` that gives the expected result. **No routine row reads
real data**: every routine fixture is built from a closed-form bank or from hand-made
arrays.

| # | Check | Fixture (seed) | What is computed | Expected result (source) | Tolerance | Tier and markers |
| --- | --- | --- | --- | --- | --- | --- |
| GA1 | The accepted presets unchanged (identity 1) | `sample_presets`, the six context presets (42) | (a) GN1 (a) and (b) unchanged; (b) the group digests of everything `save_context_scene` writes for each context preset, as the golden test of the sample presets computes them, against pins taken at the base revision of the implementation batch before its first change. | (a) as GN1; (b) equal to the pins ("Opt-in": the six presets keep their behaviour; E1). | Exact (SHA-256 of content); the floating-point values of the CSV files and JSON records to 6 significant digits, magnitudes below 1e-9 as 0 (amendment of 2026-10-19). | (a) as GN1; (b) routine for the three routine presets, `slow` for the three `slow` presets; `synthetic`, `golden`. |
| GA2 | Names, fields and the preset registry | none | The fields of `ContextSceneConfig` and of every accepted configuration, and the keys `_payload` writes, against the base revision; the new names (`generate_appearance_scene`, `AppearanceScene`, `AppearanceSceneConfig`, `BankReference`, `AugmentationConfig`, `LayoutConfig`, `CompositionConfig`, `build_appearance_bank`, `closed_form_bank`, `load_appearance_bank`, `save_appearance_scene`, `appearance_scene_preset`, `APPEARANCE_PRESETS`, `APPEARANCE_PRESET_VERSION`, `APPEARANCE_TRUTH_VERSION`) exported and on API pages; six presets, `appearance-presets-v1`, `"1"`; each preset's grid, spacing, jitter, overlap, seed, cell types and molecules against the accepted preset of its context and tier. | Accepted fields and payload keys unchanged ("Opt-in"); the names of "Opt-in" exported (`docs/check_reference.py`); PR1's six presets with the accepted values of their context and tier ("Presets"). | Exact; preset floats within 1e-9. | routine; `synthetic`, `contract`. |
| GA3 | Refusals | `cf_bank_files` and its two copies; none | `ValueError` and its message for: a bank file whose SHA-256 differs from the reference; a missing `entry_id`; an augmentation range outside its allowed interval (a scale or gain at or below 0); `floor` outside `(0, 1]`; a source refused by BR4 (both copies); a reached density below `floor` (`app_crowded`). | Each raises `ValueError` naming the field, the file or the source and its two numbers ("The appearance bank", BR4; "When placement stops"). | Exact (the type; the named values in the message). | routine; `synthetic`, `contract`. |
| GA4 | The bank build is deterministic and recorded | `cf_banks`, `cf_bank_files` (42 to 45) | Each closed-form bank built twice in memory; each file bank built twice into two files; the digest of the arrays; the SHA-256 of each file against its record; the record's inputs (preset and seeds for CF1; path, size and SHA-256 of each source file), rules, counts per rule, plateaus and digest; `load_appearance_bank` with the recorded SHA-256 and with one byte changed. | Two builds give equal arrays and an equal digest. The recorded SHA-256 equals the file's. The record lists every input with its SHA-256 (or the preset and seeds) and the counts of "Selection of entries". Loading with a changed byte raises (ST1, CF1). | Arrays and digests exact; the file bytes are not compared, because a zip archive carries a time (ST1). | routine; `synthetic`, `validation`. |
| GA5 | The selection rules hold | `bank_cases` (none); `cf_banks` (42 to 45) | For every label of every source: which rule leaves it out, recomputed by the oracle (border, size p5–p95 of the labels passing rule 1, a culture nucleus of at least 50 voxels, rim share at most 0.25 or 0.35 for culture cells); the kept set against the bank's entries; the counts against the record; the held-out image's labels against the bank's sources. | On `bank_cases` each hand-made label meets its known outcome. On `cf_banks` the kept set and the counts equal the oracle's, and no entry comes from the held-out image (SEL1, SRC2's held-out rule). | Exact (sets and counts); rim shares and size percentiles recomputed with the same arithmetic, labels within 1e-9 of a rule's limit reported, expected 0. | routine; `synthetic`, `validation`. |
| GA6 | What an entry carries | `cf_banks`, `bank_cases`, `fall_off_cases` | Per entry: the stored signed distance against `EDT(not M) − EDT(M)` with the source spacing; inside the mask the stored excess against `max(I − b, 0)` relative to the plateau; the local background `b` against the 20th percentile over the unlabelled voxels of the box; the rim level `r` against the median excess over the inner rim. Outside the mask, OUT1's four steps recomputed by the oracle from the source arrays with the same library calls: (1) `min(G_0.3 * X, r · max(0, 1 − d / λ))`; (2) the weight `v`, the indicator `d ≤ d_other` smoothed by a Gaussian of 0.6 µm and set to 0 on other labels; (3) `own`, the per-distance median of the capped values where `v > 0.5`, in bins of 0.1 µm, made non-increasing; (4) `kept · v + own · (1 − v)` blurred by a Gaussian of 0.25 µm. Also: the excess beyond `λ` + 4 × 0.25 µm; on `fall_off_cases` (a) the stored field outside against the 0.25-µm blur of `r · max(0, 1 − d / λ)`, (b) the two copies with the neighbour at `10r` and `20r` against each other, (c) the stored field against the oracle with and without the non-increasing step. For a culture entry: its nucleus `D_e` against the largest nucleus label inside the cell, and its DAPI and Flamingo channels. For each patch: its share of foreground. The stored plateau, rim share and volume. | `D_e ≤ 0` exactly on the mask ("Terms"). The excess inside equals the source excess over the plateau, with no blur (BR1; "What an entry carries"). Outside, the stored field equals OUT1's four steps. Case (a) equals the capped linear fall-off blurred. Case (b)'s two copies are equal, so the neighbour's stain never enters the entry (step 2 sets `v` to 0 there and step 3 fills with `own`). Case (c) equals the oracle with the non-increasing step and differs from it without that step. Beyond `λ` + 1.0 µm, the reach of the 0.25-µm blur truncated at four deviations, the excess is 0. A culture entry's nucleus is its largest nucleus label (the 80-voxel one in `bank_cases`), and the other nucleus's DAPI stays in the excess unlabelled (EN1; "Kinds of cells"). Every patch holds at most 10 % foreground (within 1 µm of a label or above 0.7 of the plateau; BG1's patch rule). The stored values equal the recomputation. | `D_e` within 1e-6 µm (float32 storage; the same library call); the excess inside within one float32 rounding of the value (relative 2⁻²³); outside, within 1e-6 of `r` (float32 storage of the same library calls); **Amendment of 2026-10-17 (W-516; operator decision, confirmed by Jiahao on 2026-10-18):** outside, within `max(1e-6 · r, 1e-9)` of the value over the plateau, because an entry whose rim level `r` is 0 still carries the blurred inside excess outside its mask and float32 storage cannot meet a bound of 0 (measured: `context-culture-v1` seed 43 entry 8, DAPI, error 8.2e-10); 1e-9 is the project's default absolute tolerance; case (b) exact; case (c) differs by more than 1e-3 of `r` without the step; zeros exact; counts exact. | routine; `synthetic`, `validation`. |
| GA7 | Transforms recorded and in range | `app_tiny`, `app_streams` (42); `app_cu_offset` (42) | For every row of `placements.csv`: `entry_id` in the bank; `theta`, the flips, `scale_z`, `scale_yx` and `gain` against AU1 and against the values redrawn from `context.appearance.augment` at entity `object-<k>-proposal-<j>`; with BS2, the entries placed before the bank is exhausted, and the entry of each placed proposal against a replay of BS2's queue (the permutation from `context.appearance.entry`; at each proposal `j` up to the last recorded, the head of the queue is drawn, and moved to the end when no placed object records proposal `j`); in the culture, for each placed object, its nucleus centroid in Z in the sample frame, `centre_z + (n̄_e,z − x̄_e,z)`, with `n̄_e` the centroid of the entry's nucleus mask (its nucleus `D_e ≤ 0`) and `x̄_e` the entry box's centre. This is the Z row of `T_k(n̄_e) = c_k + A_k (n̄_e − x̄_e)` ("Augmentation", the transform), since `A_k` is the identity in Z for a culture entry and `centre_z` is `c_k`, the transformed box centre. It is compared with the same nucleus's centroid z in its source image plus the bank's one shift `Δ`. Also: `Δ` against `z_mid` less the median nucleus centroid z of the bank's source image (for a bank of several images, A2 states which median, since the page names one image), and the median over the placed objects of their transformed nucleus centroids in Z against `z_mid`. | `theta` in `[0, 2π)`; `flip_y` ±1; `flip_z` ±1 in dense 3D and 1 elsewhere; scales and gain in `[0.9, 1.1]`; `scale_z = scale_yx` in dense 3D and 1 otherwise; equal to the redrawn values; no entry placed twice before the bank is exhausted, and each placed entry the one the replayed queue gives ("Augmentation", AU1; BS2); in the culture every placed nucleus centroid is its source centroid plus one constant `Δ`, and `Δ` moves the source image's median nucleus centroid onto the middle plane ("Augmentation", the culture's Z). The median of the placed centroids lies on `z_mid` on `app_cu_offset`, whose source centroids are all equal. On `app_tiny` that median is reported, because the placed objects are a subset of the source nuclei and the page states the rule for the source image's median. | Exact for the draws, the entries and the flips; the culture's Z within 1e-9 µm, or within half a source Z step (0.173 µm) if A2 rounds the shift to whole source planes as the prototype's 6-plane shift does (A2 states which). | routine; `synthetic`, `validation`. |
| GA8 | Labels against the placements (identity 2) | `app_tiny` (each copy), `app_plane`, `app_morph`, `app_cases`, `app_routine` (42) | `N` and `C` recomputed by the oracle from `placements.csv` and the bank at every voxel centre of every FOV, of the sample grid and of the morphology round's grid (`true_f + τ_f + q ⊙ s`), and by the point evaluator at the same positions; per grid the voxels that differ; per object the number of connected pieces of its label (26-connected in 3D, 8-connected in a plane), reported (limit L6). | Labels equal the recomputation at any fractional offset, at `Z=1` and on the morphology round's grid, except at boundary voxels ("Truth"); the evaluator equals the oracle. Each placed object's label is its entry's mask evaluated through its transform. | Exact; boundary voxels counted and reported, expected 0. | routine on the tiny fixtures; `slow` on `app_routine` if measured so; `synthetic`, `validation`. |
| GA9 | Ownership and anchors (identity 3) | `app_cases`, `app_tiny`, `app_plane`, `app_routine` (42) | On the sample grid each object's mask voxels inside earlier objects' masks, over its volume; each recorded anchor against `T_k` of its entry's voxel of smallest `D_e`; each anchor's label; `C` on every voxel with `N = n > 0`; the review's case of the two unit spheres at the point 0.98 µm from the first centre (each sphere centred in its entry box, so its centre is its `c_k`). | Overlap at most 1 % of each object's volume (PL2); every anchor is its entry's deepest point transformed (within 1e-9 µm) and its label is its own object; `N = n > 0` implies `C = cell(n)` at every voxel, overlaps included; at the review's point `N = C = 1` ("One ownership rule for nuclei and cells"). | Exact (labels, IDs, voxel counts); the 1 % compared on voxel counts exactly. | routine; `synthetic`, `validation`. |
| GA10 | Placement and its stop | `app_tiny`, `app_crowded`, `app_dn1`, `app_routine` (42) | The reached count or visible share, the placed objects and the proposals that `appearance.json` records, against a recount. Each placed object's share of its mask in `Ω`. In the culture, each placed nucleus against earlier nuclei. Each placed proposal replayed from `context.appearance.placement`: a contact proposal's centre against `c_i + f (r_i + r_k) u` with the redrawn partner and direction, at the first `f` of the ladder 0.5, 0.54, … below 1.4 whose overlap with the earlier masks is at most 1 %, every smaller `f` recomputed above 1 %; a uniform proposal's centre against the redrawn point in `Ω` widened by 0.3 of the box. Every centre compared is `c_k`, the transformed box centre, as the partner's `c_i` is. In the culture only Y and X are compared: the Z of a culture object is its entry's kept Z plus `Δ`, checked by GA7, as the prototype's `fit` overrides the proposal's Z (`round10/scripts/proto.py::place_all`). With DN3, the placed count. With DN1 (`app_dn1`): the recorded target and threshold against the medians over the bank's source images; the visible share of the composed stain, recomputed by the oracle after the last placement and after the one before it. The outcomes of `app_crowded` at each floor, at `max_refusals` and at `max_proposals`, and of the short copy of `app_dn1` at each floor. | The record equals the recount. Every object holds at least 35 % of its mask in `Ω`, and no culture nucleus touches an earlier one. Each centre is the one PL2's rule gives. DN3 places its count. DN1 stops at the first check whose share reaches the target: the share after the last placement is at least the target, and after the one before it below the target. Placement ends at the first of the target, `max_proposals` or `max_refusals`, records the reached value, and raises below `floor` naming both values (PL2, DN1, DN3, "When placement stops"). | Exact for counts, records and outcomes; centres within 1e-9 µm; the overlap compared on voxel counts exactly; the visible shares recomputed with the same threshold, exact as voxel counts. | routine (`slow` for `app_dn1` if measured so); `synthetic`, `validation`. |
| GA11 | Territories | `app_tiny` (dense 3D and 2D tissue: CT1; culture: CU1), `app_cases` (42) | Outside the nuclei, each cell voxel's `δ_k` against `g_k` and the smallest `δ`; in the culture, `C` against the cell entry's own mask in placement order; `cells.csv` rows against the placed objects. | CT1: `C(σ) = k` where `k` has the smallest positive `δ_k(σ) ≤ g_k` (ties to the smaller index), else 0; CU1: the first placed cell whose mask holds `σ`; one ordinary cell per placed object, no binucleate or empty cell ("Cell territories", "Kinds of cells"). | Exact; ties within 1e-9 µm are boundary voxels, reported, expected 0. | routine; `synthetic`, `validation`. |
| GA12 | Composition (identity 4) | `app_tiny` (each copy), `app_plane`, `app_morph`, `app_coarse`, `app_bg1` (42); `app_cases` (none) | Each float64 stain image (cast off) against `I(σ) = B(σ) + κ · H(σ) + S(σ)` recomputed by the oracle from `placements.csv`, the bank, `background_patches.csv` (BG1, on `app_bg1`) or the `BackgroundConfig` of a closed-form bank (BG2) and the recorded `κ`, at the voxel centres, and against the point evaluator; on `app_coarse`, with IN2's prefilter of `max(0, (t_a / (m s_e,a)) − 1) / 2` source voxels per axis in the oracle; on `app_bg1`, each lattice cell's patch in `background_patches.csv` against the index redrawn from `context.appearance.background`, and `B` against `Σ_p w_p(σ) P_p(σ − o_p) / Σ_p w_p(σ)` with the ramp weights; inside each mask, `I − B − κH` against the owner's `E` alone; the image rendered again at `true_f + k ⊙ s` for an integer `k` against the first shifted by `k`; an isolated object of `app_cases` with the identity transform, placed so that `c_k − x̄_e` is a whole number of source voxels, against its entry's excess shifted by `(c_k − x̄_e) ⊘ s_e`, times its gain. | Equal at any fractional offset and at `Z=1`, except at boundary voxels, with the prefilter active (IN2) and with real-style patches (BG1 as a point function); the patch draws equal the redrawn ones; inside a mask only the owner's signal counts; the integer shift reproduces the image; the identity transform reproduces the entry ("Composition", "Truth", IN2, BG1). | 1e-9 of the image maximum for the identity and for the shift (float rounding of an exact identity; derived below); boundary voxels counted and reported, expected 0. | routine; `synthetic`, `validation`. |
| GA13 | Out-of-focus light and the rim background | `app_tiny` (each copy), `app_routine` (42) | Per stain: the recorded `κ`; the mean over the rims of the placed masks (the mask less its face-connected erosion, on the sample grid) of `B + κH`; the mean local source background of the placed objects; `κ` recomputed by the rule of `proto.py::place_all`, `max(b̄ − mean_rim(B), 0) / mean_rim(H)`; the per-entry blurred copy against `scipy.ndimage.gaussian_filter` of the entry's excess with the deviations of H2. | Where `κ > 0` the rim mean of `B + κH` equals the mean local source background; where `κ = 0` the rim mean of `B` is at least that background, and this is reported; the recorded `κ` equals the recomputation; the blurred copy equals the library call (RA15, H2). Whether H2 equals the prototype's grid blur H1 where signals do not overlap is reported as the largest relative difference, with no bound: neither the prototype nor W-513 measured it. | 1e-9 relative (float rounding of sums over the rim voxels); the blurred copy within 1e-9 of its maximum. | routine; `synthetic`, `validation`. |
| GA14 | Brightness within a scene | `cf_banks`, `app_tiny` (42) | Each entry's excess and patches against the source image over its plateau; `g` against the median plateau of the bank's source images; each placed object's gain. | The stored arrays are relative to their source plateau (BR1); `g` is the median plateau (O8). | Within one float32 rounding (relative 2⁻²³); `g` exact. | routine; `synthetic`, `validation`. |
| GA15 | 8-bit output (identity 5) | `app_tiny_uint8` and its float64 rendering (42) | Each 8-bit image against `clip(rint(g · J), 0, 255)` of the float64 image `J` with the same streams; no noise stream drawn. | Identity 5; OB1 adds no stage ("Composition", RA17, RA18). | Exact. | routine; `synthetic`, `validation`. |
| GA16 | The morphology round (identity 6) | `app_morph`, `app_plane` (42) | `morphology_truth.csv` against the configuration as GN18; with the whole-voxel translation, the morphology `DAPI` against the reference `DAPI` shifted by `τ_f ⊘ s`; both rounds' `DAPI` and the morphology `Flamingo` against the entries' channels at the shifted points (inside GA12). | Identity 6 and identity 8 of {doc}`synthetic-cellular-scene`; MO1. | As GN18: the truth columns within 1e-9 voxels; the shifted image within 1e-9 of the maximum (as GA12), boundary voxels reported, expected 0. | routine; `synthetic`, `validation`. |
| GA17 | Molecules and types (identity 8) | `app_tiny` with the cell types of `ctx_types`, its copy with type 2's `size_scale` 1.5 (`app_streams`), `app_routine` (42) | GN8 to GN12 on these fixtures, with `C` and `N` of the appearance recomputed by the oracle; `placements.csv` and the label images of the `size_scale` copy against the scene without it. | Identities 4 to 6 of {doc}`synthetic-cellular-scene`; M2, NF1, A2 and G1 hold; the size scale of a type has no effect: placements and labels equal ("Cell territories", last paragraph). | As GN8 to GN12 (exact, 4 SE). | routine; `synthetic`, `validation`. |
| GA18 | Projection truth (identity 9) | `app_tiny` (2D tissue, culture), `app_routine` (42) | Per column of fixed `(y, x)` on a FOV grid, the sample grid and a grid at a fractional offset, the nucleus IDs; `P` against the Z maximum; in 2D tissue, the placed signal `S` (recomputed as in GA12) of every plane against that of the middle plane. | In 2D tissue no column holds two nucleus IDs, `P` is the Z maximum, and `S` is the same in every plane (TZ1: each entry a column); in the culture the columns holding two IDs are counted and reported. | Exact; columns at a boundary voxel reported. | routine; `synthetic`, `validation`. |
| GA19 | `Z=1` | `app_plane` (42) | Generation; the shapes of every image and label image; every z; `τ_z`; the label kind `import_labels` reads; GA8, GA12, GA15 and GA16 on this fixture. | Every array has one plane; every z is 0; `τ_z = 0`; `plane` label images ("Truth"; {doc}`synthetic-cellular-scene`, "`Z=1`"). | Exact. | routine; `synthetic`, `validation`. |
| GA20 | A new random stream leaves the existing draws unchanged | `app_streams`, the six context presets (42) | The components the appearance generator draws from; the descriptors against those of the formed model, `sample.*`, `stain.noise.*` and `context.*` (`_SAMPLE_COMPONENTS`, `_CONTEXT_COMPONENTS`); the draws of every other proposal when one law changes; the molecules of other types when type 2's row changes; the other stain when one stain is dropped; with GA1, the accepted presets' outputs. | Every new component starts with `context.appearance.` and equals no existing descriptor; no `context.nucleus.*`, `context.cell.kind`, `context.cell.contact`, `context.stain.*` or `context.texture.*` stream is drawn; a changed gain range changes the draws of no other proposal's entity; GN21's statements on types hold; dropping a stain changes no other stain; the accepted presets are unchanged ("Random streams"). | Exact. | routine; `synthetic`, `validation`. |
| GA21 | Truth schema and persistence | `app_tiny` (each copy), `app_plane`, `app_morph` (42) | The directory `save_appearance_scene` writes: the `save_context_scene` layout; `nuclei.csv` and `cells.csv` columns, the shape columns null, the centre and anchor against `placements.csv`, `site_radius` against the largest distance from the recorded centre `c_k` (the transformed box centre) to a voxel of the placed mask; the empty tables; `appearance.json` keys and `APPEARANCE_TRUTH_VERSION`, the bank's name, SHA-256 and digest, `κ`, `g`, the SHA-256 of every file; `placements.csv` and `background_patches.csv` columns; the tables reloaded; each stain file's name, round and file pattern against its `StainChannelConfig`; the dataset of `save_context_dataset` read by `Dataset`, `FOV.load_images` and `FOV.prepare_morphology`; the labels rebuilt from the saved files and the bank only (GA8 on the reloaded scene). | AT2 as written ("Truth schema"); hashes equal the files; reloaded tables equal; each stain keeps its `StainChannelConfig`'s name, round and file pattern ("Composition"); `context.json` keys as `_payload` writes them. | Exact (`check_exact`). | routine; `synthetic`, `contract`. |
| GA22 | The closed-form bank reads no real data | `cf_banks`, `app_routine` | In a subprocess that builds `cf_banks` and generates `app_routine`, the files opened, recorded by an `open` audit hook (`sys.addaudithook`); the bank records' inputs. | No file outside the test's `tmp_path` and the package is read; the records name a preset and seeds (CF1; "Real data stays outside the repository"). | Exact. | routine; `synthetic`, `contract`. |
| GA23 | The appearance presets in their target ranges | `app_slow`; `app_slow_acquired` for TR4 and TR5 of 2D tissue and culture (42) | Each row of "Target definitions" on the truth labels and the stain of each scene, at the pixel named; per quantity the measured value, the range, the object count and the outcome, printed as one CSV block (`preset, row, quantity, value, low, high, objects, outcome`) that the implementation issue saves in its run directory, as W-485 saved `gn23_targets.csv`. | Each asserted row inside its range; each reported row reported; a quantity on fewer than 100 objects reported (V12). | The ranges, provisional; counts exact. | `slow`; `synthetic`, `validation`; real-bank; run by the Chapter II evidence project (W-536). |
| GA24 | Generation and build cost | `app_routine`, `app_slow`, `app_slow_acquired`, the build of each real bank | Each preset generated, and each bank built, in its own process under `/usr/bin/time -v`: wall time and peak RSS, in the test output. | Within RG1 (one CPU, 16 GiB, 3,600 s per step) for the `slow` presets, their variants and the builds; a routine-tier test that generates a routine appearance preset takes less than 5 s, else it is `slow` ("Presets", "Resources"). | The bound; nothing else is asserted. | `slow`; `synthetic`, `validation`; real-bank for the `slow` presets and the builds, which the Chapter II evidence project runs (W-536). |
| GA26 | The real banks | `real_banks` | Each bank file's SHA-256 against the preset's; the build record's inputs against W-392's inventory (`inventory/`, the SHA-256 of each file); the held-out images (LN Position048, `crop_y18432_x0`, culture Position397) against the sources; the counts per rule against W-513 `tables/bank_selection_<context>.csv`; BR4's guard on every source; the entry count against the largest preset's placements (BS2). | SHA-256 equal; every input inventoried and equal; no held-out image among the sources; the counts equal W-513's rows (3,033 LN entries; 304 2D tissue; 91 culture); every source inside BR4's guard; the bank size reported against the placements. | Exact. | `slow`; `synthetic`, `validation`; real-bank; run by the Chapter II evidence project (W-536). |
| GA27 | Rebuild of a real bank | `real_banks` | The bank rebuilt from its build configuration in a temporary directory; its digest against the stored record's. | Equal digests (ST1, "a rebuild is compared by that digest"). | Exact. | `slow`; `synthetic`, `validation`; real-bank; run by the Chapter II evidence project (W-536). |

**The tolerance of GA12 and GA16 (derived).** Both sides evaluate one formula on the same
float32 bank arrays, read as float64. They differ only in how a position is rounded. A
linear interpolation of an array with largest step `Δ` between neighbours, at spacing `s`,
changes by at most `(Δ / s) · δx` for a position error `δx`. The transform `T_k⁻¹ σ` in
float64 rounds a position of size `D` by at most a few units in the last place: below
`4 · 2.2e-16 · D`, which is 2.8e-14 µm for the 31 µm of the tiny fixtures. The excess is
relative to the plateau, so `Δ` is at most the image maximum `M`. The finest grid is the
2D tissue's 0.094635 µm, which gives at most 3e-13 `M` per array and axis. A voxel sums at
most the owner's term, the fall-offs and blurred copies of the objects within reach, and
the background patches. Taking at most 100 terms on the tiny fixtures (A3's script counts
them), the bound is 1e-10 `M`.
The tolerance of 1e-9 `M` covers it tenfold. The implementing issue (A3) evaluates this
worst case on its fixtures with a script, as W-395 did, before the rows assert it. It
stays at 1e-9 `M` only if that evaluation lies below it.

## Coverage of the appearance page

Every identity and guarantee of {doc}`synthetic-real-appearance`, by section, with the
check that tests it or the reason it has none.

| Section | Identity or guarantee | Check |
| --- | --- | --- |
| "What this scene is not" | Not a calibration; masks from method labels; no accuracy; no learned model | none: statements of scope, kept by "The claim boundary"; GA22 for the routine tier's data |
| "Settled decisions" | AU1's ranges; real data outside the repository; NumPy, SciPy and scikit-image only | GA7; GA22, GA26; none for the libraries (the review checks the imports and the lock) |
| "Terms" | `D_e = EDT(not M) − EDT(M)`; `D_e ≤ 0` exactly the mask on its grid; `δ_k = m_k D_e(T_k⁻¹ σ)`; excess, rim level, plateau, visible stain | GA6 (`D_e`, the excess, the rim level, the plateau); GA8 (masks from `D_e`) and GA11 (`δ_k` against `g_k`); GA10 (the visible stain in DN1's stop) and GA23 (on the presets) |
| "Opt-in" | A scene without the new settings is the accepted scene; the six presets, their functions and fields keep their behaviour; `_payload` unchanged | GA1, GA2 |
| "Opt-in" | The new names | GA2; the reference audit |
| "The appearance bank" | EN1: what an entry is (a nucleus; in the culture a cell with its largest nucleus label and its DAPI and Flamingo) | GA6 (the entry's arrays; the culture's largest nucleus and both channels), GA11 (culture cells as their own masks) |
| "The appearance bank" | SRC2: the held-out images are never a source | GA5 (closed form), GA26 (real) |
| "The appearance bank" | SEL1: the four rules, no cap; the counts | GA5; GA26 against W-513's counts |
| "The appearance bank" | BR1 and BR4's guard | GA14; GA3 (refusal); GA26 |
| "The appearance bank" | BS2: one permutation per scene, a refused entry moved to the end, no reuse before the bank is exhausted; the build reports the bank size | GA7 (the queue replayed); GA26 (the size reported) |
| "The appearance bank" | ST1: one file, its SHA-256 checked; the build record; a rebuild compared by digest | GA3, GA4, GA26, GA27 |
| "The appearance bank" | The estimated bank size (about 0.9 GB) | none: an estimate; GA24 records the build |
| "The bank without real data" | CF1: the routine tier, the CI environment and the golden tests read no real data | GA22; "Conventions" (no routine fixture reads a bank file) |
| "What an entry carries" | Inside the mask the excess unchanged and not blurred; OUT1 outside: the capped fall-off, the weight by the nearest source label, the per-distance median on shared faces made non-increasing, the 0.25-µm blur outside the mask; `λ` and the blur as configuration | GA6 (each step recomputed; `fall_off_cases` (a) the cap, (b) the neighbour weighting, (c) the non-increasing median; zeros beyond `λ` + 1.0 µm) |
| "Augmentation" | AU1; the transform `A_k = S_k R_k F_k`; the culture's Z kept and shifted by one constant onto the middle plane | GA7 (the draws; the culture's Z shift against `z_mid` less the median nucleus centroid); GA8 (the transform evaluated) |
| "Augmentation" | IN2: the mask from `D_e`; the prefilter `max(0, (t_a / (m s_e,a)) − 1) / 2`, which depends only on the entry, its scale and the target spacing | GA8 (masks from `D_e` at fractional offsets); GA12 (the prefilter active on `app_coarse`; the integer shift: no dependence on the offset) |
| "Layout" | PL2: contact proposals and the ladder of `f`, uniform proposals, at most 1 % overlap, 35 % in the domain, a culture nucleus touching no placed nucleus, the anchor rule, the earlier object owning an overlap | GA10 (each proposal replayed; 35 %; culture nuclei), GA9 (overlap, anchors, ownership) |
| "Layout" | Kinds of cells: one ordinary cell per object; the culture's largest nucleus label | GA11; GA5 (rule 3) |
| "Layout" | DN1 and DN3; when placement stops (target, `max_proposals`, `max_refusals`); the record; `floor` | GA10 (DN1 on `app_dn1`, DN3, `app_crowded` at each stop and floor); TR2v in GA23 |
| "Layout" | CT1, CU1; one ownership rule; `N = n > 0` implies `C = cell(n)` | GA9, GA11 |
| "Layout" | The size scale of a type has no effect | GA17 (the `size_scale` copy against the scene) |
| "Layout" | TZ1: a column through the planes | GA18 (no column holds two IDs; `S` the same in every plane) |
| "Composition" | `I = B + κH + S`; only the owner's signal inside a mask; equal to the prototype's composition where masks do not overlap | GA12; the equality with the prototype is not checked (the prototype is not code of the repository) |
| "Composition" | H2: a point function; `κ` once per scene, recorded; equal to H1 where signals do not overlap | GA12, GA13 (the equality with H1 reported, no bound) |
| "Composition" | BG1 a point function, with one patch per lattice cell and ramp weights; BG2 for the closed-form bank; the patch rule (at most 10 % foreground) | GA12 (BG1 on `app_bg1` with the closed-form bank's patches, the draws and the formula; BG2 on the other fixtures); GA6 (the patch rule); GA23 measures the result on real patches (TR8r, TR8v) |
| "Composition" | OB1; O8 | GA15 (no noise stage), GA14 (`g`) |
| "Composition" | Each stain keeps its `StainChannelConfig`'s name, round and file pattern | GA21 |
| "Truth" | The placed masks are the labels; each object a function of the sample position; only the placement test and `κ` are not | GA8, GA12, GA13 |
| "Truth" | AT2 and its files | GA21 |
| "Truth" | Identities 1 to 9 | GA1 (1), GA8 (2), GA9 (3), GA12 (4), GA15 (5), GA16 (6), GA3, GA4, GA7 and GA26 (7), GA17 (8), GA18 (9) |
| "The morphology round" | MO1; the translation, its truth table and the dataset layout unchanged | GA16, GA21; CU0 |
| "Target ranges" | TG2 | "Target definitions"; GA23 |
| "Presets" | PR1: six presets, their names, version and values | GA2 |
| "Presets" | Resources: estimates; each preset measured with `/usr/bin/time -v`; a test of 5 s or more is `slow` | GA24 |
| "Random streams" | The components and entities; no existing draw changes; no `context.nucleus.*` and similar stream drawn | GA20, GA1 |
| "Known limits" | L1 to L8 | GA23 reports L1 (TR8r, TR8v), L2 (TR3), L3 (TR4); L4 is read by "The claim boundary"; L5 by GA26 (91 cells); L6 by GA8 (labels from `D_e`; the pieces per label reported; the page states no criterion for a speck, so none is asserted); L7 and L8 are accepted limits with nothing to compute |
| "What stays excluded" | No learned model, no real data in the repository, no bank delivered, no out-of-plane rotation, no real membrane stain | GA22; GA7 (rotation about Z only); none for the rest (scope) |
| "Accepted statements this page changes" | Entries 1 to 43 | None by a test: the documentation issue applies the wording and its review checks it against the list. For entries 35 to 43 (this design's page) the wording here supersedes W-513's where the two differ ("Accepted statements this page changes", below) |

## Implemented checks

The generator checks are in the modules of `src/python/test/` listed below, each marked
`synthetic` with the kind marker its row names. The implementation issues W-516 to W-519
added most of them with the code they check. W-520 adds three modules:

* `test_synthetic_appearance_checks.py`, which holds `test_every_ga_row_names_collected_tests`
  to keep this table true;
* `test_synthetic_appearance_coverage.py`, the rows on fixtures they name that no earlier
  module ran: GA13 and GA17 on `app_routine`, GA20's copy with type 2's abundance row changed,
  GA21 on the culture copies of `app_plane` and `app_morph` (drawn), GA22 with `app_routine`
  generated under the audit hook, GA15 on `app_plane` (GA19 names it) and GA2's fifteen names
  together;
* GA23 in `test_synthetic_appearance_presets.py`, with its measures in
  `cellular_appearance_measures.py`; W-536 moved it to `benchmarks/appearance_evidence.py`
  (below).

That test reads the rows GA1 to GA24 and GA26 to GA27 of the table "Generator checks" from this page and
checks four things:

* this table has exactly those rows, in that order;
* every row names at least one test of its module that pytest collects, and none of them
  is marked as an expected failure or skipped by a mark;
* each named test carries `synthetic` and the one kind marker its GA row names;
* a row whose tier is `routine` names no `slow` test, a row whose tier is `slow` names only
  `slow` tests, and a row whose tier is `routine and slow` names both.

A row whose tier is `evidence project` names its script under `benchmarks/` and no test,
and every script a row names exists.

**Checks run by the evidence project (W-536).** GA23, GA26, GA27 and the `slow` part of GA24
(the `slow` presets, their acquired variants and the bank builds) read the real banks,
rebuild them from real images and allow RG1 per step, so they are evidence workloads rather
than software checks. Jiahao accepted moving them to the Chapter II evidence project
(TM-14 of W-526, 2026-10-19). They run as `benchmarks/appearance_evidence.py`, which pytest
does not collect, with the same assertions, printed rows and bounds as the tests they were
at revision `0b9fa8b`. From `src/python`:
`STARFINDER_APPEARANCE_BANKS=<bank directory> uv run python ../../benchmarks/appearance_evidence.py --output <run directory>`;
`--check` selects checks and `--list` lists the cases, banks and seeds without reading a
bank. The script imports the helpers of this page from `src/python/test/`. GA8 to GA18, the
routine part of GA24 (each routine preset within the project's 4 GiB) and the closed-form
bank checks stay in the suite.

A test of these modules that is not named here is an additional check of the code it
tests. The helpers are:

* `cellular_appearance_oracle.py`, the test oracle;
* `cellular_appearance_scenes.py`, the fixtures;
* `cellular_appearance_measures.py`, GA23's measures: W-393's through `cellular_measures.py`,
  and the visible-stain share, the median outside the visible stain and the background at
  label rims rebuilt from the definitions of W-513 `scripts/measure.py`.

GA23 measures the scenes that GA24 generates and saves, each in its own process, read back
from their files, so that no `slow` preset is generated twice; both run in
`benchmarks/appearance_evidence.py`. TR2v, TR8v and TR8r are
measured on the stain of the sample grid, rendered by the point evaluator and cast as the
FOV images are; the other rows on each FOV's stain and truth labels, each object once, as
GN23 measures them. Each scene's rows are printed as one CSV block
(`preset, row, quantity, value, low, high, objects, outcome`). A row the page asserts,
outside its range where V12 compares it, fails the test and is a finding.

Every tolerance is the one of the table "Generator checks", unchanged. Where a row says
"Exact" for a number that is not an integer and that the test computes itself, its tests
compare it within an absolute 1e-9 (the project's default float tolerance), or within the
bound the row itself states. Counts, IDs, labels, strings, hashes and dtypes are compared
exactly, and so are the 8-bit images of GA15 against the row's expression. No test needs an
extra beyond `dev`, `local-registration` and `checkpoint`. A real-bank check skips, naming
`STARFINDER_APPEARANCE_BANKS` or the missing file, where the variable or the bank is
missing; GA24 needs `/usr/bin/time`.

Where a test differs from its row as written, the case is recorded in its issue's worker
notes, and the row is not changed:

* **Tiers.** A check on `app_routine` generates its routine preset, which takes 5 s or more
  on one CPU, so it is `slow`; so are the copies on the dense 3D closed-form bank and the
  `cf_bank_files` of dense 3D and culture. Rows the page tiers `routine` therefore have a
  `slow` part: GA3, GA4, GA8 to GA15, GA17, GA18 and GA20 to GA22.
* **GA1** names the golden tests. GN1 (b), a `validation` test of W-483, stays unchanged and
  passes in its tier.
* **GA2.** The fields of the accepted configurations are GN2's tests, and the keys `_payload`
  writes are kept by GA1 (b), whose pinned digests hold each context preset's
  `context.json`.
* **GA23.** TR4 and TR5 of 2D tissue and of the culture are measured on `app_slow_acquired`
  and reported; TR7 of `appearance-dense3d-slow-v1` is asserted against 0.25–0.41, while
  GN23 keeps reporting TR7 of `context-dense3d-slow-v1` (AM1).
* **GA23, the amendment of 2026-10-18.** TR2v of dense 3D and TR8r of dense 3D and 2D tissue
  are reported beside their range, with the outcome "reported: amendment of 2026-10-18
  (W-520), beside the range", as the amendment after "Target definitions" states. Every
  other row the page asserts still fails the test when it leaves its range.

**Retired check:** GA25 was retired on 2026-10-19 because it froze test, helper and selected
fixture source text during development, forcing later test changes to edit source pins first.
Behavioral goldens and shared-fixture checks remain.

| # | Module and tests | Fixtures | Tier | Floats |
| --- | --- | --- | --- | --- |
| GA1 | `test_synthetic_sample_golden.py`, `test_saved_preset_outputs_equal_the_pins`; `test_synthetic_appearance_golden.py`, `test_saved_context_preset_outputs_equal_the_base_revision_pins` | `sample_presets`, the six context presets | routine and slow | none compared (SHA-256 of content); floats to 6 significant digits (amendment of 2026-10-19) |
| GA2 | `test_synthetic_appearance_contract.py`, `test_new_names_are_exported`; `test_synthetic_appearance_presets_contract.py`, `test_six_presets_with_their_version`, `test_each_preset_takes_the_accepted_preset_of_its_context_and_tier`, `test_slow_presets_name_their_real_bank_by_name_and_sha256`, `test_the_accepted_presets_are_unchanged`, `test_preset_names_are_exported_and_on_the_api_pages`; `test_synthetic_appearance_coverage.py`, `test_ga2_every_new_name_is_exported_and_on_the_api_pages`; `test_synthetic_cellular_contract.py`, `test_gn2_existing_fields_and_presets_as_at_the_base`, `test_gn2_new_names_and_fields`; `test_synthetic_cellular_stains_contract.py`, `test_gn2_stain_names_and_fields` | none (the API pages by `docs/check_reference.py`) | routine | preset floats within 1e-9 |
| GA3 | `test_synthetic_appearance_bank.py`, `test_load_refuses_a_changed_byte_a_wrong_reference_and_a_missing_file`, `test_bank_reference_entries_and_build_refusals`, `test_br4_refuses_both_copies_of_the_file_bank`; `test_synthetic_appearance_scene.py`, `test_ga3_augmentation_and_layout_refusals`, `test_ga3_a_reached_count_below_floor_raises_naming_both_values`; `test_synthetic_appearance_contract.py`, `test_configuration_refusals`, `test_generation_refusals_and_a_stored_bank` | `cf_bank_files` and its two copies, `app_crowded`, none | routine and slow | none compared (the type and the named values) |
| GA4 | `test_synthetic_appearance_bank.py`, `test_closed_form_bank_build_is_deterministic_and_recorded`, `test_file_bank_build_is_deterministic_and_recorded` | `cf_banks`, `cf_bank_files` | routine and slow | arrays and digests exact |
| GA5 | `test_synthetic_appearance_bank.py`, `test_bank_cases_meet_their_known_outcomes`, `test_closed_form_bank_selection_equals_the_oracle` | `bank_cases`, `cf_banks` | routine | rim shares and percentiles by the same arithmetic; labels within 1e-9 of a limit reported |
| GA6 | `test_synthetic_appearance_bank.py`, `test_closed_form_entries_carry_what_the_page_states`, `test_bank_case_entries_and_patches`, `test_fall_off_cases` | `cf_banks`, `bank_cases`, `fall_off_cases` | routine | `D_e` within 1e-6 µm; the excess within 2⁻²³ relative; outside within `max(1e-6 · r, 1e-9)` (the amendment of 2026-10-17) |
| GA7 | `test_synthetic_appearance_scene.py`, `test_ga7_transforms_are_in_range_redrawn_and_entries_follow_bs2` | `app_tiny`, `app_streams` (gain), `app_cu_offset` | routine | the culture's Z within 1e-9 µm |
| GA8 | `test_synthetic_appearance_scene.py`, `test_ga8_labels_equal_the_oracle_on_every_grid`, `test_ga8_ga9_hand_made_placements`, `test_ga8_morphology_grid_is_the_reference_grid_moved_by_whole_voxels`; `test_synthetic_appearance_presets.py`, `test_ga8_routine_labels_equal_the_oracle_on_every_grid` | `app_tiny`, `app_plane`, `app_morph`, `app_cases`, `app_routine` | routine and slow | decisions within 1e-9 µm counted as boundary voxels |
| GA9 | `test_synthetic_appearance_scene.py`, `test_ga9_ownership_overlap_and_anchors`, `test_ga8_ga9_hand_made_placements`; `test_synthetic_appearance_presets.py`, `test_ga9_routine_ownership_overlap_and_anchors` | `app_cases`, `app_tiny`, `app_plane`, `app_routine` | routine and slow | anchors within 1e-9 µm |
| GA10 | `test_synthetic_appearance_scene.py`, `test_ga10_each_centre_is_the_one_pl2_gives`, `test_ga10_crowded_placement_stops_and_records`; `test_synthetic_appearance_stains.py`, `test_ga10_dn1_target_threshold_and_stop`, `test_ga10_dn1_short_copy_records_or_raises_at_its_floor`; `test_synthetic_appearance_presets.py`, `test_ga10_routine_placement_and_its_stop` | `app_tiny`, `app_crowded`, `app_dn1`, `app_routine` | routine and slow | centres within 1e-9 µm |
| GA11 | `test_synthetic_appearance_scene.py`, `test_ga11_territories`, `test_ga8_ga9_hand_made_placements`; `test_synthetic_appearance_presets.py`, `test_ga11_routine_territories` | `app_tiny`, `app_cases`, `app_routine` | routine and slow | ties within 1e-9 µm counted |
| GA12 | `test_synthetic_appearance_stains.py`, `test_ga12_stains_equal_the_composition`, `test_ga12_background_patches_are_the_redrawn_lattice`, `test_ga12_an_isolated_object_with_the_identity_transform_is_its_entry` | `app_tiny`, `app_plane`, `app_morph`, `app_coarse`, `app_bg1`, `app_cases` | routine and slow | 1e-9 of the image maximum |
| GA13 | `test_synthetic_appearance_stains.py`, `test_ga13_kappa_and_the_rim_background`; `test_synthetic_appearance_coverage.py`, `test_ga13_kappa_and_the_rim_background_on_the_routine_presets` | `app_tiny`, `app_plane`, `app_bg1`, `app_routine` | routine and slow | 1e-9 relative; the blurred copy within 1e-9 of its maximum |
| GA14 | `test_synthetic_appearance_bank.py`, `test_brightness_is_relative_to_the_source_plateau`; `test_synthetic_appearance_stains.py`, `test_ga14_g_is_the_median_plateau_and_each_gain_scales_its_object` | `cf_banks`, `app_tiny` | routine and slow | within 2⁻²³ relative; `g` within 1e-9 |
| GA15 | `test_synthetic_appearance_stains.py`, `test_ga15_8bit_images_are_the_cast_and_no_noise_is_drawn`; `test_synthetic_appearance_coverage.py`, `test_ga15_one_plane_8bit_images_are_the_cast` | `app_tiny_uint8`, `app_plane` with the 8-bit cast | routine and slow | none compared (8-bit images exact) |
| GA16 | `test_synthetic_appearance_stains.py`, `test_ga16_morphology_round` | `app_morph`, `app_plane` | routine | 1e-9 voxels; 1e-9 of the maximum |
| GA17 | `test_synthetic_appearance_stains.py`, `test_ga17_types_genes_and_molecules`, `test_ga17_the_count_guard`, `test_ga17_zero_spread_gives_the_median_mean`, `test_ga17_rendering_identity_of_the_reference_round`, `test_ga17_a_type_size_scale_changes_no_placement_and_no_label`; `test_synthetic_appearance_coverage.py`, `test_ga17_types_and_molecules_on_the_routine_presets` | `app_tiny` with `ctx_types`, `app_streams` (size scale), `app_routine` | routine and slow | 4 SE; row sums and `mecr` within 1e-9 |
| GA18 | `test_synthetic_appearance_scene.py`, `test_ga18_columns_hold_one_nucleus_in_2d_tissue`; `test_synthetic_appearance_presets.py`, `test_ga18_routine_projection_truth` | `app_tiny` (2D tissue, culture), `app_routine` | routine and slow | none compared (IDs exact) |
| GA19 | `test_synthetic_appearance_scene.py`, `test_ga19_one_plane_labels`; `test_synthetic_appearance_stains.py`, `test_ga19_one_plane_stains`, `test_ga19_saved_labels_import_as_plane_label_images` | `app_plane` | routine | none compared (every z exactly 0) |
| GA20 | `test_synthetic_appearance_scene.py`, `test_ga20_layout_components_start_with_context_appearance`, `test_ga20_a_changed_gain_range_changes_no_other_draw`, `test_ga20_a_type_size_scale_changes_no_placement_or_label`; `test_synthetic_appearance_stains.py`, `test_ga20_components_of_the_appearance_scene`, `test_ga20_dropping_a_stain_changes_no_other_stain`; `test_synthetic_appearance_coverage.py`, `test_ga20_a_type_abundance_changes_only_that_types_genes` | `app_streams`, `app_tiny` | routine and slow | the redrawn gain within 1e-12 |
| GA21 | `test_synthetic_appearance_contract.py`, `test_ga21_layout_versions_nulls_tables_and_hashes`, `test_ga21_dataset_registers_the_morphology_round`; `test_synthetic_appearance_coverage.py`, `test_ga21_layout_on_the_culture_plane_and_drawn_morphology` | `app_tiny`, `app_plane`, `app_morph` | routine and slow | floats round trip exactly |
| GA22 | `test_synthetic_appearance_bank.py`, `test_closed_form_bank_reads_no_file_outside_the_package`; `test_synthetic_appearance_coverage.py`, `test_ga22_a_routine_preset_reads_no_file_outside_the_package` | `cf_banks`, `app_routine` | routine and slow | none compared |
| GA23 | `benchmarks/appearance_evidence.py`, checks `GA23-slow` and `GA23-acquired` | `app_slow`, `app_slow_acquired` | evidence project | the target ranges |
| GA24 | `test_synthetic_appearance_presets.py`, `test_ga24_routine_preset_generation_cost`; the `slow` presets, the acquired variants and the builds in `benchmarks/appearance_evidence.py`, checks `GA24-slow`, `GA24-acquired` and `GA24-build` | `app_routine`; `app_slow`, `app_slow_acquired`, `real_banks` in the script | slow | the project's 4 GiB; the bound of RG1 in the script |
| GA26 | `benchmarks/appearance_evidence.py`, check `GA26` | `real_banks` | evidence project | plateaus within 1e-9 |
| GA27 | `benchmarks/appearance_evidence.py`, check `GA27` | `real_banks` | evidence project | none compared (digests) |

## Consequences for the chain checks

The chain checks CD, CT and CU of {doc}`synthetic-cellular-scene-validation` (W-487, not
started) were planned on the closed-form presets. On 2026-10-16 Jiahao judged those presets
too unlike the real images. He asked for the texture, background and shape to be fixed
before the chain checks. Each check is therefore **amended**: it runs on the appearance
presets, and its steps, matching rules and thresholds stay. None is replaced, because
nothing in the route, the segmentation call or the four steps depends on how the stain was
made. None stands unchanged on the closed-form presets, because that would run the chain
on the appearance Jiahao rejected.

| Check | Fate | Fixture | Thresholds (all provisional) | Reason |
| --- | --- | --- | --- | --- |
| CD (dense 3D, route per FOV, StarDist `3D_spleen`) | **Amended** | `chain_dense3d` on `appearance-dense3d-slow-v1` (bank `ln-sample1-v1`); `chain_dense3d_routine` on `appearance-dense3d-v1` (closed-form bank); `chain_dense3d_truth` on both | TH1 matched fraction ≥ 0.9, TH2 merged ≤ 0.10, TH3 splits 0, TH4 agreement ≥ 0.85: the values of W-393 (TH table of {doc}`synthetic-cellular-scene-validation`), unchanged and provisional, measured by W-393 on neither scene | The truth is StarDist `3D_spleen` labels, so CD1 is read as the method on its own labels ("The claim boundary"). The scene holds about 738 nuclei (estimate), against about 150, at the visible stain's density. CD5 is added |
| CT (2D tissue, route after fusion, StarDist `2D_versatile_fluo` and the seeded watershed) | **Amended** | `chain_tissue2d` on `appearance-tissue2d-slow-v1` (0.18927 µm, StarDist at the scale 0.5; bank `tissue2d-v1`); routine and truth variants as CD | TH1 to TH4 as CD; TH5 every FOV `image_supported` and every per-axis error ≤ 1 µm, unchanged and provisional | CT0's stitch now runs on the real PI texture and background; `P` holds by TZ1. About 337 nuclei (estimate) against about 76. CT5 is added |
| CU (culture, route per FOV, Cellpose `cpsam_v2` with the morphology round) | **Amended** | `chain_culture` on `appearance-culture-slow-v1` (0.388796 µm; bank `culture3d-v1`) at the seeds 42 and 43, as accepted; routine and truth variants as CD | TH1 to TH4 as CD, on the two scenes pooled | The truth cells and nuclei are Cellpose `cpsam_v2` labels extended through z (CU1 of {doc}`synthetic-real-appearance`), read as the method on its own labels. About 32 cells per scene (estimate), about 63 pooled, against about 83. A culture entry can carry the unlabelled DAPI of a second nucleus, a source of false positives (TH-FP, reported). CU0 stands: MO1 keeps both rounds registered on the shared `DAPI`. CU5 is added |

**Amendment of 2026-10-18 (W-487; operator decision, confirmed by Jiahao on 2026-10-18).** The first run of the chain
checks (W-487, the seeds fixed in advance) missed two rows on the `slow` presets:

* CD1's TH1 on `appearance-dense3d-slow-v1`: StarDist `3D_spleen` matched 499 of 875 interior truth nuclei, 0.570
  against at least 0.9 (1,381 truth nuclei, 612 labels). TH2 (10 merged of 1,381) and TH3 (no split) hold.
* CU1's TH3 on `appearance-culture-slow-v1`, the seeds 42 and 43 pooled: 2 splits against 0, both at seed 42. TH1
  (100 of 106 interior truth nuclei, 0.943) and TH2 (no merge) hold.

The causes are the following. In dense 3D, DN1 places labelled nuclei until the visible-stain share of the real
images is reached. That stain includes dim nuclei which StarDist did not label in those images (limits L1 and L4).
The scene therefore holds about three to six times the label density of the method's labels on real LN: its truth
labels fill 0.47 of the volume, against W-393's 0.07–0.16. The 16-plane stack adds to it: 772 of the 875 interior
nuclei are cut by a Z end. In the culture, one split comes from a truth mask with a detached piece that Cellpose
labels separately, which V4 counts as a split (limit L6: the pieces per label are reported, and no criterion for a
speck is given). The other comes from a two-lobed mask at the FOV border, which Cellpose labels as two. These two
rows are **reported beside their threshold, not asserted**, with their value and counts. Every other threshold stays
asserted, CD1's TH2 and TH3 and CU1's TH1 and TH2 included. No threshold, preset, bank, seed or matching rule
changes. Whether DN1 should use another threshold was a question Jiahao closed on 2026-10-18:
the rows stay known limits ("Implementation").

**Steps 0 to 4** stay as written, with these readings:

* **The truth on a grid.** On the route per FOV the truth is each FOV's label images. On
  the route after fusion it is `P` on the segmentation grid, evaluated by the appearance
  point evaluator at the grid's pixel centres (choice V3, with the evaluator of
  {doc}`synthetic-real-appearance`, "Truth").
* **Interior nuclei.** Nuclei cut by the Y or X border are counted as accepted (C21, Z1).
  The source entries never touch a Z end of their own stack (SEL1, rule 1), but a placed
  nucleus can be cut by the 16-plane stack of the preset. Those nuclei are reported as
  accepted.
* **Molecules** come from the truth positions (choice V2). The territories are CT1's,
  grown by the accepted 0.78 µm in dense 3D, beside the route's expansion of 0.7776 µm, and
  CU1's cells in the culture.

**Step 5, the scene as the method sees it (new; reported).** One row per context:

| # | Step | Fixture (seed) | What is computed | Expected result (source) | Threshold or tolerance | Tier and markers |
| --- | --- | --- | --- | --- | --- | --- |
| CD5 | The method's labels against the method's labels of real images | `chain_dense3d` (42) | On `3D_spleen`'s nucleus labels of each FOV: the touching share (TR1), the volume fraction (TR2) and the cell volume of the nuclei expanded by 0.7776 µm (TR11), with W-393's definitions; the same on the truth labels beside them. | Reported against W-393's ranges of TR1, TR2 and TR11, which are measured on the same method's labels of the real images: a like-for-like comparison of what the method sees (TG2). | Reported, with the value, the range and the object count; no threshold (no measurement of the method's labels on an appearance scene exists). | `slow`, `learned`; `integration`, `validation`. |
| CT5 | As CD5 | `chain_tissue2d` (42) | On the export's nucleus labels: TR1 and TR2 (area fraction); the truth beside them. | As CD5, against W-393's 2D tissue ranges. | Reported. | as CD5. |
| CU5 | As CD5 | `chain_culture` (42 and 43) | On Cellpose's nuclei extended through z, per scene and pooled: TR1 and TR2; the truth beside them. | As CD5, against W-393's culture ranges. | Reported. | as CD5. |

**How many nuclei the thresholds count.** About 738, 337 and 32 objects per `slow` scene,
and about 63 for the culture's two scenes pooled. These are W-513's estimates at the
prototype's reached density ({doc}`synthetic-real-appearance`, "Presets"; W-513
`tables/resource_estimates.csv`, column `objects_at_prototype_density`), not counts of a
generated scene. The routine appearance presets hold about 24, 45 and 5 objects. Step 1
reports the interior count it used, as accepted.

**The reported quantities** TH-FP, TH-UN, TH-MECR, TH-CPF, TH-REG and TH-PL stay reported
with no threshold. TH-FP adds the culture's unlabelled second nuclei as a named cause.

**What a failure means** stays as written in {doc}`synthetic-cellular-scene-validation`,
with the reading of truth masks of "The claim boundary". A miss in CD1 or CU1 is a finding
about the scene or the chain. Its options for Jiahao are a scene control within its target
range (the density target, `λ`, `κ`, the squash threshold), the threshold, or a known
limit.

## Execution conditions

**Where banks live.** A bank is one file `<bank name>.npz` with its build record and its
build configuration, outside the repository (ST1).

* **The master copy** is under the artifact root, in
  `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/appearance-banks/<bank name>/`,
  with `<bank name>.npz`, the build record and the build configuration (sources, held-out
  images, rules, patch sizes). This is configuration per dataset, not code.
* **The working copy** is on local disk, in the directory that `STARFINDER_APPEARANCE_BANKS`
  names on the lab machine, for example `/home/unix/jiahao/.cache/starfinder/appearance-banks/`,
  beside the model weights. The artifact mount is an SMB share, so a test reads the local
  copy, and the SHA-256 check makes the two equal.
* **No bank, record, configuration of sources or array derived from real images enters the
  repository.** The repository's presets name each bank by its name and SHA-256 (ST1;
  choice VA7 below).

**Which tiers need a bank built from real data.**

| Tier and selection | Checks | Bank |
| --- | --- | --- |
| Routine (the fast tier R, `not (slow or learned or extended)`: the quick gate Q and V), and the CI environment | GA1 (routine presets) to GA22 | closed-form banks built in the session, or hand-made arrays; **no real data** |
| `slow`, `synthetic` (`pytest-slow-synthetic`) | GA1 (the `slow` context presets); the routine fixtures that measure 5 s or more | closed-form banks; no real data |
| The Chapter II evidence project (`benchmarks/appearance_evidence.py`, W-536; not in the test suite) | GA23, GA24 (the `slow` presets, the acquired variants and the builds), GA26, GA27 | the three real banks |
| `slow` and `learned`, `integration` (the `learned` selection) | CD, CT and CU with a model, on the `slow` presets | the three real banks |
| `slow` and `learned`, `integration` | the routine chains (`chain_*_routine`) | closed-form banks; no real data |
| `slow`, `integration` (`pytest-slow-other`) | the truth-label chains `chain_*_truth` on the `slow` presets | the three real banks |

**The CI environment never needs a bank.** It installs the extras `dev`,
`local-registration` and `checkpoint` and has no `STARFINDER_APPEARANCE_BANKS`. Every
routine check builds its bank in the session, and every real-bank check skips there,
naming the variable. The chain checks skip there for their missing extras, as accepted. No
check fails there for want of a bank. On the lab machine the variable must be set in the
environment of the controller's `slow` and `learned` selections. Otherwise the real-bank
checks skip there too (choice VA8), and the implementation issue's criterion asks for the
selection log that shows them passed, not skipped.

**Measured cost of each tier.** Nothing of the appearance scene is implemented, so no tier
was measured as such. Each row gives what the prototype record or W-513 measured on one CPU
with `/usr/bin/time -v` (W-513 `tables/prototype_steps.csv` from `round10/logs/<context>.run.log`;
W-513 `tables/resources.csv`, final attempts), and names what is missing. "Peak RSS" is in
KiB.

| Tier | What runs | Nearest measurement (source) | Wall time | Peak RSS | Missing |
| --- | --- | --- | --- | --- | --- |
| Routine | A closed-form bank (CF1) and an appearance scene on it | **Not measured** by the prototype or W-513 (neither built a closed-form bank). Nearest operation: generating the accepted routine presets that CF1 renders, W-485 `gn24_resources.csv` rows `context-*-v1` (a measurement of that operation, not of a bank) | 2.12 s, 2.88 s, 3.70 s per preset render | 0.239, 0.205, 0.247 GiB | The bank build and the scene on it; the upper estimates of W-513 (`tables/resource_estimates.csv`, `placement_estimate_s_high`: 10 s, 39 s, 51 s) do not show that a routine test stays below 5 s. Measured by A1 to A3 with `--durations=0` |
| `slow` real-bank: build | `build_appearance_bank` on the configured sources | The prototype's bank stage, inside its scene step (`prototype_steps.csv`, `bank_stage_s`); W-513's measurement steps, which read the same FOVs and applied the four rules to every label (`resources.csv`, `measure_ln.rerun-20261016T210713Z`, `measure_tissue2d.rerun-20261016T210822Z`, `measure_culture.rerun-20261016T210902Z`) | Prototype: 96.8 s (LN), 43.5 s (2D tissue), 35.8 s (culture). W-513: 68.59 s, 39.83 s, 89.63 s | Prototype: within its steps' 3,987,588; 1,970,664; 3,442,840. W-513: 2,127,600; 1,954,536; 2,870,856 | The build with SEL1 on every label, its file write and its digest. Measured by A1 and A4 (GA24) |
| `slow` real-bank: a `slow` preset | Generating an appearance `slow` preset | The prototype's whole scene steps (`prototype_steps.csv`, `wall_s`, `peak_rss_kib`): LN 50 × 512 × 512 with 1,305 nuclei and 40,000 proposals; 2D tissue 1 × 1024 × 1024 with 210; culture 30 × 1024 × 1024 with 19 cells | 669.51 s, 228.5 s, 235.79 s | 3,987,588; 1,970,664; 3,442,840 | The presets themselves. W-513's estimates (421 s, 1,933 s, 367 s; "Presets") are estimates. Measured by A4 (GA24) |
| `slow` real-bank: GA23 | The target measures on a scene | The prototype's measurement stage (`measurement_stage_s`) | 45.6 s, 2.2 s, 1.8 s | within the steps above | GA23 on the presets; measured by A5 |
| `slow` and `learned`: chain checks | Generation, stitch, segmentation, assignment, export and report | **Not measured** on an appearance scene. The nearest operations are W-393's, as {doc}`synthetic-cellular-scene-validation` gives them ("Execution conditions"): `3D_spleen` on 3 × 3 FOVs 778.92 s and 3,263,160 KiB; the export after fusion 22.0 s and 994,216 KiB; the Cellpose nucleus call 261.22 s and 2,428,836 KiB per scene | as named | as named | Every whole chain on the appearance presets; measured by the amended W-487 |

**Bounds.** RG1 (one CPU, 16 GiB and 3,600 s per step) covers every step. These sums are
estimates: W-513's preset estimates plus W-393's nearest operations.

* **CD:** 420.71 s and 778.92 s make about 1,200 s.
* **CT:** 1,933.47 s and 22.0 s make about 1,955 s.
* **CU**, two scenes: 2 × 366.54 s of generation, plus per scene the nucleus call, the cell
  call counted at the nucleus call's time, and the extension through z (2 × 536.11 s). That
  makes about 1,805 s.

Each sum is below 3,600 s. Generation and model held at once would peak at about 7,250,748
KiB in CD, below 16 GiB. The implementation measures each whole chain under
`/usr/bin/time -v`.

**Models** are found and checked as {doc}`synthetic-cellular-scene-validation` states
("Models, by path, offline"). No GPU is used.

## Accepted statements this page changes

Each statement of {doc}`synthetic-cellular-scene-validation` that this design would change,
with the old and the proposed wording, in the order of that page. The page is not edited
here. An implementation issue (the amended W-488) applies each entry as written and
nothing else. The statements not listed stay true, because the closed-form scene and its
presets are unchanged (E1, PR1).

{doc}`synthetic-real-appearance` already lists nine statements of that page (its entries 35
to 43), with "W-514 states" where it left a point open. The entries below include all nine
with the open points settled. Where the two lists differ, the wording here is the one to
apply. Entries 3, 7, 22, 27, 28, 29, 30, 32 and 38 below are W-513's entries 35 to 43
completed (named in each).

1. **Opening paragraph, first sentence.**
   *Old:* "This page is the validation design of the second delivery of §2.12, the cellular
   scene of {doc}`synthetic-cellular-scene` (W-394, corrected by W-480, amended by W-481)."
   *Proposed:* the same, followed by "The appearance presets of
   {doc}`synthetic-real-appearance` are validated by
   {doc}`synthetic-real-appearance-validation`, which amends this page as its section
   'Accepted statements this page changes' states."
2. **"The claim boundary", "Development targets, not a D04 calibration".**
   *Old:* "The scene's parameters and the target ranges of the generator checks come from
   W-393's measurements of method labels on the densest tiles of three samples (W-393
   `notes.md` §12)."
   *Proposed:* the same, followed by "The density and background targets of the appearance
   presets come from W-513's measurements of the visible stain and of the background at
   label rims; they are development targets too."
3. **"The claim boundary", "No annotation"** (W-513 entry 35).
   *Old:* "The real labels W-393 measured are method output, read as measurements of shape
   and spacing, and nothing here is or replaces an annotation."
   *Proposed:* the same, followed by "In the appearance presets the truth masks are
   themselves method labels of real images. A chain check reads agreement with them as a
   check of the chain and of the method's consistency with its own kind of labels, never as
   accuracy ({doc}`synthetic-real-appearance-validation`, 'The claim boundary')."
4. **"Conventions", "Tiers and markers", the generator checks' markers.**
   *Old:* "the generator checks are `synthetic` (the package whose behaviour they assert),
   kind `validation`, `contract` for configuration and interface checks, `golden` for
   pinned outputs"
   *Proposed:* the same, followed by "; the appearance checks GA1 to GA24 and GA26 to GA27 carry the same
   markers, and those that read a bank built from real images are `slow`".
5. **"Conventions", "Thresholds".**
   *Old:* "each cites the W-393 table and row it comes from ("Thresholds and where they come
   from"), and **the gate W-396 approves each one** before an implementation issue asserts
   it."
   *Proposed:* the same, followed by "On the appearance presets the same thresholds and the
   target ranges of {doc}`synthetic-real-appearance-validation` are provisional again, and
   the gate W-515 approves each one for those presets."
6. **"Conventions", "Seeds".**
   *Old:* "Every scene uses the preset seed 42, with one exception fixed in advance: the
   culture chain check CU runs `context-culture-slow-v1` at the two seeds 42 and 43"
   *Proposed:* "Every scene uses the preset seed 42, with one exception fixed in advance: the
   culture chain check CU runs `appearance-culture-slow-v1` at the two seeds 42 and 43",
   with the rest of the item unchanged.
7. **"Conventions", "Fixtures"** (W-513 entry 36).
   *Old:* "**Fixtures** are generated in the test session from the values stated here; no test
   reads the network share, a run directory or a real image, and no generated image is
   committed."
   *Proposed:* "Fixtures are generated in the test session from the values stated here, and
   no generated image is committed. No routine test and no test in the CI environment reads
   the network share, a run directory or a real image. A `slow` test of an appearance preset
   reads its bank file, derived from real images, by the path of
   `STARFINDER_APPEARANCE_BANKS`, and skips, naming the variable or the file, without it."
8. **"Conventions", "Extras".**
   *Old:* "so the CI environment (extras `dev`, `local-registration`, `checkpoint`) runs
   every generator check and skips every chain check, and fails none."
   *Proposed:* "so the CI environment (extras `dev`, `local-registration`, `checkpoint`)
   runs every generator check except those that read a bank built from real images, which
   skip there naming `STARFINDER_APPEARANCE_BANKS`, skips every chain check, and fails
   none."
9. **"Fixtures", the rows `chain_dense3d`, `chain_tissue2d` and `chain_culture`, the scene
   each is built on.**
   *Old:* "`context_slow` (dense 3D) written by `save_context_dataset`"; "`context_slow` (2D
   tissue, at 0.18927 µm in Y and X) written by `save_context_dataset`";
   "`context-culture-slow-v1` (at 0.388796 µm in Y and X) generated at the two seeds 42 and
   43"; and "`chain_dense3d_routine`: the same on `context-dense3d-v1`",
   "`chain_tissue2d_routine`: the same on `context-tissue2d-v1`", "`chain_culture_routine`:
   the same on `context-culture-v1` at seed 42 only".
   *Proposed:* "`appearance-dense3d-slow-v1` on the bank `ln-sample1-v1`, written by
   `save_context_dataset` of `scene.context`"; "`appearance-tissue2d-slow-v1` (at 0.18927 µm
   in Y and X) on the bank `tissue2d-v1`, written by `save_context_dataset` of
   `scene.context`"; "`appearance-culture-slow-v1` (at 0.388796 µm in Y and X) on the bank
   `culture3d-v1`, generated at the two seeds 42 and 43"; and "`chain_dense3d_routine`: the
   same on `appearance-dense3d-v1` and its closed-form bank", "`chain_tissue2d_routine`: the
   same on `appearance-tissue2d-v1` and its closed-form bank", "`chain_culture_routine`: the
   same on `appearance-culture-v1` and its closed-form bank at seed 42 only". Every other
   part of the three rows unchanged.
10. **"Fixtures", the row `chain_*_truth`, its truth on the segmentation grid.**
    *Old:* "the truth nuclei and cells on the segmentation grid imported as `truth_import`
    and `truth_runs` of `test/sample_segmentation_fixtures.py` build them, with the truth
    of choice V3."
    *Proposed:* the same, followed by "In the appearance presets the truth on the
    segmentation grid is evaluated by the point evaluator of
    {doc}`synthetic-real-appearance` ('Truth')."
11. **"Two test helpers the implementation adds".**
    *Old:* "(1) A *test oracle* in a helper module of `src/python/test/`, written from the
    formulas of {doc}`synthetic-cellular-scene` and the truth tables alone ... (2) A *point
    evaluator* in the generator"
    *Proposed:* the same, followed by "The appearance scene adds its own oracle
    (`cellular_appearance_oracle.py`) and point evaluator, written from
    {doc}`synthetic-real-appearance` ({doc}`synthetic-real-appearance-validation`,
    'Conventions')."
12. **"Generator checks", the preamble.**
    *Old:* "Routine tier unless the row says otherwise (see "Conventions" for the 5-s rule).
    "Source" names the statement of {doc}`synthetic-cellular-scene` (by section) or of
    another page that gives the expected result."
    *Proposed:* the same, followed by "These rows check the closed-form scene and its six
    presets. A scene of real appearance is checked by GA1 to GA24 and GA26 to GA27 of
    {doc}`synthetic-real-appearance-validation`, whose table 'How the cellular checks apply'
    says which of these rows applies to it."
13. **Row GN13 (the stain identities), "Expected result".**
    *Old:* "Identity 7: equal at any fractional offset and at `Z=1`, except at boundary
    voxels (the cut-off surfaces included); on the plateau the image less the background is
    `A_n U(σ) (1 + T_n(σ)) · E_n`, ..."
    *Proposed:* the same, followed by "These stains are those of the closed-form presets. The
    stains of an appearance scene are checked by the composition identity (GA12, GA13 of
    {doc}`synthetic-real-appearance-validation`)."
14. **Row GN14 (the band), "Expected result".**
    *Old:* "`β_k = 0` on every boundary of `C = k` inside `Ω`, ..."
    *Proposed:* the same, followed by "No appearance preset uses the boundary source (O2)."
15. **Row GN15 (bright regions), "Expected result".**
    *Old:* "`N(c_b) = n` exactly for its nucleus `n`; sizes and heights positive; regions
    only in a `nucleus` stain with texture (identity 9, Tx1)."
    *Proposed:* the same, followed by "An appearance scene draws no bright region, and its
    `bright_regions.csv` is empty (GA21); its texture is compared by TR5 and TR7 in GA23."
16. **Row GN16 (the unevenness field), "Expected result".**
    *Old:* "U1: `U` is the recorded bumps' normalized log-Gaussian sum, the same at a sample
    position in every FOV that images it; with no centre `U = 1`"
    *Proposed:* the same, followed by "An appearance scene draws no field, and its
    `unevenness.csv` is empty (GA21)."
17. **Row GN17 (8-bit output), "Expected result".**
    *Old:* "Identity 10; `k` exists and equals `g / m_A` for a FOV or a scene without a
    nucleus (B1)"
    *Proposed:* the same, followed by "In an appearance scene the cast is
    `clip(rint(g · J), 0, 255)` with `g` the bank's median plateau (GA15)."
18. **Row GN23, "Expected result".**
    *Old:* "Inside the target range of each quantity, the p5–p95 of the W-393 row named in
    "The four cases, their controls and their ranges" and in each preset table (PS1)"
    *Proposed:* the same, followed by "The appearance presets are compared by GA23 with the
    target definitions of {doc}`synthetic-real-appearance-validation`."
19. **"Implemented checks", the finding "GN23, TR7 of dense 3D tissue".**
    *Old:* "This is reported and not asserted (the amendment of 2026-10-15 after the TR
    table)."
    *Proposed:* "This is reported and not asserted on `context-dense3d-slow-v1` (the
    amendment of 2026-10-15 after the TR table). On `appearance-dense3d-slow-v1` GA23
    asserts it."
20. **"Chain checks", the preamble.**
    *Old:* "The thresholds of steps 1 and 2 apply to the `slow` preset of each context (for
    the culture, to its two scenes pooled, "Seeds"); the routine preset of each context runs
    the same chain with steps 3 and 4 asserted and steps 1 and 2 reported (choice V1)."
    *Proposed:* "The thresholds of steps 1 and 2 apply to the appearance `slow` preset of
    each context ({doc}`synthetic-real-appearance`, PR1), whose bank is built from real
    images (for the culture, to its two scenes pooled, "Seeds"). The appearance routine
    preset of each context, on its closed-form bank, runs the same chain with steps 3 and 4
    asserted and steps 1 and 2 reported (choice V1). Each check adds a step 5, reported
    ({doc}`synthetic-real-appearance-validation`, 'Consequences for the chain checks')."
21. **"Chain checks", the tables CD, CT and CU.**
    *Old:* the rows CD0 to CD4, CT0 to CT4 and CU0 to CU4.
    *Proposed:* the same rows, each followed by the row CD5, CT5 or CU5 of
    {doc}`synthetic-real-appearance-validation`.
22. **"How many nuclei the thresholds count"** (W-513 entry 42).
    *Old:* "By audit A of {doc}`synthetic-cellular-scene` (W-394; re-run by W-481 for the
    two presets whose pixel changed), a `slow` scene holds about 150 nuclei in dense 3D
    tissue, about 76 in 2D tissue ... and about 42 in the culture ...; the culture chain
    check pools its two scenes, about 83 nuclei. A routine scene holds about 7, 11 and 8."
    *Proposed:* the same for the closed-form presets, followed by "The appearance `slow`
    presets, on which the chain checks run, hold about 738, 337 and 32 objects, about 63
    for the culture's two scenes pooled, and the routine appearance presets about 24, 45 and
    5: estimates at the prototype's reached density ({doc}`synthetic-real-appearance`,
    'Presets'), not counts."
23. **"Thresholds and where they come from", the first paragraph.**
    *Old:* "Every threshold below is **provisional**: it was proposed by W-393 from runs of
    the §2.9 methods on the *current* sample scene, it is approved or changed by Jiahao at
    the gate W-396, and only then does an implementation issue assert it."
    *Proposed:* the same, followed by "For the appearance presets each is provisional again
    and approved at the gate W-515; W-393 measured it on neither scene."
24. **Row TH1, "Applies to".**
    *Old:* "about 150, 76 and 83 nuclei expected before the border rule"
    *Proposed:* "about 738, 337 and 63 objects expected before the border rule in the
    appearance presets (estimates; 'How many nuclei the thresholds count')".
25. **Row TH-FP, "Why no threshold".**
    *Old:* "W-393 withdrew its first bound of 20 %: three rows exceed it, from the 8-bit clip
    and the wide edge, which the new presets change"
    *Proposed:* the same, followed by "; in the culture appearance presets a source cell can
    carry the unlabelled DAPI of a second nucleus, a further cause of false positives".
26. **"Target ranges of the generator checks (TR)", the first sentence.**
    *Old:* "GN23 compares the scene with the target ranges of {doc}`synthetic-cellular-scene`
    ("The four cases, their controls and their ranges" and the preset tables)."
    *Proposed:* the same, followed by "GA23 compares the appearance presets with the rows as
    {doc}`synthetic-real-appearance-validation` defines them ('Target definitions'): TR1,
    TR2 and TR11 reported on the truth and compared like for like in step 5 of the chain
    checks, TR2v, TR8r and TR8v added, TR3 reported beside the source entries."
27. **Row TR1** (W-513 entry 37).
    *Old:* "TR1 | Share of touching nuclei, per image | 0.44–0.57; 0.76–0.82; 0.33–0.72 |
    `geometry.csv` "share of touching nuclei, per image""
    *Proposed:* the same for the closed-form presets; for the appearance presets "reported on
    the truth labels, not compared: the truth is placed at the visible stain's density
    (round 10: 0.99 against 0.56 on the held-out LN crop); compared like for like, on the
    method's labels of the scene against W-393's range on the same method's labels of the
    real images, in step 5 of the chain checks, reported".
28. **Row TR2** (W-513 entry 38).
    *Old:* "TR2 | Nucleus volume (area) fraction, per image | 0.07–0.16; 0.30–0.44;
    0.17–0.45"
    *Proposed:* the same for the closed-form presets; for the appearance presets "TR2v |
    Visible-stain share, per scene (stain smoothed by one voxel above its Otsu threshold) |
    0.47–0.68; 0.29–0.44; DAPI Z maximum 0.12–0.14 and Flamingo Z maximum 0.41–0.53 | W-513
    `tables/visible_share_<context>.csv`, the rows of
    {doc}`synthetic-real-appearance-validation`", with the truth label share reported beside
    it and compared like for like in step 5 of the chain checks.
29. **Row TR3** (W-513 entry 39).
    *Old:* "TR3 | Share not described by one ellipsoid (IoU < 0.8), per image |
    0.064–0.074; 0.027–0.060; 0.12–0.43"
    *Proposed:* the same for the closed-form presets; for the appearance presets "reported,
    not compared: the share on the placed truth labels beside the share on the entries they
    come from and their difference, whose bound is not measured; the real range reported
    with the squash threshold of rule 4, its control (limit L2; round 10: 0.009 against
    0.047)".
30. **Row TR8** (W-513 entry 40).
    *Old:* "TR8 | Background over plateau, per image; noise over plateau, per 128-px block |
    0.01–0.34, 0–0.063; 0.022–0.031, 0–0.033; 0.002–0.047, 0–0.24"
    *Proposed:* the same for the closed-form presets; for the appearance presets the
    background is reported, and two rows are added. "TR8r | Background at label rims, median
    over the truth labels (cells in the culture, on the volume), over the plateau |
    0.33–0.40; 0.125–0.130; 0 | W-513 `tables/rim_background_<context>.csv`, column
    `local_bg_over_plateau_median`". "TR8v | Median stain outside the visible stain, over
    the plateau (the culture on the Z maximum) | 0–0.54; 0.022–0.047; DAPI 0.036–0.038 and
    Flamingo 0.022–0.074, reported | W-513 `tables/visible_share_<context>.csv`, column
    `not_visible_over_plateau`, the rows of {doc}`synthetic-real-appearance-validation`".
    The noise row unchanged.
31. **Row TR11.**
    *Old:* "TR11 | Cell volume (expanded nuclei, dense 3D) | 90–185 µm³ | `geometry.csv`
    "cell volume""
    *Proposed:* the same for the closed-form presets; for the appearance presets "reported on
    the truth; compared like for like on the method's nuclei expanded by 0.7776 µm in step 5
    of CD, reported".
32. **The paragraph "Amendment of 2026-10-15 (W-485; operator decision pending Jiahao's
    confirmation)"** (W-513 entry 41, AM1).
    *Old:* "TR7 of dense 3D tissue is therefore a known limit of the scene: GN23 reports it
    with its measured value, range and object count and does not assert it. A texture
    control that reaches it is a candidate for a later issue."
    *Proposed:* "TR7 of dense 3D tissue is therefore a known limit of the closed-form
    presets: GN23 reports it with its measured value, range and object count and does not
    assert it. The appearance presets of {doc}`synthetic-real-appearance` are the texture
    control: round 10 of the prototype gave 0.317 against the real 0.348 on the held-out
    crop, and GA23 of {doc}`synthetic-real-appearance-validation` asserts TR7 of
    `appearance-dense3d-slow-v1` against 0.25–0.41."
33. **The paragraph after the amendment.**
    *Old:* "The tissue-2D ranges come from its densest tiles and every real label is method
    output (W-393 `notes.md` §12)"
    *Proposed:* the same, followed by "; the density and background ranges of the appearance
    presets come from W-513's per-image measurements of three or four images per context,
    each range the smallest to the largest value".
34. **"Execution conditions", "Which gate runs what", the first two items.**
    *Old:* "The routine test gate (`not slow and not learned`, every issue) runs the routine
    generator checks: GN1 to GN22 on the tiny fixtures. They need no extra beyond
    the base, so the CI environment runs them too." and "The controller's `slow` selections
    run the `slow` generator checks (GN1(a) and (b) on `sample-small-v1`, the preset parts
    of GN4 to GN7, GN23, GN24) ..."
    *Proposed:* the same, followed by "The routine gate and the CI environment also run GA1
    (routine presets) to GA22, on closed-form banks. The `slow` selection also runs
    GA23, GA24, GA26 and GA27, which read the real banks through
    `STARFINDER_APPEARANCE_BANKS` on the lab machine and skip elsewhere
    ({doc}`synthetic-real-appearance-validation`, 'Execution conditions')."
35. **"Execution conditions", "Bounds".**
    *Old:* "The `slow` presets need the resource bound of C26 (RG1: one CPU, 16 GiB and
    3,600 s per step)."
    *Proposed:* the same, followed by "So do the appearance `slow` presets, their bank
    builds and their chains; their estimated sums are about 1,200 s (CD), 1,955 s (CT) and
    1,805 s (CU, two scenes) ({doc}`synthetic-real-appearance-validation`, 'Bounds')."
36. **"Coverage of the cellular scene's page", the first sentence.**
    *Old:* "Every identity and guarantee of {doc}`synthetic-cellular-scene`, by section, with
    the check that tests it or the reason it has none."
    *Proposed:* the same, followed by "Those of {doc}`synthetic-real-appearance` are covered
    in {doc}`synthetic-real-appearance-validation` ('Coverage of the appearance page')."
37. **"Implementation order".**
    *Old:* "Six draft implementation issues ... I1 the scene ...; I2 the stains and the
    morphology round; I3 the presets; I4 the generator checks; I5 the chain checks; I6 the
    documentation with a short example."
    *Proposed:* the same, followed by "I1 to I4 are implemented (W-483 to W-486). I5 (W-487)
    and I6 (W-488) are amended, and the appearance scene is implemented by the drafts A1 to
    A5 of W-514 before them ({doc}`synthetic-real-appearance-validation`, 'Implementation
    order')."
38. **"Exclusions", "No real data"** (W-513 entry 43).
    *Old:* "**No real data, no new dependency, no lock change, no marker registration.**"
    *Proposed:* "**No real data in the repository, the routine tier or the CI environment;
    no new dependency, no lock change, no marker registration.**"

## Choices for the gate

Each has its options and one recommendation in the W-514 worker notes; the design above
follows the recommendation.

| # | Choice | Recommendation |
| --- | --- | --- |
| VA1 | Where the chain checks run | On the appearance presets: the `slow` presets on real banks, the routine presets on closed-form banks. CD, CT and CU are amended, not replaced, and not kept on the closed-form presets |
| VA2 | The density target | TR2v, the visible-stain share, with the label share reported beside it; the range the smallest to the largest per-image value of W-513's rows |
| VA3 | The background target | TR8r (the rim background) asserted; TR8v (outside the visible stain) reported, because LN tiles with empty regions give 0; a block-wise TR8v later, once measured |
| VA4 | TR3, TR4, TR5 and TR7 | TR3 reported beside the source entries and the real range, its bound missing until measured; TR4, TR5 and TR7 against the real ranges; the entries' TR7 ratio reported |
| VA5 | TR1, TR2 and TR11 on the method's labels | Reported in a new step 5 of each chain check, not asserted, until a first measurement |
| VA6 | The tolerance of the composition identity | 1e-9 of the image maximum for the identity and the integer shift, derived, and evaluated on the fixtures by A3 before it is asserted |
| VA7 | Where the bank's SHA-256 is pinned | In the preset in the repository (ST1), with the bank kept outside |
| VA8 | How the controller's `slow` selection finds the banks | `STARFINDER_APPEARANCE_BANKS` set by the operator in the controller's check environment on the lab machine; the issue that adds a real-bank check shows it passed, not skipped |
| VA9 | The culture chain check's seeds | 42 and 43, as accepted (about 63 cells pooled, estimate) |
| VA10 | Golden pins of the six accepted context presets | Added by A1 at the base revision before any change (GA1 (b)) |

## Implementation order

Five draft issues implement the appearance scene and its checks, and the drafts of W-487 and
W-488 amend those two issues. All are files of the W-514 run directory
(`runs/W-514/20261016T223616Z-d34e52d0/drafts/`), in the runner's issue-contract form. None
is created in Linear.

* **A1, the bank:** entries, selection, brightness, storage and the build step, the
  closed-form bank, and the golden pins of the accepted presets.
* **A2, augmentation, layout and labels:** transforms, placement, density, territories,
  ownership and the point evaluator.
* **A3, composition and truth:** composition, output, the morphology round, the truth
  schema and the generation entry.
* **A4, the presets and the real banks:** the six presets, the three real banks built
  outside the repository, cost and the bank checks.
* **A5, the generator and preset checks:** GA23, the coverage, and the table of
  implemented checks with its test.

The dependencies are:

* A2 depends on A1, and A3 on A1 and A2;
* A4 depends on A1 to A3, and A5 on A1 to A4;
* the amended W-487 depends on A4 and A5;
* the amended W-488 depends on A1 to A5 and the amended W-487.

The work of W-483 to W-486 stays as it is. A1 to A5 add a scene beside it and replace none
of it.

## Exclusions

* **No implementation and no test code**: this page names checks; the drafts write them.
* **No new measurement**: every number is W-513's, the prototype's, W-393's or W-485's,
  cited; a number none of them measured is named as missing.
* **No comparison of segmentation methods and no accuracy claim** ("The claim boundary").
* **No real data in the repository, the routine tier or the CI environment; no new
  dependency, no lock change, no marker registration.**
* **Nothing is accepted by this page**, no accepted page is edited, and no issue is created
  in Linear.

## Implementation

The generator checks of this page are in the modules that "Implemented checks" names, by
row. The chain checks of {doc}`synthetic-cellular-scene-validation`, amended as "Consequences for the
chain checks" states, are `test_integration_cellular_chain.py` (W-487). The real-bank
checks GA23, GA24 (its `slow` part), GA26 and GA27 run in the Chapter II evidence project
as `benchmarks/appearance_evidence.py` since W-536, with `STARFINDER_APPEARANCE_BANKS` set
("Implemented checks"); the script records a skip, naming the variable, where it is unset.

**Issues.** The cellular scene was implemented by W-483 (geometry and molecules), W-484
(stains and the morphology round) and W-485 (the six context presets); W-486 completed its
generator checks. The real appearance was implemented by W-516 (the bank), W-517
(augmentation, placement, territories and labels), W-518 (composition, output, the
morphology round and truth), W-519 (the six appearance presets and the three real banks)
and W-520 (the remaining generator checks, GA23 and the table of implemented checks). W-487
ran the chain checks on the appearance presets. W-488 applied the lists of accepted
statements, W-483's four findings and the confirmations of the dated amendments, and added
these sections and the example `docs/examples/context_scene.py` ({doc}`api/examples`), run
by `test_synthetic_context_example.py`.

**Modules** (all in `src/python/starfinder/synthetic/`, exported by `starfinder.synthetic`
and listed on the API page `docs/api/synthetic.rst`):

* `_cellular.py` (W-483): the shape functions of lobed nuclei and harmonics, extents,
  contact distances and the label functions;
* `_context.py` (W-483, W-484): `ContextSceneConfig` and its parts (`NucleusShapeConfig`,
  `PackingConfig`, `TerritoryConfig`, `CellTypeConfig`, `MoleculeConfig`),
  `generate_context_scene`, `ContextScene`, `save_context_scene` and `save_context_dataset`;
* `_stains.py` (W-484): `StainChannelConfig`, `BrightRegionConfig`, `UnevennessConfig`,
  `OutputScaleConfig`, `MorphologyRoundConfig` and the stain functions;
* `_context_presets.py` (W-485): `CONTEXT_PRESETS`, `CONTEXT_PRESET_VERSION` and
  `context_scene_preset`;
* `_appearance_bank.py` (W-516): `BankReference`, `build_appearance_bank`,
  `closed_form_bank` and `load_appearance_bank`;
* `_appearance_layout.py` (W-517): `AugmentationConfig`, `LayoutConfig`, placement by PL2
  with DN1 and DN3, territories, ownership and the point evaluator;
* `_appearance.py` (W-518): `AppearanceSceneConfig`, `CompositionConfig`,
  `generate_appearance_scene`, `AppearanceScene`, `save_appearance_scene` and
  `APPEARANCE_TRUTH_VERSION`;
* `_appearance_presets.py` (W-519): `APPEARANCE_PRESETS`, `APPEARANCE_PRESET_VERSION` and
  `appearance_scene_preset`;
* `_sample.py`: the hooks by which the sample scene's generation runs with the cellular
  scene's and the appearance scene's cells (W-483, W-484, W-518); a scene without the new
  settings is the accepted sample scene.

**Tests** (in `src/python/test/`):

* the generator checks GN1 to GN24 of {doc}`synthetic-cellular-scene-validation`: `test_synthetic_cellular_contract.py`,
  `test_synthetic_cellular_scene.py` (W-483), `test_synthetic_cellular_stains.py`,
  `test_synthetic_cellular_stains_contract.py` (W-484), `test_synthetic_cellular_presets.py`,
  `test_synthetic_cellular_presets_contract.py` (W-485), `test_synthetic_cellular_routine.py`
  and `test_synthetic_cellular_checks.py` (W-486), with the helpers `cellular_oracle.py`,
  `cellular_scenes.py`, `cellular_measures.py` and `cellular_checks.py`; each row's tests are
  named in that page's "Implemented checks"; GN23 and the `slow` and acquired cases of GN24
  run in the Chapter II evidence project as `benchmarks/cellular_evidence.py` (W-536);
* the generator checks GA1 to GA24 and GA26 to GA27 of {doc}`synthetic-real-appearance-validation`:
  `test_synthetic_appearance_bank.py`, `test_synthetic_appearance_golden.py` (W-516),
  `test_synthetic_appearance_scene.py` (W-517), `test_synthetic_appearance_stains.py`,
  `test_synthetic_appearance_contract.py` (W-518), `test_synthetic_appearance_presets.py`,
  `test_synthetic_appearance_presets_contract.py` (W-519, GA23 added by W-520),
  `test_synthetic_appearance_coverage.py` and `test_synthetic_appearance_checks.py` (W-520),
  with the helpers `cellular_appearance_oracle.py`, `cellular_appearance_scenes.py` and
  `cellular_appearance_measures.py`; each row's tests are named in that page's "Implemented
  checks"; GA23, GA26, GA27 and the `slow` part of GA24 run in the Chapter II evidence
  project as `benchmarks/appearance_evidence.py` (W-536);
* the chain checks CD, CT and CU, steps 0 to 5, on the appearance presets:
  `test_integration_cellular_chain.py` (W-487), marked `integration` and `validation`; the
  chains with a model are `slow` and `learned`, the truth-label chains `slow`;
* the example's test `test_synthetic_context_example.py` (W-488), `slow`.

**Amendments during the implementation**, each with its date, its issue and its reason;
each is written where it applies:

* **2026-10-15, W-485** ({doc}`synthetic-cellular-scene-validation`, after the table of target
  ranges). TR7 of dense 3D tissue is reported, not asserted, on `context-dense3d-slow-v1`:
  the bright regions of Tx1, within TR5's ranges, do not reach the real within-nucleus
  variation (0.168 against 0.25–0.41). Operator decision, confirmed by Jiahao on 2026-10-16;
  it applies to the closed-form presets only (W-515, AM1).
* **2026-10-17, W-516** ({doc}`synthetic-real-appearance`, after CF1's options table). The
  closed-form bank's sources are rendered with noise and unevenness off: with the field
  kept, BR4 refuses two of the three culture sources and no closed-form culture bank can be
  built. Operator decision, confirmed by Jiahao on 2026-10-18.
* **2026-10-17, W-516** ({doc}`synthetic-real-appearance-validation`, GA6's row). The stored
  field outside a mask is compared within `max(1e-6 · r, 1e-9)` of the value over the
  plateau: an entry whose rim level `r` is 0 still carries the blurred inside excess
  outside its mask, and float32 storage cannot meet a bound of 0. Operator decision,
  confirmed by Jiahao on 2026-10-18.
* **2026-10-17, W-519** ({doc}`synthetic-real-appearance`, "Presets"). `appearance-culture-slow-v1`
  stops after 500 refused proposals in a row: the culture's cell masks saturate below DN1's
  target of 0.501, and a stop after 4,000 took 27 min 41 s per generation. At seed 42 it
  places 65 cells and reaches 0.480. Operator decision, confirmed by Jiahao on 2026-10-18.
* **2026-10-18, W-520** ({doc}`synthetic-real-appearance-validation`, after "Target
  definitions"). TR2v of dense 3D (0.385) and TR8r of dense 3D (0.238) and of 2D tissue
  (0.034) are reported beside their ranges, not asserted: DN1 places with a fixed threshold
  while TR2v takes the scene's own Otsu threshold, and the real rim level holds unlabelled
  stain that no entry carries. Operator decision, confirmed by Jiahao on 2026-10-18.
* **2026-10-18, W-487** ({doc}`synthetic-real-appearance-validation`, "Consequences for the
  chain checks"). CD1's TH1 on `appearance-dense3d-slow-v1` (0.570 against at least 0.9)
  and CU1's TH3 on `appearance-culture-slow-v1` (2 splits against 0, the seeds 42 and 43
  pooled) are reported beside their thresholds, not asserted: the dense 3D scene holds
  about three to six times the label density of the method's labels on real LN, and the two
  culture splits come from a mask with a detached piece and a two-lobed mask at the FOV
  border. Every other threshold stays asserted. Operator decision, confirmed by Jiahao on
  2026-10-18.
* **2026-10-18, W-488** ({doc}`synthetic-cellular-scene-validation`). W-483's four findings are
  applied with the options its notes recommend: the fixture `ctx_crowded` at `density=0.9`,
  so that it raises as the page says; GN6's anchors strict in the label function's
  arithmetic, as K1 states; the unreachable monolayer refusal dropped from GN3 and kept in
  the generator as a guard; and GN9 stating that `ctx_types` holds three sample cells. The
  rows as written could not be tested as written (W-483 notes, "Findings for Jiahao").
* **2026-10-18, W-488** (`docs/examples/context_scene.py`, {doc}`api/examples`). The example
  keeps the two presets the issue names, `context-tissue2d-v1` and `appearance-tissue2d-v1`,
  and its test `test_synthetic_context_example.py` is `slow` instead of routine: generating
  and saving the two presets takes more than 5 s on one CPU, the bound of a routine test,
  and that time is the package's. Measured on one CPU: the example 9.25 s, its test 8.57 s
  in the `pytest-slow-synthetic` selection and 8.55 s in the CI environment. The test runs
  in `pytest-slow-synthetic` and in the CI environment. Operator decision, confirmed by Jiahao on
  2026-10-18.

No threshold, tolerance, target range, preset or bank was changed by W-488.

* **2026-10-19, GA1** (`test_synthetic_appearance_golden.py`, `test_synthetic_sample_golden.py`).
  The golden digests of the saved presets compare the floating-point values of the CSV files
  and JSON records to 6 significant digits, magnitudes below 1e-9 as 0; integers, strings and
  pixel arrays stay exact. NumPy's vectorised functions differ in the last bit between CPUs:
  on GitHub runners the records of `context-culture-v1` and the CSV files of the two sample
  presets differed from the lab machine's in that bit only, and with NumPy's AVX-512 paths
  turned off on the lab machine the CSV files of every preset differed too. The pins were
  taken again at `8606450`, whose saved files equal the base revisions' exactly on the lab
  machine, and they hold on both CPU paths. Jiahao's decision,
  2026-10-19.

**Accepted statements applied (W-488, 2026-10-18).** The fourteen entries of
{doc}`synthetic-cellular-scene`'s "Accepted statements this addition changes" are applied to
{doc}`synthetic-sample-scene`, with the completions of entry 32 of {doc}`synthetic-real-appearance`'s list; entries
1 to 34 of {doc}`synthetic-real-appearance`'s "Accepted statements this page changes" are
applied to {doc}`synthetic-cellular-scene`; and entries 1 to 38 of {doc}`synthetic-real-appearance-validation`'s list,
which take the place of entries 35 to 43 of {doc}`synthetic-real-appearance`, are applied to
{doc}`synthetic-cellular-scene-validation`. Each is applied with its proposed wording; the lists
stay on their pages as the record of what changed.

**Questions Jiahao closed on 2026-10-18.** Whether DN1 should place with the scene's own Otsu
threshold, the threshold TR2v measures with, and whether TR8r gets a control: both stay known
limits, reported beside their ranges, as CD1's TH1 of W-487 is. The scene serves to check the
pipeline, not to measure segmentation accuracy. Better source labels (the source FOVs
segmented again with better StarDist settings, and the banks rebuilt) would move these rows
without a change of the generator. No preset, target or check was changed for them.
