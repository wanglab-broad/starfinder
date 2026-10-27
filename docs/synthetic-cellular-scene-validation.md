# Validation design of the synthetic cellular scene: generator checks and chain checks

Status: Accepted (W-396, 2026-10-15, at 5701cc2)
the coarser `slow` presets amended by W-482 for the decisions of 2026-10-15; for review at the human gate W-396)

This page is the validation design of the second delivery of §2.12, the cellular scene of
{doc}`synthetic-cellular-scene` (W-394, corrected by W-480, amended by W-481). It has two parts. The
**generator checks** test that the generator makes what that page specifies: every identity
and guarantee it states, the realized statistics against the configured laws, and that the
existing sample scene stays as it is. The **chain checks** run the real §2.9 segmentation
methods on one scene of each context (two of the culture, at the seeds 42 and 43), through
the real route of that context, and compare
the nuclei, the cell of each molecule, the assembly and the §2.11 report with the scene's
truth. It implements D1 and D7 of the W-152 comment "§2.12 planning decisions (2026-10-08)".
It builds on {doc}`synthetic-cellular-scene`, {doc}`synthetic-sample-scene`,
{doc}`stitching-assembly-validation`, {doc}`sample-segmentation-validation` and the
validation design of {doc}`report-algorithms`, and redefines nothing in them: every rule,
identity, column and status a check names is used as those pages state it. The appearance
presets of {doc}`synthetic-real-appearance` are validated by
{doc}`synthetic-real-appearance-validation`, whose amendments of this page are applied
here ("Implementation").

It is a page beside {doc}`synthetic-cellular-scene` rather than a section of it, as the
validation designs of §2.10 ({doc}`stitching-assembly-validation`) and of the route after
fusion ({doc}`sample-segmentation-validation`) are pages beside their contracts: the gate
reviews the scene and its validation as two decisions, and an amendment of a threshold
after acceptance is recorded here without touching the scene's page.

**Sources.** Statements about existing behaviour cite the file and function under
`src/python/` at the batch's starting revision 943bb6b. Measurements cite the W-393 run
directory `runs/W-393/20261015T003630Z-3fec2493/` under the artifact root
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/`, as "W-393" followed by a
file and a row, in the form of {doc}`synthetic-cellular-scene` ("Sources"): W-393
`routes.csv` is `results/tables/routes.csv`, W-393 `sizing.csv` is
`results/tables/sizing.csv`, W-393 `section10_values.txt` is
`results/verification/section10_values.txt`, and W-393 `notes.md` §n is a section of that
run's notes. A row of `routes.csv` is named `<scene>/<route>` (for example
`edge/spleen`), a row of `sizing.csv` by its `tag`. Where this page needs a number W-393
did not measure, it says so and gives none.

## The claim boundary

* **An engineering check of the chain.** The chain checks establish that the stain, the
  segmentation call, the assignment, the assembly and the report are wired to each other
  and to the scene's truth as the accepted pages state, on one synthetic scene per context
  (two for the culture).
  A threshold of a chain check is a bound on that wiring, not a measure of a method.
* **No accuracy of any segmentation method on real tissue.** No number of this page says
  how well StarDist `3D_spleen`, StarDist `2D_versatile_fluo` with the seeded watershed, or
  Cellpose `cpsam_v2` segments LN, tissue-2D or culture images. The three methods run on
  different inputs by design (D2), so no number here compares them, and none ranks them;
  comparisons belong to the evidence project (§2.14).
* **Development targets, not a D04 calibration.** The scene's parameters and the target
  ranges of the generator checks come from W-393's measurements of method labels on the
  densest tiles of three samples (W-393 `notes.md` §12). They are targets for a development
  fixture. They are not a calibration of the synthetic model to tissue, and passing them
  does not make the scene one. The density and background targets of the appearance
  presets come from W-513's measurements of the visible stain and of the background at
  label rims; they are development targets too.
* **No annotation.** The real labels W-393 measured are method output, read as
  measurements of shape and spacing, and nothing here is or replaces an annotation.
  Annotation stays with W-123. In the appearance presets the truth masks are themselves
  method labels of real images. A chain check reads agreement with them as a check of the
  chain and of the method's consistency with its own kind of labels, never as accuracy
  ({doc}`synthetic-real-appearance-validation`, 'The claim boundary').
* **W-93 keeps the responsibility for synthetic truth and calibration.** This design adds
  checks of one generator's truth; what a synthetic truth must be and how a synthetic model
  is calibrated stay with W-93.

## Conventions

* **Tiers and markers** are those of {doc}`contributing` ("Test markers"). The *routine
  tier* is the fast tier R, `not (slow or learned or extended)`: its `contract` and
  kind-less checks run in the quick gate Q on every change, its other `validation` checks in
  V at a batch's end and before merge ({doc}`contributing`, "Test markers"). A check
  is `slow` when it takes 5 s or more on one CPU and one thread in the locked
  environment, measured with `--durations=0`; the tier written below
  is the expected one, and a routine check measured at 5 s or more is marked `slow` and
  named in its issue's notes. A check that loads a model is `learned`. Every test file sets
  one subsystem marker and a kind marker: the generator checks are `synthetic` (the
  package whose behaviour they assert), kind `validation`, `contract` for configuration and
  interface checks, `golden` for pinned outputs; the appearance checks GA1 to GA24 and GA26 to GA27 carry
  the same markers, and those that read a bank built from real images are `slow`; the
  chain checks assert the behaviour of
  `synthetic`, `segmentation`, `assignment`, `stitching` and `report` together, so their
  file is `integration`, kind `validation`.
* **Tolerances.** Counts, IDs, strings, statuses, hashes and label images are compared
  exactly. Every other tolerance is of one of three kinds, named in its row:
  * **float rounding of an exact identity**: the two sides are the same formula on the same
    inputs, computed with different arithmetic or at two roundings of one position. A float
    the test computes by the same formula is within an absolute 1e-9 (the project's
    default) unless the row states another bound. A decision the test recomputes with its
    own arithmetic (a power face, a lobe or harmonic surface, a territory limit, a cut-off,
    a region boundary) is compared as identity 2 of {doc}`synthetic-cellular-scene` states:
    positions whose decision quantity lies within 1e-9 µm of 0 are boundary positions,
    counted and reported, and the expected count is 0;
  * **derived from a formula**: a bound that follows from the formulas of
    {doc}`synthetic-cellular-scene` at its worst case (the soft edge's tail, the
    site-skipping rule, the Lipschitz constant of the band argument), stated in the row with
    its derivation;
  * **sampling tolerance**: a realized statistic of a random law is compared with its
    configured value within four standard errors of that law at the realized count
    ("4 SE"). With one seed the outcome is fixed, so a pass is not a probability; a miss at
    a correct implementation would be a draw outside four standard errors (a two-sided
    chance of about 6e-5 per comparison under a normal approximation), which is reported to
    Jiahao and never retried with another seed.

  Every rounding and derived tolerance was evaluated with the page's formulas at its
  fixture's parameters and at the worst case it must cover, by the script
  `scripts/tolerance_check.py` of W-395's run directory (output
  `results/tolerances/tolerance_check.txt`, with one line per tolerance and its kind in
  `results/tolerances/tolerance_checklist.md`), re-run by W-482 with grids of the two
  changed `slow` pixels added (same paths in `runs/W-482/20261015T170720Z-694fab7e/`); its stain functions are the page's full formulas (the field `U`, the
  bright regions with their box cut-off, and the territory source's nuclear factor), and every deviation lies within
  its tolerance.
  The rounding bounds hold on the generator checks' fixtures, whose sample positions lie
  within 31 µm of the origin (the domain of `sample-tiny-v1`); a fixture on a larger domain
  needs them derived again, because the rounding of a power distance grows with its size.
* **Thresholds** of the chain checks and the target ranges of the generator checks are
  **provisional**: each cites the W-393 table and row it comes from ("Thresholds and where
  they come from"), and **the gate W-396 approves each one** before an implementation issue
  asserts it. W-393 measured them on the current scene, not on this one (W-393 `notes.md`
  §10), so the first run of the chain checks is also their first measurement on the new
  scene; what follows a miss is in "What a failure means". On the appearance presets the
  same thresholds and the target ranges of {doc}`synthetic-real-appearance-validation` are
  provisional again, and the gate W-515 approves each one for those presets.
* **Seeds.** Every scene uses the preset seed 42, with one exception fixed in advance: the
  culture chain check CU runs `appearance-culture-slow-v1` at the two seeds 42 and 43 (the
  preset with `AmpliconSceneConfig.seed` replaced by 43 for the second scene; Jiahao's decision
  of 2026-10-15, written on {doc}`synthetic-cellular-scene` by W-481). Its steps 1 and 2
  (nuclei against truth, the cell of each molecule) pool the counts of the two scenes before
  the thresholds TH1 to TH4 are applied; its steps 3 and 4 (the count identities of the
  assembly, the report) are asserted on each scene. The hand-built fixtures use no random
  number. No other seed is used, and no check is repeated with another seed to pass.
* **Fixtures** are generated in the test session from the values stated here, and no
  generated image is committed. No routine test and no test in the CI environment reads the
  network share, a run directory or a real image. A `slow` test of an appearance preset
  reads its bank file, derived from real images, by the path of
  `STARFINDER_APPEARANCE_BANKS`, and skips, naming the variable or the file, without it.
* **Extras.** A check that exports needs the `anndata` extra, one that fuses or stitches
  `multiview-stitcher`, one that loads a model `stardist` or `cellpose`. Each skips with a
  stated reason where its extra is not importable, so the CI environment (extras `dev`,
  `local-registration`, `checkpoint`) runs every generator check except those that read a
  bank built from real images, which skip there naming `STARFINDER_APPEARANCE_BANKS`,
  skips every chain check, and fails none.

## Fixtures

| Fixture | Construction | Seeds | Needs |
| --- | --- | --- | --- |
| `sample_presets` | `sample_scene_preset("sample-tiny-v1")` and `sample_scene_preset("sample-small-v1")`, as generated today. | 42 | none |
| `ctx_off` | `CellularSceneConfig(sample=c)` for `c` each of the two sample presets, every new part at its default (None or empty). | 42 | generator (I1) |
| `ctx_shapes` | `sample-tiny-v1` (2 × 2 FOVs of 5 × 36 × 36, fractional J1 jitter in Z, Y and X) with `NucleusShapeConfig(lobed_share=1.0)` and the preset lobe laws; once more with N2 on (`harmonics` 3, `harmonic_amplitude` set so that `harmonic_amplitude · M_Y` is 0.5); `PackingConfig(contact=1.0, overlap=0.2, gap=0.0, kind_shares=(0.6, 0.2, 0.2), on_crowded="stop")`, so that it holds touching nuclei, binucleate cells with unequal radii and empty cells; one copy per territory rule T1, T2 and T3. | 42 | generator (I1) |
| `ctx_monolayer` | `ctx_shapes` with `monolayer=True` (five planes, `z_mid` in the covered region). | 42 | generator (I1) |
| `ctx_plane` | `ctx_shapes` on the single-plane configuration of {doc}`stitching-assembly-validation` (`scene_plane`: FOVs of 1 × 36 × 36, jitter bound (0, 1.5, 1.5) voxels), with the stains and the morphology round of `ctx_stains` (`T_z` 0). | 42 | generator (I1, I2) |
| `ctx_types` | `sample-tiny-v1` with the cell types of C8 (G1, four types, the two marker pairs), `MoleculeConfig(per_cell_median=200, per_cell_spread=0.3, nuclear_fraction=0.5)` and A2 ambient molecules; a copy with type 2's abundance row changed, and one with type 3's molecules per cell changed (stream checks). | 42 | generator (I1) |
| `ctx_crowded` | `sample-tiny-v1` with `PackingConfig(density=0.9, on_crowded="stop", floor=0.9)` (more than the domain holds: raises), and with `floor=0.1` (stops and records). | 42 | generator (I1) |
| `ctx_stains` | `ctx_shapes` (each territory rule) with noise disabled, template dtype `float64` and no output scale: a `nucleus` stain `DAPI` with bright regions and unevenness, a `territory` stain `Flamingo` (`nuclear_ratio` 0.3) with unevenness, a `boundary` stain `membrane` at `band_width` 0.01 and 0.5 µm with unevenness; the bright regions and the unevenness take the culture preset's laws, the presets' largest slopes (`BrightRegionConfig(per_nucleus=2, size` lognormal median 0.17 µm, log SD 0.64, `height` median 2.0, log SD 0.69`)`; `UnevennessConfig(centres_per_area=4, amplitude=0.74)`, length one FOV width); and a morphology round `organelle` holding `DAPI` and `Flamingo`, once with a supplied whole-voxel translation `(0, 2, −3)` voxels for every FOV and once with MT1's drawn fractional translation; a copy whose supplied jitter differs between two FOVs by whole voxels (the worked example of identity 2 of {doc}`synthetic-sample-scene`, moved by `T`), and a copy with no cell (every FOV beyond every cut-off). | 42 | generator (I1, I2) |
| `ctx_uint8` | `ctx_stains` with noise on and `OutputScaleConfig(dtype="uint8", plateau=50)`; and its float64 rendering with the same streams. | 42 | generator (I1, I2) |
| `band_cases` | Truth tables written by hand, as audit D of W-480 builds them (`results/audit/stains_vs_labels.py` in `runs/W-480/20261015T070249Z-868bf70b/`): the binucleate cell with spherical sites of radii 1 and 2 µm at (0, 0, 0) and (0, 0, 2.7), reach 1; binucleate cells with sites of unequal radii; touching cells of different owners; lobed nuclei under T2; the same at a FOV at a fractional offset in Z, Y and X and at `Z=1`. | none | generator point evaluator (I2) |
| `context_routine` | The three routine presets `context-dense3d-v1`, `context-tissue2d-v1`, `context-culture-v1` as generated. | 42 | generator (I3) |
| `context_slow` | The three `slow` presets `context-dense3d-slow-v1`, `context-tissue2d-slow-v1` (at 0.18927 µm in Y and X), `context-culture-slow-v1` (at 0.388796 µm) as generated: about 150, 76 and 42 nuclei per scene (audit A of W-394, re-run by W-481 for the two presets whose pixel is twice the acquired one). | 42 | generator (I3); `slow`; the resource grant of C26 |
| `context_slow_acquired` | The two variants on which GN23 measures the edge width (TR4) and the bright regions (TR5) of 2D tissue and culture at the pixel W-393 measured them: `context-tissue2d-slow-v1` with `spacing_zyx` replaced by (0.346255, 0.094635, 0.094635) µm and `context-culture-slow-v1` with `spacing_zyx` replaced by (0.346255, 0.194398, 0.194398) µm (`dataclasses.replace`; every other value, jitter and bounds in voxels included, as the preset). They are the two presets as they stood before W-481: about 20 and 12 nuclei (audit A of W-394), and an estimated generation below 53 s and 1.4 GiB, and of 70 to 100 s and below 2 GiB (the estimates of {doc}`synthetic-cellular-scene`, "Resources", before W-481; estimates, not measurements). Not presets: they are built in the test and are not in `CELLULAR_PRESETS`. | 42 | generator (I3); `slow`; the resource grant of C26 |
| `chain_dense3d` | `appearance-dense3d-slow-v1` on the bank `ln-sample1-v1`, written by `save_cellular_dataset` of `scene.context`; per FOV the nuclei by `segment(StarDistConfig(model_path=<3D_spleen>, scale=1.0))` on the reference stain `DAPI` and the cells by `assign_molecules` with `ExpandLabelsConfig(0.7776, "um", "planar")` (W-382's LN settings, W-393 `notes.md` §4); the molecules at their truth positions (each FOV's legacy CSV of the no-segmenter path read by `import_legacy_molecules_csv`, choice V2); `Dataset.stitch` with `MultiviewStitcherConfig()` on `DAPI`, approved, its placement used when every FOV is `image_supported` and the truth placement otherwise (`placement_from_table` with the true origins, approved; SC1 of C20); `Dataset.export` with `route: per_fov`, `dimensionality: stack`; `Dataset.report` and `load_report`. `chain_dense3d_routine`: the same on `appearance-dense3d-v1` and its closed-form bank. | 42 | I3; `stardist`, `anndata`, `multiview-stitcher`; `3D_spleen` |
| `chain_tissue2d` | `appearance-tissue2d-slow-v1` (at 0.18927 µm in Y and X) on the bank `tissue2d-v1`, written by `save_cellular_dataset` of `scene.context` with `PI` as the reference stain; per FOV the nuclei–amplicon composite `composite_nuclei_amplicon` of `PI` and the reference round's channel maximum saved as an image (the route has no composite of its own: W-393's workaround, W-393 `notes.md` §4); `Dataset.stitch` on `PI`; `Dataset.export` with `route: after_fusion`, `dimensionality: projection` (Z maximum), a `segmentation` block with the nucleus run StarDist `2D_versatile_fluo` at the scale that gives it about 0.38 µm pixels (0.5 on the `slow` preset's 0.18927 µm grid, 1.0 on the routine preset's 0.37854 µm grid) and the cell run `seeded_watershed` (sigma 1.5 µm, Otsu) on the fused composite seeded by the nuclei, device CPU (W-393 `notes.md` §4); `Dataset.report` and `load_report`. `chain_tissue2d_routine`: the same on `appearance-tissue2d-v1` and its closed-form bank. | 42 | I3; `stardist`, `anndata`, `multiview-stitcher`; `2D_versatile_fluo` |
| `chain_culture` | `appearance-culture-slow-v1` (at 0.388796 µm in Y and X) on the bank `culture3d-v1`, generated at the two seeds 42 and 43 ("Seeds": the preset, and the preset with `AmpliconSceneConfig.seed` 43), each scene written by `save_cellular_dataset` with the reference stain `DAPI` and the morphology round `organelle` (`ch00` `DAPI`, `ch02` `Flamingo`); per FOV `FOV.prepare_morphology(MorphologyConfig())` (MR1: one translation on the shared `DAPI`); the nuclei by Cellpose `cpsam_v2` on the Z maximum of the registered morphology `DAPI`, extended through z with W-382's culture nucleus setting (`median_um` 1.94398, Otsu, `min_area_um2` 0.37791, `dilation_um` 0.97199, holes once; W-393 `notes.md` §4); the cells by Cellpose `cpsam_v2` on the `Flamingo` and `DAPI` projections, extended through z with W-382's culture cell setting (D2); molecules, stitch, export (`route: per_fov`, `dimensionality: stack`) and report as `chain_dense3d`, for each scene. Steps 1 and 2 pool the counts of the two scenes (about 83 nuclei, W-481's re-run of audit A) before TH1 to TH4; steps 3 and 4 are asserted on each scene. `chain_culture_routine`: the same on `appearance-culture-v1` and its closed-form bank at seed 42 only. | 42 and 43 (`slow`); 42 (routine) | I3; `cellpose`, `anndata`, `multiview-stitcher`; `cpsam_v2` |
| `chain_*_truth` | Each chain fixture with the method replaced by the truth labels: on the route per FOV, the FOV truth labels imported with `import_labels(…, relabel=False)` (the no-segmenter path of {doc}`synthetic-sample-scene`, as EE1 to EE3 of {doc}`stitching-assembly-validation` run it); on the route after fusion, the truth nuclei and cells on the segmentation grid imported as `truth_import` and `truth_runs` of `test/sample_segmentation_fixtures.py` build them, with the truth of choice V3. In the appearance presets the truth on the segmentation grid is evaluated by the point evaluator of {doc}`synthetic-real-appearance` ('Truth'). No model is called. `chain_culture_truth` has the two scenes of `chain_culture`. | 42; 42 and 43 for `chain_culture_truth` | I3; `anndata`, `multiview-stitcher`; `slow` |

**What is reused, by name.** From `test/report_scene.py`: `require_extras` (the skip
reasons), the `Chain` record and its `collect` call, and `export_after_fusion` for the
export of the route after fusion; its `build_chain` (W-384's recipe through `FOV.run`) only
under option V2-b. From `test/sample_segmentation_fixtures.py`: `segmentation_grid`,
`truth_import`, `truth_runs`, `seeded_runs` and `scene_plane_config`. From
`test/test_stitching_assembly_scene.py`: the oracle of EE2 (`oracle`, `oracle_owner`) and
the table of EE5 (`ee5_table`), moved to a helper module when a second file imports them.
From `test/test_segmentation_validation.py`: the matching of the §2.9 check L12
(`l12_agreement`: one to one at IoU ≥ 0.5, the larger IoU first). The validation tables of
{doc}`report-algorithms` (DF7, DF8, CH3, RT1, RC12, PG1) and of
{doc}`sample-segmentation-validation` (SA3, SA5, SA7) are the definitions the chain checks'
fourth and third steps apply; they are cited by row.

**Two test helpers the implementation adds.** (1) A *test oracle* in a helper module of
`src/python/test/`, written from the formulas of {doc}`synthetic-cellular-scene` and the
truth tables alone (it imports nothing from `starfinder.synthetic` beyond the table
readers): the nucleus and cell labels `N` and `C`, the three stain signals with their
cut-offs, the bright regions, the field `U` and the projection truth `P`. The generator
checks compare the generator's output with it. (2) A *point evaluator* in the generator
(a private function of `starfinder.synthetic`), which evaluates the scene's label and stain
functions at supplied sample positions; the band check bisects boundaries with it, and the
chain checks evaluate the truth on a segmentation grid with it (choice V3). The oracle and
the evaluator are compared in GN4 and GN13, so neither is checked only by itself. The
appearance scene adds its own oracle (`cellular_appearance_oracle.py`) and point
evaluator, written from {doc}`synthetic-real-appearance`
({doc}`synthetic-real-appearance-validation`, 'Conventions').

## Generator checks

Routine tier unless the row says otherwise (see "Conventions" for the 5-s rule). "Source"
names the statement of {doc}`synthetic-cellular-scene` (by section) or of another page that
gives the expected result. These rows check the closed-form scene and its six presets. A
scene of real appearance is checked by GA1 to GA24 and GA26 to GA27 of
{doc}`synthetic-real-appearance-validation`, whose table 'How the cellular checks apply' says
which of these rows applies to it.

(synthetic-cellular-scene-validation-generator-checks)=

| # | Check | Fixture (seed) | What is computed | Expected result (source) | Tolerance | Tier and markers |
| --- | --- | --- | --- | --- | --- | --- |
| GN1 | An existing preset unchanged; O1 | `sample_presets`, `ctx_off` (42) | (a) The thirteen group digests of everything `save_sample_scene` writes for each sample preset; (b) `generate_cellular_scene` with every new part at its default against `generate_sample_scene`, field for field, and their saved directories file by file. | (a) The pins of `test/test_synthetic_sample_golden.py`, computed at 943bb6b (W-394); (b) identity 1 (O1, "Opt-in"): arrays equal, tables equal with `check_exact`, `record` equal, `sample.json` and every file of the sample layout equal. | Exact (SHA-256 of content, as the golden test defines it; arrays and tables exact); in (a) the floating-point values of the CSV files and JSON records to 6 significant digits, magnitudes below 1e-9 as 0 (amendment of 2026-10-19). | (a) The existing golden test, its float comparison amended on 2026-10-19: routine for `sample-tiny-v1`, `slow` for `sample-small-v1` (4.99 s, W-394 notes §3); `synthetic`, `golden`. (b) routine on `sample-tiny-v1`, `slow` on `sample-small-v1`; `synthetic`, `validation`. |
| GN2 | Names, fields and preset registry | none | The fields of `SampleSceneConfig`, `CellsConfig` and `StainConfig`; the keys of `SAMPLE_PRESETS`; the names exported by `starfinder.synthetic`; `CELLULAR_PRESETS`, `CELLULAR_PRESET_VERSION`, `CELLULAR_TRUTH_VERSION`; the fields of `CellTypeConfig`; each context preset's configuration. | The fields and keys as at 943bb6b, written in the test ("Opt-in"); the new names of "Names" exported and on API pages (`docs/check_reference.py`); six presets, `context-presets-v1`, `"1"`; the per-type fields of `CellTypeConfig` are exactly `molecules_per_cell`, `nuclear_fraction` and `size_scale` ("Cell types"); every preset value equals the PS1 tables of "Presets", and no preset uses the `boundary` source (O2). | Exact. | routine; `synthetic`, `contract`. |
| GN3 | Configuration refusals | none | `ValueError` and its message for: `stains` given with `sample.stain` not None; `harmonic_amplitude · M_Y ≥ 1`; `overlap` outside `[0, 0.5)`; `gap < 0`; `contact` outside `[0, 1]`; `density` outside `(0, 1)`; ordinary and binucleate shares both 0 in density mode; `floor` outside `(0, 1]`; `monolayer` with one plane; a cytoplasm molecule with no room after `max_proposals`; a crowded scene below its floor (`ctx_crowded`). | Each raises `ValueError` naming the field, the seed or the cell, and both densities for the floor ("Names", "Nucleus shape", "Packing", "Molecules"). | Exact (the type; the named values present in the message). | routine; `synthetic`, `contract`. |
| GN4 | Labels against the truth (identity 2), and identities 1 to 5 of {doc}`synthetic-sample-scene` | `ctx_shapes` (each territory rule), `ctx_monolayer`, `ctx_plane`, `ctx_stains` (its fractional and whole-voxel offsets), `context_routine` (42) | `N` and `C` recomputed by the test oracle from `nuclei.csv`, `lobes.csv`, `harmonics.csv`, `cells.csv` and `cell_context.csv` at every voxel centre of every FOV and of the sample grid, and by the generator's point evaluator at the same positions; per grid the voxels that differ; per nucleus the number of connected pieces of its label (26-connected in 3D, 8-connected in a plane); the accepted identities 1 to 5 on `molecules.csv`, `cells.csv`, `cell_fovs.csv` and `molecule_fovs.csv`. | Labels equal the recomputation except at boundary voxels, at any fractional offset and at `Z=1` (identity 2; "Nucleus shape", "Label rule", "Cell territories"); the evaluator equals the oracle; a nucleus whose label has several pieces keeps one ID and is counted once ("A nucleus label with more than one piece"); identities 1 to 5 of {doc}`synthetic-sample-scene` hold with the new label functions ("Identities a test can check"). | Exact; boundary voxels within 1e-9 µm counted and reported, expected 0; the number of multi-piece nuclei reported. | routine on the tiny fixtures; `slow` on `context_routine` if it measures 5 s or more; `synthetic`, `validation`. |
| GN5 | Lobes, harmonics and one connected shape | `ctx_shapes` (with and without N2), `ctx_monolayer`, `ctx_plane`, `context_routine`, `context_slow` (42) | From `lobes.csv`, `nuclei.csv` and `harmonics.csv`: lobe 1's offset and its ellipsoid against `nuclei.csv`; for every lobe `l ≥ 2`, its offset `D_l` against `0.9 · (e_{n,1}(û_l) + ε_l(−û_l))`, the two extents recomputed by the test; `site_radius` against `max_l (‖o_l‖ + max semi-axis)` (with lobe 1's harmonic factor under N2); `Σ abs(a_lm)` against `harmonic_amplitude`; `offset_z` of every lobe in a plane or a monolayer; `D_l / a_l` of every lobe of the monolayer presets. | Lobe 1 has offset 0 and repeats `nuclei.csv`; `D_l ≤ 0.9 · (e + ε)`, so every lobe meets lobe 1 and every drawn nucleus is one connected shape before clipping (LB1; W-480); the site sphere contains every lobe and is today's radius for one lobe and no harmonics ("Bounding site"); `Σ abs(a_lm)` equals the amplitude (N2); lobe offsets lie in the YX plane in a plane or a monolayer; in the monolayer presets the bound does not bind, so `D_l / a_l` lies in the support `[0.5, 0.9]` of `lobe_offset` ("Lobe offsets and one connected shape"). | Exact for counts and zeros; floats within 1e-9 µm (1e-9 for the sum of coefficients). | routine on the tiny fixtures; the presets `slow` where measured so; `synthetic`, `validation`. |
| GN6 | No overlap between nuclei of different cells; a nucleus inside its own cell (identity 3) | `ctx_shapes` (each territory rule; contact 1, overlap 0.2), `ctx_monolayer`, `ctx_plane`, `context_routine` (42) | On every label voxel of every grid: `C` where `N = n > 0`; at every voxel centre where the oracle finds two nuclei's shapes both with `ρ ≤ 1`, the nucleus the label carries; each anchor's own evaluation; each anchor's power distances to every other site. | `N = n > 0` implies `C = cell(n)`, boundary voxels included (identity 3); where two shapes overlap the label is the nucleus of the power-nearest site (ties to the smaller site index) and never the other, so no point carries nuclei of two cells ("Nuclei of different cells do not overlap"); each anchor lies strictly inside its own power cell, in the label function's arithmetic as K1 states, and gives its own cell ("Packing", K1; "Cell territories"). | Exact (labels and IDs); the oracle's decisions (power faces, `ρ ≤ 1`) by float rounding: voxels whose decision quantity lies within 1e-9 µm of 0 reported, expected 0. | routine on the tiny fixtures; `synthetic`, `validation`. |
| GN7 | Packing | `ctx_shapes`, `ctx_monolayer`, `ctx_crowded`, `context_routine`, `context_slow` (42) | For every pair of accepted sites within reach, the K1 condition recomputed by the test from the truth (the contact distance from the two extents, the two power tests); for the two sites of every binucleate cell, the same; the summed nominal size of the placed nuclei against `φ · ‖Ω‖`; the reached density and the placed seeds that `context.json` records; the result of `ctx_crowded` with each floor. | Every pair satisfies `‖c − c_i‖ ≥ (1 − κ) · t_ik(û) + g` and both power tests (K1, P1); placement stops at the target density, or with `on_crowded="stop"` at the first seed that cannot be placed, recorded, and raises below `floor · φ` (F2, "Density"); every preset reaches its floor (audit A of W-394; W-481's re-run for the two `slow` presets whose pixel changed). | Exact for counts and records; the K1 inequality within 1e-9 µm (the test's float64 arithmetic is not the generator's). | routine on the tiny fixtures; `slow` on the presets; `synthetic`, `validation`. |
| GN8 | Cell types and their genes (identity 4) | `ctx_types`, `context_routine` (42) | Rows of `cell_context.csv` per cell; each cell's `region` against the region of its anchor recomputed from `regions.csv`; each cell molecule's gene weight in its cell's type; the genes of ambient molecules; each row of `type_abundances.csv`. | One row per cell; `region` is the Voronoi region of the anchor (ties to the smaller region index, Ty2); every cell molecule's gene has nonzero weight in its type, and with A2 no ambient molecule has a marker gene (identity 4); each abundance row sums to 1 and the presets hold G1's table ("Cell types"). | Exact; anchors within 1e-9 µm of a region boundary reported, expected 0 (rounding of the Voronoi decision); the row sums within 1e-9 (rounding of G1's exact sums). | routine; `synthetic`, `validation`. |
| GN9 | Counts and the marker answers (identities 5 and 6) | `ctx_types`, `context_routine` (42) | `cell_gene_counts.csv` against `molecules.csv` grouped by `true_cell_id` and `gene_id`, its sums against `cells.csv` `n_molecules`, its `nuclear_count` against the molecules of compartment `nucleus`; `marker_pairs.csv` against MECR recomputed from `cell_gene_counts.csv` by the report's definition ({doc}`report-algorithms`, "`mecr.csv`") over the sample cells at `positive_threshold` 1. | Identity 5 and identity 6: equal; the exclusive pair has `both` 0 and `mecr` 0 when `either > 0`; `mecr` null when `either` is 0 ("The known answers"). `ctx_types` holds three sample cells, none of type 1, so there its co-expressed pair has `either` 0 and `mecr` null; the co-expressed answer is exercised by the presets and the chain checks (W-483, finding 4). | Exact (integers; `mecr` as the float64 ratio of the same two integers). | routine; `synthetic`, `validation`. |
| GN10 | The realized gene abundance of each cell type | `ctx_types`, `context_routine` (42) | For each type `t` with `n_t` cell molecules and each gene `g`: the realized share `p̂ = count / n_t` against the configured weight `w_tg`. | `p̂` equals `w_tg` within the sampling tolerance of the categorical draw of `context.identity` ("Cell types", G1; "Random streams"); a marker gene of another type has count exactly 0. | 4 SE: `abs(p̂ − w) ≤ 4 · sqrt(w (1 − w) / n_t)`; zeros exact. | routine; `synthetic`, `validation`. |
| GN11 | Molecules per cell, and the renderer unchanged | `ctx_types`, `context_routine` (42) | `cell_context.csv` `molecule_mean` per cell; `Σ_k n_k` against `Σ_k μ_k` over the cells; the total against `max_count`; with `per_cell_spread` 0, every `molecule_mean`; the FOV amplicon images against the rendering identity of {doc}`synthetic-sample-scene` ("Rendering each FOV at its true offset"). | M2: the count of cell `k` is Poisson with mean `μ_k`, so the summed count equals the summed means within the Poisson tolerance; `max_count` guards the total; with spread 0 every mean is the median (M1) ("Molecules"); the molecules are rendered by the unchanged formed generator ("Every new object is a function of the sample position"). | 4 SE: `abs(Σ n_k − Σ μ_k) ≤ 4 · sqrt(Σ μ_k)`; the rest exact. | routine; `synthetic`, `validation`. |
| GN12 | The realized nuclear fraction of reads | `ctx_types`, `context_routine` (42) | Over the cell molecules of cells with a nucleus: the realized share `f̂` with compartment `nucleus` against `f`; per molecule, `compartment` against `N(σ_m) > 0` and `C(σ_m)` against `true_cell_id`, both recomputed by the oracle; the compartments of the molecules of empty cells. | NF1: each cell molecule is in the nucleus with probability `f`, so `f̂` equals `f` within the binomial tolerance; `compartment` is `nucleus` exactly where the molecule's position holds a nucleus of its cell; every cell molecule lies in its own cell; an empty cell's molecules are all in its territory ("Molecules", C10). | 4 SE: `abs(f̂ − f) ≤ 4 · sqrt(f (1 − f) / N)` with `N` the molecules counted; the rest exact, boundary positions within 1e-9 µm reported, expected 0. | routine; `synthetic`, `validation`. |
| GN13 | The stain plateau against the label; the stain identities (identity 7) | `ctx_stains` (each territory rule, both morphology translations, the whole-voxel pair and the copy with no cell), `ctx_plane` (42) | Each stain image (`nucleus`, `territory`, `boundary` sources) against the oracle's signal function with its background (`evaluate_background_translated` with the FOV's offset) at the voxel centres, and against the point evaluator; the image rendered again at `true_f + k ⊙ s` for an integer `k` against the first shifted by `k`; on the voxels with `N = n` and the erfc argument at most `−4w` (the plateau), the image less the background against `A_n U(σ) (1 + T_n(σ))`; the sign of the nuclear argument against `N` and of `d_k` against `C`; the image of a FOV beyond every cut-off. | Identity 7: equal at any fractional offset and at `Z=1`, except at boundary voxels (the cut-off surfaces included); on the plateau the image less the background is `A_n U(σ) (1 + T_n(σ)) · E_n`, with `E_n = ½ erfc(a / (√2 w))` at an argument `a ≤ −4w`, so `1 − ½ erfc(2√2) ≤ E_n ≤ 1`, and no other nucleus contributes there (its face term is at least `4w`, beyond its cut-off) ("Stains", the nuclear source); `N = n` exactly where the nuclear argument is at most 0 and `C = k` exactly where `d_k ≤ 0` ("Stains", the nuclear and territory sources); a FOV beyond the cut-off of every nucleus holds the background only in the nuclear stain, and the territory and boundary stains are lit by an empty cell; with no cell, every stain is the background. These stains are those of the closed-form presets. The stains of an appearance scene are checked by the composition identity (GA12, GA13 of {doc}`synthetic-real-appearance-validation`). | *Identity against the oracle and the evaluator* (float rounding of an exact identity): 1e-9 of the image maximum. *Integer shift* (float rounding of an exact identity, one point function at two roundings of a position): 1e-10 of the maximum. Both are covered by a derived worst case of the full stains (with `U`, the bright regions `T_n` and, in the territory source, the factor `1 − (1 − q) M`): relative to the image maximum, the most objects contributing at a voxel, times the largest plateau and `U`, times the sum of each factor's error, which is its slope times the rounding of a position (2 ulp per axis) plus the rounding of its own evaluation: the edge or band (slope `1 / (√(2π) w)` or `e^(−½) / b`, times the argument's Lipschitz constant, plus the rounding of a face or radial term), the field (`‖∇ log U‖ ≤ 2 a ‖z‖ D_Ω / L²`, `D_Ω` the diameter of `Ω`, plus 8 ε `a Σ abs(z_q)`), the bright regions (slope `Σ_b h_b e^(−½) / s_b`, plus 8 ε `Σ_b h_b`) and `M` (the nuclear edges' slopes, weighted by `1 − q`). On the fixtures that worst case is 7.3e-11 at `b` 0.01 µm and at most 1.3e-11 for every other source; on grids of the two changed `slow` pixels in the same domain (W-482's re-run) it is 9.1e-11 at `b` 0.01 µm (0.18927 µm) and at most 1.6e-11 for every other source (0.388796 µm), still within 1e-10; observed: at most 7.1e-14 (identity) and 2.0e-13 (shift). Identity 7 of {doc}`synthetic-cellular-scene` states 1e-10 of the maximum for the shift (amended by W-481, choice V13), which covers that worst case. Positions within 1e-9 µm of a bright region's box of `4 s_b` are boundary voxels. *Plateau* (derived from `E_n` at its worst case `a = −4w`): `0 ≤ A_n U (1 + T_n) − (image − background) ≤ ½ erfc(2√2) · A_n U (1 + T_n)`, with `½ erfc(2√2)` = 3.167e-5, plus 1e-9 of the maximum for rounding. *Site skipping* (derived from the rule of "Stains": a skipped site's face term is below `−4w` over the box): at voxels where the oracle's face term `h_n` is below `−4w` and sets the argument, the identity holds within `½ erfc(2√2) · A_n U (1 + T_n)` (choice V9; counted and reported). Boundary voxels counted and reported, expected 0. | routine; `synthetic`, `validation`. |
| GN14 | The band at every boundary | `band_cases` (none); `ctx_stains` (42) | Label boundaries bisected to 1e-9 µm between nearby positions with the point evaluator; at each, the boundary stain of each cell whose boundary it is, at `b` 0.01 and 0.5 µm; `β_k` at both ends of steps of 1e-3 and 1e-6 µm in random directions from random positions, on steps that do not cross the internal face of a binucleate cell; at positions where the internal-face term `g_k` alone lights the band (`g_k < 0.05 µm ≤ abs(d_k)`), whether a boundary of `C = k` lies within audit D's search radius. | `β_k = 0` on every boundary of `C = k` inside `Ω`, the face between the two sites of a binucleate cell included, so the band is at its peak `B_k U(σ)` there; `β_k` is continuous off that internal face ("The boundary source", W-480); `g_k` lights the band only near a boundary of `C = k` (reported: the page states no distance for "near"). No appearance preset uses the boundary source (O2). | *Continuity* (derived: a Lipschitz bound of `β_k`): `abs(Δβ_k) ≤ L_k · ‖Δσ‖ + 1e-9 µm`. `β_k` is built by `max`, `min` and `abs` from face terms, which are exact distances to planes (constant 1), and from the outer terms of the cell's sites, so `L_k` is the largest constant of those: 1 under T1 and T3 and for an empty cell (a distance to a sphere, or none); under T2 `r̄_n / a_min,l` over the lobes `l` of the site's nucleus (`(ρ − 1) · r̄_n` for an ellipsoid of smallest semi-axis `a_min,l`; the grown term has larger semi-axes), and for lobe 1 under N2 that value times `sqrt(1 / r_min² + G² / r_min⁴)`, with `r_min = 1 − harmonic_amplitude · M_Y` and `G = L · harmonic_amplitude · M_Y` the bound of `‖∇_S r‖` by Bernstein's inequality for a spherical polynomial of degree `L` (in a plane, `abs(r′) ≤ L · Σ abs(a_k)`). *Band at the boundary* (derived from the same bound): at each end of a bisected pair, at most `ε` = 1e-9 µm from the boundary, `β_k ≤ L_k ε + 1e-9 µm`, so the band is at least `exp(−½ ((L_k ε + 1e-9 µm) / b)²)` of `B_k U(σ)` (above `1 − 1e-12` at `b` 0.01 µm for every `L_k` up to 12; the largest on the fixtures is 11.7). *Converse*: reported, no tolerance (the count of such positions with no boundary within `4 × 0.05 µm × F`, audit D's search, expected 0). | routine; `synthetic`, `validation`. |
| GN15 | Bright regions (identity 9) | `ctx_stains`, `context_routine` (42) | For every row of `bright_regions.csv`, `N` at its centre by the evaluator and the oracle; its nucleus ID, stain, size and height. | `N(c_b) = n` exactly for its nucleus `n`; sizes and heights positive; regions only in a `nucleus` stain with texture (identity 9, Tx1). An appearance scene draws no bright region, and its `bright_regions.csv` is empty (GA21); its texture is compared by TR5 and TR7 in GA23. | Exact by the point evaluator (the generator's own function); by the oracle, float rounding of the decision: centres within 1e-9 µm of a boundary reported, expected 0. | routine; `synthetic`, `validation`. |
| GN16 | The unevenness field | `ctx_stains`, a copy with `centres_per_area` 0 (42) | The field `U` recomputed from `unevenness.csv` with U1's normalized formula at the voxel centres (inside GN13's comparison), and on the common voxels of the whole-voxel pair of `ctx_stains` (two FOVs whose offsets differ by whole voxels, so a common voxel images one sample position at two roundings); the field with no centre. | U1: `U` is the recorded bumps' normalized log-Gaussian sum, the same at a sample position in every FOV that images it; with no centre `U = 1` ("Unevenness across the field"). The statement that `log U` has standard deviation exactly `a` at every point, given the centres, has no check of its own: it holds by the normalization (the weights are divided by `sqrt(Σ w_q²)`), which the recomputation reproduces, and audit A of W-394 measured it (0.67 to 0.73 per preset, 200 realizations). An appearance scene draws no field, and its `unevenness.csv` is empty (GA21). | Between FOVs, the ratio within 1e-12 of 1 (float rounding of an exact identity; the tolerance script observes at most 8.9e-16 on the dense 3D presets' spacing and 1.1e-15 on the culture `slow` preset's 0.388796 µm); with no centre exactly 1; the rest inside GN13. | routine; `synthetic`, `validation`. |
| GN17 | 8-bit output (identity 10) and the image dtypes | `ctx_uint8`, its no-cell copy (42) | Each 8-bit stain image against `clip(rint(J · k), 0, 255)` of its float64 rendering `J` with the same streams; `k` from `stain_channels.csv` against `g / m_A` of the configuration; the saved and reloaded stain images' dtype and values. | Identity 10; `k` exists and equals `g / m_A` for a FOV or a scene without a nucleus (B1); float32 and uint8 images are written and read back unchanged ("Truth schema"). In an appearance scene the cast is `clip(rint(g · J), 0, 255)` with `g` the bank's median plateau (GA15). | The 8-bit image exact against the expression with the recorded `k` (the page defines the cast by it); `k` within 1e-9 relative of `g / m_A` (float rounding: `m_A = exp(μ)` of the law); the dtypes and values after reloading exact. | routine; `synthetic`, `validation`. |
| GN18 | The truth translation of the morphology round (identity 8) | `ctx_stains` (both translations), `ctx_plane` (42) | `morphology_truth.csv` against the configuration: `translation_*_vox` against `τ_f ⊘ s`, `pull_displacement_*_vox` against `−τ_f ⊘ s`, each `τ` against its bound; the morphology image against the stain functions at `true_f + τ_f + q ⊙ s` (inside GN13's comparison); with the whole-voxel translation, the morphology `DAPI` against the reference `DAPI` shifted by `τ_f ⊘ s` on the common voxels; `Ω` and `K` from `sample.json` against the same scene without a morphology round; the aligned crop of identity 2 with `T`. | Identity 8; the pull displacement is the negated translation in voxels ("The truth of the translation"); `abs(τ_{f,a}) ≤ T_a · s_a`, `τ_z = 0` at `Z=1` (MT1); `Ω` grows by `T ⊙ s` on each side and `K` is unchanged; `δ_f = nominal_f ⊘ s + j_f + J + T`, and a supplied jitter that makes it whole on every axis gives an aligned crop (entries 3, 4 and 11 of "Accepted statements this addition changes"). | The truth columns within 1e-9 voxels (float rounding of `τ_f ⊘ s`); the shifted image within 1e-10 of the maximum (float rounding of an exact identity, derived as GN13's integer shift; choice V13), boundary voxels reported, expected 0; the rest exact. | routine; `synthetic`, `validation`. |
| GN19 | Projection truth (identity 11) | `ctx_monolayer`, `context_routine` (2D tissue and culture) (42) | Every nucleus centre's z; on a FOV grid, the sample grid and a grid at a fractional offset in Z, Y and X, the nucleus IDs per column of fixed `(y, x)`; columns within 1e-9 µm of a power face. | Every centre at `z_mid`; no column holds two nucleus IDs; `P(y, x)` is the Z maximum of that grid's labels (PT1, "Monolayer"). | Exact; columns within 1e-9 µm of a face reported, expected 0. | routine; `synthetic`, `validation`. |
| GN20 | `Z=1` | `ctx_plane` (42) | Generation; the shapes of every FOV image, stain, morphology image and label image; the z of every nucleus centre, lobe offset, cell anchor, molecule, bright region and unevenness centre; `τ_z`; the label kind `import_labels` reads; GN4, GN6, GN13, GN17 and GN18 on this fixture. | Generation succeeds; every array has one plane; every z is exactly 0; `τ_z = 0`; the saved labels are read as `plane` label images ("`Z=1`"). | Exact. | routine; `synthetic`, `validation`. |
| GN21 | A new random stream leaves the existing draws unchanged | `ctx_off`, `ctx_types` and its two changed copies, `ctx_stains` and a copy with one more stain (42) | The components of every stream the context generator draws from (recorded in `context.json`); with every new part off, the streams drawn; between `ctx_types` and its copies, every molecule's position, compartment and gene; between `ctx_stains` and its copy with one more stain, the other stains' images; the noise descriptor of a reference-round `DAPI` stain. | Every new component starts with `context.`, so no new descriptor equals one of the formed model, the `sample.*` components or `stain.noise.*` (`synthetic/_sample.py::_SAMPLE_COMPONENTS`); with every part off no `context.` stream is drawn (with GN1(b)); changing type 2's abundance changes only the genes of type-2 cells' molecules; changing type 3's molecules per cell changes only the molecules of type-3 cells; adding a stain changes no other stain; a reference-round `DAPI` draws its noise from the sample scene's stain streams ("Random streams", points 1 to 4). | Exact. | routine; `synthetic`, `validation`. |
| GN22 | Truth schema and persistence | `ctx_stains`, `ctx_types`, `ctx_plane`, `ctx_off` (42) | The directory `save_cellular_scene` writes: the sample layout against `save_sample_scene` of `scene.sample`; the twelve tables' columns and order; `context.json` keys, version and SHA-256 of every file; the tables reloaded; an empty population (a type with no cell, no N2, no morphology round); the dataset `save_cellular_dataset` writes, read by `Dataset`, `FOV.load_images` and `FOV.prepare_morphology`. | V1: the sample layout unchanged and twelve new tables with the columns of "Truth schema"; `context_truth_version` `"1"`; hashes equal the files; reloaded tables equal (floats round trip); empty populations give a header and no row; the dataset holds the layout of "Files and layout" and `prepare_morphology` loads the reference stain and registers the morphology round. | Exact (`check_exact`). | routine; `synthetic`, `contract`. |
| GN23 | The four cases and the molecules in their target ranges | `context_slow`; `context_slow_acquired` for TR4 and TR5 of 2D tissue and culture (42) | With W-393's measures (definitions of W-393 `notes.md` §2, rebuilt in a test helper; the bright-region detection at factor 1.5, 0.1 µm³ or µm², one-voxel smoothing) on the truth labels and the stain of each scene, at the pixel W-393 measured each quantity: the edge width 10–90 % (TR4) and the bright regions (TR5) of 2D tissue and culture on `context_slow_acquired`, at their acquired pixels of 0.094635 and 0.194398 µm (at the presets' twice-coarser pixels a culture pixel is about 0.15 µm², near the median bright region of 0.26 µm², and the culture's edge of 0.85 µm spans about two pixels); every other quantity, and TR4 and TR5 of dense 3D tissue, on `context_slow` itself: share of touching nuclei; nucleus volume (area) fraction; share not described by one ellipsoid; edge width 10–90 %; bright regions per nucleus, size and peak; nucleus median over the image's median; within-nucleus CV; background and noise over the plateau; the 8-bit plateau grey level; reads per cell; nuclear fraction per cell; the expanded cell volume (dense 3D); the composite signal share per cell (2D tissue); the share of clipped plateaus and bright-region peaks. | Inside the target range of each quantity, the p5–p95 of the W-393 row named in "The four cases, their controls and their ranges" and in each preset table (PS1); a quantity measured on fewer than 100 objects of its kind is reported, not compared (choice V12); the clipped shares reported. Per scene the dense 3D `slow` preset holds about 150 nuclei and reaches 100; the 2D tissue (about 76) and culture (about 42) `slow` presets and their variants (about 20 and 12) do not, so their quantities measured on nuclei or cells are reported, and a quantity of another kind (bright regions, molecules) is compared only where the scene holds 100 of its objects (the culture's second scene, seed 43, belongs to the chain check only). The appearance presets are compared by GA23 with the target definitions of {doc}`synthetic-real-appearance-validation`. | The range, provisional ("Thresholds and where they come from", TR rows); counts exact. | `slow`; `synthetic`, `validation`, `slow`; run by the Chapter II evidence project (W-536). |
| GN24 | Generation cost | `context_routine`, `context_slow`, `context_slow_acquired` (42) | Each preset generated in its own process under `/usr/bin/time -v`: wall time and peak RSS, recorded in the test output. | Within the bound of C26 (one CPU, 16 GiB, 3600 s per step) for the `slow` presets and the two variants; a test that generates a routine preset in the routine tier takes less than 5 s ("Resources"). | The bound; nothing else is asserted. | `slow`; `synthetic`, `validation`, `slow`; the `slow` presets and the two variants run by the Chapter II evidence project (W-536). |

## Implemented checks

The generator checks are in the modules of `src/python/test/` listed below, each marked
`synthetic` with the kind marker its row names. The implementation issues W-483 to W-485
added most of them with the code they check. W-486 adds
`test_synthetic_cellular_routine.py`: GN4 to GN6, GN8 to GN12, GN15 and GN19 on
`context_routine`, and GN4 on `ctx_stains`. It also adds GN5 on `context_slow` in
`test_synthetic_cellular_presets.py`, and `test_synthetic_cellular_checks.py`, whose
`test_every_gn_row_names_collected_tests` keeps this table true.

That test reads the rows GN1 to GN24 of the table "Generator checks" from this page and
checks four things:

* this table has exactly those rows, in that order;
* every row names at least one test of its module that pytest collects, and none of them
  is marked as an expected failure or skipped by a mark;
* each named test carries `synthetic` and one kind marker, the one its GN row names;
* a row whose tier is `routine` names no `slow` test, a row whose tier is `slow` names only
  `slow` tests, and a row whose tier is `routine and slow` names both.

A row whose tier is `evidence project` names its script under `benchmarks/` and no test,
and every script a row names exists.

**Checks run by the evidence project (W-536).** GN23 and the `slow` and acquired-pixel cases
of GN24 generate the `slow` presets again and allow C26's bound per step, so they are
evidence workloads rather than software checks. Jiahao accepted moving them to the
Chapter II evidence project (TM-14 of W-526, 2026-10-19). They run as
`benchmarks/cellular_evidence.py`, which pytest does not collect, with the same assertions,
printed rows and bounds as the tests they were at revision `0b9fa8b`. From `src/python`:
`uv run python ../../benchmarks/cellular_evidence.py --output <run directory>`; `--check`
selects checks and `--list` lists the cases and seeds. They read no real data. The script
imports `cellular_measures.py` and the GN24 process runner of
`test_synthetic_cellular_presets.py` from `src/python/test/`. GN7 and GN5 on the `slow`
presets and GN24 on the routine presets (within the project's 4 GiB) stay in the suite.

A test of these modules that is not named here is an additional check of the code it
tests. The helpers are:

* `cellular_oracle.py`, the test oracle;
* `cellular_scenes.py`, the tiny fixtures;
* `cellular_measures.py`, W-393's measures for GN23 (run by `benchmarks/cellular_evidence.py`) and the chain checks;
* `cellular_checks.py`, the GN5 check that the two preset modules share.

The routine presets are generated once per session in `test_synthetic_cellular_routine.py`,
and the `slow` presets once per module in `test_synthetic_cellular_presets.py`.

Every tolerance is the one of the table "Generator checks", unchanged, and every float
comparison states its bound: `rtol=0` with the row's absolute bound, or the row's relative
bound alone, never a library default. Where a row says "Exact" for a number that is not an
integer and that the test computes itself, its tests compare it within an absolute 1e-9 (the
project's default float tolerance), or within the bound the row itself states. Counts, IDs,
labels, strings, hashes and dtypes are compared exactly. So are three kinds of float, each
the same stored number on both sides:

* a value with its own copy, as in GN1's arrays and tables and GN11's formed inputs;
* a value with its saved and reloaded file (GN17, GN22);
* a value with the same draw in a second scene (GN21).

The 8-bit images of GN17 against the row's expression, the zeros a row states, and the
configuration values of GN2 against the page's literals are also compared exactly.
No test needs an extra beyond `dev`, `local-registration` and `checkpoint`, so the CI
environment runs them all; GN24 needs `/usr/bin/time`.

Where a test differs from its row as written, the case is a finding recorded in its issue's
worker notes, and the row is not changed:

* **GN3 and GN7, `ctx_crowded`.** At `density=0.6` and `floor=0.9` the domain holds more
  than `0.9 · 0.6`, so placement stopped and recorded without raising (W-483, finding 1).
  W-488 amended the fixture to `density=0.9`. GN3 tests it (`crowded-below` of
  `cellular_scenes.py`): placement stops at the same seed and raises. The copy at
  `floor=0.1` is generated at `density=0.6` (`crowded-0.1`) and stops at the same seed and
  records; GN7 also keeps the copy at `density=0.6` and `floor=0.9`, which reaches its
  floor. `cellular_scenes.py` remains unchanged.
* **GN3, "`z_mid` outside the covered region's Z interval".** This cannot be reached,
  because the interval is centred on `z_mid` (W-483, finding 3). W-488 dropped the clause
  from GN3; the generator keeps the refusal as a guard, without a test.
* **GN6, the anchors.** Each anchor lies strictly inside its own power cell in the
  arithmetic K1 names, as GN6 states since W-488. An anchor that P1 placed on its
  partner's face within rounding, by the oracle's own arithmetic, is reported (W-483,
  finding 2).
* **GN6 on the presets.** These scenes need not hold a voxel inside two nuclei's shapes, so
  that count is reported rather than required to be positive.
* **GN11's rendering identity on the presets.** It runs under the identity's own conditions
  in {doc}`synthetic-sample-scene`: the formed template's noise and background off and
  float64 accumulation. The preset's readout trend and mixing are kept, through each
  molecule's realized amplitudes.
* **GN23, TR7 of dense 3D tissue.** This is reported and not asserted on
  `context-dense3d-slow-v1` (the amendment of 2026-10-15 after the TR table). On
  `appearance-dense3d-slow-v1` GA23 asserts it.

**Retired check:** GN25 was retired on 2026-10-19 because it froze test, helper and selected
fixture source text during development, forcing later test changes to edit source pins first.
Behavioral goldens and shared-fixture checks remain.

(synthetic-cellular-scene-validation-implemented-checks)=

| # | Module and tests | Fixtures | Tier | Floats |
| --- | --- | --- | --- | --- |
| GN1 | `test_synthetic_sample_golden.py`, `test_saved_preset_outputs_equal_the_pins`; `test_synthetic_cellular_scene.py`, `test_gn1b_every_part_off_is_the_sample_scene` | `sample_presets`, `ctx_off` | routine and slow | none compared (digests, arrays and tables exact; in (a) floats to 6 significant digits, amendment of 2026-10-19) |
| GN2 | `test_synthetic_cellular_contract.py`, `test_gn2_existing_fields_and_presets_as_at_the_base`, `test_gn2_new_names_and_fields`; `test_synthetic_cellular_stains_contract.py`, `test_gn2_stain_names_and_fields`; `test_synthetic_cellular_presets_contract.py`, `test_registry_names_and_version`, `test_gn2_every_value_equals_the_page_table` | none (the API pages by `docs/check_reference.py`) | routine | preset values within 1e-9 |
| GN3 | `test_synthetic_cellular_contract.py`, `test_gn3_configuration_refusals`, `test_gn3_generation_refusals`, `test_gn3_a_crowded_scene_below_its_floor`; `test_synthetic_cellular_stains_contract.py`, `test_gn3_stain_refusals`, `test_gn3_stain_generation_refusals` | none; `ctx_crowded` | routine | none compared |
| GN4 | `test_synthetic_cellular_scene.py`, `test_gn4_labels_against_the_oracle_and_the_evaluator`; `test_synthetic_cellular_routine.py`, `test_gn4_labels_on_the_routine_presets`, `test_gn4_labels_on_ctx_stains` | `ctx_shapes`, `ctx_monolayer`, `ctx_plane`, `ctx_types`, `ctx_stains`, `context_routine` | routine | decisions within 1e-9 µm counted as boundary voxels |
| GN5 | `test_synthetic_cellular_scene.py`, `test_gn5_harmonic_bound_is_the_largest_value`, `test_gn5_lobes_harmonics_and_one_connected_shape`; `test_synthetic_cellular_routine.py`, `test_gn5_lobes_on_the_routine_presets`; `test_synthetic_cellular_presets.py`, `test_gn5_lobes_on_the_slow_presets` | `ctx_shapes`, `ctx_monolayer`, `ctx_plane`, `context_routine`, `context_slow` | routine and slow | within 1e-9 µm (1e-9 for the sum of coefficients) |
| GN6 | `test_synthetic_cellular_scene.py`, `test_gn6_no_overlap_and_a_nucleus_inside_its_own_cell`; `test_synthetic_cellular_routine.py`, `test_gn6_no_overlap_on_the_routine_presets` | `ctx_shapes`, `ctx_monolayer`, `ctx_plane`, `context_routine` | routine | decisions within 1e-9 µm counted |
| GN7 | `test_synthetic_cellular_scene.py`, `test_gn7_k1_for_every_pair_and_the_record`, `test_gn7_crowded_stops_and_records_above_a_low_floor`, `test_gn7_crowded_at_the_page_floor_reaches_it`; `test_synthetic_cellular_presets.py`, `test_gn7_routine_preset_reaches_its_floor`, `test_gn7_slow_preset_reaches_its_floor` | `ctx_shapes`, `ctx_monolayer`, `ctx_crowded`, `context_routine`, `context_slow` | routine and slow | K1 within 1e-9 µm; sizes within 1e-9 |
| GN8 | `test_synthetic_cellular_scene.py`, `test_gn8_types_and_genes`; `test_synthetic_cellular_routine.py`, `test_gn8_types_and_genes_on_the_routine_presets` | `ctx_types`, `context_routine` | routine | row sums within 1e-9; anchors within 1e-9 µm counted |
| GN9 | `test_synthetic_cellular_scene.py`, `test_gn9_counts_and_marker_answers`; `test_synthetic_cellular_routine.py`, `test_gn9_counts_and_marker_answers_on_the_routine_presets` | `ctx_types`, `context_routine` | routine | `mecr` within 1e-9; counts exact |
| GN10 | `test_synthetic_cellular_scene.py`, `test_gn10_realized_gene_abundance_of_each_type`; `test_synthetic_cellular_routine.py`, `test_gn10_realized_gene_abundance_on_the_routine_presets` | `ctx_types`, `context_routine` | routine | 4 SE |
| GN11 | `test_synthetic_cellular_scene.py`, `test_gn11_molecules_per_cell_and_the_count_guard`, `test_gn11_molecules_reach_the_formed_generator_unchanged`, `test_gn11_rendering_identity_of_the_reference_round`; `test_synthetic_cellular_routine.py`, `test_gn11_molecules_per_cell_and_the_count_guard_on_the_routine_presets`, `test_gn11_rendering_identity_on_the_routine_presets` | `ctx_types`, `context_routine` | routine and slow | 4 SE; the rendering within the oracle tolerance of {doc}`synthetic-sample-scene` |
| GN12 | `test_synthetic_cellular_scene.py`, `test_gn12_nuclear_fraction_and_compartments`; `test_synthetic_cellular_routine.py`, `test_gn12_nuclear_fraction_and_compartments_on_the_routine_presets` | `ctx_types`, `context_routine` | routine | 4 SE; positions within 1e-9 µm counted |
| GN13 | `test_synthetic_cellular_stains.py`, `test_gn13_stains_against_the_oracle_and_the_evaluator`, `test_gn13_integer_shift_between_two_fovs` | `ctx_stains`, `ctx_plane` | routine | 1e-9 and 1e-10 of the image maximum; the plateau and V9 bounds |
| GN14 | `test_synthetic_cellular_stains.py`, `test_gn14_band_cases`, `test_gn14_band_on_ctx_stains` | `band_cases`, `ctx_stains` | routine | the Lipschitz and band bounds of the row |
| GN15 | `test_synthetic_cellular_stains.py`, `test_gn15_bright_regions`; `test_synthetic_cellular_routine.py`, `test_gn15_bright_regions_on_the_routine_presets` | `ctx_stains`, `context_routine` | routine | centres within 1e-9 µm counted |
| GN16 | `test_synthetic_cellular_stains.py`, `test_gn16_the_field_between_fovs_and_without_centres` | `ctx_stains` and its copy with no centre | routine | the ratio within 1e-12 of 1 |
| GN17 | `test_synthetic_cellular_stains.py`, `test_gn17_8bit_output_and_the_dtypes` | `ctx_uint8`, its no-cell copy | routine | `k` within 1e-9 relative |
| GN18 | `test_synthetic_cellular_stains.py`, `test_gn18_truth_translation`, `test_gn18_whole_voxel_shift_identity`, `test_gn18_domain_grown_covered_region_unchanged_and_the_aligned_crop` | `ctx_stains`, `ctx_plane` | routine | 1e-9 voxels; 1e-10 of the maximum |
| GN19 | `test_synthetic_cellular_scene.py`, `test_gn19_projection_truth_in_a_monolayer`; `test_synthetic_cellular_routine.py`, `test_gn19_projection_truth_on_the_routine_presets` | `ctx_monolayer`, `context_routine` (2D tissue and culture) | routine | columns within 1e-9 µm counted |
| GN20 | `test_synthetic_cellular_stains.py`, `test_gn20_one_plane` | `ctx_plane` | routine | none compared (every z exactly 0) |
| GN21 | `test_synthetic_cellular_scene.py`, `test_gn21_new_components_start_with_context`, `test_gn21_a_type_abundance_changes_only_that_types_genes`, `test_gn21_a_type_count_changes_only_that_types_molecules`; `test_synthetic_cellular_stains.py`, `test_gn21_stain_components_and_streams` | `ctx_off`, `ctx_types` and its copies, `ctx_stains` and its copy with one more stain | routine and slow | none compared |
| GN22 | `test_synthetic_cellular_contract.py`, `test_gn22_saved_layout_tables_and_record`, `test_gn22_empty_populations`; `test_synthetic_cellular_stains_contract.py`, `test_gn22_saved_stain_tables_and_images`, `test_gn22_dataset_read_and_morphology_registered`, `test_gn22_dataset_refusals` | `ctx_stains`, `ctx_types`, `ctx_plane`, `ctx_off` | routine and slow | floats round trip exactly |
| GN23 | `benchmarks/cellular_evidence.py`, checks `GN23-slow` and `GN23-acquired` | `context_slow`, `context_slow_acquired` | evidence project | the TR ranges |
| GN24 | `test_synthetic_cellular_presets.py`, `test_gn24_generation_cost_in_its_own_process` (the routine presets); the `slow` presets and the two variants in `benchmarks/cellular_evidence.py`, checks `GN24-slow` and `GN24-acquired` | `context_routine`; `context_slow`, `context_slow_acquired` in the script | slow | the project's 4 GiB; the bound of C26 in the script |

## Chain checks

One chain check per context, each through the real route of its context (D2), each with
the four steps the issue names, numbered 1 to 4 below, and a step 0 where the route has a
stage before the segmentation whose result this design records. Each step is its own test,
so a miss in one step does not hide the others. The thresholds of steps 1 and 2 apply to the
appearance `slow` preset of each context ({doc}`synthetic-real-appearance`, PR1), whose bank
is built from real images (for the culture, to its two scenes pooled, "Seeds"). The
appearance routine preset of each context, on its closed-form bank, runs the same chain
with steps 3 and 4 asserted and steps 1 and 2 reported (choice V1). Each check adds a step
5, reported ({doc}`synthetic-real-appearance-validation`, 'Consequences for the chain checks'). Each chain fixture is built once per test module. Each step writes its per-object tables to the test's output (the
nuclei matched, missed, merged and split, the false positives, and every molecule whose
assigned cell is not its true cell), so a miss can be reported with its objects.

**Where each tier runs.** The test suite runs every chain check on the routine presets. The
`slow` presets, whose banks are built from real images, run in the Chapter II evidence
project as `benchmarks/cellular_chain_evidence.py` (W-603), outside the test suite: about
2 h on one CPU and up to 13 GiB per step ("Bounds"). The script calls the same tests with
the `slow` tier, so each row below keeps its assertions and thresholds there ("Implemented
chain checks").

**How many nuclei the thresholds count.** By audit A of {doc}`synthetic-cellular-scene`
(W-394; re-run by W-481 for the two presets whose pixel changed), a `slow` scene holds about
150 nuclei in dense 3D tissue, about 76 in 2D tissue (at 0.18927 µm, twice the acquired
pixel) and about 42 in the culture (at 0.388796 µm, twice the acquired pixel); the culture
chain check pools its two scenes, about 83 nuclei. A routine scene holds about 7, 11 and 8.
At the acquired pixels the 2D tissue and culture `slow` scenes held about 20 and 12 nuclei,
too few for TH1 to tell a 5 % from a 20 % miss rate; the coarser pixels and the culture's
second seed are Jiahao's decision of 2026-10-15 for that reason. These are the expected
nuclei of a scene; step 1's fraction counts the interior ones (below), so its count differs
from them, and each step 1 reports the count it used. The appearance `slow` presets, on
which the chain checks run, hold about 738, 337 and 32 objects, about 63 for the culture's
two scenes pooled, and the routine appearance presets about 24, 45 and 5: estimates at the
prototype's reached density ({doc}`synthetic-real-appearance`, 'Presets'), not counts.

**Nuclei against truth (step 1)** uses the matching of the §2.9 check L12
(`l12_agreement` of `test/test_segmentation_validation.py`: one to one at IoU ≥ 0.5, the
larger IoU first) and W-393's definitions (W-393 `notes.md` §2): a predicted label is
*owned* by the truth nucleus that holds at least half of its voxels; a *split* is a truth
nucleus that owns at least two predicted labels; a *merge* is a predicted label that holds
at least half of each of at least two truth nuclei; *missed* is a truth nucleus not matched
and in no split or merge; a *false positive* is a predicted label not matched and owned by
no truth nucleus. A truth nucleus is counted and matched once, by its ID with all its
voxels; a predicted label that covers a detached piece only is owned by that nucleus and is
therefore not a false positive (choice V4, the open point of "A nucleus label with more than
one piece"). An *interior* truth nucleus is one not cut by the Y or X border of the image
compared (the FOV on the route per FOV, the sample's fused grid on the route after fusion);
the matched fraction is computed over interior truth nuclei, and nuclei cut by a Y or X
border or by a Z end are reported (C21, Z1). The truth on the route per FOV is each FOV's
nucleus label image, summed over the FOVs (a nucleus in an overlap counts in each FOV that
holds it, as in W-393); on the route after fusion it is the projection truth `P` on the
segmentation grid (C19, PT1), evaluated as choice V3 says.

**The cell of each molecule (step 2)**: a predicted cell names a truth cell through its
nucleus's match (label expansion, the seeded watershed and the extension through z keep
the seed nucleus's label), and a molecule *agrees* when that truth cell is its
`true_cell_id`, or when it is unassigned and has no true cell (W-393 `notes.md` §2). The
threshold is on the agreement among assigned molecules; the agreement over the molecules
with a true cell and the share of them left unassigned are reported.

**The count identities of the assembly (step 3)** are identities 1 to 5 of
{doc}`assembly-contract` ("Count accounting"), as AS6 and EE3 of
{doc}`stitching-assembly-validation` check them (route per FOV) and as SA7 of
{doc}`sample-segmentation-validation` restates them (route after fusion), with `n_obs` of
`sample.h5ad` equal to the counted cells. They hold on the method's labels and on the truth
labels alike; on `chain_*_truth`, EE1 to EE3 (route per FOV) and SA3 and SA5 (route after
fusion) hold as those pages state them.

**The report (step 4)**: `Dataset.report(sample, fov_ids, config=ReportConfig(...))` on the
export, with the marker groups of the four types and `marker_pairs` the exclusive pair
`(balanced-01, balanced-03)` and the co-expressed pair `(balanced-01, balanced-02)` (C8),
then `load_report` on the folder it wrote. The scene has no run records (choice V2, a: the
molecules come from the truth positions), so the report is made as A2 of
{doc}`report-algorithms` makes it from an export without `run.json`. Known answers, where
the scene defines them:

* **Reads assigned to cells** equal the `sample.h5ad` sum: `reads_in_cells` equals `Σ X` and
  the molecule count of identity 4 of {doc}`assembly-contract`, and the summary's
  `reads_in_cells_equal` is true (DF7, CH3).
* **The MECR of the exclusive pair.** The truth answer is 0 (`marker_pairs.csv`, identity 6).
  In the report, each pair's row of `mecr.csv` equals MECR recomputed by the test from `X`
  of `sample.h5ad` by DF8's definition. On `chain_*_truth`, every counted cell in the
  exclusive pair's `both` holds a retained marker molecule whose sampled label differs from
  its true cell (`label_agrees` false, identity 5 of {doc}`synthetic-sample-scene`; on the
  route after fusion, SA5's condition): with no such molecule the MECR is exactly 0. On the
  method, the MECR and `both` are reported with the true cells of each mixing cell's marker
  molecules: W-393 measured no MECR, so no bound is proposed. The co-expressed pair is
  reported against its truth value.
* **The cells per FOV.** `fovs.csv` `n_cells_retained` equals the counted cells of the
  export's `cells.csv` that each FOV owns (RT1's rule), and is reported against the same
  count on `chain_*_truth`. W-393 measured no cells per FOV on the routes, so no bound is
  proposed for the difference.
* **Read back.** `load_report` returns every table equal to its file, RC12's condition holds
  (every null value named by one row of `unavailable.csv`), and PG1 holds on the page.

### Dense 3D tissue (CD): route per FOV, StarDist `3D_spleen`

| # | Step | Fixture (seed) | What is computed | Expected result (source) | Threshold or tolerance | Tier and markers |
| --- | --- | --- | --- | --- | --- | --- |
| CD0 | Placement | `chain_dense3d` (42) | `Dataset.stitch` statuses and per-axis origin errors against the truth after removing the mean; the placement used. | Reported: W-393 stitched no scene on this route's grid, so no bound is proposed; a placement that does not make every FOV `image_supported` is replaced by the truth placement, and the test output says so (SC1). | Reported. | `slow`, `learned`; `integration`, `validation`. |
| CD1 | Nuclei against truth | `chain_dense3d`, `chain_dense3d_routine` (42) | Per FOV, `3D_spleen`'s nucleus labels against the FOV's truth nucleus labels, summed over the FOVs: matched, missed (interior and at the border), merged truth nuclei, splits, false positives; the nuclei cut by a Z end; the multi-piece nuclei. | On the `slow` preset: TH1, TH2 and TH3 hold; false positives reported (TH-FP). On the routine preset: reported. | TH1: matched fraction of interior truth nuclei ≥ 0.9; TH2: merged truth nuclei ≤ 0.10 of the truth count; TH3: splits 0 (provisional). | `slow`, `learned`; `integration`, `validation`. |
| CD2 | The cell of each molecule | as CD1 | Per molecule with `in_footprint`, its assigned cell (cells by `assign_molecules` with the 0.7776-µm planar expansion) against `true_cell_id`; agreement among assigned molecules, over the molecules with a true cell, and the share left unassigned; on `chain_dense3d_truth`, `cell_id` against `fov_cell_label`. | On the `slow` preset: TH4 holds; the agreement and the unassigned share reported (TH-UN). On the truth labels: EE1 holds exactly. | TH4: agreement among assigned molecules ≥ 0.85 (provisional). | `slow`, `learned`; `integration`, `validation`; the truth-label part `slow` only. |
| CD3 | Count identities | `chain_dense3d`, `chain_dense3d_routine`, `chain_dense3d_truth` (42) | The export's tables and `sample.h5ad` against identities 1 to 5 of "Count accounting"; on the truth labels, EE2 and EE3. | Every identity holds (AS6, EE3); `n_obs` equals the counted cells. | Exact. | `slow`, `learned` (the method); `slow` (the truth labels); `integration`, `validation`. |
| CD4 | The report | as CD3 | `Dataset.report` on the export and `load_report`: the known answers of "The report (step 4)". | As listed there. | Exact for the identities; the rest reported. | as CD3. |
| CD5 | The method's labels against the method's labels of real images | `chain_dense3d` (42) | On `3D_spleen`'s nucleus labels of each FOV: the touching share (TR1), the volume fraction (TR2) and the cell volume of the nuclei expanded by 0.7776 µm (TR11), with W-393's definitions; the same on the truth labels beside them. | Reported against W-393's ranges of TR1, TR2 and TR11, which are measured on the same method's labels of the real images: a like-for-like comparison of what the method sees (TG2). | Reported, with the value, the range and the object count; no threshold (no measurement of the method's labels on an appearance scene exists). | `slow`, `learned`; `integration`, `validation`. |

### 2D tissue (CT): projection sample, route after fusion, StarDist `2D_versatile_fluo` and the seeded watershed

| # | Step | Fixture (seed) | What is computed | Expected result (source) | Threshold or tolerance | Tier and markers |
| --- | --- | --- | --- | --- | --- | --- |
| CT0 | Stitch on the projection sample | `chain_tissue2d`, `chain_tissue2d_routine` (42) | `Dataset.stitch` on `PI`: the FOVs `image_supported`; per axis the origin error against the truth after removing the mean. | On the `slow` preset TH5 holds. When it does not, the step fails (a finding) and CT1 to CT4 run on the truth placement, saying so (SC1). On the routine preset: reported. | TH5: every FOV `image_supported` and every per-axis error ≤ 1 µm (provisional). | `slow`, `learned` (with its chain); `integration`, `validation`. |
| CT1 | Nuclei against truth | `chain_tissue2d`, `chain_tissue2d_routine` (42) | The export's nucleus labels (`segmentation_<nucleus run>.ome.tif`) against `P` on the segmentation grid: matched, missed (interior and at the sample's border), merged, splits, false positives. | On the `slow` preset: TH1, TH2 and TH3 hold; false positives reported. On the routine preset: reported. | TH1, TH2, TH3 (provisional). | `slow`, `learned`; `integration`, `validation`. |
| CT2 | The cell of each molecule | as CT1 | The export's retained molecules: assigned cell (the seeded watershed's cells, named through their seed nucleus) against `true_cell_id`; agreement among assigned, over those with a true cell, and the unassigned share; on `chain_tissue2d_truth`, SA5. | On the `slow` preset: TH4 holds; the rest reported. On the truth labels: SA5 holds. | TH4 (provisional). | `slow`, `learned`; `integration`, `validation`. |
| CT3 | Count identities | `chain_tissue2d`, `chain_tissue2d_routine`, `chain_tissue2d_truth` (42) | SA7's identities on the export; `check_export` (R1 to R3, F, L, S, T1 to T5); on the truth labels, SA3. | Every identity and condition holds. | Exact. | as CD3. |
| CT4 | The report | as CT3 | As CD4, with RT1's rule for the cells per FOV on the route after fusion. | As "The report (step 4)". | Exact for the identities; the rest reported. | as CD3. |
| CT5 | As CD5 | `chain_tissue2d` (42) | On the export's nucleus labels: TR1 and TR2 (area fraction); the truth beside them. | As CD5, against W-393's 2D tissue ranges. | Reported. | as CD5. |

### Sparse culture (CU): route per FOV, Cellpose `cpsam_v2` with the morphology round

| # | Step | Fixture (seed) | What is computed | Expected result (source) | Threshold or tolerance | Tier and markers |
| --- | --- | --- | --- | --- | --- | --- |
| CU0 | Registration of the morphology round; placement | `chain_culture` (42 and 43), `chain_culture_routine` (42) | On each scene, `FOV.prepare_morphology`'s recovered `displacement_zyx` against `pull_displacement_*_vox` of `morphology_truth.csv`, per FOV and axis; the stitch as CD0. | Reported: W-393 measured no round offset (the current scene has no morphology round), so no bound is proposed; the placement as CD0. | Reported. | `slow`, `learned`; `integration`, `validation`. |
| CU1 | Nuclei against truth | as CU0 | Per FOV, Cellpose's nuclei extended through z against the FOV's truth nucleus labels, as CD1; on the `slow` preset the counts of the two scenes (seeds 42 and 43) summed before the fractions are formed, and each scene's counts reported. | On the `slow` preset: TH1, TH2 and TH3 hold on the pooled counts; false positives reported. On the routine preset: reported. | TH1, TH2, TH3 (provisional). | `slow`, `learned`; `integration`, `validation`. |
| CU2 | The cell of each molecule | as CU0 | Per molecule, the cell of Cellpose's cell labels on `Flamingo` and `DAPI`, extended through z, named through the nucleus it contains, against `true_cell_id`; as CD2; on the `slow` preset the molecules of the two scenes pooled before the agreement is formed, and each scene's agreement reported. | On the `slow` preset: TH4 holds on the pooled molecules; the rest reported. On the truth labels: EE1 holds on each scene. | TH4 (provisional; no W-393 measurement on this route, see TH4). | `slow`, `learned`; `integration`, `validation`. |
| CU3 | Count identities | `chain_culture`, `chain_culture_truth` (42 and 43), `chain_culture_routine` (42) | As CD3, on each scene. | As CD3, on each scene; not pooled. | Exact. | as CD3. |
| CU4 | The report | as CU3 | As CD4, on each scene's export. | As "The report (step 4)", on each scene; not pooled. | Exact for the identities; the rest reported. | as CD3. |
| CU5 | As CD5 | `chain_culture` (42 and 43) | On Cellpose's nuclei extended through z, per scene and pooled: TR1 and TR2; the truth beside them. | As CD5, against W-393's culture ranges. | Reported. | as CD5. |

### Implemented chain checks

Each row's tests are in `test_integration_cellular_chain.py` (W-487), marked `integration`
and `validation`. The chains with a model are `slow` and `learned`, and the truth-label
chains are `slow`. The suite collects them on the routine presets only (the tier `routine`;
the culture at seed 42). Each row's `slow` presets run in the Chapter II evidence project
as the check of the same name in `benchmarks/cellular_chain_evidence.py` (W-603). The
script calls these tests with the tier `slow` and runs one case per test and seed that the
suite held before W-603, 31 in all; each case names that former node ID. The last column
states the preset tier that the suite keeps. `test_synthetic_cellular_checks.py` keeps
this table aligned with the collected tests and the script.

(synthetic-cellular-scene-validation-implemented-chain-checks)=

| # | Module and tests | Evidence project (the `slow` presets) | Tier in the suite |
| --- | --- | --- | --- |
| CD0 | `test_integration_cellular_chain.py`, `test_cd0_placement` | `benchmarks/cellular_chain_evidence.py`, check `CD0` (seed 42) | routine |
| CD1 | `test_integration_cellular_chain.py`, `test_cd1_nuclei_against_truth` | `benchmarks/cellular_chain_evidence.py`, check `CD1` (seed 42) | routine |
| CD2 | `test_integration_cellular_chain.py`, `test_cd2_cell_of_each_molecule`, `test_cd2_truth_labels_ee1` | `benchmarks/cellular_chain_evidence.py`, check `CD2` (seed 42) | routine |
| CD3 | `test_integration_cellular_chain.py`, `test_cd3_count_identities`, `test_cd3_count_identities_truth_labels` | `benchmarks/cellular_chain_evidence.py`, check `CD3` (seed 42) | routine |
| CD4 | `test_integration_cellular_chain.py`, `test_cd4_report`, `test_cd4_report_truth_labels` | `benchmarks/cellular_chain_evidence.py`, check `CD4` (seed 42) | routine |
| CD5 | `test_integration_cellular_chain.py`, `test_cd5_scene_as_the_method_sees_it` | `benchmarks/cellular_chain_evidence.py`, check `CD5` (seed 42) | routine |
| CT0 | `test_integration_cellular_chain.py`, `test_ct0_stitch_on_the_projection_sample` | `benchmarks/cellular_chain_evidence.py`, check `CT0` (seed 42) | routine |
| CT1 | `test_integration_cellular_chain.py`, `test_ct1_nuclei_against_truth` | `benchmarks/cellular_chain_evidence.py`, check `CT1` (seed 42) | routine |
| CT2 | `test_integration_cellular_chain.py`, `test_ct2_cell_of_each_molecule`, `test_ct2_truth_labels_sa5` | `benchmarks/cellular_chain_evidence.py`, check `CT2` (seed 42) | routine |
| CT3 | `test_integration_cellular_chain.py`, `test_ct3_count_identities`, `test_ct3_count_identities_truth_labels` | `benchmarks/cellular_chain_evidence.py`, check `CT3` (seed 42) | routine |
| CT4 | `test_integration_cellular_chain.py`, `test_ct4_report`, `test_ct4_report_truth_labels` | `benchmarks/cellular_chain_evidence.py`, check `CT4` (seed 42) | routine |
| CT5 | `test_integration_cellular_chain.py`, `test_ct5_scene_as_the_method_sees_it` | `benchmarks/cellular_chain_evidence.py`, check `CT5` (seed 42) | routine |
| CU0 | `test_integration_cellular_chain.py`, `test_cu0_registration_and_placement` | `benchmarks/cellular_chain_evidence.py`, check `CU0` (seeds 42 and 43) | routine |
| CU1 | `test_integration_cellular_chain.py`, `test_cu1_nuclei_against_truth` | `benchmarks/cellular_chain_evidence.py`, check `CU1` (seeds 42 and 43) | routine |
| CU2 | `test_integration_cellular_chain.py`, `test_cu2_cell_of_each_molecule`, `test_cu2_truth_labels_ee1` | `benchmarks/cellular_chain_evidence.py`, check `CU2` (seeds 42 and 43) | routine |
| CU3 | `test_integration_cellular_chain.py`, `test_cu3_count_identities`, `test_cu3_count_identities_truth_labels` | `benchmarks/cellular_chain_evidence.py`, check `CU3` (seeds 42 and 43) | routine |
| CU4 | `test_integration_cellular_chain.py`, `test_cu4_report`, `test_cu4_report_truth_labels` | `benchmarks/cellular_chain_evidence.py`, check `CU4` (seeds 42 and 43) | routine |
| CU5 | `test_integration_cellular_chain.py`, `test_cu5_scene_as_the_method_sees_it` | `benchmarks/cellular_chain_evidence.py`, check `CU5` (seeds 42 and 43) | routine |

## Thresholds and where they come from

Every threshold below is **provisional**: it was proposed by W-393 from runs of the §2.9
methods on the *current* sample scene, it is approved or changed by Jiahao at the gate
W-396, and only then does an implementation issue assert it. W-393 states that each "would
be re-measured on the new scene before they are fixed" (W-393 `notes.md` §10): the first run
of the chain checks is that re-measurement, and a miss there is a finding ("What a failure
means"). For the appearance presets each is provisional again and approved at the gate
W-515; W-393 measured it on neither scene.

| # | Threshold | Applies to | W-393 table and row it comes from | Observed by W-393 |
| --- | --- | --- | --- | --- |
| TH1 | Matched fraction at IoU ≥ 0.5 of the truth nuclei not cut by the Y or X border at least 0.9 | CD1, CT1, CU1 (`slow` presets; CU1 on the counts of the culture's two scenes, seeds 42 and 43, pooled; about 738, 337 and 63 objects expected before the border rule in the appearance presets (estimates; 'How many nuclei the thresholds count')) | W-393 `notes.md` §10, "Nuclei"; `section10_values.txt`, row "interior matched fraction on misses alone, routes per FOV"; `routes.csv` rows `<scene>/spleen` (CD1), `<scene>/cellpose_route` (CU1) and `<scene>/projection` (CT1), columns `n_truth`, `truth_at_xy_border`, `missed_not_at_xy_border` | Routes per FOV: lowest 0.9091 (one interior miss of 11: `edge/spleen`, `edge/cellpose_route`, `plane/cellpose_route`); no row below 0.9. Route after fusion: `missed_not_at_xy_border` 0 or 1 per row (1 on `background/projection`), but `truth_at_xy_border` is empty on the projection rows, so the interior fraction there is **not measured** by W-393. Merges in three rows remove nuclei from the matched set whose interior status the table does not give (`dense/projection`, `combined/cellpose_route`, `combined/projection`). |
| TH2 | Merged truth nuclei at most 10 % of the truth count | CD1, CT1, CU1 (as TH1) | W-393 `notes.md` §10, "Merges and splits"; `section10_values.txt`, row "merged truth nuclei per row"; `routes.csv` column `merged_truth` | At most 2 per row; largest share 0.0556 (2 of 36, `combined/projection`). |
| TH3 | Splits 0 | CD1, CT1, CU1 (as TH1) | W-393 `notes.md` §10, "Merges and splits"; `section10_values.txt`, row "splits, all rows"; `routes.csv` column `split` | 0 in every row. |
| TH4 | Agreement among assigned molecules at least 0.85 | CD2, CT2, CU2 (`slow` presets; CU2 on the molecules of the two culture scenes pooled) | W-393 `notes.md` §10, "Molecules"; `section10_values.txt`, row "agreement among assigned molecules"; `routes.csv` column `agree_among_assigned`, rows `<scene>/spleen` (CD2) and `<scene>/projection` (CT2) | 0.8632 (`uint8/projection`) to 1.0000 (`base/spleen`); spleen rows 0.97 to 1.00, projection rows 0.86 to 1.00. For CU2 there is **no W-393 row**: the culture cell call was not runnable on the current scene (`routes.csv` rows `<scene>/cellpose_route`, column `note`, "cells not runnable"), so 0.85 applies to the culture as W-393's proposal for every route, without a culture measurement. |
| TH5 | Every FOV `image_supported` and every per-axis origin error at most 1 µm after removing the mean | CT0 (`slow` preset) | W-393 `notes.md` §10, "Stitch on the projection sample"; `section10_values.txt`, rows "stitch placed every FOV" and "largest per-axis placement error where every FOV was placed"; `routes.csv` rows `dense/projection`, `edge/projection`, `combined/projection`, `plane/projection`, columns `stitch`, `fovs_placed`, `placement_max_abs_error_um` | Every FOV placed on 4 of 9 scenes, largest error 0.7622 µm (`combined`); stopped on 5 (3 of 9 FOVs placed on `base`, `background`, `molecules`, `uint8`; 6 of 9 on `realscale`). Kept by W-393 because SC1 makes the presets dense enough; the routes per FOV were not stitched by W-393. |

**Reported, with no threshold proposed.** These quantities are reported by the chain checks
and asserted by none, because W-393 proposes no bound for them or did not measure them:

| # | Quantity | Why no threshold | W-393 source |
| --- | --- | --- | --- |
| TH-FP | False positives per truth nucleus (steps 1) | W-393 withdrew its first bound of 20 %: three rows exceed it, from the 8-bit clip and the wide edge, which the new presets change; in the culture appearance presets a source cell can carry the unlabelled DAPI of a second nucleus, a further cause of false positives; "a bound is proposed only once the new scene is measured" | `notes.md` §10, "False positives"; `section10_values.txt`, rows "false positives per truth nucleus" (0 to 29.4 %) and "rows above 20 %" |
| TH-UN | Share of the molecules with a true cell left unassigned (steps 2) | "bounded only once the scene's territories are defined to match each route's cell rule"; T2 does that for the tissue presets, but W-393 did not measure it on such a scene | `notes.md` §10, "Molecules"; `section10_values.txt`, row "molecules with a true cell left unassigned" (12 % to 80 %) |
| TH-MECR | The exclusive pair's MECR after segmentation (steps 4) | Not measured by W-393: the current scene has no cell types | `notes.md` §8, row "Cell types with gene abundances" |
| TH-CPF | Cells per FOV against the truth-label chain (steps 4) | Not measured by W-393 | none |
| TH-REG | Registration error of the morphology round (CU0) | Not measured by W-393: no morphology round on the current scene | `notes.md` §4 (the culture cell call not runnable) |
| TH-PL | Placement on the routes per FOV (CD0, CU0) | Not measured by W-393: it stitched the projection scenes only | `notes.md` §4 |

**Target ranges of the generator checks (TR).** GN23 compares the scene with the target
ranges of {doc}`synthetic-cellular-scene` ("The four cases, their controls and their
ranges" and the preset tables). GA23 compares the appearance presets with the rows as
{doc}`synthetic-real-appearance-validation` defines them ('Target definitions'): TR1, TR2
and TR11 reported on the truth and compared like for like in step 5 of the chain checks,
TR2v, TR8r and TR8v added, TR3 reported beside the source entries. Each is the p5–p95 of a W-393 row of
`results/real/<context>/{geometry,stain,molecules}.csv`, collected in W-393 `notes.md` §9;
each is a development target, provisional, approved at the gate W-396 with the presets
(C23 to C25):

| # | Quantity | Range by context (dense 3D; 2D tissue; culture) | W-393 row |
| --- | --- | --- | --- |
| TR1 | Share of touching nuclei, per image | 0.44–0.57; 0.76–0.82; 0.33–0.72 for the closed-form presets; for the appearance presets reported on the truth labels, not compared: the truth is placed at the visible stain's density (round 10: 0.99 against 0.56 on the held-out LN crop); compared like for like, on the method's labels of the scene against W-393's range on the same method's labels of the real images, in step 5 of the chain checks, reported | `geometry.csv` "share of touching nuclei, per image" |
| TR2 | Nucleus volume (area) fraction, per image | 0.07–0.16; 0.30–0.44; 0.17–0.45 for the closed-form presets; for the appearance presets TR2v, with the truth label share reported beside it and compared like for like in step 5 of the chain checks | `geometry.csv` "nucleus volume fraction, per image" |
| TR2v | Visible-stain share, per scene (stain smoothed by one voxel above its Otsu threshold) | 0.47–0.68; 0.29–0.44; DAPI Z maximum 0.12–0.14 and Flamingo Z maximum 0.41–0.53 | W-513 `tables/visible_share_<context>.csv`, the rows of {doc}`synthetic-real-appearance-validation` |
| TR3 | Share not described by one ellipsoid (IoU < 0.8), per image | 0.064–0.074; 0.027–0.060; 0.12–0.43 for the closed-form presets; for the appearance presets reported, not compared: the share on the placed truth labels beside the share on the entries they come from and their difference, whose bound is not measured; the real range reported with the squash threshold of rule 4, its control (limit L2; round 10: 0.009 against 0.047) | `geometry.csv` "share of nuclei not described by one ellipsoid (IoU < 0.8), per image" |
| TR4 | Edge width 10–90 % (µm) | 0.20–2.20; 0.31–1.57; 0.33–1.63 | `stain.csv` "edge width 10-90 %" |
| TR5 | Bright regions per nucleus; size; peak over the nucleus median | 1–5, 0.13–5.04 µm³, 1.88–3.10; 0–8, 0.11–1.16 µm², 1.76–2.60; 0–58, 0.11–7.49 µm², 1.90–9.58 | `stain.csv` "bright regions per nucleus (factor 1.5)", "bright region size (factor 1.5)", "bright region peak over nucleus median (factor 1.5)" |
| TR6 | Nucleus median over the image's median of nucleus medians | 0.70–1.44; 0.52–1.37; 0.25–4.92 | `stain.csv` "nucleus median over the image's median of nucleus medians" |
| TR7 | Variation within a nucleus (CV of interior voxels) | 0.25–0.41; 0.20–0.38; 0.11–1.35 | `stain.csv` "variation within a nucleus (CV of interior voxels)" |
| TR8 | Background over plateau, per image; noise over plateau, per 128-px block | 0.01–0.34, 0–0.063; 0.022–0.031, 0–0.033; 0.002–0.047, 0–0.24 for the closed-form presets; for the appearance presets the background is reported, and the rows TR8r and TR8v are added; the noise row unchanged | `stain.csv` "background level over plateau, per image", "noise (sd of background) over plateau, per 128-px block" |
| TR8r | Background at label rims, median over the truth labels (cells in the culture, on the volume), over the plateau | 0.33–0.40; 0.125–0.130; 0 | W-513 `tables/rim_background_<context>.csv`, column `local_bg_over_plateau_median` |
| TR8v | Median stain outside the visible stain, over the plateau (the culture on the Z maximum) | 0–0.54; 0.022–0.047; DAPI 0.036–0.038 and Flamingo 0.022–0.074, reported | W-513 `tables/visible_share_<context>.csv`, column `not_visible_over_plateau`, the rows of {doc}`synthetic-real-appearance-validation` |
| TR9 | 8-bit plateau (grey levels) | 35–50; 64–93; 13–83 | W-393 `notes.md` §9 (the tiles' plateaus) |
| TR10 | Reads per cell; nuclear fraction of reads per cell | 1–19, 0.25–0.75; 85–310, 0.43–0.93; 86–1884, 0–0.56 | `molecules.csv` "reads per cell (kept cells)" (tissue-2D: "reads per cell (cells not touching the crop border)"), "nuclear fraction of reads, per cell with >= 10 reads" |
| TR11 | Cell volume (expanded nuclei, dense 3D) | 90–185 µm³ for the closed-form presets; for the appearance presets reported on the truth; compared like for like on the method's nuclei expanded by 0.7776 µm in step 5 of CD, reported | `geometry.csv` "cell volume" |
| TR12 | Composite signal share per cell (2D tissue) | 0.52–0.96 | `molecules.csv` "share of pixels with signal in the composite, per cell" |

**Amendment of 2026-10-15 (W-485; operator decision, confirmed by Jiahao on 2026-10-16; it applies to the closed-form presets only, W-515, AM1).** GN23
measured TR7 of dense 3D tissue on `context-dense3d-slow-v1` at seed 42 at a median
within-nucleus CV of 0.168 on 143 nuclei, outside its range 0.25–0.41. Two trials on that
preset, each changing one bright-region law, gave 0.212 (height median 2.0 instead of 1.35)
and 0.206 (5 regions per nucleus instead of 3), each with TR5 inside its ranges: the Gaussian
bright regions of Tx1, within TR5's ranges, do not reach the real within-nucleus variation.
TR7 of dense 3D tissue is therefore a known limit of the closed-form presets: GN23 reports
it with its measured value, range and object count and does not assert it. The appearance
presets of {doc}`synthetic-real-appearance` are the texture control: round 10 of the
prototype gave 0.317 against the real 0.348 on the held-out crop, and GA23 of
{doc}`synthetic-real-appearance-validation` asserts TR7 of `appearance-dense3d-slow-v1`
against 0.25–0.41. TR7 of 2D tissue and of the culture is reported,
as V12 gives it.

The tissue-2D ranges come from its densest tiles and every real label is method output
(W-393 `notes.md` §12); the density and background ranges of the appearance presets come
from W-513's per-image measurements of three or four images per context, each range the
smallest to the largest value; the routine presets take some values outside the middle of a range
on purpose ("Presets", PS1), and are therefore not compared with TR10.

## What a failure means

A chain check that misses a threshold on the synthetic scene is **a finding about the scene
or about the threshold**, not a verdict on the method, and not a defect to be made to pass.

* **The assertion is not weakened.** No threshold, tolerance, matching rule or fixture is
  changed during an implementation issue to make a check pass, and a failing check is not
  skipped, marked as an expected failure, deselected or deleted. A test that fails for an
  environmental reason (a missing extra or model, a thread count) is fixed by how it is run.
* **How it is reported.** The step's assertion message names the counts that missed and the
  threshold row (TH1 to TH5), and its output holds the per-object tables of that step. The
  implementation worker returns the issue blocked on that criterion, with the finding: the
  objects that missed (a figure per failure kind, in the form of W-393's
  `results/figures/`), the scene property it points to (for example the touching share, the
  edge width or the 8-bit plateau), and the options for Jiahao: change a scene control within
  its target range, change the threshold, or record the case as a known limit of the scene.
* **Who decides.** Jiahao, at a gate, as for an accepted page: a changed threshold or
  preset is written into this page as a dated amendment ("Amendments after acceptance", in
  the form of {doc}`sample-segmentation-validation`), with the finding that caused it. Until
  then the check stays as written and stays red.
* **What it never means.** A miss is never read as an accuracy of the method on real tissue,
  and a pass never is either (see "The claim boundary").

A generator check that misses is a defect of the generator or of the specification: a
statistical row (4 SE) that misses at a correct implementation is reported to Jiahao with
its value and is not retried with another seed.

## Execution conditions

**Models, by path, offline.** No model is downloaded; a test whose model is absent skips
with its reason, as `test/test_segmentation_learned.py` does. The SHA-256 of each file is
W-392's `inventory/models.json`, which the implementation checks before its first call.

| Chain check | Model and its path | How the test finds it | Environment and device |
| --- | --- | --- | --- |
| CD | StarDist `3D_spleen`: `/home/unix/jiahao/.cache/starfinder/user-models/stardist/3D_spleen` (`config.json`, `thresholds.json`, `weights_best.h5`) | `STARFINDER_STARDIST_3D_SPLEEN`, as `test/test_segmentation_validation.py` | The locked environment of the batch (`dev-s210-export-20261007T050852Z`, extras `stardist`, `anndata`, `multiview-stitcher`), CPU, one thread (`CUDA_VISIBLE_DEVICES=""`) |
| CT | StarDist `2D_versatile_fluo`: `/home/unix/jiahao/.cache/starfinder/weights/stardist/2D_versatile_fluo` | `STARFINDER_WEIGHTS_DIR=/home/unix/jiahao/.cache/starfinder/weights` (`_weights.py::weights_directory`) | As CD; CPU, because the export applies one device to every run of its `segmentation` block and `seeded_watershed` runs on the CPU only (open point of {doc}`synthetic-cellular-scene`, W-393 `notes.md` §4) |
| CU | Cellpose `cpsam_v2`: `/home/unix/jiahao/.cache/starfinder/weights/cellpose/cpsam_v2/cpsam_v2` | `STARFINDER_WEIGHTS_DIR`, as CT; `HF_HUB_OFFLINE=1` | As CD, extra `cellpose`; CPU |

The GPU is not used by any gate. W-393 also measured each model call on the GPU (the
`s29-spike-20261004T180758Z` environment); those numbers are given below for comparison
only, and a GPU run of a chain check would need its own grant.

**Measured cost, from W-393** (`sizing.csv`, final attempts, one CPU, `/usr/bin/time -v`;
"peak RSS" is `peak_rss_kib`). W-393 ran each operation on the nearest scene the current
generator makes, not on the presets of this page, and it timed the operations alone, not a
chain. W-393's per-operation costs are therefore the measured costs of each learned check.
Where W-393 could not run an operation, its row says **not measured** with W-393's reason
and gives beside it the measured cost of the nearest operation W-393 did run, labelled as
such; that number is a measurement of the other operation, not an estimate of this one.
The time of a whole chain check (the sum of its operations in one process) and of
generating the new presets is not measured either ({doc}`synthetic-cellular-scene`,
"Resources", gives estimates, W-481's for the two presets whose pixel changed; "Bounds" below
adds them to the measured operations, labelled as estimates). Each of these is measured by draft I5 before the gate sets a
threshold or the selection that runs it, under its criterion "the measured time and peak
RSS of each step (the Cellpose cell call on `Flamingo` and `DAPI` and each whole chain
included, which W-393 did not measure) are in the worker notes and within the granted
bound, before a threshold or the selection that runs the check is set from them"; the
presets' generation is measured by I3 (GN24).

| Tier | Operation (W-393 `sizing.csv` tag) | Wall time | Peak RSS | GPU, for comparison |
| --- | --- | --- | --- | --- |
| Routine generator checks | Generation of the nearest routine scenes: `gen_ln_routine`, `gen_t2_routine`, `gen_cu_routine` | 6.06 s, 10.42 s, 6.43 s | 296,220; 282,420; 247,036 KiB | — |
| Routine generator checks | The `ctx_*` fixtures on `sample-tiny-v1` | not measured by W-393 (the accepted `sample-tiny-v1` saves in 0.41 to 0.42 s, the golden test's time in W-394's notes §3); measured by I1 and I2 with `--durations=0` | not measured | — |
| `slow` generator checks (GN23, GN24) | Generation of the nearest `slow` scenes: `gen_ln_slow`, `gen_t2_slow`, `gen_cu_slow` | 194.12 s, 52.97 s, 63.64 s | 3,498,428; 1,477,016; 1,485,080 KiB | — |
| `learned` and `slow`: CD | `3D_spleen` per FOV, routine 2 × 2: `size_spleen_ln_routine_cpu` | 178.7 s (42.2 s per FOV) | 3,212,236 KiB | `size_spleen_ln_routine_gpu`: 43.75 s, 1,906,752 KiB, 8,666 MiB |
| `learned` and `slow`: CD | `3D_spleen` per FOV, `slow` 3 × 3: `size_spleen_ln_slow_cpu` | 778.92 s (85.4 s per FOV) | 3,263,160 KiB | `size_spleen_ln_slow_gpu`: 90.48 s, 2,028,660 KiB, 8,666 MiB |
| `learned` and `slow`: CD | Label expansion 0.7776 µm: `size_expand_ln_routine_cpu`, `size_expand_ln_slow_cpu` | 0.91 s, 1.77 s | 161,756; 181,876 KiB | not runnable (no device) |
| `learned` and `slow`: CT | Export after fusion (fuse, StarDist 2D on CPU, seeded watershed, assignment): `size_proj_t2_routine`, `size_proj_t2_slow` | 18.02 s, 22.0 s | 961,904; 994,216 KiB | not runnable in the export (device rule) |
| `learned` and `slow`: CT | StarDist `2D_versatile_fluo` alone: `size_sd2d_t2_routine_cpu`, `size_sd2d_t2_slow_cpu` | 11.48 s, 11.79 s (0.94 s, 0.92 s warm per call) | 880,780; 882,300 KiB | `size_sd2d_t2_*_gpu`: 19.57 s, 19.56 s, about 1,480,000 KiB, 490 MiB |
| `learned` and `slow`: CT | Seeded watershed alone: `size_ws_t2_routine_cpu.rerun-20261015T032049Z`, `size_ws_t2_slow_cpu.rerun-20261015T032050Z` | 0.93 s, 0.96 s | 160,588; 162,016 KiB | not runnable (CPU only) |
| `learned` and `slow`: CU | Cellpose `cpsam_v2` nuclei per FOV: `size_cellpose_cu_routine_cpu`, `size_cellpose_cu_slow_cpu` | 117.4 s, 261.22 s (27.3 s, 27.9 s per FOV) | 2,461,548; 2,428,836 KiB | `size_cellpose_cu_*_gpu`: 31.08 s, 59.2 s; 3,165,704, 3,387,024 KiB; 2,284 MiB |
| `learned` and `slow`: CU | Extension through z: `size_extend_cu_routine_cpu`, `size_extend_cu_slow_cpu` | 1.03 s, 13.67 s | 161,976; 182,804 KiB | not runnable (no device) |
| `learned` and `slow`: CU | Cellpose cell call on `Flamingo` and `DAPI` | **not measured** by W-393: not runnable on the current scene, which has no cytoplasm stain (W-393 `routes.csv`, rows `<scene>/cellpose_route`, `note` "cells not runnable"). Nearest operation W-393 ran, on the same grids and model: the Cellpose `cpsam_v2` nucleus call above, `size_cellpose_cu_routine_cpu` 117.4 s and `size_cellpose_cu_slow_cpu` 261.22 s. Measured by I5 (criterion above). | **not measured**. Nearest operation: 2,461,548 and 2,428,836 KiB (the same two rows). Measured by I5. | not measured; nearest operation `size_cellpose_cu_*_gpu` 31.08 s and 59.2 s |
| `learned` and `slow`: CD, CT, CU | A whole chain check (generation, stitch, segmentation, assignment, export, report in one process) | **not measured**: W-393 timed operations alone; the measured costs are the rows above. Measured by I5 (criterion above). | **not measured**; measured by I5 | — |
| `slow` (truth labels) | The chains with the truth labels (no model) | **not measured** by W-393; measured by I5 (criterion above) | **not measured**; measured by I5 | — |

**The two `slow` presets whose pixel changed.** `context-tissue2d-slow-v1` (0.18927 µm in Y
and X, StarDist at the scale 0.5) and `context-culture-slow-v1` (0.388796 µm) take twice
their context's acquired pixel; W-393 generated no scene at either pixel, so **no learned
check of either preset was measured at the preset's pixel**. Each row gives the nearest
operation W-393 measured, on the sizing scene of the same grid (3 × 3 FOVs of 16 × 256 ×
256) at the acquired pixel, half the preset's, labelled as a measurement of that operation
and not an estimate of this one; the cost at the preset's pixel is measured by I5 under the
criterion quoted above. The culture chain check runs two scenes (seeds 42 and 43), so its
time is that of two calls; its peak RSS is that of one call, and whether holding the two
scenes in one process raises it is not measured either (I5). The numbers come from the
script `scripts/execution_costs.py` of W-482's run directory
(`runs/W-482/20261015T170720Z-694fab7e/`, output `results/costs/execution_costs.txt`).

| Chain check, preset | Learned operation | Nearest operation W-393 measured (`sizing.csv` tag, pixel) | Wall time | Peak RSS | GPU, for comparison |
| --- | --- | --- | --- | --- | --- |
| CT, `context-tissue2d-slow-v1` at 0.18927 µm | Export after fusion with StarDist `2D_versatile_fluo` at the scale 0.5 and the seeded watershed | `size_proj_t2_slow`, on `t2_slow` at 0.094635 µm (StarDist at the scale 0.25): **nearest operation**. At 0.18927 µm **not measured** (no scene at that pixel; at the scale 0.5 StarDist sees four times the pixels it saw there) | 22.0 s | 994,216 KiB | not runnable in the export (device rule) |
| CT, `context-tissue2d-slow-v1` at 0.18927 µm | StarDist `2D_versatile_fluo` alone | `size_sd2d_t2_slow_cpu`, at 0.094635 µm: **nearest operation**; at 0.18927 µm **not measured** (as above) | 11.79 s (0.92 s warm per call) | 882,300 KiB | `size_sd2d_t2_slow_gpu`: 19.56 s, 1,478,720 KiB, 490 MiB |
| CT, `context-tissue2d-slow-v1` at 0.18927 µm | Seeded watershed alone | `size_ws_t2_slow_cpu.rerun-20261015T032050Z`, at 0.094635 µm: **nearest operation**; at 0.18927 µm **not measured** (as above) | 0.96 s | 162,016 KiB | not runnable (CPU only) |
| CU, `context-culture-slow-v1` at 0.388796 µm, two scenes | Cellpose `cpsam_v2` nuclei per FOV | `size_cellpose_cu_slow_cpu`, on `cu_slow` at 0.194398 µm: **nearest operation**; at 0.388796 µm **not measured** (no scene at that pixel; the same FOV shape, with nuclei half as many pixels across) | 261.22 s per scene; 522.44 s for the two scenes | 2,428,836 KiB | `size_cellpose_cu_slow_gpu`: 59.2 s per scene, 3,387,024 KiB, 2,284 MiB |
| CU, `context-culture-slow-v1` at 0.388796 µm, two scenes | Extension through z | `size_extend_cu_slow_cpu`, at 0.194398 µm: **nearest operation**; at 0.388796 µm **not measured** (as above) | 13.67 s per scene; 27.34 s for the two scenes | 182,804 KiB | not runnable (no device) |
| CU, `context-culture-slow-v1` at 0.388796 µm, two scenes | Cellpose cell call on `Flamingo` and `DAPI` | **not measured** by W-393 at any pixel: not runnable on the current scene, which has no cytoplasm stain (W-393 `routes.csv`, rows `<scene>/cellpose_route`, `note` "cells not runnable"). Nearest operation: the nucleus call `size_cellpose_cu_slow_cpu` at 0.194398 µm, same grid and model | **not measured**; nearest operation 261.22 s per scene, 522.44 s for the two scenes | **not measured**; nearest operation 2,428,836 KiB | not measured; nearest operation `size_cellpose_cu_slow_gpu` 59.2 s per scene |

**Which gate runs what.**

* The routine tier (the fast tier R: the quick gate Q on every change and V at a batch's
  end and before merge) runs the routine generator checks: GN1 to GN22 on the tiny
  fixtures. They need no extra beyond the
  base, so the CI environment runs them too.
* The controller's `slow` selections run the `slow` generator checks (GN1(a) and (b) on
  `sample-small-v1`, the preset parts of GN4 to GN7, GN24 on the routine presets), which
  carry the subsystem marker `synthetic`, in its `pytest-slow-synthetic` selection, and the
  truth-label chains on the routine presets (`slow`, not `learned`), which carry
  `integration`, in `pytest-slow-other`; each runs when its paths change and at the last
  issue of a batch.
* The routine tier and the CI environment also run GA1 (routine presets) to GA21,
  on closed-form banks (GA22 is `slow` since W-576). The `slow` selection also runs GA8 to GA18 and GA24 on the routine
  appearance presets.
* The Chapter II evidence project runs, outside the test suite since W-536, GN23 and the
  `slow` and acquired cases of GN24 (`benchmarks/cellular_evidence.py`, "Implemented
  checks"), and GA23, the `slow` part of GA24, GA26 and GA27, which read the real banks
  through `STARFINDER_APPEARANCE_BANKS` (`benchmarks/appearance_evidence.py`;
  {doc}`synthetic-real-appearance-validation`, 'Implemented checks'). Since W-603 it also
  runs every chain check on the `slow` presets, with a model and on the truth labels
  (`benchmarks/cellular_chain_evidence.py`, "Implemented chain checks").
* The controller's `learned` selections run the chain checks with a model on the routine
  presets (`learned` and `slow`, with `integration`) in its two chain selections, the
  culture's and the other contexts', when their inputs change and at the last issue of a
  batch. The controller selects a check by the changed paths in the project
  configuration, which the operator keeps; the chain-check file must be among that
  selection's inputs (choice V10).
* The CI environment skips every chain check, naming the missing extra.

**Bounds.** The `slow` presets need the resource bound of C26 (RG1: one CPU, 16 GiB and
3,600 s per step). So do the appearance `slow` presets, their bank builds and their
chains, which is why these run in the evidence project. W-542 measured the chain file in
the suite on one CPU on 2026-10-26: its `slow` tier took 62.8 min with a 13.1 GiB peak in
the culture's chain selection, 22.3 min and 6.8 GiB in the other chain selection, and
35.6 min and 13.2 GiB for the truth-label chains; the routine tier took 4.4, 4.3 and
1.4 min with peaks of 3.0, 3.5 and 1.0 GiB. Before that measurement, their estimated sums were about 1,200 s (CD), 1,955 s (CT) and 1,805 s (CU, two
scenes) ({doc}`synthetic-real-appearance-validation`, 'Bounds'). The nearest dense 3D `slow` scene's generation peaked at 3,498,428 KiB
and the `3D_spleen` call alone at 3,263,160 KiB; a chain check holds both in one test
process, which can exceed the project's 4 GiB default and stays within RG1's 16 GiB. The
sum of W-393's measured operations for the dense 3D `slow` chain (generation 194.12 s and
`3D_spleen` 778.92 s) is below 3,600 s; the whole chain's time and peak RSS are measured by
the implementation (I5) under `/usr/bin/time -v`. For the two presets whose pixel changed,
the same sum adds the generation estimates of {doc}`synthetic-cellular-scene` ("Resources",
W-481: 131 to 155 s and at most 2.08 GiB for 2D tissue, 252 to 296 s and at most 3.98 GiB per
culture scene) to the nearest operations above, so it is an estimate, not a measurement:
153 to 177 s for the 2D tissue `slow` chain (generation and the export after fusion, which
holds StarDist and the watershed), and 1,576 to 1,664 s for the culture `slow` chain (two
generations, and per scene the nucleus call, the cell call counted at the nucleus call's
time, and the extension through z), both below 3,600 s. Held at once, every part of the
culture chain would add to 12.77 GiB, within RG1's 16 GiB and above the project's 4 GiB
default; I5 measures it.

## Coverage of the cellular scene's page

Every identity and guarantee of {doc}`synthetic-cellular-scene`, by section, with the
check that tests it or the reason it has none. Those of {doc}`synthetic-real-appearance`
are covered in {doc}`synthetic-real-appearance-validation` ('Coverage of the appearance
page'). The statement-by-statement list (each line
of W-480's re-run of W-394's audits C and E) is in the W-395 worker notes; for the page as
amended by W-481 (W-481's re-run of audit C and the guarantee search of audit E run again on
the amended page) it is `results/coverage/coverage_checklist.md` of W-482's run directory.

| Section of {doc}`synthetic-cellular-scene` | Identity or guarantee | Check |
| --- | --- | --- |
| "Opt-in" | A scene without the new settings is exactly today's output; `generate_sample_scene` and `save_sample_scene` unchanged; the golden test's pins | GN1 |
| "Opt-in" | Identity O1 | GN1 (b) |
| "Opt-in" | The addition is reached only through new names; no field added to `SampleSceneConfig`, `CellsConfig`, `StainConfig` | GN2 |
| "Names" | The new names, with their API pages | GN2; the reference audit |
| "Names" | `stains` given requires `sample.stain` None (`ValueError`) | GN3 |
| "Every new object is a function of the sample position" | Every new object evaluated at voxel centres at any fractional offset, without resampling | GN4 (labels), GN13 (stains, bright regions, field), GN18 (morphology round), GN19 (projection truth), GN17 (8-bit, per voxel) |
| "Every new object is a function of the sample position" | Molecules rendered by the unchanged formed generator | GN11 |
| "Every new object is a function of the sample position" | The noise drawn per FOV from the FOV's streams | GN21 (descriptors); the noise itself has no recomputation (it is not a point function) |
| "Nucleus shape" | `ρ_n` the minimum over lobes; the label rule changes only in `ρ_n` and the territory test | GN4 |
| "Nucleus shape" | LB1: every drawn nucleus one connected shape; the laws unchanged where the bound does not bind, never in Y and X for the presets; the bound a function of drawn values; `lobes.csv` records the bounded offset | GN5; GN4 (the shape rebuilt from the truth) |
| "Nucleus shape" | A label with several pieces keeps one ID; the identities hold voxel by voxel | GN4 (pieces counted); step 1 of the chain checks (counted once, choice V4) |
| "Nucleus shape" | N2: coefficients scaled to the amplitude; `harmonic_amplitude · M_Y < 1` else `ValueError`; `harmonics.csv` rebuilds the shape; N2 perturbs lobe 1 only | GN5, GN3, GN4 |
| "Nucleus shape" | The bounding site contains every lobe and is today's radius for one lobe | GN5 |
| "Nucleus shape" | The extent `e_n(û)` (a definition used by packing) | GN7 (recomputed in the contact distance) |
| "Nucleus shape" | Nuclei of different cells do not overlap | GN6 |
| "Packing" | K1: the acceptance tests; every centre strictly inside its own power cell; touching on a flat face | GN7, GN6 |
| "Packing" | P1: a contact proposal passes the K1 tests against its partner by construction (the `nextafter` loop) | GN7 (every accepted pair recomputed) |
| "Packing" | A binucleate seed; with `packing` None, today's rule | GN7; GN1 |
| "Packing" | Monolayer: every centre at `z_mid`; its `ValueError`s; no two nuclei or territories overlap in a Z projection, up to rounding within a face | GN19, GN3 |
| "Packing" | Density mode: kinds drawn; seeds added until `φ · ‖Ω‖`; `ValueError` without ordinary and binucleate shares; sizes per nucleus entity | GN7, GN3, GN21 |
| "Packing" | F2: stop and record, the floor raises; dense presets reproducible without retries | GN7 (`ctx_crowded`, presets), GN3 |
| "Cell territories" | T1, T2, T3; a nucleus lies inside its own cell; the anchor in its own territory | GN4, GN6 |
| "Cell types" | Ty2: the type from the region of the anchor; with no region Ty1 | GN8 |
| "Cell types" | MECR stays a per-cell count, unaffected by where cells are | none: a property of the report's MECR definition, which DF8 of {doc}`report-algorithms` checks; GN9 applies that definition to the truth |
| "Cell types" | G1's known answers: a marker gene only in its type's cells; the exclusive pair's truth MECR 0; only type-1 cells hold either co-expressed gene | GN8, GN9; step 4 of the chain checks |
| "Cell types" | The expected co-expressed ratio above 0.99 for 40 molecules or more | none: an analytic design value under an approximation; `marker_pairs.csv` gives the realized truth (GN9) |
| "Cell types" | Only molecules per cell, nuclear fraction and size scale may depend on the type | GN2 |
| "Molecules" | M2; `max_count` guards the total | GN11 |
| "Molecules" | NF1: the realized nuclear fraction is Binomial; a cytoplasm molecule with no room raises; an empty cell places all its molecules in its territory | GN12, GN3 |
| "Molecules" | A2: no ambient molecule has a marker gene | GN8 |
| "Stains" | S1; the nuclear stain formula; zero only beyond every nucleus's cut-off; the sign agrees with the label; the cut-off; skipping a site changes `E_n` by at most 3e-5 of the plateau | GN13 |
| "Stains" | Tx1: bright regions inside their nucleus | GN15; GN23 (TR5) |
| "Stains" | U1: the normalized field; `K_U = 0` gives `U = 1` | GN16, GN13 |
| "Stains" | U1: `log U` has standard deviation `a` at every point | none: holds by the normalization (GN16 states why) |
| "Stains" | Background and noise controls unchanged | GN13 (the background recomputed), GN21 (the noise streams) |
| "Stains" | The territory source: the sign agrees with `C`; the cut-off; lit by every cell, with or without a nucleus; under T3 only a scene without cells has points beyond every territory's cut-off | GN13 |
| "Stains" | The boundary source: the band at its peak on every boundary of `C = k`, the internal face of a binucleate cell included; `β_k` jumps only there and at the cut-off; no label image, distance transform or resampling; `g_k` lights the band only near a boundary; no signal beyond the cut-off | GN14 (the converse reported, since the page states no distance for "near"), GN13 |
| "Stains" | B1: `k` exists without objects; identity 10 | GN17 |
| "Stains" | O2: the boundary source implemented, no preset uses it | GN14, GN2 |
| "The morphology round" | Rendered at `true_f + τ_f + q ⊙ s`; `Ω` grows by `T ⊙ s`; `K` unchanged; the translation truth; the layout read by `Dataset`, `FOV.load_images` and `FOV.prepare_morphology` (MR1) | GN18, GN22; CU0 |
| "`Z=1`" | Every object in the plane z = 0; `τ_z = 0`; `plane` label images | GN20 |
| "The four cases, their controls and their ranges" | A generator check measures the edge width, the non-ellipsoid share, the bright regions and the other targets | GN23 |
| "Presets" | Six presets in `CELLULAR_PRESETS`, their PS1 values; PT1 (C19); SC1 (C20); Z1 (C21) | GN2, GN19, CT0, step 1 of the chain checks |
| "Presets" | The culture `slow` preset's seed stays 42, and the culture chain check runs at the seeds 42 and 43 (the culture table's row "Seed") | GN2 (the preset's seed); CU0 to CU4 on `chain_culture` ("Seeds") |
| "Presets", "Resources" | Each preset measured with `/usr/bin/time -v`; a generator test of 5 s or more is `slow`; the bound of C26 | GN24 |
| "Truth schema" | V1: the sample layout unchanged, twelve new tables, `context.json`; empty populations; float32 and uint8 images round trip; which truth holds each parameter | GN22, GN17; GN4, GN13, GN15, GN16, GN18 (each parameter rebuilt from its table) |
| "Truth schema" | Identities 1 to 11 | GN1 (1), GN4 (2), GN6 (3), GN8 (4), GN9 (5, 6), GN13 (7), GN18 (8), GN15 (9), GN17 (10), GN19 (11) |
| "Random streams" | The descriptor unchanged; an existing draw does not change; no new descriptor equals an existing one; a part off makes no draw; within a cellular scene a changed law changes no draw | GN21, GN1 |
| "Accepted statements this addition changes" | Entries 1 to 14 | The wording is applied to {doc}`synthetic-sample-scene` by the documentation issue (I6) and checked by its review against the list; no test checks wording. The numbers they state are tested: entries 3, 4 and 11 by GN18, entry 5 by GN6, entry 8 by GN4 and GN6, entry 13 by GN22 |
| "Accepted statements this addition changes" | The statement of {doc}`stitching-assembly-validation` that stays true | none: a statement about another page's wording; nothing is computed |
| "Open points for a later route issue" | No composite operation after fusion; one device per export | none here: these are route changes for a later route issue; the CT chain uses W-393's workaround (a composite per FOV, fused; the CPU for every run) and states it |
| "Exclusions" | No accuracy claim, no comparison, no calibration, no annotation | "The claim boundary" |

## Choices for the gate

Each is set out with its options and one recommendation in the W-395 worker notes; the
design above follows the recommendation.

| # | Choice | Recommendation |
| --- | --- | --- |
| V1 | Which presets the thresholds of steps 1 and 2 apply to | The `slow` presets (about 150, 76 and 42 nuclei per scene, the culture's two scenes pooled to about 83); the routine presets (about 7, 11 and 8 nuclei) run the chain with steps 3 and 4 asserted and steps 1 and 2 reported |
| V2 | Where the chain's molecules come from | (a) The truth positions, as W-393 and the no-segmenter path; the decoding chain stays checked by §2.11's `scene_chain` |
| V3 | The truth on a segmentation grid | The generator's point evaluator at the grid's pixel centres and the fused grid's planes (PT1), not the nearest sample-grid voxel |
| V4 | A predicted label covering a detached piece of a nucleus | Counted by the nucleus's ID, owned by it, not a false positive; the multi-piece nuclei reported |
| V5 | The placement of the routes per FOV | The stitch, with the truth placement as the stated fallback (SC1); no threshold until measured |
| V6 | False positives | Reported, no bound, until the first measurement on the new scene |
| V7 | The unassigned share | Reported, no bound, until the first measurement on the new scene |
| V8 | The exclusive pair's MECR after segmentation | Reported with the mixing cells; exact only on the truth-label chain |
| V9 | Identity 7's tolerance where a site was skipped | The bound of the site-skipping rule, `½ erfc(2√2)` of the local plateau, at the voxels the rule can affect |
| V10 | Marker and selection of the chain-check file | `integration` with `validation`, `slow`, `learned`, file `test_integration_cellular_chain.py`; the operator adds it to the inputs of the `learned` selection. Since W-603 the file holds the routine tier only; the `slow` tier runs in the Chapter II evidence project as `benchmarks/cellular_chain_evidence.py` ("Implemented chain checks") |
| V11 | The statistical tolerance of GN10 to GN12 | 4 SE |
| V12 | Target ranges measured on few objects | Compared only where the scene holds at least 100 objects of the kind; reported otherwise. The rule stays: of the `slow` presets GN23 measures, dense 3D tissue (about 150 nuclei) reaches it, 2D tissue (about 76) and culture (about 42) do not, nor do the variants `context_slow_acquired` (about 20 and 12) |
| V13 | The integer-shift tolerance of identities 7 and 8 | 1e-10 of the maximum, which covers the derived rounding bound of the full stains on the fixtures (7.3e-11 at a band of 0.01 µm, at most 1.3e-11 otherwise; 9.1e-11 and 1.6e-11 on grids of the two changed `slow` pixels, W-482); {doc}`synthetic-cellular-scene` now states 1e-10 for identities 7 and 8 (amended by W-481) |

## Exclusions

* **No implementation and no test code**: this page names checks; the issues that
  "Implementation" names write them.
* **No new measurement**: every number is W-393's, W-394's, W-480's or W-481's, cited; a number
  W-393 did not measure is named as missing.
* **No comparison of segmentation methods and no accuracy claim** ("The claim boundary").
* **No microsim and no simulation of raw acquisition**: that decision is taken at the gate
  W-396 from the W-392 result, and no check or issue here prepares it.
* **No real data in the repository, the routine tier or the CI environment; no new
  dependency, no lock change, no marker registration.**
* **Nothing is accepted by this page**, and no issue is created in Linear.

## Implementation

The generator checks of this page are in the modules that "Implemented checks" names, by
row. The chain checks run on the appearance presets, as {doc}`synthetic-real-appearance-validation`
amends them ("Consequences for the chain checks"), with the step 5 it adds.

**Issues.** The cellular scene was implemented by W-483 (geometry and molecules), W-484
(stains and the morphology round) and W-485 (the six context presets); W-486 completed its
generator checks. The real appearance was implemented by W-516 (the bank), W-517
(augmentation, placement, territories and labels), W-518 (composition, output, the
morphology round and truth), W-519 (the six appearance presets and the three real banks)
and W-520 (the remaining generator checks, GA23 and the table of implemented checks). W-487
ran the chain checks on the appearance presets. W-488 applied the lists of accepted
statements, W-483's four findings and the confirmations of the dated amendments, and added
these sections and the example `docs/examples/cellular_scene.py` ({doc}`api/examples`), run
by `test_synthetic_context_example.py`.

**Modules** (all in `src/python/starfinder/synthetic/`, exported by `starfinder.synthetic`
and listed on the API page `docs/api/synthetic.rst`):

* `_cellular.py` (W-483): the shape functions of lobed nuclei and harmonics, extents,
  contact distances and the label functions;
* `_context.py` (W-483, W-484): `CellularSceneConfig` and its parts (`NucleusShapeConfig`,
  `PackingConfig`, `TerritoryConfig`, `CellTypeConfig`, `MoleculeConfig`),
  `generate_cellular_scene`, `CellularScene`, `save_cellular_scene` and `save_cellular_dataset`;
* `_stains.py` (W-484): `StainChannelConfig`, `BrightRegionConfig`, `UnevennessConfig`,
  `OutputScaleConfig`, `MorphologyRoundConfig` and the stain functions;
* `_context_presets.py` (W-485): `CELLULAR_PRESETS`, `CELLULAR_PRESET_VERSION` and
  `cellular_scene_preset`;
* `_appearance_bank.py` (W-516): `BankReference`, `build_appearance_bank`,
  `generate_closed_form_bank` and `load_appearance_bank`;
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
  chains with a model are `slow` and `learned`, the truth-label chains `slow`; the suite
  runs them on the routine presets, and on the `slow` presets they run in the Chapter II
  evidence project as `benchmarks/cellular_chain_evidence.py` (W-603);
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
* **2026-10-18, W-488** (`docs/examples/cellular_scene.py`, {doc}`api/examples`). The example
  keeps the two presets the issue names, `context-tissue2d-v1` and `appearance-tissue2d-v1`,
  and its test `test_synthetic_context_example.py` is `slow` instead of routine: generating
  and saving the two presets takes more than 5 s on one CPU, the bound of a routine test,
  and that time is the package's. Measured on one CPU: the example 9.25 s, its test 8.57 s
  in the `pytest-slow-synthetic` selection and 8.55 s in the CI environment. The test runs
  in `pytest-slow-synthetic` and in the CI environment. Operator decision, confirmed by Jiahao on
  2026-10-18.
* **2026-10-19, GN1 (a)** (`test_synthetic_sample_golden.py`). The golden digests of the two
  sample presets compare the floating-point values of their CSV files and JSON records to 6
  significant digits, magnitudes below 1e-9 as 0, because NumPy's results differ in the last
  bit between CPUs (a GitHub runner's CSV files differed from the lab machine's only there).
  Integers, strings and pixel arrays stay exact; the pins were taken again at `8606450`, where
  the saved files equal those of 943bb6b exactly on the lab machine. The same applies to GA1
  ({doc}`synthetic-real-appearance-validation`). Jiahao's
  decision, 2026-10-19.

No threshold, tolerance, target range, preset or bank was changed by W-488.

**Accepted statements applied (W-488, 2026-10-18).** The fourteen entries of
{doc}`synthetic-cellular-scene`'s "Accepted statements this addition changes" are applied to
{doc}`synthetic-sample-scene`, with the completions of entry 32 of {doc}`synthetic-real-appearance`'s list; entries
1 to 34 of {doc}`synthetic-real-appearance`'s "Accepted statements this page changes" are
applied to {doc}`synthetic-cellular-scene`; and entries 1 to 38 of {doc}`synthetic-real-appearance-validation`'s list,
which take the place of entries 35 to 43 of {doc}`synthetic-real-appearance`, are applied to
{doc}`synthetic-cellular-scene-validation`. Each is applied with its proposed wording; the lists
stay on their pages as the record of what changed.

**Lists removed (2026-10-27).** The two lists of accepted statements of the
real-appearance pages are no longer on those pages: every entry is stated on the page it
changed, and the lists remain in the repository's history. The list of
{doc}`synthetic-cellular-scene` stays on its page.
