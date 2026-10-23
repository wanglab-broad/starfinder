# Synthetic cellular scene: nucleus shapes, packing, cell types, stains, the morphology round and the cellular presets of three contexts

Status: Accepted (W-396, 2026-10-15, at 5701cc2)

This page specifies the second delivery of §2.12: a cellular scene rich enough for the
real §2.9 segmentation methods to run on the synthetic sample, so that the chain from
stain to segmentation, assignment, assembly and report is checked against truth (D1 of
the W-152 comment "§2.12 planning decisions (2026-10-08)"). It builds on
{doc}`synthetic-sample-scene` (Accepted at W-329) and {doc}`synthetic-specification`,
and redefines nothing in them. Every statement of {doc}`synthetic-sample-scene` that the
addition would change is listed, with its old and proposed wording, in "Accepted
statements this addition changes"; that page is not edited. The validation design (the
chain checks and their thresholds) and the draft implementation issues belong to the next
issue of the batch (W-395); where this page needs one of them, it names the open point.

**Sources.** Statements about existing behaviour cite the file and function at the
batch's starting revision 943bb6b as `<file>::<function>`, with files under
`src/python/starfinder/` (for example `synthetic/_sample.py::_label_functions`).
Measurements cite the W-393 run directory `runs/W-393/20261015T003630Z-3fec2493/` under
the artifact root `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/`, as
"W-393" followed by a file and a row: for example W-393 `real/ln/geometry.csv` "share of
touching nuclei, per image" is that row of `results/real/ln/geometry.csv`, read as its
median and its 5th to 95th percentiles (written p5–p95). W-393 `notes.md` §n is a section
of that run's notes. Inputs and the microsim result come from W-392
(`runs/W-392/20261009T0323Z-inputs/`).

## What this addition is not

* It is **not a calibrated tissue model.** Nuclei are unions of ellipsoids clipped to
  power cells, cells are power-diagram territories, molecules are placed by compartment
  with a gene abundance per cell type, and stains are closed-form functions. Its
  parameters are **development targets** read from the W-393 measurements, **not a D04
  calibration** (D6). The real labels W-393 measured are method output (StarDist,
  Cellpose, the legacy watershed), read as measurements of shape and spacing, not as
  annotation or truth (W-393 `notes.md` §12). With the real appearance of
  {doc}`synthetic-real-appearance`, nuclei and culture cells are real method labels with
  their stain, augmented and placed by a synthetic rule; that scene is not a calibrated
  tissue model either.
* It **establishes no segmentation accuracy.** The §2.9 methods run on it as an
  engineering check of the chain, not as a measure of a method's quality; no accuracy is
  claimed, no methods are compared, nothing is trained. A method that fails on the scene
  is a finding about the scene (D1, D7).
* It does **not use microsim** or any new dependency (D5). The W-392 spike did not keep
  microsim as a stain renderer (W-392 `spike/results.md`); whether a simulation of raw
  acquisition is opened is decided at the gate W-396 and is not prepared here. Everything
  is rendered with NumPy and SciPy, which are locked; the real appearance of
  {doc}`synthetic-real-appearance` also uses scikit-image, which is locked.
* It has **no optical model, no simulation of raw acquisition and no attenuation with
  depth.** None was requested; W-393 found no need for depth attenuation (W-393
  `notes.md` §8, row "Attenuation with depth": not shown needed, not tested).
* It has no structured tissue background (the real appearance of
  {doc}`synthetic-real-appearance` adds one from real patches), no ER stain (D2), no illumination differences
  between FOVs, no stage drift beyond the accepted jitter, and no cross-FOV cell matching.

## Settled decisions this page follows

* **D1.** The real §2.9 methods run on the synthetic scene to check the chain against
  truth; an engineering check, not a calibrated tissue model.
* **D2.** The nuclear stain first. One stain interface by geometry source (nucleus
  interiors, territory interiors, a band at territory boundaries). A cytoplasm stain and a
  morphology round with a known translation are required for the culture context; a
  membrane stain is optional; ER is not done. The three contexts and their routes are
  those of the table of D2 (repeated in "Presets"). In the appearance presets of
  {doc}`synthetic-real-appearance` the stains are channels of the bank's entries, not
  geometry sources; the culture's cytoplasm stain and morphology round stay.
* **D3.** Both regimes, dense and sparse. The nuclear fraction of reads is a control
  (Jiahao: it depends on the tissue and the assay design). Cell types with a gene
  abundance per type. More molecules per cell, so that the composite fills the cell body.
* **D4.** The four cases where Jiahao's segmentations fail most: touching nuclei, uneven
  staining, high background, nuclei that are not one ellipse. Controls: packing density
  and edge width; a texture inside the nucleus as a few bright regions per nucleus
  (Jiahao, 2026-10-09) and a smooth multiplicative unevenness; the existing background
  controls; a nucleus as a union of ellipsoids, a spherical-harmonic perturbation as an
  optional second form. An 8-bit output. No depth attenuation unless W-393 shows a need.
  On 2026-10-16 Jiahao chose appearance sampled from real images for the texture, the
  background and the shapes ({doc}`synthetic-real-appearance`); the bright regions, the
  unevenness and the union of ellipsoids remain the closed-form presets' controls.
* **D6.** Targets measured in the three contexts by W-393, not a D04 calibration.
* **D7, D8.** Validation and its thresholds belong to W-395 and the gate W-396; this
  batch implements nothing.

## Choices for the gate

Each choice below has an options table with its cost and consequence and one
recommendation. This list is the index; the W-393 choice it answers is named where there
is one (W-393 `notes.md` §11).

| # | Choice | Recommendation | Section |
| --- | --- | --- | --- |
| C1 | How the addition is reached | E1, a new entry and configuration beside the unchanged sample scene | "Opt-in" |
| C2 | Nucleus shape (W-393 choice 9) | N1, a union of ellipsoids, with LB1's bound on the lobe offsets; N2, spherical harmonics, optional | "Nucleus shape" |
| C3 | Touching nuclei (W-393 choice 1) | K1, a shape-aware contact distance, nuclei clipped to their power cells | "Packing" |
| C4 | Placement at high density | P1, random sequential addition with contact proposals | "Packing" |
| C5 | A seed that cannot be placed | F2, stop at the reached density and record it, with a floor | "Packing" |
| C6 | Territories and the truth cell per context (W-393 choice 2) | T2 for the tissue contexts, T1 for the culture | "Cell territories" |
| C7 | How a cell gets its type | Ty2, by region with a purity | "Cell types" |
| C8 | Gene abundance and marker pairs | G1, markers per type, shared genes, named exclusive and co-expressed pairs | "Cell types" |
| C9 | Molecules per cell | M2, Poisson with a lognormal mean per cell | "Molecules" |
| C10 | The nuclear fraction of reads | NF1, a compartment draw per molecule | "Molecules" |
| C11 | The genes of ambient molecules | A2, an ambient abundance without the marker genes | "Molecules" |
| C12 | The stain interface | S1, one channel configuration with a geometry source | "Stains" |
| C13 | Texture (W-393 choices 3 and 8) | Tx1, Gaussian bright regions inside the nucleus | "Stains" |
| C14 | Unevenness across the field | U1, log-Gaussian bumps in the sample frame | "Stains" |
| C15 | 8-bit output (W-393 choice 4) | B1, scale to a plateau grey level, then cast | "Stains" |
| C16 | Optional stains in the first delivery | O2, the boundary source implemented, no preset uses it | "Stains" |
| C17 | The morphology round's translation | MT1, bounded uniform per FOV, with a supplied table | "The morphology round" |
| C18 | How the morphology round is registered | MR1, a reference stain sharing the nuclear stain's name | "The morphology round" |
| C19 | Truth of the 2D tissue projection | PT1, a monolayer (every centre in the stack's mid-plane), truth from the sampled planes | "Presets" |
| C20 | Stain content for the stitch (W-393 choice 5) | SC1, presets dense enough, truth placement as a stated fallback | "Presets" |
| C21 | Z extent (W-393 choice 6) | Z1, keep 16 planes and count nuclei by the Y/X border | "Presets" |
| C22 | Truth schema | V1, new files with their own version | "Truth schema" |
| C23 | Parameter set of the dense 3D tissue presets | PS1, the proposed set | "Presets" |
| C24 | Parameter set of the 2D tissue presets | PS1, the proposed set | "Presets" |
| C25 | Parameter set of the culture presets | PS1, the proposed set | "Presets" |
| C26 | The resource bound of the `slow` presets | RG1, grant the W-393 bound to the implementation batch | "Presets", "Resources" |

## Terms

* The **sample position** `σ` is a point of the sample frame in µm, ZYX, as in
  {doc}`synthetic-sample-scene` ("Terms"). A FOV `f` with true origin `true_f` and
  spacing `s` images, at index `p`, the sample position `true_f + p ⊙ s`
  (`synthetic/_sample.py::generate_sample_scene`, the `fov_points` and `_fov_labels`
  call).
* A **point function** is a function of `σ` alone, once the scene's draws are made. Every
  new object of this page is one. A FOV evaluates it at its voxel centres
  `true_f + p ⊙ s`, at any fractional offset, without resampling; the sample grid
  evaluates the same function at its own voxel centres.
* A **lobe** is one ellipsoid of a nucleus; lobe 1 is the ellipsoid of today's nucleus.
* A **site** is the bounding sphere of one nucleus (or of an empty cell), as today; its
  **power cell** is the set of points whose power distance `‖σ − c_j‖² − R_j²` is smallest
  for that site (ties to the smaller site index), as in `synthetic/_sample.py::_label_functions`.
* A **territory** is the set of points of one cell's label.
* A **geometry source** is the set a stain channel lights up: `nucleus` (nucleus
  interiors), `territory` (territory interiors) or `boundary` (a band at territory
  boundaries).
* A **morphology round** is an additional acquisition round holding stain channels only,
  acquired with a known translation against the reference round.
* A **context** is one of the three real settings of D2: dense 3D tissue (LN sample1),
  2D tissue (tissue-2D) and sparse culture (cell-culture-3D).

## Opt-in

A scene without the new settings is exactly today's output. The addition is reached
only through new names; `generate_sample_scene`, `save_sample_scene`,
`sample_scene_preset`, `SampleSceneConfig`, `CellsConfig`, `StainConfig` and
`SAMPLE_PRESETS` keep their behaviour, fields and values.

**Why a new configuration.** `SampleSceneConfig`'s `asdict` form is the
`requested_config` of every saved `sample.json` (`synthetic/_sample.py::_config_payload`,
called by `_assemble`), so a new field on it, on `CellsConfig` or on
`StainConfig`, even one defaulting to None, would add a key to `sample.json` of both
existing presets.

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **E1. A new entry and configuration (recommended)** | `generate_cellular_scene(codebook, *, config=CellularSceneConfig(...), on_fov=None)`; `CellularSceneConfig` holds an unchanged `SampleSceneConfig` and the new parts. The sample-scene internals are shared as private functions. | One more entry; the internals of `synthetic/_sample.py` are split into shared private functions, which the golden test guards. | `generate_sample_scene` and its saved files stay unchanged by construction; the golden test pins their content. The new truth lives in new files with their own version (C22). |
| E2. New optional fields, omitted from the payload when None | `CellsConfig` and `StainConfig` gain fields defaulting to None; `_config_payload` drops None-valued new keys. | A special case in the payload; `requested_config` no longer equals `asdict`. | Today's files stay unchanged only by that special case, and every reader of `requested_config` must know it. |
| E3. New fields and a new truth version | Add the fields; `SAMPLE_TRUTH_VERSION` becomes `"2"`. | Simplest code. | Both existing presets' `sample.json` change: a scene without the new settings is no longer today's output, against the issue. |

**Recommendation: E1.**

**What a scene without the new settings produces.** Two statements, both checkable:

1. `generate_sample_scene` and `save_sample_scene` are unchanged: for `sample-tiny-v1`
   and `sample-small-v1`, every saved file has the content pinned by the golden test
   `test/test_synthetic_sample_golden.py` (thirteen groups per preset: the seven
   sample tables, the sample labels, the FOV images, stains, labels and tables, and the
   JSON records without `numpy_version` and without the file map). One temporary edit of
   the stain kernel made both presets fail it (worker notes of W-394).
2. **Identity O1.** `generate_cellular_scene(book, config=CellularSceneConfig(sample=c))`,
   with every new part at its default (None or empty), returns a `CellularScene` whose
   `sample` equals `generate_sample_scene(book, config=c)` field for field (arrays equal,
   tables equal with `check_exact`, `record` equal), and saved, its `sample.json` and
   every file of the sample layout equal those of `save_sample_scene`.

**Existing tests that must pass unchanged:** every test {doc}`synthetic-sample-scene`
names in "Opt-in, and what stays as it is"; `test/test_synthetic_sample_scene.py`; the
golden test; and the §2.10 and §2.11 tests that generate `sample-tiny-v1`
(`test/report_scene.py`, `test/test_report_chain.py`).

## Names

All new, in `starfinder.synthetic`, with the API pages that `docs/check_reference.py`
requires:

| Concept | Name | Kind |
| --- | --- | --- |
| Entry | `generate_cellular_scene(codebook, *, config=CellularSceneConfig(), on_fov=None)` | function |
| Result | `CellularScene` | dataclass |
| Configuration | `CellularSceneConfig`, `NucleusShapeConfig`, `PackingConfig`, `TerritoryConfig`, `CellTypeConfig`, `MoleculeConfig`, `StainChannelConfig`, `BrightRegionConfig`, `UnevennessConfig`, `OutputScaleConfig`, `MorphologyRoundConfig` | frozen dataclasses |
| Persistence | `save_cellular_scene(scene, directory)`, `save_cellular_dataset(scene, directory)` | functions |
| Presets | `cellular_scene_preset(name)`, `CELLULAR_PRESETS`, `CELLULAR_PRESET_VERSION` (`context-presets-v1`) | function, dict, str |
| Version | `CELLULAR_TRUTH_VERSION` (`"1"`) | str |

```python
@dataclass(frozen=True)
class CellularSceneConfig:
    sample: SampleSceneConfig = SampleSceneConfig()   # template, acquisition, seeds and size laws, molecules
    nuclei: NucleusShapeConfig | None = None          # lobes and harmonics ("Nucleus shape")
    packing: PackingConfig | None = None              # density, contact, overlap ("Packing")
    territories: TerritoryConfig | None = None        # territory rule ("Cell territories")
    types: CellTypeConfig | None = None               # cell types and abundances ("Cell types")
    molecules: MoleculeConfig | None = None           # counts, nuclear fraction, ambient genes ("Molecules")
    stains: tuple[StainChannelConfig, ...] = ()       # empty: the sample scene's StainConfig ("Stains")
    morphology: MorphologyRoundConfig | None = None   # the additional round ("The morphology round")

@dataclass
class CellularScene:
    sample: SampleScene                               # the sample-scene result, its tables and labels
    stains: dict[str, dict[str, np.ndarray]]          # fov_id -> stain name -> ZYX image (reference round)
    morphology: dict[str, np.ndarray]                 # fov_id -> ZYXC image of the morphology round
    lobes: pd.DataFrame                               # "Truth schema": one table per file of that section
    harmonics: pd.DataFrame
    cell_context: pd.DataFrame
    regions: pd.DataFrame
    type_abundances: pd.DataFrame
    cell_gene_counts: pd.DataFrame
    marker_pairs: pd.DataFrame
    stain_channels: pd.DataFrame
    stain_objects: pd.DataFrame
    bright_regions: pd.DataFrame
    unevenness: pd.DataFrame
    morphology_truth: pd.DataFrame
    record: dict                                      # versions, configuration, streams, digests
```

When `stains` is not empty, `sample.stain` must be None (`ValueError` otherwise): the
context's first `nucleus` stain of the reference round takes the place of the sample
scene's stain, as the `reference_stain` input of `stitch` and the nuclear image.

## Every new object is a function of the sample position

The accepted scene already evaluates its labels and its stain at voxel centres from
point functions (`synthetic/_sample.py::_label_functions`, `_stain_signal`), so its
labels are exact at fractional offsets (W-392 `spike/results.md`: the package's labels and
labels recomputed from the truth tables agree in every voxel at a fractional offset).
Every new object keeps that property:

| Object | Point function of `σ` | How FOV `f` evaluates it at a fractional offset |
| --- | --- | --- |
| Nucleus label `N` with lobes, harmonics and clipping | `N(σ)` ("Nucleus shape", "Packing") | at `true_f + p ⊙ s` for every index `p`; no raster, no resampling |
| Cell label `C` with the territory rules | `C(σ)` ("Cell territories") | as `N` |
| Cell type | a value per cell; with regions, the region of the cell's anchor, `region(anchor)` | a property of the cell, carried by `C`; nothing is rendered |
| Molecules | positions `σ_m` in µm, drawn once in the sample frame | at `p = (σ_m − true_f) ⊘ s` by the unchanged formed generator, as today (`synthetic/_sample.py::generate_sample_scene`) |
| Nuclear stain signal (plateau, soft edge, contact faces) | `S_nuc(σ)` ("Stains") | at `true_f + p ⊙ s`, then the existing observation stages with the FOV's streams |
| Bright regions | `T_n(σ)`, Gaussians with centres in µm | inside `S_nuc`, at the same points |
| Unevenness | `U(σ)`, centres and widths in µm | multiplies the signal at the same points; continuous across FOVs |
| Cytoplasm and membrane signals | `S_ter(σ)`, `S_bnd(σ)` ("Stains") | at `true_f + p ⊙ s`; the band from closed forms, not from a distance transform of a label image |
| Background components of each stain | `BackgroundConfig` records on the sample grid, as today | `evaluate_background_translated` with the FOV's offset, as `generate_sample_scene` does for the stain today |
| Morphology round image | the stain functions above; the round's index `q` in FOV `f` images `true_f + τ_f + q ⊙ s`, with a known translation `τ_f` | at those points; no resampling ("The morphology round") |
| 8-bit output | a scale and a cast of each voxel's value | per voxel, after the noise |
| Projection truth of the 2D tissue presets | `P(y, x) = max_q N(z_q, y, x)` over the Z planes `z_q` of the grid it is compared on | at that grid's pixel centres and planes, so it is defined by the sampled planes, at any fractional offset in Z, Y and X ("Presets", C19) |

The noise is the one thing that is not a point function: it is drawn per FOV from the
FOV's streams, as today, and is independent between overlapping FOVs
({doc}`synthetic-sample-scene`, "Rendering each FOV at its true offset"). In a scene of
{doc}`synthetic-real-appearance` the labels are the placed masks under one placement
order, and every stain (the culture's Flamingo included) is the composition of placed
entries, background patches and out-of-focus light, each a point function of `σ`. The
morphology round evaluates them at its shifted points, and the 8-bit output scales by `g`
with no noise before the cast. `P` holds for its 2D tissue and is reported for its culture.
No noise is drawn there; what is not a point function is the placement test and `κ`, each
decided once on the sample grid ('Truth' there).

## Nucleus shape

Today a nucleus is one ellipsoid: with `d = σ − c_n`, `u = cos θ · d_y + sin θ · d_x`,
`v = −sin θ · d_y + cos θ · d_x`, its normalized radius is
`ρ(σ) = sqrt((d_z / a_z)² + (u / (a_l e))² + (v / a_l)²)` and the nucleus is `ρ ≤ 1`
(`synthetic/_sample.py::_rho`); its site sphere has radius `max(a_z, a_l e)`
(`_cellular_scene`). The real nuclei are mostly near one ellipsoid, with a tail that is
not: the share of nuclei whose ellipsoid fit has an IoU below 0.8 is 0.068 per image in
LN (p5–p95 0.064–0.074), 0.038 in tissue-2D (0.027–0.060) and 0.24 in the culture
(0.12–0.43), with solidity p5 0.88, 0.90 and 0.83 (W-393 `real/<context>/geometry.csv`
"share of nuclei not described by one ellipsoid (IoU < 0.8), per image" and "convexity
(solidity)").

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **N1. A union of ellipsoids (recommended)** | Nucleus `n` has `L_n` lobes; lobe 1 is today's ellipsoid; lobe `l ≥ 2` has centre `c_n + o_l`, semi-axes, elongation and angle of its own. `ρ_n(σ) = min_l ρ_{n,l}(σ)`; the shape is `ρ_n ≤ 1`. | `L_n` evaluations of `ρ` per candidate point instead of one; about 1.1 to 1.3 times today's label cost at the target lobed shares. Closed form. | Lobed and concave nuclei (solidity below 1, ellipsoid IoU below 0.8), concave where the lobes meet; the share of such nuclei is one parameter. A point function, evaluated at the voxel centres at any fractional offset (identity 2). |
| N2. A spherical-harmonic perturbation (optional second form) | Lobe 1 becomes `‖d'‖ ≤ r(d̂')`, with `d'` the point in lobe 1's normalized frame (`d'_z = d_z / a_z`, `d'_u = u / (a_l e)`, `d'_v = v / a_l`), `d̂' = d' / ‖d'‖` and `r(d̂) = 1 + Σ_{l=2..L} Σ_m a_lm Y_lm(d̂)` with real orthonormal harmonics; in one plane or a monolayer, `r(φ) = 1 + Σ_{k=2..L} (a_k cos kφ + b_k sin kφ)` in the YX angle. | Real harmonics by `scipy.special` (locked) per candidate point; amplitudes bounded so that `r > 0`. | Smooth bumps and dents on every nucleus; star-shaped only, so no deep cleft between lobes. A point function, as N1. |
| N3. A thresholded random field per nucleus | Smooth noise added to `ρ`, thresholded. | Needs a noise field defined in the sample frame (not the per-FOV noise of the observation model). | Irregular outlines, but the shape depends on how the field is defined between voxel centres; no closed-form bound for the site sphere. |

**Recommendation: N1 first; N2 as an optional second form** for the closed-form presets
(W-393 choice 9: the real non-ellipsoid share is 4 to 7 % in LN and tissue-2D and 24 % in
the culture); in the appearance presets of {doc}`synthetic-real-appearance` a nucleus is a
placed real mask.

```python
@dataclass(frozen=True)
class NucleusShapeConfig:
    lobed_share: float = 0.0                       # probability that a nucleus has more than one lobe
    extra_lobes: tuple[int, int] = (1, 2)          # lobes beyond the first: an integer drawn uniformly in this range
    lobe_scale: ScalarDistribution = ScalarDistribution("uniform", (0.5, 0.8))  # semi-axes relative to lobe 1
    lobe_offset: ScalarDistribution = ScalarDistribution("uniform", (0.5, 0.9)) # λ: distance / lobe 1's lateral semi-axis,
                                                                                #   bounded by the two extents (LB1)
    harmonics: int = 0                             # N2: largest degree L; 0 is off
    harmonic_amplitude: float = 0.0                # N2: Σ abs(a_lm) is scaled to this value; 0 ≤ it · max abs(Y_lm) < 1
```

**Lobe offsets and one connected shape.** Lobe `l ≥ 2` of nucleus `n` has a direction
`û_l`, an angle, semi-axes `k_l` times lobe 1's (`k_l` from `lobe_scale`) and the centre
`c_n + o_l`, `o_l = D_l û_l`. With the distance written before W-480, `D_l = λ_l · a_l` (`λ_l`
from `lobe_offset`, `a_l` lobe 1's lateral semi-axis), the two lobes need not meet: the dense 3D
laws allow `a_z` 1.5 µm, `a_l` 3 µm, one lobe scaled by 0.6 at `0.85 · 3 = 2.55` µm along
Z, and with the centre at z = 1.3 the two lobes occupy the Z intervals [−0.2, 2.8] and
[2.95, 4.75], 0.15 µm apart (nearby directions fail too).

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **LB1. Bound each offset by the two extents (recommended)** | `D_l = min(λ_l · a_l, 0.9 · (e_{n,1}(û_l) + ε_l(−û_l)))`, with `e_{n,1}(û)` the length of the ray `c_n + s û`, `s ≥ 0`, inside lobe 1 (`1 / ‖d'(û)‖`, times `r(d̂'(û))` under N2) and `ε_l(−û)` the length of the ray `c_n + o_l − s û` inside lobe `l` (the same expression in lobe `l`'s frame). | Two extents per lobe, once per nucleus. | Every drawn nucleus is one connected shape. The laws are unchanged wherever the bound does not bind: never in Y and X for the presets (below), and in 3D only where the two extents sum to less than `λ_l · a_l / 0.9`, for a lobe pointing close to a short axis. |
| LB2. The offset as a fraction of the two extents | `D_l = λ_l · (e_{n,1}(û_l) + ε_l(−û_l))`, with `λ_l < 1` required of the law. | As LB1. | Connected by construction; `lobe_offset` changes its meaning and every lobed shape of the presets changes. |
| LB3. No bound; narrow the claim | `D_l = λ_l · a_l`, as written before W-480. | None. | A drawn nucleus can be two shapes; the truth would record it, and the generator check would count such nuclei as a share that depends on the laws. |

**Recommendation: LB1.** Lobe 1 contains the segment from `c_n` to
`c_n + e_{n,1}(û_l) û_l` (it is convex, or star-shaped about `c_n` under N2), and lobe `l`
the segment from `c_n + o_l` back to `c_n + (D_l − ε_l(−û_l)) û_l`. The bound gives
`D_l − ε_l(−û_l) < e_{n,1}(û_l)`, so the two segments share a stretch of length at least
`0.1 · (e_{n,1}(û_l) + ε_l(−û_l)) > 0`: every lobe meets lobe 1, and the union of the lobes,
before clipping, is one connected shape for every draw the laws allow. In the case above
the bound gives `D_2 = 0.9 · (1.5 + 0.9) = 2.16` µm, and the lobes overlap by 0.24 µm in Z.
In Y and X (one plane or a monolayer) the bound never binds for laws with an elongation of
at least 1, `lobe_scale` at least 0.5, `lobe_offset` at most 0.9 and N2 off (every preset):
`e_{n,1} ≥ a_l` and `ε_l ≥ 0.5 a_l`, so the bound is at least `1.35 a_l > 0.9 a_l`. The bound
is a function of drawn values only, so the shape stays a point function; `lobes.csv` records
the bounded offset, from which the shape is rebuilt without the rule. Audit F of W-480
(`results/audit/nucleus_draws.txt` in `runs/W-480/20261015T070249Z-868bf70b/`) draws 20,000
nuclei per preset (10,000 with the preset's lobed share, 10,000 all lobed), nuclei at the
ends of the laws and the case above, and finds none that is not one connected shape. Under
the rule written before W-480, 504 of the 8,064 dense 3D nuclei at the ends of the laws and
4 of the 20,000 dense 3D draws with every nucleus lobed are not.

**A nucleus label with more than one piece.** The label is the drawn shape clipped to the
nucleus's power cell and to `Ω` and sampled at voxel centres. Both clips are convex, but a
neighbour's power face can cross the line of centres between lobe 1 and lobe `l` while part
of lobe `l` stays on the nucleus's side, and a FOV's border or a thin neck between voxel
centres can split a label too. A nucleus label can therefore have more than one piece. For
the truth, the nucleus keeps one ID for all its pieces (one row of `nuclei.csv`, its lobes
in `lobes.csv`), and the label identities (2, 3 and 11 of "Truth schema") hold voxel by
voxel whatever the number of pieces. For the checks, a nucleus is counted and matched once,
by its ID with all its voxels: the generator check of the non-ellipsoid share measures the
ID, and W-395's nucleus matching (intersection over union by ID) states how it counts a
predicted object that covers a detached piece only (an open point of W-395). On packed
scenes of the six presets (68,967 nuclei, at least 11,278 per preset), audit F found 5
nuclei whose label has more than one piece on a grid of a twelfth of the smallest
semi-axis. On a grid four times finer (twice in 3D), 3 of them are one piece and 2 keep a
sliver of one or two grid points beside the main piece.

**N2 in detail.** The coefficients of nucleus `n` are drawn from the stream
`context.nucleus.harmonics` (entity `nucleus-<n>`), one standard normal per `(l, m)` in
order of degree then order (per `(k, cos)`, `(k, sin)` in one plane), and scaled so that
`Σ abs(a_lm) = harmonic_amplitude`; with `M_Y = max_{l ≤ L} max abs(Y_lm)`,
`harmonic_amplitude · M_Y < 1` makes `r > 0` everywhere (`ValueError` otherwise). The real
harmonics are `Y_lm = sqrt(2) (−1)^m Re(Y_l^m)` for `m > 0`, `Y_l^0` for `m = 0` and
`sqrt(2) (−1)^m Im(Y_l^{abs(m)})` for `m < 0`, from `scipy.special.sph_harm_y` (locked). At
the centre (`d' = 0`) the nucleus contains the point by definition. The coefficients are
truth: `harmonics.csv` holds `nucleus_id`, `degree`, `order` (`cos` or `sin` in one plane)
and `coefficient`, so the shape is rebuilt from `nuclei.csv`, `lobes.csv` and
`harmonics.csv`. N2 perturbs lobe 1 only; further lobes stay ellipsoids.

**Label rule.** The nucleus label keeps today's rule (`_label_functions`): at `σ`, take
the site `j*` of smallest power distance (ties to the smaller site index); `N(σ) = n` when
`j*` is a site of nucleus `n`, `σ ∈ Ω`, `σ` passes the cell's territory test ("Cell
territories"), and `ρ_n(σ) ≤ 1`; else 0. Besides the territory test, which T2 and T3
change, only `ρ_n` changes: the minimum over lobes, with
lobe 1 in the harmonic form `ρ_{n,1} = ‖d'‖ / r(d̂')` under N2 (and 0 at `d' = 0`). In the
appearance presets of {doc}`synthetic-real-appearance`, `N(σ)` is the first placed object
that holds `σ` (its signed distance at most 0), else 0; in the culture, the nucleus of the
first placed cell that holds `σ`, where that nucleus's mask holds it.

**Bounding site.** Nucleus `n`'s site sphere has centre `c_n` and radius
`R_n = max_l (‖o_l‖ + max semi-axis of lobe l)`, which contains every lobe; for N2,
`R_n` uses lobe 1's largest extent `max(a_z, a_l e) · (1 + harmonic_amplitude · M_Y)`. Lobe
1's offset is 0, so `R_n` is today's radius when a nucleus has one lobe and no harmonics.
Lobe directions are drawn in the YX plane when the scene has one plane or a monolayer
(C19), and in 3D otherwise. In the appearance presets of {doc}`synthetic-real-appearance`,
`site_radius` is the radius of the placed mask's bounding sphere about its centre, and
the shape columns are null (AT2).

**The extent along a direction.** `e_n(û)` is the length of the run of the ray
`c_n + s û`, `s ≥ 0`, that stays inside the nucleus shape from its centre: each lobe the
ray meets gives one interval `[s_in, s_out]` of `s` (the two roots of a quadratic per
ellipsoid; for lobe 1 under N2, `[0, r(d̂') / ‖d'(û)‖]` with `d'(û)` the normalized image
of `û`), and `e_n(û)` is the end of the union of these intervals that contains `s = 0`.
Every point `c_n + s û` with `0 ≤ s ≤ e_n(û)` lies in the shape. Lobe 1 contains `c_n`, so
`e_n(û) > 0` for every `û`. Packing uses it ("Packing").

**Nuclei of different cells do not overlap.** A point carries the nucleus of the site of
smallest power distance only, so two nuclei can never share a point, whatever the lobes
or the overlap of their spheres: the label is one value per point by construction. Where
two shapes reach across the face between their power cells, each is cut by that flat
face, and the two nuclei touch on it ("Packing"). In the appearance presets of
{doc}`synthetic-real-appearance` a point carries the nucleus of the first placed object
that holds it, so two nuclei never share a point either; masks may overlap by up to 1 %
of a mask, the earlier owning the overlap, and touching nuclei meet on their real
outlines, not on a flat face.

## Packing

Today seeds are placed by random sequential addition in `Ω`, each proposal accepted
when its site sphere keeps a positive `gap` to every accepted sphere, and a seed with no
accepted proposal after `max_proposals` raises `ValueError`
(`synthetic/_sample.py::_cellular_scene`, `_clear`; `CellsConfig.__post_init__`
requires `gap > 0`). The site spheres are therefore disjoint, so no two nuclei touch: W-393
measured a touching share of 0 on every scene, also at gap 0.05 µm, against 0.52 in LN,
0.79 in tissue-2D and 0.50 in the culture (W-393 `notes.md` §3 and §6,
`tables/scene_measurements.csv`). The generator also
refused 30 to 300 seeds on dense variants because no proposal fitted (W-393 `notes.md`
§13, "Generator refusals").

**The contact distance.** For an accepted site `i` and a new site `k` whose centre lies
in the direction `û` from `c_i`, with `Δ = R_i² − R_k²`, the face of their power cells
crosses the line of centres at the distance `x(d) = (d² + Δ) / (2d)` from `c_i`, for a
centre distance `d`. Nucleus `i` reaches that face along the line when `e_i(û) ≥ x(d)`
(every point of the segment from `c_i` to the face is then in its shape) and nucleus `k`
when `e_k(−û) ≥ d − x(d)`. The **contact distance** is the largest `d` at which
both hold,

```
t_ik(û) = min( e_i(û) + sqrt(e_i(û)² − Δ),  e_k(−û) + sqrt(e_k(−û)² + Δ) )
```

when both radicands are nonnegative, and `t_ik(û) = 0` otherwise (the two shapes cannot
both reach the face along this line). For an empty cell, its virtual sphere stands in
for the shape (`e = empty_radius` in every direction). At any centre distance
`d ≤ t_ik(û)` with `d² > abs(Δ)`, both nuclei cross their common face on the line of
centres, so they touch on that face after clipping. (Each condition holds on an interval
of `d` between the two roots of a quadratic whose product is `±Δ`; the smaller root is
at most `sqrt(abs(Δ))`, so `d² > abs(Δ)` and `d ≤ t_ik(û)` together put `d` inside both
intervals.)

**Touching nuclei (C3).**

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **K1. A shape-aware contact distance, nuclei clipped to their power cells (recommended)** | Accept a proposal `c` of site `k` when, for every accepted site `i` (direction `û` from `c_i` to `c`), `‖c − c_i‖ ≥ (1 − κ) · t_ik(û) + g`, and `pow_k(c) < pow_i(c)` and `pow_i(c_i) < pow_k(c_i)` computed with the label function's arithmetic (`_label_functions`; mathematically `‖c − c_i‖² > abs(R_i² − R_k²)`), with the overlap allowance `0 ≤ κ < 0.5` and the gap `g ≥ 0` (µm). The label rule is unchanged ("Nucleus shape"). | No new label code: `_label_functions` already assigns the nucleus of the power-nearest site only. Two extents per accepted neighbour within reach (`‖c − c_i‖ < R_i + R_k + g`; farther sites pass both tests, since `t_ik ≤ R_i + R_k` and `R_i + R_k > sqrt(abs(Δ))`). | With `g = 0` and `κ > 0`, a site placed at its contact distance touches its partner on a flat shared face, as touching nuclei do in real label images. The second test keeps every centre strictly inside its own power cell, so every nucleus contains its centre and every anchor lies in its own territory, which the sample membership rule needs ({doc}`synthetic-sample-scene`, "Sample membership"). Labels stay point functions. |
| K2. Random sequential addition, then a compression step | Place with a positive gap, then move the centres towards each other (a relaxation) until neighbours touch. | An iterative step with its own convergence rule and streams. | Contacts are tangent points, not faces; spheres must still not overlap, so ellipsoids touch rarely. |
| K3. Keep a positive gap | Today's rule. | None. | No touching nuclei: one of the four cases of D4 is absent. |

**Recommendation: K1** (W-393 choice 1, option (a), with the contact measured on the
shapes rather than on their bounding spheres: with spheres, two elongated nuclei placed
side by side along their short axes would not reach their common face).

**Placement at high density (C4).**

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **P1. Random sequential addition with contact proposals (recommended)** | For each proposal of seed `k`: with probability `contact` (and at least one accepted site), choose an accepted site `j` uniformly and a direction `û` uniformly on the sphere (on the circle in Y and X for one plane or a monolayer), drawn again while `t_jk(û) = 0`, at most 16 times; with a direction found, propose `c = c_j + D û` with `D = max((1 − κ) · t_jk(û) + g, sqrt(abs(R_j² − R_k²)))`, increased by `nextafter` steps until the K1 tests against `j`, evaluated exactly as acceptance evaluates them (with the direction recomputed from `c`), hold, as `_cellular_scene` does today for the two nuclei of a binucleate seed. Otherwise (no contact draw, or no direction after 16 draws) propose uniformly in `Ω` as today. Accept by the K1 test against every accepted site; a contact proposal outside `Ω` is rejected. | One uniform draw per proposal, and an index and up to 16 directions for a contact proposal; nothing iterative beyond the `nextafter` steps. | For a seed with one site, a contact proposal passes the K1 test against its partner by construction, for every allowed `κ` and `g` (a binucleate seed's contact proposal gives its centre `c`, and its two nuclei are then tested like any proposal); it touches its partner when `D ≤ t_jk(û)`, which holds when `g < κ · t_jk(û)` and `sqrt(abs(Δ)) < t_jk(û)` (the `nextafter` steps are then smaller than the margin). `contact` sets how often seeds are proposed against a neighbour; the resulting touching share is measured on the truth labels with W-393's measure. |
| P2. Random sequential addition, then relaxation | As K2. | Iterations over all pairs. | Higher density than P1 can reach, at the price of a procedure whose result depends on its stopping rule. |
| P3. A jittered lattice | Centres on a hexagonal lattice with random offsets. | Cheapest. | A regular pattern that real tissue does not have; neighbour distances are nearly constant. |

**Recommendation: P1.** Random sequential addition of disjoint spheres saturates near a
volume fraction of 0.38 in 3D (0.55 for disks in a plane); the K1 exclusion is the
contact distance shrunk by `1 − κ`, which is smaller than the bounding spheres, so the
nucleus volume fraction of LN (0.13, W-393 `real/ln/geometry.csv` "nucleus volume
fraction, per image") and the area fractions of tissue-2D (0.36) and the culture (0.26)
are within reach; audit A of W-394 (`results/audit/rules_vs_presets.txt`) evaluates each
preset. In the appearance presets, placement is the rule PL2 of
{doc}`synthetic-real-appearance`: masks overlap by at most 1 % of their volume, and the
earlier object owns the overlap.

A binucleate seed places its two nuclei at `c ± (D_12 / 2) û_φ`, with `c` its proposed
centre, `û_φ = (0, sin φ, cos φ)` from today's `sample.cell.placement` draw of `φ`, and
`D_12` the distance of P1 between its two nuclei along `û_φ` (with `nextafter` steps until
the K1 tests between the two hold); both sites are then tested against the accepted
sites. With `packing` None this is today's rule (`D_12 = R_1 + R_2 + gap`,
`_cellular_scene`). The appearance presets of {doc}`synthetic-real-appearance` have no
binucleate and no empty cells: every placed object is one ordinary cell with one nucleus
('Kinds of cells' there).

```python
@dataclass(frozen=True)
class PackingConfig:
    density: float | None = None     # target nominal nucleus volume fraction of Ω; area fraction of Ω's Y-X box
                                     #   with one plane or a monolayer; 0 < density < 1
    kind_shares: tuple[float, float, float] | None = None   # ordinary, binucleate, empty; None: from the counts
    contact: float = 0.0             # probability that a proposal is placed against an accepted site, 0 ≤ contact ≤ 1
    overlap: float = 0.0             # κ, 0 ≤ κ < 0.5
    gap: float = 0.0                 # g ≥ 0 (µm); today's CellsConfig.gap is used when packing is None
    monolayer: bool = False          # every centre in the stack's mid-plane z_mid = (n_z − 1) / 2 · s_z
    on_crowded: str = "raise"        # "raise" or "stop" (C5)
    floor: float = 0.9               # with "stop": the reached share of the density below which it still raises, 0 < floor ≤ 1
```

**Monolayer.** With `monolayer`, every centre (nuclei, lobes' parent centres and empty
cells) has `z = z_mid`, lobe offsets and contact directions lie in the YX plane, and a
uniform proposal maps its z draw to `z_mid`. `z_mid` must lie in the covered region's Z
interval, so that anchors can be sample members (`ValueError` otherwise), and the scene
must have more than one plane (one plane is the `Z=1` case). Because all centres share
one z, the power distance difference of two sites has no z term: every face of the power
diagram is vertical, so a column of fixed `(y, x)` lies in one power cell, and **no two
nuclei or territories overlap in a Z projection** ("Presets", C19). In floating point the
sums `dz² + dy² + dx²` of two sites round with the same `dz²` term, so the comparison can
differ along a column only within rounding of a face; identity 11 reports such columns.
In the appearance presets of {doc}`synthetic-real-appearance` the 2D tissue is placed in
one plane, each entry a column through the planes (TZ1), so the same holds there. The
culture cells keep their own Z extents, the bank shifted by one constant so that its
median nucleus centroid lies on the stack's middle plane (AU1), and are tested for overlap
in 3D, so a column can hold two culture nuclei.

**Density.** With `density` None, the seed counts of `CellsConfig` apply as
today. With a value `φ`, the kind of seed `k` (ordinary, binucleate or empty) is drawn
from `context.cell.kind` (entity `cell-<k>`) with the probabilities `kind_shares` (or
proportional to the counts of `CellsConfig`), and seeds are added until the summed
nominal size of the placed nuclei reaches `φ · ‖Ω‖`. `‖Ω‖` is the volume of `Ω`, or the
area of its Y-X box with one plane or a monolayer; the nominal size of a nucleus is the
sum over its lobes of `4/3 π a_z a_l² e` (with one plane or a monolayer, `π a_l² e`), with
each lobe's own semi-axes. The ordinary or binucleate share must be positive, so that the
loop ends (`ValueError` otherwise). Nucleus `n` (the `n`-th placed nucleus) draws its sizes
from its own entity `nucleus-<n>`, so the sequence of sizes does not depend on placement
outcomes. In the appearance presets of {doc}`synthetic-real-appearance` objects are added
until the composed stain's visible-stain share reaches its target (DN1, `slow` presets) or
a fixed count is placed (DN3, routine presets), with F2's record and floor.

**A seed that cannot be placed (C5).**

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| F1. Raise | As today: `ValueError` naming the seed. | None. | W-393 had to catch the error and retry with fewer seeds on five scenes ("refused" in W-393 `tables/sizing.csv` `generator_attempts`). |
| **F2. Stop and record, with a floor (recommended)** | With `on_crowded="stop"`, placement ends at the first seed with no accepted proposal after `max_proposals`; that seed is not placed; the reached density and the number of placed seeds are recorded in `context.json`; if the reached density is below `floor · φ`, `ValueError` names the seed and both densities. With `density` None, `on_crowded="stop"` stops the same way and the floor applies to the requested count. In the appearance presets of {doc}`synthetic-real-appearance` no seed is fixed in advance: each proposal draws its own entry and transform. Placement ends at the target (DN1) or the count (DN3), at `max_proposals` proposals in total, or after `max_refusals` refused proposals in a row. The reached value, the placed objects and the proposals are recorded in `appearance.json`, and below `floor` of the target a `ValueError` names both ('When placement stops' there). | One record entry. | Dense presets are reproducible without retries, and a preset that cannot reach its density fails loudly. |
| F3. Skip the seed and continue | Drop the seed, try the next. | None. | The scene holds fewer seeds than requested without a clear stopping point; later seeds can still fit, which makes the result depend on the order of sizes. |

**Recommendation: F2**, `on_crowded` defaulting to `"raise"` so that a count-mode scene
behaves as today.

**The two regimes.** *Dense tissue*: `κ` about 0.1, `g = 0`, `contact` set from the
touching share, the density from the volume (area) fraction; territories fill most of
the space between nuclei. *Sparse culture*: large nuclei far apart (median centre
distance 21.8 µm, W-393 `real/culture/geometry.csv` "nearest-neighbour distance between
centres") at a low area fraction, and territories that leave empty space. The culture
labels of W-393 are denser than "sparse" suggests (area fraction 0.17 to 0.47, touching
share 0.50; W-393 `notes.md` §6), so the culture presets keep `g = 0` and `contact` at the
measured share: their sparseness comes from the density, not from a gap. In the
appearance presets of {doc}`synthetic-real-appearance` both regimes come from the bank and
PL2: contact proposals with probability 0.75, at most 1 % overlap, and the density from
the visible-stain share of the source images (DN1).

## Cell territories

Today the cell label is the power cell of the nearest site, limited to
`‖σ − c_{j*}‖ ≤ R_{j*} + reach`, with the two sites of a binucleate cell mapping to one
cell (`synthetic/_sample.py::_label_functions`; rule L1 of {doc}`synthetic-sample-scene`).
With lobed nuclei the site sphere is larger than lobe 1, and the LN route's cells are
nuclei expanded by 0.78 µm (W-382's setting, W-393 `notes.md` §4), while today's reach of
1.5 to 2.5 µm beyond the site sphere left 55 to 80 % of the molecules with a true cell
unassigned on that route (W-393 `notes.md` §4 and choice 2).

| Option | Territory of the cell of site `j*` | Cost | Consequence |
| --- | --- | --- | --- |
| T1. Power cell within a reach of the site sphere | Today's rule: `‖σ − c_{j*}‖ ≤ R_{j*} + reach`. | None. | Cells are rounded beyond the nucleus by a distance measured from the bounding sphere, not from the nucleus; with lobes the margin differs between directions. |
| **T2. Power cell within a grown nucleus (recommended for the two tissue contexts)** | `ρ_n(σ) ≤ 1` or `ρ_n^{+δ}(σ) ≤ 1`, where `ρ_n` is the nucleus of site `j*` and `ρ_n^{+δ}` the minimum over its lobes of the ellipsoid `ρ` with every semi-axis grown by `δ` µm (a per-cell value drawn like the reach; without the harmonic factor). For an empty cell, `‖σ − c_{j*}‖ ≤ empty_radius + δ`. | One more `ρ` evaluation per lobe for the candidate site. | A cell is close to its nucleus expanded by `δ`, as the LN route's expansion and the tissue-2D cell areas describe; the reach measures from the nucleus surface. |
| T3. The whole power cell | No outer limit inside `Ω`. | None. | Every point belongs to a cell; no molecule can lie outside every cell, so ambient molecules always join a cell. |

**Recommendation: T2 for dense 3D tissue and 2D tissue, T1 for the culture**, whose
cells reach far beyond the nucleus (median cell area 896 µm² against nuclei of 259 µm²,
W-393 `real/culture/geometry.csv` "cell area (2D labels)", "nucleus area (2D labels)")
(W-393 choice 2, option (a)); in the appearance presets, CT1 (the nearest placed nucleus
within the growth) and the culture cells' own masks ({doc}`synthetic-real-appearance`).

```python
@dataclass(frozen=True)
class TerritoryConfig:
    rule: str = "reach"                        # "reach" (T1), "grown" (T2) or "power" (T3)
    growth: ScalarDistribution = ScalarDistribution("lognormal", (log(0.8), 0.15))   # µm, δ of T2
```

**A nucleus lies inside its own cell.** Under all three rules, `N(σ) = n > 0` requires
that the power-nearest site is a site of `n`, that `σ ∈ Ω` and that `ρ_n(σ) ≤ 1`, and the
cell label `C(σ)` is computed in the same evaluation from the same site. Under T1, the
shape lies in the site sphere of radius `R_n ≤ R_n + reach`; under T2, the test holds
whenever `ρ_n(σ) ≤ 1` by its first clause (the grown ellipsoids also contain the
ellipsoidal lobes, since an ellipsoid with every semi-axis grown by `δ ≥ 0` about the same
centre and axes contains the original, but the first clause makes the containment exact
in floating point and under N2); under T3 there is no outer limit. In each case
`C(σ) = cell(n)`, also at a point on a power face or a surface, because both labels come
from one evaluation. The anchor (the first site's centre) lies strictly in its own power
cell by the K1 tests and has `ρ = 0`, so it lies in its own territory, as
{doc}`synthetic-sample-scene` ("Sample membership") requires. In the appearance presets of
{doc}`synthetic-real-appearance` it holds by one placement order: on every voxel of a
nucleus label the cell label is that nucleus's cell, and the distance rule CT1 decides
only where no nucleus label holds the point ('One ownership rule for nuclei and cells'
there).

## Cell types

Today every molecule draws its gene from the template's abundances, the same for every
cell (`synthetic/_sample.py::_molecules`, stream `sample.identity`). Cell types give the
§2.11 marker figures and MECR a pattern with known answers on synthetic data (D3). MECR
is `both / either` over the counted cells for a supplied pair, with a cell positive for a
gene when its whole-cell count is at least `positive_threshold`
(`report/_measurements.py::mecr`; {doc}`report-algorithms`, "`mecr.csv`").

**How a cell gets its type (C7).**

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| Ty1. At random | Type `t` with probability `π_t`, one draw per cell. | One draw per cell. | Types are mixed everywhere; the marker figures show no spatial pattern, only counts. |
| **Ty2. By region, with a purity (recommended)** | `K_r` region centres in the sample frame (µm) define a Voronoi partition `region(σ)` (ties to the smaller region index); region `r` has a main type. With `K_r = 0` (no regions), Ty1 applies. A cell takes the main type of `region(anchor)` with probability `purity`, else a type drawn by `π`. | `K_r` centres and one or two draws per cell. | The marker figures show domains whose truth is known per cell; MECR stays a per-cell count, unaffected by where cells are. A function of the anchor, so the same in every FOV, at any offset. |
| Ty3. By a smooth field | The type of a cell is the arg-max of `K` smooth fields at its anchor. | More parameters. | As Ty2, with curved boundaries; nothing the checks need beyond Ty2. |

**Recommendation: Ty2**, with `K = 4` types, `K_r = 4` regions and purity 0.9.

**Gene abundance and the marker pairs (C8).** The development codebook of both
existing presets is `development_codebook("balanced")`, 16 genes
(`synthetic/_sample.py::sample_scene_preset`).

| Option | Abundance table (types × genes) | Cost | Consequence |
| --- | --- | --- | --- |
| **G1. Markers per type, shared genes, named pairs (recommended)** | Genes 1 to 8 are markers, two per type (type `t` has genes `2t − 1` and `2t`), with weight `w_m` in their type and 0 elsewhere; genes 9 to 16 are shared, weight `w_s` in every type. Each row is normalized. | A fixed table of 4 × 16 values; the pairs are named in the configuration. | Two markers of one type are **co-expressed** by construction; two markers of different types are **exclusive** by construction. The truth answers are integers from the truth table ("Truth schema"). |
| G2. A random abundance table per type | Dirichlet draws per type. | One draw per type. | No pair is exclusive by construction; MECR has no designed answer. |
| G3. One marker per type, nothing shared | Each type expresses one gene. | Simplest. | Cells hold one gene each; the assignment and report see an unrealistic table, and no co-expressed pair exists. |

**Recommendation: G1**, with `w_m = 0.15` and `w_s = 0.0875` (a type's two markers then
carry 30 % of its molecules), the exclusive pair `(balanced-01, balanced-03)` (types 1
and 2) and the co-expressed pair `(balanced-01, balanced-02)` (type 1), both passed to
the report as `marker_pairs`.

**The known answers.** With G1's table (a marker gene has weight 0 in every other type)
and A2 below (ambient molecules carry no marker gene), a marker gene occurs only among
molecules placed by a cell of its type, each placed inside that cell (accepted where
`C = k`), and the counts per cell are the true counts of `cell_gene_counts.csv`. Another
abundance table, or ambient abundances with marker genes, void the two answers below:

* **Exclusive pair.** No cell has a true count of at least one for both genes, so the
  truth MECR of `(balanced-01, balanced-03)` is exactly 0 at every `positive_threshold`
  (when `either > 0`). An observed MECR above 0 after segmentation and assignment counts
  cells that received a molecule of another type's marker: the chain's mixing, not the
  scene's.
* **Co-expressed pair.** Under the same conditions, only type-1 cells hold either gene. The truth MECR is
  `both / either` computed from `cell_gene_counts.csv`, an integer ratio known before any
  method runs. As a design check, with a cell's marker counts approximately independent
  Poisson of mean `λ = 0.15 · n_k` each, the expected ratio at threshold 1 is
  `(1 − e^{−λ})² / (1 − e^{−2λ})`, above 0.99 for `n_k ≥ 40` molecules.
* **Marker figures.** Each marker gene's true molecules lie in the cells of its type,
  hence mostly in the regions whose main type it is.

**What else may depend on the type (C8, second part).** The configuration allows a value
per type for the mean molecules per cell, the nuclear fraction of reads and a scale of
the nucleus size laws; every other law stays per scene. The presets use one value per
context for each (the W-393 tables do not resolve types). In the appearance presets of
{doc}`synthetic-real-appearance` the size scale has no effect, since sizes come from the
entries and the augmentation's scale.

```python
@dataclass(frozen=True)
class CellTypeConfig:
    types: tuple[str, ...] = ("type-1", "type-2", "type-3", "type-4")
    abundances: tuple[tuple[float, ...], ...] = ...    # types × codebook genes, rows normalized (G1)
    proportions: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)   # π
    regions: int = 4                                   # K_r; 0 is Ty1
    purity: float = 0.9
    molecules_per_cell: tuple[float, ...] | None = None    # per type; None: MoleculeConfig's value
    nuclear_fraction: tuple[float, ...] | None = None      # per type; None: MoleculeConfig's value
    size_scale: tuple[float, ...] | None = None            # per type, multiplies the nucleus size laws
    marker_pairs: tuple[tuple[str, str, str], ...] = (("balanced-01", "balanced-03", "exclusive"),
                                                      ("balanced-01", "balanced-02", "coexpressed"))
```

## Molecules

Today cell `k` receives `Poisson(molecules_per_cell)` molecules placed uniformly in its
territory by rejection, and ambient molecules are uniform in `Ω` at `ambient_density`
per µm³ (`synthetic/_sample.py::_molecules`). The real contexts hold far more reads per
cell and a nuclear fraction that differs by context: reads per cell 8 (1–19) in LN, 194
(85–310) in tissue-2D, 982 (86–1884) in the culture; nuclear fraction per cell 0.50
(0.25–0.75), 0.75 (0.43–0.93) and 0.26 (0–0.56) (W-393 `real/<context>/molecules.csv`
"reads per cell" and "nuclear fraction of reads, per cell with >= 10 reads"). The current
scene puts 0.13 of a cell's molecules in its nucleus, its share of the territory (W-393
`notes.md` §4).

**Molecules per cell (C9).**

| Option | Law | Cost | Consequence |
| --- | --- | --- | --- |
| M1. Poisson with one mean | Today. | None. | Counts per cell vary far less than in the real tables (LN p5–p95 1–19 at median 8 is wider than Poisson's 4–13). |
| **M2. Poisson with a lognormal mean per cell (recommended)** | `μ_k ~ lognormal(log m, s)` per cell (or per type), then `Poisson(μ_k)`. | One more draw per cell. | The spread of reads per cell is one parameter `s`, set from the real p5–p95. |
| M3. A fixed count | `n_k = m`. | None. | No spread; the composite share per cell is uniform. |

**Recommendation: M2.**

**The nuclear fraction of reads (C10).**

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **NF1. A compartment draw per molecule (recommended)** | Each cell molecule is placed in its cell's nucleus (either nucleus of a binucleate cell, `N(σ) > 0` and `C(σ) = k`) with probability `f` and in the rest of its territory (`N(σ) = 0`, `C(σ) = k`) otherwise, uniformly by rejection within the chosen compartment. | One uniform draw per molecule; rejection in the nucleus box. | The realized nuclear fraction per cell is Binomial(`n_k`, `f`): the measured quantity is the control. A cytoplasm molecule that finds no point of its territory outside the nucleus after `max_proposals` raises `ValueError` naming the cell; an empty cell (no nucleus) places all its molecules in its territory. |
| NF2. A density ratio | Uniform density in the territory, multiplied by `r` inside the nucleus. | As today, with a weighted rejection. | The nuclear fraction depends on each cell's nucleus-to-territory ratio, so one `r` gives different fractions per cell. |
| NF3. Uniform (today) | `f` is the nucleus's share of the territory. | None. | 0.13 in the current scene; not a control. |

**Recommendation: NF1** (W-393 choice 2 names the nuclear fraction as the control: 0.5 LN,
0.75 tissue-2D, 0.26 culture).

**Ambient molecules (C11).** Today they draw genes from the template's abundances and
land in cells or outside every cell as the geometry decides; an ambient molecule inside a
cell belongs to it ({doc}`synthetic-sample-scene`, "Cell truth for molecules").

| Option | Genes of ambient molecules | Cost | Consequence |
| --- | --- | --- | --- |
| A1. The template's abundances | Today. | None. | An ambient molecule of a marker gene inside a cell of another type makes the truth MECR of an exclusive pair positive: the designed answer 0 is lost. |
| **A2. An ambient abundance without the marker genes (recommended)** | The shared genes only, uniform by default. | One more abundance row. | The truth MECR of an exclusive pair is exactly 0; ambient molecules still add to every cell's shared-gene counts and to the molecules outside every cell. |
| A3. No ambient molecules | `ambient_density = 0`. | None. | No molecule outside every cell; the `unassigned` status of the assignment is never exercised by truth. |

**Recommendation: A2.**

```python
@dataclass(frozen=True)
class MoleculeConfig:
    per_cell_median: float = 8.0          # m of M2
    per_cell_spread: float = 0.0          # s of M2 (log SD); 0 is M1
    nuclear_fraction: float | None = None # f of NF1; None is NF3
    ambient_density: float | None = None  # per µm³ of Ω; None: SampleSceneConfig.ambient_density
    ambient_abundances: tuple[float, ...] | None = None   # None: A2 (shared genes, uniform)
```

`SampleSceneConfig.molecules_per_cell` is ignored when `molecules` is given, and
`max_count` still guards the total.

## Stains

Today the stain is one nuclear channel per FOV: per nucleus a soft-edged ellipsoid
`A_n · ½ erfc((ρ_n − 1) · r̄_n / (√2 w))`, then the observation stages (tissue background,
baseline, Poisson and read noise) with the stain's own streams, cast to the template's
dtype by rounding and clipping (`synthetic/_sample.py::_stain_signal`, `_observe` from
`synthetic/_observation.py`, `_cast`; option St1 of {doc}`synthetic-sample-scene`).

**The interface (C12).**

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **S1. One channel configuration with a geometry source (recommended)** | `StainChannelConfig(name, source, round, ...)`, one per channel; `source` is `nucleus`, `territory` or `boundary`; the same appearance controls (brightness, edge, unevenness, background, noise, output) for every source. | One configuration class and three signal functions sharing the distance terms. | The nuclear, cytoplasm and membrane stains are three uses of one interface (D2); a further stain is a configuration, not code. |
| S2. One configuration class per stain | `NuclearStainConfig`, `CytoplasmStainConfig`, ... | More classes, duplicated controls. | Each stain can differ; the interface of D2 is not one. |
| S3. A list of today's `StainConfig` | Reuse `StainConfig` with a source field. | A new field on `StainConfig`. | Changes the accepted `StainConfig` and the saved `sample.json` (C1). |

**Recommendation: S1.** In the appearance presets of {doc}`synthetic-real-appearance`
each stain is one channel of the bank's entries, with the name, round and file pattern of
its `StainChannelConfig`; its source and appearance controls are not used ('Composition'
there).

```python
@dataclass(frozen=True)
class StainChannelConfig:
    name: str                                   # channel name, e.g. "DAPI", "PI", "Flamingo", "membrane"
    source: str = "nucleus"                     # "nucleus", "territory" or "boundary"
    round: str = "reference"                    # "reference" (a reference stain file) or "morphology"
    channel: str = "ch04"                       # the file pattern of the channel ("The morphology round")
    brightness: ScalarDistribution = ScalarDistribution("lognormal", (log(1000), 0.2))  # plateau per object
    edge_width: float = 0.5                     # µm, w
    band_width: float = 0.5                     # µm, boundary source only
    nuclear_ratio: float = 0.3                  # territory source: brightness inside the nucleus over the cytoplasm
    texture: BrightRegionConfig | None = None   # nucleus source only
    unevenness: UnevennessConfig | None = None
    background: BackgroundConfig = BackgroundConfig(baseline_enabled=True, baseline=((100.0,),))
    noise: NoiseConfig = NoiseConfig(dependent_enabled=True, model="poisson", alpha=1.0,
                                     independent_enabled=True, sigma=3.0)
    output: OutputScaleConfig | None = None     # None: the template's dtype, cast as today
```

**The nuclear stain** (source `nucleus`), first in the delivery order. Before noise,

```
S_nuc(σ) = U(σ) · Σ_n A_n · E_n(σ) · (1 + T_n(σ))
E_n(σ)  = ½ erfc( max((ρ_n(σ) − 1) · r̄_n, h_n(σ)) / (√2 w) )
h_n(σ)  = max over sites j ≠ sites of n of (pow_i(σ) − pow_j(σ)) / (2 ‖c_i − c_j‖)
```

with `ρ_n` the nucleus's shape function, `r̄_n` the geometric mean of lobe 1's
semi-axes (as today; `(ρ_n − 1) · r̄_n` is a signed distance to the surface, exact for a
sphere and approximate otherwise), `pow_i` the power distance to `n`'s site `i`, and `h_n`
the signed distance to the faces of `n`'s power cell (negative inside; `pow_i − pow_j` is
affine in `σ`, so this is the exact distance to each face plane). `S_nuc` is 0 only
beyond every nucleus's cut-off (below): a FOV whose voxel centres hold no nucleus label
can still receive the soft edge of a nucleus just outside it or between its voxel centres,
and only a FOV beyond the cut-off of every nucleus holds the background and noise alone in
a nuclear stain (a territory or boundary stain is lit by cells, with or without a nucleus:
"The territory source", "The boundary source").
The sign of the erfc argument agrees with the nucleus label: `N(σ) = n` exactly where
`max((ρ_n − 1) · r̄_n, h_n) ≤ 0`, for `σ ∈ Ω` (a point with `ρ_n ≤ 1` passes every territory
test, "Cell territories"), apart from a tie on a power face (ties to the smaller site
index) and rounding within 1e-9 µm of a boundary. Audit D of W-394
(`results/audit/stains_vs_labels.txt`) evaluates both at more than 100,000 positions per
configuration (binucleate cells with unequal radii, lobed and harmonic nuclei, touching
nuclei, a FOV at a fractional offset in Z, Y and X, `Z=1`) and finds no disagreement
outside that band. The **plateau** is
`A_n` (per nucleus, lognormal); the **edge** is the Gaussian edge of width `w` across the
nucleus surface and, where two nuclei touch, across their shared face, so the stain
plateau coincides with the label up to the edge width at a contact too. `E_n` is set to
exactly 0 beyond `ρ_n > 1 + 4w / a_min` (`a_min` lobe 1's smallest semi-axis) or
`h_n > 4w`, as today's cut-off. Every other site enters `h_n`; an implementation may skip
a site only where its term is below `−4w` everywhere in `n`'s evaluation box (a site whose
sphere is merely far from `n`'s is not enough: the face with a much larger sphere can cut
close to `c_n`), which changes `E_n` by at most the cut-off's `3 · 10⁻⁵` of the plateau and
never the sign of its argument. Then the existing
observation stages (tissue background, baseline, dependent and read noise, optional
correlated noise; stages 8 to 10 of {doc}`synthetic-specification`) and the output
(B1 below).

**Bright regions (C13).** A real nucleus has a few clearly brighter regions; W-393
detected them as objects (smoothed stain at least 1.5 times the nucleus median, at least
0.1 µm³ or µm², W-393 `notes.md` §2): per nucleus 3 (1–5) in LN, 2 (0–8) in tissue-2D, 2
(0–58) in the culture; peak over the nucleus median 2.35 (1.88–3.10), 2.01 (1.76–2.60),
3.00 (1.90–9.58); size 0.82 µm³ (0.13–5.04), 0.19 µm² (0.11–1.16), 0.26 µm² (0.11–7.49)
(W-393 `real/<context>/stain.csv`, rows "bright regions per nucleus (factor 1.5)",
"bright region peak over nucleus median (factor 1.5)", "bright region size (factor
1.5)"). The current scene has none and a within-nucleus CV of 0.09 against 0.27 to 0.32
(W-393 `notes.md` §8).

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **Tx1. Gaussian bright regions inside the nucleus (recommended)** | `T_n(σ) = Σ_b h_b exp(−½ ‖(σ − c_b) / s_b‖²)`, with `Poisson(λ)` regions per nucleus, centres drawn uniformly inside the nucleus (rejection on `N = n`), size `s_b` (µm) and height `h_b` from laws. | A few Gaussians per nucleus, each over its own box of `4 s_b`; truth one row per region. | Matches the measured objects by count, size and peak; each region is a known object for the W-393 detection, which serves as the golden measure. |
| Tx2. A multiplicative random field inside each nucleus | Smooth noise per nucleus. | A field defined in the sample frame. | Raises the within-nucleus CV, but its bright regions are not countable objects with a truth row, and the CV cannot tell regions from noise (D4). |
| Tx3. Ellipsoidal plateaus | Brighter sub-ellipsoids with soft edges. | As Tx1, plus edges. | Flat-topped regions; real regions are peaked (peak 2.0 to 3.0, mean 1.7 to 2.0 of the nucleus median). |

**Recommendation: Tx1** for the closed-form presets, with W-393's bright-region
detection (factor 1.5, 0.1 µm³ or µm², one-voxel smoothing) as the measure a generator
check applies (W-393 choice 8; accepting the measure is part of this choice); the
appearance presets carry the real texture of their entries. The region size of a Gaussian of height
`h` above 1.5 times the plateau has radius `s · sqrt(2 ln(2h))`, which gives first values
`s ≈ 0.41 µm` (LN, `h` 1.35), `0.21 µm` (tissue-2D, `h` 1.0) and `0.17 µm` (culture, `h`
2.0); the generator check measures the scene with W-393's detection and the preset value
is kept only if the measured counts, sizes and peaks lie in the target ranges.

```python
@dataclass(frozen=True)
class BrightRegionConfig:
    per_nucleus: float = 2.0                         # λ, Poisson mean
    size: ScalarDistribution = ScalarDistribution("lognormal", (log(0.3), 0.4))     # µm, s_b
    height: ScalarDistribution = ScalarDistribution("lognormal", (log(1.0), 0.3))   # h_b; peak ≈ 1 + h_b
```

**Unevenness across the field (C14).** The real nuclei differ between each other and
across the field: the nucleus median over the image's median has p5–p95 0.70–1.44 in LN,
0.52–1.37 in tissue-2D and 0.25–4.92 in the culture (W-393 `real/<context>/stain.csv`
"nucleus median over the image's median of nucleus medians"); the culture plateaus
differ by a factor 6 between FOVs (W-393 `notes.md` §12).

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **U1. Log-Gaussian bumps in the sample frame, normalized (recommended)** | `log U(σ) = a · Σ_q z_q w_q(σ) / sqrt(Σ_q w_q(σ)²)`, with weights `w_q(σ) = exp(−½ (‖σ − u_q‖² − min_r ‖σ − u_r‖²) / L²)` (Gaussian weights divided by the largest, which the ratio does not change and which keeps the largest weight at 1, so no `L` underflows), `K_U ~ Poisson(centres_per_area · A_Ω / A_FOV)` centres `u_q` uniform in the Y-X box of `Ω` at the z of `Ω`'s centre (µm; `A_Ω` and `A_FOV` the Y-X areas of `Ω` and of one FOV), `z_q ~ N(0, 1)`, one length `L` (µm). Given the centres, `log U(σ)` is normal with SD exactly `a` at every point, near the edges of `Ω` too (without the normalization it would fall to a fraction of `a` there; audit A of W-394 measured 25 to 70 % on the presets' domains). With `K_U = 0`, `U = 1`. | `K_U` Gaussians per voxel, about 4 per FOV area, and one square root. | A staining unevenness continuous across FOVs, multiplying the signal (not the background); a point function, evaluated at the voxel centres at any offset. |
| U2. A low-order cosine series | `U = exp(Σ c_k cos(k·σ + φ_k))`. | As U1. | Periodic; a pattern a stitcher could lock onto. |
| U3. A flat field per FOV | Vignetting multiplying each FOV. | Per FOV. | An illumination effect that differs between FOVs at the same sample point, which {doc}`synthetic-sample-scene` excludes ("What this addition is not"); left out. |

**Recommendation: U1.** With it, the between-nucleus variation has two parts: the
per-nucleus lognormal spread of `A_n` and the field. The presets take the per-nucleus log
SD from the MAD of the measured ratio (`1.4826 × MAD`: 0.23 LN, 0.20 tissue-2D, 0.53
culture) and the field's `a` from the remaining width of the p5–p95 range
(`sqrt(max(0, s_range² − s_nuc²))` with `s_range = ln(p95 / p5) / 3.29`: 0 LN, 0.22
tissue-2D, 0.74 culture). W-393 measured no spatial scale of the unevenness; `L` is one
FOV width (a development value). The appearance presets take the spread between nuclei
from their entries and the gain (0.9–1.1), with no field.

```python
@dataclass(frozen=True)
class UnevennessConfig:
    centres_per_area: float = 4.0     # mean number of centres per FOV area of Y × X, ≥ 0
    amplitude: float = 0.2            # a, the SD of log U at every point, ≥ 0
    length: float | None = None       # L in µm, > 0; None: the larger of n_y · s_y and n_x · s_x
```

**Background and noise.** The existing controls, unchanged: the stain's
`BackgroundConfig` (baseline, gradient, regions, texture, in the sample grid's index
units) and `NoiseConfig`, applied by `_observe` (`synthetic/_observation.py`) as today.
The real background over the plateau is 0.19 per image in LN (per 128-pixel block p95
0.78), 0.025 in tissue-2D (0.06) and 0.012 in the culture (0.37), and the noise over the
plateau per block 0.021, 0.011 and 0.025 (W-393 `real/<context>/stain.csv` "background
level over plateau, per image", "background over plateau, per 128-px block", "noise (sd
of background) over plateau, per 128-px block"). The current stain has 0.38 and 0.039
(W-393 `notes.md` §6, scene `base`): higher than every context, so the presets lower it
(W-393 `notes.md` §8, "Existing background controls": no new control). In the appearance
presets the background is assembled from real patches and no observation stage is added
(BG1, OB1 of {doc}`synthetic-real-appearance`).

**The territory source** (cytoplasm). The stain of cell `k` is built from the same rule
as its label `C` ("Cell territories"). Let `j_k(σ)` be the site of cell `k` with the
smallest power distance at `σ` (ties to the smaller site index; for a cell with one site,
that site), `h*_k(σ) = max over sites i of other cells of (pow_{j_k}(σ) − pow_i(σ)) /
(2 ‖c_{j_k} − c_i‖)` the signed distance to the faces between `j_k` and the other cells'
sites, and `o_j(σ)` the outer term of site `j`'s territory test: `‖σ − c_j‖ − (R_j + reach)`
for T1, `min((ρ_n − 1) · r̄_n, (ρ_n^{+δ} − 1) · r̄_n)` for T2 (`n` the nucleus of `j`),
`‖σ − c_j‖ − (empty_radius + reach)` or `− (empty_radius + δ)` for an empty cell under T1
or T2, and `−∞` for T3. Then

```
d_k(σ) = max( h*_k(σ), o_{j_k(σ)}(σ) )
```

and `C(σ) = k` exactly where `d_k(σ) ≤ 0`, for `σ ∈ Ω`: `h*_k < 0` says that the
power-nearest site of all is a site of `k`, which is then `j_k`, and `o_{j_k} ≤ 0` is that
site's territory test. (Taking each site's outer term and keeping the smaller one would
not do: in a binucleate cell, a point in the power cell of its first site but beyond that
site's reach is outside the cell even when it lies within the second site's reach.) As
for the nucleus, the sign agrees with the label apart from a tie on a power face and
rounding within 1e-9 µm of a boundary (audit D: no disagreement outside it, the reviewer's
binucleate case included, where the rule of the first repair disagreed). `d_k` is a signed
distance to each face plane and,
for T1, to the outer sphere; for T2 it is the scaled `ρ` term, approximate as for the
nucleus. Across the face between the two sites of a binucleate cell, `j_k` changes and so
does the outer term: `d_k` can jump there, as the label's own rule does. With
`E_k^ter(σ) = ½ erfc(d_k(σ) / (√2 w))` and the soft nucleus indicator
`M(σ) = min(1, Σ_n E_n(σ))`:

```
S_ter(σ) = U(σ) · Σ_k B_k · E_k^ter(σ) · (1 − (1 − q) · M(σ))
```

with `B_k` the plateau per cell and `q` the `nuclear_ratio` (the cytoplasm stain is
dimmer inside the nucleus, as Flamingo is). As for the nucleus, `E_k^ter` is set to exactly
0 where `d_k > 4w`, the **cut-off of the territory** of cell `k` (a jump of at most
`3 · 10⁻⁵` of the plateau). The source is lit by every cell, with or without a nucleus: an
empty cell of radius 2.5 µm with a reach of 1 µm gives the full plateau `B_k U` at its
centre, beyond every nucleus's cut-off. A point where `d_k > 4w` for every cell `k` holds
no territory signal; under T3 every point of `Ω` lies in a territory, so with T3 only a
scene without cells has such points in `Ω`. W-393 did not measure a cytoplasm stain (its
real tables hold nuclear stains only), so `q`, `B_k` and the edge of the cytoplasm stain are
development values for the gate. The culture appearance presets carry the real Flamingo of
their cell entries.

**The boundary source** (membrane). A band of width `b` (the `band_width`) centred on
every boundary of each cell label `C = k`:

```
S_bnd(σ) = U(σ) · Σ_k B_k · exp(−½ (β_k(σ) / b)²)
β_k(σ)   = min( abs(d_k(σ)), g_k(σ) )
g_k(σ)   = max( abs(h_jj'(σ)), h*_k(σ), min(o_j(σ), o_j'(σ)), −max(o_j(σ), o_j'(σ)) )   cell k with two sites j, j'
g_k(σ)   = +∞                                                                          cell k with one site
```

with `d_k`, `h*_k` and the outer terms `o_j` of the territory source, and
`h_jj'(σ) = (pow_j(σ) − pow_j'(σ)) / (2 ‖c_j − c_j'‖)` the signed distance to the power
face between the two sites of a binucleate cell. The exponential is set to exactly 0 where
`β_k > 4b`, the **cut-off of the band** of cell `k` (a jump of at most `3.4 · 10⁻⁴` of
`B_k U`). `β_k = 0` on every boundary of `C = k` inside `Ω`, so the band reaches its peak
`B_k U(σ)` there:

* on a face between a site of `k` and a site of another cell, where `k`'s side passes its
  territory test: `h*_k = 0` and `d_k = 0`. Where two cells touch, both bands lie on their
  shared face;
* on the outer limit `o_j = 0` of a site `j` of `k` inside `j`'s power cell: `d_k = 0`;
* on the face between the two sites of a binucleate cell, where one side passes its site's
  territory test and the other does not. The label changes from `k` to 0 there, but `d_k`
  jumps past 0 without reaching it, so the rule written before W-480,
  `exp(−½ (d_k / b)²)`, had no band on it: for spherical sites of radii 1 and 2 µm at
  (0, 0, 0) and (0, 0, 2.7), reach 1, the face at y = 2 lies at x = 0.79444…, and across
  x ± 1e-6 `C` changes from 0 to 1 while `d_k` jumps from +0.152 to −0.238. On that face
  `h_jj' = 0`, `h*_k ≤ 0` and the two outer terms have opposite signs, so `g_k = 0`. Where
  both sides pass (the face inside the cell) or both fail, the `min` or the `−max` of the
  outer terms is positive, and the face carries no band.

`β_k` jumps only on the face between the two sites of a binucleate cell, where `d_k` and
`h*_k` change their site (the sign of `h*_k` does not), and at the cut-off; elsewhere every
term is continuous. `β_k` is the distance to the face planes and, under T1, to the outer
spheres, except near corners; under T2 its outer terms are the scaled `ρ` terms, approximate
as for the nucleus. The edge of `Ω`, where every label becomes 0, is not a cell boundary:
the band is not centred on it. Every term is a closed form of `σ` and the truth tables
(centres, site radii, lobes, reach or growth), so a FOV at a fractional offset evaluates
`S_bnd` at its voxel centres `true_f + p ⊙ s`, without a label image, a distance transform
or resampling, and gets the band of the sample frame.
Like the territory source, the band is lit by cells with or without a nucleus; a point
where `β_k > 4b` for every cell `k` holds no boundary signal. Audit D of W-480
(`results/audit/stains_vs_labels.txt` in `runs/W-480/20261015T070249Z-868bf70b/`) bisects
18,000 label boundaries to within 1e-9 µm between nearby sample positions (the
binucleate case above, binucleate cells with sites of unequal radii, touching cells of
different owners, lobed nuclei under T2, a FOV at a fractional offset in Z, Y and X, and
`Z=1`). At no boundary point does the band of a cell whose boundary it is lie below 0.99 of
`B_k U(σ)`, at `b` 0.01 and 0.5 µm, while the rule written before W-480 lies below it at
every such point on a binucleate face. The audit also finds no position where `g_k` alone
lights the band without a boundary of `C = k` nearby.

**8-bit output (C15).** Today the cast rounds and clips to the output dtype (`_cast`); with
`dtype uint8` the current plateau of 200 over a baseline of 100 saturates, the
within-nucleus CV falls to 0.017, and StarDist 2D found 5 background objects (W-393
`notes.md` §8, row "8-bit output"). Real 8-bit plateaus lie at 35 to 50 grey levels in
LN, 64 to 93 in tissue-2D and 13 to 83 in the culture (W-393 `notes.md` §9).

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **B1. Scale to a plateau grey level, then cast (recommended)** | After the noise, multiply the float64 plane by `k = g / m_A`, with `m_A` the median of the stain's brightness law (not of the drawn plateaus, so that `k` exists for a FOV or a scene without objects), then round half to even and clip to `[0, 255]`, as `_cast` does. A nucleus of plateau `m_A` with `U = 1` and no background lands at grey level `g`; the background adds `k` times its value (so the measured plateau, background included, is about `g · (1 + b)` for a background ratio `b`). | One multiplication per voxel. | The 8-bit image has the real plateau and keeps the noise's character; clipping touches only the brightest plateaus and bright regions (audit A of W-394 gives the expected share per preset). |
| B2. Render at the 8-bit scale | Set brightness and background in grey levels before the noise. | None. | Poisson noise at a few tens of counts is far stronger than the real noise over the plateau (0.011 to 0.025). |
| B3. Clip (today) | Cast only. | None. | Saturation, as W-393 found. |

**Recommendation: B1** (W-393 choice 4). In the appearance presets of
{doc}`synthetic-real-appearance` the entries are relative to their source plateau, so the
composed image is multiplied by `g` itself, the median plateau of the bank's source images
(O8), with no noise stage before the cast.

```python
@dataclass(frozen=True)
class OutputScaleConfig:
    dtype: str = "uint8"
    plateau: float = 50.0     # g, grey level of the brightness law's median, 0 < g < 255
```

**Optional stains in the first delivery (C16).**

| Option | What the first delivery holds | Cost | Consequence |
| --- | --- | --- | --- |
| O1. Nuclear and cytoplasm only | The `nucleus` and `territory` sources. | Least. | The interface is shown for two sources; a membrane stain needs new code later. |
| **O2. Nuclear and cytoplasm; the boundary source implemented, used by no preset (recommended)** | The three sources, with a generator check of the band; no preset uses it. | The band function (it reuses the territory distance and adds the term of a binucleate cell's internal face) and one check. | The interface of D2 is complete; a membrane stain is a configuration, as Jiahao's data sometimes have one. |
| O3. A membrane stain in a preset | As O2, with a preset using it. | One more image per FOV in that preset. | No route of D2 uses a membrane stain today. |

**Recommendation: O2.** ER is not done (D2).

## The morphology round

**What exists.** `FOV.prepare_morphology(MorphologyConfig(...))` prepares, beside
`FOV.run`, (1) the reference stains, the files of `Dataset.reference_stains` in the
reference round's folder, as the image `reference_stain`, recorded as the same
acquisition as the reference round, and (2) every other round, loaded with its own
channels (`Dataset.other_channel_order`) and registered with `FOV.register`; with
no recipe, one translation on the shared stain, the single reference stain against the
round's one channel of the same name (`dataset/fov.py::FOV.prepare_morphology`;
{doc}`coordination`, "Sequencing rounds and other rounds"). A round is read from `<input_root>/<round>/<fov_id>/`,
one file per channel matched by `*<pattern>*.tif` (`dataset/fov.py::FOV.load_images`,
`io/tiff.py::load_round`). Registration transforms are pull maps: the registered image
at reference index `p` is the moving image at `p + d` for a translation with
`displacement_zyx` `d` ({doc}`registration-contract`, "Transform types").

In the real contexts the nuclear stain is a file of the reference round in tissue-2D (PI,
`round1` `ch04`) and LN (DAPI, `round4` `ch04`, with a `flamingo` round that also holds
DAPI), and an additional `organelle` round (`ch00` DAPI, `ch01` ER, `ch02` Flamingo) in
the culture (D2; W-305 `20261004T0630Z-summary/examples.md`). The culture route needs
the morphology round and the cytoplasm stain (D2).

**The synthetic morphology round.** The stains with `round="morphology"` are rendered in
an additional round named by `MorphologyRoundConfig.name`, whose index `q` in FOV `f`
images the sample position `true_f + τ_f + q ⊙ s`, with a per-FOV translation `τ_f`
(µm). Every stain value is the point function evaluated there; nothing is resampled. The
reference round's stains (round `reference`) are rendered at `true_f + p ⊙ s`, as today.
The scene domain grows by `T_a · s_a` on each side when a morphology round is present
(`T` its translation bound in voxels), so that every morphology voxel images generated
content; the covered region `K` stays as defined by the reference rounds ("Accepted
statements this addition changes", entries 3, 4 and 11).

**Its translation (C17).**

| Option | Law | Cost | Consequence |
| --- | --- | --- | --- |
| **MT1. Bounded uniform per FOV, with a supplied table (recommended)** | `τ_{f,a} = T_a · (2u − 1) · s_a`, `u` from the stream `context.morphology.translation` of FOV `f`, drawn z, y, x; `translation="supplied"` takes a table in voxels. | Three draws per FOV. | A fractional translation on every axis, independent per FOV, as the real rounds are registered per FOV; a test can supply an integer one. |
| MT2. One translation for the sample | One `τ` for every FOV. | Three draws. | A stage offset common to all FOVs; per-FOV registration is not exercised beyond one value. |
| MT3. None | `τ = 0`. | None. | The registration of `prepare_morphology` is trivial; a defect that ignores the translation would pass. |

**Recommendation: MT1**, bound `(0.5, 6, 6)` voxels. W-393 measured no round offset; the
bound is a development value of the size of the real ones (LN `flamingo` round (0, 6, 21)
pixels at `Position020`, W-305 `examples.md`; culture frame within a few pixels, W-392
`frame/frames.md`).

**How it is registered (C18).**

| Option | Files and recipe | Cost | Consequence |
| --- | --- | --- | --- |
| **MR1. A reference stain sharing the nuclear stain's name (recommended)** | The reference round's folder holds the nuclear stain as a reference stain file (`Dataset.reference_stains`, name `DAPI`); the morphology round holds a channel of the same name; `prepare_morphology` with no recipe registers by one translation on that stain. | One more stain image per FOV in the reference round. | The default recipe applies as in LN; the registration truth is one translation per FOV. |
| MR2. No reference stain; a recipe against the reference round | `MorphologyConfig(recipe=...)` registers the morphology round's nuclear channel against the resident reference round (after `FOV.run`). | A recipe in the check. | Closer to a context whose reference round has no stain, but the registration matches a nuclear stain to amplicon spots, which the scene does not make well posed. |

**Recommendation: MR1** for the culture presets, with the limit stated: the real culture
route's own registration is not reproduced.

```python
@dataclass(frozen=True)
class MorphologyRoundConfig:
    name: str = "organelle"
    translation: str = "uniform"                              # MT1, or "supplied"
    translation_max_zyx: tuple[float, float, float] = (0.5, 6.0, 6.0)   # voxels; 0 in z for one plane
    translation_zyx: Mapping[str, tuple[float, float, float]] | None = None
```

**Files and layout.** `save_cellular_dataset(scene, directory)` writes a dataset that
`Dataset` and `FOV.load_images`, `FOV.run` and `FOV.prepare_morphology` read:

```
<directory>/
  dataset.json                       # the Dataset fields: rounds (sequencing, reference, other), channel_order,
                                     #   other_channel_order, reference_stains (pattern and name), voxel sizes,
                                     #   codebook path, FOV IDs; and the SHA-256 of every file below
  codebook.csv
  <round>/<fov_id>/<pattern>.tif     # each amplicon round: one ZYX file per channel (ch00 ... ch03)
  <reference round>/<fov_id>/<pattern>.tif   # the reference stains (round "reference"), e.g. ch04 named DAPI or PI
  <morphology round>/<fov_id>/<pattern>.tif  # the morphology stains, e.g. organelle/FOV_001/ch00.tif (DAPI),
                                             #   ch02.tif (Flamingo)
```

Each file carries the FOV's reference `ImageMetadata`, as `save_volume` writes it. The
existing case, a nuclear stain file of the reference round with no morphology round, is
the same layout without the morphology folder; `prepare_morphology` then loads it as
`reference_stain` with no transform. The no-segmenter path of
{doc}`synthetic-sample-scene` (labels and molecule CSV per FOV) stays in the
`save_cellular_scene` layout.

**The truth of the translation.** `morphology_truth.csv`, one row per FOV: `fov_id`,
`round`, `translation_z/y/x_um` (`τ_f`), `translation_z/y/x_vox` (`τ_f ⊘ s`, as drawn or
supplied) and `pull_displacement_z/y/x_vox` (`−τ_f ⊘ s`): the `displacement_zyx` that the
round's `TranslationTransform` recovers when registration is exact, because the reference
index `p` and the morphology index `p − τ_f ⊘ s` image the same sample position.

## `Z=1`

With one plane (`fov_shape_zyx[0] = 1`; the jitter bound in Z is 0, as today), every
object lies in the plane z = 0:

* **Nuclei.** Every centre and lobe offset has z = 0, so each lobe's section is the
  ellipse of its lateral semi-axes; the harmonic form (N2) is a Fourier series in the
  angle.
* **Packing.** Disks in Y and X; the density is an area fraction; contact directions lie
  on the circle.
* **Territories.** The power diagram of the plane; T1, T2 and T3 as above in two
  dimensions.
* **Molecules and bright regions.** z = 0.
* **Unevenness.** Centres at z = 0; the field does not depend on z.
* **Morphology round.** `T_z = 0`, so `τ_z = 0`.
* **Labels.** `plane` label images, as today ({doc}`synthetic-sample-scene`, "Label
  volumes").

In a scene of {doc}`synthetic-real-appearance` with one plane, every object is evaluated at
z = 0 like any point: a 2D entry is its plane and a 3D entry its section through the plane;
placement runs in the plane; territories follow CT1 or CU1 in the plane; `τ_z = 0`;
`plane` label images.

On the current scene, the 2D route ran to the end on a one-plane scene, and StarDist
`3D_spleen` refused it at its dimensionality check, which is the dense 3D route's
requirement, not a defect of the scene (W-393 `notes.md` §5). No preset has one plane;
the 2D tissue presets are projection samples (C19).

## The four cases, their controls and their ranges

| Case (D4) | Control that produces it | W-393 measurement that sets its range | Range by context (LN; tissue-2D; culture) |
| --- | --- | --- | --- |
| Touching nuclei | `PackingConfig.contact`, `overlap` (`κ`) and `density` (K1, P1) | `real/<context>/geometry.csv` "share of touching nuclei, per image", with "nucleus volume fraction, per image" and "nearest-neighbour distance between surfaces" | touching 0.44–0.57; 0.76–0.82; 0.33–0.72. Volume (area) fraction 0.07–0.16; 0.30–0.44; 0.17–0.45 |
| Uneven staining | `UnevennessConfig` (U1), the per-nucleus brightness spread of `StainChannelConfig.brightness`, and the bright regions (Tx1) | `real/<context>/stain.csv` "nucleus median over the image's median of nucleus medians", "variation within a nucleus (CV of interior voxels)", the three bright-region rows | between nuclei 0.70–1.44; 0.52–1.37; 0.25–4.92. Within a nucleus CV 0.25–0.41; 0.20–0.38; 0.11–1.35 |
| High background | `StainChannelConfig.background` (baseline, regions) and `noise`, relative to the plateau | `real/<context>/stain.csv` "background level over plateau, per image", "background over plateau, per 128-px block", "noise (sd of background) over plateau, per 128-px block" | per image 0.01–0.34; 0.022–0.031; 0.002–0.047. Per block up to 0.78; 0.06; 0.37 (p95). Noise 0–0.063; 0–0.033; 0–0.24 |
| Nuclei that are not one ellipse | `NucleusShapeConfig.lobed_share` and the lobe laws (N1); N2 optional | `real/<context>/geometry.csv` "share of nuclei not described by one ellipsoid (IoU < 0.8), per image", "convexity (solidity)", "ellipsoid fit IoU" | share 0.064–0.074; 0.027–0.060; 0.12–0.43. Solidity p5 0.88; 0.90; 0.83 |

In the appearance presets the controls are the bank, its selection rules, the density
(DN1), the fall-off length and the out-of-focus light ({doc}`synthetic-real-appearance`).

The edge width belongs to the touching case as well (a wide edge merges close nuclei in
the stain): W-393 `real/<context>/stain.csv` "edge width 10-90 %", 1.39 µm (0.20–2.20) in
LN, 0.72 (0.31–1.57) in tissue-2D, 0.85 (0.33–1.63) in the culture. W-393 measured 0.97 µm
on the current scene at `edge_width` 0.5 and 1.46 µm at 1.0, on the LN grid (`notes.md` §8), so the presets
take `w` from the line through those two points, `w ≈ (w_10–90 − 0.48) / 0.98`: 0.93 µm
(LN), 0.25 µm (tissue-2D), 0.38 µm (culture), and a generator check measures the result.
The tissue-2D crops are the densest tiles of that sample (W-393 `notes.md` §12), so its
ranges describe dense tissue.

## Presets

Six cellular presets in `CELLULAR_PRESETS` (version `context-presets-v1`), separate from
`SAMPLE_PRESETS`: one routine and one `slow` preset per context. The appearance presets
are separate again, in `APPEARANCE_PRESETS` ({doc}`synthetic-real-appearance`). Each table below gives a
value per configuration field, with its unit, and the source it was set from: a W-393
measurement (named "W-393 measured", with its row and, where the measured quantity is
another one, the conversion), or "not measured by W-393" with the source (a W-393 scene
configuration, an accepted page, the route, or a stated assumption). Which set of values
the presets take is choice C23, C24 or C25; the values below are option PS1 of each. The
expected numbers of nuclei and molecules are outcomes, not parameters: they come from
audit A of W-394 (`results/audit/rules_vs_presets.txt`), which runs the page's rules with
these values (40 trials per routine preset, 80 for 2D tissue; 5, 40 and 10 for the `slow`
presets) and measures the touching share as W-393 measures it, on the label image. For
`context-tissue2d-slow-v1` and `context-culture-slow-v1`, whose pixel is twice the acquired
one by Jiahao's decision of 2026-10-15, they come from its re-run by W-481 with the same trials
(`results/audit/rules_vs_presets.txt` in `runs/W-481/20261015T160947Z-7ca392b2/`). A
single routine scene holds few nuclei, so its touching share varies widely (0 to 1); the
ranges hold for the mean.

**Parameters common to the six presets.**

| Parameter | Field (unit) | Value | Set from |
| --- | --- | --- | --- |
| Codebook | `cellular_scene_preset` | `development_codebook("balanced")`, 16 genes, four rounds, four channels | Not measured by W-393; the codebook of `sample-small-v1` (`synthetic/_sample.py::sample_scene_preset`) |
| Seed, split, scene key | `AmpliconSceneConfig.seed`, `split`, `scene_key` | 42, `development`, the preset's name | Not measured by W-393; the convention of the sample presets |
| Per-round geometry | `GeometryConfig.translation_max_zyx` (voxels) | translation only, (1, 2, 2), about `round1` | Not measured by W-393; `sample-small-v1` and the W-393 sizing scenes' `config.json` |
| Amplicon appearance | the template's property laws, readout, background and noise; `dtype` | those of `sample-small-v1` (`_appearance` of `benchmark-presets-v1`), float32 | Not measured by W-393 (it measured stains and labels, not amplicon images); the W-393 sizing scenes |
| Stochastic readout effects | `ReadoutEffectsConfig` | off | Required by `SampleSceneConfig` |
| Seed kinds | `PackingConfig.kind_shares` | ordinary 0.9, binucleate 0.05, empty 0.05 | Not measured by W-393; assumption, so that the `slow` presets hold cells with two nuclei and without a nucleus |
| Empty site | `CellsConfig.empty_radius` (µm) | 2.5 | Not measured by W-393; the default |
| Crowding | `max_proposals`; `PackingConfig.on_crowded`, `floor` | 10,000; `stop`, 0.9 | Not measured by W-393; the default and C5 |
| Lobe laws | `NucleusShapeConfig.extra_lobes`, `lobe_scale`, `lobe_offset` | 1 or 2; uniform 0.5–0.8; uniform 0.5–0.9 of lobe 1's `a_l`, bounded by LB1 | Not measured by W-393; assumption. With LB1's bound every drawn nucleus is one connected shape before clipping ("Nucleus shape"; audit F of W-480 at 10,000 draws per preset and at the ends of the laws); a clipped label can have more than one piece, counted under its one ID. The generator check measures the resulting non-ellipsoid share |
| Harmonics | `NucleusShapeConfig.harmonics` | 0 (N2 off) | C2 |
| Overlap allowance, gap | `PackingConfig.overlap` κ, `gap` (µm) | 0.1, 0 | Not measured by W-393 as such (the nearest-surface distances are one or two pixels in every context); assumption, checked by audit A |
| Cell types | `CellTypeConfig` | G1 (4 types, two markers each, eight shared genes, `w_m` 0.15, `w_s` 0.0875), proportions 0.25 each, 4 regions, purity 0.9, the two marker pairs of C8 | Not measured by W-393 (no measurement resolves types); C7 and C8 |
| Ambient molecules | `MoleculeConfig.ambient_density` (µm⁻³), `ambient_abundances` | 0.002; A2 (the eight shared genes, uniform) | Not measured by W-393 (it counted reads in cells only); `sample-small-v1`. W-392's tissue-2D share of molecules off every cell label (45 %, `frame/frames.md`) is not used: it covers the whole sample, tissue-free areas included, while the presets reproduce the densest crops; at the 2D tissue routine preset it would need about 5,000 ambient molecules against 1,000 cell molecules (an outside-cell share `s` needs an ambient count `X` with `X (1 − f_T) = s / (1 − s) · (M_c + X f_T)`, `f_T` ≈ 0.46 the territory share of the area), which would swamp the cell types and the nuclear fraction |
| Stain plateau scale | `StainChannelConfig.brightness` median (float units) | 1000 | Not measured by W-393 (it measured ratios to the plateau); a scale: background and noise below are given as fractions of it |
| Noise model | `StainChannelConfig.noise` | Poisson (α 1) and read noise `σ = sqrt((r_n · 1000)² − b · 1000)` | W-393 measured `r_n`, "noise (sd of background) over plateau, per 128-px block", and `b`, "background level over plateau, per image": the background's own Poisson variance is removed from the measured noise |
| Bright-region spreads | `BrightRegionConfig.size`, `height` log SDs | `ln(p95 / p5) / (3 · 3.29)` of the region volume (`/ (2 · 3.29)` of an area); `ln((peak95 − 1) / (peak5 − 1)) / 3.29` | W-393 measured "bright region size (factor 1.5)" and "bright region peak over nucleus median (factor 1.5)" |
| Unevenness centres, length | `UnevennessConfig.centres_per_area`, `length` | 4, one FOV width | Not measured by W-393 (no spatial scale was measured); assumption |
| Growth, reach spread | `TerritoryConfig.growth`, `CellsConfig.reach` log SD | 0.15 (culture reach 0.3) | Not measured by W-393; the reach law SD of `sample-small-v1` |

**Truth of the 2D tissue projection (C19).** On the route after fusion the fused Z
maximum is segmented in one call; the nuclei of a 3D scene can overlap in that
projection, and W-393 had to define the projected truth by a rule (`notes.md` §2). The
continuous Y/X footprint of a nucleus is not what any sampled grid shows: a pixel whose
column meets the shape only between two sampled planes is in the footprint and in no
voxel label (for example a semi-axis of 2.1 µm, the nearest plane 0.1 µm from the centre
and a normalized lateral radius of 0.999: footprint radius² 0.998, sampled radius²
1.0003). The truth is therefore defined from the sampled planes.

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **PT1. A monolayer; the truth from the sampled planes (recommended)** | `PackingConfig.monolayer`: every centre at `z_mid`. The projected truth on a grid is `P(y, x) = max_q N(z_q, y, x)`, with `z_q` the centres of that grid's Z planes and `(y, x)` its pixel centres: the Z maximum of the grid's own nucleus labels. | None beyond the 2D packing. | Every power face is vertical ("Packing", "Monolayer"), so a column holds at most one nucleus and `P` is that nucleus or 0, never a choice between two. `P` is a point function of the pixel centre and the grid's planes, so it holds at any fractional offset in Z, Y and X; a nucleus that no sampled plane meets is absent from `P`, as from the voxel labels. |
| PT2. 3D packing, projected labels by a rule | Pack in 3D; the truth of a fused pixel is the largest ID of its column. | As W-393. | Nuclei overlapping in projection make the truth the larger ID by convention, not by geometry. |

**Recommendation: PT1.**

**Stain content for the stitch (C20).** The stitch placed 3 of 9 FOVs on the sparse
preset scene and all 9 on the denser variants (W-393 `notes.md` §4; choice 5).

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **SC1. Presets dense enough, truth placement as a stated fallback (recommended)** | The presets' densities come from the real tables; a chain check that needs the stitch on a preset whose stitch does not place every FOV states that and uses the truth placement. | None. | The stitch is exercised where the scene allows; the fallback is visible. |
| SC2. A weak structured stain background | Add texture to the stain's background so overlaps hold structure. | One more background component. | A structured tissue background, which this delivery excludes. |
| SC3. Truth placement always | Segmentation checks never depend on the stitch. | None. | The chain is not checked through the stitch. |

**Recommendation: SC1.**

**Z extent (C21).** The grant bound of 16 planes (5.5 µm at 0.346 µm) is about one LN
nucleus deep; real LN stacks hold 50 planes (W-393 `notes.md` §3 and choice 6).

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **Z1. Keep 16 planes; count nuclei by the Y/X border (recommended)** | The presets stay at 16 planes; the chain checks (W-395) count truth nuclei not cut by the FOV's Y or X border and report those cut by a Z end. | None. | Within the measured bound; most nuclei touch a Z end, which the checks state. |
| Z2. More planes | 32 or 50 planes. | Beyond the bound W-393 measured; a new resource grant. | Nuclei whole in Z, like the real stacks. |

**Recommendation: Z1.**

**The parameter sets (C23, C24, C25).** The same four options for each context; each
context's table names what differs for it.

| Option | Values | Cost | Consequence |
| --- | --- | --- | --- |
| **PS1. The proposed set (recommended)** | The tables below: medians of the W-393 rows, with the routine preset's molecules per cell (and the dense 3D routine density) moved within the measured range so that the routine generator checks can stay in the routine tier. | The resources of "Resources". | Every case of D4 is present at a typical strength; the routine and `slow` presets of a context differ only where the table says so. |
| PS2. Medians in both tiers | As PS1, with the routine preset taking the `slow` values of molecules per cell and density. | Routine generation grows with the molecules (about twice for 2D tissue, eight times for the culture), so its generator tests become `slow` by the 5-s rule. | The routine tier loses its generator checks on the context presets. |
| PS3. A stress set at the ends of the ranges | The p95 of the touching share, of the non-ellipsoid share, of the per-block background and of the edge width; the field and plateau spread at the ends of the between-nucleus range; bright-region peaks at their p95. | As PS1. | More method failures by design; the chain-check thresholds (W-395) would then describe the extremes of the real images, not a typical one. Better kept as a variant built with `dataclasses.replace` than as the preset. |
| PS4. Today's values with the new controls off | `sample-small-v1` on the context's grid. | Least. | None of the four cases: W-393 already measured this scene (touching 0, no lobes, background 0.38 of the plateau, no texture). |

**Recommendation: PS1** for each context.

### Dense 3D tissue: `context-dense3d-v1` (routine) and `context-dense3d-slow-v1` (`slow`)

Real data LN sample1; route per FOV (`fusion.dimensionality` `stack`); nuclear stain DAPI
as a reference stain file of the reference round (`ch04`, named `DAPI`; the existing
case, no morphology round); method StarDist `3D_spleen` at scale 1 on the stain per FOV;
cells by expansion of the nuclei by 0.7776 µm, planar (W-382's setting, W-393 `notes.md` §4).

| Parameter | Field (unit) | Routine | `slow` | Set from |
| --- | --- | --- | --- | --- |
| Grid, FOV shape | `AcquisitionConfig.columns`, `rows`, `fov_shape_zyx` (voxels) | 2 × 2, 16 × 64 × 64 | 3 × 3, 16 × 256 × 256 | Not measured by W-393; its sizing scenes `ln_routine`, `ln_slow` (`results/scenes/<name>/config.json`) |
| Spacing | `spacing_zyx` (µm) | 0.346255, 0.194398, 0.194398 | same | Not measured by W-393; the LN calibration of those configurations |
| Overlap, jitter bound | `overlap`; `jitter_max_zyx` (voxels) | 0.25; 0.8, 3.0, 3.0 | 0.25; 0.8, 4.5, 4.5 | Not measured by W-393; those configurations |
| Axial semi-axis | `CellsConfig.nucleus_axial_radius` (µm, lognormal median, log SD) | 2.1, 0.14 | same | W-393 measured `real/ln/geometry.csv` "semi-axis a3" 2.09 (1.66–2.61); log SD `ln(p95 / p5) / 3.29` |
| Lateral semi-axis | `nucleus_lateral_radius` (µm, median, log SD) | 2.7, 0.10 | same | W-393 measured "semi-axis a2" 2.73 (2.35–3.22) |
| Elongation | `nucleus_elongation` (folded lognormal μ, τ) | 0.089, 0.151 | same | W-393 measured "elongation a1/a2" 1.21 (p95 1.47): median `exp(μ + 0.674 τ)`, p95 `exp(μ + 1.96 τ)` |
| Lobed share | `NucleusShapeConfig.lobed_share` | 0.07 | same | W-393 measured "share of nuclei not described by one ellipsoid (IoU < 0.8), per image" 0.068 (0.064–0.074) |
| Density | `PackingConfig.density` (nominal volume fraction of `Ω`) | 0.16 | 0.13 | W-393 measured "nucleus volume fraction, per image" 0.126 (p95 0.156); the routine preset takes the upper end so that its four FOVs hold more nuclei |
| Contact; monolayer | `contact`; `monolayer` | 0.3; false | same | W-393 measured "share of touching nuclei, per image" 0.51 (0.44–0.57); `contact` set by audit A so that the touching share, measured on the label image, lies in that range (mean 0.51 routine, 0.53 `slow`) |
| Expected nuclei, cells | outcome | about 7, 7 | about 150, 150 | audit A (`‖Ω‖` 3,207 and 97,002 µm³) |
| Territory | `TerritoryConfig.rule`, `growth` (µm median, log SD) | `grown`, 0.78, 0.15 | same | Not measured by W-393: the route's expansion 0.7776 µm (W-382's setting, W-393 `notes.md` §4); W-393 measured "cell volume" 127 µm³ (90–185) for the expanded nuclei, which the generator check compares |
| Molecules per cell | `MoleculeConfig.per_cell_median`, `per_cell_spread` (log SD) | 8, 0.45 | same | W-393 measured `real/ln/molecules.csv` "reads per cell (kept cells)" 8 (1–19); the spread fitted by audit A to the p95 (model 2–19); the lower tail is not reproduced |
| Nuclear fraction | `MoleculeConfig.nuclear_fraction` | 0.5 | same | W-393 measured "nuclear fraction of reads, per cell with >= 10 reads" 0.50 (0.25–0.75) |
| Expected molecules | outcome | about 70 | about 1,500 | audit A |
| Stain | `StainChannelConfig.name`, `source`, `round`, `channel` | `DAPI`, `nucleus`, `reference`, `ch04` | same | Not measured by W-393; the route (D2) |
| Plateau spread | `brightness` log SD | 0.23 | same | W-393 measured `real/ln/stain.csv` "nucleus median over the image's median of nucleus medians", MAD 0.154: `1.4826 × MAD` |
| Unevenness | `unevenness` | None (`a` = 0) | same | W-393 measured the same row's p5–p95 0.70–1.44: `s_range` 0.22 does not exceed the per-nucleus 0.23 ("Stains", C14) |
| Edge | `edge_width` (µm) | 0.93 | same | W-393 measured "edge width 10-90 %" 1.39 µm (0.20–2.20), converted by the line of "The four cases" |
| Bright regions | `texture.per_nucleus`; `size` (µm median, log SD); `height` (median, log SD) | 3; 0.41, 0.37; 1.35, 0.27 | same | W-393 measured "bright regions per nucleus (factor 1.5)" 3 (1–5), "bright region size" 0.82 µm³ (0.13–5.04), peak 2.35 (1.88–3.10): `h = peak − 1`, `s = r / sqrt(2 ln(2h))` with `r = (3V / 4π)^(1/3)` |
| Background | `background.baseline` (float units); one region: `regions_enabled`, height, `region_sigma_zyx`, `tissue_weights` | 190; at the sample grid's centre, height 600, σ 24.9 µm in Y and X (128 LN pixels, given to the configuration in sample-grid voxels) and `n_z` planes in Z, weight 1 | same | W-393 measured "background level over plateau, per image" 0.19 and "background over plateau, per 128-px block" p95 0.78: baseline 0.19 · 1000, the region adds 0.60 · 1000 at its centre |
| Read noise | `noise.sigma` (float units) | 16 | same | W-393 measured noise per block 0.021: `sqrt(21² − 190)` |
| Output | `output` (`OutputScaleConfig`) | `uint8`, `g` 42 | same | W-393 measured 8-bit plateaus 35–50 (`notes.md` §9) |

What differs for this context in C23: PS2 would set the routine density to 0.13 (about 5
nuclei in four FOVs); PS3 would raise `contact` to reach the touching p95 0.57 and the
lobed share to 0.074.

### 2D tissue: `context-tissue2d-v1` (routine) and `context-tissue2d-slow-v1` (`slow`)

Real data tissue-2D; **projection samples through the route after fusion**
(`fusion.dimensionality` `projection`): `Dataset.stitch`, then `Dataset.export` fuses, takes
the Z maximum and segments it in one call with the settings of the `segmentation` block
of `export` ({doc}`sample-segmentation-contract`, "Where the settings live"). Nuclear
stain PI as a reference stain file of the reference round (`ch04`, named `PI`); nuclei by
StarDist `2D_versatile_fluo` at the scale that gives it about 0.38 µm pixels (1 on the
routine preset's 0.37854 µm, 0.5 on the `slow` preset's 0.18927 µm); cells by
`seeded_watershed` on the nuclei–amplicon composite, seeded by the nuclei. The composite is
an open point of the route ("Open points for a later route issue").

| Parameter | Field (unit) | Routine | `slow` | Set from |
| --- | --- | --- | --- | --- |
| Grid, FOV shape | `columns`, `rows`, `fov_shape_zyx` (voxels) | 2 × 2, 8 × 64 × 64 | 3 × 3, 16 × 256 × 256 | Not measured by W-393; its sizing scenes `t2_routine`, `t2_slow` (`t2_slow` at the acquired pixel, half the `slow` preset's) |
| Spacing | `spacing_zyx` (µm) | 0.346255, 0.37854, 0.37854 | 0.346255, 0.18927, 0.18927 | Not measured by W-393; the pixel StarDist sees on tissue-2D (routine), W-393 `notes.md` §7, and twice tissue-2D's acquired pixel of 0.094635 µm (`slow`; Jiahao's decision of 2026-10-15 before the gate W-396, so that a `slow` scene holds about 76 nuclei instead of about 20 at the acquired pixel; StarDist runs at the scale 0.5) |
| Overlap, jitter bound | `overlap`; `jitter_max_zyx` (voxels) | 0.25; 0.8, 3.0, 3.0 | 0.25; 0.8, 4.5, 4.5 | Not measured by W-393; those configurations |
| Axial semi-axis | `nucleus_axial_radius` (µm, median, log SD) | 2.1, 0.14 | same | Not measured by W-393 (2D labels have no Z extent: `real/tissue2d/geometry.csv` "nucleus volume" has no value); assumption: LN's value |
| Lateral semi-axis | `nucleus_lateral_radius` (µm, median, log SD) | 3.6, 0.34 | same | W-393 measured `real/tissue2d/geometry.csv` "semi-axis a2" 3.58 (1.46–4.44) |
| Elongation | `nucleus_elongation` (μ, τ) | 0.052, 0.301 | same | W-393 measured "elongation a1/a2" 1.29 (p95 1.90) |
| Lobed share | `lobed_share` | 0.04 | same | W-393 measured "share of nuclei not described by one ellipsoid" 0.038 (0.027–0.060) |
| Density | `density` (area fraction of `Ω`'s Y-X box; monolayer) | 0.36 | same | W-393 measured "nucleus volume fraction, per image", an area fraction for these 2D labels, 0.365 (0.30–0.44) |
| Contact; monolayer | `contact`; `monolayer` | 0.65; true | same | W-393 measured "share of touching nuclei, per image" 0.78 (0.76–0.82); `contact` set by audit A on the label image of the Z projection (mean 0.78 routine; 0.81 `slow` at its pixel of 0.18927 µm in W-481's re-run, within the range, so `contact` is not retuned); the monolayer is C19 |
| Expected nuclei | outcome | about 11 | about 76 | audit A (`‖Ω‖` 1,995 µm²); for the `slow` preset its re-run by W-481 (`‖Ω‖` 15,089 µm²) |
| Territory | `TerritoryConfig.rule`, `growth` (µm median, log SD) | `grown`, 0.5, 0.15 | same | W-393 measured "cell area" 66.1 µm² and "nucleus area" 52.2 µm²: equivalent radii 4.59 and 4.08 µm, a difference of 0.51 µm |
| Molecules per cell | `per_cell_median`, `per_cell_spread` | 85, 0.30 | 194, 0.30 | W-393 measured `real/tissue2d/molecules.csv` "reads per cell" 194 (85–310); the routine preset takes the p5 to stay in the routine tier; the spread fitted to the p95 (model 116–321 at median 194) |
| Nuclear fraction | `nuclear_fraction` | 0.75 | same | W-393 measured "nuclear fraction of reads" 0.75 (0.43–0.93) |
| Expected molecules | outcome | about 1,000 | about 15,800 | audit A; for the `slow` preset its re-run by W-481 |
| Stain | `name`, `source`, `round`, `channel` | `PI`, `nucleus`, `reference`, `ch04` | same | Not measured by W-393; the route (D2) |
| Plateau spread | `brightness` log SD | 0.20 | same | W-393 measured `real/tissue2d/stain.csv` "nucleus median over the image's median", MAD 0.135 |
| Unevenness | `unevenness.amplitude` | 0.22 | same | W-393 measured the same row's p5–p95 0.52–1.37: `sqrt(0.296² − 0.20²)` |
| Edge | `edge_width` (µm) | 0.25 | same | W-393 measured "edge width 10-90 %" 0.72 µm (0.31–1.57) |
| Bright regions | `per_nucleus`; `size`; `height` | 2; 0.21, 0.36; 1.0, 0.23 | same | W-393 measured per nucleus 2 (0–8); size 0.19 µm² (0.11–1.16), `r = sqrt(A / π)`; peak 2.01 (1.76–2.60) |
| Background | `background.baseline` (float units) | 25, no region | same | W-393 measured background 0.025 per image (0.022–0.031) and per block p95 0.06, which the baseline and noise cover |
| Read noise | `noise.sigma` | 10 | same | W-393 measured noise 0.011: `sqrt(11.3² − 25)` |
| Output | `output` | `uint8`, `g` 78 | same | W-393 measured 8-bit plateaus 64–93 |
| Composite share | a target of the generator check, not a field | 0.84 (0.52–0.96) | same | W-393 measured "share of pixels with signal in the composite, per cell" |

What differs for this context in C24: PS2 would take 194 molecules per cell in the routine
preset; PS3 would take the touching p95 0.82, the lobed share 0.06 and the field's
upper end.

### Sparse culture: `context-culture-v1` (routine) and `context-culture-slow-v1` (`slow`)

Real data cell-culture-3D; route per FOV; nuclear stain DAPI and cytoplasm stain
Flamingo in the morphology round `organelle` (`ch00` DAPI, `ch02` Flamingo; no `ch01`, ER
is not done), with a DAPI reference stain file in the reference round (MR1); nuclei by
Cellpose `cpsam_v2` on the DAPI Z maximum extended through z (W-382's culture setting),
cells by Cellpose on Flamingo and DAPI, extended through z (D2; W-393 `notes.md` §4). The
culture is a single-layer culture (W-305 `examples.md`), so its presets use the monolayer
and an area density.

| Parameter | Field (unit) | Routine | `slow` | Set from |
| --- | --- | --- | --- | --- |
| Grid, FOV shape | `columns`, `rows`, `fov_shape_zyx` (voxels) | 2 × 2, 16 × 64 × 64 | 3 × 3, 16 × 256 × 256 | Not measured by W-393; its sizing scenes `cu_routine`, `cu_slow` (`cu_slow` at the acquired pixel, half the `slow` preset's) |
| Spacing | `spacing_zyx` (µm) | 0.346255, 0.777592, 0.777592 | 0.346255, 0.388796, 0.388796 | Not measured by W-393; the culture grid binned 4 × 4 (routine: a 64-pixel FOV at 0.194 µm is smaller than one nucleus), W-393 `notes.md` §7, and binned 2 × 2, twice the acquired pixel of 0.194398 µm (`slow`; Jiahao's decision of 2026-10-15 before the gate W-396, so that a `slow` scene holds about 42 nuclei instead of about 12 at the acquired pixel) |
| Seed | `AmpliconSceneConfig.seed` | 42 | 42 | Not measured by W-393; the convention of the sample presets. The `slow` preset's own seed stays 42; the validation design runs the culture chain check at the two seeds 42 and 43 (Jiahao's decision of 2026-10-15; about 83 nuclei over the two scenes by W-481's re-run of audit A) |
| Overlap, jitter bound | `overlap`; `jitter_max_zyx` (voxels) | 0.25; 0.8, 3.0, 3.0 | 0.25; 0.8, 4.5, 4.5 | Not measured by W-393; those configurations |
| Axial semi-axis | `nucleus_axial_radius` (µm, median, log SD) | 2.1, 0.14 | same | W-393 measured "nucleus volume (extended 3D labels)" 1,081 µm³ and "nucleus area (2D labels)" 259 µm²: a height of 4.2 µm, half of it the semi-axis; the log SD is LN's (assumption) |
| Lateral semi-axis | `nucleus_lateral_radius` (µm, median, log SD) | 7.5, 0.30 | same | W-393 measured `real/culture/geometry.csv` "semi-axis a2" 7.48 (1.05–16.5); the log SD is an assumption, narrower than that range because its low end is debris (W-393 `notes.md` §9) |
| Elongation | `nucleus_elongation` (μ, τ) | 0.117, 0.408 | same | W-393 measured "elongation a1/a2" 1.48 (p95 2.50) |
| Lobed share | `lobed_share` | 0.24 | same | W-393 measured "share of nuclei not described by one ellipsoid" 0.24 (0.12–0.43) |
| Density | `density` (area fraction; monolayer) | 0.26 | same | W-393 measured "nucleus volume fraction, per image", an area fraction for these 2D labels, 0.26 (0.17–0.45) |
| Contact; monolayer | `contact`; `monolayer` | 0.4; true | same | W-393 measured "share of touching nuclei, per image" 0.48 (0.33–0.72); `contact` set by audit A on the label image of the Z projection (mean 0.55 routine; 0.60 `slow` at its pixel of 0.388796 µm in W-481's re-run, within the range, so `contact` is not retuned) |
| Expected nuclei | outcome | about 8 | about 42 | audit A (`‖Ω‖` 10,219 µm²); for the `slow` preset its re-run by W-481 (`‖Ω‖` 66,046 µm²); both grown by the morphology bound |
| Territory | `CellsConfig.reach` (µm median, log SD); `TerritoryConfig.rule` | 5.3, 0.3; `reach` | same | W-393 measured "cell area (2D labels)" 896 µm² (202–2291): equivalent radius 16.9 µm, less "semi-axis a1" 11.6 µm |
| Molecules per cell | `per_cell_median`, `per_cell_spread` | 120, 0.40 | 982, 0.40 | W-393 measured `real/culture/molecules.csv` "reads per cell" 982 (86–1884); the routine preset stays near the p5 for the routine tier; the spread fitted to the p95 (model 506–1901 at median 982); the lower tail is not reproduced |
| Nuclear fraction | `nuclear_fraction` | 0.26 | same | W-393 measured "nuclear fraction of reads" 0.26 (0–0.56) |
| Expected molecules | outcome | about 1,200 | about 44,500 | audit A; for the `slow` preset its re-run by W-481 |
| Nuclear stains | `name`, `source`, `round`, `channel` | `DAPI`, `nucleus`, `reference`, `ch04`; and `DAPI`, `nucleus`, `morphology`, `ch00`, otherwise the same configuration | same | Not measured by W-393; the route (D2) and MR1 |
| Plateau spread | `brightness` log SD | 0.53 | same | W-393 measured `real/culture/stain.csv` "nucleus median over the image's median", MAD 0.358 |
| Unevenness | `unevenness.amplitude` | 0.74 | same | W-393 measured the same row's p5–p95 0.25–4.92: `sqrt(0.909² − 0.53²)` |
| Edge | `edge_width` (µm) | 0.38 | same | W-393 measured "edge width 10-90 %" 0.85 µm (0.33–1.63) |
| Bright regions | `per_nucleus`; `size`; `height` | 2; 0.17, 0.64; 2.0, 0.69 | same | W-393 measured per nucleus 2 (0–58); size 0.26 µm² (0.11–7.49); peak 3.0 (1.90–9.58) |
| Background | `background.baseline` (float units); one region as in dense 3D tissue | 12; height 350, σ 24.9 µm | same | W-393 measured background 0.012 per image and per block p95 0.37 |
| Read noise | `noise.sigma` | 25 | same | W-393 measured noise 0.025: `sqrt(25.3² − 12)` |
| Output | `output` | `uint8`, `g` 20 | same | W-393 measured 8-bit plateaus 13–83; audit A, with U1 as written: at `g` 20, 0.03 % (routine) and 0.2 % (`slow`, at its pixel of 0.388796 µm in W-481's re-run) of plateaus clip, and 6 to 8 % of bright-region peaks, given the wide plateau spread (at `g` 30, 1 % of plateaus clipped with U1 before its normalization) |
| Cytoplasm stain | `name`, `source`, `round`, `channel`; `brightness` (median, log SD); `nuclear_ratio`; `edge_width` (µm); `unevenness`; background, noise, output | `Flamingo`, `territory`, `morphology`, `ch02`; 1000, 0.3; 0.3; 0.5; as DAPI's; as DAPI's | same | Not measured by W-393 (its real tables hold nuclear stains only); assumptions for the gate |
| Morphology round | `MorphologyRoundConfig.name`, `translation`, `translation_max_zyx` (voxels) | `organelle`, `uniform`, 0.5, 6, 6 | same | Not measured by W-393; the round name of the culture data and the offsets of W-305 and W-392 ("The morphology round") |

What differs for this context in C25: PS2 would take 982 molecules per cell in the
routine preset (about eight times the routine generation); PS3 would take the touching
p95 0.72, the lobed share 0.43 and the field at the ends of the 0.25–4.92 range.

### Resources

Generation estimates scale the nearest scene W-393 generated and measured (`tables/sizing.csv`,
final attempts, one CPU, `/usr/bin/time -v`) by what changes: the molecule count (each
cell molecule is placed by rejection with one label evaluation per proposal,
`synthetic/_sample.py::_molecules`), the lobes (1.1 to 1.3 times the label cost), and each
extra stain image per FOV (about the cost of today's stain). The numbers of nuclei and
molecules are audit A's. They are estimates, not measurements; the implementation
measures each preset with `/usr/bin/time -v`, and a generator test that takes 5 s or more
is marked `slow` ({doc}`contributing`, "Test markers"). The segmentation calls are W-393's
measurements on the nearest scene. For the tissue-2D and culture `slow` presets, at twice
the acquired pixel of their nearest scenes, the arithmetic is in
`results/audit/resources_estimate.txt` of `runs/W-481/20261015T160947Z-7ca392b2/`: the
whole time of the nearest scene is scaled by the molecule ratio and the lobe factor (an
over-estimate, since the voxels per FOV do not change), each extra stain image is allowed a
seventeenth of that time (one of the 17 images per FOV the nearest scene rendered), and the
peak RSS grows by the largest growth per molecule between W-393's `slow` generations
(`ln_slow` against `cu_slow`, 80 KiB) plus the extra float64 stain images.

| Preset | Nearest W-393 scene: generation | Estimate of the preset's generation | Segmentation call on the nearest scene (CPU; GPU) | Within the bound |
| --- | --- | --- | --- | --- |
| `context-dense3d-v1` | `ln_routine`: 6.06 s, 0.28 GiB (14 nuclei, 1,887 molecules) | 2 to 4 s, below 0.4 GiB (about 7 nuclei, 64 molecules; textures and the background region add about one stain evaluation) | `3D_spleen`, 4 FOVs: 179 s, 3.1 GiB; 44 s, 1.8 GiB, 8,666 MiB GPU | routine generator checks if measured below 5 s; the model call is a `learned` and `slow` chain check |
| `context-dense3d-slow-v1` | `ln_slow`: 194 s, 3.3 GiB (264 nuclei, about 40,000 molecules) | below 194 s and 3.3 GiB (about 150 nuclei, 1,500 molecules) | 779 s, 3.1 GiB; 90 s, 1.9 GiB | `slow`; within 3,600 s and 16 GiB per step |
| `context-tissue2d-v1` | `t2_routine`: 10.4 s, 0.27 GiB (26 nuclei, about 3,900 molecules) | 3 to 6 s, below 0.4 GiB (about 11 nuclei, 1,000 molecules) | export after fusion 18 s, 0.92 GiB; StarDist 2D 0.94 s warm (GPU 2.9 s) | routine if measured below 5 s, else `slow` with the routine preset unchanged |
| `context-tissue2d-slow-v1` | `t2_slow`, at the acquired pixel 0.094635 µm (half the preset's): 53 s, 1.4 GiB (49 nuclei, 7,010 molecules) | 130 to 155 s, about 2.1 GiB at most (about 76 nuclei, 15,800 molecules: 53 s × 2.25 × 1.1 to 1.3) | export 22 s, 0.95 GiB; seeded watershed 0.96 s (on `t2_slow` at 0.094635 µm, StarDist at the scale 0.25; at the scale 0.5 it sees four times the pixels) | `slow`; within the bound |
| `context-culture-v1` | `cu_routine`: 6.4 s, 0.24 GiB (8 nuclei, 2,400 molecules) | 3 to 7 s, below 0.4 GiB (about 8 nuclei, 1,200 molecules; two more stain images and the morphology round per FOV) | Cellpose nuclei, 4 FOVs: 117 s, 2.3 GiB; 31 s, 3.0 GiB, 2,284 MiB GPU; extension through z 1.0 s | as above; the cell call was not runnable on the current scene, estimated at the nucleus call's time (same model and image size) |
| `context-culture-slow-v1` | `cu_slow`, at the acquired pixel 0.194398 µm (half the preset's): 64 s, 1.4 GiB (44 nuclei, 12,732 molecules) | 250 to 300 s, about 4.0 GiB at most (about 42 nuclei, 44,500 molecules, two more stain images and the morphology round: 64 s × 3.49 × 1.1 to 1.3, plus 2 × 3.7 s for the stain images) | 261 s, 2.3 GiB; 59 s, 3.2 GiB; extension through z 13.7 s (on `cu_slow` at 0.194398 µm, nuclei twice as many pixels across as at the preset's pixel) | `slow`; within the bound; each of the two seeds of the chain check is one such scene |

**The resource bound of the `slow` presets (C26).** They use 3 × 3 FOVs of 16 × 256 ×
256, the bound of Jiahao's resource exception for W-393 (granted 2026-10-08, recorded on
W-392), not the §2.10 bound of `sample-small-v1` (one scene of at most 16 × 256 × 256 cut
into at most 3 × 3 tiles). Every final W-393 attempt lay within 3,600 s and 16 GiB
(`notes.md` §7). The tissue-2D and culture `slow` presets take twice their context's
acquired pixel (Jiahao's decision of 2026-10-15 before the gate W-396), so that a scene
holds about 76 and 42 nuclei instead of about 20 and 12; their nearest W-393 scenes
`t2_slow` and `cu_slow` were measured at the acquired pixel, and their estimates above
(at most 155 s and 2.1 GiB, 300 s and 4.0 GiB) lie within the same bound.

| Option | Bound for the implementation batch | Cost | Consequence |
| --- | --- | --- | --- |
| **RG1. Grant the W-393 bound again (recommended)** | Scenes up to 3 × 3 FOVs of 16 × 256 × 256; one CPU, 16 GiB and 3,600 s per step; the GPU for the model calls of the chain checks. | A grant at the gate. | The `slow` presets as specified; each nearest scene was measured inside it (`t2_slow` and `cu_slow` at half the pixel of their presets), and the estimates of the presets lie inside it. |
| RG2. Keep the §2.10 bound | One scene of at most 16 × 256 × 256 cut into 3 × 3 tiles (FOVs of about 16 × 96 × 96 to 16 × 120 × 120). | None. | The `slow` presets shrink to about a sixth of the area: about 25 dense 3D nuclei, 13 tissue-2D nuclei and 7 culture nuclei, and a culture nucleus (22 µm long) fills about half of a FOV of 96 to 120 pixels at 0.389 µm. |
| RG3. No `slow` presets | Routine presets only. | None. | The chain checks run on 7 to 9 nuclei per scene, too few for the stitch (C20) and for thresholds with any resolution. |

**Recommendation: RG1** (entry 12 of "Accepted statements this addition changes").

## Truth schema

| Option | Where the new truth goes | Cost | Effect on existing outputs |
| --- | --- | --- | --- |
| **V1. New files beside the unchanged sample layout, with their own version (recommended)** | `save_cellular_scene` writes the `save_sample_scene` layout unchanged, plus the files below and `context.json` with `context_truth_version` `"1"`. | Consumers join by `nucleus_id`, `cell_id`, `amplicon_id`, `fov_id` and the stain name; twelve new tables and the stain images. | None: no existing table gains a column; `sample.json` keeps `sample_truth_version` `"1"`. |
| V2. New columns on the sample tables | `cells.csv` gains `type`, `nuclei.csv` gains lobe columns. | Fewer files. | A table's columns would depend on how the scene was made, as {doc}`synthetic-sample-scene` rejected (option V2 there). |
| V3. `SAMPLE_TRUTH_VERSION` `"2"` for every scene | One schema with the new columns. | One version. | Changes every saved sample scene, including today's presets. |

**Recommendation: V1.** A scene of {doc}`synthetic-real-appearance` writes this layout
with the ellipsoid columns null, and adds `appearance.json`, `placements.csv` and
`background_patches.csv`.

```
<directory>/
  <the save_sample_scene layout, unchanged>     # sample.json, the seven tables, labels, fovs/<fov_id>/...
  context.json                 # context_truth_version "1", context_preset_version, the requested and effective
                               #   CellularSceneConfig, the reached density and placed seeds (F2), the stream
                               #   scheme and descriptors, the SHA-256 of every file of the directory
  lobes.csv                    # nucleus_id, lobe, offset_z/y/x (µm), axial_radius, lateral_radius (µm),
                               #   elongation, angle (rad); lobe 1 repeats nuclei.csv's ellipsoid with offset 0
  harmonics.csv                # nucleus_id, degree, order (an integer, or cos / sin in one plane or a
                               #   monolayer), coefficient; empty without N2
  cell_context.csv             # cell_id, kind, type, region, molecule_mean (M2's μ_k), growth (µm, T2's δ;
                               #   null otherwise)
  regions.csv                  # region_id, centre_z/y/x (µm), main_type
  type_abundances.csv          # type, gene_id, weight, role (marker, shared); and the ambient row (type "ambient")
  cell_gene_counts.csv         # cell_id, gene_id, count, nuclear_count: the molecules whose true_cell_id is the
                               #   cell, by gene (long form, nonzero rows only)
  marker_pairs.csv             # gene_a, gene_b, relation (exclusive, coexpressed), cells_a, cells_b, both,
                               #   either, mecr (null when either is 0): the truth values at positive_threshold 1
                               #   over the sample cells
  stain_channels.csv           # stain name, source, round, channel, output dtype, scale k of B1 (1 without an
                               #   output scale), m_A
  stain_objects.csv            # stain name, object (nucleus or cell), object ID, plateau (A_n or B_k)
  bright_regions.csv           # region_id, nucleus_id, stain, centre_z/y/x (µm), size (µm), height
  unevenness.csv               # stain, bump_id, centre_z/y/x (µm), z (U1's standard normal), a, length (µm)
  morphology_truth.csv         # "The morphology round"
  fovs/<fov_id>/stains/<name>.ome.tif          # each reference-round stain (ZYX, its output dtype)
  fovs/<fov_id>/morphology.ome.tif             # the morphology round (ZYXC, its stains in configuration order)
```

The raw dataset of `save_cellular_dataset` is a separate directory ("The morphology
round"). `molecules.csv` keeps its columns; its `compartment` column already records the
nucleus or cytoplasm placement, and its `gene_id` the type's draw. An empty population
gives a table with its header and no rows (a type with no cell, a cell with no molecule,
a scene without a morphology round). Stain images are saved by `save_volume` as OME-TIFF
in their own dtype (float32 or uint8, both written and read back unchanged); CSV floats
use round-trip precision, as {doc}`synthetic-sample-scene` states.

**Which truth holds each parameter.** Every parameter of a new object is either a column
of a table above or recomputed from the seed and a named stream ("Random streams"):

| Object | Parameters of its function | Truth table and column, or stream |
| --- | --- | --- |
| Nucleus lobes | centre, semi-axes, elongation, angle per lobe | `nuclei.csv` (lobe 1, `site_radius`), `lobes.csv` |
| Harmonic perturbation | coefficients `a_lm` | `harmonics.csv` |
| Packing outcome | centres, site radii, kinds, reached density | `nuclei.csv`, `cells.csv` (`anchor_*`, `kind`), `cell_context.csv` `kind`, `context.json` |
| Territories | rule, reach or growth per cell | `cells.csv` `reach`, `cell_context.csv` `growth`, `context.json` (the rule) |
| Cell types | regions, type per cell, abundances | `regions.csv`, `cell_context.csv` `type`, `region`, `type_abundances.csv` |
| Molecules | positions, genes, compartment, mean per cell | `molecules.csv`, `cell_context.csv` `molecule_mean` |
| Ambient molecules | positions, genes, abundance | `molecules.csv` (`population` `ambient`), `type_abundances.csv` row `ambient` |
| Nuclear, territory and boundary stains | plateau per object, edge, band, nuclear ratio, scale | `stain_objects.csv`, `stain_channels.csv`, `context.json` (the configuration) |
| Bright regions | centre, size, height | `bright_regions.csv` |
| Unevenness field | centres, `z_q`, `a`, length | `unevenness.csv` |
| Stain background | `BackgroundConfig` components | `context.json` (effective configuration), as `sample.json` records the stain's today |
| 8-bit cast | `k`, `g`, `m_A` | `stain_channels.csv` |
| Stain noise | per-FOV residuals | not a table: recomputed from the seed and the streams `stain.noise.*` with scene key `[scene_key, fov_id]` |
| Morphology round and its translation | `τ_f` | `morphology_truth.csv` |
| Projection truth | none beyond the nuclei and the grid's planes | recomputed from the nucleus truth (`P` of C19) |

In a scene of {doc}`synthetic-real-appearance` the placed objects' parameters are in
`placements.csv`, with the bank named by its SHA-256 in `appearance.json`; the background
patches are in `background_patches.csv`, and `κ` and `g` in `appearance.json`.
`lobes.csv`, `harmonics.csv`, `stain_objects.csv`, `bright_regions.csv` and
`unevenness.csv` are empty, and no stain noise is drawn (OB1).

**Identities a test can check** (in addition to identities 1 to 5 of
{doc}`synthetic-sample-scene`, which hold with the new label functions). A "boundary
voxel" below is a voxel whose centre lies within 1e-9 µm of a power face, of a lobe or
harmonic surface, of a territory limit, or of a stain's cut-off surface (`ρ_n = 1 + 4w / a_min`,
`h_n = 4w` and the territory's `d_k = 4w`, where the value jumps by up to about `3 · 10⁻⁵` of
the plateau, as today's stain does; the box of `4 s_b` of a bright region and the band's
`β_k = 4b`, where it jumps by up to `3.4 · 10⁻⁴` of the region's height or of the band's
peak); as for the aligned crop of
{doc}`synthetic-sample-scene` (identity 2), a test that recomputes a label with its own
arithmetic reports the number of boundary voxels where it differs and expects 0.
Identities 2 (labels from the shape tables), 7 (stains equal their closed forms), 9
(bright regions) and 10 (`k = g / m_A`) are those of the closed-form scene. A scene of
{doc}`synthetic-real-appearance` states its own ('Identities a test can check' there).
Identities 1, 3 to 6 and 8 hold there as written; identity 11 holds for its 2D tissue and
is reported for its culture.

1. **O1.** "Opt-in" above.
2. **Labels.** Per-FOV and sample labels equal `N` and `C` (with lobes, harmonics,
   clipping and the territory rule) recomputed from `nuclei.csv`, `lobes.csv`,
   `harmonics.csv`, `cells.csv` and `cell_context.csv` at the voxel centres, at any
   fractional offset in Z, Y and X and at `Z=1`, except at boundary voxels (reported). A
   nucleus or cell smaller than a voxel may hold no voxel centre of a grid; it then has no
   value there, as identity 2 of {doc}`synthetic-sample-scene` states.
3. **Nucleus inside cell.** On every label voxel, `N = n > 0` implies `C = cell(n)`; it
   holds on boundary voxels too, because both labels come from one evaluation. Every
   anchor's own evaluation gives its own cell (the anchor lies strictly in its power
   cell).
4. **Types and genes.** Every cell has one row of `cell_context.csv`; every molecule placed
   by a cell (`population` `cell`) has a gene of nonzero weight in its cell's type; with
   A2 no ambient molecule has a marker gene. A type with no cell has rows in
   `type_abundances.csv` only.
5. **Counts.** `cell_gene_counts.csv` equals the counts of `molecules.csv` grouped by
   `true_cell_id` and `gene_id` (integers, exact), and its sum per cell equals
   `cells.csv` `n_molecules` (0 for a cell without rows).
6. **Marker answers.** `marker_pairs.csv` equals MECR recomputed from
   `cell_gene_counts.csv` by the report's definition over the sample cells (integers;
   `mecr` null when `either` is 0); the exclusive pair has `both` 0.
7. **Stains.** Rendered with noise disabled, no output scale and the template dtype
   `float64`, each stain image equals its signal function (with background) at the voxel
   centres within 1e-9 of the image maximum; rendered again at `true_f + k ⊙ s` for an
   integer vector `k`, it equals the first image shifted by `k` on the common voxels within
   1e-10 of the maximum (the no-interpolation identity). The tolerance covers the rounding
   bound that W-395's `scripts/tolerance_check.py` derives for the full stains (with the
   field `U`, the bright regions and the territory source's nuclear factor): 7.3e-11 of the
   maximum at a band of 0.01 µm and at most 1.3e-11 for the other sources
   (`results/tolerances/tolerance_check.txt` in `runs/W-395/20261015T080623Z-62fad7ad/`);
   an interpolation changes values by many orders more. Both hold at any fractional offset
   in Z, Y and X and at `Z=1`, except at boundary voxels (reported). For the nuclear
   source, a FOV beyond the cut-off of every nucleus holds the background only; a FOV whose
   voxel centres hold no nucleus label can still hold the soft edge of a nearby or sub-voxel
   nucleus, which the signal function gives. The territory and boundary sources are lit by
   cells with or without a nucleus (an empty cell lights them), so a FOV beyond every
   nucleus's cut-off can hold them. A FOV beyond the cut-off of every cell territory
   (`d_k > 4w` for every cell `k` at each voxel centre) holds the background only for the
   territory source, and a FOV beyond the band's cut-off of every cell (`β_k > 4b`) holds
   the background only for the boundary source; a FOV whose voxel centres hold no cell label
   can still hold the soft edge or the band of a nearby cell. Under T3 every point of `Ω`
   lies in a territory, so only a scene without cells has a FOV beyond every territory's
   cut-off. With float32 output or the 8-bit cast the comparison is not made: a rounding
   near a half grey level can differ.
8. **Morphology round.** Rendered as in identity 7, the morphology image at index `q`
   equals the stain functions at `true_f + τ_f + q ⊙ s`; with a supplied `τ_f ⊘ s` that is
   a whole number of voxels on every axis, a morphology stain whose configuration equals a
   reference stain's apart from `round` and `channel` (as the two DAPI channels of the
   culture presets) equals that reference stain shifted by `τ_f ⊘ s` on the common voxels,
   within 1e-10 of the maximum (the derived bound of identity 7), except at boundary voxels
   (reported). At `Z=1`, `τ_z = 0`.
9. **Bright regions.** Every region's centre was accepted where `N = n` in the same
   evaluation that labels voxels, so `N(c_b) = n` exactly; a centre on a boundary is
   evaluated by the same function.
10. **8-bit.** The 8-bit image equals `clip(rint(J · k), 0, 255)` of the float64 image `J`
    of identity 7 rendered with noise and with the same streams, `k = g / m_A` from
    `stain_channels.csv`; `k` is a number from the configuration, defined also for a FOV
    without a nucleus.
11. **Projection truth (PT1).** On any grid (a FOV's, the sample grid or the fused grid),
    no column of fixed `(y, x)` holds two nucleus IDs, except in columns within 1e-9 µm of
    a power face (reported, expected 0), and `P(y, x)` is the Z maximum of that grid's
    nucleus labels by definition. `P` is not compared with a continuous footprint.

The comparisons of the chain checks, their tolerances and thresholds are W-395's (open
point).

## Random streams

Every new draw uses the descriptor of {doc}`synthetic-specification`, unchanged:
`["starfinder.synthetic/1", split, seed, scene_key, component, entity, round_label,
channel_label]` (`synthetic/_sample.py::_streams`), with new component names, all
prefixed `context.`:

| Component | Entity | Draws |
| --- | --- | --- |
| `context.cell.kind` | `cell-<k>` | one draw of the kind (density mode) |
| `context.nucleus.lobed` | `nucleus-<n>` | one uniform (lobed or not) |
| `context.nucleus.lobes`, `context.nucleus.lobe.scale`, `context.nucleus.lobe.offset`, `context.nucleus.lobe.direction`, `context.nucleus.lobe.angle` | `nucleus-<n>` (count); `nucleus-<n>-lobe-<l>` (the rest) | one value each; the direction is two values (one in a plane or a monolayer) |
| `context.nucleus.harmonics` | `nucleus-<n>` | one standard normal per coefficient, in order of degree and order |
| `context.cell.contact` | `cell-<k>` | per proposal: one uniform; for a contact proposal also the partner index and up to 16 directions |
| `context.cell.growth` | `cell-<k>` | one value (T2) |
| `context.region.placement`, `context.region.type` | `region-<r>` | z, y, x; one type |
| `context.cell.type` | `cell-<k>` | one uniform (purity), then one type draw when it fails |
| `context.molecule.mean` | `cell-<k>` | one lognormal value (M2) |
| `context.molecule.compartment` | molecule ID | one uniform (NF1) |
| `context.identity`, `context.ambient.identity` | molecule ID | one gene draw with the type's or the ambient abundance |
| `context.stain.brightness` | `nucleus-<n>` or `cell-<k>` | one value; `channel_label` the stain name |
| `context.texture.count`, `context.texture.placement`, `context.texture.size`, `context.texture.height` | `nucleus-<n>` (count); `nucleus-<n>-region-<b>` | as named; `channel_label` the stain name |
| `context.unevenness.count`, `context.unevenness.placement`, `context.unevenness.amplitude` | `field` (count); `bump-<q>` | one Poisson value; z, y, x; one value; `channel_label` the stain name |
| `context.morphology.translation` | FOV ID | `u_z, u_y, u_x` (MT1) |
| `stain.noise.dependent`, `stain.noise.independent`, `stain.noise.correlated` (per FOV, scene key `[scene_key, fov_id]`) | none | as today, with the stain's name as `channel_label`; for a morphology stain also the round's name as `round_label` |

**An existing draw does not change.**

1. **The sample scene's code path is unchanged** (E1): `generate_sample_scene` draws
   exactly what it draws today; the golden test pins its saved outputs for both presets.
2. **No new descriptor equals an existing one.** Existing components are those of the
   formed model (`geometry.*`, `noise.*`, `background.*`, the property and identity
   components; {doc}`synthetic-specification`, "Random streams"), the `sample.*`
   components and `stain.noise.*` of {doc}`synthetic-sample-scene`
   (`synthetic/_sample.py::_SAMPLE_COMPONENTS`). Every new component starts with
   `context.`, which no existing component does, so a new descriptor differs from every
   existing one in its component field. The per-FOV stain noise keeps its component names
   and differs by `channel_label` (a stain name other than the sample scene's `DAPI`, or the
   same name in another round, which sets `round_label`); a cellular scene whose nuclear
   stain is named `DAPI` in the reference round draws its noise from the same streams as the
   sample scene's stain.
3. **With a new part off, its draws are not made.** A part at its default (None) consumes
   no draw (as disabled controls do in {doc}`synthetic-specification`, "Stages and effect
   order"); with `packing` None, placement is today's rule with today's draws. With
   `packing` given, the uniform proposals of P1 still come from `sample.cell.placement`
   in today's order of draws, and every new draw (kind, contact, direction) comes from a
   `context.` stream.
4. **Within a cellular scene**, each new draw has its own entity, so changing a law changes
   no draw of any stream; what the draws produce can change. Changing a type's gene
   abundance changes only the genes of that type's molecules. Changing a type's molecules
   per cell or nuclear fraction changes the number of that type's molecules and which
   compartment each is placed in, so their positions change; molecules of other cells keep
   theirs. Changing a type's size scale, a nucleus's lobes or any size law can move later
   seeds (sequential placement), and with them every molecule of the moved cells, as
   changing a size does today.

## Accepted statements this addition changes

Each statement of {doc}`synthetic-sample-scene` (Accepted, W-329) that the addition would
change, with the old and the proposed wording, in the order of that page. That page is
not edited here; an implementation issue changes each statement as written and nothing
else. Audit B of W-394 checks this list statement by statement: a scan of that page
prints every line with a formula, a code expression with `=`, `≤`, `≥`, `⊙` or `⊘`, or the
words "equals", "must", "raises", "never", "always", "exactly"
(`results/audit/accepted_statements_scan.txt`, 145 lines), and its checklist marks each
line unchanged, with the reason, or changed by an entry below
(`results/audit/accepted_statements_checklist.md`); `accepted_sections.txt` summarizes it
by section.

1. **"What this addition is not", first item.**
   *Old:* "It does **not establish the accuracy of a learned segmenter.** No segmenter is
   run on this scene as a measure of its quality, and no segmentation accuracy is claimed
   from it."
   *Proposed:* "It does **not establish the accuracy of a segmenter.** The §2.9 methods,
   learned ones included, run on the cellular scene of {doc}`synthetic-cellular-scene` as
   an engineering check of the chain (D1 of the §2.12 planning decisions), never as a
   measure of a method's quality, and no segmentation accuracy is claimed from it."
2. **"What this addition is not", second item.**
   *Old:* "It is **not a calibrated tissue model.** Nuclei are smooth ellipsoids, cells are
   power-diagram territories, molecules are uniform within them; sizes and densities are
   development choices, not fitted to tissue, and no empirical profile is used."
   *Proposed:* "It is **not a calibrated tissue model.** In this scene nuclei are smooth
   ellipsoids, cells are power-diagram territories and molecules are uniform within them;
   the cellular scene of {doc}`synthetic-cellular-scene` adds lobed and touching nuclei,
   cell types and placement by compartment; its real-appearance scene places real nuclei
   and culture cells instead. Sizes and densities are development targets,
   set there from W-393's measurements of method labels, not fitted to tissue and not a
   D04 calibration."
3. **"Terms", the halo.**
   *Old:* "The **halo** is `Ω \ K`, a band of width `2 · J_a · s_a` on each outer side."
   *Proposed:* "The **halo** is `Ω \ K`, a band of width `2 · J_a · s_a` on each outer side
   (`(2 · J_a + T_a) · s_a` with the morphology round of {doc}`synthetic-cellular-scene`)."
4. **"The sample scene", "Scene domain", "Covered region" and "Independence from the
   drawn jitter".**
   *Old:* "`Ω_a = [−s_a/2 − J_a·s_a, N_a + (n_a − ½)·s_a + J_a·s_a)`", "that is `Ω` shrunk
   by `2 · J_a · s_a` on each side" and "`Ω`, `K` and membership depend on the nominal grid
   and the jitter bound, not on the drawn jitter."
   *Proposed:* the first two the same, followed by "With a morphology round of translation
   bound `T_a` voxels ({doc}`synthetic-cellular-scene`), `Ω` grows by `T_a · s_a` on each
   side, so that the shifted round images generated content; `K` is unchanged, and is then
   `Ω` shrunk by `(2 · J_a + T_a) · s_a`."; and "`Ω`, `K` and membership depend on the
   nominal grid, the jitter bound and the morphology round's translation bound, not on the
   drawn jitter or translation."
5. **"The sample scene", "Sample membership".**
   *Old:* "The anchor always lies inside its own territory, because it is inside its site
   sphere ("Cell territories")."
   *Proposed:* "The anchor always lies inside its own territory: here because it is inside
   its site sphere ("Cell territories"); in the cellular scene of
   {doc}`synthetic-cellular-scene`, whose site spheres may overlap, because its placement
   test keeps every centre strictly inside its own power cell; in the real-appearance
   scene, because a proposal whose anchor lies in a placed mask is refused."
6. **"Nuclei", first paragraph.**
   *Old:* "Nucleus `n` is an ellipsoid with centre `c_n` (µm), axial semi-axis `a_z`,
   lateral semi-axes `a_l · e` and `a_l`, ..."
   *Proposed:* the same, followed by "In the cellular scene of
   {doc}`synthetic-cellular-scene` a nucleus is a union of ellipsoids, lobe 1 optionally
   perturbed by spherical harmonics, and its site sphere bounds every lobe; in the
   real-appearance scene a nucleus is a placed real mask."
7. **"Nuclei", "Placement (random sequential addition)".**
   *Old:* "Seeds are placed in order: ordinary cells, then binucleate cells, then empty
   cells. Each proposal draws a centre uniformly in `Ω` from the seed's own
   `sample.cell.placement` stream; it is accepted when its site spheres keep a gap of at
   least `gap` to every accepted sphere (`‖c_i − c_j‖ ≥ R_i + R_j + gap`). ... A seed with
   no accepted proposal after `max_proposals` raises `ValueError` naming it: nothing is
   dropped silently."
   *Proposed:* the same, followed by "The cellular scene of {doc}`synthetic-cellular-scene`
   replaces this rule when its packing settings are given: the kind of each seed may be
   drawn, a proposal may be placed at its contact distance against an accepted site, it is
   accepted by a shape-aware contact test that lets site spheres overlap, and with
   `on_crowded="stop"` placement ends at the first seed that cannot be placed, recorded in
   `context.json`, and raises `ValueError` only below a stated share of the target
   density; the real-appearance scene places objects by PL2 and stops as 'When placement
   stops' states. Nothing is dropped silently there either."
8. **"Cell territories and the label-generation rule": "Nucleus label" and "Cell label".**
   *Old:* "`N(σ) = n` where `σ ∈ Ω` and `ρ_n(σ) ≤ 1`, else 0"; "Nuclei never overlap,
   because their site spheres keep a positive gap."; "Then `C(σ) = cell(j*)` when `σ ∈ Ω` and `‖σ − c_{j*}‖ ≤ R_{j*} + reach(cell(j*))`, else
   0."; and "Because the site spheres are disjoint, each lies inside its own power cell
   (`pow_i ≤ 0` inside sphere `i`, `pow_j > 0` outside sphere `j`), so **every nucleus lies
   inside its own cell** and `N(σ) = n` implies `C(σ) = cell(n)`."
   *Proposed:* "`N(σ) = n` where `σ ∈ Ω`, `ρ_n(σ) ≤ 1` and a site of `n` is `j*`, else 0 (with
   disjoint site spheres every point with `ρ_n ≤ 1` has `j*` a site of `n`, so this is the
   rule above; the cellular scene of {doc}`synthetic-cellular-scene` needs the condition,
   because its site spheres may overlap)"; "Nuclei never overlap: a point carries only the
   nucleus of its power-nearest site. Here the site spheres keep a positive gap, so nuclei do not touch; the cellular
   scene of {doc}`synthetic-cellular-scene` lets site spheres overlap and nuclei touch on
   the faces of their power cells; in the real-appearance scene a point carries the
   nucleus of the first placed object that holds it, and on every nucleus voxel the cell
   label is that nucleus's cell."; the cell label the same, followed by "(the cellular
   scene adds two other outer limits: a grown nucleus, and none)"; and "A nucleus is
   labelled only where its own site is the power-nearest one and its shape function is at
   most 1, and its shape lies within its territory's outer limit, so **every nucleus lies
   inside its own cell** and `N(σ) = n` implies `C(σ) = cell(n)`; here the site spheres are
   disjoint, so no nucleus is cut by a power face."
9. **"Stain", the paragraph after `StainConfig`.**
   *Old:* "The stain is one ZYX image per FOV, in the template's dtype, saved as
   `fovs/<fov_id>/stain.ome.tif` ... It is the `reference_stain` input of `stitch` ... and
   the nuclear image of the cellular scene."
   *Proposed:* the same, followed by "The cellular scene of
   {doc}`synthetic-cellular-scene` renders several stains per FOV, each in its own output
   dtype; its first nuclear stain of the reference round takes this role."
10. **"Cell truth for molecules", the template's laws and "Populations".**
    *Old:* "Its property laws (`brightness`, widths, `elongation`, `angle`) and
    `abundances` are used for every generated molecule." and "Cell `k` receives
    `Poisson(molecules_per_cell)` molecules ... and is accepted where `C(σ) = k`".
    *Proposed:* the same, followed by "In the cellular scene of
    {doc}`synthetic-cellular-scene`, a molecule's gene is drawn from its cell type's
    abundance (an ambient molecule's from the ambient abundance), a cell's count is
    Poisson with a mean drawn per cell, and a cell molecule is placed in its nucleus with a
    set probability, else in the rest of its territory."
11. **"Identities a test can check", identity 2, "Aligned crop".**
    *Old:* "Since `origin_sample = −J ⊙ s`, `δ_f = nominal_f ⊘ s + j_f + J`. For a nominal step
    that is a whole number of voxels (24 in `sample-tiny-v1`, 72 in `sample-small-v1`), the
    condition holds exactly when `j_{f,a} + J_a` is a whole number on every axis. Within the
    bound that means `j_{f,a} ∈ {−J_a, −J_a + 1, …}` up to `+J_a`." and "Zero jitter is not
    aligned unless `J` is whole".
    *Proposed:* the same, followed by "With the morphology round of
    {doc}`synthetic-cellular-scene`, `Ω` grows by `T ⊙ s` on each side, so
    `origin_sample = −(J + T) ⊙ s` and `δ_f = nominal_f ⊘ s + j_f + J + T`; the condition then
    holds exactly when `j_{f,a} + J_a + T_a` is a whole number on every axis, and zero jitter
    is aligned only when `J + T` is whole. For the culture presets (`J_z = 0.8`, `T_z = 0.5`)
    the supplied jitter `j_z = −0.8` gives `δ_z = 0.5`, not 0; `j_z = −0.3` aligns."
12. **"Presets and resources", "Not in either tier".**
    *Old:* "Larger grids, LN-sized FOVs (50 × 1496 × 1496) and whole-sample scenes are
    outside this addition".
    *Proposed:* "Larger grids, LN-sized FOVs (50 × 1496 × 1496) and whole-sample scenes are
    outside this addition; the `slow` context presets of {doc}`synthetic-cellular-scene`
    use 3 × 3 FOVs of 16 × 256 × 256, under their own resource grant."
13. **"Truth schema", the version rule.**
    *Old:* "`sample.json` names `SAMPLE_TRUTH_VERSION`; any change to a column or rule above
    bumps it."
    *Proposed:* the same, followed by "A scene of {doc}`synthetic-cellular-scene` writes the
    same files with the same columns, made by the rules of that page (lobes, clipping,
    territories, molecules by compartment); its `context.json` names
    `CELLULAR_TRUTH_VERSION`, which versions those rules, and `sample.json` keeps
    `sample_truth_version` `"1"`; a real-appearance scene writes them with the ellipsoid
    columns null, versioned by `APPEARANCE_TRUTH_VERSION` in `appearance.json`."
14. **"Exclusions".**
    *Old:* "no calibration to real tissue and no empirical profiles; no claim about
    segmentation accuracy, and no learned segmenter run on this scene as a measure of its
    quality"
    *Proposed:* "no calibration to real tissue (the development targets of
    {doc}`synthetic-cellular-scene` are not one); no claim about segmentation accuracy,
    and no segmenter run on this scene as a measure of its quality"

The other accepted pages were read for statements the scene changes. The statement of
{doc}`stitching-assembly-validation` that the sample scene's cells are "power-diagram
territories around ellipsoids, not a tissue model" stays true of that scene and of this
one, and is not listed.

## Open points for a later route issue

These are W-393 choice 7; they are not scene changes and not statements of
{doc}`synthetic-sample-scene`. The route after fusion has no composite operation: a run
reads the fused projections it names and stacks them as its input channels
(`export/_segment.py::segment_fused`), and `seeded_watershed` takes one channel, so
W-393 made the nuclei–amplicon composite per FOV and fused it, where tissue-2D's legacy
route made it from the fused images. `SampleSegmentationConfig.device` applies to every
run of the block and is refused when a run's method does not run on it
(`export/_config.py::SampleSegmentationConfig.__post_init__`), and
`seeded_watershed` runs on the CPU only (its entry in
`segmentation/_methods.py::SEGMENTATION_METHODS`), so StarDist cannot run on the GPU
beside it (W-393 `notes.md` §4). Proposed: a composite made from the fused images, and a
device per run.

## Exclusions

* **No implementation** in this batch: no change to the generator, its presets or its
  tests, except the golden test that pins today's outputs. The draft implementation
  issues are W-395's.
* **No validation design:** which comparisons the chain checks make, their tolerances and
  thresholds belong to W-395 and the gate; this page names the identities only.
* **No microsim and no new dependency** (D5); the W-392 spike's result is cited, not acted
  on.
* **No raw acquisition, no optical model, no depth attenuation:** not requested; W-393
  showed no need (`notes.md` §8).
* **No calibration:** the presets are development targets; the real labels are method
  output, the tissue-2D crops are the densest tiles of that sample, and one seed per scene
  gives small counts (W-393 `notes.md` §12).
* **No ER stain** (D2), no per-FOV illumination, no cross-FOV cell matching: outside this
  delivery. A structured background from real patches is part of
  {doc}`synthetic-real-appearance`.
* **No accuracy claim, comparison of methods, training or annotation** (D1, D7; W-93 keeps
  synthetic truth and calibration, W-123 annotation).
* **No real data** is read by this page's golden test, the routine tier or the CI
  environment. The appearance banks of {doc}`synthetic-real-appearance` are built from real
  images outside the repository, and only `slow` tests on the lab machine read them.
* **No change to an accepted page:** the changes are listed above, for the gate.

## Implementation

This page's scene is implemented in `starfinder.synthetic` by W-483 to W-485 and checked by
the generator checks of {doc}`synthetic-cellular-scene-validation` (W-486). The real-appearance
scene of {doc}`synthetic-real-appearance` (W-516 to W-520) sits beside it and changes none of
it (E1, PR1 there).

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

* the generator checks GN1 to GN24 (GN25 retired) of {doc}`synthetic-cellular-scene-validation`: `test_synthetic_cellular_contract.py`,
  `test_synthetic_cellular_scene.py` (W-483), `test_synthetic_cellular_stains.py`,
  `test_synthetic_cellular_stains_contract.py` (W-484), `test_synthetic_cellular_presets.py`,
  `test_synthetic_cellular_presets_contract.py` (W-485), `test_synthetic_cellular_routine.py`
  and `test_synthetic_cellular_checks.py` (W-486), with the helpers `cellular_oracle.py`,
  `cellular_scenes.py`, `cellular_measures.py` and `cellular_checks.py`; each row's tests are
  named in that page's "Implemented checks";
* the generator checks GA1 to GA27 of {doc}`synthetic-real-appearance-validation`:
  `test_synthetic_appearance_bank.py`, `test_synthetic_appearance_golden.py` (W-516),
  `test_synthetic_appearance_scene.py` (W-517), `test_synthetic_appearance_stains.py`,
  `test_synthetic_appearance_contract.py` (W-518), `test_synthetic_appearance_presets.py`,
  `test_synthetic_appearance_presets_contract.py` (W-519, GA23 added by W-520),
  `test_synthetic_appearance_coverage.py` and `test_synthetic_appearance_checks.py` (W-520),
  with the helpers `cellular_appearance_oracle.py`, `cellular_appearance_scenes.py` and
  `cellular_appearance_measures.py`; each row's tests are named in that page's "Implemented
  checks";
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
* **2026-10-18, W-488** (`docs/examples/cellular_scene.py`, {doc}`api/examples`). The example
  keeps the two presets the issue names, `context-tissue2d-v1` and `appearance-tissue2d-v1`,
  and its test `test_synthetic_context_example.py` is `slow` instead of routine: generating
  and saving the two presets takes more than 5 s on one CPU, the bound of a routine test,
  and that time is the package's. Measured on one CPU: the example 9.25 s, its test 8.57 s
  in the `pytest-slow-synthetic` selection and 8.55 s in the CI environment. The test runs
  in `pytest-slow-synthetic` and in the CI environment. Operator decision, confirmed by Jiahao on
  2026-10-18.

No threshold, tolerance, target range, preset or bank was changed by W-488.

**Accepted statements applied (W-488, 2026-10-18).** The fourteen entries of
{doc}`synthetic-cellular-scene`'s "Accepted statements this addition changes" are applied to
{doc}`synthetic-sample-scene`, with the completions of entry 32 of {doc}`synthetic-real-appearance`'s list; entries
1 to 34 of {doc}`synthetic-real-appearance`'s "Accepted statements this page changes" are
applied to {doc}`synthetic-cellular-scene`; and entries 1 to 38 of {doc}`synthetic-real-appearance-validation`'s list,
which take the place of entries 35 to 43 of {doc}`synthetic-real-appearance`, are applied to
{doc}`synthetic-cellular-scene-validation`. Each is applied with its proposed wording; the lists
stay on their pages as the record of what changed.
