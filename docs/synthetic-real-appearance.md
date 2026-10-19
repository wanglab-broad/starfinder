# Synthetic cellular scene with real appearance: nuclei and culture cells sampled from real images

Status: Accepted (W-515, 2026-10-16, at 71ef8a1)

This page specifies an appearance model for the cellular scene of
{doc}`synthetic-cellular-scene` (Accepted at W-396, implemented by W-483 to W-486). In that
scene every stain is a closed form. On the `slow` presets its nuclei are smooth, its
texture stays below the real variation within a nucleus, its backgrounds miss the real
ones and its shapes are too regular. On 2026-10-16 Jiahao judged the scenes too unlike
the real images and chose appearance sampled from real images. Real nuclei (in the
culture context, whole cells with their nucleus and both stains) are taken from held-in
images, augmented by rotation, flips, a small scale and a small gain, and placed by a
synthetic layout on a background assembled from real patches. The placed masks are the
truth. An operator prototype built such scenes for the three contexts, and Jiahao judged
its round 10 good on the same day.

The page builds on {doc}`synthetic-cellular-scene` and {doc}`synthetic-sample-scene` and
edits neither. Every statement of {doc}`synthetic-cellular-scene` and
{doc}`synthetic-cellular-scene-validation` that the change would alter is listed, with the
old and the proposed wording, in "Accepted statements this page changes". The validation
design (target definitions, generator checks, consequences for the chain checks) and the
draft implementation issues belong to the next issue, W-514. Where this page needs one of
them, it names the open point. The gate W-515 reviews both pages before any
implementation.

**Sources.**

* **Existing behaviour.** Statements cite the file and function at the batch's starting
  revision `eb333bc` as `<file>::<function>`, with files under `src/python/starfinder/`
  (for example `synthetic/_context.py::generate_context_scene`).
* **The prototype.** "The prototype" is the operator's record
  `runs/W-485/20261016T1714Z-operator-prototype/` under the artifact root
  `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/`:
  * round 10, the reference result, is `round10/scripts/proto.py`, with its logs
    `round10/logs/<context>.log.json` (keys `proto` for the synthetic scene and `real_crop`
    for the held-out crop) and `round10/logs/<context>.run.log` (its timestamps and
    `/usr/bin/time -v`);
  * the earlier rounds are `round10/scripts/proto.v<n>.py`, with their logs in
    `round10/logs/v<n>-<context>.log.json`; round 1's logs are `logs/v1-<context>.log.json`
    and round 2's are `logs/<context>.log.json` at the top of the folder;
  * `<context>` is `ln` (dense 3D), `tissue2d` or `culture`. A function of the prototype is
    cited as `proto.py::<function>`.
* **This issue's measurements.** They are in the W-513 run directory
  `runs/W-513/20261016T204608Z-f983442a/`, cited as "W-513" followed by a file under
  `results/` and its row, for example W-513 `tables/visible_share_ln.csv` row
  `ln/Position048/whole/DAPI-3D/nuclei-3D`. The definitions are in the docstring of its
  `scripts/measure.py`.
* **Other measurements.** W-393 is cited as {doc}`synthetic-cellular-scene` cites it
  ("Sources").

## What this scene is not

* It is **not a calibrated tissue model.** It is a collage of real nuclei and cells,
  augmented and placed by a synthetic rule. Their density, spacing and background are
  development choices, matched to measured targets. They are not fitted to tissue and they
  are not a D04 calibration (D6). The **parameters are development targets.**
* **Its masks come from method labels.** The placed masks, which are the truth, are the
  labels a method made on the source images: StarDist `3D_spleen` in LN, StarDist via the
  legacy pipeline (`PI_label.tif`) in tissue-2D, and Cellpose `cpsam_v2` extended through z
  in the culture. They are neither annotation nor ground truth of the real images.
  StarDist's labels are star-convex and sit inside the visible nucleus: in the LN FOVs,
  0.93 to 0.97 of the label voxels are visible stain, while 0.75 to 0.85 of the visible
  stain lies outside every label (W-513 `tables/visible_share_ln.csv`, rows
  `ln/<FOV>/whole/DAPI-3D/nuclei-3D`, columns `label_voxels_visible` and
  `visible_outside_labels`).
* It **establishes no segmentation accuracy.** The §2.9 methods run on it as an
  engineering check of the chain (D1), never as a measure of a method's quality. No
  accuracy is claimed, no methods are compared and nothing is trained (D7). A method that
  segments the scene well has segmented a collage of its own kind of labels; how W-514's
  chain checks read that is W-514's claim boundary (open point).
* It is **not a learned generative model.** Nothing is trained, no GPU is used and no
  dependency is added. NumPy, SciPy and scikit-image, which are locked, do the work.

## Settled decisions this page follows

* **Direction (Jiahao, 2026-10-16).** Appearance is sampled from real images: real nuclei,
  and in the culture context whole cells with their nucleus and both stains. They are
  augmented, placed by a synthetic layout, and the placed masks are the truth. The
  closed-form appearance of {doc}`synthetic-cellular-scene` is not tuned further. Round 10
  of the prototype is the reference result.
* **Augmentation (Jiahao: rotations and small scaling).** Rotation about Z by any angle,
  flips, a scale of 0.9 to 1.1 and a gain of 0.9 to 1.1.
* **Real data stays outside the repository (Jiahao, W-329).** No real-data record, bank
  file or array derived from real images is committed, not even a small text file. Which
  images and labels a bank comes from is configuration per dataset, not code. The routine
  tier and the CI environment read no real data.
* **Libraries over in-house code.** NumPy, SciPy and scikit-image functions are preferred
  to new in-house algorithms. There is no new dependency and no lock change.
* **Unchanged from W-396.** D1, D6 and D7 as above. The resource grant RG1 covers scenes
  up to 3 × 3 FOVs of 16 × 256 × 256, with one CPU, 16 GiB and 3,600 s per step.

## Choices for the gate

Each choice below has an options table with its cost and consequence, one recommendation,
and the prototype's choice named among the options. This list is the index.

| # | Choice | Recommendation | Prototype (round 10) | Section |
| --- | --- | --- | --- | --- |
| RA1 | How the real appearance is reached | E1, a new entry and configuration around an unchanged `ContextSceneConfig` | not applicable (a script) | "Opt-in" |
| RA2 | What an entry is | EN1, a nucleus (dense 3D, 2D tissue) or a cell with its nucleus and both stains (culture) | EN1 | "The appearance bank" |
| RA3 | Bank sources and held-out images | SRC2, the prototype's sources with the culture FOVs Position407 and Position408 added | SRC1 | "The appearance bank" |
| RA4 | Selection of entries | SEL1, the prototype's four rules with no cap | SEL1 with a cap per FOV | "The appearance bank" |
| RA5 | Brightness within a scene | BR1 with the guard of BR4, the plateau taken on the labels the entries come from | BR1 | "The appearance bank" |
| RA6 | Bank size and repetition | BS2, every kept entry, drawn without replacement within a scene | BS1 | "The appearance bank" |
| RA7 | Storage and build | ST1, one file per bank outside the repository, found by an environment variable and checked by its SHA-256 | not applicable | "The appearance bank" |
| RA8 | The bank without real data | CF1, entries rendered by the accepted closed-form generator | not applicable | "The bank without real data" |
| RA9 | What an entry carries outside its mask | OUT1, its own capped fall-off, the per-distance median on shared faces, a blur outside the mask | OUT1 | "What an entry carries" |
| RA10 | Augmentation ranges | AU1, as the prototype | AU1 | "Augmentation" |
| RA11 | Interpolation of masks and intensities | IN2, the mask from the entry's signed distance; linear intensities with an anti-aliasing prefilter | IN1 | "Augmentation" |
| RA12 | Layout | PL2, the prototype's placement, with an anchor rule | PL2 | "Layout" |
| RA13 | Density | DN1, the visible-stain share, with its target from the bank's own images | DN1, target from the held-out image | "Layout" |
| RA14 | Cell territories | CT1 in the dense contexts (nearest placed nucleus within a growth); the culture cells' own masks (CU1) | CU1; no territories in the dense contexts | "Layout" |
| RA15 | Out-of-focus light | H2, each placed object's own blurred signal, scaled once per scene | H1 | "Composition" |
| RA16 | Background | BG1, real patches, as a point function | BG1 | "Composition" |
| RA17 | Observation stages | OB1, none added to sampled appearance | OB1 | "Composition" |
| RA18 | Output | O8, scale relative signal to a plateau grey level, round and clip | O8 | "Composition" |
| RA19 | 2D tissue entries in a stack | TZ1, each 2D entry a column through the planes of the monolayer | not applicable (one plane) | "Layout" |
| RA20 | Truth schema | AT2, new files beside the context layout, the ellipsoid columns null | not applicable | "Truth" |
| RA21 | The morphology round | MO1, both rounds from the entry's own channels | not applicable | "The morphology round" |
| RA22 | Target ranges and the visible stain | TG2, visible-stain targets for density and background, the other W-393 ranges kept | TG1 (measured beside the real crop) | "Target ranges" |
| RA23 | Presets | PR1, new appearance presets beside the six accepted ones, which stay closed-form | not applicable | "Presets" |
| RA24 | The W-485 amendment on TR7 | AM1, it stands for the closed-form presets and does not apply to the appearance presets | not applicable | "Accepted statements this page changes" |

## Terms

* An **entry** is one object of a bank: arrays on its own grid (its **entry frame**, voxel
  centres at `q ⊙ s_e` µm, `s_e` the source spacing), with a mask, its signal per channel
  and the record of where it came from.
* A **bank** is a set of entries, with background patches, built from configured real
  images (or from the closed-form generator, "The bank without real data") and stored as one
  file with its SHA-256.
* A **source image** is a real image a bank is built from. Its **source labels** are the
  method labels on it. A **held-out image** is a real image no bank uses; it serves
  comparisons only.
* The **excess** of an entry is its signal above its **local source background**, the 20th
  percentile of the source stain over the unlabelled voxels of its box
  (`proto.py::extract_sprites`, `b`). Fluorescence adds, so a placed signal is an excess.
* The **rim level** is the median excess over the inner rim of the mask, the mask less its
  face-connected erosion.
* A **placed object** `k` is an entry `e_k` with a transform `T_k` from its entry frame to
  the sample frame. Placed objects are numbered in placement order.
* The **signed distance** of an entry with mask `M` on its grid of spacing `s_e` is

  ```
  D_e = EDT(not M) − EDT(M)        EDT(A) = scipy.ndimage.distance_transform_edt(A, sampling=s_e)
  ```

  in µm, where `EDT(A)` gives at each voxel of `A` the distance to the nearest voxel
  outside `A`, and 0 outside `A`. On a mask voxel `D_e = −EDT(M) < 0`, and on a voxel
  outside `D_e = EDT(not M) > 0`, so on the entry's own grid `D_e ≤ 0` is exactly the mask.
  Interpolated linearly, `D_e` crosses 0 between a mask voxel and its neighbour outside
  (midway for a face neighbour, where the two values are `−s` and `+s` for their step `s`).
  A placed object `k` **holds** `σ`
  where `D_{e_k}(T_k⁻¹ σ) ≤ 0`. Its distance in the sample frame is
  `δ_k(σ) = m_k · D_{e_k}(T_k⁻¹ σ)`, with `m_k` its scale in Y and X (rotations and flips
  keep distances). This is the one signed distance of the page, used by the mask (IN2), the
  bank (ST1), the anchor (PL2), the territories (CT1) and the truth.
* The **visible stain** of an image is the stain smoothed by a Gaussian of one voxel,
  above the Otsu threshold of that smoothed image (`proto.py::vis_share`;
  `skimage.filters.threshold_otsu`).
* The **plateau** of an image is the median over its labels of each label's median stain
  (W-393's definition; `proto.py::plateau_of`).
* The sample position `σ`, a FOV's voxel centres `true_f + p ⊙ s` and a point function are
  as in {doc}`synthetic-cellular-scene` ("Terms").

## Opt-in

A scene without the new settings is exactly the accepted cellular scene. The six
`context-*-v1` presets, `generate_context_scene`, `save_context_scene`,
`save_context_dataset`, `ContextSceneConfig` and `CONTEXT_PRESETS` keep their behaviour,
fields and values.

`ContextSceneConfig`'s payload is the `requested_config` of every saved `context.json`
(`synthetic/_context.py::_payload`, called by `generate_context_scene`). `_payload` writes
`sample`, `nuclei`, `packing`, `territories`, `types`, `molecules`, `stains` and
`morphology` by name, so a new field on `ContextSceneConfig` changes `context.json` only if
`_payload` is changed to write it. Options E2 and E3 each need such a change.

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **E1. A new entry and configuration (recommended)** | `generate_appearance_scene(codebook, *, config=AppearanceSceneConfig(...), on_fov=None)`; `AppearanceSceneConfig` holds an unchanged `ContextSceneConfig` and the new parts. | One more entry; shared private functions, as W-483 shared those of the sample scene. | The accepted presets' outputs stay unchanged by construction; the new truth lives in new files ("Truth"). |
| E2. A new optional field `appearance` on `ContextSceneConfig`, left out of the payload when None | As named. | A special case in `_payload`. | With `_payload` changed to write the field only when it is set, the accepted presets' `context.json` stays as it is, but `requested_config` holds the field for some configurations and not for others; every reader must know the case. |
| E3. A new field and a new context truth version | As named. | Least code. | With `_payload` changed to write the field for every configuration, every accepted preset's `context.json` gains a key and a new version, against the exclusions of this batch. |

**Recommendation: E1.**

| Concept | Name | Kind |
| --- | --- | --- |
| Entry | `generate_appearance_scene(codebook, *, config=AppearanceSceneConfig(...), on_fov=None)` | function |
| Result | `AppearanceScene` | dataclass |
| Configuration | `AppearanceSceneConfig`, `BankReference`, `AugmentationConfig`, `LayoutConfig`, `CompositionConfig` | frozen dataclasses |
| Bank | `build_appearance_bank(config, path)`, `closed_form_bank(preset, n)`, `load_appearance_bank(reference)` | functions |
| Persistence | `save_appearance_scene(scene, directory)`; the dataset by `save_context_dataset` of `scene.context` | function |
| Presets | `appearance_scene_preset(name)`, `APPEARANCE_PRESETS`, `APPEARANCE_PRESET_VERSION` (`appearance-presets-v1`) | function, dict, str |
| Version | `APPEARANCE_TRUTH_VERSION` (`"1"`) | str |

```python
@dataclass(frozen=True)
class AppearanceSceneConfig:
    context: ContextSceneConfig            # unchanged: acquisition, seeds, cell types, molecules, morphology round;
                                           #   its nuclei, packing and stains parts are not used (below)
    bank: BankReference                    # a stored bank by name and SHA-256, or a closed-form bank ("The bank
                                           #   without real data")
    augmentation: AugmentationConfig = AugmentationConfig()
    layout: LayoutConfig = LayoutConfig()  # placement, density, territories
    composition: CompositionConfig = CompositionConfig()   # out-of-focus light, background, observation, output

@dataclass
class AppearanceScene:
    context: ContextScene                  # the context scene's tables and layout; its stains are the composed ones
    placements: pd.DataFrame               # "Truth"
    record: dict                           # versions, bank SHA-256, configuration, reached density, streams
```

**What stays of W-483 to W-486.** The acquisition layout, `Ω`, `K` and the FOV jitter
(`synthetic/_sample.py::_layout`, `generate_sample_scene`) stay. So do the cell types and
molecules, placed by rejection with the label functions of this page
(`synthetic/_context.py::_Context.molecules`), the morphology round's translation and
dataset layout (`synthetic/_context.py::_stain_part`, `save_context_dataset`), the cast
(`synthetic/_sample.py::_cast`) and the stream scheme (`synthetic/_sample.py::_streams`).
Under PL2 the packing (`synthetic/_context.py::_Context._place`), the nucleus shapes
(`synthetic/_cellular.py::_shapes`), the closed-form stains
(`synthetic/_stains.py::_Model.signal`), the bright regions, the unevenness and the stain
background (`synthetic/_observation.py::_prepare_background`) are not used by a real
bank. A closed-form bank uses them to render its entries.

## The appearance bank

**What an entry is (RA2).**

| Option | Entry | Cost | Consequence |
| --- | --- | --- | --- |
| **EN1. A nucleus; in the culture a cell with its nucleus and both stains (recommended; prototype)** | Dense 3D: one StarDist nucleus with its DAPI. 2D tissue: one nucleus with its PI. Culture: one Cellpose cell (extended through z) with its largest nucleus label inside it, and its DAPI and Flamingo of the organelle round (`proto.py::extract_sprites`, called with `nucleus_labels`). | One kind of entry per context. | The culture keeps the real relation between its nucleus and cytoplasm stains; the dense contexts need territories from a rule (RA14). |
| EN2. A nucleus with its expanded territory in every context | Entries carry a cell mask in the dense contexts too (LN's W-382 expanded territories; tissue-2D's `labeled_cells.tif`). | Larger entries; these cell masks are themselves a rule (expansion, legacy watershed). | Cells of an unreal shape cut out of a packed tissue; their neighbours' stain inside the box has to be removed. |

**Recommendation: EN1.**

**Bank sources and held-out images (RA3).** The inputs are those W-393 inventoried (W-392
`inventory/`, by the paths of W-393 `scripts/common.py`): four LN FOVs, four culture FOVs,
and three crops of 2048 × 2048 pixels of the tissue-2D sample images. W-513 checked the
SHA-256 of all 34 files against W-392's inventory, and all are equal (W-513
`inventory/inputs.csv`, column `equal_to_w392`). The sources of a bank are configuration
per dataset ("Storage and the build step").

| Option | Sources (held-out image) | Cost | Consequence |
| --- | --- | --- | --- |
| SRC1. The prototype's (prototype) | LN Position049, Position057, Position058 (Position048); tissue-2D crops `crop_y20480_x2048`, `crop_y20480_x4096` (`crop_y18432_x0`); culture Position398 (Position397) (`round10/logs/<context>.log.json`, keys `held_out`, `bank`). | None. | A culture bank of 27 cells (below). |
| **SRC2. SRC1 with the culture FOVs Position407 and Position408 added (recommended)** | As SRC1; the culture bank takes Position398, Position407 and Position408. | Two more FOVs read by the build. | A culture bank of 91 cells. The two FOVs were set aside because W-393 measured plateaus of 13 and 19 against 79 and 83. W-513 shows that this comes from their 2D nucleus labels, which cover 0.47 and 0.33 of the Z projection while the visible DAPI covers 0.14 and 0.13. On the labels the entries are cut from (extended through z) the four plateaus are 46, 48, 46 and 45, and the median DAPI over the visible stain of the Z projection is 79, 80, 87 and 82 (W-513 `tables/bank_selection_culture.csv` columns `plateau`, `plateau_DAPI_zmax_2D_labels`; `tables/visible_share_culture.csv` rows `culture/<FOV>/whole/DAPI-2D/nuclei-2D`, columns `visible_share`, `label_share`, `median_raw_visible`). |
| SRC3. SRC2 and further crops of the same tissue-2D images, or the culture's fused sample images with their legacy labels | More crops of `PI.tif` and `PI_label.tif` (inventoried whole); `fused/DAPI.tif`, `Flamingo.tif`, `DAPI_label.tif`, `Cell_label.tif` of the culture (inventoried by W-392, not read by W-393). | A measurement of each new source before use; the legacy culture labels are another method's. | More entries in the two small banks; not measured by this issue. |
| SRC4. No held-out image | Every image is a source. | None. | Comparisons are made on images the bank was cut from, so a match says less. |

**Recommendation: SRC2**, the held-out images as in the prototype, never a source of any
bank, for comparisons only. SRC3 is the route if the 2D tissue bank proves too small
(RA6).

**Selection of entries (RA4).** The prototype keeps a label when it passes four rules in
order (`proto.py::extract_sprites`):

1. it does not touch the image border in Y or X, and in LN not a Z end either (the
   prototype tests every axis of a 3D LN label);
2. its size lies between the 5th and 95th percentile of the sizes of the labels that pass
   rule 1;
3. in the culture, the cell holds a nucleus label of at least 50 voxels;
4. it is not squashed: at most 25 % of its rim lies against a neighbour (35 % for culture
   cells). The rim share is the share of the label's rim voxels inside the face-connected
   dilation of the other labels in its box.

The prototype kept at most 350 entries per LN FOV and 400 otherwise, in a random order, and
counted rule 4 only until it reached that cap (`round10/logs/ln.run.log`: "sprites kept
350 left out as squashed ... 43", 20 and 29). W-513 applies the four rules to every label
(W-513 `tables/bank_selection_<context>.csv`, one row per image):

| Image | Labels | Left out by rule 1 (Z end alone) | by rule 2 | by rule 3 | by rule 4 | Kept | Plateau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LN Position048 (held out) | 3,015 | 995 (956) | 202 | — | 120 | 1,698 | 35 |
| LN Position049 | 2,475 | 1,018 (993) | 146 | — | 110 | 1,201 | 45 |
| LN Position057 | 2,405 | 856 (838) | 156 | — | 102 | 1,291 | 39 |
| LN Position058 | 1,207 | 565 (558) | 66 | — | 35 | 541 | 50 |
| tissue-2D `crop_y18432_x0` (held out) | 349 | 27 | 34 | — | 114 | 174 | 64 |
| tissue-2D `crop_y20480_x2048` | 291 | 23 | 28 | — | 69 | 171 | 80 |
| tissue-2D `crop_y20480_x4096` | 242 | 34 | 22 | — | 53 | 133 | 92.5 |
| culture Position397 (held out) | 59 | 20 | 4 | 6 | 0 | 29 | 46 |
| culture Position398 | 54 | 23 | 4 | 0 | 0 | 27 | 48 |
| culture Position407 | 56 | 17 | 4 | 0 | 0 | 35 | 46 |
| culture Position408 | 57 | 22 | 4 | 2 | 0 | 29 | 45 |

The columns are `labels`, `left_out_border` (`left_out_border_z_alone`),
`left_out_size_after_border`, `left_out_nucleus_after_size`, `left_out_squashed_after_rest`,
`kept` and `plateau`. The culture plateau is the DAPI's on the labels extended through z.
The tissue-2D counts of the two bank crops equal the prototype's (171 and 133 kept; 69 and
53 squashed; `round10/logs/tissue2d.run.log`). In LN the rim share of the candidates after
rules 1 and 2 has a median of 0 to 0.008, and 6 to 8 % lie above 0.25 (columns
`rim_share_median`, `share_rim_above_0.25`). In the tissue-2D crops the median is 0.15 to
0.20, and 28 to 40 % lie above 0.25.

| Option | Rules | Cost | Consequence |
| --- | --- | --- | --- |
| **SEL1. The prototype's four rules, no cap (recommended; the prototype with a cap)** | As above, on every label. | The build processes every label (LN about 3,000 per bank; W-513's step `measure_ln` measured the rules on all four FOVs in 69 s). | 3,033 LN entries instead of the prototype's 1,050. Shapes are more regular than real (limit L2), because squashed labels are left out. |
| SEL2. SEL1 without the Z-end rule in LN | Keep labels cut by a Z end of the 50-plane stack. | None. | About 2,800 more LN entries, each a nucleus with a flat face where the stack ended, placed anywhere in depth. |
| SEL3. SEL1 keeping squashed labels, their shared faces given the synthetic fall-off | Rule 4 off. | None. | The shapes of packed nuclei come back (the held-out share not described by one ellipsoid is 0.047 against 0.009 in round 10, `round10/logs/ln.log.json`, `real_crop` and `proto` `non_ellipsoid_share`), along with flattened faces whose neighbour is gone. |

**Recommendation: SEL1**, with the squash threshold as the control of limit L2.

**Brightness within a scene (RA5).** One scene mixes entries of several images. The
prototype scales each entry and the background patches by the target plateau over its
source image's plateau. The target is the held-out image's plateau: 37 in LN and 59 in
tissue-2D. The culture is not scaled (`proto.py`, the lines that multiply by
`P_T / plat[...]`). Within an image, the kept entries' medians over its plateau have p5–p95
0.64–1.31 in LN, 0.54–1.46 in tissue-2D and 0.65–1.73 in the culture (nucleus DAPI)
(W-513 `tables/bank_selection_<context>.csv` columns `kept_median_over_plateau_p5`,
`_p95`): the spread between nuclei is kept.

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **BR1. Scale each image's entries and patches by the target over its plateau (recommended; prototype)** | The build stores excess and patches relative to the source image's plateau; the scene multiplies by its plateau grey level `g` and the entry's gain. | One number per source image. | The real spread between nuclei stays; images of different exposure blend. |
| BR2. No scaling | Raw grey levels. | None. | Steps in brightness between nuclei of different images (LN plateaus 39 to 50). |
| BR3. Each entry scaled to a common median | Per entry. | None. | The real spread between nuclei (TR6) is lost. |
| BR4. BR1, with a guard on the plateaus | A source image whose plateau lies outside a factor of 1.5 of the bank's median, or whose plateau and median over the visible stain disagree by more than that factor, is refused by the build with a message. | Two numbers per image. | Catches a source scaled by more than 1.5, where its 8-bit steps would show, and a plateau taken on wrong labels. With the 2D culture labels, Position407 would be refused (13 against 87 over the visible stain). |

**Recommendation: BR1 with BR4's guard**, the plateau taken on the labels the entries are
cut from. The culture's then lie within 45 to 48, and LN's within 39 to 50.

**Bank size and repetition (RA6).** Repetition shows when one entry appears more than
once in a scene. With a bank of `N` entries and `n` placements drawn with replacement, the
expected share of placements that reuse an entry is `1 − N (1 − (1 − 1/N)^n) / n`.
Without replacement it is `max(0, n − N) / n`. At the prototype's reached density a
`slow` preset holds about 738 dense 3D nuclei, 337 2D tissue nuclei and 32 culture cells
(W-513 `tables/resource_estimates.csv`, column `objects_at_prototype_density`):

| Context | Bank | Entries | Reused share with replacement | Without replacement |
| --- | --- | --- | --- | --- |
| Dense 3D | prototype's (capped) | 1,050 | 0.282 | 0 |
| Dense 3D | SEL1 on the same FOVs | 3,033 | 0.112 | 0 |
| 2D tissue | prototype's | 304 | 0.395 | 0.099 |
| Culture | prototype's | 27 | 0.406 | 0.148 |
| Culture | SRC2 | 91 | 0.152 | 0 |

These are rows `<context>/context-<context>-slow-v1/<bank>` of W-513
`tables/bank_repetition.csv`, with `<context>` `ln`, `tissue2d` or `culture`.

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| BS1. A capped bank drawn with replacement (prototype) | 350 or 400 entries per image; one uniform index per proposal (`proto.py::place_all`). | None. | 28 to 41 % of the placements of a `slow` scene reuse an entry, under a new rotation, flip, scale and gain. |
| **BS2. Every kept entry, drawn without replacement within a scene (recommended)** | A permutation of the bank per scene; an entry rejected by placement goes back to the end of the order; reuse only after the bank is exhausted. The build reports the bank size against the largest preset's expected placements. | One permutation per scene. | No reuse in dense 3D and culture `slow` scenes; about 10 % reuse in 2D tissue, where SRC3 adds entries. |
| BS3. A floor on the bank size | The build refuses a bank smaller than twice the largest preset's expected placements. | None. | The 2D tissue and SRC1 culture banks would be refused until SRC3. |

**Recommendation: BS2.**

**Storage and the build step (RA7).**

| Option | Where | Cost | Consequence |
| --- | --- | --- | --- |
| **ST1. One file per bank outside the repository, found by an environment variable (recommended)** | `<STARFINDER_APPEARANCE_BANKS>/<bank name>.npz` (NumPy's archive, locked), master copy under the artifact root, with its build record. The preset names the bank and its SHA-256; the scene refuses a file whose SHA-256 differs. The build configuration (sources, held-out images, rules, patch sizes) is a file kept with the bank, outside the repository. | One build step per bank, on the lab machine. | No real-derived array or record enters the repository. A test that needs a real bank skips with a reason where the variable or the file is missing, as `src/python/test/test_segmentation_learned.py` skips without its model. |
| ST2. The configuration in the repository's `config/`, the file outside | As ST1, with the source paths in the repository. | None. | The repository names lab paths and FOVs of real data: a record of real data, against W-329. |
| ST3. No stored bank; the scene reads the real images | The generator cuts entries at each run. | Every scene reads three FOVs (the prototype's bank stage, 96.8 s in LN, `round10/logs/ln.run.log`). | Scenes depend on the network share; slower; the bank is not one identified object. |

**Recommendation: ST1.** The build is one named step,
`build_appearance_bank(config, path)`, run once per bank in its own process with
`/usr/bin/time -v`, reading only the configured inputs. It writes the bank file and its
record: the inputs with their SHA-256, the rules and their counts as in the table above,
the plateaus, and a digest of the arrays. A rebuild is compared by that digest, because a
zip archive's bytes carry a time. The bank holds per entry: the mask's signed distance
`D_e` of "Terms" (float32, µm, negative inside); the excess per channel relative to the
source plateau (float32); for culture entries the signed distance `D_e` of its nucleus
mask; the box origin and spacing; the source image ID, label and plateau; the local
background; the rim share; and the volume. It also holds the background patches per
source image (relative to the plateau).
An LN entry box with its margins is about 21 × 43 × 43 voxels, so a bank of 3,033 LN
entries with two float32 arrays each holds about 0.9 GB before compression (an estimate).

**What a bank holds per stain, and the prototype's stages.** The prototype built its bank
inside the scene's process (`proto.py`, the loops over `bank_fovs` and `extract_sprites`).
In round 10 that stage took 96.8 s in LN, 43.5 s in tissue-2D and 35.8 s in the culture,
within steps of peak RSS 3,987,588, 1,970,664 and 3,442,840 KiB (W-513
`tables/prototype_steps.csv`, columns `bank_stage_s`, `peak_rss_kib`, from
`round10/logs/<context>.run.log`).

## The bank without real data

The CI environment and the golden tests read no real data. A test that needs a bank uses
a **closed-form bank**: the build step run on scenes of today's closed-form generator
instead of real images.

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **CF1. Entries rendered by the accepted closed-form generator (recommended)** | `closed_form_bank(preset, n)` generates an accepted `context-*-v1` preset (`synthetic/_context.py::generate_context_scene`) with noise off, and runs the build step on its stain and truth labels as if they were a source image, in memory, with no file. Its record names the preset and seed instead of the SHA-256 of inputs. | One small accepted scene per bank; the build code is exercised on it. | The routine tier, the CI environment and the golden tests read no real data, and every code path of the bank, placement, composition and truth is tested on it. Its appearance is the closed form's, so it checks the machinery, not the realism. |
| CF2. A bank of hand-made arrays | Ellipsoids with a constant excess, written in the test. | Least. | The build step is not exercised. |
| CF3. A small real bank committed as a fixture | As named. | None. | Against W-329: no array derived from real images enters the repository. |

**Amendment of 2026-10-17 (W-516; operator decision, confirmed by Jiahao on 2026-10-18).** CF1
as written (noise off) conflicts with BR4. With the accepted unevenness field U1 kept, the
routine culture preset `context-culture-v1` has plateaus of 31, 4 and 53 grey levels at the
seeds 42, 43 and 44: its field scales a whole routine scene. BR4 refuses a source whose
plateau lies outside a factor 1.5 of the bank's median (31), so two of the three sources
are refused and no closed-form culture bank can be built. The closed-form bank's sources
are therefore rendered with **noise and unevenness off**; BR4 stays as it is. The
closed-form bank exercises the bank's machinery in the routine tier; the real banks keep
the real images' own variation.

**Recommendation: CF1.** **No real-data record, bank file or array derived from real
images enters the repository.** The real banks live outside it (ST1), only `slow` tests on
the lab machine read them, and they skip with a stated reason elsewhere.

## What an entry carries

**Inside its mask** an entry carries its excess with the source intensities unchanged
(`proto.py::extract_sprites`, `np.where(selfm, X, ...)`): `X = max(I − b, 0)`, with `b` the
local source background.

**Outside its mask (RA9)** it carries only its own fall-off. The prototype's round 10
defines it, with `d` the distance to the mask in µm and `r` the rim level:

1. the surroundings, smoothed by a Gaussian of 0.3 µm, capped by a linear fall-off from
   the rim level to 0 over 0.6 µm: `kept = min(G_0.3 * X, r · max(0, 1 − d / 0.6))`;
2. kept only on voxels nearer to the entry than to any other source label: a weight `v`,
   the indicator `d ≤ d_other` smoothed by a Gaussian of 0.6 µm, set to 0 on the other
   labels;
3. on faces it shared with a source neighbour (`v` small), the per-distance median of the
   capped values on its own side (`kept` where `v > 0.5`, in bins of 0.1 µm, made
   non-increasing), interpolated over `d`; this is `own`;
4. outside the mask the result `kept · v + own · (1 − v)` is blurred by a Gaussian of 0.25
   µm, so that the rim does not follow the source label's pixel staircase. Inside the mask
   nothing is blurred.

The alternatives the prototype tried and dropped, with their results (median edge width
10–90 %, visible share of the scene, background over plateau, from the logs named; LN
unless stated):

| Round | Outside the mask | Result | Why it was dropped |
| --- | --- | --- | --- |
| 2 | A **flat fill of the source neighbours**: the other labels in the box set to the median of the free voxels (`proto.v2.py::extract_sprites`), the surroundings kept up to 1.2 µm (round 1 had given the neighbours zero weight, over 0.8 µm) | Round 2: edge 0.864 µm, visible 0.589, background 0.118 (`logs/ln.log.json`) | The fill leaves a flat plateau where a neighbour was; round 3 replaced it by the capped fall-off. |
| 3 to 9 | A **1.2 µm halo**: the capped fall-off over 1.2 µm (`RAMP_UM = 1.2`, `proto.v3.py` to `proto.v9.py`) | Round 9: edge 1.115 µm, visible 0.628, background 0.242 (`round10/logs/v9-ln.log.json`) | The halos of packed nuclei add up into hazy gaps; round 10 halves it to 0.6 µm. |
| 8 | A **fall-off measured on free faces**: the entry's own median profile over distance, measured only where no other label lies within 1 µm, used everywhere, the surroundings capped at 1.5 times it (`proto.v8.py::extract_sprites`) | Edge 1.740 µm, 1,020 nuclei placed against 1,291 to 1,305 in rounds 7, 9 and 10, visible 0.638 (`round10/logs/v8-ln.log.json`) | Wider edges and fewer placed nuclei; round 9 takes the profile on its own side of the shared faces instead. |
| 10 (kept) | The rule above, over 0.6 µm | Edge 0.824 µm, visible 0.508, background 0.212 (`round10/logs/ln.log.json`, `proto`) | Judged good by Jiahao (2026-10-16). Edges narrower than the held-out crop's 1.536 µm (limit L3). |

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **OUT1. Round 10's rule (recommended; prototype)** | As above, with the fall-off length `λ` = 0.6 µm and the blur 0.25 µm as configuration. | Two distance transforms per entry at build time (`scipy.ndimage.distance_transform_edt`). | Neighbours and unlabelled fragments are dropped, not replaced by a flat value. `λ` is the control of edges (L3) against haze (L1): round 9 at 1.2 µm gave wider edges and more haze. |
| OUT2. Nothing outside the mask | The excess is 0 outside. | None. | A hard cut at the label, whose edges are the method label's, not the stain's. |
| OUT3. The flat fill (rounds 1 and 2) | As above. | None. | Flat plateaus where neighbours were. |

**Recommendation: OUT1.**

## Augmentation

**Ranges (RA10).**

| Option | Rotation | Flips | Scale | Gain | Cost | Consequence |
| --- | --- | --- | --- | --- | --- | --- |
| **AU1. As the prototype (recommended; prototype)** | About Z, uniform in `[0, 2π)` | Y with probability ½; Z with probability ½ in dense 3D; none in X, which a rotation by π with a Y flip covers | Uniform 0.9–1.1, isotropic in dense 3D and 2D tissue, in Y and X only for culture cells, which keep their Z position | Uniform 0.9–1.1 | None. | Jiahao's ranges. A flip in Y followed by any rotation gives every mirror image in the plane. |
| AU2. Wider scale and gain (0.8–1.25) | As AU1 | As AU1 | 0.8–1.25 | 0.8–1.25 | None. | More variety from small banks, at the price of sizes and brightness outside the sources' p5–p95 (rule 2, TR6). |
| AU3. Rotations about every axis in 3D | Any rotation | As AU1 | As AU1 | As AU1 | Resampling across the anisotropic Z step (0.346 against 0.194 µm). | Nuclei tilted out of the acquired plane, blurred by the coarser axis; the point-spread function's elongation in Z would turn with them, which a real nucleus does not show. |

**Recommendation: AU1.** In the culture, "keeps its Z position" means that the entry's Z
is not scaled or flipped. The whole bank is shifted by one constant, so that the median
nucleus centroid of its source image lands on the scene stack's middle plane. The
prototype used a shift of 6 planes for Position398 in its 30-plane canvas
(`round10/logs/culture.run.log`, "Z shift 6"; `proto.py`, `Z_SHIFT`).
The culture is therefore not a monolayer in the accepted sense, where every centre lies at
`z_mid` ({doc}`synthetic-cellular-scene`, "Monolayer"). Its cells keep their own Z
extents, and PL2 tests their overlap in 3D, so two cells whose projected masks overlap can
stand at different depths. The 2D tissue, placed in one plane as columns (TZ1), keeps
the monolayer's property that a column holds one nucleus at most.

**The transform.** Placed object `k` with centre `c_k` (µm, sample frame) maps an
entry-frame point `x` (µm) to `σ = c_k + A_k (x − x̄_e)`, with `x̄_e` the entry box's centre
and `A_k = S_k R_k F_k`. Here `F_k = diag(f_z, f_y, 1)` (each ±1), `R_k` is the rotation by
`θ_k` in the Y–X plane, and `S_k = diag(s_z, s, s)` (`s_z = s` in dense 3D, 1 for culture
and 2D tissue). The inverse is `x = x̄_e + A_k⁻¹ (σ − c_k)`. The prototype applies the same
map in voxels with `scipy.ndimage.affine_transform` (`proto.py::transform`).

**Interpolation and anisotropic voxels (RA11).** The rotation is about Z only, so it
never mixes the Z step with the lateral step: Y and X are isotropic in every source. Z is
only flipped and, in dense 3D, scaled. An entry is evaluated at `x / s_e` in its own voxel
index.

| Option | Mask | Intensities | Cost | Consequence |
| --- | --- | --- | --- | --- |
| IN1. Linear on both (prototype) | Linear interpolation of the 0/1 mask, labelled where above 0.5 | Linear (`order=1`) | None. | One-pixel specks and notches where a rotated staircase crosses 0.5 (limit L6). |
| **IN2. The mask from its signed distance; intensities linear with a prefilter (recommended)** | The entry's signed distance `D_e = EDT(not M) − EDT(M)` of "Terms" (µm, negative inside), interpolated linearly, the object holding `σ` where it is at most 0 | Linear, after a Gaussian prefilter of `σ_a = max(0, (t_a / (m s_e,a)) − 1) / 2` source voxels on each axis `a` where the target spacing `t_a` is coarser than the scaled source spacing (`m` the scale); this is `skimage.transform.resize`'s anti-aliasing rule | One distance transform per entry at build time; a prefilter per entry and target spacing (cached per scene). | Smooth masks without staircase specks. The 2D tissue presets at 0.18927 and 0.37854 µm and the culture presets at 0.388796 and 0.777592 µm (2 to 4 times the source pixel) are rendered without aliasing. |
| IN3. Cubic intensities | Order 3 | — | As IN2. | Overshoot and negative values at sharp edges. |

**Recommendation: IN2.** The prefilter depends only on the entry, its scale and the target
spacing, never on a FOV's offset, so it keeps every object a point function.

## Layout

**How entries meet the accepted layout (RA12).** The accepted layout of W-483 places
sites by the shape-aware contact distance K1 and by random sequential addition with
contact proposals P1. It clips nuclei to their power cells, gives cells territories by
T1 to T3, and assigns types and molecules by cell
(`synthetic/_context.py::_Context._place`, `_Context.cellular`, `_Context.molecules`).

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| PL1. Entries fitted to the accepted sites, clipped to their power cells | K1 and P1 place sites as today; each site takes the entry closest in size and elongation, rotated to the site's angle and scaled to its semi-axes; its mask and signal are clipped to the site's power cell. | The accepted packing kept; a matching step; scales well outside 0.9–1.1 to fit. | Real texture cut by flat power faces, a visible straight edge through a real nucleus. The density follows the accepted laws, set from method labels (TR2), so the scene would be about as sparse as the StarDist labels, not the visible stain (RA13). |
| **PL2. The prototype's placement (recommended; prototype)** | Per proposal: draw an entry and its transform; with probability `contact` (0.75), if an object is placed, choose a placed object `i` uniformly and a direction `u` (uniform on the sphere in dense 3D, on the circle in a plane, a monolayer or for culture cells) and try `c = c_i + f (r_i + r_k) u` for `f` = 0.5, 0.54, … below 1.4, with `r` the equivalent radius of each mask in µm; the first `f` at which the mask overlaps placed masks by at most 1 % of its volume, evaluated on the sample grid, is taken. Otherwise propose `c` uniformly in `Ω` widened by 0.3 of the box. A proposal is refused when less than 35 % of its mask lies in the domain, when its overlap exceeds 1 %, when a culture cell's nucleus touches a placed nucleus, or (new) when its anchor lies in a placed mask. A new object never relabels or paints over a placed one: where masks overlap, the earlier one owns the voxel (`proto.py::place_all`). | Proposals capped (40,000 in the prototype, with 4,000 failures in a row); each proposal slices its box on the sample grid. | Touching, packed nuclei at the visible stain's density; whole real masks, never clipped by a flat face. The accepted packing is not used. |
| PL3. Accepted sites as centres, PL2's ownership | K1 and P1 propose centres; each takes an entry and is accepted by PL2's overlap rule. | Both rules. | Two packing rules whose outcomes interact; nothing gained over PL2, because K1's contact distance is defined on ellipsoids. |

**Recommendation: PL2**, with one addition. A placed object's **anchor** is the
transformed position of its entry's deepest point, where `D_e` is smallest (most negative).
A proposal whose anchor lies in a placed mask is refused, so every anchor lies in its own
label: later objects never paint over earlier ones. The sample membership rule needs this
({doc}`synthetic-sample-scene`, "Sample membership").

**Kinds of cells.** Every placed object is one ordinary cell with one nucleus. The
appearance scenes hold no binucleate and no empty cells, and `kind_shares` is not used. A
culture cell's nucleus is its largest nucleus label inside the cell
(`proto.py::extract_sprites`, `nid`). Another nucleus label inside a source cell is not
labelled, while its DAPI stays in the entry's signal.

**Density (RA13).** The prototype stops placing when the visible-stain share of the
composed image reaches the held-out image's visible share, with a fixed threshold, the
held-out image's Otsu threshold (`proto.py::place_all`, `vis_fn`). It checks every 15
placements (30 in LN, 1 in the culture). It also stops at 40,000 proposals or after 4,000
refused proposals in a row (`max_attempts`, `max_fail`). In round 10 dense 3D stopped at
the cap of 40,000 proposals, with 0.508 against 0.628. 2D tissue stopped after 4,000
refusals in a row, at 37,804 proposals, with 0.566 against 0.667. The culture stopped at
its target, 0.271 against 0.263, after 125 proposals (W-513 `tables/prototype_steps.csv`,
columns `proposals`, `visible_share`, `target_visible_share`; `round10/logs/<context>.log.json`,
`attempts`).

The method labels miss nuclei, which is why a label-based density fails. On the held-out
LN centre crop (50 × 512 × 512) the StarDist labels cover 0.163 of the volume, while the
visible stain covers 0.628 (W-513 `tables/visible_share_ln.csv` row
`ln/Position048/centre512/DAPI-3D/nuclei-3D`; the prototype's 0.16 and 0.63). In round 10
the truth labels covered 0.46 of the volume and the visible stain 0.51 (`proto`,
`volume_fraction`; `visible`). In tissue-2D the labels and the visible stain agree: the
label shares of the whole crops are 0.29 to 0.45 and their visible shares 0.29 to 0.44
(W-513 `tables/visible_share_tissue2d.csv`, rows `tissue2d/<crop>/whole/PI-2D/nuclei-2D`).
In the culture the 2D nucleus labels cover more than the visible DAPI, 0.17 against 0.12
in Position397 (`tables/visible_share_culture.csv` row
`culture/Position397/whole/DAPI-2D/nuclei-2D`).

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **DN1. The visible-stain share, its target from the bank's own images (recommended; prototype with the held-out target)** | Place until the composed stain's visible share, with a fixed threshold relative to the plateau, reaches the target; the target and the threshold are the medians over the bank's source images on the region matching the preset's Z extent (LN: the middle 16 planes, visible 0.681, 0.539, 0.472 for Position049, 057, 058, W-513 `tables/visible_share_ln.csv` rows `ln/<FOV>/slab16_z17/DAPI-3D/nuclei-3D`). Placement stops as "When placement stops" states. | A rendering of the stain on the sample grid every `m` placements. | Density matched on what the eye and the methods see, measured the same way on real and synthetic images; the held-out image stays out of the scene. |
| DN2. The label volume fraction (TR2) | Place until the truth labels reach W-393's fraction (0.13 in LN). | None beyond counting. | A scene as sparse as StarDist's labels: about a quarter of the visible density in LN. |
| DN3. A fixed count per preset | Place `n` objects, `n` set once from DN1's reached count. | None. | Cheapest and reproducible; the density follows whatever the bank's sizes give. Suited to the routine fixtures on the closed-form bank. |

**Recommendation: DN1** for the `slow` presets, and DN3 for the routine presets, whose
closed-form bank has no visible-stain target.

**When placement stops.** Objects are placed one at a time, and each proposal draws its
own entry and transform; no seed is fixed in advance. Placement ends at the first of:

* the target reached (DN1) or the count placed (DN3);
* `max_proposals` proposals in total (40,000 in the prototype);
* `max_refusals` refused proposals in a row (4,000 in the prototype; `proto.py::place_all`,
  `max_attempts` and `max_fail`).

The reached visible share or count, the placed objects and the proposals made are
recorded in `appearance.json`. Below `floor` (0.9) of the target or the count, a
`ValueError` names both. This replaces F2 of {doc}`synthetic-cellular-scene`, which
stops at the first seed that finds no accepted proposal after `max_proposals`.

**Cell territories (RA14).** The prototype has no cells in the dense contexts. Its
culture cells are the entries' own cell masks.

| Option | Territory of placed nucleus `k` | Cost | Consequence |
| --- | --- | --- | --- |
| **CT1. The nearest placed nucleus within a growth (recommended for dense 3D and 2D tissue)** | Where a nucleus label holds `σ`, `C(σ) = N(σ)`: the cell of the nucleus that owns the voxel by placement order ("Truth"). Elsewhere `C(σ) = k` where `k` minimizes `δ_k(σ)` ("Terms"; positive there, the distance to the placed mask), ties to the smaller index, and `δ_k(σ) ≤ g_k`, with `g_k` the accepted growth law (`synthetic/_context.py::TerritoryConfig`, `growth`: 0.78 µm dense 3D, 0.5 µm 2D tissue); else 0. | One more interpolation per object near a point. | The cells of `skimage.segmentation.expand_labels`, the LN route's rule (W-382's expansion by 0.7776 µm); a point function. A nucleus lies inside its own cell by construction (below). |
| CT2. Power cells of the placed objects | The accepted T2 with sites at the anchors and the equivalent radii. | As T2. | Flat faces cut through the gaps; the nucleus can cross its power face, breaking "a nucleus lies inside its own cell". |
| CT3. No territories | Cells equal nuclei. | None. | The nuclear fraction of reads is 1; the assignment's expansion is not checked. |
| **CU1. The culture cells' own masks (recommended; prototype)** | The placed cell entry's mask is the cell; its nucleus mask, cut to the cell, the nucleus. | None. | Real cell shapes with their real Flamingo. |
| CU2. Culture cells by CT1 from the nuclei | Cells grown from placed nuclei. | As CT1. | Loses the real cytoplasm stain's shape. |

**One ownership rule for nuclei and cells.** Placement allows masks to overlap by up to
1 % of a mask's volume (PL2), and the earlier object owns the overlap. Inside such an
overlap the signed distances can rank the two objects the other way, so `C` must not be
decided there by distance. Take two unit spheres (radius 1 µm, scale 1) with centres 1.9
µm apart, placed in that order. Their overlap is a lens of volume
`π (4 + 1.9) (2 − 1.9)² / 12 = 0.0154` µm³, which is 0.37 % of a sphere's 4.19 µm³, so PL2
permits it. At the point 0.98 µm from the first centre on the line of centres,
`δ_1 = 0.98 − 1 = −0.02` µm and `δ_2 = 0.92 − 1 = −0.08` µm. Both spheres hold the point,
and `N` names the first, the earlier one. A rule that took the smallest `δ` everywhere
would set `C` to the second, so `N = 1` would not imply `C = 1`. Under CT1 as written
above the point lies in a nucleus label, so `C = N = 1`, and distances decide only where
no nucleus label holds `σ`. Hence `N(σ) = n > 0` implies `C(σ) = cell(n)` at every point,
overlaps included. In the culture (CU1) the same order rules both labels: `C(σ)` is the
first placed cell whose cell mask holds `σ`, and `N(σ)` is that cell's nucleus where the
nucleus's own `D_e ≤ 0` holds, else 0. So a nucleus voxel always belongs to its own cell,
as the prototype labels it (`proto.py::place_all`: a cell's nucleus is written only on the
voxels the cell itself takes).

**Recommendation: CT1 for dense 3D and 2D tissue, CU1 for the culture.** Cell types,
molecules per cell, the nuclear fraction and ambient molecules are the accepted ones
(`synthetic/_context.py::_Context.molecules`). They are placed by rejection with `C` and
`N` of this page. A per-type size scale (`synthetic/_context.py::CellTypeConfig`,
`size_scale`) has no effect, since sizes come from the entries and the scale of AU1.

**2D tissue entries in a stack (RA19).** The tissue-2D entries are 2D, and the 2D tissue
presets are monolayers of 8 or 16 planes projected by the route after fusion.

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **TZ1. A column through the planes (recommended)** | The entry's mask and signal are the same in every plane of the monolayer's stack; placement runs in one plane. | None; placement in 2D (W-513 `tables/resource_estimates.csv` column `r_v_one_plane`). | The Z maximum equals the 2D composition, so the projection truth `P` of PT1 is the placed 2D masks exactly. Planes away from the middle are as bright as the middle, which no route reads. |
| TZ2. An axial profile from the accepted laws | Signal times the accepted soft edge in Z with an axial semi-axis drawn per object; mask where inside. | One law per object. | A more plausible stack; the Z maximum differs from the entry where the profile dims it. |

**Recommendation: TZ1.**

## Composition

Each stain of an appearance preset is one channel of the bank's entries: DAPI in dense
3D and the culture, PI in 2D tissue, and Flamingo in the culture. It keeps the name, round
and file pattern of the accepted preset's `StainChannelConfig`
(`synthetic/_stains.py::StainChannelConfig`). Its geometry source, brightness, edge,
texture, unevenness, background and noise fields are not used.

The image of a stain at `σ` is the background, plus the out-of-focus light, plus the
placed signal. Only the owner's signal counts inside a mask:

```
I(σ) = B(σ) + κ · H(σ) + S(σ)
S(σ) = E_{O(σ)}(σ)                    where some mask holds σ (O(σ) the owner, first in placement order)
S(σ) = Σ_k E_k(σ)                     elsewhere (each object's fall-off outside its mask)
E_k(σ) = g_k · ε_{e_k}(T_k⁻¹ σ)       the entry's excess, prefiltered and interpolated (IN2), times its gain
```

The owner `O(σ)` is the first placed object that holds `σ`: `N(σ)` in the dense contexts,
the cell `C(σ)` in the culture, whose entries are cells ("One ownership rule for nuclei and
cells").

This is the prototype's composition (`proto.py::place_all`). Inside a placed mask it sets
the owner's excess. Outside, it adds a new object's halo only where no mask lies. The
point form above equals it where masks do not overlap; where they do (at most 1 % of a
mask), the owner's excess is taken.

**Out-of-focus light (RA15).** The prototype adds a Gaussian of the placed signal with
`σ` of 2 µm in Z and 1 µm in Y and X. It scales the Gaussian by `κ` so that the background
plus that light, averaged over the rims of the placed masks, equals the mean local source
background of the placed objects, the level each excess was taken over
(`proto.py::place_all`, `images`; `HAZE_UM`). Round 10 found `κ` 0.68 in LN, 0.13 in
tissue-2D and 0 in the culture, whose local background is 0 (`round10/logs/<context>.log.json`,
`trace`, `haze_kappa`). The local background at the label rims is, over the plateau, a
median of 0.33 to 0.40 in the LN FOVs (p95 0.69 to 0.82), 0.125 to 0.130 in the tissue-2D
crops, and 0 in every culture FOV (W-513 `tables/rim_background_<context>.csv`, rows
`<context>/<image>/<stain>`, columns `local_bg_over_plateau_median`, `_p95`). This is the
level the light is matched to. The background W-393 measured farther than 2 µm from every
label (0.34 in Position048) contains the unlabelled nuclei.

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| H1. Blur the composed signal on a grid (prototype) | `H = G * S` on the canvas. | One Gaussian filter of the scene. | Not a point function: a FOV at a fractional offset would resample the blurred grid. |
| **H2. Each object's own blurred signal (recommended)** | `H(σ) = Σ_k g_k · (G_k * ε_{e_k})(T_k⁻¹ σ)`, where `G_k` has the deviations `(2 / s_z, 1 / s, 1 / s)` µm in the entry frame; the convolution is precomputed per entry on its box grown by three deviations. A Gaussian isotropic in Y and X commutes with the rotation and the flips, and the scale becomes this narrower kernel, so `H` equals the blur of `Σ_k E_k` exactly. `κ` is computed once per scene on the sample grid, as the prototype computes it, and recorded. | A blurred copy per entry and scale (the scale has a few values per bank if quantized, or is computed per object). | A point function; equal to H1 wherever the placed signals do not overlap. |
| H3. No out-of-focus light | `κ = 0`. | None. | The gaps of dense 3D drop to the background patches: round 5, before the light, had visible 0.343 and background 0.125 (`round10/logs/v5-ln.log.json`). |

**Recommendation: H2.**

**Background (RA16).** The prototype assembles the background from real patches. LN uses
patches of 24 pixels and 10 planes, overlapping by 12 pixels and 2 planes. The culture
uses 32 pixels and all planes, overlapping by 10. Patches are cut where the source has
little foreground (at most 10 % of the patch within 1 µm of a label or above 0.7 of the
plateau). Their holes are filled by the patch's local mean plus residuals drawn from its
own background, and their levels are equalized per depth. They are quilted with linear
ramps over the overlaps (`proto.py::bg_bank`, `fill_holes`, `quilt`).

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **BG1. Real patches, as a point function (recommended; prototype on a grid)** | The bank stores the patches; a scene draws one patch per cell of a lattice fixed in the sample frame (period: patch size less overlap, in µm), and `B(σ) = Σ_p w_p(σ) P_p(σ − o_p) / Σ_p w_p(σ)` with the prototype's ramp weights and linear interpolation of each patch. | Patches stored per source image; a few patches per voxel. | The real background texture, the same at a sample position in every FOV. |
| BG2. The accepted `BackgroundConfig` | Baseline, gradient and regions (`synthetic/_observation.py::evaluate_background_translated`). | None. | Smooth closed-form backgrounds; what the closed-form bank uses (CF1). |
| BG3. A flat level | The mean local background. | None. | No texture in the gaps. |

**Recommendation: BG1 for real banks, BG2 for the closed-form bank.**

**Observation stages (RA17).** The real crops carry their own noise and their 8-bit
rounding. The accepted stages add baseline, Poisson and read noise per FOV
(`synthetic/_observation.py::_observe`).

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **OB1. None added (recommended; prototype)** | The composed image is cast; no stage of `_observe`. | None. | The noise is the sources' (round 10's noise over plateau per block 0.032 against 0.028 on the held-out crop, `round10/logs/ln.log.json`, `noise_block_rel_med`). It is fixed in the sample frame, so two overlapping FOVs agree at a sample position up to interpolation, unlike real overlaps (limit L7). |
| OB2. A small read noise per FOV | The accepted `independent` stage only, at a fraction of the measured noise. | One draw per voxel. | Overlaps become independent; the total noise rises above the real one. |
| OB3. The accepted stages unchanged | As the closed-form scene. | As today. | Noise counted twice. |

**Recommendation: OB1**, with OB2 the control of limit L7.

**Output (RA18).** Entries are relative to their source plateau (BR1). A scene multiplies
the composed float64 image by its plateau grey level `g`. It then rounds half to even and
clips to `[0, 255]`, as `synthetic/_sample.py::_cast` does: the 8-bit output.

| Option | `g` | Cost | Consequence |
| --- | --- | --- | --- |
| **O8. The median plateau of the bank's source images (recommended)** | LN 45 (45, 39, 50), tissue-2D 86 (80, 92.5), culture 46 (48, 46, 45 on the extended labels) (W-513 `tables/bank_selection_<context>.csv`, `plateau`) | None. | Scale factors near 1 for every source, so the sources' 8-bit steps stay as they were. |
| O8-H. The held-out image's plateau (prototype) | 37 LN, 59 tissue-2D (`round10/logs/<context>.run.log`, "held-out ... plateau") | None. | Ties the scene to the comparison image. |
| O8-A. The accepted presets' `g` | 42, 78, 20 | None. | The culture's 20 came from W-393's 2D-label plateaus, which W-513 shows are low in two FOVs. |

**Recommendation: O8.**

## Truth

**The placed masks are the labels.** In the dense contexts, nucleus label `N(σ) = k` for
the first placed object `k` that holds `σ` ("Terms"), else 0, and the cell label is `N(σ)`
where `N(σ) > 0` and CT1's distance rule elsewhere. In the culture, the cell label is the
first placed cell whose mask holds `σ`, and the nucleus label is that cell's nucleus where
its nucleus mask holds `σ`, else 0. Both follow one placement order ("One ownership rule
for nuclei and cells").

**Each placed object is a function of the sample position.** Object `k`'s mask, signal,
nucleus mask (culture), out-of-focus light and distance are its entry's arrays evaluated
at `T_k⁻¹ σ`. The entry arrays, `T_k` and `g_k` are fixed once the scene's draws are made.
The interpolation is linear on the entry's grid, after the prefilter of IN2 for the
target spacing. A mask holds `σ` where the interpolated `D_{e_k}(T_k⁻¹ σ)` ("Terms") is
at most 0. A FOV evaluates every object at its voxel centres `true_f + p ⊙ s`, at any fractional
offset in Z, Y and X, without resampling a rendered image. The morphology round evaluates
the same functions at `true_f + τ_f + q ⊙ s` ("The morphology round"). With one plane
(`Z=1`) every object is evaluated at z = 0 like any point: a 2D entry is its plane and a 3D
entry its section through the plane; placement runs in the plane, `τ_z = 0`, and the labels
are `plane` label images. The background (BG1)
and the out-of-focus light (H2) are point functions too. The only parts that are not are
the placement test, which is decided once on the sample grid, and `κ`, a scalar computed
once on the sample grid. Both are recorded.

**Truth schema (RA20).**

| Option | Where the new truth goes | Cost | Consequence |
| --- | --- | --- | --- |
| AT1. New files; the accepted shape columns filled with the moment ellipsoid of each placed mask | `nuclei.csv` keeps its columns with fitted values. | A fit per object. | Columns that read as the shape but are not: recomputing the labels from them gives other labels (identity 2 of {doc}`synthetic-cellular-scene` would fail). |
| **AT2. New files beside the context layout; the ellipsoid columns null (recommended)** | `save_appearance_scene` writes the `save_context_scene` layout with `nuclei.csv` and `cells.csv` keeping their columns, centre and anchor from the placement, the shape columns (`axial_radius`, `lateral_radius`, `elongation`, `angle`) null and `site_radius` the radius of the placed mask's bounding sphere; `lobes.csv`, `harmonics.csv`, `stain_objects.csv`, `bright_regions.csv` and `unevenness.csv` empty (the gains are in `placements.csv`); plus the files below. | Readers of the shape columns must accept nulls. | The labels are rebuilt from `placements.csv` and the bank only, and nothing pretends to be an ellipsoid. |
| AT3. A new sample truth version | `SAMPLE_TRUTH_VERSION` `"2"` for these scenes. | One version. | Two meanings of `sample.json` in one layout. |

**Recommendation: AT2**, with `APPEARANCE_TRUTH_VERSION` `"1"`:

```
<directory>/
  <the save_context_scene layout>     # nuclei.csv and cells.csv as above; context.json unchanged in meaning
  appearance.json                     # appearance_truth_version "1", appearance_preset_version, the requested and
                                      #   effective AppearanceSceneConfig, the bank's name, SHA-256, build digest and
                                      #   build record path, the reached density and placed objects, the
                                      #   proposals made, κ per stain, the stream scheme, the SHA-256 of every file
  placements.csv                      # object_id, kind (nucleus or cell), entry_id, source_image, source_label,
                                      #   source_plateau, centre_z/y/x (µm), anchor_z/y/x (µm), theta (rad), flip_z,
                                      #   flip_y, scale_z, scale_yx, gain, proposal, overlap_share, volume (µm³ or µm²
                                      #   on the sample grid)
  background_patches.csv              # stain, lattice_z/y/x, patch_id (bank), source_image
  fovs/<fov_id>/stains/<name>.ome.tif # as the context layout
```

**Identities a test can check.** A "boundary voxel" is a voxel whose centre has an
interpolated `D_{e_k}(T_k⁻¹ σ)` within 1e-9 µm of 0, or a tie between two objects'
distances `δ_k` under CT1. A test that recomputes a label reports the boundary voxels where
it differs and expects 0.

1. **Opt-in.** The six accepted context presets and the two sample presets give, after
   the change, the files the golden test and identity O1 of
   {doc}`synthetic-cellular-scene` pin.
2. **Labels.** Per-FOV and sample labels equal `N` and `C` recomputed from
   `placements.csv` and the bank at the voxel centres, at any fractional offset and at
   `Z=1`, except at boundary voxels.
3. **Ownership and anchors.** On the sample grid, each object's mask overlaps the masks of
   earlier objects by at most 1 % of its volume. Every anchor's label is its own object.
   `N = n > 0` implies `C = cell(n)` at every voxel, overlaps included: CT1 sets `C = N` on
   every nucleus voxel, and CU1 writes a nucleus only on its own cell's voxels ("One
   ownership rule for nuclei and cells").
4. **Composition.** With the output cast off and float64, each stain image equals `I(σ)`
   recomputed from `placements.csv`, the bank, `background_patches.csv` and `κ` at the
   voxel centres, within a rounding tolerance W-514 states (open point). Rendered again at
   `true_f + k ⊙ s` for an integer vector `k`, it equals the first image shifted by `k` on
   the common voxels (the no-resampling identity).
5. **Output.** The 8-bit image equals `clip(rint(g · J), 0, 255)` of the float64 image `J`
   of identity 4.
6. **Morphology round.** With a supplied whole-voxel translation, the morphology DAPI
   equals the reference DAPI shifted by it, as identity 8 of
   {doc}`synthetic-cellular-scene`.
7. **Bank.** The bank file's SHA-256 equals the recorded one; each `entry_id` exists in it;
   each transform parameter lies in its range (AU1); with BS2, no entry is placed twice
   while the bank is not exhausted.
8. **Molecules.** Identities 4 to 6 of {doc}`synthetic-cellular-scene` (types and genes,
   counts, marker answers) hold with `C` and `N` of this page.
9. **Projection truth.** On any grid of a 2D tissue scene, no column of fixed `(y, x)`
   holds two nucleus IDs (TZ1), and `P(y, x)` is the Z maximum of that grid's nucleus
   labels, as identity 11 of {doc}`synthetic-cellular-scene` states. In the culture, whose
   cells keep their own depths, the columns holding two nucleus IDs are counted and
   reported. The culture's route compares the FOVs' 3D labels, not `P`.

The comparisons, tolerances and which tier runs each are W-514's (open point).

## The morphology round

Culture entries carry the DAPI and Flamingo of the organelle round, cut from the same
source cell, so the two channels are registered with each other.

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **MO1. Both rounds from the entry's own channels (recommended)** | The reference round's DAPI stain file and the morphology round's DAPI and Flamingo are the entry's channels, evaluated at `true_f + p ⊙ s` and at `true_f + τ_f + q ⊙ s` with the accepted MT1 translation (`synthetic/_context.py::_stain_part`). | None. | MR1's registration on the shared DAPI applies as accepted, and identity 6 holds. |
| MO2. The reference DAPI from the closed form | As named. | None. | Two different DAPI images of one sample position: the registration has no truth. |

**Recommendation: MO1.** The translation, its truth table `morphology_truth.csv` and the
dataset layout of `save_context_dataset` are unchanged.

## Target ranges

**Which W-393 ranges stay (RA22).** The ranges are TR1 to TR12 of
{doc}`synthetic-cellular-scene-validation`. Those measured on the truth labels assume
that real labels and truth labels mean the same. Under this page the truth labels are
method labels at the visible stain's density, so a label-based quantity of a real image
undercounts what is there. That makes TR1 (touching) and TR2 (volume fraction) of dense 3D
incomparable.

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| TG1. Keep TR1 to TR12, compare with the held-out crop beside them (prototype) | As round 10 measured (`proto` against `real_crop`). | None. | TR1, TR2 and TR8 of dense 3D fail by construction (touching 0.99 against 0.44–0.57 and labels 0.46 against 0.07–0.16, since the truth is placed at the visible stain's density; background against unlabelled nuclei), and TR3 misses by limit L2 (0.009 against 0.064–0.074). |
| **TG2. Visible-stain targets for density and background; the rest kept (recommended)** | TR2 is replaced in every context by the visible-stain share per image, and TR8's background by the median stain outside the visible stain, over the plateau; both measured the same way on real and synthetic images (the definitions of W-513 `scripts/measure.py`). TR4 to TR7 and TR9 to TR12 stay as written. TR1 is reported on the truth labels and, as like for like, compared on a method's labels of the scene against its labels of the real image in the chain checks (W-514). TR3 is measured on the truth labels; a miss caused by leaving out squashed sources (limit L2) is reported with the squash threshold, its control, and W-514 states whether it is asserted. | One measure added to the generator check. | Density and background targets that do not depend on what a method missed. |
| TG3. Targets from the held-out images only | Each target the held-out image's value. | None. | One image per context; no range. |

**Recommendation: TG2.** The visible-stain measurements per real image are W-513's:

| Context | Image (region) | Visible share | Label share | Median outside the visible stain over the plateau | W-513 row |
| --- | --- | --- | --- | --- | --- |
| Dense 3D | Position048 (whole; held out) | 0.590 | 0.161 | 0.086 | `ln/Position048/whole/DAPI-3D/nuclei-3D` |
| Dense 3D | Position049 (whole) | 0.535 | 0.130 | 0 | `ln/Position049/whole/DAPI-3D/nuclei-3D` |
| Dense 3D | Position057 (whole) | 0.570 | 0.124 | 0.051 | `ln/Position057/whole/DAPI-3D/nuclei-3D` |
| Dense 3D | Position058 (whole) | 0.371 | 0.060 | 0 | `ln/Position058/whole/DAPI-3D/nuclei-3D` |
| Dense 3D | Position048 (centre 50 × 512 × 512) | 0.628 | 0.163 | 0.162 | `ln/Position048/centre512/DAPI-3D/nuclei-3D` |
| 2D tissue | `crop_y18432_x0` (held out) | 0.443 | 0.450 | 0.047 | `tissue2d/crop_y18432_x0/whole/PI-2D/nuclei-2D` |
| 2D tissue | `crop_y20480_x2048` | 0.361 | 0.365 | 0.038 | `tissue2d/crop_y20480_x2048/whole/PI-2D/nuclei-2D` |
| 2D tissue | `crop_y20480_x4096` | 0.288 | 0.294 | 0.022 | `tissue2d/crop_y20480_x4096/whole/PI-2D/nuclei-2D` |
| Culture | Position397, DAPI Z max (held out) | 0.118 | 0.171 | 0.038 | `culture/Position397/whole/DAPI-2D/nuclei-2D` |
| Culture | Position398, DAPI Z max | 0.139 | 0.183 | 0.036 | `culture/Position398/whole/DAPI-2D/nuclei-2D` |
| Culture | Position397, Flamingo Z max (centre 1024; held out) | 0.263 | 0.328 (cells) | — | `culture/Position397/centre1024/Flamingo-2D/cells-2D` |

The rows are in W-513 `tables/visible_share_<context>.csv`, columns `visible_share`,
`label_share` and `not_visible_over_plateau`. The last row reproduces the prototype's
culture target (`round10/logs/culture.run.log`, "visible share 0.2634..., cell label
share 0.3276..."). LN's median outside the visible stain is 0 where the tile holds empty
regions. Whether TR8 then needs a block-wise background (as W-393's per-block row) is an
open point of W-514. The middle 16 planes give 0.47 to 0.68 in the three LN bank FOVs (rows
`ln/<FOV>/slab16_z17/DAPI-3D/nuclei-3D`). The ranges, their thresholds and the checks
that assert them are W-514's.

## Presets

**Presets (RA23).**

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **PR1. New appearance presets beside the accepted six (recommended)** | `APPEARANCE_PRESETS` (version `appearance-presets-v1`): `appearance-dense3d-v1`, `appearance-tissue2d-v1` and `appearance-culture-v1` (routine, closed-form bank of their context, DN3); `appearance-dense3d-slow-v1`, `appearance-tissue2d-slow-v1` and `appearance-culture-slow-v1` (`slow`, the real banks `ln-sample1-v1`, `tissue2d-v1`, `culture3d-v1` of SRC2, DN1). Each takes the grid, spacing, jitter, overlap, seed, cell types and molecules of the accepted preset of its context and tier. | Six presets and three bank builds. | The accepted six stay closed-form, as the routine fixtures and the golden reference, and keep the known limit on TR7. |
| PR2. Replace the accepted `slow` presets | As named. | Changes accepted presets. | Against this batch's exclusions; the closed-form `slow` presets are lost as a reference. |
| PR3. The acquired pixel for 2D tissue and culture | Grids at 0.094635 and 0.194398 µm. | None. | Fewer objects per scene, the reason Jiahao coarsened them on 2026-10-15. |

**Recommendation: PR1**, on the accepted grids.

**Amendment of 2026-10-17 (W-519; operator decision, confirmed by Jiahao on 2026-10-18).** The
culture's whole cell masks saturate below DN1's target. Its target, the median over the
bank's sources of the Flamingo visible share on the middle 16 planes, is 0.501 (0.291,
0.505 and 0.501 for Position398, 407 and 408). With at most 1 % overlap per mask, placement
reached 0.480 after 65 cells, and every later cell took hundreds of proposals. With the
prototype's stop of 4,000 refused proposals in a row, and with the mask test made faster
without changing any outcome, one generation of `appearance-culture-slow-v1` placed 68
cells in 10,184 proposals and reached 0.499. It took 27 min 41 s on one CPU, with a peak
RSS of 13,844,268 KiB, and stopped by its refusals. Because a generation of more than
15 minutes is too long for the checks that repeat it, `appearance-culture-slow-v1` stops after
500 refused proposals in a row. Every other rule and its target stay unchanged. It then
places 65 cells in 2,419 proposals, reaches 0.480 (above its floor, 0.9 · 0.501 = 0.451)
and records that share without raising, in 11 min 19 s with a peak RSS of 13,424,852 KiB.
The other presets keep 4,000.

**Resources.** These are estimates, not measurements. The basis is the prototype's
measured steps (one CPU, `/usr/bin/time -v`, `round10/logs/<context>.run.log`): LN
669.5 s and 3,987,588 KiB; tissue-2D 228.5 s and 1,970,664 KiB; culture 235.8 s and
3,442,840 KiB. The prototype's placement is not optimized: it stopped at its cap of 40,000
proposals in dense 3D (W-513 `tables/prototype_steps.csv`). Each preset's objects are the
prototype's reached density times the preset's `|Ω|`, quoted on
{doc}`synthetic-cellular-scene` ("Presets"). The placement time is the prototype's
placement stage times the object ratio `r_n`, times `max(1, r_v)` with `r_v` the ratio of
the voxels each density check evaluates. The bank build is the prototype's bank stage
(once per bank; a scene loads the stored bank). The peak RSS is bounded by the
prototype's step, which also held three source FOVs. W-513's measurement steps read and
processed whole FOVs, as a build does, within 69 s and 2,127,600 KiB in LN, 40 s and
1,954,536 KiB in tissue-2D, and 90 s and 2,870,856 KiB in the culture (W-513
`tables/resources.csv`, the final attempts of `measure_ln`, `measure_tissue2d` and
`measure_culture`).

| Preset | Objects (estimate) | Placement estimate | Whole scene, upper estimate | Peak RSS bound | Within RG1 |
| --- | --- | --- | --- | --- | --- |
| `appearance-dense3d-slow-v1` | 738 | 298 s | 421 s | 3.99 GB | yes |
| `appearance-tissue2d-slow-v1` | 337 | 294 s (placed in one plane, TZ1) to 1,888 s (checks on the 16-plane stack) | 1,933 s | 1.97 GB | yes |
| `appearance-culture-slow-v1` | 32 | 330 s | 367 s | 3.44 GB | yes |
| `appearance-dense3d-v1` | 24 | 10 s (upper) | not estimated for the closed-form bank | as above | yes |
| `appearance-tissue2d-v1` | 45 | 39 s (upper) | not estimated | as above | yes |
| `appearance-culture-v1` | 5 | 51 s (upper) | not estimated | as above | yes |

The columns are W-513 `tables/resource_estimates.csv` rows `context-<context>-...`:
`objects_at_prototype_density`, `placement_estimate_s_low` / `_high`,
`scene_estimate_s_high`, `peak_rss_kib_bound` and `within_rg1`. They are rounded. The
routine estimates scale the prototype's per-proposal cost, which its whole-scene density
checks dominate (1.6 s per proposal in the culture, checked after every placement on 31
million voxels). They are therefore upper bounds, and they do not show that a routine
generation stays below the routine tier's 5 s. The implementation measures every preset
with `/usr/bin/time -v`. A generator test of 5 s or more is `slow`
({doc}`contributing`, "Test markers"), and DN3 with a small closed-form bank is the
control.

## Random streams

Every new draw uses the accepted descriptor (`synthetic/_sample.py::_streams`) with new
component names, all prefixed `context.appearance.`:

| Component | Entity | Draws |
| --- | --- | --- |
| `context.appearance.entry` | `bank` | one permutation of the bank (BS2) |
| `context.appearance.augment` | `object-<k>-proposal-<j>` | `θ`, the flips, the scale, the gain |
| `context.appearance.placement` | `object-<k>-proposal-<j>` | the contact uniform; for a contact proposal the partner index and the direction; else z, y, x |
| `context.appearance.background` | `patch-<z>-<y>-<x>` (lattice cell) | one patch index; one per lattice cell, shared by the stains of a culture scene, as the prototype shares them |
| `context.appearance.growth` | `cell-<k>` | one value of CT1's growth |

**An existing draw does not change.** The accepted context generator's code path is
unchanged (E1), and its golden pins and identity O1 hold. Every new component starts with
`context.appearance.`, which no accepted component does (`synthetic/_context.py`,
`_CONTEXT_COMPONENTS`; `synthetic/_sample.py`, `_SAMPLE_COMPONENTS`). A new descriptor
therefore differs from every existing one in its component field. An appearance scene
draws no `context.nucleus.*`, `context.cell.kind`, `context.cell.contact`,
`context.stain.*` or `context.texture.*` stream. Its molecule and type draws keep the
accepted components and entities, with `cell-<k>` the `k`-th placed cell. Each proposal
has its own entity, so changing a law changes the draws of no other proposal. What they
produce can change, since placement is sequential.

## Known limits

Each is either a control or accepted as a limit:

| # | Limit | Measured | Control or limit |
| --- | --- | --- | --- |
| L1 | The gaps of dense 3D are hazier than real and hold fewer distinct dim nuclei | Round 10's background over plateau 0.212 against the held-out crop's 0.432, which contains unlabelled dim nuclei (`round10/logs/ln.log.json`, `background_rel`) | **Control:** the fall-off length `λ` and `κ`. **Accepted** for the dim nuclei: StarDist did not label them, so no entry holds one. TG2's background measure shows it. |
| L2 | Shapes are more regular once squashed sources are dropped | Share not described by one ellipsoid 0.009 against 0.047 on the held-out crop (`proto`, `real_crop`, `non_ellipsoid_share`) | **Control:** the squash threshold of rule 4 (SEL3 at its end). **Accepted** at 0.25 for the first delivery. |
| L3 | Edges are narrower | Dense 3D 0.82 µm against 1.54 µm (`proto`, `real_crop`, `edge_width_p5_med_p95`) | **Control:** `λ` (1.2 µm gave 1.115 µm in round 9) against L1. **Accepted** at 0.6 µm, within TR4's 0.20–2.20. |
| L4 | The masks come from method labels: StarDist's are star-convex and sit inside the visible nucleus | 0.93 to 0.97 of LN label voxels visible; 0.75 to 0.85 of visible stain outside labels (W-513, "What this scene is not") | **Accepted**; the claim boundary of W-514 says how truth from a method's labels is read. |
| L5 | A culture bank from one FOV holds 27 cells | W-513 `tables/bank_selection_culture.csv` | **Control:** SRC2 (91 cells) and BS2. |
| L6 | Occasional one-pixel specks and notches | Seen in round 10's figures (`round10/figures/zoom_v10.png`) | **Control:** IN2 removes the specks of mask interpolation. The notches where an earlier mask owns a voxel are **accepted** (at most 1 % per mask). |
| L7 | Noise is fixed in the sample frame, so overlapping FOVs agree | By construction (OB1) | **Control:** OB2. **Accepted** in the first delivery. |
| L8 | The 2D tissue stack is a column (TZ1) | By construction | **Accepted:** no route reads planes other than through the projection. |

## What stays excluded

* **A learned generative model, the GPU and new dependencies**: Jiahao chose sampled
  appearance, and the prototype needed only NumPy, SciPy and scikit-image.
* **Real data in the repository** (Jiahao, W-329): no bank, record, configuration of
  sources or array derived from real images.
* **A bank as a deliverable of this batch**: banks are built by the implementation.
* **Rotations out of the acquired plane** (AU3): they need resampling across the
  anisotropic Z step.
* **A membrane stain with real appearance**: no inventoried image holds one, and the
  boundary source stays closed-form and unused by presets (O2 of
  {doc}`synthetic-cellular-scene`).
* **An ER stain** (D2), **an optical model, raw acquisition and depth attenuation**: as on
  {doc}`synthetic-cellular-scene`.
* **Calibration, accuracy claims, comparisons of methods, training and annotation** (D6,
  D7; W-93 keeps synthetic truth and calibration, W-123 annotation).
* **Any change to an accepted page or preset**: the changes are listed below, for the
  gate.

## Accepted statements this page changes

Each statement of {doc}`synthetic-cellular-scene` and of
{doc}`synthetic-cellular-scene-validation` that the change would alter, with the old and
the proposed wording, in the order of those pages. Neither page is edited here; an
implementation issue changes each statement as written and nothing else. The other
statements of the two pages stay true, because the accepted scene and presets are
unchanged (E1, PR1). They describe the closed-form appearance, which the new scene sits
beside.

**{doc}`synthetic-cellular-scene`**

1. **"What this addition is not", first item.**
   *Old:* "It is **not a calibrated tissue model.** Nuclei are unions of ellipsoids clipped
   to power cells, cells are power-diagram territories, molecules are placed by compartment
   with a gene abundance per cell type, and stains are closed-form functions."
   *Proposed:* the same, followed by "With the real appearance of
   {doc}`synthetic-real-appearance`, nuclei and culture cells are real method labels with
   their stain, augmented and placed by a synthetic rule; that scene is not a calibrated
   tissue model either."
2. **"What this addition is not", third item.**
   *Old:* "Everything is rendered with NumPy and SciPy, which are locked."
   *Proposed:* "Everything is rendered with NumPy and SciPy, which are locked; the real
   appearance of {doc}`synthetic-real-appearance` also uses scikit-image, which is locked."
3. **"What this addition is not", last item.**
   *Old:* "It has no structured tissue background, no ER stain (D2), ..."
   *Proposed:* "It has no structured tissue background (the real appearance of
   {doc}`synthetic-real-appearance` adds one from real patches), no ER stain (D2), ..."
4. **"Settled decisions this page follows", D2.**
   *Old:* "One stain interface by geometry source (nucleus interiors, territory interiors,
   a band at territory boundaries)."
   *Proposed:* the same, followed by "In the appearance presets of
   {doc}`synthetic-real-appearance` the stains are channels of the bank's entries, not
   geometry sources; the culture's cytoplasm stain and morphology round stay."
5. **"Settled decisions this page follows", D4.**
   *Old:* "Controls: packing density and edge width; a texture inside the nucleus as a few
   bright regions per nucleus (Jiahao, 2026-10-09) and a smooth multiplicative unevenness;
   the existing background controls; a nucleus as a union of ellipsoids, a
   spherical-harmonic perturbation as an optional second form."
   *Proposed:* the same, followed by "On 2026-10-16 Jiahao chose appearance sampled from
   real images for the texture, the background and the shapes
   ({doc}`synthetic-real-appearance`); the bright regions, the unevenness and the union of
   ellipsoids remain the closed-form presets' controls."
6. **"Every new object is a function of the sample position": the rows "Nucleus label
   `N` ...", "Cell label `C` ...", "Nuclear stain signal ...", "Bright regions",
   "Unevenness", "Cytoplasm and membrane signals", "Background components of each stain",
   "Morphology round image", "8-bit output" and "Projection truth of the 2D tissue
   presets", and the sentence after the table.**
   *Old:* the point functions `N(σ)`, `C(σ)`, `S_nuc(σ)`, `T_n(σ)`, `U(σ)`, `S_ter(σ)` and
   `S_bnd(σ)`, the `BackgroundConfig` records, the morphology image from "the stain
   functions above", the 8-bit output "per voxel, after the noise", `P(y, x) = max_q N(z_q,
   y, x)`; and "The noise is the one thing that is not a point function: it is drawn per
   FOV from the FOV's streams, as today, and is independent between overlapping FOVs".
   *Proposed:* the same, followed by "In a scene of {doc}`synthetic-real-appearance` the
   labels are the placed masks under one placement order, and every stain (the culture's
   Flamingo included) is the composition of placed entries, background patches and
   out-of-focus light, each a point function of `σ`. The morphology round evaluates them at
   its shifted points, and the 8-bit output scales by `g` with no noise before the cast.
   `P` holds for its 2D tissue and is reported for its culture. No noise is drawn there;
   what is not a point function is the placement test and `κ`, each decided once on the
   sample grid ('Truth' there)."
7. **"Nucleus shape", the recommendation of C2.**
   *Old:* "**Recommendation: N1 first; N2 as an optional second form**"
   *Proposed:* the same, followed by "for the closed-form presets; in the appearance
   presets of {doc}`synthetic-real-appearance` a nucleus is a placed real mask."
8. **"Nucleus shape", "Label rule".**
   *Old:* "The nucleus label keeps today's rule (`_label_functions`): at `σ`, take the site
   `j*` of smallest power distance (ties to the smaller site index); `N(σ) = n` when `j*`
   is a site of nucleus `n`, `σ ∈ Ω`, `σ` passes the cell's territory test ("Cell
   territories"), and `ρ_n(σ) ≤ 1`; else 0."
   *Proposed:* the same, followed by "In the appearance presets of
   {doc}`synthetic-real-appearance`, `N(σ)` is the first placed object that holds `σ` (its
   signed distance at most 0), else 0; in the culture, the nucleus of the first placed cell
   that holds `σ`, where that nucleus's mask holds it."
9. **"Nucleus shape", "Bounding site".**
   *Old:* "Nucleus `n`'s site sphere has centre `c_n` and radius
   `R_n = max_l (‖o_l‖ + max semi-axis of lobe l)`, which contains every lobe ..."
   *Proposed:* the same, followed by "In the appearance presets of
   {doc}`synthetic-real-appearance`, `site_radius` is the radius of the placed mask's
   bounding sphere about its centre, and the shape columns are null (AT2)."
10. **"Nucleus shape", "Nuclei of different cells do not overlap".**
    *Old:* "A point carries the nucleus of the site of smallest power distance only, so two
    nuclei can never share a point ... Where two shapes reach across the face between their
    power cells, each is cut by that flat face, and the two nuclei touch on it."
    *Proposed:* the same, followed by "In the appearance presets of
    {doc}`synthetic-real-appearance` a point carries the nucleus of the first placed object
    that holds it, so two nuclei never share a point either; masks may overlap by up to 1 %
    of a mask, the earlier owning the overlap, and touching nuclei meet on their real
    outlines, not on a flat face."
11. **"Packing", the recommendations of C3 and C4.**
    *Old:* "**Recommendation: K1**" and "**Recommendation: P1.**"
    *Proposed:* the same, followed by "In the appearance presets, placement is the rule
    PL2 of {doc}`synthetic-real-appearance`: masks overlap by at most 1 % of their volume,
    and the earlier object owns the overlap."
12. **"Packing", the binucleate seed.**
    *Old:* "A binucleate seed places its two nuclei at `c ± (D_12 / 2) û_φ`, ..."
    *Proposed:* the same, followed by "The appearance presets of
    {doc}`synthetic-real-appearance` have no binucleate and no empty cells: every placed
    object is one ordinary cell with one nucleus ('Kinds of cells' there)."
13. **"Packing", "Monolayer".**
    *Old:* "With `monolayer`, every centre (nuclei, lobes' parent centres and empty cells)
    has `z = z_mid`, ... so **no two nuclei or territories overlap in a Z projection**"
    *Proposed:* the same, followed by "In the appearance presets of
    {doc}`synthetic-real-appearance` the 2D tissue is placed in one plane, each entry a
    column through the planes (TZ1), so the same holds there. The culture cells keep their
    own Z extents, the bank shifted by one constant so that its median nucleus centroid
    lies on the stack's middle plane (AU1), and are tested for overlap in 3D, so a column
    can hold two culture nuclei."
14. **"Packing", "Density".**
    *Old:* "With a value `φ`, ... seeds are added until the summed nominal size of the
    placed nuclei reaches `φ · ‖Ω‖`."
    *Proposed:* the same, followed by "In the appearance presets of
    {doc}`synthetic-real-appearance` objects are added until the composed stain's
    visible-stain share reaches its target (DN1, `slow` presets) or a fixed count is placed
    (DN3, routine presets), with F2's record and floor."
15. **"Packing", "A seed that cannot be placed (C5)", F2.**
    *Old:* "With `on_crowded="stop"`, placement ends at the first seed with no accepted
    proposal after `max_proposals`; that seed is not placed; the reached density and the
    number of placed seeds are recorded in `context.json`; if the reached density is below
    `floor · φ`, `ValueError` names the seed and both densities."
    *Proposed:* the same, followed by "In the appearance presets of
    {doc}`synthetic-real-appearance` no seed is fixed in advance: each proposal draws its
    own entry and transform. Placement ends at the target (DN1) or the count (DN3), at
    `max_proposals` proposals in total, or after `max_refusals` refused proposals in a row.
    The reached value, the placed objects and the proposals are recorded in
    `appearance.json`, and below `floor` of the target a `ValueError` names both ('When
    placement stops' there)."
16. **"Packing", "The two regimes".**
    *Old:* "*Dense tissue*: `κ` about 0.1, `g = 0`, `contact` set from the touching share,
    the density from the volume (area) fraction; territories fill most of the space
    between nuclei. *Sparse culture*: ... their sparseness comes from the density, not
    from a gap."
    *Proposed:* the same, followed by "In the appearance presets of
    {doc}`synthetic-real-appearance` both regimes come from the bank and PL2: contact
    proposals with probability 0.75, at most 1 % overlap, and the density from the
    visible-stain share of the source images (DN1)."
17. **"Cell territories", the recommendation.**
    *Old:* "**Recommendation: T2 for dense 3D tissue and 2D tissue, T1 for the culture**"
    *Proposed:* the same, followed by "; in the appearance presets, CT1 (the nearest
    placed nucleus within the growth) and the culture cells' own masks
    ({doc}`synthetic-real-appearance`)."
18. **"Cell territories", "A nucleus lies inside its own cell".**
    *Old:* "Under all three rules, `N(σ) = n > 0` requires that the power-nearest site is
    a site of `n`, that `σ ∈ Ω` and that `ρ_n(σ) ≤ 1`, and the cell label `C(σ)` is
    computed in the same evaluation from the same site."
    *Proposed:* the same, followed by "In the appearance presets of
    {doc}`synthetic-real-appearance` it holds by one placement order: on every voxel of a
    nucleus label the cell label is that nucleus's cell, and the distance rule CT1 decides
    only where no nucleus label holds the point ('One ownership rule for nuclei and cells'
    there)."
19. **"Cell types", "What else may depend on the type (C8, second part)".**
    *Old:* "The configuration allows a value per type for the mean molecules per cell, the
    nuclear fraction of reads and a scale of the nucleus size laws; every other law stays
    per scene."
    *Proposed:* the same, followed by "In the appearance presets of
    {doc}`synthetic-real-appearance` the size scale has no effect, since sizes come from the
    entries and the augmentation's scale."
20. **"Stains", "The interface (C12)", option S1 and its recommendation.**
    *Old:* "`StainChannelConfig(name, source, round, ...)`, one per channel; `source` is
    `nucleus`, `territory` or `boundary`; the same appearance controls (brightness, edge,
    unevenness, background, noise, output) for every source." and "**Recommendation:
    S1.**"
    *Proposed:* the same, followed by "In the appearance presets of
    {doc}`synthetic-real-appearance` each stain is one channel of the bank's entries, with
    the name, round and file pattern of its `StainChannelConfig`; its source and appearance
    controls are not used ('Composition' there)."
21. **"Stains", "Bright regions (C13)", the recommendation.**
    *Old:* "**Recommendation: Tx1**"
    *Proposed:* the same, followed by "for the closed-form presets; the appearance presets
    carry the real texture of their entries."
22. **"Stains", "Unevenness across the field (C14)", the recommendation.**
    *Old:* "**Recommendation: U1.**"
    *Proposed:* the same, followed by "The appearance presets take the spread between
    nuclei from their entries and the gain (0.9–1.1), with no field."
23. **"Stains", "Background and noise".**
    *Old:* "The existing controls, unchanged: the stain's `BackgroundConfig` ... and
    `NoiseConfig`, applied by `_observe` ... as today."
    *Proposed:* the same, followed by "In the appearance presets the background is
    assembled from real patches and no observation stage is added (BG1, OB1 of
    {doc}`synthetic-real-appearance`)."
24. **"Stains", "The territory source", last sentence.**
    *Old:* "W-393 did not measure a cytoplasm stain ..., so `q`, `B_k` and the edge of the
    cytoplasm stain are development values for the gate."
    *Proposed:* the same, followed by "The culture appearance presets carry the real
    Flamingo of their cell entries."
25. **"Stains", "8-bit output (C15)", option B1 and its recommendation.**
    *Old:* "After the noise, multiply the float64 plane by `k = g / m_A`, with `m_A` the
    median of the stain's brightness law ..." and "**Recommendation: B1** (W-393 choice 4)."
    *Proposed:* the same, followed by "In the appearance presets of
    {doc}`synthetic-real-appearance` the entries are relative to their source plateau, so the
    composed image is multiplied by `g` itself, the median plateau of the bank's source
    images (O8), with no noise stage before the cast."
26. **"`Z=1`", the list of what lies in the plane z = 0.**
    *Old:* "With one plane (`fov_shape_zyx[0] = 1`; the jitter bound in Z is 0, as today),
    every object lies in the plane z = 0" and the items "Nuclei", "Packing", "Territories",
    "Molecules and bright regions", "Unevenness", "Morphology round" and "Labels".
    *Proposed:* the same, followed by "In a scene of {doc}`synthetic-real-appearance` with
    one plane, every object is evaluated at z = 0 like any point: a 2D entry is its plane
    and a 3D entry its section through the plane; placement runs in the plane; territories
    follow CT1 or CU1 in the plane; `τ_z = 0`; `plane` label images."
27. **"The four cases, their controls and their ranges", the column "Control that produces
    it".**
    *Old:* the controls of the closed form (packing, `UnevennessConfig`, bright regions,
    `BackgroundConfig`, lobes).
    *Proposed:* the same, followed by a note: "In the appearance presets the controls are
    the bank, its selection rules, the density (DN1), the fall-off length and the
    out-of-focus light ({doc}`synthetic-real-appearance`)."
28. **"Presets", first sentence.**
    *Old:* "Six presets in `CONTEXT_PRESETS` (version `context-presets-v1`), separate from
    `SAMPLE_PRESETS`: one routine and one `slow` preset per context."
    *Proposed:* the same, followed by "The appearance presets are separate again, in
    `APPEARANCE_PRESETS` ({doc}`synthetic-real-appearance`)."
29. **"Truth schema", the recommendation V1.**
    *Old:* "`save_context_scene` writes the `save_sample_scene` layout unchanged, plus the
    files below and `context.json` with `context_truth_version` `"1"`."
    *Proposed:* the same, followed by "A scene of {doc}`synthetic-real-appearance` writes
    this layout with the ellipsoid columns null, and adds `appearance.json`,
    `placements.csv` and `background_patches.csv`."
30. **"Truth schema", "Which truth holds each parameter".**
    *Old:* "Every parameter of a new object is either a column of a table above or
    recomputed from the seed and a named stream ("Random streams")", and the rows of the
    table that follows.
    *Proposed:* the same, followed by "In a scene of {doc}`synthetic-real-appearance` the
    placed objects' parameters are in `placements.csv`, with the bank named by its SHA-256
    in `appearance.json`; the background patches are in `background_patches.csv`, and `κ`
    and `g` in `appearance.json`. `lobes.csv`, `harmonics.csv`, `stain_objects.csv`,
    `bright_regions.csv` and `unevenness.csv` are empty, and no stain noise is drawn (OB1)."
31. **"Truth schema", "Identities a test can check", the preamble.**
    *Old:* "**Identities a test can check** (in addition to identities 1 to 5 of
    {doc}`synthetic-sample-scene`, which hold with the new label functions)."
    *Proposed:* the same, followed by "Identities 2 (labels from the shape tables), 7
    (stains equal their closed forms), 9 (bright regions) and 10 (`k = g / m_A`) are those
    of the closed-form scene. A scene of {doc}`synthetic-real-appearance` states its own
    ('Identities a test can check' there). Identities 1, 3 to 6 and 8 hold there as
    written; identity 11 holds for its 2D tissue and is reported for its culture."
32. **"Accepted statements this addition changes", the proposed wordings for
    {doc}`synthetic-sample-scene` in its entries 2, 5, 6, 7, 8 and 13.**
    *Old:* entry 2, "the cellular scene of {doc}`synthetic-cellular-scene` adds lobed and
    touching nuclei, cell types and placement by compartment"; entry 5, "in the cellular
    scene ..., because its placement test keeps every centre strictly inside its own power
    cell"; entry 6, "In the cellular scene ... a nucleus is a union of ellipsoids, lobe 1
    optionally perturbed by spherical harmonics, and its site sphere bounds every lobe";
    entry 7, "The cellular scene ... replaces this rule when its packing settings are given
    ..."; entry 8, "Nuclei never overlap: a point carries only the nucleus of its
    power-nearest site ..."; entry 13, "A scene of {doc}`synthetic-cellular-scene` writes
    the same files with the same columns, made by the rules of that page ...".
    *Proposed:* each the same, followed by the appearance scene of
    {doc}`synthetic-real-appearance`:
    * entry 2: "; its real-appearance scene places real nuclei and culture cells instead";
    * entry 5: "; in the real-appearance scene, because a proposal whose anchor lies in a
      placed mask is refused";
    * entry 6: "; in the real-appearance scene a nucleus is a placed real mask";
    * entry 7: "; the real-appearance scene places objects by PL2 and stops as 'When
      placement stops' states";
    * entry 8: "; in the real-appearance scene a point carries the nucleus of the first
      placed object that holds it, and on every nucleus voxel the cell label is that
      nucleus's cell";
    * entry 13: "; a real-appearance scene writes them with the ellipsoid columns null,
      versioned by `APPEARANCE_TRUTH_VERSION` in `appearance.json`".
33. **"Exclusions", "No real data".**
    *Old:* "**No real data** is read by this page or its golden test."
    *Proposed:* "**No real data** is read by this page's golden test, the routine tier or
    the CI environment. The appearance banks of {doc}`synthetic-real-appearance` are built
    from real images outside the repository, and only `slow` tests on the lab machine read
    them."
34. **"Exclusions", "No ER stain".**
    *Old:* "**No ER stain** (D2), no structured tissue background, no per-FOV illumination,
    no cross-FOV cell matching: outside this delivery."
    *Proposed:* "**No ER stain** (D2), no per-FOV illumination, no cross-FOV cell matching:
    outside this delivery. A structured background from real patches is part of
    {doc}`synthetic-real-appearance`."

**{doc}`synthetic-cellular-scene-validation`**

35. **"The claim boundary", "No annotation".**
    *Old:* "The real labels W-393 measured are method output, read as measurements of
    shape and spacing, and nothing here is or replaces an annotation."
    *Proposed:* the same, followed by "In the appearance presets the truth masks are
    themselves method labels of real images ({doc}`synthetic-real-appearance`); W-514
    states how a chain check reads them."
36. **"Conventions", "Fixtures".**
    *Old:* "Fixtures are generated in the test session from the values stated here; no
    test reads the network share, a run directory or a real image, and no generated image
    is committed."
    *Proposed:* the same for the routine tier and the CI environment, followed by "A `slow`
    test of an appearance preset reads its bank file, derived from real images, by the
    path of `STARFINDER_APPEARANCE_BANKS`, and skips with a stated reason without it."
37. **"Thresholds and where they come from", row TR1.**
    *Old:* "TR1 | Share of touching nuclei, per image | 0.44–0.57; 0.76–0.82; 0.33–0.72 |
    `geometry.csv` "share of touching nuclei, per image""
    *Proposed:* the same for the closed-form presets; for the appearance presets "TR1 |
    Share of touching nuclei, per image | reported on the truth labels, not compared: the
    truth is placed at the visible stain's density (round 10: 0.99 against 0.56 on the
    held-out LN crop); the touching share is compared like for like, on a method's labels of
    the scene against the same method's labels of the real image, in the chain checks
    (W-514)".
38. **"Thresholds and where they come from", row TR2.**
    *Old:* "TR2 | Nucleus volume (area) fraction, per image | 0.07–0.16; 0.30–0.44;
    0.17–0.45"
    *Proposed:* the same for the closed-form presets; for the appearance presets "TR2 |
    Visible-stain share, per image (stain smoothed by one voxel above its Otsu threshold) |
    the ranges W-514 sets from W-513 `tables/visible_share_<context>.csv`", with the truth
    label share reported.
39. **Row TR3.**
    *Old:* "TR3 | Share not described by one ellipsoid (IoU < 0.8), per image |
    0.064–0.074; 0.027–0.060; 0.12–0.43"
    *Proposed:* the same for the closed-form presets; for the appearance presets the same
    row on the truth labels, followed by "a miss caused by leaving out squashed sources
    (limit L2 of {doc}`synthetic-real-appearance`; round 10: 0.009 against 0.047 on the
    held-out LN crop) is reported with the squash threshold, its control; W-514 states
    whether the row is asserted".
40. **Row TR8.**
    *Old:* "TR8 | Background over plateau, per image; noise over plateau, per 128-px block
    | 0.01–0.34, 0–0.063; ..."
    *Proposed:* the same for the closed-form presets; for the appearance presets the
    background is "the median stain outside the visible stain, over the plateau", with the
    ranges W-514 sets from the same table, column `not_visible_over_plateau`; the noise row
    unchanged.
41. **The paragraph "Amendment of 2026-10-15 (W-485; operator decision pending Jiahao's
    confirmation)".**
    *Old:* "TR7 of dense 3D tissue is therefore a known limit of the scene: GN23 reports it
    with its measured value, range and object count and does not assert it. A texture
    control that reaches it is a candidate for a later issue."
    *Proposed (AM1):* "TR7 of dense 3D tissue is a known limit of the closed-form presets:
    GN23 reports it there and does not assert it. The appearance presets of
    {doc}`synthetic-real-appearance` are the texture control: round 10 of the prototype
    gave 0.317 against the real 0.348 (held-out crop), and their TR7 is compared as W-514
    states."

| Option for the W-485 amendment (RA24) | Cost | Consequence |
| --- | --- | --- |
| **AM1. It stands for the closed-form presets; it does not apply to the appearance presets (recommended)** | One sentence. | The accepted presets keep their recorded limit, and the appearance presets meet TR7 (round 10: 0.317, 0.285, 0.218 against 0.348, 0.266, 0.343 real; `round10/logs/<context>.log.json`, `within_cv_p5_med_p95`). |
| AM2. Withdraw it | Edits the accepted page's record. | GN23 on the closed-form dense 3D preset fails again, for a known reason. |
| AM3. Leave it as written | None. | It names "a later issue" for a texture control that this page is. |

42. **"How many nuclei the thresholds count".**
    *Old:* "a `slow` scene holds about 150 nuclei in dense 3D tissue, about 76 in 2D tissue
    ... and about 42 in the culture"
    *Proposed:* the same for the closed-form presets, followed by "the appearance presets
    hold about 738, 337 and 32 objects (estimates of {doc}`synthetic-real-appearance`,
    'Resources')."
43. **"Exclusions", "No real data".**
    *Old:* "**No real data, no new dependency, no lock change, no marker registration.**"
    *Proposed:* "**No real data in the repository, the routine tier or the CI environment;
    no new dependency, no lock change, no marker registration.**"

The generator checks GN13 (stain identities), GN15 (bright regions), GN16 (field) and
GN23 (targets) and the chain checks CD, CT and CU are amended, kept or replaced by W-514.
Which of them stands for the appearance presets, and how W-487 and W-488 change, is
W-514's to state (open point).

## Implementation

This page's scene is implemented in `starfinder.synthetic` by W-516 to W-519 and checked by
the generator checks GA1 to GA27 of {doc}`synthetic-real-appearance-validation` (W-516 to W-520). The
three real banks `ln-sample1-v1`, `tissue2d-v1` and `culture3d-v1` were built by W-519
outside the repository ("Execution conditions" of {doc}`synthetic-real-appearance-validation`); the
presets name each by its name and SHA-256, and only `slow` tests on the lab machine read
them, through `STARFINDER_APPEARANCE_BANKS`.

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

* the generator checks GN1 to GN25 of {doc}`synthetic-cellular-scene-validation`: `test_synthetic_cellular_contract.py`,
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
* **2026-10-18, W-488** (`docs/examples/context_scene.py`, {doc}`api/examples`). The example
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
