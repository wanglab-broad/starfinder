# Synthetic sample scene: multi-FOV acquisition and a minimal cellular scene

% Status: Accepted (W-329, 2026-10-07, at 9033d46)

This page specifies a bounded, opt-in addition to the synthetic generator of
[synthetic specification](amplicon-scene.md): one scene of sample size in continuous coordinates, a
nominal grid of FOVs acquired from it at true offsets with a known fractional jitter, a
minimal cellular scene (nucleus and cell label volumes and a nuclear stain), cell truth for
every molecule, per-FOV truth, and the path by which per-FOV truth labels enter the
segmentation's mask import. It gives stitching and assembly the two kinds of truth that
real data cannot give: the true offset of every FOV, in 3D and with a fractional part, and
the knowledge of which FOVs see each true cell and molecule. It is also a fixture that the
processing report's measurement QC can use later.

% the §2.9 mask import; It gives §2.10; §2.11 measurement QC

It builds on these pages and redefines none of them:

* [synthetic specification](amplicon-scene.md): the formed-amplicon model, its stages, truth tables and
  random streams, used unchanged for every FOV;
* [stitching-contract](../../../docs/stitching-contract.md): the sample frame, the tile position table and
  `positions_from_grid`, the placement record (`PlacementResult`, `origin_z/y/x` in µm,
  the gauge G1) and the index-to-sample rule;
* [assembly-contract](../../../docs/assembly-contract.md): footprints, ownership, retention, statuses and the count
  identities;
* [segmentation-contract](../segmentation.md) ("External-mask import", "Label image") and
  [assignment-contract](../assignment.md) (`MoleculeTable`, `import_legacy_molecules_csv`, "Coordinate
  sampling").

It is implemented in `starfinder.synthetic`, no existing generator behavior changes and
nothing is renamed. The validation designs that use the scene are
[stitching and assembly validation design](../../../docs/stitching-assembly-validation.md) and [synthetic cellular scene validation](../../validation/synthetic-cellular-scene.md); this page
states the identities a test can check, not the validation plan.

## What this addition is not

* It does **not establish the accuracy of a segmenter.** The segmentation methods, learned ones
  included, run on the cellular scene of [synthetic cellular scene](cellular-scene.md) as an engineering
  check of the chain (the first of the synthetic data's planning decisions), never as a
  measure of a method's quality, and no segmentation accuracy is claimed from it.
* It is **not a calibrated tissue model.** In this scene nuclei are smooth ellipsoids,
  cells are power-diagram territories and molecules are uniform within them; the cellular
  scene of [synthetic cellular scene](cellular-scene.md) adds lobed and touching nuclei, cell types and
  placement by compartment; its real-appearance scene places real nuclei and culture cells
  instead. Sizes and densities are development targets, set there from the context
  measurements of method labels, not fitted to tissue and not a calibration.
* It does **not use microsim** or any other new rendering dependency (Jiahao,
  2026-10-05). Everything is rendered by the existing analytic model or by closed-form
  functions evaluated at voxel centres, with NumPy and SciPy, which are already locked.
* It does not model stage drift, rotation, illumination differences between FOVs,
  photobleaching in overlaps or cross-FOV cell matching.

% The §2.9 methods; D1 of the §2.12 planning decisions; W-393's measurements; not a D04 calibration

## Settled decisions this page follows

* **The synthetic data's scope is amended** for multi-FOV and cellular synthetic truth,
  without microsim (the stitching planning decisions of 2026-10-05). The synthetic data's
  milestone had excluded cellular masks and assignment truth; this page is the bounded
  addition. Canonical responsibility for synthetic truth and reproducibility stays with the
  synthetic truth work.
* **Two sample-level entries** outside `PipelineConfig` and `FOV.run`: `Dataset.stitch`
  and `Dataset.export` (a stitching planning decision, as amended at this page's review of
  2026-10-06);
  `Dataset.export` calls `assemble`, `fuse_images`, `fuse_labels` and `export_spatialdata` in
  order ([sample-output-contract](../../../docs/sample-output-contract.md), "The export entry"); ownership regions are
  computed in assemble. The scene feeds those entries; it adds no entry of its own to the
  pipeline.
* **Placement runs on the CPU only** and is translation only. The scene's FOV offsets are
  translations; no rotation or deformation of a FOV against the sample is generated.
* **Retention** (2026-09-27, [assembly-contract](../../../docs/assembly-contract.md)): cells by centroid, no cross-FOV
  matching. The scene provides truth to measure that policy, not a replacement for it.

% The §2.12 scope is amended (D7 of the W-152 comment "§2.10 planning decisions (2026-10-05)"); the §2.12 milestone;
% W-93 keeps canonical responsibility; (D3, as amended at the W-329 review of 2026-10-06)

## Terms

* The **sample frame** is that of [stitching-contract](../../../docs/stitching-contract.md): ZYX, µm, identity direction,
  `frame_id` `sample:<sample>`.
* The **scene domain** `Ω` is the box of the sample frame in which scene content is
  generated; it contains every footprint for any allowed jitter ("The sample scene").
* The **covered region** `K ⊂ Ω` is the box that at least one footprint covers for every
  jitter within the bound. **The sample** is the content of `K`: the **sample molecules**
  `M` are the molecules whose position lies in `K`, and the **sample cells** are the cells
  whose anchor lies in `K` ("The sample scene", "Sample membership").
* The **halo** is `Ω \ K`, a band of width `2 · J_a · s_a` on each outer side
  (`(2 · J_a + T_a) · s_a` with the morphology round of [synthetic cellular scene](cellular-scene.md)). Its content
  is rendered so that FOV edges are not empty where a jitter exposes them, but it is not
  part of the sample, and nothing about it is claimed beyond what is stated for it.
* A FOV's **nominal origin** is its position from the nominal grid; its **true origin** is
  where it was actually rendered; the **jitter** is true minus nominal, the true
  correction in the language of the placement record.
* A FOV's **footprint** is that of [assembly-contract](../../../docs/assembly-contract.md): per axis
  `[origin − s/2, origin + (n − ½)·s)`. In index units, a position `p` is in the
  footprint exactly when `−0.5 ≤ p < n − 0.5` on every axis, which is the "inside" test of
  the assignment's coordinate sampling (`0 ≤ floor(p + 0.5) < n`).
* A **seed** is one generated cell: an ordinary cell has one nucleus, a binucleate cell
  two, an empty cell none.
* A **site** is one sphere of the power diagram: one per nucleus, and one virtual sphere
  per empty cell.

% §2.9 coordinate sampling

## Opt-in, and what stays as it is

The addition is **opt-in**: it is reached only through new names ("Names"). Nothing that
exists today calls it, and a caller who does not call it gets exactly today's behavior.
The implementation must keep, by construction:

* **No existing signature or config changes.** `generate_amplicon_scene`, `generate_amplicon_dataset`,
  `generate_registration_pair`, `save_amplicon_scene`, `amplicon_scene_preset` and every preset
  function keep their parameters. `AmpliconSceneConfig`, `ReadoutEffectsConfig`,
  `BackgroundConfig`, `TextureConfig`, `NoiseConfig`, `GeometryConfig` and
  `ScalarDistribution` gain no field: their `asdict` form is the `requested_config` of
  every provenance record, so a new field would change every saved `provenance.json`.
* **No registry entry changes.** `SCENE_PRESETS`, `BENCHMARK_PRESETS`,
  `DEFORMATION_PRESETS`, `DEVELOPMENT_FIXTURES`, `DEVELOPMENT_SIZES` and
  `CALIBRATED_CONDITIONS` keep their keys and values; the sample presets live in a
  separate registry, because `test_one_registry_holds_every_tier` calls
  `amplicon_scene_preset` on every `SCENE_PRESETS` name.
* **No version changes.** `_GENERATOR_VERSION` stays `"6"`, `PRESET_VERSION` stays
  `benchmark-presets-v1`, `HISTORICAL_TRUTH_VERSION` stays `"2.0"` and the stream
  namespace stays `starfinder.synthetic/1`. Every FOV scene of a sample scene is produced
  by the unchanged formed generator and records generator version 6; the sample-level
  additions carry their own versions ("Truth schema").
* **No stream descriptor of an existing component changes.** The new draws use new
  component names ("Random streams"), so no existing draw moves.
* **The package stays light.** Importing `starfinder.synthetic` still imports no
  `starfinder.benchmark`, `starfinder.provenance`, `starfinder.artifacts`,
  `starfinder.io.checkpoints`, `starfinder.io.candidates` or `starfinder.io.molecules`
  module (`test_old_generator_api_is_removed`,
  `test_package_imports_no_persistence_or_checkpoint_modules`), and the synthetic sources
  name no issue or artifact contract (`test_no_artifact_contract_or_issue_references`). The
  no-segmenter path therefore writes plain files, and the consumer builds the segmentation
  and assignment inputs from them.

% the §2.9 inputs

**Existing tests that must pass unchanged** (no edit to any assertion, fixture or pinned
value):

* every file with the `synthetic` marker: `test_synthetic_contracts.py`,
  `test_formed_scene.py`, `test_formed_geometry.py`, `test_formed_streams.py`,
  `test_development_fixtures.py`, `test_development_presets.py`,
  `test_calibrated_presets.py`, `test_background_noise.py`, `test_readout_effects.py` and
  `test_benchmark_synthetic.py`;
* the tests that consume generator output of existing presets and pin results computed
  from it: `test_readout_validation.py` (R6, `PINNED_CAL_141C093` and
  `test/data/readout_r6_cal_141c093/`, from `calibrated_scene_preset` at seed 103),
  `test_spot_finding_validation.py` and `test_spot_finding_rounds.py` (scenes of
  `spot_finding_scenes.py` and `spot_finding_fixtures.py`), `test_assignment.py` (the
  `development_scene_preset("clean", size="small")` FOV run),
  `test_preprocessing_synthetic_evaluation.py`,
  `test_preprocessing_calibrated_evaluation.py`, `test_image_statistics.py`,
  `test_segmentation_labels.py`, `test_segmentation_segment.py` and
  `test_dataset_channels.py`; (on 2026-10-25 R6's `PINNED_CAL_141C093` and that folder
  were replaced with the equal explicit expected values of `test/data/expected/test_readout_validation/`.)
* the session fixture `small_dataset` of `conftest.py` with its recorded
  `SMALL_DATASET_GENERATOR` (`generator_version` 6, `benchmark-presets-v1`).

% (W-573 later replaced R6's `PINNED_CAL_141C093`)

**Preset outputs whose digests stay bit-identical**, for the same seed, NumPy build and
CPU (the repeatability limit of [synthetic specification](amplicon-scene.md), "Random streams"):

* the per-round `image_sha256`, the `formed`, `round_truth` and signal tables, and the
  whole `provenance` record of `formed-small-v1`, `formed-z1-v1`, the four development
  fixtures (`z1-clean`, `z1-combined`, `small-clean`, `small-combined`), every
  `calibrated_scene_preset` condition and dtype, and one FOV of each benchmark preset
  (`tiny`, `small`, `medium`, `large`, `tissue`, `thick_medium`) from
  `benchmark_scene_preset` and `registration_scene_preset`;
* the files written by `starfinder synthetic generate` in both modes (`--mode e2e` and
  `--mode registration`), including `ground_truth.json` and `generation.json`;
* the files of `save_amplicon_scene` for the fixtures above.

The implementation shows this by running the files above unchanged; the cross-process
test (`test_cross_process_repeatability_with_hash_seeds`, which generates every preset and
fixture above, the dataset-level path and the effect paths in two processes with different
`PYTHONHASHSEED` values, in an empty working directory that it leaves empty)
compares the provenance and digests of these presets between processes, path by path, and the pinned R6
and spot-finding digests compare them with values pinned before this addition.

## Names

All new, in `starfinder.synthetic`, with `__all__` and the autosummary list of
`docs/api/synthetic.rst` gaining the exports that `docs/check_reference.py` requires:

| Concept | Name | Kind |
| --- | --- | --- |
| Entry | `generate_sample_scene(codebook, *, config=SampleSceneConfig(), on_fov=None)` | function |
| Result | `SampleScene` | dataclass |
| Config | `SampleSceneConfig`, `AcquisitionConfig`, `CellsConfig`, `StainConfig` | frozen dataclasses |
| Persistence | `save_sample_scene(scene, directory)` | function |
| Presets | `sample_scene_preset(name)`, `SAMPLE_PRESETS`, `SAMPLE_PRESET_VERSION` (`sample-presets-v1`) | function, dict, str |
| Version | `SAMPLE_TRUTH_VERSION` (`"1"`) | str |

```python
@dataclass
class SampleScene:
    sample: str                                   # the sample frame is sample:<sample>
    fovs: dict[str, AmpliconScene]                  # one amplicon scene per FOV, in table order
    stains: dict[str, np.ndarray]                 # ZYX stain image per FOV (empty without a stain)
    fov_metadata: dict[str, ImageMetadata]        # each FOV's reference grid metadata (FOV frame)
    placement: pd.DataFrame                       # placement truth ("Placement truth")
    positions: pd.DataFrame                       # the nominal tile position table
    nuclei: pd.DataFrame                          # sample-level tables ("Truth schema")
    cells: pd.DataFrame
    molecules: pd.DataFrame
    cell_fovs: pd.DataFrame
    molecule_fovs: pd.DataFrame
    sample_grid: ImageMetadata                    # the sample label grid
    sample_labels: dict[str, np.ndarray]          # "nucleus", "cell": uint32 ZYX on the sample grid
    fov_labels: dict[str, dict[str, np.ndarray]]  # fov_id -> "nucleus", "cell": uint32 ZYX on the FOV grid
    record: dict                                  # versions, requested and effective config, streams, digests
```

With `on_fov(fov_id, scene, stain, labels)`, each FOV is handed over as soon as it is
complete and not retained, as `generate_amplicon_dataset`'s `on_round` does for rounds; `fovs`,
`stains` and `fov_labels` are then empty and the tables are complete.

## The sample scene

One scene of sample size is defined once, in continuous sample-frame coordinates (µm),
and every FOV is rendered from it.

* **Scene domain.** With the nominal grid of "Multi-FOV acquisition", per axis `a`, FOV
  size `n_a`, spacing `s_a`, the largest nominal origin `N_a` and the jitter bound `J_a`
  (voxels):
  `Ω_a = [−s_a/2 − J_a·s_a, N_a + (n_a − ½)·s_a + J_a·s_a)`.
  `Ω` contains every footprint for any jitter within the bound. All scene content (seeds,
  nuclei, cells, molecules, background) is generated in `Ω`.
* **Covered region.** Per axis
  `K_a = [−s_a/2 + J_a·s_a, N_a + (n_a − ½)·s_a − J_a·s_a)`,
  that is `Ω` shrunk by `2 · J_a · s_a` on each side (for Z=1, `J_z = 0`, so
  `K_z = Ω_z = [−s_z/2, s_z/2)`, and every centre and molecule has z = 0). With a
  morphology round of translation bound `T_a` voxels ([synthetic cellular scene](cellular-scene.md)),
  `Ω` grows by `T_a · s_a` on each side, so that the shifted round images generated
  content; `K` is unchanged, and is then `Ω` shrunk by `(2 · J_a + T_a) · s_a`. **Every point of `K` lies in at least one footprint, for every jitter within
  the bound.** Proof, per axis: the nominal footprint intervals of grid index `i` are
  `I_i = [i·step_a − s_a/2, i·step_a + (n_a − ½)·s_a)`. Shrink each by `J_a·s_a` on both
  sides to its core. Adjacent cores still overlap, because the overlap rule gives
  `overlap · n_a ≥ 2·J_a + 1` voxels, so the union of the cores is exactly `K_a`. A point of
  `K` therefore has, on each axis, an index `i_a` whose core holds it. The FOV at
  `(i_y, i_x)` exists (the grid is a full rectangle with one layer in Z), and its true
  footprint is its nominal interval moved by at most `J_a·s_a` per axis, so it still
  contains the core and the point. `K` is also the largest such box: a point outside it
  is uncovered when every FOV on that edge is jittered inward by the bound.
* **Sample membership.** A molecule belongs to the sample when its position `σ` lies in
  `K` (`in_sample` true), whatever its population or cell. A cell belongs to the sample
  when its **anchor** lies in `K`. The anchor is the centre of its first site: the
  nucleus centre of an ordinary cell, the first nucleus centre of a binucleate cell, or
  the virtual centre of an empty cell. The anchor always lies inside its own territory:
  here because it is inside its site sphere ("Cell territories"); in the cellular scene of
  [synthetic cellular scene](cellular-scene.md), whose site spheres may overlap, because its placement
  test keeps every centre strictly inside its own power cell; in the real-appearance
  scene, because a proposal whose anchor lies in a placed mask is refused. A nucleus belongs to the
  sample when its cell does. Everything else generated in `Ω` is **halo** content
  (`in_sample` false).
  * **A cell that straddles the boundary of `K`** keeps the one membership its anchor
    gives it. Its molecules are members by their own positions: a sample cell can have
    halo molecules where its territory crosses into the halo, and a halo cell can have
    sample molecules in `K`. Those molecules keep that cell as their `true_cell_id`.
* **Independence from the drawn jitter.** `Ω`, `K` and membership depend on the nominal
  grid, the jitter bound and the morphology round's translation bound, not on the drawn
  jitter or translation. Redrawing the jitter moves no cell
  and no molecule, and changes no membership.
* **Sample grid.** The sample label volumes are evaluated on the grid of spacing `s` whose
  voxel `q = (0, 0, 0)` has its centre at `origin_sample = lo(Ω) + s/2`, with shape
  `ceil((hi(Ω) − lo(Ω)) ⊘ s − 1e-9)` (the tolerance keeps a whole number of voxels from
  rounding up). Its `ImageMetadata` has `frame_id` `sample:<sample>`, that spacing, origin
  `origin_sample`, identity direction and unit `µm`. FOV `f`'s index `p` is sample-grid
  index `δ_f + p`, with `δ_f = (true_f − origin_sample) ⊘ s`, fractional in general. The
  sample grid covers the whole of `Ω`, halo included. When `Ω` is not a whole number of
  voxels long, its last voxel centres lie beyond `hi(Ω)`, where both labels are 0.
  `sample.json` records the bounds of
  `Ω` and `K`, and a sample-grid voxel is in `K` when its centre is.
* **Content.** The cellular scene (nuclei, cells) and the molecules (amplicons with their
  formed properties), as positions in µm, shapes and amplitudes; and the background
  components of the amplicon channels and of the stain, as `BackgroundConfig` records in
  the index units of the sample grid (centres and widths in voxels, a gradient normalized
  by the sample grid's shape). All are analytic records, never a rendered sample image.
* **Z=1.** A sample with `n_z = 1` is a plane at z = 0: the jitter bound in Z must be 0,
  every centre has z = 0, and the label functions ignore z, as the formed model does for
  singleton Z.

**The halo.** The covered region needs a decision about the content outside it.

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **H1. Keep a halo, mark it (recommended)** | Generate in `Ω`; the sample is the content of `K`; halo content carries `in_sample` false. | One more column on three tables. About 40 % of `sample-tiny-v1`'s molecules and 30 % of `sample-small-v1`'s lie in the halo, mostly in Z ("Presets and resources"). | FOV images look the same up to their edges for every jitter: no FOV shows an empty strip that a stitcher could register on. The identities are exact for the sample, and halo content is reported, not claimed. |
| H2. No halo: generate in `K` only | `Ω := K`; every generated molecule and cell is a sample member. | No marking; fewer molecules. | A FOV jittered outward images up to `2 · J_a` voxels of nothing at its outer edge. Cells and the stain end in a straight cut along `∂K`, an artificial edge in exactly the overlap-free border, and the stitching input then differs by FOV for a reason that real data does not have. |
| H3. Acquire a ring of extra FOVs around the sample | Add one row and one column of FOVs on each side, so the nominal union covers `Ω`. | A 2 × 2 sample becomes 4 × 4 FOVs, four times the rendering, and outside the routine bound. | The outer ring has the same problem one ring further out. It moves the halo rather than removing it. |

**Recommendation: H1.** It is the only option that keeps every FOV image natural at its
edge and still states exact coverage for a population fixed before the jitter is drawn.

## Multi-FOV acquisition

### Nominal grid

```python
@dataclass(frozen=True)
class AcquisitionConfig:
    columns: int = 2
    rows: int = 2
    order: str = "column_major_down_right"
    fov_shape_zyx: tuple[int, int, int] = (5, 36, 36)      # voxels
    spacing_zyx: tuple[float, float, float] = (1.0, 0.5, 0.5)   # µm per voxel
    overlap: float | tuple[float, float] = 1 / 3            # fraction of the FOV, (y, x) or one value
    jitter: str = "uniform"                                 # "uniform" (J1) or "supplied"
    jitter_max_zyx: tuple[float, float, float] = (0.4, 1.5, 1.5)   # voxels, J1 bound
    jitter_zyx: Mapping[str, tuple[float, float, float]] | None = None   # voxels, per FOV, "supplied"
    fov_ids: tuple[str, ...] | None = None                  # None: FOV_001, FOV_002, ... in k order
```

| Parameter | Unit | Default | Rule |
| --- | --- | --- | --- |
| `columns`, `rows` | – | 2, 2 | integers ≥ 1 |
| `order` | – | `column_major_down_right` | FOV `k` at column `k // rows`, row `k % rows`, the order of `positions_from_grid` ([stitching-contract](../../../docs/stitching-contract.md)) |
| `fov_shape_zyx` | voxels | (5, 36, 36) | positive integers; every FOV has this reference grid |
| `spacing_zyx` | µm | (1.0, 0.5, 0.5) | finite, > 0 |
| `overlap` | fraction | 1/3 | `0 ≤ overlap < 1`; the nominal step is `(1 − overlap) · n · s` in Y and X, as in `positions_from_grid` |
| `jitter_max_zyx` | voxels | (0.4, 1.5, 1.5) | finite, ≥ 0; `0` in Z when `n_z = 1` |
| `jitter_zyx` | voxels | None | with `jitter="supplied"`: one finite triple per FOV ID, `|j_a| ≤ jitter_max_a` |

The nominal origin of FOV `k` is `(0, row · step_y, col · step_x)` µm: every FOV has
nominal z 0 (one layer of tiles, option Z1 of [assembly-contract](../../../docs/assembly-contract.md)). The nominal
positions equal, value for value, those of
`positions_from_grid(columns=…, rows=…, order=…, tile_shape_yx=fov_shape_zyx[1:],
spacing_yx=spacing_zyx[1:], overlap=…, fov_ids=…)`.

**Validation.** In Y and X, wherever there are at least two columns (or rows),
`overlap · n_a ≥ 2 · jitter_max_a + 1` voxels, so that neighbouring footprints overlap by at
least one voxel for every jitter: every nominal neighbour pair stays a neighbour pair and
the union of footprints has no interior gap. In Z, `jitter_max_z < n_z / 2`. A violation
raises `ValueError` naming the axis.

### True offsets and the form of the jitter

The true origin of FOV `f` is `true_f = nominal_f + j_f ⊙ s` with `j_f` in voxels.

| Option | Rule | Cost | What it tests |
| --- | --- | --- | --- |
| **J1. Bounded uniform, independent per FOV and axis (recommended)** | `j_{f,a} = J_a · (2u − 1)`, `u ~ U[0, 1)` from the stream `sample.jitter` of FOV `f`, drawn z, y, x in order. Continuous draws have a fractional part on every axis with probability 1; a preset test checks that every axis of every FOV has a non-integer jitter. | Three draws per FOV; the bound makes the overlap rule above a guarantee. | Sub-voxel recovery in every axis, independent of the grid structure; the corrections the stitching parity runs compared ("removing the mean translation"). |
| J2. Truncated Gaussian | `j ~ N(0, σ_a²)` truncated at `±J_a` by redrawing | As J1, plus a redraw loop; the same bound. | As J1, with most offsets small and a few near the bound. |
| J3. Linear trend plus J1 residual | `j_f = g ⊙ (col_f, row_f, 0) + J1 residual` with a supplied per-column and per-row slope `g` | Two more parameters; the bound applies to the total. | A systematic per-column drift like the 1.7 px per column trend the stitching parity runs measured between the backends on LN (their notes, section 2.2); a gauge that removes only the mean leaves the trend visible. |

% J1: the corrections W-324 compared; J3: the trend W-324 measured (notes section 2.2)

**Recommendation: J1**, with `jitter="supplied"` always available for hand-chosen cases
(for example an integer jitter against the nominal grid, as the stitching parity runs' cut fixtures used,
or a known trend). J1 is the simplest law whose bound
guarantees the overlap rule, it isolates sub-voxel placement error from systematic error,
and its truth is three numbers per FOV. J3 is the natural next step if the validation
design needs a trend; it can be written as a supplied table today.

% as the W-324 cut fixtures used

### Rendering each FOV at its true offset

Each FOV is rendered by the unchanged formed generator, as one `AmpliconScene` on its own
grid, with **no interpolation of a rendered image**: every voxel value is the analytic
model evaluated at that voxel's centre.

* **Molecules.** A molecule at sample position `σ` (µm) is at FOV index position
  `p = (σ − true_f) ⊘ s` (float64). FOV `f` receives every molecule whose `p` lies within
  `[−0.5 − M_a, n_a − 0.5 + M_a]` on every axis, with the margin
  `M_a = 4 · w_a + D_a + 1` voxels, `w_a` the largest axis-aligned kernel extent over the
  generated molecules (sample and halo) ([synthetic specification](amplicon-scene.md), `support_truncated`) and `D_a` the
  bound of the FOV's per-round displacement on that axis. Molecules beyond the margin
  cannot touch the grid in any round; including them would change nothing but the truth
  rows. Each molecule enters with its sample ID and its sample-level gene and properties.
* **Background of the amplicon channels.** The template's background components are
  sample-grid records ("The sample scene"). FOV `f` evaluates them at sample-grid index
  `δ_f + F_r⁻¹(p)`: for a translation round exactly per axis, as
  `evaluate_background_translated` does, with the translation `t_r − δ_f`; for a deformed
  round through the inverse map of the formed model. A gradient is normalized by the
  sample grid, not the FOV grid, so it is continuous across FOVs.
* **Rounds.** Each FOV keeps its own per-round geometry (translation, local, affine,
  polynomial), drawn from its own streams, with the reference round at identity in the FOV
  frame: in an acquisition, round-to-round registration is per FOV. The FOV's true origin
  applies to the reference frame, so every round of the FOV is placed by it.
* **Noise.** Every residual of every FOV is drawn from that FOV's streams (scene key
  `[scene_key, fov_id]`, as `generate_amplicon_dataset` gives), so **the noise is independent per
  FOV**: two FOVs that image the same molecule in an overlap see independent noise.
* **Stochastic readout effects** (`loss`, `dropout`, `weakening`) are per-ID draws of each
  FOV's scene; drawn per FOV, the same molecule could be lost in one FOV and kept in its
  neighbour. A sample scene therefore requires them disabled in the template
  (`ValueError` otherwise); deterministic effects (trend, gains, mixing) are allowed. A
  later version may draw them once per molecule at sample level (worker notes).
* **Metadata.** Each FOV's reference `ImageMetadata` is in the FOV frame
  (`<formed namespace>/reference`, as today) with the acquisition spacing, origin
  `(0, 0, 0)`, identity direction and unit `µm`, which is what `stitch` and `assemble`
  require ("Relation to `ImageMetadata.origin`", [stitching-contract](../../../docs/stitching-contract.md)). The true
  origin is never written into it; it is truth.

**Implementation route.**

| Option | How | Cost | Effect on existing outputs |
| --- | --- | --- | --- |
| **R1. Compose public configs per FOV (recommended)** | For each FOV, `replace(template, FOV_id=f, scene_key=[scene_key, f], shape_zyx=n, coordinates=P_f, amplicon_ids=…, gene_ids=…, brightness=ScalarDistribution("supplied", …), …)`, with the background components translated into the FOV frame (region and texture centres minus `δ_f`, supplied texture widths and heights, the gradient's intercept and slopes re-expressed for the FOV grid) and the FOV metadata above; then the unchanged `_generate`. | No edit of `_formed.py`, `_observation.py` or `_geometry.py`. The gradient can be re-expressed only while the FOV intercept stays ≥ 0 (true for nonnegative slopes; validated, `ValueError` otherwise). | None, by construction: no existing code path changes. |
| R2. An internal offset hook | `_generate` gains a private argument carrying sample-frame components and the FOV offset, evaluated with `evaluate_background_translated` directly. | A change inside the formed generator's control flow; every gradient allowed. | None if the default path is untouched, but that has to be shown by the digests rather than by construction. |

**Recommendation: R1.** It meets "no change to existing outputs" without argument, and
its one restriction (a gradient whose FOV intercept would be negative) does not occur in
the presets.

## Placement truth

`SampleScene.placement`, saved as `placement_truth.csv`, has one row per FOV in tile
position table order:

| Column | Type, unit | Content |
| --- | --- | --- |
| `fov_id` | str | The FOV, as in `positions` and every per-FOV table |
| `grid_column`, `grid_row` | int | The nominal grid cell |
| `nominal_z`, `nominal_y`, `nominal_x` | float64, µm | The nominal origin, equal to `positions` |
| `true_origin_z`, `true_origin_y`, `true_origin_x` | float64, µm | The sample-frame position of index `(0, 0, 0)` of the FOV's reference grid |
| `true_correction_z/y/x` | float64, µm | `true_origin − nominal`, the jitter in µm |
| `jitter_z_vox`, `jitter_y_vox`, `jitter_x_vox` | float64, voxels | The jitter in the FOV's index units, as drawn |
| `true_origin_g1_z/y/x` | float64, µm | `nominal + c − mean(c)` with `c` the true correction and the mean over all FOVs: the truth in gauge G1 |
| `shape_z/y/x`, `spacing_z/y/x` | int, float64 µm | The FOV's reference grid |

**Units and direction.** Positions are in µm in the sample frame. The direction is the
push sense of the placement record ([stitching-contract](../../../docs/stitching-contract.md), "Index to sample
coordinates"): a FOV index position `p` is at `sample = true_origin + p ⊙ spacing`, the
identity direction of the sample frame. A `PlacementResult` is therefore compared with the
truth column for column: its `origin_*` with `true_origin_g1_*` under G1 (both have zero
mean correction over the FOVs, when every FOV is `image_supported`), its `correction_*`
with `true_correction_* − mean`, and its `nominal_*` with `nominal_*`. `positions`, saved as
`positions.csv` with `fov_id`, `nominal_z/y/x` in µm, is a supplied tile position table
that `positions_from_table` reads unchanged.

**The identity a test can check.** For every FOV `f`, every row of `f`'s `formed` table
(the molecules rendered in `f`, at their reference-round index positions `p`) and the
placement-truth row of `f`:

```
true_origin_f + p ⊙ spacing_f  ==  (sample_z, sample_y, sample_x) of that molecule in molecules.csv
```

on every axis within **1e-9 µm** (absolute); equivalently
`ImageMetadata("sample:<sample>", spacing, true_origin, identity, "µm").index_to_world(p)`,
which is what `placed_metadata` returns for a placement holding the true origins. The
tolerance covers float64 rounding: the generator computes `p` with one subtraction and one
division and the test inverts them with one multiplication and one addition, each with
relative error at most 2⁻⁵³, on positions below 10⁴ µm, so the error is below 1e-11 µm;
1e-9 µm is 1e-8 of the smallest spacing in use (0.0946 µm, the stitching input
inventory's `inputs.md`), so no voxel-level error can hide inside it.

% W-323 `inputs.md`

Two further identities tie the rendered images to the truth, with noise disabled and
float64 accumulation:

* **Rendering.** FOV `f`'s reference-round image equals the independent oracle of the
  formed model (`test/formed_oracle.py`) evaluated at the `p` above with the molecules'
  properties, within the oracle tolerance of `test_development_fixtures.py`.
* **No interpolation.** Rendering FOV `f` again at `true_f + k ⊙ s`, for an integer
  vector `k`, gives the first image shifted by `k` voxels on the voxels both cover, within
  1e-12 of the image maximum. This holds for fractional true offsets because each voxel
  value is evaluated, not resampled.

## The cellular scene

```python
@dataclass(frozen=True)
class CellsConfig:
    nuclei: int = 7                       # ordinary cells, one nucleus each
    binucleate: int = 1                   # cells with two nuclei
    empty: int = 0                        # cells without a nucleus
    nucleus_axial_radius: ScalarDistribution = ScalarDistribution("lognormal", (log(1.5), 0.1))   # µm, semi-axis along z
    nucleus_lateral_radius: ScalarDistribution = ScalarDistribution("lognormal", (log(2.5), 0.1)) # µm, minor lateral semi-axis
    nucleus_elongation: ScalarDistribution = ScalarDistribution("folded_lognormal", (0.0, 0.1))   # ≥ 1
    nucleus_angle: ScalarDistribution = ScalarDistribution("uniform", (0.0, π))                   # rad, in the YX plane
    empty_radius: float = 2.5             # µm, virtual site radius of an empty cell
    gap: float = 1.0                      # µm, minimum gap between site spheres
    reach: ScalarDistribution = ScalarDistribution("lognormal", (log(3.0), 0.15))                 # µm, territory beyond the site sphere
    max_proposals: int = 10_000           # per seed and per molecule
```

### Nuclei

Nucleus `n` is an ellipsoid with centre `c_n` (µm), axial semi-axis `a_z`, lateral
semi-axes `a_l · e` and `a_l`, rotated by `θ` in the YX plane, with the convention of the
amplicon kernel ([synthetic specification](amplicon-scene.md), stage 7): with `d = σ − c_n`,
`u = cos θ · d_y + sin θ · d_x`, `v = −sin θ · d_y + cos θ · d_x`, the normalized radius is
`ρ_n(σ) = sqrt((d_z / a_z)² + (u / (a_l e))² + (v / a_l)²)`, and the nucleus is
`ρ_n ≤ 1`. Its **site sphere** has radius `R_n = max(a_z, a_l · e)`, which contains it.
Sizes are drawn before placement, each from its own stream. In the cellular scene of
[synthetic cellular scene](cellular-scene.md) a nucleus is a union of ellipsoids, lobe 1 optionally
perturbed by spherical harmonics, and its site sphere bounds every lobe; in the
real-appearance scene a nucleus is a placed real mask.

**Placement (random sequential addition).** Seeds are placed in order: ordinary cells,
then binucleate cells, then empty cells. Each proposal draws a centre uniformly in `Ω`
from the seed's own `sample.cell.placement` stream; it is accepted when its site spheres
keep a gap of at least `gap` to every accepted sphere (`‖c_i − c_j‖ ≥ R_i + R_j + gap`).
A binucleate seed draws a centre `c` and an angle `φ ~ U[0, π)` from the same stream and
places its two nuclei at `c ± (d/2)(0, sin φ, cos φ)`, with
`d = R_1 + R_2 + gap`, both tested. An empty seed is one virtual sphere of radius
`empty_radius`. A seed with no accepted proposal after `max_proposals` raises
`ValueError` naming it: nothing is dropped silently. Because acceptance depends on the
seeds before it, changing one seed's size can move later seeds; streams stay separate,
outcomes do not. The cellular scene of [synthetic cellular scene](cellular-scene.md) replaces this rule
when its packing settings are given: the kind of each seed may be drawn, a proposal may be
placed at its contact distance against an accepted site, it is accepted by a shape-aware
contact test that lets site spheres overlap, and with `on_crowded="stop"` placement ends at
the first seed that cannot be placed, recorded in `context.json`, and raises `ValueError`
only below a stated share of the target density; the real-appearance scene places objects
by PL2 and stops as 'When placement stops' states. Nothing is dropped silently there
either.

### Cell territories and the label-generation rule

**Nucleus label.** `N(σ) = n` where `σ ∈ Ω`, `ρ_n(σ) ≤ 1` and a site of `n` is `j*`, else 0
(with disjoint site spheres every point with `ρ_n ≤ 1` has `j*` a site of `n`, so this is
the rule above; the cellular scene of [synthetic cellular scene](cellular-scene.md) needs the condition,
because its site spheres may overlap); like `C`, it is 0 outside `Ω`, where a nucleus near
the edge of `Ω` may extend. Nuclei never overlap: a point carries only the nucleus of its
power-nearest site. Here the site spheres keep a positive gap, so nuclei do not touch; the
cellular scene of [synthetic cellular scene](cellular-scene.md) lets site spheres overlap and nuclei touch
on the faces of their power cells; in the real-appearance scene a point carries the nucleus
of the first placed object that holds it, and on every nucleus voxel the cell label is that
nucleus's cell.

**Cell label (rule L1 below).** With `pow_j(σ) = ‖σ − c_j‖² − R_j²` over all sites `j`, let
`j*` be the site of smallest `pow_j(σ)` (ties to the smaller site index). Then
`C(σ) = cell(j*)` when `σ ∈ Ω` and `‖σ − c_{j*}‖ ≤ R_{j*} + reach(cell(j*))`, else 0 (the
cellular scene adds two other outer limits: a grown nucleus, and none). The two
sites of a binucleate cell map to one cell, so its territory is the union of their power
cells. A nucleus is labelled only where its own site is the power-nearest one and its shape
function is at most 1, and its shape lies within its territory's outer limit, so **every
nucleus lies inside its own cell** and `N(σ) = n` implies `C(σ) = cell(n)`; here the site
spheres are disjoint, so no nucleus is cut by a power face.

Both are point functions of the sample position. They are what "the label at a position"
means everywhere on this page, and they are evaluated at voxel centres only to fill label
volumes.

| Option | Rule | Cost | Consequence |
| --- | --- | --- | --- |
| **L1. Ellipsoidal nuclei, power-diagram territories with a reach (recommended)** | As above. | Brute force over sites: one `(points × sites)` power-distance evaluation in blocks of 2¹⁸ points; about 4.3·10⁷ point-site pairs for the `slow` preset. Closed form, exact, no new dependency. | A label is defined at every point, so labels at fractional offsets need no resampling; cells have flat shared faces and rounded outer edges; a nucleus is always inside its cell. |
| L2. Ellipsoidal nuclei, Voronoi territories of the centres | `j*` by `‖σ − c_j‖` instead of the power distance | As L1. | A large nucleus beside a small one can cross the bisector, so the gap rule must grow to `2 · max R`, which wastes space; otherwise the same. |
| L3. Raster on the sample grid, then `expand_labels` | Rasterize the nuclei on the sample grid and grow territories with the segmentation's `expand_labels` | Imports `starfinder.segmentation` into the generator; cost of one distance transform of the sample grid. | Labels exist only at sample-grid voxels: a FOV at a fractional offset would need a resampled label image, which breaks "the per-FOV labels are the sample labels restricted to the FOV", and the truth would depend on the code it is meant to check. |

% L3: the §2.9 `expand_labels`

**Recommendation: L1.** It is the only option whose labels are exact at every FOV offset,
it keeps the generator independent of the segmentation code it validates, and its
guarantees (one cell per point, nucleus inside cell) are proved by the construction
rather than tested after the fact.

**Optional cases.** `empty` (default 0) gives cells without a nucleus, which assignment
reports as `excluded_no_nucleus`; `binucleate` (default 1 in the library, as in the tiny
preset) gives cells with two nuclei, which assignment counts with `n_nuclei` 2. Both are
ordinary seeds otherwise: the same territory rule, molecules and truth.

% which §2.9 assign reports; which §2.9 counts

### Label volumes

* **Sample level.** `labels/sample_nuclei.tif` and `labels/sample_cells.tif`: uint32 ZYX
  on the sample grid, `N` and `C` evaluated at each voxel centre; 0 is background; values
  are the nucleus and cell IDs (`1, 2, …` in seed order; nuclei numbered in order of
  their seeds, two per binucleate seed). An object that contains no voxel centre of a
  grid has no value in that grid's image (identity 2).
* **Per FOV.** `fovs/<fov_id>/labels/nuclei.tif` and `cells.tif`: uint32 ZYX on the FOV's
  reference grid, `N` and `C` evaluated at `true_f + p ⊙ s` for every index `p`, written
  with `save_volume` and the FOV's reference `ImageMetadata`. A cell at a FOV border is
  partial: only the voxels inside the grid carry its value. Values are the sample IDs; no
  relabelling.

The label dtype and geometry follow the label-image contract (uint32, option L1 of
[segmentation-contract](../segmentation.md); `volume` for Z>1, `plane` for Z=1).

### Stain

```python
@dataclass(frozen=True)
class StainConfig:
    channel_label: str = "DAPI"
    brightness: ScalarDistribution = ScalarDistribution("lognormal", (log(200), 0.2))  # intensity, per nucleus
    edge_width: float = 0.5                                  # µm, Gaussian edge (w)
    background: BackgroundConfig = BackgroundConfig(baseline_enabled=True, baseline=[[100.0]])
    noise: NoiseConfig = NoiseConfig(dependent_enabled=True, model="poisson", alpha=1.0,
                                     independent_enabled=True, sigma=3.0)
```

The stain is one ZYX image per FOV, in the template's dtype, saved as
`fovs/<fov_id>/stain.ome.tif` with the FOV's reference metadata. It is the
`reference_stain` input of `stitch` ([stitching-contract](../../../docs/stitching-contract.md), "Input image") and the
nuclear image of the cellular scene. The cellular scene of [synthetic cellular scene](cellular-scene.md)
renders several stains per FOV, each in its own output dtype; its first nuclear stain of
the reference round takes this role.

| Option | Pre-noise signal | Cost | Consequence |
| --- | --- | --- | --- |
| **St1. Analytic soft-edged ellipsoid per nucleus (recommended)** | `S(σ) = Σ_n A_n · ½ erfc((ρ_n(σ) − 1) · r̄_n / (√2 · w))`, `r̄_n` the geometric mean of the semi-axes, `w = edge_width`; exactly 0 where `ρ_n > 1 + 4w / min(a)`. Evaluated at each FOV voxel centre `true_f + p ⊙ s`, then the existing observation stages (tissue background, baseline, dependent and read noise, optional correlated noise, cast; stages 8 to 11 of [synthetic specification](amplicon-scene.md)) with the stain's own streams. | One new closed-form kernel (`scipy.special.erfc`, locked) over each nucleus's bounding box; per FOV a few nuclei times their box voxels. | The stain plateau coincides with the nucleus label up to the edge width, so a threshold of the stain reproduces the labels to within the edge width for nuclei much larger than it; exact at fractional offsets. |
| St2. One Gaussian per nucleus through the amplicon kernel | Each nucleus is a formed molecule with `sz = a_z / 2`, `sl = a_l / 2`, elongation and angle of the nucleus, rendered by stage 7 into a one-channel image, then the same observation stages | No new kernel at all. | Nuclei are blobs without an edge; the label boundary is not visible in the stain, so stain-derived masks disagree with the truth by construction. Enough for stitching, not for the processing report's measurement QC. |
| St3. Partial-volume raster and blur per FOV | Average the indicator of each nucleus over `k³` sub-voxel points of every FOV voxel, then convolve with a Gaussian on a grid padded by the kernel radius | `k³` evaluations per voxel (27 at `k = 3`) and one padded convolution per FOV. | The closest to an optical image; exact at fractional offsets because the raster is per FOV, but the blur's padding must reach beyond the FOV and costs more than the rest of the stain. |

% St2: not for §2.11 measurement QC

**Recommendation: St1.** It renders the stain from the same nuclei as the labels, with
one closed-form expression evaluated at voxel centres, and it reuses the observation
model for everything after the signal, which is what "through the existing observation
model" requires. St2 is the fallback if a new kernel is unwanted; it serves stitching but
not measurement QC.

## Cell truth for molecules

```python
@dataclass(frozen=True)
class SampleSceneConfig:
    fov: AmpliconSceneConfig = AmpliconSceneConfig(     # template: dtype, laws, readout, background, noise,
        dataset_version="sample-scene-v1", sample_id="sample",   # geometry, seed, split, scene_key, sample_id
        scene_key="sample-scene-v1", shape_zyx=(5, 36, 36), dtype="float32", accumulation="float64")
    acquisition: AcquisitionConfig = AcquisitionConfig()
    cells: CellsConfig | None = CellsConfig()   # None: no cellular scene
    stain: StainConfig | None = StainConfig()                   # None: no stain image
    molecules_per_cell: float = 8.0      # Poisson mean per cell
    ambient_density: float = 0.002       # molecules per µm³ of Ω
    max_count: int = 100_000             # guard on the number of molecules generated in Ω
```

The template's `shape_zyx` must equal `acquisition.fov_shape_zyx`; its population fields
(`count`, `density`, `coordinates`, `amplicon_ids`, `gene_ids`, non-uniform `placement`)
must be at their defaults, because the sample scene owns the population (`ValueError`
otherwise). Its property laws (`brightness`, widths, `elongation`, `angle`) and
`abundances` are used for every generated molecule. In the cellular scene of
[synthetic cellular scene](cellular-scene.md), a molecule's gene is drawn from its cell type's abundance
(an ambient molecule's from the ambient abundance), a cell's count is Poisson with a mean
drawn per cell, and a cell molecule is placed in its nucleus with a set probability, else
in the rest of its territory.

**Populations.**

* **Cell molecules.** Cell `k` receives `Poisson(molecules_per_cell)` molecules
  (stream `sample.molecule.count`, entity the cell). Each draws proposals uniformly in the
  box of its cell's site spheres grown by the reach, clipped to `Ω`, from its own
  `sample.molecule.placement` stream, and is accepted where `C(σ) = k`; after
  `max_proposals` without acceptance it raises `ValueError`.
* **Ambient molecules.** `Poisson(ambient_density · |Ω|)` molecules (entity `ambient`),
  uniform in `Ω`, with no rejection: they land in cells or outside every cell as the
  geometry decides.
* Without a cellular scene (`cells=None`), only the ambient population exists and no
  label or cell truth is produced.
* Both populations are generated throughout `Ω`, so the halo is filled at the same
  density as `K`. Sample membership is then decided by position ("Sample membership"); it
  does not change how anything is drawn.

**Truth.** The true cell of a molecule is `C(σ)` at its own continuous sample position
(the reference-round position), not at a voxel: `true_cell_id = C(σ)`, null where
`C(σ) = 0`. A molecule outside every cell is **kept** with a null cell. Likewise
`true_nucleus_id = N(σ)`, null where `N(σ) = 0`, and `compartment` is `nucleus` where
`N > 0`, `cytoplasm` where `N = 0 < C`, and `outside` where `C = 0`. How these null IDs
are compared with the value 0 of a label image is stated in "Per-FOV truth" ("Sampled
labels and nullable truth IDs"). The `population` column (`cell`, `ambient`)
records how it was placed; it is not the truth: an ambient molecule inside a cell
belongs to that cell. Accordingly, `n_molecules` of `cells.csv` is the number of molecules
whose `true_cell_id` is the cell: every row of `molecules.csv` with that `true_cell_id`,
sample and halo molecules alike and whatever their `population`. A cell's own generated
molecules always count, because each is accepted only where `C(σ) = k`, and so do the
ambient molecules that land in it; it is not the cell's Poisson draw. `in_sample` (on
`molecules.csv`, `cells.csv` and `nuclei.csv`) records sample membership, and `true_cell_in_sample` on `molecules.csv` records the
membership of the molecule's true cell (null without one).

IDs: `c<k>-m<j>` for the `j`-th molecule of cell `k` and `a-m<j>` for ambient molecules
(`j` from 0), so adding molecules to one cell changes no other molecule's draws. Each
molecule keeps its ID in every FOV that renders it (the FOV namespace differs).

## Per-FOV truth

* **Label volumes cut to each FOV** ("Label volumes" above), in which a cell at a border is
  partial.
* **`cell_fovs`**, one row per (cell, FOV) where the FOV's cell label holds the cell or
  its anchor is in the FOV's footprint: `cell_id`, `fov_id`, `in_sample` (the cell's),
  `anchor_in_footprint`, `n_voxels` (in that FOV, possibly 0), `fov_edge` (the cell
  touches the first or last Y or X index of the FOV grid) and `z_edge` (first or last Z
  index, for Z>1). The two edge flags are the same tests as the `fov_edge` and `z_edge`
  flags of [assembly-contract](../../../docs/assembly-contract.md), computed on the truth labels. A cell with
  `n_voxels > 0` that touches no edge of a FOV is **complete** in it by this label test,
  which is how assembly's flags judge it. A thin part of its territory that lies between
  voxel centres can still cross the FOV's footprint.
* **`molecule_fovs`**, one row per (molecule, FOV) rendered in that FOV:
  * `amplicon_id`, `fov_id` and `in_sample` (the molecule's);
  * `z`, `y`, `x`: the reference-round index position `p`;
  * `in_footprint`: `−0.5 ≤ p < n − 0.5` on every axis;
  * `fov_cell_label` and `fov_nucleus_label` (uint32 or null): the value of the FOV's
    cell and nucleus label image at the voxel `floor(p + 0.5)`, which is what assignment
    samples. The value 0 is kept as 0: the voxel is in the grid and in no cell (no
    nucleus). Null means that nothing was sampled: the voxel is outside the grid, or the
    scene has no cellular scene and so no label image. With a cellular scene,
    `fov_cell_label` is null exactly on the rows without `in_footprint`, because
    `in_footprint` is the in-grid test of the sampling rule ("Terms");
  * `label_agrees` (boolean or null): `fov_cell_label` compared with `true_cell_id` by the
    rule below; null where `fov_cell_label` is null;
  * `csv_row`: its row in the FOV's molecule CSV ("The path that needs no segmenter").
* **Sampled labels and nullable truth IDs.** A label image stores "in no cell" as 0; the
  truth tables store it as null (`true_cell_id` is null where `C(σ) = 0`, and
  `true_nucleus_id` where `N(σ) = 0`, "Cell truth for molecules"). A sampled value `v` is
  compared with a truth ID by this rule, the same for the cell and the nucleus label:

  | Sampled value `v` | Meaning | Assignment (without expansion) | Compared with the truth ID |
  | --- | --- | --- | --- |
  | `k > 0` | in the grid, in cell (nucleus) `k` | cell label: `assigned` or `excluded_cell` with `cell_id` `k`; nucleus label: `nucleus_id` `k` | agrees exactly when the truth ID is `k`; a null truth ID disagrees |
  | `0` | in the grid, in no cell (no nucleus) | cell label: `unassigned`, `cell_id` null; nucleus label: `nucleus_id` 0 | agrees exactly when the truth ID is null |
  | null | not sampled: outside the grid | `outside_grid`, `cell_id` and `nucleus_id` null | not compared: no agreement value |

  Equivalently, where `v` is not null, the two agree exactly when `v` equals the truth ID
  read with null as 0, that is `v == C(σ)` for the cell label and `v == N(σ)` for the
  nucleus label. A null `v` is never equal to a null truth ID: it says that nothing was
  sampled, not that the molecule is in no cell. The statuses, the sampled voxel and the
  in-grid test are those of [assignment-contract](../assignment.md) ("Coordinate sampling", "Direct
  assignment and statuses"), which this page cites and does not redefine; there, too, a
  null `cell_id` stands for both `unassigned` and `outside_grid`, and the status tells
  them apart. `label_agrees` records the rule for the cell label only; `molecule_fovs`
  has no agreement column for the nucleus label, and a test that needs one computes it
  from `fov_nucleus_label` and `true_nucleus_id` by the same rule.
* **Sample-level tables that say which FOVs see each object.** A molecule is **seen** by
  a FOV when it is `in_footprint` there. A cell is **seen** by a FOV when its anchor is
  in that FOV's footprint. `molecules.csv` and `cells.csv` have `fov_ids` (`;`-joined,
  table order) and `n_fovs` by these rules, for sample and halo content alike.
  `cells.csv` also has `label_fov_ids`: the FOVs whose label image holds the cell.

% which is what §2.9 assign samples; the column "§2.9 assign (without expansion)"

### Identities a test can check

With `G` every generated molecule (sample and halo), `M = {m ∈ G : σ_m ∈ K}` the sample
molecules, `M_f` the molecules of `G` seen by FOV `f`, `n_fovs(m) = #{f : σ_m ∈
footprint_f}`, and `L` the label functions. Without a cellular scene (`cells=None`) only
identity 3 applies: there are no label images, no `nuclei.csv` and no cell rows, and
`true_cell_id`, `true_nucleus_id`, `compartment` and the label columns of
`molecule_fovs` are null.

1. **One cell or none.** `molecules.csv` has one row per molecule, keyed by `amplicon_id`;
   `true_cell_id` is a single value or null, and every non-null value is a `cell_id` of
   `cells.csv`. A non-null `true_nucleus_id` belongs to a nucleus of that cell
   (`nuclei.csv` `cell_id`).
2. **Per-FOV labels are the sample labels restricted to the FOV.** For every FOV and every
   index `p`, `fov_labels[f][t][p] == L_t(true_f + p ⊙ s)` for `t` in nucleus and cell, and
   `sample_labels[t][q] == L_t(origin_sample + q ⊙ s)`: both volumes are the same label
   functions sampled at their own voxel centres.
   * **IDs.** Every non-zero value of a FOV or sample nucleus image is a `nucleus_id` of
     `nuclei.csv`, and every non-zero value of a FOV or sample cell image is a `cell_id`
     of `cells.csv`.
   * **Presence differs between grids.** The two grids sample different points, so an ID
     can occur in a FOV image and not in the sample image, or the other way round. A
     nucleus of radius 0.1 µm can contain one FOV voxel centre and no sample voxel centre.
     The same holds between two overlapping FOVs at a fractional relative offset. Where
     an object has voxels is recorded as follows:
     * for cells in FOV images, `cell_fovs` `n_voxels` and `cells.csv` `label_fov_ids`;
     * for cells in the sample image, `cells.csv` `n_voxels` (0 when no sample voxel
       centre falls in the cell);
     * nuclei have no such column, so a test reads nucleus presence from the images
       themselves.
   * **Aligned crop.** Let `δ_f = (true_f − origin_sample) ⊘ s` be FOV `f`'s offset on the
     sample grid, in voxels. When `δ_f` is a whole number on every axis, the FOV volume
     equals the crop `sample_labels[t][δ_f : δ_f + n]`, which lies inside the sample grid,
     except at voxels whose centre lies within 1e-9 µm of a label boundary. The test
     reports the number of those voxels and expects 0.
     * Since `origin_sample = −J ⊙ s`, `δ_f = nominal_f ⊘ s + j_f + J`. For a nominal step
       that is a whole number of voxels (24 in `sample-tiny-v1`, 72 in `sample-small-v1`),
       the condition holds exactly when `j_{f,a} + J_a` is a whole number on every axis.
       Within the bound that means `j_{f,a} ∈ {−J_a, −J_a + 1, …}` up to `+J_a`.
     * Zero jitter is not aligned unless `J` is whole: on `sample-tiny-v1` it gives
       `δ = (0.4, 1.5, 1.5)` voxels for the first FOV.
     * With the morphology round of [synthetic cellular scene](cellular-scene.md), `Ω` grows by `T ⊙ s`
       on each side, so `origin_sample = −(J + T) ⊙ s` and
       `δ_f = nominal_f ⊘ s + j_f + J + T`; the condition then holds exactly when
       `j_{f,a} + J_a + T_a` is a whole number on every axis, and zero jitter is aligned
       only when `J + T` is whole. For the culture presets (`J_z = 0.8`, `T_z = 0.5`) the
       supplied jitter `j_z = −0.8` gives `δ_z = 0.5`, not 0; `j_z = −0.3` aligns.
     * Worked example on `sample-tiny-v1` (`J = (0.4, 1.5, 1.5)`): the supplied jitter
       `(−0.4, 0.5, −1.5)` voxels for `FOV_001` gives `δ = (0, 2, 0)` voxels. The FOV's
       5 × 36 × 36 volume then equals `sample_labels[t][0:5, 2:38, 0:36]`.
     * Two overlapping FOVs whose `δ` differ by a whole number of voxels agree on their
       common voxels in the same sense.
3. **The FOVs cover the sample with the stated multiplicity.** For the sample molecules
   `M`, whatever the drawn jitter:
   * **coverage**: every `m ∈ M` has `n_fovs(m) ≥ 1`, so `∪_f (M_f ∩ M) = M` (proof:
     "Covered region");
   * **multiplicity**: `Σ_f |M_f ∩ M| = Σ_{m ∈ M} n_fovs(m)`;
   * **nominal multiplicity**: a sample molecule farther than `J_a · s_a` on every axis
     from every nominal footprint boundary has the nominal multiplicity, 1 outside the
     overlaps, 2 in an overlap of two neighbours and 4 in a corner overlap of four.

   For the halo molecules `G \ M`, only the counting identity
   `Σ_f |M_f \ M| = Σ_{m ∈ G \ M} n_fovs(m)` is claimed. A halo molecule may be seen by
   no FOV (`n_fovs = 0`), and every molecule with `n_fovs = 0` is a halo molecule. A test
   checks all of these on `molecules.csv` and `molecule_fovs.csv`, splitting by
   `in_sample`. It also checks that `in_sample` equals `σ ∈ K` recomputed from the bounds
   in `sample.json`, and is unchanged when the same scene is generated with another
   jitter.
4. **Cells.** For the sample cells, whatever the drawn jitter:
   * every sample cell has `n_fovs ≥ 1`, because its anchor is a point of `K`;
   * the number of `cell_fovs` rows with `anchor_in_footprint` and `in_sample` equals
     `Σ_{sample cells} n_fovs`;
   * the nominal multiplicity of identity 3 holds for anchors.

   For every cell, sample or halo:
   * `cells.csv` `fov_ids` equals the FOVs of its `cell_fovs` rows with
     `anchor_in_footprint`;
   * `label_fov_ids` equals the FOVs of its rows with `n_voxels > 0`.

   A sample cell is not guaranteed to have label voxels in a FOV that sees its anchor:
   the label image is sampled at voxel centres. The validation reads `label_fov_ids` for
   that.
5. **Sampling agreement is measured, not assumed.** On rows with `in_footprint`,
   `fov_cell_label` is `C(v_m)` at the centre `v_m = true_f + floor(p + 0.5) ⊙ s` of the
   sampled voxel (identity 2), and the truth is `C(σ_m)`, `true_cell_id` read with null
   as 0 ("Sampled labels and nullable truth IDs"). `label_agrees` is false exactly when
   `C(σ_m) ≠ C(v_m)`: the molecule and that centre lie in different cells, or one of them
   lies in a cell and the other in none. When both lie outside every cell, the row has
   `fov_cell_label` 0 and a null `true_cell_id`, and `label_agrees` is true. The number of
   rows with `label_agrees` false is reported per FOV. On the other rows (rendered in the
   margin, outside the grid) nothing is sampled: `fov_cell_label` and `label_agrees` are
   null, and these rows are excluded from the count. It separates the assignment's
   sampling rule from assignment and assembly errors in the validation.

% the §2.9 sampling rule

## The path that needs no segmenter

Per FOV, the truth labels enter through the segmentation's mask import, so that assignment and
assembly can be compared with sample-level truth without running a segmenter:

% the §2.9 mask import

1. **Labels.** `import_labels(fovs/<f>/labels/cells.tif, grid=ReferenceGrid(n,
   fov_metadata[f], source="declared"), target="cell",
   label_namespace=["<dataset>", "<sample>", "<f>", null, "truth"])`, and the same for
   `nuclei.tif` with `target="nucleus"`. The files carry `starfinder_metadata` equal to the
   grid's, so the import's metadata check passes; `relabel=False` keeps the sample IDs,
   so an assigned molecule's `cell_id` is directly comparable with `true_cell_id`. Assign
   writes a null `cell_id` for both `unassigned` and `outside_grid`, so a comparison of
   those molecules reads the status: `unassigned` is the sampled value 0 and
   `outside_grid` is a molecule not sampled, and each compares with `true_cell_id` by the
   rule of "Per-FOV truth" ("Sampled labels and nullable truth IDs").
2. **Molecules.** `fovs/<f>/molecules.csv` is the legacy one-based `x, y, z, gene` CSV of
   the reference-round positions of the molecules that FOV renders (`p + 1`, round-trip
   float precision), in `molecule_fovs` order. `import_legacy_molecules_csv(path,
   spot_namespace=["<dataset>", "<sample>", "<f>", null], genes=codebook.genes)` reads it
   unchanged; its `spot_id` `csv:<row>` maps to `amplicon_id` through `csv_row`. The
   synthetic package writes the CSV and never imports `starfinder.assignment`. This is
   a different, one-based file from the zero-based `fovs/<fov_id>/molecules.csv` that
   `FOV.export_spots` writes under an output root ([migration](../../../docs/migration.md)).
3. **Assign and assemble.** `assign_molecules` per FOV, then `assemble(placement,
   inputs)`: with the placement from `stitch` on the stain images (the full chain), or with
   one built from the placement truth by `placement_from_table` (assembly alone; see
   below). Retained cells and molecules are
   then compared with `cells.csv`, `molecules.csv`, `cell_fovs.csv` and
   `molecule_fovs.csv` by cell ID and amplicon ID. The sample truth is the rows with
   `in_sample` true. Halo rows are observations at the edge of the acquisition that the
   FOVs may or may not see, and they are reported separately.

% (W-562; [migration](../../../docs/migration.md))

The validation itself (which comparisons, which numbers) belongs to the validation
design; this page fixes only the inputs and their identities. The placement from the
placement truth is built by `placement_from_table` of [stitching-contract](../../../docs/stitching-contract.md)
("Supplied placement") from the scene's nominal positions and its true origins: the tile
position table is `positions.csv` (read by `positions_from_table`), whose `nominal_z/y/x`
are the nominal grid's; `origins` maps each `fov_id` to the `(true_origin_z,
true_origin_y, true_origin_x)` of its row of `placement_truth.csv`; and `grids` holds each
FOV's reference grid. Every FOV is then `metadata_only` at its true origin, flagged
`supplied`, with `correction_*` equal to the `true_correction_*` of `placement_truth.csv`
within 1e-9 µm (both are `true_origin − nominal`), and the review is `required` with the reason
`supplied_placement`, so the caller approves it with `approve_placement` before `assemble`
accepts it. Because the nominal grid puts every FOV at nominal z 0 ("Nominal grid"), a
scene with Z jitter then still has one `nominal_z`, and its truth placement passes check 8
of [assembly-contract](../../../docs/assembly-contract.md) ("Checks the entry applies"); the jitter in Z moves the
FOVs' sample z, not ownership (option Z1). One need of this
path is still not met by the current contracts and is recorded in the worker notes: a
molecule input that keeps the sample IDs.

## Random streams and seeds

Every new draw uses the descriptor of [synthetic specification](amplicon-scene.md), unchanged:
`["starfinder.synthetic/1", split, seed, scene_key, component, entity, round_label, channel_label]`,
with the template's `split` and `seed`.

* **Sample-level draws** use the template's `scene_key` and new components, all prefixed
  `sample.`, so they can never equal an existing descriptor:

  | Component | Entity | Draws |
  | --- | --- | --- |
  | `sample.jitter` | FOV ID | `u_z, u_y, u_x` in order (J1) |
  | `sample.nucleus.axial`, `sample.nucleus.lateral`, `sample.nucleus.elongation`, `sample.nucleus.angle` | `nucleus-<n>` | one value each |
  | `sample.cell.placement` | `cell-<k>` | proposals, z, y, x per proposal; then `φ` for a binucleate seed |
  | `sample.cell.reach` | `cell-<k>` | one value |
  | `sample.molecule.count` | `cell-<k>` or `ambient` | one Poisson draw |
  | `sample.molecule.placement` | molecule ID | proposals, z, y, x per proposal |
  | `sample.identity`, `sample.brightness`, `sample.width.axial`, `sample.width.lateral`, `sample.elongation`, `sample.angle` | molecule ID | one value each, with the template's laws |
  | `sample.background.count`, `sample.background.placement`, `sample.background.width`, `sample.background.brightness` | as for texture today | the sample-frame texture of the amplicon channels |
  | `sample.stain.brightness` | `nucleus-<n>` | one value |
  | `sample.stain.background.*` | as above | the stain's sample-frame texture |

* **Per-FOV draws** use the scene key `[scene_key, fov_id]` of `generate_amplicon_dataset`: the
  existing `geometry.*` and `noise.*` components for the amplicon rounds, and
  `stain.noise.dependent`, `stain.noise.independent` and `stain.noise.correlated` with the
  stain's channel label for the stain.
* **Consequences.** Adding a FOV leaves the draws of the other FOVs as they were: their
  jitter (`sample.jitter`, entity the FOV ID) and their per-FOV draws (scene key
  `[scene_key, fov_id]`). Nothing more is claimed for an added FOV: the domain depends on
  the grid, so adding a column changes `Ω` and `K`, which changes placement outcomes and
  membership, and with them what the other FOVs image. Changing the jitter law or its
  draws moves no cell or molecule and changes no membership; no stream entity depends on membership;
  changing one
  molecule's law changes only that molecule; changing a nucleus size can move later seeds
  (sequential placement). Repeatability is that of the formed model: exact for a pinned
  NumPy build on one CPU.
* **Seeds.** The template's root seed, default 42; `split` `development`. Each preset
  fixes its scene key to its name.

## Presets and resources

`SAMPLE_PRESETS` (version `sample-presets-v1`), separate from `SCENE_PRESETS`. Both use
`development_codebook("balanced")` (16 genes, four rounds, four channels), seed 42, split
`development`, float32 output and the scene key of their name; stochastic readout effects
are off (required), and the per-round geometry is a translation only.

| | `sample-tiny-v1` (routine tier) | `sample-small-v1` (`slow` tier) |
| --- | --- | --- |
| Grid | 2 × 2 FOVs | 3 × 3 FOVs |
| FOV shape (voxels) | 5 × 36 × 36 | 14 × 96 × 96 |
| Spacing (µm) | 1.0, 0.5, 0.5 | 0.346255, 0.194398, 0.194398 (the raw LN calibration, the stitching input inventory's `inputs.md`) |
| Overlap | 1/3 (12 voxels) | 0.25 (24 voxels) |
| Jitter bound (voxels, J1) | 0.4, 1.5, 1.5 | 0.8, 4.5, 4.5 |
| Per-round translation bound (voxels) | 0, 1, 1 | 1, 2, 2 |
| Sample grid (voxels) | 6 × 63 × 63 | 16 × 249 × 249 |
| `Ω` (µm, Z; Y and X) | 5.8; 31.5 | 5.40; 48.4 |
| `K` (µm, Z; Y and X) | 4.2; 28.5 | 4.29; 44.9 |
| Share of `Ω`'s volume in `K` (the expected share of molecules in the sample) | 0.59 | 0.68 |
| Seeds | 7 ordinary, 1 binucleate, 1 empty | 14 ordinary, 2 binucleate, 2 empty |
| Nucleus radii, median (µm) | axial 1.5, lateral 2.5 | axial 2.0, lateral 3.0 |
| Reach, median (µm) | 3.0 | 2.5 |
| Molecules | 8 per cell, ambient 0.002 µm⁻³ (about 80 generated in `Ω`, about 50 sample molecules) | 30 per cell, ambient 0.002 µm⁻³ (about 560 generated in `Ω`, about 380 sample molecules) |
| Amplicon appearance | constant A 100, sz = sl = 1, no readout effect, no background, Gaussian read noise σ 1; float64 accumulation (for the oracle identities) | the benchmark appearance of `benchmark-presets-v1` (`_appearance` at float32: lognormal brightness and widths, trend 0.95, 5 % crosstalk, baseline, gradient and regions placed by fractions of the sample grid, Poisson and read noise); float32 accumulation |
| Stain | `StainConfig()` | `StainConfig()` |
| Voxel values rendered | 440,640 (4 FOVs × 6,480 voxels × 17 images) | 19,740,672 (9 × 129,024 × 17) |

% W-323 `inputs.md`

**Resource estimate** (from array sizes and the existing peak formula
`_estimate_peak_bytes` of `_presets.py`; nothing here is a measurement):

* **`sample-tiny-v1`, inside the routine test bound.** Every array is within the default
  bound (at most 32 × 64 × 64 voxels, four channels, four rounds): FOV images 5 × 36 × 36
  × 4, the sample label grid 6 × 63 × 63. Working memory per FOV and round is at most
  132 MiB by the peak formula, almost all of it the formula's fixed noise-chunk term;
  labels and tables are below 1 MiB. It renders 440,640 voxel values, fewer than the
  524,288 of one `calibrated-development-v1` scene (8 × 64 × 64, four rounds, four
  channels), which `test_calibrated_generation_stays_within_bounds` generates in the
  default tier, so it is expected below the 5 s `slow` threshold on one CPU. The label
  functions add `(4 × 6,480 + 23,814) × 10` point-site evaluations, about 5·10⁵. Saved,
  it is about 2.2 MB (images 1.7 MB, stain 0.1 MB, labels 0.4 MB). The
  implementation measures it with `--durations=0` and `/usr/bin/time -v`; if a test that
  generates it takes 5 s or more, that test is marked `slow`, and the routine tier uses
  the same grid with 3 × 36 × 36 FOVs.
* **`sample-small-v1`, inside the `slow` bound of the stitching exceptions.** Its sample grid
  of 16 × 249 × 249 is within the bound of one scene of at most 16 × 256 × 256 cut into at
  most a 3 × 3 grid of overlapping tiles, the synthetic cut-fixture bound of the batch
  (the stitching parity runs' fixture used one 16 × 256 × 256 scene and 3 × 3 tiles of
  14 × 120 × 120, their notes, section 2.5). Peak working memory per FOV and round: 153 MiB by the formula;
  label volumes 7.6 MiB (sample) plus 1 MiB per FOV; the power-diagram evaluation in
  blocks of 2¹⁸ points holds about 0.2 GiB of temporaries; all FOV images held in memory
  (without `on_fov`) are 71 MiB in float32. Estimated peak RSS below 1 GiB beside the
  interpreter and its imports, against the 4 GiB stop target, on one CPU and one thread.
  It renders 19.7 million voxel values, 0.59 of the 33.6 million of the benchmark `small`
  dataset that `conftest.py` generates once per session; the label functions add about
  4.3·10⁷ point-site evaluations. Saved, it is about 96 MB uncompressed (images 74 MB,
  stain 4.6 MB, labels 17 MB) plus the tables.
* **Not in either tier.** Larger grids, LN-sized FOVs (50 × 1496 × 1496) and whole-sample
  scenes are outside this addition; the `slow` context presets of
  [synthetic cellular scene](cellular-scene.md) use 3 × 3 FOVs of 16 × 256 × 256, under their own
  resource grant; the generator sets no upper bound, as today, and the caller owns memory
  and time.

% the §2.10 exceptions; the W-324 fixture (W-324 notes section 2.5)

## Truth schema

| Option | Where the truth goes | Cost | Effect on existing outputs |
| --- | --- | --- | --- |
| **V1. New sample-level files with their own version (recommended)** | Per FOV the unchanged `save_amplicon_scene` layout; beside it the sample tables and label images below, described by `sample.json` with `sample_truth_version` `"1"`. No existing table gains a column. | Consumers join the sample tables with the FOV tables by `fov_id` and `amplicon_id`; seven new files. | None: every existing file keeps its columns and bytes. |
| V2. New columns on the per-FOV tables | `formed` gains `sample_z/y/x`, `true_cell_id`, `compartment`; `round_truth` gains `in_footprint` | Fewer files. | The columns would exist only for sample scenes, so the schema of `formed.csv` would depend on how it was made; or they exist always and every existing `formed.csv` changes. |
| V3. `ground_truth.json` version 3 | Extend the historical MATLAB-compatible record with the sample keys | One file for MATLAB readers. | `HISTORICAL_TRUTH_VERSION` would move from `"2.0"`, which changes the files `starfinder synthetic generate` writes and `test_benchmark_synthetic.py` checks. |

**Recommendation: V1.** It is the only option that changes no existing file, keeps one
schema per table, and versions the addition independently of generator version 6.

**Layout** written by `save_sample_scene(scene, directory)` into a new or empty directory
(nothing is overwritten, as `save_amplicon_scene`):

```
<directory>/
  sample.json                   # record: sample_truth_version "1", sample_preset_version, generator versions,
                                #   requested and effective SampleSceneConfig, seed, stream scheme and
                                #   descriptors, sample grid, the bounds of Ω and K (µm), codebook,
                                #   SHA-256 of every file below
  positions.csv                 # tile position table: fov_id, nominal_z/y/x (µm)
  placement_truth.csv           # "Placement truth"
  nuclei.csv                    # nucleus_id, cell_id, in_sample, centre_z/y/x (µm), axial_radius,
                                #   lateral_radius (µm), elongation, angle (rad), site_radius (µm), stain_brightness
  cells.csv                     # cell_id, kind (ordinary, binucleate, empty), in_sample, anchor_z/y/x (µm),
                                #   n_nuclei, reach (µm), centroid_z/y/x (µm, of the sample-grid voxels; NaN when n_voxels is 0),
                                #   n_voxels (sample grid), n_molecules (the molecules whose true_cell_id
                                #   is the cell), fov_ids, n_fovs, label_fov_ids
  molecules.csv                 # amplicon_id, population, in_sample, gene_id, color_sequence, sample_z/y/x (µm),
                                #   A, sz, sl, e, theta, true_cell_id, true_cell_in_sample, true_nucleus_id,
                                #   compartment, fov_ids, n_fovs
  cell_fovs.csv                 # "Per-FOV truth": cell_id, fov_id, in_sample, anchor_in_footprint,
                                #   n_voxels, fov_edge, z_edge
  molecule_fovs.csv             # "Per-FOV truth": amplicon_id, fov_id, in_sample, z, y, x, in_footprint,
                                #   fov_cell_label, fov_nucleus_label, label_agrees, csv_row
  labels/sample_nuclei.tif, labels/sample_cells.tif
  fovs/<fov_id>/                # save_amplicon_scene layout: images/<round>.ome.tif, formed.csv,
                                #   round_truth.csv, signals.csv, provenance.json
  fovs/<fov_id>/stain.ome.tif
  fovs/<fov_id>/labels/nuclei.tif, fovs/<fov_id>/labels/cells.tif
  fovs/<fov_id>/molecules.csv   # legacy one-based x, y, z, gene (the no-segmenter path)
```

This scene file is one-based and differs from the zero-based `molecules.csv` that
`FOV.export_spots` writes.

% `FOV.export_spots` writes (W-562)

Units are in the column names' tables above: µm in the sample frame, voxel indices in
the FOV frame, radians for angles; IDs are strings except the uint32 label values
`cell_id` and `nucleus_id`. CSV floats use round-trip precision; empty cells are nulls.
`sample.json` names `SAMPLE_TRUTH_VERSION`; any change to a column or rule above bumps it.
A scene of [synthetic cellular scene](cellular-scene.md) writes the same files with the same columns,
made by the rules of that page (lobes, clipping, territories, molecules by compartment);
its `context.json` names `CELLULAR_TRUTH_VERSION`, which versions those rules, and
`sample.json` keeps `sample_truth_version` `"1"`; a real-appearance scene writes them with
the ellipsoid columns null, versioned by `APPEARANCE_TRUTH_VERSION` in `appearance.json`.

## The tests of the scene

* **Keeps**: every test named in "Opt-in, and what stays as it is", unchanged.
* The scene's own tests are in `src/python/test/test_synthetic_sample_scene.py`
  (marker `synthetic`; `slow` where measured at 5 s or more; no test reads the network share):
  the configuration rules, the nominal grid and placement truth, the rendering identities,
  the cellular identities, stream isolation, the single-plane case below, the saved files
  read without a segmenter and cross-process repeatability; among them is
  one `slow` test that generates `sample-small-v1` under `/usr/bin/time -v` and records its
  time and peak RSS, as `test_medium_cli_both_modes_time_and_peak_rss` does for the
  benchmark presets.
* **The single-plane case.** No preset has Z=1, and none is added. The test supplies the
  configuration it generates from, built in the test with `dataclasses.replace` from
  `sample_scene_preset("sample-tiny-v1")`: that configuration with
  `fov_shape_zyx` and the template's `shape_zyx` set to `(1, 36, 36)` and
  `jitter_max_zyx` to `(0.0, 1.5, 1.5)`, everything else unchanged (its per-round
  translation bound in Z is already 0). It asserts that generation succeeds, with no
  error; that the sample grid has one plane and every FOV image, stain and label image
  has the shape `(1, 36, 36)`; that every nucleus centre, cell anchor and molecule has
  z = 0, and every FOV's true offset in Z is 0; that no `cell_fovs` row has `z_edge`;
  that cellular identities 1 to 5 hold; that `import_labels` reads one FOV's saved
  labels as a `plane` label image; and that `SAMPLE_PRESETS` keeps its two keys. It runs
  in the routine tier, with fewer voxels than `sample-tiny-v1`.

## Exclusions

No implementation and no change to the generator, its presets or its tests; no microsim
or other new rendering dependency, no calibration to real tissue (the development targets
of [synthetic cellular scene](cellular-scene.md) are not one); no claim about segmentation accuracy, and
no segmenter run on this scene as a measure of its quality; no change to the segmentation, assignment, stitching or assembly contracts (the need of the
no-segmenter path that they still do not meet is listed in the worker notes); no rotation, illumination model, stage
drift (beyond a supplied jitter table) or cross-FOV cell matching.

% the §2.9 or §2.10 contracts
