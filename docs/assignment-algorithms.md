# Assignment algorithm specification and §2.9 validation design

Status: Accepted (W-309, 2026-10-05, at 3550723)

This page specifies the numerical rules of §2.9 assignment (sampling, expansion, overlap
correspondence, compartment partition and count accounting), each with its parameters,
units, defaults, failure behavior and resource estimate. It then gives the task-group-6
engineering validation design for the whole of §2.9, segmentation and assignment, and the
plan for the bounded real examples. The entry, records and files are in
{doc}`assignment-contract`; the current behavior is in {doc}`assignment-baseline`; the
segmentation methods and label functions are specified in {doc}`segmentation-algorithms`.
Nothing here is implemented, and nothing here ranks a method, sets a default model or
makes an accuracy claim.

## Evidence

Assignment has no model and W-308 ran nothing but the golden test and its pin script, so
every resource figure below is a **code-derived estimate** (memory from array sizes and
dtypes, time from the operations' complexity), not a measurement. The whole-FOV reference
size is the W-306 estimate's 50×1496×1496 grid (111.9 million voxels; one `uint32` label
volume is 448 MB). Segmentation figures quoted in the real-example plan are W-306's
(`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-306/20261004T194036Z-74e12949`,
`tables/cost-summary.csv`; per-call wall time, one CPU thread unless stated, model loaded
before the call), with the qualifications listed in {doc}`segmentation-contract`
("Limitations of the evidence") and {doc}`segmentation-algorithms` ("How to read the
resource figures").

## What each element addresses

| Element | Problem addressed |
| --- | --- |
| Sampling | Turn a zero-based subpixel position into one voxel, with defined bounds, for integer and subpixel detectors alike |
| Expansion | Territories around nuclei or cells when molecules lie just outside the segmented outline, applied once |
| Overlap correspondence | Which nucleus belongs to which cell, from two independent label images, with every doubtful case flagged |
| Compartment partition | Nuclear and cytoplasmic counts that sum to the whole-cell counts, only where the correspondence supports them |
| Count accounting | One status per molecule and counts that sum, so nothing is lost between the molecule table and the matrix |

## Sampling

**Rule** (option P1 of the contract). Per sampled axis with size `n` and zero-based
coordinate `c` (float64): `i = floor(c + 0.5)`; the molecule is inside when `0 ≤ i < n` on
every axis of the molecule run's grid, else `outside_grid`. `plane` labels read the
territory at `y, x` (`z` is bounds-checked against the molecule grid but selects nothing);
`volume` and `extended` labels sample `z, y, x`. The territory value, the nucleus value and, when assign
expanded, the original territory value are read at the same voxel.

**Parameters.** None. Units: voxel indices of the reference grid.

**Failure behavior.** Non-finite coordinates are refused when the `MoleculeTable` is built
(`ValueError` naming the first row). Nothing else fails: off-grid molecules get a status.
`floor(c + 0.5)` is computed in float64; coordinates beyond ±2⁵² voxels lose integer
precision, which no image reaches.

**Resources** (code-derived). For `m` molecules: the float64 coordinates (24 bytes), the
int64 indices (24 bytes), a validity mask and the gathered values (4 to 8 bytes per label
image), about 70 bytes per molecule: 70 MB for 10⁶ molecules. One vectorized gather per
label image; time linear in `m`, independent of the grid size.

## Expansion

**Rule.** `expand_labels` of {doc}`segmentation-algorithms`, called once, by assign, on
the cell run's labels (the original territories) with an `ExpandLabelsConfig`
(`distance`, `unit`, `mode`). `planar` reproduces `reads_assignment.py` (scikit-image
`expand_labels` per Z plane); `volumetric` grows in 3D with the ZYX spacing. Labels are
never overwritten or renumbered, so every original territory is contained in its expanded
territory under the same value, and every nucleus inside its territory stays there. Both
label images are kept ({doc}`assignment-contract`, "Label expansion").

**Parameters.** `distance` (no default), `unit` `pixel` or `um`, `mode` `planar` or
`volumetric`. On a calibrated grid the distance is in µm; on an uncalibrated grid it must
be in pixels. The legacy translation is `planar`, `pixel`, `dilation_distance`, with
`AssignmentConfig.legacy_pixel_expansion=True` on its calibrated declared grid.

**Failure behavior.** A cell run whose record lists `expand_labels` is refused, with or
without an assign expansion, because its original mask was not kept (`ValueError`,
contract check 7). `unit="um"` on an uncalibrated grid raises; `unit="pixel"` on a
calibrated grid raises unless `legacy_pixel_expansion=True`. An empty label image stays empty; distance 0 changes nothing (W-307 probe).
Two successive planar expansions are not one: 4 then 2 pixels differs from 6 pixels by 585
voxels on the assignment golden fixture (`test_two_expansions_are_not_one`), which is why
the legacy pair cannot be folded silently.

**Resources** (code-derived; W-307's estimate for the function). `planar`: about 22 bytes
per pixel of one plane in temporaries (about 50 MB for a 1496² plane) plus the output
volume, 4 bytes per voxel (448 MB for the whole FOV). Keeping both territories doubles the
label memory (896 MB). `volumetric`: about 26 bytes per voxel (2.9 GB for the whole FOV),
which with the label volumes passes the 4 GiB stop target, so whole-FOV volumetric
expansion needs blocks or the §2.10 tiling.

## Overlap correspondence

**Rule.** Given the nucleus image `N` and the territory image `T` on one grid:

1. Nucleus sizes `s_n`: the count of each positive value of `N`.
2. Overlaps `o(n, c)`: the count of each pair `(N, T)` over voxels with `N > 0`
   (including `c = 0`, the background part).
3. Shares `f(n, c) = o(n, c) / s_n`; `c*` is the cell with the largest share, ties broken
   by the smaller `cell_id` (a tie can only occur at a share ≤ 0.5, where it cannot
   produce a match).
4. Status: `matched` to `c*` if `f(n, c*) > match_fraction`; else `ambiguous` if
   `Σ_{c>0} o(n, c) > 0`; else `no_cell`. `outside` if matched and
   `1 − f(n, c*) > outside_tolerance`.
5. Cell flags: `several_nuclei` (two or more matched nuclei), `ambiguous_nucleus` (an
   ambiguous nucleus has `o(n, c) > 0`), `nucleus_outside_cell` (a matched nucleus of
   `c` is `outside`), `foreign_nucleus` (`o(n, c) > 0` for a nucleus matched to another
   cell). Any of the last three makes the compartments of a `matched` or `ambiguous` cell
   `withheld`.
6. Cell correspondence: `matched` with at least one matched nucleus; else `ambiguous` with
   an ambiguous nucleus over it; else `no_nucleus`.

A supplied table replaces step 4's match only: a nucleus in the table is matched to its
row's cell; a nucleus absent from it is `ambiguous` if it overlaps a cell and `no_cell`
otherwise, as in step 4. Steps 1, 2, the `outside` test, 5 and 6 are unchanged, so a table
equal to the derived matches reproduces the derived result.

**Parameters.** `match_fraction` (dimensionless share of the nucleus's voxels, default
0.5, allowed [0.5, 1); strict `>`); `outside_tolerance` (dimensionless share, default 0.1,
allowed [0, 1); strict `>`, so a nucleus with exactly a tenth outside its cell is not
flagged). 0.5 is the smallest value that makes a match unique. 0.1 is contract option C2,
chosen on 2026-10-05 as a provisional value from one culture crop of W-320 (largest outside
share of a matched nucleus 0.073), not fitted and not an accuracy statement; 0.0 is exact
containment (option C1). Shares are voxel counts, so anisotropic spacing does not change
them.

**Failure behavior.** No nucleus image: no correspondence, every cell `unavailable`. A
nucleus image with no object: every cell is `no_nucleus` (all excluded by default, and the
record says the nucleus run's outcome was `empty`). A supplied table with an unknown
value or a repeated nucleus raises `ValueError`. Nothing is repaired: no nucleus is
clipped to its cell, no cell is merged or split, no nucleus is re-matched by distance.

**Resources** (code-derived). Accumulated per Z plane, so only the label volumes are
whole-FOV: per plane, the pairs at nuclear voxels as one uint64 key (`n × (max_cell + 1) +
c`) and `np.unique(…, return_counts=True)`, at most 2.24 million keys (18 MB) and a sort
of the same size for a 1496² plane. Whole FOV: the two `uint32` volumes (896 MB), plus the
expanded territory when assign expanded (448 MB more), plus per-plane temporaries under
100 MB; about 1.4 GB in total, under the 4 GiB target. Time: one sort per plane, `O(k log
k)` in the nuclear voxels of the plane, a few seconds for the whole FOV by this count, not
measured.

## Compartment partition

**Rule.** For an `assigned` molecule of cell `c`: `unavailable` without nuclei;
`no_nucleus` when `c`'s correspondence is `no_nucleus` (never `cytoplasm`: a cell without a
detected nucleus gets no compartment); `withheld` when `c` is withheld; otherwise `nucleus`
when its `nucleus_id` is a nucleus matched to `c`, else `cytoplasm`. Because an `available` cell has no `foreign_nucleus` flag, a
nonzero `nucleus_id` inside it is always one of its own nuclei, so the rule partitions its
molecules exactly. Molecules inside a nucleus but outside every territory stay
`unassigned`.

**Parameters.** None beyond those of the correspondence.

**Failure behavior.** None: every molecule gets exactly one compartment value (or null
when not `assigned`).

**Resources** (code-derived). One lookup per molecule in a map from nucleus to matched cell
(the nucleus table), linear in molecules and nuclei.

## Count accounting

**Rule.** Group the `assigned` molecules by `(cell_id, gene_id)` for `whole`, and by
`(cell_id, gene_id, compartment)` for the `nucleus` and `cytoplasm` molecules of
`available` cells; keep nonzero groups as the long `counts` table. Cell sizes and
centroids: per positive territory value, the voxel count and the mean of the voxel
indices (float64 sums accumulated per plane), from the original territory, and
`expanded_size_voxels` and `expanded_centroid_*` from the expanded one; physical sizes multiply by the voxel volume
(or area) when the calibration is known. The identities of the contract ("Count
accounting", 1 to 5) are checked after counting; a violation raises `AssertionError`
naming the identity, because it can only come from a defect.

**Parameters.** None. No size or count filter.

**Failure behavior.** A territory image with no object gives an empty cell table, an empty
`counts` table, every in-grid molecule `unassigned` and the outcome `empty`; it is not an
error, and unlike today a FOV whose molecules all miss the cells keeps its cells.

**Resources** (code-derived). The group-by is linear in molecules. Sizes and centroids:
one pass over the territory image per plane with `np.unique` and weighted sums, so about
the label volume plus per-plane temporaries. Centroid sums of up to 10⁸ voxels with
indices below 1496 stay below 2⁵³, so float64 sums are exact and the centroids
deterministic.

## Engineering validation design (task group 6)

Task group 6 of the outline, "Validation and inspection", has three parts: this
engineering validation, the bounded real examples below and the usage documentation (the
tour notebook). The engineering validation is, as in §2.7 and §2.8, known-answer fixtures
with pass/fail tolerances fixed before the run, in default-tier pytest modules, plus
`learned`-tier modules for the checks that need StarDist or Cellpose. It has no
comparison matrix, no parameter sweep, no ranking of methods, no default chosen from
comparative data and no accuracy claim. Two fixed settings of one parameter in a check
(A7, A8) test the rule at its boundary; they are not a sweep.

Rules:

* **Fixtures are hand-specified geometry**: boxes, ellipsoids and single voxels at stated
  positions, with stated molecules, built in the test, each at most 32×64×64 voxels. Every
  expected value is derived from the geometry by hand, by a formula in the test, or from a
  pinned golden or parity output that the row names; never from the output of the code
  under test.
* **Seeds.** The two golden fixtures keep their own: `seg_golden` uses seed 20261005 (the
  W-307 golden test) and `assign_golden` uses no random numbers. The new fixtures use seed
  101 for the noise of the `seeded` stain, seed 102 for the noise of the L11 stains, and
  seed 100 for the row permutation of A20. No other seed and no other random number is
  used.
* **The §2.12 molecular fixtures are not cellular truth.** The calibrated synthetic scenes
  of §2.12 (and §2.8's `cal` scenes) place amplicons, not cells, nuclei or membranes: no
  check derives a cell, a territory, a nucleus or a correspondence from them, and none uses
  them to judge a segmentation.
* **Tolerances.** Exact equality for every label image, status, count, identity and hash;
  1e-12 relative for physical sizes (a product of three spacings); the W-306 CPU-against-GPU
  tolerances for the learned methods (label count within max(1, 1 %), at least 99 % of
  labels matched at IoU ≥ 0.5, median matched IoU at least 0.99), which are engineering
  bounds from one host, not accuracy. A tolerance is not adjusted during the run; a correct
  implementation that cannot meet one goes to Jiahao.
* Every test runs on CPU with one thread in the routine gate; L12 runs only in a batch whose
  guidance grants the GPU (D6).

### Fixtures

| Fixture | Construction | Used by |
| --- | --- | --- |
| `seg_golden` | The W-307 segmentation golden fixture (16×64×64 uint8 DAPI, amplicon, Flamingo; seed 20261005). | L4, L6, L7 |
| `assign_golden` | The W-308 assignment golden fixture (16×64×64 `uint16` labels with cells 3, 7, 12, 20, 25; 19 molecules; five genes), on an uncalibrated grid (`ImageMetadata("assign_golden")`); its Z maximum as a plane on the projection of that grid. | A2, A3, A4, A19 |
| `boxes` | 8×32×32 grid, spacing (0.35, 0.1, 0.1) µm, unit `micrometer`. Cells are boxes; nuclei are boxes; every box includes the plane z = 4 and the shares below hold in 3D: cell 1 with one nucleus inside; cell 2 with two nuclei inside; cells 3 and 4 adjacent, with nucleus 31 of 100 voxels split 50 / 50 across their border; cell 5 with nucleus 51 of 10 voxels, 6 inside and 4 in the background; cells 6 and 7 adjacent, with nucleus 61 of 10 voxels, 8 in cell 6 and 2 in cell 7; cell 8 without a nucleus; nucleus 91 entirely in the background. Nucleus 91 lies at least 4 voxels in Y and X from every cell box. Molecules: in every cell two molecules outside every nucleus, and two inside each nucleus part that lies in the cell (nucleus 31's halves in cells 3 and 4, nucleus 61's parts in cells 6 and 7; none in cell 8); one in the background; one in nucleus 91; one in the background part of nucleus 51; three off the grid. | A2, A5–A13, A15, A20 |
| `boxes_51_49` | `boxes` with nucleus 31 split 51 / 49 (51 of its 100 voxels in cell 3). | A7 |
| `boxes_nocal` | `boxes` with metadata `ImageMetadata("boxes")` (no spacing, no unit). | A13 |
| `bounds` | 8×8×8 labels, value 1 everywhere except a 0 at voxel (2, 3, 3); molecules at `c ∈ {−0.5, −0.5 − 1e−9, 2.5, 3.5, n − 0.5 − 1e−9, n − 0.5, 1e6}` on each axis in turn (`n` that axis's size), the other two at 1.0, and one molecule at (2.0, 3.0, 3.0). | A1 |
| `plane` | `boxes` reduced to its plane z = 4 (1×32×32), on the projection of the `boxes` grid; molecules of `boxes` with `z` from 0 to 7, and one at z = 1000 (off the 8-plane molecule grid). | A16 |
| `culture` | 8×32×32 culture layer, spacing (0.35, 0.1, 0.1) µm: 2D cell and nucleus labels (1×32×32 boxes, each nucleus inside its cell) and a noise-free stain of 200 in z 2–4 and 10 elsewhere; extension with a numeric `threshold` strictly between the two stain levels on the scale `ZExtensionConfig` documents (W-307 leaves the scale open; 10 < t < 200 in grey levels, 10/255 < t < 200/255 on the [0, 1] scale), `median_um` 0.1, `min_area_um2` 0.01, `dilation_um` 0 and one hole filling. Every plane is constant, so the median keeps it, and the threshold separates the levels whether the comparison is `>` or `≥`: the extended labels are the 2D labels in z 2–4 and 0 elsewhere. Molecules inside the layer, above it and below it. | L5, A17 |
| `seeded` | The `boxes` grid; the `boxes` nuclei as seeds (nucleus 91 among them, in the background); a stain of 200 inside the `boxes` cells and 20 elsewhere plus Gaussian noise of standard deviation 5 (seed 101); `SeededWatershedConfig(sigma_um=0.1)` (one pixel in Y and X), other fields at their defaults. | L9, L13 |
| `imports` | TIFF files written in the test: big-endian `uint16` labels with the values 4, 9 and 30, `int32`, `float32` and `bool` masks, a YX file, a file with `starfinder_metadata` that differs from the grid, a file of another shape. | L3 |
| `many_labels` | 32×64×64 `int32` labels whose first 70,000 voxels in C order hold the values 1 to 70,000 and the rest 0. | L2 |
| `parity` | The W-306 parity outputs, copied as `test/data/segmentation_parity.npz` (357 KB, sha256 `4a5df997…`), {doc}`segmentation-contract` ("Tests the implementation changes"). | L10 |

### Checks

| # | Check | Fixture | Metric | Pass/fail tolerance |
| --- | --- | --- | --- | --- |
| L1 | Label contract | hand-built arrays | `SegmentationResult.__post_init__` and `ReferenceGrid` | A `uint32` ZYX array on its grid constructs; each violation (a dtype other than `uint32`, among them an `int32` array with a negative value; a 2D array; `labels.shape ≠ grid.shape_zyx`; a target other than `nucleus` or `cell`; `plane` with Z>1; `volume` or `extended` with Z=1) raises at construction (the contract names no error type for these checks). Exact. |
| L2 | Label dtype rule | `many_labels`; int32, uint16 and uint32 versions of `assign_golden`'s labels | Converted array and values | All 70,000 values survive the conversion to `uint32` (none wraps; label 65,536 stays 65,536); the three dtypes give identical `uint32` arrays; a negative value raises `ValueError`. Exact. |
| L3 | Import | `imports` | `import_labels` result and record | Big-endian data read with native values; a YX file becomes 1×Y×X and needs a Z=1 grid; `float32` and `bool` raise `TypeError`; the file of another shape raises `IncompatibleGeometryError`; differing metadata raises `ValueError`; missing metadata is recorded as `declared`; `relabel=True` maps 4, 9, 30 to 1, 2, 3 with the map recorded; the file and array SHA-256 are recorded. Exact. |
| L4 | `expand_labels` | `seg_golden` stand-in labels; single-voxel label in 9×21×21 with spacing (0.3, 0.1, 0.1) | Labels | `planar`, `pixel`, distance 4, cast to `uint16`, equals W-307's `LABELS_3D[(False, True)]`, and on the 1×Y×X form of the 2D stand-in labels, reshaped to Y×X, `LABELS_2D[(False, True)]`; `volumetric`, `um`, distance 0.5 labels exactly the voxels whose physical distance to the seed is ≤ 0.5 µm (computed in the test); distance 0 is the identity; an empty image stays empty; `um` without spacing raises. Exact. |
| L5 | `extend_labels_through_z` | `culture` | Extended labels | Equals the 2D labels in z 2–4 and 0 in the other planes; geometry `extended`; labels with Z>1 raise; a stain of 10 everywhere with the same threshold gives an all-zero image with outcome `empty`. Exact. **Provisional**: no MATLAB parity (MATLAB not run). |
| L6 | `labels_to_grid` | `seg_golden` | Labels | The W-307 shrunk stand-in labels, 16×32×32 → 16×64×64, equal the values of W-307's `RESTORED_LABELS` array (its 2×2 block repetition) after a cast to `int32`; a 1×30×32 image → exactly 1×61×63 by the index rule ⌊(i + 0.5) × n_s / n_t⌋, computed in the test. Exact. |
| L7 | Input functions | `seg_golden`; constant and zero images | Digests and values | `composite_nuclei_amplicon` and `enhance_with_flamingo` equal W-307's `COMPOSITE_DIGESTS`, `FLAMINGO_DIGEST` and `CONSTANT_DIGESTS`; `normalize_percentiles` equals `(x − p_low) / (p_high − p_low + 1e-20)` with NumPy's linear percentiles (the formula of {doc}`segmentation-algorithms`), within 1e-6 relative and 1e-6 absolute (float32), and a constant image gives zeros; `rescale_input` divides the spacing by the factors and records the rescale in `frame_id`. Exact except the stated float32 bound. |
| L8 | Every method at contract level: the stage wrapper | a test-only method registered for the test; `seg_golden` | Raised errors and the record | Each raising check of the 11 wrapper checks of {doc}`segmentation-contract` (config, target, input, device, dependencies, model, dimensionality, seeds; output after the run) raises, with the error the contract names where it names one (config `TypeError`, target `ValueError`, input shape `IncompatibleGeometryError`, dependencies `SegmentationBackendUnavailableError`, model `MissingModelError`, dimensionality and seed grid `IncompatibleGeometryError`, output `ValueError` for a negative value), and an input violating two checks raises the earlier one; the test-only spec declares a missing dependency and `models=True` where checks 5 and 6 need them; a method returning another shape, a negative value or a float array is refused by the output check; a dropped Z axis for one plane is restored; the record (check 11) holds the uniform provenance entry; `segment` refuses `LabelImportConfig` with `TypeError`. Exact. |
| L9 | `seeded_watershed` at contract level | `seeded` | Labels and record | As {doc}`segmentation-algorithms` ("`seeded_watershed`") and the W-306 prototype `scripts/seeded_watershed.py` (lines 43–45: the mask is the foreground united with every seed voxel, and the seeds are the markers) state: the set of output values equals the set of all seed values; every seed voxel keeps its seed's value; nucleus 91, in the background and at least 4 voxels from every stained cell, gives a cell equal to its seed exactly; no seeds give an all-zero image with outcome `empty`; a grid without spacing raises `ValueError`; the result is on the input grid. Exact. |
| L10 | `stardist` at contract level (`learned`) | `parity` | Labels | The six `rescale0` arrays of P1 to P3, recomputed on CPU with `scale` 1.0, the stored thresholds and, for `expand1`, planar expansion by 4, equal the saved arrays after a cast to `uint16`; a 3D model on Z=1 and a 2D model on Z>1 raise `IncompatibleGeometryError` at check 7 of {doc}`segmentation-contract`, that is after the dependency import (check 5) and the model-file resolution (check 6) and before the method's `run` is called, so no model object is built; `threshold_source` is `stored`. Exact on CPU. |
| L11 | `cellpose` at contract level (`learned`) | Z=1 stains of the `boxes` geometry (its plane z = 4; 200 in cells, 20 elsewhere, noise of standard deviation 5 with seed 102); a missing model path | Labels and errors | A missing model path raises `MissingModelError` before any library call; a Z=1 input returns 1×Y×X; the result is `uint32`; two calls in one process give identical labels (the normalization mapping is not mutated); `diameter` is required. Exact on CPU. |
| L12 | CPU against GPU (`learned`, GPU batches only) | `parity` P1, P2 | Label count, matched fraction, median matched IoU | Within max(1, 1 %); at least 99 % at IoU ≥ 0.5; median at least 0.99 (W-306 tolerances, one host). |
| L13 | Segmentation persistence | an imported run of the `seg_golden` stand-in labels; a `seeded_watershed` run on `seeded` | Files and record | `labels.tif` and `segmentation.json` written for both runs, and `input.ome.tif` for the `seeded_watershed` run (an import has no segmentation input; {doc}`segmentation-contract`, "Method input" and "External-mask import"); `FOV.load_segmentation` returns an equal result (array equality, record equality); a changed label file raises naming the hash. Exact. |
| L14 | Model resolution (`learned` for the files) | the cached `2D_versatile_fluo` files; a copy with one byte changed | Hashes and errors | Known-model hashes equal `KNOWN_MODELS`; the changed copy raises `ModelHashMismatchError` naming both hashes; with the network disabled nothing is fetched. Exact. |
| A1 | Sampling at the bounds | `bounds` | Sampled voxel and status | On each axis: −0.5 → voxel 0; −0.5 − 1e−9 → `outside_grid`; 2.5 → 3; 3.5 → 4; n − 0.5 − 1e−9 → n − 1; n − 0.5 and 1e6 → `outside_grid`; every in-grid molecule of these is `assigned` to cell 1; the molecule at (2.0, 3.0, 3.0) is `unassigned`. Exact. |
| A2 | Statuses and accounting | `assign_golden` (without gene `Z`), `boxes` | Molecule statuses; identities 1–5 of the contract | Every molecule has the status written in the fixture; the four status counts sum to the number of molecules; whole-cell counts sum to `n_assigned`; `nucleus + cytoplasm = whole` per available cell and gene, and the compartment values of the assigned molecules sum to `n_assigned`; excluded molecules equal the excluded cells' `n_molecules`. Exact. |
| A3 | Legacy equivalence | `assign_golden` (without gene `Z`, as A2) | Per-molecule cell and counts | With no expansion and with `planar`/`pixel`/4, `cell_id` equals the pinned `SEG_LABELS` of the remaining molecules (0 as null; the entry of the gene-`Z` molecule is left out) and the whole-cell matrix the pinned counts, which never held gene `Z`, in 3D and 2D. The pinned `volume` and `fov_*` columns (computed after the legacy expansion) equal `size_voxels` and the truncated `centroid_*` without expansion, and `expanded_size_voxels` and the truncated `expanded_centroid_*` with it (for cell 3 in 3D: 771 and 1866 voxels). Exact. |
| A4 | One expansion | `assign_golden` (without gene `Z`, as A2) | Territories, `in_expansion`, errors | `AssignmentResult.territories` equals `expand_labels` of the cell run and `cell_labels` equals the cell run's labels; `in_expansion` is true exactly for the three band molecules (3D and 2D), whose `original_cell_id` is 0; a cell run whose record lists `expand_labels` raises `ValueError` with and without `expansion`; `unit="um"` on this uncalibrated grid raises `ValueError`; the record's `expansion` entry holds both masks' SHA-256 and the voxels added (no file paths: `assign_molecules` writes nothing). Exact. |
| A5 | Correspondence, one nucleus | `boxes` cell 1 | Nucleus and cell rows | Nucleus `matched`, `share_in_cell` 1.0; cell `n_nuclei` 1, correspondence `matched`, no flag, `available`. Exact. |
| A6 | Correspondence, several nuclei | `boxes` cell 2 | Cell row | `n_nuclei` 2, flag `several_nuclei` only, `available`; its nuclear count is the sum over both nuclei. Exact. |
| A7 | Correspondence, ambiguous | `boxes` cells 3, 4; `boxes_51_49` | Nucleus status and flags | 50/50: nucleus 31 `ambiguous` (0.5 is not more than 0.5); cells 3 and 4 have correspondence `ambiguous`, flag `ambiguous_nucleus`, compartments `withheld`, and are kept by default (ambiguous is not absence of a nucleus). 51/49: nucleus 31 matched to cell 3 and flagged `outside`; cell 3 `matched`, `nucleus_outside_cell`, `withheld`; cell 4 `no_nucleus` with `foreign_nucleus`, excluded by default. Exact. |
| A8 | Correspondence, nucleus outside its cell | `boxes` cell 5 | Flags | With `outside_tolerance` 0.0 (passed explicitly) and with the default 0.1, nucleus 51 (outside share 0.4) is `matched`, `outside`, cell 5 `withheld`; with 0.5, not `outside` and cell 5 `available`; the molecule in its background part is `unassigned` with `nucleus_id` 51. The default at its boundary, on a hand-built pair on the `boxes` grid: a 30-voxel nucleus with exactly 3 voxels outside its cell (0.1) is not `outside` and its cell `available`; with one voxel more (4 / 30), `outside` and `withheld`; when the outside part lies in a neighbouring cell, that cell has `foreign_nucleus` at 0.1 and at 0.5. Exact. |
| A9 | Foreign nucleus | `boxes` cells 6, 7, with `exclude_cells_without_nucleus=False` | Flags | Nucleus 61 matched to cell 6 (share 0.8), `outside`; cell 6 `matched`, `nucleus_outside_cell`, compartments `withheld`; cell 7 correspondence `no_nucleus`, flag `foreign_nucleus`, compartments `no_nucleus`; both keep their whole-cell counts and have no nuclear or cytoplasmic rows. Exact. |
| A10 | Compartment partition | `boxes` | Molecule compartments; counts | The available cells are 1 and 2; in each, the nuclear molecules are exactly those placed in its nuclei and `nucleus + cytoplasm = whole` per gene; withheld cells (3, 4, 5, 6) have no compartment rows (absent, not zero); with the exclusion off, cells 7 and 8 have no compartment rows and their molecules are `no_nucleus`, none `cytoplasm`; without nuclei every assigned molecule is `unavailable`. Exact. |
| A11 | Exclusion and its statuses | `boxes` with nuclei (default), with `exclude_cells_without_nucleus=False`, without nuclei | Cell status, molecule status, counts, totals | Default: cells 7 and 8 (correspondence `no_nucleus`) are `excluded_no_nucleus` with reason `no_matched_nucleus`, their molecules `excluded_cell` with their `cell_id`, no count rows, `exclusion_source` `default`; cells 3 and 4 (`ambiguous`) stay kept; the complete cell table keeps cells 7 and 8, and the totals before and after differ by exactly those cells and molecules. `False`: all kept; cells 7 and 8 have compartments `no_nucleus` with whole-cell counts only. Without nuclei: nothing excluded, every cell `unavailable`. Exact. |
| A12 | Supplied correspondence | `boxes` | Result tables and arrays; errors | A supplied table equal to the derived matches gives the derived result: the `molecules`, `cells`, `counts` and `nuclei` tables equal under `assert_frame_equal(check_exact=True)` and the label arrays equal. The records are compared with these provenance fields excluded: `inputs.correspondence` (its `source` and the supplied table's `sha256`); every other record field is equal. An unknown value or a repeated nucleus raises `ValueError`. Exact. |
| A13 | Cell metadata | `boxes`, `boxes_nocal` | Sizes, centroids, calibration | `size_voxels` equal the box volumes; `size_physical` equals voxels × 0.35 × 0.1 × 0.1 µm³ within 1e-12 relative, unit `micrometer^3`; centroids equal the box centres exactly; `boxes_nocal` gives NaN sizes and `calibration` `unknown`. Exact except the stated relative bound. |
| A14 | Grid and identity checks | `boxes` with a shifted frame, another shape, a plane on the wrong projection, nuclei on another grid, another FOV's namespace | Raised errors | Each raises its named error before any sampling. Exact. |
| A15 | Persistence round trip | `boxes` on a synthetic FOV, CSV and Parquet, in the cases of the table "Label images of a checkpointed assignment" of {doc}`assignment-contract`: cell run saved under its run or unsaved; nucleus run absent, saved or unsaved; without expansion and with a `planar` 0.1 µm expansion | Files, links, tables, images, record | In each case exactly the files of that table exist (linked masks are not copied; `cell_labels.tif`, `nucleus_labels.tif` and `territories.tif` only in their rows); every `file` link in `assignment.json` resolves to an existing file whose SHA-256 equals the recorded one; `FOV.load_assignment` returns tables equal under `assert_frame_equal(check_exact=True)`, label arrays equal to the result's and an equal record; a changed written file, or a linked `labels.tif` changed or removed, raises `ValueError` naming the path; the files of `FOV.run` and of segmentation are byte-identical before and after `FOV.assign`. Exact. |
| A16 | Z=1 case | `plane` | Statuses, counts, calibration | Each molecule with `z` from 0 to 7 has the `cell_id` of the plane label at its `(y, x)`; the z = 1000 molecule is `outside_grid`; `size_physical` equals pixels × 0.1 × 0.1 µm² (unit `micrometer^2`, `calibration_source` `projection_source`). Exact. |
| A17 | Culture case | `culture` | Statuses and compartments | Molecules in z 2–4 are assigned to the cell whose 2D label holds their `(y, x)`; those in the other planes are `unassigned`; every cell is `matched` and `available`, and its nuclear molecules are those inside its extended nucleus. Exact. |
| A18 | Workflow translation | legacy configurations; `assign_golden` written as a goodSpots CSV, integer and float | Config, errors, outputs | The legacy keys translate as the contract's table states (the legacy `dilation_distance` with `legacy_pixel_expansion=True`); `stardist_segmentation.expand_labels: true`, alone or with the assignment key, raises naming the key; float coordinates are accepted; a gene mismatch with the codebook raises; where `anndata` is importable, `raw.h5ad`'s legacy `obs` columns and `X` equal the frozen helper's values for the kept cells. Exact. |
| A19 | Determinism | `assign_golden` (without gene `Z`, as A2), `boxes` | SHA-256 of every result table, three single-thread processes | Identical. |
| A20 | Row order | `boxes` with the molecule rows permuted by `numpy.random.default_rng(100).permutation` | Per-key results and `MoleculeTable.sha256` | Identical per `(spot_namespace, spot_id)` and the same hash. Exact. |

### What the default tier covers and what it does not

L1–L9, L13, A1–A17, A19 and A20 run in the default tier, and A18 there except its
`raw.h5ad` comparison, which runs where `anndata` is installed. L10, L11 and L14 are
`learned` (StarDist, Cellpose and model files on CPU) and L12 runs only on a GPU batch.
Nothing here measures segmentation quality: a hand-built fixture shows that a rule is
implemented as written, not that a method finds cells.

### Resource plan

Every default-tier fixture is at most 32×64×64 voxels and every check takes well under a
second (the golden tests take 0.2–0.3 s each); the learned checks take what W-306 measured
for its parity inputs on CPU (a few seconds per call after the model load). Each run
records wall time and peak RSS with `/usr/bin/time -v` against the 4 GiB stop target.

### Implemented checks (W-318)

The checks are in the modules of `src/python/test/` listed below. The task groups added
most of them with the code they check; `test_segmentation_validation.py` and
`test_assignment_validation.py` add L12, L4's identity, empty-image and missing-spacing
clauses on the row's own fixtures, L7's rescale clause on `seg_golden`, A2's five
identities on `assign_golden`, and A15's check of the `FOV.run` and segmentation files on
the `boxes` FOV. Each of the two modules also checks that every row of the table below
names tests that exist. A test named for a row but built on another fixture (the `boxes`
cases of `test_l4_*` in `test_assignment.py`, the ramp of `test_l7_rescaling_*`, the ring
of `test_l5_hole_filling_*`, the plane of `test_l9_the_plane_z4_*`, and the development
preset of `test_fov_run_and_segmentation_files_are_untouched_by_assign`, a W-315
integration test) is an additional check and is not listed.

The identities of A2 are checked by `check_accounting` (`test_assignment.py`), which
recomputes every count row from the `molecules` table (whole per kept cell and gene,
nucleus and cytoplasm per available cell and gene) and compares the `counts` table, the
cell table and `record["counts"]` with them exactly; it also runs in A11, A16 and A17.
Data-frame comparisons of the rows marked exact use `check_exact=True`. Physical sizes
(A13, A16) are compared within 1e-12 relative, the bound the rule "Tolerances" sets for
them.

Seeds are those of the rules above: 20261005 inside `seg_golden`, 101 for the `seeded`
stain (`segmentation_fixtures.py`), 102 for the L11 stains, 100 for the A20 permutation, and
none for `assign_golden` or the other hand-built fixtures. Every tolerance is the one in
the table "Checks", unchanged; no check is skipped in the default tier or marked as an
expected failure. Rows A2, A3, A4 and A19 use `assign_golden` without the gene-`Z` molecule
and the matching subset of the pinned labels; the unmodified fixture serves the
unknown-gene rejection and the legacy pins (`test_assignment_golden.py`).

Two rows are not in the default tier. A19 starts three Python processes, each of which
imports the package (about 2.2 s on one CPU), so its test is `slow` under the rule of
{doc}`contributing` ("Test markers"); its three hash lists for each change are kept in
the run's notes. L12 is `learned` and `slow` and runs only where `STARFINDER_L12_GPU_PYTHON`
names the Python of a GPU environment, that is in a batch whose guidance grants the GPU;
otherwise it skips with that reason. The learned checks need the extras and the cached
models (L10's P1 and L12 also `STARFINDER_STARDIST_3D_SPLEEN`) and skip without them.

| # | Module and tests | Tier |
| --- | --- | --- |
| L1 | `test_segmentation_labels.py`, `test_l1_*` | default |
| L2 | `test_segmentation_labels.py`, `test_l2_*` | default |
| L3 | `test_segmentation_labels.py`, `test_l3_*` | default |
| L4 | `test_segmentation_golden.py`, `test_expand_labels_function`; `test_assignment.py`, `test_l4_volumetric_expansion_in_um_labels_the_voxels_within_the_distance`; `test_segmentation_validation.py`, `test_l4_distance_zero_is_the_identity_and_an_empty_image_stays_empty`, `test_l4_um_without_spacing_raises` | default |
| L5 | `test_segmentation_labels.py`, `test_l5_the_labels_extend_through_the_stained_planes`, `test_l5_labels_with_z_above_1_raise`, `test_l5_a_stain_of_10_everywhere_gives_an_empty_image` | default |
| L6 | `test_segmentation_labels.py`, `test_l6_the_shrunk_stand_in_labels_return_to_the_restored_values`, `test_l6_an_odd_grid_is_reached_exactly` | default |
| L7 | `test_segmentation_inputs.py`, `test_l7_composite_and_enhancement_equal_the_pinned_digests`, `test_l7_constant_and_zero_inputs_equal_the_pinned_digests`, `test_l7_normalization_follows_the_formula`, `test_l7_normalization_of_a_constant_image_gives_zeros`; `test_segmentation_validation.py`, `test_l7_rescaling_seg_golden_divides_the_spacing_and_records_the_rescale` | default |
| L8 | `test_segmentation_segment.py`, `test_l8_*` | default |
| L9 | `test_segmentation_segment.py`, `test_l9_every_seed_keeps_its_value_and_no_other_value_appears`, `test_l9_no_seeds_give_an_empty_label_image`, `test_l9_a_grid_without_spacing_raises` | default |
| L10 | `test_segmentation_learned.py`, `test_l10_*` | `learned`; P1 (`3D_spleen`) also `slow` |
| L11 | `test_segmentation_learned.py`, `test_l11_*` | `learned`; the two Cellpose calls also `slow` |
| L12 | `test_segmentation_validation.py`, `test_l12_cpu_and_gpu_labels_agree_within_the_w306_bounds`, `test_the_l12_metrics_on_hand_built_labels` | `learned` and `slow`, GPU batches only; the metric test on hand-built labels default |
| L13 | `test_segmentation_persistence.py`, `test_l13_*` | default |
| L14 | `test_segmentation_learned.py`, `test_l14_the_cached_files_equal_the_table_and_a_changed_copy_raises` | `learned` |
| A1 | `test_assignment.py`, `test_a1_sampling_at_the_bounds` | default |
| A2 | `test_assignment.py`, `test_a2_boxes_statuses_and_accounting`; `test_assignment_golden.py`, `test_the_package_statuses_and_count_identities`; `test_assignment_validation.py`, `test_a2_assign_golden_statuses_and_identities_1_to_5` | default |
| A3 | `test_assignment_golden.py`, `test_the_package_reproduces_the_pins` | default |
| A4 | `test_assignment.py`, `test_a4_*` | default |
| A5 | `test_assignment.py`, `test_a5_to_a7_nucleus_table_and_cell_rows` | default |
| A6 | `test_assignment.py`, `test_a5_to_a7_nucleus_table_and_cell_rows`, `test_a6_several_nuclei_count_both` | default |
| A7 | `test_assignment.py`, `test_a5_to_a7_nucleus_table_and_cell_rows`, `test_a7_a_nucleus_split_51_49` | default |
| A8 | `test_assignment.py`, `test_a8_a_nucleus_outside_its_cell`, `test_the_default_correspondence_is_option_c2`, `test_the_default_tolerance_at_its_boundary` | default |
| A9 | `test_assignment.py`, `test_a9_a_foreign_nucleus` | default |
| A10 | `test_assignment.py`, `test_a10_compartment_partition` | default |
| A11 | `test_assignment.py`, `test_a11_exclusion_and_its_statuses` | default |
| A12 | `test_assignment.py`, `test_a12_*` | default |
| A13 | `test_assignment.py`, `test_a13_cell_metadata` | default |
| A14 | `test_assignment.py`, `test_a14_grid_and_identity_checks_raise_before_sampling` | default |
| A15 | `test_assignment_persistence.py`, `test_a15_*`; `test_assignment_validation.py`, `test_a15_fov_run_and_segmentation_files_of_boxes_are_untouched_by_assign` | default |
| A16 | `test_assignment.py`, `test_a16_a_plane_on_a_projected_grid` | default |
| A17 | `test_assignment.py`, `test_a17_culture_labels_extended_through_z` | default |
| A18 | `test_assignment_workflow.py`, `test_a18_*` | default; the `raw.h5ad` comparison where `anndata` is installed |
| A19 | `test_assignment.py`, `test_a19_three_single_thread_processes_give_identical_tables` | `slow` |
| A20 | `test_assignment.py`, `test_a20_row_order` | default |

## Bounded real examples

These run after the segmentation methods (task group 3) and assignment (task group 4)
exist and inspect the three contexts on the W-305 crops, volumetric first. They are inspections, not comparisons: no annotation exists
(W-123), the culture reference labels are earlier workflow results, and nothing is ranked
or called accurate. Every input is a W-305 crop listed in
`runs/W-305/20261004T0630Z-summary/examples.md` with its SHA-256 in the run's
`metrics.json` and `manifest.json`.

**Molecules.** W-305 staged morphology only. Each example needs the molecule table of the
same FOV window: the existing goodSpots of that FOV cut to the crop window, or a `FOV.run`
of the window. Neither is a W-305 crop, so the issue that runs the examples needs an
authorization naming those files; without it, the examples run segmentation, correspondence
and the territory statistics only, and report the molecule outputs as not run.

| Order | Context | Crop (W-305 run) | Runs | Resources |
| --- | --- | --- | --- | --- |
| 1 | Volumetric tissue (LN, `Position020`) | `20261004T0530Z-ln-3d-spleen/crop_dapi_round4.tif` (50×512×512 uint8), and `crop_dapi_enhanced_with_flamingo.tif` for the Flamingo-assisted nuclei | Nuclei: `stardist`, `3D_spleen`, `scale` 1.0, stored thresholds. Territories: (a) assign's planar expansion of the nuclei by 0.7776 µm (4 pixels at 0.1944 µm, the legacy parity distance), with the original nuclei and the expanded territories both kept, (b) `seeded_watershed` on `crop_flamingo.tif`. Assign with nuclei, C1 thresholds, exclusion on and off. Spacing 0.3463 × 0.1944 × 0.1944 µm (TIFF tags). | StarDist 226 s per call on one CPU thread (W-306 `ln_dapi_round4@1.0`; 58 s on four threads, 55 s on the GPU), 2,245 MiB peak RSS per call including the loaded model; assignment code-derived: two `uint32` volumes of 52 MB plus the expanded one, under 0.5 GB, seconds. |
| 2 | 2D tissue (tissue-2D, `round1/tile_1`) | `20261004T0600Z-tissue2d-watershed/crop_PI.tif` and `crop_amplicon_merged.tif` (1024×1024 uint8) | Nuclei: `stardist`, `2D_versatile_fluo`, `scale` 0.25. Cells: `seeded_watershed` on the PI–amplicon composite (`composite_nuclei_amplicon` of `crop_PI.tif` and `crop_amplicon_merged.tif` with the default `CompositeConfig`, role `composite`), seeds the nuclei, `sigma_um` 1.5, spacing 0.0946 µm (corrected 2026-10-05, W-321 review; W-320 ran it on the amplicon signal alone). Assign as a `plane` (Z=1 grid), with nuclei, C2 thresholds (the default since W-334; W-320 ran C1, which gives the same result here because every nucleus lies wholly inside its cell), exclusion on and off. | StarDist 1.2 s per call on one CPU thread, 868 MiB peak RSS including the loaded model (W-306 `tissue_pi@0.25`); watershed 0.25 s (W-306 `tissue2d_amplicon`); assignment code-derived: 4 MB per label image, under 1 s. |
| 3 | Single-layer culture (stitched sample) | `20261004T0615Z-culture-fused-crop/crop_DAPI_3d_42x512x512.tif`, `crop_Flamingo_3d_42x512x512.tif`, the projections `crop_DAPI_max_2d.tif`, `crop_Flamingo_max_2d.tif`, and the references `reference_Cell_label_2d.tif`, `reference_DAPI_label_2d.tif`, `reference_Cell_3d_42x512x512.tif` (big-endian; imported) | (a) The imported 2D references, extended through z with `extend_labels_through_z` on the 3D crops; (b) Cellpose `cpsam_v2` on the projections, cells at `diameter` 240 and nuclei at an explicit recorded diameter (W-306 ran 60 and 240 on the DAPI projection), then extended the same way. Assign with nuclei and compartments, C1, exclusion on and off. The 2D projections are 1024×1024 and the 3D crops their 512×512 centre, so the extension uses the matching centre window. | Cellpose 203 s per call on one CPU thread for the two-channel cells, 1,718 MiB peak RSS including the loaded model, or 1.5 s on the GPU (W-306 `culture_cells_2d@d240`); nuclei 229 s on one CPU thread at diameter 60 (`culture_nuclei_2d@d60`; diameter 240 was measured on the GPU only, 0.38 s); import and extension code-derived: 42×512×512 `uint32` is 44 MB per volume; assignment seconds. |

**Outputs to inspect**, per example and per territory option, written to the run directory
(at most ten PNGs in total):

* an overlay: territory outlines (Z maximum, and the middle plane for the volumes) over
  the stain, with molecules coloured by status (`assigned`, `unassigned`, `excluded_cell`,
  `outside_grid`);
* distributions: `size_voxels` and `size_physical`, molecules per cell and nuclei per cell
  (histograms and quantile tables);
* correspondence warnings: nuclei per status, cells per correspondence status and per
  flag, the distribution of the matched nuclei's outside share (the evidence a C2
  tolerance would need), and the cells whose compartments are withheld or `no_nucleus`;
* totals before and after the exclusion: cells, molecules `assigned`, molecules
  `excluded_cell`, and the whole-cell, nuclear and cytoplasmic totals;
* `summarize_assignment` as JSON, with the input hashes.

Budget: one CPU process at a time within 4 GiB RSS; the whole plan is about 15 minutes on
one CPU thread, dominated by the StarDist 3D calls (two LN inputs at about 226 s) and the
Cellpose projection calls (203 s and 229 s); on a GPU batch, a few minutes.

**Whole-FOV measurement** (authorized once at W-309 on 2026-10-05; outside the budget
above). After the three examples, one whole volumetric field of view is segmented
block-wise, to replace the estimate of {doc}`segmentation-algorithms` by a measurement:

* Input: the whole `round4` `ch04` (DAPI) image of LN `Position020`, 50×1496×1496, the
  image `crop_dapi_round4.tif` was cut from, with its SHA-256 recorded.
* Call: `stardist`, `3D_spleen`, `scale` 1.0, stored thresholds, with the block fields
  ({doc}`segmentation-contract`, "Block-wise prediction") fixed in the script before the
  run; on the GPU, with one CPU thread. No assignment follows.
* Limits: the process is stopped at 16 GiB RSS or after 3600 s for the call, and a stop is
  recorded as the outcome and not retried larger; artifacts stay within 1 GiB.
* Recorded: peak RSS, framework and process GPU memory, wall and CPU time, the label
  count, the requested and the effective block parameters, and the SHA-256 of the labels.
* Not run: any other whole field of view, a whole-FOV call of Cellpose or of the seeded
  watershed, and a CPU-only whole-FOV call, which is a row with the outcome not run.

The result is one measurement on one host. It sets no default block size and supports no
accuracy statement.

## Limitations

* Every assignment resource figure is code-derived; nothing was measured on a whole FOV.
* The correspondence thresholds are not fitted to data. Exact containment (0.0) withheld
  the compartments of nearly every cell of the W-320 culture crop; the default 0.1 (option
  C2) is provisional, from that one crop without annotation, and is not claimed to suit
  other samples.
* The validation design checks that rules are implemented as written; it measures no
  segmentation or assignment quality, which needs annotation (W-123) and E04.
* The bounded examples need molecule tables that W-305 did not stage.
