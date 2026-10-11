# Cells seen by two FOVs: measurements and a proposal

Status: Proposed (W-399, 2026-10-09)

At the implementation review W-377 (2026-10-08) Jiahao saw that a cell at a seam can be in
the sample twice, and decided that this has to be handled. A cell is retained by the FOV
that owns its ownership centroid ({doc}`assembly-contract`, "Retention"); a cell cut by a
FOV edge is seen in part by each FOV, the two centroids differ, and both observations can
be retained. This page answers, with measurements on the two cellular 2 × 2 blocks of
W-382 and W-376, the five questions W-399 asks, and makes one recommendation for each.
It proposes; it changes no accepted page, no default and no rule. Jiahao decides.

Three ways were discussed at the review: a rule in the assembly that keeps one of two
observations of a cell; merging observations across FOVs; and segmenting the fused image
and assigning the sample's molecules to those cells. Jiahao agrees in outline with the
third for samples exported as projections and the first for dense 3D tissue, and does not
want two orders of processing if one can do.

**Read every number on this page with four limits.**

* **Every number comes from one 2 × 2 block, not from a whole sample, and a whole sample
  has more seam per FOV than a block.** A regular grid of N × M FOVs has
  (N − 1)·M + (M − 1)·N inner seams of one FOV side each, which is 2 − 1/N − 1/M per FOV:
  1 for a 2 × 2 block, about 1.76 for the cell-culture-3D sample (7 × 10 FOVs, 123 inner
  seams) and 1.73 for LN sample1 (64 FOVs on an 8 × 9 grid with blank cells, 111 inner
  seams, counted from W-376's placement), and up to 2 for a large grid. For the same
  tissue, FOV size and overlap, the share of cells near a seam in a whole sample is
  therefore expected to be larger than in a block, by a factor of up to about 1.8 for
  these samples. This factor is an expectation from the geometry, not a measurement; the
  counts measured on a block are not extrapolated to a whole sample on this page.
* **Pairs are candidates, inferred from geometry.** No annotation exists. A pair of
  retained cells of two FOVs whose territories overlap strongly, with nearby centroids, is
  called here a candidate duplicate (one cell seen by two FOVs, by this measure); that two
  observations are one biological cell is an inference from the overlap of their
  territories and the distance of their centroids, never a measurement.
* **Nothing here measures whether a segmentation is right.** The measurements count
  cells, nuclei, molecules, overlaps and times. No annotation exists, so no number on this
  page is an accuracy, and no segmentation method or scale is ranked. The nucleus labels
  of the cell culture are known to be faulty (one label over the nuclei of two cells,
  W-382 comment of 2026-10-08); they are used as they are.
* **One method per sample, unchanged.** The methods, models and parameters are W-382's:
  StarDist `3D_spleen` block by block for LN, Cellpose `cpsam_v2` on Z maxima extended
  through Z for the cell culture. What differs is only the image (the fused block instead
  of one FOV) and, in the reduced runs, its scale. Nothing was tuned.

## Evidence

The run directory of W-399 is
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-399/20261009T073706Z-367758c2`
(found with `timeout 30 find …/runs/W-399 -maxdepth 2 -name notes.md`). This page cites
its files as "`results/<path>`", relative to that directory. `notes.md` there describes
how each step ran; `results/manifest.json` holds the SHA-256 of every file read and
written; `results/tables/resources.csv` the time, memory and GPU memory of every step.

| Measurement | Tables | Figures |
| --- | --- | --- |
| Inputs, first | `results/inventory.json`, `results/inventory.csv` | — |
| 1. Pairs of retained cells of two FOVs | `results/tables/m1_pairs_{culture,ln}.csv` and `.json` | `results/figures/fig3_pair_overlap_distribution.png` |
| 2. cell-culture-3D after fusion | `results/tables/routes_culture.csv` and `.json` | `results/figures/fig1_culture_cells_two_routes.png` |
| 3. LN sample1 after fusion | `results/tables/routes_ln.csv` and `.json`, `results/tables/m3_nuclei_seams_ln.csv`, `results/tables/m3_nuclei_seam_profile_ln.csv` | `results/figures/fig2_ln_nuclei_two_routes.png`, `results/figures/fig4_ln_nuclei_seam_density.png` |
| 4. Molecules in the overlaps | `results/tables/m4a_strips_{ln,culture}.csv`, `results/tables/m4a_strips_by_line_{ln,culture}.csv`, `results/tables/m4bc_matches_{ln,culture}.csv` | `results/figures/fig5_ln_molecule_seam_density.png`, `results/figures/fig6_culture_molecule_seam_density.png` |

The inputs are the files W-382 and W-376 used, each checked against its record before
the first model call: the DAPI images of the four LN FOVs (`Position048`, `049`, `057`,
`058`), the DAPI and Flamingo images of the four culture FOVs (`Position397`, `398`, `407`,
`408`), their molecule tables, the two models, W-382's tables, and W-376's placements,
block assemblies, label images and pair lists. The LN placement is BigStitcher's, the
culture placement `multiview_stitcher`'s, as W-376 used them.

The terms are those of the two routes this page compares:

* **The route per FOV** is the present one: each FOV is segmented and assigned on its own
  (W-382), and the assembly retains each cell by its ownership centroid (W-376).
* **The route after fusion** fuses the block's stain images with the package's `fuse`,
  segments the fused image once, takes the block's molecules from a `molecules_only`
  assembly (so that each position comes from one FOV) and assigns them to the cells of the
  fused image with `assign_molecules` on the fused grid.

## What the four measurements show

### 1. Pairs of retained cells of two FOVs

For each pair of retained cells of different FOVs whose territories share a voxel on the
sample grid, `results/tables/m1_pairs_{culture,ln}.csv` gives the shared voxels as a share
of the smaller territory and of the union, the distance of the two ownership centroids,
the flags and the molecule counts. The pairs equal W-376's lists, 34 in the culture block
and 93 in the LN block (`pairs_equal_w376` in the `.json`).

| Block | Pairs | Above 0.25 of the smaller territory | Above 0.5 | Above 0.75 | Cells of `sample.h5ad` in pairs above 0.5 | Their counts |
| --- | --- | --- | --- | --- | --- | --- |
| cell-culture-3D | 34 | 8 | 8 | 5 | 16 of 158 (10.1%) | 16,910 of 167,358 (10.1%) |
| LN sample1 | 93 | 5 | 4 | 2 | 8 of 8,194 (0.10%) | 48 of 71,881 (0.07%) |

* **In the culture the shares fall into two groups.** 26 pairs share at most 3% of the
  smaller territory: by this measure two different cells that touch across the seam. 8
  pairs share 63% to 90%, with ownership centroids 1.5 µm to 9.8 µm apart: candidate
  duplicates, one cell seen by both FOVs by this measure (an inference from geometry). No
  pair lies between 3% and 63% (`fig3_pair_overlap_distribution.png`).
* **In LN the shares are small.** 88 pairs share at most 21%: by this measure neighbouring
  nuclei whose expanded territories touch across the seam. 5 pairs share 29% to 86%;
  their centroids are 0.5 µm to 4 µm apart (candidate duplicates by this measure).
* **The review's figures are upper bounds.** The review counted 34 of the 158 cells of the
  culture's `sample.h5ad` in a pair (21.5%); counting the cells of every pair gives 35 of
  158 here (22.2%), and only 16 of them belong to pairs above 0.25: by this measure the
  others touch a cell of the other FOV. In LN the 157 cells of every pair (1.9%) fall to
  10 above 0.25.
* **The present flag `conflict` marks exactly the cells of the pairs above 0.5** in both
  blocks: 16 retained culture cells carry it and 8 retained LN cells, and these are the 16
  and 8 cells of the pairs above 0.5.
* **Every culture pair above 0.25 has an observation cut by a FOV edge** (`fov_edge` on
  one observation in 4 pairs, on both in 4): the cells were segmented with a diameter of
  240 pixels (47 µm), and the overlap of two FOVs is about 27 µm to 30 µm wide. In LN no
  observation of the 4 pairs above 0.5 is cut: a nucleus lies whole inside both FOVs, and
  both observations are retained only when their centroids straddle the boundary.

### 2. cell-culture-3D after fusion

The DAPI and Flamingo images of the four FOVs were fused with the block's saved placement
at full resolution (33 × 2,844 × 2,842 voxels) and at 1/4 in Y and X (33 × 711 × 711).
Cellpose ran on the Z maxima of the fused stacks with W-382's diameters (240 and 60
pixels), divided by 4 at 1/4; the labels were extended through Z with W-382's settings,
which are lengths in µm, so their pixel values fall by the same factor (median window 10
→ 3 pixels, dilation 10 and 5 → 2.5 and 1.25 pixels, areas 200 and 10 → 13 and 1 pixels).
The block's 190,541 retained molecules were assigned on the fused grid.

| | After fusion, 1/1 | After fusion, 1/4 | Per FOV (W-382, W-376) |
| --- | --- | --- | --- |
| Cells | 173 | 168 | 226 in the four FOVs; 185 retained |
| Cells without a matched nucleus (excluded) | 27 | 22 | 37 in the four FOVs; 27 of the retained |
| Kept cells (the cells of a `sample.h5ad`) | 146 | 146 | 158, of which 8 are candidate second observations by the overlap measure (pairs above 0.5) |
| Molecules assigned | 165,155 of 190,541 | 168,086 of 190,541 | 167,358 counts in `sample.h5ad`, from 195,974 retained molecules |
| Counts per kept cell, median (mean) | 1,186 (1,131) | 1,185 (1,151) | 1,117 (1,059) |
| Cells touching the outer border of the block | 32 (24 kept) | 32 (22 kept) | 31 retained; 18 more retained cells touch an inner FOV edge |
| Model calls: wall time (`segment`) | 22.6 s cells, 17.8 s nuclei | 13.3 s, 14.2 s | 8 calls of 34 s to 38 s each (process wall, model loading included) |
| Peak RSS of the calls | 3.1 GB, 3.3 GB | 2.3 GB, 2.3 GB | 2.3 GB to 2.8 GB |
| GPU memory (nvidia-smi peak; torch peak) | 4,144 and 3,056 MiB; 2,095 and 1,760 MiB | 2,214 and 2,906 MiB; 1,760 MiB | 2,548 to 2,904 MiB; 1,760 MiB |
| Extension through Z (CPU, one thread) | 201 s and 154 s | 3.3 s and 2.4 s | 95 s to 192 s per FOV |

Source: `results/tables/routes_culture.csv` and `.json`; the call records under
`results/calls/` (tags `21_` to `24_`, `32_`, `34_`); W-382 `notes.md` ("Segmentation")
for the route per FOV.

**Times measured beside other work.** The steps did not all run one after another: the
Cellpose calls at 1/4 (`23_`, `24_`) ran while the extension `32_culture_extend_k1` ran on
another CPU of the same host (CPU 5; the calls were bound to CPU 1). No step shared a CPU
with another step; the CPUs, start and end of every step and the steps beside it are in
`results/tables/resources.csv` (columns `cpus_bound`, `start_utc`, `end_utc`,
`ran_beside`, `shared_cpus_with`). The disk and the memory bandwidth of the host were
shared, so these times can differ by some percent from a run alone.

* **No cell of the route after fusion is in the table twice** (the script checks that each
  label of the image is one row and each molecule one row: `w399_unique_check.json` in
  each assignment folder). The route per FOV holds 8 candidate second observations among
  its 158 (measurement 1, inferred from geometry).
* **Both scales give the same number of kept cells, 146**, and almost the same counts per
  cell; at 1/4 the model call and the extension take a fraction of the time
  (`fig1_culture_cells_two_routes.png` shows the cells of both scales and of the route per
  FOV on the fused middle plane). That the cells are the same cells is not measured.
* **The assignment ran unchanged on the fused grid**, with one workaround in the script:
  `MoleculeTable` holds the molecules of one FOV (one `spot_namespace`), so the block's
  retained molecules were given one block namespace and the identity `<fov_id>|<spot_id>`.
  A sample-level route needs a molecule table whose rows keep their FOV identities.
* **The faulty nucleus labels act on both routes alike**: 27 cells after fusion and 27
  retained cells per FOV have no matched nucleus and are excluded by the default rule.

### 3. LN sample1 after fusion

The DAPI images of the four FOVs were fused with the block rows of the saved BigStitcher
placement as a stack at full resolution (52 × 2,855 × 2,837 voxels, 421 million) and at
1/2 in Y and X (52 × 1,428 × 1,419, 105 million). StarDist `3D_spleen` ran block by block
on each with W-382's stored thresholds and block fields, the block covering all 52 planes
and the block size kept at 256 × 256 voxels at 1/2. Territories are the nuclei expanded by
0.7776 µm, as in W-382. To compare times like for like, `Position057` was segmented again
as W-382 did, with the 4 threads of the fused runs: its labels equal W-382's byte for byte.

| | After fusion, 1/1 | After fusion, 1/2 | Per FOV (W-382, W-376) |
| --- | --- | --- | --- |
| Nuclei (= cells) | 8,932 | 1,695 | 9,102 in the four FOVs; 8,194 retained |
| Molecules assigned | 76,807 of 306,106 | 47,939 of 306,106 | 71,881 counts in `sample.h5ad` |
| Counts per cell, median (mean) | 8 (8.6) | 26 (28.3) | 8 (8.8) |
| Cells touching the outer border of the block | 91 | 25 | 98 retained; none touches an inner FOV edge |
| Nuclei per 1000 µm² within 20 µm of a boundary / in the rest of the block | 29.5 / 29.2 (ratio 1.01) | 5.1 / 5.6 (0.92) | 27.5 / 26.7 (1.03) |
| Model call: `segment` wall, CPU time, threads | 2,727 s, 4,930 s, 4 | 507 s, 750 s, 4 | `Position057` again: 659 s, 1,141 s, 4; W-382: 805 s to 1,309 s per FOV at 1 thread |
| Time per million voxels (`segment` wall) | 6.5 s | 4.8 s | 5.9 s (`Position057`, 4 threads); 9.4 s (W-382, 1 thread) |
| Peak RSS of the call | 7.7 GB | 3.3 GB | 3.4 GB |
| GPU memory (nvidia-smi peak; TensorFlow peak) | 8,666 MiB; 2,707 MiB | 8,666 MiB; 2,707 MiB | 8,666 MiB; 2,707 MiB |

Source: `results/tables/routes_ln.csv` and `.json`, `results/tables/m3_nuclei_seams_ln.csv`,
`results/tables/m3_nuclei_seam_profile_ln.csv` and the call records (tags
`20_ln_stardist_k1.rerun-…`, `25_`, `26_`); `fig2_ln_nuclei_two_routes.png` (the middle
plane 26 of 52, whole block and a window of 80 µm at the crossing) and
`fig4_ln_nuclei_seam_density.png`.

**Times measured beside other work.** The three StarDist calls ran bound to CPUs 0 to 3
while other steps ran on other CPUs of the same host: `20_ln_stardist_k1.rerun-…` beside
the fusions (CPUs 8 to 11) and the measurement and figure steps (CPU 4);
`25_ln_stardist_k2` beside the culture extensions and assignments and the LN assignment
(CPU 5) and helper steps (CPU 4); `26_ln_stardist_fov_Position057` beside the LN
assignments (CPU 5) and helper steps (CPU 4). No step shared a CPU with a model call; the
session's own short commands ran on CPU 0, one of the four CPUs of these calls, and their
effect is not measured. The disk and memory bandwidth were shared. A difference of 10%
between two of these times is within what such sharing can explain; the times are fit for
"about the same cost per voxel", not for a finer comparison (`results/tables/resources.csv`,
columns `cpus_bound`, `ran_beside`, `shared_cpus_with`).

* **At full resolution, a voxel costs about as much after fusion as per FOV**: 6.5 s against
  5.9 s per million voxels with 4 threads (10% more, within what the work beside the calls
  can explain). The block's four FOVs hold 448
  million voxels against the fused block's 421 million, because each overlap is segmented
  twice per FOV; the two routes took about the same time for the block. The CPU time was 1.8
  times the wall time with 4 threads; W-382 found most of a call's time to be CPU work.
* **At 1/2 the model finds far fewer nuclei**: 1,695 against 8,932. With the block size kept
  in voxels, the nuclei are half as large in voxels as at full resolution, and the model
  does not find most of them (the window in `fig2` shows large outlines over groups of
  nuclei). The run at 1/2 is therefore not a cheaper equivalent for this model.
* **No deficit or excess of nuclei at the seams on either route.** Within 20 µm of a
  boundary the density is 1.01 times (after fusion) and 1.03 times (per FOV) that of the
  rest of the block. The candidate duplicates of measurement 1 (4 pairs above 0.5) are too
  few to show in this density, and the 361 excluded LN cells flagged `unclaimed` do not
  show as a deficit either.
* **The after-fusion run finds 8,932 nuclei where the route per FOV retains 8,194**, 9%
  more on the same block with the same model. Whether either number is right is not
  measured.
* **10 molecules fall outside the fused grid** (`outside_grid`): their sample Z lies outside
  the 52 planes the fusion grid spans. The route per FOV has none.
* **Memory grows with the image**: 7.7 GB for the fused block against 3.4 GB for one FOV.

### 4. Molecules in the overlaps

`results/tables/m4a_strips_*.csv` count the retained molecules of the `molecules_only`
assembly of each block per unit area in 1 µm strips from 20 µm on one side of each
ownership boundary to 20 µm on the other; the crossing of the two boundaries is left out.
`results/tables/m4bc_matches_*.csv` match, for each pair of FOVs whose footprints overlap,
the molecules of one FOV inside the overlap to those of the other as mutual nearest
neighbours of the same gene in 3D, for radii of 0.5, 1 and 2 µm. Nothing is corrected and
no status is changed.

* **The retained density has no step of doubling and no gap at the boundary.** On each
  side it is the density of the FOV that owns that side: in LN about 100 molecules per
  100 µm² on one side and 100 to 120 on the other; all observations together, retained and
  excluded, reach about 210 in the overlap (`fig5_ln_molecule_seam_density.png`,
  `fig6_culture_molecule_seam_density.png`). Where the two FOVs detect different numbers of
  molecules in the same overlap (`Position057` more than `Position048`), the retained
  density steps by that difference (about 14%) at the boundary: that is the detection of
  the two FOVs, not the rule.
* **Only part of the molecules of an overlap are found by both FOVs.** At 1 µm, 49% to 79%
  of the molecules of one LN FOV in a side overlap have a mutual partner in the other FOV,
  and 35% to 56% in the culture; at 2 µm, 64% to 83% and 40% to 57%; at 0.5 µm, 4% to 64%
  and 11% to 38% (`share_of_a_matched`, `share_of_b_matched`). The offsets between matched
  molecules are a few tenths of a µm: in LN the mean offset per axis is at
  most 0.6 µm in Z and 0.3 µm in Y and X, with spreads of 0.2 µm to 0.4 µm.
* **The rule counts very few matched molecules twice or not at all.** A matched pair is a
  candidate for one molecule seen by both FOVs, inferred from position and gene. Of the
  matched pairs at 1 µm in the four side overlaps, both observations are retained in 154 of 22,006 in LN
  and 23 of 12,174 in the culture: counted twice. Neither is retained while both positions
  belong to the two FOVs in 101 and 140: counted by neither. The other pairs that neither
  FOV retains lie where a third FOV owns the position, near the crossing. Against the
  306,106 and 190,541 retained molecules of the blocks these are 0.05% and 0.01% counted
  twice and 0.03% and 0.07% counted by neither.

## Question 1: one route for every sample?

**Decided on 2026-10-09** (Jiahao): both routes stay, segmentation per FOV followed by
assembly and segmentation of the fused image (option C, not the recommendation A); the
route after fusion is built for projection samples, specified by W-401
({doc}`sample-segmentation-contract`), and the 3D block-wise route is not built now.

**Options.**

* **(A) After fusion for every sample.** Each sample's stain images are fused, the fused
  image (projection or stack) is segmented once, the sample's molecules come from a
  `molecules_only` assembly and are assigned to the fused image's cells. Segmentation per
  FOV stays as it exists (`FOV.segment`, `FOV.assign`) for work on single FOVs, and is not
  a route to the sample's cells.
* **(B) Per FOV for every sample**, with the duplicate rule of question 3.
* **(C) Two routes:** after fusion for samples exported as projections, per FOV with the
  rule for dense 3D tissue (the outline Jiahao agreed with).

**What each changes.**

* (A) adds a sample-level segmentation and assignment: a segmentation input on the fused
  grid (the package's `segment` took the fused image unchanged here), a molecule table of
  the sample that keeps each row's FOV identity (`MoleculeTable` holds one FOV today, see
  measurement 2), a sample cell table without FOV keys, and a label image that is the
  segmentation itself. For a large 3D stack it needs block-by-block segmentation on a
  chunked fused stack and a chunked label file (question 2). The outputs gain one
  observation per cell by construction; the per-FOV cellular assembly is no longer needed
  for them (question 5).
* (B) changes only the assembly (question 3); the outputs keep cells cut by FOV edges.
* (C) keeps both orders of processing, and both sets of code and checks.

**The measured cost.** Every number is from one 2 × 2 block; the whole-sample figures are
**extrapolations** from the time per million voxels, not measurements. The model calls ran
while other steps used other CPUs of the host (`20_ln_stardist_k1.rerun-…`, `25_`, `26_`,
`23_`, `24_`; measurement 3, "Times measured beside other work"), so the 10% between the
two StarDist rates is within what that can explain: the two routes cost about the same per
voxel, and no finer statement rests on these times.

| | Time per million voxels | Block | Whole LN sample (extrapolation) |
| --- | --- | --- | --- |
| StarDist after fusion, 1/1, 4 threads | 6.5 s | 2,727 s for 421 M voxels | 8,849 M voxels (66 × 12,334 × 10,870, the grid of W-376's LN stack): about 57,000 s (16 h) on one GPU |
| StarDist per FOV, 4 threads (`Position057`) | 5.9 s | about 2,640 s for 448 M voxels (four FOVs at 659 s; only `Position057` measured) | 64 FOVs × 112 M = 7,162 M voxels: about 42,000 s (12 h), in independent jobs |
| Cellpose after fusion, culture projection 1/1 | 23 s and 18 s for 8.1 M pixels | 41 s for the two calls; extension 355 s | — |
| Cellpose after fusion, culture projection 1/4 | 13 s and 14 s for 0.5 M pixels | 27 s; extension 6 s | — |

* The fused LN grid is larger than the sum of the FOVs (8,849 against 7,162 million voxels)
  because its bounding box holds the blank cells of the LN grid; the fused route pays for
  them unless the blocks of blank voxels are skipped. Per voxel imaged, the two routes cost
  about the same (the measured rates differ by 10%, within what the work beside the calls
  can explain).
* **A whole tissue-2D sample treated as 3D (extrapolation, with an assumed plane count).**
  56 tiles of 3,072 × 3,072 pixels in 7 × 8 with 10% overlap make a fused plane of about
  19,661 × 22,426 pixels (441 million). At 6.5 s per million voxels, each plane costs about
  2,900 s (0.8 h), so 30 planes, an assumption, would take about 24 h on one GPU. Its pixel
  (0.0946 µm) is half LN's, and the LN model is not calibrated to it; the figure is a
  voxel count times a rate, nothing more.
* Memory, not time, is the limit of the fused route for 3D: the fused block's call peaked
  at 7.7 GB RSS; a whole LN stack has 21 times its voxels, so it cannot be segmented as one
  array in memory (W-376's whole LN stack was written at 1/2 only).

**Not measured.** A whole sample by either route; block-by-block segmentation with a
chunked label file; the per-FOV times of the other three LN FOVs at 4 threads; the
tissue-2D sample; any accuracy.

**Recommendation.** (A): segmentation after fusion as the one route to a sample's cells,
with segmentation per FOV kept only as it exists. The measured cost per voxel is about the
same, the route after fusion holds no cell twice, and the molecules keep the present rule
(question 4). A whole sample has more seam per FOV than the block (first limit), so the
candidate duplicates the route per FOV leaves are expected to be a larger share there,
which adds to this recommendation rather than weakening it. For projection samples it can
follow now. For dense 3D tissue it needs the block-by-block segmentation and the chunked
label file of question 2 first; until those exist, the duplicate rule of question 3 is the
interim for 3D samples, not a second route kept for good.

## Question 2: how to segment after fusion

**Decided on 2026-10-09** (Jiahao): for projection samples, the fused Z maximum reduced
in Y and X is segmented in one call (specified by W-401); the block-by-block segmentation
of a fused 3D stack with a chunked label file is not built now.

**Options and measurements.**

* **Image and scale, projection samples (cell culture).** At full resolution and at 1/4 in
  Y and X the culture gave the same 146 kept cells and about the same counts per cell
  (measurement 2). Cellpose resizes its input by 30 / diameter before the network, so with
  the diameters divided by the scale the network sees an image of the same size at both
  scales (cells: 1/8 of the full-resolution projection, or 1/2 of the 1/4 projection);
  only the detail lost in the reduction differs. The 1/4 run took 13 s to 14 s per call
  and 6 s for the extension, against 18 s to 23 s and 355 s.
* **Image and scale, dense 3D (LN).** At 1/2 the StarDist model found 1,695 nuclei against
  8,932 (measurement 3): a reduced stack does not serve a model trained at full resolution.
  Z is never reduced (accepted decision), so 3D runs at full resolution.
* **Whole or block by block.** The LN calls already run block by block inside one process
  (`predict_instances_big`, blocks of all planes × 256 × 256 voxels with an overlap of 64
  and a context of 52 voxels, about 12.4 µm and 10 µm, against nuclei of 5 µm to 8 µm). The
  whole fused LN block took 7.7 GB; a whole LN stack in one array would need about 21
  times that. A whole culture projection at full resolution is about 13,636 × 9,576 pixels
  (130 million, from W-376's projection at 1/2), 16 times the block's, which one call is
  expected to take; this is not measured.
* **Block overlap.** Larger than the largest object, as StarDist requires (`min_overlap`
  above the object size): the present 64 voxels (12.4 µm) for LN nuclei; for culture cells
  of up to about 47 µm, more than 47 µm at the segmented scale if a projection is ever
  split.
* **Label file of a large 3D sample.** One OME-Zarr label image (OME-NGFF 0.5 on Zarr
  format 3, the format the store already uses), `uint32`, chunks of 16 × 512 × 512 voxels as
  the standalone fused OME-Zarr, written block by block. A whole LN stack at full
  resolution is 8,849 million voxels, 35 GB uncompressed; the saved per-FOV label files of
  W-382 compress about 100 times (4.4 MB for 450 MB). A single OME-TIFF is not proposed for
  3D: it cannot be written by blocks of a segmentation running out of order.

**What it changes.** For projections: none beyond question 1 (the segmentation input is the
fused projection at a configured scale). For 3D: a block-wise segmentation driver over a
chunked fused stack that writes labels into the OME-Zarr and resolves labels across blocks
(StarDist's own `predict_instances_big` does this inside one process; across processes it
is new), and a `labels` entry of the export that names the file.

**Not measured.** Any block-wise run across processes, the OME-Zarr label writer, a whole
culture projection in one call, and whether the cells at 1/4 are the same cells as at full
resolution.

**Recommendation.** Projection samples: the fused Z maximum reduced in Y and X with the
diameters divided by the same factor, the whole projection in one call; for the cell
culture 1/4 (measured). For another projection sample, the largest whole factor that keeps
the cell diameter at 30 pixels or more, Cellpose's working size (a rule, not measured on
tissue-2D). Dense 3D: the full-resolution
fused stack, block by block with StarDist's present blocks (all planes × 256 × 256, overlap
64, context 52), labels written into one chunked OME-Zarr label image.

## Question 3: if cells of several FOVs stay a route, the rule that keeps one observation

**Decided on 2026-10-09** (Jiahao): option (b), as a part of the route per FOV for good;
implemented by W-400 ({doc}`assembly-contract`, "Retention", rule 8).

**Options.**

* **(a) No rule** (the present state): both observations stay; `conflict` marks them.
* **(b) Keep one of two observations by their overlap.** Among retained cells of
  different FOVs, a pair whose shared voxels are at least a share *t* of the smaller
  territory is taken as one cell (a candidate duplicate, inferred from geometry); one
  observation stays retained, the other becomes excluded with
  a new reason.
* **(c) Merge the two observations** into one cell (union of territories, molecules of
  both). Not proposed: it needs a new cell key across FOVs, a rule for molecules seen by
  both FOVs and a new label image, and the issue excludes prototyping it.

**The measure, the threshold and which observation stays (option b).**

* **Measure:** shared voxels on the sample grid divided by the smaller territory, as
  `results/tables/m1_pairs_*.csv` computes it. The union share gives the same split here
  but is smaller for a cut cell, whose part is much smaller than the whole.
* **Threshold:** *t* = 0.5, one of the three values the issue fixed before the
  measurement. **The choice among them is informed by the result**, and this page says so:
  in the culture no pair lies between 0.03 and 0.63, so 0.25, 0.5 and 0.75 would give 8, 8
  and 5 pairs; in LN 0.5 keeps the 4 pairs whose centroids are at most 2.2 µm apart,
  consistent with one nucleus seen twice (an inference from geometry, not measured) and
  leaves out one pair at 0.29 with centroids 4 µm apart. At 0.5 the rule picks exactly the
  cells `conflict` already marks in both blocks, so it can be stated as "`conflict` and at
  least half of the smaller territory shared".
* **Which observation stays:** the one with the larger territory on the sample grid, ties
  by placement table order. A cut observation is smaller than an uncut one, so this keeps
  the more complete view. An alternative is the observation without `fov_edge`; in the
  culture both carry `fov_edge` in 4 of 8 pairs, so it does not decide alone.
* **The molecules of the other observation** become excluded with it (reason, for
  example, `duplicate_observation`), as the molecules of a `not_owner` cell are today. They
  are not given to the kept cell: the kept cell's FOV imaged the same region and assigned
  its own molecules there. A molecule of the excluded observation outside the kept
  territory stays excluded; this is counted in the record.

**What it changes.** In the package: `assemble` gains the pair computation (the territory
images placed on the sample grid, as W-376's `scripts/overlap.py` and W-399's
`scripts/analysis.py` do), a configuration value for *t*, an assembly reason and a count
in the record; `check_identities` and the accounting gain the reason. In the outputs of
these blocks (`rule_reading` in `results/tables/m1_pairs_*.json`, read from the table, not
applied): 8 of the 158 cells of the culture's `sample.h5ad` would drop, with 7,994 of its
167,358 counts (4.8%), and 4 of the 8,194 LN cells, with 25 of 71,881 counts (0.03%); the
fused label image by retained cells loses the rim or half of another colour at those
cells. These shares are a 2 × 2 block's. A whole sample has more seam per FOV, so its
share of candidate duplicates, and of the counts they hold, is expected to be larger, by a
factor of up to about 1.76 (cell culture, 7 × 10 FOVs) and 1.73 (LN, 64 FOVs on 8 × 9),
from the seam count of the first limit; this is an expectation, not a measurement.

**What it leaves wrong.** The kept observation of a culture cell cut by its FOV's edge is
still the part that FOV saw: the rule removes the candidate second observation, it does
not complete the cell. In the culture 6 of the 8 kept observations would carry `fov_edge`. A cell retained
by no FOV (`unclaimed`: 6 of 41 excluded culture cells, 361 of 908 excluded LN cells) is
not affected.

**Not measured.** Whether a pair above 0.5 is the same biological cell (no annotation), and
the rule on a whole sample.

**Recommendation.** If cells of several FOVs stay a route, option (b) with the smaller
territory share, *t* = 0.5, the larger observation kept, and the other's molecules excluded
with it.

## Question 4: the molecules of an overlap

**Options.**

* **(a) A position belongs to one FOV** (the present rule, nearest footprint centre): each
  observation is retained when its own position is owned by its own FOV.
* **(b) Scan order:** the earlier FOV keeps all its molecules in an overlap. It moves the
  boundary from the middle of the overlap to its edge; counted twice and not at all happen
  in the same way, at another line.
* **(c) Match and merge** the observations of two FOVs (mutual nearest neighbours of the
  same gene within a radius) and keep one of each pair.

**What the measurement says** (measurement 4): the present rule neither doubles nor drops
density at the boundary; matched observations counted twice are 0.05% (LN) and 0.01%
(culture) of the retained molecules of a block, and counted by neither 0.03% and 0.07%.
Matching finds a partner for only 35% to 79% of the molecules of an overlap at 1 µm, and
the share changes strongly between 0.5 µm and 1 µm, so option (c) would remove only part
of the candidate duplicates it aims at and would have to choose a radius; it adds a second,
uncertain decision for a small effect. Option (b) has no advantage in these numbers and
makes the kept molecules of an overlap depend on the acquisition order.

**What it changes.** Nothing under (a). Under the recommendation for question 1, the
after-fusion route takes its molecules from a `molecules_only` assembly, so the rule is
the one route for molecules in both orders of processing.

**Not measured.** Duplicates whose two observations are more than 2 µm apart, and a whole
sample.

**Recommendation.** Keep (a), "a position belongs to one FOV".

## Question 5: what becomes of the parts of §2.10 for cells of several FOVs

The parts that exist for cells of several FOVs are the cellular assembly (`assemble` with
assignment inputs, mode `cells`, retention by ownership centroid), the flags `fov_edge`,
`z_edge`, `conflict`, `unclaimed`, `outside_owner_region` and `cell_unclaimed`, and the
fused label image drawn by retained cells (rule LR4, `fuse_labels`).

**Decided on 2026-10-09** (Jiahao): both routes stay (question 1), so these parts stay
with the route per FOV, with the duplicate rule of question 3, and the route after fusion
uses the `molecules_only` assembly; the 3D block-wise route is not built now.

**Options.**

* **(a) Keep them all as they are** beside a new route after fusion.
* **(b) Under the recommendation of question 1:** the route after fusion uses the
  `molecules_only` assembly (unchanged) and its own cell table and label image; the
  cellular assembly, its flags and LR4 stay for samples segmented per FOV, which in the
  recommendation are the 3D samples until the block-wise route exists, with the rule of
  question 3 added; they are retired for a sample type once its route after fusion exists.
* **(c) Remove them now.** Not proposed: 3D samples need them until question 2 is built.

**What each changes.** Under (b): `assemble` gains the duplicate rule and its reason;
`conflict` keeps its rule and is the candidate set of the duplicate rule (it marked exactly
the pairs above 0.5 in both blocks, measurement 1); `unclaimed` keeps its rule and meaning
(cells possibly missing, which the duplicate rule does not touch); `fov_edge` keeps marking
cut cells; `fuse_labels` keeps drawing retained cells, now without the candidate second
observations.
For a sample on the route after fusion, `fuse_labels` is not run: the segmentation of the
fused image is the label image, and the cell flags do not apply. No existing name, format
version or default changes; the new parts are additions.

**Not measured.** The flags on a route after fusion (they are not defined there), and how
often `unclaimed` cells are truly missing.

**Recommendation.** (b): keep the cellular assembly, the flags and the label image by
retained cells for the samples segmented per FOV, with the duplicate rule of question 3;
the route after fusion uses only the `molecules_only` assembly; retire the cellular parts
for a sample type when its route after fusion exists.

## Statements of accepted pages this proposal puts in question

These are named, not edited:

* {doc}`assembly-contract`, "Settled decisions this page follows" ("Retention"): "There is
  no matching or merging of cells across FOVs". Option (b) of question 3 matches
  observations by their overlap, without merging them.
* {doc}`assembly-contract`, "Flags": "No matching, merging or reassignment repairs them".
* {doc}`sample-output-contract`, the exclusions of its introduction: "cross-FOV cell
  matching" is excluded. Under option (b) the label image of "Assembled label export"
  still shows the retained cells, but a candidate second observation would no longer be
  retained;
  the rule LR4 itself does not change.
* {doc}`sample-output-contract`, the same exclusions: "reassignment of molecules or
  recomputation of counts from a fused image or a label raster". The route after fusion of
  question 1 assigns the sample's molecules to the cells of a segmentation of the fused
  image, which this exclusion of the first delivery leaves out; {doc}`assembly-contract`
  defines no sample-level segmentation either.

## What was not measured

* Any whole sample: every number is from a 2 × 2 block; whole-sample costs are
  extrapolations, marked as such.
* Accuracy of any segmentation, of the assignment or of the duplicate rule.
* Any second segmentation method or parameter set, and the tissue-2D sample.
* The behaviour of the after-fusion route where FOVs differ in brightness or focus, beyond
  what the two blocks show.

## Cells retained by no FOV, measured (this issue)

W-434 (2026-10-10) measured, on the same two blocks, the cells that their own FOV
excludes `not_owner` and whether any FOV retains them. It measures; it decides and
implements no rule. Its run directory is
`/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-434/20261010T234018Z-ca5cf87f`;
its worker notes, tables (`results/tables/`), figures (`results/figures/`) and manifest
(`results/manifest.json`) are there.

**Method.** The present `assemble` was run on W-376's block inputs and placements, once
with the duplicate rule off (every cell then has the status, reason and flags of W-376's
block export, which W-399 measured) and once at its default 0.5, which excludes 8 culture
and 4 LN observations `duplicate_observation`, one per candidate pair of W-399. Every cell
excluded `not_owner` in the second assembly is put in one class by the owner FOV's
territory at its ownership centroid, read as the package reads it for `unclaimed`: the
owner retains a cell there; the owner has no cell there; the owner's cell there is
excluded `duplicate_observation`; or it is excluded `not_owner` and owned by this cell's
FOV (the two observations exclude each other). The classes without a retained cell equal
the `unclaimed` flag in both blocks, with no disagreement. A cell has **no counterpart**
when it shares less than 0.5 of the smaller territory, the duplicate rule's measure, with
every retained cell of any FOV. A synthetic pair of FOVs with a known answer gave the
expected class and counterpart for each of its cells.

| Class (assembly with the duplicate rule on) | Culture: excluded | Culture: no counterpart (0.3 / 0.5 / 0.7) | LN: excluded | LN: no counterpart (0.3 / 0.5 / 0.7) |
| --- | --- | --- | --- | --- |
| Owner retains a cell there | 32 | 0 / 1 / 3 | 547 | 1 / 24 / 109 |
| Owner has no cell there | 4 | 2 / 3 / 3 | 348 | 339 / 346 / 347 |
| Owner's cell there excluded `duplicate_observation` | 3 | 0 / 0 / 1 | 0 | — |
| Both observations exclude each other | 2 | 2 / 2 / 2 | 13 | 13 / 13 / 13 |
| All (`unclaimed`: culture 9, LN 361) | 41 | 4 / 6 / 9 | 908 | 353 / 383 / 469 |

* **What they are.** In the culture, 5 of the 6 cells without a counterpart carry
  `fov_edge`: they are parts of cells cut by their own FOV's edge, which the owner
  segmented differently. In LN they are nuclei of ordinary size (median 0.98 of the
  retained median), spread over the whole overlap past the boundary (median 7.0 µm into
  the owner's region) and over Z. They carry `fov_edge` and `z_edge` less often than the
  excluded cells that have a counterpart (9% against 12%, and 30% against 41%). What
  distinguishes them is that the owner's segmentation has no
  nucleus there; in the panels the owner's stain often shows one. The segmentation of
  the fused block (W-399, scale 1/1) has a cell sharing at least 0.5 with 114 of the 383,
  against 464 of the 525 excluded LN cells that have a counterpart. That segmentation is
  a second observation, not a truth.
* **Molecules and counts.** Every molecule a FOV assigned to such a cell is excluded with
  it (`cell_not_owner`). In LN they are 3,390 molecules, 4.7% of the molecules in
  retained cells, and the cells are 383 of the 8,190 cells (4.7%) and 4.7% of the counts
  of `sample.h5ad`. The owner FOV's own molecules at those places are mostly retained
  without a cell (3,241 of 4,311). In the culture they are 3,494 molecules (2.0%) and
  5 of 150 cells with 2.1% of the counts.
* **What two rules would bring back** (counted, not built). Rule r1 keeps an excluded
  observation that no retained cell shares at least 0.5 with, and where two such
  observations share at least 0.5 with each other it keeps one by the duplicate rule's
  order, whether or not they exclude each other at their centroids. Rule r2 does the same
  for observations without `fov_edge` only. In the culture, r1
  returns 5 cells (4 cut by a FOV edge, 1.3% of the counts) and r2 returns 1. In LN, r1
  returns 369 nuclei (33 cut by a FOV edge, 4.5% of the counts) and r2 returns 337 (4.2%).
  Neither creates a pair at or above 0.5 among the cells it returns. If the order were
  applied only to observations that exclude each other at their centroids, LN would get
  377 and 342 cells back, and 9 and 5 pairs at or above 0.5 would remain among them.
  Of the 383 LN cells without a counterpart, 27 are one of two excluded observations of
  the same place, and 356 were seen by one FOV only. In the culture the figures are 2 and 4.

**Limits of these numbers.**

* They are counts on two 2 × 2 blocks. A whole sample has more seam per FOV
  (2 − 1/N − 1/M side lengths against 1), so the shares are expected to be larger there.
  This is an expectation, not a measurement.
* No annotation says whether a label is a real cell. These are counts, not accuracies.
* Jiahao considers the culture's nucleus labels faulty (one label over the nuclei of two
  cells). The LN nucleus segmentation is incomplete, which is a likely reason why an owner
  has no nucleus where a neighbour has one. Whether a returned observation is a real
  nucleus was not measured.
* Not measured: a whole sample, and which rule, if any, is right. Jiahao decides on a
  rule.
