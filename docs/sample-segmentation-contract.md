# Sample segmentation contract: segmentation after fusion for projection samples, the route setting and the consistency check of the two routes

Status: Accepted (W-377, 2026-10-09, at 779ea77)

## Amendments after acceptance

* 2026-10-09: `segment_fused` gains the keyword-only `identity=None`, the pair
  `(dataset_id, sample_id)` of its label namespaces ("Names", "The parts on each route"),
  because nothing the function was given holds the dataset's identity; without it the
  labels do not carry that identity. Found by W-426. Operator decision, pending Jiahao's
  confirmation.
* 2026-10-09: check 7 states that `cells` may name a run of either target and `nuclei`
  only a run of the target `nucleus` ("The entry"), in so many words, as the row RS4 of
  {doc}`sample-segmentation-validation` now refuses. Found by W-426. Operator decision,
  pending Jiahao's confirmation.
* 2026-10-10: the status `empty` of `segment` is recorded when the part is called alone;
  through `Dataset.export` with the assemble part on, a placement without an included FOV
  is refused before anything is written, as the accepted "No included FOV" of
  {doc}`sample-output-contract` settles for `fuse`; the `molecules_only` fallback with
  `complete` false is about `not_run`, and `complete` keeps its one rule ("The entry").
  Found by W-427 and its review. Operator decision, pending Jiahao's confirmation.
* 2026-10-10: the `segmentation` section holds `assignment`, the assignment settings the
  sample's tables are made with, set to the current export's when the segmentation is
  reused ("The record"), because `assemble(..., segmentation=…)` has no other argument
  that holds them. Found by W-427 and its review. Operator decision, pending Jiahao's
  confirmation.
* 2026-10-10: L6 on the route after fusion is checked when the number of cells is first
  known, after `segment` and before `assemble` writes on a fresh export, before anything
  is written with a reused segmentation ("The label images"), in place of "L6 holds as
  stated", because the cells do not exist before the segmentation. Found by W-428 and its
  review. Operator decision, pending Jiahao's confirmation.

This page specifies the second route to a sample's cells, so that it can be implemented
after Jiahao approves it: the fused Z maximum of a projection sample is segmented once,
and the sample's molecules are assigned to those cells. It states how the route is called,
which images and settings it takes, how the sample's molecules reach `assign_molecules`,
what every column, key and file of the outputs holds on this route, what the record says,
how a dataset chooses its route, and how the two routes are checked against each other.
The validation design is {doc}`sample-segmentation-validation`. Nothing here is
implemented; this page changes no accepted page and names, in "What changes in the
accepted pages", the sentences an implementation would change.

It builds on these pages and redefines none of them: {doc}`sample-output-contract` (the
export entry, `export/<sample>/`, the fused image, the label export, the SpatialData
export, the read-back checks), {doc}`assembly-contract` (`assemble`, the sample tables,
`sample.h5ad`, the duplicate rule of 2026-10-09), {doc}`segmentation-contract` (`segment`,
`SegmentationInput`, the method configs), {doc}`assignment-contract` (`assign_molecules`,
`MoleculeTable`, the grid rule) and {doc}`stitching-contract` (the placement and the
sample frame). The measurements it uses are those of {doc}`cells-across-fovs-proposal`,
cited there with their limits.

Evidence, cited by path:

* W-399, run directory
  `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-399/20261009T073706Z-367758c2`:
  `notes.md` section 4 (how the block's molecules were assigned on the fused grid and
  where the scripts worked around the package) and `scripts/w399.py` (functions
  `step_fuse`, `step_culture_cellpose`, `fused_molecules`, `step_assign`).
* W-401, run directory
  `/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-401/20261009T212422Z-b4380edb`:
  `scripts/grounding_calls.py` and its record `calls/grounding_calls.json`. Every statement
  on this page about what a present function accepts was checked by one of these calls on
  a small synthetic input (two FOVs of 3 × 24 × 24 voxels, and `Dataset.export` on a
  synthetic dataset of the same two FOVs), cited as "call C1" to "call C25" (with
  letters for the further calls of one statement, such as C11b). A statement about the
  present code that no call checks is marked "read from the source" with its file and
  line, at revision `d24eb11`. No real data was read and no model was loaded or run:
  where a statement would need a model, the call is the refusal `segment` makes before
  any inference. That refusal comes after `segment` has imported the method's libraries:
  it calls `require` before it resolves the model (read from the source,
  `src/python/starfinder/segmentation/_segment.py:261` and `:262`;
  `src/python/starfinder/_registry.py:84–88` imports each dependency), and call C15c
  showed Cellpose and torch imported after a refused `segment` call.

## What Jiahao decides

**Decided by Jiahao on 2026-10-09: every choice below as recommended**, and a dataset with
`projection: sum` is refused on this route. The options stay on the page as the record of
what was considered.

The settled points below are not reopened. The open choices of this page, each with its
options further down and one recommendation:

| # | Choice | Recommendation | Section |
| --- | --- | --- | --- |
| 1 | How the route is called | E1: one new part `segment` in `Dataset.export` and one keyword `segmentation` of `assemble`; on this route the parts run `fuse`, `segment`, `assemble`, `fuse_labels`, `export_spatialdata` | "The entry" |
| 2 | The scale the route segments at | SC1: the fused projection the `fuse` part writes, at the dataset's `scale_yx` | "The stains and the scale" |
| 3 | How pixel lengths of the method settings are given | LS2: in pixels of the acquired grid, as for the route per FOV, divided by the scale factor by the route, both values recorded | "Lengths in pixels" |
| 4 | Where the method settings live | G2: a `segmentation` block inside the `export` block, in the run format of §2.9 with fused image names as inputs | "Where the settings live" |
| 5 | How the sample's molecules reach `assign_molecules` | MT1: a private conversion to one `MoleculeTable` on the segmentation grid; no §2.9 type changes | "The sample's molecule table" |
| 6 | What `fov_id` of a cell holds on this route | F1: the owning FOV of the cell's ownership centroid (options F1 to F3) | "The cells" |
| 7 | How a molecule names its cell | CK1: a new column `cell_namespace` in the sample molecule table on both routes, and every reader links by the cell key | "The molecules" |
| 8 | Which label files the route writes | K1: one OME-TIFF per segmentation run in `export/<sample>/`, and the expanded territories when assign expands | "The label images" |
| 9 | The cell flags on this route | FL1: none applies; `assembly_flags` is empty and the record says so | "The cells" |
| 10 | The default of the route setting | RD1: by `dimensionality`, unconditional; a projection dataset without the segmentation settings is refused, naming them | "The route setting" |
| 11 | Where the consistency check lives | CC1: a package function `compare_routes` that reads two export folders and decides nothing | "The consistency check of the two routes" |

The defaults the operator proposed and this page recommends for Jiahao to confirm: a
sample exported as a projection is segmented after fusion; a sample exported as a 3D stack
is segmented per FOV.

## Settled decisions this page follows

Decided by Jiahao on 2026-10-08 and 2026-10-09 (on {doc}`cells-across-fovs-proposal`).
They are stated as settled and not reopened:

* **Both routes to a sample's cells stay**: segmentation per FOV followed by assembly (the
  **route per FOV**, which exists), and segmentation of the fused image (the **route after
  fusion**, specified here). Segmentation per FOV is the route that still works when a
  thick section cannot be fused on the hardware at hand, and earlier samples were
  processed that way.
* **The route per FOV keeps the duplicate rule** of question 3, option (b), for good
  ({doc}`assembly-contract`, "Retention", rule 8; W-400).
* **A dataset chooses its route by one setting with a default.** The two routes give the
  same outputs: the same files of `export/<sample>/`, the same tables and the same
  `sample.h5ad`, so that what reads them need not know the route. The two routes are
  checked against each other.
* **The route after fusion is for projection samples only, now.** The fused Z maximum,
  reduced in Y and X by a configured factor, is segmented in one call. Segmentation of a
  fused 3D stack block by block, with labels in a chunked file, is not specified now.
* **The segmented image is the Z maximum** (operator decision of 2026-10-09 for Jiahao,
  pending his confirmation). A dataset whose route is `after_fusion`, set or by default,
  and that sets `projection: sum` is refused before anything is written, naming
  `projection` and `route`: one file cannot be both the sum the dataset asked for and the
  maximum the route segments, and a second projection file would be a new output. The
  option that writes no new file and refuses nothing is in "The route setting".
* **A projection sample is assigned by Y and X only** (W-390): the labels are a plane on a
  grid of one plane, and a molecule's z does not select a label (call C7b).
* **The molecules keep their rule**: a position belongs to one FOV. The route after fusion
  takes the sample's molecules from the assembly of the molecule tables (`molecules_only`).
* **No merging of cells across FOVs.**
* Standing preferences (W-329, 2026-10-06): reuse what a library or the package already
  does; one `Dataset.export` and one `export/<sample>/`, no new output folder; choices that
  differ by dataset are configuration; no real-data record in the repository.

Excluded: segmentation of a fused 3D stack, a block-wise driver, a chunked label file,
merging of cells, any change to the rule for molecules, any change to the route per FOV
beyond the one column of choice 7, a model call or a real-data run in the validation.

## Terms

* The **route per FOV** and the **route after fusion** are defined above. A sample's route
  is recorded as `per_fov` or `after_fusion`.
* A **projection sample** is a sample whose export writes the 2D projection
  (`dimensionality` `projection` or `both`, {doc}`sample-output-contract`, "The fused image
  by dataset"). Its tiles may be stacks (cell-culture-3D) or planes (tissue-2D).
* The **segmentation grid** is the grid of the fused projection file the route segments:
  shape `(1, N_y, N_x)`, frame `sample:<sample>`, Y and X spacing `k·s` and origin
  `o + s·(k − 1)/2` of the output level at factor `k`, and in Z the slab
  (`S_z = n0_z·s_z`, `O_z = o_z + s_z·(n0_z − 1)/2`), all as the `fusion` section records
  them ({doc}`sample-output-contract`, "The scale factor"). Call C4 fused two FOVs at
  `k = 2` and recorded `(1, 12, 20)`, spacing `(3.0, 1.0, 1.0)` and origin
  `(1.0, 0.25, 0.25)` µm; call C5 read the same grid back from the file alone.
* A **segmentation run** is one `segment` call (or one import) of the route after fusion,
  named as the §2.9 runs are named.
* The **sample segmentation** is the result of the runs: one label image per run on the
  segmentation grid, the run records and the `segmentation` section of `export.json`.

## Names

All new names are additions; no existing public name, module or format version changes.

| Concept | Name | Kind | Module |
| --- | --- | --- | --- |
| Route values | `ROUTES = ("per_fov", "after_fusion")` | tuple | `starfinder.sample_output` |
| Settings of the route | `SampleSegmentationConfig(runs, cells, nuclei=None, assignment=AssignmentConfig(), device="cpu")` | frozen dataclass | `starfinder.sample_output` |
| One run | `SampleSegmentationRun(name, target, inputs, method, seeds=None)` | frozen dataclass | `starfinder.sample_output` |
| The segment part | `segment_fused(output_dir, *, config, sample, identity=None)` returning `SampleSegmentation` | function | `starfinder.sample_output` |
| Its result | `SampleSegmentation(grid, runs, record)` | frozen dataclass | `starfinder.sample_output` |
| Its loader | `load_sample_segmentation(output_dir)` | function | `starfinder.sample_output` |
| The assembly of this route | the keyword `segmentation: SampleSegmentation | None = None` of `assemble` | keyword | `starfinder.assembly` |
| The consistency check | `compare_routes(per_fov, after_fusion, *, threshold=0.5, output=None)` returning `RouteComparison` | function, frozen dataclass | `starfinder.sample_output` |
| Export settings | `ExportConfig.route`, `ExportConfig.segment`, `ExportConfig.segmentation` | fields | `starfinder.sample_output` |
| Record | the `segmentation` section of `export.json`; the key `route` in the `assembly` section | JSON | — |

`docs/api/sample_output.rst`, `docs/api/inventory.rst` and `docs/api/python-index.rst`
gain the new exports when they are implemented, which `docs/check_reference.py` requires.

## The entry

How the route is called from Python and from the workflow, in the conventions of
`Dataset.export` and the §2.9 entries. Today `ExportConfig` has no route field (call C11), and
`Dataset.export` called on a synthetic dataset runs the four parts `assemble`, `fuse`,
`fuse_labels` and `export_spatialdata` and writes no route and no `segmentation` section
(calls C11b, C12).

### Options

| Option | Python | Workflow | What it changes in the package | What it changes in the outputs |
| --- | --- | --- | --- | --- |
| **E1. One new part of `Dataset.export` (recommended)** | `Dataset.export(sample, *, placement, assignment, config, output_dir)` unchanged; `config.route` chooses the route. On the route after fusion it calls `fuse`, `segment_fused`, `assemble(..., segmentation=…)` with `save_assembly`, `fuse_labels` and `export_spatialdata`, in that order. Each is callable alone. | The one rule `export_sample`; its inputs follow the route. | One part, one function and one keyword of `assemble`; `PARTS` gains `segment` and its section `segmentation`. | The same folder; one more section in `export.json` and the label files of "The label images". |
| E2. A second sample-level entry | `Dataset.segment_sample(sample, …)` writes `segmentation/<sample>/`; `Dataset.export` reads it. | A second rule before `export_sample`. | A method, a folder and a rule. | A second output folder per sample. It buys a separate rule for the GPU step, so that the export itself never needs a GPU node. |
| E3. Segment and assign inside `fuse` | `fuse` segments the image it fuses and assigns. | No new key. | `fuse` gains segmentation, assignment and the sample tables. | The `fusion` section holds cells; a part that "fuses" also counts. |

**Recommendation: E1.** It keeps one entry and one folder, reuses `fuse`, `segment`,
`assign_molecules`, `assemble` and `save_assembly` as they are (calls C4, C6, C7, C3 and
C19 used each on the synthetic input), and adds one part. Its
cost is that the `export_sample` rule needs the device of the segmentation methods (a GPU
for a learned model on `cuda`); with E2 only the segmentation rule would.

### The parts on each route

| Order | Route per FOV (as today) | Route after fusion |
| --- | --- | --- |
| 1 | `assemble` from the saved §2.9 assignment folders | `fuse`: the fused images, the segmentation's stains among them (`images`) |
| 2 | `fuse` | `segment`: `segment_fused` segments the fused projection and writes the label images and the `segmentation` section |
| 3 | `fuse_labels` | `assemble`: the `molecules_only` assembly of the molecule tables, then `assign_molecules` of its retained molecules to the sample segmentation, then `save_assembly`: the four tables, `sample.h5ad` and the `assembly` section |
| 4 | `export_spatialdata` | `fuse_labels`: `cell_labels.ome.tif` from the sample's territories ("The label images") |
| 5 | — | `export_spatialdata`, as on the route per FOV |

On the route per FOV the part `segment` is recorded `off` with the reason `route_per_fov`.
The order differs because the cells of the route after fusion exist only after the fused
image does. Fusion uses no ownership and records an assembly only when one is given
({doc}`sample-output-contract`, "Fused image inputs"), so `fuse` before `assemble` changes
nothing in the fused files.

**What each part does on the route after fusion.**

* **`fuse`** is unchanged. The segmentation's stains are image names of `images`
  (`reference_stain`, `reference_merged` or `<name>=file:<pattern>`), so they are fused
  like any other channel, with the same placement, grid and blending (call C4 fused two
  named stains as projections at `k = 2` in one call).
* **`segment`** (`segment_fused(output_dir, *, config, sample, identity=None)`) reads the fused
  projection of each input image from the `fusion` section (path and SHA-256), builds one
  `SegmentationInput` per run on the segmentation grid, calls `segment` (or
  `import_labels` with that grid for a run whose method is `import`: `segment` refuses a
  `LabelImportConfig`, call C17c, and `import_labels` takes a YX file on the segmentation
  grid, keeps its values with `relabel=False` and refuses a file of another shape, calls
  C17 and C17b), with seeds where a
  run names them, and writes one label OME-TIFF per run and the `segmentation` section. It
  applies no label operation: an expansion is assign's ({doc}`assignment-contract`,
  "Label expansion"), and no extension through Z is applied, because the labels stay a
  plane. Call C6 ran `segment` on a fused projection unchanged: a `(1, 12, 20, 1)` input
  on the segmentation grid, `seeded_watershed` with nuclei on the same grid as seeds and
  the namespace `[dataset_id, sample_id, null, null, "cell"]` gave a `plane` label image;
  the µm setting `sigma_um` 1.5 became 1.5 pixels at 1 µm spacing and 0 in Z.
  `identity` is the pair `(dataset_id, sample_id)` of the label namespaces
  `[dataset_id, sample_id, null, null, <run>]`; `Dataset.export` always passes the
  dataset's. Called without it, the namespace is `[null, <sample>, null, null, <run>]`:
  such labels do not carry the dataset's identity and are for use outside a dataset.
* **`assemble`** with `segmentation` given and molecule-table inputs: it computes the
  `molecules_only` assembly (unchanged rules), converts its retained molecules to one
  `MoleculeTable` on the segmentation grid ("The sample's molecule table"), calls
  `assign_molecules` with the cell run, the nucleus run and the dataset's
  `AssignmentConfig`, and builds the four sample tables of "Outputs, column by column".
  `save_assembly` writes them unchanged (call C19 saved and reloaded a `molecules_only`
  result; call C3 gave its table).
* **`fuse_labels`** writes `cell_labels.ome.tif` from the sample's territories ("The label
  images").
* **`export_spatialdata`** is unchanged in its elements; the readers of "Readers that
  change" apply.

**What the segmentation was made from.** The `segmentation` section records its inputs in
a form that survives a new fusion: per input image the fused file's name and SHA-256, the
segmentation grid (shape, spacing, origin, frame) and the placement's `fovs` and `links`
SHA-256 as the `fusion` section records them. It does not record a hash of the `fusion`
section, because that section holds wall time, CPU time and peak RSS (call C11b), so
fusing again, or turning `fuse` off, would change the hash of a section whose
segmentation is still valid. Likewise the `assembly` and `labels` sections of this route
record the SHA-256 of each label file they read (`segmentation_files`), not a hash of the
`segmentation` section, which also holds times. This follows the rule of
{doc}`sample-output-contract` ("The export entry", the paragraph after "The parts, in
order": "each section records the SHA-256 of the sections it was computed from, so
`check_export` finds a section made from an earlier assembly"), with files in place of
sections where a section holds times.

**What is reused when a part has already run.** One rule, which `Dataset.export`, the
read-back checks (T1 and T5 of "The record") and the validation (SG4) follow:

| Setting | Route after fusion |
| --- | --- |
| `segment: false` | The saved sample segmentation is loaded with `load_sample_segmentation` and recorded `reused`. It is refused before anything is written (`ValueError` naming the folder and the reason) when the folder holds no saved segmentation, when a label file is missing or its SHA-256 differs from the record, when the recorded placement SHA-256 differs from the placement's, when the recorded grid differs from the grid `fused_grid` gives for this export's placement and `scale_yx` (it is known before any fusion, call C22), or when an input image name is not in `images`. `fuse` may be on or off. |
| `segment: false` with `fuse: true` | After `fuse` has written, each fused input file's SHA-256 is compared with the recorded one; a difference records the part `segment` as `failed`, reason `fused_input_changed` with the image name and both SHA-256 values, and raises; the files of `fuse` stay, as for any error inside a part ({doc}`sample-output-contract`, "When a part cannot run"). |
| `segment: false` with `fuse: false` | `fuse` removes the fused files the previous record lists ({doc}`sample-output-contract`, "How the folder is kept equal to its record"; call C11d), so the reused segmentation's inputs are no longer in the folder; its recorded input SHA-256 values stay as provenance. |
| `assemble: false` | Allowed only with `segment: false`: the saved tables are loaded with `load_assembly` and recorded `reused` when the `assembly` section's `segmentation_files` equal the reused segmentation's label files; otherwise refused before anything is written. With `segment: true` the cells may change, so the tables are made again. |
| `fuse: false` with `segment: true` | Refused before anything is written: the part segments the fused images of this export, and `fuse: false` removes them. |

**The statuses of `segment`.** `written`, `reused`, `off` (route per FOV, reason
`route_per_fov`; or a sample without cells, `assignment=None`, reason `no_cells`),
`empty` (no included FOV, reason `no_included_fov`, as `fuse` records, when the part is
called alone on a folder without an included FOV; through `Dataset.export` with the
assemble part on, such a placement has no owning FOV and is refused before anything is
written, as on the route per FOV: {doc}`sample-output-contract`, "No included FOV"), `not_run`
(`part_not_run` when `fuse` did not run; `extra_missing` naming the method's extra, for
example `cellpose`, when its backend is not importable) and `failed`. A model that is
missing or whose SHA-256 differs is a setting error: it is refused before anything is
written (check 7 below). `segment` itself refuses it too, before any inference but after
importing the method's libraries (calls C15, C15b and C15c).

When `segment` is `not_run`, `assemble` still writes the tables of the
`molecules_only` assembly, `sample.h5ad` without a row and its section with `mode`
`molecules_only` and the reason, so that the files of the parts that ran exist, as
{doc}`sample-output-contract` ("When a part cannot run") asks; `complete` is false.
`complete` keeps its one rule (no part `not_run` or `failed`), so a part that is `empty`
does not by itself make it false, as for `fuse`.

**Check 7 before anything is written** (after checks 1 to 6 of
{doc}`sample-output-contract`, "Checks before anything is written"): the route is a value
of `ROUTES` or null; `route: after_fusion` with `dimensionality: stack` or with
`projection: sum` is refused ("The route setting"); on the route after fusion with an `assignment` name, `segmentation` is
given, each run's input images are names of `images`, `dimensionality` writes the
projection, `fuse` is on unless `segment` is off, `assemble` is on unless `segment` is
off, the reuse conditions of the table above hold when `segment` is off, the runs' configs
validate, their
seeds name earlier runs, `cells` and `nuclei` name runs of the targets `cell` (or
`nucleus` as territories) and `nucleus` (`cells` may name a run of either target, `nuclei`
only a run of the target `nucleus`), and every model resolves with its SHA-256
(`resolve_model` called by check 7 itself, not through `segment`: called alone in a fresh
process it refused a missing model and imported neither Cellpose nor torch, call C15c; a
missing file and a SHA-256 mismatch are refused, calls C15 and C15b). Each failure raises `ValueError` (or the
model errors of {doc}`segmentation-contract`) naming the key, and the folder stays
unchanged. What check 7 promises about models: a missing or mismatched model is refused
before anything is written, also on a machine without the method's extra, because
`resolve_model` imports no segmentation library. What it does not promise: that the
method's backend imports; that is `segment`'s `require`, which records the part `segment`
`not_run` with `extra_missing`. An implementation calls `resolve_model` in check 7; the
order of `require` and model resolution inside `segment` need not change, and no choice
of "What Jiahao decides" depends on it.

## The stains and the scale

**Which images.** The runs name fused images by role: for the cell culture, cells from
`cytoplasm: flamingo` with `nuclear: dapi`, nuclei from `nuclear: dapi`, where `dapi`
and `flamingo` are image names of `images` (for example
`dapi=file:images/organelle/DAPI/{fov_id}.tif`), as W-399 fused them. The file of each
FOV is the one `fuse` reads for that name ({doc}`sample-output-contract`, "Fused image
inputs"). The input of `segment` is the fused projection itself, so the segmentation input
is a saved artifact with its SHA-256, as a §2.9 run's `input.ome.tif` is
({doc}`segmentation-contract`, "Method input").

**Which projection.** The fused Z maximum (`projection: max`, the default), as settled. A
dataset on this route that sets `projection: sum` is refused before anything is written
("Settled decisions this page follows", "The route setting").

### Options for the scale

| Option | Rule | What it changes in the package | What it changes in the outputs |
| --- | --- | --- | --- |
| **SC1. The fused projection at `scale_yx` (recommended)** | The runs read `<image>_projection.ome.tif`, which `fuse` writes at the dataset's `scale_yx` (`1/k`, default 0.5). | Nothing beyond reading the files. | No new file. A dataset that segments at 1/4 writes its fused images at 1/4 too. |
| SC2. A separate `segmentation_scale_yx` | `fuse` writes the segmentation's stains also at that scale, as `<image>_segmentation.ome.tif`. | A second output level per stain in `fuse`. | One more file per stain; the written image and the segmented image can differ in scale. It buys a fine fused image beside a coarse segmentation. |

**Recommendation: SC1.** One setting decides the scale of the image and of the cells, and
the segmented image is a file of the folder. On the culture block W-399 found 146 kept
cells at 1/1 and at 1/4, with about the same counts per cell, and the 1/4 calls took 13 s
to 14 s against 18 s to 23 s ({doc}`cells-across-fovs-proposal`, measurement 2: one 2 × 2
block; that they are the same cells is not measured; no accuracy). For another projection
sample, the proposal's rule is the largest whole factor that keeps the cell diameter at
30 pixels or more, Cellpose's working size, a rule not measured on tissue-2D (question 2).

The whole culture projection at full resolution is about 13,636 × 9,576 pixels, which one
call is expected to take; this is not measured ({doc}`cells-across-fovs-proposal`,
question 2). The route segments the whole projection in one call; no block-wise driver is
specified.

### Lengths in pixels

The method settings of §2.9 hold some lengths in pixels of the input grid. On the
segmentation grid a pixel is `k` acquired pixels wide, so a setting given for the
acquired grid has to be divided by `k`. W-399 divided Cellpose's diameters by 4 by hand
(240 and 60 became 60 and 15), and its settings in µm needed nothing
({doc}`cells-across-fovs-proposal`, measurement 2). The pixel fields of the two learned
configs (call C14), and what `StarDistConfig` accepts (call C14b):

| Method | Field | Unit | Value used at factor `k` |
| --- | --- | --- | --- |
| `cellpose` | `diameter` | pixels | `diameter / k` |
| `cellpose` | `min_size` | pixels (an area) | `max(1, round(min_size / k²))` |
| `stardist` | `scale` | a factor of Y and X | `scale · k` in Y and X, Z unchanged; `StarDistConfig` accepts any positive finite scale (it refused 0 and −1 and accepted 8.0), so the product is always accepted |
| `stardist` | `block_size`, `min_overlap`, `context` | pixels | `ceil(value / k)` in Y and X. `StarDistConfig` refuses block fields with a scale other than 1 at construction; a block run is given with `scale` 1, so at `k > 1` its used scale `k` is refused, and block runs are possible at `k = 1` only |
| `stardist` | `n_tiles` | a count | unchanged |
| `seeded_watershed` | `sigma_um` | µm | unchanged: `segment` converts it with the grid's spacing (call C6) |
| `import` | — | — | the file must be on the segmentation grid |
| the assignment's `expansion` | `distance` | µm or pixels | µm unchanged (call C7c expanded the sample cells by 1 µm on the segmentation grid); pixels are refused on that calibrated grid without `legacy_pixel_expansion` (call C7c), and with it divided by `k` |

| Option | Rule | What it changes | Consequence |
| --- | --- | --- | --- |
| LS1. Given as used | The settings are pixels of the segmentation grid. | Nothing. | The settings change with `scale_yx`, and differ from those of the route per FOV for the same method. |
| **LS2. Given at the acquired pixel, divided by the route (recommended)** | The settings are those of the acquired grid, as for the route per FOV; the route computes the values of the table above and records both. | One table of pixel fields per method, kept equal to the registry by a test. | One method setting serves both routes, so the consistency check compares two routes with the same model settings. |

**Recommendation: LS2.**

### Where the settings live

| Option | Settings | What it changes | Consequence |
| --- | --- | --- | --- |
| G1. The per-FOV `segmentation` block, with a map from roles to fused image names | The runs of the §2.9 plan, their `projection` and `operations` dropped. | A map key; rules for dropping what does not apply. | One block for both routes, but a run of the route per FOV (a Z maximum extended through Z, as W-382's culture runs) silently becomes another run. |
| **G2. A `segmentation` block inside `export` (recommended)** | Runs in the format of §2.9 (`name`, `target`, `seeds`, `method` and its keys, LS2 lengths), `inputs` as `{role, image}`; `cells`, `nuclei`, `device`. The assignment settings (`expansion`, `correspondence`, `exclude_cells_without_nucleus`) are those of the dataset's `assignment` block. | One sub-block. | The route's settings are explicit; the assignment settings are shared by both routes. |

**Recommendation: G2.**

```python
@dataclass(frozen=True)
class SampleSegmentationRun:
    name: str                            # snake_case; the label file and the namespace's run
    target: str                          # nucleus or cell
    inputs: tuple[tuple[str, str], ...]  # (role, image name of FusionConfig.images), in channel order
    method: object                       # a SEGMENTATION_METHODS config or a LabelImportConfig; lengths as LS2
    seeds: str | None = None             # an earlier run of the same config

@dataclass(frozen=True)
class SampleSegmentationConfig:
    runs: tuple[SampleSegmentationRun, ...]
    cells: str                           # the run whose labels are the cells
    nuclei: str | None = None            # the nucleus run for correspondence and compartments
    assignment: AssignmentConfig = AssignmentConfig()
    device: str = "cpu"                  # "cpu" or "cuda", as FOV.segment
```

```yaml
export:
  route: null                  # per_fov | after_fusion; null: by dimensionality ("The route setting")
  segment: true                # false: reuse the saved sample segmentation
  images: [reference_stain, dapi=file:images/organelle/DAPI/{fov_id}.tif,
           flamingo=file:images/organelle/Flamingo/{fov_id}.tif]
  scale_yx: 0.25
  segmentation:
    device: cuda
    cells: cell
    nuclei: nucleus
    runs:
      - name: nucleus
        target: nucleus
        inputs: [{role: nuclear, image: dapi}]
        method: cellpose
        model: cpsam_v2
        diameter: 60           # pixels of the acquired grid (LS2): 15 at 1/4
      - name: cell
        target: cell
        inputs: [{role: cytoplasm, image: flamingo}, {role: nuclear, image: dapi}]
        method: cellpose
        model: cpsam_v2
        diameter: 240          # 60 at 1/4
```

The values illustrate the layout (W-382's diameters for the culture, which W-399 used);
they are not recommended settings.

## The sample's molecule table

The sample's molecules are the `molecules_only` assembly of the per-FOV molecule tables:
every molecule of every FOV is retained or excluded by its own position (a position
belongs to one FOV). Its table already keeps each row's `fov_id`, `spot_namespace` and
`spot_id` with `sample_z/y/x` (call C3: six molecules of two FOVs, four retained, two
excluded `not_owner`, no cell). That table is the sample's molecule table; what is
missing is the way into `assign_molecules`. `MoleculeTable` holds the molecules of one
FOV: rows of two `spot_namespace` values are refused (call C2), and `assign_molecules`
refuses labels whose namespace does not begin with the molecules' identity (call C8).

### Options

| Option | How `assign_molecules` takes the sample's molecules | What it changes in the package | What it changes in the outputs |
| --- | --- | --- | --- |
| **MT1. A private conversion (recommended)** | The retained rows become one `MoleculeTable`: `spot_namespace` `[dataset_id, sample_id, null, null]` (accepted, call C1), `spot_id` the row's position in the sample molecule table, `z, y, x = (sample_zyx − O) ⊘ S` on the segmentation grid in float64, not rounded, `gene_id` and the sample's genes. `assign_molecules` runs unchanged (call C7) and returns one row per input in input order ({doc}`assignment-contract`, "Molecule table"), which the route maps back to the original keys by position. | One private function in the route. No §2.9 type or check changes. | None: the written tables keep every molecule's original `spot_namespace`, `spot_id` and `fov_id`; the temporary keys never leave memory, and the record names the conversion and its table's SHA-256. |
| MT2. A public `SampleMoleculeTable` | A new type whose rows keep their own namespaces, and an `assign_molecules` that accepts it. | A new type and a changed §2.9 entry ({doc}`assignment-contract`, "Inputs", check 6). | None beyond MT1. It buys a public sample-level molecule type for other sample-level callers. |
| MT3. A relaxed `MoleculeTable` | `MoleculeTable` accepts several namespaces when its source says so. | A changed accepted validation. | None beyond MT1. |

**Recommendation: MT1.** It is what W-399 did (`fused_molecules`, with `<fov_id>|<spot_id>`
as the identity, which works as well), and call C7 confirms it on the package: the four
retained molecules of two FOVs were assigned on the fused grid, and the cell across the
seam received the retained molecule of FOV1 and the one of FOV2.

**Coordinates.** On a grid of one plane `assign_molecules` samples `y` and `x` only
(`sampled_axes` `yx`), so a molecule's `z` index never makes it `outside_grid`: the same
molecules with every `z` set to 7.3 were assigned exactly as before (call C7b). For a
projection, `(sample_z − O_z)/S_z` lies in `[−½, ½)` for every z of the slab anyway (call
C7: index 0). A molecule outside the segmentation grid in Y or X is `outside_grid`: by the
grid rule of "The fusion grid" this can only happen within one fusion-grid voxel of the
far edge of a FOV's footprint, where the union grid can end one voxel short (derived from
the grid formulas, not measured).

## Outputs, column by column

The route after fusion writes the same tables, with the same columns and dtypes, the same
`sample.h5ad` and the same files. For each column, key and file of the present outputs,
this section says "the same" or what it holds on this route and why. The column lists are
those of {doc}`assembly-contract` ("Sample tables", "Sample-level AnnData") and of the
package (call C13).

### The cells

One row per cell of the cell run's label image, in increasing `cell_id`. Every row is
retained: no cell of the sample segmentation is a second observation, so nothing excludes
one.

| Column | Route after fusion |
| --- | --- |
| `fov_id` | The owning FOV of the cell's ownership centroid (choice 6, F1, below). |
| `cell_namespace` | `[dataset_id, sample_id, null, null, <cell run name>]`: the label namespace of the cell run, with no FOV (accepted by `segment`, call C6). |
| `cell_id` | The cell run's label value, unique in the sample. |
| `obs_name` | The same format, `<sample>_<fov_id>_<cell_id>`; unique because `cell_id` is. |
| `status`, `exclusion_reason` | The same, from the sample assignment (`excluded_no_nucleus` with nuclei and the default exclusion). |
| `size_voxels`, `expanded_size_voxels` | Pixels of the segmentation grid (at `1/k`), not of a FOV's grid. |
| `size_physical`, `expanded_size_physical` | The same meaning: µm² for the plane (call C7: 35 pixels of 1 µm² gave 35.0). On the route per FOV a cell extended through Z (the culture of W-382) has µm³; a projection sample's cells are planes on this route. |
| `centroid_z/y/x`, `expanded_centroid_z/y/x` | Mean pixel index on the segmentation grid (0.0 in Z for the plane), not on a FOV's grid. |
| `n_molecules`, `n_nuclei`, `correspondence`, `correspondence_flags`, `compartments` | The same, from the sample assignment. |
| `sample_centroid_z/y/x` | The same rule: the original territory's centroid in sample coordinates, through the segmentation grid's `index_to_world` (call C23); `sample_centroid_z` NaN for the plane. |
| `sample_expanded_centroid_z/y/x` | The same; NaN unless assign expanded. |
| `ownership_centroid` | The same rule (`expanded` or `original` by `AssemblyConfig.centroid`); it names the centroid `fov_id` is derived from. |
| `sample_ownership_centroid_z/y/x` | The same. |
| `owner_fov_id` | The same rule: the owner of the ownership centroid; equal to `fov_id`, or null as above. |
| `assembly_status` | `retained` on every row. |
| `assembly_reason` | Null on every row. |
| `assembly_flags` | Empty on every row (choice 9, FL1). |
| `conflict_with` | Empty on every row. |
| `duplicate_of` | Null on every row: the duplicate rule does not apply. |

**What `fov_id` holds (choice 6).** On the route per FOV `fov_id` is the FOV that
observed the cell. After fusion no single FOV observed it, but the column is a key that
readers use: `obs_name` is `<sample>_<fov_id>_<cell_id>`; identity 1 counts the cell rows
per `fov_id` against `fovs.n_cells` (read from the source,
`src/python/starfinder/assembly/_assemble.py:166`); the counts table and `sample.h5ad`
carry it; `lookup_source(cell=…)` finds the record's FOV entry by it (read from the
source, `src/python/starfinder/sample_output/_spatialdata.py:755–757`); and
`equivalent_radii` takes the cell's grid spacing by it (call C10).

| Option | `fov_id` on this route | What it changes in the package | What it changes in the outputs | Readers of the present outputs |
| --- | --- | --- | --- | --- |
| **F1. The owning FOV of the ownership centroid (recommended)** | The FOV that `ownership_map` gives for `sample_ownership_centroid_*` (call C21: FOV1 at x 9.7 µm, FOV2 at x 9.8 µm, none at 30 µm); where no owning footprint holds the point, the owning FOV with the nearest footprint centre, ties by table order, and `owner_fov_id` null. | One `ownership_map` call per sample (it exists) and the nearest-centre rule for a point outside every footprint (new). | One real FOV ID per cell. On the route per FOV a retained cell's `fov_id` is also the owner of its ownership centroid ({doc}`assembly-contract`, "Retention", rule 1), so for every counted cell the column has one meaning on both routes. `fovs.n_cells` counts the cells each FOV owns. | Kept: `obs_name`, identity 1, the counts key, `sample.h5ad`, a reader that groups cells by FOV, `lookup_source`'s entry lookup. `equivalent_radii` and `lookup_source` still change for other reasons ("Readers that change"). |
| F2. Null | No value. | `obs_name` needs a rule for a missing FOV (a new format); identity 1 skips the cells; `lookup_source` needs a branch, because its entry lookup finds no FOV entry for a null value (read from the source, `_spatialdata.py:757`). | `fov_id` null on every cell row, count row and `obs` row; `fovs.n_cells` 0 for every FOV. | Broken without changes: `obs_name` by its stated format, `lookup_source(cell=…)`; a reader that groups or filters cells by FOV loses every cell. Kept: readers that only copy the column. |
| F3. The FOVs whose footprints the cell touches | The `;`-joined `fov_id`s of every owning FOV whose footprint holds a pixel of the cell's territory, as `conflict_with` joins names. | A footprint test per cell (new); identity 1, the counts key and `lookup_source` change to split the value. | `fov_id` is no longer one FOV: a cell at a seam holds two or four names, `obs_name` contains `;`, and `fovs.n_cells` sums to more than the cell rows. | Broken without changes: identity 1 (the joined value is no row of `fovs`), `lookup_source`, any join of the cell or count table with `fovs` on `fov_id`; `sample.h5ad`'s `fov_id` becomes a category per combination. It buys a direct view of the cells at seams, which a new column beside F1 would give without breaking the key. |

**Recommendation: F1.** It keeps `fov_id` a key of the `fovs` table with the meaning it
already has for retained cells, and breaks no reader that F2 or F3 would break.

**The cell flags on this route.**

| Option | Rule | What it changes | Consequence |
| --- | --- | --- | --- |
| **FL1. None applies (recommended)** | `fov_edge`, `z_edge`, `conflict` and `unclaimed` are not computed; the record's `flags.computed` is empty and `flags.applicable` is false. | Nothing. | A cell touching the outer border of the sample is not marked; on the culture block 32 cells of the fused image touched it, against 31 retained cells plus 18 at an inner FOV edge on the route per FOV ({doc}`cells-across-fovs-proposal`, measurement 2, one 2 × 2 block). |
| FL2. A new flag `sample_edge` | A cell whose territory touches the first or last pixel of the segmentation grid. | A new value of `CELL_ASSEMBLY_FLAGS`. | It marks cells cut by the sample border, which exist on both routes; on the route per FOV it would need its own rule. |

**Recommendation: FL1**, and FL2 as a later addition for both routes if the cells at the
sample border matter.

### The molecules

One row per molecule of every FOV, in placement table order, then input order, as the
`molecules_only` assembly gives them.

| Column | Route after fusion |
| --- | --- |
| `fov_id`, `spot_namespace`, `spot_id`, `z`, `y`, `x`, `gene_id` | The same: the observing FOV, the §2.9 keys and the index position on the FOV's grid. |
| `voxel_z/y/x` | The sampled pixel on the segmentation grid (`voxel_z` 0), not on the FOV's grid; null for an excluded molecule, which is not assigned. |
| `assignment_status` | The sample assignment's status for a retained molecule; null for an excluded one, as in a `molecules_only` assembly. |
| `cell_id`, `in_expansion`, `original_cell_id`, `nucleus_id`, `compartment` | The same meaning, on the sample segmentation; null for an excluded molecule. |
| `cell_namespace` (new, both routes; choice 7) | The cell's namespace for a molecule with a `cell_id`, null otherwise. |
| `sample_z/y/x`, `owner_fov_id` | The same. |
| `assembly_status`, `assembly_reason` | The molecules' rule, unchanged: retained when its position's owner is its FOV, else `not_owner`; `outside_grid` (outside its FOV's grid, or outside the segmentation grid in Y or X) and `fov_not_placed` as today. A molecule of a cell is not retained "with its cell" (rule 2 of the route per FOV): each position has one observation. |
| `assembly_flags` | Empty: `outside_owner_region` and `cell_unclaimed` cannot arise. |

**How a molecule names its cell.** Today the readers find a molecule's cell by the
molecule's own `fov_id` and its `cell_id`: the SpatialData export's `cell_links` (call C9)
and identities 3 and 4 of `AssemblyResult` (call C16). On the route after fusion a cell's
`fov_id` is one FOV, while its molecules come from every FOV that owns a part of it, so
both refuse the tables: call C9 failed with "links to cell 1 of FOV FOV1, which the sample
cell table does not hold", and call C16 failed at identity 3.

| Option | Rule | What it changes in the package | What it changes in the outputs |
| --- | --- | --- | --- |
| **CK1. A column `cell_namespace` on both routes (recommended)** | The sample molecule table gains `cell_namespace`; every reader links a molecule to its cell by the §2.9 key `(cell_namespace, cell_id)`. On the route per FOV the column holds the FOV's cell namespace. | `assemble`, `check_identities` (identities 3 and 4 keyed by the cell key), `cell_links`, `check_export`. | One column more in `tables/molecules` on both routes; its SHA-256 changes on the route per FOV, and every pinned value of a test that hashes that table changes. A reader that names columns does not fail. |
| CK2. No new column, readers by route | On the route after fusion readers link by `cell_id` alone (one namespace). | The same readers, each with a branch. | No column. A reader outside the package that joins on `(fov_id, cell_id)` gives no cell, without an error, to the molecules of a cell whose `fov_id` is another FOV. |

**Recommendation: CK1.** The link becomes the key the contracts already name, on both
routes.

### The counts

The same columns. `fov_id` is the cell's `fov_id`, `cell_namespace` the cell run's.
Identity 4 holds as stated once it is keyed by the cell key: the whole counts of a counted
cell equal its retained `assigned` molecules.

### The FOVs

| Column | Route after fusion |
| --- | --- |
| `fov_id`, `placement_status`, `origin_z/y/x`, `footprint_*`, `centre_y/x`, `owns_points` | The same. |
| `kind` | `molecules`, the kind of the inputs. |
| `n_cells`, `n_cells_retained` | The cells whose `fov_id` is this FOV (both equal). |
| `n_molecules`, `n_molecules_retained` | The same. |

### `sample.h5ad`

| Part | Route after fusion |
| --- | --- |
| `obs` rows | The counted cells, the same rule (`retained` and §2.9 `kept`), in the cell table's order. |
| `obs` columns | The same names, with the values of "The cells". |
| `var`, `X` | The same. |
| `layers` | `nucleus` and `cytoplasm` when the sample segmentation has a nucleus run. Today `sample_anndata` decides this from the FOV entries of the record (`fovs[*].nuclei`: call C20 gave no layer with every entry false and both with every entry true), which are false for molecule inputs, so it reads the `segmentation` record on this route. |
| `obsm["spatial"]` | The same (`sample_ownership_centroid_x/y`). |
| `uns["assembly"]` | The record, with `route` and `segmentation_files`. |
| `uns["spatial_unit"]`, `uns["frame_id"]`, `uns["spatial_centroid"]` | The same. |

### `export.json`

| Key | Route after fusion |
| --- | --- |
| `format_version`, `stage`, `sample`, `frame_id`, `complete`, `viewer_qualification` | The same. |
| `parts` | The same keys and `segment` (on both routes; `off` on the route per FOV). |
| `assembly` | The same layout ({doc}`assembly-contract`, "What assemble records"), with `mode` `cells`, `route` `after_fusion`, `route_source`, `segmentation_files` (the SHA-256 of each label file it read), the sample assignment's §2.9 record as `assignment`, `config.duplicate_threshold` null and `flags` as FL1; `fovs[*].kind` `molecules`, `assignment` null, `territories` null. On the route per FOV `route` is `per_fov`. |
| `fusion` | The same. |
| `segmentation` | New ("The record"). |
| `labels` | The same layout, with the rule `segmentation` and `segmentation_files` ("The label images"). |
| `spatialdata` | The same. |

### The files of `export/<sample>/`

File by file against the output profiles of {doc}`sample-output-contract` ("The export
folder and the output profiles"). `<fmt>` is `csv` or `parquet`; `<image>` is an image
name of `images`.

| File | Present profile that writes it | Route after fusion |
| --- | --- | --- |
| `export.json` | every profile | The same, with the `segmentation` section. |
| `tables/fovs.<fmt>`, `tables/cells.<fmt>`, `tables/molecules.<fmt>`, `tables/counts.<fmt>` | Default (`csv`); Parquet tables (`parquet`) | The same files, with the content above. |
| `sample.h5ad` | Default | The same file. |
| `<image>_projection.ome.tif` | Default; Sum projection; Another scale factor; Further channels; Pyramid in OME-TIFF | The same; the segmentation's stains are among the image names. The sum projection is refused on this route (settled). |
| `<image>_stack.ome.tif` | 3D stack alone (`dimensionality: stack`); Stack and projection (`both`) | The same with `both`; `stack` alone is refused on this route ("The route setting"). |
| `<image>_projection.ome.zarr/` | Standalone OME-Zarr (`ome_zarr: true`) | The same. |
| `<image>_stack.ome.zarr/` | Standalone OME-Zarr with `dimensionality: both` (or `stack`) | The same with `both`. |
| `cell_labels.ome.tif` | Label export (`fuse_labels: true`) | The same meaning ("The label images"). |
| `sample.zarr/` | SpatialData (`export_spatialdata: true`) | The same elements. |
| the files of the earlier run | Assembly reused (`assemble: false`) | The same, under the reuse rule of "The entry". |
| the files of the parts that ran | A part that cannot run | The same, with the part `segment` among the parts. |
| `segmentation_<run>.ome.tif` | — | New, one per run ("The label images"). |
| `territories.ome.tif` | — | New, when assign expands ("The label images"). |

### Readers that change

Each reader the package has of these outputs, and what it does on the route after fusion:

| Reader | Today | Route after fusion |
| --- | --- | --- |
| `load_assembly`, `AssemblyResult.__post_init__` (`check_identities`) | Identities keyed by `(fov_id, cell_id)` (call C16; read from the source, `src/python/starfinder/assembly/_assemble.py:235` and `:275`) | Keyed by the cell key (CK1); identity 1 compares cells with the sample segmentation's labels instead of per-FOV inputs, and identity 3 lets molecules follow their own position only. |
| `sample_anndata` | `layers` from the FOV entries (call C20) | From the `segmentation` record on this route. |
| `cell_links` (SpatialData) | `(fov_id, cell_id)` | `(cell_namespace, cell_id)` (CK1). |
| `equivalent_radii` (SpatialData shapes) | `size_voxels` with the cell's FOV spacing | With the segmentation grid's spacing: with a FOV's spacing a cell counted at 1/2 would get half its radius (call C10: 2.82 µm against 5.64 µm). |
| `lookup_source(cell=…)` | The FOV's assignment folder and the row's position among the cells of its FOV (read from the source, `src/python/starfinder/sample_output/_spatialdata.py:755–762`); `lookup_source(spot=…)` of a `molecules_only` export returns the FOV, no assignment folder and the row (call C11c) | `assignment_folder` null, `row` the cell's position in the sample cell table, and two added keys, `label_file` (the cell run's file) and `label_value` (`cell_id`). `lookup_source(spot=…)` is the same as for a `molecules_only` sample. |
| `check_export` | R1 to R3, F, L, S | R1 with the part `segment`, L3 and L5 by the rule `segmentation`, S2 with the new column, and the conditions T1 to T5 of "The record". |
| `fov_ownership`, the assignment figures of `export_sample` (W-330) | Redrawn for each FOV whose `fovs.kind` is `assignment` and whose saved assignment is an input (read from the source, `src/python/starfinder/dataset/workflow.py:2090`) | Not drawn: every FOV's `kind` is `molecules` on this route. |

A reader outside the package that reads a column by name does not fail. One that maps
`centroid_*` or `voxel_*` through a FOV's grid, or that joins molecules to cells on
`(fov_id, cell_id)`, gives other results on this route; the record's `route` tells it
which.

## The label images

On the route after fusion the label image is the segmentation itself.

| Option | Files in `export/<sample>/` | What it changes | Consequence |
| --- | --- | --- | --- |
| **K1. One OME-TIFF per run (recommended)** | `segmentation_<run>.ome.tif` per run (uint32, the run's label values, the segmentation grid's geometry, the OME-TIFF layout of {doc}`sample-output-contract`, written by the package's OME-TIFF writer `write_ome_tiff`, which call C18 used for a uint32 label plane read back equal in values and metadata); with an expansion, `territories.ome.tif` (the territories assign sampled, values `cell_id`), written by the `assemble` part. | One writer call per file. | Every label image the tables refer to is a file of the folder with its SHA-256; nothing is duplicated. |
| K2. The §2.9 folders inside the export folder | `segmentation/<run>/` and `assignment/<name>/`, written by the §2.9 writers `write_run` and `write_assignment`, as W-399 did (read from the source, `src/python/starfinder/segmentation/_persist.py:95`, `src/python/starfinder/assignment/_persist.py:53`; `read_assignment` at `:120`). | No writer of its own. | Two folders, and the assignment folder repeats the molecule, cell and count tables of `tables/` in other keys. It buys `read_assignment` and the §2.9 records as files. |

**Recommendation: K1.** No `segmentation_<run>.ome.tif` file name can equal
`cell_labels.ome.tif` or a fused image's name, because only these files begin with
`segmentation_`.

**`cell_labels.ome.tif`** keeps its meaning: the `instance_id`s of the retained cells,
placed without blending at `labels_scale_yx`, written by `fuse_labels` when it is on (off
by default). On this route `fuse_labels` passes the sample's territories (the cell run's
labels, or the expanded territories) with `cell_id` mapped to `instance_id` as its one
view to the same library call (interpolation order 0, `_first_positive`,
`output_spacing`; the call `fuse_labels` makes today is read from the source,
`src/python/starfinder/sample_output/_labels.py:429`), so no second placement is written.
Call C25 passed the sample cell labels of the synthetic input as one view to that call at
twice the spacing: it returned every second pixel of the input, uint32, with no value
outside the input's. The grid therefore has the label scale's spacing and the
territories' origin; the `labels` section records the rule
`segmentation`, and conditions L1, L2 and L4 hold as stated. L6 is checked when the
number of cells is first known, because no cell exists before the segmentation: on a fresh
export directly after the part `segment`, from the number of labels of the cell run, and
before `assemble` writes any file (the files of `fuse` and `segment` stay, as for a part
that fails, and the record names the part `fuse_labels` and the error); with a reused
segmentation (`segment: false`) before anything is written. L3 reads the sample's
territories instead of each FOV's, and L5 checks the spacing and origin of this grid.

The relation of the two route's label images: on the route per FOV `cell_labels.ome.tif`
is drawn from each FOV's retained cells, and two observations of one cell that the
duplicate rule keeps (below 0.5 of the smaller territory) show two IDs; on this route each
cell of the segmentation has one ID by construction.

## The record

The `segmentation` section of `export.json`, written by `segment_fused`:

* `status`, and the reason for `off`, `empty`, `not_run` and `failed`;
* `route`;
* `inputs`: the segmentation grid (shape, spacing, origin, frame, unit, the factor `k`,
  the projection `max`), per input image the fused file's name and SHA-256, and the
  placement's `fovs` and `links` SHA-256 ("What the segmentation was made from"); no
  hash of the `fusion` section;
* per run: `name`, `target`, `geometry` (`plane`), `label_namespace`, the inputs (role,
  image name), the seeds' run, the method record `segment` returns (the uniform
  provenance entry with `artifacts`, `execution` and `effective`; the import record for an
  import), `lengths` (each pixel field as given and as used, LS2), the label file's path
  and SHA-256, `n_labels`, `max_label` and the outcome;
* `cells`, `nuclei`, `device`;
* `assignment`: the assignment settings (the `AssignmentConfig` of
  `SampleSegmentationConfig`) that the sample's tables are made with, which `assemble`
  reads; with `segment: false` it holds the settings of the current export, because the
  labels do not depend on them and the tables do. The `assembly` section's `assignment`
  stays the §2.9 record of the assignment that was made;
* `software` as in `run.json` ({doc}`checkpoints`), wall and CPU time and peak RSS.

The `assembly` section records `route`, `route_source` and, on this route,
`segmentation_files`: per run the label file's name and SHA-256 that it read (and
`territories.ome.tif` when it wrote one). The `labels` section records
`segmentation_files` beside `assembly_sha256`. These are file hashes, compared by T5; R1
compares the section hashes it compares today.

**What `check_export` checks for it** (new conditions, beside those of
{doc}`sample-output-contract`, "Read-back checks"):

* T1. **The segmentation and what it was made from.** Always: the `segmentation`
  section's label files exist with their SHA-256, are uint32 and have the recorded grid's
  shape and metadata. When the `fusion` section is `written`: for each input image it
  lists the fused projection file, that file's SHA-256 equals the recorded input SHA-256
  and its grid equals the recorded grid (shape, spacing and origin within 1e-9 µm);
  otherwise the condition fails naming the image, a segmentation made from another
  fusion. When the `fusion` section is `off`, `not_run` or `empty`, the fused files are
  absent and the recorded input SHA-256 values are provenance only; T1 checks the label
  files alone. This is the rule of "What is reused when a part has already run".
* T2. **Each cell once.** The positive values of the cell run's label image are exactly
  the `cell_id`s of the sample cell table, one row each, all with the run's namespace.
* T3. **Assigned where it lies.** For every retained molecule, the territory image at the
  pixel `floor(j + ½)` of `j = (sample_yx − O_yx) ⊘ S_yx` holds its `cell_id` (0 for an
  `unassigned` molecule), and a molecule outside the grid is excluded `outside_grid`.
* T4. Every cell's `fov_id` and `owner_fov_id` follow the rule of "The cells", recomputed
  with `ownership_map`.
* T5. The `assembly` and `labels` sections' `segmentation_files` equal the label files
  and SHA-256 values of the `segmentation` section (and of `territories.ome.tif`), so a
  table made from another segmentation is found.

R2 lists the new files, and R3 loads the tables with the identities as restated.

## The route setting

One setting of the dataset: `route` in the `export` block, `ExportConfig.route`.

| Value | Route |
| --- | --- |
| `per_fov` | The route per FOV: `assemble` of the saved §2.9 assignment folders, with retention by ownership centroid, the flags and the duplicate rule of W-400 (`duplicate_threshold` 0.5), as `Dataset.export` does today. |
| `after_fusion` | The route after fusion of this page. |
| null (not set) | The default by `dimensionality` below. |

**Default by `dimensionality`**: `projection` → `after_fusion`; `stack` → `per_fov`;
`both` → `per_fov`, because a dataset that asks for the stack is treated as a 3D sample.
The record says whether the route was set or taken from the default (`route_source`
`set` or `default`).

**A 3D stack is refused.** `route: after_fusion` with `dimensionality: stack` raises
`ValueError` before anything is written: "segmentation after fusion is specified for
projection samples only; a dataset with dimensionality 'stack' is segmented per FOV
(route: per_fov)". The reasons: segmentation of a fused stack block by block with a
chunked label file is not built now (decided on 2026-10-09); the fused LN 2 × 2 block
already took 7.7 GB in one call, and a whole LN stack has about 21 times its voxels; and a
stack reduced to 1/2 is no cheaper equivalent for the LN model, which found 1,695 nuclei
there against 8,932 at full resolution ({doc}`cells-across-fovs-proposal`, measurement 3
and question 1: one 2 × 2 block; the whole-stack figure is an extrapolation; no
accuracy). With `both` the route after fusion segments the projection.

**A sum projection is refused** (settled; operator decision for Jiahao, pending his
confirmation). A dataset whose route is `after_fusion`, set or by default, with
`projection: sum` raises `ValueError` before anything is written, naming `projection` and
`route`: the route segments the Z maximum, the dataset asked for the sum as its
projection file, and one file cannot be both; a second projection file would be a new
output.

| Option | A dataset with `projection: sum` | What it changes | Consequence |
| --- | --- | --- | --- |
| **PS1. Refused when its route is `after_fusion` (recommended)** | The error above, for a set route and for the default alike. | One condition of check 7. | No new file; the dataset chooses `route: per_fov` or `projection: max`. |
| PS2. Default `per_fov` for a sum dataset | With `route` null a sum dataset takes `per_fov`; a set `after_fusion` is refused as in PS1. | The default looks at `projection` as well as `dimensionality`. | No new file and no refusal for a dataset that does not set the route; the default then depends on two settings. |

**A sample without segmentation** (`assignment=None`, `assembly.assignment: null`) is a
`molecules_only` export on either route; the route is recorded and `segment` is `off`,
reason `no_cells`.

### Options for the default

| Option | A dataset that does not set `route` | What it changes | Consequence |
| --- | --- | --- | --- |
| **RD1. By `dimensionality`, unconditional (recommended)** | The default above. A projection dataset with an `assignment` name and no `export.segmentation` is refused before anything is written, naming `export.segmentation` and `route: per_fov`. | One rule. | Every existing configuration with `stitching` and `assembly` blocks that exports a projection must add the segmentation settings or `route: per_fov`; none changes its results silently. |
| RD2. By `dimensionality` where the settings exist | `after_fusion` for a projection dataset with `export.segmentation`, else `per_fov`. | One rule with a condition. | Existing configurations keep the route per FOV; the route depends on whether a block is present. |
| RD3. `per_fov` unless set | — | Nothing. | The route after fusion is opt-in; the default does not follow `dimensionality`. |

**Recommendation: RD1.** The default is the outline Jiahao agreed with, and no run
changes route without a key that says so.

## The consistency check of the two routes

A check that runs both routes on the same sample or block and reports. It decides
nothing: no output of either route changes, and no threshold of the report rejects or
accepts anything. It is for finding a fault: a molecule assigned to different cells by two
routes that agree on a cell, a cell that one route has twice, a cell that one route misses
or cuts, or counts that differ for a matched cell. It is not an accuracy: neither route
is a truth, and agreement between them does not show that either is right.

### How it is run

The same sample is exported twice into two folders, once with `route: per_fov` and once
with `route: after_fusion` (`output_dir` of `Dataset.export` is free in Python; the
workflow writes one route per sample). Then:

```python
report = compare_routes(per_fov_folder, after_fusion_folder, threshold=0.5, output=None)
```

`compare_routes` checks that the two folders hold one sample with one placement (the
placement SHA-256 of both `assembly` sections), one gene list and one molecule population
(the same `(spot_namespace, spot_id)` keys), and raises `ValueError` naming the
difference otherwise. It writes nothing unless `output` names a folder outside both export
folders; then it writes the tables below as CSV and the summary as JSON there.

| Option | Where | What it changes | Consequence |
| --- | --- | --- | --- |
| **CC1. A package function (recommended)** | `compare_routes` in `starfinder.sample_output`, called by hand on a block or a sample; no workflow rule. | One function and its result type. | Repeatable on any two folders; tested on fixtures. |
| CC2. A validation script only | Scripts in a run directory, as W-399's. | Nothing in the package. | Not repeatable by a user; not tested. |
| CC3. Both routes inside one export | `Dataset.export` runs both routes. | A second order of processing in every export. | Doubles the cost of every export; Jiahao does not want two orders of processing where one can do. |

**Recommendation: CC1.**

### How two cells are matched

By the measure of the duplicate rule ({doc}`assembly-algorithms`, "Duplicate rule"),
applied between the routes: each route's cells are placed on one grid, the segmentation
grid of the route after fusion, and the shared pixels of a pair of cells are counted.

* A cell of the route after fusion is its territory on that grid as it is.
* A counted cell of the route per FOV is its territory (the one assign sampled) placed as
  the label export places it: a grid pixel `v` shows the FOV pixel `floor(c + ½)` with
  `c = (O + v ⊙ S − origin) ⊘ s` in Y and X inside the FOV's voxel-centre extent; a
  `volume` or `extended` territory shows a cell at `v` when any of its planes does. Each
  cell is placed by itself, so two observations of one cell both count.
* The share of a pair is its shared pixels divided by the pixels of the smaller cell.
  A pair is **matched** when its share is at least `threshold` (0.5, the duplicate
  rule's default) and each cell is the other's partner of highest share (ties by the
  smaller cell key). The threshold only names which pairs the report counts as matched;
  the report also gives every pair with a share above 0.

### What the report holds

* **Summary**: counted cells of each route; matched pairs; cells of each route in no
  matched pair; cells of the route after fusion that share at least `threshold` with two
  or more cells of the route per FOV (a cell seen twice that the duplicate rule kept) and
  the reverse (a cell the route after fusion split); the share distribution in bins of
  0.05; retained molecules of each route and of both.
* **Pairs**: per pair with a share above 0, both cell keys and `obs_name`s, both sizes in
  pixels of the common grid, the share, matched or not, the distance of the two
  `sample_ownership_centroid_*`, the flags of the cell of the route per FOV (`fov_edge`,
  `conflict`) and its `duplicate_of`.
* **Counts per matched cell**: both whole counts, their difference, and the sum over genes
  of the absolute per-gene differences.
* **Molecules**: per molecule key (the two routes hold the same keys), its assembly
  status and its cell on each route (a cell when `cell_id` is not null), and exactly one
  category. The category is decided in this order, so no two can apply: first retention,
  then the cell, then the match.

  | Retained on | Cell on | Category |
  | --- | --- | --- |
  | both routes | both routes, and the two cells are a matched pair | `same_cell` |
  | both routes | both routes, and the two cells are not a matched pair | `different_cells` |
  | both routes | one route only (the route named) | `cell_on_one_route` |
  | both routes | neither route (`unassigned` or `outside_grid` on both) | `no_cell_on_both` |
  | one route only (the route named) | — | `retained_on_one_route` |
  | neither route | — | `excluded_on_both` |

  On the route per FOV a molecule is retained with its cell wherever it lies; on the route
  after fusion only by its position, so the retained populations differ at seams by
  design, and the report counts this rather than hiding it. A molecule in a cell of the
  route per FOV that is not matched only because the route after fusion's cell shares the
  threshold with two of them (a cell seen twice that the duplicate rule kept) is
  `different_cells`; the summary's count of such cells says why
  ({doc}`sample-segmentation-validation`, `fused_pair_seams`).

W-399's comparison of the culture block is an example of what the report holds: 146 kept
cells after fusion against 158 counted cells per FOV, of which 8 are candidate second
observations by the overlap measure ({doc}`cells-across-fovs-proposal`, measurement 2:
one 2 × 2 block; candidates inferred from geometry; no accuracy). Those numbers came from
labels extended through Z and assigned in 3D; the route of this page assigns the
projection's plane labels by Y and X, so its counts on that block are not measured.

## Resources

What W-399 measured on 2 × 2 blocks is in {doc}`cells-across-fovs-proposal`; nothing
here is a new measurement. A whole sample has more seam per FOV than a 2 × 2 block
(`2 − 1/N − 1/M` side lengths per FOV against 1), so the share of cells near a seam is
expected to be larger there, by up to about 1.8 for the two samples measured; this is an
expectation from the geometry, not a measurement. The route after fusion's cost is the
fusion of the stains (part of the default export, W-354), one segmentation call on the
projection and one assignment; the projection's size scales with `k⁻²`. The export rule
needs the segmentation methods' device and memory; for a learned model on `cuda` that is
a GPU node, which is a §2.13 resource of the rule.

## What changes in the accepted pages

Named here sentence by sentence, not edited. An implementation issue changes each
sentence as described and nothing else.

{doc}`sample-output-contract`:

1. Introduction and "Settled decisions this page follows", the "Excluded" paragraph:
   "reassignment of molecules or recomputation of counts from a fused image or a label
   raster" is excluded. The route after fusion assigns the sample's molecules to the cells
   of a segmentation of the fused image; the exclusion stays for the route per FOV.
   The same words in "Exclusions" likewise.
2. "Names": the table of new names under N1 gains the names of "Names" above.
3. "The export entry", "The parts, in order": the table of four parts gains `segment`
   and the order of the route after fusion ("The parts on each route"); the paragraph
   after it, "each section records the SHA-256 of the sections it was computed from",
   gains the file hashes of the route after fusion ("What the segmentation was made
   from").
4. "Checks before anything is written": check 7.
5. "When a part cannot run": rows for `segment`; "The statuses of a part … `reused`
   (assemble only)" becomes "assemble and segment".
6. "The export folder and the output profiles": a profile "Route after fusion" with
   `segmentation_<run>.ome.tif` and `territories.ome.tif`; the rows "3D stack alone" and
   "Sum projection" gain that the route after fusion refuses them.
7. "Assembled label export", first paragraph: "it shows the retained cells of the sample,
   each drawn whole as its FOV segmented it" gains the route after fusion, whose cells are
   drawn as the sample segmentation holds them; "Inputs" gains the sample's territories.
8. "Read-back checks": R1 (the part `segment`), L3 and L5 by the rule `segmentation`, S2
   (the column `cell_namespace`), and T1 to T5.
9. "SpatialData export", "Identities and source lookup": "returns its FOV, the FOV's
   assignment folder, its `assignment.json` SHA-256 and the row's position in the §2.9
   table" gains the cell of the route after fusion ("Readers that change").
10. "Configuration": `ExportConfig` gains `route`, `segment` and `segmentation`.
11. "Workflow configuration": the `export` block gains `route`, `segment` and
    `segmentation`; "runs `Dataset.export` after the placement exists and after
    `reads_assignment` of every FOV of the sample" holds for the route per FOV; on the
    route after fusion the rule needs the molecule tables and the source images, and the
    assignment figures are not redrawn.

{doc}`assembly-contract`:

12. "Settled decisions this page follows", **Retention**: "a cell is retained by its
    centroid; all molecules of a retained cell are kept wherever they lie" holds for the
    route per FOV; on the route after fusion every cell of the sample segmentation is
    retained and molecules are retained by their position.
13. "The assemble entry", "Function": `assemble` gains `segmentation=None`; "It never
    reads or writes files except to load a saved assignment named by path, never runs
    stitch, assign or any per-FOV stage" gains: with a segmentation it calls
    `assign_molecules` once on the sample.
14. "Inputs", "Per FOV, one of two kinds": with a segmentation the inputs are molecule
    tables.
15. "Sample-level identities", **Cells**: "`cell_namespace` already names `[dataset_id,
    sample_id, fov_id, subtile_id, run]`, so the key … traces back to the FOV" gains the
    namespace with a null FOV of the route after fusion.
16. "Count accounting", identities 1, 3 and 4: keyed by the cell key; identity 1 against
    the sample segmentation's labels and identity 3 by position on the route after
    fusion.
17. "Sample tables": `cells` "one row per cell of every input FOV" gains the cells of the
    sample segmentation; the meaning of `fov_id` on that route; `molecules` gains
    `cell_namespace` (both routes).
18. "What assemble records": the keys `route`, `route_source`, `segmentation_files` and
    `assignment`.
19. "Sample-level AnnData": "`layers` When the inputs gave nuclei" gains the nucleus run
    of the sample segmentation.
20. "Workflow configuration": "It runs once per sample, after `reads_assignment` of every
    FOV of the sample" holds for the route per FOV.

{doc}`assignment-contract`:

21. "Boundary with §2.10": "Assignment is FOV-local: it reads one FOV's molecules and
    label images on that FOV's reference grid, and nothing else." The function stays as it
    is; the route after fusion calls it once on the segmentation grid with the sample's
    molecules in one sample-level namespace.
22. "Inputs", **Molecules**: "one FOV identity (the `spot_namespace` list `[dataset_id,
    sample_id, fov_id, subtile_id]`)" gains the sample identity with null `fov_id` and
    `subtile_id`, which the constructor already accepts (call C1).

{doc}`segmentation-contract`:

23. "The segment entry", the parameter `label_namespace`: "a JSON list `[dataset_id,
    sample_id, fov_id, subtile_id, run name]` for a FOV run" gains the sample run with
    null `fov_id` and `subtile_id`, which `segment` already accepts (call C6).

{doc}`stitching-assembly-validation`:

24. Introduction: "This page is the task-group-6 validation design for the whole of
    §2.10" gains a pointer to {doc}`sample-segmentation-validation` for the route after
    fusion.

`docs/migration.md` gains entries for the new names, the `export` keys, the column
`cell_namespace` and the route default (RD1). The status lines of the accepted pages stay
`Accepted`.

## Exclusions

No implementation and no prototype in the package; no segmentation of a fused 3D stack,
no block-wise driver, no chunked label file; no merging of cells across FOVs; no change
to the rule for molecules; no change to the route per FOV beyond the column
`cell_namespace` (choice 7) and the readers' link; no new output folder; no model call, no
GPU and no real data in the validation (every number is cited from
{doc}`cells-across-fovs-proposal` with its limits, and every statement about a present
function from W-401's call record).
