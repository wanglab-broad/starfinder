# Expected values of the golden baseline's stitching, export and report

The rest of the golden baseline of the current Snakemake workflow (`backend: python`): stitching
with `multiview_stitcher`, the export on each route the fixture allows and the report after the
export, which the workflow orchestration refactor must reproduce (W-643, open point P1 of W-642;
decision D10 of the §2.13 planning; `design/specs/orchestration.md`, "The rest of the golden
baseline"). It sits beside W-642's `test_workflow_orchestration_golden/`, which pins the per-FOV
core, morphology, assignment and the BigStitcher macro and is not changed. No test reads these
files yet: the comparison test lands with the lock change that adds Snakemake 9 to a `workflow`
dependency group. Until then `scripts/workflow_golden.py compare` is the comparison.

They were recorded by W-643 with `src/python/scripts/workflow_golden.py` (SHA-256
`34cc99152706b7357d65db2972d190e7843629debdfaea21208635fd691255dd` when the runs were made; its
later change touches the comparison only) at Starfinder `56c35b8b560bfac7bd8b03e0ec7b308ead9a7d89`
(the package of the worktree equal to it), on host GP099-29C (Intel Xeon Gold 6234), one CPU and
one numerical thread. Two environments:

* the fixture is generated and the outputs are read in the locked environment
  `dev-s210-export-20261007T050852Z` (Python 3.12.12, NumPy 2.2.6, pandas 3.0.0);
* the workflow runs in the golden scratch environment `w642-golden-20261029T1750Z`: the project
  lock at `56c35b8` with the extras dev, ome, local-registration, checkpoint, anndata,
  multiview-stitcher and spatialdata, and Snakemake 9.15.0 outside the lock (identity
  `runs/W-642/20261029T1750Z-golden-env/environment.json`); its rule scripts import Starfinder
  from the checkout that `starfinder_path` names.

From `src/python`, with `PYTHONPATH` naming this checkout's `src/python`:

    python scripts/workflow_golden.py fixture <work>
    python scripts/workflow_golden.py run-export <work> --snakemake <Snakemake 9 executable>
    python scripts/workflow_golden.py extract-export <work> <expected folder>
    python scripts/workflow_golden.py compare <this folder> <expected folder>

The fixture is W-642's (`sample_scene_preset("sample-tiny-v1")`, seed 42). Invocation D makes
three Snakemake invocations (`--cores 1`, target `all`), one per export route, each in its own
output root: `golden-per-fov` (configuration B of W-642 with the report block and `export:
{route: per_fov, images: [reference_merged], fuse_labels: true, labels_scale_yx: 0.5}`, the truth
labels copied first as the legacy label files), `golden-molecules-only` (configuration A with the
stitching block, `assembly: {assignment: null}`, `export: {images: [reference_merged]}`) and
`golden-after-fusion` (configuration A with the stitching block and an export segmentation that
imports one cell run: the Z maximum of the per-FOV route's fused cell labels on the fused
projection grid). The stitching block is B's with `config: {binning_zyx: [1, 2, 2],
quality_threshold: -1.0}`. A second run of all of it in the same environment reproduced every
value.

The files, in the layout of W-642's folder, with each name prefixed by its route
(`per_fov`, `molecules_only`, `after_fusion`):

* `values.json`: the fixture's file digests; the shape, dtype, minimum, maximum and sum of each
  TIFF of the export folders and of the import file (`labels/fused-cells.ome.tif`); and the
  SHA-256 of each file of `stitching/`, `export/` and `report/` that holds no time or absolute path
  (`compare: same_environment`).
* `records/<route>/`: the JSON records of `stitching/`, `export/` and `report/`, normalized as
  W-642's (no fields of where, when and with which software, no times or memory, the work folder
  written `<work>`), and without the digests of the files that hold absolute paths:
  `placement.json`, `export.json`, `sample.h5ad`, and the export record's `assembly_sha256`.
* `<route>-stitching-*.csv`, `<route>-export-*.csv`, `<route>-report-*.csv` with `dtypes.json`:
  the tables, without the report's time columns, the report's intensity histograms as one row per
  histogram.
* `arrays.npz`: every integer image whole (the fused projections, the fused and imported labels,
  the segmentation of the route after fusion).

The comparison follows `docs/contributing.md` ("Expected values and tolerances"), as W-642's:
integers, strings, booleans, integer images and the fixture's digests exactly, computed float64
values with `rtol=1e-9`, `atol=1e-12`, and the file digests only in the recording environment
(`compare --other-environment` skips them). Since W-643, `--other-environment` also skips the digest a
run record holds of each checkpoint stage record (`run.json` `checkpoints.<stage>.sha256` in W-642's
folder): `candidates.json` names the NumPy version it ran with, so that digest is an identity of the
recording environment too. W-642's values, compared across the two environments this way, agree.
