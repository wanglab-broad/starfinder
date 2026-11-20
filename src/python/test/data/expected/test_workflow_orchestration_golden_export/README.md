# Expected values of the golden baseline's stitching, export and report

W-884 (2026-11-19) adds the report's object sizes to each route: `<route>-report-sample-tables-sizes.csv` (the cell sizes
of the per-FOV and after-fusion routes, whose assignments hold no nucleus label image, and a header only for
`molecules_only`), the rule `small_nucleus_fraction` and part `assembly.sizes` rows of `unavailable.csv` and of the
`unavailable` entries of `report.json` (`no_segmentation`), the two new keys of `config.limits`, the rule's entry of
`rules`, the `sizes.csv` entry of `files` and the digests of the two tables. They were taken from a run of the migrated
invocation D at this change, extracted with the script. The report of the per-FOV route now reads the label image of
every FOV for the sizes, so its `inputs` list gains the entry of FOV_003's `cell_labels.tif` (the fourth FOV; the
examples read the other three): its SHA-256 is the one the FOV's `assignment.json` and `export.json` record, and its size
is not recorded anywhere in this folder, so 2107 bytes is derived: a rebuild of the file from the same fixture is 16 bytes
shorter than each of the three recorded ones (1934, 1873 and 2079 against 1950, 1889 and 2095), and FOV_003's rebuild is 2091
bytes. The comparison leaves the label entries' SHA-256 and size out (D6). Nothing else changes.

W-879 (2026-11-19) removes from each route's report the rows of `unavailable.csv` and the `unavailable` entries of
`report.json` that said no record holds the time of `segment`, `assign_molecules`, `stitch` or the assembly: the
records now hold `wall_s`, so the report shows those times (`steps.csv` keeps its time column out of the comparison, as
before). The size and SHA-256 of `unavailable.csv` in `report.json` and the dtypes of its empty `table` and `column`
columns of the per-FOV and after-fusion routes follow. Nothing else changes: a run of the migrated invocation D at
this change, extracted with the script, equals the other expected values as before.

W-790 (2026-11-03) updates detection-dependent molecule identities, summaries
and record fields after regional-maximum collapse, using the maintained
migrated task commands and extractor. Only differences between the same
fixtures at base and result are applied; existing migration exceptions,
unaffected values and tolerances remain unchanged. The per-file ledger is
retained with W-790's implementation evidence.

The rest of the golden baseline of the current Snakemake workflow (`backend: python`): stitching
with `multiview_stitcher`, the export on each route the fixture allows and the report after the
export, which the workflow orchestration refactor must reproduce (W-643, open point P1 of W-642;
decision D10 of the §2.13 planning; `design/specs/orchestration.md`, "The rest of the golden
baseline"). It sits beside W-642's `test_workflow_orchestration_golden/`, which pins the per-FOV
core, morphology, assignment and the BigStitcher macro and is not changed. No test reads these
files yet: the comparison test lands with the lock change that adds Snakemake 9 to a `workflow`
dependency group. Until then `scripts/workflow_golden.py compare` is the comparison.

They were recorded by W-643 with `src/python/scripts/workflow_golden.py` at Starfinder
`56c35b8b560bfac7bd8b03e0ec7b308ead9a7d89`: invocation D ran with the script of SHA-256
`3730d87826bea72bf8f84cecc7a037b083c2381b6547511b582f1abcb7eec760` (its `run-export` as in the
first recording) and was extracted with the script of SHA-256
`5c0365d4f3838f685c8b7f2bf89f97e6362ff6525cb7af43914c9d231a72e789`, which adds the AnnData
content (repair 1); the package of the worktree equal to `56c35b8`; on host GP099-29C (Intel Xeon Gold 6234), one CPU and
one numerical thread. Two environments:

* the fixture is generated in the locked environment `dev-s210-export-20261007T050852Z` (Python
  3.12.12, NumPy 2.2.6, pandas 3.0.0); since repair 1 the outputs are read in the golden scratch
  environment below, which has `anndata` for `sample.h5ad`;
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
  the segmentation of the route after fusion), and each route's `sample.h5ad` matrices whole: `X`,
  every layer and every `obsm`, `varm`, `obsp` and `varp` entry (`<route>/export/sample/sample.h5ad/X`,
  `.../obsm/spatial`, ...), with their shape, dtype and format in `values.json`.
* `<route>-export-sample-sample.h5ad-obs.csv` and `-var.csv`: the AnnData's `obs` and `var`, its
  names as the first column `_index`; and `records/<route>/export/sample/sample.h5ad.uns.json`: its
  `uns`, the JSON strings (`assembly`, `spatial_centroid`) parsed and normalized as the records
  are. The file's own digest is left out (its `uns` holds absolute paths and times); its content is
  compared instead (W-643, repair 1).

The comparison follows `docs/contributing.md` ("Expected values and tolerances"), as W-642's:
integers, strings, booleans, integer images and the fixture's digests exactly, computed float64
values (and the float matrices of the AnnData files) with `rtol=1e-9`, `atol=1e-12`, and the file digests only in the recording environment
(`compare --other-environment` skips them). Since W-643, `--other-environment` also skips the digest a
run record holds of each checkpoint stage record (`run.json` `checkpoints.<stage>.sha256` in W-642's
folder): `candidates.json` names the NumPy version it ran with, so that digest is an identity of the
recording environment too. W-642's values, compared across the two environments this way, agree.
