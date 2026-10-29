# Expected values of the workflow orchestration golden baseline

The golden baseline of the current Snakemake workflow (`backend: python`), which the workflow
orchestration refactor must reproduce (W-642; decision D10 of the §2.13 planning;
`design/baselines/orchestration.md`, "The golden baseline"). No test reads these files yet: the
batch environment has no Snakemake, so the comparison test lands with the lock change that adds
Snakemake 9 to a `workflow` dependency group (task group 2). Until then
`scripts/workflow_golden.py compare` is the comparison.

They were recorded by W-642 with `src/python/scripts/workflow_golden.py` (SHA-256
`34f79811f99f74d8cf61841a5032b4b59b46fd732e872ede8f76fe2eb3da4533`) at Starfinder
`56c35b8b560bfac7bd8b03e0ec7b308ead9a7d89`, on host GP099-29C (Intel Xeon Gold 6234), one CPU and
one numerical thread. Two environments:

* the fixture is generated and the outputs are read in the locked environment
  `dev-s210-export-20261007T050852Z` (Python 3.12.12, NumPy 2.2.6, pandas 3.0.0, tifffile
  2026.1.28);
* the workflow runs with Snakemake 9.15.0 from `~/miniforge3/envs/starfinder-v9`, outside the lock
  (Python 3.12.12, NumPy 2.3.5, pandas 2.3.3, SciPy 1.17.0, scikit-image 0.26.0, tifffile
  2026.1.14, anndata 0.12.7), as installed; its rule scripts import Starfinder from the checkout
  that `starfinder_path` names.

From `src/python`, with `PYTHONPATH` naming this checkout's `src/python`:

    python scripts/workflow_golden.py fixture <work>
    python scripts/workflow_golden.py run <work> --snakemake <Snakemake 9 executable>
    python scripts/workflow_golden.py extract <work> <expected folder>
    python scripts/workflow_golden.py compare <this folder> <expected folder>

The fixture is `sample_scene_preset("sample-tiny-v1")` (seed 42: 2 × 2 FOVs of 5 × 36 × 36, four
rounds of four channels). The run makes four Snakemake invocations (`--cores 1`): A, target `all`
(`rsf_single_fov` with the checkpoint stages `candidates` and `pre_qc`, `rotate_nuclei`,
`nuclei_registration`, `create_nuclei_amplicon_overlay`, `report_sample`, `report_dataset`); A2,
the files of `enhance_dapi_with_flamingo`; B, the saved assignment `default` of each FOV
(`reads_assignment` of the assembly block on the truth cell labels imported as the legacy label
file); C, the tiles, macro and view map of `bigstitcher_macro_sample`. A second run of all of it in
the same environment reproduced every value.

The files:

* `values.json`: one entry per value, `{value, compare}`. The fixture's file digests; the shape,
  dtype, minimum, maximum and sum of the morphology images and the BigStitcher tiles; the macro's
  text (the work folder written `<work>`); and the SHA-256 of each output file that holds no time
  or absolute path.
* `records/`: the JSON records (`run.json`, `nuclei_registration.json`, `assignment.json`,
  `report.json`), each without the fields of where, when and with which software it was made and its
  measured times and memory (`started_at`, `ended_at`, `generated_at`, `code`, `environment`,
  `software`, `input_root`, `output_root`, `workflow_config`, `seconds`, `wall_s`, `cpu_s`,
  `peak_rss_kib`), without the digests of records and files that hold them (`record_sha256`, and
  `sha256`/`bytes` of an entry naming such a file), and with the work folder written `<work>`.
* `<fov>-molecules.csv`, `<fov>-assignment-*.csv`, `report-*.csv`, `bigstitcher-view_map.csv`
  with `dtypes.json`: the tables, in the CSV convention of `test/expected_values.py`, without the
  report's time columns; the report's intensity histograms as one row per histogram
  (`report-sample-tables-intensity-summary.csv`).
* `arrays.npz`: each FOV's reference image (`<fov>/reference`).

The comparison follows `docs/contributing.md` ("Expected values and tolerances"): integers,
strings, statuses, booleans, integer images, floats that hold integers and the fixture's digests
exactly (every image of this run is an integer image, so its statistics are exact); computed
float64 values (a record's floats and the tables' non-integer float columns) with `rtol=1e-9`,
`atol=1e-12` (the script compares a float32 image's statistics with `rtol=1e-6`, `atol=1e-6`; this
run has none); the output file digests (`compare: same_environment`) only in the recording
environment (`compare --other-environment` skips them).
