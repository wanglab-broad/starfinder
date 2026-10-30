# Expected values of the per-FOV core on the session dataset

The per-FOV core values of the current workflow (`backend: python`) on the pytest session's small
synthetic dataset, recorded before any §2.13 refactor lands, which the refactored `fov` task must
reproduce: `core_values` of check X4 (`design/validation/orchestration.md`). No test reads them yet:
the X4 test, `test_workflow_core_golden.py`, lands with task group 2 (TG2-1). Until then
`record_core_values.py compare` is the comparison.

They were recorded by W-734 with `record_core_values.py` (this folder) at Starfinder
`67e1f4a08da6605e438856a8d85c8572335a224b` (`dev` before the §2.13 implementation batch), with the
four files of the lock commit `d3dba4d` applied uncommitted; under `src/python/starfinder/`,
`workflow/` and `profile/` the checkout differed from `67e1f4a` only in the lines of
`starfinder/__init__.py` that read the version from the package metadata. Host GP099-29C, one CPU
and one numerical thread, in the locked environment `dev-s213-workflow-20261030T193139Z` built from
that lock (Python 3.12.12, NumPy 2.2.6, Snakemake 9.15.0), which generates the dataset, runs the
workflow and reads the outputs. A second recording in the same environment reproduced every value,
the file digests included.

From `src/python`, with `PYTHONPATH` naming this checkout's `src/python`:

    python test/data/expected/test_workflow_core_golden/record_core_values.py fixture <work>
    python test/data/expected/test_workflow_core_golden/record_core_values.py run <work> --snakemake <Snakemake 9 executable>
    python test/data/expected/test_workflow_core_golden/record_core_values.py extract <work> <expected folder>
    python test/data/expected/test_workflow_core_golden/record_core_values.py compare <this folder> <expected folder>

The fixture is the dataset of `test/conftest.py` (`small_dataset`), generated with its command line
(`starfinder synthetic generate --mode e2e --preset small --seed 42 --owner pytest`): two FOVs of
16 × 256 × 256, four rounds of four channels, uint16, the twelve-gene `codebook.csv`, laid out
`<fov>/<round>/`, with the workflow's `fovs.csv` added. The run is one Snakemake invocation
(`--cores 1`) of `rsf_single_fov` for each FOV, the one rule allowed, with the processing parameters
of W-642's golden configuration A (`scripts/workflow_golden.py`, `configurations`: min-max
normalization to uint8 and a white top-hat of radius 3, translation registration on the maximum
signal, local maxima, extraction with voxel size (1, 2, 2), the end bases `CC` and start base `C`,
WTA decoding with `clip_negative`, deduplication, the checkpoint stages `candidates` and `pre_qc`),
`image_dir_template: "{fov}/{round}"` and `codebook: codebook.csv`.

The files:

* `values.json`: one entry per value, `{value, compare}`. The fixture's file digests (the
  generator's `manifest.json`, which records its absolute output folder, has none); each reference
  image's shape and dtype; and the SHA-256 of each output file that holds no time or absolute path
  (`molecules.csv`, the `pre_qc` stage record and both stage tables). `reference.tif` and the
  `candidates` stage record hold the reference round's frame id, which names the absolute input
  folder (the session dataset's TIFFs store no frame; the contract's OC4), so their digests are
  left out here and in the records that list them; their content is kept (the image in
  `arrays.npz`, the record normalized). Two recordings in different folders differed in nothing
  else.
* `records/fovs/<fov>/run.json` and `records/fovs/<fov>/checkpoints/<stage>.json`: the run record
  (its counts, steps, transforms and stage records' digests) and the two checkpoint stage records,
  normalized as `workflow_golden.py` normalizes records: without the fields of where, when and with
  which software a record was made and its measured times and memory, and with the work folder
  written `<work>`.
* `<fov>-molecules.csv` and `<fov>-checkpoints-<stage>.csv` with `dtypes.json`: the molecule table
  and the two stage tables, read with the dtypes their records state, in the CSV convention of
  `test/expected_values.py`.
* `arrays.npz`: each FOV's reference image (`<fov>/reference`).
* `record_core_values.py`: the recording script; `README.md`: this file.

The comparison is `workflow_golden.py`'s (`docs/contributing.md`, "Expected values and
tolerances"): integers, strings, statuses, booleans, the reference images, floats that hold
integers and the fixture's digests exactly; computed float64 values (a record's floats and the
tables' non-integer float columns) with `rtol=1e-9`, `atol=1e-12`; the output file digests
(`compare: same_environment`) and the run records' digests of the stage records only in the
recording environment (`compare --other-environment` skips them). X4 compares the refactored `fov`
task's outputs with these values and leaves out the contract's exceptions (OC3: the frame ids carry
`/rotate:0`; OC4: the frame ids and inputs name absolute paths, written `<work>` here).
