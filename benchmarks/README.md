# Benchmark scripts

Evidence and measurement scripts kept beside the package. Experiments run through
`starfinder experiment expand|evaluate|report` and `starfinder trial run`; see the
[experiments guide](../docs/benchmark/experiments.md). W-781 retired the old
`starfinder benchmark run/evaluate/report` lifecycle and its scripts here: the case
builders `recipes.py` and `configs/` (registration and pipeline profiles, the source map),
`report_saved.py`, and the §2.5 preprocessing comparison `preprocessing_synthetic.py` and
its report `preprocessing_report.py`. Their source stays in Git at
[`8094b30`](https://github.com/wanglab-broad/starfinder/tree/8094b30c6c57486f307aa6198dfb5e33b86c5241/benchmarks) and
[the migration guide](../docs/migration.md#removed-benchmark-lifecycle) names each replacement.

- `image_statistics.py`: intensity, puncta and noise statistics of one real TIFF or synthetic channel
  volume, and the read-only real-data measurement behind the synthetic calibration targets (W-238);
  see [image statistics](../design/notes/image-statistics.md).
- `appearance_evidence.py`: the real-bank appearance checks GA23, GA24 (the `slow` presets, the
  acquired variants and the bank builds), GA26 and GA27 of
  [the appearance validation page](../design/validation/synthetic-real-appearance-scene.md), moved out of the
  test suite for the Chapter II evidence project (W-536). Reads the real banks through
  `STARFINDER_APPEARANCE_BANKS`; RG1 allows one CPU, 16 GiB and 3,600 s per step.
- `cellular_evidence.py`: GN23 and the `slow` and acquired-pixel cases of GN24 of
  [the cellular validation page](../design/validation/synthetic-cellular-scene.md), moved likewise
  (W-536). Reads no real data; within C26's bound per step.
- `cellular_chain_evidence.py`: the chain checks CD, CT and CU of
  [the cellular validation page](../design/validation/synthetic-cellular-scene.md) on the appearance
  `slow` presets, moved out of the test suite for the Chapter II evidence project (W-603); it calls
  the tests of `test_integration_cellular_chain.py`, which the suite runs on the routine presets, with
  the `slow` tier. Reads the real
  banks through `STARFINDER_APPEARANCE_BANKS` and the models through `STARFINDER_WEIGHTS_DIR` and
  `STARFINDER_STARDIST_3D_SPLEEN`; about 2 h on one CPU and up to 13 GiB per step.
- `workflow_engine_evidence.py`: the workflow engine checks that W-898 moved out of the test suite
  (W-893's categories K1, O2, K2 and K3): X11's `restored-twice` case on both engines and its `raw-size` and `raw-older`
  cases on Nextflow of [the orchestration validation page](../design/validation/orchestration.md), the Nextflow cache
  test on real tasks, and the uncalibrated run on Nextflow. It protects the engines' reruns after a changed
  configuration or a replaced raw image, compared with a fresh run, and the declared spacing of an uncalibrated run's
  grids. Needs Snakemake 9 (the `workflow` group), Nextflow 26.10.4 or newer on Java 17 or newer, and the `anndata`
  and `multiview-stitcher` extras; a case whose engine is missing skips with the reason. It writes its own inputs
  (`tiny_sample`) and reads no real data; about 40 minutes on one CPU, under 1 GiB beside Nextflow's JVM. It runs at
  the end of a batch whose changes touch the workflow package or `nextflow/`, and with every evidence script before a
  tag or a release (Jiahao, 2026-11-19), on UGER by the coordinating session.
- `orchestration_golden_evidence.py`: G3 of [the orchestration validation page](../design/validation/orchestration.md),
  the refactored workflow against the golden values that W-642 and W-643 recorded with the workflow before the
  dispatcher, with the tests of its comparison `compare_migrated` (W-893's category K7, moved by W-898). It protects
  W-647's refactor; the expected folders stay in `src/python/test/data/expected/`. Needs Snakemake 9 and, for G2's
  part, the `anndata` and `multiview-stitcher` extras; writes its own fixture, reads no real data, about 2.5 minutes on
  one CPU. It runs when the engine script does, and also after a change to `workflow/` or `scripts/workflow_golden.py`.
- `evidence_runner.py`: the runner of the two workflow evidence scripts, which hold their moved tests as pytest
  tests: it runs pytest on the script with the suite's configuration and writes `results.json` (each case's former
  node ID, outcome and time, the revision and the engines' versions) and `junit.xml` to `--output`; `--list` collects
  the cases without running them.
- `revision.py`: the source revision record that `image_statistics.py` writes into its manifest
  (W-288; the retired `preprocessing_synthetic.py` wrote it too): HEAD, dirty state, the untracked
  paths and the sha256 of `git diff HEAD --binary` followed by each untracked path's name and bytes;
  an untracked directory entry, such as a nested Git repository, contributes its name only.

Original scripts, results and restricted reference snapshots remain external.
These new implementations follow the Python package's declared MIT terms;
no external source license or redistribution right is inferred. All generated
configs, inputs, runs and reports belong outside this checkout. Real/large,
Postcode, hybrid and MATLAB experiments require their documented prerequisites
and separate execution scope. No default sweep is provided.
