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
