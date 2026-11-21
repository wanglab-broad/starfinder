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

## Evidence scripts

Tests that left the suite run here, on request. Each evidence script runs at the end of a batch whose
changes touch its subsystem, and all of them run before a tag or a release (Jiahao, 2026-11-19). They run
on UGER, through the coordinating session, in the image with the commit's code laid over it. Each runs as
one command from `src/python`; the only host paths are the environment variables below, and the scripts
that need none of them generate all their inputs.

| Script | Protects | Needs | Runs after a batch that changes |
| --- | --- | --- | --- |
| `appearance_evidence.py` | GA23, GA24 (`slow` presets, acquired variants, bank builds), GA26 and GA27: the real-bank appearance scenes and the banks' rebuilds (W-536) | the real banks through `STARFINDER_APPEARANCE_BANKS`; up to 3,600 s and 16 GiB per step | the synthetic generator (`src/python/starfinder/synthetic/`) or the appearance banks |
| `cellular_evidence.py` | GN23 and the `slow` and acquired cases of GN24: the cellular scene's `slow` presets in their target ranges and resource bound (W-536) | no extra, no real data; within C26's bound per step | the synthetic generator |
| `cellular_chain_evidence.py` | CD, CT and CU on the appearance `slow` presets: the whole cellular chain (W-603) | the banks, `STARFINDER_WEIGHTS_DIR` and `STARFINDER_STARDIST_3D_SPLEEN`; about 2 h and up to 13 GiB per step | a stage of the chain: synthetic generation, stitching, segmentation, assignment, export or the report |
| `workflow_engine_evidence.py` | the engines' reruns after a changed configuration or a replaced raw image, compared with a fresh run; the grids of an uncalibrated run on Nextflow (W-898: K1, O2, K2, K3) | Snakemake 9, Nextflow 26.10.4 or newer on Java 17 or newer, the `anndata` and `multiview-stitcher` extras; about 40 minutes | the workflow package or `nextflow/` |
| `orchestration_golden_evidence.py` | G3: the refactored workflow against the values of the workflow before the dispatcher (W-898: K7) | Snakemake 9, the `anndata` and `multiview-stitcher` extras; about 2.5 minutes | what the engine script runs after, `workflow/` or `scripts/workflow_golden.py` |
| `spot_finding_evidence.py` | the spot-finding validation rows on seeds 101 and 102 and on Piscis `20230905` (W-899: K4, O3) | the `spotiflow` and `piscis` extras, the six models through `STARFINDER_WEIGHTS_DIR`; about 35 minutes and 1.7 GB | spot finding, its evaluation or these tests' helpers |
| `synthetic_scene_evidence.py` | GN7, GN5 and GA1's digests on the three `slow` context presets (W-899: K5, O1) | no extra, no weights, no bank; about 13 minutes | the synthetic generator or these tests' helpers |

The scripts and the other files here:

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
- `spot_finding_evidence.py`: the spot-finding validation rows that W-899 moved out of the test suite (W-893's
  categories K4 and O3): seeds 101 and 102 of S1 to S8, S10 and S16 for every method (S10's Piscis seed 102 stays in
  the suite), and every row of Piscis `20230905` in S1 to S15, of
  [the spot-finding specification](../design/specs/spot-finding-algorithms.md) ("Implemented checks", "Where the rows
  run"). It protects the engineering validation on the seeds and the model the suite no longer runs (the suite keeps
  seed 100 and Piscis `20251212`); the cases are the test functions of `test_spot_finding_validation.py` called with
  those rows, strict expected failures included. Needs the `spotiflow` and `piscis` extras and the six models' weights
  through `STARFINDER_WEIGHTS_DIR`; no appearance bank, no real data, no network. About 35 minutes on one CPU and
  1.7 GB. It runs at the end of a batch whose changes touch spot finding (`src/python/starfinder/spot_finding/`, the
  evaluation of spot finding or these tests' helpers), and with every evidence script before a tag or a release
  (Jiahao, 2026-11-19), on UGER by the coordinating session.
- `synthetic_scene_evidence.py`: the synthetic scene's checks on the three `slow` context presets that W-899 moved
  out of the test suite: GN7 and GN5 (W-893's category K5) of
  [the cellular validation page](../design/validation/synthetic-cellular-scene.md) and GA1's digests (option O1) of
  [the appearance validation page](../design/validation/synthetic-real-appearance-scene.md). It protects the packing
  and lobes of the `slow` presets and their identity contract (the digests and assertions unchanged; the routine
  presets' checks and pins stay in the suite). Needs no extra, no `STARFINDER_WEIGHTS_DIR` and no appearance bank;
  every case generates its preset, and it reads no real data. About 13 minutes on one CPU. It runs at the end of a
  batch whose changes touch the synthetic generator (`src/python/starfinder/synthetic/`) or these tests' helpers, and
  with every evidence script before a tag or a release (Jiahao, 2026-11-19), on UGER by the coordinating session.
- `evidence_runner.py`: the runner of the four evidence scripts that hold their moved tests as pytest tests (the two
  workflow scripts, the spot-finding and the synthetic scene scripts): it runs pytest on the script with the suite's
  configuration and writes `results.json` (each case's former node ID, outcome and time, the revision, the engines'
  versions and the evidence variables as set) and `junit.xml` to `--output`; `--list` collects the cases without
  running them. A strict expected failure that fails as expected counts as passed.
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
