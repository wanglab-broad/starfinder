# Experiment report

`starfinder experiment report` renders a saved evaluation as a review page, so
that you can understand an experiment before you inspect its provenance. It
reads the folder `starfinder experiment evaluate` wrote and the saved records
and files that evaluation names, checking each record and file that has a
recorded SHA-256 against it (see Limits). It processes, detects and generates
nothing, and computes no metric. It is not
the sample-processing report of `starfinder.report`. The
{doc}`commands reference <commands>` lists its arguments and output files.

## Quick start

From `src/python`, write the bounded example's inputs into a new folder outside
Git, run its experiment and evaluation, then render the report:

```bash
uv run python ../../docs/examples/experiment_report_inputs.py /tmp/starfinder-report
uv run starfinder experiment expand --config /tmp/starfinder-report/experiment.yaml --output /tmp/starfinder-report/trials
uv run starfinder trial run --config /tmp/starfinder-report/trials/trial-000000.json
uv run starfinder trial run --config /tmp/starfinder-report/trials/trial-000001.json
uv run starfinder trial run --config /tmp/starfinder-report/trials/trial-000002.json
uv run starfinder experiment evaluate --sources /tmp/starfinder-report/sources.yaml --metrics /tmp/starfinder-report/metrics.yaml --output /tmp/starfinder-report/evaluation
uv run starfinder experiment report --evaluation /tmp/starfinder-report/evaluation --output /tmp/starfinder-report/report --protocol /tmp/starfinder-report/protocol.yaml
```

The example is an 8 × 16 × 16 impulse pair shifted by (1, −2, 1) voxels and one
translation method under three conditions. The `scipy` and `skimage` trials
succeed. The `mismatched` trial's moving image has another shape, so it fails
with `IncompatibleGeometryError` and exits 1. The evaluation then exits 1: its
metric for that trial is `skipped`, and the `skimage` metric requests no
tolerance, so its `passed` value is undefined ("tolerance gate not requested").
The report exits 0. Open `report/report.html` to see both states. In the
translation card, the skipped trial is shown in red with its error, the
undefined value in italics with its reason. The `scipy` row shows `max_error`
0 voxels and `passed` True. This is a software example, not an accuracy
experiment.

## What the page holds

The summary comes first, in the order of the W-237 review format:

1. **Setup.** The qualification (pilot or scientific), the route and the
   input and truth provenance. Then each trial's resolved method,
   configuration, condition and seed, and the setups. Then the
   conditions, seeds, modes and dtype. Then each metric's meaning, population
   and frozen parameters, the fixed cut-offs and the missing prerequisites.
   Last come the resources by scope, and the revision check.
2. **One card per method or mode.** Each card gives the purpose and the
   targeted conditions, and those absent from the evaluation. The key metrics
   follow as ranges across seeds and FOVs, in three columns: before, after
   and without a side, so a value with no side is shown beside the others.
   Each range keeps its unit, denominators, missing seeds or FOVs, undefined
   values with their reasons, failures, and the `values.csv` rows it comes
   from. Then come the supplied precondition and harm results, a verdict only
   when the protocol supplied one, with its failing clause, and the
   limitations.
3. **Cross-cutting findings and anomalies.** Each item names its source table.
   The report lists what the tables show as findings: failed sources, metrics
   not computed, undefined values, missing seeds or FOVs, unpaired rows,
   different resource scopes, collapsed precision-recall curves and an
   unresolved revision. The protocol's own findings and hypotheses follow,
   with the hypotheses set apart.
4. **Reading order**, linking every card and figure.

A card's anchor, its details anchor and its figure file names carry a readable
slug of its method and mode followed by the first 8 hexadecimal characters of
the SHA-256 of the exact pair, as in `card-translation-c437d4bb` (the bounded example's card). Cards whose
names differ only in punctuation stay apart, and two cards that would still
share an identifier refuse the report.

The method details and figures follow. Each card has a dot plot of its numeric
key metrics, one dot per saved row, with undefined values counted, not drawn.
Each image figure the protocol names draws saved images on one linear display
range, with the saved detections and, given truth points, the representative
puncta. The dim punctum, the lowest realized 3 × 3 × 3 peak, comes first. The
appendix keeps the identity, revision and checksums, every file the report
read, the sources and consumed files, every value per seed and FOV, the before
and after pairs with unpaired rows kept, the resources, the diagnostics and
saved curves, and what was omitted or reduced. Wide and long tables are split
into pieces that each fit one screen, with the key columns repeated.

## The protocol

The evaluation holds values, not their meaning for an experiment. An optional
`--protocol` mapping (YAML or JSON) supplies that text. Every key is optional,
and without a protocol the page says what was not supplied. The
{doc}`commands reference <commands>` lists the keys: the qualification, route,
inputs, truth, dtype and resources notes; condition descriptions; fixed
cut-offs; missing prerequisites; metric notes; `rows`; the cards in `methods`;
`findings`; and `figures`. The bounded example writes a short `protocol.yaml`
with a card, conditions, a cut-off, a hypothesis and an image figure.

Rows take their method, seed and condition name from the trial records, and
their FOV from the workflow task records (else from the workflow manifest's
task entry, else the task id). A metric the evaluation did not compute may have
stopped before recording which task it reads. Its tasks then come from the
metrics specification the evaluation recorded, checked against its SHA-256;
when that file is missing or changed, the FOV stays empty and a finding says
why. A protocol row assigns a metric to a
card (`method`, `mode`) and a `side`, and fills a condition, seed or FOV that
the records do not give. It cannot change one they do give. A key metric is
written `ID` or `function` for all its values, or `ID:value` for one, such as
`M03:f1`. A figure reads a trial's listed `output`, a trial's recorded `input`
or a files source's `file`, by name.

## Limits

* The renderer recomputes nothing. It derives only ranges of saved values and,
  for a list-valued value, the mean over its items, labelled with their
  number.
* It selects no threshold, default or optimum. Rows of different setups, units
  or parameter sets are never pooled into one range. A trial's setup is its
  method, configuration and condition. A workflow task's setup is its kind and
  the configuration and blocks hashes of its task record, else of the
  workflow manifest. The setups are listed under Setup, and the before and
  after pairs name each side's setup. Checkpoint and files sources save no
  configuration identity, so rows reading only them have no setup and are
  told apart only by the protocol's cards and sides.
* Assigned-read counts and agreement scores are shown with their populations;
  they are not accuracy.
* The trial records, workflow manifests and task records, and every image or
  table a figure draws, are checked against the SHA-256 that the evaluation or
  a checked record holds. A changed or missing one refuses the report, and
  nothing is written. `values.csv` must equal the rows of `evaluation.json`.
  `evaluation.json` itself, each metric's details file and the protocol have no
  recorded SHA-256: the report reads them, hashes them and lists them in
  `report.json` as read only. The recorded metrics specification is read, and
  checked, only for a metric that was not computed (see the protocol section).
* The evaluation's recorded commit must be an ancestor of this checkout's
  HEAD. For an evaluation of a clean tree, `starfinder/benchmark/_evaluate.py`
  and `starfinder/evaluation/` must be unchanged since that commit. A dirty
  evaluation records no checksum of its changes, so it cannot be resolved to a
  commit. The report labels it unresolved. The renderer needs a Git checkout
  of starfinder.
* A figure draws only `.npy` or `.tif` images of trial or files sources.
* Exit status 0 means the report was written. 2 means a usage or input error:
  a saved file or revision that does not check, an invalid protocol, or an
  output folder that is not new or empty, or lies inside the evaluation or a
  source. Nothing is written then.

## How it is checked

`test_experiment_report.py` renders a hand-built study of eight saved trials
(one failed) in two fixed threshold modes, with processing and evaluation
entry points patched to raise. It hashes the source files before and after,
and checks the known F1, correct-decode and assigned-read values, units,
denominators, seeds, missing rows and unpaired rows. It checks the page order
and reading-order links, the refusal of every tampered file, the revision
check, and the Quick start above.

[Benchmark specification](https://github.com/wanglab-broad/starfinder/blob/dev/design/specs/benchmark.md).
