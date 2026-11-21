# Contributing and local preview

Build from `src/python` with Python 3.12 and the locked environment. The
[installation guide](get-started/installation.md) covers package setup.
Keep code, docstrings, public API lists and examples together when an interface changes.

## Quick start

```bash
uv run --group docs sphinx-build -n -W --keep-going -b html ../../docs /tmp/starfinder-docs-html
uv run python ../../docs/check_reference.py
```

Choose a fresh output directory outside the checkout. The strict build checks
internal references; the reference audit checks exports and navigation.
Generated API stubs in `docs/api/generated/` are disposable: edit `__all__`,
authored autosummary lists and implementation docstrings instead.
For a preview, serve the HTML directory with Python's `http.server` and open
its localhost URL. Check navigation, search and API source links.

## Test environment

Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`NUMEXPR_NUM_THREADS`, `NUMBA_NUM_THREADS` and
`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS` to 1 before you run the tests. The first five
are `starfinder._execution.THREAD_VARIABLES`, the source of the list, and each task
records them in its execution entry; a test that compares bytes across runs or
engines needs the same values in every run. Keep `TMPDIR` on local disk, and keep
library caches off `HOME` by setting `MPLCONFIGDIR` and `NUMBA_CACHE_DIR` on local
scratch.

The suite's `conftest.py` sets `SOURCE_DATE_EPOCH` to the session's start time when
it is unset. pytest imports it before pytest-xdist starts its workers, so the workers
and every process a test starts record one version for a modified checkout, whose
`.d<date>` part comes from that variable ({ref}`the code identity <checkpoints-code-identity>`).
A value you set is kept.

## Documentation pages

A guide is about 1,500 words at most. Where applicable, use these sections in
order: a short lead, Quick start, What it does, Methods or Presets, What you get,
Main options, Limits, How it is checked, then a link to its specification.
Show a minimal public-API example. Explain choices in plain words; link precise
rules to the API or specification.

A reference holds lookup tables with brief introductions. Generate tables from
code where possible using the directives in `docs/_ext/`: `method-table`,
`config-options` and `synthetic-presets`. Link config classes to their API
pages for further details rather than copying parameter descriptions.

Every config dataclass and every public function, method and dataclass on the
API pages requires a complete numpydoc `Parameters` section, with an entry for
every public field or parameter (excluding `self` and `cls`): what it sets, its
unit and simple valid values, in one or two plain sentences. Variadic parameters
keep their stars. Add a `Returns` or `Yields` section when the callable returns a
value. Keep precise rules in the specification and the first docstring line
unchanged. `config-options` reads these descriptions; the strict build rejects
missing, unknown, duplicate or empty entries for its targets and API objects,
and requires a return section when an API callable's annotation denotes a value.
Unannotated callables have no automated return check. Exceptions, warnings,
enums, constants and private dataclass fields are exempt. The quick-gate contract
test checks all public config dataclasses, every `config-options` target and
every API object, including public methods defined by listed classes.

A class page follows scanpy's class pages and describes each parameter once.
The signature comes first, without fields whose names start with `_`. Then the
docstring, whose `Parameters` section gives each field's type, default and
description. An `Attributes` table lists the properties and other public
attributes that `Parameters` does not name, and is left out when there are none;
a `Methods` table lists the public methods, one line each. The full entry of
each method and attribute follows. A dataclass field named in `Parameters` gets
no entry of its own, and a reference to it, such as
`` :attr:`Dataset.input_root` ``, links to the class. The `class_pages` extension
in `docs/_ext/` applies these rules to every class from `dataclasses.fields` and
the `Parameters` names; no class needs a setting of its own.

Specifications, validation designs, baselines and proposals are design records
in [design/](https://github.com/wanglab-broad/starfinder/blob/dev/design/README.md),
never site pages. An implementation batch's last issue writes the section's
guide and reference from the implemented code. Check facts against source and
run examples on small inputs. Tests read labelled tables in design records
with `docs_pages.design("path/under/design.md")`; keep their labels and row shapes.

## Test markers

Markers are registered in `src/python/pyproject.toml`.

* Set a subsystem marker on the file with `pytestmark`; use `integration` for
  behavior across three or more subsystems. Prefer an existing test module;
  name a new one `test_<subsystem>_<topic>.py`.
* Use `contract` for interfaces, dtypes, errors, registries and workflow keys;
  `golden` for pinned outputs; `validation` for checks in design tables;
  `e2e` for the full pipeline. A validation of a public interface also gets
  `contract`. Unmarked kinds are ordinary unit tests.
* Use `learned` for tests loading detector models or needing
  `STARFINDER_WEIGHTS_DIR`.
* Use `nextflow` for tests that run the Nextflow workflow. They skip, naming the
  reason, without Nextflow 26.10.4 or newer on `PATH` or without the Java 17 or newer
  that it runs on (`test/workflow_nextflow.py`), and also carry `slow`, so that each stays in the slow or
  extended selection.
* Mark a test `slow` when it takes at least 5 seconds on one CPU and one
  numerical thread in the locked environment, excluding one-time imports.
  Include cached setup cost paid by the first selected user. Measure with
  `--durations=0`; mark individual slow parameters when appropriate.
* Put page-reading tests in docs-contract modules without slow, learned or
  extended tests, and mark them `contract`. Those modules are not imported by
  expensive tests. Find parsed tables by MyST labels with `test/docs_pages.py`,
  rather than headings. Use a shared helper without page paths when needed.
  Executable files under `docs/examples/` remain check inputs.

## Expected values and tolerances

Keep exact assertions where the contract requires identity: file, record, export
and run hashes; checkpoint and old-format round trips; models and weights;
stream descriptors; OME bytes and content-derived UUIDs; same-environment
repeatability; integer and discrete outputs. Replace historical numerical pins
with explicit expected values and tolerances. Use analytic formulas where
available; remove obsolete source locks while retaining behavioral checks.

| Quantity | Comparison |
| --- | --- |
| integers, counts, labels, integer images, strings, statuses, booleans | exact |
| floats that hold integer values (sums of integer pixels, integer-valued coordinates and peaks) | exact |
| float64 computed values (scores, thresholds, quantiles, transforms, displacement fields, random draws through `exp` or `log`) | `rtol=1e-9`, `atol=1e-12` |
| float32 computed values | `rtol=1e-6`, `atol=1e-6` |
| physical coordinates and lengths with a declared grid precision | the declared precision (1e-9 µm) |
| solver guarantees and analytic bounds that a contract states | the stated bound, unchanged |
| analytic expectations (a formula the test evaluates) | the tolerance of the computed quantity's dtype |

Use the compared result's dtype. `assert_allclose` checks
`|actual - expected| <= atol + rtol * |expected|`; give both `rel` and `abs`
with `pytest.approx`. These measure software precision, not scientific accuracy.
Do not widen tolerances or change expectations to make a test pass.

Write about 20 expected values or fewer in the test. Larger tables and arrays
go under `test/data/expected/<test module>/` as CSV or NPZ with a README naming
the generating revision and command; state per-column tolerances in the test.
Before replacing a digest, save a base-revision comparison showing the new
expectations reproduce the old digest through the old encoding. Do not take
expectations from changed code. A negative control compares changed output
with the explicit expectation and checks that it lies outside tolerance.

## Test selections

Run from `src/python` with the `dev` dependency group installed. A command-line
`-m` replaces pytest's default exclusion of `extended`.

| Selection | Marker expression | Cadence |
| --- | --- | --- |
| Q | `not (slow or learned or extended) and (contract or not validation)` | every change |
| V | `validation and not contract and not (slow or learned or extended)` | batch end and before merge |
| R = Q ∪ V | `not (slow or learned or extended)` | shared paths, batch canary, batch end and before merge |
| Slow | `slow and not (learned or extended) and <subsystem>` | affected subsystems; all at batch end |
| Learned | `learned` | changed dependencies; batch end if changed |
| Extended | `extended and not learned` | batch end and before merge |
| Nextflow | `nextflow` | `nextflow/` or the workflow package changed; batch end and before merge |
| Documentation | reference audit and strict Sphinx build | audit every change; build for docs/public interfaces, batch end and before merge |

Shared paths are `image`, `io`, `dataset`, `conftest.py`, `pyproject.toml` and
`uv.lock`. The slow groups are benchmark; synthetic excluding benchmark;
registration; spot_finding; and other excluding those four.
`scripts/selection_coverage.py` checks that R, slow, learned and extended
cover collected tests once. `scripts/marker_drift.py` reads a durations log to
review cost markers. Review new and changed test markers before merging.

`-m nextflow` needs Nextflow 26.10.4 or newer on `PATH` and Java 17 or newer, the Java that
`nextflow info` reports it runs on. CI
runs it in its own job with the minimal extras, where the tests of the real
per-FOV route skip for want of `anndata` and `multiview-stitcher`.

The selections are independent: they may run at the same time when each has its
own core and temporary directory.

### Evidence scripts

Some tests left the suite for evidence scripts under `benchmarks/` (W-893's audit, phase 2).
Each script holds its moved tests unchanged, generates its own inputs and runs as one
command. The phase-2 scripts and the subsystem each protects:

| Script | Protects | Runs after a batch that changes |
| --- | --- | --- |
| `workflow_engine_evidence.py` | the engines' reruns after a changed configuration or a replaced raw image, compared with a fresh run, and the grids of an uncalibrated run on Nextflow | the workflow package or `nextflow/` |
| `orchestration_golden_evidence.py` | G3: the refactored workflow against the values of the workflow before the dispatcher | what the engine script runs after, `workflow/` or `scripts/workflow_golden.py` |
| `spot_finding_evidence.py` | the spot-finding validation rows on seeds 101 and 102 and on the Piscis model `20230905` | spot finding, its evaluation or these tests' helpers |
| `synthetic_scene_evidence.py` | the `slow` context presets' packing, lobes and GA1 digests | the synthetic generator or these tests' helpers |

Jiahao's rule of 2026-11-19: each evidence script runs at the end of a batch whose changes
touch its subsystem, and all of them run before a tag or a release. They run on UGER,
through the coordinating session, with the commit's code laid over the image. The
[benchmark scripts' README](https://github.com/wanglab-broad/starfinder/blob/dev/benchmarks/README.md)
lists every evidence script with its needs.

### The runner's checks

The controller runs the selections as checks, each on one CPU, chosen by the paths a
change touches. W-893's phase 2 set three rules for them:

* **A full validation runs Q once.** A validation that runs the fast tier R does not
  run Q again, since R holds Q (K9). Q stays the gate of every other change.
* **The fast tier is two checks (S1).** `pytest-fast-tier-1` runs
  `not (slow or learned or extended) and (report or segmentation or synthetic or workflow)`
  and `pytest-fast-tier-2` runs `not (slow or learned or extended) and not (report or
  segmentation or synthetic or workflow)`: together the fast tier, in two parts of about
  equal time.
* **`pytest-extended-recovery` is two checks (S2),** split by test:
  `pytest-extended-recovery-1` adds `-k "test_x7_sigkill_partial_task_recovers_with_only_its_downstream
  or test_x11_stale_results_rerun_only_readers_and_downstream"` to its selection, and
  `pytest-extended-recovery-2` adds the same expression under `not (...)`.

A check's inputs are the test files it runs and the modules they import (H3).
`scripts/check_inputs.py` collects each check's selection and computes the import
closure of its test files, following the scripts a test loads by path and, for a file
that puts a folder on `sys.path`, the modules it imports from that folder. It writes them
to `test/check_inputs.json`, which `test/test_check_inputs.py` keeps equal to the import
graph. Run it again after you add, move or re-mark tests:

```bash
uv run python scripts/check_inputs.py            # writes test/check_inputs.json
uv run python scripts/check_inputs.py --check    # exit status 1 when the file is out of date
```

## Limits

A local HTML build does not execute MATLAB or optional algorithms. Record
unavailable checks in the handoff. Publication is a separate authorized action;
the [release instructions](https://github.com/wanglab-broad/starfinder/blob/dev/design/notes/contributing.md#publish-and-verify-a-release)
explain the deployment workflow.
