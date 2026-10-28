# Contributing and local preview

## Install the documentation environment

The package requires Python 3.11 or newer; documentation CI uses **Python 3.12**
and **uv 0.9.28** with the committed lockfile. From the repository root:

```bash
cd src/python
export UV_PYTHON=python3.12
export PYTHONDONTWRITEBYTECODE=1
export UV_LOCKED=true UV_NO_DEFAULT_GROUPS=true
uv sync --locked --no-default-groups --group docs
```

The `docs` dependency group in `src/python/pyproject.toml` declares Sphinx 8.x,
MyST Parser 4.x, PyData Sphinx Theme 0.16.x, sphinx-design 0.6.x, and
sphinxcontrib-matlabdomain 0.22.x. Site styling lives in `docs/_static/custom.css`
and the logo in `docs/_static/logo.png`; theme options are set in `docs/conf.py`.
`src/python/uv.lock` records exact
resolved versions and platform/Python markers. Use `--locked` to detect stale lock
metadata; no optional package extras are needed for the current site. A first
installation needs access to the package registry (or an existing uv cache).
The exported settings keep later `uv run` commands locked and omit the default
development group. On hosts with several interpreters, set `UV_PYTHON` to the
absolute Python 3.12 executable. A host's broken binary-library installation is
not fixed by changing documentation dependencies; see
[environment troubleshooting](troubleshooting.md).

Sphinx imports the installed `starfinder` checkout, including its top-level
imports, so install the package's runtime dependencies as well as the `docs`
group. The build does not mock imports, download datasets, execute notebooks,
run Snakemake, or start MATLAB. SimpleITK, SpatialData, napari, and a MATLAB
license are not required to build the API references. The MATLAB extension
parses only `src/matlab/`; it needs `sphinx.ext.autodoc` enabled alongside it.
Keep MATLAB help immediately after the function/class signature. Reference
pages in `docs/api/matlab/` use `mat:currentmodule:: .` for root-level sources
and explicitly prefixed `mat:autoclass`, `mat:automethod`, and
`mat:autofunction` directives so Python and MATLAB objects remain distinct.
When adding a MATLAB interface, update its help and the relevant inventory page.
See the [extension's upstream documentation](https://github.com/sphinx-contrib/matlabdomain)
for directive syntax. Source parsing does not validate MATLAB execution.

## Build HTML with warnings treated as errors

From `src/python`, choose an absolute output directory **outside the checkout**:

```bash
DOCS_OUTPUT=/absolute/path/outside/checkout/starfinder-docs-html
uv run --group docs sphinx-build -n -W --keep-going -b html ../../docs "$DOCS_OUTPUT"
```

Replace the example path before running. Use a new or empty directory for a clean
build. `-n` checks references, `-W` treats warnings as errors, and `--keep-going`
reports all warnings before exiting. A successful build exits with status 0 and
writes `index.html`, `search.html`, and `searchindex.js` under the output directory.
For a full reread into an existing output directory, add `-E -a`.

The module autosummary lists use reStructuredText (`docs/api/*.rst`) so Sphinx can
discover it before parsing the Markdown pages. Autosummary writes disposable
`.rst` stubs into `docs/api/generated/`; Git ignores
these files. Do not edit them: update the API list or source docstrings instead.
When removing or renaming an export, also remove its obsolete generated stub;
autosummary does not delete old pages, which otherwise retain broken imports
and appear as orphan documents on subsequent builds.
Keep generated HTML, logs, and run-specific validation artifacts outside Git.
Run `uv run python ../../docs/check_reference.py` to check export parity,
alphabetical references and the five-section navigation before building.

## Preview and check the result

Keep the same `DOCS_OUTPUT` value and working directory:

```bash
uv run python -m http.server 8000 --bind 127.0.0.1 --directory "$DOCS_OUTPUT"
```

Open <http://127.0.0.1:8000/> on the same host. For a remote build host, forward
the preview port through your normal SSH connection. Stop the server with Ctrl-C.
For a project-path preview, build into a `starfinder` subdirectory of your
external preview root, serve that parent directory, and open
<http://127.0.0.1:8000/starfinder/>. This checks the same relative asset and source
paths used by Pages.
Check the five navigation sections (Get Started, Workflow, API, Wiki and Benchmark), search for `normalize_min_max`, open its
generated API page, and follow the `[source]` link beside its signature. The
separate page-source link shows the documentation markup.

## Edit and validate documentation

Keep documentation and behavior changes in the **same pull request**:

1. Update Python docstrings at the implementation in `src/python/starfinder/`.
   Use NumPy-style parameters and returns; describe defaults, exceptions,
   shapes/axes, dtypes, units, coordinate origin and optional-backend behavior.
   Use Sphinx roles such as `{py:func}` in MyST guides and `:func:` in reST
   docstrings for references to documented Python objects.
2. List a public export in two places: the `__all__` of its namespace and that
   namespace's autosummary list in `docs/api/*.rst`. A root export (in
   `starfinder.__all__`) is also listed in the root section of the
   [public inventory](../../docs/api/inventory.rst). `docs/check_reference.py` checks
   these lists. Generated stubs are disposable.
   Update [contracts](../../docs/api/contracts.md), [backend requirements](../../docs/api/backends.md)
   and executable examples when their documented guarantees change. MATLAB
   help and inventory maintenance follow the source-parsing rules above.
3. Edit MyST guides in `docs/`; add new pages to the appropriate toctree
   (`docs/index.md` for top-level sections). Use relative page links or explicit
   Sphinx labels, preserve existing README/example entry points, and check
   navigation, search and source links in the preview.
4. Keep recipe code in `docs/examples/` and display it with `literalinclude`
   where practical. Use fixed seeds and small inputs; document resource bounds,
   output schemas and limitations. Run changed examples and the strict build.
5. In the PR, record the commands, interpreter, results, skipped checks and
   output/artifact location. Do not commit generated HTML, logs, datasets or
   local environment files.

### Maintain dependencies

Keep build dependencies in the **`docs`** group in `src/python/pyproject.toml`;
keep algorithm/runtime dependencies and optional extras in their existing
sections. When a documentation dependency changes, update its constraint and
run `UV_LOCKED=false uv lock` from `src/python` (temporarily allowing the intended
lockfile update), then repeat the locked install, strict build
and smoke checks. Commit `pyproject.toml` and `uv.lock` together. Review the
lockfile diff for unrelated upgrades; do not change scientific dependency pins
to silence documentation warnings. CI pins Actions to commit SHAs and uv to a
version in `.github/workflows/docs.yml`; review upstream release changes when
updating them and revalidate the workflow.

### Reproduce the bounded CI smoke checks

From `src/python`, use a new absolute directory outside the checkout:

```bash
DOCS_RUN=/absolute/path/outside/checkout/docs-validation
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$DOCS_RUN/matplotlib"
uv run --group docs python ../../docs/examples/quickstart.py "$DOCS_RUN/quickstart"
uv run --group docs python ../../docs/examples/recipes.py "$DOCS_RUN/quickstart" "$DOCS_RUN/recipes"
```

These are the existing [quickstart](python-quickstart.md) and
[recipes](recipes.md), with assertions for detected shifts, nonempty decoded
output, TIFF round-trips and CSV coordinate conversion. The tiny preset uses
seed **42**, two FOVs, four rounds and `(8,128,128,4)` uint16 volumes; the recipes
reuse FOV_001 and a deterministic `(12,24,24)` single-spot volume. Allow one CPU,
1 GiB RAM and 50 MiB disk. CI limits the combined smoke step to five minutes.
Each script refuses an existing output directory. Success ends with
`Quickstart checks passed.` and `All development recipes passed.`; inspect each
`summary.json`. These checks validate software examples, not biological accuracy.

### CI triggers and evidence

`.github/workflows/docs.yml` runs on all pull requests, pushes to `main`
(the verified GitHub default branch), `dev` and `codex/docs-autonomous`, and
manual dispatch. There is no path filter: source and dependency changes can
break autodoc even when Markdown is unchanged. The read-only job uses Ubuntu
24.04, installs the locked runtime plus `docs` group, builds with
`-n -W --keep-going`, then runs the bounded examples. A failure in a command
fails its step even when output is piped to `tee`. The job timeout is 15 minutes.

Open the commit's **Documentation** Actions run and inspect every step, not
only whether the workflow was triggered. The always-run artifact step retains
installation/build/example logs, interpreter/package versions, seed/config and
example summaries for 14 days. It also retains linkcheck output when requested.
Record the run URL, tested commit and result in the PR/issue before calling CI
verified. Copy cited evidence to durable external storage before artifact expiry.
Successful push runs on `codex/docs-autonomous` also publish the checked HTML
as described below. Pull requests, `main`, `dev` and manual dispatch validate
only; they do not publish.

### Publish and verify a release

The publication URL is <https://wanglab-broad.github.io/starfinder/>. The selected
source branch is **`codex/docs-autonomous`**; it is independent of the repository's
default branch, `main`. Repository **Settings → Pages → Build and deployment**
must use **GitHub Actions**. The `github-pages` environment must permit the
selected branch. Changing the publication branch requires updating both push
guards in the build job, the deploy-job guard and the environment's deployment
branch policy. Keep the branch in the workflow's push trigger list as well.
See [GitHub's custom workflow requirements](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

Every push to the selected branch triggers publishing, including documentation,
docstring and dependency edits. The workflow first passes the strict build and
bounded examples, adds `build-info.json` with the source revision and Actions run
URL, then uploads the HTML through `actions/upload-pages-artifact`. The dependent
`deploy` job uses `actions/deploy-pages`, with `pages: write` and `id-token: write`
restricted to that job. PR checks have no deployment permissions. There is no
generated-site branch and no Jekyll transformation; Sphinx's relative links keep
`_static`, `_modules`, API and search assets under `/starfinder/`.

After a reviewed change is committed and pushed to the selected branch:

1. Open its **Documentation** Actions run. Confirm the `docs` and `deploy` jobs
   succeeded for the expected SHA, and follow the `github-pages` environment URL.
   Merely enabling Pages or passing a local build does not establish publication.
2. Open `build-info.json` under the site URL. Match `revision` and `run_url` to
   the pushed commit/run; allow deployment propagation before diagnosing a stale
   response. Record the public URL, revision and run URL in the release handoff.
3. Open the home page and all five navigation sections (Get Started, Workflow, API, Wiki and Benchmark). Check a nested
   [Python API page](../../docs/api/python.rst), its function anchor and `[source]` backlink;
   open [MATLAB dataset methods](../../docs/api/matlab/dataset.rst) and
   [workflow configuration](../../docs/workflows.md). Follow links between guides and APIs.
4. Search for `normalize_min_max` in the site's search UI and open a matching
   API result. In browser developer tools, check for JavaScript errors and failed
   requests to CSS, fonts, scripts and `searchindex.js`. Check nested pages as
   well as the home page: assets must resolve beneath `/starfinder/`, not at the
   host root. A downloaded search index alone does not verify interactive search.
5. Retain the validation logs, published-page checks and limitations with the
   release record. Confirm a documentation-changing **push** produced the
   deployment; a manual run or settings change does not test that trigger.

If the deployment fails, inspect Pages availability, the Actions permissions,
the environment's branch policy and the failed job before changing code. A
failed build must not publish; the previous successful site remains the release
until a replacement deployment succeeds. Revert an unwanted documentation
change in the selected branch and let normal checks/deployment run again. Do not
force-push or mark an inaccessible site as verified. Keep this source branch
until a reviewed migration updates the workflow and environment together.

Publishing documents supported software usage. It does not validate optional
backends, MATLAB execution, UGER submission, real-data provenance, biological
accuracy or a cell-level endpoint; those need separate evidence.

### Optional imports and external links

CI installs no optional extras and does not mock imports. Missing required
runtime imports must fail the build. Optional backend execution (SimpleITK,
SpatialData, napari), MATLAB execution, Snakemake/UGER jobs and large datasets
are explicitly outside docs CI. When documenting an optional feature, describe
its install extra and fallback/error behavior; test it separately when changing
that behavior and record any unavailable runtime. Do not add broad
`suppress_warnings`, `nitpick_ignore` or import mocks to hide broken references.

The HTML build checks internal references without requesting external websites.
External link checking is **skipped by default** to avoid making each PR depend
on remote availability, authentication or rate limits. The Actions summary
records this skip. For a network-enabled check, choose **Run workflow** and
enable `check_external_links`, or from `src/python` run:

```bash
uv run --group docs sphinx-build -n -W --keep-going -b linkcheck ../../docs "$DOCS_RUN/linkcheck"
```

The workflow must first exist on the default branch for GitHub's manual-dispatch
UI to expose it; the local command is also available before then. Public checks
use a 15-second per-request timeout, one retry and five workers; the CI step
has a five-minute bound. Only `http://127.0.0.1/` preview URLs (including ports)
are explicitly excluded in `docs/conf.py`, because they refer to the reader's
local server. No public domain is broadly excluded. Inspect `output.txt` and
`output.json`; repair broken links and record network/authentication failures
as unresolved or skipped, rather than claiming external links passed. Any new
exclusion needs a specific URL pattern and documented reason.

### Python regression tests

When Python behavior changes, run the tests from `src/python`. The explicit `dev`
group is needed with `UV_NO_DEFAULT_GROUPS=true` above.

```bash
uv run --group dev pytest test/ -v -m "not (slow or learned or extended) and (contract or not validation)"  # Q
uv run --group dev pytest test/ -v -m "registration"    # one subsystem, slow tests included
uv run --group dev pytest test/ -v                      # default run
uv run --group dev pytest test/ -v -m extended          # extended run
```

The default run excludes tests marked `extended`; a `-m` on the command line
replaces that default, so each selection below names `extended` itself. Select
by marker while developing and leave the other selections to the cadences of
"Test selections" below.

#### Test markers

Markers are registered in `src/python/pyproject.toml` and checked with
`--strict-markers`; an unregistered marker is an error.

| Group | Markers | Set on |
| --- | --- | --- |
| Subsystem | `core`, `io`, `dataset`, `preprocessing`, `registration`, `spot_finding`, `readout`, `segmentation`, `stitching`, `report`, `synthetic`, `evaluation`, `benchmark`, `workflow`, `integration` | the file, with `pytestmark` |
| Cost and requirement | `slow`, `learned`, `extended` | the test or the parameter (`extended`: the file or the test) |
| Kind | `contract`, `golden`, `validation`, `e2e` | the file or the test |

Rules for a new or changed test:

1. Every test file sets at least one subsystem marker. The subsystem is the
   package whose behavior the test asserts, not every package it imports. A
   test that asserts behavior across three or more subsystems is `integration`.
   Collection stops with an error that names each file without one.
2. Put a new test in the existing file of its subsystem where one fits. A new
   file is named `test_<subsystem>_<topic>.py`.
3. Kind: `contract` for interface, dtype, error, registry and workflow-key
   behavior; `golden` for comparison with pinned outputs; `validation` for a
   check named in a design table of these pages (the S and V series); `e2e` for
   the full pipeline. A test without a kind marker is an ordinary unit test. A
   `validation` test that is also an ordinary test of a public interface is
   marked `contract` too, which puts it in the quick gate Q.
4. `learned` for a test that loads a detector model or needs
   `STARFINDER_WEIGHTS_DIR`.
5. `slow` by measurement: 5 s or more on one CPU and one thread in the locked
   environment, not counting the one-time import of a library. Time a new test
   with `--durations=0` and report the time in the handoff. A test that uses a
   cached or module-scoped result which takes that long is `slow` as well,
   because it pays the cost when the first user is deselected. For a
   parametrized test, mark the slow parameters with `pytest.param(...,
   marks=pytest.mark.slow)` when only some are slow. An existing test changes
   its cost marker only with a saved measurement that separates the one-time
   import and setup from the execution.
6. A test that reads a documentation page, a file under `docs/` other than
   `docs/examples/`, lives in a docs-contract module: a file that holds no
   `slow`, `learned` or `extended` test and that no module holding one
   imports. The test is marked `contract`, so it runs in Q; a new such file is
   named `test_<subsystem>_docs_contract.py`. A `slow`, `learned` or
   `extended` test reads no documentation page, and neither do the modules
   and helpers it imports: a helper that both need goes to a test helper
   module that names no page. A table or passage that a test parses carries a
   MyST target label right before it, such as `(report-algorithms-checks)=`,
   and the test finds it by that label with `test/docs_pages.py`, never by
   heading text, so a heading may change while the label and the table's row
   shape stay. The files of `docs/examples/` are configurations and scripts
   that tests execute and stay inputs of the checks that run them
   (`scripts/learned_closure.py` lists what a check's tests read).
7. Reviewers check the markers of new and changed tests.

% W-602

#### Expected values and tolerances

Each expectation a test states belongs to one of three classes (Jiahao's
decision of 2026-10-25):

% W-526, TM-20, TM-22, TM-27

1. **Contract identity: exact.** Bit equality, or a digest of the computed
   values, stays where a contract requires identity: file, record, export and
   run-record hashes; checkpoint and old-format round trips; model and weight
   identity; stream descriptors; OME bytes and content-derived UUIDs;
   same-environment repeatability (two runs or two processes in one
   environment); integer and discrete outputs; and every place these pages
   promise bit identity. These assertions stay exact, and an integer or
   discrete output may keep its digest.
2. **Historical numerical pins: explicit values with tolerances.** A
   floating-point value that a test pinned at an earlier revision, as a digest
   or as a literal, is written as explicit expected values and compared with
   the tolerance of its quantity below. A value that a formula gives is written
   as that formula, an analytic expectation.
3. **Unnecessary: removed.** A pin that only records history, such as a frozen
   obsolete test-side implementation or a source-line lock, is removed; the
   behaviors tested beside it stay.

| Quantity | Comparison |
| --- | --- |
| integers, counts, labels, integer images, strings, statuses, booleans | exact |
| floats that hold integer values (sums of integer pixels, integer-valued coordinates and peaks) | exact |
| float64 computed values (scores, thresholds, quantiles, transforms, displacement fields, random draws through `exp` or `log`) | `rtol=1e-9`, `atol=1e-12` |
| float32 computed values | `rtol=1e-6`, `atol=1e-6` |
| physical coordinates and lengths with a declared grid precision | the declared precision (1e-9 µm) |
| solver guarantees and analytic bounds that a contract states | the stated bound, unchanged |
| analytic expectations (a formula the test evaluates) | the tolerance of the computed quantity's dtype |

The dtype is that of the compared result: a float32 image uses the float32 row
even when the test evaluates its expected value in float64. A tolerance passes
when `|actual - expected| <= atol + rtol * |expected|`, as
`numpy.testing.assert_allclose(actual, expected, rtol=..., atol=...)` checks it.
With `pytest.approx`, give both `rel` and `abs`: `abs` alone turns the relative
tolerance off. These are bounds on software precision, not scientific accuracy
thresholds. Never widen a tolerance or change an expected value to make a
result pass.

Expected values are written in the test when it needs about 20 values or
fewer. Larger expectations are a CSV file (tables) or an `.npz` file (arrays)
under `src/python/test/data/expected/<test module>/`, with a `README.md` that
names the generating revision and command. The test states the tolerances, per
column.

A digest pin is replaced only after a saved comparison on the base revision
shows that the new expected values, encoded as the old helper encoded them,
give the old pinned digest. Expected values are never taken from a new run of
the changed code.

A negative control, a test that a changed setting changes the result, compares
the changed output with the explicit expectation and asserts that it lies
outside the tolerance.

The `golden` marker keeps its meaning: a comparison with pinned outputs, which
are now explicit values.

#### Test selections

The tests run in the selections below, each with its cadence. "A batch" is a
run of issues with one canary issue first; "before merge" is the gate of a pull
request.

| Selection | Marker expression | When it runs |
| --- | --- | --- |
| Q, the quick gate | `not (slow or learned or extended) and (contract or not validation)` | every change |
| V, the rest of the fast tier | `validation and not contract and not (slow or learned or extended)` | at a batch's end and before merge |
| R = Q ∪ V, the fast tier | `not (slow or learned or extended)` | for a change to a shared path (`starfinder/image.py`, `starfinder/io/`, `starfinder/dataset/`, `test/conftest.py`, `pyproject.toml`, `uv.lock`), at a batch's canary, at a batch's end and before merge |
| Slow, per subsystem | `slow and not (learned or extended) and <subsystem>`, in the groups below | when that subsystem changes, and at a batch's end |
| Learned | `learned` | when a dependency of the learned tests changes, and at a batch's end if one changed |
| Extended | `extended and not learned` | at a batch's end and before merge |
| Documentation | not pytest: the reference audit `docs/check_reference.py`; the strict build `sphinx-build -n -W` | the reference audit on every change; the strict build at a batch's end, before merge, and on a change to `docs/` or to a public interface |

The slow selection runs in five groups, one per subsystem check:

| Group | Marker expression |
| --- | --- |
| benchmark | `slow and not (learned or extended) and benchmark` |
| synthetic | `slow and not (learned or extended) and synthetic and not benchmark` |
| registration | `slow and not (learned or extended) and registration` |
| spot_finding | `slow and not (learned or extended) and spot_finding` |
| other | `slow and not (learned or extended) and not (benchmark or synthetic or registration or spot_finding)` |

Q and V share no test, and together they are R. R, slow (`slow and not (learned
or extended)`), learned and extended hold each collected test exactly once, so
a batch's end runs every test. The `learned` tests run in the controller's
learned checks, split by model and chain; the `extended` tests that are
`learned` run there too. The chain selections and the slow group other hold
the cellular chain checks on the routine presets only: their `slow` presets,
built from real images, run outside the suite as
`benchmarks/cellular_chain_evidence.py`, as the other real-data workloads run
as `benchmarks/appearance_evidence.py` and `benchmarks/cellular_evidence.py`. A test that is both `contract` and `validation` runs in
Q. To check the coverage, collect the node IDs of each selection (no test runs)
and compare them:

```bash
uv run python scripts/selection_coverage.py --output <run-dir>/selections
```

It writes each selection's node IDs and `coverage.json`, and exits with 1 when
Q and V overlap or when a collected test is missing from the four selections or
in two of them. To review the cost markers, run a selection with
`--durations=0` and pass its log to the drift report, which lists the tests
without `slow` that took 5 s or more and the `slow` tests under 2 s. It leaves out
a `slow` test under 2 s only when an entry of `EXEMPTIONS` in
`scripts/marker_drift.py` covers it. An entry names one cached source (a
wider-scoped fixture or a cached helper), the tests that share it and a saved
measurement of what its first user pays:

```bash
uv run pytest test/ -v --durations=0 -m "<selection>" > <run-dir>/durations.log
uv run python scripts/marker_drift.py <run-dir>/durations.log
```

The `Tests` GitHub Actions workflow runs on pushes and pull requests to `dev`
and `main` in two jobs:

| Job | Marker expression |
| --- | --- |
| `fast` | `not (slow or learned or extended)`, the fast tier R |
| `slow-extended` | `(slow or extended) and not learned` |

Together they run every test that is not `learned`. Each job installs the
locked `dev`, `local-registration` and `checkpoint` extras, sets the numerical
thread variables to one thread, lists which optional backends are installed,
and runs its pytest step under `/usr/bin/time -v` for the peak RSS, with `-rs`
for the reason of each skipped test. Each pytest step has a limit of 30
minutes. A measured overrun is split by file or node ID into further steps that
together hold the same tests; neither job is split before an overrun is
measured. The workflow installs neither the `spotiflow` and `piscis` extras
with their weights nor the `registration-elastix` extra, so the `learned` tests
and the elastix registration tests do not run there. MATLAB is never executed;
the launcher tests mock `subprocess`.
The full Python suite is separate from bounded documentation CI. Local build
success does not establish a successful Actions run, deployment, optional-backend
execution or external link availability.
