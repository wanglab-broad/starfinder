# Contributing and local preview

## Install the documentation environment

The package supports Python 3.10 or newer; documentation CI uses **Python 3.12**
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
Keep generated HTML, logs, and run-specific validation artifacts outside Git.

## Preview and check the result

Keep the same `DOCS_OUTPUT` value and working directory:

```bash
uv run python -m http.server 8000 --bind 127.0.0.1 --directory "$DOCS_OUTPUT"
```

Open <http://127.0.0.1:8000/> on the same host. For a remote build host, forward
the preview port through your normal SSH connection. Stop the server with Ctrl-C.
Check the eight navigation sections, search for `normalize_intensity`, open its
generated API page, and follow the `[source]` link beside its signature. The
separate page-source link shows the documentation markup.

## Edit and validate documentation

Keep documentation and behavior changes in the **same pull request**:

1. Update Python docstrings at the implementation in `src/python/starfinder/`.
   Use NumPy-style parameters and returns; describe defaults, exceptions,
   shapes/axes, dtypes, units, coordinate origin and optional-backend behavior.
   Use Sphinx roles such as `{py:func}` in MyST guides and `:func:` in reST
   docstrings for references to documented Python objects.
2. For a public export, update the relevant `docs/api/*.rst` autosummary list
   and [public inventory](api/inventory.rst). Generated stubs are disposable.
   Update [contracts](api/contracts.md), [backend requirements](api/backends.md)
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

These are the existing [quickstart](getting-started.md) and
[recipes](recipes.md), with assertions for detected shifts, nonempty decoded
output, TIFF round-trips and CSV coordinate conversion. The tiny preset uses
seed **42**, two FOVs, four rounds and `(8,128,128,4)` uint8 volumes; the recipes
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
3. Open the home page and all eight navigation sections. Check a nested
   [Python API page](api/python.rst), its function anchor and `[source]` backlink;
   open [MATLAB dataset methods](api/matlab/dataset.rst) and
   [workflow configuration](workflows.md). Follow links between guides and APIs.
4. Search for `normalize_intensity` in the site's search UI and open a matching
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

When Python behavior changes, follow the repository test guidance from `src/python`:

```bash
uv run --group dev pytest test/ -v
```

The explicit `dev` group is needed with `UV_NO_DEFAULT_GROUPS=true` above.
The full Python suite is separate from bounded documentation CI. Local build
success does not establish a successful Actions run, deployment, optional-backend
execution or external link availability.
