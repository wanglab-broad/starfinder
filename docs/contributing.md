# Contributing and local preview

## Install the documentation environment

Use a checkout, `uv`, and Python 3.10 or newer. From the repository root:

```bash
cd src/python
uv sync --locked --no-default-groups --group docs
```

The `docs` dependency group in `src/python/pyproject.toml` declares Sphinx 8.x,
MyST Parser 4.x, and PyData Sphinx Theme 0.16.x. `src/python/uv.lock` records exact
resolved versions and platform/Python markers. Use `--locked` to detect stale lock
metadata; no optional package extras are needed for the current site. A first
installation needs access to the package registry (or an existing uv cache).

Sphinx imports the installed `starfinder` checkout, including its top-level
imports, so install the package's runtime dependencies as well as the `docs`
group. The build does not mock imports, download datasets, execute notebooks,
run Snakemake, or start MATLAB. SimpleITK, SpatialData, napari, and a MATLAB
license are not required to build the Python API reference.

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
Check the eight navigation sections, search for `min_max_normalize`, open its
generated API page, and follow the `[source]` link beside its signature. The
separate page-source link shows the documentation markup.

## Edit and validate documentation

Edit MyST Markdown under `docs/`. The configuration enables autodoc, autosummary,
napoleon (NumPy/Google docstrings), and viewcode (Python source links). Add pages
to the toctree in `index.md`; label unfinished material explicitly. Preserve the
README and example entry points. Run the strict build after documentation edits.

When Python behavior changes, follow the repository test guidance from `src/python`:

```bash
uv run pytest test/ -v
```

**Coverage planned:** automated documentation CI, publishing, and the full
contributor checklist. Local build success does not imply deployment or complete
API/example coverage.
