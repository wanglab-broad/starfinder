# Project documentation

This directory is for maintained software documentation: installation, usage, API behavior, architecture, and reproducible examples.

The documentation publication URL is <https://wanglab-broad.github.io/starfinder/>.
See [publishing and release verification](contributing.md#publish-and-verify-a-release)
for the selected branch, deployment checks and how to identify the live revision.

## Build and preview the site

The site uses Sphinx, MyST Markdown, and the PyData theme. From the repository root:

```bash
cd src/python
uv sync --locked --no-default-groups --group docs
DOCS_OUTPUT=/absolute/path/outside/checkout/starfinder-docs-html
uv run --group docs sphinx-build -n -W --keep-going -b html ../../docs "$DOCS_OUTPUT"
uv run python -m http.server 8000 --bind 127.0.0.1 --directory "$DOCS_OUTPUT"
```

Replace the output path with a fresh external directory, then open
<http://127.0.0.1:8000/> on the build host. Stop the preview with Ctrl-C.
See [contributing](contributing.md) for import requirements and validation details,
and [the site index](index.md) for current coverage. No microscopy data or MATLAB
license is needed to build the site. Exact dependency resolutions are in
`src/python/uv.lock`; the dependency group is named `docs`.

## Site structure

The five sections are Get started, Workflow, Benchmark, API (Python/MATLAB),
and Wiki / Convention. Start with [architecture](architecture.md) and the
[Python migration guide](migration.md) when updating existing callers.
[CONTEXT.md](../CONTEXT.md) is concise orientation; [AGENTS.md](../AGENTS.md)
owns operations. Detailed interfaces remain canonical in the site.
Run `uv run python ../../docs/check_reference.py` from `src/python` to check
export parity and alphabetical reference lists.

## Software entry points

- [Minimal Python image-to-molecule quickstart](getting-started.md)
- [Project overview and setup](../README.md)
- [Python package documentation](../src/python/README.md)
- [Workflow examples](../example/README.md)
- [Downstream examples](../example/downstream/README.md)

## Development records

Historical files formerly stored in this folder were moved out of the repository. An index of them, with their original paths, checksums and preserved source snapshots, is kept outside the repository. Historical results require evidence review before reuse.

Keep benchmark runners, reusable configurations, evaluation code, and small fixtures in Git. Store large or run-specific outputs outside the checkout and record their provenance with the run. See [agent instructions](../AGENTS.md) for the handoff and artifact convention.
