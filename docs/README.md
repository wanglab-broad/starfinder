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

## Software entry points

- [Minimal Python image-to-molecule quickstart](getting-started.md)
- [Project overview and setup](../README.md)
- [Python package documentation](../src/python/README.md)
- [Workflow examples](../example/README.md)
- [Downstream examples](../example/downstream/README.md)

## Development records

Task plans, progress, and run-specific reports are maintained in [Linear](https://linear.app/jiahaoh/document/thesis-and-implementation-workflow-c12f30bffe9f). Historical files formerly stored here are listed in the [migration index](https://linear.app/jiahaoh/document/historical-record-index-and-migration-provenance-881cc5edc52b), with their original paths, checksums, and preserved source snapshots. Historical results require evidence review before reuse.

Keep benchmark runners, reusable configurations, evaluation code, and small fixtures in Git. Store large or run-specific outputs outside the checkout and link their provenance from the corresponding issue. See [agent instructions](../AGENTS.md) for the handoff and artifact convention.
