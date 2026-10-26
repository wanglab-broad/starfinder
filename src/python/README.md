# STARfinder Python Package

Python backend for the STARfinder spatial transcriptomics pipeline.

Documentation publication URL: <https://wanglab-broad.github.io/starfinder/>.

See the [documentation build guide](../../docs/README.md) for the local Sphinx site
and generated Python API reference.
Start with the [Python image-to-molecule quickstart](../../docs/getting-started.md)
for a complete fixed-seed example using tiny synthetic TIFFs, registration,
spot detection, barcode extraction/filtering, and molecule-level CSV inspection.

## Installation

```bash
cd src/python
uv sync --locked --no-default-groups
```

[Installation](../../docs/get-started/installation.md) lists the optional extras, the
Snakemake environment and the MATLAB requirement.

## Development

```bash
# Run tests
uv run pytest test/ -v

# Generate synthetic inputs only (choose a new external directory)
uv run starfinder synthetic generate --mode e2e --owner Jiahao --preset tiny --seed 42 --output /absolute/path/outside/checkout/synthetic-tiny
```

## Architecture and migration

See [project context](../../CONTEXT.md), [Python migration](../../docs/migration.md),
and [the documentation site](../../docs/index.md), in five sections: Get Started, Workflow, API, Wiki and Benchmark.
Detailed contracts live in the site; operational instructions live in
[AGENTS.md](../../AGENTS.md).
