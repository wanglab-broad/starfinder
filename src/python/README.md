# STARfinder Python Package

Python backend for the STARfinder spatial transcriptomics pipeline.

See the [documentation build guide](../../docs/README.md) for the local Sphinx site
and generated Python API reference.

## Installation

```bash
cd src/python
uv sync
```

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Generate synthetic test data
uv run python -m starfinder.testing --preset mini --output tests/fixtures/synthetic/mini
```
