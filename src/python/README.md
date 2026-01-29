# STARfinder Python Package

Python backend for the STARfinder spatial transcriptomics pipeline.

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
