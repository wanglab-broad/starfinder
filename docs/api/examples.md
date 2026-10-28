# Small API examples

Small scripts for exploring the Python API. The
[output guide](../recipes.md) shows a TIFF-to-molecule workflow; the
[array reference](contracts.md) lists coordinate and channel conventions.

## API contracts

From `src/python`, choose a new output directory outside the checkout:

::::{dropdown} Exercise independent contracts with small arrays and a TIFF/CSV

```bash
uv run python ../../docs/examples/api_contracts.py /absolute/path/to/api-example
```

```{literalinclude} ../examples/api_contracts.py
:language: python
```

::::

## Cellular and real-appearance scenes

The [cellular scene guide](../synthetic-cellular-scene.md) and
[real-appearance scene guide](../synthetic-real-appearance.md) introduce these scenes;
the [synthetic reference](../synthetic-reference.md) lists their presets and files.

::::{dropdown} Generate two scenes and print their marker-pair truth

From `src/python`, supply a new output directory outside the checkout, or omit it
to use a temporary directory:

```bash
uv run python ../../docs/examples/cellular_scene.py /absolute/path/to/cellular-example
```

```{literalinclude} ../examples/cellular_scene.py
:language: python
```

::::
