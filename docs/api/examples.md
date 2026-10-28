# Small API examples

This script exercises independent API contracts using in-memory arrays and a
small TIFF/CSV. It is not an end-to-end microscopy tutorial. Its only random
input is a `(12, 24, 24)` synthetic volume with seed 97.
For a complete TIFF-to-molecule workflow, use the
[minimal Python quickstart](../recipes.md#quick-start).

From `src/python`, choose a new output directory outside the checkout:

```bash
uv run python ../../docs/examples/api_contracts.py /absolute/path/to/api-example
```

The script exits nonzero if a contract assertion fails. It checks detected versus
applied shifts, backward field sampling, normalization/projection, TIFF dtype
round-trip, spot detection, barcode extraction/decoding, dataset CSV origin, and
a small synthetic volume. If SimpleITK is absent, it checks the documented
error and TPS no-fallback behavior. It does not install or execute optional demons.

```{literalinclude} ../examples/api_contracts.py
:language: python
```

## A cellular scene and a real-appearance scene

`cellular_scene.py` generates the routine 2D tissue presets of the cellular scene
(`context-tissue2d-v1`, {doc}`../synthetic-cellular-scene`) and of the real-appearance
scene (`appearance-tissue2d-v1`, {doc}`../synthetic-real-appearance`), saves each, and
prints the truth answers of the two marker pairs (`marker_pairs.csv`) and the numbers of
nuclei, cells and molecules. The routine appearance presets place entries of the
closed-form bank, which is built in memory from scenes of the closed-form generator, so
the example reads no real data and needs no `STARFINDER_APPEARANCE_BANKS`. It takes
about 10 s on one CPU, almost all of it the generation of the two presets. From
`src/python`, with an optional new output directory outside the checkout (without one, a
temporary directory is used and removed):

```bash
uv run python ../../docs/examples/cellular_scene.py /absolute/path/to/cellular-example
```

```{literalinclude} ../examples/cellular_scene.py
:language: python
```
