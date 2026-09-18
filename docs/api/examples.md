# Small API examples

This script exercises independent API contracts using in-memory arrays and a
small TIFF/CSV. It is not an end-to-end microscopy tutorial. Its only random
input is a `(12, 24, 24)` synthetic volume with seed 97.
For a complete TIFF-to-molecule workflow, use the
[minimal Python quickstart](../getting-started.md).

From `src/python`, choose a new output directory outside the checkout:

```bash
uv run python ../../docs/examples/api_contracts.py /absolute/path/to/api-example
```

The script exits nonzero if a contract assertion fails. It checks detected versus
applied shifts, backward field sampling, normalization/projection, TIFF dtype
round-trip, spot detection, barcode extraction/decoding, dataset CSV origin, and
a small benchmark measurement. If SimpleITK is absent, it checks the documented
error and TPS no-fallback behavior. It does not install or execute optional demons.

```{literalinclude} ../examples/api_contracts.py
:language: python
```
