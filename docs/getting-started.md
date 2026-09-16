# Getting started

## Install the Python package

From a STARfinder checkout with Python 3.10 or newer and `uv` installed:

```bash
cd src/python
uv sync --locked --no-default-groups
```

This installs the package and its runtime dependencies. Building the documentation
also needs the separate `docs` group; see [contributing](contributing.md) for the
complete install, build, and preview commands.

## Try a small array

From `src/python`, run this deterministic, in-memory example:

```bash
uv run python - <<'PY'
import numpy as np
from starfinder.preprocessing import min_max_normalize

volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
normalized = min_max_normalize(volume)
assert normalized.shape == volume.shape
assert normalized.dtype == np.uint8
assert normalized.min() == 0 and normalized.max() == 255
print(normalized.shape, normalized.dtype)
PY
```

Expected output: `(2, 3, 4) uint8`. This only demonstrates the
{py:func}`~starfinder.preprocessing.min_max_normalize` API; it uses no image
files or random seed.

## Coverage still planned

A tested minimal image-to-reads quickstart is not yet included. Consult the
[existing examples](https://github.com/wanglab-broad/starfinder/blob/dev/example/README.md)
for current entry points and their individual data/software requirements.
