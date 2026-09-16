# Conventions

Python image arrays use `(Z, Y, X, C)` ordering (channel last). A single-channel
volume may use `(Z, Y, X)` where supported by the function, as in
{py:func}`~starfinder.preprocessing.min_max_normalize`.

The Python API's [array and coordinate contracts](api/contracts.md) document
coordinate origins, channel ordering, dtype behavior and registration signs.
[Small API examples](api/examples.md) exercise these conventions.

**Coverage planned:** broader workflow and cross-backend development recipes.
