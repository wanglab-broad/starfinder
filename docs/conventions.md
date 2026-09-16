# Conventions

Python image arrays use `(Z, Y, X, C)` ordering (channel last). A single-channel
volume may use `(Z, Y, X)` where supported by the function, as in
{py:func}`~starfinder.preprocessing.min_max_normalize`.

**Coverage planned:** comprehensive coordinate, channel, registration-sign, and
cross-backend conventions with tested examples. Until then, check the function's
docstring and the [repository guidance](https://github.com/wanglab-broad/starfinder/blob/dev/AGENTS.md)
before translating data between backends.
