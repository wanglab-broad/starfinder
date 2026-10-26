# Get Started

Install the Python package, then run a small example on a local CPU. The two demos are
executed notebooks in `example/introduction/`, shown here with their outputs. The
single-FOV demo processes one tiny synthetic field of view to a table of molecules; the
multi-FOV / sample demo processes a small synthetic sample of four FOVs to cells, places,
exports and reports it. Both use classical methods only and need no microscopy data,
MATLAB, Snakemake, GPU, model or download; the sample demo needs the `multiview-stitcher`
and `anndata` extras. The same single-FOV route as a script is the
[quickstart](../getting-started.md).

```{toctree}
:maxdepth: 1

Installation <installation>
Single FOV demo <single-fov-demo>
Multi-FOV / sample demo <multi-fov-demo>
```
