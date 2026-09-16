# STARfinder documentation

STARfinder processes STARmap-related spatial transcriptomics images with
Python and MATLAB backends coordinated by Snakemake.

Start with [getting started](getting-started.md), or look up the
[Python API](api/python.rst). To build this site locally, follow
[contributing](contributing.md).

```{note}
The Python API reference includes an export inventory and small executable
examples. Getting started provides a tested synthetic image-to-molecule
quickstart. The other pages identify coverage that is still planned, including
the MATLAB reference and workflow guide.
```

```{toctree}
:maxdepth: 2
:caption: Documentation

getting-started
workflows
api/python
api/matlab
recipes
conventions
troubleshooting
contributing
```

## Existing entry points

The repository's [project README](https://github.com/wanglab-broad/starfinder/blob/dev/README.md),
[Python README](https://github.com/wanglab-broad/starfinder/blob/dev/src/python/README.md),
[workflow examples](https://github.com/wanglab-broad/starfinder/blob/dev/example/README.md),
and [downstream examples](https://github.com/wanglab-broad/starfinder/blob/dev/example/downstream/README.md)
remain available. These links follow the development branch; use your checkout's
copies when working with a different revision.
