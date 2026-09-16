# STARfinder documentation

STARfinder processes STARmap-related spatial transcriptomics images with
Python and MATLAB backends coordinated by Snakemake.

Start with [getting started](getting-started.md), or look up the
[Python API](api/python.rst) and [MATLAB API](api/matlab.md). To build this site locally, follow
[contributing](contributing.md).

```{note}
The Python API reference includes an export inventory and small executable
examples. Getting started provides a tested synthetic image-to-molecule
quickstart. The MATLAB reference parses project-owned source and links workflow
stages to Python operations; it does not certify MATLAB runtime behavior.
The workflow guide maps rules and configuration to source, with bounded DAG
examples and explicit downstream and cluster limitations. Development recipes
exercise small Python inputs; conventions and troubleshooting explain axes,
channels, thresholds, backend requirements and workflow diagnostics.
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
