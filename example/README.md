# STARfinder usage examples

For a small Python-only example requiring no downloaded microscopy data, start
with the [canonical image-to-molecule quickstart](../docs/getting-started.md).
It runs the maintained [quickstart script](../docs/examples/quickstart.py) and
produces molecule-level CSVs; it does not perform segmentation or cell assignment.

The [foundation tour notebook](introduction/starfinder_foundation_tour.ipynb) introduces the package design, synthetic scenes, pipeline API, checkpoints and evaluation on one small synthetic field of view.

The dataset-specific and cluster examples below have separate requirements:

1. ```downstream``` - Downstream demos for the dataset-specific analyses
2. ```sequential_workflow``` - Sequential workflow demo for image processing, including registration, spot finding, stitching, segmentation, and reads assignment
3. ```wanglab``` - Additional examples / scripts for WangLab users 

