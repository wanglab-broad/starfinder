# STARfinder usage examples

For a small Python-only example requiring no downloaded microscopy data, start
with the [canonical image-to-molecule quickstart](../docs/recipes.md#quick-start).
It runs the maintained [quickstart script](../docs/examples/quickstart.py) and
produces molecule-level CSVs; it does not perform segmentation or cell assignment.

Two executed notebooks in `introduction/` are the Get Started demos, also shown as pages of the documentation:
the [single FOV demo](introduction/single_fov_demo.ipynb) runs one synthetic field of view to its molecule table, and
the [multi-FOV / sample demo](introduction/sample_demo.ipynb) runs a synthetic sample of four FOVs to cells, then
stitches, exports and reports it.

The dataset-specific and cluster examples below have separate requirements:

1. ```downstream``` - Downstream demos for the dataset-specific analyses
2. ```sequential_workflow``` - Sequential workflow demo for image processing, including registration, spot finding, stitching, segmentation, and reads assignment
3. ```wanglab``` - Additional examples / scripts for WangLab users 

