"""Python workflow adapter of a shared rule: stardist_segmentation (the translated StarDist run; needs the stardist extra)."""
import sys
if 'starfinder_path' in snakemake.config:
    sys.path.insert(0, snakemake.config['starfinder_path'] + '/src/python')
from starfinder.dataset.workflow import _run_stardist_segmentation
_run_stardist_segmentation(snakemake)
