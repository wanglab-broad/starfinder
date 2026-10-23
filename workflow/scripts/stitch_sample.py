"""Python workflow adapter of the stitching block: stitch_sample (Dataset.stitch, or the BigStitcher import and the legacy tile_config writer)."""
import sys
if 'starfinder_path' in snakemake.config:
    sys.path.insert(0, snakemake.config['starfinder_path'] + '/src/python')
from starfinder._snakemake import _run_stitching
_run_stitching(snakemake)
