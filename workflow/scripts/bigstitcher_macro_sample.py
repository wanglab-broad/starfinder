"""Python workflow adapter of the stitching block: bigstitcher_macro_sample (the tiles, the macro and the view map)."""
import sys
if 'starfinder_path' in snakemake.config:
    sys.path.insert(0, snakemake.config['starfinder_path'] + '/src/python')
from starfinder._snakemake import _run_bigstitcher_macro
_run_bigstitcher_macro(snakemake)
