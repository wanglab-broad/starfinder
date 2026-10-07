"""Python workflow adapter of the stitching block: bigstitcher_engine (Fiji on the macro, then the success check of dataset.xml)."""
import sys
if 'starfinder_path' in snakemake.config:
    sys.path.insert(0, snakemake.config['starfinder_path'] + '/src/python')
from starfinder.dataset.workflow import _run_bigstitcher_engine
_run_bigstitcher_engine(snakemake)
