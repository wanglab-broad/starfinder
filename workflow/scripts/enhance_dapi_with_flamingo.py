"""Python workflow adapter of a shared rule: enhance_dapi_with_flamingo (enhance_with_flamingo)."""
import sys
if 'starfinder_path' in snakemake.config:
    sys.path.insert(0, snakemake.config['starfinder_path'] + '/src/python')
from starfinder.dataset.workflow import _run_enhance_dapi_with_flamingo
_run_enhance_dapi_with_flamingo(snakemake)
