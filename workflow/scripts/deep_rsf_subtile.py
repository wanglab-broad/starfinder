"""Python workflow adapter: deep_rsf_subtile. Shared keys are translated once."""
import sys
sys.path.insert(0, snakemake.config['starfinder_path'] + '/src/python')
from starfinder._snakemake import _run_workflow
_run_workflow(snakemake, 'deep_rsf_subtile')
