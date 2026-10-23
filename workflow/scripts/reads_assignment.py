"""Python workflow adapter of a shared rule: reads_assignment (assign_molecules on the legacy label file; needs anndata)."""
import sys
if 'starfinder_path' in snakemake.config:
    sys.path.insert(0, snakemake.config['starfinder_path'] + '/src/python')
from starfinder._snakemake import _run_reads_assignment
_run_reads_assignment(snakemake)
