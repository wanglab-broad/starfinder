"""Python workflow adapter of the assembly and export blocks: export_sample (Dataset.export of one sample)."""
import sys
if 'starfinder_path' in snakemake.config:
    sys.path.insert(0, snakemake.config['starfinder_path'] + '/src/python')
from starfinder._snakemake import _run_export
_run_export(snakemake)
