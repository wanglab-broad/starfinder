"""Python workflow adapter of the report block: report_sample (Dataset.report of one sample)."""
import sys
if 'starfinder_path' in snakemake.config:
    sys.path.insert(0, snakemake.config['starfinder_path'] + '/src/python')
from starfinder.dataset.workflow import _run_report_sample
_run_report_sample(snakemake)
