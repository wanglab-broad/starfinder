"""Python workflow adapter of a shared rule: create_nuclei_amplicon_overlay (composite_nuclei_amplicon)."""
import sys
if 'starfinder_path' in snakemake.config:
    sys.path.insert(0, snakemake.config['starfinder_path'] + '/src/python')
from starfinder.dataset.workflow import _run_nuclei_amplicon_overlay
_run_nuclei_amplicon_overlay(snakemake)
