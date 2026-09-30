"""Python workflow adapter: nuclei_registration (other rounds registered to the reference round's ch04 stain)."""
import sys
sys.path.insert(0, snakemake.config['starfinder_path'] + '/src/python')
from starfinder.dataset.workflow import _run_nuclei_registration
_run_nuclei_registration(snakemake)
