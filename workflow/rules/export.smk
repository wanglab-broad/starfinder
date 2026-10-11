"""
    The one per-sample export rule of the assembly and export blocks (docs/sample-output-contract.md and
    docs/assembly-contract.md, "Workflow configuration"; option W1), included when the configuration has an
    assembly block: export_sample runs every part of Dataset.export and writes export/{sample}/. It replaces
    create_sample_h5ad and create_sample_reads_assignment, which do not run with the block. Its inputs follow
    the route (docs/sample-segmentation-contract.md, "The entry"): on the route per FOV the per-FOV outputs of
    reads_assignment; on the route after fusion the molecule tables and the reference and source images.
    Uses variables from common.smk; the blocks and the route are validated there (_export_config,
    _check_export_route).
"""

from starfinder.dataset.workflow import _export_sample_files

### ==================== [ Helper Functions ] =========================

def get_export_input(wildcards):
    """The sample's placement.json, then by route its per-FOV assignment outputs (route per FOV), or its goodSpots
    tables (molecules only) with the reference and source images (route after fusion)."""
    return _export_sample_files(config, OUTPUT_DIR, SAMPLE_TO_FOV[wildcards.sample], wildcards.sample)[0]

### ==================== [ Export Sample ] =========================

# Its declared output is export.json, which lists every other file, so a part that cannot run
# (an extra not installed) is recorded not_run and does not fail the rule.
rule export_sample:
    input:
        get_export_input
    output:
        _export_sample_files(config, OUTPUT_DIR, [])[1]
    resources:
        mem_mb=get_rule_config('create_sample_h5ad', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb'])
    script:
        "../scripts/export_sample.py"
