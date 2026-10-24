"""
    The two rules of the report block (docs/report-contract.md, "Workflow configuration"; option WR1), included
    when the configuration has a report block: report_sample writes report/{sample}/ with Dataset.report after the
    sample's last per-sample output, and report_dataset writes the comparison page report/index.html and
    report/samples.csv with Dataset.compare_samples over every sample's report.
    Uses variables from common.smk; the block is validated there (_report_block).
"""

from starfinder._snakemake import _report_dataset_files, _report_sample_files

# Every sample of the run, in the order of fovs.csv (the report block is Python-only).
REPORT_SAMPLES = list(SAMPLE)

### ==================== [ Helper Functions ] =========================

def get_report_input(wildcards):
    """The sample's last per-sample output: export.json (assembly block), else placement.json (stitching block),
    else its FOVs' run.json records."""
    return _report_sample_files(config, OUTPUT_DIR, SAMPLE_TO_FOV[wildcards.sample], wildcards.sample)[0]

### ==================== [ Sample Report ] =========================

# Its declared output is report.json, which lists every other file of report/{sample}/.
rule report_sample:
    input:
        get_report_input
    output:
        _report_sample_files(config, OUTPUT_DIR, [])[1]
    resources:
        mem_mb=DEFAULT_RESOURCES['mem_mb']
    script:
        "../scripts/report_sample.py"

### ==================== [ Comparison Page ] =========================

rule report_dataset:
    input:
        _report_dataset_files(OUTPUT_DIR, REPORT_SAMPLES)[0]
    output:
        _report_dataset_files(OUTPUT_DIR, REPORT_SAMPLES)[1]
    resources:
        mem_mb=DEFAULT_RESOURCES['mem_mb']
    script:
        "../scripts/report_dataset.py"
