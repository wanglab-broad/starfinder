"""
    Utility rules: MAF creation, etc. Included with backend matlab only: the Python stitching block reads
    its explicit stage file (stitching.positions.stage_file) and writes no per-sample copy.
    Uses variables from common.smk
"""

### ==================== [ Create Sample MAF ] =========================

rule create_sample_maf:
    input:
        expand("{output_dir}/sample-annotation.csv", output_dir=DOC_DIR),
        expand("{output_dir}/raw.maf", output_dir=DOC_DIR),
    output:
        expand("{output_dir}/maf/{sample}.maf", output_dir=DOC_DIR, sample=SAMPLE),
    resources:
        mem_mb=8000,
        runtime=2
    script:
        "../scripts/create_sample_maf.py"
