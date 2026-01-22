"""
    Reads assignment rules: Reads assignment, sample h5ad creation
    Uses variables from common.smk
"""

### ==================== [ Reads Assignment ] =========================

rule reads_assignment:
    input:
        expand("{output_dir}/documents/sample-annotation.csv", output_dir=OUTPUT_DIR),
        expand("{dapi_dir}/{{fovID}}.tif", dapi_dir=DAPI_DIR, fovID=FOVS),
        expand("{output_dir}/images/stardist_segmentation/{{fovID}}.tif", output_dir=OUTPUT_DIR, fovID=FOVS),
        expand("{output_dir}/signal/{{fovID}}_goodSpots.csv", output_dir=OUTPUT_DIR, fovID=FOVS),
        expand("{output_dir}/documents/genes.csv", output_dir=OUTPUT_DIR),
        expand("{output_dir}/output/tile_config_{sample}.csv", output_dir=OUTPUT_DIR, sample=SAMPLE),
    output:
        expand("{output_dir}/expr/{{fovID}}/raw.h5ad", output_dir=OUTPUT_DIR, fovID=FOVS),
        expand("{output_dir}/expr/{{fovID}}/reads_assignment.csv", output_dir=OUTPUT_DIR, fovID=FOVS),
    resources:
        mem_mb=config['rules']['reads_assignment']['resources']['mem_mb']
    script:
        "../scripts/reads_assignment.py"

### ==================== [ Create Sample H5AD ] =========================

def get_h5ad_input(wildcards):
    return expand("{output_dir}/expr/{fovID}/raw.h5ad", output_dir=OUTPUT_DIR, fovID=SAMPLE_TO_FOV[wildcards.sample])

rule create_sample_h5ad:
    input:
        expand("{output_dir}/documents/sample-annotation.csv", output_dir=OUTPUT_DIR),
        get_h5ad_input
    output:
        expand("{output_dir}/expr/{{sample}}_raw.h5ad", output_dir=OUTPUT_DIR, sample=SAMPLE)
    resources:
        mem_mb=config['rules']['create_sample_h5ad']['resources']['mem_mb']
    script:
        "../scripts/create_sample_h5ad.py"

### ==================== [ Create Sample Reads Assignment ] =========================

def get_reads_input(wildcards):
    return expand("{output_dir}/expr/{fovID}/reads_assignment.csv", output_dir=OUTPUT_DIR, fovID=SAMPLE_TO_FOV[wildcards.sample])

rule create_sample_reads_assignment:
    input:
        expand("{output_dir}/documents/sample-annotation.csv", output_dir=OUTPUT_DIR),
        get_reads_input
    output:
        expand("{output_dir}/expr/{{sample}}_reads_assignment.csv", output_dir=OUTPUT_DIR, sample=SAMPLE)
    resources:
        mem_mb=config['rules']['create_sample_reads_assignment']['resources']['mem_mb']
    script:
        "../scripts/create_sample_reads_assignment.py"
