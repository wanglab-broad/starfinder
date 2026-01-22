"""
    Stitching rules: BigStitcher, tile config
    Uses variables from common.smk
"""

### ==================== [ Helper Functions ] =========================

def get_stitching_input(wildcards):
    return expand("{dapi_dir}/{fovID}.tif", dapi_dir=DAPI_DIR, fovID=SAMPLE_TO_FOV[wildcards.sample])

### ==================== [ Stitching Preparation ] =========================

rule stitching_preparation:
    input:
        expand("{output_dir}/documents/sample-annotation.csv", output_dir=OUTPUT_DIR),
        expand("{output_dir}/documents/maf/{{sample}}.maf", output_dir=OUTPUT_DIR),
        get_stitching_input
    output:
        expand("{output_dir}/images/fused/{{sample}}/blank.tif", output_dir=OUTPUT_DIR),
        expand("{output_dir}/images/fused/{{sample}}/grid.csv", output_dir=OUTPUT_DIR),
        expand("{output_dir}/images/fused/{{sample}}/grid.png", output_dir=OUTPUT_DIR),
    resources:
        mem_mb=config['rules']['stitching_preparation']['resources']['mem_mb']
    script:
        "../scripts/stitching_preparation.py"

### ==================== [ BigStitcher Macro Creation ] =========================

rule create_BigStitcher_macro:
    input:
        expand("{output_dir}/images/fused/{{sample}}/grid.csv", output_dir=OUTPUT_DIR)
    output:
        expand("{output_dir}/images/fused/{{sample}}/BigStitcher_macro.ijm", output_dir=OUTPUT_DIR)
    resources:
        mem_mb=config['rules']['create_BigStitcher_macro']['resources']['mem_mb']
    script:
        "../scripts/create_BigStitcher_macro_1.py"

### ==================== [ Run BigStitcher Macro ] =========================

rule run_BigStitcher_macro:
    input:
        expand("{output_dir}/images/fused/{{sample}}/BigStitcher_macro.ijm", output_dir=OUTPUT_DIR),
        expand("{output_dir}/images/fused/{{sample}}/grid.csv", output_dir=OUTPUT_DIR)
    output:
        expand("{output_dir}/images/fused/{{sample}}/DAPI/dataset.xml", output_dir=OUTPUT_DIR)
    params:
        temp_dir=expand("{output_dir}/images/fused/{{sample}}/DAPI", output_dir=OUTPUT_DIR)
    threads: 4
    resources:
        mem_mb=config['rules']['run_BigStitcher_macro']['resources']['mem_mb'],
        runtime=config['rules']['run_BigStitcher_macro']['resources']['runtime']
    benchmark:
        f"{OUTPUT_DIR}/log/benchmark/run_BigStitcher_macro/{{sample}}.txt"
    shell:
        f"{config['fiji_path']} -Djava.io.tmpdir={{params.temp_dir}} --ij2 --headless --run {{input[0]}}"

### ==================== [ Create Tile Config ] =========================

rule create_tile_config:
    input:
        expand("{output_dir}/images/fused/{{sample}}/DAPI/dataset.xml", output_dir=OUTPUT_DIR, sample=SAMPLE),
        expand("{output_dir}/images/fused/{{sample}}/grid.csv", output_dir=OUTPUT_DIR, sample=SAMPLE)
    output:
        expand("{output_dir}/output/tile_config_{{sample}}.csv", output_dir=OUTPUT_DIR, sample=SAMPLE),
        expand("{output_dir}/output/tile_config_{{sample}}.html", output_dir=OUTPUT_DIR, sample=SAMPLE)
    resources:
        mem_mb=config['rules']['create_tile_config']['resources']['mem_mb']
    script:
        "../scripts/create_tile_config.py"
