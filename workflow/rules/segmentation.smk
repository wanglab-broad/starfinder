"""
    Segmentation rules: StarDist, DAPI enhancement
    Uses variables from common.smk
"""

### ==================== [ DAPI Enhancement with Flamingo ] =========================

rule enhance_dapi_with_flamingo:
    input:
        dapi_img=expand("{output_dir}/images/flamingo/DAPI/{{fovID}}.tif", output_dir=OUTPUT_DIR),
        flamingo_img=expand("{output_dir}/images/flamingo/Flamingo/{{fovID}}.tif", output_dir=OUTPUT_DIR)
    output:
        eh_dapi_img=expand("{output_dir}/images/flamingo/enhanced_DAPI/{{fovID}}.tif", output_dir=OUTPUT_DIR)
    resources:
        mem_mb=get_rule_config('enhance_dapi_with_flamingo', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb'])
    script:
        "../scripts/enhance_dapi_with_flamingo.py"

### ==================== [ StarDist Segmentation ] =========================

rule stardist_segmentation:
    input:
        expand("{seg_input_dir}/{{fovID}}.tif", seg_input_dir=SEG_INPUT_DIR)
    output:
        expand("{output_dir}/images/stardist_segmentation/{{fovID}}.tif", output_dir=OUTPUT_DIR)
    conda:
        f"{config['envs_path']}/stardist"
    threads: 2
    resources:
        mem_mb=get_rule_config('stardist_segmentation', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb']),
        runtime=get_rule_config('stardist_segmentation', 'resources.runtime', DEFAULT_RESOURCES['runtime'])
    benchmark:
        f"{OUTPUT_DIR}/log/benchmark/stardist_segmentation/{{fovID}}.txt"
    script:
        "../scripts/stardist_segmentation.py"

### ==================== [ Nuclei Amplicon Overlay ] =========================

rule create_nuclei_amplicon_overlay:
    input:
        dapi_img=expand("{output_dir}/images/DAPI/{{fovID}}.tif", output_dir=OUTPUT_DIR),
        amplicon_img=expand("{output_dir}/images/ref_merged/{{fovID}}.tif", output_dir=OUTPUT_DIR)
    output:
        expand("{output_dir}/images/overlay/{{fovID}}.tif", output_dir=OUTPUT_DIR)
    resources:
        mem_mb=get_rule_config('create_nuclei_amplicon_overlay', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb'])
    script:
        "../scripts/create_nuclei_amplicon_overlay.py"
