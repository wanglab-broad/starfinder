"""
    Segmentation rules: StarDist, DAPI enhancement
    Uses variables from common.smk, whose SHARED_IMAGE_INPUTS names the images these rules read: the reference
    image and the channel folders of the flamingo round (docs/workflows.md, "Nuclei registration").
"""

### ==================== [ DAPI Enhancement with Flamingo ] =========================

rule enhance_dapi_with_flamingo:
    input:
        dapi_img=SHARED_IMAGE_INPUTS['dapi_img'],
        flamingo_img=SHARED_IMAGE_INPUTS['flamingo_img']
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
        amplicon_img=SHARED_IMAGE_INPUTS['amplicon_img']
    output:
        expand("{output_dir}/images/overlay/{{fovID}}.tif", output_dir=OUTPUT_DIR)
    resources:
        mem_mb=get_rule_config('create_nuclei_amplicon_overlay', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb'])
    script:
        "../scripts/create_nuclei_amplicon_overlay.py"
