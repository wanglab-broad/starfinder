"""
    Python backend: Spot-finding rules
    Mirrors spot-finding.smk but uses Python scripts via Snakemake script: directive.
    The subtile rules declare the subtile's run.json (SUBTILE_RUN_RECORD, fovs/<fov>/subtile_<n>/run.json)
    instead of MATLAB's score logs; stitch_subtile reads the FOV's reference image (REFERENCE_IMAGE_INPUT, one path,
    its input reference). The codebook input is CODEBOOK (common.smk).
    Uses variables from common.smk
"""

### ==================== [ Local RSF with Subtiles ] =========================

rule lrsf_single_fov_subtile:
    input:
        config['config_path'].replace('.yaml', '.json'),
        CODEBOOK,
        expand("{output_dir}/output/subtile/{{fovID}}/subtile_data_{{n_subtile}}.npz", output_dir=OUTPUT_DIR),
    output:
        SUBTILE_RUN_RECORD,
        temp(expand("{output_dir}/output/subtile/{{fovID}}/subtile_goodSpots_{{n_subtile}}.csv", output_dir=OUTPUT_DIR)),
    threads: 4
    resources:
        mem_mb=get_rule_config('lrsf_single_fov_subtile', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb']),
        runtime=make_get_runtime('lrsf_single_fov_subtile')
    benchmark:
        f"{OUTPUT_DIR}/log/benchmark/lrsf_single_fov_subtile/{{fovID}}_{{n_subtile}}.txt"
    script:
        "../scripts/lrsf_single_fov_subtile.py"

### ==================== [ Deep RSF Subtile ] =========================

rule deep_rsf_subtile:
    input:
        config['config_path'].replace('.yaml', '.json'),
        CODEBOOK,
        expand("{output_dir}/output/subtile/{{fovID}}/subtile_data_{{n_subtile}}.npz", output_dir=OUTPUT_DIR),
    output:
        SUBTILE_RUN_RECORD,
        temp(expand("{output_dir}/output/subtile/{{fovID}}/subtile_goodSpots_{{n_subtile}}.csv", output_dir=OUTPUT_DIR)),
    threads: 4
    resources:
        mem_mb=get_rule_config('deep_rsf_subtile', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb']),
        runtime=make_get_runtime('deep_rsf_subtile')
    benchmark:
        f"{OUTPUT_DIR}/log/benchmark/deep_rsf_subtile/{{fovID}}_{{n_subtile}}.txt"
    script:
        "../scripts/deep_rsf_subtile.py"

### ==================== [ Stitch Subtile Data ] =========================

rule stitch_subtile:
    input:
        expand("{output_dir}/output/subtile/{{fovID}}/subtile_coords.csv", output_dir=OUTPUT_DIR),
        expand("{output_dir}/output/subtile/{{fovID}}/subtile_goodSpots_{n_subtile}.csv", output_dir=OUTPUT_DIR, n_subtile=N_SUBTILE),
        reference=REFERENCE_IMAGE_INPUT,
    output:
        expand("{output_dir}/signal/{{fovID}}_goodSpots.csv", output_dir=OUTPUT_DIR),
        expand("{output_dir}/signal/{{fovID}}_goodSpots.png", output_dir=OUTPUT_DIR),
    resources:
        mem_mb=get_rule_config('stitch_subtile', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb']),
        runtime=get_rule_config('stitch_subtile', 'resources.runtime', DEFAULT_RESOURCES['runtime'])
    script:
        "../scripts/stitch_subtile.py"
