"""
    Python backend: Spot-finding rules
    Mirrors spot-finding.smk but uses Python scripts via Snakemake script: directive.
    Uses variables from common.smk
"""

### ==================== [ Local RSF with Subtiles ] =========================

rule lrsf_single_fov_subtile:
    input:
        config['config_path'].replace('.yaml', '.json'),
        expand("{input_dir}/genes.csv", input_dir=INPUT_DIR),
        expand("{output_dir}/output/subtile/{{fovID}}/subtile_data_{{n_subtile}}.npz", output_dir=OUTPUT_DIR),
    output:
        expand("{output_dir}/log/sf_scores/{{fovID}}_{{n_subtile}}.txt", output_dir=OUTPUT_DIR),
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
        expand("{input_dir}/genes.csv", input_dir=INPUT_DIR),
        expand("{output_dir}/output/subtile/{{fovID}}/subtile_data_{{n_subtile}}.npz", output_dir=OUTPUT_DIR),
    output:
        expand("{output_dir}/log/sf_scores/{{fovID}}_{{n_subtile}}.txt", output_dir=OUTPUT_DIR),
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
    output:
        expand("{output_dir}/signal/{{fovID}}_goodSpots.csv", output_dir=OUTPUT_DIR),
        expand("{output_dir}/signal/{{fovID}}_goodSpots.png", output_dir=OUTPUT_DIR),
    resources:
        mem_mb=get_rule_config('stitch_subtile', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb']),
        runtime=get_rule_config('stitch_subtile', 'resources.runtime', DEFAULT_RESOURCES['runtime'])
    script:
        "../scripts/stitch_subtile.py"
