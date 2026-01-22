"""
    Spot-finding rules: RSF variants, deep RSF, spot filtering
    Uses variables from common.smk
"""

### ==================== [ Local RSF with Subtiles ] =========================

rule lrsf_single_fov_subtile:
    input:
        config['config_path'].replace('.yaml', '.json'),
        expand("{input_dir}/genes.csv", input_dir=INPUT_DIR),
        expand("{output_dir}/output/subtile/{{fovID}}/subtile_data_{{n_subtile}}.mat", output_dir=OUTPUT_DIR),
    output:
        expand("{output_dir}/log/sf_scores/{{fovID}}_{{n_subtile}}.txt", output_dir=OUTPUT_DIR),
        temp(expand("{output_dir}/output/subtile/{{fovID}}/subtile_goodSpots_{{n_subtile}}.csv", output_dir=OUTPUT_DIR)),
    threads: 4
    resources:
        mem_mb=config['rules']['lrsf_single_fov_subtile']['resources']['mem_mb'],
        runtime=make_get_runtime('lrsf_single_fov_subtile')
    benchmark:
        f"{OUTPUT_DIR}/log/benchmark/lrsf_single_fov_subtile/{{fovID}}_{{n_subtile}}.txt"
    run:
        param_string = (f"'{input[0]}', "
                        f"'{input[2]}'"
                        )
        matlab_script_name = 'lrsf_single_fov_subtile'
        run_matlab_scripts(param_string, matlab_script_name)

### ==================== [ Deep Create Subtile ] =========================

rule deep_create_subtile:
    input:
        config['config_path'].replace('.yaml', '.json'),
        expand("{input_dir}/{rounds}/{{fovID}}", input_dir=INPUT_DIR, rounds=ROUND),
    output:
        expand("{output_dir}/images/ref_merged/{{fovID}}.tif", output_dir=OUTPUT_DIR),
        temp(expand("{output_dir}/output/subtile/{{fovID}}/subtile_coords.csv", output_dir=OUTPUT_DIR)),
        temp(expand("{output_dir}/output/subtile/{{fovID}}/subtile_data_{n_subtile}.mat", output_dir=OUTPUT_DIR, n_subtile=N_SUBTILE)),
    threads: 4
    resources:
        mem_mb=config['rules']['deep_create_subtile']['resources']['mem_mb'],
        runtime=make_get_runtime('deep_create_subtile')
    benchmark:
        f"{OUTPUT_DIR}/log/benchmark/deep_create_subtile/{{fovID}}.txt"
    run:
        param_string = (f"'{input[0]}', "
                        f"'{wildcards.fovID}'"
                        )
        matlab_script_name = 'deep_create_subtile'
        run_matlab_scripts(param_string, matlab_script_name)

### ==================== [ Deep RSF Subtile ] =========================

rule deep_rsf_subtile:
    input:
        config['config_path'].replace('.yaml', '.json'),
        expand("{input_dir}/genes.csv", input_dir=INPUT_DIR),
        expand("{output_dir}/output/subtile/{{fovID}}/subtile_data_{{n_subtile}}.mat", output_dir=OUTPUT_DIR),
    output:
        expand("{output_dir}/log/sf_scores/{{fovID}}_{{n_subtile}}.txt", output_dir=OUTPUT_DIR),
        temp(expand("{output_dir}/output/subtile/{{fovID}}/subtile_goodSpots_{{n_subtile}}.csv", output_dir=OUTPUT_DIR)),
    threads: 4
    resources:
        mem_mb=config['rules']['deep_rsf_subtile']['resources']['mem_mb'],
        runtime=make_get_runtime('deep_rsf_subtile')
    benchmark:
        f"{OUTPUT_DIR}/log/benchmark/deep_rsf_subtile/{{fovID}}_{{n_subtile}}.txt"
    run:
        param_string = (f"'{input[0]}', "
                        f"'{input[2]}'"
                        )
        matlab_script_name = 'deep_rsf_subtile'
        run_matlab_scripts(param_string, matlab_script_name)

### ==================== [ RSF Single FOV Sequential ] =========================

rule rsf_single_fov_seq:
    input:
        config['config_path'].replace('.yaml', '.json'),
        expand("{input_dir}/{rounds}/{{fovID}}", input_dir=INPUT_DIR, rounds=ROUND),
    output:
        expand("{output_dir}/signal/{{fovID}}_allSpots.csv", output_dir=OUTPUT_DIR),
    threads: 2
    resources:
        mem_mb=config['rules']['rsf_single_fov_seq']['resources']['mem_mb'],
        runtime=config['rules']['rsf_single_fov_seq']['resources']['runtime']
    run:
        param_string = (f"'{input[0]}', "
                        f"'{wildcards.fovID}'"
                        )
        matlab_script_name = 'rsf_single_fov_seq'
        run_matlab_scripts(param_string, matlab_script_name)

### ==================== [ Stitch Subtile Data ] =========================

rule stitch_subtile:
    input:
        expand("{output_dir}/output/subtile/{{fovID}}/subtile_coords.csv", output_dir=OUTPUT_DIR),
        expand("{output_dir}/output/subtile/{{fovID}}/subtile_goodSpots_{n_subtile}.csv", output_dir=OUTPUT_DIR, n_subtile=N_SUBTILE),
    output:
        expand("{output_dir}/signal/{{fovID}}_goodSpots.csv", output_dir=OUTPUT_DIR),
        expand("{output_dir}/signal/{{fovID}}_goodSpots.png", output_dir=OUTPUT_DIR),
    resources:
        mem_mb=config['rules']['stitch_subtile']['resources']['mem_mb'],
        runtime=config['rules']['stitch_subtile']['resources']['runtime']
    script:
        "../scripts/stitch_subtile.py"
