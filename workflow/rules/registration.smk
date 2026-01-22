"""
    Registration rules: Global/local registration, nuclei registration
    Uses variables from common.smk
"""

### ==================== [ Registration and Spot Finding (RSF) pipeline ] =========================

rule rsf_single_fov:
    input:
        config['config_path'].replace('.yaml', '.json'),
        expand("{input_dir}/genes.csv", input_dir=INPUT_DIR),
        expand("{input_dir}/{rounds}/{{fovID}}", input_dir=INPUT_DIR, rounds=ROUND),
    output:
        expand("{output_dir}/log/{{fovID}}_rsf.txt", output_dir=OUTPUT_DIR),
        expand("{output_dir}/log/sf_scores/{{fovID}}.txt", output_dir=OUTPUT_DIR),
        expand("{output_dir}/images/ref_merged/{{fovID}}.tif", output_dir=OUTPUT_DIR),
        expand("{output_dir}/signal/{{fovID}}_goodSpots.csv", output_dir=OUTPUT_DIR),
    threads: 4
    resources:
        mem_mb=get_rule_config('rsf_single_fov', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb']),
        runtime=get_rule_config('rsf_single_fov', 'resources.runtime', DEFAULT_RESOURCES['runtime'])
    benchmark:
        f"{OUTPUT_DIR}/log/benchmark/rsf_single_fov/{{fovID}}.txt"
    run:
        param_string = (f"'{input[0]}', "
                        f"'{wildcards.fovID}'"
                        )
        matlab_script_name = 'rsf_single_fov'
        run_matlab_scripts(param_string, matlab_script_name)

### ==================== [ Global Registration with Subtiles ] =========================

rule gr_single_fov_subtile:
    input:
        config['config_path'].replace('.yaml', '.json'),
        expand("{input_dir}/genes.csv", input_dir=INPUT_DIR),
        expand("{input_dir}/{rounds}/{{fovID}}", input_dir=INPUT_DIR, rounds=ROUND),
    output:
        expand("{output_dir}/log/{{fovID}}_gr.txt", output_dir=OUTPUT_DIR),
        expand("{output_dir}/images/ref_merged/{{fovID}}.tif", output_dir=OUTPUT_DIR),
        temp(expand("{output_dir}/output/subtile/{{fovID}}/subtile_coords.csv", output_dir=OUTPUT_DIR)),
        temp(expand("{output_dir}/output/subtile/{{fovID}}/subtile_data_{n_subtile}.mat", output_dir=OUTPUT_DIR, n_subtile=N_SUBTILE)),
    params:
        uger_log=f"{OUTPUT_DIR}/log/gr_single_fov_subtile/{{fovID}}.txt"
    threads: 4
    resources:
        mem_mb=get_rule_config('gr_single_fov_subtile', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb']),
        runtime=make_get_runtime('gr_single_fov_subtile')
    benchmark:
        f"{OUTPUT_DIR}/log/benchmark/gr_single_fov_subtile/{{fovID}}.txt"
    run:
        param_string = (f"'{input[0]}', "
                        f"'{wildcards.fovID}'"
                        )
        matlab_script_name = 'gr_single_fov_subtile'
        run_matlab_scripts(param_string, matlab_script_name)

### ==================== [ Nuclei Registration ] =========================

additional_round_names = get_additional_round_names()

rule nuclei_registration:
    input:
        config['config_path'].replace('.yaml', '.json'),
        expand("{input_dir}/{rounds}/{{fovID}}", input_dir=INPUT_DIR, rounds=additional_round_names),
    output:
        expand("{output_dir}/log/{{fovID}}_nr.txt", output_dir=OUTPUT_DIR),
        expand("{output_dir}/log/gr_shifts/{{fovID}}_nr.txt", output_dir=OUTPUT_DIR),
    resources:
        mem_mb=get_rule_config('nuclei_registration', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb'])
    run:
        param_string = (f"'{input[0]}', "
                        f"'{wildcards.fovID}'"
                        )
        matlab_script_name = 'nuclei_registration'
        run_matlab_scripts(param_string, matlab_script_name)

### ==================== [ Rotate Nuclei ] =========================

def get_dapi_input(wildcards):
    ff = glob.glob(f"{INPUT_DIR}/{config['dapi_round']}/{wildcards.fovID}/*ch04.tif")
    return ff

rule rotate_nuclei:
    input:
       get_dapi_input
    output:
        expand("{dapi_dir}/{{fovID}}.tif", dapi_dir=DAPI_DIR, fovID=FOVS),
    resources:
        mem_mb=get_rule_config('rotate_nuclei', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb'])
    run:
        from scipy.ndimage import rotate
        from skimage.io import imread, imsave
        current_img = imread(input[0])
        if len(current_img.shape) == 3:
            current_img = rotate(current_img, config['rotate_angle'], axes=(1, 2))
        else:
            current_img = rotate(current_img, config['rotate_angle'])

        if config['maximum_projection']:
            current_img = current_img.max(axis=0)

        imsave(output[0], current_img)
