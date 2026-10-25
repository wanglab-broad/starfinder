"""
    Python backend: Registration rules
    Mirrors registration.smk but uses Python scripts via Snakemake script: directive.
    The processing rules declare their FOV's run.json (RUN_RECORD, in the FOV folder fovs/<fov>/) instead of
    MATLAB's log files, and the reference image they write (REFERENCE_IMAGE_OUTPUT: the file reference_image
    names, none with reference_image_import); nuclei_registration declares its fovs/<fov>/nuclei_registration.json.
    The codebook input is CODEBOOK (common.smk), and the image folders are those of image_dir_template
    (_image_dir_inputs: <INPUT>/<round>/<fov>, or <INPUT>/<fov>/<round>). The inputs are named, and the adapters
    read them by name (snakemake.input.<name>); the rules take no JSON copy of the configuration, which the
    adapters read from snakemake.config (the MATLAB rules keep it, registration.smk).
    Uses variables from common.smk
"""

### ==================== [ Registration and Spot Finding (RSF) pipeline ] =========================

rule rsf_single_fov:
    input:
        codebook=CODEBOOK,
        images=_image_dir_inputs(config, ROUND),
    output:
        RUN_RECORD,
        REFERENCE_IMAGE_OUTPUT,
        expand("{output_dir}/fovs/{{fovID}}/molecules.csv", output_dir=OUTPUT_DIR),
    threads: 4
    resources:
        mem_mb=get_rule_config('rsf_single_fov', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb']),
        runtime=get_rule_config('rsf_single_fov', 'resources.runtime', DEFAULT_RESOURCES['runtime'])
    benchmark:
        f"{OUTPUT_DIR}/log/benchmark/rsf_single_fov/{{fovID}}.txt"
    script:
        "../scripts/rsf_single_fov.py"

### ==================== [ Global Registration with Subtiles ] =========================

rule gr_single_fov_subtile:
    input:
        codebook=CODEBOOK,
        images=_image_dir_inputs(config, ROUND),
    output:
        RUN_RECORD,
        REFERENCE_IMAGE_OUTPUT,
        temp(expand("{output_dir}/output/subtile/{{fovID}}/subtile_coords.csv", output_dir=OUTPUT_DIR)),
        temp(expand("{output_dir}/output/subtile/{{fovID}}/subtile_data_{n_subtile}.npz",
                    output_dir=OUTPUT_DIR, n_subtile=N_SUBTILE)),
    threads: 4
    resources:
        mem_mb=get_rule_config('gr_single_fov_subtile', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb']),
        runtime=make_get_runtime('gr_single_fov_subtile')
    benchmark:
        f"{OUTPUT_DIR}/log/benchmark/gr_single_fov_subtile/{{fovID}}.txt"
    script:
        "../scripts/gr_single_fov_subtile.py"

### ==================== [ Deep Create Subtile ] =========================

rule deep_create_subtile:
    input:
        images=_image_dir_inputs(config, ROUND),
    output:
        RUN_RECORD,
        REFERENCE_IMAGE_OUTPUT,
        temp(expand("{output_dir}/output/subtile/{{fovID}}/subtile_coords.csv", output_dir=OUTPUT_DIR)),
        temp(expand("{output_dir}/output/subtile/{{fovID}}/subtile_data_{n_subtile}.npz",
                    output_dir=OUTPUT_DIR, n_subtile=N_SUBTILE)),
    threads: 4
    resources:
        mem_mb=get_rule_config('deep_create_subtile', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb']),
        runtime=make_get_runtime('deep_create_subtile')
    benchmark:
        f"{OUTPUT_DIR}/log/benchmark/deep_create_subtile/{{fovID}}.txt"
    script:
        "../scripts/deep_create_subtile.py"

### ==================== [ Nuclei Registration ] =========================

additional_round_names = get_additional_round_names()

rule nuclei_registration:
    input:
        images=_image_dir_inputs(config, additional_round_names),
    output:
        expand("{output_dir}/fovs/{{fovID}}/nuclei_registration.json", output_dir=OUTPUT_DIR),
    resources:
        mem_mb=get_rule_config('nuclei_registration', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb'])
    script:
        "../scripts/nuclei_registration.py"

### ==================== [ Rotate Nuclei ] =========================

def get_dapi_input(wildcards):
    from starfinder import _layout
    from starfinder.dataset.workflow import _nuclear_stain
    template = config.get('image_dir_template') or _layout.IMAGE_DIR_TEMPLATE
    # The reference round's file of the declared reference stain (_nuclear_stain: reference_stains), where the
    # MATLAB rule reads dapi_round's *ch04.tif.
    stain = _nuclear_stain(config).channel
    pattern = f"{_layout.image_dir(INPUT_DIR, config['ref_round'], wildcards.fovID, template)}/*{stain}.tif"
    ff = sorted(glob.glob(pattern))
    if len(ff) != 1:
        raise ValueError(f"rotate_nuclei needs exactly one file matching {pattern}; found {len(ff)}: {ff}")
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
