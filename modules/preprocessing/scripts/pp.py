"""
    Python wrapper for MATLAB code as subprocess for running in Snakemake
    basically parsing parameters and calling matlab functions in core_matlab.m
"""

import sys
import time
import subprocess

### ==================== [ Helper functions ] =========================
def print_current_time(message=None):
    """
        Prints current time. Used for logging.
    :param message: optional message to show
    :return: none
    """
    current_time = time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(time.time()))
    if message is None:
        print(current_time)
    else:
        print(current_time + " " + message)


def print_check_run_info(run):
    """
        Print and checks run result. Used for verifying subprocess status.
    :param run: subprocess.run object
    """
    print("STDOUT------")
    print(run.stdout)
    print("STDERR------")
    print(run.stderr)
    run.check_returncode()


def rsf(mode, sample, pos):
    """
        Master function for rules in registration.smk including:
            - global_registration
            - local_registration
            - spot_finding
            - stitch
            - nuclei_protein_registration

        Code logic is directed by mode parameter given by the rule that calls this fn
        Parses parameters and calls core_matlab.m function as subprocess
    """
    # Set core_matlab script directory path from config.yaml to add path to MATLAB run
    core_matlab_path = f"{snakemake.config['user_dir']}/{snakemake.config['sample']}/starfinder/modules/rsf/scripts"
    
    # MATLAB dotkit loaded on cluster in broad-uger/broad-jobscript.sh 

    # Parse required params into param_string 
    param_string = (
        f"'{sample}', "
        f"'{mode}', "
        f"'{pos}', "
        f"{snakemake.config['xy']}, "
        f"{snakemake.config['z']}, "
        f"{snakemake.config['ref_round']}, "
        f"{snakemake.config['n_chs']}, "
        f"{snakemake.config['n_rounds']}, "
        f"'{snakemake.config['user_dir']}', "
        f"'{snakemake.config['subdirs']['source_data']}', "
        f"'{snakemake.config['subdirs']['registration']}', "
        f"'{snakemake.config['log_dir']}'"
    )

    # Add mode-specific params into param_string based on rule
    if mode == 'global_registration':
        param_string += f", 'sqrt_pieces', {snakemake.config['sqrt_pieces']}"
    elif mode == 'local_registration':
        param_string += (
            f", 'spotfinding_method', '{snakemake.config['spotfinding_method']}', "
            f"'sqrt_pieces', {snakemake.config['sqrt_pieces']}, "
            f"'subtile', {snakemake.wildcards['subtile']}, "
            f"'voxel_size', {snakemake.config['voxel_size']}, "
            f"'end_bases', {snakemake.config['end_bases']}, "
            f"'barcode_mode', '{snakemake.config['barcode_mode']}', "
            f"'intensity_threshold', {snakemake.config['intensity_threshold']}"
        ) 
        if snakemake.config['barcode_mode'] == 'duo':
            param_string += f", 'split_loc', {snakemake.config['split_loc']}"
    elif mode == 'stitch':
        param_string += (
            f", 'spotfinding_method', '{snakemake.config['spotfinding_method']}', "
            f"'sqrt_pieces', {snakemake.config['sqrt_pieces']}"
        )
    elif mode == 'nuclei_protein_registration':
        param_string += (
            f", 'protein_round', '{snakemake.config['protein_round']}', "
            # Protein stains need to be written in one-by-one since snakemake automatically passes in with single quotes
            f"{parse_protein_stains(snakemake.config['protein_stains'])}"
        )
    else:
        print("invalid rule entered. exiting now.")
        sys.exit()
    
    # Call MATLAB for execution
    # print_current_time(f"Running MATLAB {mode} for {param_string}") #; run ID: {run_id}")
    matlab_run_string = f"addpath('{core_matlab_path}'); core_matlab({param_string})"
    print(matlab_run_string)
    subprocess.run(["matlab", "-nodisplay -nosplash -nodesktop", "-r", matlab_run_string])
    # print_check_run_info(run)
    # print_current_time(f"Finished RSF: {mode}")


## RUN
mode = snakemake.params['mode']
sample = snakemake.config['sample']
pos = snakemake.wildcards['pos']
rsf(mode, sample, pos)
    





    
