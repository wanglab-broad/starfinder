
import sys
import subprocess

def reg_test(input_dir, sample, output_dir, fovID, ref_round, dapi_round):

    # Set core_matlab script directory path from config.yaml to add path to MATLAB run
    param_string = (f"'{input_dir}', "
                    f"'{sample}', "
                    f"'{output_dir}', "
                    f"'{fovID}', "
                    f"'{ref_round}', "
                    f"'{dapi_round}'"
                    )
        
    core_matlab_path = f"{snakemake.config['user_dir']}/starfinder/modules/preprocessing/scripts"
    matlab_run_string = f"addpath('{core_matlab_path}'); registration_test({param_string})"
    print(matlab_run_string)
    subprocess.run(["matlab", "-nodisplay -nosplash -nodesktop", "-r", matlab_run_string])

## RUN
input_dir = snakemake.config['input_dir']
sample = snakemake.config['sample']
output_dir = snakemake.config['output_dir']
fovID = snakemake.wildcards['fovID']
ref_round = snakemake.config['ref_round']
dapi_round = snakemake.config['dapi_round']
reg_test(input_dir, sample, output_dir, fovID, ref_round, dapi_round)
    