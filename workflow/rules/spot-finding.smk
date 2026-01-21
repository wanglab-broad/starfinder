""" 
    This snakemake file includes Image Registration workflows
"""

### ==================== [ Basic setting ] =========================

import os 
import glob
import random
import pandas as pd
from pathlib import Path

### ==================== [ Helper functions ] =========================
def run_matlab_scripts(param_string, matlab_script_name):
    matlab_script_path = f"{config['starfinder_path']}/workflow/scripts/spot-finding"
    matlab_run_string = f"addpath('{matlab_script_path}'); {matlab_script_name}({param_string});exit;"
    print(matlab_run_string)
    import subprocess
    subprocess.run(["matlab", "-nodisplay -nosplash -nodesktop", "-r", matlab_run_string])


### ==================== [ Parameters ] =========================
INPUT_DIR = Path(config['root_input_path'], config['dataset_id'], config['sample_id'])
OUTPUT_DIR = Path(config['root_output_path'], config['dataset_id'], config['output_id'])
DAPI_DIR = Path(OUTPUT_DIR, 'images', 'DAPI')

ROUND = [f"round{i}" for i in range(1, config['n_rounds']+1)]
SAMPLE_ANNOT = pd.read_csv(f"{OUTPUT_DIR}/documents/sample-annotation.csv")
SAMPLE_DICT = {}
for current_sample in SAMPLE_ANNOT['sample_id'].unique():
    fov_list = []

    current_start = SAMPLE_ANNOT.loc[SAMPLE_ANNOT['sample_id'] == current_sample, 'fov_start'].values[0]
    current_end = SAMPLE_ANNOT.loc[SAMPLE_ANNOT['sample_id'] == current_sample, 'fov_end'].values[0]

    for i in range(current_start, current_end+1):
        current_fov = config['fov_id_pattern'].format(i=i)
        fov_list.append(current_fov)
    
    SAMPLE_DICT[current_sample] = fov_list

SAMPLE = list(SAMPLE_DICT.keys())
FOVS = sorted([config['fov_id_pattern'].format(i=j) for j in range(1, config['n_fovs']+1)])
if config['rules']['global_registration']['run'] | config['rules']['local_registration']['run']:
    N_SUBTILE = [i for i in range(1, (config['rules']['global_registration']['parameters']['create_subtiles']['sqrt_pieces'])**2+1)]
else:
    N_SUBTILE = [1]
    
## Define specific list of positions tos run the pipeline for testing/re-runs
if config['subset_list']:
    FOVS = [config['fov_id_pattern'].format(i=j) for j in config['subset_list']] 
elif config['subset_range']:   
    FOVS = [config['fov_id_pattern'].format(i=j) for j in range(config['subset_start'], config['subset_end']+1)]
elif config['subset_random']:
    FOVS = random.sample(FOVS, config['n_random_tests'])

### ==================== [ Rules ] =========================

# Global registration
def get_runtime(wildcards, attempt):
    return attempt * config['rules']['spot_finding']['resources']['runtime']

rule spot_finding:
    input:
        config['config_path'].replace('.yaml', '.json'),
        expand("{input_dir}/{rounds}/{{fovID}}", input_dir=INPUT_DIR, rounds=ROUND),
    output:
        expand("{output_dir}/log/{{fovID}}_gr.txt", output_dir=OUTPUT_DIR),
        expand("{output_dir}/log/gr_shifts/{{fovID}}.txt", output_dir=OUTPUT_DIR),
        expand("{output_dir}/images/ref_merged/{{fovID}}.tif", output_dir=OUTPUT_DIR),
        temp(expand("{output_dir}/output/subtile/{{fovID}}/subtile_coords.csv", output_dir=OUTPUT_DIR)),
        temp(expand("{output_dir}/output/subtile/{{fovID}}/subtile_data_{n_subtile}.mat", output_dir=OUTPUT_DIR, n_subtile=N_SUBTILE)),
    params:
        uger_log=f"{OUTPUT_DIR}/log/spot_finding/{{fovID}}.txt"
    threads: 4
    resources:
        mem_mb=config['rules']['spot_finding']['resources']['mem_mb'],
        runtime=get_runtime
    benchmark:
        f"{OUTPUT_DIR}/log/benchmark/spot_finding/{{fovID}}.txt"
    run:
        param_string = (f"'{input[0]}', "
                        f"'{wildcards.fovID}'"
                        )
        matlab_script_name = 'spot_finding'
        run_matlab_scripts(param_string, matlab_script_name)
