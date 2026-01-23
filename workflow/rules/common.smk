"""
    Common imports, helper functions, and parameters shared across all rule files.
    This file should be included first in the main Snakefile.
"""

### ==================== [ Imports ] =========================

import os
import glob
import random
import pandas as pd
from pathlib import Path

### ==================== [ Workflow Mode Configuration ] =========================

# Workflow mode: "free" allows mix-and-match, presets enable predefined rule combinations
WORKFLOW_MODE = config.get('workflow_mode', 'free')

# Preset workflow definitions
WORKFLOW_PRESETS = {
    'direct': [
        'rsf_single_fov',
    ],
    'subtile': [
        'gr_single_fov_subtile',
        'lrsf_single_fov_subtile',
        'stitch_subtile',
    ],
    'deep': [
        'deep_create_subtile',
        'deep_rsf_subtile',
        'stitch_subtile',
    ],
}

# Rules that are always available (utilities, preparation, etc.)
# These rules check their individual run flags regardless of workflow_mode
ALWAYS_AVAILABLE_RULES = [
    'rsf_preparation',
    'nuclei_registration',
    'rotate_nuclei',
    'create_nuclei_amplicon_overlay',
    'stardist_segmentation',
    'stitching_preparation',
    'create_BigStitcher_macro',
    'run_BigStitcher_macro',
    'create_tile_config',
    'reads_assignment',
    'create_sample_h5ad',
    'create_sample_reads_assignment',
]

def is_rule_enabled(rule_name):
    """Check if a rule should be enabled based on workflow_mode.

    - workflow_mode: 'free' -> use individual run: True/False flags from config
    - workflow_mode: 'direct'/'subtile'/'deep' -> use preset, ignore individual flags
    - Rules in ALWAYS_AVAILABLE_RULES check their individual run flags regardless of mode
    """
    # Always-available rules use their individual config flags
    if rule_name in ALWAYS_AVAILABLE_RULES:
        return config.get('rules', {}).get(rule_name, {}).get('run', False)

    # Free mode: use individual run flags
    if WORKFLOW_MODE == 'free':
        return config.get('rules', {}).get(rule_name, {}).get('run', False)

    # Preset mode: check if rule is in the preset list
    return rule_name in WORKFLOW_PRESETS.get(WORKFLOW_MODE, [])

def get_rule_config(rule_name, key, default=None):
    """Safely get config value for a rule with default fallback.

    Args:
        rule_name: Name of the rule
        key: Config key path (e.g., 'resources.mem_mb' or 'parameters.threshold')
        default: Default value if key doesn't exist

    Returns:
        Config value or default
    """
    rule_cfg = config.get('rules', {}).get(rule_name, {})

    # Navigate nested keys (e.g., 'resources.mem_mb')
    keys = key.split('.')
    value = rule_cfg
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k)
        else:
            return default
        if value is None:
            return default

    return value if value is not None else default

# Default resource values
DEFAULT_RESOURCES = {
    'mem_mb': 8000,
    'runtime': 30,
}

### ==================== [ Helper functions ] =========================

def yaml_to_json(yaml_file):
    import yaml
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    import json
    json_file = yaml_file.replace('.yaml', '.json')
    with open(json_file, 'w') as f:
        f.write(json.dumps(data, indent=4))

    return json_file

def run_matlab_scripts(param_string, matlab_script_name):
    matlab_script_path = f"{config['starfinder_path']}/workflow/scripts"
    matlab_run_string = f"addpath('{matlab_script_path}'); {matlab_script_name}({param_string});exit;"
    print(matlab_run_string)
    import subprocess

    # Source the Broad useuse script and load MATLAB before running the command
    # This is needed because subprocess spawns a fresh bash that doesn't have
    # the environment from the jobscript's 'use Matlab'
    cmd = f'source /broad/software/scripts/useuse && use Matlab && matlab -nodisplay -nosplash -nodesktop -r "{matlab_run_string}"'
    subprocess.run(cmd, shell=True, executable='/bin/bash', check=True)


def run_fiji_macros(fiji_path, macro_path):
    import subprocess
    process = subprocess.Popen([fiji_path, "--ij2 --headless --run -macro", macro_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stdout, stderr)

def get_additional_round_names():
    return [current_round['round_name'] for current_round in config['additional_round']]

def make_get_runtime(rule_name):
    """Factory function to create rule-specific get_runtime functions.

    This avoids duplicate function definitions with different config keys.
    Usage: resources: runtime=make_get_runtime('rule_name')
    """
    def get_runtime(wildcards, attempt):
        return attempt * get_rule_config(rule_name, 'resources.runtime', DEFAULT_RESOURCES['runtime'])
    return get_runtime

### ==================== [ Parameters ] =========================

INPUT_DIR = Path(config['root_input_path'], config['dataset_id'], config['sample_id'])
OUTPUT_DIR = Path(config['root_output_path'], config['dataset_id'], config['output_id'])
DAPI_DIR = Path(OUTPUT_DIR, 'images', 'DAPI')
FUSE_DIR = Path(OUTPUT_DIR, 'images', 'fused')
SEG_INPUT_DIR = Path(OUTPUT_DIR, 'images', get_rule_config('stardist_segmentation', 'parameters.segmentation_input_folder', 'overlay'))
DOC_DIR = Path(OUTPUT_DIR, 'documents')

ROUND = [f"round{i}" for i in range(1, config['n_rounds']+1)]

SAMPLE_ANNOT = pd.read_csv(f"{OUTPUT_DIR}/documents/sample-annotation.csv")
SAMPLE_TO_FOV = {}
FOV_TO_SAMPLE = {}
for current_sample in SAMPLE_ANNOT['sample_id'].unique():
    fov_list = []

    current_start = SAMPLE_ANNOT.loc[SAMPLE_ANNOT['sample_id'] == current_sample, 'fov_start'].values[0]
    current_end = SAMPLE_ANNOT.loc[SAMPLE_ANNOT['sample_id'] == current_sample, 'fov_end'].values[0]

    for i in range(current_start, current_end+1):
        current_fov = config['fov_id_pattern'].format(i=i)
        fov_list.append(current_fov)
        FOV_TO_SAMPLE[current_fov] = current_sample

    SAMPLE_TO_FOV[current_sample] = fov_list

SAMPLE = list(SAMPLE_TO_FOV.keys())
FOVS = sorted([config['fov_id_pattern'].format(i=j) for j in range(1, config['n_fovs']+1)])

# Determine N_SUBTILE based on enabled rules
# Only gr_single_fov_subtile and deep_create_subtile have sqrt_pieces parameter
if is_rule_enabled('gr_single_fov_subtile'):
    N_SUBTILE = [i for i in range(1, (get_rule_config('gr_single_fov_subtile', 'parameters.create_subtiles.sqrt_pieces', 4))**2+1)]
elif is_rule_enabled('deep_create_subtile'):
    N_SUBTILE = [i for i in range(1, (get_rule_config('deep_create_subtile', 'parameters.create_subtiles.sqrt_pieces', 4))**2+1)]
else:
    N_SUBTILE = [1]

## Define specific list of positions to run the pipeline for testing/re-runs
if config['subset_list']:
    FOVS = [config['fov_id_pattern'].format(i=j) for j in config['subset_list']]
    SAMPLE = list(set([FOV_TO_SAMPLE[fov] for fov in FOVS]))
elif config['subset_range']:
    FOVS = [config['fov_id_pattern'].format(i=j) for j in range(config['subset_start'], config['subset_end']+1)]
elif config['subset_random']:
    FOVS = random.sample(FOVS, config['n_random_tests'])

### ==================== [ Dynamic Output Function ] =========================

def get_overall_output(wildcards):
    """Create dynamic output list for all rules based on workflow_mode and config."""
    output_list = []
    for current_rule in config['rules']:
        if is_rule_enabled(current_rule):
            if current_rule == 'rsf_single_fov':
                output_list += [f"{OUTPUT_DIR}/log/{fovID}_rsf.txt" for fovID in FOVS]
                output_list += [f"{OUTPUT_DIR}/log/sf_scores/{fovID}.txt" for fovID in FOVS]
                output_list += [f"{OUTPUT_DIR}/images/ref_merged/{fovID}.tif" for fovID in FOVS]
                output_list += [f"{OUTPUT_DIR}/signal/{fovID}_goodSpots.csv" for fovID in FOVS]
            elif current_rule == 'rsf_single_fov_seq':
                output_list += [f"{OUTPUT_DIR}/signal/{fovID}_allSpots.csv" for fovID in FOVS]
            elif current_rule == 'gr_single_fov_subtile':
                output_list += [f"{OUTPUT_DIR}/log/{fovID}_gr.txt" for fovID in FOVS]
                output_list += [f"{OUTPUT_DIR}/images/ref_merged/{fovID}.tif" for fovID in FOVS]
                output_list += [f"{OUTPUT_DIR}/output/subtile/{fovID}/subtile_coords.csv" for fovID in FOVS]
                output_list += [f"{OUTPUT_DIR}/output/subtile/{fovID}/subtile_data_{i}.mat" for fovID in FOVS for i in N_SUBTILE]
            elif current_rule == 'lrsf_single_fov_subtile':
                output_list += [f"{OUTPUT_DIR}/log/sf_scores/{fovID}_{i}.txt" for fovID in FOVS for i in N_SUBTILE]
                output_list += [f"{OUTPUT_DIR}/output/subtile/{fovID}/subtile_goodSpots_{i}.csv" for fovID in FOVS for i in N_SUBTILE]
            elif current_rule == 'deep_create_subtile':
                output_list += [f"{OUTPUT_DIR}/output/subtile/{fovID}/subtile_coords.csv" for fovID in FOVS]
                output_list += [f"{OUTPUT_DIR}/output/subtile/{fovID}/subtile_data_{i}.mat" for fovID in FOVS for i in N_SUBTILE]
            elif current_rule == 'deep_rsf_subtile':
                output_list += [f"{OUTPUT_DIR}/log/sf_scores/{fovID}_{i}.txt" for fovID in FOVS for i in N_SUBTILE]
                output_list += [f"{OUTPUT_DIR}/output/subtile/{fovID}/subtile_goodSpots_{i}.csv" for fovID in FOVS for i in N_SUBTILE]
            elif current_rule == 'stitch_subtile':
                output_list += [f"{OUTPUT_DIR}/signal/{fovID}_goodSpots.csv" for fovID in FOVS]
            elif current_rule == 'nuclei_registration':
                output_list += [f"{OUTPUT_DIR}/log/{fovID}_nr.txt" for fovID in FOVS]
                output_list += [f"{OUTPUT_DIR}/log/gr_shifts/{fovID}_nr.txt" for fovID in FOVS]
            elif current_rule == 'rotate_nuclei':
                output_list += [f"{DAPI_DIR}/{fovID}.tif" for fovID in FOVS]
            elif current_rule == 'create_nuclei_amplicon_overlay':
                output_list += [f"{OUTPUT_DIR}/images/overlay/{fovID}.tif" for fovID in FOVS]
            elif current_rule == 'enhance_dapi_with_flamingo':
                output_list += [f"{OUTPUT_DIR}/images/flamingo/enhanced_DAPI/{fovID}.tif" for fovID in FOVS]
            elif current_rule == 'stardist_segmentation':
                output_list += [f"{OUTPUT_DIR}/images/stardist_segmentation/{fovID}.tif" for fovID in FOVS]
            elif current_rule == 'stitching_preparation':
                output_list += [f"{OUTPUT_DIR}/images/fused/{sample}/blank.tif" for sample in SAMPLE]
                output_list += [f"{OUTPUT_DIR}/images/fused/{sample}/grid.csv" for sample in SAMPLE]
                output_list += [f"{OUTPUT_DIR}/images/fused/{sample}/grid.png" for sample in SAMPLE]
            elif current_rule == 'create_BigStitcher_macro':
                output_list += [f"{OUTPUT_DIR}/images/fused/{sample}/BigStitcher_macro.ijm" for sample in SAMPLE]
            elif current_rule == 'run_BigStitcher_macro':
                output_list += [f"{OUTPUT_DIR}/images/fused/{sample}/DAPI/dataset.xml" for sample in SAMPLE]
            elif current_rule == 'create_tile_config':
                output_list += [f"{OUTPUT_DIR}/output/tile_config_{sample}.csv" for sample in SAMPLE]
                output_list += [f"{OUTPUT_DIR}/output/tile_config_{sample}.html" for sample in SAMPLE]
            elif current_rule == 'reads_assignment':
                output_list += [f"{OUTPUT_DIR}/expr/{fovID}/raw.h5ad" for fovID in FOVS]
                output_list += [f"{OUTPUT_DIR}/expr/{fovID}/reads_assignment.csv" for fovID in FOVS]
            elif current_rule == 'create_sample_h5ad':
                output_list += [f"{OUTPUT_DIR}/expr/{sample}_raw.h5ad" for sample in SAMPLE]
            elif current_rule == 'create_sample_reads_assignment':
                output_list += [f"{OUTPUT_DIR}/expr/{sample}_reads_assignment.csv" for sample in SAMPLE]

    return output_list

### ==================== [ Preparation Rule ] =========================

rule rsf_preparation:
    input:
        yaml_config = config['config_path']
    output:
        json_config = config['config_path'].replace('.yaml', '.json')
    run:
        json_path = yaml_to_json(config['config_path'])
