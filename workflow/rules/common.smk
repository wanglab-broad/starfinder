"""
    Common imports, helper functions, and parameters shared across all rule files.
    This file should be included first in the main Snakefile.
"""

### ==================== [ Imports ] =========================

import os
import sys
import glob
import random
import pandas as pd
from pathlib import Path

# Standard-library MATLAB launcher kept next to this file so tests can import it.
sys.path.insert(0, str(Path(workflow.basedir) / 'rules'))
import matlab_launcher

### ==================== [ Backend Selection ] =========================

BACKEND = config.get("backend", "matlab")
if BACKEND not in ("python", "matlab"):
    raise ValueError(f"Unknown backend '{BACKEND}'. Valid options: 'python', 'matlab'")

# The Python-only §2.10 blocks: with a stitching block, stitching-py.smk replaces the four legacy
# stitching rules; with an assembly block, export_sample replaces the legacy sample rules (option W1).
STITCHING_BLOCK = "stitching" in config
ASSEMBLY_BLOCK = "assembly" in config
LEGACY_STITCHING_RULES = ('stitching_preparation', 'create_BigStitcher_macro', 'run_BigStitcher_macro',
                          'create_tile_config')
LEGACY_SAMPLE_RULES = ('create_sample_h5ad', 'create_sample_reads_assignment')
# The Python-only §2.11 report block: with it, report.smk adds report_sample and report_dataset (option WR1).
REPORT_BLOCK = "report" in config

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

def validate_workflow_mode_dependencies():
    """Ensure required rules are configured for each workflow mode.

    In preset modes (direct/subtile/deep), this validates that the required
    rules have their configuration sections defined in the config file.
    This catches configuration errors early rather than at rule execution time.
    """
    mode = WORKFLOW_MODE

    if mode == 'free':
        # Free mode allows any combination - no validation needed
        return

    if mode not in WORKFLOW_PRESETS:
        raise ValueError(
            f"Unknown workflow_mode '{mode}'. "
            f"Valid options: 'free', 'direct', 'subtile', 'deep'"
        )

    # Check that required rules have configuration defined
    required_rules = WORKFLOW_PRESETS[mode]
    rules_config = config.get('rules', {})

    missing_rules = []
    for rule in required_rules:
        if rule not in rules_config:
            missing_rules.append(rule)

    if missing_rules:
        raise ValueError(
            f"Workflow mode '{mode}' requires configuration for rules: {missing_rules}. "
            f"Please add these rule sections to your config file under 'rules:'."
        )

# Run workflow mode validation at module load time
validate_workflow_mode_dependencies()

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
    """Run a MATLAB entry point with the configured launcher.

    matlab_launcher ('path' or 'broad') and matlab_single_thread select the
    invocation; see workflow/rules/matlab_launcher.py.
    """
    matlab_launcher.run_matlab_scripts(config, param_string, matlab_script_name)


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

# FOV membership. The Python backend reads <INPUT>/fovs.csv (sample,fov_id and an optional position_id) in row
# order, restricted by selected_fovs, for every rule (_membership); the MATLAB backend keeps its sample annotation,
# fov_id_pattern, n_fovs and subset keys. MEMBERSHIP is the file the rules that read it declare.
if BACKEND == "python":
    sys.path.insert(0, str(Path(config["starfinder_path"]) / "src" / "python"))
    from starfinder import _layout
    from starfinder.dataset.workflow import _membership, _reference_file
    MEMBERSHIP = str(_layout.fovs_file(INPUT_DIR))
    _MEMBERS = _membership(config, MEMBERSHIP)
    SAMPLE_TO_FOV = {sample: list(fovs) for sample, fovs in _MEMBERS.samples.items()}
    FOV_TO_SAMPLE = {fov: sample for sample, fovs in SAMPLE_TO_FOV.items() for fov in fovs}
    SAMPLE = list(SAMPLE_TO_FOV)
    FOVS = list(_MEMBERS.fovs)
    # The fovID wildcard of every rule matches the run's FOVs only (the Snakefile's wildcard_constraints), so that an
    # explicit target of a FOV outside fovs.csv or selected_fovs resolves to no rule.
    import re as _re
    FOV_CONSTRAINT = "|".join(_re.escape(fov) for fov in FOVS)
    # The codebook of the sequencing rules and reads_assignment: the file the Python-only key codebook (direct_panel
    # in readout_mode direct) names, absolute or relative to INPUT_DIR, else <INPUT>/genes.csv (_reference_file).
    CODEBOOK = str(_reference_file(config))
else:
    # MATLAB reads <INPUT>/genes.csv and the round-first image layout <INPUT>/<round>/<fov>; the keys that name
    # another codebook or layout are Python-only.
    _PYTHON_INPUT_KEYS = [key for key in ('codebook', 'direct_panel', 'image_dir_template')
                          if config.get(key) is not None]
    if _PYTHON_INPUT_KEYS:
        raise ValueError(f"{', '.join(_PYTHON_INPUT_KEYS)} {'is' if len(_PYTHON_INPUT_KEYS) == 1 else 'are'} "
                         "Python-only and need backend: python")
    CODEBOOK = f"{INPUT_DIR}/genes.csv"
    MEMBERSHIP = f"{OUTPUT_DIR}/documents/sample-annotation.csv"
    SAMPLE_ANNOT = pd.read_csv(MEMBERSHIP)
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

## Define specific list of positions to run the pipeline for testing/re-runs (MATLAB backend; the Python backend's
## selected_fovs is applied by _membership above)
if BACKEND == "matlab":
    # The schema gives the subset switches no default (the Snakefile's validate would insert it into Python
    # configurations, which refuse them), so MATLAB's defaults are applied here.
    config.setdefault('subset_range', False)
    config.setdefault('subset_random', False)
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
    if BACKEND == "python":
        # The FOV folder's run record and the reference image of the Python processing rules (none with
        # reference_image_import), as the package lays them out.
        from starfinder._snakemake import _reference_image_outputs, _run_record_outputs
    if STITCHING_BLOCK:
        output_list += [f"{OUTPUT_DIR}/stitching/{sample}/placement.json" for sample in SAMPLE]
    if ASSEMBLY_BLOCK:
        output_list += [f"{OUTPUT_DIR}/export/{sample}/export.json" for sample in SAMPLE]
    if REPORT_BLOCK:
        output_list += [f"{OUTPUT_DIR}/report/{sample}/report.json" for sample in SAMPLE]
        output_list += [f"{OUTPUT_DIR}/report/samples.csv"]
    for current_rule in config['rules']:
        if STITCHING_BLOCK and current_rule in LEGACY_STITCHING_RULES:
            continue
        if ASSEMBLY_BLOCK and current_rule in LEGACY_SAMPLE_RULES:
            continue
        if is_rule_enabled(current_rule):
            if current_rule == 'rsf_single_fov':
                # The Python rsf_single_fov writes the FOV folder's run.json, reference image (none with
                # reference_image_import) and zero-based molecules.csv, MATLAB's its logs, images/ref_merged/
                # and the legacy goodSpots.
                if BACKEND == "python":
                    output_list += _run_record_outputs(OUTPUT_DIR, FOVS)
                    output_list += _reference_image_outputs(config, OUTPUT_DIR, FOVS)
                    output_list += [f"{OUTPUT_DIR}/fovs/{fovID}/molecules.csv" for fovID in FOVS]
                else:
                    output_list += [f"{OUTPUT_DIR}/log/{fovID}_rsf.txt" for fovID in FOVS]
                    output_list += [f"{OUTPUT_DIR}/log/sf_scores/{fovID}.txt" for fovID in FOVS]
                    output_list += [f"{OUTPUT_DIR}/images/ref_merged/{fovID}.tif" for fovID in FOVS]
                    output_list += [f"{OUTPUT_DIR}/signal/{fovID}_goodSpots.csv" for fovID in FOVS]
            elif current_rule == 'rsf_single_fov_seq':
                output_list += [f"{OUTPUT_DIR}/signal/{fovID}_allSpots.csv" for fovID in FOVS]
            elif current_rule == 'gr_single_fov_subtile':
                subtile_ext = "npz" if BACKEND == "python" else "mat"
                if BACKEND == "python":
                    output_list += _run_record_outputs(OUTPUT_DIR, FOVS)
                    output_list += _reference_image_outputs(config, OUTPUT_DIR, FOVS)
                else:
                    output_list += [f"{OUTPUT_DIR}/log/{fovID}_gr.txt" for fovID in FOVS]
                    output_list += [f"{OUTPUT_DIR}/images/ref_merged/{fovID}.tif" for fovID in FOVS]
                output_list += [f"{OUTPUT_DIR}/output/subtile/{fovID}/subtile_coords.csv" for fovID in FOVS]
                output_list += [f"{OUTPUT_DIR}/output/subtile/{fovID}/subtile_data_{i}.{subtile_ext}" for fovID in FOVS for i in N_SUBTILE]
            elif current_rule in ('lrsf_single_fov_subtile', 'deep_rsf_subtile'):
                # The Python subtile rules write each subtile's run.json, MATLAB's the score logs.
                if BACKEND == "python":
                    output_list += _run_record_outputs(OUTPUT_DIR, FOVS, N_SUBTILE)
                else:
                    output_list += [f"{OUTPUT_DIR}/log/sf_scores/{fovID}_{i}.txt" for fovID in FOVS for i in N_SUBTILE]
                output_list += [f"{OUTPUT_DIR}/output/subtile/{fovID}/subtile_goodSpots_{i}.csv" for fovID in FOVS for i in N_SUBTILE]
            elif current_rule == 'deep_create_subtile':
                subtile_ext = "npz" if BACKEND == "python" else "mat"
                output_list += [f"{OUTPUT_DIR}/output/subtile/{fovID}/subtile_coords.csv" for fovID in FOVS]
                output_list += [f"{OUTPUT_DIR}/output/subtile/{fovID}/subtile_data_{i}.{subtile_ext}" for fovID in FOVS for i in N_SUBTILE]
                if BACKEND == "python":
                    output_list += _run_record_outputs(OUTPUT_DIR, FOVS)
            elif current_rule == 'stitch_subtile':
                output_list += [f"{OUTPUT_DIR}/signal/{fovID}_goodSpots.csv" for fovID in FOVS]
            elif current_rule == 'nuclei_registration':
                # The Python rule writes one registration record per FOV, MATLAB's its log and shift table.
                if BACKEND == "python":
                    output_list += [f"{OUTPUT_DIR}/fovs/{fovID}/nuclei_registration.json" for fovID in FOVS]
                else:
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

# The FOV molecule table that reads_assignment and export_sample read, chosen by its producer
# (docs/workflows.md, "Molecule tables"): fovs/{fovID}/molecules.csv of the Python rsf_single_fov, or the
# legacy one-based signal/{fovID}_goodSpots.csv of MATLAB's rsf_single_fov and of stitch_subtile (the
# subtile and deep modes, the ruleorder of the Snakefile). The Python backend asks the package (_molecule_table).
MOLECULE_TABLE = f"{OUTPUT_DIR}/signal/{{fovID}}_goodSpots.csv"
# The images the shared segmentation rules read (segmentation.smk): MATLAB's reference image
# images/ref_merged/ and its channel-name folders of the additional round flamingo. The Python backend
# reads its reference image and the channel-label folders nuclei_registration writes (_shared_image_inputs).
SHARED_IMAGE_INPUTS = {"amplicon_img": f"{OUTPUT_DIR}/images/ref_merged/{{fovID}}.tif",
                       "dapi_img": f"{OUTPUT_DIR}/images/flamingo/DAPI/{{fovID}}.tif",
                       "flamingo_img": f"{OUTPUT_DIR}/images/flamingo/Flamingo/{{fovID}}.tif"}

# Validate only active Python adapters; shared MATLAB configuration stays intact.
if BACKEND == "python":
    from starfinder.dataset import from_workflow_config
    for python_rule in ("rsf_single_fov", "gr_single_fov_subtile",
                        "lrsf_single_fov_subtile", "deep_create_subtile", "deep_rsf_subtile"):
        if is_rule_enabled(python_rule):
            from_workflow_config(config, python_rule)
    # The §2.10 blocks raise here, naming the key, rather than when their rules run; so does the route
    # (check 7 of docs/sample-segmentation-contract.md, without the reuse conditions and the models).
    from starfinder.dataset.workflow import (_check_export_route, _check_sample_rules, _export_config, _membership,
                                             _molecule_table, _report_block, _stage_position_ids, _stitching_block)
    # The annotation-driven legacy sample rules run with backend matlab only; the blocks replace them.
    _check_sample_rules(config)
    MOLECULE_TABLE = str(_molecule_table(config, OUTPUT_DIR, "{fovID}"))
    # The FOV folder's run record of the processing rules (fovs/{fovID}/run.json, a subtile's
    # fovs/{fovID}/subtile_{n_subtile}/run.json), the reference image they write (the file reference_image
    # names; none with reference_image_import, which they never write) and the one its readers read, a single
    # path pattern, so that a rule input named by it is one path (snakemake.input.<name> a string).
    from starfinder._snakemake import (_image_dir_inputs, _reference_image_inputs, _reference_image_outputs,
                                       _run_record_outputs, _shared_image_inputs)
    RUN_RECORD = _run_record_outputs(OUTPUT_DIR, ["{fovID}"])[0]
    SUBTILE_RUN_RECORD = _run_record_outputs(OUTPUT_DIR, ["{fovID}"], ["{n_subtile}"])[0]
    REFERENCE_IMAGE_OUTPUT = _reference_image_outputs(config, OUTPUT_DIR, ["{fovID}"])
    REFERENCE_IMAGE_INPUT = _reference_image_inputs(config, OUTPUT_DIR, ["{fovID}"])[0]
    SHARED_IMAGE_INPUTS = _shared_image_inputs(config, OUTPUT_DIR)
    if STITCHING_BLOCK:
        _stitching_block(config)
        _stage_position_ids(config, _membership(config, MEMBERSHIP))
    if ASSEMBLY_BLOCK:
        _export_config(config)
        _check_export_route(config)
    if REPORT_BLOCK:
        _report_block(config)
