"""
    Stitching rules of the Python-only stitching block (docs/stitching-contract.md, "Workflow configuration"),
    included instead of stitching.smk when the configuration has the block. One stitch rule per sample writes
    stitching/{sample}/; with method bigstitcher the macro rule, the Fiji engine rule and the import come first,
    and without an assembly block the import also writes output/tile_config_{sample}.csv with the legacy writer
    (with one nothing reads it, so it is neither written nor declared).
    Uses variables from common.smk; the block is validated there (_stitching_block).
"""

from starfinder._snakemake import _reference_image_inputs, _stitching_position_inputs

STITCHING_METHOD = config['stitching'].get('method', 'bigstitcher')

### ==================== [ Helper Functions ] =========================

def get_stitching_images(wildcards):
    """The sample's reference images (the files reference_image names), whose grids Dataset.stitch and the import
    read."""
    return _reference_image_inputs(config, OUTPUT_DIR, SAMPLE_TO_FOV[wildcards.sample])

def get_stitching_positions(wildcards):
    """The position file the placement reads (the explicit stage file or supplied table; none for a regular grid), as
    _stitching_position_inputs gives it from the same expression as Dataset.stitch."""
    return _stitching_position_inputs(config)

if STITCHING_METHOD != 'bigstitcher' and not ASSEMBLY_BLOCK and is_rule_enabled('reads_assignment'):
    raise ValueError(f"stitching.method {STITCHING_METHOD} writes no tile_config_{{sample}}.csv, which the legacy "
                     "reads_assignment rule reads; add an assembly block (export_sample) or use method bigstitcher")

if STITCHING_METHOD == 'bigstitcher':
    BIGSTITCHER_DIR = Path(config['stitching']['bigstitcher_save_dir'])
    # The legacy reads_assignment rule reads the tile configuration; with an assembly block no rule does.
    STITCH_SAMPLE_TILE_CONFIG = [] if ASSEMBLY_BLOCK else [f"{OUTPUT_DIR}/output/tile_config_{{sample}}.csv"]

    ### ==================== [ BigStitcher Macro ] =========================

    rule bigstitcher_macro_sample:
        input:
            MEMBERSHIP,
            get_stitching_images,
            get_stitching_positions
        output:
            f"{BIGSTITCHER_DIR}/{{sample}}/BigStitcher_macro.ijm",
            f"{BIGSTITCHER_DIR}/{{sample}}/view_map.csv",
        resources:
            mem_mb=get_rule_config('create_BigStitcher_macro', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb'])
        script:
            "../scripts/bigstitcher_macro_sample.py"

    ### ==================== [ BigStitcher Engine (Fiji) ] =========================

    # fiji_path is read by the script when the rule runs, not here; the rule fails unless
    # dataset.xml holds a Stitching Transform for every view, whatever Fiji's exit status.
    rule bigstitcher_engine:
        input:
            f"{BIGSTITCHER_DIR}/{{sample}}/BigStitcher_macro.ijm",
            f"{BIGSTITCHER_DIR}/{{sample}}/view_map.csv",
        output:
            f"{BIGSTITCHER_DIR}/{{sample}}/dataset.xml"
        threads: 4
        resources:
            mem_mb=get_rule_config('run_BigStitcher_macro', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb']),
            runtime=get_rule_config('run_BigStitcher_macro', 'resources.runtime', DEFAULT_RESOURCES['runtime'])
        benchmark:
            f"{OUTPUT_DIR}/log/benchmark/bigstitcher_engine/{{sample}}.txt"
        script:
            "../scripts/bigstitcher_engine.py"

    ### ==================== [ Import and Legacy Tile Config ] =========================

    rule stitch_sample:
        input:
            MEMBERSHIP,
            f"{BIGSTITCHER_DIR}/{{sample}}/dataset.xml",
            f"{BIGSTITCHER_DIR}/{{sample}}/view_map.csv",
            f"{BIGSTITCHER_DIR}/{{sample}}/BigStitcher_macro.ijm",
            get_stitching_images,
            get_stitching_positions
        output:
            f"{OUTPUT_DIR}/stitching/{{sample}}/placement.json",
            STITCH_SAMPLE_TILE_CONFIG,
        resources:
            mem_mb=get_rule_config('create_tile_config', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb'])
        script:
            "../scripts/stitch_sample.py"

else:

    ### ==================== [ In-process Placement ] =========================

    rule stitch_sample:
        input:
            MEMBERSHIP,
            get_stitching_images,
            get_stitching_positions
        output:
            f"{OUTPUT_DIR}/stitching/{{sample}}/placement.json"
        resources:
            mem_mb=get_rule_config('stitching_preparation', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb'])
        script:
            "../scripts/stitch_sample.py"
