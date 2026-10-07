"""
    Stitching rules of the Python-only stitching block (docs/stitching-contract.md, "Workflow configuration"),
    included instead of stitching.smk when the configuration has the block. One stitch rule per sample writes
    stitching/{sample}/; with method bigstitcher the macro rule, the Fiji engine rule and the import come first,
    and the import also writes output/tile_config_{sample}.csv with the legacy writer.
    Uses variables from common.smk; the block is validated there (_stitching_block).
"""

STITCHING_METHOD = config['stitching'].get('method', 'bigstitcher')

### ==================== [ Helper Functions ] =========================

def get_stitching_images(wildcards):
    """The sample's reference images, whose grids Dataset.stitch and the import read."""
    return expand("{output_dir}/images/ref_merged/{fovID}.tif", output_dir=OUTPUT_DIR,
                  fovID=SAMPLE_TO_FOV[wildcards.sample])

if STITCHING_METHOD != 'bigstitcher' and not ASSEMBLY_BLOCK and is_rule_enabled('reads_assignment'):
    raise ValueError(f"stitching.method {STITCHING_METHOD} writes no tile_config_{{sample}}.csv, which the legacy "
                     "reads_assignment rule reads; add an assembly block (export_sample) or use method bigstitcher")

if STITCHING_METHOD == 'bigstitcher':
    BIGSTITCHER_DIR = Path(config['stitching']['bigstitcher_save_dir'])

    ### ==================== [ BigStitcher Macro ] =========================

    rule bigstitcher_macro_sample:
        input:
            expand("{output_dir}/documents/sample-annotation.csv", output_dir=OUTPUT_DIR),
            get_stitching_images
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
            expand("{output_dir}/documents/sample-annotation.csv", output_dir=OUTPUT_DIR),
            f"{BIGSTITCHER_DIR}/{{sample}}/dataset.xml",
            f"{BIGSTITCHER_DIR}/{{sample}}/view_map.csv",
            f"{BIGSTITCHER_DIR}/{{sample}}/BigStitcher_macro.ijm",
            get_stitching_images
        output:
            f"{OUTPUT_DIR}/stitching/{{sample}}/placement.json",
            f"{OUTPUT_DIR}/output/tile_config_{{sample}}.csv",
        resources:
            mem_mb=get_rule_config('create_tile_config', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb'])
        script:
            "../scripts/stitch_sample.py"

else:

    ### ==================== [ In-process Placement ] =========================

    rule stitch_sample:
        input:
            expand("{output_dir}/documents/sample-annotation.csv", output_dir=OUTPUT_DIR),
            get_stitching_images
        output:
            f"{OUTPUT_DIR}/stitching/{{sample}}/placement.json"
        resources:
            mem_mb=get_rule_config('stitching_preparation', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb'])
        script:
            "../scripts/stitch_sample.py"
