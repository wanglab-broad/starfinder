"""
    Reads assignment rules: Reads assignment, sample h5ad creation
    Uses variables from common.smk. reads_assignment reads the run's FOV membership (MEMBERSHIP: fovs.csv
    for the Python backend, MATLAB's sample annotation); the two legacy sample rules run with backend
    matlab only (the Python backend's assembly block replaces them).
"""

### ==================== [ Reads Assignment ] =========================

# Without an assembly block the legacy filter reads every sample's tile configuration. With one
# (option W1) the per-FOV files are written without the filter, so no tile configuration is read,
# and the FOV's assignment folder fovs/{fovID}/assignment/<name>/, which export_sample assembles, is saved
# as well; the molecule grid is then that of the FOV's reference image (the file reference_image_import or
# reference_image names, default fovs/{fovID}/reference.tif), which the adapter reads (_assembly_cells).
# READS_ASSIGNMENT_GEOMETRY is the one input that places the FOV: the tile configuration, or that grid.
# The molecule input is MOLECULE_TABLE (common.smk), the file of the configured producer.
if ASSEMBLY_BLOCK:
    from starfinder._snakemake import _assignment_record_outputs, _reference_image_inputs
    READS_ASSIGNMENT_GEOMETRY = _reference_image_inputs(config, OUTPUT_DIR, ["{fovID}"])
    READS_ASSIGNMENT_NAME = (config['assembly'] or {}).get('assignment', 'default')
    READS_ASSIGNMENT_FOLDER = ([] if READS_ASSIGNMENT_NAME is None else
                               _assignment_record_outputs(OUTPUT_DIR, ["{fovID}"], READS_ASSIGNMENT_NAME))
else:
    READS_ASSIGNMENT_GEOMETRY = expand("{output_dir}/output/tile_config_{sample}.csv", output_dir=OUTPUT_DIR,
                                          sample=SAMPLE)
    READS_ASSIGNMENT_FOLDER = []

rule reads_assignment:
    input:
        MEMBERSHIP,
        expand("{dapi_dir}/{{fovID}}.tif", dapi_dir=DAPI_DIR, fovID=FOVS),
        expand("{output_dir}/images/stardist_segmentation/{{fovID}}.tif", output_dir=OUTPUT_DIR, fovID=FOVS),
        MOLECULE_TABLE,
        expand("{output_dir}/documents/genes.csv", output_dir=OUTPUT_DIR),
        READS_ASSIGNMENT_GEOMETRY,
    output:
        expand("{output_dir}/expr/{{fovID}}/raw.h5ad", output_dir=OUTPUT_DIR, fovID=FOVS),
        expand("{output_dir}/expr/{{fovID}}/reads_assignment.csv", output_dir=OUTPUT_DIR, fovID=FOVS),
        READS_ASSIGNMENT_FOLDER,
    resources:
        mem_mb=get_rule_config('reads_assignment', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb'])
    script:
        "../scripts/reads_assignment.py"

if BACKEND == "matlab":

    ### ==================== [ Create Sample H5AD ] =========================

    def get_h5ad_input(wildcards):
        return expand("{output_dir}/expr/{fovID}/raw.h5ad", output_dir=OUTPUT_DIR, fovID=SAMPLE_TO_FOV[wildcards.sample])

    rule create_sample_h5ad:
        input:
            expand("{output_dir}/documents/sample-annotation.csv", output_dir=OUTPUT_DIR),
            get_h5ad_input
        output:
            expand("{output_dir}/expr/{{sample}}_raw.h5ad", output_dir=OUTPUT_DIR, sample=SAMPLE)
        resources:
            mem_mb=get_rule_config('create_sample_h5ad', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb'])
        script:
            "../scripts/create_sample_h5ad.py"

    ### ==================== [ Create Sample Reads Assignment ] =========================

    def get_reads_input(wildcards):
        return expand("{output_dir}/expr/{fovID}/reads_assignment.csv", output_dir=OUTPUT_DIR, fovID=SAMPLE_TO_FOV[wildcards.sample])

    rule create_sample_reads_assignment:
        input:
            expand("{output_dir}/documents/sample-annotation.csv", output_dir=OUTPUT_DIR),
            get_reads_input
        output:
            expand("{output_dir}/expr/{{sample}}_reads_assignment.csv", output_dir=OUTPUT_DIR, sample=SAMPLE)
        resources:
            mem_mb=get_rule_config('create_sample_reads_assignment', 'resources.mem_mb', DEFAULT_RESOURCES['mem_mb'])
        script:
            "../scripts/create_sample_reads_assignment.py"
