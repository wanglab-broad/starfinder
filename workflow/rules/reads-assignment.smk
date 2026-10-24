"""
    Reads assignment rules: Reads assignment, sample h5ad creation
    Uses variables from common.smk. reads_assignment reads the run's FOV membership (MEMBERSHIP: fovs.csv
    for the Python backend, MATLAB's sample annotation); the two legacy sample rules run with backend
    matlab only (the Python backend's assembly block replaces them).
"""

### ==================== [ Reads Assignment ] =========================

# Without an assembly block the legacy filter reads every sample's tile configuration and the rule writes
# expr/{fovID}/raw.h5ad and reads_assignment.csv (and, undeclared, its figures and log.txt). With one
# (option W1) no tile configuration is read; the molecule grid is then that of the FOV's reference image (the
# file reference_image_import or reference_image names, default fovs/{fovID}/reference.tif), which the adapter
# reads (_assembly_cells). With an assignment name the rule writes and declares the FOV's assignment folder
# fovs/{fovID}/assignment/<name>/ alone, which export_sample assembles, and with assembly.previews on the folder
# of its figures fovs/{fovID}/previews/ (a directory); with assignment null (molecules only) the per-FOV files
# without the filter (_reads_assignment_outputs).
# READS_ASSIGNMENT_GEOMETRY is the one input that places the FOV: the tile configuration, or that grid.
# The molecule input is MOLECULE_TABLE (common.smk), the file of the configured producer. The count matrix's genes,
# in their order, are those of the codebook the sequencing rules read (CODEBOOK, common.smk; _reference_genes).
if ASSEMBLY_BLOCK:
    from starfinder._snakemake import _reads_assignment_outputs, _reference_image_inputs
    READS_ASSIGNMENT_GEOMETRY = _reference_image_inputs(config, OUTPUT_DIR, ["{fovID}"])
    READS_ASSIGNMENT_OUTPUTS, READS_ASSIGNMENT_PREVIEWS = _reads_assignment_outputs(config, OUTPUT_DIR)
else:
    READS_ASSIGNMENT_GEOMETRY = expand("{output_dir}/output/tile_config_{sample}.csv", output_dir=OUTPUT_DIR,
                                          sample=SAMPLE)
    READS_ASSIGNMENT_OUTPUTS = [
        expand("{output_dir}/expr/{{fovID}}/raw.h5ad", output_dir=OUTPUT_DIR, fovID=FOVS),
        expand("{output_dir}/expr/{{fovID}}/reads_assignment.csv", output_dir=OUTPUT_DIR, fovID=FOVS),
    ]
    READS_ASSIGNMENT_PREVIEWS = []

rule reads_assignment:
    input:
        MEMBERSHIP,
        expand("{dapi_dir}/{{fovID}}.tif", dapi_dir=DAPI_DIR, fovID=FOVS),
        expand("{output_dir}/images/stardist_segmentation/{{fovID}}.tif", output_dir=OUTPUT_DIR, fovID=FOVS),
        MOLECULE_TABLE,
        CODEBOOK,
        READS_ASSIGNMENT_GEOMETRY,
    output:
        READS_ASSIGNMENT_OUTPUTS,
        [directory(folder) for folder in READS_ASSIGNMENT_PREVIEWS],
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
