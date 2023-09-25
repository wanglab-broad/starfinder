"""
This is the Snakefile defining rules to run computational image processing.

Do not attempt to run the entire pipeline (aka rule: "starfinder") unless all previous 
modules have been completed, as there will be missing intermediary files from FIJI.

Customize copies of this Snakefile and config.yaml in your analysis directory 
as needed.
"""

from os.path import join
configfile: "config/config.yaml"

# [======================== Set tiles to run ========================]

## Define specific list of positions to run the pipeline for testing/re-runs

# for all tiles: 
TILES = [f"Position{str(x).zfill(3)}" for x in range(1, config['n_tiles']+1)]

# for subset of tiles:
# TILE_List = [1, 3]
# TILES = [f"Position{str(x).zfill(3)}" for x in TILE_List] 


# [======================== Import modules ========================]

module rsf:
    snakefile:
        # here, it is also possible to provide a plain raw URL 
        "modules/rsf/Snakefile"
    config: config     

module seg:
    snakefile:
        "modules/seg/Snakefile"
    config: config

module prestitch:
    snakefile:
        "modules/stitch/Snakefile"
    config: config

use rule * from rsf 
use rule * from seg 
use rule * from prestitch


# [======================== Define final stitching rules  ========================]

# Helper function to output filepath string 
def getOutpath(subdir):
    path = join(config['user_dir'], config['sample'], config['subdirs'][subdir])
    return path

# Define a new default target that collects targets from the imported module as well as
# the final stitching rules 

rule starfinder:
    input:
        rules.rsf.input, # "all" rule for rsf
        rules.seg.input, # "all" rule for seg
        rules.prestitch.input, # "all" rule for stitch
        directory(getOutpath('source_data')), # result of setup
        f"{getOutpath('final')}/{config['sample']}.h5ad" # final sample h5ad
    default_target: True

rule stitch_reads:
    input: 
        # tile registration txts 
        expand(
            "{outpath}/stitchlinks/{files}",
            outpath = getOutpath('stitch'),
            files = ['TileConfiguration.registered.txt', 'TileConfiguration.txt']
        ),
        # remain reads
        expand(
            "{outpath}/{seg_method}/{pos}/{files}",
            outpath = getOutpath('segmentation'),
            seg_method = config['seg_method'],
            pos = TILES,
            files = ["remain_reads.csv", "cell_center.csv"]
        ),
        # orderlist
        f"{getOutpath('stitch')}/orderlist"

    output:
        # final remain reads csv
        f"{getOutpath('stitch')}/remain_reads.csv",
        # final cell center csv
        f"{getOutpath('stitch')}/cell_center.csv",
        # final image
        f"{getOutpath('stitch')}/cell_reads_profile.png"
    script:
        "modules/stitch/scripts/stitch.py"
        
rule parse_to_h5ad:
    input: 
        # final remain reads csv
        remain_reads = f"{getOutpath('stitch')}/remain_reads.csv",
        # final cell center csv
        cell_center = f"{getOutpath('stitch')}/cell_center.csv"
    output:
        # cell x gene matrix
        gene_by_cell = f"{getOutpath('final')}/gene_by_cell.csv",
        cell_by_gene = f"{getOutpath('final')}/cell_by_gene.csv",
        # obs
        cell_meta = f"{getOutpath('final')}/cell_metadata.csv",
        # vars
        gene_meta = f"{getOutpath('final')}/gene_metadata.csv",
        # h5ad
        adata = f"{getOutpath('final')}/{config['sample']}.h5ad"
    script:
        "modules/stitch/scripts/parse_results.py"
        





    







    




