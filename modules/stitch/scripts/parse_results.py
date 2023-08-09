## Parse remain_reads.csv and cell_center.csv into three files (cell_by_gene.csv, cell_metadata.csv, gene_metadata.csv) that can be used to construct a Seurat object or Scanpy anndata
## In this snakemake execution form, additionally parse into h5ad file

import pandas as pd
import numpy as np
import anndata as ad
import copy
from scipy.sparse import csr_matrix

cluster_dir = snakemake.config['user_dir']
tissue = snakemake.config['sample']

# for sample_name in samples:
    # Get I/O paths
# if args.sub:
#     inputpath = os.path.join(cluster_dir, tissue, '04.stitched_results', sample, sample_name)
# else:
#  inputpath = os.path.join(cluster_dir, tissue, '04.stitched_results', sample_name)

# Read in reads assignment results
cell_center = pd.read_csv(snakemake.input.cell_center, index_col=0)
remain_reads = pd.read_csv(snakemake.input.remain_reads, index_col=0, na_filter=False)
cell_center_index = copy.deepcopy(cell_center)
cell_center_index.set_index('cell_barcode',inplace = True,drop = True) 
remain_reads_t = remain_reads.loc[:,['cell_barcode','gene']]
remain_reads_t['value'] = 1

# Create cell-by-gene expression matrix
exp_matrix = pd.pivot_table(remain_reads_t,index='cell_barcode',columns='gene',aggfunc='count',fill_value = 0)
var_raw = [str(s2) for (s1,s2) in exp_matrix.columns.tolist()]
exp_matrix.set_axis(var_raw,axis = 1,inplace=True)

# Create cell obs matrix with cell barcodes as index values
obs = cell_center_index.loc[exp_matrix.index.values,['column','row','z']]
#obs.index = [f"{sample.split('.')[0]}-{barcode}" for barcode in exp_matrix.index.values] # name each cell with its sample name and barcode
#obs.rename_axis('cell_barcode', inplace=True)

# Create gene var matrix as just gene names (no gene metadata)
var = pd.DataFrame(index=var_raw)  ## index as gene name

# if args.sub:
#     print(f"{sample_name}: {len(var)} genes detected")
# else:
print(f"{tissue}: {len(var)} genes detected")

# Save exp_matrix (read counts matrix) as csv file
exp_matrix.to_csv(snakemake.output.cell_by_gene)
exp_matrix.transpose().to_csv(snakemake.output.gene_by_cell)

# Save cell center locations as cell_metadata.csv
obs.to_csv(snakemake.output.cell_meta)

# Save var as gene_metadata.csv
var.to_csv(snakemake.output.gene_meta)

# Make adata
adata = ad.AnnData(
    X = csr_matrix(np.array(exp_matrix)),
    var = var, 
    obs = obs
)
#adata.var.set_index(keys='gene', drop=True, inplace=True)
adata.write_h5ad(snakemake.output.adata)
