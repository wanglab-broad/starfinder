#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import scanpy as sc
import anndata as ad

base_path = os.path.join(snakemake.config['root_output_path'], snakemake.config['dataset_id'], snakemake.config['output_id'])
expr_path = os.path.join(base_path, 'expr')

sample_annotation_df = pd.read_csv(snakemake.input[0])

adata_list = []
current_sample = snakemake.wildcards['sample']
current_start = sample_annotation_df.loc[sample_annotation_df['sample_id'] == current_sample, 'fov_start'].values[0]
current_end = sample_annotation_df.loc[sample_annotation_df['sample_id'] == current_sample, 'fov_end'].values[0]

for i in range(current_start, current_end+1):
    
    current_fov = snakemake.config['fov_id_pattern'].format(i=i)
    adata_file = os.path.join(expr_path, current_fov, f'raw.h5ad')

    if os.path.exists(adata_file):  
        adata = sc.read_h5ad(adata_file)

        # # combine Ms4a1 and Ms4a1-1
        adata.var_names_make_unique()
        # adata[:, 'Ms4a1'].X = adata[:, 'Ms4a1'].X + adata[:, 'Ms4a1-1'].X
        # adata = adata[:, adata.var_names != 'Ms4a1-1']

        adata_list.append(adata)
    else:
        print(f"File not found: {adata_file}")

cdata = ad.concat(adata_list, index_unique="_")

# add annotation
annot = sample_annotation_df.loc[:, ~sample_annotation_df.columns.isin(['fov_start', 'fov_end'])]
cdata.obs = cdata.obs.merge(annot, left_on='sample', right_on='sample_id', how='left')
cdata.obs['unique_index'] = cdata.obs['sample'].astype(str) + '_' + cdata.obs['fov_id'].astype(str) + '_' +  cdata.obs['seg_label'].astype(str)
cdata.obs.index = cdata.obs['unique_index']

cdata.write_h5ad(snakemake.output[0])

    

