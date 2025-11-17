#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd

base_path = os.path.join(snakemake.config['root_output_path'], snakemake.config['dataset_id'], snakemake.config['output_id'])
expr_path = os.path.join(base_path, 'expr')

sample_annotation_df = pd.read_csv(snakemake.input[0])

df_list = []
current_sample = snakemake.wildcards['sample']
current_start = sample_annotation_df.loc[sample_annotation_df['sample_id'] == current_sample, 'fov_start'].values[0]
current_end = sample_annotation_df.loc[sample_annotation_df['sample_id'] == current_sample, 'fov_end'].values[0]

for i in range(current_start, current_end+1):
    
    current_fov = snakemake.config['fov_id_pattern'].format(i=i)
    reads_file = os.path.join(expr_path, current_fov, f'reads_assignment.csv')

    if os.path.exists(reads_file):  
        current_reads_df = pd.read_csv(reads_file)

        if current_reads_df.shape[0] == 0:
            print(f"Empty file: {reads_file}")
        else:   
            current_reads_df['fov_id'] = current_fov
            current_reads_df['sample'] = current_sample

        df_list.append(current_reads_df)
    else:
        print(f"File not found: {reads_file}")

reads_df = pd.concat(df_list)

# add annotation
annot = sample_annotation_df.loc[:, ~sample_annotation_df.columns.isin(['fov_start', 'fov_end'])]
reads_df = reads_df.merge(annot, left_on='sample', right_on='sample_id', how='left')
reads_df['unique_index'] = reads_df['sample'].astype(str) + '_' + reads_df['fov_id'].astype(str) + '_' +  reads_df['seg_label'].astype(str)

reads_df.to_csv(snakemake.output[0])

    

