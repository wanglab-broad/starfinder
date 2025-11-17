#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tifffile import imread

current_fov_id = snakemake.wildcards.fovID

# set path
base_path = os.path.join(snakemake.config['root_output_path'], snakemake.config['dataset_id'], snakemake.config['output_id'])
image_path = os.path.join(base_path, 'images')
signal_path = os.path.join(base_path, 'signal')
output_path = os.path.join(base_path, 'output')

# load subtile_coords.csv
subtile_coords_df = pd.read_csv(snakemake.input[0])

# stitching reads of subtiles
reads_df = pd.DataFrame()
for i in range(subtile_coords_df.shape[0]):
    # print(i)
    current_subtile_index = subtile_coords_df.loc[i, 't']
    current_scoords_x = subtile_coords_df.loc[i, 'scoords_x']
    current_scoords_y = subtile_coords_df.loc[i, 'scoords_y']
    # current_upperleft_x = subtile_coords_df.loc[i, 'upperleft_x']
    # current_upperleft_y = subtile_coords_df.loc[i, 'upperleft_y']

    current_reads_df = pd.read_csv(os.path.join(output_path, 'subtile', current_fov_id, f'subtile_goodSpots_{current_subtile_index}.csv'))
    current_reads_df['x'] = current_reads_df['x'] + current_scoords_x - 1
    current_reads_df['y'] = current_reads_df['y'] + current_scoords_y - 1
    
    if reads_df.empty:
        reads_df = current_reads_df
    else:
        reads_df = reads_df.loc[(reads_df['x'] <= current_scoords_x) | (reads_df['y'] <= current_scoords_y), :]
        current_reads_df = current_reads_df.loc[(current_reads_df['x'] > current_scoords_x) & (current_reads_df['y'] > current_scoords_y), :]
        reads_df = pd.concat([reads_df, current_reads_df], axis=0)

# save reads_df
reads_df.to_csv(snakemake.output[0], index=False)

# visualize reads on ref_merged
ref_merged_img_path = os.path.join(image_path, 'ref_merged', f'{current_fov_id}.tif')
ref_merged_img = imread(ref_merged_img_path)  
if ref_merged_img.ndim == 3:
    ref_merged_img = np.max(ref_merged_img, axis=0)

plt.figure(figsize=(15,15))
plt.imshow(ref_merged_img, cmap='gray')
plt.plot(reads_df.x, reads_df.y, '.', color='red', markersize=1)
plt.axis('off')
plt.savefig(snakemake.output[1])
plt.close() 
