#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from tifffile import imwrite
from scipy.sparse import coo_matrix

def generate_stitching_order(maf_file_path, output_path):
    
    df = pd.read_xml(maf_file_path)
    df = df.loc[:, ["StageXPos", "StageYPos", "PositionID", "Sections"]]
    n_layers = df['Sections'].max()

    # get the steps 
    temp = df['StageYPos'].shift(-1) - df['StageYPos']
    step = pd.value_counts(temp).index.values[0]
    
    # get the relative positions
    point_relative_min = [df.loc[:,'StageXPos'].min(), df.loc[:,'StageYPos'].min()]
    point_relative_max = [df.loc[:,'StageXPos'].max(), df.loc[:,'StageYPos'].max()]
    coo_shape = np.array(point_relative_max) - np.array(point_relative_min)
    coo_shape = (coo_shape / step + 0.5 + 1).astype(int)
    df['relative_x'] = np.array((df['StageXPos'] - point_relative_min[0])/step + 0.5, dtype=int)
    df['relative_y'] = np.array((df['StageYPos'] - point_relative_min[1])/step + 0.5, dtype=int)
    df = df.loc[:, ['PositionID', 'relative_x', 'relative_y']]
    
    # construct coo matrix
    value = df.loc[:,'PositionID'].values
    row = df.loc[:,'relative_y'].values
    col = df.loc[:,'relative_x'].values
    matrix_shape = [coo_shape[1], coo_shape[0]]

    ordering_matrix = coo_matrix((value, (row,col)), shape=matrix_shape).toarray()

    ## Warning 
    # ordering_matrix = np.rot90(ordering_matrix)
    # ordering_matrix = np.flip(ordering_matrix, axis=1)
    
    ordering_df = pd.DataFrame(ordering_matrix)
    ordering_list = ordering_df.unstack(1)
    output_df = pd.DataFrame(ordering_list)
    output_df = output_df.reset_index()
    output_df.columns = ["col", "row", "id"]
    output_df['grid'] = "tile_" + output_df['row'].astype(str) + "_" + output_df['col'].astype(str)

    fig, ax = plt.subplots(figsize=(max(col), max(row)))
    plt.imshow(ordering_matrix > 0)
    plt.xticks(np.arange(-0.5, max(col)+0.5, 1.0))
    plt.yticks(np.arange(-0.5, max(row)+0.5, 1.0))
    plt.grid()
    
    for i in range(output_df.shape[0]):
        plt.text(output_df.iloc[i, 0]-0.1, output_df.iloc[i, 1]-0.1, output_df.iloc[i, 2], color='blue', size=10)
        plt.text(output_df.iloc[i, 0]-0.25, output_df.iloc[i, 1]+0.1, output_df.iloc[i, 3], color='blue', size=10)

    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path))
    plt.show()

    return output_df, n_layers

sample_annotation_df = pd.read_csv(snakemake.input[0])

base_path = os.path.join(snakemake.config['root_output_path'], snakemake.config['dataset_id'], snakemake.config['output_id'])

current_sample = snakemake.wildcards['sample']
current_output_path = os.path.join(base_path, f'images/fused/{current_sample}')
if not os.path.exists(current_output_path):
    os.mkdir(current_output_path)

current_grid_fname = os.path.join(current_output_path, 'grid.png')
current_df, n_layers = generate_stitching_order(snakemake.input[1], current_grid_fname)
current_df.to_csv(os.path.join(current_output_path, 'grid.csv'))

x, y, z = [snakemake.config['img_col'], snakemake.config['img_row'], n_layers]
blank_3d = np.zeros((z, x, y), dtype=np.uint8)
imwrite(os.path.join(current_output_path, 'blank.tif'), blank_3d)

# current_DAPI_input_path = os.path.join(base_path, 'images/flamingo/DAPI')
current_DAPI_input_path = os.path.join(base_path, 'images/DAPI')
current_DAPI_output_path = os.path.join(current_output_path, 'DAPI')
if not os.path.exists(current_DAPI_output_path):
    os.mkdir(current_DAPI_output_path)

for j in range(0, current_df.shape[0]):
    current_id = current_df.iloc[j, 2]
    current_grid = current_df.iloc[j, 3]
    current_position_id = f"Position{current_id:03}"

    if current_id == 0:
        src_dapi = os.path.join(base_path, 'images/fused', current_sample, 'blank.tif')
        dest_dapi = os.path.join(current_DAPI_output_path, f"tile_{j}.tif")

        shutil.copyfile(src_dapi, dest_dapi)
        # os.symlink(src, dest_dapi)
    else:
        src_dapi = os.path.join(current_DAPI_input_path, f"{current_position_id}.tif")
        dest_dapi = os.path.join(current_DAPI_output_path, f"tile_{j}.tif")

        shutil.copyfile(src_dapi, dest_dapi)
        # os.symlink(src_dapi, dest_dapi)