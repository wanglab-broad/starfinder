#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import plotly.express as px

# define the path
base_path = os.path.join(snakemake.config['root_output_path'], snakemake.config['dataset_id'], snakemake.config['output_id'])
image_path = os.path.join(base_path, 'images')
signal_path = os.path.join(base_path, 'signal')
output_path = os.path.join(base_path, 'output')

# parse the dataset.xml
current_sample = snakemake.wildcards.sample
current_dataset_xml = os.path.join(snakemake.input[0])

setup_df = pd.read_xml(current_dataset_xml, xpath=".//ViewRegistration")
setup_max = setup_df['setup'].max()

setup_list = []
for i in range(setup_max+1):
    setup_list.append([i] * 3)
setup_list = np.array(setup_list).flatten()

transform_df = pd.read_xml(current_dataset_xml, xpath=".//ViewTransform")
transform_df['setup'] = setup_list
transform_df = transform_df.pivot(index='setup', columns='Name', values='affine')
transform_df = transform_df.loc[:, ["Stitching Transform", "Translation to Regular Grid"]]
transform_df['x_st'] = transform_df["Stitching Transform"].str.split(' ').apply(lambda x: x[3]).astype(float)
transform_df['y_st'] = transform_df["Stitching Transform"].str.split(' ').apply(lambda x: x[7]).astype(float)
transform_df['z_st'] = transform_df["Stitching Transform"].str.split(' ').apply(lambda x: x[11]).astype(float)

transform_df['x_trg'] = transform_df["Translation to Regular Grid"].str.split(' ').apply(lambda x: x[3]).astype(float)
transform_df['y_trg'] = transform_df["Translation to Regular Grid"].str.split(' ').apply(lambda x: x[7]).astype(float)
transform_df['z_trg'] = transform_df["Translation to Regular Grid"].str.split(' ').apply(lambda x: x[11]).astype(float)

transform_df['x'] = transform_df['x_st'] + transform_df['x_trg']
transform_df['y'] = transform_df['y_st'] + transform_df['y_trg']
transform_df['z'] = transform_df['z_st'] + transform_df['z_trg']

transform_df['x'] = transform_df['x'].astype(int)
transform_df['y'] = transform_df['y'].astype(int)
transform_df['z'] = transform_df['z'].astype(int)

transform_df['x'] = transform_df['x'] + np.abs(transform_df['x'].min())
transform_df['y'] = transform_df['y'] + np.abs(transform_df['y'].min())
transform_df['z'] = transform_df['z'] + np.abs(transform_df['z'].min())
transform_df['fov_index'] = transform_df.index

transform_df = transform_df.loc[:, ['x', 'y', 'z', 'fov_index']]

# parse the grid.csv
grid_df = pd.read_csv(snakemake.input[1], index_col=0)

# merge the two dataframes  
tile_config_df = pd.concat([transform_df, grid_df], axis=1)

start_x_list = []
start_y_list = []
end_x_list = []
end_y_list = []

for i in range(tile_config_df.shape[0]):
    current_record = tile_config_df.iloc[i]
    current_id = current_record['id']

    if current_id == 0:
        start_x_list.append(0)
        start_y_list.append(0)
        end_x_list.append(0)
        end_y_list.append(0)
    else:
        print(f"Processing tile {current_id}")
        current_row = current_record['row']
        current_col = current_record['col']
        current_x = current_record['x']
        current_y = current_record['y']

        left_tile = f"tile_{current_row}_{current_col - 1}"
        right_tile = f"tile_{current_row}_{current_col + 1}"
        up_tile = f"tile_{current_row - 1}_{current_col}"
        down_tile = f"tile_{current_row + 1}_{current_col}"

        if left_tile in tile_config_df.grid.values and tile_config_df.loc[tile_config_df.grid == left_tile, 'id'].values != 0:
            left_x = tile_config_df.loc[tile_config_df.grid == left_tile, 'x'].values
            me_start_x = int((left_x + snakemake.config['img_col'] - current_x)/2 + 0.5) + current_x
        else:
            me_start_x = current_x

        if up_tile in tile_config_df.grid.values and tile_config_df.loc[tile_config_df.grid == up_tile, 'id'].values != 0:
            left_y = tile_config_df.loc[tile_config_df.grid == up_tile, 'y'].values
            me_start_y = int((left_y + snakemake.config['img_row'] - current_y)/2 + 0.5) + current_y
        else:
            me_start_y = current_y

        if right_tile in tile_config_df.grid.values and tile_config_df.loc[tile_config_df.grid == right_tile, 'id'].values != 0:
            right_x = tile_config_df.loc[tile_config_df.grid == right_tile, 'x'].values
            me_end_x = int((current_x + snakemake.config['img_col'] - right_x)/2 + 0.5 + right_x)
        else:
            me_end_x =  current_x + snakemake.config['img_col']
            
        if down_tile in tile_config_df.grid.values and tile_config_df.loc[tile_config_df.grid == down_tile, 'id'].values != 0:
            right_y = tile_config_df.loc[tile_config_df.grid == down_tile, 'y'].values
            me_end_y = int((current_y + snakemake.config['img_col'] - right_y)/2 + 0.5 + right_y)
        else:
            me_end_y = current_y + snakemake.config['img_col']

        current_start_point = [me_start_x, me_start_y]
        current_end_point = [me_end_x, me_end_y]
        start_x_list.append(me_start_x)
        start_y_list.append(me_start_y)
        end_x_list.append(me_end_x)
        end_y_list.append(me_end_y)

tile_config_df['start_x'] = start_x_list
tile_config_df['start_y'] = start_y_list
tile_config_df['end_x'] = end_x_list    
tile_config_df['end_y'] = end_y_list

tile_config_df['start_x_norm'] = tile_config_df['start_x'] - tile_config_df['x']
tile_config_df['start_y_norm'] = tile_config_df['start_y'] - tile_config_df['y']
tile_config_df['end_x_norm'] = tile_config_df['end_x'] - tile_config_df['x']
tile_config_df['end_y_norm'] = tile_config_df['end_y'] - tile_config_df['y']

tile_config_df.loc[tile_config_df.id == 0, 'start_x_norm'] = 0
tile_config_df.loc[tile_config_df.id == 0, 'start_y_norm'] = 0
tile_config_df.loc[tile_config_df.id == 0, 'end_x_norm'] = 0
tile_config_df.loc[tile_config_df.id == 0, 'end_y_norm'] = 0

tile_config_df.to_csv(snakemake.output[0])

# plot the tile configuration in 3D
fig = px.scatter_3d(tile_config_df, x='x', y='y', z='z', color='id')
fig.update_traces(marker_size = 5)
fig.update_scenes(zaxis_autorange="reversed")
fig.update_scenes(xaxis_autorange="reversed")
# fig.show()
fig.write_html(snakemake.output[1])
