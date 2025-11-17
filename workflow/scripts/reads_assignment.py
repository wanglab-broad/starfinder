#!/usr/bin/env python
# coding: utf-8

import os
import parse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tifffile import imread
from skimage.measure import regionprops
from skimage.segmentation import expand_labels
from skimage.color import label2rgb
from anndata import AnnData

current_fov_id = snakemake.wildcards.fovID
current_id = int(parse.parse(snakemake.config['fov_id_pattern'], current_fov_id)['i'])
print(current_fov_id)

# get sample id for current fov
sample_annotation_df = pd.read_csv(snakemake.input[0])
for current_sample in sample_annotation_df['sample_id']:
    current_start = sample_annotation_df.loc[sample_annotation_df['sample_id'] == current_sample, 'fov_start'].values[0]
    current_end = sample_annotation_df.loc[sample_annotation_df['sample_id'] == current_sample, 'fov_end'].values[0]
    if current_id in range(current_start, current_end+1):
        print(current_sample)
        break   

# set path
base_path = os.path.join(snakemake.config['root_output_path'], snakemake.config['dataset_id'], snakemake.config['output_id'])
image_path = os.path.join(base_path, 'images')
signal_path = os.path.join(base_path, 'signal')
output_path = os.path.join(base_path, 'output')
expr_path = os.path.join(base_path, 'expr')
if not os.path.exists(expr_path):
    os.mkdir(expr_path)

# load tile_config
tile_config_df = pd.read_csv(os.path.join(output_path, f'tile_config_{current_sample}.csv'), index_col=0)
current_record = tile_config_df[tile_config_df['id'] == int(current_id)]
current_record = current_record.iloc[0]

current_expr_path = os.path.join(expr_path, current_fov_id)
if not os.path.exists(current_expr_path):
    os.mkdir(current_expr_path)

# Load images
current_gray_img = imread(snakemake.input[1])
current_label_img = imread(snakemake.input[2])

if len(current_label_img.shape) == 3:
    current_gray_max = np.max((current_gray_img), axis=0)
    # Segmentation dialation
    if snakemake.config['rules']['reads_assignment']['parameters']['expand_labels']:
        for z in range(current_label_img.shape[0]):
            current_label_img[z,:,:] = expand_labels(current_label_img[z,:,:], distance=snakemake.config['rules']['reads_assignment']['parameters']['dilation_distance'])

    middle_layer_index = snakemake.config['img_z'] // 2
    current_middle_layer_label = label2rgb(current_label_img[middle_layer_index, :, :], image=current_gray_img[middle_layer_index, :, :], bg_label=0)
    current_label_max = np.max((current_label_img > 0), axis=0)
    current_seg_coverage = (current_label_img > 0).sum() / (current_gray_img > 40).sum() * 100
else:
    current_gray_max = current_gray_img
    # Segmentation dialation
    if snakemake.config['rules']['reads_assignment']['parameters']['expand_labels']:
        current_label_img = expand_labels(current_label_img, distance=snakemake.config['rules']['reads_assignment']['parameters']['dilation_distance'])

    current_middle_layer_label = label2rgb(current_label_img, image=current_gray_max, bg_label=0)
    current_label_max = current_label_img
    current_seg_coverage = (current_label_img > 0).sum() / (current_gray_img > 40).sum() * 100

# Load signal
reads_df = pd.read_csv(snakemake.input[3])
reads_df['x'] = reads_df['x'] - 1
reads_df['y'] = reads_df['y'] - 1
reads_df['z'] = reads_df['z'] - 1
reads_df['global_x'] = reads_df['x'] + current_record['x']
reads_df['global_y'] = reads_df['y'] + current_record['y']
reads_df['global_z'] = reads_df['z'] + current_record['z']

if reads_df.shape[0] != 0:
    # Reads assignment to cell
    points = reads_df.loc[:, ["x", "y", "z"]].values
    bases = reads_df['gene'].values
    if len(current_label_img.shape) == 3:
        reads_assignment = current_label_img[points[:, 2], points[:, 1], points[:, 0]]
    else:
        reads_assignment = current_label_img[points[:, 1], points[:, 0]]

    reads_df['seg_label'] = reads_assignment
else:
    reads_assignment = np.array([0])

# Create empty cell list
cell_locs = []
total_cells = len(np.unique(current_label_img)) - 1
areas = []
seg_labels = []

# Load genes.csv
genes_df = pd.read_csv(snakemake.input[4], header=None)
genes_df.columns = ['gene', 'barcode']

genes = genes_df['gene'].values
gene_seq_to_index = {}  # map from sequence to index into matrix

for i, k in enumerate(genes):
    gene_seq_to_index[k] = i

if total_cells == 0 or (len(np.unique(reads_assignment)) == 1 and np.unique(reads_assignment)[0] == 0):
    cell_by_gene = np.zeros((0, len(genes)))

    print("No cells found in the current fov, creating empty AnnData object...")
    cell_barcode_names = pd.DataFrame(index=genes)
    if len(current_label_img.shape) == 3:
        current_meta = pd.DataFrame({'sample': current_sample, 'fov_id': current_fov_id, 'volume': 0, 'fov_x': 0, 'fov_y': 0, 'fov_z': 0, 'seg_label': 0,
                                'global_x': 0, 'global_y': 0, 'global_z': 0}, index=[])
    else:
        current_meta = pd.DataFrame({'sample': current_sample, 'fov_id': current_fov_id, 'volume': 0, 'fov_x': 0, 'fov_y': 0, 'seg_label': 0,
                                'global_x': 0, 'global_y': 0}, index=[])
    adata = AnnData(X=cell_by_gene, obs=current_meta, var=cell_barcode_names)
    adata.write(os.path.join(current_expr_path, "raw.h5ad"))
    reads_df.to_csv(os.path.join(current_expr_path, "reads_assignment.csv"), index=False)

else:
    cell_by_gene = np.zeros((total_cells, len(genes)))
    
# Iterate through cells
    print('Iterate cells...')
    for i, region in enumerate(regionprops(current_label_img)):
        areas.append(region.area)
        cell_locs.append(region.centroid)
        seg_labels.append(region.label)

        assigned_reads = bases[np.argwhere(reads_assignment == region.label).flatten()]
        for j in assigned_reads:
            if j in gene_seq_to_index:
                cell_by_gene[i, gene_seq_to_index[j]] += 1
        
    cell_locs = np.array(cell_locs).astype(int)
    if len(current_label_img.shape) == 3:
        global_cell_locs = cell_locs + np.array([current_record['z'], current_record['y'], current_record['x']])
        current_meta = pd.DataFrame({'sample': current_sample, 'fov_id': current_fov_id, 'volume': areas, 'fov_x': cell_locs[:, 2], 'fov_y': cell_locs[:, 1], 'fov_z': cell_locs[:, 0], 'seg_label': seg_labels,
                                    'global_x': global_cell_locs[:, 2], 'global_y': global_cell_locs[:, 1], 'global_z': global_cell_locs[:, 0]})
    else:
        global_cell_locs = cell_locs + np.array([current_record['y'], current_record['x']])
        current_meta = pd.DataFrame({'sample': current_sample, 'fov_id': current_fov_id, 'volume': areas, 'fov_x': cell_locs[:, 1], 'fov_y': cell_locs[:, 0], 'seg_label': seg_labels,
                                    'global_x': global_cell_locs[:, 1], 'global_y': global_cell_locs[:, 0]})
    cell_barcode_names = pd.DataFrame(index=genes)

    # Create scanpy object
    adata = AnnData(X=cell_by_gene, obs=current_meta, var=cell_barcode_names)

    # Filter cells based on location 
    adata = adata[adata.obs['fov_x'].isin(range(current_record['start_x_norm'], current_record['end_x_norm'])), ]
    adata = adata[adata.obs['fov_y'].isin(range(current_record['start_y_norm'], current_record['end_y_norm'])), ]
    adata.obs = adata.obs.reset_index(drop=True)

    seg_label_left = list(adata.obs['seg_label'].unique())
    reads_df_cells = reads_df.loc[reads_df['seg_label'].isin(seg_label_left), :].copy()
    reads_df_background = reads_df.loc[reads_df['seg_label'] == 0, :].copy()
    reads_df_background = reads_df_background.loc[reads_df_background['x'].isin(range(current_record['start_x_norm'], current_record['end_x_norm'])), ]
    reads_df_background = reads_df_background.loc[reads_df_background['y'].isin(range(current_record['start_y_norm'], current_record['end_y_norm'])), ]
    reads_df_filtered = pd.concat([reads_df_cells, reads_df_background])
    
    # Visualize the data
    # cell centers on the segmentation
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(current_label_max, cmap='gray')
    rect = patches.Rectangle((current_record.start_x_norm, current_record.start_y_norm), 
                            current_record.end_x_norm - current_record.start_x_norm, 
                            current_record.end_y_norm - current_record.start_y_norm,
                            linewidth=.5, edgecolor='y', facecolor='none')
    ax.add_patch(rect)

    ax.plot(current_meta.fov_x, current_meta.fov_y, 'k.', markersize=1, )
    ax.plot(adata.obs.fov_x, adata.obs.fov_y, 'r.', markersize=2, ) 
    plt.savefig(os.path.join(current_expr_path, f"cell_centers_on_label.png"))
    plt.clf()
    plt.close()
    # plt.show()

    # cell centers on the dapi
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(current_gray_max, cmap='gray')
    rect = patches.Rectangle((current_record.start_x_norm, current_record.start_y_norm), 
                            current_record.end_x_norm - current_record.start_x_norm, 
                            current_record.end_y_norm - current_record.start_y_norm,
                            linewidth=.5, edgecolor='y', facecolor='none')
    ax.add_patch(rect)

    ax.plot(current_meta.fov_x, current_meta.fov_y, 'k.', markersize=1, )
    ax.plot(adata.obs.fov_x, adata.obs.fov_y, 'r.', markersize=2, ) 
    plt.savefig(os.path.join(current_expr_path, f"cell_centers_on_dapi.png"))
    plt.clf()
    plt.close()
    # plt.show()

    # reads on the segmentation
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(current_label_max, cmap='gray')
    rect = patches.Rectangle((current_record.start_x_norm, current_record.start_y_norm), 
                            current_record.end_x_norm - current_record.start_x_norm, 
                            current_record.end_y_norm - current_record.start_y_norm,
                            linewidth=.5, edgecolor='y', facecolor='none')
    ax.add_patch(rect)

    ax.plot(reads_df.x, reads_df.y, 'k.', markersize=1, )
    ax.plot(reads_df_filtered.x, reads_df_filtered.y, 'r.', markersize=1, )
    plt.savefig(os.path.join(current_expr_path, f"reads_on_label.png"))
    plt.clf()
    plt.close()
    # plt.show()

    # reads on middle layer segmentation
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(current_middle_layer_label)
    rect = patches.Rectangle((current_record.start_x_norm, current_record.start_y_norm), 
                            current_record.end_x_norm - current_record.start_x_norm, 
                            current_record.end_y_norm - current_record.start_y_norm,
                            linewidth=.5, edgecolor='y', facecolor='none')
    ax.add_patch(rect)

    ax.plot(reads_df.loc[reads_df.z == 15, 'x'].values, reads_df.loc[reads_df.z == 15, 'y'].values, 'k.', markersize=1, )
    ax.plot(reads_df_filtered.loc[reads_df_filtered.z == 15, 'x'].values, reads_df_filtered.loc[reads_df_filtered.z == 15, 'y'].values, 'r.', markersize=1, )
    plt.savefig(os.path.join(current_expr_path, f"reads_on_label_middle_layer.png"))
    plt.clf()
    plt.close()
    # plt.show()

    # Output
    # log
    with open(os.path.join(current_expr_path, "log.txt"), 'w') as f:
        msg = "{:.2%} percent [{} out of {}] reads were assigned to {} cells\n".format(cell_by_gene.sum()/len(bases), cell_by_gene.sum(), len(bases), total_cells)
        f.write(msg)
        f.write(f"segmentation coverage: {current_seg_coverage:.2f}%")

    # adata
    adata.write(os.path.join(current_expr_path, "raw.h5ad"))

    # reads assignment
    reads_df_filtered.to_csv(os.path.join(current_expr_path, "reads_assignment.csv"), index=False)