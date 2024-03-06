#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tifffile import imread, imwrite
from skimage.measure import regionprops
from skimage.segmentation import expand_labels
from skimage.color import label2rgb
from anndata import AnnData

current_id = int(sys.argv[1])
current_fov_id = f"Position{current_id:03}"
print(current_fov_id)

sample_dict = {}
for i in range(1, 401 + 1):
    sample_dict[f"Position{i:03}"] = "sample4"

for i in range(402, 706 + 1):
    sample_dict[f"Position{i:03}"] = "sample5"

for i in range(707, 1078 + 1):
    sample_dict[f"Position{i:03}"] = "sample6"

# get sample id
current_sample = sample_dict[current_fov_id]

# set path
base_path = '/stanley/WangLab/Data/Analyzed/2024-02-23-Hongyu-Covid_Spleen_replicate_2/'
image_path = os.path.join(base_path, 'images')
signal_path = os.path.join(base_path, 'signal')
output_path = os.path.join(base_path, 'output')
expr_path = os.path.join(base_path, 'expr')
if not os.path.exists(expr_path):
    os.mkdir(expr_path)

# morph_path = os.path.join(image_path, 'morph')
# if not os.path.exists(morph_path):
#     os.mkdir(morph_path)

# load tile_config
tile_config_df = pd.read_csv(os.path.join(output_path, f'tile_config_{current_sample}.csv'), index_col=0)
current_record = tile_config_df[tile_config_df['id'] == int(current_id)]
current_record = current_record.iloc[0]

# reads assignment
# current_morph_path = os.path.join(morph_path, current_fov_id)
# if not os.path.exists(current_morph_path):
#     os.mkdir(current_morph_path)

current_expr_path = os.path.join(expr_path, current_fov_id)
if not os.path.exists(current_expr_path):
    os.mkdir(current_expr_path)

# Load images
current_gray_img = imread(os.path.join(image_path, "flamingo", 'DAPI', f"{current_fov_id}.tif"))
current_gray_max = np.max((current_gray_img), axis=0)
current_label_img = imread(os.path.join(image_path, "flamingo", 'stardist_segmentation', f"{current_fov_id}.tif"))

# Segmentation dialation
for z in range(current_label_img.shape[0]):
    current_label_img[z,:,:] = expand_labels(current_label_img[z,:,:], distance=5)

current_middle_layer_label = label2rgb(current_label_img[15, :, :], image=current_gray_img[15, :, :], bg_label=0)
current_label_max = np.max((current_label_img > 0), axis=0)
current_seg_coverage = (current_label_img > 0).sum() / (current_gray_img > 40).sum() * 100
print(current_seg_coverage)

# Load signal
reads_df = pd.read_csv(os.path.join(signal_path, f'{current_fov_id}_goodSpots.csv'))
reads_df['x'] = reads_df['x'] - 1
reads_df['y'] = reads_df['y'] - 1
reads_df['z'] = reads_df['z'] - 1
reads_df['global_x'] = reads_df['x'] + current_record['x']
reads_df['global_y'] = reads_df['y'] + current_record['y']
reads_df['global_z'] = reads_df['z'] + current_record['z']

# Load genes.csv
genes_df = pd.read_csv(os.path.join(base_path, "genes.csv"), header=None)
genes_df.columns = ['gene', 'barcode']

# Reads assignment to cell
points = reads_df.loc[:, ["x", "y", "z"]].values
bases = reads_df['gene'].values
reads_assignment = current_label_img[points[:, 2], points[:, 1], points[:, 0]]
reads_df['seg_label'] = reads_assignment

cell_locs = []
total_cells = len(np.unique(current_label_img)) - 1
areas = []
seg_labels = []

genes = genes_df['gene'].values
cell_by_gene = np.zeros((total_cells, len(genes)))
gene_seq_to_index = {}  # map from sequence to index into matrix

for i, k in enumerate(genes):
    gene_seq_to_index[k] = i
    
# Iterate through cells
print('Iterate cells...')
for i, region in enumerate(regionprops(current_label_img, current_gray_img)):
    areas.append(region.area)
    cell_locs.append(region.centroid)
    seg_labels.append(region.label)
    current_cell_label = region.image
    current_cell_image = region.image_intensity
    # imwrite(os.path.join(current_morph_path, f"mask_{region.label}.tif"), current_cell_label)
    # imwrite(os.path.join(current_morph_path, f"img_{region.label}.tif"), current_cell_image)

    assigned_reads = bases[np.argwhere(reads_assignment == region.label).flatten()]
    for j in assigned_reads:
        if j in gene_seq_to_index:
            cell_by_gene[i, gene_seq_to_index[j]] += 1
    
cell_locs = np.array(cell_locs).astype(int)
global_cell_locs = cell_locs + np.array([current_record['z'], current_record['y'], current_record['x']])
current_meta = pd.DataFrame({'sample': current_sample, 'fov_id': current_fov_id, 'volume': areas, 'fov_x': cell_locs[:, 2], 'fov_y': cell_locs[:, 1], 'fov_z': cell_locs[:, 0], 'seg_label': seg_labels,
                            'global_x': global_cell_locs[:, 2], 'global_y': global_cell_locs[:, 1], 'global_z': global_cell_locs[:, 0]})
cell_barcode_names = pd.DataFrame({'gene': genes})
cell_barcode_names.index = cell_barcode_names['gene']

# Create scanpy object
adata = AnnData(X=cell_by_gene, obs=current_meta, var=cell_barcode_names)

# Filter cells based on location 
adata = adata[adata.obs['fov_x'].isin(range(current_record['start_x_norm'], current_record['end_x_norm'])), ]
adata = adata[adata.obs['fov_y'].isin(range(current_record['start_y_norm'], current_record['end_y_norm'])), ]
adata.obs = adata.obs.reset_index(drop=True)

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

ax.plot(reads_df.x, reads_df.y, 'r.', markersize=1, )
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

ax.plot(reads_df.loc[reads_df.z == 15, 'x'].values, reads_df.loc[reads_df.z == 15, 'y'].values, 'r.', markersize=1, )
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
reads_df.to_csv(os.path.join(current_expr_path, "reads_assignment.csv"), index=False)


