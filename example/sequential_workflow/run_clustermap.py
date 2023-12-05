# run ClusterMp with 2D mouse tissue section 
# user will define:
# base_path
# pi_path
# number_of_fovs

import sys
base_path = sys.argv[1]
pi_path = sys.argv[2]
number_of_fovs = sys.argv[3]

# test block
base_path = '/home/unix/jiahao/wanglab/Data/Analyzed/2023-10-01-Jiahao-Test/mAD_64/'
pi_path = '/home/unix/jiahao/wanglab/Data/Processed/2023-10-01-Jiahao-Test/mAD_64/round1'
number_of_fovs = 56

# Import packages 
import os
import timeit
import math
import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ClusterMap.clustermap import *
from skimage.transform import rotate

# IO path 
start = timeit.default_timer()
signal_path = os.path.join(base_path, 'signal')
output_path = os.path.join(base_path, 'expr/clustermap')
if not os.path.exists(output_path):
    os.mkdir(output_path)
    
# Set ClusterMap parameters
xy_radius = 50 
z_radius = 10
pct_filter = 0.1
dapi_grid_interval = 5
min_spot_per_cell = 5
cell_num_threshold = 0.02
window_size = 512

# Iterate through each tile
fovs = [f"tile_{f}" for f in range(1, number_of_fovs+1)]
for current_fov in fovs:

    # Read dapi: col, row, z
    pi_file = [f for f in os.listdir(os.path.join(pi_path, current_fov)) if 'ch04' in f]
    pi = tifffile.imread(os.path.join(pi_path, current_fov, pi_file[0]))
    pi = np.transpose(pi, (1,2,0))
    pi = rotate(pi, angle=-90)
    
    # Read spots
    spots = pd.read_csv(os.path.join(signal_path, f'{current_fov}_goodSpots.csv'))
    spots.columns = ['spot_location_1', 'spot_location_2', 'spot_location_3', 'gene']

    # Add gene code
    genes_df = pd.read_csv(os.path.join(base_path, 'genes.csv'), header=None)
    genes_df.columns = ['gene', 'barcode']
    genes_df.gene = genes_df.gene.astype('category')
    gene_order = genes_df.gene.cat.categories
    spots['gene_name'] = spots.gene.copy()
    spots.gene = spots.gene.astype('category')
    spots.gene = spots.gene.cat.reorder_categories((gene_order))
    spots['gene'] = spots.gene.cat.codes + 1
    spots.gene = spots.gene.astype(int)

    # Run ClusterMap
    number_of_genes = len(gene_order)
    gene_list = np.arange(1, number_of_genes+1)
    num_dims = len(pi.shape)
    model = ClusterMap(spots=spots, dapi=pi, gene_list=gene_list, num_dims=num_dims, gauss_blur=True, sigma=3,
                xy_radius=xy_radius, z_radius=z_radius, fast_preprocess=True)

    model.preprocess(dapi_grid_interval=dapi_grid_interval, pct_filter=pct_filter)

    # Since tiles are large, split data into smaller tiles and re-stitch after segmentation
    label_img = get_img(pi, model.spots, window_size=window_size, margin=math.ceil(window_size*0.1))
    subtiles = split(pi, label_img, model.spots, window_size=window_size, margin=math.ceil(window_size*0.1))

    # Process each tile and stitch together
    cell_info = {'cellid': [], 'cell_center': []}
    model.spots['clustermap'] = -1

    for tile_split in range(subtiles.shape[0]):
        print(f'tile: {tile_split}')
        spots_tile = subtiles.loc[tile_split, 'spots']
        dapi_tile = subtiles.loc[tile_split, 'img']

        # instantiate model
        model_tile = ClusterMap(spots=spots_tile, dapi=dapi_tile, gene_list=gene_list, num_dims=num_dims,
                        xy_radius=xy_radius, z_radius=z_radius, fast_preprocess=False)

        if model_tile.spots.shape[0] < min_spot_per_cell:
            print(f"Less than {min_spot_per_cell} spots found in subtile. Skipping and continuing...")
            continue
        else:
            if sum(model_tile.spots['is_noise'] == 0) < min_spot_per_cell:
                print(f"Less than {min_spot_per_cell} non-noisy spots found in subtile. Skipping and continuing...")
                continue
            else:
                # Segmentation
                model_tile.min_spot_per_cell = min_spot_per_cell
                model_tile.segmentation(cell_num_threshold=cell_num_threshold, dapi_grid_interval=dapi_grid_interval, add_dapi=True, use_genedis=True)
                # Check if segmentation successful 
                if 'clustermap' not in model_tile.spots.columns:
                    continue
                else:
                    # Check unique cell centers in tile
                    if len(np.unique(model_tile.spots['clustermap'])) == 0: 
                        print("No unique cell centers found in the cell. Skipping stitching...")
                        continue
                    elif len(np.unique(model_tile.spots['clustermap'])) == 1 and np.unique(model_tile.spots['clustermap']) == [-1]:
                        print("All cell centers found were noise. Skipping stitching...")
                        continue
                    else:
                        # Stitch tiles together
                        cell_info=model.stitch(model_tile, subtiles, tile_split)

    print("Finished analyzing all subtiles")

    # If tile is completely noise, throw subtiles
    if not hasattr(model, 'all_points_cellid'):
        print("No denoised cell centers found in this tile.")
        sys.exit()

    # Save reads assignment as csv
    model.save_segmentation(os.path.join(output_path, f'{current_fov}_spots.csv'))
    cell_center_df = pd.DataFrame({'cell_barcode': model.cellid_unique.astype(int),
                                   'column': model.cellcenter_unique[:,1],
                                   'row': model.cellcenter_unique[:,0],
                                   'z_axis': model.cellcenter_unique[:,2]})
    cell_center_df.to_csv(os.path.join(output_path, f'{current_fov}_cell_center.csv'))

    # Save figs for QC
    figure_path = os.path.join(output_path, 'figures')
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)

    # Reads preprocessing
    plt.figure(figsize=(10,10))
    plt.imshow(pi.max(axis=2), cmap=plt.cm.gray)
    plt.scatter(model.spots.loc[model.spots['is_noise'] == 0, 'spot_location_1'], model.spots.loc[model.spots['is_noise'] == 0, 'spot_location_2'], s=0.5, c='g', alpha=.5)
    plt.scatter(model.spots.loc[model.spots['is_noise'] == -1, 'spot_location_1'], model.spots.loc[model.spots['is_noise'] == -1, 'spot_location_2'], s=0.5, c='r', alpha=.5)
    plt.savefig(os.path.join(figure_path, f'{current_fov}_pp.png'))
    plt.clf()
    plt.close()

    # Cell segmentation 
    cell_ids = model.spots['clustermap']
    cells_unique = np.unique(cell_ids)
    spots_repr = np.array(model.spots[['spot_location_2', 'spot_location_1']])[cell_ids>=0]
    cell_ids = cell_ids[cell_ids>=0]                
    cmap = np.random.rand(int(max(cell_ids)+1), 3)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(np.zeros(pi.max(axis=2).shape), cmap='Greys_r')
    ax.scatter(spots_repr[:,1],spots_repr[:,0], c=cmap[[int(x) for x in cell_ids]], s=1, alpha=.5)
    ax.scatter(model.cellcenter_unique[:,1], model.cellcenter_unique[:,0], c='r', s=3)
    plt.axis('off')
    plt.savefig(os.path.join(figure_path, f'{current_fov}_cell_seg.png'))
    plt.clf()
    plt.close()

    # Cell segmentation with image
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(pi.max(axis=2), cmap='Greys_r')
    ax.scatter(spots_repr[:,1],spots_repr[:,0], c=cmap[[int(x) for x in cell_ids]], s=1, alpha=.5)
    ax.scatter(model.cellcenter_unique[:,1], model.cellcenter_unique[:,0], c='r', s=3)
    plt.axis('off')
    plt.savefig(os.path.join(figure_path, f'{current_fov}_cell_seg_nuclei.png'))
    plt.clf()
    plt.close()

    stop = timeit.default_timer()
    computation_time = round((stop - start) / 60, 2)

    # Save log as csv
    log_dict = {'number_of_spots': spots.shape[0], 
            'number_of_spots_after_pp': model.spots.loc[model.spots['is_noise'] == 0, :].shape[0],
            'number_of_cells': model.cellcenter_unique.shape[0],
            'computation_time': computation_time}
    log = pd.DataFrame(log_dict, index=[current_fov])
    log.to_csv(os.path.join(output_path,  f'{current_fov}_log.csv'))