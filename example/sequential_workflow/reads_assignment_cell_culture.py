# run reads assignment & segmentation workflow with 2D mouse tissue section 
# user will define:
# config_path

import sys, json
config_path = sys.argv[1]

# test block
config_file = open(config_path)
config = json.load(config_file)
base_path = config['output_path']

# import packages 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.measure import regionprops
from anndata import AnnData
from tifffile import imread
from tqdm import tqdm

# IO path
image_path = os.path.join(base_path, 'images/fused');
signal_path = os.path.join(base_path, 'signal');
output_path = os.path.join(base_path, 'expr')
if not os.path.exists(output_path):
    os.mkdir(output_path)

# Load reads 
reads_df = pd.read_csv(os.path.join(signal_path, 'fused_goodSpots.csv'))
reads_df['x'] = reads_df['x'] - 1
reads_df['y'] = reads_df['y'] - 1
reads_df['z'] = reads_df['z'] - 1

# Load images
overlay = imread(os.path.join(image_path, 'MAX_DAPI.tif'))

structure_dict = {'whole_cell': 'Cell.tif',
                 'nucleus': 'Nuclei.tif',
                 'cytoplasm': 'Cyto.tif'}

total_cells = None
for current_structure in structure_dict.keys():
    print(f"====Processing: {current_structure}====")
    current_output_path = os.path.join(output_path, current_structure)
    if not os.path.exists(current_output_path):
        os.mkdir(current_output_path)

    # Load segmentation 
    current_seg = imread(os.path.join(image_path, structure_dict[current_structure]))
    print(current_seg.shape)

    # Load genes.csv
    genes_df = pd.read_csv(os.path.join(base_path, "documents", "genes.csv"), header=None)
    genes_df.columns = ['gene', 'barcode']

    # Reads assignment to cell
    points = reads_df.loc[:, ["x", "y", "z"]].values
    bases = reads_df['gene'].values
    reads_assignment = current_seg[points[:, 2], points[:, 1], points[:, 0]]
    
    if not total_cells:
        total_cells = len(np.unique(current_seg)) - 1
    print(f"Total number of cells: {total_cells}")

    genes = genes_df['gene'].values
    cell_by_gene = np.zeros((total_cells, len(genes)))
    gene_seq_to_index = {}  # map from sequence to index into matrix

    for i, k in enumerate(genes):
        gene_seq_to_index[k] = i
        
    # Iterate through all regions
    areas = []
    cell_locs = []
    seg_labels = []
    for i, region in enumerate(tqdm(regionprops(current_seg))):
        areas.append(region.area)
        cell_locs.append(region.centroid)
        seg_labels.append(region.label)
        
    # Iterate through cells
    print('Iterate cells...')
    areas_valid = []
    cell_locs_valid = []
    for i in tqdm(range(total_cells)):
        current_label = i+1
        if current_label in seg_labels:
            areas_valid.append(areas[seg_labels.index(current_label)])
            cell_locs_valid.append(cell_locs[seg_labels.index(current_label)])
        else:
            areas_valid.append(0)
            cell_locs_valid.append([0, 0, 0])
            
        assigned_reads = bases[np.argwhere(reads_assignment == current_label).flatten()]
        for j in assigned_reads:
            if j in gene_seq_to_index:
                cell_by_gene[i, gene_seq_to_index[j]] += 1

    cell_locs = np.array(cell_locs).astype(int)
    current_meta = pd.DataFrame({'area': areas, 'x':cell_locs[:, 1], 'y':cell_locs[:, 0], 'seg_label': seg_labels})

    # Output
    with open(os.path.join(current_output_path, "log.txt"), 'w') as f:
        msg = "{:.2%} percent [{} out of {}] reads were assigned to {} cells".format(cell_by_gene.sum()/len(bases), cell_by_gene.sum(), len(bases), total_cells)
        print(msg)
        f.write(msg)
    np.savetxt(os.path.join(current_output_path, "cell_barcode_count.csv"), cell_by_gene.astype(int), delimiter=',', fmt="%d")
    cell_barcode_names = pd.DataFrame({'gene': genes})
    cell_barcode_names.to_csv(os.path.join(current_output_path, "cell_barcode_names.csv"), header=False)
    current_meta.to_csv(os.path.join(current_output_path, "meta.csv"))

    # Visualization
    if current_structure == 'whole_cell':
        
        # Plot cell number 
        figsize = (np.floor(overlay.shape[1] / 1000 * 5), np.floor(overlay.shape[0] / 1000 * 5))
        t_size = 10
        plt.figure(figsize=figsize)
        plt.imshow(overlay)
        for i, region in enumerate(regionprops(current_seg)):
            plt.plot(region.centroid[2], region.centroid[1], '.', color='red', markersize=4)
            plt.text(region.centroid[2], region.centroid[1], str(i), fontsize=t_size, color='red')

        plt.axis('off')
        plt.savefig(os.path.join(image_path,  "cell_ids.png"))
        plt.clf()
        plt.close()

        # # Plot dots on segmentation mask
        # plt.figure(figsize=figsize)
        # plt.imshow(labels > 0, cmap='gray')
        # plt.plot(reads_df['x'], reads_df['y'], '.', color='red', markersize=1)
        # plt.axis('off')
        # points_seg_path = os.path.join(image_path, "spots_on_seg.png")
        # plt.savefig(points_seg_path)
        # plt.clf()
        # plt.close()

        # Get assigned reads 
        assigned_index = np.argwhere(reads_assignment != 0).flatten()
        assigned_bases = bases[assigned_index]
        assigned_points = points[assigned_index, :]

        # Plot gene expression patterns with provided gene list as QC
        selected_genes = ['MALAT1', 'CLU']
        expr_figure_out_path = os.path.join(output_path, 'figures')
        if not os.path.exists(expr_figure_out_path):
            os.mkdir(expr_figure_out_path)
            
        print('Plotting gene expression maps...')
        for i, gene in enumerate(tqdm(selected_genes)):
            
            curr_index = np.argwhere(assigned_bases == gene).flatten()
            curr_points = assigned_points[curr_index, :]
            n_reads = curr_points.shape[0]

            # Plot
            plt.figure(figsize=(10, 10))
            plt.imshow(overlay, cmap='gray')
            plt.plot(curr_points[:, 0], curr_points[:, 1], '.', color='red', markersize=.5)
            plt.axis('off')
            expr_figure_path = os.path.join(expr_figure_out_path, f"{i+1}.{gene}_{n_reads}.png")
            plt.savefig(expr_figure_path)
            plt.clf()
            plt.close()

# Create AnnData object 
# Load whole cell data
primary_dataset = 'whole_cell'
expr_path = os.path.join(output_path, primary_dataset, 'cell_barcode_count.csv')
var_path = os.path.join(output_path, primary_dataset, 'cell_barcode_names.csv')
obs_path = os.path.join(output_path, primary_dataset, 'meta.csv')

# Add expression data to the AnnData object 
expr_x = np.loadtxt(expr_path, delimiter=',')
var = pd.read_csv(var_path, header=None)
var = pd.DataFrame(index=var.iloc[:,1].to_list())
obs = pd.read_csv(obs_path, index_col=0)

adata = AnnData(X=expr_x, var=var, obs=obs)
adata.layers['nucleus'] = np.loadtxt(os.path.join(output_path, 'nucleus', 'cell_barcode_count.csv'), delimiter=',')
adata.layers['cytoplasm'] = np.loadtxt(os.path.join(output_path, 'cytoplasm', 'cell_barcode_count.csv'), delimiter=',')

from datetime import datetime
date = datetime.today().strftime('%Y-%m-%d')
adata.write_h5ad(os.path.join(output_path, f"{date}-cell_culture.h5ad"))