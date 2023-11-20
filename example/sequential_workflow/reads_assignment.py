# run registration and spot finding workflow with 2D mouse tissue section 
# user will define:

# import packages 
import os
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu, gaussian
from skimage.measure import regionprops
from skimage.morphology import binary_dilation, disk
from skimage.segmentation import watershed
from anndata import AnnData
from tifffile import imread, imwrite
from tqdm import tqdm


# IO path 
base_path = '/home/unix/jiahao/wanglab/Data/Analyzed/2023-10-01-Jiahao-Test/mAD_64/'
output_path = os.path.join(base_path, 'expr')
if not os.path.exists(output_path):
    os.mkdir(output_path)
    
image_path = os.path.join(base_path, 'images/stitching')
signal_path = os.path.join(base_path, 'signal')

# Load reads 
reads_df = pd.read_csv(os.path.join(signal_path, 'merged_goodSpots_max3d.csv'))
reads_df['x'] = reads_df['x'] - 1
reads_df['y'] = reads_df['y'] - 1
reads_df['z'] = reads_df['z'] - 1

# Load images
overlay = imread(os.path.join(image_path, 'overlay.tif'))
dapi = imread(os.path.join(image_path, 'PI_label.tif'))

# Get cell locations 
centroids = []
areas = []

for i, region in enumerate(tqdm(regionprops(dapi))):
    centroids.append(region.centroid)
    areas.append(region.area)

centroids = np.array(centroids)
areas = np.array(areas)

# Segmentation
print("Gaussian & Thresholding")
overlay_blurred = gaussian(overlay, 5)
threhold = threshold_otsu(overlay_blurred)
overlay_bw = overlay_blurred > threhold
overlay_bw = binary_dilation(overlay_bw, footprint=disk(10))

print("Assigning markers")
centroids = centroids.astype(int)
markers = np.zeros(overlay_bw.shape, dtype=np.uint8)
for i in range(centroids.shape[0]):
    x, y = centroids[i, :]
    if x < overlay_bw.shape[0] and y < overlay_bw.shape[1]:
        markers[x-1, y-1] = 1
markers = ndi.label(markers)[0]

print("Watershed")
labels = watershed(overlay_bw, markers, mask=overlay_bw, watershed_line=True)
print(f"Labeled {len(np.unique(labels)) - 1} cells")
print(f"Saving files to {image_path}")
imwrite(os.path.join(image_path, "labeled_cells.tif"), labels.astype(np.uint16))

figsize = (np.floor(dapi.shape[1] / 1000 * 5), np.floor(dapi.shape[0] / 1000 * 5))
figsize

# Plot cell number 
t_size = 10
plt.figure(figsize=figsize)
plt.imshow(overlay)
for i, region in enumerate(regionprops(labels)):
    plt.plot(region.centroid[1], region.centroid[0], '.', color='red', markersize=4)
    plt.text(region.centroid[1], region.centroid[0], str(i), fontsize=t_size, color='red')

plt.axis('off')
plt.savefig(os.path.join(image_path,  "cell_nums.png"))
plt.clf()
plt.close()

# Plot dots on segmentation mask
plt.figure(figsize=figsize)
plt.imshow(labels > 0, cmap='gray')
plt.plot(reads_df['x'], reads_df['y'], '.', color='red', markersize=1)
plt.axis('off')
points_seg_path = os.path.join(image_path, "points_seg.png")
print(f"Saving points_seg.png")
plt.savefig(points_seg_path)
plt.clf()
plt.close()

# Load genes.csv
genes_df = pd.read_csv(os.path.join(base_path, "genes.csv"), header=None)
genes_df.columns = ['Gene', 'Barcode']

# Reads assignment to cell
points = reads_df.loc[:, ["x", "y"]].values
bases = reads_df['Gene'].values
reads_assignment = labels[points[:, 1], points[:, 0]]
    
cell_locs = []
total_cells = len(np.unique(labels)) - 1
areas = []
seg_labels = []

genes = genes_df['Gene'].values
cell_by_gene = np.zeros((total_cells, len(genes)))
gene_seq_to_index = {}  # map from sequence to index into matrix

for i, k in enumerate(genes):
    gene_seq_to_index[k] = i
    
# Iterate through cells
print('Iterate cells...')
for i, region in enumerate(tqdm(regionprops(labels))):
    # print(region.label)
    areas.append(region.area)
    cell_locs.append(region.centroid)
    seg_labels.append(region.label)
    
    assigned_reads = bases[np.argwhere(reads_assignment == region.label).flatten()]
    for j in assigned_reads:
        if j in gene_seq_to_index:
            cell_by_gene[i, gene_seq_to_index[j]] += 1
    
     
# Keep the good cells 
cell_locs = np.array(cell_locs).astype(int)
current_meta = pd.DataFrame({'area': areas, 'x':cell_locs[:, 1], 'y':cell_locs[:, 0], 'seg_label': seg_labels})

# Output
with open(os.path.join(output_path, "log.txt"), 'w') as f:
    msg = "{:.2%} percent [{} out of {}] reads were assigned to {} cells".format(cell_by_gene.sum()/len(bases), cell_by_gene.sum(), len(bases), total_cells)
    print(msg)
    f.write(msg)
np.savetxt(os.path.join(output_path, "cell_barcode_count.csv"), cell_by_gene.astype(int), delimiter=',', fmt="%d")
cell_barcode_names = pd.DataFrame({'gene': genes})
cell_barcode_names.to_csv(os.path.join(output_path, "cell_barcode_names.csv"), header=False)
current_meta.to_csv(os.path.join(output_path, "meta.csv"))

# Get assigned reads 
assigned_index = np.argwhere(reads_assignment != 0).flatten()
assigned_bases = bases[assigned_index]
assigned_points = points[assigned_index, :]

selected_genes = ['Slc6a11', 'Olig1']
expr_figure_out_path = os.path.join(output_path, 'figures')
if not os.path.exists(expr_figure_out_path):
    os.mkdir(expr_figure_out_path)
    
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
cell_barcode_names.index = cell_barcode_names.gene
cell_barcode_names = cell_barcode_names.drop('gene', axis=1)
adata = AnnData(X=cell_by_gene, var=cell_barcode_names, obs=current_meta)

from datetime import datetime
date = datetime.today().strftime('%Y-%m-%d')
adata.write_h5ad(os.path.join(output_path, f"{date}-mAD_64.h5ad"))