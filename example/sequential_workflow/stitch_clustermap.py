import sys
import os, copy
import numpy as np
import pandas as pd
import tifffile as tif
import matplotlib.pyplot as plt
from scipy.spatial import distance
from tqdm.notebook import tqdm, trange
import anndata as ad
from starmap.sequencing import *
from skimage.measure import regionprops
from skimage.color import label2rgb
from skimage.filters import median, gaussian, threshold_otsu
from skimage.morphology import disk, binary_dilation
import tifffile

# Load genes.csv
def load_genes(base_path):
    genes2seqs = {}
    seqs2genes = {}
    with open(os.path.join(base_path, "genes.csv"), encoding='utf-8-sig') as f:
        for l in f:
            fields = l.rstrip().split(",")
            curr_seg = "".join([str(s+1) for s in encode_SOLID(fields[1][::-1])])
            curr_seg = curr_seg[5:] + curr_seg[:4]
            # print(curr_seg)
            genes2seqs[fields[0]] = curr_seg
            seqs2genes[genes2seqs[fields[0]]] = fields[0]
            
    return genes2seqs, seqs2genes

def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index], closest_index

############################## SET FILE I/O ##############################

### extract the coords information
subdir = 'RIBOmap'
img_c, img_r = [2048, 2048]

data_dir = f'Z:/Data/Processed/2021-11-23-Hu-MouseBrain{subdir}'
stitch_dir = 'Z:/jiahao/Github/RIBOmap/segmentation-stitching/'
clustermap_dir = os.path.join(data_dir, 'output/clustermap/')

orderlist = os.path.join(stitch_dir, subdir, 'orderlist.csv') # list of integers denoting tiles and their positions sequentially
inputpath = os.path.join(stitch_dir, subdir, 'round1_merged') # directory containing tile pointers and TileConfiguration outputs
dapipath = os.path.join(stitch_dir, subdir, 'dapi') # directory dapi
readspath = os.path.join(stitch_dir, 'output', 'segmentation') # segmentation folder containing tile subdirectories and clustermap results within each subdir
outputpath = os.path.join(stitch_dir, subdir, 'results')

if not os.path.exists(outputpath):
    os.mkdir(outputpath)


    ############################## READ IN TILE COORDINATES ##############################

coords_file = os.path.join(inputpath, 'TileConfiguration.registered.txt')
coords_file2 = os.path.join(inputpath, 'TileConfiguration.txt')
print("Reading in coordinates...")

## precise coords
f = open(coords_file)
line = f.readline()
list = []
while line:
    if line.startswith('Position'):
        a = np.array(line.replace('Position','').replace('.tif; ; (',',').replace(', ',',').replace(')\n','').split(','))
        a = (a.astype(float)+0.5).astype(int).tolist()
        list.append(a)
    line = f.readline()
coords_df = np.array(list)
f.close

## relative coords
f = open(coords_file2)
line = f.readline()
list = []
while line:
    if line.startswith('Position'):
        a = np.array(line.replace('Position','').replace('.tif; ; (',',').replace(', ',',').replace(')\n','').split(','))
        a = (a.astype(float)+0.5).astype(int).tolist()
        a[1:3] = (np.divide(a[1:3],int(img_c*0.9+0.5)) + 0.5).astype(int).tolist()
        list.append(a)
    line = f.readline()
coords_df_v2 = np.array(list)
f.close

## order list
order_df = pd.read_csv(orderlist, header = None)
order_df.index = range(1,order_df.shape[0]+1)
order_df.columns = ['tile']

print("Combining read configurations...")
### combine two datasets
coords_df2 = pd.DataFrame(coords_df, columns=['index','column','row'], index = coords_df[:,0])
coords_df2.drop(columns=coords_df2.columns[0], axis=1, inplace=True)
coords_df2_v2 = pd.DataFrame(coords_df_v2, columns=['index','column_count','row_count'], index = coords_df_v2[:,0])
coords_df2_v2.drop(columns=coords_df2_v2.columns[0], axis=1, inplace=True)
coords_df2['column_count'] = coords_df2_v2.loc[coords_df2.index,'column_count']
coords_df2['row_count'] = coords_df2_v2.loc[coords_df2.index,'row_count']

## rearrange the index and add tile information
coords_df3 = coords_df2.loc[range(1,coords_df2.shape[0] + 1),:]
coords_df3['tile'] = order_df['tile']

## save
coords_df3.to_csv(os.path.join(outputpath,'coords.csv'))

print("Tuning coordinates...")
## find origin and tuning the coords
coords_df3_without_blank = coords_df3.loc[coords_df3['tile'] > 0,:]

min_column, min_row = [np.min(coords_df3_without_blank['column']), np.min(coords_df3_without_blank['row'])]
max_column, max_row = [np.max(coords_df3_without_blank['column']), np.max(coords_df3_without_blank['row'])]
shape_column, shape_row = [max_column - min_column + img_c, max_row - min_row + img_r]

coords_df4 = copy.deepcopy(coords_df3)
coords_df4['column'] = coords_df4['column'] - min_column
coords_df4['row'] = coords_df4['row'] - min_row

# save
# coords_df4.to_csv(os.path.join(outputpath,'tuned_coords.csv'))

order = 575
tilenum = 394

# Get remain_reads.csv for each tile
dfpath = os.path.join(clustermap_dir, f"Position{tilenum:03}")

remain_reads_t = pd.read_csv(os.path.join(clustermap_dir, f"Position{tilenum:03}", 'spots.csv'))
remain_reads_t['gene'] = remain_reads_t['gene'] - 1
remain_reads_t['spot_location_1'] = remain_reads_t['spot_location_1'] - 1
remain_reads_t['spot_location_2'] = remain_reads_t['spot_location_2'] - 1
remain_reads_t['spot_location_3'] = remain_reads_t['spot_location_3'] - 1

# rotate
temp1 = remain_reads_t['spot_location_1'].values.copy()
temp2 = remain_reads_t['spot_location_2'].values.copy()

remain_reads_t['spot_location_1'] = 2048 - temp2
remain_reads_t['spot_location_2'] = temp1

### read genes.csv
genes2seqs, seqs2genes = load_genes(data_dir)

### read genelist from clustermap
gene_list = pd.read_csv(os.path.join(clustermap_dir, f"Position{tilenum:03}", 'genelist.csv'), header=None)
gene_list.columns = ['barcode']
gene_list['barcode'] = gene_list['barcode'].astype(str)
gene_list['gene'] = gene_list['barcode'].map(seqs2genes)

### map genes 
nums2genes = dict(zip(gene_list.index.to_list(), gene_list.gene.to_list()))
remain_reads_t['gene'] = remain_reads_t['gene'].map(nums2genes)

### remove noise spots
remain_reads_t = remain_reads_t.loc[remain_reads_t['clustermap'] != -1, :]

# Label with coordinates/tilenum and barcode
# remain_reads_t['gridc_gridr_tilenum'] = str(t_grid_c)+","+str(t_grid_r)+","+str(tilenum)

## add cell barcode
remain_reads_t['cell_barcode'] =  remain_reads_t['clustermap'].values
remain_reads_t = remain_reads_t.drop(columns=['clustermap'])

# get cell center info
label_img = np.zeros([img_c, img_r, 35], dtype=np.uint16)
label_img[remain_reads_t['spot_location_2'].values, remain_reads_t['spot_location_1'].values, remain_reads_t['spot_location_3'].values] = remain_reads_t['cell_barcode'].values + 1

cell_barcode = []
region_centroid = []
for i, region in enumerate(tqdm(regionprops(label_img))):
    cell_barcode.append(region.label - 1)
    region_centroid.append(region.centroid)

region_centroid = np.array(region_centroid)
cell_center_t = pd.DataFrame({'cell_barcode': cell_barcode, 'column': region_centroid[:, 1], 'row': region_centroid[:, 0], 'z': region_centroid[:, 2]})
cell_center_t = cell_center_t.astype(int)
print(cell_center_t['cell_barcode'].nunique())

## filter cell center based on the dapi mask
current_dapi_path = os.path.join(dapipath, f"Position{order:03}.tif")
current_dapi = tif.imread(current_dapi_path)
# current_dapi = median(current_dapi, disk(5))
current_dapi = gaussian(current_dapi, sigma=5)
current_threshold = threshold_otsu(current_dapi)
current_dapi_mask = current_dapi > current_threshold
current_dapi_mask = binary_dilation(current_dapi_mask, disk(30))

fig, ax = plt.subplots(figsize=(10,10))
plt.imshow(current_dapi_mask)
plt.scatter(cell_center_t.loc[:,'column'], cell_center_t.loc[:,'row'], s=3, c='red', alpha = 0.7)
plt.show()

current_good_cells = current_dapi_mask[cell_center_t.loc[:,'row'], cell_center_t.loc[:,'column']]
cell_center_t = cell_center_t.loc[current_good_cells, :]
remain_reads_t = remain_reads_t.loc[remain_reads_t['cell_barcode'].isin(cell_center_t['cell_barcode']), :]

# remap cell barcode
cell_barcode_dict = {}
for i, k in enumerate(cell_center_t['cell_barcode'].unique()):
    cell_barcode_dict[k] = i
cell_center_t['cell_barcode'] = cell_center_t['cell_barcode'].map(cell_barcode_dict)
remain_reads_t['cell_barcode'] = remain_reads_t['cell_barcode'].map(cell_barcode_dict)

print(cell_center_t['cell_barcode'].nunique())

fig, ax = plt.subplots(figsize=(10,10))
plt.imshow(current_dapi_mask)
plt.scatter(cell_center_t.loc[:,'column'], cell_center_t.loc[:,'row'], s=3, c='red', alpha = 0.7)
plt.show()

############################## STITCH TOGETHER TILE COORDINATES ##############################

print("Adjusting cell center and read coordinates by tile position...")
alignment_thresh = 0.1
cell_barcode_min = 0
middle_edge = 0

# generate empty dataframe
remain_reads = pd.DataFrame({'spot_location_1':[],'spot_location_2':[],'spot_location_3':[],'gene':[],'cell_barcode':[],'gridc_gridr_tilenum':[]})
cell_center = pd.DataFrame({'cell_barcode':[], 'column':[], 'row':[], 'gridc_gridr_tilenum':[]})

# get grid 
grid_c, grid_r = (np.max(coords_df4.loc[:,['column_count','row_count']]) + 1).tolist()
print(grid_c, grid_r)

for t_grid_c in trange(0, grid_c):
# for t_grid_c in trange(0, 2): # test

    median_col_coord = np.median(coords_df4[(coords_df4.column_count == t_grid_c) & (coords_df4.tile != 0)]['column'])
    
    for t_grid_r in trange(0, grid_r):
    # for t_grid_r in trange(23, 25): # test
    
        median_row_coord = np.median(coords_df4[(coords_df4.row_count == t_grid_r) & (coords_df4.tile != 0)]['row'])
        print('\t[t_grid_c, t_grid_r]: ',str(t_grid_c),' ',str(t_grid_r))
        order = t_grid_c * grid_r + t_grid_r + 1

        tilenum = coords_df4['tile'][order]

        # skip the tile if the tilenum == 0, blank tile
        if tilenum == 0:
            print("\tBlank tile")
            continue

        # get upper left coordinates
        upper_left = coords_df4.loc[order, ['column', 'row']]
        upper_left_new = copy.deepcopy(upper_left)

        # check that tile is appproximately aligned where expected -- otherwise throw out
        if upper_left[0] >= (1+alignment_thresh)*median_col_coord and upper_left[0] <= (1-alignment_thresh)*median_col_coord:
            if upper_left[1] >= (1+alignment_thresh)*median_row_coord and upper_left[1] <= (1-alignment_thresh)*median_col_coord:
                print("f\tTile is aligned too far away from its expected position.")
                print("f\tTile coord: [{upper_left[0]}, {upper_left[1]}]. Median coord: [{median_col_coord}, {median_row_coord}]")
                continue

        # judgment
        t_grid_c_previous = t_grid_c - 1
        t_grid_r_previous = t_grid_r - 1

        # condition1: if left one is not blank, then calculate middle overlap
        if t_grid_c_previous >= 0: # if a left tile exists
            order_t = t_grid_c_previous * grid_r + t_grid_r + 1 # order of left tile
            if coords_df4.loc[order_t,'tile'] != 0: # if it's not blank, calculate new middle edge. Otherwise, use old middle edge
                middle_edge = np.int((coords_df4.loc[order_t,'column'] + img_c - upper_left[0])/2 + 0.5) # calculate middle overlap

                if middle_edge >= 0: 
                    upper_left_new[0] = middle_edge + upper_left[0]

        # condition2: if upper one is empty or blank
        if t_grid_r_previous >= 0:
            order_t = t_grid_c * grid_r + t_grid_r_previous + 1
            if coords_df4.loc[order_t,'tile'] != 0:
                middle_edge = np.int((coords_df4.loc[order_t,'row'] + img_c - upper_left[1])/2 + 0.5)

                if middle_edge >= 0: 
                    upper_left_new[1] = middle_edge + upper_left[1]

        ### stitch
        # Get remain_reads.csv for each tile
        dfpath = os.path.join(clustermap_dir, f"Position{tilenum:03}")
        if not os.path.exists(os.path.join(dfpath, 'spots.csv')):
            print('\tNo reads file for this tile')
            continue

        remain_reads_t = pd.read_csv(os.path.join(clustermap_dir, f"Position{tilenum:03}", 'spots.csv'))
        remain_reads_t['gene'] = remain_reads_t['gene'] - 1
        remain_reads_t['spot_location_1'] = remain_reads_t['spot_location_1'] - 1
        remain_reads_t['spot_location_2'] = remain_reads_t['spot_location_2'] - 1
        remain_reads_t['spot_location_3'] = remain_reads_t['spot_location_3'] - 1

        # rotate
        temp1 = remain_reads_t['spot_location_1'].values.copy()
        temp2 = remain_reads_t['spot_location_2'].values.copy()

        remain_reads_t['spot_location_1'] = 2048 - temp2
        remain_reads_t['spot_location_2'] = temp1

        ### read genes.csv
        genes2seqs, seqs2genes = load_genes(data_dir)

        ### read genelist from clustermap
        gene_list = pd.read_csv(os.path.join(clustermap_dir, f"Position{tilenum:03}", 'genelist.csv'), header=None)
        gene_list.columns = ['barcode']
        gene_list['barcode'] = gene_list['barcode'].astype(str)
        gene_list['gene'] = gene_list['barcode'].map(seqs2genes)

        ### map genes 
        nums2genes = dict(zip(gene_list.index.to_list(), gene_list.gene.to_list()))
        remain_reads_t['gene'] = remain_reads_t['gene'].map(nums2genes)

        ### remove noise spots
        remain_reads_t = remain_reads_t.loc[remain_reads_t['clustermap'] != -1, :]

        # skip current tile if no reads left
        if remain_reads_t.shape[0] == 0:
            print("\tNo reads found in remain_reads.csv for this tile")
            continue

        # Label with coordinates/tilenum and barcode
        remain_reads_t['gridc_gridr_tilenum'] = str(t_grid_c)+","+str(t_grid_r)+","+str(tilenum)

        # remap cell barcode
        remain_reads_t['cell_barcode'] =  remain_reads_t['clustermap'].values
        remain_reads_t = remain_reads_t.drop(columns=['clustermap'])

        # get cell center info
        label_img = np.zeros([img_c, img_r, 35], dtype=np.uint16)
        label_img[remain_reads_t['spot_location_2'].values, remain_reads_t['spot_location_1'].values, remain_reads_t['spot_location_3'].values] = remain_reads_t['cell_barcode'].values + 1

        cell_barcode = []
        region_centroid = []
        for i, region in enumerate(tqdm(regionprops(label_img))):
            cell_barcode.append(region.label - 1)
            region_centroid.append(region.centroid)

        region_centroid = np.array(region_centroid)
        cell_center_t = pd.DataFrame({'cell_barcode': cell_barcode, 'column': region_centroid[:, 1], 'row': region_centroid[:, 0], 'z': region_centroid[:, 2]})
        cell_center_t = cell_center_t.astype(int)

        ## filter cell center based on the dapi mask
        current_dapi_path = os.path.join(dapipath, f"Position{order:03}.tif")
        current_dapi = tif.imread(current_dapi_path)
        # current_dapi = median(current_dapi, disk(5))
        current_dapi = gaussian(current_dapi, sigma=5)
        current_threshold = threshold_otsu(current_dapi)
        current_dapi_mask = current_dapi > current_threshold
        current_dapi_mask = binary_dilation(current_dapi_mask, disk(30))

        current_good_cells = current_dapi_mask[cell_center_t.loc[:,'row'], cell_center_t.loc[:,'column']]
        cell_center_t = cell_center_t.loc[current_good_cells, :]
        remain_reads_t = remain_reads_t.loc[remain_reads_t['cell_barcode'].isin(cell_center_t['cell_barcode']), :]

        # remap cell barcode
        cell_barcode_dict = {}
        for i, k in enumerate(cell_center_t['cell_barcode'].unique()):
            cell_barcode_dict[k] = i
        cell_center_t['cell_barcode'] = cell_center_t['cell_barcode'].map(cell_barcode_dict)
        remain_reads_t['cell_barcode'] = remain_reads_t['cell_barcode'].map(cell_barcode_dict)

        # change cell barcode
        remain_reads_t['cell_barcode'] = remain_reads_t['cell_barcode'] + cell_barcode_min
        cell_center_t['cell_barcode'] = cell_center_t['cell_barcode'] + cell_barcode_min

        # modify cell center
        cell_center_t['gridc_gridr_tilenum'] = str(t_grid_c)+","+str(t_grid_r)+","+str(tilenum)

        ### stitch
        # Adjust spot_location_1 (column)
        remain_reads_t['spot_location_1'] = remain_reads_t['spot_location_1'] + upper_left[0]
        cell_center_t['column'] = cell_center_t['column']  + upper_left[0]

        # Adjust spot_location_2 (row)
        remain_reads_t['spot_location_2'] = remain_reads_t['spot_location_2'] + upper_left[1]
        cell_center_t['row'] = cell_center_t['row']  + upper_left[1]

        # print(f"\ttile: Position{tilenum:03}")
        # print(f"\tInitial remain_reads: {len(remain_reads_t)}")
        # print(f"\tInitial cell centers: {len(cell_center_t)}")

        ## keep the cells within upper_left_new for `remain_reads`, `cell_center`
        cell_center = cell_center.loc[(cell_center['column'] <= upper_left_new[0])|(cell_center['row'] <= upper_left_new[1]) | (cell_center['row'] >= upper_left_new[1]+img_r),:] 
        remain_reads = remain_reads.loc[remain_reads['cell_barcode'].isin(cell_center['cell_barcode']),:]

        ## keep the cells beyond upper_left_new for `remain_reads_t`, `cell_center_t`
        cell_center_t = cell_center_t.loc[(cell_center_t['column'] > upper_left_new[0])&(cell_center_t['row'] > upper_left_new[1]),:]
        remain_reads_t = remain_reads_t.loc[remain_reads_t['cell_barcode'].isin(cell_center_t['cell_barcode']),:]
        # print(f"\tReads beyond upper_left_new: {len(remain_reads_t)}")
        # print(f"\tCell centers beyond upper_left_new: {len(cell_center_t)}")

        ## append
        cell_center = pd.concat((cell_center, cell_center_t), axis=0)
        print(f"\tNew total number of cell centers: {len(cell_center)}")
        remain_reads = pd.concat((remain_reads, remain_reads_t), axis=0)
        print(f"\tNew total number of reads: {len(remain_reads)}")

        # Update minimum cell barcode
        if cell_center_t.shape[0] > 0:
            cell_barcode_min = np.max(cell_center_t['cell_barcode']) + 1

remain_reads.to_csv(f'Z:/Data/Analyzed/2021-11-23-Hu-MouseBrain/{subdir}/remain_reads.csv')
cell_center.to_csv(f'Z:/Data/Analyzed/2021-11-23-Hu-MouseBrain/{subdir}/cell_center.csv')

### polish after stitch
print("Polishing stitch...")

# filter the repeated reads
remain_reads = remain_reads.drop(columns=['is_noise'])
remain_reads = remain_reads.drop_duplicates(subset = None, keep = 'first')

# reset index
cell_center.reset_index(inplace = True, drop = True)
remain_reads.reset_index(inplace = True, drop = True)

# transfer float to integer
remain_reads['spot_location_1'] = remain_reads['spot_location_1'].astype(int)
remain_reads['spot_location_2'] = remain_reads['spot_location_2'].astype(int)
remain_reads['spot_location_3'] = remain_reads['spot_location_3'].astype(int)
remain_reads['cell_barcode'] = remain_reads['cell_barcode'].astype(int)

cell_center['column'] = cell_center['column'].astype(int)
cell_center['row'] = cell_center['row'].astype(int)
cell_center['z'] = cell_center['z'].astype(int)
cell_center['cell_barcode'] = cell_center['cell_barcode'].astype(int)

### deal with multi-assigned reads
print("Removing reads assigned to multiple cell centers:")
# find duplicated reads
remain_reads_check = remain_reads.loc[:, ['spot_location_1', 'spot_location_2', 'spot_location_3']]
remain_reads_check.columns = ['col','row','z']
remain_reads_check['coors'] = remain_reads_check['col'].apply(str).str.cat(remain_reads_check['row'].apply(str),sep='-').str.cat(remain_reads_check['z'].apply(str),sep='-')
remain_reads_check_counts = remain_reads_check['coors'].value_counts()
repeat_reads = remain_reads_check_counts[remain_reads_check_counts>1]

# assign the duplicated reads to the closest cell
filter_index = []
for i in trange(len(repeat_reads)):

    repeat_reads_index = remain_reads_check.index[remain_reads_check['coors'] == repeat_reads.index[i]]
    vec1 = remain_reads.loc[repeat_reads_index[0], ['spot_location_1', 'spot_location_2', 'spot_location_3']].tolist()
    repeat_reads_cell_index = cell_center['cell_barcode'].isin(remain_reads.loc[repeat_reads_index, 'cell_barcode'])
    repeat_reads_cell = cell_center.loc[repeat_reads_cell_index, :]
    closest_index = closest_node(vec1,np.array(repeat_reads_cell.loc[:, ['column','row','z']]).tolist())[1]
    selected_cell = repeat_reads_cell.iloc[closest_index, 0]
    filter_index.extend(repeat_reads_index[np.logical_not(remain_reads.loc[repeat_reads_index, 'cell_barcode'] == selected_cell)].tolist())

print("read counts before filtering multi-assigned reads: " + str(remain_reads.shape[0]))
remain_reads.drop(index = filter_index, inplace = True)
remain_reads.reset_index(inplace = True, drop = True)
print("read counts after filtering multi-assigned reads: " + str(remain_reads.shape[0]))

# print("Saving cell_center.csv and remain_reads.csv")
# cell_center.to_csv(os.path.join(outputpath, 'cell_center.csv'))
# remain_reads.to_csv(os.path.join(outputpath, 'remain_reads.csv'))

remain_reads.to_csv(f'Z:/Data/Analyzed/2021-11-23-Hu-MouseBrain/{subdir}/remain_reads_polished.csv')
cell_center.to_csv(f'Z:/Data/Analyzed/2021-11-23-Hu-MouseBrain/{subdir}/cell_center_polished.csv')

############################## STITCH TOGETHER TILE COORDINATES ##############################

print("Adjusting cell center and read coordinates by tile position...")
alignment_thresh = 0.1
cell_barcode_min = 0
middle_edge = 0

# generate empty dataframe
remain_reads = pd.DataFrame({'spot_location_1':[],'spot_location_2':[],'spot_location_3':[],'gene':[],'cell_barcode':[],'gridc_gridr_tilenum':[]})
cell_center = pd.DataFrame({'cell_barcode':[], 'column':[], 'row':[], 'gridc_gridr_tilenum':[]})

# get grid 
grid_c, grid_r = (np.max(coords_df4.loc[:,['column_count','row_count']]) + 1).tolist()
print(grid_c, grid_r)

for t_grid_c in trange(0, grid_c):
# for t_grid_c in trange(0, 2): # test

    median_col_coord = np.median(coords_df4[(coords_df4.column_count == t_grid_c) & (coords_df4.tile != 0)]['column'])
    
    for t_grid_r in trange(0, grid_r):
    # for t_grid_r in trange(23, 25): # test
    
        median_row_coord = np.median(coords_df4[(coords_df4.row_count == t_grid_r) & (coords_df4.tile != 0)]['row'])
        print('\t[t_grid_c, t_grid_r]: ',str(t_grid_c),' ',str(t_grid_r))
        order = t_grid_c * grid_r + t_grid_r + 1

        tilenum = coords_df4['tile'][order]

        # skip the tile if the tilenum == 0, blank tile
        if tilenum == 0:
            print("\tBlank tile")
            continue

        # get upper left coordinates
        upper_left = coords_df4.loc[order, ['column', 'row']]
        upper_left_new = copy.deepcopy(upper_left)

        # check that tile is appproximately aligned where expected -- otherwise throw out
        if upper_left[0] >= (1+alignment_thresh)*median_col_coord and upper_left[0] <= (1-alignment_thresh)*median_col_coord:
            if upper_left[1] >= (1+alignment_thresh)*median_row_coord and upper_left[1] <= (1-alignment_thresh)*median_col_coord:
                print("f\tTile is aligned too far away from its expected position.")
                print("f\tTile coord: [{upper_left[0]}, {upper_left[1]}]. Median coord: [{median_col_coord}, {median_row_coord}]")
                continue

        # judgment
        t_grid_c_previous = t_grid_c - 1
        t_grid_r_previous = t_grid_r - 1

        # condition1: if left one is not blank, then calculate middle overlap
        if t_grid_c_previous >= 0: # if a left tile exists
            order_t = t_grid_c_previous * grid_r + t_grid_r + 1 # order of left tile
            if coords_df4.loc[order_t,'tile'] != 0: # if it's not blank, calculate new middle edge. Otherwise, use old middle edge
                middle_edge = np.int((coords_df4.loc[order_t,'column'] + img_c - upper_left[0])/2 + 0.5) # calculate middle overlap

                if middle_edge >= 0: 
                    upper_left_new[0] = middle_edge + upper_left[0]

        # condition2: if upper one is empty or blank
        if t_grid_r_previous >= 0:
            order_t = t_grid_c * grid_r + t_grid_r_previous + 1
            if coords_df4.loc[order_t,'tile'] != 0:
                middle_edge = np.int((coords_df4.loc[order_t,'row'] + img_c - upper_left[1])/2 + 0.5)

                if middle_edge >= 0: 
                    upper_left_new[1] = middle_edge + upper_left[1]

        ### stitch
        # Get remain_reads.csv for each tile
        dfpath = os.path.join(clustermap_dir, f"Position{tilenum:03}")
        if not os.path.exists(os.path.join(dfpath, 'spots.csv')):
            print('\tNo reads file for this tile')
            continue

        remain_reads_t = pd.read_csv(os.path.join(clustermap_dir, f"Position{tilenum:03}", 'spots.csv'))
        remain_reads_t['gene'] = remain_reads_t['gene'] - 1
        remain_reads_t['spot_location_1'] = remain_reads_t['spot_location_1'] - 1
        remain_reads_t['spot_location_2'] = remain_reads_t['spot_location_2'] - 1
        remain_reads_t['spot_location_3'] = remain_reads_t['spot_location_3'] - 1

        # rotate
        temp1 = remain_reads_t['spot_location_1'].values.copy()
        temp2 = remain_reads_t['spot_location_2'].values.copy()

        remain_reads_t['spot_location_1'] = 2048 - temp2
        remain_reads_t['spot_location_2'] = temp1

        ### read genes.csv
        genes2seqs, seqs2genes = load_genes(data_dir)

        ### read genelist from clustermap
        gene_list = pd.read_csv(os.path.join(clustermap_dir, f"Position{tilenum:03}", 'genelist.csv'), header=None)
        gene_list.columns = ['barcode']
        gene_list['barcode'] = gene_list['barcode'].astype(str)
        gene_list['gene'] = gene_list['barcode'].map(seqs2genes)

        ### map genes 
        nums2genes = dict(zip(gene_list.index.to_list(), gene_list.gene.to_list()))
        remain_reads_t['gene'] = remain_reads_t['gene'].map(nums2genes)

        ### get background spots
        remain_reads_t = remain_reads_t.loc[remain_reads_t['clustermap'] == -1, :]

        # skip current tile if no reads left
        if remain_reads_t.shape[0] == 0:
            print("\tNo reads found in remain_reads.csv for this tile")
            continue

        # Label with coordinates/tilenum and barcode
        remain_reads_t['gridc_gridr_tilenum'] = str(t_grid_c)+","+str(t_grid_r)+","+str(tilenum)

        ### stitch
        # Adjust spot_location_1 (column)
        remain_reads_t['spot_location_1'] = remain_reads_t['spot_location_1'] + upper_left[0]

        # Adjust spot_location_2 (row)
        remain_reads_t['spot_location_2'] = remain_reads_t['spot_location_2'] + upper_left[1]

        ## append
        remain_reads = pd.concat((remain_reads, remain_reads_t), axis=0)
        print(f"\tNew total number of reads: {len(remain_reads)}")

remain_reads.to_csv(f'Z:/Data/Analyzed/2021-11-23-Hu-MouseBrain/{subdir}/background_reads.csv')
### polish after stitch

# filter the repeated reads
remain_reads['cell_barcode'] = -1
remain_reads = remain_reads.drop(columns=['is_noise', 'clustermap'])
remain_reads = remain_reads.drop_duplicates(subset=['spot_location_1', 'spot_location_2', 'spot_location_3'], keep='first')

# reset index
remain_reads.reset_index(inplace = True, drop = True)

remain_reads
remain_reads.to_csv(f'Z:/Data/Analyzed/2021-11-23-Hu-MouseBrain/{subdir}/background_reads_polished.csv')


# # Read in reads assignment results
gene_path = os.path.join('Z:/Data/Analyzed/2021-11-23-Hu-MouseBrain/genes.csv')
gene_names = pd.read_csv(gene_path, header=None, names=["Gene Name", "Barcode"])["Gene Name"]

# cell_center = pd.read_csv(os.path.join(inputpath, 'results/cell_center.csv'),index_col=0)
# remain_reads = pd.read_csv(os.path.join(inputpath,'results/remain_reads.csv'),index_col=0, na_filter=False)
cell_center_index = copy.deepcopy(cell_center)
cell_center_index.set_index('cell_barcode', inplace = True, drop = True) 
remain_reads_t = remain_reads.loc[:,['cell_barcode','gene']]
remain_reads_t['value'] = 1

# ## Create cell-by-gene expression matrix
exp_matrix = pd.pivot_table(remain_reads_t, index='cell_barcode', columns='gene', aggfunc='count', fill_value = 0)
var_raw = [str(s2) for (s1,s2) in exp_matrix.columns.tolist()]
exp_matrix.set_axis(var_raw,axis = 1,inplace=True)
obs = cell_center_index.loc[exp_matrix.index.values,['column','row','z']]
obs.reset_index(inplace = True,drop = True) ### obs as cell location
var = pd.DataFrame(index=var_raw)  ## index as gene name

# Store in anndata object
adata = ad.AnnData(X=np.array(exp_matrix),
                var=var,
                obs=obs)

from datetime import datetime
date = datetime.today().strftime('%Y-%m-%d')
adata.write_h5ad(f'Z:/Data/Analyzed/2021-11-23-Hu-MouseBrain/{subdir}/{date}-{subdir}-raw.h5ad')


# plot spots
cell_ids = remain_reads['cell_barcode']
cells_unique = np.unique(cell_ids)
spots_repr = np.array(remain_reads[['spot_location_2', 'spot_location_1']])[cell_ids>=0]
cell_ids = cell_ids[cell_ids>=0]                
cmap = np.random.rand(int(max(cell_ids)+1), 3)
fig, ax = plt.subplots(figsize=(40,40))
ax.scatter(spots_repr[:,1], spots_repr[:,0], c=cmap[[int(x) for x in cell_ids]], s=1, alpha=.5)
ax.scatter(cell_center.loc[:,'column'], cell_center.loc[:,'row'], c='r', s=3)
plt.axis('off')
plt.show()

fig, ax = plt.subplots(figsize=[40,40])
plt.scatter(remain_reads.loc[:,'spot_location_1'], remain_reads.loc[:,'spot_location_2'], s=1, alpha=0.2)
plt.scatter(cell_center.loc[:,'column'], cell_center.loc[:,'row'], s=3, c='red', alpha = 0.7)

# plt.scatter(remain_reads.loc[:,'spot_location_1'], shape_row - remain_reads.loc[:,'spot_location_2'], s=1, alpha=0.2)
# plt.scatter(cell_center.loc[:,'column'], shape_row - cell_center.loc[:,'row'], s=3, c='red', alpha = 0.7)

ax.set_aspect('equal')
ax.axis('off')
plt.show()

fig, ax = plt.subplots(figsize=[40,40])
plt.scatter(remain_reads.loc[:,'spot_location_1'], remain_reads.loc[:,'spot_location_2'], s=1, alpha=0.2)

ax.set_aspect('equal')
ax.axis('off')
plt.show()