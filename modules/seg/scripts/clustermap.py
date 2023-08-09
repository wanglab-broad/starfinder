import sys,os
from glob import glob
import copy, psutil, math
import numpy as np
import pandas as pd
import tifffile as tif
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import ndimage
from ClusterMap.clustermap import *
from skimage.exposure import adjust_gamma
from skimage.filters import gaussian
from sklearn import preprocessing
from skimage.morphology import disk,erosion,dilation,closing,ball
from tqdm import tqdm

# [======================== Parse Parameters ========================]

tissue = snakemake.config['sample']
tile_dir = snakemake.wildcards.pos
img_z = snakemake.config['z']
cell_num_threshold = snakemake.config['cell_num_threshold'] # Threshold for determining number of cells. Larger value -> less cells 
dapi_grid_interval = snakemake.config['dapi_grid_interval'] # sample interval in DAPI image. A large value will reduce computation resources. Default: 3. 
[xy_radius, z_radius] =  snakemake.config['cell_radius']
img_c = snakemake.config['xy'] 
img_r = snakemake.config['xy'] 
dapi_round = snakemake.config['ref_round'] # Round with DAPI 
pct_filter = snakemake.config['pct_filter'] # Percent filter (percent of noise reads filtered out -- larger number will keep less reads) 
useown_dapi_bi = snakemake.config['extra_preprocess']

# These params will usually stay constant
left_rotation = 270 # 90-degree CW rotation 
reads_filter = 5 # minimum number of reads for a subtile to be kept
overlap_percent = 0.2
window_size = 550

# File I/O
proj_dir = os.path.join(snakemake.config['user_dir'], tissue)
gene_path = os.path.join(proj_dir, snakemake.config['subdirs']['source_data'], "genes.csv")
dapi_path = glob(os.path.join(proj_dir, snakemake.config['subdirs']['source_data'], f"round{dapi_round}", tile_dir, "*ch04.tif"))[0]
input_path = os.path.join(proj_dir, snakemake.config['subdirs']['registration'], tile_dir)
output_path = os.path.join(proj_dir, snakemake.config['subdirs']['segmentation'], "clustermap", tile_dir)
os.makedirs(output_path, exist_ok=True)


# [======================== Helper Functions ========================]

def rotate_around_point_highperf(xy, radians, origin=(0, 0)):
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    qx = (qx + 0.5).astype(int)
    qy = (qy + 0.5).astype(int)
    rotated_matrix = pd.DataFrame({'column':qx.astype(int),
                                    'row':qy.astype(int)})
    return rotated_matrix


# [======================== Rotate and max proj DAPI  ========================]

filter_value2 = 1 # np.percentile(dapi_f2,50)

dapi = tif.imread(dapi_path) ## [z,row,col] [z,y,x]
img_z = dapi.shape[0]
dapi = dapi[0:img_z,:,:]   ## let all the used dapi file be the same layer number.

dapi_f2 = dapi.copy()
for i in tqdm(range(img_z)):
    ## rotation
    dapi_f2[i,:,:] = ndimage.rotate(dapi_f2[i,:,:], left_rotation, reshape=False)

# Write rotated dapi max proj for later FIJI stitching
dapi_rotate_max = dapi_f2.max(axis = 0)
tif.imwrite(os.path.join(output_path, 'max_rotated_dapi.tif'), dapi_rotate_max, bigtiff=False)
dapi_f2 = np.transpose(dapi_f2, (1,2,0))

plt.figure(figsize=[20,20])
plt.subplot(2,2,1)
plt.imshow(ndimage.rotate(dapi[14,:,:], left_rotation, reshape=False),cmap = 'gray')
plt.subplot(2,2,2)
plt.imshow(dapi_f2[:,:,14],cmap = 'gray')
plt.subplot(2,2,3)
plt.imshow(ndimage.rotate(dapi.max(0), left_rotation, reshape=False),cmap = 'gray')
plt.subplot(2,2,4)
plt.imshow(dapi_f2.max(2),cmap = 'gray')

plt.savefig(os.path.join(output_path, "filtered_dapi.png"))


# [======================== Load spot-finding reads ========================]

points_df_t = pd.read_csv(os.path.join(input_path, "goodPoints_max3d.csv"))
points_df_t.columns = ['column','row','z_axis','gene']
if left_rotation%360 != 0:
    rotate_cor = rotate_around_point_highperf(np.array([points_df_t.loc[:,'column'],points_df_t.loc[:,'row']],dtype=int),math.radians(left_rotation),[int(img_c/2+0.5),int(img_r/2+0.5)])
else:
    rotate_cor = copy.deepcopy(points_df_t)
points_df_t.loc[:,'column'] = np.array(rotate_cor.loc[:,'column'])
points_df_t.loc[:,'row'] = np.array(rotate_cor.loc[:,'row'])
points_df_t = points_df_t.loc[points_df_t['z_axis'] <= img_z,:]
points_df_t.reset_index(drop = True, inplace = True)


# [======================== Clustermap Model Initialization ========================]

# Initialize spots df
spots = pd.DataFrame({'gene_name' : points_df_t['gene'],'spot_location_1' : points_df_t['column'],'spot_location_2': points_df_t['row'],'spot_location_3':points_df_t['z_axis']})
# If less than reads_filter (default 5) spots found, then write empty file and exit
if(spots.shape[0] < reads_filter):
    remain_reads = pd.DataFrame({'gene_name':[],'spot_location_1':[],'spot_location_2':[],'spot_location_3':[],'gene':[],'is_noise':[],'clustermap':[]})
    cell_center_df = pd.DataFrame({'cell_barcode':[],'column':[],'row':[],'z_axis':[]})
    remain_reads.to_csv(os.path.join(output_path,'remain_reads.csv'))
    cell_center_df.to_csv(os.path.join(output_path,'cell_center.csv'))
    sys.exit()

### convert gene_name to gene identity
genes=pd.DataFrame(spots['gene_name'].unique())
at1=list(genes[0])
gene=list(map(lambda x: at1.index(x)+1, spots['gene_name']))
spots['gene']=gene
spots['gene']=spots['gene'].astype('int')

### Initialize clustermap model
num_gene=np.max(spots['gene'])
gene_list=np.arange(1,num_gene+1)
num_dims=len(dapi_f2.shape)

# Plug in a previously pre-processed binarized DAPI 
if useown_dapi_bi:
    model = ClusterMap(spots=spots, dapi=None, gene_list=gene_list, num_dims=num_dims,
                    xy_radius=xy_radius,z_radius=z_radius,fast_preprocess=True,gauss_blur = True)
    model.dapi, model.dapi_binary, model.dapi_stacked = [dapi_f2, dapi_bi, dapi_bi_max] ### use own binary dapi
# Else, rely on clustermap DAPI pre-processing
else:
    model = ClusterMap(spots=spots, dapi=dapi_f2, gene_list=gene_list, num_dims=num_dims,
                xy_radius=xy_radius,z_radius=z_radius,fast_preprocess=False)

### Plot preprocessing results
model.preprocess(dapi_grid_interval=dapi_grid_interval,pct_filter=pct_filter)
model.spots['is_noise']=model.spots['is_noise']+1
model.plot_segmentation(figsize=(5,5),s=0.6,method='is_noise',
                        cmap=np.array(((0,1,0),(1,0,0))),
                        plot_dapi=True,save = True,savepath = os.path.join(output_path, 'cellseg_noisecheck.png'))
model.spots['is_noise']=model.spots['is_noise']-min(model.spots['is_noise'])-1
model.min_spot_per_cell=reads_filter

# [======================== Clustermap Segmentation ========================]

if window_size == 0:
    model.segmentation(cell_num_threshold=cell_num_threshold,dapi_grid_interval=dapi_grid_interval,add_dapi=True,use_genedis=True)
else:
    img = dapi_f2
    label_img = get_img(img, model.spots, window_size=window_size, margin=math.ceil(window_size*overlap_percent))
    out = split(img, label_img, model.spots, window_size=window_size, margin=math.ceil(window_size*overlap_percent))
    
    # Process each tile and stitch together
    cell_info={'cellid':[],'cell_center':[]}
    model.spots['clustermap']=-1

    for tile_split in range(out.shape[0]):
        print(f'tile: {tile_split}')
        spots_tile=out.loc[tile_split,'spots']
        dapi_tile=out.loc[tile_split,'img']

        # Initialize subtile clustermap model using dapi version
        if useown_dapi_bi:
            model_tile = ClusterMap(spots=spots_tile, dapi=None, gene_list=gene_list, num_dims=num_dims,
                            xy_radius=xy_radius,z_radius=z_radius,fast_preprocess=True,gauss_blur = True)
            dapi_bi_tile = dapi_tile > filter_value2
            dapi_bi_max_tile = dapi_bi_tile.max(2)
            model_tile.dapi, model_tile.dapi_binary, model_tile.dapi_stacked = [dapi_tile, dapi_bi_tile, dapi_bi_max_tile] ### use own binary dapi
        else:
            model_tile = ClusterMap(spots=spots_tile, dapi=dapi_tile, gene_list=gene_list, num_dims=num_dims,
                xy_radius=xy_radius,z_radius=z_radius,fast_preprocess=False)

        if (model_tile.spots.shape[0] < reads_filter) | (sum(model_tile.spots['is_noise'] == 0) < reads_filter):
            print(f"Less than {reads_filter} spots found in subtile. Skipping and continuing...")
            continue

        if (np.sum(model_tile.dapi_binary) == 0):
            print(f"No dapi spots found in subtile. Skipping and continuing...")
            continue

        else:
            ### segmentation
            model_tile.min_spot_per_cell=reads_filter
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
                    ### add subtile into full tile model reuslts
                    cell_info=model.stitch(model_tile, out, tile_split)


print("Finished analyzing all subtiles")

# If tile is completely noise, throw out
if not hasattr(model, 'all_points_cellid'):
    print("No denoised cell centers found in this tile.")
    remain_reads = pd.DataFrame({'gene_name':[],'spot_location_1':[],'spot_location_2':[],'spot_location_3':[],'gene':[],'is_noise':[],'clustermap':[]})
    cell_center_df = pd.DataFrame({'cell_barcode':[],'column':[],'row':[],'z_axis':[]})
    remain_reads.to_csv(os.path.join(output_path,'remain_reads.csv'))
    cell_center_df.to_csv(os.path.join(output_path,'cell_center.csv'))
    sys.exit()

# [======================== Clustermap Results -- Plot and Save ========================]

model.plot_segmentation(figsize=(10,10),s=3,plot_with_dapi=True,plot_dapi=True, show=False)
plt.scatter(model.cellcenter_unique[:,1],model.cellcenter_unique[:,0],c='r',s=5)
plt.savefig(os.path.join(output_path, 'cellseg_result.png'))

# Save results
model.spots['clustermap'] = model.spots['clustermap'].astype(int)
remain_reads = model.spots.loc[model.spots['clustermap'] >= 0,:]
remain_reads.reset_index(inplace = True,drop = True)
remain_reads_raw = model.spots
remain_reads_raw.reset_index(inplace = True,drop = True)
cell_center_df = pd.DataFrame({'cell_barcode' : model.cellid_unique.astype(int),'column' : model.cellcenter_unique[:,1],'row': model.cellcenter_unique[:,0],'z_axis':model.cellcenter_unique[:,2]})
remain_reads_raw.to_csv(os.path.join(output_path , 'remain_reads_raw.csv'))
remain_reads.to_csv(os.path.join(output_path , 'remain_reads.csv'))
cell_center_df.to_csv(os.path.join(output_path, 'cell_center.csv'))

if(remain_reads.shape[0] == 0):
    sys.exit()

cmap=np.random.rand(int(max(remain_reads['clustermap'])+1),3)
binary_dapi = np.flipud(model.dapi_binary)
s = 5
plt.figure(figsize=[20,20])
plt.subplot(2,2,1)
plt.imshow(ndimage.rotate(dapi.max(0), left_rotation, reshape=False) ,cmap = 'gray')
plt.gca().set_title('dapi (max projection)',fontsize=15)
plt.subplot(2,2,2)
plt.imshow(dapi_f2[:,:,14],cmap = 'gray')
plt.gca().set_title('Pre-filtered dapi (layer 15)',fontsize=15)
plt.subplot(2,2,3)
plt.imshow(ndimage.rotate(dapi.max(0), left_rotation, reshape=False),cmap = 'gray')
plt.scatter(remain_reads['spot_location_1'],remain_reads['spot_location_2'],c=cmap[[int(x) for x in remain_reads['clustermap']]],s=s,alpha = 0.1)
plt.scatter(model.cellcenter_unique[:,1],model.cellcenter_unique[:,0],c = 'red', s=20)
plt.gca().set_title('dapi (max projection) + reads',fontsize=15)
plt.subplot(2,2,4)
plt.imshow(binary_dapi.max(2),origin='lower',cmap = 'gray')
plt.scatter(remain_reads['spot_location_1'],binary_dapi.shape[1] - remain_reads['spot_location_2'],c=cmap[[int(x) for x in remain_reads['clustermap']]],s=s,alpha = 0.1)
plt.scatter(model.cellcenter_unique[:,1],binary_dapi.shape[1] - model.cellcenter_unique[:,0],c = 'red', s=20)
plt.gca().set_title('Clustermap-filtered dapi \n (max projection) + reads',fontsize=15)
plt.savefig(os.path.join(output_path,'final_segmentation_results.png'))

model.plot_segmentation(figsize=(30,15),s=0.05,plot_with_dapi=True,plot_dapi=True, show=False)
plt.savefig(os.path.join(output_path, f'clustermap_segmentation.png'))
