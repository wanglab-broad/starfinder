"""
3D Segmentation and Read Assignment
Functions to perform 2.5D segmentation of DAPI stains and DAPI/amplicon channel overlays using StarDist

NOTE: The functions in this script have been formatted to be compatible with Snakemake StarFinder pipeline. 
"""

# Import packages
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from glob import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tifffile import imread, imwrite
from skimage import filters, morphology, segmentation
from skimage.measure import regionprops, regionprops_table
from scipy import ndimage as ndi
from scipy.io import loadmat
import argparse
import math
import timeit


#================================= LOAD AND ROTATE DATA =================================

# 0. Load channels of reference round 
def getChannels(input_path, output_path, r1merged_path):
    """
    Load data 
        - load dapi channel 
        - load each of the 4 channels
        - take maximum of 4 channels and DAPI = overlay
    Return:
        - dapi channel (3D arrays)
        - overlay (3D array maximum of 5 channels)
    """
    # Get channel paths
    dapi_path = glob(os.path.join(input_path, '*ch04.tif'))

    # Import and store images as 3D arrays
    dapi = imread(dapi_path)

    print("Rotating DAPI and round 1 channels...")
    # Rotate dapi by 90 degrees CW and write to file
    dapi_rotate = dapi.copy()
    for i in range(dapi_rotate.shape[0]):
        dapi_rotate[i,:,:] = ndi.rotate(dapi_rotate[i,:,:], 270, reshape=False)
    dapi_rotate_max = dapi_rotate.max(axis=0)
    imwrite(os.path.join(output_path, 'max_rotated_dapi.tif'), dapi_rotate_max, bigtiff=False)
    print("Saved 'max_rotated_dapi.tif'")

    # Take maximum of all 4 channels and DAPI = overlay
    # overlay = dapi_rotate 
    # for ch in chs:
    #     # rotate and take maximum 
    #     ch_rotate = ch.copy()
    #     for i in range(ch_rotate.shape[0]):
    #         ch_rotate[i,:,:] = ndi.rotate(ch_rotate[i,:,:], 270, reshape=False)
    #     overlay = np.maximum(overlay, ch_rotate) 

    r1merged = imread(r1merged_path)
    for i in range(r1merged.shape[0]):
        r1merged[i,:,:] = ndi.rotate(r1merged[i,:,:], 270, reshape=False)
    
    overlay = np.maximum(r1merged, dapi_rotate)
    imwrite(os.path.join(output_path, 'max_rotated_overlay.tif'), overlay.max(axis=0), bigtiff=False)

    return dapi_rotate, overlay


#================================= HELPER FNS =================================

# Log function
def log(msg, output_path, first=False):
    """
    Given a message, print message to screen for visual tracking and appends to log.txt
    @param msg: string to print and log
    @param output_path: path to tile folder
    @param first: boolean for whether to write (first log of the run, overwrites existing log) or append
    """
    print(msg)
    if first:
        fid = open(os.path.join(output_path, 'log.txt'), "w")
    else: 
        fid = open(os.path.join(output_path, 'log.txt'), "a")
    fid.write(msg)
    fid.close()

#================================= DAPI SEGMENTATION =================================

# 1. Perform 2D segmentation using StarDist2D pre-trained model
def segmentDapi2D(dapi, preprocess=False, save_path=None):
    """
    2D segmentation of nuclei
    returns a 2D label array
    """
    # Get StarDist2D pretrained versatile fluorescent nuclei model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    # Take 2D maximum projection of DAPI and segment with model
    dapi_max = dapi.max(axis=0)
    if preprocess:
        dapi_max = filters.gaussian(dapi_max, sigma=4)
    labels2D, _ = model.predict_instances(normalize(dapi_max))

    if save_path:
        plt.figure(figsize=(20,10))
        plt.subplot(121)
        plt.title('DAPI')
        plt.imshow(dapi.max(axis=0))
        plt.subplot(122)
        plt.title('StarDist labels')
        plt.imshow(labels2D)
        plt.savefig(os.path.join(save_path, 'dapi_segmentation.png'))

    return labels2D

# 2. Process and binarize DAPI
def im_process(image, strel, show=False):
    """
    2D or 3D Image processing function 
    Performs sequential gaussian blur, median filter, thresholding, and morphological closing
    @param image is the image being processed
    @param strel is a structuring elem (2D or 3D depending on needs, ie ball(7) or disk(5))
    """
    
    blur = filters.gaussian(image, sigma=3)
    denoised = ndi.median_filter(blur, size=2)
    thresholded = denoised > filters.threshold_otsu(denoised)
    closed = morphology.closing(thresholded, selem=strel)
    
    if show == True:
        plt.figure(figsize=(10,10))
        plt.imshow(closed)
        
    return closed
    
# 3. 2.5D segment by multiplying each layer of binary threshold DAPI z-stack with StarDist 2D labels
def segmentDapi2_5D(dapi, labels2D, strel):
    """
    Perform 2.5D segmentation on 3D DAPI channel:
    - Extract each layer of DAPI z-stack
    - Process and binarize layer
    - Multiply binarized layer with 2D labels array to transfer labels to DAPI
    Return z-stack (3D array) of labeled DAPI channel
    """
    labels3D = np.zeros(dapi.shape)
    for z in range(0, dapi.shape[0]):
        layer = dapi[z] # extract
        processed = im_process(layer, strel, show=False)
        labels3D[z] = processed * labels2D

    return labels3D

# 4. Get centroid labels from labeled image (DAPI or cell) and use as watershed seeds for later use in overlay segmentation
def getCentroids(labels3D, save_df_path=None):
    """
    Use skimage regionprops to locate centroid of each labeled nuclei
    Returns a nx3 array, where n is the total number of nuclei labeled, and columns are z,y,x coords respectively
    Returns a df with centroid coordinates, area/volume, and corresponding region label
    """
    centroids = []
    labels3D = labels3D.astype(int)
    for i, region in enumerate(regionprops(labels3D)):
        centroids.append(region.centroid)
    centroids = np.array(centroids).astype(int)

    centroids_df = pd.DataFrame(
        regionprops_table(labels3D.astype(int),
                          properties = ('label', 'centroid', 'area'))
    )
    centroids_df.columns = ['cell_barcode', 'z', 'y', 'x', 'volume']
    centroids_df = centroids_df.astype({
        'z':int, 
        'y':int, 
        'x':int
    })
    if save_df_path:
        centroids_df.to_csv(save_df_path)

    return centroids



#================================= CELL SEGMENTATION =================================

# 5. Assign DAPI centroid as marker in shape of image
def getMarkerArray(overlay, centroids):
    """
    Returns an array of markers in the shape of the overlay
    Each corresponding labeled nuclei in labels3D has a single label at its centroid location, or marker
    """
    numCells = centroids.shape[0]
    markers = np.zeros(overlay.shape, dtype=np.uint8)
    for i in range(numCells):
        z,y,x = centroids[i,:]
        if z < overlay.shape[0] and y < overlay.shape[1] and x < overlay.shape[2]:
            markers[z,y,x] = i+1

    return markers, numCells

# 6. Process overlay image and watershed segment
def segmentCells3D(overlay, markers, strel3D):
    """
    Perform seed-based watershed segmentation of DAPI/amplicon overlay using markers
    @param strel is a 3D structuring element defining dilation parameter (i.e. ball)
    Returns a dilated 3D labeled image of cells
    """
    overlay_processed = im_process(overlay, strel3D)
    cellLabels = segmentation.watershed(overlay_processed, markers, mask=overlay_processed)
    cellLabels = morphology.dilation(cellLabels, selem=strel3D)
    
    return cellLabels

# 6.1 If no overlay image or DAPI signals are extremely dense, just dilate dapi labels to cell labels
def dilateCells3D(labels3D, strel3D):
    return morphology.dilation(labels3D, strel3D)



#================================= READ ASSIGNMENT =================================

# 8. Load reads.csv (output)
def loadReads(base_path, output_path, output_tile_path, tile_num):
    """
    Loads and returns a file containing amplicon read coordinates and genes
    """
    reads = pd.read_csv(os.path.join(base_path, f'output/tile_{tile_num}.csv'), index_col=0)
    return reads

# 8.1 Load goodPoints_max3D.mat (matlab output)
def loadReadsMat(base_path, tile_num, output_path, assign_dir):
    dots = loadmat(base_path, f'output/max/tile_{tile_num}')
    bases = [i[0] for i in dots["goodReads"]]
    bases = np.array(bases)
    temp = dots["goodSpots"]
    temp = temp[:,:3]
    points = np.zeros(temp.shape)
    points[:,0] = temp[:,0]-1
    points[:,1] = temp[:,1]-1
    points[:,2] = temp[:,2]-1

    # Log
    assign_output_path = os.path.join(output_path, assign_dir)
    if not os.path.exists(assign_output_path):
        os.makedirs(assign_output_path)
    #log(f"\tNumber of reads: {len(bases)}\n", output_tile_path)
    
    return bases, points

# 8.2.5 Helper function for 8.2 Load goodPoints_max3d.csv
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
    rotated_matrix = pd.DataFrame({'x':qx.astype(int),
                                   'y':qy.astype(int)})
    return rotated_matrix

# 8.2 Load goodPoints_max3d.csv (output of 3-part registration pipeline)
def loadReadsNew(reads_path, dapi):
    # Load reads (currently not 0-based, so adjust)
    # reads = pd.read_csv(os.path.join(base_path, f'02.processed_data/Position{tile_num}/goodPoints_max3d.csv'))
    
    reads = pd.read_csv(reads_path)
    for c in ['x', 'y', 'z']:
        reads[c] -= 1

    # Rotate reads by 90 degrees CW
    reads_rotated = rotate_around_point_highperf(
        np.array([reads.loc[:,'x'],reads.loc[:,'y']],dtype=int),
        math.radians(270),
        [int(dapi.shape[1]/2+0.5),int(dapi.shape[2]/2+0.5)]
    )

    # Add z column and genes back in 
    reads_rotated['z'] = reads['z'].copy()
    reads_rotated['Gene'] = reads['Gene'].copy()

    return reads_rotated

# 9. Load genes.csv (with the raw data)
def loadGenes(base_path):
    """
    Loads genes.csv located in root data directory
    """
    gene_path = os.path.join(base_path, '01.data/genes.csv') 
    genes = pd.read_csv(gene_path, header=None, names=["Gene Name", "Barcode"])["Gene Name"]
    gene2idx = {gene:idx for idx, gene in enumerate(genes)}
    return genes, gene2idx

# 10. Assign reads and write to file
def assignReads(numCells, reads, genes, cellLabels, gene2idx, output_path):
    matrix = np.zeros((numCells, len(genes)))
    for i in range(len(reads)):
        x,y,z,gene = reads.iloc[i].values
        cell = cellLabels[z,y,x]
        if cell != 0:
            matrix[cell.astype(int)-1, gene2idx[gene]] += 1

    # Write matrix to file
    np.savetxt(os.path.join(output_path, 'readMatrix.csv'), matrix, fmt='%d', delimiter=',')

    return matrix # just for logging purposes

# 10.1 Assign reads and write to file (for MATLAB input)
def assignReadsMat(numCells, bases, points, cellLabels, gene2idx, output_path, assign_dir, tile_num):
    matrix = np.zeros((numCells, len(gene2idx.keys())))
    genecounts_mat = {}
    for i in range(len(points)):
        x,y,z = points[i,:].astype(int)
        gene = ''.join(bases[i])
        cell = cellLabels[z,y,x]
        if cell != 0:
            matrix[cell.astype(int)-1, gene2idx[gene]] += 1
            genecounts_mat[gene] += 1
    matrix = matrix.astype(int)

    # Write matrix to file
    np.savetxt(os.path.join(output_path, assign_dir, f'tile{tile_num}_readmatrix_mat.csv'))

    return matrix

# 10.2 Assign reads and create a remain_reads.csv file 
def assignRemainReads(reads, cellLabels, output_path):
    cell_barcode = []
    for i in range(len(reads)):
        x,y,z,gene = reads.iloc[i].values
        cell = cellLabels[z,y,x] # extract whether read overlaps with a cell
        if cell != 0:
            cell_barcode.append(cell) # append corresponding cell label
        else:
            cell_barcode.append(0)
    reads['cell_barcode'] = cell_barcode

    # Save assigned reads
    remain_reads = reads[reads['cell_barcode']!=0]
    remain_reads.to_csv(os.path.join(output_path, 'remain_reads.csv'))

    print(f"Total reads: {len(cell_barcode)}")
    print(f"Assigned reads: {len(remain_reads)}")
    pct_assigned = round((len(remain_reads)) / len(cell_barcode) * 100, 2)
    print(f"Assignment percentage: {pct_assigned}%")
    
    return remain_reads, pct_assigned

# 11. Plot results
def plotSegAssignResults(dapi, cell_centroids, cellLabels, reads, remain_reads, output_path):
    fig = plt.figure(figsize=(30,30))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    # Plot dapi max projection
    plt.subplot(221)
    plt.title('DAPI max projection', fontdict={'fontsize':30})
    plt.axis('off')
    plt.imshow(dapi.max(axis=0))

    # Plot cell segmentation 2D max projection
    plt.subplot(222)
    plt.title(f'Cell segmentation ({len(cell_centroids)} cells)', fontdict={'fontsize':30})
    plt.axis('off')
    plt.imshow(cellLabels.max(axis=0))

    # Plot dapi with cell centers
    plt.subplot(223)
    plt.title('DAPI with cell centers', fontdict={'fontsize':30})
    plt.axis('off')
    plt.imshow(dapi.max(axis=0))
    plt.scatter(cell_centroids[:,2], cell_centroids[:,1], s=25, c='red')

    # Plot dapi with remain reads and all reads
    plt.subplot(224)
    pct_assigned = round((len(remain_reads) / len(reads) * 100), 2)
    plt.title(f'DAPI with all/assigned reads ({pct_assigned}%)', fontdict={'fontsize':30})
    plt.axis('off')
    plt.imshow(dapi.max(axis=0))
    plt.scatter(reads['x'], reads['y'], alpha=0.1, c='red', s=10)
    plt.scatter(remain_reads['x'],remain_reads['y'],s = 10,alpha = 0.8,c=pd.Categorical(np.array(remain_reads['cell_barcode'])).codes, cmap= matplotlib.colors.ListedColormap ( np.random.rand ( 256,3)))

    plt.savefig(os.path.join(output_path, 'segmentation_assignment_results.png'))


#================================= MASTER FUNCTIONS =================================

# Segmentation 
def segment(input_path, output_tile_path, strel2D, strel3D):
    """
    Wrapping together all segmentation functions
    @param input_path: base_path/round1/tile_num
    @param output_tile_path: base_path/output/segassign/tile_X
    @param strel2D: structuring element used in morphological closing for 2.5D DAPI segmentation
    @param strel3D: structuring element used in morphological closing and dilation for 3D overlay segmentation
    #@param assign: boolean for whether assignment will be performed in this job
    """
    start = timeit.default_timer()

    # DAPI segmentation
    log("Performing DAPI Segmentation...", output_tile_path)
    dapi, overlay = getChannels(input_path)
    labels2D = segmentDapi2D(dapi)
    labels3D = segmentDapi2_5D(dapi, labels2D, strel2D)
    centroids = getDapiCentroids(labels3D)

    # Cell segmentation
    log("Performing cell segmentation...", output_tile_path)
    markers, numCells = getMarkerArray(overlay, centroids)
    cellLabels = segmentCells3D(overlay, markers, strel3D)
    saveCellLabels(cellLabels, output_tile_path)
    
    stop = timeit.default_timer()

    # Log
    msg = f"Segmentation results:\n\tNumber of cells: {numCells}\n\tTime: {round(stop-start,3)} seconds\n"
    log(msg, output_tile_path)

# Read assignment
def readAssignment(base_path, output_path, output_tile_path, tile_num, mat):
    """
    Wrapping together all read assignment functions 
    @param output_tile_path: base_path/output/segassign/tile_X
    @param tile_num: number of tile
    @param mat: boolean for whether reads locations are output from MATLAB
    @param numCells: number of cells labeled in segmentation mask
    @param seg: boolean for whether or not seg was performed in this job
    @param seg_dir: base_path/output/segassign/seg
    """
    start = timeit.default_timer()

    log("Performing read assignment...", output_tile_path)
    
    # Load segmentation mask
    if os.path.exists(output_tile_path): 
        cellLabels = imread(os.path.join(output_tile_path, 'cellLabels.tiff'))
        numCells = np.unique(cellLabels).max().astype(int)
    else:
        raise ValueError("Cannot perform read assignment without segmentation mask")
        
    if not mat:
        reads = loadReads(base_path, output_path, output_tile_path, tile_num)
        genes, gene2idx = loadGenes(base_path)
        matrix = assignReads(numCells, reads, genes, cellLabels, gene2idx, output_tile_path)
    else:
        bases, points = loadReadsMat(base_path, tile_num, output_path)
        _, gene2idx = loadGenes(base_path)
        matrix = assignReadsMat(numCells, bases, points, cellLabels, gene2idx, output_path, tile_num)

    stop = timeit.default_timer()

    # Log
    msg = f"Read assignment result:\n\t{round(matrix.sum()/len(reads) * 100, 4)}% reads assigned ({matrix.sum().astype(int)} out of {len(reads)} reads)\n\tTime: {round(stop-start, 3)} seconds\n"
    log(msg, output_tile_path)

# Segmentation and assignment v2 (using results from 3-part pipeline):
def segAssign3D(base_path, tile, source_data_path, r1merged_path, reads_path, output_path, strel2D, strel3D):

    # DAPI segmentation
    start = timeit.default_timer()

    print("Performing DAPI Segmentation...")
    dapi, overlay = getChannels(source_data_path, output_path, r1merged_path)
    labels2D = segmentDapi2D(dapi, preprocess=True, save_path=os.path.join(output_path))
    labels3D = segmentDapi2_5D(dapi, labels2D, strel2D)
    dapi_centroids = getCentroids(labels3D, save_df_path=os.path.join(output_path, 'dapi_centroids.csv'))

    stop = timeit.default_timer()
    msg = f"Finished DAPI segmentation and saved files. Time: {round(stop-start,3)} seconds\n"
    print(msg)

    # Cell segmentation
    print("Performing cell segmentation...")
    start = timeit.default_timer()

    # cellLabels = dilateCells3D(labels3D, strel3D)
    # markers, numCells = getMarkerArray(overlay, dapi_centroids)
    # cellLabels = segmentCells3D(overlay, markers, strel3D)
    cellLabels = dilateCells3D(labels3D, strel3D)
    cell_centroids = getCentroids(cellLabels, save_df_path=os.path.join(output_path, 'cell_center.csv'))

    # Save cell labels and dapi_segmentation
    np.save(os.path.join(output_path, 'cellLabels.npy'), cellLabels)
    np.save(os.path.join(output_path, 'dapiLabels.npy'), labels3D)  

    stop = timeit.default_timer()
    msg = f"Segmentation results:\n\tNumber of cells: {cell_centroids.shape[0]}\n\tTime: {round(stop-start,3)} seconds\n"
    print(msg)

    # Reads assignment
    start = timeit.default_timer()

    print("Performing read assignment...")
    
    reads = loadReadsNew(reads_path, dapi)
    remain_reads, pct_assigned = assignRemainReads(reads, cellLabels, output_path)
    plotSegAssignResults(dapi, cell_centroids, cellLabels, reads, remain_reads, output_path)

    stop = timeit.default_timer()

    msg = f"Read assignment result:\n\t{pct_assigned}% reads assigned ({len(remain_reads)} out of {len(reads)} reads)\n\tTime: {round(stop-start, 3)} seconds\n"
    print(msg)
    print("Segmentation and reads assignment done!")


#================================= SNAKEMAKE EXECUTION =================================

# Parse arguments
base_path = os.path.join(snakemake.config['user_dir'], snakemake.config['sample'])
tile = snakemake.wildcards.pos
# can parameterize structuring elements for morph ops in future -- for now, keep default
strel2D = morphology.disk(snakemake.config['disk_radius'])
strel3D = morphology.ball(snakemake.config['ball_radius'])
source_data_path = os.path.join(base_path, f"{snakemake.config['subdirs']['source_data']}/round1/{tile}") 
r1merged_path = os.path.join(base_path, f"{snakemake.config['subdirs']['registration']}/{tile}/interm/r1merged.tif")
reads_path = os.path.join(base_path, f"{snakemake.config['subdirs']['registration']}/{tile}/goodPoints_{snakemake.config['spotfinding_method']}.csv")
output_path = os.path.join(base_path, f"{snakemake.config['subdirs']['segmentation']}/watershed/{tile}")
os.makedirs(output_path, exist_ok=True)

# Check if tile exists
if not os.path.exists(source_data_path):
    raise NameError('Tile data does not exist or is not accessible')

print(f"=================================== {tile} ===================================\n")

segAssign3D(
    base_path, 
    tile, 
    source_data_path, 
    r1merged_path,
    reads_path, 
    output_path, 
    strel2D, 
    strel3D
)