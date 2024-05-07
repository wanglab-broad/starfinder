#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tifffile import imread, imwrite
from skimage.util import img_as_ubyte, img_as_float, invert
from skimage.exposure import rescale_intensity
from skimage.measure import regionprops
from skimage.morphology import disk
from skimage.filters import median, gaussian 

def getCentroids(labels3D, save_df_path=None):
    from skimage.measure import label, regionprops, regionprops_table
    """
    Use skimage regionprops to locate centroid of each labeled nuclei
    Returns a nx3 array, where n is the total number of nuclei labeled, and columns are z,y,x coords respectively
    Returns a df with centroid coordinates, area/volume, and corresponding region label
    """
    labels3D = labels3D.astype(int)
    centroids_df = pd.DataFrame(
        regionprops_table(labels3D.astype(int),
                          properties = ('label', 'centroid', 'area', 'axis_major_length', 'axis_minor_length'))
    )
    if len(labels3D.shape) == 3:
        centroids_df.columns = ['cell_barcode', 'z', 'y', 'x', 'volume', 'axis_major_length', 'axis_minor_length']
        centroids_df = centroids_df.astype({
            'z':int, 
            'y':int, 
            'x':int
        })
    else:
        centroids_df.columns = ['cell_barcode', 'y', 'x', 'volume', 'axis_major_length', 'axis_minor_length']
        centroids_df = centroids_df.astype({
            'y':int, 
            'x':int
        })
    if save_df_path is not None:
        centroids_df.to_csv(save_df_path)

    return centroids_df

def MarkerArray_build(overlay, centroids_df):
    """
    Returns an array of markers in the shape of the overlay
    Each corresponding labeled nuclei in labels3D has a single label at its centroid location, or marker
    """
    centroids_df_f1 = np.array(centroids_df.loc[:,['cell_barcode','z','y','x']])
    markers = np.zeros(overlay.shape, dtype=np.uint16)
    for i in range(centroids_df_f1.shape[0]):
        c_barcode,z,y,x = centroids_df_f1[i,:]
        if z < overlay.shape[0] and y < overlay.shape[1] and x < overlay.shape[2]:
            markers[z,y,x] = c_barcode
    return markers

def seed_watershed(overlay, markers, strel3D = None):
    """
    Perform seed-based watershed segmentation of DAPI/amplicon overlay using markers
    @param strel is a 3D structuring element defining dilation parameter (i.e. ball)
    Returns a dilated 3D labeled image of cells
    """
    from skimage import segmentation,morphology
    if strel3D is None:
        strel3D = morphology.ball(7)
    overlay_processed = im_process(overlay, strel3D)
    cellLabels = segmentation.watershed(overlay_processed, markers, mask=overlay_processed)
    return cellLabels

def im_process(image, strel, show=False):
    """
    2D or 3D Image processing function 
    Performs sequential gaussian blur, median filter, thresholding, and morphological closing
    @param image is the image being processed
    @param strel is a structuring elem (2D or 3D depending on needs, ie ball(7) or disk(5))
    """
    from skimage import filters, morphology
    from scipy import ndimage as ndi
    blur = filters.gaussian(image, sigma=3)
    denoised = ndi.median_filter(blur, size=2)
    thresholded = denoised > filters.threshold_otsu(denoised)
    closed = morphology.closing(thresholded, footprint=strel)
    if show == True:
        plt.figure(figsize=(10,10))
        plt.imshow(closed)
    return closed

current_id = int(sys.argv[1])
current_fov_id = f"Position{current_id:03}"
print(current_fov_id)

# define the path to the images
base_path = '/stanley/WangLab/Data/Analyzed/2024-03-12-Mingrui-PFC/'
dapi_path = os.path.join(base_path, 'images/DAPI')
stardist_path = os.path.join(base_path, 'images/stardist_segmentation')
amplicon_path = os.path.join(base_path, 'images/ref_merged_round1')
output_path = os.path.join(base_path, 'images/cell_segmentation')
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
# load the images
current_dapi_img = imread(os.path.join(dapi_path, current_fov_id + '.tif'))
current_amplicon_img = imread(os.path.join(amplicon_path, current_fov_id + '.tif'))
current_stardist_img = imread(os.path.join(stardist_path, current_fov_id + '.tif'))

# enhance the contrast of the images (amplicon)
eh_value = 0.005
vmin = np.quantile(current_amplicon_img, eh_value)
vmax = np.quantile(current_amplicon_img, 1-eh_value)
vrange = (vmin, vmax)
current_amplicon_img_eh = rescale_intensity(current_amplicon_img, vrange)

# enhance the contrast of the images (dapi)
eh_value = 0.001
vmin = np.quantile(current_dapi_img, eh_value)
vmax = np.quantile(current_dapi_img, 1-eh_value)
vrange = (vmin, vmax)
current_dapi_img_eh = rescale_intensity(current_dapi_img, vrange)

# Create overlay image 
current_overlay = np.stack((current_amplicon_img_eh, current_dapi_img_eh), axis=0)
current_overlay = np.max(current_overlay, axis=0)
current_overlay = gaussian(current_overlay, sigma=3)

# Cell segmentation
centroids_df = getCentroids(current_stardist_img)
# marker_seed = MarkerArray_build(current_overlay, centroids_df)
current_segmentation_output = seed_watershed(current_overlay, current_stardist_img)
imwrite(os.path.join(output_path, f'{current_fov_id}.tif'), current_segmentation_output)