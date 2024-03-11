#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
from tifffile import imread, imwrite
from skimage.util import img_as_ubyte, img_as_float, invert
from skimage.exposure import rescale_intensity
from skimage.morphology import disk
from skimage.filters import median

current_id = int(sys.argv[1])
current_fov_id = f"Position{current_id:03}"
print(current_fov_id)

# define the path to the images
base_path = '/stanley/WangLab/Data/Analyzed/2024-03-08-Hongyu-Covid_LN/'
dapi_path = os.path.join(base_path, 'images/flamingo/DAPI')
flamingo_path = os.path.join(base_path, 'images/flamingo/Flamingo')
output_path = os.path.join(base_path, 'images/flamingo/output')
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
# load the images
current_dapi_img = imread(os.path.join(dapi_path, current_fov_id + '.tif'))
current_flamingo_img = imread(os.path.join(flamingo_path, current_fov_id + '.tif'))

# enhance the contrast of the images (flamingo)
eh_value = 0.005
vmin = np.quantile(current_flamingo_img, eh_value)
vmax = np.quantile(current_flamingo_img, 1-eh_value)
vrange = (vmin, vmax)
current_flamingo_img_eh = rescale_intensity(current_flamingo_img, vrange)
for z in range(current_flamingo_img_eh.shape[0]):
    current_flamingo_img_eh[z] = median(current_flamingo_img_eh[z], disk(1))

# enhance the contrast of the images (dapi)
eh_value = 0.001
vmin = np.quantile(current_dapi_img, eh_value)
vmax = np.quantile(current_dapi_img, 1-eh_value)
vrange = (vmin, vmax)
current_dapi_img_eh = rescale_intensity(current_dapi_img, vrange)

# create the output image
current_dapi_img_eh = img_as_float(current_dapi_img_eh)
current_flamingo_img_eh = img_as_float(current_flamingo_img_eh)
current_flamingo_invert = invert(current_flamingo_img_eh)

current_output_img = current_dapi_img_eh * current_flamingo_invert
current_output_img = img_as_ubyte(current_output_img)

imwrite(os.path.join(output_path, f'{current_fov_id}.tif'), current_output_img)