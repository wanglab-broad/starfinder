#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from tifffile import imread, imwrite
from skimage.util import img_as_ubyte, img_as_float, invert
from skimage.exposure import rescale_intensity
from skimage.morphology import disk
from skimage.filters import median

# load the images
current_dapi_img = imread(snakemake.input['dapi_img'])
current_flamingo_img = imread(snakemake.input['flamingo_img'])

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

imwrite(snakemake.output[0], current_output_img)