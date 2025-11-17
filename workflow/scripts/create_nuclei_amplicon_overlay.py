#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tifffile import imread, imwrite
from skimage.util import img_as_ubyte, img_as_float
from skimage.exposure import rescale_intensity

# load the images
current_dapi_img = imread(snakemake.input['dapi_img'])
current_amplicon_img = imread(snakemake.input['amplicon_img'])

# enhance the contrast of the images (amplicon)
eh_value = 0.001
vmin = np.quantile(current_amplicon_img, eh_value)
vmax = np.quantile(current_amplicon_img, 1-eh_value)
vrange = (vmin, vmax)
current_amplicon_img_eh = rescale_intensity(current_amplicon_img, vrange)

# enhance the contrast of the images (dapi)
eh_value = 0.005
vmin = np.quantile(current_dapi_img, eh_value)
vmax = np.quantile(current_dapi_img, 1-eh_value)
vrange = (vmin, vmax)
current_dapi_img_eh = rescale_intensity(current_dapi_img, vrange)

# create the output image
current_dapi_img_eh = img_as_float(current_dapi_img_eh)
current_amplicon_img_eh = img_as_float(current_amplicon_img_eh)
current_overlay_img = np.zeros(current_dapi_img_eh.shape + (2,))
current_overlay_img[..., 0] = current_dapi_img_eh
current_overlay_img[..., 1] = current_amplicon_img_eh
current_overlay_img = current_overlay_img.max(axis=3)

if snakemake.config['rules']['create_nuclei_amplicon_overlay']['parameters']['maximum_projection']:
    current_overlay_img = current_overlay_img.max(axis=0)

current_output_img = img_as_ubyte(current_overlay_img)

imwrite(snakemake.output[0], current_output_img)