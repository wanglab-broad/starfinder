#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tifffile import imread, imwrite
from skimage.util import img_as_ubyte
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation

# load the images
current_overlay_img = imread(snakemake.input['overlay_img'])
current_segmentation = imread(snakemake.input['segmentation'])

# create label boundaries 
boundaries = find_boundaries(current_segmentation, mode='otter')
boundaries = dilation(boundaries)
boundaries = img_as_ubyte(boundaries)

# create boundaries overlay
boundaries_overlay = np.zeros(current_overlay_img.shape + (3,), dtype=np.uint8)
boundaries_overlay[..., 0] = current_overlay_img
boundaries_overlay[..., 1] = boundaries

boundaries_overlay = img_as_ubyte(boundaries_overlay)
imwrite(snakemake.output[0], boundaries_overlay, compress=6)