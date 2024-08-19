#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
from tifffile import imread, imsave
from csbdeep.utils import normalize
from csbdeep.io import save_tiff_imagej_compatible
from stardist.models import StarDist3D
from skimage.morphology import disk, dilation

model = StarDist3D(None,
                    name=snakemake.config['rules']['stardist_segmentation']['parameters']['stardist_model_name'], 
                    basedir=snakemake.config['rules']['stardist_segmentation']['parameters']['stardist_base_path'])

se = disk(1, dtype=np.int32)
axis_norm = (0,1,2)
prob_thresh = snakemake.config['rules']['stardist_segmentation']['parameters']['prob_thresh']
nms_thresh = snakemake.config['rules']['stardist_segmentation']['parameters']['nms_thresh']

current_img = imread(snakemake.input[0])
current_img = normalize(current_img, 1, 99.8, axis=axis_norm)
labels, details = model.predict_instances(current_img, n_tiles=[1, 4, 4], prob_thresh=prob_thresh, nms_thresh=nms_thresh)

for z in range(labels.shape[0]):
    current_slice = labels[z,:,:]
    labels[z,:,:] = dilation(current_slice, se)

imsave(snakemake.output[0], labels.astype('uint16'), compression='zlib')
# save_tiff_imagej_compatible(snakemake.output[0], labels, axes='ZYX')

            