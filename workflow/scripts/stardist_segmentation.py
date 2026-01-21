#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tifffile import imread, imsave
from csbdeep.utils import normalize
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from stardist.models import StarDist2D, StarDist3D
from skimage.transform import rescale
from skimage.segmentation import expand_labels

prob_thresh = snakemake.config['rules']['stardist_segmentation']['parameters']['prob_thresh']
nms_thresh = snakemake.config['rules']['stardist_segmentation']['parameters']['nms_thresh']

current_img = imread(snakemake.input[0])
threshold = threshold_otsu(current_img)
bw_img = current_img > threshold
test_img = label(bw_img)
props = regionprops(test_img)
areas = np.array([prop.area for prop in props])

if areas.max() > 100:
    if len(current_img.shape) == 3:
        model = StarDist3D(None,
                            name=snakemake.config['rules']['stardist_segmentation']['parameters']['stardist_model_name'], 
                            basedir=snakemake.config['rules']['stardist_segmentation']['parameters']['stardist_base_path'])

        axis_norm = (0,1,2)

        if snakemake.config['rules']['stardist_segmentation']['parameters']['rescale']:
            current_img = rescale(current_img, [1, .5, .5])
            current_img = normalize(current_img, 1, 99.8, axis=axis_norm)
            labels, details = model.predict_instances(current_img, n_tiles=[1, 4, 4], prob_thresh=prob_thresh, nms_thresh=nms_thresh)
            current_label = rescale(labels, [1, 2, 2], order=0, preserve_range=True)
        else:
            current_img = normalize(current_img, 1, 99.8, axis=axis_norm)
            current_label, details = model.predict_instances(current_img, n_tiles=[1, 4, 4], prob_thresh=prob_thresh, nms_thresh=nms_thresh)

        if snakemake.config['rules']['stardist_segmentation']['parameters']['expand_labels']:
            for z in range(current_label.shape[0]):
                current_label[z,:,:] = expand_labels(current_label[z,:,:], distance=snakemake.config['rules']['stardist_segmentation']['parameters']['distance'])

    else:
        model = StarDist2D(None,
                            name=snakemake.config['rules']['stardist_segmentation']['parameters']['stardist_model_name'], 
                            basedir=snakemake.config['rules']['stardist_segmentation']['parameters']['stardist_base_path'])

        axis_norm = (0,1)

        if snakemake.config['rules']['stardist_segmentation']['parameters']['rescale']:
            current_img = rescale(current_img, [.5, .5])
            current_img = normalize(current_img, 1, 99.8, axis=axis_norm)
            labels, details = model.predict_instances(current_img, n_tiles=[2, 2], prob_thresh=prob_thresh, nms_thresh=nms_thresh)
            current_label = rescale(labels, [2, 2], order=0, preserve_range=True)
        else:
            current_img = normalize(current_img, 1, 99.8, axis=axis_norm)
            current_label, details = model.predict_instances(current_img, n_tiles=[2, 2], prob_thresh=prob_thresh, nms_thresh=nms_thresh)

        if snakemake.config['rules']['stardist_segmentation']['parameters']['expand_labels']:
            current_label = expand_labels(current_label, distance=snakemake.config['rules']['stardist_segmentation']['parameters']['distance'])

    imsave(snakemake.output[0], current_label.astype('uint16'), compression='zlib')
else:
    current_label = np.zeros(current_img.shape, dtype='uint16')
    imsave(snakemake.output[0], current_label.astype('uint16'), compression='zlib')
    
            