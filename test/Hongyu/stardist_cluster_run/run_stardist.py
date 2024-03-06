#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible
from stardist.models import StarDist3D
from skimage.morphology import disk, dilation

current_id = int(sys.argv[1])
current_fov_id = f"Position{current_id:03}"

base_path = '/stanley/WangLab/Data/Analyzed/2024-02-23-Hongyu-Covid_Spleen_replicate_2/images/flamingo/'
data_path = os.path.join(base_path, 'output')
output_path = os.path.join(base_path, 'stardist_segmentation')
if not os.path.exists(output_path):
    os.mkdir(output_path)

model_path = '/stanley/WangLab/Tools/stardist_models/'
model = StarDist3D(None, name='3D_spleen', basedir=model_path)

se = disk(1, dtype=np.int32)
axis_norm = (0,1,2)
prob_thresh = 0.6
nms_thresh = 0.1

current_img = imread(os.path.join(data_path, f"{current_fov_id}.tif"))
current_img = normalize(current_img, 1, 99.8, axis=axis_norm)
labels, details = model.predict_instances(current_img, n_tiles=[1, 4, 4], prob_thresh=prob_thresh, nms_thresh=nms_thresh)

for z in range(labels.shape[0]):
    current_slice = labels[z,:,:]
    labels[z,:,:] = dilation(current_slice, se)

current_output = os.path.join(output_path, f"{current_fov_id}.tif")
save_tiff_imagej_compatible(current_output, labels, axes='ZYX')
ncells = np.unique(labels).shape[0] - 1

            