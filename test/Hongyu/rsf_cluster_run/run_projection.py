#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
from tifffile import imread, imwrite
from skimage.util import img_as_ubyte

current_id = int(sys.argv[1])
current_fov_id = f"Position{current_id:03}"
print(current_fov_id)

# set path
base_path = '/stanley/WangLab/Data/Analyzed/2024-02-23-Hongyu-Covid_Spleen_replicate_2/'
input_path = os.path.join(base_path, 'images/flamingo/DAPI')
output_path = os.path.join(base_path, 'images/flamingo/DAPI_MAX')

current_img = imread(os.path.join(input_path, f"{current_fov_id}.tif"))
current_max_img = current_img.sum(axis=0)
current_max_img = current_max_img / current_max_img.max()
current_max_img = img_as_ubyte(current_max_img)
imwrite(os.path.join(output_path, f"{current_fov_id}.tif"), current_max_img)