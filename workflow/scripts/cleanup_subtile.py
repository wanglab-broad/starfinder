#!/usr/bin/env python
# coding: utf-8

import os
current_fov_id = snakemake.wildcards.fovID

# set path
base_path = os.path.join(snakemake.config['root_output_path'], snakemake.config['dataset_id'], snakemake.config['output_id'])
image_path = os.path.join(base_path, 'images')
signal_path = os.path.join(base_path, 'signal')
output_path = os.path.join(base_path, 'output')
current_subtile_path = os.path.join(output_path, 'subtile', current_fov_id)

# remove subtile folder
if os.path.exists(current_subtile_path):
    os.system('rm -r ' + current_subtile_path)



