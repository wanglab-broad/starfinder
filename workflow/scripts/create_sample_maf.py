#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

# define the path
base_path = os.path.join(snakemake.config['root_output_path'], snakemake.config['dataset_id'], snakemake.config['output_id'])
doc_path = os.path.join(base_path, 'documents')

# load sample annotation 
sample_annotation = pd.read_csv(snakemake.input[0])

# create sample dictionary 
sample_dict = {}
for current_sample in sample_annotation['sample_id'].unique():
    fov_list = []

    current_start = sample_annotation.loc[sample_annotation['sample_id'] == current_sample, 'fov_start'].values[0]
    current_end = sample_annotation.loc[sample_annotation['sample_id'] == current_sample, 'fov_end'].values[0]
    
    sample_dict[current_sample] = range(current_start, current_end+1)

# create link disctionary
link_dict = {}
maf_parsed = BeautifulSoup(open(snakemake.input[1]), 'xml')
for link in maf_parsed.find_all('XYZStagePointDefinition'):
    link_dict[link.get('PositionID')] = link

# create new maf files
for current_sample in sample_dict.keys():
    sample_range = sample_dict[current_sample]
    sample_link = []
    for i in sample_range:
        sample_link.append(str(link_dict[str(i)]))

    base_string = f"""
<!--Leica Application Suite X (LAS X)-->
<!--Leica Microsystems CMS GmbH-->
<!--http://www.confocal-microscopy.com-->
<!--LAS X 3.5.5.19976-->
<XYZStagePointDefinitionList StageOrderNumber="0">
{''.join(sample_link)}
</XYZStagePointDefinitionList>
"""
    new_maf = BeautifulSoup(base_string, 'xml')
    new_maf = new_maf.prettify()

    with open(os.path.join(doc_path, 'maf', f'{current_sample}.maf'), 'w') as file:
        file.write(new_maf)