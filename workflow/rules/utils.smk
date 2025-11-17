""" 
    This snakemake file includes Google Cloud backup workflows
"""

### ==================== [ Basic setting ] =========================

import os 
import glob
import random
import pandas as pd
from pathlib import Path

### ==================== [ Helper functions ] =========================



### ==================== [ Parameters ] =========================
INPUT_DIR = Path(config['root_input_path'], config['dataset_id'], config['sample_id'])
OUTPUT_DIR = Path(config['root_output_path'], config['dataset_id'], config['output_id'])
DOC_DIR = Path(OUTPUT_DIR, 'documents')

SAMPLE_ANNOT = pd.read_csv(f"{OUTPUT_DIR}/documents/sample-annotation.csv")
SAMPLE_DICT = {}
for current_sample in SAMPLE_ANNOT['sample_id'].unique():
    fov_list = []

    current_start = SAMPLE_ANNOT.loc[SAMPLE_ANNOT['sample_id'] == current_sample, 'fov_start'].values[0]
    current_end = SAMPLE_ANNOT.loc[SAMPLE_ANNOT['sample_id'] == current_sample, 'fov_end'].values[0]

    for i in range(current_start, current_end+1):
        current_fov = config['fov_id_pattern'].format(i=i)
        fov_list.append(current_fov)
    
    SAMPLE_DICT[current_sample] = fov_list

SAMPLE = list(SAMPLE_DICT.keys())

### ==================== [ Rules ] =========================

rule create_sample_maf:
    input:
        expand("{output_dir}/sample-annotation.csv", output_dir=DOC_DIR),
        expand("{output_dir}/raw.maf", output_dir=DOC_DIR),
    output:
        expand("{output_dir}/maf/{sample}.maf", output_dir=DOC_DIR, sample=SAMPLE),
    resources:
        mem_mb=8000,
        runtime=2
    script:
        "../scripts/create_sample_maf.py"

# rule create_segmentation_preview:
#     input:
#         overlay_img=expand("{output_dir}/images/overlay/{{fovID}}.tif", output_dir=OUTPUT_DIR),
#         segmentation=expand("{output_dir}/images/stardist_segmentation/{{fovID}}.tif", output_dir=OUTPUT_DIR)
#     output:
#         expand("{output_dir}/images/segmentation_preview/{{fovID}}.tif", output_dir=OUTPUT_DIR)
#     resources:
#         mem_mb=config['rules']['create_segmentation_preview']['resources']['mem_mb']
#     script:
#         "scripts/create_segmentation_preview.py"