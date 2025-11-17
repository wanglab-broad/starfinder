#!/usr/bin/env python
# coding: utf-8

import pandas as pd

grid_df = pd.read_csv(snakemake.input[0])
tile_x = grid_df['col'].max() + 1
tile_y = grid_df['row'].max() + 1    

macro_template = f"""
run("Define dataset ...", 
"define_dataset=[Automatic Loader (Bioformats based)] project_filename=dataset.xml " +
"path={snakemake.config['root_output_path']}/{snakemake.config['dataset_id']}/{snakemake.config['output_id']}/images/fused/{snakemake.wildcards.sample}/DAPI/*.tif exclude=10 pattern_0=Tiles " +
"modify_voxel_size? voxel_size_x={snakemake.config['voxel_size_xy']} voxel_size_y={snakemake.config['voxel_size_xy']} voxel_size_z={snakemake.config['voxel_size_z']} voxel_size_unit=µm " +
"move_tiles_to_grid_(per_angle)?=[Move Tile to Grid (Macro-scriptable)] grid_type=[Down & Right             ] tiles_x={tile_x} tiles_y={tile_y} tiles_z=1 overlap_x_(%)=10 overlap_y_(%)=10 overlap_z_(%)=10 " +
"keep_metadata_rotation how_to_load_images=[Re-save as multiresolution HDF5] load_raw_data_virtually " +
"dataset_save_path={snakemake.config['root_output_path']}/{snakemake.config['dataset_id']}/{snakemake.config['output_id']}/images/fused/{snakemake.wildcards['sample']}/DAPI manual_mipmap_setup " +
"subsampling_factors=[{{2,2,2}}, {{4,4,4}}] hdf5_chunk_sizes=[{{16,16,16}}, {{16,16,16}}] timepoints_per_partition=1 setups_per_partition=0 use_deflate_compression");

run("Calculate pairwise shifts ...", 
"select={snakemake.config['root_output_path']}/{snakemake.config['dataset_id']}/{snakemake.config['output_id']}/images/fused/{snakemake.wildcards['sample']}/DAPI/dataset.xml " + 
"process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] " +
"method=[Phase Correlation] downsample_in_x=4 downsample_in_y=4 downsample_in_z=4");

run("Filter pairwise shifts ...", 
"select={snakemake.config['root_output_path']}/{snakemake.config['dataset_id']}/{snakemake.config['output_id']}/images/fused/{snakemake.wildcards['sample']}/DAPI/dataset.xml " +
"filter_by_link_quality min_r=0.7 max_r=1 max_shift_in_x=0 max_shift_in_y=0 max_shift_in_z=0 max_displacement=0");

run("Optimize globally and apply shifts ...", 
"select={snakemake.config['root_output_path']}/{snakemake.config['dataset_id']}/{snakemake.config['output_id']}/images/fused/{snakemake.wildcards['sample']}/DAPI/dataset.xml " +
"process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] " + 
"relative=2.5 absolute=3.5 global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links] show_expert_grouping_options " +
"how_to_treat_timepoints=[treat individually] how_to_treat_channels=group how_to_treat_illuminations=group how_to_treat_angles=[treat individually] how_to_treat_tiles=compare fix_group_0-0");

eval("script", "System.exit(0);");
run("Quit");
"""
macro_template = macro_template.replace("//", "/")

with open(snakemake.output[0], 'w') as f:
    f.write(macro_template)



