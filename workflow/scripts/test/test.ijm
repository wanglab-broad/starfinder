run("Define dataset ...", 
"define_dataset=[Automatic Loader (Bioformats based)] project_filename=dataset.xml " + 
"path=Z:/Data/Analyzed/2024-03-08-Hongyu-Covid_LN/starfinder-test/images/fused/sample1/DAPI exclude=10 pattern_0=Tiles " +
"modify_voxel_size? voxel_size_x=0.19 voxel_size_y=0.19 voxel_size_z=0.35 voxel_size_unit=µm " +
"move_tiles_to_grid_(per_angle)?=[Move Tile to Grid (Macro-scriptable)] grid_type=[Down & Right             ] tiles_x=8 tiles_y=9 tiles_z=1 overlap_x_(%)=10 overlap_y_(%)=10 overlap_z_(%)=10 " +
"keep_metadata_rotation how_to_load_images=[Re-save as multiresolution HDF5] load_raw_data_virtually " +
"dataset_save_path=Z:/Data/Analyzed/2024-03-08-Hongyu-Covid_LN/starfinder-test/images/fused/sample1/DAPI " +
"subsampling_factors=[{ {1,1,1}, {2,2,2}, {4,4,4}, {8,8,8} }] hdf5_chunk_sizes=[{ {16,16,16}, {16,16,16}, {16,16,16}, {32,16,8} }] timepoints_per_partition=1 setups_per_partition=0 use_deflate_compression");

run("Calculate pairwise shifts ...", 
"select=Z:/Data/Analyzed/2024-03-08-Hongyu-Covid_LN/starfinder-test/images/fused/sample1/DAPI/dataset.xml " + 
"process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] " +
"method=[Phase Correlation] downsample_in_x=4 downsample_in_y=4 downsample_in_z=4");

run("Filter pairwise shifts ...", 
"select=Z:/Data/Analyzed/2024-03-08-Hongyu-Covid_LN/starfinder-test/images/fused/sample1/DAPI/dataset.xml " +
"filter_by_link_quality min_r=0.7 max_r=1 max_shift_in_x=0 max_shift_in_y=0 max_shift_in_z=0 max_displacement=0");

run("Optimize globally and apply shifts ...", 
"select=Z:/Data/Analyzed/2024-03-08-Hongyu-Covid_LN/starfinder-test/images/fused/sample1/DAPI/dataset.xml " +
"process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] " + 
"relative=5 absolute=7 global_optimization_strategy=[Two-Round using metadata to align unconnected Tiles] show_expert_grouping_options " +
"how_to_treat_timepoints=[treat individually] how_to_treat_channels=group how_to_treat_illuminations=group how_to_treat_angles=[treat individually] how_to_treat_tiles=compare fix_group_0-0");

