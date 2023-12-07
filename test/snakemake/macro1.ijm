input_path = "/stanley/WangLab/Data/Analyzed/2023-10-01-Jiahao-Test/mAD_64/images/protein/PI"
output_file_path = "/stanley/WangLab/Data/Analyzed/2023-10-01-Jiahao-Test/mAD_64_config/test.tif"

// input_path = "/home/unix/jiahao/wanglab/Data/Analyzed/2023-10-01-Jiahao-Test/mAD_64/images/protein/PI"
// output_file_path = "/home/unix/jiahao/wanglab/Data/Analyzed/2023-10-01-Jiahao-Test/mAD_64_config/test.tif"

run("Grid/Collection stitching", "type=[Positions from file] order=[Defined by TileConfiguration] directory=" + input_path + " layout_file=TileConfiguration.registered.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save computation time (but use more RAM)] image_output=[Fuse and display]");
// open(input_path)
saveAs("tiff", output_file_path);