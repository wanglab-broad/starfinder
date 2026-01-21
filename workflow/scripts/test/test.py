def ImageJ_stitching(prefix_path, output_path,grid_size_x, grid_size_y, strategy_t = 'column-by-column', order_t = "Down & Right", saveplot = 'Write to disk', macro_content = None):
    import subprocess, os

    # Define the ImageJ path and the macro you want to run
    imagej_path = f"{prefix_path}/tangzefang/04.tools/Fiji.app/ImageJ-linux64"  # You need to update this to the actual path to your ImageJ executable
    macro_filename = f"stitching.ijm"
    macro_path = os.path.join(output_path, macro_filename)
    # Generate the ImageJ macro script
    if macro_content is None:
        macro_content = """
            run("Grid/Collection stitching", "type=[Grid: {}] order=[{}] grid_size_x={} grid_size_y={} tile_overlap=10 first_file_index_i=1 directory={} file_names=tile_{}.tif output_textfile_name=TileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 compute_overlap computation_parameters=[Save computation time (but use more RAM)] image_output=[{}]");
        """.format(strategy_t,order_t,grid_size_x,grid_size_y, output_path,'{i}',saveplot)

    with open(macro_path, 'w') as macro_file:
        macro_file.write(macro_content)

    # Execute the ImageJ command using subprocess
    command = [imagej_path, '--headless', '-macro', macro_path]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(stdout, stderr)