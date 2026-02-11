# Temporary prompt history 
Note: this is not a TODO list.

--- 
SSIM on 2D MIP if image is too large. 

let's implement the MIP-based spot detection and do a test run to validate the performance gain. 


Issue: synthetic dataset design is not relecting the real data
Things to fix:
1. make images in uint8 by default 
2. The background noise is so prominent that I cannot visually identify the spot location, try generate a test image set without noise
3. generate the new test image dataset in ~/wanglab/jiahao/test/<test_set_id>, so that I can inspect it interactively


For the ground truth, can you also generate an image with annotation:
1. Viualize all spots on a 2D maximum projection 
2. Add a bounding box annotation to each spot with the gene and color sequence annotated  


nice work, here are some modifications needed:
1. remove the legend on the ground_truth_annotation.png
2. generate the ground_truth_annotation.png in the same folder level as the ground_truth.json
3. make the generation of the ground_truth_annotation.png default behavior when creating a testing dataset

1. load_multipage_tiff should return unit8 by default 
2. what are the other options for loaing tiff file effectively？ what about OME-TIFF format


I noticed that some of the points disappeared after create_local_deformation, do you know why 

Based on your understanding of this project, revise the following plan and save it to docs folder:

I plan to conduct a systematic benchmark of the registration module, following this outline:
**Task 1: Data Preparation**
- Create pairs of 3D reference and moving images (single images with ZYX dimensions). For each composite images, generate a maximum projection across channels (ZYXC → ZYX).
- Reference images:
  - Synthetic datasets with varying sizes (to be determined).
  - Real datasets from test sets: tissue-2D, cell-culture-3D, and LN.
    - Use four channels in the first field of view during initial testing.
- Moving images:
  - **Global registration set:** Apply known global shifts to reference images; record ground truth shifts.
  - **Local registration set:** Apply local deformations to reference images; record ground truth deformation fields.

**Task 2: Performance Benchmarking**
- Evaluate both global and local registration methods using different backends:
  - All available Python implementations
  - Original MATLAB version
- Record execution time and memory usage for each test case.

**Task 3: Reporting**
- Compile results into comprehensive reports featuring tables and visualizations for performance comparison.


one issue found in the global shift application: there is no shift applied to the Z-axis currently, can you add the Z-axis shift and regenerate affected files. 

Okay, everything looks good now, let's mark Task 1 in the plan as completed and also the checkpoint. FOr the last step of the checkpoint, since the datasets are too large, let's just skip the copy and keep the data in the drive. 

Now, Modify the plan:
1. during Task 2, each benchmark run should save the registered images and create inpsection.png comparing the states before and after registration
2. add another checkpoint after Task 2 for visual inspection of the registered imamges


Here are my preferences:
1. I agree that we should simplify the Parameter Tuning task, but I want to test the multi-pyramid strategy on real data since it was used in the original MATLAB implementation.
2. Always save inspection images (small, ~100 KB each). Save registered volumes only for failed cases and for the best/worst results per preset.
3. Running benchmarks in size order with early stopping is acceptable.
4. The MATLAB comparison is a necessary step and can proceed once the Python version is stable.
5. For registration_runner.py, try extending the original runner module from the benchmark to keep the codebase neat and consistent.

Please modify the plan accordingly

I have couple questions:
1. what is the values showed in the third and the forth panel in the inpsection png, what does the mean reperents
2. for the saved shift.json, when I compared the results from two different backends, only the method, time, and memory were having differences, but other qc metrics are exactly the same, is this true? 


Please review the registration benchmark plan and proceed to Task 3. Key points to address include:
1. Summarize the overall benchmark workflow.
2. Describe how our testing datasets are constructed and designed, covering:
   - Synthetic and real datasets
   - Voxel size scaling and ranking
   - Generation of global registration shifts
   - Creation of local registration deformations
3. Outline the benchmark design, including:
   - Metrics selection
   - Workflow for global registration benchmarking
     - Available backend options (e.g., NumPy vs. scikit-image, NumPy vs. MATLAB)
     - Considerations regarding metrics, time, and memory usage
   - Workflow for local registration benchmarking
     - Available backend options (e.g., Python vs. MATLAB)
     - Reasons for poor performance using Python
     - Performance with real datasets—where it partially succeeds
     - Comparing Python versus MATLAB implementations concerning time and memory efficiency

Additionally, suggest potential future improvements based on these insights.

Since I need to include these benchmark in a lab meeting, can you help create the following visualization:
General visualization:
1. line plot showing the number of voxels of the datasets, including synthetic and real, also the thick_large
2. create visualizations showing the five deformation type generated for synthetic data, two panels each type (left: 2D deformation MIP, right: 2D MIP of the mov image)
3. visualization explaining the four benchmark metrics (MCC, SSIM, Spot IoU, and Match Rate) if possbile 

For global registration:
1. line plot showing the speed of global registration for 3 backends (numpy_fft, skiamge, DFT) using synthetic data, where x-axis is the dataset size and y-axis is time usage
2. line plot showing the memory of global registration for 3 backends (numpy_fft, skiamge, DFT) using synthetic data, where x-axis is the dataset size and y-axis is memory usage

For local registration:
1. line plot showing the speed of local registration for 2 backends (python vs. matlab) using synthetic data, where x-axis is the dataset size and y-axis is time usage
2. line plot showing the memory of local registration for 2 backends (python vs. matlab) using synthetic data, where x-axis is the dataset size and y-axis is memory usage

Save figures in `/home/unix/jiahao/wanglab/jiahao/test/registration_benchmark/results/figures`

To better organize the benchmark results, I reorgnized the folders and did the following modifications:

1. change the folder name to starfinder_benchmark, reason: make it the default output folder for the future benchmark tasks
2. deleted overview.png, reason: panels are too small, not readable
3. created a data folder and moved synthetic and real dataset folder into it
4. moved all registration results under starfinder_benchmark/results/registration
5. created starfinder_benchmark/results/registration/scripts folder to store all MATLAB/Python comparison scripts
6. changed foler global/ to global_python/, this is a good example folder, results grouped by dataset and every test has an inspection image
7. changed folder matlab_global/ to global_matlab/
8. changed folder tuning/ to local_tuning/
9. changed folder matlab/ to local_matlab/
10. created folder local_python for future local registration benchmark w/ python backend

Please rescan the starfinder_benchmark folder and update notes.md and CLAUDE.md accordingly. 

