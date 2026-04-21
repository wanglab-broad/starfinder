# Temporary prompt history 
Note: this is not a TODO list.

--- 

2026-03-30 
paused cpd registraion 
start snaakeamke transtion 

Can you help make the following modifications?
1. Instead of a prefix, use a suffix in the smk files, for example, registration-py.smk
2. for the scripts, drop the prefix, just use "xxx.py"
3. for the Common Python Helper, use Snakemake `script:` directive instead of subprocess
4. use sdata as a variable name instead of ds, for example, sdata = STARMapDataset.from_config(snakemake.config)
5. also enable steaming mode in gr_single_fov_subtile

interpolation artifact issues
subsample region for fft
when finish a plan, mark finished 
iterative CPD?

I also want:
1. some randomness of the gaussian spot generation (i,e, slightly variant intensity and size across rounds for the same spot)
2. for step 6, modify the network-mount scripts based on our new implementation, do not need to keep the old import path 
3. Remove old unnecessary tests if possible 
The ultimate goal of this iteration for the synthetic data generation process is to:
1. unify these two systems
  - the primary synthetic data generation codebase should be starfinder.benchmark.data but its primary function is to generate a multi-round, multi-channel dataset for E2E tests just like starfinder.testdata. The registration benchmark data generation is a single round, single channel special case of the primary use. Thus, move starfinder.testdata to starfinder.benchmark and unify these two. 
  - make the size presets consistent, including tiny, small, medium, large, tissue, thick_medium 
2. re-design the generation of images with local deformation 
  - the previous workflow directly applies the deformation on the image which could cause un-natural blank / noise region on the image, I suggest the following workflow: modify the initial coordinates of the synthetic spots based on the local deformation, then generate spots on the new locations after deformation

Please create an implementation plan according to my input. 


local/block CPD?
diffusion-based method to match two images?
find a way to not change spot morphology 

can you help modify the synthetic data generation for our e2e_LR test:
1. Replace linear_small with d = c0 + c1*x + c2*y + c3*z + c4*x*y
2. Deformation caps at 10px instead of 5px 
what do you think?


for the maximum shifs along xy, make it 50 for all datasets, max z shift 10 for thick_medium
some questions:
1. for registration, spotfinding, and extraction, you can do one-round-at-a-time, but what about the filtering process how do you plan to handle that? do you need to save any intermidiate results?
2. I think the MIP-based FFT not making sense, the z-shift is very critical to make sure correct color extraction 
3. I think Parallelization will be handled via Snakemake later, right?

for the e2e benchmark script using real data, modify it so that:
1. refer to the most updated script for synthetic data benchmark, modify file saving locations
2. the qc metrics matches the setting when using synthetic data, since we don't have the groud truth, just simply remove those ones. Also, to achieve a better consistentcy, don't include the matlab comparison metrics in the log/{fov}.csv qc files. 
3. keep the following metrics:
  - gene_coverage
  - mean_color_score
4. save the matlab comparison metrics in log/matlab_comparison/{fov}.csv
  - include n_all_spots, n_good_spots, codebook_match_rate of two beckends



for the e2e benchmark script using synthetic data, modify it so that:
1. save the registration inspection image under log/gr_inspect folder
2. save the signal inspection image under log/signal_inspect folder
3. change the {fov}_qc.csv to {fov}.csv
4. rename the following qc metrics (old, new):
  - spot_recall, detection_recall
  - spot_precision, detection_precision
  - n_correct_form_CNNNNC, n_correct_form
5. remove the following qc metrics:
  - spot_mean_distance_px



can we make it consistent?
1. for mini, match it with the small dataset size in the registration benchmark 
2. for standard,  match it with the medium dataset size in the registration benchmark
3. use the same naming scheme, replace mini with small, standard with medium 

Here are some additional context:
1. for the tissue-2D and cell-culture-3D datasets, the fifth channel is the DAPI staining for nuclei, for decoding benchmark you only need the first 4 channels
2. the max projection of tissue-2D is just for visualization and downstream segmentation 


what about SNR-gated normalization + round_max spot finding threshold + md=2

Yeah, the real issue is per-channel min_max_normalize inflates noise in channels with no signal. what if we calculate SNR for each channel, if it is too low, we skip the normalization for that channel? Does this make sense 


Still many false positive spots. Create a plan to change spot finding implementation in python so that it matches MATLAB's imregionalmax + regionprops3 workflow. Save the plan to docs/plans


Please consider the following comments and revise the v2 plan:
1. Don't need to differentiate the dataset in the validation test, make it one e2e fixture with a test config as input. Use the mini dataset as the default. 
2. No need to test multi-fov behavior using pytest

Please consider the following comments and revise the v2 plan:
1. Save the e2e validation results and intermediate files into the starfinder_benchmark/results/e2e_validation folder for my visual inspection 
2. Create synthetic datasets in starfinder_benchmark/data/synthetic for e2e validation, use size medium as the default  
3. don't need to differentiate the dataset in the validation test, make it one e2e fixture with a test config as input.


Please consider the following comments and revise the design:
1. LayerState should be part of the dataset level metadata (STARmapDataset.layers) and inherit by FOV class (FOV.layers), current validation is good 
2. RegistrationResult should be a FOV level results (FOV.registration). global_shifts can be simplified as a dict where keys are round names and values are shift in that round. Also fix the name collision
3. Codebook should be part of the dataset level metadata (STARmapDataset.codebook) and inherit by FOV class (FOV.codebook)
4. CropWindow should be renamed as Subtile and as part of the dataset level attribute (STARmapDataset.subtile). Subtile should include number of subtile, subtile_id and corresponding subtile window
5. Fix the Logging Strategy and use it in key processing steps such as preprocessing, registration, and spot finding 
6. Fix #7, #5, #2, #4, #10, #12 and minor issues on your list 
7. Do a documentation reorg / clean up for the main object design


I have the following comments:
1. I agree that the there are so many dataclasses, maybe we can simplify some of them
2. For the codebook class, I suggest we keep it 
3. For the registration, there might be multi-step registration happening in production (e.g. global first, then local), so the dataclass need to handle that situation. However, the most important results to keep is the global shift

Please revise the plan again based on my comments 



SSIM on 2D MIP if image is too large. 

let's implement the MIP-based spot detection and do a test run to validate the performance gain. 

I think the python_matlab one works great, what is the config setting of that one? 
can we plan another benchmark test to compare different setting of python_matlab and the matlab orignal, with large synthetic data and two smaller real datasets?
1. same iteration setting for both python_matlab and matlab original
2. record time and memeory usage for each test run 
3. for python_matlab, try method=demons and method=diffeomorphic
4. always use pyramid_mode="antialias"

Save results in starfinder_benchmark/registration/local_comparison folder 



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

