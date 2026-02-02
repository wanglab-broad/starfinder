
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

