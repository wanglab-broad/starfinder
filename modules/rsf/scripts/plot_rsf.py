"""
    For QC purposes, create two plots:
        1) final spot-finding results on all round1 signal (max proj across z-stacks)
        2) final spot-finding results on DAPI signal (max proj across z-stacks)
"""
import os
from tifffile import imread
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from glob import glob

# Load goodReads
points = pd.read_csv(snakemake.input.goodReads)

# Load DAPI image 2d max proj
dapi_path = glob(os.path.join(snakemake.input.dapi, "*ch04.tif"))[0]    
dapi = imread(dapi_path)  
dapi = np.max(dapi, axis=0)

# Load r1max 2d max proj
r1max = imread(snakemake.input.r1max)
r1max = np.max(r1max, axis=0)

# Create and save reads on DAPI plot
plt.figure(figsize=(15,15))
plt.imshow(dapi)
plt.plot(points.x, points.y, '.', color='red', markersize=1)
plt.axis('off')
plt.savefig(snakemake.output.goodPoints_dapi)
plt.close() 

# Create and save reads on r1max plot
plt.figure(figsize=(15,15))
plt.imshow(r1max)
plt.plot(points.x, points.y, '.', color='red', markersize=1)
plt.axis('off')
plt.savefig(snakemake.output.goodPoints_r1max)
plt.close() 
