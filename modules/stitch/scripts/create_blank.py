# import argparse
import numpy as np
from tifffile import imwrite
import os

# # Parse arguments
# parser = argparse.ArgumentParser(description='Create a blank 2D or 3D tile with specified pixel dimensions x y z or x y')
# parser.add_argument('output_dir', type=str, help='output directory where blank tile is saved')
# parser.add_argument('x', type=int, help='x dimension of piels')
# parser.add_argument('y', type=int, help='y dimension of pixels')
# parser.add_argument('z', nargs='?', type=int, default=None, help='z dimension of pixels')
# args = parser.parse_args()

# Create blank tile
# if args.z: # 3D
#     blank = np.zeros((args.z, args.x, args.y), dtype=np.int8)
# else:
blank = np.zeros((snakemake.config['xy'], snakemake.config['xy']), dtype=np.int8)

# Save blank tile to output directory
imwrite(snakemake.output.blankpath, blank)




