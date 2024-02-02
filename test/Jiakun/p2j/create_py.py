import os 
import pandas as pd 

base_path = '/home/unix/jiahao/wanglab/Data/Analyzed/2024-01-08-Jiakun-MouseSpleen64Gene/'
signal_path = os.path.join(base_path, 'signal')
current_sample = 'Position655'
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
all_genes = sorted(current_signal.gene.unique())

library_block = f"""
import os
import pandas as pd
from tifffile import imread
import matplotlib.pyplot as plt
current_sample = '{current_sample}'
base_path = '/home/unix/jiahao/wanglab/Data/Analyzed/2024-01-08-Jiakun-MouseSpleen64Gene/'
signal_path = os.path.join(base_path, 'signal')
output_path = os.path.join(base_path, "output")
if not os.path.exists(output_path):
    os.makedirs(output_path)
print(current_sample)
"""

with open("test/Jiakun/p2j/test.py", "w") as f:
    f.write(library_block)

    for gene in all_genes:
        f.write(f"\n# {gene}")
        code_block = f"""
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "{current_sample}.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "{gene}"]
n_spots = current_spots.shape[0]
print("{gene}", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        """
        f.write(code_block)
        f.write(f"\n")