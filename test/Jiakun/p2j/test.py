
import os
import pandas as pd
from tifffile import imread
import matplotlib.pyplot as plt
current_sample = 'Position655'
base_path = '/home/unix/jiahao/wanglab/Data/Analyzed/2024-01-08-Jiakun-MouseSpleen64Gene/'
signal_path = os.path.join(base_path, 'signal')
output_path = os.path.join(base_path, "output")
if not os.path.exists(output_path):
    os.makedirs(output_path)
print(current_sample)

# Adgre1
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Adgre1"]
n_spots = current_spots.shape[0]
print("Adgre1", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Bcl11a
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Bcl11a"]
n_spots = current_spots.shape[0]
print("Bcl11a", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Bcl11b
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Bcl11b"]
n_spots = current_spots.shape[0]
print("Bcl11b", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Bcl2
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Bcl2"]
n_spots = current_spots.shape[0]
print("Bcl2", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Ccl17
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Ccl17"]
n_spots = current_spots.shape[0]
print("Ccl17", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Ccl22
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Ccl22"]
n_spots = current_spots.shape[0]
print("Ccl22", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Ccl25
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Ccl25"]
n_spots = current_spots.shape[0]
print("Ccl25", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Ccr4
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Ccr4"]
n_spots = current_spots.shape[0]
print("Ccr4", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Ccr7
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Ccr7"]
n_spots = current_spots.shape[0]
print("Ccr7", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Cd19
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Cd19"]
n_spots = current_spots.shape[0]
print("Cd19", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Cd22
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Cd22"]
n_spots = current_spots.shape[0]
print("Cd22", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Cd247
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Cd247"]
n_spots = current_spots.shape[0]
print("Cd247", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Cd28
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Cd28"]
n_spots = current_spots.shape[0]
print("Cd28", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Cd3d
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Cd3d"]
n_spots = current_spots.shape[0]
print("Cd3d", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Cd3e
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Cd3e"]
n_spots = current_spots.shape[0]
print("Cd3e", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Cd3g
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Cd3g"]
n_spots = current_spots.shape[0]
print("Cd3g", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Cd4
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Cd4"]
n_spots = current_spots.shape[0]
print("Cd4", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Cd40
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Cd40"]
n_spots = current_spots.shape[0]
print("Cd40", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Cd44
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Cd44"]
n_spots = current_spots.shape[0]
print("Cd44", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Cd68
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Cd68"]
n_spots = current_spots.shape[0]
print("Cd68", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Cd79a
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Cd79a"]
n_spots = current_spots.shape[0]
print("Cd79a", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Cd83
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Cd83"]
n_spots = current_spots.shape[0]
print("Cd83", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Cd86
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Cd86"]
n_spots = current_spots.shape[0]
print("Cd86", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Cd8a
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Cd8a"]
n_spots = current_spots.shape[0]
print("Cd8a", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Csf1r
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Csf1r"]
n_spots = current_spots.shape[0]
print("Csf1r", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Ets1
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Ets1"]
n_spots = current_spots.shape[0]
print("Ets1", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Flt3
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Flt3"]
n_spots = current_spots.shape[0]
print("Flt3", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Foxp3
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Foxp3"]
n_spots = current_spots.shape[0]
print("Foxp3", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Gata3
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Gata3"]
n_spots = current_spots.shape[0]
print("Gata3", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Gfi1
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Gfi1"]
n_spots = current_spots.shape[0]
print("Gfi1", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# H2-Aa
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "H2-Aa"]
n_spots = current_spots.shape[0]
print("H2-Aa", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# H2-K1
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "H2-K1"]
n_spots = current_spots.shape[0]
print("H2-K1", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Hes1
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Hes1"]
n_spots = current_spots.shape[0]
print("Hes1", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Hhex
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Hhex"]
n_spots = current_spots.shape[0]
print("Hhex", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Hoxa9
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Hoxa9"]
n_spots = current_spots.shape[0]
print("Hoxa9", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Ifng
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Ifng"]
n_spots = current_spots.shape[0]
print("Ifng", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Ikzf1
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Ikzf1"]
n_spots = current_spots.shape[0]
print("Ikzf1", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Il17a
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Il17a"]
n_spots = current_spots.shape[0]
print("Il17a", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Il2ra
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Il2ra"]
n_spots = current_spots.shape[0]
print("Il2ra", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Il4
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Il4"]
n_spots = current_spots.shape[0]
print("Il4", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Il7r
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Il7r"]
n_spots = current_spots.shape[0]
print("Il7r", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Irf4
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Irf4"]
n_spots = current_spots.shape[0]
print("Irf4", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Irf8
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Irf8"]
n_spots = current_spots.shape[0]
print("Irf8", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Itgam
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Itgam"]
n_spots = current_spots.shape[0]
print("Itgam", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Itgax
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Itgax"]
n_spots = current_spots.shape[0]
print("Itgax", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Kit
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Kit"]
n_spots = current_spots.shape[0]
print("Kit", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Lef1
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Lef1"]
n_spots = current_spots.shape[0]
print("Lef1", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Lmo2
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Lmo2"]
n_spots = current_spots.shape[0]
print("Lmo2", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Lyl1
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Lyl1"]
n_spots = current_spots.shape[0]
print("Lyl1", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Lyz2
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Lyz2"]
n_spots = current_spots.shape[0]
print("Lyz2", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Mef2c
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Mef2c"]
n_spots = current_spots.shape[0]
print("Mef2c", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Meis1
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Meis1"]
n_spots = current_spots.shape[0]
print("Meis1", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Ms4a1
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Ms4a1"]
n_spots = current_spots.shape[0]
print("Ms4a1", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Pdcd1
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Pdcd1"]
n_spots = current_spots.shape[0]
print("Pdcd1", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Rag1
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Rag1"]
n_spots = current_spots.shape[0]
print("Rag1", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Rag2
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Rag2"]
n_spots = current_spots.shape[0]
print("Rag2", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Runx1
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Runx1"]
n_spots = current_spots.shape[0]
print("Runx1", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Sell
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Sell"]
n_spots = current_spots.shape[0]
print("Sell", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Sirpa
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Sirpa"]
n_spots = current_spots.shape[0]
print("Sirpa", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Tbx21
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Tbx21"]
n_spots = current_spots.shape[0]
print("Tbx21", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Tcf12
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Tcf12"]
n_spots = current_spots.shape[0]
print("Tcf12", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Tcf3
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Tcf3"]
n_spots = current_spots.shape[0]
print("Tcf3", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Tcf7
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Tcf7"]
n_spots = current_spots.shape[0]
print("Tcf7", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        

# Xcr1
current_dapi_img = imread(os.path.join(base_path, "images/flamingo/DAPI_MAX", "Position655.tif"))
current_signal = pd.read_csv(os.path.join(signal_path, current_sample + '_goodSpots.csv'))
current_signal['x'] = current_signal['x'] - 1
current_signal['y'] = current_signal['y'] - 1
current_signal['z'] = current_signal['z'] - 1
current_spots = current_signal[current_signal['gene'] == "Xcr1"]
n_spots = current_spots.shape[0]
print("Xcr1", n_spots)
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(current_dapi_img, cmap='gray')
ax.plot(current_spots['x'], current_spots['y'], 'r.', markersize=3)
plt.axis('off')
plt.tight_layout()
plt.show()
        
