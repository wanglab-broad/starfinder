% IO test
base_path = fullfile('Z:/Data/Processed/2023-10-01-Jiahao-Test/');
data_path = fullfile(base_path, 'MR_BA/');
output_path = fullfile('Z:/Data/Analyzed/2023-10-01-Jiahao-Test/MR_BA/02_pp');


current_fov = 'tile_0';
useGPU = false;
sdata = STARMapDataset(data_path, output_path, 'useGPU', useGPU);

channel_order_dict(1).wavelength = 488;
channel_order_dict(1).channel = "ch00";
channel_order_dict(1).name = "ch00";

channel_order_dict(2).wavelength = 594;
channel_order_dict(2).channel = "ch01";
channel_order_dict(2).name = "ch01";

channel_order_dict(3).wavelength = 546;
channel_order_dict(3).channel = "ch02";
channel_order_dict(3).name = "ch02";

channel_order_dict(4).wavelength = 647;
channel_order_dict(4).channel = "ch03";
channel_order_dict(4).name = "ch03";

sdata = sdata.LoadRawImages('sub_dir', current_fov, 'channel_order_dict', channel_order_dict);