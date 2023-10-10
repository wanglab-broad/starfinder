% run test CNS_Well07
base_path = fullfile('Z:/Data/Processed/2023-10-01-Jiahao-Test/');
data_path = fullfile(base_path, 'CNS_Well07/');
output_path = fullfile('Z:/Data/Analyzed/2023-10-01-Jiahao-Test/CNS_Well07/02_pp');

current_fov = 'tile_1';
sdata = STARMapDataset(data_path, output_path);
sdata = sdata.LoadRawImages('sub_dir', current_fov);