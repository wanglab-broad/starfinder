% run test mAD_64
if ispc
    base_path = fullfile('Z:/Data/Processed/2023-10-01-Jiahao-Test/');
    output_path = fullfile('Z:/Data/Analyzed/2023-10-01-Jiahao-Test/mAD_64/02_pp');
elseif isunix
    base_path = fullfile('/home/unix/jiahao/wanglab/Data/Processed/2023-10-01-Jiahao-Test/');
    output_path = fullfile('/home/unix/jiahao/wanglab/Data/Analyzed/2023-10-01-Jiahao-Test/mAD_64/02_pp');
    addpath('/home/unix/jiahao/wanglab/jiahao/Github/starfinder/code-base/new/')
end 

% define subsample folder
data_path = fullfile(base_path, 'mAD_64/');

current_fov = 'tile_1';
useGPU = false;

sdata = STARMapDataset(data_path, output_path, 'useGPU', useGPU);
diary_file = fullfile(output_path, 'log.txt');
if exist(diary_file, 'file'); delete(diary_file); end
diary(diary_file);

tic;

sdata = sdata.LoadRawImages('fovID', current_fov);
sdata.layers.ref = ["round1"];

add_channel_order_dict(1).wavelength = 488;
add_channel_order_dict(1).channel = "ch00";
add_channel_order_dict(1).name = "plaque";

add_channel_order_dict(2).wavelength = 594;
add_channel_order_dict(2).channel = "ch01";
add_channel_order_dict(2).name = "tau";

add_channel_order_dict(3).wavelength = 546;
add_channel_order_dict(3).channel = "ch02";
add_channel_order_dict(3).name = "DAPI";

add_channel_order_dict(4).wavelength = 647;
add_channel_order_dict(4).channel = "ch03";
add_channel_order_dict(4).name = "Gfap";
sdata = sdata.LoadRawImages('fovID', current_fov, 'folder_list', ["protein"], ...
                            'channel_order_dict', add_channel_order_dict, 'update_layer_slot', "other");

% Preprocessing
sdata = sdata.EnhanceContrast("min-max");
sdata = sdata.EnhanceContrast("min-max", 'layer', sdata.layers.other);
sdata = sdata.HistEqualize;

% Registration
sdata = sdata.GlobalRegistration;
round1_merged_fname = fullfile(output_path, 'round1_merged.tif');
SaveSingleStack(sdata.registration{sdata.layers.ref}, round1_merged_fname);

refernce_dapi_fname = dir(fullfile(data_path, 'round1', current_fov, '*ch04.tif'));
sdata.registration{sdata.layers.ref} = LoadMultipageTiff(fullfile(refernce_dapi_fname.folder, refernce_dapi_fname.name), false);
sdata = sdata.GlobalRegistration('layer', ["protein"], 'mov_img', 'single-channel');
sdata = sdata.LocalRegistration;

% Spot finding 
sdata = sdata.SpotFinding;
sdata = sdata.ReadsExtraction;
sdata = sdata.LoadCodebook;
sdata = sdata.ReadsFiltration;

% Output 
sdata = sdata.MakeProjection;
projection_preview_path = fullfile(output_path, 'projection_montage.tif');
sdata = sdata.ViewProjection('save', true, 'output_path', projection_preview_path);
sdata = sdata.SaveImages('layer', sdata.layers.seq, 'output_path', output_path, 'folder_format', "nested");
sdata = sdata.SaveImages('layer', sdata.layers.other, 'output_path', output_path, 'folder_format', "single");
sdata = sdata.SaveSignal;

toc;
diary off;