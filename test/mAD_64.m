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

sdata = sdata.LoadRawImages('sub_dir', current_fov, 'image_slot', "seq");

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
sdata = sdata.LoadRawImages('sub_dir', current_fov, 'image_slot', "add", 'channel_order_dict', add_channel_order_dict);

% sdata = sdata.EnhanceContrast("min-max");
% sdata = sdata.HistEqualize;
% sdata = sdata.MorphRecon;
% sdata = sdata.Tophat;

% sdata = sdata.GlobalRegistration;
refernce_dapi_fname = dir(fullfile(data_path, 'round1', current_fov, '*ch04.tif'));
sdata.referenceImageAdd = LoadMultipageTiff(fullfile(refernce_dapi_fname.folder, refernce_dapi_fname.name), false);
sdata = sdata.GlobalRegistrationAdd;

% sdata = sdata.LocalRegistration;

sdata = sdata.Projection('image_slot', "add");
projection_preview_path = fullfile(output_path, 'projection_montage.tif');
sdata = sdata.ViewProjection('image_slot', "add", 'save', true, 'output_path', projection_preview_path);
sdata = sdata.SaveImages('image_slot', "add", 'output_path', output_path, 'folder_format', "single");
toc;
diary off;