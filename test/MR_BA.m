% IO test
if ispc
    base_path = fullfile('Z:/Data/Processed/2023-10-01-Jiahao-Test/');
    output_path = fullfile('Z:/Data/Analyzed/2023-10-01-Jiahao-Test/MR_BA/02_pp');
elseif isunix
    base_path = fullfile('/home/unix/jiahao/wanglab/Data/Processed/2023-10-01-Jiahao-Test/');
    output_path = fullfile('/home/unix/jiahao/wanglab/Data/Analyzed/2023-10-01-Jiahao-Test/MR_BA/02_pp');
    addpath('/home/unix/jiahao/wanglab/jiahao/Github/starfinder/code-base/new/')
end 

data_path = fullfile(base_path, 'MR_BA/');


current_fov = 'tile_0';
useGPU = false;

channel_order_dict(1).wavelength = 488;
channel_order_dict(1).channel = "Ch1";
channel_order_dict(1).name = "seq";

channel_order_dict(2).wavelength = 594;
channel_order_dict(2).channel = "Ch2";
channel_order_dict(2).name = "seq";

channel_order_dict(3).wavelength = 546;
channel_order_dict(3).channel = "Ch3";
channel_order_dict(3).name = "seq";

channel_order_dict(4).wavelength = 647;
channel_order_dict(4).channel = "Ch5";
channel_order_dict(4).name = "seq";


% tic;

sdata = STARMapDataset(data_path, output_path, 'useGPU', useGPU);

% diary_file = fullfile(output_path, 'log.txt');
% if exist(diary_file, 'file'); delete(diary_file); end
% diary(diary_file);

sdata = sdata.LoadRawImages('fovID', current_fov, 'channel_order_dict', channel_order_dict);
sdata = sdata.EnhanceContrast("min-max");
sdata = sdata.EnhanceContrast("imadjustn");

% sdata = sdata.HistEqualize;
% sdata = sdata.MorphRecon;
% sdata = sdata.Tophat;
% sdata = sdata.Projection('image_slot', "raw");
% sdata = sdata.GlobalRegistration;
% sdata = sdata.LocalRegistration;


% projection_preview_path = fullfile(output_path, 'projection_montage.tif');
% sdata = sdata.ViewProjection('save', true, 'output_path', projection_preview_path);
% sdata = sdata.SaveImages('image_slot', "raw", 'output_path', output_path);
% toc;
% diary off;