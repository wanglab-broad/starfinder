% run test CNS_Well07
if ispc
    base_path = fullfile('Z:/Data/Processed/2023-10-01-Jiahao-Test/');
    output_path = fullfile('Z:/Data/Analyzed/2023-10-01-Jiahao-Test/CNS_Well07/02_pp');
elseif isunix
    base_path = fullfile('~/wanglab/Data/Processed/2023-10-01-Jiahao-Test/');
    output_path = fullfile('~/wanglab/Data/Analyzed/2023-10-01-Jiahao-Test/CNS_Well07/02_pp');
    addpath('~/wanglab/jiahao/Github/starfinder/code-base/new/')
end 

% define subsample folder
data_path = fullfile(base_path, 'CNS_Well07/');
diary_file = fullfile(output_path, 'log.txt');
if exist(diary_file, 'file'); delete(diary_file); end

current_fov = 'tile_1';
useGPU = false;

diary(diary_file);
tic;
sdata = STARMapDataset(data_path, output_path, 'useGPU', useGPU);
sdata = sdata.LoadRawImages('sub_dir', current_fov);
sdata = sdata.EnhanceContrast("min-max");
sdata = sdata.HistEqualize;
sdata = sdata.MorphRecon;
sdata = sdata.Tophat;
sdata = sdata.Projection('image_slot', "raw");
sdata = sdata.GlobalRegistration;
sdata = sdata.LocalRegistration;


% projection_preview_path = fullfile(output_path, 'projection_montage.tif');
% sdata = sdata.ViewProjection('save', true, 'output_path', projection_preview_path);
% sdata = sdata.SaveImages('image_slot', "raw", 'output_path', output_path);
toc;
diary off;