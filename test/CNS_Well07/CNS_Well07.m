% run test CNS_Well07
if ispc
    base_path = fullfile('Z:/Data/Processed/2023-10-01-Jiahao-Test/');
    output_path = fullfile('Z:/Data/Analyzed/2023-10-01-Jiahao-Test/CNS_Well07/02_pp');
elseif isunix
    base_path = fullfile('/home/unix/jiahao/wanglab/Data/Processed/2023-10-01-Jiahao-Test/');
    output_path = fullfile('/home/unix/jiahao/wanglab/Data/Analyzed/2023-10-01-Jiahao-Test/CNS_Well07/02_pp');
    addpath('/home/unix/jiahao/wanglab/jiahao/Github/starfinder/code-base/new/')
end 

% Input
data_path = fullfile(base_path, 'CNS_Well07/');

current_fov = 'tile_1';
useGPU = false;

tic;

sdata = STARMapDataset(data_path, output_path, 'useGPU', useGPU);
sdata.layers.ref = ["round1"];

diary_file = fullfile(output_path, 'log.txt');
if exist(diary_file, 'file'); delete(diary_file); end
diary(diary_file);

sdata = sdata.LoadRawImages('fovID', current_fov);

% Preprocessing
sdata = sdata.EnhanceContrast("min-max");
sdata = sdata.HistEqualize;
sdata = sdata.MorphRecon;

% Registration
sdata = sdata.GlobalRegistration;
sdata = sdata.LocalRegistration;

% Read calling
sdata = sdata.SpotFinding;
sdata = sdata.ReadsExtraction;
sdata = sdata.LoadCodebook;
sdata = sdata.ReadsFiltration;

% Save output
sdata = sdata.SaveSignal;
sdata = sdata.Projection('image_slot', "raw");
projection_preview_path = fullfile(output_path, 'projection_montage.tif');
sdata = sdata.ViewProjection('save', true, 'output_path', projection_preview_path);
sdata = sdata.SaveImages('image_slot', "raw", 'output_path', output_path);
toc;
diary off;