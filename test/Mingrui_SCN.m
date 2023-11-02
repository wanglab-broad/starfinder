% run test Mingrui_SCN

if ispc
    base_path = fullfile('Z:/Data/Processed/2023-09-19-Mingrui-Mouse-SCN-120Gene/');
    output_path = fullfile('Z:/Data/Analyzed/2023-09-19-Mingrui-Mouse-SCN-120Gene/sample14/');
elseif isunix
    base_path = fullfile('/home/unix/jiahao/wanglab/Data/Processed/2023-09-19-Mingrui-Mouse-SCN-120Gene/');
    output_path = fullfile('/home/unix/jiahao/wanglab/Data/Analyzed/2023-09-19-Mingrui-Mouse-SCN-120Gene/sample14/');
    addpath('/home/unix/jiahao/wanglab/jiahao/Github/starfinder/code-base/new/')
end 

% Input
data_path = fullfile(base_path, 'sample14/');

current_fov = 'Position013';
useGPU = false;

tic;

sdata = STARMapDataset(data_path, output_path, 'useGPU', useGPU);
sdata.layers.ref = ["round5"];

sdata = sdata.LoadRawImages('fovID', current_fov);

% Preprocessing
sdata = sdata.EnhanceContrast("min-max");
sdata = sdata.HistEqualize;
% sdata = sdata.MorphRecon;

% % Registration
sdata = sdata.GlobalRegistration;

% Save output
sdata = sdata.SaveImages;
toc;