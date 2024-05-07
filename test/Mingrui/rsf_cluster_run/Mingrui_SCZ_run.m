% run registration and spot finding workflow with 2D mouse tissue section 
% user will define:
% input_path 
% output_path
% ref_round
% fov_id_pattern
% number_of_fovs

% test block
input_path = fullfile('/stanley/WangLab/Data/Processed/2024-03-12-Mingrui-PFC/');
output_path = fullfile('/stanley/WangLab/Data/Analyzed/2024-03-22-Mingrui-PFC-test/');
ref_round = ["round1"];

% add path for .m files
addpath('/stanley/WangLab/jiahao/Github/starfinder/code-base/new/') % pwd is the location of the starfinder folder

sdata = STARMapDataset(input_path, output_path, 'useGPU', false);

log_folder = fullfile(output_path, "log");
if ~exist(log_folder, 'dir')
    mkdir(log_folder);
end
diary_file = fullfile(log_folder, sprintf("%s.txt", current_fov));
if exist(diary_file, 'file'); delete(diary_file); end
diary(diary_file);

starting = tic;

sdata = sdata.LoadRawImages('fovID', current_fov, 'rotate_angle', -90);
sdata.layers.ref = ref_round;

i = sscanf(current_fov,'Position%d');

if i > 435
    sdata.images{"round4"} = imrotate(sdata.images{"round4"}, 0.75, 'bilinear', 'crop');
end

% Preprocessing
sdata = sdata.EnhanceContrast("min-max");
sdata = sdata.HistEqualize;
sdata = sdata.MorphRecon('radius', 6);

% Registration
sdata = sdata.GlobalRegistration;
% sdata = sdata.LocalRegistration;

% Save round 1 ref image
ref_merged_folder = fullfile(output_path, "images", "ref_merged_round1");
if ~exist(ref_merged_folder, 'dir')
    mkdir(ref_merged_folder);
end
ref_merged_fname = fullfile(ref_merged_folder, sprintf('%s.tif', current_fov));
round1_ref_merge = max(sdata.images{"round1"}, [], 4);
SaveSingleStack(round1_ref_merge, ref_merged_fname);

% Save round 4 DAPI image
dapi_fname = dir(fullfile(input_path, "round4", current_fov, '*ch04.tif'));
reg_dapi_img = LoadMultipageTiff(fullfile(dapi_fname.folder, dapi_fname.name), false);
reg_dapi_img = imrotate(reg_dapi_img, -90);

if i > 435
    reg_dapi_img = imrotate(reg_dapi_img, 0.75, 'bilinear', 'crop');
end
reg_dapi_img = DFTApply3D(reg_dapi_img, sdata.registration{"round4"}, false);
reg_dapi_img = uint8(reg_dapi_img);

reg_dapi_folder = fullfile(output_path, "images", "DAPI");
if ~exist(reg_dapi_folder, 'dir')
    mkdir(reg_dapi_folder);
end
reg_dapi_fname = fullfile(reg_dapi_folder, sprintf('%s.tif', current_fov));
SaveSingleStack(reg_dapi_img, reg_dapi_fname);

% Spot finding 
sdata = sdata.SpotFinding('ref_layer', "round1");
sdata = sdata.ReadsExtraction('voxel_size', [1 1 1]);
sdata = sdata.LoadCodebook('split_index', 5);
sdata = sdata.ReadsFiltration('end_base', ["CC", "TT"], 'n_barcode_segments', 2, 'split_index', 5);

% Output 
% sdata = sdata.MakeProjection;
% preview_folder = fullfile(output_path, "images", "montage_preview");
% if ~exist(preview_folder, 'dir')
%     mkdir(preview_folder);
% end
% projection_preview_path = fullfile(preview_folder, sprintf("%s.tif", current_fov));
% sdata = sdata.ViewProjection('save', true, 'output_path', projection_preview_path);

round1_ref_merge_max = max(round1_ref_merge, [], 3);
sdata = sdata.ViewSignal('bg_img', round1_ref_merge_max, 'save', true);
sdata = sdata.SaveSignal;

toc(starting);
diary off;
