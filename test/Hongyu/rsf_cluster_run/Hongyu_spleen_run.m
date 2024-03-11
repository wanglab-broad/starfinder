% run registration and spot finding workflow with 2D mouse tissue section 
% user will define:
% input_path 
% output_path
% ref_round
% fov_id_pattern
% number_of_fovs

% test block
input_path = fullfile('/stanley/WangLab/Data/Processed/2024-03-08-Hongyu-Covid_LN/');
output_path = fullfile('/stanley/WangLab/Data/Analyzed/2024-03-08-Hongyu-Covid_LN/');
ref_round = ["round4"];

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

add_channel_order_dict(1).wavelength = 488;
add_channel_order_dict(1).channel = "ch00";
add_channel_order_dict(1).name = "Flamingo";

add_channel_order_dict(2).wavelength = 546;
add_channel_order_dict(2).channel = "ch01";
add_channel_order_dict(2).name = "RBD";

add_channel_order_dict(3).wavelength = 546;
add_channel_order_dict(3).channel = "ch02";
add_channel_order_dict(3).name = "DAPI";

sdata = sdata.LoadRawImages('fovID', current_fov, ... 
                            'rotate_angle', -90, ...
                            'folder_list', ["flamingo"], ...
                            'channel_order_dict', add_channel_order_dict, ...
                            'update_layer_slot', "other");

% Preprocessing
sdata = sdata.EnhanceContrast("min-max");

% Registration
sdata = sdata.GlobalRegistration;

% Save round 1 ref image
ref_merged_folder = fullfile(output_path, "images", "ref_merged_round1");
if ~exist(ref_merged_folder, 'dir')
    mkdir(ref_merged_folder);
end
ref_merged_fname = fullfile(ref_merged_folder, sprintf('%s.tif', current_fov));
round1_ref_merge = max(sdata.images{"round1"}, [], 4);
SaveSingleStack(round1_ref_merge, ref_merged_fname);

refernce_dapi_fname = dir(fullfile(input_path, ref_round, current_fov, '*ch04.tif'));
current_ref_img = LoadMultipageTiff(fullfile(refernce_dapi_fname.folder, refernce_dapi_fname.name), false);
current_ref_img = imrotate(current_ref_img, -90);
sdata = sdata.GlobalRegistration('layer', ["flamingo"], ...
                                'ref_img', 'input_image', ...
                                'input_image', current_ref_img, ...
                                'mov_img', 'single-channel');

% Spot finding 
sdata = sdata.SpotFinding('ref_layer', "round1", 'intensity_threshold', 0.4);
sdata = sdata.ReadsExtraction('voxel_size', [1 1 1]);
sdata = sdata.LoadCodebook;
sdata = sdata.ReadsFiltration('end_base', ["AC"]);
round1_ref_merge_max = max(round1_ref_merge, [], 3);
sdata = sdata.ViewSignal('bg_img', round1_ref_merge_max, 'save', true);

% Output 
sdata = sdata.MakeProjection;
preview_folder = fullfile(output_path, "images", "montage_preview");
if ~exist(preview_folder, 'dir')
    mkdir(preview_folder);
end
projection_preview_path = fullfile(preview_folder, sprintf("%s.tif", current_fov));
sdata = sdata.ViewProjection('save', true, 'output_path', projection_preview_path);
sdata = sdata.SaveImages('layer', sdata.layers.other, 'output_path', output_path, 'folder_format', "single", 'maximum_projection', false);

sdata = sdata.SaveSignal;

toc(starting);
diary off;
