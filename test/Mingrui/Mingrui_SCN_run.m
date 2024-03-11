% run registration and spot finding workflow with 2D mouse tissue section 
% user will define:
% input_path 
% output_path
% ref_round
% fov_id_pattern
% number_of_fovs

% test block
input_path = fullfile('/stanley/WangLab/Data/Processed/2024-02-28-Mingrui-4-DEG_pilot_test/');
output_path = fullfile('/stanley/WangLab/Data/Analyzed/2024-02-28-Mingrui-4-DEG_pilot_test/');
ref_round = ["RIBO"];

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

add_channel_order_dict(1).wavelength = 488;
add_channel_order_dict(1).channel = "ch00";
add_channel_order_dict(1).name = "gene1";

add_channel_order_dict(2).wavelength = 546;
add_channel_order_dict(2).channel = "ch01";
add_channel_order_dict(2).name = "gene2";

add_channel_order_dict(3).wavelength = 546;
add_channel_order_dict(3).channel = "ch02";
add_channel_order_dict(3).name = "gene3";

add_channel_order_dict(4).wavelength = 546;
add_channel_order_dict(4).channel = "ch03";
add_channel_order_dict(4).name = "gene4";

sdata = sdata.LoadRawImages('fovID', current_fov, ... 
                            'rotate_angle', -90, ...
                            'folder_list', ref_round, ...
                            'channel_order_dict', add_channel_order_dict, ...
                            'update_layer_slot', "seq");
sdata.layers.ref = ref_round;

% Preprocessing
sdata = sdata.EnhanceContrast("min-max");

% Spot finding 
sdata = sdata.SpotFinding('ref_layer', ref_round, 'intensity_threshold', 0.3);

% Output 
ref_merge = max(sdata.images{ref_round}, [], 4);
ref_merge_max = max(ref_merge, [], 3);

% % Save ref image
% ref_merged_folder = fullfile(output_path, "images", "ref_merged");
% if ~exist(ref_merged_folder, 'dir')
%     mkdir(ref_merged_folder);
% end
% ref_merged_fname = fullfile(ref_merged_folder, sprintf('%s.tif', current_fov));
% SaveSingleStack(ref_merge_max, ref_merged_fname);

sdata = sdata.ViewSignal('signal_slot', "allSpots", 'bg_img', ref_merge_max, 'save', true);

sdata = sdata.SaveSignal('signal_slot', "allSpots", 'field_to_keep', ["x", "y", "z", "MaxIntensity", "Channel"]);

toc(starting);
diary off;
