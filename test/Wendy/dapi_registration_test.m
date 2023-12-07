% run test CNS_Well07
if ispc
    base_path = fullfile('Z:/Data/Processed/2023-09-19-wendyw-WW_SC_005/');
    output_path = fullfile('Z:/Data/Analyzed/2023-09-19-wendyw-WW_SC_005/');
elseif isunix
    base_path = fullfile('/home/unix/jiahao/wanglab/Data/Processed/2023-09-19-wendyw-WW_SC_005/');
    output_path = fullfile('/home/unix/jiahao/wanglab/Data/Analyzed/2023-09-19-wendyw-WW_SC_005/');
    addpath('/home/unix/jiahao/wanglab/jiahao/Github/starfinder/code-base/new/')
end 

% Input
data_path = fullfile(base_path);
fovID_list = dir(strcat(base_path, "round2/Position*"));
fovID_list = string({fovID_list(:).name});
fovID_list = fovID_list(1:152);
% fovID = 'Position001';
useGPU = false;
ref_round = ["round1"];
dapi_round = ["round5"];

for fovID=fovID_list
    sdata = STARMapDataset(data_path, output_path, 'useGPU', useGPU);
    sdata.layers.ref = ref_round;

    % load ref round 
    ref_channel_order_dict(1).wavelength = 488;
    ref_channel_order_dict(1).channel = "ch00";
    ref_channel_order_dict(1).name = "seq";

    ref_channel_order_dict(2).wavelength = 546;
    ref_channel_order_dict(2).channel = "ch02";
    ref_channel_order_dict(2).name = "seq";

    ref_channel_order_dict(3).wavelength = 594;
    ref_channel_order_dict(3).channel = "ch01";
    ref_channel_order_dict(3).name = "seq";

    ref_channel_order_dict(4).wavelength = 647;
    ref_channel_order_dict(4).channel = "ch03";
    ref_channel_order_dict(4).name = "seq";
    sdata = sdata.LoadRawImages('fovID', fovID, 'folder_list', ref_round, ...
                                'channel_order_dict', ref_channel_order_dict, 'update_layer_slot', "other");


    % load dapi round 
    dapi_channel_order_dict(1).wavelength = 488;
    dapi_channel_order_dict(1).channel = "ch00";
    dapi_channel_order_dict(1).name = "seq";

    dapi_channel_order_dict(2).wavelength = 546;
    dapi_channel_order_dict(2).channel = "ch02";
    dapi_channel_order_dict(2).name = "seq";

    dapi_channel_order_dict(3).wavelength = 594;
    dapi_channel_order_dict(3).channel = "ch01";
    dapi_channel_order_dict(3).name = "seq";

    dapi_channel_order_dict(4).wavelength = 647;
    dapi_channel_order_dict(4).channel = "ch03";
    dapi_channel_order_dict(4).name = "seq";

    dapi_channel_order_dict(5).wavelength = 647;
    dapi_channel_order_dict(5).channel = "ch04";
    dapi_channel_order_dict(5).name = "DAPI";
    sdata = sdata.LoadRawImages('fovID', fovID, 'folder_list', dapi_round, ...
                                'channel_order_dict', dapi_channel_order_dict, 'update_layer_slot', "other");


    % Registration
    sdata = sdata.GlobalRegistration('layer', ["round5"], 'ref_img', 'merged-image', 'mov_img', 'merged-image');

    % % Save output
    merged_projection = max(sdata.registration{"round1"}, [], 3);
    dapi_projection = max(sdata.images{"round5"}(:,:,:,5), [], 3);

    merged_img_folder = fullfile(output_path, 'images', 'ref_merged');
    if ~exist(merged_img_folder, 'dir')
        mkdir(merged_img_folder);
    end

    dapi_img_folder = fullfile(output_path, 'images', 'DAPI');
    if ~exist(dapi_img_folder, 'dir')
        mkdir(dapi_img_folder);
    end

    merged_img_fname = fullfile(merged_img_folder, sprintf("%s.tif", fovID));
    imwrite(merged_projection, merged_img_fname);

    dapi_img_fname = fullfile(dapi_img_folder, sprintf("%s.tif", fovID));
    imwrite(dapi_projection, dapi_img_fname);
end