% run registration and spot finding workflow 
% user need to define the location of the dataet configuration file
% config_path 

function sdata = nuclei_registration(config_path, current_fov)

    % load dataset info 
    config_raw = fileread(config_path);
    config = jsondecode(config_raw);

    % add path for .m files
    addpath(fullfile(config.starfinder_path, 'src/matlab/'))
    addpath(genpath(fullfile(config.starfinder_path, 'src/matlab-addon/')))

    % create object instance
    input_path = fullfile(config.root_input_path, config.dataset_id, config.sample_id);
    output_path = fullfile(config.root_output_path, config.dataset_id, config.output_id);
    sdata = STARMapDataset(input_path, output_path, 'useGPU', false);

    % create log folder and file path
    log_folder = fullfile(output_path, "log");
    if ~exist(log_folder, 'dir')
        mkdir(log_folder);
    end
    diary_file = fullfile(log_folder, sprintf("%s_nr.txt", current_fov));
    if exist(diary_file, 'file'); delete(diary_file); end
    diary(diary_file);

    starting = tic;

    % load additional images
    for i = 1:length(config.additional_round)
        sdata = sdata.LoadRawImages('fovID', current_fov, ... 
                                    'rotate_angle', config.rotate_angle, ...
                                    'folder_list', [string(config.additional_round(i).round_name)], ...
                                    'channel_order_dict', config.additional_round(i).channel_order, ...
                                    'update_layer_slot', "other");
    end

    sdata = sdata.EnhanceContrast("min-max", 'layer', sdata.layers.other);

    % registration
    % % load reference nuclei image for additional registration 
    refernce_dapi_fname = dir(fullfile(input_path, config.ref_round, current_fov, '*ch04.tif'));
    current_ref_img = LoadMultipageTiff(fullfile(refernce_dapi_fname.folder, refernce_dapi_fname.name), false);
    current_ref_img = imrotate(current_ref_img, config.rotate_angle);
    sdata.layers.ref = config.ref_round;
    sdata = sdata.GlobalRegistration('layer', sdata.layers.other, ...
                                    'ref_img', 'input_image', ...
                                    'input_image_ref', current_ref_img, ...
                                    'mov_img', 'single-channel', ...
                                    'ref_channel', config.ref_channel, ...
                                    'log_suffix', "nr");

    % output 
    sdata = sdata.SaveImages('layer', sdata.layers.other, 'output_path', output_path, 'folder_format', "single", 'maximum_projection', config.maximum_projection);

    toc(starting);
    diary off;

end