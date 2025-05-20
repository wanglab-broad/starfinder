% run registration and spot finding workflow 
% user need to define the location of the dataet configuration file
% config_path 

function sdata = rsf_single_fov_seq(config_path, current_fov)

    % load dataset info 
    config_raw = fileread(config_path);
    config = jsondecode(config_raw);

    % add path for .m files
    addpath(fullfile(config.starfinder_path, 'code-base/src/'))
    addpath(genpath(fullfile(starfinder_path, 'code-base/matlab-addon/')))

    % create object instance
    input_path = fullfile(config.root_input_path, config.dataset_id, config.sample_id);
    output_path = fullfile(config.root_output_path, config.dataset_id, config.output_id);
    sdata = STARMapDataset(input_path, output_path, 'useGPU', false);

    % create log folder and file path
    log_folder = fullfile(output_path, "log");
    if ~exist(log_folder, 'dir')
        mkdir(log_folder);
    end
    diary_file = fullfile(log_folder, sprintf("%s_rsf.txt", current_fov));
    if exist(diary_file, 'file'); delete(diary_file); end
    diary(diary_file);

    starting = tic;

    % load sequencing images 
    sdata = sdata.LoadRawImages('fovID', current_fov, 'rotate_angle', config.rotate_angle, ...
                                'channel_order_dict', config.seq_channel_order);

    sdata.layers.ref = config.ref_round;

    % registration
    if config.rules.rsf_single_fov_seq.parameters.nuclei_registration.run
        % load reference nuclei image for registration 
        refernce_dapi_fname = dir(fullfile(config.rules.rsf_single_fov_seq.parameters.nuclei_registration.ref_dapi_path, current_fov, ...
                                config.rules.rsf_single_fov_seq.parameters.nuclei_registration.ref_dapi_pattern));
        current_ref_img = LoadMultipageTiff(fullfile(refernce_dapi_fname.folder, refernce_dapi_fname.name), false);
        current_ref_img = imrotate(current_ref_img, config.rotate_angle);
        sdata.layers.ref = config.ref_round;

        moving_dapi_fname = dir(fullfile(config.rules.rsf_single_fov_seq.parameters.nuclei_registration.mov_dapi_path, current_fov, ...
                                config.rules.rsf_single_fov_seq.parameters.nuclei_registration.mov_dapi_pattern));
        % '*ch04.tif'));
        current_mov_img = LoadMultipageTiff(fullfile(moving_dapi_fname.folder, moving_dapi_fname.name), false);
        current_mov_img = imrotate(current_mov_img, config.rotate_angle);

        sdata = sdata.GlobalRegistration('layer', sdata.layers.seq, ...
                                        'layers_to_register', config.rules.rsf_single_fov_seq.parameters.nuclei_registration.layers_to_register, ...
                                        'ref_img', 'input_image', ...
                                        'input_image_ref', current_ref_img, ...
                                        'mov_img', 'input_image', ...
                                        'input_image_mov', current_mov_img, ...
                                        'log_suffix', "nr");
    end

    % spot finding 
    if config.rules.rsf_single_fov_seq.parameters.spot_finding.run
        sdata = sdata.SpotFinding('ref_layer', config.rules.rsf_single_fov_seq.parameters.spot_finding.ref_round, ...
                                  'intensity_threshold', config.rules.rsf_single_fov_seq.parameters.spot_finding.intensity_threshold);
    end

    % output
    sdata.registration{sdata.layers.ref} = max(sdata.images{sdata.layers.ref}, [], 4);
    ref_merge_max = max(sdata.registration{sdata.layers.ref}, [], 3);
    sdata = sdata.ViewSignal('signal_slot', "allSpots", 'bg_img', ref_merge_max, 'save', true);
    sdata = sdata.SaveSignal('signal_slot', "allSpots", 'field_to_keep', "all");

    toc(starting);
    diary off;

end