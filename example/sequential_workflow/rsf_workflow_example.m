% run registration and spot finding workflow with 2D mouse tissue section 
% use need to define the location of the configuration file
% config_path 

function sdata = rsf_workflow_example(config_path)

    % load dataset info 
    config_raw = fileread(config_path);
    config = jsondecode(config_raw);

    % add path for .m files
    addpath(fullfile(pwd, '../code-base/new/')) % pwd is the location of the starfinder folder

    % test block
    addpath(fullfile('/home/unix/jiahao/wanglab/jiahao/Github/starfinder/code-base/new/'))

    for n=config.starting_fov_id:config.starting_fov_id + config.number_of_fovs - 1

        current_fov = sprintf(config.fov_id_pattern, n);

        sdata = STARMapDataset(config.input_path, config.output_path, 'useGPU', false);

        log_folder = fullfile(config.output_path, "log");
        if ~exist(log_folder, 'dir')
            mkdir(log_folder);
        end
        diary_file = fullfile(log_folder, sprintf("%s.txt", current_fov));
        if exist(diary_file, 'file'); delete(diary_file); end
        diary(diary_file);

        tic;

        % load sequencing images 
        sdata = sdata.LoadRawImages('fovID', current_fov, 'rotate_angle', config.rotate_angle);
        sdata.layers.ref = config.ref_round;

        % load additional images
        sdata = sdata.LoadRawImages('fovID', current_fov, ... 
                                    'rotate_angle', config.rotate_angle, ...
                                    'folder_list', string(config.additional_round), ...
                                    'channel_order_dict', config.channel_order, ...
                                    'update_layer_slot', "other");

        % Preprocessing
        sdata = sdata.EnhanceContrast("min-max");
        sdata = sdata.EnhanceContrast("min-max", 'layer', sdata.layers.other);
        sdata = sdata.HistEqualize;

        % Registration
        sdata = sdata.GlobalRegistration;

        ref_merged_folder = fullfile(config.output_path, "images", "ref_merged");
        if ~exist(ref_merged_folder, 'dir')
            mkdir(ref_merged_folder);
        end
        ref_merged_fname = fullfile(ref_merged_folder, sprintf('%s.tif', current_fov));
        if config.maximum_projection
            SaveSingleStack(max(sdata.registration{sdata.layers.ref}, [], 3), ref_merged_fname);
        else
            SaveSingleStack(sdata.registration{sdata.layers.ref}, ref_merged_fname);
        end

        refernce_dapi_fname = dir(fullfile(config.input_path, 'round1', current_fov, '*ch04.tif'));
        current_ref_img = LoadMultipageTiff(fullfile(refernce_dapi_fname.folder, refernce_dapi_fname.name), false);
        current_ref_img = imrotate(current_ref_img, config.rotate_angle);
        sdata = sdata.GlobalRegistration('layer', sdata.layers.other, ...
                                        'ref_img', 'input_image', ...
                                        'input_image', current_ref_img, ...
                                        'mov_img', 'single-channel', ...
                                        'ref_channel', config.ref_channel);
        sdata = sdata.LocalRegistration;

        % Spot finding 
        sdata = sdata.SpotFinding;
        sdata = sdata.ReadsExtraction;
        sdata = sdata.LoadCodebook;
        sdata = sdata.ReadsFiltration;

        % Output 
        sdata = sdata.MakeProjection;
        preview_folder = fullfile(config.output_path, "images", "montage_preview");
        if ~exist(preview_folder, 'dir')
            mkdir(preview_folder);
        end
        projection_preview_path = fullfile(preview_folder, sprintf("%s.tif", current_fov));
        sdata = sdata.ViewProjection('save', true, 'output_path', projection_preview_path);

        sdata = sdata.SaveImages('layer', sdata.layers.other, 'output_path', config.output_path, 'folder_format', "single", 'maximum_projection', config.maximum_projection);
        sdata = sdata.SaveSignal;

        toc;
        diary off;

    end

end