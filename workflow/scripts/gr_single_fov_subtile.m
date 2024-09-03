% run registration and spot finding workflow 
% user need to define the location of the dataet configuration file
% config_path 

function sdata = gr_single_fov_subtile(config_path, current_fov)

    % load dataset info 
    config_raw = fileread(config_path);
    config = jsondecode(config_raw);

    % add path for .m files
    addpath(fullfile(config.starfinder_path, 'code-base/src/'))

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
    sdata = sdata.LoadRawImages('fovID', current_fov, 'rotate_angle', config.rotate_angle);
    sdata.layers.ref = config.ref_round;

    % preprocessing
    if config.rules.rsf_single_fov.parameters.enhance_contrast.run
        sdata = sdata.EnhanceContrast("min-max");
    end

    if config.rules.rsf_single_fov.parameters.hist_equalize.run
        sdata = sdata.HistEqualize;
    end

    if config.rules.rsf_single_fov.parameters.morph_recon.run
        sdata = sdata.MorphRecon('radius', config.rules.rsf_single_fov.parameters.morph_recon.radius);
    end

    % global registration
    if config.rules.rsf_single_fov.parameters.global_registration.run
        sdata = sdata.GlobalRegistration('ref_layer', config.rules.rsf_single_fov.parameters.global_registration.ref_round);
    end

    % save reference images 
    ref_merged_folder = fullfile(output_path, "images", "ref_merged");
    if ~exist(ref_merged_folder, 'dir')
        mkdir(ref_merged_folder);
    end
    ref_merged_fname = fullfile(ref_merged_folder, sprintf('%s.tif', current_fov));
    if config.maximum_projection
        SaveSingleStack(max(sdata.registration{sdata.layers.ref}, [], 3), ref_merged_fname);
    else
        SaveSingleStack(sdata.registration{sdata.layers.ref}, ref_merged_fname);
    end

    % output 
    sdata = sdata.MakeProjection;
    preview_folder = fullfile(output_path, "images", "montage_preview");
    if ~exist(preview_folder, 'dir')
        mkdir(preview_folder);
    end
    projection_preview_path = fullfile(preview_folder, sprintf("%s.tif", current_fov));
    sdata = sdata.ViewProjection('save', true, 'output_path', projection_preview_path);

    % save subtile 
    sdata = sdata.CreateSubtiles('save', true);

    toc(starting);
    diary off;

end