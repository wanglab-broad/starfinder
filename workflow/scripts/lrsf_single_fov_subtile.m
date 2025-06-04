% run registration and spot finding workflow 
% user need to define the location of the dataet configuration file
% config_path 

function subtile_data = lrsf_single_fov_subtile(config_path, subtile_path)

    % load dataset info 
    config_raw = fileread(config_path);
    config = jsondecode(config_raw);

    % add path for .m files
    addpath(fullfile(config.starfinder_path, 'code-base/src/'))
    addpath(genpath(fullfile(config.starfinder_path, 'code-base/matlab-addon/')))
    starting = tic;

    % create object instance
    % output_path = fullfile(config.root_output_path, config.dataset_id, config.output_id, "output", "subtile", current_fov);
    % subtile_data = load(fullfile(output_path, sprintf('subtile_data_%d.mat', subtile_index)));
    load(subtile_path);

    % registration
    if config.rules.lrsf_single_fov_subtile.parameters.local_registration.run
        subtile_data = subtile_data.LocalRegistration('ref_layer', config.rules.lrsf_single_fov_subtile.parameters.local_registration.ref_round);
    end

    if config.rules.lrsf_single_fov_subtile.parameters.morph_recon.run
        subtile_data = subtile_data.MorphRecon('radius', config.rules.lrsf_single_fov_subtile.parameters.morph_recon.radius);
    end

    % spot finding 
    if config.rules.lrsf_single_fov_subtile.parameters.spot_finding.run
        subtile_data = subtile_data.SpotFinding('ref_layer', config.rules.lrsf_single_fov_subtile.parameters.spot_finding.ref_round, ...
                                  'intensity_threshold', config.rules.lrsf_single_fov_subtile.parameters.spot_finding.intensity_threshold);
    end

    % reads extraction
    if config.rules.lrsf_single_fov_subtile.parameters.reads_extraction.run
        subtile_data = subtile_data.ReadsExtraction('voxel_size', config.rules.lrsf_single_fov_subtile.parameters.reads_extraction.voxel_size);
    end

    % load codebook
    if config.rules.lrsf_single_fov_subtile.parameters.load_codebook.run
        subtile_data = subtile_data.LoadCodebook('split_index', config.rules.lrsf_single_fov_subtile.parameters.load_codebook.split_index);
    end

    % filter reads
    if config.rules.lrsf_single_fov_subtile.parameters.reads_filtration.run
        subtile_data = subtile_data.ReadsFiltration('end_base', config.rules.lrsf_single_fov_subtile.parameters.reads_filtration.end_base, ...
                                  'n_barcode_segments', config.rules.lrsf_single_fov_subtile.parameters.reads_filtration.n_barcode_segments, ...
                                  'split_index', config.rules.lrsf_single_fov_subtile.parameters.reads_filtration.split_index);  
    end

    % output 
    subtile_data = subtile_data.SaveSignal;

    toc(starting);

end