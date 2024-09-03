% run registration and spot finding workflow 
% user need to define the location of the dataet configuration file
% config_path 

function subtile_data = lrsf_single_fov_subtile(config_path, current_fov, subtile_index)

    % load dataset info 
    config_raw = fileread(config_path);
    config = jsondecode(config_raw);

    % add path for .m files
    addpath(fullfile(config.starfinder_path, 'code-base/src/'))
    starting = tic;

    % create object instance
    output_path = fullfile(config.root_output_path, config.dataset_id, config.output_id, "output", "subtile", current_fov);
    subtile_data = load(fullfile(output_path, sprintf('subtile_data_%d.mat', subtile_index)));

    % registration
    if config.rules.rsf_single_fov.parameters.local_registration.run
        subtile_data = subtile_data.LocalRegistration('ref_layer', config.rules.rsf_single_fov.parameters.local_registration.ref_round);
    end

    % spot finding 
    if config.rules.rsf_single_fov.parameters.spot_finding.run
        subtile_data = subtile_data.SpotFinding('ref_layer', config.rules.rsf_single_fov.parameters.spot_finding.ref_round, ...
                                  'intensity_threshold', config.rules.rsf_single_fov.parameters.spot_finding.intensity_threshold);
    end

    % reads extraction
    if config.rules.rsf_single_fov.parameters.reads_extraction.run
        subtile_data = subtile_data.ReadsExtraction('voxel_size', config.rules.rsf_single_fov.parameters.reads_extraction.voxel_size);
    end
  
    % load codebook and filter reads
    if config.rules.rsf_single_fov.parameters.reads_filtration.run
        subtile_data = subtile_data.LoadCodebook;
        subtile_data = subtile_data.ReadsFiltration('end_base', config.rules.rsf_single_fov.parameters.reads_filtration.end_base);
    end

    % output 
    subtile_data = subtile_data.SaveSignal;

    toc(starting);

end