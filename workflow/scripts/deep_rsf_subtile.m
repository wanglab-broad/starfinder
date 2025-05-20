% run registration and spot finding workflow 
% user need to define the location of the dataet configuration file
% config_path 

function subtile_data = deep_rsf_subtile(config_path, subtile_path)

    % load dataset info 
    config_raw = fileread(config_path);
    config = jsondecode(config_raw);

    % add path for .m files
    addpath(fullfile(config.starfinder_path, 'code-base/src/'))
    addpath(genpath(fullfile(starfinder_path, 'code-base/matlab-addon/')))
    starting = tic;

    % create object instance
    % output_path = fullfile(config.root_output_path, config.dataset_id, config.output_id, "output", "subtile", current_fov);
    % subtile_data = load(fullfile(output_path, sprintf('subtile_data_%d.mat', subtile_index)));
    load(subtile_path);

    input_path = fullfile(config.root_input_path, config.dataset_id, config.sample_id);
    output_path = fullfile(config.root_output_path, config.dataset_id, config.output_id);
    subtile_data.inputPath = input_path;
    subtile_data.outputPath = output_path;

    % % registration
    % if config.rules.deep_rsf_subtile.parameters.global_registration.run
    %     subtile_data = subtile_data.GlobalRegistration('ref_layer', config.rules.deep_rsf_subtile.parameters.global_registration.ref_round);
    % end

    % if config.rules.deep_rsf_subtile.parameters.local_registration.run
    %     subtile_data = subtile_data.LocalRegistration('ref_layer', config.rules.deep_rsf_subtile.parameters.local_registration.ref_round);
    % end

    if config.rules.deep_rsf_subtile.parameters.morph_recon.run
        subtile_data = subtile_data.MorphRecon('radius', config.rules.deep_rsf_subtile.parameters.morph_recon.radius);
    end

    % spot finding 
    if config.rules.deep_rsf_subtile.parameters.spot_finding.run
        subtile_data = subtile_data.SpotFinding('ref_layer', config.rules.deep_rsf_subtile.parameters.spot_finding.ref_round, ...
                                                'intensity_estimation', config.rules.deep_rsf_subtile.parameters.spot_finding.intensity_estimation, ...
                                                'intensity_threshold', config.rules.deep_rsf_subtile.parameters.spot_finding.intensity_threshold);
    end

    if ~isempty(subtile_data.signal.allSpots)
        % reads extraction
        if config.rules.deep_rsf_subtile.parameters.reads_extraction.run
            subtile_data = subtile_data.ReadsExtraction('voxel_size', config.rules.deep_rsf_subtile.parameters.reads_extraction.voxel_size);
        end

        % load codebook
        if config.rules.deep_rsf_subtile.parameters.load_codebook.run
            subtile_data = subtile_data.LoadCodebook('split_index', config.rules.deep_rsf_subtile.parameters.load_codebook.split_index);
        end

        % filter reads
        if config.rules.deep_rsf_subtile.parameters.reads_filtration.run
            subtile_data = subtile_data.ReadsFiltration('end_base', config.rules.deep_rsf_subtile.parameters.reads_filtration.end_base, ...
                                    'n_barcode_segments', config.rules.deep_rsf_subtile.parameters.reads_filtration.n_barcode_segments, ...
                                    'split_index', config.rules.deep_rsf_subtile.parameters.reads_filtration.split_index);  
        end

        % output 
        subtile_data = subtile_data.SaveSignal('field_to_keep', "all");
    else
        % output 
        score_log_folder = fullfile(subtile_data.outputPath, "log", "sf_scores");
        current_fname = fullfile(score_log_folder, sprintf("%s_%s.txt", subtile_data.fovID, string(subtile_data.subtile.index)));
        subtile_data.signal.scores = [0 0 0 0 0 0 0];
        headers = ["fov_id" "subtile_id" "total_spots" "no_color" "multi_color" "spots_in_codebook" "spots_in_correctform" "correctform_in_codebook" "good_spots"];
        scores_to_save = [string(subtile_data.fovID) string(subtile_data.subtile.index) string(subtile_data.signal.scores)];
        writematrix(headers, current_fname, 'Delimiter', ',');
        writematrix(scores_to_save, current_fname, 'Delimiter', ',', 'WriteMode', 'append');

        subtile_data.signal.goodSpots = subtile_data.signal.allSpots;
        subtile_data = subtile_data.SaveSignal('field_to_keep', "all");
    end

    toc(starting);

end