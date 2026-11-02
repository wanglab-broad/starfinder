function compare_matlab_stages(config_path)
% Run the MATLAB side of the Python/MATLAB stage comparison on its literal fixture.
%
% config_path names a JSON file with starfinder_path and directory, written by
% src/python/scripts/matlab_stage_comparison.py prepare. For Z=5 and Z=1 it reads
% <directory>/z<Z>-input.mat and writes z<Z>-matlab.mat and z<Z>-r<r>-filtered.csv
% (r = 0, 1); it also writes matlab-runtime.json and matlab-ver.txt. It calls the
% src/matlab functions unchanged; design/validation/matlab-stage-comparison.md
% fixes the fixture, the mappings and the checks.
    cfg = jsondecode(fileread(config_path));
    addpath(fullfile(cfg.starfinder_path, 'src', 'matlab'));

    licenses = struct('Image_Toolbox', license('checkout', 'Image_Toolbox'), ...
                      'Statistics_Toolbox', license('checkout', 'Statistics_Toolbox'));
    assert(licenses.Image_Toolbox == 1 && licenses.Statistics_Toolbox == 1, 'toolbox licence checkout failed');
    runtime = struct();
    runtime.version = version;
    runtime.release = version('-release');
    runtime.matlabroot = matlabroot;
    runtime.computer = computer;
    runtime.threads = maxNumCompThreads;
    runtime.licenses = licenses;
    runtime.licenses_in_use = license('inuse');
    runtime.toolboxes = ver;
    runtime.date = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'''));
    assert(runtime.threads == 1, 'expected one computational thread (matlab_single_thread: true)');
    write_text(fullfile(cfg.directory, 'matlab-ver.txt'), evalc('ver'));
    write_text(fullfile(cfg.directory, 'matlab-runtime.json'), jsonencode(runtime, 'PrettyPrint', true));

    % Color symbol m is MATLAB input channel m after this reorder (Python color_to_channel {1:1, 2:0, 3:3, 4:2}).
    channel_order = [2 1 4 3];
    seq_to_gene = dictionary([EncodeBases("AAC") EncodeBases("ACC")], ["gene-A" "gene-B"]);
    for depth = [5 1]
        s = load(fullfile(cfg.directory, sprintf('z%d-input.mat', depth)));
        out = struct();
        if depth > 1
            props = SpotFindingMax3D(s.first, "adaptive", 0.1);
            out.detection_xyz = double(props.Centroid);
            out.detection_channel = double(props.Channel);
            params = DFTRegister3D(s.reference, s.moving);
            out.registered = DFTApply3D(s.moving, params);
        else
            % SpotFindingMax3D is volumetric (regionprops3): Z=1 detection is not compared.
            out.detection_xyz = zeros(0, 3);
            out.detection_channel = zeros(0, 1);
            params = DFTRegister2D(s.reference, s.moving);
            out.registered = DFTApply2D(s.moving, params);
        end
        out.shifts = params.shifts;

        % Supplied centers, one-based x, y, z: A, B, zero, tie.
        spots = array2table(double(s.centers_xyz), 'VariableNames', {'x', 'y', 'z'});
        rounds = {s.extract_first, s.second};
        for radius = 0:1
            calls = strings(4, 2);
            scores = zeros(4, 2);
            for r = 1:2
                [calls(:, r), scores(:, r)] = ExtractFromLocation(rounds{r}(:, :, :, channel_order), spots, ...
                                                                  [radius radius radius]);
            end
            out.(sprintf('calls_r%d', radius)) = cellstr(calls);
            out.(sprintf('scores_r%d', radius)) = scores;

            % DecodeCS does not accept "M": only A and B, uniquely called, are filtered.
            obj = struct();
            obj.signal.allSpots = spots(1:2, :);
            obj.signal.allSpots.color_seq = calls(1:2, 1) + calls(1:2, 2);
            obj.signal.scores = [];
            obj.codebook.seqToGene = seq_to_gene;
            obj = FilterReads(obj, ["AC"]);
            writetable(obj.signal.goodSpots, fullfile(cfg.directory, sprintf('z%d-r%d-filtered.csv', depth, radius)));
        end
        out.encoded = cellstr([EncodeBases("AAC") EncodeBases("ACC")]);
        save(fullfile(cfg.directory, sprintf('z%d-matlab.mat', depth)), '-struct', 'out', '-v7');
    end
end


function write_text(path, text)
    fid = fopen(path, 'w');
    assert(fid > 0, 'cannot write %s', path);
    fprintf(fid, '%s', text);
    fclose(fid);
end
