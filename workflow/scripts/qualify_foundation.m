function qualify_foundation(config_path)
% Bounded development comparison adapter; does not change processing APIs.
    cfg = jsondecode(fileread(config_path));
    addpath(fullfile(cfg.starfinder_path, 'src/matlab'));
    assert(license('checkout', 'Image_Toolbox') == 1);
    assert(license('checkout', 'Statistics_Toolbox') == 1);
    runtime.version = version;
    runtime.release = version('-release');
    runtime.toolboxes = ver;
    runtime.threads = maxNumCompThreads;
    assert(runtime.threads == 1);
    for depth = [5 1]
        s = load(fullfile(cfg.directory, sprintf('z%d-input.mat', depth)));
        if depth > 1
            props = SpotFindingMax3D(s.first, 'adaptive', 0.1);
            detection_xyz = double(props.Centroid);
            detection_channel = props.Channel;
            [params, registered] = DFTRegister3D(s.reference, s.moving);
            shifts_yxz = params.shifts;
        else
            detection_xyz = zeros(0, 3);
            detection_channel = zeros(0, 1);
            [params, registered] = DFTRegister2D(s.reference, s.moving);
            shifts_yxz = [params.shifts 0];
        end
        spots = array2table(s.centers_xyz, 'VariableNames', {'x','y','z'});
        calls = strings(4, 2, 2);
        scores = zeros(4, 2, 2);
        for radius = 0:1
            [calls(:,1,radius+1), scores(:,1,radius+1)] = ExtractFromLocation(s.extract_first(:,:,:,[2 1 4 3]), spots, [radius radius radius]);
            [calls(:,2,radius+1), scores(:,2,radius+1)] = ExtractFromLocation(s.second(:,:,:,[2 1 4 3]), spots, [radius radius radius]);
        end
        called = spots(1:2,:);
        called.color_seq = calls(1:2,1,1) + calls(1:2,2,1);
        obj.signal.allSpots = called;
        obj.signal.scores = [];
        obj.codebook.seqToGene = dictionary(["12" "21"], ["gene-A" "gene-B"]);
        obj = FilterReads(obj, {'AC'});
        writetable(obj.signal.goodSpots, fullfile(cfg.directory, sprintf('z%d-filtered.csv', depth)));
        encoded = [EncodeBases('AAC') EncodeBases('ACC')];
        result = struct('calls', calls, 'encoded', encoded, 'runtime', runtime);
        fid = fopen(fullfile(cfg.directory, sprintf('z%d-result.json', depth)), 'w');
        fprintf(fid, '%s', jsonencode(result));
        fclose(fid);
        save(fullfile(cfg.directory, sprintf('z%d-result.mat', depth)), 'registered', 'shifts_yxz', 'detection_xyz', 'detection_channel', 'scores', '-v7');
    end
end
