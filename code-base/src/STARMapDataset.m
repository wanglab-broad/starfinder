classdef STARMapDataset
% STARMapDataset, the primary class for the STARMap imaging analysis pipeline
% ====Properties====
% ...

% ====Methods====
% ...
    
    properties
        
        % Basic parameters
        inputPath;
        outputPath;
        fovID;
        useGPU;
        
        % Images 
        images;
        projections;
        layers;

        % Metadata
        metadata;
        subtile;
        
        % Registration
        registration;
     
        % Spots
        signal;

        % Codebook
        codebook;

        % Workflow
        jobToDo;
        jobFinished;
        
    end
    

    methods
        
        % 1.Construction method of Pipeline object
        function obj = STARMapDataset( inputPath, outputPath, varargin )
            % the construction method of pipeline object, use this to create
            % an object to start analysis by providing an inputPath
            % useGPU: default == false
            
            % Input parser
            p = inputParser;
            
            addRequired(p, 'inputPath');
            addRequired(p, 'outputPath');

            defaultuseGPU = false;
            addOptional(p, 'useGPU', defaultuseGPU);
            parse(p, inputPath, outputPath, varargin{:});
            
            % setup IO
            obj.inputPath = p.Results.inputPath;
            obj.outputPath = p.Results.outputPath;
            
            % make output folder
            if ~exist(obj.outputPath, 'dir')
                mkdir(obj.outputPath)
            end
            
            % setup GPU usage
            obj.useGPU = p.Results.useGPU;
            
            % initiate core properties 
            obj.images = dictionary();
            obj.projections = dictionary();
            obj.metadata = dictionary();
            obj.subtile = struct();
            obj.layers = struct();
            obj.registration = dictionary();
            obj.signal = struct();
            obj.codebook = struct();
            
            obj.subtile.index = 0;
            obj.layers.seq = [];
            obj.layers.other = [];
            obj.layers.ref = [];

            % show message
            fprintf('Pipeline Obj is generated...\n');
            
        end


        % 2.Load raw images 
        function obj = LoadRawImages( obj, varargin )

            % Input parser
            p = inputParser;
            
            defaultfovID = '';
            addOptional(p, 'fovID', defaultfovID);

            defaultupdateLayer = "seq";
            addOptional(p, 'update_layer_slot', defaultupdateLayer);

            default_dirs = dir(strcat(obj.inputPath, '/round*'));
            defaultfolderList = string({default_dirs(:).name});
            addOptional(p, 'folder_list', defaultfolderList);

            defaultDict(1).wavelength = 488;
            defaultDict(1).channel = "ch00";
            defaultDict(1).name = "seq";

            defaultDict(2).wavelength = 546;
            defaultDict(2).channel = "ch02";
            defaultDict(2).name = "seq";

            defaultDict(3).wavelength = 594;
            defaultDict(3).channel = "ch01";
            defaultDict(3).name = "seq";

            defaultDict(4).wavelength = 647;
            defaultDict(4).channel = "ch03";
            defaultDict(4).name = "seq";

            addOptional(p, 'channel_order_dict', defaultDict);

            defaultzrange = []; 
            addOptional(p, 'zrange', defaultzrange);

            defaultconvert = false;
            addOptional(p, 'convert_uint8', defaultconvert);

            defaultAngle = 0;
            addOptional(p, 'rotate_angle', defaultAngle);

            defaultFlip = "";
            addOptional(p, 'flip', defaultFlip);

            defaultuseGPU = false;
            addOptional(p, 'useGPU', defaultuseGPU);
            parse(p, varargin{:});
            
            % Load tiff stacks
            all_dir = dir(obj.inputPath);
            folder_names = {all_dir(:).name};
            round_dir = all_dir(ismember(folder_names, p.Results.folder_list));
            for r=1:numel(round_dir)
                current_round_dir = round_dir(r);
                current_layer = current_round_dir.name;
                fprintf(sprintf('====Loading %s images====\n', current_layer));
                [obj.images{current_layer}, current_dims] = LoadImageStacks(current_round_dir, ...
                    p.Results.fovID, ...
                    p.Results.channel_order_dict, ...
                    p.Results.convert_uint8);

                current_metadata = struct();
                current_metadata.dims = current_dims;
                current_metadata.dimX = current_dims(1);
                current_metadata.dimY = current_dims(2);
                current_metadata.dimZ = current_dims(3);
                current_metadata.Nchannel = current_dims(4);
                current_metadata.BitDepth = current_dims(5);
                obj.fovID = p.Results.fovID;
                current_metadata.ChannelInfo = p.Results.channel_order_dict;
                obj.metadata{current_layer} = current_metadata;
               
                if p.Results.rotate_angle ~= 0
                    obj.images{current_layer} = imrotate(obj.images{current_layer}, p.Results.rotate_angle);
                end

                if ~isempty(p.Results.flip)
                    switch p.Results.flip
                        case "vertical"
                            obj.images{current_layer} = flip(obj.images{current_layer}, 1);
                        case "horizontal"
                            obj.images{current_layer} = flip(obj.images{current_layer}, 2);
                        case "both"
                            obj.images{current_layer} = flip(obj.images{current_layer}, 1);
                            obj.images{current_layer} = flip(obj.images{current_layer}, 2);
                    end
                end
            end
            
            switch p.Results.update_layer_slot
                case "seq"
                    obj.layers.seq = p.Results.folder_list; 
                case "other"
                    obj.layers.other = [obj.layers.other p.Results.folder_list]; 
            end

            obj.layers.all = transpose(obj.images.keys);
            [obj.images, obj.metadata] = AdjustSizeAcrossRound(obj.images, obj.metadata, obj.layers.all, p.Results.zrange);

        end
        

    % ====Preprocessing====

        % 3.Enhance Contrast
        function obj = EnhanceContrast( obj, method, varargin )

            % Input parser
            p = inputParser;
            
            addRequired(p, 'method');

            defaultLayer = obj.layers.seq; 
            addOptional(p, 'layer', defaultLayer);

            defaultLow_in= 0.01;
            addParameter(p, 'low_in', defaultLow_in);

            defaultHigh_in = 0.95;
            addParameter(p, 'high_in', defaultHigh_in);

            defaultLow_out = 0;
            addParameter(p, 'low_out', defaultLow_out);

            defaultHigh_out = 1;
            addParameter(p, 'high_out', defaultHigh_out);

            defaultGamma = 1;
            addParameter(p, 'gamma', defaultGamma);

            parse(p, method, varargin{:});

            switch p.Results.method
                case "min-max"
                    fprintf("====Min-Max intensity normalization====\n");
                    for current_layer=p.Results.layer
                        fprintf(sprintf("Normalizing %s...", current_layer))
                        obj.images(current_layer) = MinMaxNorm(obj.images(current_layer));
                    end
                case "imadjustn"
                    fprintf("====Running imadjustn====\n");
                    for current_layer=p.Results.layer
                        tic
                        fprintf(sprintf("Processing %s...", current_layer))
                        
                        for c=1:obj.metadata{current_layer}.Nchannel 
                            current_channel = obj.images{current_layer}(:, :, :, c);
                            current_channel_adjusted = imadjustn(current_channel,...
                                [p.Results.low_in p.Results.high_in],...
                                [p.Results.low_out p.Results.high_out],...
                                p.Results.gamma);
                            obj.images{current_layer}(:, :, :, c) = current_channel_adjusted;
                        end
                
                        fprintf(sprintf('[time = %.2f s]\n', toc));
                    end
            end


            % change job log
            obj.jobFinished.EnhanceContrast = true;

        end 
        

        % 4.Hitogram Equalization
        function obj = HistEqualize( obj, varargin )
            
            % Input parser
            p = inputParser;

            defaultLayer = obj.layers.seq; 
            addOptional(p, 'layer', defaultLayer);

            defaultReferenceChannel = 1;
            addParameter(p, 'reference_channel', defaultReferenceChannel);

            defaultReferenceLayer = "round1";
            addParameter(p, 'reference_layer', defaultReferenceLayer);

            defaultNbins = 64;
            addParameter(p, 'nbins', defaultNbins);

            parse(p, varargin{:});
            
            fprintf("====Histogram Equalization====\n");
            fprintf(sprintf('Reference: %s - channel%d\n', p.Results.reference_layer, p.Results.reference_channel));
            reference_image = obj.images{p.Results.reference_layer}(:,:,:,p.Results.reference_channel);

            for current_layer=p.Results.layer
                for c=1:obj.metadata{current_layer}.Nchannel 
                    fprintf('Equalizing %s - channel%d\n', current_layer, c);
                    obj.images{current_layer}(:, :, :, c) = imhistmatchn(obj.images{current_layer}(:, :, :, c), reference_image, p.Results.nbins);
                end
            end

            obj.jobFinished.HistogramEqualization = true;

        end


        % 5.Morphological reconstruction
        function obj = MorphRecon( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultLayer = obj.layers.seq; 
            addOptional(p, 'layer', defaultLayer);

            defaultRadius = 3;
            addOptional(p, 'radius', defaultRadius);

            parse(p, varargin{:});
            
            fprintf("====Morphological Reconstruction====\n");
            
            for current_layer=p.Results.layer
                fprintf(sprintf("Processing %s...", current_layer));
                obj.images(current_layer)= MorphologicalReconstruction(obj.images(current_layer), p.Results.radius);
            end
            
            % change metadata
            obj.jobFinished.MorphologicalReconstruction = true;
            
        end
        
        
        % 5.5 Whilte tophat
        function obj = Tophat( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultLayer = obj.layers.seq; 
            addOptional(p, 'layer', defaultLayer);

            defaultRadius = 3;
            addOptional(p, 'radius', defaultRadius);

            parse(p, varargin{:});
            
            fprintf("====Tophat Filtering====\n");
            % setup structure element
            se = strel('disk', p.Results.radius);

            for current_layer=p.Results.layer
                tic
                fprintf(sprintf("Processing %s...", current_layer))
                
                for c=1:obj.metadata{current_layer}.Nchannel 
                    current_channel = obj.images{current_layer}(:, :, :, c);
                    for z=1:obj.metadata{current_layer}.dimZ
                        current_channel(:,:,z) = imtophat(current_channel(:,:,z), se);
                    end
                    obj.images{current_layer}(:, :, :, c) = uint8(current_channel);
                end
        
                fprintf(sprintf('[time = %.2f s]\n', toc));
            end

            % change metadata
            obj.jobFinished.Tophat = true;
            
        end

        
        % Make Projections
        function obj = MakeProjection( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultLayer = obj.layers.all; 
            addOptional(p, 'layer', defaultLayer);

            defaultMethod = "max";
            addOptional(p, 'method', defaultMethod);

            parse(p, varargin{:});
            
            fprintf("====Generate Projection Images====\n");
            for current_layer=p.Results.layer
                obj.projections{current_layer} = MakeProjections(obj.images(current_layer), p.Results.method);
            end
            
        end


        % View Projections
        function obj = ViewProjection( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultLayer = obj.layers.all; 
            addOptional(p, 'layer', defaultLayer);

            defaultEnhance = false;
            addOptional(p, 'enhance_contrast', defaultEnhance);

            defaultSave = false;
            addOptional(p, 'save', defaultSave);

            defaultOutputPath = fullfile(obj.outputPath, 'projection_montage.tif');
            addOptional(p, 'output_path', defaultOutputPath);

            parse(p, varargin{:});

            montage_img = MakeMontage(obj.projections, p.Results.layer, p.Results.enhance_contrast);
            
            if p.Results.save
                current_output_folder_msg = strrep(p.Results.output_path, '\', '\\');
                fprintf(sprintf('Saving projection montage to %s\n', current_output_folder_msg));
                exportgraphics(montage_img, p.Results.output_path, 'Resolution', 300, 'ContentType', 'image');
            end

        end

        
        % Save Stack Images
        function obj = SaveImages( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultLayer = obj.layers.all; 
            addOptional(p, 'layer', defaultLayer);

            defaultFormat = "nested";
            addOptional(p, 'folder_format', defaultFormat);

            defaultGroupChannel = true;
            addOptional(p, 'group_channel', defaultGroupChannel);

            defaultMaxProjection = true;
            addOptional(p, 'maximum_projection', defaultMaxProjection);

            defaultOutputPath = obj.outputPath;
            addOptional(p, 'output_path', defaultOutputPath);

            parse(p, varargin{:});

            for current_layer=p.Results.layer
                current_output_folder = fullfile(p.Results.output_path, "images/");
                if ~exist(current_output_folder, 'dir')
                    mkdir(current_output_folder)
                end
                current_output_folder_msg = strrep(current_output_folder, '\', '\\');
                fprintf(sprintf('Saving %s images to %s\n', current_layer, current_output_folder_msg));
                switch p.Results.folder_format
                    case "nested"
                        SaveImageNestedFolder(obj.images(current_layer), current_layer, current_output_folder, obj.fovID, p.Results.maximum_projection);
                    case "single"
                        SaveImageSingleFolder(obj.images(current_layer), current_layer, current_output_folder, obj.fovID, p.Results.group_channel, obj.metadata{current_layer}.ChannelInfo, p.Results.maximum_projection);
                end
            end

        end


    % ====Image Registration====
            
        % 6.Global registration
        function obj = GlobalRegistration( obj, varargin )

            % Input parser
            p = inputParser;

            % Defaults
            defaultLayer = obj.layers.seq; 
            addOptional(p, 'layer', defaultLayer);

            defaultRefLayer = obj.layers.ref; % round1
            addOptional(p, 'ref_layer', defaultRefLayer);

            defaultRegLayer = [];
            addOptional(p, 'layers_to_register', defaultRegLayer);

            defaultRefChannel = "DAPI"; 
            addOptional(p, 'ref_channel', defaultRefChannel);

            defaultRefImg= "merged-image"; % single-channel input_image
            addOptional(p, 'ref_img', defaultRefImg);

            defaultMovImg= "merged-image"; % single-channel input_image
            addOptional(p, 'mov_img', defaultMovImg);

            defaultInputImageRef= ""; % 
            addOptional(p, 'input_image_ref', defaultInputImageRef);
            
            defaultInputImageMov= ""; % 
            addOptional(p, 'input_image_mov', defaultInputImageMov);

            defaultScale= 1; % 
            addOptional(p, 'scale', defaultScale);

            defaultSaveShifts = true;
            addParameter(p, 'save_shifts', defaultSaveShifts);

            defaultLogSuffix = "";
            addParameter(p, 'log_suffix', defaultLogSuffix);

            parse(p, varargin{:});

            fprintf('====Global Registration====\n');
            switch p.Results.ref_img
                case "merged-image"
                    obj.registration{p.Results.ref_layer} = max(obj.images{p.Results.ref_layer}, [], 4);
                case "single-channel"
                    obj.registration{p.Results.ref_layer} = obj.images{p.Results.ref_layer}(:,:,:,p.Results.ref_channel);
                case "input_image"
                    obj.registration{p.Results.ref_layer} = p.Results.input_image_ref;
            end

            if p.Results.scale ~= 1
                obj.registration{p.Results.ref_layer} = imresize3(obj.registration{p.Results.ref_layer}, p.Results.scale);
            end
            
            if isempty(p.Results.layers_to_register)
                layers_to_register = p.Results.layer(p.Results.layer ~= p.Results.ref_layer);
            else
                layers_to_register = p.Results.layers_to_register;
            end
            
            for current_layer=layers_to_register
                switch p.Results.mov_img
                    case "merged-image"
                        mov_img = max(obj.images{current_layer}, [], 4);
                    case "single-channel"
                        current_metadata = obj.metadata{current_layer}.ChannelInfo;
                        ref_channel_index = find(contains({current_metadata(:).name}, p.Results.ref_channel) == 1);
                        mov_img = obj.images{current_layer}(:,:,:,ref_channel_index);
                    case "input_image"
                        mov_img = p.Results.input_image_mov;
                end

                if p.Results.scale ~= 1
                    mov_img = imresize3(mov_img, p.Results.scale);
                end

                starting = tic;
                [obj.images{current_layer}, obj.registration{current_layer}] = RegisterImagesGlobal(obj.images{current_layer},...
                                                                                    obj.registration{p.Results.ref_layer},...
                                                                                    mov_img,...
                                                                                    p.Results.scale);
    
                if iscell(current_layer)
                    current_layer = current_layer{1};
                end
                fprintf(sprintf('%s vs. %s finished [time=%02f]\n', current_layer, p.Results.ref_layer, toc(starting)));
                fprintf(sprintf('Shifted by %s\n', num2str(obj.registration{current_layer}.shifts)));
            end

            if p.Results.save_shifts
                shift_log_folder = fullfile(obj.outputPath, "log", "gr_shifts");
                if ~exist(shift_log_folder, 'dir')
                    mkdir(shift_log_folder);
                end
                
                if p.Results.log_suffix ~= ""
                    current_fname = fullfile(shift_log_folder, sprintf("%s_%s.txt", obj.fovID, p.Results.log_suffix));
                else
                    current_fname = fullfile(shift_log_folder, sprintf("%s.txt", obj.fovID));
                end
                if exist(current_fname, 'file'); delete(current_fname); end

                current_output_folder_msg = strrep(current_fname, '\', '\\');
                fprintf(sprintf('Saving shifts to %s\n', current_output_folder_msg));

                headers = ["fov_id" "round" "row" "col" "z"];
                shifts_to_save = [];
                for current_layer=layers_to_register
                    current_shifts = obj.registration{current_layer}.shifts;
                    current_shifts = [repmat(string(obj.fovID), size(current_shifts, 1), 1) repmat(string(current_layer), size(current_shifts, 1), 1) string(current_shifts)];
                    shifts_to_save = [shifts_to_save; current_shifts];
                end

                writematrix(headers, current_fname, 'Delimiter', ',');
                writematrix(shifts_to_save, current_fname, 'Delimiter', ',', 'WriteMode', 'append');
            end

            % change metadata
            obj.jobFinished.GlobalRegistration = true;
            
        end
        

        % 7.Local (Non-rigid) registration test
        function obj = LocalRegistration( obj, varargin )

            % Input parser
            p = inputParser;

            % Defaults
            defaultLayer = obj.layers.seq; 
            addOptional(p, 'layer', defaultLayer);

            defaultRefLayer = obj.layers.ref; % round1
            addOptional(p, 'ref_layer', defaultRefLayer);
            
            defaultRefChannel = "DAPI"; 
            addOptional(p, 'ref_channel', defaultRefChannel);

            defaultRefImg= "merged-image"; % single-image
            addOptional(p, 'ref_img', defaultRefImg);

            defaultMovImg= "merged-image"; % single-image
            addOptional(p, 'mov_img', defaultMovImg);

            defaultInputImage= ""; 
            addOptional(p, 'input_image', defaultInputImage);

            defaultIter = 10;
            addParameter(p, 'Iterations', defaultIter);

            defaultAFS = 1;
            addParameter(p, 'AccumulatedFieldSmoothing', defaultAFS);

            parse(p, varargin{:});
            
            fprintf('====Local (Non-rigid) Registration====\n');
            switch p.Results.ref_img
                case "merged-image"
                    obj.registration{p.Results.ref_layer} = max(obj.images{p.Results.ref_layer}, [], 4);
                case "single-channel"
                    obj.registration{p.Results.ref_layer} = obj.images{p.Results.ref_layer}(:,:,:,p.Results.ref_channel);
                case "input_image"
                    obj.registration{p.Results.ref_layer} = p.Results.input_image;
            end

            layers_to_register = p.Results.layer(p.Results.layer ~= p.Results.ref_layer);
            for current_layer=layers_to_register
                switch p.Results.mov_img
                    case "merged-image"
                        mov_img = max(obj.images{current_layer}, [], 4);
                    case "single-channel"
                        current_metadata = obj.metadata{current_layer}.ChannelInfo;
                        ref_channel_index = find(contains([current_metadata(:).name], p.Results.ref_channel) == 1);
                        mov_img = obj.images{current_layer}(:,:,:,ref_channel_index);
                end

                starting = tic;
                [obj.images{current_layer}, ~] = RegisterImagesLocal(obj.images{current_layer}, ...
                                                                                    obj.registration{p.Results.ref_layer}, ...
                                                                                    mov_img, ...
                                                                                    p.Results.Iterations, ...
                                                                                    p.Results.AccumulatedFieldSmoothing);
    
                fprintf(sprintf('%s vs. %s finished [time=%02f]\n', current_layer, p.Results.ref_layer, toc(starting)));
            end

            % change metadata
            obj.jobFinished.LocalRegistration = true;
            
        end
        
        
    % ====Reads Calling====

        % 7.Spot finding
        function obj = SpotFinding( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultRefLayer = obj.layers.ref; % round1
            addOptional(p, 'ref_layer', defaultRefLayer);

            defaultMethod = "max3d";
            addOptional(p, 'method', defaultMethod);

            defaultIntensityEstimation = "adaptive";
            addOptional(p, 'intensity_estimation', defaultIntensityEstimation);

            defaultIntensityThreshold = 0.2;
            addOptional(p, 'intensity_threshold', defaultIntensityThreshold);
            
            parse(p, varargin{:});
            
            fprintf('====Spot Finding====\n');
            fprintf(sprintf('Method: %s\n', p.Results.method));
            fprintf(sprintf('Reference: %s\n', p.Results.ref_layer));
            
            tic;
            switch p.Results.method
                case "max3d"
                    obj.signal.allSpots = SpotFindingMax3D(obj.images{p.Results.ref_layer}, p.Results.intensity_estimation, p.Results.intensity_threshold);
            end
            fprintf(sprintf('Number of spots found: %d\n', size(obj.signal.allSpots, 1)));
            fprintf(sprintf('[time = %.2f s]\n', toc));

            if size(obj.signal.allSpots, 1) == 0
                obj.signal.allSpots = cell2table(cell(0,4), 'VariableNames', ["x", "y", "z", "gene"]);
            else
                obj.signal.allSpots = splitvars(obj.signal.allSpots, "Centroid", 'NewVariableNames', ["x", "y", "z"]);
            end
            obj.jobFinished.SpotFinding = true;
        end
        
        
        % 8.Reads extraction
        function obj = ReadsExtraction( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultLayer = obj.layers.seq; 
            addOptional(p, 'layer', defaultLayer);

            defaultvoxelSize = [2 2 1];
            addParameter(p, 'voxel_size', defaultvoxelSize);

            parse(p, varargin{:});
            
            fprintf('====Reads Extraction====\n');
            fprintf(sprintf('voxel size: %d x %d x %d\n', p.Results.voxel_size));

            if ~isempty(obj.signal.allSpots)
                complete_color_seq = [];
                for current_layer=p.Results.layer
                    tic;
                    fprintf(sprintf("Processing %s...", current_layer))
                    [current_color_seq, current_color_score] = ExtractFromLocation( obj.images{current_layer}, obj.signal.allSpots, p.Results.voxel_size ); 
                    obj.signal.allSpots{:, sprintf("%s_color", current_layer)} = current_color_seq; 
                    obj.signal.allSpots{:, sprintf("%s_score", current_layer)} = current_color_score; 
                    if isempty(complete_color_seq)
                        complete_color_seq = current_color_seq;
                    else
                        % complete_color_seq = strcat(complete_color_seq, current_color_seq);
                        complete_color_seq = complete_color_seq + current_color_seq;
                    end
                    fprintf(sprintf('[time = %.2f s]\n', toc));
                end                                          
                
                obj.signal.allSpots{:, "color_seq"} = complete_color_seq;
            end

            obj.jobFinished.ReadsExtraction = true;

        
        end
        
        
        % 9.Load codebook
        function obj = LoadCodebook( obj, varargin )
        
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultdInputPath = obj.inputPath;
            addParameter(p, 'input_path', defaultdInputPath);

            defaultReverse = true;
            addParameter(p, 'do_reverse', defaultReverse);

            defaultSplitIndex = [];
            addParameter(p, 'split_index', defaultSplitIndex);
            
            parse(p, varargin{:});
            
            fprintf('====Load Codebook====\n');
            fprintf(sprintf('reverse: %d\n', p.Results.do_reverse));
            % load hash tables of gene name -> seq and seq -> gene name
            % where 'seq' is the string representation of the barcode in colorspace
            [obj.codebook.geneToSeq, obj.codebook.seqToGene] = LoadCodebook(p.Results.input_path, p.Results.split_index, p.Results.do_reverse);  

            % change metadata
            obj.jobFinished.LoadCodebook = true;
        
        end
        
        
        % 10.Reads filtration
        function obj = ReadsFiltration( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultthreshold = 0.5;
            addParameter(p, 'q_score_thershold', defaultthreshold);

            defaultNBarcodeSegments = 1;
            addParameter(p, 'n_barcode_segments', defaultNBarcodeSegments);

            defaultendBases = ["CC"]; % multiple endbase pairs ["CC", "CT"]
            addParameter(p, 'end_base', defaultendBases);

            defaultSplitIndex = [5];
            addParameter(p, 'split_index', defaultSplitIndex);

            defaultSaveScores = true;
            addParameter(p, 'save_scores', defaultSaveScores);

            parse(p, varargin{:});
            
            fprintf('====Reads Filtration====\n');
            end_base = string(p.Results.end_base);

            % remove reads with incorrect colors
            n_spots = size(obj.signal.allSpots, 1);
            no_color_spots = contains(obj.signal.allSpots.color_seq, "N");
            multi_color_spots = contains(obj.signal.allSpots.color_seq, "M");
            to_keep = ~or(no_color_spots, multi_color_spots);
            obj.signal.allSpots = obj.signal.allSpots(to_keep, :); 
            fprintf(sprintf('Number of spots were dropped because of no color: %d\n', sum(no_color_spots)));
            fprintf(sprintf('Number of spots were dropped because of multi-max color: %d\n', sum(multi_color_spots)));
            fprintf('Comparing with codebook...\n');
            fprintf(sprintf('Number of barcode segments: %d\n', p.Results.n_barcode_segments));
            obj.signal.scores = [n_spots sum(no_color_spots) sum(multi_color_spots)]; 

            if ~isempty(obj.signal.allSpots)
                if numel(end_base) > 1
                    end_base_msg = strjoin(end_base, " or ");
                else
                    end_base_msg = end_base;
                end
                fprintf(sprintf('Barcode ends with: %s\n', end_base_msg));

                if p.Results.n_barcode_segments == 1
                    obj = FilterReads(obj, end_base);  
                else
                    obj = FilterReadsMultiSegment(obj, end_base, p.Results.split_index);
                end

                n_good_spots = size(obj.signal.goodSpots, 1);
                obj.signal.scores = [obj.signal.scores n_good_spots];
            else
                obj.signal.goodSpots = cell2table(cell(0,4), 'VariableNames', ["x", "y", "z", "gene"]);
                obj.signal.scores = [obj.signal.scores 0 0 0 0];
            end

            if p.Results.save_scores
                score_log_folder = fullfile(obj.outputPath, "log", "sf_scores");
                if ~exist(score_log_folder, 'dir')
                    mkdir(score_log_folder);
                end

                if obj.subtile.index > 0
                    current_fname = fullfile(score_log_folder, sprintf("%s_%s.txt", obj.fovID, string(obj.subtile.index)));
                    if exist(current_fname, 'file'); delete(current_fname); end
                    current_output_folder_msg = strrep(current_fname, '\', '\\');
                    fprintf(sprintf('Saving scores to %s\n', current_output_folder_msg));

                    if numel(end_base) > 1
                        headers = ["fov_id" "subtile_id" "total_spots" "no_color" "multi_color" "spots_in_codebook" "correctform_1" "correctform_2" "good_spots"];
                    else
                        headers = ["fov_id" "subtile_id" "total_spots" "no_color" "multi_color" "spots_in_codebook" "spots_in_correctform" "correctform_in_codebook" "good_spots"];
                    end
                    scores_to_save = [string(obj.fovID) string(obj.subtile.index) string(obj.signal.scores)];
                else
                    current_fname = fullfile(score_log_folder, sprintf("%s.txt", obj.fovID));
                    if exist(current_fname, 'file'); delete(current_fname); end
                    current_output_folder_msg = strrep(current_fname, '\', '\\');
                    fprintf(sprintf('Saving scores to %s\n', current_output_folder_msg));

                    if numel(end_base) > 1
                        headers = ["fov_id" "total_spots" "no_color" "multi_color" "spots_in_codebook" "correctform_1" "correctform_2" "good_spots"];
                    else
                        headers = ["fov_id" "total_spots" "no_color" "multi_color" "spots_in_codebook" "spots_in_correctform" "correctform_in_codebook" "good_spots"];
                    end
                    scores_to_save = [string(obj.fovID) string(obj.signal.scores)];
                end

                writematrix(headers, current_fname, 'Delimiter', ',');
                writematrix(scores_to_save, current_fname, 'Delimiter', ',', 'WriteMode', 'append');

            end

            % change metadata
            obj.jobFinished.ReadsFiltration = true;
            
        end


        % 11.Preview reads
        function obj = ViewSignal( obj, varargin )
    
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultSlot = "goodSpots";
            addOptional(p, 'signal_slot', defaultSlot);

            if isConfigured(obj.registration)
                if isKey(obj.registration, obj.layers.ref)
                    defaultBgImg = max(obj.registration{obj.layers.ref}, [], 3);
                else
                    defaultBgImg = [];
                end
            else
                defaultBgImg = [];
            end
            addOptional(p, 'bg_img', defaultBgImg);

            defaultSpotsColor = "red";
            addOptional(p, 'spots_color', defaultSpotsColor);

            defaultSpotsZize = 1;
            addOptional(p, 'spots_size', defaultSpotsZize);

            defaultSave = false;
            addOptional(p, 'save', defaultSave);

            defaultOutputFolder = fullfile(obj.outputPath, "signal");
            addOptional(p, 'output_path', defaultOutputFolder);

            parse(p, varargin{:});
            
            if ~exist(p.Results.output_path, 'dir')
                mkdir(p.Results.output_path)
            end

            switch p.Results.signal_slot
                case "goodSpots"
                    signal_preview_img = PlotCentroids(obj.signal.goodSpots, p.Results.bg_img, p.Results.spots_color, p.Results.spots_size);
            
                    if p.Results.save
                        current_fname = fullfile(p.Results.output_path, sprintf("%s_goodSpots.png", obj.fovID));
                        current_output_folder_msg = strrep(current_fname, '\', '\\');
                        fprintf(sprintf('Saving goodSpots preview image to %s\n', current_output_folder_msg));
                        % fileattrib(p.Results.output_path, '+w');
                        exportgraphics(signal_preview_img, current_fname, 'Resolution', 300, 'ContentType', 'image');
                    end
                case "allSpots"
                    signal_preview_img = PlotCentroids(obj.signal.allSpots, p.Results.bg_img, p.Results.spots_color, p.Results.spots_size);
            
                    if p.Results.save
                        current_fname = fullfile(p.Results.output_path, sprintf("%s_allSpots.png", obj.fovID));
                        current_output_folder_msg = strrep(current_fname, '\', '\\');
                        fprintf(sprintf('Saving allSpots preview image to %s\n', current_output_folder_msg));
                        % fileattrib(p.Results.output_path, '+w');
                        exportgraphics(signal_preview_img, current_fname, 'Resolution', 300, 'ContentType', 'image');
                    end
            end

            % change metadata
            obj.jobFinished.ViewSignal = true;   
        
            
        end

        
        % 12.Save reads
        function obj = SaveSignal( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultSlot = "goodSpots";
            addOptional(p, 'signal_slot', defaultSlot);

            defaultOutputFolder = fullfile(obj.outputPath, "signal");
            addOptional(p, 'output_path', defaultOutputFolder);

            defaultFieldToKeep = ["x", "y", "z", "gene"];
            addOptional(p, 'field_to_keep', defaultFieldToKeep);

            parse(p, varargin{:});
            
            if ~exist(p.Results.output_path, 'dir')
                mkdir(p.Results.output_path)
            end

            switch p.Results.signal_slot
                case "goodSpots"
                    if obj.subtile.index > 0
                        current_fname = fullfile(obj.outputPath, "output", "subtile", obj.fovID, sprintf("subtile_goodSpots_%d.csv", obj.subtile.index));
                    else
                        current_fname = fullfile(p.Results.output_path, sprintf("%s_goodSpots.csv", obj.fovID));
                    end
                    current_output_folder_msg = strrep(current_fname, '\', '\\');
                    fprintf(sprintf('Saving goodSpots to %s\n', current_output_folder_msg));
                    if p.Results.field_to_keep == "all"
                        output_table = obj.signal.goodSpots;
                    else
                        output_table = obj.signal.goodSpots(:, p.Results.field_to_keep);
                    end
                    writetable(output_table, current_fname);
                case "allSpots"
                    if obj.subtile.index > 0
                        current_fname = fullfile(obj.outputPath, "output", "subtile", obj.fovID, sprintf("subtile_allSpots_%d.csv", obj.subtile.index));
                    else
                        current_fname = fullfile(p.Results.output_path, sprintf("%s_allSpots.csv", obj.fovID));
                    end
                    current_output_folder_msg = strrep(current_fname, '\', '\\');
                    fprintf(sprintf('Saving allSpots to %s\n', current_output_folder_msg));
                    if p.Results.field_to_keep == "all"
                        output_table = obj.signal.allSpots;
                    else
                        output_table = obj.signal.allSpots(:, p.Results.field_to_keep);
                    end
                    writetable(output_table, current_fname);
            end

            % change metadata
            obj.jobFinished.SaveSignal = true;   
        
            
        end
        
        % 13.Create subtiles
        function obj = CreateSubtiles( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultSqrtPieces = 4;
            addOptional(p, 'sqrt_pieces', defaultSqrtPieces);

            defaultOverlapRatio = 0.1;
            addOptional(p, 'overlap_ratio', defaultOverlapRatio);

            defaultSave = false;
            addOptional(p, 'save', defaultSave);

            defaultOutputFolder = fullfile(obj.outputPath, "output");
            addOptional(p, 'output_path', defaultOutputFolder);

            defaultRefLayer = obj.layers.ref;
            addOptional(p, 'ref_layer', defaultRefLayer);

            parse(p, varargin{:});
            
            if ~exist(p.Results.output_path, 'dir')
                mkdir(p.Results.output_path)
            end

            current_subtile_folder = fullfile(p.Results.output_path, "subtile");
            if ~exist(current_subtile_folder, 'dir')
                mkdir(current_subtile_folder)
            end

            current_metadata = obj.metadata{p.Results.ref_layer};
            obj.subtile.coords = MakeSubtileTable(current_metadata.dims, p.Results.sqrt_pieces, p.Results.overlap_ratio);     

            if p.Results.save
                current_output_folder = fullfile(current_subtile_folder, obj.fovID);
                if ~exist(current_output_folder, 'dir')
                    mkdir(current_output_folder)
                end
                % save the coords of subtiles
                writetable(obj.subtile.coords, fullfile(current_output_folder, "subtile_coords.csv"), 'Delimiter',',', 'QuoteStrings',false);

                % save the object of subtiles
                for i=1:size(obj.subtile.coords)
                    tile_idx = obj.subtile.coords.t(i);
                    start_coords_x = obj.subtile.coords.scoords_x(i);
                    start_coords_y = obj.subtile.coords.scoords_y(i);

                    end_coords_x = obj.subtile.coords.ecoords_x(i);
                    end_coords_y = obj.subtile.coords.ecoords_y(i);

                    subtile_data = obj;
                    for l=1:length(obj.layers.seq)
                        current_layer = obj.layers.seq{l};
                        subtile_data.images{current_layer} = subtile_data.images{current_layer}(start_coords_y:end_coords_y, start_coords_x:end_coords_x, :, :);
                    end
                    subtile_data.subtile.index = tile_idx;
                    
                    save(fullfile(current_output_folder, sprintf('subtile_data_%d.mat', tile_idx)), "subtile_data", "-v7.3");
                end

            end

            % change metadata
            obj.jobFinished.CreateSubtiles = true;   
        
            
        end

    
    end
    
    
end


