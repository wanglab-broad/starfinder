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
        
        % Registration
        registration;
     
        % Spots
        signal;

        % Codebook
        codebook;

        % Metadata
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
            obj.layers = struct();
            obj.registration = dictionary();
            obj.signal = struct();
            obj.codebook = struct();
            
            obj.layers.seq = [];
            obj.layers.other = [];

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

            default_dirs = dir(strcat(obj.inputPath, 'round*'));
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
                    obj.images{current_layer} = imrotate(obj.images{current_layer}, p.Results.rotate_angl);
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
                exportgraphics(montage_img, p.Results.output_path, 'Resolution', 300);
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
                        SaveImageNestedFolder(obj.images(current_layer), current_layer, current_output_folder, obj.fovID);
                    case "single"
                        SaveImageSingleFolder(obj.images(current_layer), current_layer, current_output_folder, obj.fovID, p.Results.group_channel, obj.metadata{current_layer}.ChannelInfo);
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

            defaultRefChannel = "DAPI"; 
            addOptional(p, 'ref_channel', defaultRefChannel);

            defaultRefImg= "merged-image"; % single-channel input_image
            addOptional(p, 'ref_img', defaultRefImg);

            defaultMovImg= "merged-image"; % single-channel input_image
            addOptional(p, 'mov_img', defaultMovImg);

            defaultInputImage= ""; % 
            addOptional(p, 'input_image', defaultInputImage);

            parse(p, varargin{:});

            fprintf('====Global Registration====\n');
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
                [obj.images{current_layer}, obj.registration{current_layer}] = RegisterImagesGlobal(obj.images{current_layer},...
                                                                                    obj.registration{p.Results.ref_layer},...
                                                                                    mov_img);
    
                fprintf(sprintf('%s vs. %s finished [time=%02f]\n', current_layer, p.Results.ref_layer, toc(starting)));
                fprintf(sprintf('Shifted by %s\n', num2str(obj.registration{current_layer}.shifts)));
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

            defaultIntensityThreshold = 0.2;
            addOptional(p, 'intensity_threshold', defaultIntensityThreshold);
            
            parse(p, varargin{:});
            
            fprintf('====Spot Finding====\n');
            fprintf(sprintf('Method: %s\n', p.Results.method));
            fprintf(sprintf('Reference: %s\n', p.Results.ref_layer));
            
            tic;
            switch p.Results.method
                case "max3d"
                    obj.signal.allSpots = SpotFindingMax3D(obj.images{p.Results.ref_layer}, p.Results.intensity_threshold);
            end
            fprintf(sprintf('Number of spots found: %d\n', size(obj.signal.allSpots, 1)));
            fprintf(sprintf('[time = %.2f s]\n', toc));

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
            obj.signal.allSpots = splitvars(obj.signal.allSpots, "Centroid", 'NewVariableNames', ["x", "y", "z"]);


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

            defaultSplitIndex = [4];
            addParameter(p, 'split_index', defaultSplitIndex);

            parse(p, varargin{:});
            
            fprintf('====Reads Filtration====\n');

            % remove reads with incorrect colors
            no_color_spots = contains(obj.signal.allSpots.color_seq, "N");
            multi_color_spots = contains(obj.signal.allSpots.color_seq, "M");
            to_keep = ~or(no_color_spots, multi_color_spots);
            obj.signal.allSpots = obj.signal.allSpots(to_keep, :); 
            fprintf(sprintf('Number of spots were dropped because of no color: %d\n', sum(no_color_spots)));
            fprintf(sprintf('Number of spots were dropped because of multi-max color: %d\n', sum(multi_color_spots)));
            fprintf('Comparing with codebook...\n');
            fprintf(sprintf('Number of barcode segments: %s\n', p.Results.n_barcode_segments));

            if numel(p.Results.end_base) > 1
                end_base_msg = strjoin(p.Results.end_base, " or ");
            else
                end_base_msg = p.Results.end_base;
            end
            fprintf(sprintf('Barcode ends with: %s\n', end_base_msg));

            if p.Results.n_barcode_segments == 1
                obj = FilterReads(obj, p.Results.end_base);  
            else
                obj = FilterReadsMultiSegment(obj, p.Results.end_base, p.Results.split_index);
            end
            
            % change metadata
            obj.jobFinished.ReadsFiltration = true;
            
        end
        
        
        % 11.Save reads
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
                    current_fname = fullfile(p.Results.output_path, sprintf("%s_goodSpots.csv", obj.fovID));
                    current_output_folder_msg = strrep(current_fname, '\', '\\');
                    fprintf(sprintf('Saving goodSpots to %s\n', current_output_folder_msg));
                    output_table = obj.signal.goodSpots(:, p.Results.field_to_keep);
                    writetable(output_table, current_fname);
                case "allSpots"
                    current_fname = fullfile(p.Results.output_path, sprintf("%s_allSpots.csv", obj.fovID));
                    current_output_folder_msg = strrep(current_fname, '\', '\\');
                    fprintf(sprintf('Saving allSpots to %s\n', current_output_folder_msg));
                    output_table = obj.signal.allSpots(:, p.Results.field_to_keep);
                    writetable(output_table, current_fname);
            end

            % change metadata
            obj.jobFinished.SaveSignal = true;   
        
            
        end
       
    
    end
    
    
end


