    classdef STARMapDataset
    % STARMapDataset, the primary class for the STARMap imaging analysis pipeline
    % ====Properties====
    % ...
    
    % ====Methods====
    % ...
    
    properties
        
        % IO
        inputPath;
        outputPath;
        fovID;
        
        % GPU
        useGPU;
        
        % Images 
        images;

        rawImages;
        registeredImages;
        additionalImages;

        projections;
        projectionImages;

        metadata;
        dims;
        dimX;
        dimY;
        dimZ;
        Nchannel;
        Nround;
        seqChannelOrderDict;
        addChannelOrderDict;
        
        % Registration
        registration;
        referenceImageSeq;
        globalParamsSeq;
        referenceImageAdd;
        globalParamsAdd;        
        
        % Spots
        signal;

        allSpots;
        goodSpots;
        
        % Reads
        allReads;
        goodReads;
        goodReadsLoc;
        allScores;
        goodScores;
        
        % Codebook
        codebook;
        seqToGene;
        geneToSeq;
        barcodeMat;
        barcodeNames;
        barcodeSeqs;
        basecsMat;
        CodebookSplitIndex;
        
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
            obj.images = containers.Map();
            obj.projections = containers.Map();
            obj.metadata = containers.Map();

            % show message
            fprintf('Pipeline Obj is generated...\n');
            
        end

        % 2.Load raw images 
        function obj = LoadRawImages( obj, varargin )

            % Input parser
            p = inputParser;

            defaultLayer = "seq"; % or others such as "protein", "round2"
            addOptional(p, 'layer', defaultLayer);
            
            defaultsubDir = '';
            addOptional(p, 'sub_dir', defaultsubDir);

            defaultotherDir = ["protein"];
            addOptional(p, 'additional_dir', defaultotherDir);

            defaultDict(1).wavelength = 488;
            defaultDict(1).channel = "ch00";
            defaultDict(1).name = "seq";

            defaultDict(2).wavelength = 594;
            defaultDict(2).channel = "ch01";
            defaultDict(2).name = "seq";

            defaultDict(3).wavelength = 546;
            defaultDict(3).channel = "ch02";
            defaultDict(3).name = "seq";

            defaultDict(4).wavelength = 647;
            defaultDict(4).channel = "ch03";
            defaultDict(4).name = "seq";

            addOptional(p, 'channel_order_dict', defaultDict);

            defaultzrange = [];
            addOptional(p, 'zrange', defaultzrange);

            defaultconvert = false;
            addOptional(p, 'convert_uint8', defaultconvert);

            defaultuseGPU = false;
            addOptional(p, 'useGPU', defaultuseGPU);
            parse(p, varargin{:});
            
            % Load tiff stacks
            if p.Results.layer == "seq"
                fprintf('====Loading sequencing raw images====\n');
                round_dir = dir(strcat(obj.inputPath, 'round*'));
                [obj.images("seq"), dims_seq] = LoadImageStacks(round_dir, ...
                                                                p.Results.sub_dir, ...
                                                                p.Results.channel_order_dict, ...
                                                                p.Results.zrange, ...
                                                                p.Results.convert_uint8);
                obj.metadata("seq") = struct();
                obj.metadata("seq").dims = dims_seq;
                obj.metadata("seq").dimX = dims_seq(1);
                obj.metadata("seq").dimY = dims_seq(2);
                obj.metadata("seq").dimZ = dims_seq(3);
                obj.metadata("seq").Nchannel = dims_seq(4);
                obj.metadata("seq").Nround = dims_seq(5);
                obj.metadata("seq").BitDepth = dims_seq(6);
                obj.metadata("seq").fovID = p.Results.sub_dir;
                obj.metadata("seq").ChannelInfo = p.Results.channel_order_dict;
            else
                all_dir = dir(obj.inputPath);
                folder_names = {all_dir(:).name};
                round_dir = all_dir(ismember(folder_names, p.Results.additional_dir));
                for r=1:numel(round_dir)
                    current_round_dir = round_dir(r);
                    current_layer = current_round_dir.name;
                    fprintf(sprintf('====Loading %s images into layer %s====\n', current_layer, current_layer));
                    [obj.images(current_layer), dims_other] = LoadImageStacks(current_round_dir, ...
                        p.Results.sub_dir, ...
                        p.Results.channel_order_dict, ...
                        p.Results.zrange, ...
                        p.Results.convert_uint8);

                    current_metadata = struct();
                    current_metadata.dims = dims_other;
                    current_metadata.dimX = dims_other(1);
                    current_metadata.dimY = dims_other(2);
                    current_metadata.dimZ = dims_other(3);
                    current_metadata.Nchannel = dims_other(4);
                    current_metadata.Nround = dims_other(5);
                    current_metadata.BitDepth = dims_other(6);
                    current_metadata.fovID = p.Results.sub_dir;
                    current_metadata.ChannelInfo = p.Results.channel_order_dict;
                    obj.metadata(current_layer) = current_metadata;
                end
               
            end
            
        end
        
        
        % 2.5.Swap channels (2 & 3, optional)
        function obj = SwapChannels( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            defaultLayer = "seq"; 
            addOptional(p, 'layer', defaultLayer);
            
            defaultChannel_1 = 2;
            defaultChannel_2 = 3;
            addOptional(p, 'channel_1', defaultChannel_1);
            addOptional(p, 'channel_2', defaultChannel_2);
            parse(p, varargin{:});
            
            % swap channels
            fprintf('====Swap Channels====\n');
            fprintf(sprintf('Channel %d <==> Channel %d\n', p.Results.channel_1, p.Results.channel_2));
            obj.images(p.Results.layer) = SwapTwoChannels(obj.images(p.Results.layer), p.Results.channel_1, p.Results.channel_2);
            
            % change metadata
            obj.jobFinished.SwapChannels = true;
            
        end
        

    % ====Preprocessing====

        % 3.Enhance Contrast
        function obj = EnhanceContrast( obj, method, varargin )

            % Input parser
            p = inputParser;
            
            
            addRequired(p, 'method');

            defaultLayer = "seq"; 
            addOptional(p, 'layer', defaultLayer);

            defaultLow_in= 0.05;
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
                    obj.images(p.Results.layer) = MinMaxNorm(obj.images(p.Results.layer));
                case "regular"
                    fprintf("====Running imadjustn====\n");
                    for r=1:obj.metadata(p.Results.layer).Nround
                        tic
                        fprintf(sprintf("Processing Round %d...", r))
                        
                        for c=1:obj.Nchannel 
                            current_channel = obj.rawImages{r}(:, :, :, c);
                            current_channel_adjusted = imadjustn(current_channel,...
                                [p.Results.low_in p.Results.high_in],...
                                [p.Results.low_out p.Results.high_out],...
                                p.Results.gamma);
                            obj.rawImages{r}(:, :, :, c) = current_channel_adjusted;
                        end
                
                        fprintf(sprintf('[time = %.2f s]\n', toc));
                    end
            end


            % change metadata
            obj.jobFinished.EnhanceContrast = 1;
            obj.jobFinished.EnhanceContrastMethod = p.Results.method;
        end 
        

        % 4.Hitogram Equalization
        function obj = HistEqualize( obj, varargin )
            
            % Input parser
            p = inputParser;

            defaultReferenceChannel = 1;
            addParameter(p, 'reference_channel', defaultReferenceChannel);

            defaultReferenceRound = 1;
            addParameter(p, 'reference_round', defaultReferenceRound);

            defaultNbins = 64;
            addParameter(p, 'nbins', defaultNbins);

            parse(p, varargin{:});
            
            fprintf("====Histogram Equalization====\n");
            fprintf(sprintf('Reference: Round %d - Channel %d\n', p.Results.reference_round, p.Results.reference_channel));
            reference_image = obj.rawImages{p.Results.reference_round}(:,:,:,p.Results.reference_channel);

            for r=1:obj.Nround 
                for c=1:obj.Nchannel      
                    fprintf('Equalizing Round %d - Channel %d\n', r, c);
                    obj.rawImages{r}(:,:,:,c) = imhistmatchn(obj.rawImages{r}(:,:,:,c), reference_image, p.Results.nbins);
                end    
            end     

            obj.jobFinished.HistogramEqualization = 1;

        end


        % 5.Morphological reconstruction
        function obj = MorphRecon( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultRadius = 3;
            addOptional(p, 'radius', defaultRadius);

            parse(p, varargin{:});
            
            fprintf("====Morphological Reconstruction====\n");
            
            obj.rawImages = MorphologicalReconstruction(obj.rawImages, p.Results.radius);
            
            % change metadata
            obj.jobFinished.MorphologicalReconstruction = 1;
            
        end
        
        
        % 5.5 Whilte tophat
        function obj = Tophat( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultRadius = 3;
            addOptional(p, 'radius', defaultRadius);

            parse(p, varargin{:});
            
            fprintf("====Tophat Filtering====\n");
            % setup structure element
            se = strel('disk', p.Results.radius);

            for r=1:obj.Nround
                tic
                fprintf(sprintf("Processing Round %d...", r));

                for c=1:obj.Nchannel
                    current_channel = obj.rawImages{r}(:,:,:,c);
                    for z=1:obj.dimZ
                        current_channel(:,:,z) = imtophat(current_channel(:,:,z), se);
                    end
                    obj.rawImages{r}(:,:,:,c) = uint8(current_channel);
                end
                fprintf(sprintf('[time = %.2f s]\n', toc));
            end 

            % change metadata
            obj.jobFinished.Tophat = 1;
            
        end

        
        % Make Projections
        function obj = Projection( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultMethod = "max";
            addOptional(p, 'method', defaultMethod);

            defaultSlot = "raw";
            addOptional(p, 'image_slot', defaultSlot);

            parse(p, varargin{:});
            
            fprintf("====Generate Projection Images====\n");
            obj.projectionImages = containers.Map();

            switch p.Results.image_slot
                case "raw"
                    obj.projectionImages("raw") = MakeProjections(obj.rawImages, p.Results.method);
                case "registered"
                    obj.projectionImages("registered") = MakeProjections(obj.registeredImages, p.Results.method);
                case "add"
                    obj.projectionImages("additional") = MakeProjections(obj.additionalImages, p.Results.method);
            end
            
        end


        % View Projections
        function obj = ViewProjection( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultSlot = "raw";
            addOptional(p, 'image_slot', defaultSlot);

            defaultEnhance = false;
            addOptional(p, 'enhance_contrast', defaultEnhance);

            defaultSave = false;
            addOptional(p, 'save', defaultSave);

            defaultOutputPath = fullfile(obj.outputPath, 'projection_montage.tif');
            addOptional(p, 'output_path', defaultOutputPath);

            parse(p, varargin{:});

            switch p.Results.image_slot
                case "raw"
                    montage_img = MakeMontage(obj.projectionImages("raw"), obj.Nround, obj.Nchannel, p.Results.enhance_contrast);
                case "registered"
                    montage_img = MakeMontage(obj.projectionImages("registered"), obj.Nround, obj.Nchannel, p.Results.enhance_contrast);
                case "add"
                    montage_img = MakeMontage(obj.projectionImages("additional"), obj.Nround, obj.Nchannel, p.Results.enhance_contrast);
            end
            
            if p.Results.save
                current_output_folder_msg = strrep(p.Results.output_path, '\', '\\');
                fprintf(sprintf('Saving %s projections to %s\n', p.Results.image_slot, current_output_folder_msg));
                exportgraphics(montage_img, p.Results.output_path, 'Resolution', 300);
            end

        end

        
        % Save Stack Images
        function obj = SaveImages( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultSlot = "raw";
            addOptional(p, 'image_slot', defaultSlot);

            defaultFormat = "nested";
            addOptional(p, 'folder_format', defaultFormat);

            defaultGroupChannel = true;
            addOptional(p, 'group_channel', defaultGroupChannel);

            defaultOutputPath = obj.outputPath;
            addOptional(p, 'output_path', defaultOutputPath);

            parse(p, varargin{:});

            switch p.Results.image_slot
                case "raw"
                    current_output_folder = fullfile(p.Results.output_path, "raw_images/");
                    if ~exist(current_output_folder, 'dir')
                        mkdir(current_output_folder)
                    end
                    current_output_folder_msg = strrep(current_output_folder, '\', '\\');
                    fprintf(sprintf('Saving %s images to %s\n', p.Results.image_slot, current_output_folder_msg));
                    switch p.Results.folder_format
                        case "nested"
                            SaveImageNestedFolder(obj.rawImages, current_output_folder, obj.fovID);
                        case "single"
                            SaveImageSingleFolder(obj.rawImages, current_output_folder, obj.fovID, p.Results.group_channel, obj.seqChannelOrderDict);
                    end
                case "registered"
                    current_output_folder = fullfile(p.Results.output_path, "registered_images/");
                    if ~exist(current_output_folder, 'dir')
                        mkdir(current_output_folder)
                    end
                    current_output_folder_msg = strrep(current_output_folder, '\', '\\');
                    fprintf(sprintf('Saving %s images to %s\n', p.Results.image_slot, current_output_folder_msg));
                    switch p.Results.folder_format
                        case "nested"
                            SaveImageNestedFolder(obj.registeredImages, current_output_folder, obj.fovID);
                        case "single"
                            SaveImageSingleFolder(obj.registeredImages, current_output_folder, obj.fovID, p.Results.group_channel, obj.seqChannelOrderDict);
                    end
                case "add"
                    current_output_folder = fullfile(p.Results.output_path, "additional_images/");
                    if ~exist(current_output_folder, 'dir')
                        mkdir(current_output_folder)
                    end
                    current_output_folder_msg = strrep(current_output_folder, '\', '\\');
                    fprintf(sprintf('Saving %s images to %s\n', p.Results.image_slot, current_output_folder_msg));
                    switch p.Results.folder_format
                        case "nested"
                            SaveImageNestedFolder(obj.additionalImages, current_output_folder, obj.fovID);
                        case "single"
                            SaveImageSingleFolder(obj.additionalImages, current_output_folder, obj.fovID, p.Results.group_channel, obj.addChannelOrderDict);
                    end
            end
            

        end


    % ====Image Registration====
            
        % 6.Global registration
        function obj = GlobalRegistration( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultRef = 1;
            addOptional(p, 'ref_round', defaultRef);

            parse(p, varargin{:});
        
            fprintf('====Global Registration====\n');
            obj.registeredImages = cell(obj.Nround, 1);
            obj.registeredImages{p.Results.ref_round} = obj.rawImages{p.Results.ref_round};

            fprintf(sprintf('Create reference image with round%d\n', p.Results.ref_round));
            if isempty(obj.referenceImageSeq)
                obj.referenceImageSeq = max(obj.rawImages{p.Results.ref_round}, [], 4);
            end

            rounds = 1:obj.Nround;
            rounds = rounds(rounds ~= p.Results.ref_round);

            for r=rounds
                mov_img = max(obj.rawImages{r}, [], 4);
                starting = tic;
                [obj.registeredImages{r}, obj.globalParamsSeq] = RegisterImagesGlobal(obj.rawImages{r}, obj.referenceImageSeq, mov_img);
                fprintf(sprintf('Round %d vs. Round %d finished [time=%02f]\n', r, p.Results.ref_round, toc(starting)));
                fprintf(sprintf('Shifted by %s\n', num2str(obj.globalParamsSeq.shifts)));
            end

            % change metadata
            obj.jobFinished.GlobalRegistration = 1;
            
        end
        

        % 7.Local (Non-rigid) registration test
        function obj = LocalRegistration( obj, varargin )

            % Input parser
            p = inputParser;

            % Defaults
            defaultRef = 1;
            addOptional(p, 'ref_round', defaultRef);

            defaultIter = 10;
            addParameter(p, 'Iterations', defaultIter);

            defaultAFS = 1;
            addParameter(p, 'AccumulatedFieldSmoothing', defaultAFS);

            parse(p, varargin{:});
            
            fprintf('====Local (Non-rigid) Registration====\n');

            fprintf(sprintf('Create reference image with round%d\n', p.Results.ref_round));
            if isempty(obj.referenceImageSeq)
                obj.referenceImageSeq = max(obj.rawImages{p.Results.ref_round}, [], 4);
            end

            rounds = 1:obj.Nround;
            rounds = rounds(rounds ~= p.Results.ref_round);

            for r=rounds
                mov_img = max(obj.registeredImages{r}, [], 4);
                starting = tic;
                [obj.registeredImages{r}, ~] = RegisterImagesLocal(obj.registeredImages{r}, obj.referenceImageSeq, mov_img, ...
                                                                        p.Results.Iterations, p.Results.AccumulatedFieldSmoothing);

                fprintf(sprintf('Round %d vs. Round %d finished [time=%02f]\n', r, p.Results.ref_round, toc(starting)));
            end

            % change metadata
            obj.jobFinished.LocalRegistration = [1 floor(log2(obj.dimZ)) p.Results.AccumulatedFieldSmoothing];
            
        end


         % 6.Global registration with additional reference 
         function obj = GlobalRegistrationAdd( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            % Defaults
            if ~isempty(obj.addChannelOrderDict)
                channel_names = {obj.addChannelOrderDict(:).name};
                defaultRef = find([channel_names{:}] == "DAPI");
            else
                defaultRef = 1;
            end
            addOptional(p, 'ref_channel', defaultRef);

            parse(p, varargin{:});
        
            fprintf('====Global Registration with Additional Reference====\n');
            Nround_add = numel(obj.additionalImages);

            fprintf(sprintf('Using ch%d as reference...\n', p.Results.ref_channel));
            
            if isempty(obj.referenceImageAdd)
                fprintf("Need a reference image stored in sdata.referenceImageAdd slot!")
            end

            for r=1:Nround_add
                mov_img = obj.additionalImages{r}(:,:,:,p.Results.ref_channel);
                starting = tic;
                [obj.additionalImages{r}, obj.globalParamsAdd] = RegisterImagesGlobal(obj.additionalImages{r}, obj.referenceImageAdd, mov_img);
                fprintf(sprintf('Round %d finished [time=%02f]\n', r, toc(starting)));
                fprintf(sprintf('Shifted by %s\n', num2str(obj.globalParamsAdd.shifts)));
            end

            % change metadata
            obj.jobFinished.GlobalRegistrationAdd = 1;
            
        end
        
        
    % ====Reads Calling====

        % 7.Spot finding
        function obj = SpotFinding( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultMethod = "max3d";
            defaultrefIndex = 1;
            defaultfsize = [5 5 3];
            defaultfsigma = 1;
            defaultqualityThreshold = 0.7;
            defaultvolumeThreshold = 10;
            defaultbarcodeMethod = "image";
            defaultshowPlots = true;

            addOptional(p,'Method',defaultMethod);
            addParameter(p, 'ref_index', defaultrefIndex);
            addParameter(p, 'fsize', defaultfsize);
            addParameter(p, 'fsigma', defaultfsigma);
            addParameter(p, 'qualityThreshold', defaultqualityThreshold);
            addParameter(p, 'volumeThreshold', defaultvolumeThreshold);
            addParameter(p, 'barcodeMethod', defaultbarcodeMethod);
            addParameter(p, 'showPlots', defaultshowPlots);
            
            parse(p, varargin{:});
            
            
            fprintf('====Spot Finding====\n');
            fprintf(sprintf('Method: %s\n', p.Results.Method));
            fprintf(sprintf('Reference round: %d\n', p.Results.ref_index));
            
            tic
            switch p.Results.Method
                case "max3d"
                    obj.allSpots = SpotFindingMax3D(obj.registeredImages, p.Results.ref_index);
                    obj.jobFinished.SpotFinding = [1 p.Results.Method];
                case "ex_max3d"
                    obj.allSpots = SpotFindingExtendedMax3D(obj.registeredImages);
                    obj.jobFinished.SpotFinding = [1 p.Results.Method];
                case "log3d"
                    obj.allSpots = SpotFindingLog3D(obj.registeredImages, p.Results.fsize, p.Results.fsigma);
                    obj.jobFinished.SpotFinding = [1 p.Results.Method p.Results.fsize p.Results.fsigma];
                case "barcode"
                    obj.allSpots = SpotFindingBarcode(obj.registeredImages, ...
                        obj.seqToGene, ...
                        p.Results.qualityThreshold, ...
                        p.Results.volumeThreshold, ...
                        p.Results.barcodeMethod, ...
                        p.Results.showPlots, ...
                        obj.useGPU);
                    obj.jobFinished.SpotFinding = [1 p.Results.Method ...
                        p.Results.qualityThreshold ...
                        p.Results.volumeThreshold ...
                        p.Results.barcodeMethod, ...
                        p.Results.showPlots ...
                        ];
                case "barcode_test"
                    [obj.allReads, obj.allSpots, obj.allScores, obj.basecsMat] = test_SpotFindingBarcode(obj.registeredImages, ...
                        obj.seqToGene, ...
                        p.Results.qualityThreshold, ...
                        p.Results.volumeThreshold, ...
                        p.Results.showPlots, ...
                        obj.useGPU);
                    obj.jobFinished.SpotFinding = [1 p.Results.Method ...
                        p.Results.qualityThreshold ...
                        p.Results.volumeThreshold ...
                        p.Results.barcodeMethod, ...
                        p.Results.showPlots ...
                        ];
                case "will"
                    obj.allSpots = SpotFindingWill(obj.registeredImages);
                    obj.jobFinished.SpotFinding = [1 p.Results.Method];
            end
            fprintf(sprintf('Number of spots found by %s: %d\n', p.Results.Method, size(obj.allSpots, 1)));
            fprintf(sprintf('[time = %.2f s]\n', toc));
            
            if ~isempty(obj.log)
                fprintf(obj.log, '====Spot Finding====\n');
                fprintf(obj.log, sprintf('Method: %s\n', p.Results.Method));
                fprintf(obj.log, sprintf('Reference round: %d\n', p.Results.ref_index));
                fprintf(obj.log, sprintf('Number of spots found by %s: %d\n', p.Results.Method, size(obj.allSpots, 1)));
            end
            
        end
        
        
        % 8.Reads extraction
        function obj = ReadsExtraction( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultvoxelSize = [3 3 1];

            addParameter(p, 'voxelSize', defaultvoxelSize);


            parse(p, varargin{:});
            
            fprintf('====Reads Extraction====\n');
            fprintf(sprintf('voxel size: %d x %d x %d\n', p.Results.voxelSize));
            
            obj = ExtractFromLocation( obj, p.Results.voxelSize ); 
                                                                                
            obj.jobFinished.ReadsExtraction = [1 p.Results.voxelSize];
        
        end
        
        
        % 9.Load codebook
        function obj = LoadCodebook( obj, varargin )
        
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultdoReverse = true;
            defaultremoveIndex = [];

            addParameter(p, 'remove_index', defaultremoveIndex);
            addParameter(p, 'doReverse', defaultdoReverse);

            parse(p, varargin{:});
            
            fprintf('====Load Codebook====\n');
            fprintf(sprintf('doReverse: %d\n', p.Results.doReverse));
            % load hash tables of gene name -> seq and seq -> gene name
            % where 'seq' is the string representation of the barcode in colorspace
            [obj.geneToSeq, obj.seqToGene] = new_LoadCodebook(obj.inputPath, p.Results.remove_index, p.Results.doReverse);  
            obj.CodebookSplitIndex = p.Results.remove_index;
            
            seqStrs = obj.seqToGene.keys;
            seqCS = []; % color sequences in matrix form for computing hamming distances ie: Nbarcode x Nround double
            for i=1:numel(seqStrs)
                % seqStrs{i}
                seqCS(end+1, :) = Str2Colorseq(seqStrs{i});
            end
            obj.barcodeMat = seqCS;
            obj.barcodeNames = obj.seqToGene.values; % cell array of seq names
            obj.barcodeSeqs = obj.seqToGene.keys; % str color seqs
            
            % change metadata
            obj.jobFinished.LoadCodebook = 1;
        
        end
        
        
        % 10.Reads filtration
        function obj = ReadsFiltration( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultthreshold = 0.5;
            defaultmode = "regular";
            defaultendBases = ['C', 'C'];
            defaultsplitLoc = 4;
            defaultshowPlots = true;
            defaultendBasesMix = ['C', 'A'];

            addParameter(p, 'q_score_thers', defaultthreshold);
            addParameter(p, 'mode', defaultmode);
            addParameter(p, 'endBases', defaultendBases);
            addParameter(p, 'split_loc', defaultsplitLoc);
            addParameter(p, 'showPlots', defaultshowPlots);
            addParameter(p, 'endBases_mix', defaultendBasesMix);

            parse(p, varargin{:});
            
            fprintf('====Reads Filtration====\n');
            fprintf(sprintf('mode: %s\n', p.Results.mode));
            fprintf(sprintf('Base in both ends: %s --- %s\n', p.Results.endBases(1), p.Results.endBases(2)));

            switch p.Results.mode
                case "regular"
                    obj = new_FilterReads(obj, p.Results.endBases, p.Results.q_score_thers, p.Results.showPlots);  
                case "duo"
                    obj = new_FilterReads_Duo(obj, p.Results.endBases, p.Results.split_loc, p.Results.q_score_thers, p.Results.showPlots);
                case "mix"
                    obj = new_FilterReads_Mix(obj, p.Results.endBases, p.Results.endBases_mix, p.Results.showPlots);
                    
            end
            
            % change metadata
            obj.jobFinished.ReadsFiltration = [1 p.Results.mode];
            
        end
        
        
        % 11.Save reads
        function obj = SaveReads( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultinputId = '';

            addOptional(p, 'inputId', defaultinputId);

            parse(p, varargin{:});
            
            fprintf('====Save Reads====\n');

%             obj.allCounts = SaveGoodReads( obj, p.Results.inputId );  
            
            % change metadata
            obj.jobFinished.SaveReads = 1;   
            
            
        end
       
    
    end


    % ====Other Methods====
    methods
        
        % 1.Load registered images
        function obj = LoadImages( obj, inputPath, varargin )

            % Input parser
            p = inputParser;
            
            defaultsubdir = '';
            defaultinputDim = [];
            defaultinputFormat = 'uint8';
            defaultzrange = '';
            defaultclass = "mat";
            defaultuseGPU = false;
            addOptional(p, 'sub_dir', defaultsubdir);
            addOptional(p, 'input_dim', defaultinputDim);
            addOptional(p, 'input_format', defaultinputFormat);
            addOptional(p, 'zrange', defaultzrange);
            addOptional(p, 'output_class', defaultclass);
            addOptional(p, 'useGPU', defaultuseGPU);
            parse(p, varargin{:});
            
            % Load tiff stacks from inputPath
            fprintf('====Loading raw images====\n');
            [obj.registeredImages, ~] = test_LoadImageStacks(inputPath, ...
                                                            p.Results.sub_dir, ...
                                                            p.Results.input_dim, ...
                                                            p.Results.input_format, ...
                                                            p.Results.zrange, ...
                                                            p.Results.output_class, ...
                                                            false);
            % change metadata
            obj.jobFinished.LoadRegisteredImages = 1;
            
        end
        
        
            
    end
    
    
    end
