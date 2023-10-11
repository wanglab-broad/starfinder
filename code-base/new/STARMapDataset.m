    classdef STARMapDataset
    % STARMapDataset, the primary class for the STARMap imaging analysis pipeline
    % ====Properties====
    % *IO*
    % inputPath: primary path of the input folder
    % outputPath: primary path of the input folder, usually inputPath/output
    
    % *GPU*
    % useGPU: state of GPU utilization
    
    % *Images*
    % rawImages: 5-D array raw cDNA amplicon images of multiple rounds 
    % registeredImages: 5-D array registered images
    % gpuImages: 5-D array registered images on GPU
    % dims: dimension of the rawImages
    % dimX
    % dimY
    % dimZ
    % Nchannel: number of channel of image
    % Nround: number of imaging round
    
    % *Spots*
    % allSpots: all spots/points found in the image
    % goodSpots: spots/points left after filtration
    
    % *Reads*
    % allReads: all reads extracted from all spots/points
    % goodReads: reads left after filtration 
    % goodReadsLoc: locations of good reads
    % allScores: scores of all reads
    % goodScores: scores of good reads

    % Codebook
    % seqToGene: map(dictionary) color sequence --> gene
    % geneToSeq: map(dictionary) gene --> color sequence
    % barcodeMat:
    % barcodeNames:
    % barcodeSeqs:
    % basecsMat:

    % Metadata
    % jobFinished:
    
    % ====Methods====
    % ...
    
    properties
        
        % IO
        inputPath;
        outputPath;
        
        % GPU
        useGPU;
        
        % Images 
        rawImages;
        registeredImages;
        gpuImages;
        proteinImages;
        projectionImages;
        dims;
        dimX;
        dimY;
        dimZ;
        Nchannel;
        Nround;
        
        % Spots
        allSpots;
        goodSpots;
        
        % Reads
        allReads;
        goodReads;
        goodReadsLoc;
        allScores;
        goodScores;
        
        % Codebook
        seqToGene;
        geneToSeq;
        barcodeMat;
        barcodeNames;
        barcodeSeqs;
        basecsMat;
        CodebookSplitIndex;
        
        % Metadata
        jobFinished;
        log;
        
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
            
            % setup metadata
            obj.jobFinished = struct('LoadRawImages', false);
            
            % show message
            fprintf('Pipeline Obj is generated...\n');
            
        end

        % 2.Load raw images 
        function obj = LoadRawImages( obj, varargin )

            % Input parser
            p = inputParser;
            
            defaultsubdir = '';
            addOptional(p, 'sub_dir', defaultsubdir);

            defaultDict(1).wavelength = 488;
            defaultDict(1).id = "ch00";
            defaultDict(2).wavelength = 594;
            defaultDict(2).id = "ch01";
            defaultDict(3).wavelength = 546;
            defaultDict(3).id = "ch02";
            defaultDict(4).wavelength = 647;
            defaultDict(4).id = "ch03";

            addOptional(p, 'channel_order_dict', defaultDict);

            defaultzrange = [];
            addOptional(p, 'zrange', defaultzrange);

            defaultconvert = false;
            addOptional(p, 'convert_uint8', defaultconvert);

            defaultuseGPU = false;
            addOptional(p, 'useGPU', defaultuseGPU);
            parse(p, varargin{:});
            
            % Load tiff stacks from inputPath
            fprintf('====Loading raw images====\n');
            [obj.rawImages, obj.dims] = LoadImageStacks(obj.inputPath, ...
                                                            p.Results.sub_dir, ...
                                                            p.Results.channel_order_dict, ...
                                                            p.Results.zrange, ...
                                                            p.Results.convert_uint8);

            obj.dimX = obj.dims(1);
            obj.dimY = obj.dims(2);
            obj.dimZ = obj.dims(3);
            obj.Nchannel = obj.dims(4);
            obj.Nround = obj.dims(5);

            % change metadata
            obj.jobFinished.LoadRawImages = true;
            obj.jobFinished.ImageFormat = class(obj.rawImages{1});
            
        end
        
        
        % 2.5.Swap channels (2 & 3, optional)
        function obj = SwapChannels( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            defaultChannel_1 = 2;
            defaultChannel_2 = 3;
            addOptional(p, 'channel_1', defaultChannel_1);
            addOptional(p, 'channel_2', defaultChannel_2);
            parse(p, varargin{:});
            
            % swap channels
            fprintf('====Swap Channels====\n');
            fprintf(sprintf('Channel %d <==> Channel %d\n', p.Results.channel_1, p.Results.channel_2));
            obj.rawImages = SwapTwoChannels(obj.rawImages, p.Results.channel_1, p.Results.channel_2);
            
            % change metadata
            obj.jobFinished.SwapChannels = [1 p.Results.channel_1 p.Results.channel_2];
            
        end
        
    % ====Preprocessing====

        % 3.Enhance Contrast
        function obj = EnhanceContrast( obj, method, varargin )

            % Input parser
            p = inputParser;
            
            addRequired(p, 'method');

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
                    obj.rawImages = MinMaxNorm(obj.rawImages);
                case "regular"
                    fprintf("====Running imadjustn====\n");
                    for r=1:obj.Nround
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
                case "protein"
                    obj.projectionImages("protein") = MakeProjections(obj.proteinImages, p.Results.method);
            end
            
        end

        % View Projections
        function obj = ViewProjection( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultSlot = "raw";
            addOptional(p, 'image_slot', defaultSlot);

            defaultSave = false;
            addOptional(p, 'save', defaultSave);

            defaultOutputPath = './projection_montage.tif';
            addOptional(p, 'output_path', defaultOutputPath);

            parse(p, varargin{:});

            switch p.Results.image_slot
                case "raw"
                    montage_img = MakeMontage(obj.projectionImages("raw"));
                case "registered"
                    montage_img = MakeMontage(obj.projectionImages("registered"));
                case "protein"
                    montage_img = MakeMontage(obj.projectionImages("protein"));
            end
            
            if p.Results.save
                imwrite(montage_img, p.Results.output_path);
            end

        end


    % ====Image Registration====

        % 5.Global registration
        function obj = GlobalRegistration( obj, varargin )
            
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultRef = 1;
            defaultnblocks = [1 1];
            defaultuseOverlay = false;

            addOptional(p,'ref_round',defaultRef);
            addOptional(p,'nblocks',defaultnblocks);
            addOptional(p,'useOverlay',defaultuseOverlay);
            parse(p, varargin{:});
            
            
            fprintf('====Global Registration====\n');
            fprintf(sprintf('Reference round: %d\n', p.Results.ref_round));
            fprintf(sprintf('Use overlay: %d\n', p.Results.useOverlay));
            
            obj.registeredImages = will_JointRegister3D(obj.rawImages, p.Results.ref_round, p.Results.nblocks, p.Results.useOverlay, obj.log);


            % change metadata
            obj.jobFinished.GlobalRegistration = [1 p.Results.ref_round, p.Results.nblocks p.Results.useOverlay];
            
        end
        
        
        % 5.5.Global registration
        function obj = test_GlobalRegistration( obj, varargin )
            % Input parser
            p = inputParser;
            
            % Defaults
            defaultRef = 1;
            defaultuseGPU = obj.useGPU;

            addOptional(p,'ref_round',defaultRef);
            addOptional(p, 'useGPU', defaultuseGPU);
            parse(p, varargin{:});
            
            
            fprintf('====Global Registration====\n');
            output_reg = zeros(size(obj.rawImages), 'uint8');
            output_reg(:,:,:,:,p.Results.ref_round) = obj.rawImages(:,:,:,:,p.Results.ref_round);
            
            rounds = 1:obj.Nround;
            rounds = rounds(rounds ~= p.Results.ref_round);
            
            if p.Results.useGPU
                
                for r=rounds
                    tic;
%                     ref_round = gpuArray(obj.rawImages(:,:,:,:,p.Results.ref_round));
%                     fix = max(ref_round, [], 4);

                    ref_round = obj.rawImages(:,:,:,:,p.Results.ref_round);
                    fix = gpuArray(max(ref_round, [], 4));
                    
%                     curr_round = gpuArray(obj.rawImages(:,:,:,:,r));
%                     curr_mov = max(curr_round, [], 4);
                    
                    curr_round = obj.rawImages(:,:,:,:,r);
                    curr_mov = gpuArray(max(curr_round, [], 4));

                    params = DFTRegister3D(fix, curr_mov, false);
                    % disp("DFTRegister success!");
                    for c=1:4
                        curr_reg = DFTApply3D(gpuArray(curr_round(:,:,:,c)), params, false);
                        curr_round(:,:,:,c) = curr_reg;
                    end

                    output_reg(:,:,:,:,r) = gather(curr_round);
                    fprintf(sprintf('Round %d vs. Round %d finished [time=%02f]\n', r, p.Results.ref_round, toc));
                    fprintf(obj.log, sprintf('Round %d vs. Round %d finished [time=%02f]\n', r, p.Results.ref_round, toc));
                    fprintf(sprintf('Shifted by %s\n', num2str(params.shifts)));
                    fprintf(obj.log, sprintf('Shifted by %s\n', num2str(params.shifts)));
                    reset(gpuDevice);
                end
               
            else
                for r=rounds
                    starting = tic;
                    ref_round = obj.rawImages(:,:,:,:,p.Results.ref_round);
                    fix = max(ref_round, [], 4);

                    curr_round = obj.rawImages(:,:,:,:,r);
                    curr_mov = max(curr_round, [], 4);

                    params = DFTRegister3D(fix, curr_mov, false);
                    % disp("DFTRegister success!");
                    fprintf(sprintf('DFT register finished [time=%02f]\n', toc(starting)));
                    
                    starting_apply = tic;
                    for c=1:4
                        curr_reg = DFTApply3D(curr_round(:,:,:,c), params, false);
                        curr_round(:,:,:,c) = curr_reg;
                    end
                    fprintf(sprintf('DFT apply finished [time=%02f]\n', toc(starting_apply)));
                    
                    output_reg(:,:,:,:,r) = curr_round;
                    fprintf(sprintf('Round %d vs. Round %d finished [time=%02f]\n', r, p.Results.ref_round, toc(starting)));
                    fprintf(obj.log, sprintf('Round %d vs. Round %d finished [time=%02f]\n', r, p.Results.ref_round, toc(starting)));
                    fprintf(sprintf('Shifted by %s\n', num2str(params.shifts)));
                    fprintf(obj.log, sprintf('Shifted by %s\n', num2str(params.shifts)));
                end
                
            end
            
            obj.registeredImages = output_reg;


            % change metadata
            obj.jobFinished.test_GlobalRegistration = 1;
            
        end
        
        % 6.Local (Non-rigid) registration
        function obj = LocalRegistration( obj, varargin )

            % Input parser
            p = inputParser;
            % Defaults
            defaultRef = 1;
            defaultMethod = "max";
            defaultIter = 60;
            defaultAFS = 1;

            %addRequired(p,'object');
            addOptional(p,'ref_round',defaultRef);
            addOptional(p,'Method',defaultMethod);
            addParameter(p,'Iterations',defaultIter);
            addParameter(p,'AccumulatedFieldSmoothing',defaultAFS);

            parse(p, varargin{:});
            
            % WARNING
            if obj.useGPU
                obj.gpuImages = obj.registeredImages;
                obj.registeredImages = gather(obj.registeredImages);
            else
                obj.gpuImages = obj.registeredImages;
            end
            
            fprintf('====Local (Non-rigid) Registration====\n');
            obj = new_LocalRegistration(obj, p.Results.ref_round, 'Method', p.Results.Method, 'Iterations', p.Results.Iterations, 'AccumulatedFieldSmoothing', p.Results.AccumulatedFieldSmoothing);

            
            % change metadata
            obj.jobFinished.LocalRegistration = [1 p.Results.Method floor(log2(obj.dimZ)) p.Results.AccumulatedFieldSmoothing];
            
        end
        
        % 6.1.Local (Non-rigid) registration test
        function obj = xxx_LocalRegistration( obj, varargin )

            % Input parser
            p = inputParser;
            % Defaults
            defaultRef = 1;
            defaultMethod = "max";
            defaultIter = 60;
            defaultAFS = 1;

            %addRequired(p,'object');
            addOptional(p,'ref_round',defaultRef);
            addOptional(p,'Method',defaultMethod);
            addParameter(p,'Iterations',defaultIter);
            addParameter(p,'AccumulatedFieldSmoothing',defaultAFS);

            parse(p, varargin{:});
            
            fprintf('====Local (Non-rigid) Registration====\n');
            obj = test_LocalRegistration(obj, p.Results.ref_round, 'Method', p.Results.Method, 'Iterations', p.Results.Iterations, 'AccumulatedFieldSmoothing', p.Results.AccumulatedFieldSmoothing);

            % change metadata
            obj.jobFinished.LocalRegistration = [1 p.Results.Method floor(log2(obj.dimZ)) p.Results.AccumulatedFieldSmoothing];
            
        end
        
        
        % 6.5 DAPI registration  
        function obj = NucleiRegistration( obj, ref_dapi, move_dapi )
            
            fprintf('====Nuclei-based Registration====\n');
            

            round1_img = obj.rawImages(:,:,:,:,1);
            
            if obj.useGPU
                
                tic;
                
                fix = gpuArray(ref_dapi);
                mov = gpuArray(move_dapi);

                params = DFTRegister3D(fix, mov, false);

                for c=1:4
                    curr_reg = DFTApply3D(gpuArray(round1_img(:,:,:,c)), params, false);
                    round1_img(:,:,:,c) = curr_reg;
                end

                obj.rawImages(:,:,:,:,1) = gather(round1_img);
                fprintf(sprintf('Move nuclei vs. Ref nuclei finished [time=%02f]\n', toc));
                fprintf(obj.log, sprintf('Move nuclei vs. Ref nuclei finished [time=%02f]\n', toc));
                fprintf(sprintf('Shifted by %s\n', num2str(params.shifts)));
                fprintf(obj.log, sprintf('Shifted by %s\n', num2str(params.shifts)));
                reset(gpuDevice);
               
            else
                tic;
                params = DFTRegister3D(ref_dapi, move_dapi, false);

                for c=1:4
                    curr_reg = DFTApply3D(round1_img(:,:,:,c), params, false);
                    round1_img(:,:,:,c) = curr_reg;
                end

                obj.rawImages(:,:,:,:,2) = round1_img;
                fprintf(sprintf('Move nuclei vs. Ref nuclei finished [time=%02f]\n', toc));
                fprintf(obj.log, sprintf('Move nuclei vs. Ref nuclei finished [time=%02f]\n', toc));
                fprintf(sprintf('Shifted by %s\n', num2str(params.shifts)));
                fprintf(obj.log, sprintf('Shifted by %s\n', num2str(params.shifts)));
                reset(gpuDevice);
            end

            % change metadata
            obj.jobFinished.NucleiRegistration = 1;
            
        end
        
        
        % 6.5 use DAPI register protein images  
        function obj = NucleiRegistrationProtein( obj, protein_folder, sub_dir, dapi_channel )
            
            fprintf('====Nuclei-based Registration====\n');
            

            ref_dapi_path = dir(fullfile(obj.inputPath, "round1", sub_dir, "*.tif"));
            ref_dapi = new_LoadMultipageTiff(fullfile(ref_dapi_path(5).folder, ref_dapi_path(5).name), 'uint8', 'uint8', false);
            
%             ref_dapi_path = dir(fullfile(obj.inputPath, "ref_dapi", sub_dir, "*.tif"));
%             ref_dapi = new_LoadMultipageTiff(fullfile(ref_dapi_path(1).folder, ref_dapi_path(1).name), 'uint8', 'uint8', false);
            
            ref_dapi = ref_dapi(:,:,1:30); %%%
            
            protein_path = fullfile(obj.inputPath, protein_folder, sub_dir);
            protein_files = dir(fullfile(protein_path, '*.tif'));
            nfiles = numel(protein_files);
            protein_imgs = cell(nfiles, 1);

            % Load all channels
            for c=1:nfiles 
                curr_path = strcat(protein_files(c).folder, '/', protein_files(c).name);
                curr_img = new_LoadMultipageTiff(curr_path, 'uint8', 'uint8', false);
                protein_imgs{c} = curr_img(:,:,1:30);
            end


            if obj.useGPU
                
                tic;
                
                fix = gpuArray(ref_dapi);
                mov = gpuArray(protein_imgs{dapi_channel});

                params = DFTRegister3D(fix, mov, false);

                for c=1:nfiles
                    curr_reg = DFTApply3D(gpuArray(protein_imgs{c}), params, false);
                    protein_imgs{c} = uint8(gather(curr_reg));
                end

                obj.proteinImages = protein_imgs;
                fprintf(sprintf('Move nuclei vs. Ref nuclei finished [time=%02f]\n', toc));
                fprintf(obj.log, sprintf('Move nuclei vs. Ref nuclei finished [time=%02f]\n', toc));
                fprintf(sprintf('Shifted by %s\n', num2str(params.shifts)));
                fprintf(obj.log, sprintf('Shifted by %s\n', num2str(params.shifts)));
                reset(gpuDevice);
               
            else
                tic;
                params = DFTRegister3D(ref_dapi, move_dapi, false);

                for c=1:4
                    curr_reg = DFTApply3D(protein_imgs{c}, params, false);
                    protein_imgs{c} = uint8(curr_reg);
                end

                obj.proteinImages = protein_imgs;
                fprintf(sprintf('Move nuclei vs. Ref nuclei finished [time=%02f]\n', toc));
                fprintf(obj.log, sprintf('Move nuclei vs. Ref nuclei finished [time=%02f]\n', toc));
                fprintf(sprintf('Shifted by %s\n', num2str(params.shifts)));
                fprintf(obj.log, sprintf('Shifted by %s\n', num2str(params.shifts)));
            end

            % change metadata
            obj.jobFinished.NucleiRegistrationProtein = 1;
            
        end
        
        % 6.5 use DAPI register protein images  
        function obj = DotsRegistrationProtein( obj, protein_folder, sub_dir, move_channel )
            
            fprintf('====Dots-based Registration====\n');
            
            protein_path = fullfile(obj.inputPath, protein_folder, sub_dir);
            protein_files = dir(fullfile(protein_path, '*.tif'));
            nfiles = numel(protein_files);
            protein_imgs = cell(nfiles, 1);

            % Load all channels
            for c=1:nfiles 
                curr_path = strcat(protein_files(c).folder, '/', protein_files(c).name);
                curr_img = new_LoadMultipageTiff(curr_path, 'uint8', 'uint8', false);
                protein_imgs{c} = curr_img;
            end

            ref_img = max(obj.rawImages(:,:,:,:,1), [], 4);
            
            if obj.useGPU
                
                tic;
                
                fix = gpuArray(ref_img);
                mov = gpuArray(protein_imgs{move_channel});
                params = DFTRegister3D(fix, mov, false);

                for c=1:4
                    curr_reg = DFTApply3D(gpuArray(protein_imgs{c}), params, false);
                    protein_imgs{c} = uint8(gather(curr_reg));
                end

                obj.proteinImages = protein_imgs;
                fprintf(sprintf('Move image vs. Ref image finished [time=%02f]\n', toc));
                fprintf(obj.log, sprintf('Move image vs. Ref image finished [time=%02f]\n', toc));
                fprintf(sprintf('Shifted by %s\n', num2str(params.shifts)));
                fprintf(obj.log, sprintf('Shifted by %s\n', num2str(params.shifts)));
                reset(gpuDevice);
               
            else
                tic;
                mov = protein_imgs{move_channel};
                params = DFTRegister3D(ref_img, mov, false);

                for c=1:4
                    curr_reg = DFTApply3D(protein_imgs{c}, params, false);
                    protein_imgs{c} = uint8(curr_reg);
                end

                obj.proteinImages = protein_imgs;
                fprintf(sprintf('Move image vs. Ref image finished [time=%02f]\n', toc));
                fprintf(obj.log, sprintf('Move image vs. Ref image finished [time=%02f]\n', toc));
                fprintf(sprintf('Shifted by %s\n', num2str(params.shifts)));
                fprintf(obj.log, sprintf('Shifted by %s\n', num2str(params.shifts)));
            end

            % change metadata
            obj.jobFinished.DotsRegistrationProtein = 1;
            
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
%             defaultthreshold = 0.5;
%             defaultshowPlots = true;

            addParameter(p, 'voxelSize', defaultvoxelSize);
%             addParameter(p, 'q_score_thers', defaultthreshold);
%             addParameter(p, 'showPlots', defaultshowPlots);

            parse(p, varargin{:});
            
            fprintf('====Reads Extraction====\n');
            fprintf(sprintf('voxel size: %d x %d x %d\n', p.Results.voxelSize));
            
%             [obj.allReads, obj.allSpots, obj.allScores, obj.basecsMat] = ExtractFromLocation( obj.registeredImages, obj.allSpots, ...
%                                                                                     p.Results.voxelSize, p.Results.q_score_thers, ...
%                                                                                     p.Results.showPlots, obj.log ); 

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
        function obj = LoadRegisteredImages( obj, inputPath, varargin )

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
