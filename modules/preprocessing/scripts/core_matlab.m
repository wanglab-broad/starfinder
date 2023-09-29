% This function takes parameters from config.yaml, passed by the rsf.py script as arguments 
% Depending on the parameters, the logic will route to one of 5 chunks:
    % 1 global_registration: performs global reg over a whole tile
    % 2 split: splits globally registered rounds into subtiles to parallelize following steps 
    % 3 local_registration: performs local reg over one subtile and performs spot-finding and filtering
    % 4 stitch: aggregates spot-finding results across subtiles and "re-stitches" the full tile
    % 5 nuclei_protein_registration: 

function out = core_matlab( sample, mode, tile, xy, z, ref_round, n_chs, n_rounds, ...
                            user_dir, source_data_dir, registration_dir, log_dir, ...
                            varargin )  

    %% Input parser
    p = inputParser;

    % Required parameters necessary for all modes
    addRequired(p, 'sample');
    addRequired(p, 'mode');
    %addRequired(p, 'run_id');
    addRequired(p, 'tile');
    addRequired(p, 'xy');
    addRequired(p, 'z');
    addRequired(p, 'ref_round');
    addRequired(p, 'n_chs');
    addRequired(p, 'n_rounds');

    addRequired(p, 'user_dir');
    addRequired(p, 'source_data_dir');
    addRequired(p, 'registration_dir');
    addRequired(p, 'log_dir');

    disp("required parameters parsed")
    
    % "Parameter" parameters are mode-specific
    % These may not be included in function call, so dummy values are defaults
    defaultSubtile = 0;
    defaultendBases = [];
    defaultbarcodeMode = "";
    defaultsplitLoc = 0;
    defaultvoxelSize = [];
    defaultsqrtPieces = 0;
    defaultMethod = "";
    defaultintensityThreshold = 0;
    defaultqScoreThers = 0;
    defaultproteinRound = "";
    defaultproteinStains = [];
    addParameter(p, 'subtile', defaultSubtile);
    addParameter(p, 'end_bases', defaultendBases);
    addParameter(p, 'barcode_mode', defaultbarcodeMode);
    addParameter(p, 'split_loc', defaultsplitLoc);
    addParameter(p, 'voxel_size', defaultvoxelSize);
    addParameter(p, 'sqrt_pieces', defaultsqrtPieces);
    addParameter(p, 'spotfinding_method', defaultMethod);
    addParameter(p, 'intensity_threshold', defaultintensityThreshold);
    % addParameter(p, 'q_score_thers', defaultqScoreThers);
    addParameter(p, 'protein_round', defaultproteinRound);
    addParameter(p, 'protein_stains', defaultproteinStains);
 
    parse(p, sample, mode, tile, xy, z, ref_round, n_chs, n_rounds, ...
            user_dir, source_data_dir, registration_dir, log_dir, ...
            varargin{:}); 

    disp("additional parameters parsed")

    % Parse dimensions
    input_dim = [p.Results.xy p.Results.xy p.Results.z p.Results.n_chs p.Results.n_rounds];

    % File I/O
    input_path = fullfile(p.Results.user_dir, p.Results.sample, p.Results.source_data_dir);
    output_path = fullfile(p.Results.user_dir, p.Results.sample, p.Results.registration_dir); %, p.Results.run_id);
    
    % add path to the matlab code base 
    addpath(fullfile('/stanley/WangLab/morgan/code/starmap-matlab/src'));
    addpath(fullfile('/stanley/WangLab/morgan/code/starmap-matlab/archive'));

    % tile directory
    curr_out_path = fullfile(output_path, p.Results.tile); 
    if ~exist(curr_out_path, 'dir')
        mkdir(curr_out_path);
        fileattrib(curr_out_path, '+w', 'g'); % allows write permlogissions for group of users (755)
    end
    % log directory within tile
    curr_out_path_log = fullfile(curr_out_path, p.Results.log_dir);
    if ~exist(curr_out_path_log, 'dir')
        mkdir(curr_out_path_log);
        fileattrib(curr_out_path_log, '+w', 'g'); % allows write permissions for group of users (755)
    end
    % intermediary output directory within tile (includes registered image mat files, subtile outputs, and r1max)      
    interm_output_dir = fullfile(curr_out_path, 'interm');
    if ~exist(interm_output_dir, 'dir')
        mkdir(interm_output_dir);
        fileattrib(interm_output_dir, '+w', 'g'); % allows write permissions for group of users (755)
    end
    
    % Global Registration
    if strcmp(p.Results.mode,'global_registration')
        sdata = new_STARMapDataset_zf(input_path, output_path, 'useGPU', false); 
        sdata.log = fopen(fullfile(curr_out_path_log, 'log_global.txt'), 'w');

        %%% preprocess
        sdata = sdata.LoadRawImages('sub_dir', p.Results.tile, 'input_dim', input_dim);
        sdata = sdata.SwapChannels; % !!
        sdata = sdata.MinMaxNormalize;
        sdata = sdata.HistEqualize('Method', "inter_round");
        sdata = sdata.HistEqualize('Method', "intra_round");
        sdata = sdata.MorphoRecon('Method', "2d", 'radius', 6);
    
        %%% register
        sdata = sdata.test_GlobalRegistration('useGPU', false, 'ref_round', p.Results.ref_round); 
    
        %%% save round 1 merged tif and registered whole image
        try % round 1 merged
           r1_img = max(sdata.registeredImages(:,:,:,:,p.Results.ref_round), [], 4);
           r1_img_name = fullfile(interm_output_dir, "r1merged.tif");
           SaveSingleTiff(r1_img, r1_img_name);
           disp(strcat("Wrote ", r1_img_name, " to file"))
        catch
           disp('Did not write round1 merged tif. Probably already exists, but double-check');
        end
        fclose(sdata.log);

        % Split into subtiles for local registration and spot-finding
        coords_mat = table([],[],[],[],[],[],[],[],[],[],[],'VariableNames',{'t','ind_x','ind_y','scoords_x','scoords_y','ecoords_x','ecoords_y','upperleft_x','upperleft_y','inputdim_x','inputdim_y'});
        sub_order = [];
        for i = 0:(p.Results.sqrt_pieces-1)
            for j = 0:(p.Results.sqrt_pieces-1)
                sub_order = [sub_order;[i,j]];
            end
        end
        tile_size = floor(p.Results.xy / p.Results.sqrt_pieces);
        overlap_half = floor(tile_size * 0.1);
        upper_left = [0,0];
        for t=1:size(sub_order,1)
            tile_idx = sub_order(t,:);
            start_coords_x = tile_idx(1) * tile_size - overlap_half + 1;
            end_coords_x = (tile_idx(1)+1) * tile_size + overlap_half;
            start_coords_y = tile_idx(2) * tile_size - overlap_half + 1;
            end_coords_y = (tile_idx(2)+1) * tile_size + overlap_half;
            %% compensate in edge
            if tile_idx(1) == 0
                start_coords_x = start_coords_x + overlap_half;
            end
            if tile_idx(2) == 0
                start_coords_y = start_coords_y + overlap_half;
            end
            %% compensate in edge
            if tile_idx(1) == p.Results.sqrt_pieces - 1
                end_coords_x = input_dim(1);
            end
            if tile_idx(2) == p.Results.sqrt_pieces - 1
                end_coords_y = input_dim(2);
            end
            upper_left(1) = tile_idx(1) * tile_size;
            upper_left(2) = tile_idx(2) * tile_size;    
    
            input_dim_t = input_dim;
            input_dim_t(1:2) = [end_coords_x - start_coords_x + 1,end_coords_y - start_coords_y + 1];
            disp([tile_idx,start_coords_x,end_coords_x,start_coords_y,end_coords_y,upper_left(1:2),input_dim_t(1:2)]);
            coords_mat_t = table(t,tile_idx(1),tile_idx(2),start_coords_x,start_coords_y,end_coords_x,end_coords_y,upper_left(1),upper_left(2),input_dim_t(1),input_dim_t(2),'VariableNames',{'t','ind_x','ind_y','scoords_x','scoords_y','ecoords_x','ecoords_y','upperleft_x','upperleft_y','inputdim_x','inputdim_y'});
            coords_mat = [coords_mat;coords_mat_t];
            t_output = sdata.registeredImages(start_coords_y:end_coords_y,start_coords_x:end_coords_x,:,:,:); %% row - y , col - x [row, col, z, :,:]

            %%% save each subtile registered images in following format: registeredImages_t{subtile}_{total_subtiles}.mat
            save(fullfile(interm_output_dir, strcat('registeredImages_t',num2str(t),'_',num2str(p.Results.sqrt_pieces^2),'.mat')), "t_output");
        end
        writetable(coords_mat, fullfile(interm_output_dir,strcat('coords_mat_',num2str(p.Results.sqrt_pieces^2),'.csv')),'Delimiter',',','QuoteStrings',false);

    end
    
    % Local Registration and Spot Finding
    if strcmp(p.Results.mode,'local_registration')
        %%% get subtile coordinate position data
        coords_mat =readtable(fullfile(interm_output_dir,strcat('coords_mat_',num2str(p.Results.sqrt_pieces^2),'.csv')),'ReadVariableNames',true,'TextType','string');
        goodSpots = table([],[],[],[],'VariableNames',{'x','y','z','Gene'});
        
        t = p.Results.subtile;
        input_dim_t = input_dim;
        tile_idx = table2array(coords_mat(t,2:3));
        start_coords_x = table2array(coords_mat(t,4));
        start_coords_y = table2array(coords_mat(t,5));
        upper_left = table2array(coords_mat(t,8:9));
        input_dim_t(1:2) = table2array(coords_mat(t,10:11));
    
        %%% initialize and load registered subtile 
        sdata_t = new_STARMapDataset_zf(input_path, output_path, 'useGPU', false);
        sdata_t.log = fopen(fullfile(curr_out_path_log, strcat('log_t',num2str(p.Results.subtile),'_',num2str(p.Results.sqrt_pieces^2),'.txt')), 'w');
        fprintf(sdata_t.log, strcat('log_t',num2str(p.Results.subtile),'_',num2str(p.Results.sqrt_pieces^2),':\n'));
        load(fullfile(interm_output_dir, strcat('registeredImages_','t',num2str(p.Results.subtile),'_',num2str(p.Results.sqrt_pieces^2),'.mat')));
        sdata_t.registeredImages = t_output;
        t_output = [];
        sdata_t.dims = input_dim_t;
        sdata_t.dimX = input_dim_t(1);
        sdata_t.dimY = input_dim_t(2);
        sdata_t.dimZ = input_dim_t(3);
        sdata_t.Nchannel = p.Results.n_chs;
        sdata_t.Nround = p.Results.n_rounds;
    
        %%% locally register across rounds    
        sdata_t = sdata_t.xxx_LocalRegistration('Iterations', 50, 'AccumulatedFieldSmoothing', 1, 'ref_round',p.Results.ref_round);
    
        %%% spot finding
        sdata_t = sdata_t.SpotFinding('Method', p.Results.spotfinding_method, 'ref_index', p.Results.ref_round, 'intensityThreshold', p.Results.intensity_threshold, 'showPlots', false);
        sdata_t = sdata_t.ReadsExtraction('voxelSize', p.Results.voxel_size);
        if strcmp(p.Results.barcode_mode, "duo")
            sdata_t = sdata_t.LoadCodebook('remove_index', p.Results.split_loc);
            sdata_t = sdata_t.ReadsFiltration('mode', "duo", 'endBases', p.Results.end_bases, 'split_loc', p.Results.split_loc, 'showPlots', false);
        elseif strcmp(p.Results.barcode_mode, "regular")
            sdata_t = sdata_t.LoadCodebook();
            sdata_t = sdata_t.ReadsFiltration('mode', "regular", 'endBases', p.Results.end_bases, 'showPlots', false);
        else
            fprintf(sdata_t.log, "Reads filtration incomplete: invalid mode entered (valid options include 'regular' and 'duo'");
        end
        
        %%% save results
        if size(sdata_t.goodSpots,1) > 0
            sdata_t.goodSpots(:,1) = sdata_t.goodSpots(:,1) + start_coords_x - 1;
            sdata_t.goodSpots(:,2) = sdata_t.goodSpots(:,2) + start_coords_y - 1;
            goodSpots_t = [table(sdata_t.goodSpots(:,1),sdata_t.goodSpots(:,2),sdata_t.goodSpots(:,3),'VariableNames',{'x','y','z'}),cell2table(cellfun(@(x) sdata_t.seqToGene(x), sdata_t.goodReads, 'UniformOutput', false),'VariableNames',{'Gene'})];
        else
            goodSpots_t = table([],[],[],[],'VariableNames',{'x','y','z','Gene'});
        end

        writetable(goodSpots_t,fullfile(interm_output_dir, strcat('goodPoints_', p.Results.spotfinding_method, '_t',num2str(p.Results.subtile),'_',num2str(p.Results.sqrt_pieces^2),'.csv')),'Delimiter',',','QuoteStrings',false);
    
        fclose(sdata_t.log);
    end
    
    % Stitch
    if strcmp(p.Results.mode,'stitch')
        %%% get subtile configuration
        coords_mat = readtable(fullfile(interm_output_dir,strcat('coords_mat_',num2str(p.Results.sqrt_pieces^2),'.csv')),'ReadVariableNames',true,'TextType','string');
        goodSpots = table([],[],[],[],'VariableNames',{'x','y','z','Gene'});
        
        %%% iteratively aggregate subtile spots
        for t=1:size(coords_mat,1)
            start_coords_x = table2array(coords_mat(t,4));
            start_coords_y = table2array(coords_mat(t,5));
            upper_left = table2array(coords_mat(t,8:9));
            goodSpots_t = readtable(fullfile(interm_output_dir,strcat('goodPoints_',p.Results.spotfinding_method,'_t',num2str(t),'_',num2str(p.Results.sqrt_pieces^2),'.csv')),'ReadVariableNames',true,'TextType','string');
        
            % filter based on overlap region
            if size(goodSpots,1) > 0
                goodSpots = goodSpots((table2array(goodSpots(:,1)) <= upper_left(1)) | (table2array(goodSpots(:,2)) <= upper_left(2)),:);
            end
            if size(goodSpots_t,1) > 0
                goodSpots_t = goodSpots_t((table2array(goodSpots_t(:,1)) > upper_left(1)) & (table2array(goodSpots_t(:,2)) > upper_left(2)),:);
            else
                goodSpots_t = table([],[],[],[],'VariableNames',{'x','y','z','Gene'});
            end
            goodSpots = [goodSpots;goodSpots_t];
        end
    
        writetable(goodSpots,fullfile(curr_out_path, strcat('goodPoints_',p.Results.spotfinding_method,'.csv')),'Delimiter',',','QuoteStrings',false);
        disp(strcat(input_path,"  Work Finished!!!!!"))
    end

    % Nuclei Protein Registration
    if strcmp(p.Results.mode, 'nuclei_protein_registration')
        %%% initialize
        sdata = new_STARMapDataset_zf(input_path, output_path, 'useGPU', false);
        sdata.log = fopen(fullfile(curr_out_path_log, 'log_protein_registration.txt'), 'w');

        %%% perform DAPI-based registration of protein round
        sdata = sdata.NucleiRegistrationProtein(p.Results.protein_round, p.Results.tile, 1);

        %%% save cell images
        protein_output_dir = fullfile(output_path, p.Results.protein_round);
        if ~exist(protein_output_dir, 'dir')
            mkdir(protein_output_dir);
        end
        sub_dirs = string(p.Results.protein_stains)
        SaveCellImg(protein_output_dir, sdata.proteinImages, p.Results.tile, sub_dirs)
    end




