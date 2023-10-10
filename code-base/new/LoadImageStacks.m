function [output_imgs, dims] = LoadImageStacks( inputPath, sub_dir, zrange, convert_uint8 )
%LOADIMAGESTACKS Load image stacks for each round
        
    % Suppress all warnings 
    warning('off','all');
    
    % Get directories containing all images 
    dirs = dir(strcat(inputPath, 'round*'));
    Nround = numel(dirs);

    % Set output_imgs
    output_imgs = cell(Nround, 1);
    channel_numes = [];

    for r=1:Nround

        current_dir = dirs(r).name;
        current_files = dir(fullfile(inputPath, current_dir, sub_dir, '*.tif'));
        current_Nchannel = numel(current_files);
        channel_numes(r) = current_Nchannel;

    end

    Nchannel = mode(channel_numes);

    for r=1:Nround
        tic
        fprintf('Loading round %d...', r);

        current_dir = dirs(r).name;

        current_files = dir(fullfile(inputPath, current_dir, sub_dir, '*.tif'));
        
        InfoImage=imfinfo(fullfile(current_files(1).folder, current_files(1).name));
        dimX=InfoImage(1).Width;
        dimY=InfoImage(1).Height;
        dimZ=length(InfoImage);
        imageFormat = sprintf("uint%d", InfoImage(1).BitDepth);

        round_img = zeros([dimX dimY dimZ Nchannel], imageFormat);

        % Load all channels
        for c=1:Nchannel 
            current_path = fullfile(current_files(c).folder, current_files(c).name);
            current_img = LoadMultipageTiff(current_path, convert_uint8);
            round_img(:,:,:,c) = current_img;
        end

        output_imgs{r} = round_img;

        fprintf(sprintf('[time = %.2f s]\n', toc));

    end

    % Collapse to common sized array
    maxX = 1E10; maxY = 1E10;
    for r=1:Nround
       
        current_round = output_imgs{r};
        [currX, currY, ~] = size(current_round); 
       if currX < maxX
           maxX = currX;
       end
       if currY < maxY
           maxY = currY;
       end

    end

    if isempty(zrange)
       zrange = 1:size(output_imgs{1}, 3);
    end

    % Show message for re-sizing
    fprintf('Collapsed to size %d by %d by %d\n', maxX, maxY, numel(zrange));

    for r=1:Nround
        fprintf('Collapsing round %d\n', r);
        curr_round = output_imgs{r};
        output_imgs{r} = curr_round(1:maxX, 1:maxY, zrange, :);
    end
    

    dims = [maxX maxY numel(zrange) Nchannel Nround];


    fprintf('Raw image as cell'); 
  
end

