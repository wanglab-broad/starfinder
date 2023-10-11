function [output_imgs, dims] = LoadImageStacks( inputPath, sub_dir, channel_order_dict, zrange, convert_uint8 )
%LOADIMAGESTACKS Load image stacks for each round
        
    % Suppress all warnings 
    warning('off','all');
    
    % Get directories containing all images 
    dirs = dir(strcat(inputPath, 'round*'));
    Nround = numel(dirs);

    % Set output_imgs
    output_imgs = cell(Nround, 1);
    channel_names = {channel_order_dict(:).name};
    Nchannel = numel(find([channel_names{:}] == "seq"));
    

    for r=1:Nround
        tic
        fprintf('Loading round %d...\n', r);

        current_dir = dirs(r).name;

        current_files = dir(fullfile(inputPath, current_dir, sub_dir, '*.tif'));
        
        InfoImage=imfinfo(fullfile(current_files(1).folder, current_files(1).name));
        dimZ=length(InfoImage);
        imageFormat = sprintf("uint%d", InfoImage(1).BitDepth);

        % Load all channels
        temp_round_imgs = cell(Nchannel, 1);
        maxX_ch = 1E10; maxY_ch = 1E10;
        for c=1:Nchannel 
            current_ch_id = channel_order_dict(c).channel;
            current_file_index = find(contains({current_files(:).name}, current_ch_id) == 1);
            current_path = fullfile(current_files(current_file_index).folder, current_files(current_file_index).name);
            current_img = LoadMultipageTiff(current_path, convert_uint8);
            temp_round_imgs{c} = current_img;

            [currX, currY, ~] = size(current_img); 
            if currX < maxX_ch
                maxX_ch = currX;
            end
            if currY < maxY_ch
                maxY_ch = currY;
            end
        end

        fprintf('Collapsed to size %d by %d by %d\n', maxX_ch, maxY_ch);

        for c=1:Nchannel
            current_channel = temp_round_imgs{c};
            temp_round_imgs{c} = current_channel(1:maxX_ch, 1:maxY_ch, :);
        end

        round_img = zeros([maxX_ch maxY_ch dimZ Nchannel], imageFormat);
        for c=1:Nchannel
            round_img(:, :, :, c) = temp_round_imgs{c};
        end
        output_imgs{r} = round_img;

        fprintf(sprintf('[time = %.2f s]\n', toc));

    end

    % Collapse to common sized array across each round 
    fprintf('Adjust size across each round...\n');
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
  
end

