function [output_img, dims] = LoadImageStacks( round_dir, sub_dir, channel_order_dict, convert_uint8 )
%LOADIMAGESTACKS Load image stacks for each round
        
    % Suppress all warnings 
    warning('off','all');

    % Set output_imgs
    channel_names = {channel_order_dict(:).name};
    Nchannel = numel(channel_names);

    tic;
    current_dir = round_dir.name;

    current_files = dir(fullfile(round_dir.folder, current_dir, sub_dir, '*.tif'));
    
    % InfoImage=imfinfo(fullfile(current_files(1).folder, current_files(1).name));
    % dimZ=length(InfoImage);

    % Load all channels
    temp_round_img = cell(Nchannel, 1);
    maxX_ch = 1E10; maxY_ch = 1E10; dimZ = 1E10;
    for c=1:Nchannel 
        InfoImage=imfinfo(fullfile(current_files(c).folder, current_files(c).name));
        currZ = length(InfoImage);
        current_ch_id = channel_order_dict(c).channel;
        current_file_index = find(contains({current_files(:).name}, current_ch_id) == 1);
        current_path = fullfile(current_files(current_file_index).folder, current_files(current_file_index).name);
        current_img = LoadMultipageTiff(current_path, convert_uint8);
        temp_round_img{c} = current_img;

        [currX, currY, ~] = size(current_img); 
        if currX < maxX_ch
            maxX_ch = currX;
        end
        if currY < maxY_ch
            maxY_ch = currY;
        end
        if currZ < dimZ
            dimZ = currZ;
        end
    end

    fprintf('Collapsed to size %d by %d by %d ', maxX_ch, maxY_ch, dimZ);

    if convert_uint8
        finalBitDepth = 8;
    else
        finalBitDepth = InfoImage(1).BitDepth;
    end

    imageFormat = sprintf("uint%d", finalBitDepth);
    dims = [maxX_ch maxY_ch dimZ Nchannel finalBitDepth];

    output_img = zeros([maxX_ch maxY_ch dimZ Nchannel], imageFormat);
    
    for c=1:Nchannel
        output_img(:, :, :, c) = temp_round_img{c}(1:maxX_ch, 1:maxY_ch, 1:dimZ);
    end

    fprintf(sprintf('[time = %.2f s]\n', toc));

end

