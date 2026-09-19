function SaveImageSingleFolder(input_img, layer, output_folder, fovID, group_channel, channel_order_dict, maximum_projection)
% Save a cell-wrapped (row, column, Z, C) image using bundled saveastiff.
%
% layer, output_folder and fovID set output paths. With group_channel=true,
% channel_order_dict(c).name chooses a subfolder; maximum_projection selects
% Z maxima or full volumes. Channels sharing a name overwrite the same FOV file.
% The group_channel=false branch references undefined r and is not a usable
% recipe in this checkout. Returns no value.
   
    Nchannel = size(input_img{1}, 4);
    options.overwrite = true;
    options.compress = 'lzw';

    if group_channel
        current_round_folder = fullfile(output_folder, layer);
        if ~exist(current_round_folder, 'dir')
            mkdir(current_round_folder);
        end

        for c=1:Nchannel
            current_output_folder = fullfile(current_round_folder, sprintf("%s/", channel_order_dict(c).name));
            if ~exist(current_output_folder, 'dir')
                mkdir(current_output_folder);
            end

            fname = fullfile(current_output_folder, sprintf('%s.tif', fovID));
            if maximum_projection
                % SaveSingleStack(max(input_img{1}(:,:,:,c), [], 3), fname);
                saveastiff(max(input_img{1}(:,:,:,c), [], 3), char(fname), options);
            else
                % SaveSingleStack(input_img{1}(:,:,:,c), fname);
                saveastiff(input_img{1}(:,:,:,c), char(fname), options);
            end
        end
    else
        for c=1:Nchannel
            fname = fullfile(output_folder, sprintf('round%d_ch%02d_%s.tif', r, c, fovID));
            if maximum_projection
                % SaveSingleStack(max(input_img{1}(:,:,:,c), [], 3), fname);
                saveastiff(max(input_img{1}(:,:,:,c), [], 3), char(fname), options);
            else
                % SaveSingleStack(input_img{1}(:,:,:,c), fname);
                saveastiff(input_img{1}(:,:,:,c), char(fname), options);
            end
        end
    end

end

