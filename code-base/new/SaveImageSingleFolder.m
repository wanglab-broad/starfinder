function SaveImageSingleFolder(input_img, layer, output_folder, fovID, group_channel, channel_order_dict)
   
    Nchannel = size(input_img{1}, 4);
        
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
            SaveSingleStack(input_img{1}(:,:,:,c), fname);
        end
    else
        for c=1:Nchannel
            fname = fullfile(output_folder, sprintf('round%d_ch%02d_%s.tif', r, c, fovID));
            SaveSingleStack(input_img{1}(:,:,:,c), fname);
        end
    end

end

