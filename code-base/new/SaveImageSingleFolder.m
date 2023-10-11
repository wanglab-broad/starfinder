function SaveImageSingleFolder(input_img, output_folder, fovID, group_channel, channel_order_dict)
   
    Nround = numel(input_img);
    Nchannel = size(input_img{1}, 4);
        
    if group_channel
        for r=1:Nround
            current_round_folder = fullfile(output_folder, sprintf('round%d', r));
            if ~exist(current_round_folder, 'dir')
               mkdir(current_round_folder);
            end

            for c=1:Nchannel
                current_output_folder = fullfile(current_round_folder, sprintf("%s/", channel_order_dict(c).name));
                if ~exist(current_output_folder, 'dir')
                    mkdir(current_output_folder);
                end

                fname = fullfile(current_output_folder, sprintf('%s.tif', fovID));
                SaveSingleStack(input_img{r}(:,:,:,c), fname);
            end
        end
    else
        for r=1:Nround
            for c=1:Nchannel
                fname = fullfile(output_folder, sprintf('round%d_ch%02d_%s.tif', r, c, fovID));
                SaveSingleStack(input_img{r}(:,:,:,c), fname);
            end
        end
    end

end

