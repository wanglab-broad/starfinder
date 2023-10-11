function SaveImageNestedFolder(input_img, output_folder, fovID)

    Nround = numel(input_img);
    Nchannel = size(input_img{1}, 4);

    for r=1:Nround
        current_round_folder = fullfile(output_folder, sprintf('round%d', r));
        if ~exist(current_round_folder, 'dir')
           mkdir(current_round_folder);
        end

        current_fov_folder = fullfile(current_round_folder, sprintf("%s/", fovID));
        if ~exist(current_fov_folder, 'dir')
            mkdir(current_fov_folder)
        end

        for c=1:Nchannel
            fname = fullfile(current_fov_folder, sprintf('ch%02d.tif', c));
            SaveSingleStack(input_img{r}(:,:,:,c), fname);
        end
    end

end

