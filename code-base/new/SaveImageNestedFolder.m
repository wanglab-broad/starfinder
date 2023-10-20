function SaveImageNestedFolder(input_img, layer, output_folder, fovID)


    Nchannel = size(input_img{1}, 4);

    current_round_folder = fullfile(output_folder, layer);
    if ~exist(current_round_folder, 'dir')
        mkdir(current_round_folder);
    end

    current_fov_folder = fullfile(current_round_folder, fovID);
    if ~exist(current_fov_folder, 'dir')
        mkdir(current_fov_folder)
    end

    for c=1:Nchannel
        fname = fullfile(current_fov_folder, sprintf('ch%02d.tif', c));
        SaveSingleStack(input_img{1}(:,:,:,c), fname);
    end


end

