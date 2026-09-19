function SaveImageNestedFolder(input_img, layer, output_folder, fovID, maximum_projection)
% Save the first cell in input_img under output_folder/layer/fovID.
%
% input_img is a cell array holding a (row, column, Z, C) image. layer and fovID
% are path components; maximum_projection selects a Z maximum or full stack.
% Creates folders and writes ch01.tif, ch02.tif, ... (1-based output names,
% not the original acquisition channel IDs). Returns no value.


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
        if maximum_projection
            SaveSingleStack(max(input_img{1}(:,:,:,c), [], 3), fname);
        else
            SaveSingleStack(input_img{1}(:,:,:,c), fname);
        end
    end


end

