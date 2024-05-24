% run 3D segmentation workflow for cell culture example dataset 
% user will define:
% config_path

function create_3d_segmentation(config_path)

    % load dataset info 
    config_raw = fileread(config_path);
    config = jsondecode(config_raw);

    % add path for .m files
    addpath(genpath(fullfile(pwd, './code-base/src/'))) % pwd is the location of the starfinder folder
    addpath(genpath(fullfile(pwd, './code-base/matlab-addon/')))

    % test block
    % addpath(fullfile('/stanley/WangLab/jiahao/Github/starfinder/code-base/new/'))
    % addpath(genpath(fullfile('/stanley/WangLab/jiahao/Github/starfinder/code-base/matlab-addon/')))

    % load images
    image_path = fullfile(config.output_path, 'images/fused');
    reference_cell_label = imread(fullfile(image_path, 'Cell_label.tif'));
    overlay_img = imread_big(fullfile(image_path, 'overlay.tif'));

    % Cell
    % filter and threshold
    overlay_filtered = zeros(size(overlay_img), 'uint8');
    for z=1:size(overlay_img, 3)
        overlay_filtered(:,:,z) = medfilt2(overlay_img(:,:,z), [10 10]);
    end

    cell_threshold = graythresh(overlay_filtered);
    overlay_bw = imbinarize(overlay_filtered, cell_threshold);

    % generate segmentation
    cell_seg = zeros(size(overlay_img), 'uint16');
    se = strel('disk', 10);
    for z=1:size(overlay_img, 3)
        current_mask = imfill(overlay_bw(:,:,z), 'holes');
        current_mask = bwareaopen(current_mask, 200);
        current_mask = imdilate(current_mask, se);
        current_mask = imfill(current_mask, 'holes');
        cell_seg(:,:,z) = uint16(current_mask) .* reference_cell_label;
    end

    % save 3D cell segmentation 
    output_path = fullfile(image_path, 'Cell.tif');
    options.compress = 'lzw';
    options.overwrite = true;
    options.big = true;
    saveastiff(cell_seg, output_path, options);
    fprintf("Cell segmentation saved!");

    % Nuclei 
    % load images
    reference_dapi_label = imread(fullfile(image_path, 'DAPI_label.tif'));
    dapi_img = imread_big(fullfile(image_path, 'DAPI.tif'));

    % filter and threshold
    dapi_filtered = zeros(size(dapi_img), 'uint8');
    for z=1:size(dapi_img, 3)
        dapi_filtered(:,:,z) = medfilt2(dapi_img(:,:,z), [10 10]);
    end

    dapi_threshold = graythresh(dapi_filtered);
    dapi_bw = imbinarize(dapi_filtered, dapi_threshold);

    % generate segmentation
    dapi_seg = zeros(size(dapi_img), 'uint16');
    se = strel('disk', 5);

    for z=1:size(dapi_img, 3)
        current_mask = imfill(dapi_bw(:,:,z), 'holes');
        current_mask = bwareaopen(current_mask, 10);
        current_mask = imdilate(current_mask, se);
        dapi_seg(:,:,z) = uint16(current_mask) .* reference_dapi_label;
    end

    % save 3D cell segmentation
    output_path = fullfile(image_path, 'Nuclei.tif');
    saveastiff(dapi_seg, output_path, options);
    fprintf("Nuclei segmentation saved!")

    clear dapi_img
    clear dapi_filtered
    clear dapi_bw

    clear overlay_img
    clear overlay_filtered
    clear overlay_bw


    % Cyto
    cyto_seg = cell_seg - dapi_seg;

    % save 3D cell segmentation
    output_path = fullfile(image_path, 'Cyto.tif');
    saveastiff(cyto_seg, output_path, options);
    fprintf("Cytoplasm segmentation saved!")

end



