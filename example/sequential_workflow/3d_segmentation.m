% Cell segmentation
current_sample = 'starmap';

% load images
ref_cell_seg = imread(fullfile(input_path, 'segmentation/', current_sample, 'overlay_label.tiff'));

overlay_img = imread_big(fullfile(input_path, 'stitched_images/3D', current_sample, 'overlay_new.tif'));

% filter and threshold
cell_filter = zeros(size(overlay_img), 'uint8');
for z=1:size(overlay_img, 3)
    cell_filter(:,:,z) = medfilt2(overlay_img(:,:,z), [10 10]);
end

% cell_threshold = graythresh(cell_filter);
cell_threshold = 0.013  %% starmap 0.013, ribomap 0.005
cell_bw = imbinarize(cell_filter, cell_threshold);

% generate segmentation
cell_seg = zeros(size(overlay_img), 'uint16');
se = strel('disk', 10);
for z=1:size(overlay_img, 3)
    current_mask = imfill(cell_bw(:,:,z), 'holes');
    current_mask = bwareaopen(current_mask, 200);
    current_mask = imdilate(current_mask, se);
    current_mask = imfill(current_mask, 'holes');
    cell_seg(:,:,z) = uint16(current_mask) .* ref_cell_seg;
end

% output
output_path = fullfile(input_path, 'segmentation', current_sample, 'cell.tif')
options.compress = 'lzw';
options.overwrite = true;
options.big = true;
saveastiff(cell_seg, output_path, options);


% DAPI 
% load images
ref_dapi_seg = imread(fullfile(input_path, 'segmentation/', current_sample, 'dapi_label.tiff'));

dapi_img = imread_big(fullfile(input_path, 'stitched_images/3D', current_sample, 'dapi.tif'));

% filter and threshold
dapi_filter = zeros(size(dapi_img), 'uint8');
for z=1:size(dapi_img, 3)
    dapi_filter(:,:,z) = medfilt2(dapi_img(:,:,z), [10 10]);
%     dapi_filter(:,:,z) = imgaussfilt(dapi_img(:,:,z));
end

dapi_threshold = graythresh(dapi_filter);
dapi_bw = imbinarize(dapi_filter, dapi_threshold);

% generate segmentation
dapi_seg = zeros(size(dapi_img), 'uint16');
se = strel('disk', 5);

for z=1:size(dapi_img, 3)
    current_mask = imfill(dapi_bw(:,:,z), 'holes');
    current_mask = bwareaopen(current_mask, 10);
    current_mask = imdilate(current_mask, se);
    dapi_seg(:,:,z) = uint16(current_mask) .* ref_dapi_seg;
end

% output
output_path = fullfile(input_path, 'segmentation', current_sample, 'nuclei.tif')
saveastiff(dapi_seg, output_path, options);
% if exist(output_path, 'file') == 2
%     delete(output_path);
% end
% for j=1:size(dapi_seg, 3)
%     imwrite(dapi_seg(:,:,j), output_path, 'writemode', 'append');        
% end

clear dapi_bw
clear dapi_filter
clear dapi_img
clear cell_bw
clear cell_filter
clear overlay_img


% ER 
% load images
er_img = imread_big(fullfile(input_path, 'stitched_images/3D', current_sample, 'er.tif'));

% filter and threshold
er_filter = zeros(size(er_img), 'uint8');
for z=1:size(er_img, 3)
    er_filter(:,:,z) = imadjust(er_filter(:,:,z), stretchlim(er_filter(:,:,z), [0.003 0.997]), []);
    er_filter(:,:,z) = medfilt2(er_img(:,:,z), [10 10]);
%     dapi_filter(:,:,z) = imgaussfilt(dapi_img(:,:,z));
end

er_threshold = graythresh(er_filter) * 2
er_bw = imbinarize(er_filter, er_threshold);

% generate segmentation
er_seg = zeros(size(er_img), 'uint16');
se = strel('disk', 3);

for z=1:size(er_img, 3)
    current_mask = imfill(er_bw(:,:,z), 'holes');
    current_mask = bwareaopen(current_mask, 10);
    current_mask = imdilate(current_mask, se);
    er_seg(:,:,z) = uint16(current_mask) .* ref_cell_seg;
end

er_seg = er_seg - dapi_seg;

% output
output_path = fullfile(input_path, 'segmentation', current_sample, 'er.tif')
saveastiff(er_seg, output_path, options);
% if exist(output_path, 'file') == 2
%     delete(output_path);
% end
% for j=1:size(er_seg, 3)
%     imwrite(er_seg(:,:,j), output_path, 'writemode', 'append');        
% end

clear er_bw
clear er_filter

%cyto
cyto_seg = cell_seg - dapi_seg;

% output
output_path = fullfile(input_path, 'segmentation', current_sample, 'cyto.tif')
saveastiff(cyto_seg, output_path, options);
% if exist(output_path, 'file') == 2
%     delete(output_path);
% end
% for j=1:size(cyto_seg, 3)
%     imwrite(cyto_seg(:,:,j), output_path, 'writemode', 'append');        
% end


%outer
outer_seg = cell_seg - dapi_seg - er_seg;

% output
output_path = fullfile(input_path, 'segmentation', current_sample, 'outer_cyto.tif')
saveastiff(outer_seg, output_path, options);
% if exist(output_path, 'file') == 2
%     delete(output_path);
% end
% for j=1:size(outer_seg, 3)
%     imwrite(outer_seg(:,:,j), output_path, 'writemode', 'append');        
% end
