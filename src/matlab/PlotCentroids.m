function output_img = PlotCentroids(input_centroid, input_img, color, msize, input_title)
% Overlay spot-table x/y coordinates on a 2-D background image.
%
% input_centroid is a table with 1-based x (column) and y (row); input_img is a
% 2-D background. Pass color and msize explicitly (the short-argument default
% handling does not initialize msize). input_title is optional, default ''.
% Returns output_img as a figure handle; Z is not shown.

    if nargin < 4
        color = 'red';
    end

    if nargin < 5
        input_title = '';
    end

    figure('Position', [0 0 size(input_img, 1) size(input_img, 2)])
    imshow(input_img, [])
    hold on
    plot(input_centroid.x, input_centroid.y, '.', "Color", color, "MarkerSize", msize)
    title(input_title)
    hold off
    
    output_img = gcf;
end