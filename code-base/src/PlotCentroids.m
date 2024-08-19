function output_img = PlotCentroids(input_centroid, input_img, msize, color, input_title)
% Plot centroids on image

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