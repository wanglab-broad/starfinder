function [color_seq, color_score] = ExtractFromLocation( input_img, allSpots, voxel_size )
%ExtractFromLocation

    % get dims
    [dimX, dimY, dimZ, Nchannel] = size(input_img);
    Npoint = size(allSpots, 1);
    color_matrix = zeros(Npoint, Nchannel); % "color value" of each dot in each channel of each sequencing round 
    color_seq = string(zeros(Npoint, 1));
    color_score = zeros(Npoint, 1);

    for i=1:Npoint
        
        % Get voxel for each dot
        % current_point = allSpots.Centroid(i,:);
        current_point = table2array(allSpots(i, ["x", "y", "z"])); 
        extentsX = GetExtents(current_point(2), voxel_size(1), dimX);
        extentsY = GetExtents(current_point(1), voxel_size(2), dimY);                    
        extentsZ = GetExtents(current_point(3), voxel_size(3), dimZ);    
        
        current_voxel = input_img(extentsX, extentsY, extentsZ, :); % 4-D array
        color_matrix(i, :) = single(squeeze(sum(current_voxel, [1 2 3]))); % sum along row,col,z
        color_matrix(i, :) = color_matrix(i, :) ./ (sqrt(sum(squeeze(color_matrix(i, :)).^2)) + 1E-6); % +1E-6 avoids denominator equaling to 0
        color_max = max(color_matrix(i, :), [], 2);

        if ~isnan(color_max)
            m = find(color_matrix(i, :) == color_max);
            if numel(m) ~= 1
                color_seq(i) = "M";
                color_score(i) = Inf;
            else
                color_seq(i) = string(m(1));
                color_score(i) = -log(color_max);
            end
        else
            color_seq(i) = "N";
            color_score(i) = Inf;
        end
    end


end


function e = GetExtents(pos, voxelSize, lim)

if pos-voxelSize < 1 
    e1 = 1;
else
    e1 = pos-voxelSize;
end

if pos+voxelSize > lim
    e2 = lim;
else
    e2 = pos+voxelSize;
end

e = e1:e2;

end

