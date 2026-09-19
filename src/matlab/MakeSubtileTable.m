function output_table = MakeSubtileTable( dims, sqrt_pieces, overlap_ratio )
% Construct overlapping XY crop bounds for a square tiling grid.
%
% dims supplies image dimensions, sqrt_pieces is grid side count, overlap_ratio
% sets overlap on each side relative to floor(dims(1)/sqrt_pieces). Returns a
% table with 1-based inclusive scoords/ecoords, zero-based ind_x/ind_y and
% upperleft offsets, tile IDs t, and inputdim sizes. Both axes use a tile size
% computed from dims(1); rectangular-image behavior needs separate validation.

    column_headers = {'t', 'ind_x', 'ind_y',...
                    'scoords_x', 'scoords_y', 'ecoords_x', 'ecoords_y',...
                    'upperleft_x', 'upperleft_y', 'inputdim_x', 'inputdim_y'};

    output_table = table([],[],[],[],[],[],[],[],[],[],[],...
                        'VariableNames', column_headers);

    sub_order = [];
    for i = 0:(sqrt_pieces-1)
        for j = 0:(sqrt_pieces-1)
            sub_order = [sub_order; [i,j]];
        end
    end

    tile_size = floor(dims(1) / sqrt_pieces);
    overlap_half = floor(tile_size * overlap_ratio);
    upper_left = [0, 0];

    for t=1:size(sub_order, 1)
        tile_idx = sub_order(t, :);
        start_coords_x = tile_idx(1) * tile_size - overlap_half + 1;
        end_coords_x = (tile_idx(1) + 1) * tile_size + overlap_half;
        start_coords_y = tile_idx(2) * tile_size - overlap_half + 1;
        end_coords_y = (tile_idx(2) + 1) * tile_size + overlap_half;
        
        % compensate in edge
        if tile_idx(1) == 0
            start_coords_x = start_coords_x + overlap_half;
        end
        if tile_idx(2) == 0
            start_coords_y = start_coords_y + overlap_half;
        
        end
        % compensate in edge
        if tile_idx(1) == sqrt_pieces - 1
            end_coords_x = dims(1);
        end
        if tile_idx(2) == sqrt_pieces - 1
            end_coords_y = dims(2);
        
        end
        upper_left(1) = tile_idx(1) * tile_size;
        upper_left(2) = tile_idx(2) * tile_size;    

        dims_t = dims;
        dims_t(1:2) = [end_coords_x - start_coords_x + 1,end_coords_y - start_coords_y + 1];
        % disp([tile_idx,start_coords_x,end_coords_x,start_coords_y,end_coords_y,upper_left(1:2),dims_t(1:2)]);
        output_table_t = table(t, tile_idx(1), tile_idx(2),...
                                start_coords_x, start_coords_y, end_coords_x, end_coords_y,...
                                upper_left(1), upper_left(2), dims_t(1), dims_t(2),...
                                'VariableNames', column_headers);
        output_table = [output_table; output_table_t];
    end

end