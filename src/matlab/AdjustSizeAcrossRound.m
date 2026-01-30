function [input_img, metadata] = AdjustSizeAcrossRound( input_img, metadata, layer_list, zrange )
%AdjustSizeAcrossRound 
          
    % Collapse to common sized array across each round 
    fprintf('Adjust size across each round...\n');
    tic;
    dimX_list = [];
    dimY_list = [];
    dimZ_list = [];

    for current_layer=layer_list
        current_meta = metadata{current_layer};
        dimX_list = [current_meta.dimX, dimX_list];
        dimY_list = [current_meta.dimY, dimY_list];
        dimZ_list = [current_meta.dimZ, dimZ_list];
    end

    minX = min(dimX_list);
    minY = min(dimY_list);
    minZ = min(dimZ_list);

    if isempty(zrange)
       zrange = 1:minZ;
    end

    for current_layer=layer_list
        input_img{current_layer} = input_img{current_layer}(1:minX, 1:minY, zrange, :);
        current_meta = metadata{current_layer};
        current_meta.dimX = minX;
        current_meta.dimY = minY;
        current_meta.dimZ = max(zrange);
        metadata{current_layer} = current_meta;
    end
  
    % Show message for re-sizing
    fprintf('Collapsed to final size %d by %d by %d ', minX, minY, numel(zrange));
    fprintf(sprintf('[time = %.2f s]\n', toc));
end

