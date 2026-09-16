function [input_img, params] = RegisterImagesLocal( input_img, ref_img, mov_img, iterations, afs)
% Estimate a demons field and warp each channel of input_img.
%
% input_img is (row, column, Z, C); ref_img/mov_img are matching scalar volumes.
% iterations and afs are passed to imregdemons as iteration count and accumulated
% field smoothing. Pyramid levels derive from input Z depth (minimum one).
% Returns warped input_img in its original array dtype and the displacement
% field params from imregdemons (row, column, Z, 3), with X/Y/Z components.
% Requires Image Processing Toolbox; not interchangeable with Python TPS/CPD.

    Nchannel = size(input_img, 4);
    dimZ = size(input_img, 3);
    pyd_level = floor(log2(dimZ)); 
    if pyd_level == 0
        pyd_level = 1;
    end

    % Calculate shift
    % starting = tic;
    [params, ~] = imregdemons(mov_img, ref_img, iterations, ...
        'PyramidLevels', pyd_level, ...
        'AccumulatedFieldSmoothing', afs, ...
        'DisplayWaitbar', false);
    % fprintf(sprintf('Local registeration finished [time=%02f]\n', toc(starting)));

    % Apply shift to each channel
    % starting_apply = tic;
    for c=1:Nchannel
        current_reg = imwarp(input_img(:,:,:,c), params);
        input_img(:,:,:,c) = current_reg;
    end

    
end 

    

