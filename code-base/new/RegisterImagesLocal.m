function [input_img, params] = RegisterImagesLocal( input_img, ref_img, mov_img, iterations, afs)
%RegisterImagesLocal is used to do local (non-rigid) registration for images

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

    

