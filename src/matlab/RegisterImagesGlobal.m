function [input_img, params] = RegisterImagesGlobal( input_img, ref_img, mov_img, scale )
% Estimate translation on ref_img/mov_img and apply it to every channel.
%
% input_img has shape (row, column, Z, C); ref_img and mov_img are matching 3-D
% registration volumes. scale is their spatial resize factor relative to input.
% Returns the registered input_img (assigned into its original dtype) and params
% with correction shifts in (row, column, Z) order and diffphase. Shifts are
% divided by scale before application. No extra sign negation is applied.

    Nchannel = size(input_img, 4);

    % Calculate shift
    % starting = tic;
    params = DFTRegister3D(ref_img, mov_img, false);
    % fprintf(sprintf('DFT register finished [time=%02f]\n', toc(starting)));
    
    if scale ~= 1
        params.shifts = params.shifts / scale;
    end

    % Apply shift to each channel
    % starting_apply = tic;
    for c=1:Nchannel
        current_reg = DFTApply3D(input_img(:,:,:,c), params, false);
        input_img(:,:,:,c) = current_reg;
    end
    % fprintf(sprintf('DFT apply finished [time=%02f]\n', toc(starting_apply)));

end
        
