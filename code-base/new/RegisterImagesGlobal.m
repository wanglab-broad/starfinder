function [input_img, params] = RegisterImagesGlobal( input_img, ref_img, mov_img )
% RegisterImagesGlobal

    Nchannel = size(input_img, 4);

    % Calculate shift
    % starting = tic;
    params = DFTRegister3D(ref_img, mov_img, false);
    % fprintf(sprintf('DFT register finished [time=%02f]\n', toc(starting)));
    
    % Apply shift to each channel
    % starting_apply = tic;
    for c=1:Nchannel
        current_reg = DFTApply3D(input_img(:,:,:,c), params, false);
        input_img(:,:,:,c) = current_reg;
    end
    % fprintf(sprintf('DFT apply finished [time=%02f]\n', toc(starting_apply)));

end
        
