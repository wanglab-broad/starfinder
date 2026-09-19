function input_img = MinMaxNorm( input_img )
% Normalize each channel of cell-wrapped (row, column, Z, C) images.
%
% input_img is a cell array, typically a slice of the dataset image dictionary.
% Returns the updated cell array. stretchlim(...,0) determines channel limits
% across planes and imadjustn rescales intensities. Requires Image Processing
% Toolbox; this is not Python's direct floating-point min/max division.


    Nround = numel(input_img);
    Nchannel = size(input_img{1}, 4);
    
    for r=1:Nround
        tic
        for c=1:Nchannel 
            current_channel = input_img{r}(:, :, :, c);
            current_limits = stretchlim(current_channel, 0);
            low_in = min(current_limits(1, :));
            high_in = max(current_limits(2, :));
            lowhigh_in = [low_in high_in];
            if class(lowhigh_in) == "gpuArray"
                lowhigh_in = gather(lowhigh_in);
            end
            current_channel_adjusted = imadjustn(current_channel, lowhigh_in);
            input_img{r}(:, :, :, c) = current_channel_adjusted;
        end

        fprintf(sprintf('[time = %.2f s]\n', toc));

    end

end

