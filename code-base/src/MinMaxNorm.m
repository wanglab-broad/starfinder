function input_img = MinMaxNorm( input_img )
%MinMaxNorm is used to normalize the intensity profile of each channel
%   -----IO-----
%   input_img: mat with input images 
%   output_img: mat with normalized images


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

