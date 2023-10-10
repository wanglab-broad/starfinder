function input_img = SwapTwoChannels( input_img, channel_1, channel_2 )
%SwapTwoChannels

    Nround = numel(input_img);
    
    for r=1:Nround
        
        temp = input_img{r};
        swap_1 = temp(:,:,:,channel_1);
        swap_2 = temp(:,:,:,channel_2);
        temp(:,:,:,channel_2) = swap_1;
        temp(:,:,:,channel_1) = swap_2;
        input_img{r} = temp;
        
    end


end
