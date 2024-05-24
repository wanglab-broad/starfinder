function output_img = MakeProjections( input_img, method )
%MakeProjections

    Nround = numel(input_img);
    Nchannel = size(input_img{1}, 4);

    switch method
        case "max"
            output_img = {};
            a = 1;
            for r=1:Nround
                for c=1:Nchannel
                    output_img{a} = max(input_img{r}(:,:,:,c), [], 3);
                    a = a + 1;
                end
            end
        
        case "sum"
            output_img = {};
            a = 1;
            for r=1:Nround
                for c=1:Nchannel
                    output_img{a} = im2uint8(sum(uint32(input_img{r}{:,:,:,c}), [], 3));
                    a = a + 1;
                end
            end
    end

end
