function output_img = MakeProjections( input_img, method )
%MakeProjections

Nround = numel(input_img);
Nchannel = size(input_img{1}, 4);

    switch method
        case "max"
            output_img = {};
            a = 1;
            for r=1:Nround
                current_img = max(input_img{r}, [], 3);
                for c=1:Nchannel
                    output_img{a} = current_img(:,:,1,c);
                    a = a + 1;
                end
            end
        
        case "sum"
            output_img = {};
            a = 1;
            for r=1:Nround
                current_img = im2uint8(sum(uint32(input_img{r}), [], 3));
                for c=1:Nchannel
                    output_img{a} = current_img(:,:,1,c);
                    a = a + 1;
                end
            end
    end

end
