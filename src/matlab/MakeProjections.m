function output_img = MakeProjections( input_img, method )
% Project cell-wrapped (row, column, Z, C) images along Z.
%
% input_img is a cell array; method="max" returns a cell per round/channel
% containing a (row, column) maximum projection, ordered round then channel.
% The "sum" branch uses nested cell indexing on image data and is unvalidated
% for the numeric arrays produced by the loader. Use "max" for workflow previews.

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
