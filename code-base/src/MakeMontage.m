function output_img = MakeMontage( input_img, layer_list, enhance_contrast )
%MakeMontage
        
    img_list = {};
    Nchannel_list = [];
    for current_layer=layer_list
        img_list = horzcat(img_list, input_img{current_layer}); 
        Nchannel_list = [Nchannel_list, numel(input_img{current_layer})];
    end

    Nround = numel(layer_list);
    Nchannel = max(Nchannel_list);

    figure 
    if enhance_contrast
        img = montage(img_list, "Size", [Nround Nchannel], 'DisplayRange', []);
    else
        img = montage(img_list, "Size", [Nround Nchannel]);
    end
    img = img.CData;
    row_size = size(img, 1);
    col_size = size(img, 2);
    
    for row = 0 : row_size/Nround : row_size-1
        yline(row, 'Color', 'r', 'LineWidth', 1);
    end
    
    for col = 0 : col_size/Nchannel : col_size-1
        xline(col, 'Color', 'r', 'LineWidth', 1);
    end
    
    output_img = gcf;
    
end

