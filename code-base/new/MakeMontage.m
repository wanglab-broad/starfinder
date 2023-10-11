function output_img = MakeMontage( input_img, Nround, Nchannel, enhance_contrast )
%MakeMontage
        
    figure 
    if enhance_contrast
        img = montage(input_img, "Size", [Nround Nchannel], 'DisplayRange', []);
    else
        img = montage(input_img, "Size", [Nround Nchannel]);
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

