function output_img = MakeMontage( input_img )
%MakeMontage
        
    figure 
    img = montage(input_img, "Size", [6 4]);
    output_img = img.CData;
    row_size = size(output_img, 1);
    col_size = size(output_img, 2);
    
    for row = 0 : row_size/6 : row_size-1
        yline(row, 'Color', 'r', 'LineWidth', 1);
    end
    
    for col = 0 : col_size/4 : col_size-1
        xline(col, 'Color', 'r', 'LineWidth', 1);
    end
  
end

