function SaveSingleStack(input_img, filename)
% Write a (row, column, Z) array to a multipage TIFF at filename.
%
% A 2-D array produces one page. Deletes an existing destination before writing;
% the parent directory must exist. Returns no value; pixels retain input class.

    if exist(filename, 'file') == 2
        delete(filename);
    end

    for j=1:size(input_img, 3)
        imwrite(input_img(:,:,j), filename, 'writemode', 'append');        
    end
        
end

