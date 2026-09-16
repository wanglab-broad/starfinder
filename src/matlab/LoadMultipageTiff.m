function FinalImage = LoadMultipageTiff( fname, convert_uint8 )
% Read a multipage grayscale TIFF into a row-by-column-by-Z integer array.
%
% fname is a TIFF path. convert_uint8 defaults to false (preserve bit depth);
% true applies im2uint8. FinalImage contains one TIFF page per Z plane.

    % Suppress all warnings 
    warning('off','all');
    
    if nargin < 2
        convert_uint8 = false;
    end

    InfoImage=imfinfo(fname);
    mImage=InfoImage(1).Width;
    nImage=InfoImage(1).Height;
    NumberImages=length(InfoImage);
    imageFormat = sprintf("uint%d", InfoImage(1).BitDepth);

    FinalImage=zeros(nImage, mImage, NumberImages, imageFormat);

    TifLink = Tiff(fname, 'r');
    for i=1:NumberImages
       TifLink.setDirectory(i);
       FinalImage(:,:,i)=TifLink.read();
    end
    
    if convert_uint8
        % Convert to uint8
        FinalImage = im2uint8(FinalImage);
    end

    TifLink.close();
    
end
