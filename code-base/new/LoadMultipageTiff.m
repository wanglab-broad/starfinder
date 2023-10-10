function FinalImage = LoadMultipageTiff( fname, convert_uint8 )

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

