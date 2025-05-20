function SaveSingleStack_test(input_img, filename)
    
    % make sure the vector returned by size() is of length 4
    dims = size(input_img, 1:3);

    % Set data type specific fields
    if isa(input_img, 'single')
        bitsPerSample = 32;
        sampleFormat = Tiff.SampleFormat.IEEEFP;
    elseif isa(input_img, 'uint16')
        bitsPerSample = 16;
        sampleFormat = Tiff.SampleFormat.UInt;
    elseif isa(input_img, 'uint8')
        bitsPerSample = 8;
        sampleFormat = Tiff.SampleFormat.UInt;
    else
        % if you want to handle other numeric classes, add them yourself
        disp('Unsupported data type');
        return;
    end
    
    % Open TIFF file in write mode
    outtiff = Tiff(filename,'w');
    
    % Loop through frames
    for f = 1:dims(3)
        % Set tag structure for each frame
        tagstruct.ImageLength = dims(1);
        tagstruct.ImageWidth = dims(2);
        tagstruct.SamplesPerPixel = dims(3);
        tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
        tagstruct.BitsPerSample = bitsPerSample;
        tagstruct.SampleFormat = sampleFormat;
        tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
        
        % Set the tag for the current frame
        outtiff.setTag(tagstruct);
        
        % Write the frame
        outtiff.write(input_img(:,:,f));
    end
    
    % Close the file
    outtiff.close();
end