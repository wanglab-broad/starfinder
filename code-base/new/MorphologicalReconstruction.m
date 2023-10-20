function input_img = MorphologicalReconstruction( input_img, radius )
% MorphologicalReconstruction
% Morphological opening is useful for removing small objects from an image while preserving the shape and size of larger objects in the image
    
    % Get dims 
    Nround = numel(input_img);
    Nchannel = size(input_img{1}, 4);
    Nslice = size(input_img{1}, 3);

    % Setup structure element
    se = strel('disk', radius);
    
    for r=1:Nround
        
        tic

        for c=1:Nchannel
            
            current_channel = input_img{r}(:,:,:,c);
            
            for z=1:Nslice
                
                current_slice = current_channel(:,:,z);
                marker = imerode(current_slice, se); 
                obr = imreconstruct(marker, current_slice);
                current_out = current_slice - obr;
                current_out = imsubtract(imadd(current_out, imtophat(current_out, se)), imbothat(current_out, se));
                current_channel(:,:,z) = current_out;

            end

            input_img{r}(:,:,:,c) = uint8(current_channel);
        end

        fprintf(sprintf('[time = %.2f s]\n', toc));
    end 


end
    

