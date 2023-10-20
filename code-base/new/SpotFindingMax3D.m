function props = SpotFindingMax3D( input_img, intensity_threshold )
%SpotFindingMax3D 

    props = [];
    Nchannel = size(input_img, 4);
    
    for c=1:Nchannel
        current_channel = input_img(:,:,:,c);
        current_max = imregionalmax(current_channel);

        max_intensity = max(current_channel, [], 'all');
        current_output = current_max & current_channel > intensity_threshold * max_intensity;

        current_props = regionprops3(current_output, current_channel, ["Centroid", "MaxIntensity"]);
        current_props.Centroid = int16(current_props.Centroid);
        props = vertcat(props, current_props);
    end

end

