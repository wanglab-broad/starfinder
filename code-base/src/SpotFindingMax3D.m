function props = SpotFindingMax3D( input_img, intensity_estimation, intensity_threshold )
%SpotFindingMax3D 

    props = [];
    Nchannel = size(input_img, 4);
    
    for c=1:Nchannel
        current_channel = input_img(:,:,:,c);
        current_max = imregionalmax(current_channel);

        switch intensity_estimation
            case "adaptive"
                max_intensity = max(current_channel, [], 'all');
                current_threshold = max_intensity * intensity_threshold;
            case "global"
                if class(current_channel) == "uint8"
                    current_threshold = intensity_threshold * 255;
                elseif class(current_channel) == "uint16"
                    current_threshold = intensity_threshold * 65535;
                else
                    error("Unsupported image type");
                end
        end
        current_output = current_max & current_channel > current_threshold;

        current_props = regionprops3(current_output, current_channel, ["Centroid", "MaxIntensity"]);
        current_props.Centroid = int16(current_props.Centroid);
        current_props.Channel = repmat(c, size(current_props, 1), 1);
        props = vertcat(props, current_props);
    end

end

