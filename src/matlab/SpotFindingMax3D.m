function props = SpotFindingMax3D( input_img, intensity_estimation, intensity_threshold )
% Find per-channel 3-D regional maxima above an intensity threshold.
%
% input_img has shape (row, column, Z, C). intensity_estimation is "adaptive"
% (fraction of each channel maximum) or "global" (fraction of uint8/uint16
% full scale). intensity_threshold is that fraction; always supply both options.
% There is no "noise" mode. Maxima must be strictly above the threshold.
% Returns per-channel regionprops3 rows with Centroid, MaxIntensity and Channel.
% Centroid is cast to int16 and contains 1-based (x=column, y=row, z=plane).
% Channels are concatenated without cross-channel deduplication.

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

