% run reads (signal) stitching workflow 
% user will define:
% config_path

function reads_stitching(config_path)

    % load dataset info 
    config_raw = fileread(config_path);
    config = jsondecode(config_raw);

    % add path for .m files
    addpath(fullfile(pwd, './code-base/src/')) % pwd is the location of the starfinder folder

    image_path = fullfile(config.output_path, 'images');
    signal_path = fullfile(config.output_path, 'signal');

    % load TileConfiguration from Fiji
    tile_config_file = fullfile(image_path, string(config.additional_round), config.ref_channel, "TileConfiguration.registered.txt");
    tile_config = ParseFijiTileConfiguration(tile_config_file);

    % Merge dots 
    % load stitched nuclei image
    nuclei_file = fullfile(image_path, sprintf("fused/%s.tif", config.ref_channel));
    nuclei_img_metadata = imfinfo(nuclei_file);
    nuclei_img_dims = [nuclei_img_metadata(1).Height nuclei_img_metadata(1).Width length(nuclei_img_metadata)];

    if nuclei_img_dims(3) > 1
        amplicon_file = fullfile(image_path, "fused/MAX_ref_merged.tif"); % need a Maximum projection for visualziation 
    else
        amplicon_file = fullfile(image_path, "fused/ref_merged.tif");
    end
    amplicon_img = imread(amplicon_file);

    % create empty holders 
    fused_spots = table();
    merged_region = [];

    for r=1:size(tile_config,1)

        % get each row of tileconfig
        current_row = table2cell(tile_config(r, ["fov", "x", "y", "z"]));
        [fov, x, y, z] = current_row{:};
        
        % load dots of each tile
        % fname_pattern = strcat(config.fov_id_pattern, "_goodSpots.csv");
        current_spots_file = fullfile(signal_path, sprintf("%s_goodSpots.csv", fov));
        current_spots = readtable(current_spots_file);
        current_spots.gene = string(current_spots.gene);
        current_spots.x = int32(current_spots.x);
        current_spots.y = int32(current_spots.y);
        current_spots.z = int32(current_spots.z);

        if ~isempty(current_spots)
            
            current_spots.x = current_spots.x + x;
            current_spots.y = current_spots.y + y;
            current_spots.z = current_spots.z + z;

            % construct dots region
            current_min = min(current_spots(:, ["x", "y", "z"]), [], 1);
            current_max = max(current_spots(:, ["x", "y", "z"]), [], 1);
            if current_max.x > nuclei_img_dims(2)

                current_max.x = nuclei_img_dims(2);
                toKeep = current_spots.x <= current_max.x;
                current_spots = current_spots(toKeep, :);
            end

            if current_max.y > nuclei_img_dims(1)

                current_max.y = nuclei_img_dims(1);   
                toKeep = current_spots.y <= current_max.y;
                current_spots = current_spots(toKeep, :);
                
            end
            
            if current_max.z > nuclei_img_dims(3) && nuclei_img_dims(3) ~= 1

                current_max.z = nuclei_img_dims(3);   
                toKeep = current_spots.z <= current_max.z;
                current_spots = current_spots(toKeep, :);
                
            end

            current_region = zeros(nuclei_img_dims);

            if nuclei_img_dims(3) > 1 % 3D
                current_region(current_min.y:current_max.y, current_min.x:current_max.x, current_min.z:current_max.z) = 1;
            else % 2D
                current_region(current_min.y:current_max.y, current_min.x:current_max.x) = 1;
            end
            
            % merge dots 
            if isempty(merged_region)
                merged_region = current_region;
            else
                current_overlap = merged_region & current_region;
                merged_region = merged_region | current_region;
                current_region = current_region - current_overlap; 
                
                current_spots_locs = table2array(current_spots(:, ["x", "y", "z"]));
                temp_cell = num2cell(current_spots_locs, 2);

                if nuclei_img_dims(3) > 1
                    current_lindex = cellfun(@(x) sub2ind(nuclei_img_dims, x(2), x(1), x(3)), temp_cell);
                else
                    current_lindex = cellfun(@(x) sub2ind(nuclei_img_dims, x(2), x(1)), temp_cell);
                end

                current_logical = logical(current_region(current_lindex));
                current_spots = current_spots(current_logical, :);
            end
            
            % save current 
            if isempty(fused_spots)
                fused_spots = current_spots;
            else
                fused_spots = vertcat(fused_spots, current_spots);
            end
        end

    end

    writetable(fused_spots, fullfile(signal_path, 'fused_goodSpots.csv'));
    PlotCentroids(fused_spots(:, ["x", "y"]), amplicon_img, "red", .1);
    exportgraphics(gcf, fullfile(image_path, 'fused/goodSpots.tif'), 'Resolution', 1000, 'ContentType', 'image')
end