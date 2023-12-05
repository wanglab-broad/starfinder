% run reads (signal) stitching workflow with 2D mouse tissue section 
% user will define:
% base_path

% test block
base_path = fullfile('/home/unix/jiahao/wanglab/Data/Analyzed/2023-10-01-Jiahao-Test/mAD_64/');

% add path for .m files
addpath(genpath(fullfile(pwd, '../code-base/new/'))) % pwd is the location of the starfinder folder

image_path = fullfile(base_path, 'images');
signal_path = fullfile(base_path, 'signal');
signal_files = dir(fullfile(signal_path, "*.csv"));
signal_files = struct2table(signal_files);
signal_files = natsort(signal_files.name);
number_of_fovs = numel(signal_files);

% load TileConfiguration from Fiji
% merge dots 
stitch_file = fullfile(image_path, "protein/PI/TileConfiguration.registered.txt");
opts = delimitedTextImportOptions("NumVariables", 6);

% specify range and delimiter
opts.DataLines = [5, Inf];
opts.Delimiter = ["(", ")", ",", ";"];

% specify column names and types
opts.VariableNames = ["Definetheimagecoordinates", "Var2", "Var3", "VarName4", "VarName5"];
opts.SelectedVariableNames = ["Definetheimagecoordinates", "VarName4", "VarName5"];
opts.VariableTypes = ["double", "string", "string", "double", "double"];

% specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts.ConsecutiveDelimitersRule = "join";

% specify variable properties
opts = setvaropts(opts, ["Var2", "Var3"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["Var2", "Var3"], "EmptyFieldRule", "auto");
opts = setvaropts(opts, "Definetheimagecoordinates", "TrimNonNumeric", true);
opts = setvaropts(opts, "Definetheimagecoordinates", "ThousandsSeparator", ",");

% import the data
TileConfiguration = readtable(stitch_file, opts);
TileConfiguration.Properties.VariableNames = {'fov' 'x' 'y'};

% Clear temporary variables
clear opts

% Global offsets 
offset_x = abs(min(TileConfiguration.x));
offset_y = abs(min(TileConfiguration.y));
TileConfiguration.x = int32(TileConfiguration.x + offset_x);
TileConfiguration.y = int32(TileConfiguration.y + offset_y);

% Merge dots 
% load stitched dapi image
dapi_file = fullfile(image_path, "fused/PI.tif");
amplicon_file = fullfile(image_path, "fused/ref_merged.tif");
dapi_img = imread(dapi_file);
amplicon_img = imread(amplicon_file);

% create empty holders 
fused_spots = table();
merged_region = [];

for r=1:size(TileConfiguration,1)

    % get each row of tileconfig
    current_row = table2cell(TileConfiguration(r, ["fov", "x", "y"]));
    [fov, x, y] = current_row{:};
    
    % load dots of each tile
    current_spots_file = fullfile(signal_path, sprintf("tile_%d_goodSpots.csv", fov));
    current_spots = readtable(current_spots_file);
    current_spots.gene = string(current_spots.gene);
    current_spots.x = int32(current_spots.x);
    current_spots.y = int32(current_spots.y);
    
    if ~isempty(current_spots)
        
        current_spots.x = current_spots.x + x;
        current_spots.y = current_spots.y + y;

        % construct dots region
        current_min = min(current_spots(:, ["x", "y"]), [], 1);
        current_max = max(current_spots(:, ["x", "y"]), [], 1);
        if current_max.x > size(dapi_img, 2)

            current_max.x = size(dapi_img, 2);
            toKeep = current_spots.x <= current_max.x;
            current_spots = current_spots(toKeep, :);
        end

        if current_max.y > size(dapi_img, 1)

            current_max.y = size(dapi_img, 1);   
            toKeep = current_spots.y <= current_max.y;
            current_spots = current_spots(toKeep, :);
            
        end
        
        current_region = zeros(size(dapi_img));
        current_region(current_min.y:current_max.y, current_min.x:current_max.x) = 1;
        
        % merge dots 
        if isempty(merged_region)
            merged_region = current_region;
        else
            current_overlap = merged_region & current_region;
            merged_region = merged_region | current_region;
            current_region = current_region - current_overlap; 
            
            current_spots_locs = table2array(current_spots(:, ["x", "y"]));
            temp_cell = num2cell(current_spots_locs, 2); 
            current_lindex = cellfun(@(x) sub2ind([size(dapi_img)], x(2), x(1)), temp_cell);
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
PlotCentroids(table2array(fused_spots(:, ["x", "y"])), amplicon_img, 1);
saveas(gcf, fullfile(image_path, 'fused/goodSpots.tif'));