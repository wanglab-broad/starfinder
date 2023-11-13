% run reads stitching workflow with 2D mouse tissue section 
% user will define:
% image_path  
% reads_path
% input_dim

% image_path = '/home/unix/jiahao/wanglab/Data/Analyzed/2023-09-19-wendyw-WW_SC_005/images/stitching/';
% reads_path = '/home/unix/jiahao/wanglab/wendyw/WW_SC_005/02.processed_data/2023-09-19-wendyw-WW_SC_005/';
% input_dim = [2048 2048 58 4];

% add path for .m files
addpath(genpath(fullfile(pwd, 'code-base/new/'))) % pwd is the location of the starfinder folder
addpath(genpath(fullfile(pwd, 'example/')))

data_dirs = dir(fullfile(reads_path, "Position*"));
data_dirs = struct2table(data_dirs);
data_dirs = natsort(data_dirs.name);

Ndirs = numel(data_dirs);

% load TileConfiguration from Fiji
% merge dots 
stitch_file = fullfile(image_path, "DAPI/TileConfiguration.txt");
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
TileConfiguration.Properties.VariableNames = {'tile' 'x' 'y'};

% load grid file
grid_file = fullfile(image_path, current_sample, "grid.csv");
grid = readtable(grid_file);

TileConfiguration.grid = arrayfun(@(x) sprintf("tile_%02d", x), TileConfiguration.tile);

TileConfiguration = join(TileConfiguration, grid);

% Clear temporary variables
clear opts

% Global offsets 
offset_x = abs(min(TileConfiguration.x));
offset_y = abs(min(TileConfiguration.y));
% offset_z = abs(min(TileConfiguration.z));

TileConfiguration.x = TileConfiguration.x + offset_x + 1;
TileConfiguration.y = TileConfiguration.y + offset_y + 1;
% TileConfiguration.z = TileConfiguration.z + offset_z + 1;

% Merge dots 
% load stitched dapi image
dapi_file = fullfile(image_path, current_sample, "DAPI_fused.tif");
dapi_max = imread_big(dapi_file);

% create empty holders 
merged_points = table();
merged_region = [];

upd = textprogressbar(size(TileConfiguration,1), 'updatestep', 2);

for r=1:size(TileConfiguration,1)

    
    % get each row of tileconfig
    current_row = table2cell(TileConfiguration(r, ["x", "y", "id"]));
    [x, y, id] = current_row{:};
    
    if id ~= 0
        % load dots of each tile
        current_position_dir = sprintf("Position%03d", id);
        current_dot_file = fullfile(reads_path, current_position_dir, "goodPoints_max3d.csv");
        current_dot = readtable(current_dot_file);
        current_dot.Gene = string(current_dot.Gene);
        
        if ~isempty(current_dot)
            
            current_dot.x = current_dot.x + x;
            current_dot.y = current_dot.y + y;

            % construct dots region
            current_min = min(current_dot(:, ["x", "y"]), [], 1);
            current_max = max(current_dot(:, ["x", "y"]), [], 1);
            if current_max.x > size(dapi_max, 2)

                current_max.x = size(dapi_max, 2);
                toKeep = current_dot.x <= current_max.x;
                current_dot = current_dot(toKeep, :);
            end

            if current_max.y > size(dapi_max, 1)

                current_max.y = size(dapi_max, 1);   
                toKeep = current_dot.y <= current_max.y;
                current_dot = current_dot(toKeep, :);
                
            end
            
            current_region = zeros(size(dapi_max));
            current_region(current_min.y:current_max.y, current_min.x:current_max.x) = 1;
            
            % merge dots 
            if isempty(merged_region)
                merged_region = current_region;
            else
                current_overlap = merged_region & current_region;
                merged_region = merged_region | current_region;
                current_region = current_region - current_overlap; 
                
                current_dot_locs = table2array(current_dot(:, ["x", "y"]));
                temp_cell = num2cell(current_dot_locs, 2); 
                current_lindex = cellfun(@(x) sub2ind([size(dapi_max)], x(2), x(1)), temp_cell);
                current_logical = logical(current_region(current_lindex));
                current_dot = current_dot(current_logical, :);
            end
            
            % save current 
            if isempty(merged_points)
                merged_points = current_dot;
            else
                merged_points = vertcat(merged_points, current_dot);
            end
        end

    end

    upd(r);
end

writetable(merged_points, fullfile(image_path, current_sample, 'merged_goodPoints_max3d.csv'));
PlotCentroids(table2array(merged_points(:, ["x", "y"])), dapi_max, 1);
saveas(gcf, fullfile(image_path, current_sample, 'test.tif'));


