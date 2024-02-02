function tile_config = ParseFijiTileConfiguration(tile_config_file)
% load TileConfiguration from Fiji

    opts = delimitedTextImportOptions("NumVariables", 6);

    % specify range and delimiter
    opts.DataLines = [5, Inf];
    opts.Delimiter = ["(", ")", ",", ";"];

    % specify column names and types
    opts.VariableNames = ["Definetheimagecoordinates", "Var2", "Var3", "VarName4", "VarName5", "VarName6"];
    opts.SelectedVariableNames = ["Definetheimagecoordinates", "VarName4", "VarName5", "VarName6"];
    opts.VariableTypes = ["string", "string", "string", "double", "double", "double"];

    % specify file level properties
    opts.ExtraColumnsRule = "ignore";
    opts.EmptyLineRule = "read";
    opts.ConsecutiveDelimitersRule = "join";

    % specify variable properties
    opts = setvaropts(opts, ["Var2", "Var3"], "WhitespaceRule", "preserve");
    opts = setvaropts(opts, ["Var2", "Var3"], "EmptyFieldRule", "auto");
    % opts = setvaropts(opts, "Definetheimagecoordinates", "TrimNonNumeric", true);
    % opts = setvaropts(opts, "Definetheimagecoordinates", "ThousandsSeparator", ",");

    % import the data
    tile_config = readtable(tile_config_file, opts);
    tile_config.Properties.VariableNames = {'grid' 'x' 'y' 'z'};
    tile_config.grid = extractBefore(tile_config.grid, '.');

    % Clear temporary variables
    clear opts

    % Global offsets 
    offset_x = abs(min(tile_config.x));
    offset_y = abs(min(tile_config.y));
    offset_z = abs(min(tile_config.z));
    tile_config.x = int32(tile_config.x + offset_x);
    tile_config.y = int32(tile_config.y + offset_y);
    tile_config.z = int32(tile_config.z + offset_z);
end
