function entries = parse_exploration_info(filename)
%PARSE_EXPLORATION_INFO Parse exploration txt with 3 columns.
% Keeps only RobotArm entries.

    T = readtable(filename, 'FileType', 'text', 'Delimiter', '\t');
    T = T(strcmpi(T.ExplorationType, 'RobotArm'), :);

    n = height(T);
    entries = struct('anchorPoint', {}, 'entryHoleLabel', {}, 'anchorXY', {}, 'entryXY', {});
    for i = 1:n
        entries(end+1) = struct( ...
            'anchorPoint', string(T.AnchorPoint{i}), ...
            'entryHoleLabel', string(T.ExplorationNum{i}), ...
            'anchorXY', grid_label_to_xy(T.AnchorPoint{i}), ...
            'entryXY', hole_label_to_xy(T.ExplorationNum{i})); %#ok<AGROW>
    end
end
