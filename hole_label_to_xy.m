function xy = hole_label_to_xy(holeLabel)
%HOLE_LABEL_TO_XY Convert hole label like '6D-1' to XY.

    tok = regexp(strtrim(holeLabel), '^(\d+[A-Z])-(\d)$', 'tokens', 'once');
    assert(~isempty(tok), 'Invalid hole label: %s', holeLabel);
    anchor = tok{1};
    idx = str2double(tok{2});

    base = grid_label_to_xy(anchor);
    half = 101.6 / 2;

    switch idx
        case 1
            offset = [-half, +half];
        case 2
            offset = [+half, +half];
        case 3
            offset = [+half, -half];
        case 4
            offset = [-half, -half];
        otherwise
            error('Hole index must be 1..4');
    end

    xy = base + offset;
end
