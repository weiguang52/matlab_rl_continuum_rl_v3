function xy = grid_label_to_xy(label)
%GRID_LABEL_TO_XY Convert anchor/support label like '8H' to XY.
% Origin is cell-center label 8H -> (0,0).
% Numeric index changes x. Letter index changes y.
% User rule verified by example: 6D -> (215*2, 215*4).

    letters = {'A','B','C','D','E','F','G','H','J','K','L','M','N','P','R'};
    pitch = 215.0;

    tok = regexp(strtrim(label), '^(\d+)([A-Z])$', 'tokens', 'once');
    assert(~isempty(tok), 'Invalid grid label: %s', label);
    rowNum = str2double(tok{1});
    colLetter = tok{2};

    row0 = 8;
    col0 = 'H';

    colIdx = find(strcmp(letters, colLetter), 1);
    colIdx0 = find(strcmp(letters, col0), 1);
    assert(~isempty(colIdx), 'Invalid column letter: %s', colLetter);

    dx = (row0 - rowNum) * pitch;
    dy = (colIdx0 - colIdx) * pitch;

    xy = [dx, dy];
end
