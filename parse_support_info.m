function supports = parse_support_info(filename)
%PARSE_SUPPORT_INFO Parse support file lines like 8K_38.1,8H_0,...
% Returns struct array with fields: anchorLabel, xy, radius, holeLabels, holeXY.
%
% NOTE:
%   radius <= 0 means there is NO support pillar at that anchor.
%   Such entries are skipped completely, so downstream logic will not add
%   collision inflation / safety layer / avoidance behavior there.

    txt = fileread(filename);
    txt = strrep(txt, sprintf('\r'), '');
    rows = regexp(txt, ';\n|;|\n', 'split');
    rows = rows(~cellfun(@isempty, strtrim(rows)));

    supports = struct('anchorLabel', {}, 'xy', {}, 'radius', {}, ...
                      'holeLabels', {}, 'holeXY', {});
    for i = 1:numel(rows)
        parts = regexp(strtrim(rows{i}), ',', 'split');
        for j = 1:numel(parts)
            token = strtrim(parts{j});
            if isempty(token)
                continue;
            end
            kv = regexp(token, '^(\d+[A-Z])_([\d\.]+)$', 'tokens', 'once');
            if isempty(kv)
                continue;
            end

            anchor = kv{1};
            radius = str2double(kv{2});

            % radius == 0 (or invalid / negative) means no support pillar.
            if ~isfinite(radius) || radius <= 0
                continue;
            end

            hlabels = strcat(anchor, {'-1','-2','-3','-4'});
            hxy = zeros(4,2);
            for k = 1:4
                hxy(k,:) = hole_label_to_xy(hlabels{k});
            end
            supports(end+1) = struct( ...
                'anchorLabel', anchor, ...
                'xy', grid_label_to_xy(anchor), ...
                'radius', radius, ...
                'holeLabels', {hlabels}, ...
                'holeXY', hxy); %#ok<AGROW>
        end
    end
end