function Q = run_upper_qlearning_demo(scene, numEpisodes)
%RUN_UPPER_QLEARNING_DEMO Simple upper-layer tabular Q-learning.
% State is coarse target region + layer. Action is entry choice.

    if nargin < 2, numEpisodes = 400; end
    alpha = 0.15; gamma = 0.95; eps0 = 0.25;

    maxActions = max(arrayfun(@(s) numel(s.entries), scene.layers));
    Q = zeros(4, 9, maxActions); % 4 layers x 3x3 target bins

    for ep = 1:numEpisodes
        layerId = randi(4);
        target = sample_target(scene, layerId);
        binId = target_bin_id(target(1:2));
        nA = numel(scene.layers(layerId).entries);

        if rand < max(0.02, eps0*(1-ep/numEpisodes))
            a = randi(nA);
        else
            [~, a] = min(Q(layerId, binId, 1:nA)); % smaller score better
        end

        entry = scene.layers(layerId).entries(a);
        reward = entry_reward(scene, layerId, entry, target);
        Q(layerId, binId, a) = Q(layerId, binId, a) + alpha*(reward - Q(layerId, binId, a));
    end
end

function r = entry_reward(scene, layerId, entry, target)
    supports = scene.layers(layerId).supports;

    entryXY = entry.entryXY(:);
    targetXY = target(1:2).';
    targetXY = targetXY(:);

    d = norm(entryXY - targetXY);
    anchorPenalty = 1.0 * (~strcmp(string(entry.anchorPoint), "8H"));
    blocked = 0;
    for i = 1:numel(supports)
        ctr = supports(i).xy(:);
        ds = point_to_segment_distance_2d(ctr, entryXY, targetXY);
        if ds < supports(i).radius + 120
            blocked = blocked + 1;
        end
    end
    if blocked > 1
        r = 30 + 0.005*d + 12*anchorPenalty + 25*blocked;
    else
        r = 0.005*d + 12*anchorPenalty + 8*blocked;
    end
end

function t = sample_target(scene, layerId)
    zMid = scene.layers(layerId).zRange(2);
    candidates = [];
    for i = 1:numel(scene.layers(layerId).supports)
        hxy = scene.layers(layerId).supports(i).holeXY;
        candidates = [candidates; hxy]; %#ok<AGROW>
    end
    idx = randi(size(candidates,1));
    pt = candidates(idx,:);
    noise = 30 * randn(1,2);
    t = [pt + noise, zMid];
end

function id = target_bin_id(xy)
    edges = [-inf -200 200 inf];
    [~, bx] = histc(xy(1), edges);
    [~, by] = histc(xy(2), edges);
    bx = min(max(bx,1),3);
    by = min(max(by,1),3);
    id = (bx-1)*3 + by;
end

function d = point_to_segment_distance_2d(pt, a, b)
    pt = pt(:);
    a = a(:);
    b = b(:);

    ab = b - a;
    if norm(ab) < 1e-9
        d = norm(pt - a);
        return;
    end
    t = dot(pt - a, ab) / dot(ab, ab);
    t = max(0, min(1, t));
    proj = a + t * ab;
    d = norm(pt - proj);
end