function [bestEntry, allScores] = choose_entry_rule_based(scene, layerId, target)
%CHOOSE_ENTRY_RULE_BASED Choose one RobotArm entry for a target.
% Score favors: fewer anchor relocations, short xy distance, and less obstacle density.

    entries = scene.layers(layerId).entries;
    supports = scene.layers(layerId).supports;
    n = numel(entries);
    allScores = nan(n,1);

    tgtXY = target(1:2);
    tgtXY = tgtXY(:).';   % force 1x2 row vector

    for i = 1:n
        e = entries(i);
        entryXY = e.entryXY;
        entryXY = entryXY(:).';  % force 1x2 row vector

        anchorPenalty = 0.0;
        if ~strcmp(string(e.anchorPoint), "8H")
            anchorPenalty = 1.0;
        end

        d = norm(entryXY - tgtXY);
        localObs = count_obstacles_near_line(entryXY, tgtXY, supports, 180);
        allScores(i) = 8*anchorPenalty + 0.003*d + 6*localObs;
    end

    [~, idx] = min(allScores);
    bestEntry = entries(idx);
end

function c = count_obstacles_near_line(p0, p1, supports, clearance)
    p0 = p0(:);
    p1 = p1(:);

    c = 0;
    for k = 1:numel(supports)
        ctr = supports(k).xy;
        ctr = ctr(:);
        r = supports(k).radius + 27.5 + clearance;
        d = point_to_segment_distance_2d(ctr, p0, p1);
        if d < r
            c = c + 1;
        end
    end
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