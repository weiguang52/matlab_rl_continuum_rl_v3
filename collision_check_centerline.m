function [isCollision, minClearance] = collision_check_centerline(centerlinePts, supports, armRadius, margin)
%COLLISION_CHECK_CENTERLINE Check centerline against inflated cylinders.
% Entries with radius <= 0 are treated as non-existent supports.

    isCollision = false;
    minClearance = inf;
    inflate = armRadius + margin;

    for i = 1:numel(supports)
        if ~isfield(supports(i), 'radius') || ~isfinite(supports(i).radius) || supports(i).radius <= 0
            continue;
        end

        ctr = supports(i).xy;
        safeR = supports(i).radius + inflate;
        d = sqrt((centerlinePts(:,1)-ctr(1)).^2 + (centerlinePts(:,2)-ctr(2)).^2);
        cur = min(d - safeR);
        minClearance = min(minClearance, cur);
        if any(d <= safeR)
            isCollision = true;
            return;
        end
    end
end