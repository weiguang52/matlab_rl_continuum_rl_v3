function yes = point_inside_any_support(supports, xy, armRadius)
%POINT_INSIDE_ANY_SUPPORT Check whether an XY point is inside any support obstacle.
%
% Inputs:
%   supports  : support struct array, each support should have fields:
%               supports(i).xy
%               supports(i).radius
%   xy        : [x, y]
%   armRadius : robot arm radius / safety radius contribution
%
% Output:
%   yes       : true if xy is inside any inflated support region

    yes = false;

    for i = 1:numel(supports)
        if norm(xy(:) - supports(i).xy(:)) < supports(i).radius + armRadius + 5
            yes = true;
            return;
        end
    end
end