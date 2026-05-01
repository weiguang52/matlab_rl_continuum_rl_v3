function [x, y] = sample_target_xy_by_layer(scene, layerId)
%SAMPLE_TARGET_XY_BY_LAYER Sample one target XY inside layer-specific region.
%
% 采用面积均匀采样：
%   圆形区域:   r = R * sqrt(rand)
%   圆环区域:   r = sqrt(rInner^2 + (rOuter^2-rInner^2)*rand)

    [rInner, rOuter] = get_layer_target_region(scene, layerId);

    rho = sqrt(rInner^2 + (rOuter^2 - rInner^2) * rand());
    ang = 2*pi*rand();

    x = rho * cos(ang);
    y = rho * sin(ang);
end