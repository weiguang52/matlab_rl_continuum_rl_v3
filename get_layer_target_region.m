function [rInner, rOuter] = get_layer_target_region(scene, layerId)
%GET_LAYER_TARGET_REGION Return valid foreign-object radial range for a layer.
%
% Output:
%   rInner : inner radius, mm
%   rOuter : outer radius, mm
%
% Layer regions:
%   CSP: disk, diameter 3358
%   LGP: annulus, outer diameter 3254, inner diameter 1933
%   SGP: annulus, outer diameter 2190, inner diameter 358
%   BP : annulus, outer diameter 1158, inner diameter 570

    layerName = upper(char(string(scene.layers(layerId).name)));

    % 优先使用 scene.targetRegions，方便以后在 build_scene_model 中统一改参数
    if isfield(scene, 'targetRegions') && isfield(scene.targetRegions, layerName)
        rr = scene.targetRegions.(layerName);
        rInner = rr(1);
        rOuter = rr(2);
        return;
    end

    % 兜底默认值
    switch layerName
        case 'CSP'
            rInner = 0;
            rOuter = 3358/2;
        case 'LGP'
            rInner = 1933/2;
            rOuter = 3254/2;
        case 'SGP'
            rInner = 358/2;
            rOuter = 2190/2;
        case 'BP'
            rInner = 570/2;
            rOuter = 1158/2;
        otherwise
            error('Unknown layer name: %s', layerName);
    end
end