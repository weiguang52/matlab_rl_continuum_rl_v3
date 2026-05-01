function scene = build_scene_model(layerData)
%BUILD_SCENE_MODEL Convert parsed layer data into geometry scene.

    scene.layers = layerData;
    scene.letters = {'A','B','C','D','E','F','G','H','J','K','L','M','N','P','R'};
    scene.gridPitch = 215.0;
    scene.holeRadius = 34.5;
    scene.armRadius = 55/2;
    scene.safetyMargin = 10.0;
    scene.safeInflation = scene.armRadius + scene.safetyMargin;

    % Z geometry (+Z points downward).
    % layerBoundaryZ includes the top insertion plane plus the four target boards.
    % boardZ are the four physical boards where foreign objects may lie.
    scene.topZ = 0;
    scene.boardZ = [736.42, 1714.02, 2209.62, 2783.78];
    scene.layerBoundaryZ = [scene.topZ, scene.boardZ];
    scene.plateZ = scene.layerBoundaryZ;  % backward-compatible alias for drawing only

    % ---------------------------------------------------------
    % 异物可能出现的平面区域约束，单位 mm
    % 格式: [innerRadius, outerRadius]
    %
    % CSP: 直径 3358 的圆
    % LGP: 外圈直径 3254，内圈直径 1933 的圆环
    % SGP: 外圈直径 2190，内圈直径 358 的圆环
    % BP : 外圈直径 1158，内圈直径 570 的圆环
    % ---------------------------------------------------------
    scene.targetRegions = struct();
    scene.targetRegions.CSP = [0,        3358/2];
    scene.targetRegions.LGP = [1933/2,   3254/2];
    scene.targetRegions.SGP = [358/2,    2190/2];
    scene.targetRegions.BP  = [570/2,    1158/2];

    % Reach LUT / 目标采样网格范围需要覆盖最大异物区域。
    % 原来 [-7,7]*215 = [-1505,1505]，无法完整覆盖 CSP 半径 1679。
    % 这里扩展到 [-8,8]*215 = [-1720,1720]。
    scene.gridBoundsX = [-8, 8] * scene.gridPitch;
    scene.gridBoundsY = [-8, 8] * scene.gridPitch;
end