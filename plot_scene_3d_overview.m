function fig = plot_scene_3d_overview(scene, varargin)
%PLOT_SCENE_3D_OVERVIEW  三维可视化整个场景结构
%
% 用法：
%   fig = plot_scene_3d_overview(scene);
%   fig = plot_scene_3d_overview(scene, 'showEntries', true, 'showLabels', true);
%
% 功能：
%   - 绘制四层板（CSP / LGP / SGP / BP）
%   - 绘制各层支撑柱
%   - 绘制入口点
%   - 按项目中的“z 向下为正”方式显示
%
% 可选参数：
%   'showEntries'   : 是否绘制入口点，默认 true
%   'showLabels'    : 是否绘制文字标签，默认 true
%   'plateAlpha'    : 板面透明度，默认 0.18
%   'supportAlpha'  : 支撑柱透明度，默认 0.35
%   'fig'           : 指定 figure 句柄，默认新建
%
% 注意：
%   这里假定 scene.layers(i) 至少包含：
%       .name
%       .zRange
%       .supports
%       .entries
%
%   并结合你当前项目里已经改好的“目标区域”定义：
%       CSP: 直径 3358 的圆
%       LGP: 外径 3254，内径 1933 的圆环
%       SGP: 外径 2190，内径 358 的圆环
%       BP : 外径 1158，内径 570 的圆环

    p = inputParser;
    addParameter(p, 'showEntries', true);
    addParameter(p, 'showLabels', true);
    addParameter(p, 'plateAlpha', 0.18);
    addParameter(p, 'supportAlpha', 0.35);
    addParameter(p, 'fig', []);
    parse(p, varargin{:});
    opt = p.Results;

    if isempty(opt.fig) || ~isgraphics(opt.fig)
        fig = figure('Color', 'w', 'Name', '3D Scene Overview');
    else
        fig = opt.fig;
        figure(fig); clf(fig);
    end

    ax = axes(fig);
    hold(ax, 'on');
    axis(ax, 'equal');
    grid(ax, 'on');
    view(ax, 3);

    xlabel(ax, 'X / mm');
    ylabel(ax, 'Y / mm');
    zlabel(ax, 'Z / mm');
    title(ax, 'Scene 3D Overview');

    % 为了符合“z 向下为正”的直观显示效果
    set(ax, 'ZDir', 'reverse');

    % 每层给一个颜色
    layerColors = [
        0.20 0.55 0.95;   % CSP
        0.20 0.75 0.35;   % LGP
        0.95 0.60 0.20;   % SGP
        0.80 0.25 0.25    % BP
    ];

    th = linspace(0, 2*pi, 240);

    % 记录所有 z 用于限制视图
    allZ = [];

    for li = 1:numel(scene.layers)
        layer = scene.layers(li);
        c = layerColors(min(li,size(layerColors,1)), :);

        % zRange = [entryStartZ, targetZ]。
        % 入口/下探起点画在 entryStartZ；异物所在板面画在 targetZ。
        entryStartZ = layer.zRange(1);
        zBoard = layer.zRange(2);
        allZ = [allZ, entryStartZ, zBoard]; %#ok<AGROW>

        [rIn, rOut] = get_layer_region_by_name(layer.name);

        % 1. 画该层真实异物板面（CSP/LGP/SGP/BP 对应四个 zBoard）
        draw_ring_plate(ax, rIn, rOut, zBoard, c, opt.plateAlpha);

        % 画层名称
        if opt.showLabels
            text(ax, rOut*0.72, 0, zBoard, ...
                sprintf('%s board z=%.2f, entry z=%.2f', layer.name, zBoard, entryStartZ), ...
                'Color', c, 'FontWeight', 'bold', 'FontSize', 11, ...
                'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
        end

        % 2. 画支撑柱，柱子占据两层板之间的空间
        if isfield(layer, 'supports') && ~isempty(layer.supports)
            for si = 1:numel(layer.supports)
                ctr = layer.supports(si).xy(:)';
                rr  = layer.supports(si).radius;
                draw_cylinder_z(ax, ctr(1), ctr(2), rr, ...
                    entryStartZ, zBoard, [0.35 0.35 0.35], opt.supportAlpha);
            end
        end

        % 3. 画入口点，入口在上层板面 entryStartZ，不在目标板面 zBoard
        if opt.showEntries && isfield(layer, 'entries') && ~isempty(layer.entries)
            for ei = 1:numel(layer.entries)
                exy = layer.entries(ei).entryXY(:)';
                plot3(ax, exy(1), exy(2), entryStartZ, 'm.', 'MarkerSize', 14);

                if opt.showLabels && mod(ei, max(1, ceil(numel(layer.entries)/15))) == 1
                    nameStr = get_entry_name(layer.entries(ei));
                    text(ax, exy(1), exy(2), entryStartZ, ['  ' char(nameStr)], ...
                        'Color', [0.45 0 0.55], 'FontSize', 8);
                end
            end
        end

        % 4. 画板面圆/圆环边界
        plot3(ax, rOut*cos(th), rOut*sin(th), zBoard*ones(size(th)), ...
            '-', 'Color', c, 'LineWidth', 1.5);
        if rIn > 0
            plot3(ax, rIn*cos(th), rIn*sin(th), zBoard*ones(size(th)), ...
                '-', 'Color', c, 'LineWidth', 1.2);
        end
    end

    % 坐标范围
    if isfield(scene, 'gridBoundsX')
        xlim(ax, scene.gridBoundsX);
    end
    if isfield(scene, 'gridBoundsY')
        ylim(ax, scene.gridBoundsY);
    end
    if ~isempty(allZ)
        zmin = min(allZ) - 100;
        zmax = max(allZ) + 100;
        zlim(ax, [zmin zmax]);
    end

    % 原点
    plot3(ax, 0, 0, 0, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6);
    if opt.showLabels
        text(ax, 0, 0, 0, '  Origin', 'FontSize', 9, 'Color', 'k');
    end

    legend(ax, {'Layer plate / boundary', 'Support', 'Entry'}, 'Location', 'bestoutside');
    hold(ax, 'off');
end

% =========================================================
% 画圆 / 圆环板
% =========================================================
function draw_ring_plate(ax, rIn, rOut, z0, colorVal, faceAlphaVal)
    nr = 2;
    nt = 180;
    rr = linspace(rIn, rOut, nr);
    tt = linspace(0, 2*pi, nt);
    [TT, RR] = meshgrid(tt, rr);

    XX = RR .* cos(TT);
    YY = RR .* sin(TT);
    ZZ = z0 * ones(size(XX));

    surf(ax, XX, YY, ZZ, ...
        'FaceColor', colorVal, ...
        'EdgeColor', 'none', ...
        'FaceAlpha', faceAlphaVal);
end

% =========================================================
% 画沿 z 方向的支撑柱
% =========================================================
function draw_cylinder_z(ax, xc, yc, r, z1, z2, colorVal, faceAlphaVal)
    n = 40;
    [X, Y, Z] = cylinder(r, n);
    Z = Z * (z2 - z1) + z1;
    X = X + xc;
    Y = Y + yc;

    surf(ax, X, Y, Z, ...
        'FaceColor', colorVal, ...
        'EdgeColor', 'none', ...
        'FaceAlpha', faceAlphaVal);

    % 上下盖
    th = linspace(0, 2*pi, 100);
    x = xc + r*cos(th);
    y = yc + r*sin(th);

    patch(ax, x, y, z1*ones(size(x)), colorVal, ...
        'EdgeColor', 'none', 'FaceAlpha', faceAlphaVal);
    patch(ax, x, y, z2*ones(size(x)), colorVal, ...
        'EdgeColor', 'none', 'FaceAlpha', faceAlphaVal);
end

% =========================================================
% 根据层名返回真实目标区域（半径）
% =========================================================
function [rInner, rOuter] = get_layer_region_by_name(layerName)
    nm = upper(string(layerName));

    switch nm
        case "CSP"
            rInner = 0;
            rOuter = 3358 / 2;
        case "LGP"
            rInner = 1933 / 2;
            rOuter = 3254 / 2;
        case "SGP"
            rInner = 358 / 2;
            rOuter = 2190 / 2;
        case "BP"
            rInner = 570 / 2;
            rOuter = 1158 / 2;
        otherwise
            error('Unknown layer name: %s', layerName);
    end
end

% =========================================================
% 获取入口名称
% =========================================================
function nameStr = get_entry_name(entry)
    if isfield(entry, 'holeName')
        nameStr = string(entry.holeName);
    elseif isfield(entry, 'entryHoleLabel')
        nameStr = string(entry.entryHoleLabel);
    elseif isfield(entry, 'label')
        nameStr = string(entry.label);
    else
        nameStr = "entry";
    end
end