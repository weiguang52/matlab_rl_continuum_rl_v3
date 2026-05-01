function stats = plot_reachability_lut_debug(scene, reach, varargin)
%PLOT_REACHABILITY_LUT_DEBUG Visualize Reach LUT sparsity / correctness.
%
% 用途
% ----
% 这个函数专门用来检查离线可达性查找表（Reach LUT）是否过于稀疏，
% 以及它和各层真实异物区域（圆/圆环）是否匹配。
%
% 它会输出两类图：
%   1) summary 图：四层总体可达区域（任一入口可达即为 true）
%   2) per-entry 图：某一层若干入口各自的可达区域
%
% 输入
% ----
%   scene : build_scene_model(...) 得到的 scene 结构体
%   reach : precompute_reachability(...) 得到的 reach 结构体
%
% 可选参数（Name-Value）
% ----------------------
%   'saveDir'         : 图片保存目录，默认 'reach_lut_debug'
%   'layerIds'        : 要绘制的层号，默认 1:numel(scene.layers)
%   'makeSummary'     : 是否生成四层总体图，默认 true
%   'makePerEntry'    : 是否生成单入口图，默认 true
%   'perEntryCount'   : 每层绘制前多少个入口，默认 6
%   'entryIndices'    : 指定某层入口编号，格式 cell，默认 {}
%   'supportMargin'   : 叠加支撑柱禁入圈时额外膨胀距离，默认 armRadius + 5
%   'visible'         : 'on' / 'off'，默认 'off'
%
% 输出
% ----
%   stats : 统计信息结构体，包含每层：
%       .name
%       .nEntries
%       .gridOccupancyAll
%       .gridOccupancyTargetRegion
%       .nReachableCellsTargetRegion
%       .nCellsTargetRegion
%       .summaryFile
%       .perEntryFiles
%
% 示例
% ----
%   dataDir = pwd;
%   layers  = load_all_layer_data(dataDir);
%   scene   = build_scene_model(layers);
%   S = load('reachability_lut.mat', 'reach');
%   reach = S.reach;
%   stats = plot_reachability_lut_debug(scene, reach, ...
%              'saveDir', 'reach_debug_figs', ...
%              'perEntryCount', 8, ...
%              'visible', 'on');
%
% 说明
% ----
% 图中：
%   - 背景蓝色块：Reach LUT 中被标为可达的网格
%   - 黑色外/内圈：该层异物允许出现的真实区域边界
%   - 红色圆：支撑柱禁入区（support radius + armRadius + 5）
%   - 品红点：入口位置
%
% 如果你觉得图太稀疏，优先增大：
%   precompute_reachability 里的 opts.N_samples
% 其次可酌情调：
%   opts.dilateRadius
% 不建议盲目把 cellSize 调太小，否则网格会更“碎”。

    p = inputParser;
    addParameter(p, 'saveDir', 'reach_lut_debug');
    addParameter(p, 'layerIds', 1:numel(scene.layers));
    addParameter(p, 'makeSummary', true);
    addParameter(p, 'makePerEntry', true);
    addParameter(p, 'perEntryCount', 6);
    addParameter(p, 'entryIndices', {});
    addParameter(p, 'supportMargin', scene.armRadius + 5);
    addParameter(p, 'visible', 'off');
    parse(p, varargin{:});
    opt = p.Results;

    if ~exist(opt.saveDir, 'dir')
        mkdir(opt.saveDir);
    end

    % 某些情况下 reach 是从 mat 文件直接 load 出来的，query 句柄可能不存在
    if ~isfield(reach, 'gridX') || ~isfield(reach, 'gridY') || ~isfield(reach, 'grid')
        error('reach 结构体缺少必要字段：gridX / gridY / grid');
    end

    xs = reach.gridX(:)';
    ys = reach.gridY(:)';
    [XX, YY] = meshgrid(xs, ys);

    layerIds = opt.layerIds(:)';
    stats = struct([]);

    % =====================================================
    % 1) 逐层统计
    % =====================================================
    for ii = 1:numel(layerIds)
        L = layerIds(ii);
        layer = scene.layers(L);
        gridL = logical(reach.grid{L});

        [rInner, rOuter] = i_get_layer_target_region(scene, L);
        RR = hypot(XX, YY);
        targetMask = (RR >= rInner) & (RR <= rOuter);

        nAll = numel(gridL);
        nTarget = nnz(targetMask);
        nReachAll = nnz(gridL);
        nReachTarget = nnz(gridL & targetMask);

        stats(ii).layerId = L; %#ok<AGROW>
        stats(ii).name = layer.name; %#ok<AGROW>
        stats(ii).nEntries = numel(layer.entries); %#ok<AGROW>
        stats(ii).gridOccupancyAll = nReachAll / max(nAll, 1); %#ok<AGROW>
        stats(ii).gridOccupancyTargetRegion = nReachTarget / max(nTarget, 1); %#ok<AGROW>
        stats(ii).nReachableCellsTargetRegion = nReachTarget; %#ok<AGROW>
        stats(ii).nCellsTargetRegion = nTarget; %#ok<AGROW>
        stats(ii).summaryFile = ''; %#ok<AGROW>
        stats(ii).perEntryFiles = {}; %#ok<AGROW>
    end

    % =====================================================
    % 2) 四层总体图
    % =====================================================
    if opt.makeSummary
        nL = numel(layerIds);
        nRow = ceil(sqrt(nL));
        nCol = ceil(nL / nRow);

        fig = figure('Color', 'w', 'Visible', opt.visible, ...
                     'Name', 'Reach LUT summary by layer');
        tiledlayout(nRow, nCol, 'Padding', 'compact', 'TileSpacing', 'compact');

        for ii = 1:nL
            L = layerIds(ii);
            layer = scene.layers(L);
            gridL = logical(reach.grid{L});
            nexttile;
            i_plot_single_grid(gca, scene, layer, xs, ys, gridL, ...
                               'showEntries', true, ...
                               'titleText', sprintf('%s | entries=%d | target occupancy=%.2f%%', ...
                               layer.name, numel(layer.entries), 100*stats(ii).gridOccupancyTargetRegion), ...
                               'supportMargin', opt.supportMargin);
        end

        sgtitle(sprintf('Reach LUT summary (cellSize = %.1f mm)', reach.cellSize), ...
                'FontWeight', 'bold');

        summaryFile = fullfile(opt.saveDir, 'reach_lut_summary.png');
        exportgraphics(fig, summaryFile, 'Resolution', 180);
        close(fig);

        for ii = 1:nL
            stats(ii).summaryFile = summaryFile;
        end
    end

    % =====================================================
    % 3) 每层若干单入口图
    % =====================================================
    if opt.makePerEntry
        for ii = 1:numel(layerIds)
            L = layerIds(ii);
            layer = scene.layers(L);
            nEntry = numel(layer.entries);

            if isempty(opt.entryIndices)
                idxList = 1:min(opt.perEntryCount, nEntry);
            elseif numel(opt.entryIndices) >= L && ~isempty(opt.entryIndices{L})
                idxList = opt.entryIndices{L};
                idxList = idxList(idxList >= 1 & idxList <= nEntry);
            else
                idxList = 1:min(opt.perEntryCount, nEntry);
            end

            if isempty(idxList)
                continue;
            end

            nP = numel(idxList);
            nRow = ceil(sqrt(nP));
            nCol = ceil(nP / nRow);

            fig = figure('Color', 'w', 'Visible', opt.visible, ...
                         'Name', sprintf('Reach LUT per-entry - %s', layer.name));
            tiledlayout(nRow, nCol, 'Padding', 'compact', 'TileSpacing', 'compact');

            for jj = 1:nP
                ei = idxList(jj);
                gridE = logical(reach.per_entry{L}{ei});
                entry = layer.entries(ei);
                nexttile;
                i_plot_single_grid(gca, scene, layer, xs, ys, gridE, ...
                                   'showEntries', false, ...
                                   'highlightEntryXY', entry.entryXY, ...
                                   'titleText', sprintf('#%d  %s', ei, char(entry.entryHoleLabel)), ...
                                   'supportMargin', opt.supportMargin);
            end

            sgtitle(sprintf('Per-entry Reach LUT - %s', layer.name), 'FontWeight', 'bold');
            file = fullfile(opt.saveDir, sprintf('reach_lut_per_entry_%s.png', layer.name));
            exportgraphics(fig, file, 'Resolution', 180);
            close(fig);
            stats(ii).perEntryFiles = {file};
        end
    end

    % =====================================================
    % 4) 同时写一个简洁文本报告
    % =====================================================
    reportFile = fullfile(opt.saveDir, 'reach_lut_stats.txt');
    fid = fopen(reportFile, 'w');
    if fid > 0
        fprintf(fid, 'Reach LUT statistics\n');
        fprintf(fid, '====================\n');
        fprintf(fid, 'cellSize = %.3f mm\n\n', reach.cellSize);
        for ii = 1:numel(stats)
            fprintf(fid, 'Layer %d (%s)\n', stats(ii).layerId, stats(ii).name);
            fprintf(fid, '  nEntries                 = %d\n', stats(ii).nEntries);
            fprintf(fid, '  gridOccupancyAll         = %.4f (%.2f%%%%)\n', ...
                stats(ii).gridOccupancyAll, 100*stats(ii).gridOccupancyAll);
            fprintf(fid, '  gridOccupancyTargetRegion= %.4f (%.2f%%%%)\n', ...
                stats(ii).gridOccupancyTargetRegion, 100*stats(ii).gridOccupancyTargetRegion);
            fprintf(fid, '  reachableCellsTarget     = %d / %d\n', ...
                stats(ii).nReachableCellsTargetRegion, stats(ii).nCellsTargetRegion);
            if ~isempty(stats(ii).summaryFile)
                fprintf(fid, '  summaryFile              = %s\n', stats(ii).summaryFile);
            end
            if ~isempty(stats(ii).perEntryFiles)
                fprintf(fid, '  perEntryFile             = %s\n', stats(ii).perEntryFiles{1});
            end
            fprintf(fid, '\n');
        end
        fclose(fid);
    end

    fprintf('[Reach Debug] 图已保存到: %s\n', opt.saveDir);
    fprintf('[Reach Debug] 统计报告: %s\n', reportFile);
end

% ======================================================================
function i_plot_single_grid(ax, scene, layer, xs, ys, gridL, varargin)
    p = inputParser;
    addParameter(p, 'showEntries', true);
    addParameter(p, 'highlightEntryXY', []);
    addParameter(p, 'titleText', '');
    addParameter(p, 'supportMargin', scene.armRadius + 5);
    parse(p, varargin{:});
    opt = p.Results;

    axes(ax); %#ok<LAXES>
    imagesc(xs, ys, double(gridL));
    set(ax, 'YDir', 'normal');
    axis(ax, 'equal');
    hold(ax, 'on');
    colormap(ax, [1 1 1; 0.2 0.55 0.95]);

    % 叠加真实异物区域边界
    [rInner, rOuter] = i_get_layer_target_region_from_name(scene, layer.name);
    th = linspace(0, 2*pi, 400);
    plot(ax, rOuter*cos(th), rOuter*sin(th), 'k-', 'LineWidth', 1.6);
    if rInner > 0
        plot(ax, rInner*cos(th), rInner*sin(th), 'k-', 'LineWidth', 1.6);
    end

    % 支撑柱禁入区
    for i = 1:numel(layer.supports)
        cxy = layer.supports(i).xy;
        rr = layer.supports(i).radius + opt.supportMargin;
        plot(ax, cxy(1) + rr*cos(th), cxy(2) + rr*sin(th), '-', ...
             'Color', [0.85 0.2 0.2], 'LineWidth', 0.8);
    end

    % 入口位置
    if opt.showEntries && isfield(layer, 'entries') && ~isempty(layer.entries)
        entryXY = reshape([layer.entries.entryXY], 2, []).';
        plot(ax, entryXY(:,1), entryXY(:,2), 'm.', 'MarkerSize', 9);
    end

    if ~isempty(opt.highlightEntryXY)
        plot(ax, opt.highlightEntryXY(1), opt.highlightEntryXY(2), 'mo', ...
             'MarkerSize', 8, 'LineWidth', 2);
    end

    grid(ax, 'on');
    xlabel(ax, 'X (mm)');
    ylabel(ax, 'Y (mm)');
    title(ax, opt.titleText, 'Interpreter', 'none');

    % 统一范围：优先覆盖场景网格边界
    xlim(ax, [xs(1), xs(end)]);
    ylim(ax, [ys(1), ys(end)]);
    hold(ax, 'off');
end

% ======================================================================
function [rInner, rOuter] = i_get_layer_target_region(scene, layerId)
    [rInner, rOuter] = i_get_layer_target_region_from_name(scene, scene.layers(layerId).name);
end

function [rInner, rOuter] = i_get_layer_target_region_from_name(scene, layerName)
    nm = upper(char(string(layerName)));

    if isfield(scene, 'targetRegions') && isfield(scene.targetRegions, nm)
        rr = scene.targetRegions.(nm);
        rInner = rr(1);
        rOuter = rr(2);
        return;
    end

    switch nm
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
            error('Unknown layer name: %s', nm);
    end
end
