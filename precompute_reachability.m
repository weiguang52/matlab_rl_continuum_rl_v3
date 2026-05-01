function reach = precompute_reachability(scene, robot, opts)
%PRECOMPUTE_REACHABILITY  离线预计算"(入口, 目标XY) -> 是否可达"查找表
%
% z 语义修正版：
%   scene.layers(L).zRange = 两块板之间的空间范围 [entryStartZ, targetZ]
%   entryStartZ = zRange(1)，即该层下探起点所在板面
%   targetZ     = zRange(2)，即该层异物所在板面
%   q(1)=zProbe 是从 entryStartZ 开始计算的相对下探长度。
%
% Reach LUT 只把末端 z 接近 targetZ 的样本计入可达区域。

    if nargin < 3, opts = struct(); end
    if ~isfield(opts, 'cellSize'),     opts.cellSize     = 15;    end
    if ~isfield(opts, 'N_samples'),    opts.N_samples    = 20000; end
    if ~isfield(opts, 'dilateRadius'), opts.dilateRadius = 1;     end
    if ~isfield(opts, 'cacheFile'),    opts.cacheFile    = 'reachability_lut.mat'; end
    if ~isfield(opts, 'useCache'),     opts.useCache     = true;  end
    if ~isfield(opts, 'useParallel'),  opts.useParallel  = false; end %#ok<NASGU>
    if ~isfield(opts, 'zTargetTol'),   opts.zTargetTol   = 40;    end

    if opts.useCache && isfile(opts.cacheFile)
        fprintf('[Reach] 发现缓存 %s，直接加载。\n', opts.cacheFile);
        S = load(opts.cacheFile, 'reach');
        reach = S.reach;
        reach.query = build_query_fn(reach);
        return;
    end

    fprintf('[Reach] 开始预计算可达性查找表...\n');
    t_all = tic;

    cs = opts.cellSize;
    xs = scene.gridBoundsX(1):cs:scene.gridBoundsX(2);
    ys = scene.gridBoundsY(1):cs:scene.gridBoundsY(2);
    H  = numel(ys);
    W  = numel(xs);

    reach.gridX    = xs;
    reach.gridY    = ys;
    reach.cellSize = cs;
    reach.grid      = cell(numel(scene.layers), 1);
    reach.per_entry = cell(numel(scene.layers), 1);

    for L = 1:numel(scene.layers)
        layer    = scene.layers(L);
        supports = layer.supports;
        entries  = layer.entries;
        nEntry   = numel(entries);

        entryStartZ = layer.zRange(1);
        targetZ     = layer.zRange(2);
        gap         = targetZ - entryStartZ;

        [q_low, q_high, zProbe_min, zProbe_max] = get_layer_q_bounds_relative(gap, robot);

        fprintf('[Reach] 层 %d (%s): %d 个入口  ', L, layer.name, nEntry);
        fprintf('(entryStartZ=%.2f targetZ=%.2f zProbe ∈ [%.0f, %.0f]) ', ...
            entryStartZ, targetZ, zProbe_min, zProbe_max);
        t_layer = tic;

        per_entry_L = cell(nEntry, 1);
        agg_L = false(H, W);

        for ei = 1:nEntry
            baseXY = entries(ei).entryXY(:)';
            grid_ei = sample_one_entry(baseXY, robot, supports, q_low, q_high, ...
                entryStartZ, targetZ, opts, xs, ys, H, W);
            per_entry_L{ei} = grid_ei;
            agg_L = agg_L | grid_ei;
        end

        reach.per_entry{L} = per_entry_L;
        reach.grid{L} = agg_L;

        cov_ratio = nnz(agg_L) / numel(agg_L);
        fprintf(' done in %.1fs  | 层内 XY 可达率 = %.2f%%\n', toc(t_layer), cov_ratio*100);
        if cov_ratio < 0.01
            warning('precompute_reachability:层 %d (%s) 几乎不可达 (%.3f%%)', ...
                L, layer.name, cov_ratio*100);
        end
    end

    reach.opts = opts;
    reach.query = build_query_fn(reach);

    fprintf('[Reach] 全部完成，用时 %.1fs，缓存到 %s\n', toc(t_all), opts.cacheFile);
    save(opts.cacheFile, 'reach', '-v7.3');
end

function [q_low, q_high, zProbe_min, zProbe_max] = get_layer_q_bounds_relative(gap, robot)
    chain_len_bent = robot.baseLen + 0.27 * (robot.seg1Len + robot.link1Len ...
                   + robot.seg2Len + robot.link2Len);

    zProbe_min = 0;
    zProbe_max = max(0, gap - chain_len_bent + 30);

    q_low  = [zProbe_min; -pi; -pi; -pi; -pi; -pi];
    q_high = [zProbe_max;  pi;  pi;  pi;  pi;  pi];
end

function grid_ei = sample_one_entry(baseXY, robot, supports, q_low, q_high, ...
        entryStartZ, targetZ, opts, xs, ys, H, W)

    N = opts.N_samples;
    cs = opts.cellSize;
    xMin = xs(1);
    yMin = ys(1);
    zTol = opts.zTargetTol;

    grid_ei = false(H, W);
    Q = q_low(:) + (q_high(:) - q_low(:)) .* rand(6, N);

    for k = 1:N
        q = Q(:, k);
        try
            model = continuum_forward_model(q, robot, baseXY, entryStartZ);
        catch
            continue;
        end

        tip = model.tipPos;

        % 异物只在该层板面上，因此只统计末端接近 targetZ 的样本。
        if abs(tip(3) - targetZ) > zTol
            continue;
        end

        [isCol, ~] = collision_check_centerline(model.centerline, supports, ...
            robot.armRadius, 10);
        if isCol
            continue;
        end

        ix = floor((tip(1) - xMin) / cs) + 1;
        iy = floor((tip(2) - yMin) / cs) + 1;
        if ix >= 1 && ix <= W && iy >= 1 && iy <= H
            grid_ei(iy, ix) = true;
        end
    end

    r = opts.dilateRadius;
    if r > 0
        grid_ei = dilate_bool(grid_ei, r);
    end
end

function out = dilate_bool(in, r)
    [H, W] = size(in);
    out = in;
    for dy = -r:r
        for dx = -r:r
            if dx == 0 && dy == 0, continue; end
            shifted = false(H, W);
            y1 = max(1, 1+dy);  y2 = min(H, H+dy);
            x1 = max(1, 1+dx);  x2 = min(W, W+dx);
            sy1 = max(1, 1-dy); sy2 = min(H, H-dy);
            sx1 = max(1, 1-dx); sx2 = min(W, W-dx);
            shifted(y1:y2, x1:x2) = in(sy1:sy2, sx1:sx2);
            out = out | shifted;
        end
    end
end

function fn = build_query_fn(reach)
    fn = @(layerId, entryIdx, targetXY) query_impl(reach, layerId, entryIdx, targetXY);
end

function ok = query_impl(reach, layerId, entryIdx, targetXY)
    xs = reach.gridX;
    ys = reach.gridY;
    cs = reach.cellSize;
    ix = floor((targetXY(1) - xs(1)) / cs) + 1;
    iy = floor((targetXY(2) - ys(1)) / cs) + 1;
    if ix < 1 || ix > numel(xs) || iy < 1 || iy > numel(ys)
        ok = false;
        return;
    end
    if isempty(entryIdx)
        ok = reach.grid{layerId}(iy, ix);
    else
        ok = reach.per_entry{layerId}{entryIdx}(iy, ix);
    end
end
