function results = joint_evaluation(scene, robot, upper_results, lower_results_by_layer, n_eval, reach)
%JOINT_EVALUATION  多目标覆盖评估 + 下层 SAC 实际控制
%
%  新版流程（v2）
%  --------------
%  每次评估生成一批目标（与训练时分布一致），
%  上层 DQN 贪心地序贯选择入口，直到所有目标被入口可达域覆盖（或达上限）。
%  然后对"入口 → 其覆盖的目标子集"调用下层 SAC 做真实 rollout，
%  统计定位点数、下探孔数、实际到达成功率、碰撞率。
%
%  输入
%  ----
%   scene, robot          场景与机器人参数
%   upper_results         train_upper_dqn 输出（需含 net_online, maxActions, anchorTable, opts）
%   lower_results_by_layer cell(4,1)，每层下层 SAC 结果
%   n_eval                每层评估的 episode 数（默认 20）
%   reach                 可达性查找表（必需）
%
%  输出
%  ----
%   results.overall_success_rate   全目标的到达成功率（下层 SAC 最终考核）
%   results.coverage_rate          上层入口规划的覆盖率
%   results.collision_rate         下层 rollout 碰撞率
%   results.mean_anchors           平均每 ep 使用的定位点数
%   results.mean_holes             平均每 ep 使用的下探孔数
%   results.mean_steps_per_goal    平均每目标的步数
%   results.example_traj           一条示例成功轨迹（首条）

    if nargin < 5 || isempty(n_eval), n_eval = 20; end
    if nargin < 6 || isempty(reach)
        error('joint_evaluation:需要 reach 可达性查找表（传入 precompute_reachability 的结果）');
    end

    total_ep           = 0;
    sum_anchors        = 0;
    sum_holes          = 0;
    sum_coverage       = 0;
    total_goals        = 0;
    reached_goals      = 0;
    col_goals          = 0;
    sum_steps          = 0;

    example_traj       = [];
    example_info       = [];

    upper_opts = upper_results.opts;

    for layerId = 1:numel(scene.layers)
        lower = lower_results_by_layer{layerId};
        if isempty(lower)
            fprintf('[Eval] 层 %d 无下层结果，跳过。\n', layerId);
            continue;
        end

        actor    = lower.actor;
        act_low  = lower.act_low;
        act_high = lower.act_high;

        for ep = 1:n_eval
            % ---- 生成目标集（与训练分布一致）----
            targets = sample_eval_targets(scene, layerId, reach, upper_opts);
            nT = size(targets, 1);
            if nT == 0, continue; end

            % ---- 上层序贯选入口 ----
            [entry_list, target_assign, covered] = upper_sequential_plan( ...
                scene, layerId, targets, upper_results, reach);

            nAnchorsUsed = numel(unique(arrayfun(@(e) string(e.anchorPoint), entry_list)));
            nHolesUsed   = numel(entry_list);
            coverage_rate = mean(covered);

            sum_anchors  = sum_anchors + nAnchorsUsed;
            sum_holes    = sum_holes   + nHolesUsed;
            sum_coverage = sum_coverage + coverage_rate;
            total_ep     = total_ep + 1;
            total_goals  = total_goals + nT;

            % ---- 对每个"入口 → 其分配的目标子集" 做下层 rollout ----
            for eiIdx = 1:numel(entry_list)
                entry     = entry_list(eiIdx);
                gidx_list = find(target_assign == eiIdx);
                for gi = gidx_list(:)'
                    tgt = targets(gi, :);
                    [traj, ok, col_hit, steps] = lower_rollout( ...
                        scene, layerId, robot, entry, tgt, actor, ...
                        act_low, act_high, 150);
                    sum_steps = sum_steps + steps;
                    if ok, reached_goals = reached_goals + 1; end
                    if col_hit, col_goals = col_goals + 1; end

                    if ok && isempty(example_traj)
                        example_traj = traj;
                        example_info.layerId = layerId;
                        example_info.entry   = entry;
                        example_info.target  = tgt;
                    end
                end
            end
        end
    end

    if total_ep == 0
        warning('joint_evaluation: 没有有效评估 episode');
        results.overall_success_rate = 0;
        results.coverage_rate        = 0;
        results.collision_rate       = 0;
        results.mean_anchors         = 0;
        results.mean_holes           = 0;
        results.mean_steps_per_goal  = 0;
        results.example_traj         = [];
        results.example_info         = [];
        return;
    end

    results.overall_success_rate = reached_goals / max(total_goals, 1);
    results.coverage_rate        = sum_coverage / total_ep;
    results.collision_rate       = col_goals / max(total_goals, 1);
    results.mean_anchors         = sum_anchors / total_ep;
    results.mean_holes           = sum_holes / total_ep;
    results.mean_steps_per_goal  = sum_steps / max(total_goals, 1);
    results.example_traj         = example_traj;
    results.example_info         = example_info;
end

% =========================================================
%  上层序贯规划：贪心地按 DQN 输出选入口，直到全覆盖或达上限
% =========================================================
function [entry_list, target_assign, covered] = upper_sequential_plan( ...
        scene, layerId, targets, upper_results, reach)

    net         = upper_results.net_online;
    maxA        = upper_results.maxActions;
    anchorTable = upper_results.anchorTable;
    opts        = upper_results.opts;

    nEntry = numel(scene.layers(layerId).entries);
    nT     = size(targets, 1);
    nAnch  = numel(anchorTable);

    covered       = false(nT, 1);
    used_anchors  = false(nAnch, 1);
    used_holes    = false(nEntry, 1);
    target_assign = zeros(nT, 1);

    entry_list = [];

    for step = 1:opts.max_steps
        if all(covered), break; end

        obs = encode_state_eval(scene, layerId, targets, covered, used_anchors, ...
                                 used_holes, anchorTable, opts);
        x = dlarray(single(obs(:)), 'CB');
        qvals = double(extractdata(predict(net, x)));
        qvals_valid = qvals(1:nEntry);

        % 屏蔽已用 hole（与训练一致）
        if opts.mask_used_holes
            qvals_valid(used_holes) = +inf;
            if all(isinf(qvals_valid))
                break;   % 所有 hole 都试过了
            end
        end
        [~, a] = min(qvals_valid);

        entry     = scene.layers(layerId).entries(a);
        anchorId  = find(anchorTable == string(entry.anchorPoint), 1);
        if isempty(anchorId), anchorId = 1; end

        % 计算本步覆盖的目标
        newly = false(nT, 1);
        for t = 1:nT
            if ~covered(t) && reach.query(layerId, a, targets(t, 1:2))
                newly(t) = true;
            end
        end

        if ~any(newly) && opts.mask_used_holes
            % 屏蔽当前入口后继续探索（防止陷死）
            used_holes(a) = true;
            continue;
        end

        target_assign(newly) = numel(entry_list) + 1;
        covered = covered | newly;
        used_anchors(anchorId) = true;
        used_holes(a) = true;

        if isempty(entry_list)
            entry_list = entry;
        else
            entry_list(end+1) = entry; %#ok<AGROW>
        end
    end
end

% =========================================================
%  评估阶段目标采样
% =========================================================
function targets = sample_eval_targets(scene, layerId, reach, opts)
    zTarget = scene.layers(layerId).zRange(2);   % +Z 向下后，目标取该层下表面/较深边界

    nT = randi([opts.n_targets_min, opts.n_targets_max]);
    targets = zeros(nT, 3);
    filled = 0;
    tries = 0;
    max_tries = nT * 200;

    while filled < nT && tries < max_tries
        tries = tries + 1;

        [x, y] = sample_target_xy_by_layer(scene, layerId);

        if point_inside_any_support(scene.layers(layerId).supports, [x, y], scene.armRadius)
            continue;
        end

        if ~reach.query(layerId, [], [x, y])
            continue;
        end

        filled = filled + 1;
        targets(filled, :) = [x, y, zTarget];
    end

    targets = targets(1:filled, :);
end

function yes = point_inside_any_support(supports, xy, armRadius)
    yes = false;
    for i = 1:numel(supports)
        if norm(xy(:) - supports(i).xy(:)) < supports(i).radius + armRadius + 5
            yes = true; return;
        end
    end
end

% =========================================================
%  上层状态编码（与训练保持一致）
% =========================================================
function obs = encode_state_eval(scene, layerId, targets, covered, used_anchors, ...
                                  used_hole_set, anchorTable, opts)
    nLayers = numel(scene.layers);
    layer_oh = zeros(nLayers, 1);
    layer_oh(layerId) = 1;

    unmet_idx = find(~covered);
    nUnmet = numel(unmet_idx);
    unmet_count_norm = nUnmet / opts.n_targets_max;

    if nUnmet > 0
        cen = mean(targets(unmet_idx, 1:2), 1);
        bb  = max(targets(unmet_idx, 1:2), [], 1) - min(targets(unmet_idx, 1:2), [], 1);
    else
        cen = [0, 0]; bb = [0, 0];
    end
    xR = scene.gridBoundsX; yR = scene.gridBoundsY;
    cen_n = [(cen(1) - mean(xR))/(diff(xR)/2+1e-6); ...
             (cen(2) - mean(yR))/(diff(yR)/2+1e-6)];
    bb_n  = [bb(1)/(diff(xR)+1e-6); bb(2)/(diff(yR)+1e-6)];
    used_hole_norm = sum(used_hole_set) / max(numel(used_hole_set), 1);

    obs = [layer_oh; unmet_count_norm; cen_n; bb_n; used_anchors(:); used_hole_norm];
end

% =========================================================
%  下层 rollout（使用训练好的 actor，保持与原版一致）
% =========================================================
function [traj, ep_success, ep_col, ep_steps] = lower_rollout( ...
        scene, layerId, robot, entry, target, actor, act_low, act_high, max_steps)

    % zRange = [entryStartZ, targetZ]；zProbe 是从 entryStartZ 开始的相对下探长度
    entryStartZ = scene.layers(layerId).zRange(1);
    targetZ = scene.layers(layerId).zRange(2);
    gap = targetZ - entryStartZ;
    chain_len_bent = robot.baseLen + 0.27 * (robot.seg1Len + robot.link1Len ...
                   + robot.seg2Len + robot.link2Len);
    zProbe_min = 0;
    zProbe_max = max(0, gap - chain_len_bent + 30);

    q = [0; 0; 0; 0; 0; 0];
    model = continuum_forward_model(q, robot, entry.entryXY, entryStartZ);
    prev_action = zeros(6,1);
    obs = build_obs_lower(q, model.tipPos, target, ...
                          scene.layers(layerId).supports, prev_action);

    q_low  = [zProbe_min; -pi; -1.3; -pi; -1.3; -pi];
    q_high = [zProbe_max;  pi;  1.3;  pi;  1.3;  pi];

    traj.q   = zeros(6, max_steps);
    traj.tip = zeros(max_steps, 3);
    traj.col = false(max_steps, 1);

    ep_success = false;
    ep_col     = false;
    ep_steps   = max_steps;

    for step = 1:max_steps
        x   = dlarray(single(obs(:)), 'CB');
        out = double(extractdata(predict(actor, x)));
        half = numel(out)/2;
        mu   = out(1:half);
        squashed = tanh(mu);
        action = act_low + (act_high - act_low) .* (squashed + 1) / 2;

        q_new = max(min(q + action, q_high), q_low);
        q_new(2) = wrapToPi(q_new(2));
        q_new(4) = wrapToPi(q_new(4));
        q_new(6) = wrapToPi(q_new(6));

        model_new = continuum_forward_model(q_new, robot, entry.entryXY, entryStartZ);
        [isCol, ~] = collision_check_centerline(model_new.centerline, ...
            scene.layers(layerId).supports, robot.armRadius, 10);

        traj.q(:, step)   = q_new;
        traj.tip(step, :) = model_new.tipPos;
        traj.col(step)    = isCol;

        q = q_new; model = model_new;
        obs = build_obs_lower(q, model.tipPos, target, ...
                              scene.layers(layerId).supports, action);

        if isCol
            ep_col   = true;
            ep_steps = step;
            break;
        end
        if norm(target(:) - model.tipPos(:)) < 35
            ep_success = true;
            ep_steps   = step;
            traj.q   = traj.q(:, 1:step);
            traj.tip = traj.tip(1:step, :);
            traj.col = traj.col(1:step);
            break;
        end
    end
end

% =========================================================
%  下层观测构建（与 train_lower_sac.m 保持一致）
% =========================================================
function obs = build_obs_lower(q, tipPos, target, supports, prev_action)
    err = target(:) - tipPos(:);
    dmin = inf; rel2 = [0; 0];
    for i = 1:numel(supports)
        ctr = supports(i).xy;
        d = norm(tipPos(1:2) - ctr) - supports(i).radius;
        if d < dmin
            dmin = d;
            rel2 = (ctr(:) - tipPos(1:2)') / max(norm(ctr(:) - tipPos(1:2)'), 1);
        end
    end
    dmin_norm = min(max(dmin, 0), 500) / 500;
    dist3d = norm(err);
    z_err  = abs(target(3) - tipPos(3));
    obs = [err(:); q(:); dmin_norm; rel2(:); dist3d; z_err; prev_action(:)];
    obs = obs(1:20);
end
