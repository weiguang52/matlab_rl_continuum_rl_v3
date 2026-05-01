function results = joint_evaluation_universal(scene, robot, upper_results, lower_results, n_eval, reach)
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
%   lower_results 通用下层 SAC 结果（train_lower_sac_universal 输出）
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

    actor    = lower_results.actor;
    act_low  = lower_results.act_low;
    act_high = lower_results.act_high;
    lower_opts = lower_results.opts;

    for layerId = 1:numel(scene.layers)
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
                    [traj, ok, col_hit, steps] = lower_rollout_universal( ...
                        scene, layerId, robot, entry, tgt, actor, ...
                        act_low, act_high, lower_opts, 150);
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
%  下层 rollout（使用通用 SAC actor）
% =========================================================
function [traj, ep_success, ep_col, ep_steps] = lower_rollout_universal( ...
        scene, layerId, robot, entry, target, actor, act_low, act_high, lower_opts, max_steps)

    [q_low, q_high, zProbe_min, zProbe_max, entryStartZ] = get_layer_q_bounds(scene, layerId, robot);
    supports = scene.layers(layerId).supports;

    q = [0; 0; 0; 0; 0; 0];
    model = continuum_forward_model(q, robot, entry.entryXY, entryStartZ);
    prev_action = zeros(6,1);
    obs = build_obs_lower_universal(q, model.tipPos, target, supports, prev_action, ...
        entry, scene, zProbe_min, zProbe_max, lower_opts, act_high);

    traj.q   = zeros(6, max_steps);
    traj.tip = zeros(max_steps, 3);
    traj.col = false(max_steps, 1);

    ep_success = false;
    ep_col     = false;
    ep_steps   = max_steps;

    for step = 1:max_steps
        x = dlarray(single(obs(:)), 'CB');
        out = double(extractdata(predict(actor, x)));
        half = numel(out)/2;
        mu = out(1:half);
        squashed = tanh(mu);  % 评估时用均值动作，不采样
        action = double(act_low) + (double(act_high) - double(act_low)) .* (squashed + 1) / 2;

        q_new = clamp_q(q + action, q_low, q_high);
        model_new = continuum_forward_model(q_new, robot, entry.entryXY, entryStartZ);

        [isCol, ~] = collision_check_centerline(model_new.centerline, ...
            supports, robot.armRadius, lower_opts.safety_margin);

        traj.q(:, step)   = q_new;
        traj.tip(step, :) = model_new.tipPos;
        traj.col(step)    = isCol;

        q = q_new;
        model = model_new;
        obs = build_obs_lower_universal(q, model.tipPos, target, supports, action, ...
            entry, scene, zProbe_min, zProbe_max, lower_opts, act_high);

        if isCol
            ep_col = true;
            ep_steps = step;
            traj.q   = traj.q(:,1:step);
            traj.tip = traj.tip(1:step,:);
            traj.col = traj.col(1:step);
            break;
        end
        if norm(target(:) - model.tipPos(:)) < lower_opts.goal_radius
            ep_success = true;
            ep_steps = step;
            traj.q   = traj.q(:,1:step);
            traj.tip = traj.tip(1:step,:);
            traj.col = traj.col(1:step);
            break;
        end
    end
end

function [q_low, q_high, zProbe_min, zProbe_max, entryStartZ] = get_layer_q_bounds(scene, layerId, robot)
    entryStartZ = scene.layers(layerId).zRange(1);
    targetZ     = scene.layers(layerId).zRange(2);
    gap         = targetZ - entryStartZ;

    chain_len_bent = robot.baseLen + 0.27 * (robot.seg1Len + robot.link1Len ...
                   + robot.seg2Len + robot.link2Len);

    zProbe_min = 0;
    zProbe_max = max(0, gap - chain_len_bent + 30);

    q_low  = [zProbe_min; -pi; -pi; -pi; -pi; -pi];
    q_high = [zProbe_max;  pi;  pi;  pi;  pi;  pi];
end

% =========================================================
%  通用下层观测构建（与 train_lower_sac_universal.m 保持一致，27维）
% =========================================================
function obs = build_obs_lower_universal(q, tipPos, target, supports, prev_action, ...
        entry, scene, zProbe_min, zProbe_max, opts, act_high)

    posScale = opts.obs_pos_scale;
    err = (target(:) - tipPos(:)) / posScale;

    q_norm = q(:);
    zMid = (zProbe_min + zProbe_max)/2;
    zHalf = max((zProbe_max - zProbe_min)/2, 1);
    q_norm(1) = (q(1) - zMid) / zHalf;
    q_norm(2:end) = q(2:end) / pi;

    dmin = inf;
    rel2 = [0;0];
    nearestRadius = 0;
    for i = 1:numel(supports)
        ctr = supports(i).xy;
        d = norm(tipPos(1:2) - ctr) - supports(i).radius;
        if d < dmin
            dmin = d;
            v = ctr(:) - tipPos(1:2)';
            rel2 = v / max(norm(v), 1);
            nearestRadius = supports(i).radius;
        end
    end
    if isinf(dmin), dmin = 500; end
    dmin_norm = min(max(dmin,0),500) / 500;
    dist3d_norm = norm(target(:) - tipPos(:)) / posScale;
    z_err_norm = abs(target(3) - tipPos(3)) / posScale;

    actScale = max(abs(double(act_high(:))), 1e-6);
    prev_action_norm = prev_action(:) ./ actScale;

    xR = scene.gridBoundsX;
    yR = scene.gridBoundsY;
    entryXY = entry.entryXY(:);
    entryXY_norm = [(entryXY(1) - mean(xR))/(diff(xR)/2 + 1e-6); ...
                    (entryXY(2) - mean(yR))/(diff(yR)/2 + 1e-6)];
    target_rel_entry_norm = (target(1:2)' - entryXY) / posScale;

    zProbe_bounds_norm = [zProbe_min; zProbe_max] / opts.zprobe_norm_scale;
    radius_norm = nearestRadius / opts.radius_norm_scale;

    obs = [err(:); q_norm(:); dmin_norm; rel2(:); dist3d_norm; z_err_norm; ...
           prev_action_norm(:); entryXY_norm(:); target_rel_entry_norm(:); ...
           zProbe_bounds_norm(:); radius_norm];
    obs = obs(1:27);
end

function q = clamp_q(q, q_low, q_high)
    q = max(min(q, q_high), q_low);
    q(2) = wrapToPi(q(2));
    q(4) = wrapToPi(q(4));
    q(6) = wrapToPi(q(6));
end
