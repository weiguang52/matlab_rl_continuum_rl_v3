function [traj, info] = rollout_lower_sac_universal(scene, robot, lower_results, layerId, entry, target, varargin)
%ROLLOUT_LOWER_SAC_UNIVERSAL  用训练好的通用下层 SAC 执行一次确定性 rollout。
%
% 用法：
%   [traj, info] = rollout_lower_sac_universal(scene, robot, lower_results, layerId, entry, target)
%   [traj, info] = rollout_lower_sac_universal(..., 'maxSteps', 150, 'useMeanAction', true)
%
% 输入：
%   scene, robot, lower_results  : 场景、机器人、通用 SAC 训练结果
%   layerId                      : 层号
%   entry                        : 入口结构体 scene.layers(layerId).entries(i)
%   target                       : 目标点 [x,y,z]
%
% 输出：
%   traj.q         : 6 x T 的关节轨迹
%   traj.tip       : T x 3 的末端轨迹
%   traj.col       : T x 1 的碰撞标记
%   traj.clearance : T x 1 的最小净空
%   traj.dist      : T x 1 的目标距离
%   traj.actions   : 6 x T 的动作序列
%   traj.model{k}  : 每一步的前向模型结果
%
%   info.success   : 是否到达
%   info.collision : 是否碰撞
%   info.steps     : 实际执行步数
%   info.goalDist  : 最终距离目标的距离

    p = inputParser;
    addParameter(p, 'maxSteps', 150);
    addParameter(p, 'useMeanAction', true);
    parse(p, varargin{:});
    maxSteps = p.Results.maxSteps;
    useMeanAction = p.Results.useMeanAction;

    actor      = lower_results.actor;
    act_low    = lower_results.act_low;
    act_high   = lower_results.act_high;
    lower_opts = lower_results.opts;

    if ~isfield(lower_opts, 'goal_radius')
        lower_opts.goal_radius = 35.0;
    end
    if ~isfield(lower_opts, 'safety_margin')
        lower_opts.safety_margin = 10.0;
    end
    if ~isfield(lower_opts, 'obs_pos_scale')
        lower_opts.obs_pos_scale = 1000.0;
    end
    if ~isfield(lower_opts, 'zprobe_norm_scale')
        lower_opts.zprobe_norm_scale = 3000.0;
    end
    if ~isfield(lower_opts, 'radius_norm_scale')
        lower_opts.radius_norm_scale = 500.0;
    end

    [q_low, q_high, zProbe_min, zProbe_max, entryStartZ] = get_layer_q_bounds(scene, layerId, robot);
    supports = scene.layers(layerId).supports;

    q = [0; 0; 0; 0; 0; 0];
    model = continuum_forward_model(q, robot, entry.entryXY, entryStartZ);
    prev_action = zeros(6,1);

    obs = build_obs_lower_universal(q, model.tipPos, target, supports, prev_action, ...
        entry, scene, zProbe_min, zProbe_max, lower_opts, act_high);

    traj.q         = zeros(6, maxSteps);
    traj.tip       = zeros(maxSteps, 3);
    traj.col       = false(maxSteps, 1);
    traj.clearance = nan(maxSteps, 1);
    traj.dist      = nan(maxSteps, 1);
    traj.actions   = zeros(6, maxSteps);
    traj.model     = cell(maxSteps, 1);

    info.success   = false;
    info.collision = false;
    info.steps     = maxSteps;
    info.goalDist  = inf;
    info.entry     = entry;
    info.target    = target;
    info.layerId   = layerId;
    info.goalRadius = lower_opts.goal_radius;

    for step = 1:maxSteps
        action = choose_action(actor, obs, act_low, act_high, useMeanAction);

        q_new = clamp_q(q + action, q_low, q_high);
        model_new = continuum_forward_model(q_new, robot, entry.entryXY, entryStartZ);

        [isCol, minClr] = collision_check_centerline(model_new.centerline, ...
            supports, robot.armRadius, lower_opts.safety_margin);

        distNow = norm(target(:) - model_new.tipPos(:));

        traj.q(:, step)         = q_new;
        traj.tip(step, :)       = model_new.tipPos;
        traj.col(step)          = isCol;
        traj.clearance(step)    = minClr;
        traj.dist(step)         = distNow;
        traj.actions(:, step)   = action;
        traj.model{step}        = model_new;

        q = q_new;
        model = model_new; %#ok<NASGU>
        obs = build_obs_lower_universal(q, model_new.tipPos, target, supports, action, ...
            entry, scene, zProbe_min, zProbe_max, lower_opts, act_high);

        if isCol
            info.collision = true;
            info.steps = step;
            info.goalDist = distNow;
            traj = trim_traj(traj, step);
            return;
        end

        if distNow < lower_opts.goal_radius
            info.success = true;
            info.steps = step;
            info.goalDist = distNow;
            traj = trim_traj(traj, step);
            return;
        end
    end

    info.goalDist = traj.dist(end);
end

% =========================================================
% Actor 动作选择
% =========================================================
function action = choose_action(actor, obs, act_low, act_high, useMeanAction)
    x = dlarray(single(obs(:)), 'CB');
    out = double(extractdata(predict(actor, x)));

    half = numel(out) / 2;
    mu = out(1:half);
    log_std = max(min(out(half+1:end), 2), -20);
    std_val = exp(log_std);

    if useMeanAction
        z = mu;
    else
        z = mu + std_val .* randn(size(mu));
    end

    sq = tanh(z);
    action = double(act_low) + (double(act_high) - double(act_low)) .* (sq + 1) / 2;
end

% =========================================================
% 裁剪轨迹到实际长度
% =========================================================
function traj = trim_traj(traj, step)
    traj.q         = traj.q(:,1:step);
    traj.tip       = traj.tip(1:step,:);
    traj.col       = traj.col(1:step);
    traj.clearance = traj.clearance(1:step);
    traj.dist      = traj.dist(1:step);
    traj.actions   = traj.actions(:,1:step);
    traj.model     = traj.model(1:step);
end

% =========================================================
% 获取当前层 q 范围
% =========================================================
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
% 通用下层观测构建，27 维
% 需要和 train_lower_sac_universal.m 中保持一致
% =========================================================
function obs = build_obs_lower_universal(q, tipPos, target, supports, prev_action, ...
        entry, scene, zProbe_min, zProbe_max, opts, act_high)

    posScale = opts.obs_pos_scale;

    % 1. 目标相对末端误差
    err = (target(:) - tipPos(:)) / posScale;

    % 2. q 归一化
    q_norm = q(:);
    zMid = (zProbe_min + zProbe_max)/2;
    zHalf = max((zProbe_max - zProbe_min)/2, 1);
    q_norm(1) = (q(1) - zMid) / zHalf;
    q_norm(2:end) = q(2:end) / pi;

    % 3. 最近障碍信息
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

    if isinf(dmin)
        dmin = 500;
    end

    dmin_norm = min(max(dmin,0),500) / 500;

    % 4. 目标距离
    dist3d_norm = norm(target(:) - tipPos(:)) / posScale;
    z_err_norm = abs(target(3) - tipPos(3)) / posScale;

    % 5. 上一步动作归一化
    actScale = max(abs(double(act_high(:))), 1e-6);
    prev_action_norm = prev_action(:) ./ actScale;

    % 6. 入口信息
    xR = scene.gridBoundsX;
    yR = scene.gridBoundsY;
    entryXY = entry.entryXY(:);

    entryXY_norm = [
        (entryXY(1) - mean(xR))/(diff(xR)/2 + 1e-6);
        (entryXY(2) - mean(yR))/(diff(yR)/2 + 1e-6)
    ];

    target_rel_entry_norm = (target(1:2)' - entryXY) / posScale;

    % 7. 当前层 zProbe 范围信息
    zProbe_bounds_norm = [zProbe_min; zProbe_max] / opts.zprobe_norm_scale;

    % 8. 最近障碍半径
    radius_norm = nearestRadius / opts.radius_norm_scale;

    obs = [
        err(:);
        q_norm(:);
        dmin_norm;
        rel2(:);
        dist3d_norm;
        z_err_norm;
        prev_action_norm(:);
        entryXY_norm(:);
        target_rel_entry_norm(:);
        zProbe_bounds_norm(:);
        radius_norm
    ];

    obs = obs(1:27);
end

% =========================================================
% q 限幅
% =========================================================
function q = clamp_q(q, q_low, q_high)
    q = max(min(q, q_high), q_low);

    q(2) = wrapToPi(q(2));
    q(4) = wrapToPi(q(4));
    q(6) = wrapToPi(q(6));
end