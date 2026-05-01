function results = train_lower_sac_universal(scene, robot, reach, opts)
%TRAIN_LOWER_SAC_UNIVERSAL  跨层通用下层 SAC 训练脚本
%
%  训练目标
%  --------
%  训练一个通用的下层 SAC 控制器，而不是每层单独训练一个 SAC。
%  每个 episode 随机选择 layerId、entry 和 target，使 Actor 学会：
%       给定入口 entry + 目标 target + 当前构型 q + 障碍信息 supports
%       输出连续动作 Δq，使连续体机械臂无碰撞到达目标。
%
%  与旧版 train_lower_sac 的主要区别
%  -------------------------------
%  1. 不再固定 layerId，不再固定默认 entry。
%  2. 每个 episode 随机采样一个可达的 entry-target pair。
%  3. 状态从 20 维扩展为 27 维，加入 entry 和层深度上下界信息。
%  4. 输出一个共享 actor，联合评估时所有层共用该 actor。
%
%  观测 obs = 27 维：
%    err_norm(3)              : target - tipPos，按 obs_pos_scale 归一化
%    q_norm(6)                : q 归一化，zProbe 按本层范围，角度按 pi
%    dmin_norm(1)             : 最近障碍距离归一化
%    rel2(2)                  : 最近障碍相对方向
%    dist3d_norm(1)           : 末端到目标距离归一化
%    z_err_norm(1)            : z 方向误差归一化
%    prev_action_norm(6)      : 上一步动作归一化
%    entryXY_norm(2)          : 入口坐标归一化
%    target_rel_entry_norm(2) : 目标相对入口坐标归一化
%    zProbe_bounds_norm(2)    : 本层 zProbe_min / zProbe_max 归一化
%    nearest_support_radius(1): 最近障碍半径归一化

    if nargin < 4, opts = struct(); end
    opts = default_opts(opts);

    C       = opts.curriculum;
    N_EP    = opts.num_episodes;
    MAX_ST  = opts.max_steps;
    BATCH   = opts.batch;
    LR_ACT  = opts.lr_actor;
    LR_CRI  = opts.lr_critic;
    LR_AL   = opts.lr_alpha;
    GAMMA   = opts.gamma;
    TAU     = opts.tau;
    BUF_CAP = opts.buf_cap;

    ACT_DIM = 6;
    OBS_DIM = 27;
    TARGET_ENTROPY = -ACT_DIM;

    fprintf('[Universal SAC] 初始化共享下层 SAC 网络 obs=%d act=%d\n', OBS_DIM, ACT_DIM);

    [actor, c1_online, c2_online, c1_target, c2_target] = ...
        build_sac_networks(OBS_DIM, ACT_DIM, opts.hidden);

    log_alpha_val = 0.0;

    % 这里的动作是每一步的 Δq，而不是绝对 q。
    % 为了更平滑和更容易训练，角度增量不建议过大；如需更激进，可在 main 中覆盖。
    act_low  = single(opts.act_low(:));
    act_high = single(opts.act_high(:));

    buf = SACReplayBuffer(BUF_CAP, OBS_DIM, ACT_DIM);

    reward_hist      = zeros(N_EP, 1);
    success_hist     = zeros(N_EP, 1);
    layer_hist       = zeros(N_EP, 1);
    loss_actor_hist  = [];
    loss_critic_hist = [];

    for ep = 1:N_EP
        [layerId, entry, target, q_low, q_high, zProbe_min, zProbe_max, entryStartZ] = ...
            sample_universal_episode(scene, robot, reach, ep, C, opts);

        supports = scene.layers(layerId).supports;
        layer_hist(ep) = layerId;

        q = [0; 0; 0; 0; 0; 0];
        model = continuum_forward_model(q, robot, entry.entryXY, entryStartZ);
        prev_action = zeros(ACT_DIM, 1);

        obs = build_obs_lower_universal(q, model.tipPos, target, supports, ...
            prev_action, entry, scene, zProbe_min, zProbe_max, opts, act_high);

        ep_reward = 0;
        ep_success = false;

        for step = 1:MAX_ST
            action = actor_sample_action(actor, obs, act_low, act_high);

            q_raw = q + double(action);
            q_new = clamp_q(q_raw, q_low, q_high);
            model_new = continuum_forward_model(q_new, robot, entry.entryXY, entryStartZ);

            [isCol, minClr] = collision_check_centerline(model_new.centerline, ...
                supports, robot.armRadius, opts.safety_margin);

            local_opts = opts;
            local_opts.zProbe_min = zProbe_min;
            local_opts.zProbe_max = zProbe_max;

            [reward, done, success] = compute_lower_reward(q, q_new, ...
                model.tipPos, model_new.tipPos, target, isCol, minClr, ...
                step, MAX_ST, local_opts);

            obs_next = build_obs_lower_universal(q_new, model_new.tipPos, target, ...
                supports, action, entry, scene, zProbe_min, zProbe_max, opts, act_high);

            buf.push(obs, action, reward, obs_next, double(done));

            ep_reward = ep_reward + reward;
            q = q_new;
            model = model_new;
            obs = obs_next;

            if buf.size >= BATCH
                [actor, c1_online, c2_online, c1_target, c2_target, ...
                 log_alpha_val, la, lc] = sac_update(actor, c1_online, c2_online, ...
                    c1_target, c2_target, log_alpha_val, buf, BATCH, ...
                    LR_ACT, LR_CRI, LR_AL, GAMMA, TAU, TARGET_ENTROPY, act_low, act_high);

                loss_actor_hist(end+1)  = la; %#ok<AGROW>
                loss_critic_hist(end+1) = lc; %#ok<AGROW>
            end

            if success, ep_success = true; end
            if done, break; end
        end

        reward_hist(ep)  = ep_reward;
        success_hist(ep) = ep_success;

        if mod(ep, 100) == 0
            phase = get_phase(ep, C);
            win = max(1, ep-99):ep;
            sr = mean(success_hist(win));
            fprintf('[Universal SAC] ep=%d phase=%d meanR=%.1f succRate=%.2f | layers=%s\n', ...
                ep, phase, mean(reward_hist(win)), sr, mat2str(unique(layer_hist(win))'));
        end
    end

    results.actor        = actor;
    results.c1           = c1_online;
    results.c2           = c2_online;
    results.reward_hist  = reward_hist;
    results.success_hist = success_hist;
    results.layer_hist   = layer_hist;
    results.loss_actor   = loss_actor_hist(:);
    results.loss_critic  = loss_critic_hist(:);
    results.obs_dim      = OBS_DIM;
    results.act_low      = act_low;
    results.act_high     = act_high;
    results.is_universal = true;
    results.opts         = opts;

    fprintf('[Universal SAC] 训练完成。最终成功率 (last 200 ep) = %.2f\n', ...
        mean(success_hist(max(1,end-199):end)));

    plot_lower_curves(reward_hist, success_hist, loss_actor_hist, loss_critic_hist);
end

% =========================================================
%  Episode 采样：随机层 + 可达 entry-target pair
% =========================================================
function [layerId, entry, target, q_low, q_high, zProbe_min, zProbe_max, entryStartZ] = ...
        sample_universal_episode(scene, robot, reach, ep, C, opts)

    phase = get_phase(ep, C);
    nLayers = numel(scene.layers);
    maxTries = opts.sample_max_tries;

    for tries = 1:maxTries
        layerId = randi(nLayers);
        entries = scene.layers(layerId).entries;
        if isempty(entries), continue; end

        [q_low, q_high, zProbe_min, zProbe_max, entryStartZ, zTarget] = get_layer_q_bounds(scene, layerId, robot);

        if phase == 0 || phase == 1
            % 前两个阶段：先随机入口，再在入口附近采样目标。
            % 这样更像“从某个下探孔局部控制到附近异物”的简单课程。
            radius = opts.local_radius_phase0;
            if phase == 1, radius = opts.local_radius_phase1; end

            ei = randi(numel(entries));
            entry = entries(ei);
            [ok, target] = sample_target_near_entry(scene, reach, layerId, ei, entry, zTarget, radius, opts);
            if ok, return; end
        else
            % 后期：在该层完整异物区域中采样 target，再从 Reach LUT 中找可达入口。
            [x, y] = sample_target_xy_by_layer(scene, layerId);
            if point_inside_any_support(scene.layers(layerId).supports, [x, y], scene.armRadius)
                continue;
            end
            candidate = find_reachable_entries(reach, layerId, [x, y], numel(entries));
            if isempty(candidate), continue; end
            ei = candidate(randi(numel(candidate)));
            entry = entries(ei);
            target = [x, y, zTarget];
            return;
        end
    end

    % 兜底：如果一直没采到可达 pair，就随机层、随机入口、层内随机目标。
    layerId = randi(nLayers);
    entries = scene.layers(layerId).entries;
    entry = entries(randi(numel(entries)));
    [x, y] = sample_target_xy_by_layer(scene, layerId);
    [q_low, q_high, zProbe_min, zProbe_max, entryStartZ, zTarget] = get_layer_q_bounds(scene, layerId, robot);
    target = [x, y, zTarget];
end

function [ok, target] = sample_target_near_entry(scene, reach, layerId, entryIdx, entry, zTarget, radius, opts)
    ok = false;
    target = [];
    for t = 1:opts.local_sample_tries
        r = radius * sqrt(rand());
        a = 2*pi*rand();
        x = entry.entryXY(1) + r*cos(a);
        y = entry.entryXY(2) + r*sin(a);

        if ~is_xy_in_layer_target_region(scene, layerId, [x, y])
            continue;
        end
        if point_inside_any_support(scene.layers(layerId).supports, [x, y], scene.armRadius)
            continue;
        end
        if ~reach.query(layerId, entryIdx, [x, y])
            continue;
        end

        target = [x, y, zTarget];
        ok = true;
        return;
    end
end

function idx = find_reachable_entries(reach, layerId, targetXY, nEntry)
    idx = [];
    for ei = 1:nEntry
        if reach.query(layerId, ei, targetXY)
            idx(end+1) = ei; %#ok<AGROW>
        end
    end
end

function [q_low, q_high, zProbe_min, zProbe_max, entryStartZ, targetZ] = get_layer_q_bounds(scene, layerId, robot)
    % zRange = [entryStartZ, targetZ]，表示两层板之间的空间；
    % zProbe 是从 entryStartZ 开始的相对下探长度。
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
%  网络构建
% =========================================================
function [actor, c1, c2, c1t, c2t] = build_sac_networks(obsDim, actDim, hidden)
    actor_layers = [
        featureInputLayer(obsDim,'Normalization','none','Name','obs')
        fullyConnectedLayer(hidden(1),'Name','fc1')
        reluLayer('Name','r1')
        fullyConnectedLayer(hidden(2),'Name','fc2')
        reluLayer('Name','r2')
        fullyConnectedLayer(2*actDim,'Name','out')
    ];
    actor = dlnetwork(layerGraph(actor_layers));
    c1    = build_critic(obsDim, actDim, hidden);
    c2    = build_critic(obsDim, actDim, hidden);
    c1t   = c1;
    c2t   = c2;
end

function net = build_critic(obsDim, actDim, hidden)
    layers = [
        featureInputLayer(obsDim+actDim,'Normalization','none','Name','sa')
        fullyConnectedLayer(hidden(1),'Name','fc1')
        reluLayer('Name','r1')
        fullyConnectedLayer(hidden(2),'Name','fc2')
        reluLayer('Name','r2')
        fullyConnectedLayer(1,'Name','q')
    ];
    net = dlnetwork(layerGraph(layers));
end

% =========================================================
%  动作采样
% =========================================================
function action = actor_sample_action(actor, obs, act_low, act_high)
    x = dlarray(single(obs(:)), 'CB');
    out = double(extractdata(predict(actor, x)));
    half = numel(out) / 2;
    mu = out(1:half);
    log_std = max(min(out(half+1:end), 2), -20);
    std_val = exp(log_std);
    z = mu + std_val .* randn(size(mu));
    sq = tanh(z);
    action = double(act_low) + (double(act_high) - double(act_low)) .* (sq+1)/2;
end

% =========================================================
%  SAC 更新
% =========================================================
function [actor, c1, c2, c1t, c2t, log_alpha_val, la, lc] = sac_update( ...
        actor, c1, c2, c1t, c2t, log_alpha_val, buf, ...
        batch, lr_a, lr_c, lr_al, gamma, tau, target_ent, act_low, act_high)

    [s_raw, a_raw, r_raw, s2_raw, done_raw] = buf.sample(batch);
    s    = single(s_raw);
    a    = single(a_raw);
    r    = single(r_raw(:)');
    s2   = single(s2_raw);
    done = single(done_raw(:)');
    alpha = single(exp(log_alpha_val));

    next_act = zeros(numel(act_low), batch, 'single');
    next_lp  = zeros(1, batch, 'single');
    for i = 1:batch
        x_i = dlarray(s2(:,i), 'CB');
        out_i = double(extractdata(predict(actor, x_i)));
        half = numel(out_i)/2;
        mu_i = out_i(1:half);
        lst_i = max(min(out_i(half+1:end), 2), -20);
        std_i = exp(lst_i);
        z_i = mu_i + std_i .* randn(size(mu_i));
        sq_i = tanh(z_i);
        na_i = double(act_low) + (double(act_high)-double(act_low)).*(sq_i+1)/2;
        lp_i = sum(-0.5*((z_i-mu_i)./std_i).^2) - sum(lst_i) ...
             - numel(mu_i)*0.5*log(2*pi) ...
             - sum(log(max(1 - sq_i.^2, 1e-6)));
        next_act(:,i) = single(na_i);
        next_lp(i) = single(lp_i);
    end

    sa2_dl = dlarray([s2; next_act], 'CB');
    q1_tgt = extractdata(predict(c1t, sa2_dl));
    q2_tgt = extractdata(predict(c2t, sa2_dl));
    q_min_t = min(q1_tgt, q2_tgt);
    td_target = r + gamma .* (1-done) .* (q_min_t - alpha .* next_lp);
    td_dl = dlarray(td_target, 'CB');
    sa_dl = dlarray([s; a], 'CB');

    [g1, lc1_val] = dlfeval(@critic_loss_fn, c1, sa_dl, td_dl);
    lc1 = double(extractdata(lc1_val));
    c1 = sgd_update(c1, g1, lr_c);

    [g2, lc2_val] = dlfeval(@critic_loss_fn, c2, sa_dl, td_dl);
    lc2 = double(extractdata(lc2_val));
    c2 = sgd_update(c2, g2, lr_c);
    lc = (lc1 + lc2)/2;

    s_dl = dlarray(s, 'CB');
    [g_act, la_val] = dlfeval(@actor_loss_fn, actor, c1, c2, s_dl, alpha, act_low, act_high);
    la = double(extractdata(la_val));
    actor = sgd_update(actor, g_act, lr_a);

    mean_lp = double(mean(next_lp));
    grad_log_alpha = -exp(log_alpha_val) * (mean_lp + double(target_ent));
    log_alpha_val = log_alpha_val - lr_al * grad_log_alpha;

    c1t = soft_update(c1t, c1, tau);
    c2t = soft_update(c2t, c2, tau);
end

function [grads, loss] = critic_loss_fn(net, sa, target)
    qpred = forward(net, sa);
    loss = mean((qpred - target).^2, 'all');
    grads = dlgradient(loss, net.Learnables);
end

function [grads, loss] = actor_loss_fn(actor, c1, c2, s_dl, alpha, act_low, act_high)
    out = forward(actor, s_dl);
    half = size(out,1)/2;
    mu = out(1:half,:);
    lst = max(min(out(half+1:end,:), 2), -20);
    std_ = exp(lst);
    noise = dlarray(randn(size(extractdata(mu)), 'single'), 'CB');
    z = mu + std_ .* noise;
    sq = tanh(z);

    al = dlarray(act_low, 'CB');
    ah = dlarray(act_high, 'CB');
    action = al + (ah - al) .* (sq + 1) / 2;

    log_pg = -0.5 * sum(((z - mu) ./ std_).^2, 1) ...
             - sum(lst, 1) - size(mu,1)*0.5*log(2*pi);
    log_p = log_pg - sum(log(max(1 - sq.^2, 1e-6)), 1);

    sa_in = [s_dl; action];
    q1 = predict(c1, sa_in);
    q2 = predict(c2, sa_in);
    q_min = min(q1, q2);

    loss = mean(alpha .* log_p - q_min, 'all');
    grads = dlgradient(loss, actor.Learnables);
end

function net = sgd_update(net, grads, lr)
    for i = 1:height(net.Learnables)
        g = grads.Value{i};
        if ~isempty(g)
            net.Learnables.Value{i} = net.Learnables.Value{i} - lr * g;
        end
    end
end

function net_t = soft_update(net_t, net, tau)
    for i = 1:height(net_t.Learnables)
        w_t = net_t.Learnables.Value{i};
        w = net.Learnables.Value{i};
        net_t.Learnables.Value{i} = (1-tau)*w_t + tau*w;
    end
end

% =========================================================
%  奖励函数
% =========================================================
function [reward, done, success] = compute_lower_reward(q_old, q_new, tip_old, tip_new, ...
        target, isCol, minClr, step, maxStep, opts)

    dPrev = norm(target - tip_old);
    dNow  = norm(target - tip_new);
    reward = opts.k_reach * (dPrev - dNow);

    if dNow < 100
        reward = reward + opts.k_near * (100 - dNow) / 100;
    end
    reward = reward + opts.k_clear * min(max(minClr,0),200)/200;
    reward = reward - opts.k_smooth * sum((q_new - q_old).^2);

    done = false;
    success = false;

    if isCol
        reward = reward - opts.R_collision;
        done = true;
        return;
    end
    if dNow < opts.goal_radius
        reward = reward + opts.R_goal;
        done = true;
        success = true;
        return;
    end
    if q_new(1) < opts.zProbe_min || q_new(1) > opts.zProbe_max
        reward = reward - opts.R_limit;
        done = true;
        return;
    end
    if step >= maxStep
        done = true;
    end
end

% =========================================================
%  通用观测构建（27 维）
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

function p = get_phase(ep, C)
    if ep < C.phase1
        p = 0;
    elseif ep < C.phase2
        p = 1;
    else
        p = 2;
    end
end

function opts = default_opts(opts)
    if ~isfield(opts,'num_episodes'),      opts.num_episodes = 5000; end
    if ~isfield(opts,'max_steps'),         opts.max_steps = 150; end
    if ~isfield(opts,'lr_actor'),          opts.lr_actor = 3e-4; end
    if ~isfield(opts,'lr_critic'),         opts.lr_critic = 3e-4; end
    if ~isfield(opts,'lr_alpha'),          opts.lr_alpha = 3e-4; end
    if ~isfield(opts,'gamma'),             opts.gamma = 0.99; end
    if ~isfield(opts,'tau'),               opts.tau = 5e-3; end
    if ~isfield(opts,'batch'),             opts.batch = 256; end
    if ~isfield(opts,'buf_cap'),           opts.buf_cap = 1e6; end
    if ~isfield(opts,'hidden'),            opts.hidden = [256 256]; end
    if ~isfield(opts,'safety_margin'),     opts.safety_margin = 10; end
    if ~isfield(opts,'k_reach'),           opts.k_reach = 5.0; end
    if ~isfield(opts,'k_near'),            opts.k_near = 30.0; end
    if ~isfield(opts,'k_clear'),           opts.k_clear = 2.0; end
    if ~isfield(opts,'k_smooth'),          opts.k_smooth = 0.005; end
    if ~isfield(opts,'R_goal'),            opts.R_goal = 200.0; end
    if ~isfield(opts,'R_collision'),       opts.R_collision = 150.0; end
    if ~isfield(opts,'R_limit'),           opts.R_limit = 30.0; end
    if ~isfield(opts,'goal_radius'),       opts.goal_radius = 35.0; end
    if ~isfield(opts,'curriculum')
        opts.curriculum.phase1 = 1000;
        opts.curriculum.phase2 = 3000;
    end
    if ~isfield(opts,'act_low')
        opts.act_low = [-8; -0.35; -0.25; -0.35; -0.25; -0.35];
    end
    if ~isfield(opts,'act_high')
        opts.act_high = [8; 0.35; 0.25; 0.35; 0.25; 0.35];
    end
    if ~isfield(opts,'local_radius_phase0'), opts.local_radius_phase0 = 300; end
    if ~isfield(opts,'local_radius_phase1'), opts.local_radius_phase1 = 600; end
    if ~isfield(opts,'sample_max_tries'),    opts.sample_max_tries = 500; end
    if ~isfield(opts,'local_sample_tries'),  opts.local_sample_tries = 150; end
    if ~isfield(opts,'obs_pos_scale'),       opts.obs_pos_scale = 1000; end
    if ~isfield(opts,'zprobe_norm_scale'),   opts.zprobe_norm_scale = 3000; end
    if ~isfield(opts,'radius_norm_scale'),   opts.radius_norm_scale = 500; end
end

function plot_lower_curves(reward_hist, success_hist, la_hist, lc_hist)
    figure('Name','通用下层 SAC 训练曲线','Color','w','Position',[100 100 1200 400]);
    subplot(1,3,1);
    plot(reward_hist,'Color',[0.7 0.7 0.9]); hold on;
    plot(movmean(reward_hist,100),'b-','LineWidth',2);
    xlabel('Episode'); ylabel('回报'); title('Episode 回报'); grid on;
    legend('原始','滑动均值(100)');

    subplot(1,3,2);
    plot(movmean(success_hist,100),'g-','LineWidth',2);
    xlabel('Episode'); ylabel('成功率'); title('成功率 (滑动100)');
    ylim([0 1]); grid on;

    subplot(1,3,3);
    if ~isempty(la_hist)
        semilogy(abs(la_hist),'r-','LineWidth',1,'DisplayName','Actor'); hold on;
    end
    if ~isempty(lc_hist)
        semilogy(abs(lc_hist),'b-','LineWidth',1,'DisplayName','Critic');
    end
    xlabel('更新步'); ylabel('Loss'); title('SAC Loss');
    legend; grid on;
    sgtitle('通用下层 SAC 训练曲线');
end
