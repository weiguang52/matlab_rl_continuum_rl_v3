function results = train_lower_sac(scene, layerId, robot, opts)
%TRAIN_LOWER_SAC  下层连续体机械臂 SAC 训练脚本
%
%  特性
%  ----
%  · 课程学习：
%      阶段0 (ep < C.phase1)  : 目标集中在入口孔附近小区域（半径 <= 300）
%      阶段1 (ep < C.phase2)  : 目标扩展至层内中等范围（半径 <= 600）
%      阶段2 (ep >= C.phase2) : 全层随机目标
%  · 安全层碰撞：柱半径 + 机器臂外径/2 + margin -> 直接终止 + 大惩罚
%  · 观测 (20 维)：
%      目标相对末端的 3D 误差(3)、当前 q(6)、到最近障碍归一化距离(1)、
%      最近障碍相对方向(2)、3D 欧氏距离(1)、末端到目标层 z 距离(1)、
%      上一步动作(6) -> 共 20 维
%  · 动作 (6 维)：Delta[zProbe, alpha1, theta1, gamma, theta2, alpha3]
%  · SAC：两个 Critic + Gaussian Actor + 自动熵调节
%
%  Bug fix（相比上一版本）
%  -----------------------
%  原版在 dlfeval 的闭包里同时捕获了 log_alpha（dlarray 标量）和
%  dlnetwork 对象，触发 MATLAB 自动微分引擎的 recordNary 错误。
%  修复策略：
%    1. log_alpha 改为普通 double，用解析梯度手动更新，完全不进 dlfeval。
%    2. 每个 dlfeval 只追踪一个网络的 Learnables；
%       其他网络的输出用 predict()（不追踪梯度）预先算好再传入。
%    3. 去掉 assignin 写法，sac_update 直接返回所有更新后的网络。

    if nargin < 4, opts = struct(); end
    opts = default_opts(opts);

    C      = opts.curriculum;
    N_EP   = opts.num_episodes;
    MAX_ST = opts.max_steps;
    BATCH  = opts.batch;
    LR_ACT = opts.lr_actor;
    LR_CRI = opts.lr_critic;
    LR_AL  = opts.lr_alpha;
    GAMMA  = opts.gamma;
    TAU    = opts.tau;
    BUF_CAP = opts.buf_cap;
    TARGET_ENTROPY = -6;   % = -|A|，固定值

    OBS_DIM = 20;
    ACT_DIM = 6;

    fprintf('[Lower SAC] 初始化 网络 obs=%d act=%d\n', OBS_DIM, ACT_DIM);

    [actor, c1_online, c2_online, c1_target, c2_target] = ...
        build_sac_networks(OBS_DIM, ACT_DIM, opts.hidden);

    % log_alpha 用普通 double，避免与 dlnetwork 混入 dlfeval
    log_alpha_val = 0.0;

    act_low  = single([-8; -pi;  -pi; -pi;  -pi; -pi]);
    act_high = single([ 8;  pi;   pi;  pi;   pi;  pi]);

    % zRange = [entryStartZ, targetZ]；zProbe 是从 entryStartZ 开始的相对下探长度
    entryStartZ = scene.layers(layerId).zRange(1);
    targetZ = scene.layers(layerId).zRange(2);
    gap = targetZ - entryStartZ;
    chain_len_bent = robot.baseLen + 0.27 * (robot.seg1Len + robot.link1Len ...
                   + robot.seg2Len + robot.link2Len);
    zProbe_min = 0;
    zProbe_max = max(0, gap - chain_len_bent + 30);
    fprintf('[Lower SAC] 层 %d entryStartZ=%.2f targetZ=%.2f, zProbe ∈ [%.0f, %.0f] mm\n', ...
        layerId, entryStartZ, targetZ, zProbe_min, zProbe_max);

    q_low  = [zProbe_min; -pi; -pi; -pi; -pi; -pi];
    q_high = [zProbe_max;  pi;  pi;  pi;  pi;  pi];

    % 把 zProbe 范围塞进 opts，供 compute_lower_reward 里判断越限用
    opts.zProbe_min = zProbe_min;
    opts.zProbe_max = zProbe_max;

    buf = SACReplayBuffer(BUF_CAP, OBS_DIM, ACT_DIM);

    entry = pick_default_entry(scene, layerId);
    fprintf('[Lower SAC] 使用入口: %s @ [%.1f, %.1f]\n', ...
        string(entry.entryHoleLabel), entry.entryXY(1), entry.entryXY(2));

    reward_hist      = zeros(N_EP, 1);
    success_hist     = zeros(N_EP, 1);
    loss_actor_hist  = [];
    loss_critic_hist = [];

    for ep = 1:N_EP
        target = curriculum_sample_lower(scene, layerId, entry, ep, C);

        % 从当前入口所在板面开始下探，初始相对下探量为 0
        q     = [0; 0; 0; 0; 0; 0];
        model = continuum_forward_model(q, robot, entry.entryXY, entryStartZ);
        prev_action = zeros(ACT_DIM, 1);
        obs   = build_obs_lower(q, model.tipPos, target, ...
                                scene.layers(layerId).supports, prev_action);

        ep_reward  = 0;
        ep_success = false;

        for step = 1:MAX_ST
            action = actor_sample_action(actor, obs, act_low, act_high);

            q_new     = clamp_q(q + double(action), q_low, q_high);
            model_new = continuum_forward_model(q_new, robot, entry.entryXY, entryStartZ);

            [isCol, minClr] = collision_check_centerline( ...
                model_new.centerline, scene.layers(layerId).supports, ...
                robot.armRadius, opts.safety_margin);

            [reward, done, success] = compute_lower_reward( ...
                q, q_new, model.tipPos, model_new.tipPos, target, ...
                isCol, minClr, step, MAX_ST, opts);

            obs_next = build_obs_lower(q_new, model_new.tipPos, target, ...
                                       scene.layers(layerId).supports, action);

            buf.push(obs, action, reward, obs_next, double(done));

            ep_reward   = ep_reward + reward;
            q           = q_new;
            model       = model_new;
            prev_action = action;
            obs         = obs_next;

            if buf.size >= BATCH
                [actor, c1_online, c2_online, c1_target, c2_target, ...
                 log_alpha_val, la, lc] = sac_update( ...
                    actor, c1_online, c2_online, c1_target, c2_target, ...
                    log_alpha_val, buf, BATCH, LR_ACT, LR_CRI, LR_AL, ...
                    GAMMA, TAU, TARGET_ENTROPY, act_low, act_high);
                loss_actor_hist(end+1)  = la; %#ok<AGROW>
                loss_critic_hist(end+1) = lc; %#ok<AGROW>
            end

            if success, ep_success = true; end
            if done,    break;             end
        end

        reward_hist(ep)  = ep_reward;
        success_hist(ep) = ep_success;

        if mod(ep, 100) == 0
            phase = get_phase(ep, C);
            sr = mean(success_hist(max(1,ep-99):ep));
            fprintf('[Lower SAC] ep=%d phase=%d meanR=%.1f succRate=%.2f\n', ...
                ep, phase, mean(reward_hist(max(1,ep-99):ep)), sr);
        end
    end

    results.actor        = actor;
    results.c1           = c1_online;
    results.c2           = c2_online;
    results.reward_hist  = reward_hist;
    results.success_hist = success_hist;
    results.loss_actor   = loss_actor_hist(:);
    results.loss_critic  = loss_critic_hist(:);
    results.entry        = entry;
    results.obs_dim      = OBS_DIM;
    results.act_low      = act_low;
    results.act_high     = act_high;

    fprintf('[Lower SAC] 训练完成。最终成功率 (last 200 ep) = %.2f\n', ...
        mean(success_hist(max(1,end-199):end)));

    plot_lower_curves(reward_hist, success_hist, loss_actor_hist, loss_critic_hist);
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
%  动作采样（推断阶段，不追踪梯度）
% =========================================================
function action = actor_sample_action(actor, obs, act_low, act_high)
    x   = dlarray(single(obs(:)), 'CB');
    out = double(extractdata(predict(actor, x)));
    half    = numel(out) / 2;
    mu      = out(1:half);
    log_std = max(min(out(half+1:end), 2), -20);
    std_val = exp(log_std);
    z       = mu + std_val .* randn(size(mu));
    sq      = tanh(z);
    action  = double(act_low) + (double(act_high) - double(act_low)) .* (sq+1)/2;
end

% =========================================================
%  SAC 更新
%  核心原则：每个 dlfeval 只追踪一个目标网络的 Learnables，
%            其余网络输出用 predict()（不进计算图）预先取出。
%            log_alpha 用 double + 解析梯度，完全不进 dlfeval。
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

    % --- 1. 下一状态动作 & log_prob（用 predict，不追踪梯度）---
    next_act = zeros(numel(act_low), batch, 'single');
    next_lp  = zeros(1, batch, 'single');
    for i = 1:batch
        x_i   = dlarray(s2(:,i), 'CB');
        out_i = double(extractdata(predict(actor, x_i)));
        half  = numel(out_i)/2;
        mu_i  = out_i(1:half);
        lst_i = max(min(out_i(half+1:end), 2), -20);
        std_i = exp(lst_i);
        z_i   = mu_i + std_i .* randn(size(mu_i));
        sq_i  = tanh(z_i);
        na_i  = double(act_low) + (double(act_high)-double(act_low)).*(sq_i+1)/2;
        lp_i  = sum(-0.5*((z_i-mu_i)./std_i).^2) - sum(lst_i) ...
                - numel(mu_i)*0.5*log(2*pi) ...
                - sum(log(max(1 - sq_i.^2, 1e-6)));
        next_act(:,i) = single(na_i);
        next_lp(i)    = single(lp_i);
    end

    % --- 2. TD target（用 target critic predict，不追踪梯度）---
    sa2_dl  = dlarray([s2; next_act], 'CB');
    q1_tgt  = extractdata(predict(c1t, sa2_dl));
    q2_tgt  = extractdata(predict(c2t, sa2_dl));
    q_min_t = min(q1_tgt, q2_tgt);
    td_target = r + gamma .* (1-done) .* (q_min_t - alpha .* next_lp);
    td_dl   = dlarray(td_target, 'CB');
    sa_dl   = dlarray([s; a], 'CB');

    % --- 3. Critic 1 更新（只追踪 c1.Learnables）---
    [g1, lc1_val] = dlfeval(@critic_loss_fn, c1, sa_dl, td_dl);
    lc1 = double(extractdata(lc1_val));
    c1  = sgd_update(c1, g1, lr_c);

    % --- 4. Critic 2 更新（只追踪 c2.Learnables）---
    [g2, lc2_val] = dlfeval(@critic_loss_fn, c2, sa_dl, td_dl);
    lc2 = double(extractdata(lc2_val));
    c2  = sgd_update(c2, g2, lr_c);
    lc  = (lc1 + lc2) / 2;

    % --- 5. Actor 更新（只追踪 actor.Learnables；c1/c2 在内部用 predict）---
    s_dl = dlarray(s, 'CB');
    [g_act, la_val] = dlfeval(@actor_loss_fn, actor, c1, c2, s_dl, ...
                               alpha, act_low, act_high);
    la    = double(extractdata(la_val));
    actor = sgd_update(actor, g_act, lr_a);

    % --- 6. log_alpha 解析梯度更新（不进 dlfeval）---
    %   loss_alpha = -exp(la) * (mean_lp + target_ent)
    %   d/d(la) = -exp(la) * (mean_lp + target_ent)
    mean_lp       = double(mean(next_lp));
    grad_log_alpha = -exp(log_alpha_val) * (mean_lp + double(target_ent));
    log_alpha_val  = log_alpha_val - lr_al * grad_log_alpha;

    % --- 7. Soft target 更新 ---
    c1t = soft_update(c1t, c1, tau);
    c2t = soft_update(c2t, c2, tau);
end

% ─── 独立损失函数 ──────────────────────────────────────────────────

function [grads, loss] = critic_loss_fn(net, sa, target)
    qpred = forward(net, sa);
    loss  = mean((qpred - target).^2, 'all');
    grads = dlgradient(loss, net.Learnables);
end

function [grads, loss] = actor_loss_fn(actor, c1, c2, s_dl, alpha, act_low, act_high)
    % forward(actor,...) 追踪 actor 梯度；
    % predict(c1/c2,...) 不追踪，c1/c2 作为"常数估值器"
    out  = forward(actor, s_dl);
    half = size(out, 1) / 2;
    mu   = out(1:half, :);
    lst  = max(min(out(half+1:end, :), 2), -20);
    std_ = exp(lst);
    noise = dlarray(randn(size(extractdata(mu)), 'single'), 'CB');
    z    = mu + std_ .* noise;
    sq   = tanh(z);

    al     = dlarray(act_low,  'CB');
    ah     = dlarray(act_high, 'CB');
    action = al + (ah - al) .* (sq + 1) / 2;

    log_pg = -0.5 * sum(((z - mu) ./ std_).^2, 1) ...
             - sum(lst, 1) ...
             - size(mu,1) * 0.5 * log(2*pi);
    log_p  = log_pg - sum(log(max(1 - sq.^2, 1e-6)), 1);

    sa_in = [s_dl; action];
    q1    = predict(c1, sa_in);   % 不追踪梯度
    q2    = predict(c2, sa_in);
    q_min = min(q1, q2);

    loss  = mean(alpha .* log_p - q_min, 'all');
    grads = dlgradient(loss, actor.Learnables);
end

% =========================================================
%  SGD 参数更新
% =========================================================
function net = sgd_update(net, grads, lr)
    for i = 1:height(net.Learnables)
        g = grads.Value{i};
        if ~isempty(g)
            net.Learnables.Value{i} = net.Learnables.Value{i} - lr * g;
        end
    end
end

% =========================================================
%  Soft target 更新
% =========================================================
function net_t = soft_update(net_t, net, tau)
    for i = 1:height(net_t.Learnables)
        w_t = net_t.Learnables.Value{i};
        w   = net.Learnables.Value{i};
        net_t.Learnables.Value{i} = (1-tau)*w_t + tau*w;
    end
end

% =========================================================
%  奖励函数
% =========================================================
function [reward, done, success] = compute_lower_reward( ...
        q_old, q_new, tip_old, tip_new, target, isCol, minClr, step, maxStep, opts)

    dPrev  = norm(target - tip_old);
    dNow   = norm(target - tip_new);
    reward = opts.k_reach * (dPrev - dNow);

    if dNow < 100
        reward = reward + opts.k_near * (100 - dNow) / 100;
    end
    reward  = reward + opts.k_clear * min(max(minClr,0),200)/200;
    reward  = reward - opts.k_smooth * sum((q_new - q_old).^2);

    done    = false;
    success = false;

    if isCol
        reward = reward - opts.R_collision;
        done   = true;  return;
    end
    if dNow < opts.goal_radius
        reward  = reward + opts.R_goal;
        done    = true;
        success = true;  return;
    end
    if q_new(1) < opts.zProbe_min || q_new(1) > opts.zProbe_max
        reward = reward - opts.R_limit;
        done   = true;  return;
    end
    if step >= maxStep
        done = true;
    end
end

% =========================================================
%  观测构建（20 维）
% =========================================================
function obs = build_obs_lower(q, tipPos, target, supports, prev_action)
    err  = target(:) - tipPos(:);
    dmin = inf;  rel2 = [0;0];
    for i = 1:numel(supports)
        ctr = supports(i).xy;
        d   = norm(tipPos(1:2) - ctr) - supports(i).radius;
        if d < dmin
            dmin = d;
            v    = ctr(:) - tipPos(1:2)';
            nv   = max(norm(v), 1);
            rel2 = v / nv;
        end
    end
    dmin_norm = min(max(dmin,0),500) / 500;
    dist3d    = norm(err);
    z_err     = abs(target(3) - tipPos(3));
    obs = [err(:); q(:); dmin_norm; rel2(:); dist3d; z_err; prev_action(:)];
    obs = obs(1:20);
end

% =========================================================
%  课程采样
% =========================================================
function target = curriculum_sample_lower(scene, layerId, entry, ep, C)
    phase = get_phase(ep, C);
    zTarget = scene.layers(layerId).zRange(2);

    max_tries = 200;

    for tries = 1:max_tries
        if phase == 0
            r = 300*sqrt(rand());
            ang = 2*pi*rand();
            x = entry.entryXY(1) + r*cos(ang);
            y = entry.entryXY(2) + r*sin(ang);

        elseif phase == 1
            r = 600*sqrt(rand());
            ang = 2*pi*rand();
            x = entry.entryXY(1) + r*cos(ang);
            y = entry.entryXY(2) + r*sin(ang);

        else
            [x, y] = sample_target_xy_by_layer(scene, layerId);
        end

        % 下层训练目标也限制在对应层真实异物区域内
        if ~is_xy_in_layer_target_region(scene, layerId, [x, y])
            continue;
        end

        % 避免采到支撑柱内部
        if point_inside_any_support(scene.layers(layerId).supports, [x, y], scene.armRadius)
            continue;
        end

        target = [x, y, zTarget];
        return;
    end

    % 兜底：如果局部 entry 周围采样一直失败，则直接从该层合法异物区域内采样
    for tries = 1:max_tries
        [x, y] = sample_target_xy_by_layer(scene, layerId);

        if point_inside_any_support(scene.layers(layerId).supports, [x, y], scene.armRadius)
            continue;
        end

        target = [x, y, zTarget];
        return;
    end

    % 极端兜底
    target = [entry.entryXY(1), entry.entryXY(2), zTarget];
end

function entry = pick_default_entry(scene, layerId)
    entries = scene.layers(layerId).entries;
    for i = 1:numel(entries)
        if strcmp(string(entries(i).anchorPoint), '8H')
            entry = entries(i);  return;
        end
    end
    entry = entries(1);
end

function q = clamp_q(q, q_low, q_high)
    q    = max(min(q, q_high), q_low);
    q(2) = wrapToPi(q(2));
    q(4) = wrapToPi(q(4));
    q(6) = wrapToPi(q(6));
end

function p = get_phase(ep, C)
    if ep < C.phase1,  p = 0;
    elseif ep < C.phase2, p = 1;
    else, p = 2;
    end
end

function opts = default_opts(opts)
    if ~isfield(opts,'num_episodes'),   opts.num_episodes   = 5000;  end
    if ~isfield(opts,'max_steps'),      opts.max_steps      = 150;   end
    if ~isfield(opts,'lr_actor'),       opts.lr_actor       = 3e-4;  end
    if ~isfield(opts,'lr_critic'),      opts.lr_critic      = 3e-4;  end
    if ~isfield(opts,'lr_alpha'),       opts.lr_alpha       = 3e-4;  end
    if ~isfield(opts,'gamma'),          opts.gamma          = 0.99;  end
    if ~isfield(opts,'tau'),            opts.tau            = 5e-3;  end
    if ~isfield(opts,'batch'),          opts.batch          = 256;   end
    if ~isfield(opts,'buf_cap'),        opts.buf_cap        = 1e6;   end
    if ~isfield(opts,'hidden'),         opts.hidden         = [256 256]; end
    if ~isfield(opts,'safety_margin'),  opts.safety_margin  = 10;    end
    if ~isfield(opts,'k_reach'),        opts.k_reach        = 5.0;   end
    if ~isfield(opts,'k_near'),         opts.k_near         = 30.0;  end
    if ~isfield(opts,'k_clear'),        opts.k_clear        = 2.0;   end
    if ~isfield(opts,'k_smooth'),       opts.k_smooth       = 0.005; end
    if ~isfield(opts,'R_goal'),         opts.R_goal         = 200.0; end
    if ~isfield(opts,'R_collision'),    opts.R_collision    = 150.0; end
    if ~isfield(opts,'R_limit'),        opts.R_limit        = 30.0;  end
    if ~isfield(opts,'goal_radius'),    opts.goal_radius    = 35.0;  end
    if ~isfield(opts,'curriculum')
        opts.curriculum.phase1 = 1000;
        opts.curriculum.phase2 = 3000;
    end
end

% =========================================================
%  训练曲线
% =========================================================
function plot_lower_curves(reward_hist, success_hist, la_hist, lc_hist)
    figure('Name','下层 SAC 训练曲线','Color','w','Position',[100 100 1200 400]);
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
    sgtitle('下层 SAC 训练曲线');
end
