function results = train_upper_dqn(scene, robot, reach, opts)
%TRAIN_UPPER_DQN  上层离散 Double DQN：多目标序贯覆盖 + 定位点/下探孔最小化
%
%  任务重定义（v2）
%  ----------------
%  每个 episode 给定一批随机生成的目标点（同一层），策略需要**序贯地**选择
%  一系列 (定位点, 下探孔) 入口，直到所有目标都被"某个已选入口的可达域"覆盖。
%
%  优化目标的优先级（通过奖励权重编码）
%     优先级1（主）: 使用的**定位点数量**最少
%     优先级2（次）: 使用的**下探孔数量**最少
%     优先级3（约束）: 所有目标都被覆盖
%
%  MDP 定义
%  --------
%   状态 s (变长目标集 → 固定维编码):
%     [layer_onehot(4);
%      unmet_count_norm(1);          % 未覆盖目标数/max
%      unmet_centroid_xy_norm(2);
%      unmet_bbox_size_norm(2);
%      used_anchor_mask(maxAnchors); % 已使用的定位点（one-hot 并集）
%      used_hole_count_norm(1)]      % 已用下探孔数
%
%   动作 a: 当前层所有入口（RobotArm 类型）的离散索引，a ∈ {1..nEntry_L}
%
%   转移:
%     选定 entry = entries(a) 后:
%       用 reach 查表，将 entry 可达域内的所有未覆盖目标标为已覆盖
%       将 entry.anchorPoint 加入已用定位点集合
%       将 entry 加入已用下探孔集合
%       若所有目标覆盖 → terminal
%       若超出 max_steps → terminal
%
%   奖励（学"最小"问题，采用负成本设计，最小化即最优）:
%     r_step = R_NEW_ANCHOR  if 本步引入新定位点  else 0
%            + R_NEW_HOLE                                     % 每个新下探孔
%            + R_NO_PROGRESS if 本步未覆盖任何新目标 else 0
%     r_term = -R_COVER_BONUS  if 全部覆盖（负 = 奖励）
%            + R_FAIL if 超时仍未全覆盖
%
%  Double DQN，最小化 Q：选动作用 argmin q_online，估值用 q_target[argmin_online]。
%
%  输入
%  ----
%   scene : build_scene_model 输出
%   robot : 机器臂参数（FK 需要，已在 reach 中使用，但保留以便自洽调用）
%   reach : precompute_reachability 返回的查找表（必需！）
%   opts  : 超参数结构体（见 default_opts）

    if nargin < 4, opts = struct(); end
    opts = default_opts(opts);

    % ---- 超参数 ----
    % C         课程学习参数
    % LR        学习率
    % GAMMA     折扣因子
    % EPS0      初始探索率
    % EPS_MIN   最小探索率
    % BATCH     batch size
    % BUF_CAP   经验回放池容量
    % N_EP      episode 总数
    % MAX_STEPS 每个 episode 最多选多少个入口
    C        = opts.curriculum;
    LR       = opts.lr;
    GAMMA    = opts.gamma;
    EPS0     = opts.eps0;
    EPS_MIN  = opts.eps_min;
    BATCH    = opts.batch;
    BUF_CAP  = opts.buf_cap;
    UPDATE_TARGET_FREQ = opts.update_target_freq;
    N_EP     = opts.num_episodes;
    MAX_STEPS = opts.max_steps;

    nLayers    = numel(scene.layers);
    maxActions = max(arrayfun(@(s) numel(s.entries), scene.layers));

    % ---- 定位点编码表（所有层并集）----
    anchorTable = build_anchor_table(scene);
    nAnchors    = numel(anchorTable);

    % ---- 状态维度 ----
    % layer_onehot + 未覆盖目标数比例 + 未覆盖目标 xy质心 + 未覆盖目标xy包围盒尺寸 + 已使用定位点 one-hot
    % + 已使用下探孔数量归一化
    stateDim = 4 + 1 + 2 + 2 + nAnchors + 1;    % 10 + nAnchors

    fprintf('[Upper DQN v2] stateDim=%d maxActions=%d nAnchors=%d\n', ...
        stateDim, maxActions, nAnchors);

    useGPU = resolve_use_gpu(opts.useGPU, 'Upper DQN');

    net_online = build_dqn(stateDim, maxActions, opts.hidden);
    net_target = net_online;
    if useGPU
        net_online = move_network_to_gpu(net_online);
        net_target = move_network_to_gpu(net_target);
    end

    buf = ReplayBuffer(BUF_CAP, stateDim);
    % 训练历史记录
    reward_hist       = zeros(N_EP, 1);     % 每个 ep 的累积奖励
    loss_hist         = [];                 % DQN 更新损失
    anchors_hist      = zeros(N_EP, 1);     % 每 ep 用了多少定位点
    holes_hist        = zeros(N_EP, 1);     % 每个 ep 用了多少下探孔
    coverage_hist     = zeros(N_EP, 1);     % 覆盖率
    best_reward       = -inf;               % 当前最好的 ep reward
    results_best      = net_online;         % reward 最好时的网络

    for ep = 1:N_EP
        % ---- 课程：采样本 episode 的层 & 目标集 ----
        [layerId, targets] = curriculum_sample_episode(scene, reach, ep, C, opts);
        nT = size(targets, 1);
        if nT == 0, continue; end

        % ---- episode 状态（可覆盖性判定基于 reach 表）----
        covered        = false(nT, 1);
        used_anchors   = false(nAnchors, 1);
        used_hole_set  = false(numel(scene.layers(layerId).entries), 1);

        obs = encode_state(scene, layerId, targets, covered, used_anchors, ...
                           used_hole_set, anchorTable, opts);
        ep_reward = 0;

        for step = 1:MAX_STEPS
            nA  = numel(scene.layers(layerId).entries);
            eps = max(EPS_MIN, EPS0 * (1 - ep/N_EP));

            if rand < eps
                a = randi(nA);
            else
                qvals = forward_net(net_online, obs, useGPU);
                qvals_valid = qvals(1:nA);
                % 用 mask 避免选已使用过的 hole（可选：不屏蔽，让策略自己学）
                if opts.mask_used_holes
                    qvals_valid(used_hole_set) = +inf;  % 最小化 → +inf 阻止
                    % 若本层所有入口都已经被屏蔽，说明当前 episode 无可选新孔，提前结束
                    if all(isinf(qvals_valid))
                        break;
                    end
                end
                [~, a] = min(qvals_valid);
            end

            entry = scene.layers(layerId).entries(a);
            anchorId = find_anchor_id(entry.anchorPoint, anchorTable);

            % --- 转移，判断这一步是否使用了新资源 ---
            is_new_anchor = ~used_anchors(anchorId);
            is_new_hole   = ~used_hole_set(a);

            newly_covered = false(nT, 1);
            for t = 1:nT
                if ~covered(t)
                    if reach.query(layerId, a, targets(t, 1:2))
                        newly_covered(t) = true;
                    end
                end
            end
            n_newly = sum(newly_covered);
            covered = covered | newly_covered;
            used_anchors(anchorId) = true;
            used_hole_set(a) = true;

            % --- 奖励 ---
            r = 0;
            if is_new_anchor, r = r + opts.R_NEW_ANCHOR; end     % 负
            if is_new_hole,   r = r + opts.R_NEW_HOLE;   end     % 负
            if n_newly == 0,  r = r + opts.R_NO_PROGRESS; end    % 负
            r = r + opts.R_PROGRESS_SCALE * n_newly;             % 正（小）

            done = all(covered);
            if done
                r = r + opts.R_COVER_BONUS;                      % 终局大奖
            elseif step >= MAX_STEPS
                r = r + opts.R_FAIL;
                done = true;
            end
            ep_reward = ep_reward + r;

            obs_next = encode_state(scene, layerId, targets, covered, used_anchors, ...
                                    used_hole_set, anchorTable, opts);

            % 放入经验池。NOTE: 存入缓冲区时，cost = -r 以配合 "argmin Q = argmax r"
            buf.push(obs, a, -r, obs_next, double(done));
            obs = obs_next;

            if buf.size >= BATCH
                % update_dqn 会返回更新后的 net_online；否则 MATLAB 函数内的局部更新不会回写到主循环
                % 额外传入 scene，用于在 Double DQN target 中屏蔽不同层的 padding 无效动作
                [net_online, loss] = update_dqn(net_online, net_target, buf, BATCH, LR, GAMMA, maxActions, scene, useGPU);
                loss_hist(end+1) = loss; %#ok<AGROW>
            end

            if done, break; end
        end

        if mod(ep, UPDATE_TARGET_FREQ) == 0
            net_target = net_online;
        end

        reward_hist(ep)   = ep_reward;
        anchors_hist(ep)  = sum(used_anchors);
        holes_hist(ep)    = sum(used_hole_set);
        coverage_hist(ep) = mean(covered);

        if ep_reward > best_reward
            best_reward = ep_reward;
            results_best = net_online;
        end

        if mod(ep, 50) == 0
            ph = get_phase(ep, C);
            fprintf(['[Upper DQN] ep=%d phase=%d eps=%.3f meanR=%.1f ' ...
                     '| near50 avgAnchors=%.2f avgHoles=%.2f cov=%.2f\n'], ...
                ep, ph, eps, mean(reward_hist(max(1,ep-49):ep)), ...
                mean(anchors_hist(max(1,ep-49):ep)), ...
                mean(holes_hist(max(1,ep-49):ep)), ...
                mean(coverage_hist(max(1,ep-49):ep)));
        end
    end

    % 保存/评估默认使用 CPU 网络，避免后续在无 GPU 环境加载 mat 时受限。
    if useGPU
        net_online = move_network_to_cpu(net_online);
        results_best = move_network_to_cpu(results_best);
    end

    results.net_online    = net_online;
    results.net_best      = results_best;
    results.reward_hist   = reward_hist;
    results.loss_hist     = loss_hist(:);
    results.anchors_hist  = anchors_hist;
    results.holes_hist    = holes_hist;
    results.coverage_hist = coverage_hist;
    results.maxActions    = maxActions;
    results.stateDim      = stateDim;
    results.anchorTable   = anchorTable;
    results.opts          = opts;

    fprintf('[Upper DQN] 训练完成。最后 100 ep: avgAnchors=%.2f avgHoles=%.2f cov=%.2f\n', ...
        mean(anchors_hist(max(1,end-99):end)), ...
        mean(holes_hist(max(1,end-99):end)), ...
        mean(coverage_hist(max(1,end-99):end)));
    plot_training_curves(reward_hist, anchors_hist, holes_hist, coverage_hist, loss_hist);
end

% =========================================================
%  网络构建
% =========================================================
function net = build_dqn(inDim, outDim, hidden)
    layers = [
        featureInputLayer(inDim, 'Normalization','none','Name','obs')
        fullyConnectedLayer(hidden(1), 'Name','fc1')
        reluLayer('Name','relu1')
        fullyConnectedLayer(hidden(2), 'Name','fc2')
        reluLayer('Name','relu2')
        fullyConnectedLayer(outDim, 'Name','qout')
    ];
    net = dlnetwork(layerGraph(layers));
end

function qvals = forward_net(net, obs, useGPU)
    if nargin < 3, useGPU = false; end
    xData = single(obs(:));
    if useGPU, xData = gpuArray(xData); end
    x = dlarray(xData, 'CB');
    y = predict(net, x);
    qvals = double(to_cpu(extractdata(y)));
end

function qvals = forward_net_batch(net, obsBatch, useGPU)
    if nargin < 3, useGPU = false; end
    xData = single(obsBatch);
    if useGPU, xData = gpuArray(xData); end
    x = dlarray(xData, 'CB');
    y = predict(net, x);
    qvals = single(to_cpu(extractdata(y)));
end

% =========================================================
%  Double DQN 更新（代价式：最小化 cost）
% =========================================================
function [net_online, loss] = update_dqn(net_online, net_target, buf, batch, lr, gamma, maxA, scene, useGPU)
    [s, a, r, s2, done] = buf.sample(batch);

    % 实际 batch 大小以采样结果为准，避免最后/异常情况下 batch 维度假设不一致
    actualBatch = size(s, 2);

    q2_online = forward_net_batch(net_online, s2, useGPU);
    q2_target = forward_net_batch(net_target, s2, useGPU);
    for i = 1:actualBatch
        % 不同层入口数量不同，但网络输出统一是 maxActions。
        % 因此下一状态 s2 中，超过该层真实入口数量的动作是 padding 无效动作，
        % Double DQN 选 next action 时必须屏蔽，否则 TD target 可能选到不存在的入口。
        [~, layerId2] = max(s2(1:numel(scene.layers), i));
        nA2 = numel(scene.layers(layerId2).entries);
        if nA2 < maxA
            q2_online(nA2+1:end, i) = +inf;  % 最小化 cost → +inf 表示不可选
            q2_target(nA2+1:end, i) = +inf;
        end
    end
    [~, best_a] = min(q2_online, [], 1);          % 最小化 cost
    td_targets  = single(r(:)') + single(gamma) * (1 - single(done(:)')) .* ...
                  q2_target(sub2ind(size(q2_target), best_a, 1:actualBatch));

    XData = single(s);
    tdData = td_targets;
    if useGPU
        XData = gpuArray(XData);
        tdData = gpuArray(tdData);
    end
    X  = dlarray(XData, 'CB');
    td = dlarray(tdData, 'CB');
    a_idx = a(:)';

    [grads, loss_val] = dlfeval(@(net) huber_loss_fn(net, X, a_idx, td, maxA), net_online);
    loss = double(to_cpu(extractdata(loss_val)));

    % 注意：net_online 必须作为输出参数返回给主循环，否则这里的更新只停留在函数局部变量中
    net_online = dlupdate(@(p, g) p - lr * g, net_online, grads);
end

function [grads, loss] = huber_loss_fn(net, X, a_idx, targets, maxA)
    Q = forward(net, X);
    batch = size(X, 2);
    lin_idx = sub2ind([maxA, batch], a_idx, 1:batch);
    q_pred = Q(lin_idx);
    delta = q_pred - targets;
    loss = mean(huber(delta, 1.0));
    grads = dlgradient(loss, net.Learnables);
end

function y = huber(x, delta)
    absx = abs(x);
    y = (absx <= delta) .* (0.5 * x.^2) + ...
        (absx >  delta) .* (delta * (absx - 0.5*delta));
end

% =========================================================
%  状态编码
% =========================================================
function obs = encode_state(scene, layerId, targets, covered, used_anchors, ...
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
        cen = [0, 0];
        bb  = [0, 0];
    end
    xR = scene.gridBoundsX; yR = scene.gridBoundsY;
    cen_n = [(cen(1) - mean(xR))/(diff(xR)/2+1e-6); ...
             (cen(2) - mean(yR))/(diff(yR)/2+1e-6)];
    bb_n  = [bb(1)/(diff(xR)+1e-6); bb(2)/(diff(yR)+1e-6)];

    used_hole_norm = sum(used_hole_set) / max(numel(used_hole_set), 1);

    obs = [layer_oh; unmet_count_norm; cen_n; bb_n; used_anchors(:); used_hole_norm];
end

% =========================================================
%  定位点编码表（跨层并集）
% =========================================================
function tbl = build_anchor_table(scene)
    s = strings(0, 1);
    for L = 1:numel(scene.layers)
        for i = 1:numel(scene.layers(L).entries)
            s(end+1, 1) = string(scene.layers(L).entries(i).anchorPoint); %#ok<AGROW>
        end
    end
    tbl = unique(s);
end

function idx = find_anchor_id(anchorStr, anchorTable)
    idx = find(anchorTable == string(anchorStr), 1);
    if isempty(idx), idx = 1; end
end

% =========================================================
%  Episode 目标集采样（课程学习）
% =========================================================
function [layerId, targets] = curriculum_sample_episode(scene, reach, ep, C, opts)
    phase = get_phase(ep, C);

    % 层选择
    if phase == 0
        layerId = 2;    % LGP
    elseif phase == 1
        layerId = randi(numel(scene.layers));
    else
        layerId = randi(numel(scene.layers));
    end

    zTarget = scene.layers(layerId).zRange(2);   % +Z 向下后，目标取该层下表面/较深边界

    % 每个 ep 的目标数量
    if phase == 0
        nT = randi([opts.n_targets_min, max(opts.n_targets_min, round(opts.n_targets_max/2))]);
    elseif phase == 1
        nT = randi([opts.n_targets_min, opts.n_targets_max]);
    else
        nT = randi([opts.n_targets_min, opts.n_targets_max]);
    end

    % 采样 XY：
    %   不再从正方形全域随机采样；
    %   改为按照各层真实异物存在区域采样：
    %     CSP: 圆
    %     LGP/SGP/BP: 圆环
    %
    % 同时仍然保留：
    %   1) 避开支撑柱
    %   2) require_reachable=true 时，只保留至少一个入口理论可达的点
    targets = zeros(nT, 3);
    filled = 0;
    max_tries = nT * 200;
    tries = 0;

    while filled < nT && tries < max_tries
        tries = tries + 1;

        [x, y] = sample_target_xy_by_layer(scene, layerId);

        % 拒绝落在障碍内的点
        if point_inside_any_support(scene.layers(layerId).supports, [x, y], scene.armRadius)
            continue;
        end

        % 只保留"理论上存在某个入口可达"的目标，避免无解样本
        if opts.require_reachable && ~reach.query(layerId, [], [x, y])
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

function p = get_phase(ep, C)
    if ep < C.phase1, p = 0;
    elseif ep < C.phase2, p = 1;
    else, p = 2;
    end
end

% =========================================================
%  默认超参数
% =========================================================
function opts = default_opts(opts)
    % --- RL 基础 ---
    if ~isfield(opts,'num_episodes'),        opts.num_episodes        = 2000; end
    if ~isfield(opts,'max_steps'),           opts.max_steps           = 15;   end  % 单 ep 最多选多少次入口
    if ~isfield(opts,'lr'),                  opts.lr                  = 3e-4; end
    if ~isfield(opts,'gamma'),               opts.gamma               = 0.98; end
    if ~isfield(opts,'eps0'),                opts.eps0                = 0.9;  end
    if ~isfield(opts,'eps_min'),             opts.eps_min             = 0.05; end
    if ~isfield(opts,'batch'),               opts.batch               = 64;   end
    if ~isfield(opts,'buf_cap'),             opts.buf_cap             = 50000;end
    if ~isfield(opts,'update_target_freq'),  opts.update_target_freq  = 50;   end
    if ~isfield(opts,'hidden'),              opts.hidden              = [256 256]; end

    % --- 目标集 ---
    if ~isfield(opts,'n_targets_min'),       opts.n_targets_min       = 5;    end
    if ~isfield(opts,'n_targets_max'),       opts.n_targets_max       = 15;   end
    if ~isfield(opts,'require_reachable'),   opts.require_reachable   = true; end

    % --- 奖励权重（关键：体现优先级）---
    %   新开一个定位点代价最大 → 优先级1
    %   新开一个下探孔代价较小 → 优先级2
    %   无进展 step → 惩罚防止循环
    %   全覆盖 → 大奖
    %   超时 → 大罚
    if ~isfield(opts,'R_NEW_ANCHOR'),        opts.R_NEW_ANCHOR        = -15.0; end
    if ~isfield(opts,'R_NEW_HOLE'),          opts.R_NEW_HOLE          = -1.5;  end
    if ~isfield(opts,'R_NO_PROGRESS'),       opts.R_NO_PROGRESS       = -3.0;  end
    if ~isfield(opts,'R_PROGRESS_SCALE'),    opts.R_PROGRESS_SCALE    = 0.2;   end  % 每覆盖 1 个新目标的小奖
    if ~isfield(opts,'R_COVER_BONUS'),       opts.R_COVER_BONUS       = 20.0;  end
    if ~isfield(opts,'R_FAIL'),              opts.R_FAIL              = -25.0; end

    % --- 动作屏蔽 ---
    if ~isfield(opts,'mask_used_holes'),     opts.mask_used_holes     = true; end
    if ~isfield(opts,'useGPU'),              opts.useGPU              = "auto"; end

    % --- 课程 ---
    if ~isfield(opts,'curriculum')
        opts.curriculum.phase1 = 400;    % 阶段0：LGP 小区域 + 少量目标
        opts.curriculum.phase2 = 1200;   % 阶段1：全层 + 中等目标数
    end
end

function useGPU = resolve_use_gpu(requested, label)
    if isstring(requested) || ischar(requested)
        mode = lower(string(requested));
        if mode == "auto"
            useGPU = can_use_gpu();
        elseif mode == "true" || mode == "on" || mode == "gpu"
            useGPU = can_use_gpu();
            if ~useGPU
                warning('train_upper_dqn:useGPU', '%s 请求使用 GPU，但 MATLAB 未检测到可用 GPU，已回退 CPU。', label);
            end
        else
            useGPU = false;
        end
    else
        useGPU = logical(requested) && can_use_gpu();
        if logical(requested) && ~useGPU
            warning('train_upper_dqn:useGPU', '%s 请求使用 GPU，但 MATLAB 未检测到可用 GPU，已回退 CPU。', label);
        end
    end

    if useGPU
        g = gpuDevice();
        fprintf('[%s] 使用 GPU 训练网络: %s\n', label, g.Name);
    else
        fprintf('[%s] 使用 CPU 训练网络。\n', label);
    end
end

function ok = can_use_gpu()
    ok = false;
    try
        if exist('canUseGPU', 'file') == 2 || exist('canUseGPU', 'builtin') == 5
            ok = canUseGPU();
            return;
        end
    catch
    end
    try
        ok = gpuDeviceCount() > 0;
    catch
        try
            gpuDevice();
            ok = true;
        catch
            ok = false;
        end
    end
end

function net = move_network_to_gpu(net)
    net = dlupdate(@gpuArray, net);
end

function net = move_network_to_cpu(net)
    net = dlupdate(@gather, net);
end

function x = to_cpu(x)
    if isa(x, 'gpuArray')
        x = gather(x);
    end
end

% =========================================================
%  训练曲线
% =========================================================
function plot_training_curves(rh, ah, hh, ch, lh)
    figure('Name','Upper DQN v2 training curves','Color','w','Position',[100 100 1300 650]);

    subplot(2,3,1);
    plot(rh, 'Color', [0.7 0.7 0.9]); hold on;
    plot(movmean(rh, 50), 'b-', 'LineWidth', 2);
    xlabel('Episode'); ylabel('Return'); title('Episode Return'); grid on;

    subplot(2,3,2);
    plot(movmean(ah, 50), 'r-', 'LineWidth', 2);
    xlabel('Episode'); ylabel('# Anchors'); title('Anchors used (ma50)'); grid on;

    subplot(2,3,3);
    plot(movmean(hh, 50), 'Color',[0.8 0.4 0.1], 'LineWidth', 2);
    xlabel('Episode'); ylabel('# Holes'); title('Holes used (ma50)'); grid on;

    subplot(2,3,4);
    plot(movmean(ch, 50), 'g-', 'LineWidth', 2);
    xlabel('Episode'); ylabel('Coverage'); title('Target coverage (ma50)');
    ylim([0 1.05]); grid on;

    subplot(2,3,5);
    if ~isempty(lh)
        semilogy(lh, 'k-', 'LineWidth', 0.8);
    end
    xlabel('update step'); ylabel('Huber Loss'); title('Training Loss'); grid on;

    sgtitle('Upper DQN v2: multi-target coverage training');
end
