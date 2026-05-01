%% MAIN_RL_TRAIN.m
% 双层 RL 训练主入口（v2：多目标覆盖 + 可达性 LUT）
%
% 运行顺序：
%   1. 解析场景数据
%   2. 离线预计算可达性查找表（首次慢，后续读缓存）
%   3. 训练上层 DQN（多目标序贯覆盖，最少定位点 > 最少下探孔）
%   4. 训练下层 SAC（连续臂控制）
%   5. 联合评估（序贯选入口 → 下层 rollout → 统计）
%   6. 可视化一条轨迹
%
% 快速体验（CPU）：可先将 opts_upper.num_episodes = 400 / opts_lower.num_episodes = 500
% 正式训练建议：upper 2000 ep / lower 5000 ep

clc; clear; close all;

thisFile = mfilename('fullpath');
thisDir  = fileparts(thisFile);
if isempty(thisDir), thisDir = pwd; end
addpath(thisDir);

%% ─────────────────────────────────────────────
%  1. 场景构建
% ──────────────────────────────────────────────
dataDir   = thisDir;
layerData = load_all_layer_data(dataDir);
scene     = build_scene_model(layerData);

robot.baseLen  = 25;
robot.seg1Len  = 288;
robot.link1Len = 20;
robot.seg2Len  = 288;
robot.link2Len = 20;
robot.armRadius = 55/2;
robot.nProbe = 20; robot.nBase = 10;
robot.nSeg1  = 80; robot.nLink1 = 10;
robot.nSeg2  = 80; robot.nLink2 = 10;

fprintf('==============================\n');
fprintf('  场景加载完成\n');
for i = 1:numel(scene.layers)
    fprintf('  层 %d (%s): %d 个支撑柱, %d 个可选入口\n', ...
        i, scene.layers(i).name, ...
        numel(scene.layers(i).supports), ...
        numel(scene.layers(i).entries));
end
fprintf('==============================\n\n');

%% ─────────────────────────────────────────────
%  2. 可达性查找表（离线预计算，缓存可复用）
% ──────────────────────────────────────────────
fprintf('[Main] 预计算可达性查找表（首次约 10~25 分钟，后续直接读缓存）...\n');
reach_opts = struct();
reach_opts.cellSize     = 15;       % 30mm 网格
reach_opts.N_samples    = 20000;     % 每入口蒙特卡洛采样数
reach_opts.dilateRadius = 1;        % 形态学填洞
reach_opts.zTargetTol   = 40;       % 只统计末端接近目标板面的样本，单位 mm
reach_opts.cacheFile    = fullfile(thisDir, 'reachability_lut.mat');
reach_opts.useCache     = true;
reach = precompute_reachability(scene, robot, reach_opts);
fprintf('[Main] 可达性表就绪。\n\n');

%% ─────────────────────────────────────────────
%  3. 上层 DQN 训练（多目标覆盖任务）
% ──────────────────────────────────────────────
opts_upper = struct();
opts_upper.num_episodes = 2000;  % 快速测试可调到 400
opts_upper.max_steps    = 15;    % 单 ep 最多选 15 次入口
opts_upper.hidden       = [256 256];
opts_upper.n_targets_min = 5;
opts_upper.n_targets_max = 15;
opts_upper.curriculum.phase1 = 400;
opts_upper.curriculum.phase2 = 1200;
% 奖励优先级（新定位点 ≫ 新下探孔）
opts_upper.R_NEW_ANCHOR  = -15.0;
opts_upper.R_NEW_HOLE    = -1.5;

fprintf('[Main] 开始上层 DQN 训练（多目标覆盖）...\n');
upper_results = train_upper_dqn(scene, robot, reach, opts_upper);
fprintf('[Main] 上层训练完成。\n\n');

% 保存
save(fullfile(thisDir, 'upper_dqn_result.mat'), 'upper_results', 'reach_opts');

%% ─────────────────────────────────────────────
%  4. 通用下层 SAC 训练（跨层共享控制器）
% ──────────────────────────────────────────────
opts_lower = struct();
opts_lower.num_episodes = 5000;   % ← 调小可快速测试（500）
opts_lower.max_steps    = 150;
opts_lower.hidden       = [256 256];
opts_lower.curriculum.phase1 = 1000;
opts_lower.curriculum.phase2 = 3000;
% 奖励尺度调整：降低碰撞惩罚、提高到达奖励，避免碰撞信号完全压制学习
opts_lower.R_collision  = 50.0;   % 原 150，过大导致回报全为负
opts_lower.R_goal       = 300.0;  % 原 200，增强到达正向激励
opts_lower.k_reach      = 8.0;    % 原 5，加强势差引导
% 优化器参数
opts_lower.lr_actor     = 5e-4;
opts_lower.lr_critic    = 5e-4;
opts_lower.lr_alpha     = 1e-4;
% 通用 SAC 每个 episode 会随机层、随机入口、随机目标。
% 动作是每一步的 Δq，建议比旧版更小，便于连续控制稳定学习。
opts_lower.act_low      = [-8; -0.35; -0.25; -0.35; -0.25; -0.35];
opts_lower.act_high     = [ 8;  0.35;  0.25;  0.35;  0.25;  0.35];
opts_lower.local_radius_phase0 = 300;
opts_lower.local_radius_phase1 = 600;
opts_lower.sample_max_tries    = 500;
opts_lower.local_sample_tries  = 150;

fprintf('[Main] 开始通用下层 SAC 训练（跨层共享）...\n');
lower_results = train_lower_sac_universal(scene, robot, reach, opts_lower);
fprintf('[Main] 通用下层 SAC 训练完成。\n\n');
save(fullfile(thisDir, 'lower_sac_universal.mat'), 'lower_results');

%% ─────────────────────────────────────────────
%  5. 联合评估（多目标覆盖 + 通用下层 SAC 实际 rollout）
% ──────────────────────────────────────────────
fprintf('[Main] 开始联合评估...\n');
eval_results = joint_evaluation_universal(scene, robot, upper_results, lower_results, 20, reach);

fprintf('[Main] 联合评估完成。\n');
fprintf('  下层到达成功率       = %.2f\n', eval_results.overall_success_rate);
fprintf('  上层规划覆盖率       = %.2f\n', eval_results.coverage_rate);
fprintf('  下层碰撞率           = %.2f\n', eval_results.collision_rate);
fprintf('  平均定位点数 / ep    = %.2f\n', eval_results.mean_anchors);
fprintf('  平均下探孔数 / ep    = %.2f\n', eval_results.mean_holes);
fprintf('  平均步数 / 目标      = %.1f\n', eval_results.mean_steps_per_goal);

%% ─────────────────────────────────────────────
%  6. 可视化一条评估轨迹
% ──────────────────────────────────────────────
if ~isempty(eval_results.example_traj)
    traj = eval_results.example_traj;
    ex   = eval_results.example_info;
    gifFile = fullfile(thisDir, 'rl_rollout.gif');

    fig = figure('Color','w','Position',[100 60 1200 760]);
    for k = 1:size(traj.q, 2)
        render_scene_and_robot(scene, ex.layerId, ex.entry, ex.target, ...
            robot, traj.q(:,k), fig, true, gifFile);
        drawnow;
    end
    fprintf('[Main] 轨迹动画已保存至: %s\n', gifFile);
end