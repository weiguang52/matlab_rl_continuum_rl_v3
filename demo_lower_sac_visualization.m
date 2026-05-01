function results = demo_lower_sac_visualization(varargin)
%DEMO_LOWER_SAC_VISUALIZATION  一键可视化下层通用 SAC 的机器人运动效果。
%
% 推荐用法：
%   results = demo_lower_sac_visualization();
%
% 它会：
%   1) 读取场景数据与 robot 参数
%   2) 加载 lower_sac_universal.mat / reachability_lut.mat
%   3) 自动选一个可视化案例，也可以手动指定 layer / entry / target
%   4) 播放机器人运动动画，并可保存 GIF
%
% 可选参数：
%   'dataDir'        工程目录，默认 pwd
%   'lowerFile'      下层 SAC 结果 mat 文件，默认 lower_sac_universal.mat
%   'reachFile'      Reach LUT 文件，默认 reachability_lut.mat
%   'layerId'        指定层号
%   'entryIdx'       指定入口序号
%   'target'         指定目标 [x y z]
%   'gifFile'        保存 GIF 路径
%   'visible'        'on' / 'off'
%   'maxSteps'       rollout 最大步数
%   'useMeanAction'  true: 用确定性均值动作；false: 用随机采样动作

    p = inputParser;
    addParameter(p, 'dataDir', pwd);
    addParameter(p, 'lowerFile', 'lower_sac_universal.mat');
    addParameter(p, 'reachFile', 'reachability_lut.mat');
    addParameter(p, 'layerId', []);
    addParameter(p, 'entryIdx', []);
    addParameter(p, 'target', []);
    addParameter(p, 'gifFile', fullfile(pwd, 'lower_sac_demo.gif'));
    addParameter(p, 'visible', 'on');
    addParameter(p, 'maxSteps', 150);
    addParameter(p, 'useMeanAction', true);
    parse(p, varargin{:});
    opt = p.Results;

    cwd = pwd;
    cleanupObj = onCleanup(@() cd(cwd)); %#ok<NASGU>
    cd(opt.dataDir);

    % 1. 加载场景
    layerData = load_all_layer_data(opt.dataDir);
    scene = build_scene_model(layerData);

    % 2. 加载机器人参数
    % 注意：你的 continuum_arm_3d_control.m 是脚本，不是函数，不能写 robot = continuum_arm_3d_control()
    % 这里直接使用和 main_rl_train.m / 训练过程一致的 robot 参数。
    robot.baseLen  = 25;
    robot.seg1Len  = 288;
    robot.link1Len = 20;
    robot.seg2Len  = 288;
    robot.link2Len = 20;
    robot.armRadius = scene.armRadius;
    
    robot.nProbe = 20;
    robot.nBase  = 10;
    robot.nSeg1  = 80;
    robot.nLink1 = 10;
    robot.nSeg2  = 80;
    robot.nLink2 = 10;

    % 3. 加载通用下层 SAC
    if exist(opt.lowerFile, 'file') ~= 2
        error('找不到下层 SAC 文件：%s。请先训练并保存 lower_sac_universal.mat。', opt.lowerFile);
    end
    S1 = load(opt.lowerFile, 'lower_results');
    lower_results = S1.lower_results;

    % 4. 加载 Reach LUT
    if exist(opt.reachFile, 'file') ~= 2
        error('找不到 Reach LUT 文件：%s。请先生成 reachability_lut.mat。', opt.reachFile);
    end
    S2 = load(opt.reachFile, 'reach');
    reach = S2.reach;

    % 5. 自动或手动选择一个案例
    [layerId, entryIdx, entry, target] = pick_demo_case( ...
        scene, reach, opt.layerId, opt.entryIdx, opt.target);

    fprintf('[Lower SAC Vis] layer=%d (%s), entryIdx=%d, entry=%s, target=[%.1f, %.1f, %.1f]\n', ...
        layerId, scene.layers(layerId).name, entryIdx, get_entry_name(entry), ...
        target(1), target(2), target(3));

    % 6. 播放动画
    vis = animate_lower_sac_rollout(scene, robot, lower_results, layerId, entry, target, ...
        'gifFile', opt.gifFile, ...
        'visible', opt.visible, ...
        'maxSteps', opt.maxSteps, ...
        'useMeanAction', opt.useMeanAction);

    results = vis;
    results.layerId = layerId;
    results.entryIdx = entryIdx;
    results.entry = entry;
    results.target = target;

    fprintf('[Lower SAC Vis] success=%d, collision=%d, steps=%d, finalDist=%.2f mm\n', ...
        results.info.success, results.info.collision, ...
        results.info.steps, results.info.goalDist);

    if ~isempty(opt.gifFile)
        fprintf('[Lower SAC Vis] GIF 已保存到：%s\n', opt.gifFile);
    end
end

% =========================================================
% 自动选择一个可视化 case
% =========================================================
function [layerId, entryIdx, entry, target] = pick_demo_case(scene, reach, layerId, entryIdx, target)

    if isempty(layerId)
        layerId = randi(numel(scene.layers));
    end

    entries = scene.layers(layerId).entries;
    if isempty(entries)
        error('第 %d 层没有 entries。', layerId);
    end

    zTarget = scene.layers(layerId).zRange(2);

    if ~isempty(entryIdx)
        entryIdx = max(1, min(entryIdx, numel(entries)));
        entry = entries(entryIdx);
    else
        entryIdx = randi(numel(entries));
        entry = entries(entryIdx);
    end

    if ~isempty(target)
        target = target(:).';
        return;
    end

    % 优先找当前 entry 可达的目标点
    maxTries = 800;
    for t = 1:maxTries
        [x, y] = sample_target_xy_by_layer(scene, layerId);

        if point_inside_any_support(scene.layers(layerId).supports, [x, y], scene.armRadius)
            continue;
        end

        if reach.query(layerId, entryIdx, [x, y])
            target = [x, y, zTarget];
            return;
        end
    end

    % 如果当前 entry 一直找不到，则换入口再找
    for ei = 1:numel(entries)
        for t = 1:300
            [x, y] = sample_target_xy_by_layer(scene, layerId);

            if point_inside_any_support(scene.layers(layerId).supports, [x, y], scene.armRadius)
                continue;
            end

            if reach.query(layerId, ei, [x, y])
                entryIdx = ei;
                entry = entries(ei);
                target = [x, y, zTarget];
                return;
            end
        end
    end

    % 极端兜底
    warning('没有找到 Reach LUT 可达的随机目标，使用入口正下方作为目标。');
    target = [entry.entryXY(1), entry.entryXY(2), zTarget];
end

% =========================================================
% 工具函数：获取 entry 名称
% =========================================================
function name = get_entry_name(entry)
    if isfield(entry, 'holeName')
        name = string(entry.holeName);
    elseif isfield(entry, 'entryHoleLabel')
        name = string(entry.entryHoleLabel);
    elseif isfield(entry, 'label')
        name = string(entry.label);
    else
        name = "<unknown>";
    end
end