function out = animate_lower_sac_rollout(scene, robot, lower_results, layerId, entry, target, varargin)
%ANIMATE_LOWER_SAC_ROLLOUT  下层通用 SAC 可视化：动画 + 指标曲线。
%
% 用法：
%   out = animate_lower_sac_rollout(scene, robot, lower_results, layerId, entry, target)
%
% 可选参数：
%   'gifFile'       : 保存 GIF 路径，空则不保存
%   'fig'           : 主动画 figure 句柄
%   'metricFig'     : 指标 figure 句柄
%   'pauseTime'     : 每帧暂停时间，默认 0.05 s
%   'frameSkip'     : 每隔几帧绘制一次，默认 1
%   'visible'       : 'on' / 'off'
%   'maxSteps'      : rollout 最长步数
%   'useMeanAction' : true 使用确定性均值动作；false 使用采样动作
%
% 输出：
%   out.traj
%   out.info
%   out.gifFile

    p = inputParser;
    addParameter(p, 'gifFile', '');
    addParameter(p, 'fig', []);
    addParameter(p, 'metricFig', []);
    addParameter(p, 'pauseTime', 0.05);
    addParameter(p, 'frameSkip', 1);
    addParameter(p, 'visible', 'on');
    addParameter(p, 'maxSteps', 150);
    addParameter(p, 'useMeanAction', true);
    parse(p, varargin{:});
    opt = p.Results;

    % 1. 先执行一次 rollout，获得完整轨迹
    [traj, info] = rollout_lower_sac_universal(scene, robot, lower_results, ...
        layerId, entry, target, ...
        'maxSteps', opt.maxSteps, ...
        'useMeanAction', opt.useMeanAction);

    % 2. 创建动画窗口
    if isempty(opt.fig) || ~isgraphics(opt.fig)
        fig = figure('Color', 'w', 'Visible', opt.visible, 'Name', 'Lower SAC Animation');
    else
        fig = opt.fig;
        set(fig, 'Visible', opt.visible);
    end

    % 3. 创建指标窗口
    if isempty(opt.metricFig) || ~isgraphics(opt.metricFig)
        metricFig = figure('Color', 'w', 'Visible', opt.visible, 'Name', 'Lower SAC Metrics');
    else
        metricFig = opt.metricFig;
        set(metricFig, 'Visible', opt.visible);
    end

    % 4. GIF 初始化
    if ~isempty(opt.gifFile) && exist(opt.gifFile, 'file') == 2
        delete(opt.gifFile);
    end

    nStep = size(traj.q, 2);
    frameSkip = max(1, round(opt.frameSkip));

    for k = 1:frameSkip:nStep
        draw_frame(fig, metricFig, scene, layerId, entry, target, robot, traj, info, k);

        if ~isempty(opt.gifFile)
            append_gif_frame(fig, opt.gifFile);
        end

        if strcmpi(opt.visible, 'on')
            pause(opt.pauseTime);
        end
    end

    % 确保最后一帧一定绘制
    if mod(nStep-1, frameSkip) ~= 0
        draw_frame(fig, metricFig, scene, layerId, entry, target, robot, traj, info, nStep);

        if ~isempty(opt.gifFile)
            append_gif_frame(fig, opt.gifFile);
        end
    end

    out.traj = traj;
    out.info = info;
    out.gifFile = opt.gifFile;
end

% =========================================================
% 绘制单帧
% =========================================================
function draw_frame(fig, metricFig, scene, layerId, entry, target, robot, traj, info, k)

    % 3D 动画
    render_scene_and_robot(scene, layerId, entry, target, robot, traj.q(:,k), fig, false, '');
    annotate_animation(fig, traj, info, k);
    draw_tip_trace(fig, traj, k);

    % 指标曲线
    figure(metricFig);
    clf(metricFig);

    t = 1:k;

    subplot(4,1,1);
    plot(t, traj.dist(t), 'LineWidth', 1.5);
    grid on;
    ylabel('dist / mm');
    title(sprintf('Lower SAC rollout metrics | layer=%d | step=%d/%d', ...
        info.layerId, k, size(traj.q,2)));
    if isfield(info, 'goalRadius')
        yline(info.goalRadius, '--');
    end

    subplot(4,1,2);
    plot(t, traj.clearance(t), 'LineWidth', 1.5);
    grid on;
    ylabel('clearance / mm');
    yline(0, '--');

    subplot(4,1,3);
    plot(t, traj.q(1,t), 'LineWidth', 1.2);
    hold on;
    plot(t, traj.q(3,t), 'LineWidth', 1.2);
    plot(t, traj.q(5,t), 'LineWidth', 1.2);
    grid on;
    ylabel('q');
    legend({'zProbe','theta1','theta2'}, 'Location', 'best');

    subplot(4,1,4);
    plot(t, vecnorm(traj.actions(:,t), 2, 1), 'LineWidth', 1.2);
    grid on;
    xlabel('step');
    ylabel('||action||');
end

% =========================================================
% 动画文字标注
% =========================================================
function annotate_animation(fig, traj, info, k)
    figure(fig);
    ax = gca;

    txt = sprintf(['step = %d / %d\n' ...
                   'goal dist = %.1f mm\n' ...
                   'clearance = %.1f mm\n' ...
                   'success = %d\n' ...
                   'collision = %d'], ...
                   k, size(traj.q,2), ...
                   traj.dist(k), traj.clearance(k), ...
                   info.success, info.collision);

    text(ax, 0.02, 0.98, txt, ...
        'Units', 'normalized', ...
        'VerticalAlignment', 'top', ...
        'BackgroundColor', 'w', ...
        'Margin', 6, ...
        'FontSize', 10);
end

% =========================================================
% 画末端轨迹
% =========================================================
function draw_tip_trace(fig, traj, k)
    figure(fig);
    ax = gca;
    hold(ax, 'on');

    idx = 1:k;
    plot3(ax, traj.tip(idx,1), traj.tip(idx,2), traj.tip(idx,3), ...
        'm-', 'LineWidth', 1.8);

    plot3(ax, traj.tip(k,1), traj.tip(k,2), traj.tip(k,3), ...
        'mo', 'MarkerSize', 8, 'MarkerFaceColor', 'm');

    hold(ax, 'off');
end

% =========================================================
% 追加 GIF 帧
% =========================================================
function append_gif_frame(fig, gifFile)
    drawnow;
    frame = getframe(fig);
    [im, map] = rgb2ind(frame2im(frame), 256);

    if exist(gifFile, 'file') ~= 2
        imwrite(im, map, gifFile, 'gif', ...
            'LoopCount', inf, ...
            'DelayTime', 0.06);
    else
        imwrite(im, map, gifFile, 'gif', ...
            'WriteMode', 'append', ...
            'DelayTime', 0.06);
    end
end