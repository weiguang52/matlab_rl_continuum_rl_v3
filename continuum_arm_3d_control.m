clc; clear; close all;

%% =========================
% 连续体机器臂三维显示（手动输入参数版）
% 不做自动求解，只根据手动输入的构型参数显示场景
% 机器臂本体 = 基座段 + 第一段连续体 + 第一连接段 + 第二段连续体 + 第二连接段
% 下探装置单独显示
%% =========================

%% -------------------------
% 参数定义
%% -------------------------
p.baseLen   = 25;     % 基座段长度（属于机器臂本体）
p.seg1Len   = 288;    % 第一段连续体基线长度
p.link1Len  = 20;     % 第一连接段长度
p.seg2Len   = 288;    % 第二段连续体基线长度
p.link2Len  = 20;     % 第二连接段长度

p.plateGap  = 750;    % 上下板间距，z正方向向下
p.holeR     = 35;     % 上层板圆孔半径
p.plateSize = 400;    % 板尺寸范围
p.probeMax  = 250;    % 仅用于参考，不参与自动控制

% 曲线绘制采样数
p.nProbe = 20;
p.nBase  = 10;
p.nSeg1  = 80;
p.nLink1 = 10;
p.nSeg2  = 80;
p.nLink2 = 10;

%% -------------------------
% 手动输入构型参数
% 直接改这里即可
%% -------------------------
zProbe = 280;              % 下探量
alpha1 = deg2rad(0);     % 第一段旋转角
theta1 = deg2rad(45);     % 第一段弯曲角

gamma  = deg2rad(0);    % 第二段旋转角
theta2 = deg2rad(-45);     % 第二段弯曲角

alpha3 = deg2rad(0);      % 末端旋转角

q = [zProbe; alpha1; theta1; gamma; theta2; alpha3];

%% -------------------------
% 可选目标点：只做显示参考，不参与求解
%% -------------------------
showTarget = true;
target = [180; -60; p.plateGap];

%% -------------------------
% 计算形状与末端位姿
%% -------------------------
[probePts, basePts, seg1Pts, link1Pts, seg2Pts, link2Pts, Ttip] = forwardShapeSeparated(q, p);
tipPos = Ttip(1:3,4);
tipEulerZYX = rotm2eulZYX(Ttip(1:3,1:3));

%% -------------------------
% 终端输出末端位姿
%% -------------------------
fprintf('\n=========== 机器臂末端位姿 ===========\n');
fprintf('末端位置:\n');
fprintf('  x = %.3f\n', tipPos(1));
fprintf('  y = %.3f\n', tipPos(2));
fprintf('  z = %.3f\n', tipPos(3));

fprintf('末端姿态（ZYX Euler）:\n');
fprintf('  yaw   = %.6f rad (%.2f deg)\n', tipEulerZYX(1), rad2deg(tipEulerZYX(1)));
fprintf('  pitch = %.6f rad (%.2f deg)\n', tipEulerZYX(2), rad2deg(tipEulerZYX(2)));
fprintf('  roll  = %.6f rad (%.2f deg)\n', tipEulerZYX(3), rad2deg(tipEulerZYX(3)));

fprintf('\n当前手动输入参数:\n');
fprintf('  zProbe = %.3f\n', zProbe);
fprintf('  alpha1 = %.6f rad (%.2f deg)\n', alpha1, rad2deg(alpha1));
fprintf('  theta1 = %.6f rad (%.2f deg)\n', theta1, rad2deg(theta1));
fprintf('  gamma  = %.6f rad (%.2f deg)\n', gamma,  rad2deg(gamma));
fprintf('  theta2 = %.6f rad (%.2f deg)\n', theta2, rad2deg(theta2));
fprintf('  alpha3 = %.6f rad (%.2f deg)\n', alpha3, rad2deg(alpha3));

%% -------------------------
% 绘图
%% -------------------------
fig = figure('Color','w','Name','连续体机器臂三维显示（手动输入参数）');
ax = axes('Parent', fig);
hold(ax, 'on'); grid(ax, 'on'); axis(ax, 'equal');

view(ax, 42, 22);
xlabel('X');
ylabel('Y');
zlabel('Z (downward)');
title('连续体机器臂三维显示（手动输入构型）');

set(gca, 'FontSize', 11);
set(gca, 'ZDir', 'reverse');

xlim([-p.plateSize, p.plateSize]);
ylim([-p.plateSize, p.plateSize]);
zlim([-40, p.plateGap + 120]);

% 场景
drawUpperPlateWithHole(p);
drawLowerPlate(p);

% 圆孔中心
plot3(0, 0, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');

% 目标点（仅显示）
if showTarget
    plot3(target(1), target(2), target(3), 'rp', ...
        'MarkerSize', 13, 'MarkerFaceColor', 'r');
end

% 下探装置：黑色
plot3(probePts(:,1), probePts(:,2), probePts(:,3), ...
    '-', 'LineWidth', 4, 'Color', [0.15 0.15 0.15]);

% 基座段：深蓝
plot3(basePts(:,1), basePts(:,2), basePts(:,3), ...
    '-', 'LineWidth', 4, 'Color', [0.00 0.20 0.70]);

% 第一段连续体：蓝色
plot3(seg1Pts(:,1), seg1Pts(:,2), seg1Pts(:,3), ...
    '-', 'LineWidth', 3, 'Color', [0.00 0.45 0.95]);

% 第一连接段：青色
plot3(link1Pts(:,1), link1Pts(:,2), link1Pts(:,3), ...
    '-', 'LineWidth', 4, 'Color', [0.00 0.75 0.75]);

% 第二段连续体：红色
plot3(seg2Pts(:,1), seg2Pts(:,2), seg2Pts(:,3), ...
    '-', 'LineWidth', 3, 'Color', [0.90 0.20 0.20]);

% 第二连接段：橙色
plot3(link2Pts(:,1), link2Pts(:,2), link2Pts(:,3), ...
    '-', 'LineWidth', 4, 'Color', [1.00 0.55 0.10]);

% 末端点
plot3(tipPos(1), tipPos(2), tipPos(3), ...
    'mo', 'MarkerSize', 8, 'MarkerFaceColor', 'm');

% 末端坐标系
tipFrameScale = 35;
drawFrame(Ttip, tipFrameScale);

legendItems = {'Entry', 'Target', 'Probe device', 'Base', 'Section 1', ...
               'Link 1', 'Section 2', 'Link 2', 'Robot tip'};
legend(legendItems, 'Location', 'northeastoutside');

%% =========================
% 局部函数
%% =========================

function [probePts, basePts, seg1Pts, link1Pts, seg2Pts, link2Pts, Ttip] = forwardShapeSeparated(q, p)
    zProbe = q(1);
    alpha1 = q(2);
    theta1 = q(3);
    gamma  = q(4);
    theta2 = q(5);
    alpha3 = q(6);

    % 1) 下探装置
    T0 = eye(4);
    probePts = sampleStraight(T0, zProbe, p.nProbe);

    % 2) 到达机器臂本体起点
    T = T0 * transZ(zProbe);

    % 3) 基座段
    basePts = sampleStraight(T, p.baseLen, p.nBase);
    T = T * transZ(p.baseLen);

    % 4) 第一段连续体
    [seg1Pts, T] = sampleCurvedSection(T, p.seg1Len, theta1, alpha1, p.nSeg1);

    % 5) 第一连接段
    link1Pts = sampleStraight(T, p.link1Len, p.nLink1);
    T = T * transZ(p.link1Len);

    % 6) 第二段连续体
    [seg2Pts, T] = sampleCurvedSection(T, p.seg2Len, theta2, gamma, p.nSeg2);

    % 7) 第二连接段
    link2Pts = sampleStraight(T, p.link2Len, p.nLink2);
    T = T * transZ(p.link2Len);

    % 8) 末端旋转
    T = T * rotZ(alpha3);

    % 真正机器臂末端位姿
    Ttip = T;
end

function P = sampleStraight(T0, L, N)
    if N < 2
        N = 2;
    end
    s = linspace(0, L, N).';
    P = zeros(N,3);
    for i = 1:N
        Ti = T0 * transZ(s(i));
        P(i,:) = Ti(1:3,4).';
    end
end

function [P, Tend] = sampleCurvedSection(T0, L, theta, alpha, N)
    s = linspace(0, L, N).';
    P = zeros(N,3);

    if abs(theta) < 1e-10
        for i = 1:N
            Ti = T0 * transZ(s(i));
            P(i,:) = Ti(1:3,4).';
        end
        Tend = T0 * transZ(L);
        return;
    end

    R = L / theta;
    for i = 1:N
        phi = theta * (s(i)/L);
        Ti = T0 * ...
             rotZ(alpha) * ...
             transZ(R*sin(phi)) * ...
             transX(R*(1-cos(phi))) * ...
             rotY(phi) * ...
             rotZ(-alpha);
        P(i,:) = Ti(1:3,4).';
    end

    Tend = T0 * ...
           rotZ(alpha) * ...
           transZ(R*sin(theta)) * ...
           transX(R*(1-cos(theta))) * ...
           rotY(theta) * ...
           rotZ(-alpha);
end

function drawUpperPlateWithHole(p)
    N = 160;
    x = linspace(-p.plateSize, p.plateSize, N);
    y = linspace(-p.plateSize, p.plateSize, N);
    [X, Y] = meshgrid(x, y);
    Z = zeros(size(X));

    mask = (X.^2 + Y.^2) <= p.holeR^2;
    Z(mask) = NaN;

    surf(X, Y, Z, ...
        'FaceColor', [0.65 0.82 1.00], ...
        'EdgeColor', 'none', ...
        'FaceAlpha', 0.85);

    t = linspace(0, 2*pi, 200);
    plot3(p.holeR*cos(t), p.holeR*sin(t), zeros(size(t)), ...
        'k-', 'LineWidth', 2);
end

function drawLowerPlate(p)
    [X,Y] = meshgrid(linspace(-p.plateSize, p.plateSize, 2));
    Z = p.plateGap * ones(size(X));
    surf(X, Y, Z, ...
        'FaceColor', [0.80 0.95 0.80], ...
        'EdgeColor', 'none', ...
        'FaceAlpha', 0.90);
end

function drawFrame(T, s)
    o = T(1:3,4);
    x = o + s*T(1:3,1);
    y = o + s*T(1:3,2);
    z = o + s*T(1:3,3);

    plot3([o(1),x(1)], [o(2),x(2)], [o(3),x(3)], 'r-', 'LineWidth', 2);
    plot3([o(1),y(1)], [o(2),y(2)], [o(3),y(3)], 'g-', 'LineWidth', 2);
    plot3([o(1),z(1)], [o(2),z(2)], [o(3),z(3)], 'b-', 'LineWidth', 2);
end

function eul = rotm2eulZYX(R)
    if abs(R(3,1)) < 1 - 1e-9
        pitch = -asin(R(3,1));
        roll  = atan2(R(3,2), R(3,3));
        yaw   = atan2(R(2,1), R(1,1));
    else
        pitch = -asin(max(min(R(3,1),1),-1));
        roll  = 0;
        yaw   = atan2(-R(1,2), R(2,2));
    end
    eul = [yaw, pitch, roll];
end

function T = transX(x)
    T = [1 0 0 x;
         0 1 0 0;
         0 0 1 0;
         0 0 0 1];
end

function T = transZ(z)
    T = [1 0 0 0;
         0 1 0 0;
         0 0 1 z;
         0 0 0 1];
end

function T = rotY(th)
    c = cos(th); s = sin(th);
    T = [ c 0 s 0;
          0 1 0 0;
         -s 0 c 0;
          0 0 0 1];
end

function T = rotZ(th)
    c = cos(th); s = sin(th);
    T = [ c -s 0 0;
          s  c 0 0;
          0  0 1 0;
          0  0 0 1];
end