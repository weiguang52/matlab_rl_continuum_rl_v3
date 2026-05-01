% debug_plot_reach_lut.m
% 用于单独检查 Reach LUT 可达性稀疏度和区域正确性

clear; clc; close all;

dataDir = pwd;

% 1. 加载场景
layerData = load_all_layer_data(dataDir);
scene = build_scene_model(layerData);

% 2. 加载已经预计算好的 Reach LUT
S = load('reachability_lut.mat', 'reach');
reach = S.reach;

% 3. 绘制 Reach LUT 调试图
stats = plot_reachability_lut_debug(scene, reach, ...
    'saveDir', 'reach_debug_figs', ...
    'perEntryCount', 8, ...
    'visible', 'on');

% 4. 打印统计结果
disp(struct2table(stats));