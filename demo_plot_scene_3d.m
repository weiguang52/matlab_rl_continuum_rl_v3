clear; clc; close all;

dataDir = pwd;

% 1. 载入层数据
layerData = load_all_layer_data(dataDir);

% 2. 构建场景
scene = build_scene_model(layerData);

% 3. 三维显示
fig = plot_scene_3d_overview(scene, ...
    'showEntries', true, ...
    'showLabels', true, ...
    'plateAlpha', 0.20, ...
    'supportAlpha', 0.35);

% 4. 可选：保存图片
exportgraphics(fig, 'scene_3d_overview.png', 'Resolution', 300);

disp('场景三维总览已绘制完成，并保存为 scene_3d_overview.png');