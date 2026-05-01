clc; clear; close all;

thisFile = mfilename('fullpath');
thisDir = fileparts(thisFile);
if isempty(thisDir), thisDir = pwd; end
addpath(thisDir);

dataDir = thisDir;
layerData = load_all_layer_data(dataDir);
scene = build_scene_model(layerData);

% Robot parameters (based on user's validated 3D script)
robot.baseLen  = 25;
robot.seg1Len  = 288;
robot.link1Len = 20;
robot.seg2Len  = 288;
robot.link2Len = 20;
robot.armRadius = 55/2;
robot.nProbe = 20;
robot.nBase  = 10;
robot.nSeg1  = 80;
robot.nLink1 = 10;
robot.nSeg2  = 80;
robot.nLink2 = 10;

% Example: choose target in LGP layer (layer 2)
layerId = 2;
zTarget = scene.layers(layerId).zRange(2);
% use a target close to one accessible region while forcing a slight detour
candidateTarget = [450, 120, zTarget];

[entry, scores] = choose_entry_rule_based(scene, layerId, candidateTarget);
fprintf('Chosen layer: %s\n', scene.layers(layerId).name);
fprintf('Chosen anchor: %s\n', string(entry.anchorPoint));
fprintf('Chosen entry hole: %s\n', string(entry.entryHoleLabel));
fprintf('Entry XY = [%.2f, %.2f]\n', entry.entryXY(1), entry.entryXY(2));
fprintf('Target XYZ = [%.2f, %.2f, %.2f]\n', candidateTarget(1), candidateTarget(2), candidateTarget(3));

traj = simulate_demo_rollout(scene, layerId, entry, candidateTarget, robot);

gifFile = fullfile(thisDir, 'demo_rollout.gif');
if exist(gifFile,'file') == 2
    delete(gifFile);
end

fig = figure('Color','w', 'Position', [100 60 1200 760]);
for k = 1:size(traj.q,2)
    render_scene_and_robot(scene, layerId, entry, candidateTarget, robot, traj.q(:,k), fig, true, gifFile);
    drawnow;
end

fprintf('Animation written to: %s\n', gifFile);

% Optional: upper-layer tabular demo
Q = run_upper_qlearning_demo(scene, 300);
disp('Upper-layer Q-learning demo finished.');
disp(size(Q));
