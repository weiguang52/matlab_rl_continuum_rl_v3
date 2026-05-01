function [env, agent] = run_lower_sac_setup(scene, layerId, entry, target, robot)
%RUN_LOWER_SAC_SETUP Create rlFunctionEnv + SAC agent.
% Requires Reinforcement Learning Toolbox.

    obsInfo = rlNumericSpec([18 1], 'LowerLimit', -inf, 'UpperLimit', inf);
    obsInfo.Name = 'obs';

    actLow = [-8; -0.10; -0.08; -0.10; -0.08; -0.08];
    actUpp = [ 8;  0.10;  0.08;  0.10;  0.08;  0.08];
    actInfo = rlNumericSpec([6 1], 'LowerLimit', actLow, 'UpperLimit', actUpp);
    actInfo.Name = 'du';

    resetFcn = @() localReset(scene, layerId, entry, target, robot);
    stepFcn  = @(action, logSig) localStep(action, logSig, scene, layerId, robot);
    env = rlFunctionEnv(obsInfo, actInfo, stepFcn, resetFcn);

    statePath = [featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'state')
                 fullyConnectedLayer(128)
                 reluLayer
                 fullyConnectedLayer(128)
                 reluLayer];
    actionPath = [featureInputLayer(actInfo.Dimension(1), 'Normalization', 'none', 'Name', 'action')
                  fullyConnectedLayer(128)];
    commonPath = [additionLayer(2, 'Name', 'add')
                  reluLayer
                  fullyConnectedLayer(128)
                  reluLayer
                  fullyConnectedLayer(1, 'Name', 'qout')];
    criticNet = layerGraph(statePath);
    criticNet = addLayers(criticNet, actionPath);
    criticNet = addLayers(criticNet, commonPath);
    criticNet = connectLayers(criticNet, 'relu_2', 'add/in1');
    criticNet = connectLayers(criticNet, 'fullyconnected_3', 'add/in2');
    critic1 = rlQValueFunction(dlnetwork(criticNet), obsInfo, actInfo);
    critic2 = rlQValueFunction(dlnetwork(criticNet), obsInfo, actInfo);

    actorNet = [featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none')
                fullyConnectedLayer(128)
                reluLayer
                fullyConnectedLayer(128)
                reluLayer
                fullyConnectedLayer(2*actInfo.Dimension(1))];
    actor = rlContinuousGaussianActor(dlnetwork(actorNet), obsInfo, actInfo, ...
        'ActionMeanOutputNames', 'fullyconnected_3', ...
        'ActionStandardDeviationOutputNames', 'fullyconnected_3');

    agentOpts = rlSACAgentOptions('SampleTime', 1, 'TargetSmoothFactor', 5e-3, ...
        'MiniBatchSize', 256, 'ExperienceBufferLength', 1e6);
    agent = rlSACAgent(actor, [critic1 critic2], agentOpts);
end

function [obs, logSig] = localReset(scene, layerId, entry, target, robot)
    entryStartZ = scene.layers(layerId).zRange(1);
    q0 = [0; 0; 0; 0; 0; 0];
    model = continuum_forward_model(q0, robot, entry.entryXY, entryStartZ);
    obs = build_obs(q0, model.tipPos, target, scene.layers(layerId).supports);
    logSig.q = q0;
    logSig.target = target;
    logSig.entry = entry;
    logSig.tip = model.tipPos;
    logSig.step = 0;
    logSig.entryStartZ = entryStartZ;
end

function [nextObs, reward, isDone, logSig] = localStep(action, logSig, scene, layerId, robot)
    q = logSig.q + action(:);
    q = clamp_q(q);
    model = continuum_forward_model(q, robot, logSig.entry.entryXY, logSig.entryStartZ);
    [isCol, minClr] = collision_check_centerline(model.centerline, scene.layers(layerId).supports, robot.armRadius, 10);

    dPrev = norm(logSig.target - logSig.tip);
    dNow  = norm(logSig.target - model.tipPos);
    reward = 4*(dPrev - dNow) - 0.02*sum(action(:).^2);
    reward = reward + 0.005*min(minClr, 150);

    isDone = false;
    if isCol
        reward = reward - 120;
        isDone = true;
    elseif dNow < 35
        reward = reward + 150;
        isDone = true;
    elseif q(1) > 620 || q(1) < 20
        reward = reward - 30;
        isDone = true;
    end

    logSig.q = q;
    logSig.tip = model.tipPos;
    logSig.step = logSig.step + 1;
    if logSig.step >= 120
        isDone = true;
    end

    nextObs = build_obs(q, model.tipPos, logSig.target, scene.layers(layerId).supports);
end

function obs = build_obs(q, tipPos, target, supports)
    [dmin, rel] = nearest_obstacle(tipPos(1:2), supports);
    obs = [target(:)-tipPos(:); q(:); dmin; rel(:); norm(target(:)-tipPos(:)); 1];
    obs = obs(:);
    if numel(obs) < 18
        obs(end+1:18,1) = 0;
    else
        obs = obs(1:18);
    end
end

function q = clamp_q(q)
    q(1) = min(max(q(1), 20), 620);
    q(2) = wrapToPi(q(2));
    q(3) = min(max(q(3), -1.3), 1.3);
    q(4) = wrapToPi(q(4));
    q(5) = min(max(q(5), -1.3), 1.3);
    q(6) = wrapToPi(q(6));
end

function [dmin, rel] = nearest_obstacle(xy, supports)
    dmin = inf; rel = [0 0];
    for i = 1:numel(supports)
        ctr = supports(i).xy;
        d = norm(xy - ctr) - supports(i).radius;
        if d < dmin
            dmin = d;
            rel = ctr - xy;
        end
    end
end
