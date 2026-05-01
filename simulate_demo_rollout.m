function traj = simulate_demo_rollout(scene, layerId, entry, target, robot)
%SIMULATE_DEMO_ROLLOUT Heuristic rollout for visualization.
% Generates feasible-ish q sequence to show 3D motion.

    supports = scene.layers(layerId).supports;
    entryStartZ = scene.layers(layerId).zRange(1);
    q = [0; 0; 0; 0; 0; 0];

    entryXY = entry.entryXY(:).';
    target = target(:).';
    targetXY = target(1:2);

    steps = 90;
    traj.q = zeros(6, steps);
    traj.tip = zeros(steps, 3);
    traj.collision = false(steps,1);
    traj.entryXY = entryXY;
    traj.target = target;

    % find nearest troublesome obstacle between entry and target
    avoid = pick_dominant_obstacle(entryXY, targetXY, supports, robot.armRadius + 20);
    sideSign = 1;
    if ~isempty(avoid)
        v1 = targetXY - entryXY;
        v2 = avoid.xy(:).' - entryXY;
        sideSign = sign(v1(1)*v2(2) - v1(2)*v2(1));
        if sideSign == 0, sideSign = 1; end
    end

    for k = 1:steps
        t = (k-1)/(steps-1);

        % desired tip path: descend first, then bend in xy, finally refine
        pdes = (1-t)*[entryXY, 100] + t*target;
        if ~isempty(avoid)
            bump = exp(-((t-0.55)/0.18)^2);
            dir = targetXY - entryXY;
            dir = dir / max(norm(dir),1e-8);
            nrm = sideSign * [-dir(2), dir(1)];
            pdes(1:2) = pdes(1:2) + bump * 180 * nrm;
        end

        zNeed = min(pdes(3) - 150, 580);
        zNeed = max(zNeed, 40);
        q(1) = q(1) + 0.18*(zNeed - q(1));

        rel = pdes(1:2) - entryXY;
        rho = norm(rel);
        az = atan2(rel(2), rel(1));

        thetaMag = min(1.25, rho / 280);
        q(2) = az;
        q(3) = 0.65*thetaMag;
        q(4) = az + 0.25*sin(2*pi*t);
        q(5) = 0.95*thetaMag;
        q(6) = 0;

        model = continuum_forward_model(q, robot, entryXY, entryStartZ);
        [isCol, ~] = collision_check_centerline(model.centerline, supports, robot.armRadius, 10);

        if isCol
            q(3) = q(3) * 0.94;
            q(5) = q(5) * 1.03;
            q(2) = q(2) + 0.08*sideSign;
            q(4) = q(4) + 0.12*sideSign;
            model = continuum_forward_model(q, robot, entryXY, entryStartZ);
            isCol = collision_check_centerline(model.centerline, supports, robot.armRadius, 10);
        end

        traj.q(:,k) = q;
        traj.tip(k,:) = model.tipPos;
        traj.collision(k) = isCol;
    end
end

function obs = pick_dominant_obstacle(p0, p1, supports, extra)
    p0 = p0(:);
    p1 = p1(:);

    bestScore = inf;
    obs = [];
    for i = 1:numel(supports)
        if ~isfield(supports(i), 'radius') || ~isfinite(supports(i).radius) || supports(i).radius <= 0
            continue;
        end

        ctr = supports(i).xy(:);
        r = supports(i).radius + extra;
        d = point_to_segment_distance_2d(ctr, p0, p1);
        proj = projection_ratio(ctr, p0, p1);
        if d < r + 120 && proj > 0.15 && proj < 0.85
            score = d - r;
            if score < bestScore
                bestScore = score;
                obs = supports(i);
            end
        end
    end
end

function d = point_to_segment_distance_2d(pt, a, b)
    pt = pt(:);
    a = a(:);
    b = b(:);

    ab = b - a;
    if norm(ab) < 1e-9
        d = norm(pt-a); return;
    end
    t = dot(pt-a, ab) / dot(ab,ab);
    t = max(0,min(1,t));
    proj = a + t*ab;
    d = norm(pt-proj);
end

function t = projection_ratio(pt, a, b)
    pt = pt(:);
    a = a(:);
    b = b(:);

    ab = b - a;
    if norm(ab) < 1e-9
        t = 0; return;
    end
    t = dot(pt-a, ab) / dot(ab,ab);
end