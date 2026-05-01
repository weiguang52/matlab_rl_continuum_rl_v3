function render_scene_and_robot(scene, layerId, entry, target, robot, q, fig, doCapture, gifFile)
%RENDER_SCENE_AND_ROBOT Draw plates, supports, entry, target, robot.

    if nargin < 7 || isempty(fig)
        fig = figure('Color','w');
    else
        figure(fig);
    end
    clf(fig);
    ax = axes('Parent', fig); hold(ax,'on'); grid(ax,'on'); axis(ax,'equal');
    view(ax, 38, -22);
    xlabel(ax,'X'); ylabel(ax,'Y'); zlabel(ax,'Z / depth (+down)');
    title(ax, sprintf('Layer %d (%s) demo', layerId, scene.layers(layerId).name));

    xlim(ax, [scene.gridBoundsX(1)-120, scene.gridBoundsX(2)+120]);
    ylim(ax, [scene.gridBoundsY(1)-120, scene.gridBoundsY(2)+120]);
    zlim(ax, [-100, scene.plateZ(end)+150]);

    draw_all_plates(scene, ax);
    draw_grid_lines(scene, ax);
    draw_supports(scene.layers(layerId).supports, scene.layers(layerId).zRange, scene, ax);

    entryStartZ = scene.layers(layerId).zRange(1);
    plot3(ax, entry.entryXY(1), entry.entryXY(2), entryStartZ, 'ko', 'MarkerSize', 7, 'MarkerFaceColor', 'y');
    plot3(ax, target(1), target(2), target(3), 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r');

    arm = continuum_forward_model(q, robot, entry.entryXY, entryStartZ);
    plot3(ax, arm.probePts(:,1), arm.probePts(:,2), arm.probePts(:,3), 'k-', 'LineWidth', 4);
    plot3(ax, arm.basePts(:,1), arm.basePts(:,2), arm.basePts(:,3), '-', 'LineWidth', 4, 'Color', [0.0 0.2 0.7]);
    plot3(ax, arm.seg1Pts(:,1), arm.seg1Pts(:,2), arm.seg1Pts(:,3), '-', 'LineWidth', 3, 'Color', [0.0 0.45 0.95]);
    plot3(ax, arm.link1Pts(:,1), arm.link1Pts(:,2), arm.link1Pts(:,3), '-', 'LineWidth', 4, 'Color', [0.0 0.75 0.75]);
    plot3(ax, arm.seg2Pts(:,1), arm.seg2Pts(:,2), arm.seg2Pts(:,3), '-', 'LineWidth', 3, 'Color', [0.90 0.20 0.20]);
    plot3(ax, arm.link2Pts(:,1), arm.link2Pts(:,2), arm.link2Pts(:,3), '-', 'LineWidth', 4, 'Color', [1.00 0.55 0.10]);
    plot3(ax, arm.tipPos(1), arm.tipPos(2), arm.tipPos(3), 'mo', 'MarkerSize', 8, 'MarkerFaceColor', 'm');

    if nargin >= 8 && doCapture
        fr = getframe(fig);
        [im, map] = rgb2ind(frame2im(fr), 256);
        if exist(gifFile, 'file') ~= 2
            imwrite(im, map, gifFile, 'gif', 'LoopCount', inf, 'DelayTime', 0.06);
        else
            imwrite(im, map, gifFile, 'gif', 'WriteMode', 'append', 'DelayTime', 0.06);
        end
    end
end

function draw_all_plates(scene, ax)
    colors = [0.85 0.92 1.00; 0.88 0.97 0.88; 0.95 0.92 0.82; 0.92 0.86 0.96; 0.86 0.94 0.94];
    for i = 1:numel(scene.plateZ)
        z = scene.plateZ(i);
        X = [scene.gridBoundsX(1), scene.gridBoundsX(2); scene.gridBoundsX(1), scene.gridBoundsX(2)];
        Y = [scene.gridBoundsY(1), scene.gridBoundsY(1); scene.gridBoundsY(2), scene.gridBoundsY(2)];
        Z = z * ones(2);
        surf(ax, X, Y, Z, 'FaceAlpha', 0.12, 'EdgeColor', 'none', 'FaceColor', colors(i,:));
    end
end

function draw_grid_lines(scene, ax)
    pitches = (-7:7) * scene.gridPitch;
    for i = 1:numel(scene.plateZ)
        z = scene.plateZ(i);
        for x = pitches
            plot3(ax, [x x], scene.gridBoundsY, [z z], '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.5);
        end
        for y = pitches
            plot3(ax, scene.gridBoundsX, [y y], [z z], '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.5);
        end
    end
end

function draw_supports(supports, zRange, scene, ax)
    zTop = zRange(1);
    zBot = zRange(2);
    th = linspace(0, 2*pi, 50);
    for i = 1:numel(supports)
        if ~isfield(supports(i), 'radius') || ~isfinite(supports(i).radius) || supports(i).radius <= 0
            continue;
        end

        ctr = supports(i).xy;
        r = supports(i).radius;
        rs = r + scene.safeInflation;

        [X,Y,Z] = cylinder(r, 40);
        Z = zBot + (zTop-zBot)*Z;
        surf(ax, X+ctr(1), Y+ctr(2), Z, 'FaceAlpha', 0.35, 'EdgeColor', 'none', 'FaceColor', [0.4 0.4 0.45]);

        [Xs,Ys,Zs] = cylinder(rs, 40);
        Zs = zBot + (zTop-zBot)*Zs;
        surf(ax, Xs+ctr(1), Ys+ctr(2), Zs, 'FaceAlpha', 0.08, 'EdgeColor', 'none', 'FaceColor', [1.0 0.2 0.2]);

        for k = 1:4
            hp = supports(i).holeXY(k,:);
            plot3(ax, hp(1)+scene.holeRadius*cos(th), hp(2)+scene.holeRadius*sin(th), zTop*ones(size(th)), 'b-');
        end
    end
end