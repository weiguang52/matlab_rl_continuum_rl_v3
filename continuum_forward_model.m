function out = continuum_forward_model(q, robot, baseXY, entryStartZ)
%CONTINUUM_FORWARD_MODEL Constant-curvature forward shape model.
% q = [zProbe; alpha1; theta1; gamma; theta2; alpha3]
% Coordinate convention: scene +Z points downward.
%
% entryStartZ is the board z-coordinate where the insertion starts.
% zProbe is a relative insertion length measured from entryStartZ, not
% an absolute global z coordinate.
%
% Examples:
%   CSP target board: entryStartZ = 0
%   LGP target board: entryStartZ = 736.42
%   SGP target board: entryStartZ = 1714.02
%   BP  target board: entryStartZ = 2209.62

    if nargin < 4 || isempty(entryStartZ)
        entryStartZ = 0;
    end

    zProbe = q(1);
    alpha1 = q(2);
    theta1 = q(3);
    gamma  = q(4);
    theta2 = q(5);
    alpha3 = q(6);

    T0 = eye(4);
    T0(1:2,4) = baseXY(:);
    T0(3,4) = entryStartZ;

    probePts = sampleStraight(T0, zProbe, robot.nProbe);
    T = T0 * transZ(zProbe);

    basePts = sampleStraight(T, robot.baseLen, robot.nBase);
    T = T * transZ(robot.baseLen);

    [seg1Pts, T] = sampleCurvedSection(T, robot.seg1Len, theta1, alpha1, robot.nSeg1);
    link1Pts = sampleStraight(T, robot.link1Len, robot.nLink1);
    T = T * transZ(robot.link1Len);

    [seg2Pts, T] = sampleCurvedSection(T, robot.seg2Len, theta2, gamma, robot.nSeg2);
    link2Pts = sampleStraight(T, robot.link2Len, robot.nLink2);
    T = T * transZ(robot.link2Len);

    T = T * rotZ(alpha3);

    out.probePts = probePts;
    out.basePts = basePts;
    out.seg1Pts = seg1Pts;
    out.link1Pts = link1Pts;
    out.seg2Pts = seg2Pts;
    out.link2Pts = link2Pts;
    out.centerline = [probePts; basePts(2:end,:); seg1Pts(2:end,:); link1Pts(2:end,:); seg2Pts(2:end,:); link2Pts(2:end,:)];
    out.tipPos = T(1:3,4)';
    out.Ttip = T;
end

function P = sampleStraight(T0, L, N)
    N = max(N,2);
    s = linspace(0, L, N)';
    P = zeros(N,3);
    for i = 1:N
        Ti = T0 * transZ(s(i));
        P(i,:) = Ti(1:3,4)';
    end
end

function [P, Tend] = sampleCurvedSection(T0, L, theta, alpha, N)
    s = linspace(0, L, N)';
    P = zeros(N,3);
    if abs(theta) < 1e-10
        for i = 1:N
            Ti = T0 * transZ(s(i));
            P(i,:) = Ti(1:3,4)';
        end
        Tend = T0 * transZ(L);
        return;
    end

    R = L / theta;
    for i = 1:N
        phi = theta * (s(i)/L);
        Ti = T0 * rotZ(alpha) * transZ(R*sin(phi)) * transX(R*(1-cos(phi))) * rotY(phi) * rotZ(-alpha);
        P(i,:) = Ti(1:3,4)';
    end
    Tend = T0 * rotZ(alpha) * transZ(R*sin(theta)) * transX(R*(1-cos(theta))) * rotY(theta) * rotZ(-alpha);
end

function T = transX(x)
    T = [1 0 0 x; 0 1 0 0; 0 0 1 0; 0 0 0 1];
end
function T = transZ(z)
    T = [1 0 0 0; 0 1 0 0; 0 0 1 z; 0 0 0 1];
end
function T = rotY(th)
    c = cos(th); s = sin(th);
    T = [c 0 s 0; 0 1 0 0; -s 0 c 0; 0 0 0 1];
end
function T = rotZ(th)
    c = cos(th); s = sin(th);
    T = [c -s 0 0; s c 0 0; 0 0 1 0; 0 0 0 1];
end
