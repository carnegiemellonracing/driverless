% SLAM2D A 2D EKF−SLAM algorithm with simulation and graphics.
%
% HELP NOTES:
% 1. The robot state is defined by [xr;yr;ar] with [xr;yr] the position
% and [ar] the orientation angle in the plane.
% 2. The landmark states are simply Li=[xi;yi]. There are a number of N
% landmarks organized in a 2−by−N matrix W=[L1 L2 ... Ln]
% so that Li = W(:,i).
% 3. The control signal for the robot is U=[dx;da] where [dx] is a forward
% motion and [da] is the angle of rotation.
% 4. The motion perturbation is additive Gaussian noise n=[nx;na] with
% covariance Q, which adds to the control signal.
% 5. The measurements are range−and−bearing Yi=[di;ai], with [di] the
% distance from the robot to landmark Li, and [ai] the bearing angle from
% the robot's x−axis.
% 6. The simulated variables are written in capital letters,
% R: robot
% W: set of landmarks or 'world'
% Y: set of landmark measurements Y=[Y1 Y2 ... YN]
% 7. The true map is [xr;yr;ar;x1;y1;x2;y2;x3;y3; ... ;xN;yN]
% 8. The estimated map is Gaussian, defined by
% x: mean of the map
% P: covariances matrix of the map
% 9. The estimated entities (robot and landmarks) are extracted from {x,P}
% via pointers, denoted in small letters as follows:
% r: pointer to robot state. r=[1,2,3]
% l: pointer to landmark i. We have for example l=[4,5] if i=1,
% l=[6,7] if i=2, and so on.
% m: pointers to all used landmarks.
% rl: pointers to robot and one landmark.
% rm: pointers to robot and all landmarks (the currently used map).
% Therefore: x(r) is the robot state,
% x(l) is the state of landmark i
% P(r,r) is the covariance of the robot
% P(l,l) is the covariance of landmark i
% P(r,l) is the cross−variance between robot and lmk i
% P(rm,rm) is the current full covariance −− the rest is
% unused.
% NOTE: Pointers are always row−vectors of integers.
% 10. Managing the map space is done through the variable mapspace.
% mapspace is a logical vector the size of x. If mapspace(i) = false,
% then location i is free. Oterwise mapspace(i) = true. Use it as
% follows:
% * query for n free spaces: s = find(mapspace==false, n);
% * block positions indicated in vector s: mapspace(s) = true;
% * liberate positions indicated in vector s: mapspace(s) = false;
% 11. Managing the existing landmarks is done through the variable landmarks.
% landmarks is a 2−by−N matrix of integers. l=landmarks(:,i) are the
% pointers of landmark i in the state vector x, so that x(l) is the
% state of landmark i. Use it as follows:
% * query 1 free space for a new landmark: i = find(landmarks(1,:)==0,1)
% * associate indices in vector s to landmark i: landmarks(:,i) = s
% * liberate landmark i: landmarks(:,i) = 0;
% 12. Graphics objects are Matlab 'handles'. See Matlab doc for information.
% 13. Graphic objects include:
% RG: simulated robot
% WG: simulated set of landmarks
% rG: estimated robot
% reG: estimated robot ellipse
% lG: estimated landmarks
% leG: estimated landmark ellipses
% (c) 2010, 2011, 2012 Joan Sola.
% I. INITIALIZE
% I.1 SIMULATOR −− use capital letters for variable names
% W: set of external landmarks
W = cloister(-4,4,-4,4,7); % Type 'help cloister' for help
% N: number of landmarks
N = size(W,2);
% R: robot pose [x ; y ; alpha]
R = [0;-2;0];
% U: control [d x ; d alpha]
U = [0.3 ; 0.05]; % fixing advance and turn increments creates a circle (create a own path for use here)
% Y: measurements of all landmarks
Y = zeros(2, N);
% I.2 ESTIMATOR
% Map: Gaussian {x,P}
% x: state vector's mean
x = zeros(numel(R)+numel(W), 1);
% P: state vector's covariances matrix
P = zeros(numel(x),numel(x));
% System noise: Gaussian {0,Q}
q = [.01;.02]; % amplitude or standard deviation
Q = diag(q.^2); % covariances matrix
% Measurement noise: Gaussian {0,S}
s = [.1;1*pi/180]; % amplitude or standard deviation
S = diag(s.^2); % covariances matrix
% Map management
mapspace = false(1,numel(x)); % See Help Note #10 above.
% Landmarks management
landmarks = zeros(2, N); % See Help Note #11 above
% Place robot in map
r = find(mapspace==false, numel(R) ); % set robot pointer
mapspace(r) = true; % block map positions
x(r) = R; % initialize robot states
P(r,r) = 0; % initialize robot covariance

% I.3 GRAPHICS −− use the variable names of simulated and estimated
% variables, followed by a capital G to indicate 'graphics'.
% NOTE: the graphics code is long but absolutely necessary.
% Set figure and axes for Map
mapFig = figure(1); % create figure
cla % clear axes
axis([-6 6 -6 6]) % set axes limits
axis square % set 1:1 aspect ratio

% Simulated World −− set of all landmarks, red crosses
WG = line(...
    'linestyle','none',...
    'marker','+',...
    'color','r',...
    'xdata',W(1,:),...
    'ydata',W(2,:));
% Simulated robot, red triangle
Rshape0 = .2*[...
    2 -1 -1 2; ...
    0 1 -1 0]; % a triangle at the origin
Rshape = fromFrame(R, Rshape0); % a triangle at the robot pose
RG = line(...
    'linestyle','-',...
    'marker','none',...
    'color','r',...
    'xdata',Rshape(1,:),...
    'ydata',Rshape(2,:));

% Estimated robot, blue triangle
rG = line(...
    'linestyle','-',...
    'marker','none',...
    'color','b',...
    'xdata',Rshape(1,:),...
    'ydata',Rshape(2,:));

% Estimated robot ellipse, magenta
reG = line(...
    'linestyle','-',...
    'marker','none',...
    'color','m',...
    'xdata',[ ],...
    'ydata',[ ]);

% Estimated landmark means, blue crosses
lG = line(...
    'linestyle','none',...
    'marker','+',...
    'color','b',...
    'xdata',[ ],...
    'ydata',[ ]);

% Estimated landmark ellipses, green
leG = zeros(1,N);
for i = 1:numel(leG)
leG(i) = line(...
    'linestyle','-',...
    'marker','none',...
    'color','g',...
    'xdata',[ ],...
    'ydata',[ ]);
end

% II. TEMPORAL LOOP
for t = 1:200
    % II.1 SIMULATOR
    % a. motion
    n = q .* randn(2,1); % perturbation vector
    R = move(R, U, zeros(2,1) ); % we will perturb the estimator
    % instead of the simulator

    % b. observations
    for i = 1:N % i: landmark index
        v = s .* randn(2,1); % measurement noise
        Y(:,i) = observe(R, W(:,i)) + v;
    end

    % II.2 ESTIMATOR
    % a. create dynamic map pointers to be used hereafter
    m = landmarks(landmarks ~= 0)'; % all pointers to landmarks %maybe fix
    rm = [r , m]; % all used states: robot and landmarks
    % ( also OK is rm = find(mapspace); )
    % b. Prediction −− robot motion

    [x(r), R_r, R_n] = move(x(r), U, n); % Estimator perturbed with n
    P(r,m) = R_r * P(r,m); % See PDF notes 'SLAM course.pdf'
    P(m,r) = P(r,m)';
    P(r,r) = R_r * P(r,r) * R_r' + R_n * Q * R_n';

    % c. Landmark correction −− known landmarks
    lids = find( landmarks(1,:) ); % returns all indices of existing landmarks

    for i = lids
        % expectation: Gaussian {e,E}
        l = landmarks(:, i)'; % landmark pointer
        [e, E_r, E_l] = observe(x(r), x(l) ); % this is h(x) in EKF
        rl = [r , l]; % pointers to robot and lmk.
        E_rl = [E_r , E_l]; % expectation Jacobian
        E = E_rl * P(rl, rl) * E_rl';
        % measurement of landmark i
        Yi = Y(:, i);

        % innovation: Gaussian {z,Z}
        z = Yi - e; % this is z = y − h(x) in EKF
        % we need values around zero for angles:
        if z(2) > pi
            z(2) = z(2) - 2*pi;
        end
        if z(2) < -pi
            z(2) = z(2) + 2*pi;
        end
        Z = S + E;

        % Individual compatibility check at Mahalanobis distance of 3−sigma
        % (See appendix of documentation file 'SLAM course.pdf')
        if z' * Z^-1 * z < 9
            % Kalman gain
            K = P(rm, rl) * E_rl' * Z^-1; % this is K = P*H'*Zˆ−1 in EKF

            % map update (use pointer rm)
            x(rm) = x(rm) + K*z;
            P(rm,rm) = P(rm,rm) - K*Z*K';
        end
    end

    % d. Landmark Initialization −− one new landmark only at each iteration
    lids = find(landmarks(1,:)==0); % all non−initialized landmarks
    if ~isempty(lids) % there are still landmarks to initialize
        i = lids(randi(numel(lids))); % pick one landmark randomly, its index is i
        l = find(mapspace==false, 2); % pointer of the new landmark in the map
        if ~isempty(l) % there is still space in the map
            mapspace(l) = true; % block map space
            landmarks(:,i) = l; % store landmark pointers

            % measurement
            Yi = Y(:,i);

            % initialization
            [x(l), L_r, L_y] = invObserve(x(r), Yi);
            P(l,rm) = L_r * P(r,rm);
            P(rm,l) = P(l,rm)';
            P(l,l) = L_r * P(r,r) * L_r' + L_y * S * L_y';
        end
    end

    % II.3 GRAPHICS

    % Simulated robot
    Rshape = fromFrame(R, Rshape0);
    set(RG, 'xdata', Rshape(1,:), 'ydata', Rshape(2,:));

    % Estimated robot
    Rshape = fromFrame(x(r), Rshape0);
    set(rG, 'xdata', Rshape(1,:), 'ydata', Rshape(2,:));

    % Estimated robot ellipse
    re = x(r(1:2)); % robot position mean
    RE = P(r(1:2),r(1:2)); % robot position covariance
    [xx,yy] = cov2elli(re,RE,3,16); % x− and y− coordinates of contour
    set(reG, 'xdata', xx, 'ydata', yy);

    % Estimated landmarks
    lids = find(landmarks(1,:)); % all indices of mapped landmarks
    lx = x(landmarks(1,lids)); % all x−coordinates
    ly = x(landmarks(2,lids)); % all y−coordinates
    set(lG, 'xdata', lx, 'ydata', ly);

    % Estimated landmark ellipses −− one per landmark
    for i = lids
        l = landmarks(:,i);
        le = x(l);
        LE = P(l,l);
        [xx,yy] = cov2elli(le,LE,3,16);
        set(leG(i), 'xdata', xx, 'ydata', yy);
    end
    % force Matlab to draw all graphic objects before next iteration
    drawnow
    pause(0.1) %change speed
end

function f = cloister(xmin,xmax,ymin,ymax,n)
% CLOISTER Generates features in a 2D cloister shape.
% CLOISTER(XMIN,XMAX,YMIN,YMAX,N) generates a 2D cloister in the limits
% indicated as parameters.
%
% N is the number of rows and columns; it defaults to N = 9.
% Copyright 2008−2009−2010 Joan Sola @ LAAS−CNRS.
% Copyright 2011−2012−2013 Joan Sola
if nargin < 5
    n = 9;
end
% Center of cloister
x0 = (xmin+xmax)/2;
y0 = (ymin+ymax)/2;

% Size of cloister
hsize = xmax-xmin;
vsize = ymax-ymin;
tsize = diag([hsize vsize]);

% Integer ordinates of points
outer = (-(n-3)/2 : (n-3)/2);
inner = (-(n-3)/2 : (n-5)/2);

% Outer north coordinates
No = [outer; (n-1)/2*ones(1,numel(outer))];

% Inner north
Ni = [inner ; (n-3)/2*ones(1,numel(inner))];

% East (rotate 90 degrees the North points)
E = [0 -1;1 0] * [No Ni];

% South and West are negatives of N and E respectively.
points = [No Ni E -No -Ni -E];

% Rescale
f = tsize*points/(n-1);
% Move
f(1,:) = f(1,:) + x0;
f(2,:) = f(2,:) + y0;
end

