function [X,Y] = cov2elli(x,P,n,NP)
% COV2ELLI Ellipse contour from Gaussian mean and covariances matrix.
% [X,Y] = COV2ELLI(X0,P) returns X and Y coordinates of the contour of
% the 1−sigma ellipse of the Gaussian defined by mean X0 and covariances
% matrix P. The contour is defined by 16 points, thus both X and Y are
% 16−vectors.
%
% [X,Y] = COV2ELLI(X0,P,n,NP) returns the n−sigma ellipse and defines the
% contour with NP points instead of the default 16 points.
%
% The ellipse can be plotted in a 2D graphic by just creating a line
% with 'line(X,Y)' or 'plot(X,Y)'.
% Copyright 2008−2009 Joan Sola @ LAAS−CNRS.
if nargin < 4
    NP = 16;
    if nargin < 3
        n = 1;
    end
end

%done after here

alpha = 2*pi/NP*(0:NP); % NP angle intervals for one turn
circle = [cos(alpha);sin(alpha)]; % the unit circle
% SVD method, P = R*D*R' = R*d*d*R'
[R,D]=svd(P);
d = sqrt(D);
% n−sigma ellipse <− rotated 1−sigma ellipse <− aligned 1−sigma ellipse <− unit circle
ellip = n * R * d * circle;
% output ready for plotting (X and Y line vectors)
X = x(1)+ellip(1,:);
Y = x(2)+ellip(2,:);
