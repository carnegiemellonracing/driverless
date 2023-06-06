function [y, Y_p] = scan (p)
% SCAN perform a range−and−bearing measure of a 2D point.
%
% In:
% p : point in sensor frame p = [p x ; p y]
% Out:
% y : measurement y = [range ; bearing]
% Y p: Jacobian wrt p
% (c) 2010, 2011, 2012 Joan Sola
px = p(1);
py = p(2);
d = sqrt(px^2+py^2);
a = atan2(py,px);
% a = atan(py/px); % use this line if you are in symbolic mode.
y = [d;a];
if nargout > 1 % Jacobians requested
    Y_p = [...
        px/sqrt(px^2+py^2) , py/sqrt(px^2+py^2)
        -py/(px^2*(py^2/px^2 + 1)), 1/(px*(py^2/px^2 + 1)) ];
end
end
