function [p, P_y] = invScan(y)
% INVSCAN Backproject a range−and−bearing measure into a 2D point.
%
% In:
% y : range−and−bearing measurement y = [range ; bearing]
% Out:
% p : point in sensor frame p = [p x ; p y]
% P y: Jacobian wrt y
% (c) 2010, 2011, 2012 Joan Sola
d = y(1);
a = y(2);
px = d*cos(a);
py = d*sin(a);
p = [px;py];
if nargout > 1 % Jacobians requested
    P_y = [...
        cos(a) , -d*sin(a)
        sin(a) , d*cos(a)];
end