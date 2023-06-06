function [p, P_r, P_y] = invObserve(r, y)
% INVOBSERVE Backproject a range−and−bearing measurement and transform
% to map frame.
%
% In:
% r : robot frame r = [r x ; r y ; r alpha]
% y : measurement y = [range ; bearing]
% Out:
% p : point in sensor frame
% P r: Jacobian wrt r
% P y: Jacobian wrt y
% (c) 2010, 2011, 2012 Joan Sola
if nargout == 1 % No Jacobians requested
    p = fromFrame(r, invScan(y));
else % Jacobians requested
    [p_r, PR_y] = invScan(y);
    [p, P_r, P_pr] = fromFrame(r, p_r);
    % here the chain rule !
    P_y = P_pr * PR_y;
end
end
