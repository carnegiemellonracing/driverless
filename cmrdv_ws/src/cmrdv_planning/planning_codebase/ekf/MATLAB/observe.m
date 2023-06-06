function [y, Y_r, Y_p] = observe(r, p)
% OBSERVE Transform a point P to robot frame and take a
% range−and−bearing measurement.
%
% In:
% r : robot frame r = [r x ; r y ; r alpha]
% p : point in global frame p = [p x ; p y]
% Out:
% y: range−and−bearing measurement
% Y r: Jacobian wrt r
% Y p: Jacobian wrt p
% (c) 2010, 2011, 2012 Joan Sola
if nargout == 1 % No Jacobians requested
    y = scan(toFrame(r,p));
else % Jacobians requested
    [pr, PR_r, PR_p] = toFrame(r, p);
    [y, Y_pr] = scan(pr);
    % The chain rule!
    Y_r = Y_pr * PR_r;
    Y_p = Y_pr * PR_p;
end
