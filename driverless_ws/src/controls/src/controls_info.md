## Cones

Blue cones on the left, yellow cones on the right

## Arrays

State: [x, y, yaw, speed]

Note: we only need scalar speed since steering angle serves as a proxy for direction

Control: [steering angle, torque]

Torque refers to the total torque on all wheels, so we divide it evenly amongst the front two wheels since we are in FWD.


## Coordinate Frames

Cartesian coordinate frame (y-axis is 90 deg counterclockwise from x-axis)

Yaw follows this convention, so counterclockwise from x-axis to y-axis

In slow lap, we center the coordinate frame around the car, with the y-axis pointing in the same direction as the car. So, a straight would have blue cones at a roughly constant negative x-coordinate and increasing y-coordinate, and yellow cones at a roughly constant positive x-coordinate and increasing y-coordinate.

A positive steering angle means the car is turning right.










