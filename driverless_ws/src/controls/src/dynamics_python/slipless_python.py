from constants import *
import math
import numpy as np

def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


class Slipless:
    def __init__(self, name="slipless", car_mass=car_mass_default, rolling_drag=rolling_drag_default, long_tractive_capability=long_tractive_capability_default, understeer_slope=understeer_slope_default):
        self.name = name
        self.car_mass = car_mass
        self.rolling_drag = rolling_drag
        self.long_tractive_capability = long_tractive_capability
        self.understeer_slope = understeer_slope
        self.saturating_motor_torque = (self.long_tractive_capability + self.rolling_drag / self.car_mass) * self.car_mass * whl_radius / gear_ratio


    def kinematic_swangle(self, speed, swangle):
        return swangle / (1 + self.understeer_slope * speed)

    def angular_speed(self, speed, kinematic_swangle_):
        if kinematic_swangle_ == 0:
            return 0

        back_whl_to_center_of_rot = (cg_to_front + cg_to_rear) / math.tan(kinematic_swangle_)
        return math.copysign(
            speed / math.sqrt(cg_to_rear**2 + back_whl_to_center_of_rot**2),
            kinematic_swangle_
        )

    def centripedal_accel(self, speed, swangle):
        if swangle == 0:
            return 0

        kinematic_swangle_ = self.kinematic_swangle(speed, swangle)
        back_whl_to_center_of_rot = (cg_to_front + cg_to_rear) / math.tan(kinematic_swangle_)
        return math.copysign(
            speed**2 / math.sqrt(cg_to_rear**2 + back_whl_to_center_of_rot**2),
            kinematic_swangle_
        )

    def slip_angle(self, kinematic_swangle):
        return math.atan(cg_to_rear / (cg_to_front + cg_to_rear) * math.tan(kinematic_swangle))

    def dynamics(self, state, action):
        x, y, yaw, speed = state[:slipless_state_dim]

        swangle = action[0]
        torque = action[1] * gear_ratio

        kinematic_swangle_ = self.kinematic_swangle(speed, swangle)
        slip_angle_ = self.slip_angle(kinematic_swangle_)

        saturating_tire_torque = self.saturating_motor_torque * 0.5 * gear_ratio

        if torque_mode == TorqueMode.AWD:
            torque_front = clamp(torque * 0.5, -saturating_tire_torque, saturating_tire_torque)
            torque_rear = clamp(torque * 0.5, -saturating_tire_torque, saturating_tire_torque)
        elif torque_mode == TorqueMode.FWD:
            torque_front = clamp(torque, -saturating_tire_torque, saturating_tire_torque)
            torque_rear = 0
        elif torque_mode == TorqueMode.RWD:
            torque_front = 0
            torque_rear = clamp(torque, -saturating_tire_torque, saturating_tire_torque)

        next_speed_raw = speed + (
            (torque_front * math.cos(swangle - slip_angle_) + torque_rear * math.cos(slip_angle_)) / 
            (whl_radius * self.car_mass) - self.rolling_drag / self.car_mass
        ) * timestep
        next_speed = max(0.0, next_speed_raw)

        speed2 = speed**2
        next_speed2 = next_speed**2
        dist_avg_speed = (
            2.0/3.0 * (next_speed2 * next_speed - speed2 * speed) / (next_speed2 - speed2) 
            if speed != next_speed else speed
        )

        kinematic_swangle_ = self.kinematic_swangle(dist_avg_speed, swangle)
        angular_speed_ = self.angular_speed(dist_avg_speed, kinematic_swangle_)
        next_yaw = yaw + angular_speed_ * timestep

        rear_to_center = (cg_to_rear + cg_to_front) / math.tan(kinematic_swangle_) +  1e-10
        next_x = (
            x + dist_avg_speed * math.cos(yaw) * timestep if angular_speed_ == 0
            else x + cg_to_rear * (math.cos(next_yaw) - math.cos(yaw)) + 
            rear_to_center * (math.sin(next_yaw) - math.sin(yaw))
        )
        next_y = (
            y + dist_avg_speed * math.sin(yaw) * timestep if angular_speed_ == 0
            else y + cg_to_rear * (math.sin(next_yaw) - math.sin(yaw)) +
            rear_to_center * (-math.cos(next_yaw) + math.cos(yaw))
        )

        next_state = np.array([next_x, next_y, next_yaw, next_speed])
        return next_state

def main():
    pass
    # state = np.array([0.0, 0.0, 0.0, 0.0])
    # action = np.array([-0.01, 8.0])
    # next_state = dynamics(state, action)
    # print(next_state)

if __name__ == "__main__":
    main()