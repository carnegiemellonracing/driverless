from constants import *
import numpy as np

def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def cross2d(a, b, c, d):
    return a * d - b * c

def calculate_slip_ratio(wheel_speed, velocity):
    velocity = abs(velocity)

    if velocity == 0:
        return np.sign(wheel_speed) * slip_ratio_saturation

    tangential_velo = wheel_speed * whl_radius
    return np.clip(
        (tangential_velo - velocity) / velocity,
        -slip_ratio_saturation,
        slip_ratio_saturation
    )

def tire_model(slip_ratio, slip_angle, load, forces):
    if abs(slip_ratio) < abs(slip_ratio_max_x):
        numerator = load * slip_ratio * max_force_y_at_1N
        within_sqrt = np.square(np.tan(slip_angle)) + \
            np.square(max_force_y_at_1N / max_force_x_at_1N)
        denominator = slip_ratio_max_x * np.sqrt(within_sqrt)
        forces[0] = numerator / denominator
    else:
        numerator = load * post_saturation_force_x * max_force_y_at_1N
        within_sqrt = np.square(np.tan(slip_angle)) + \
            np.square(max_force_y_at_1N / max_force_x_at_1N)
        denominator = max_force_x_at_1N * np.sqrt(within_sqrt)
        forces[0] = np.sign(slip_ratio) * numerator / denominator

    if abs(slip_angle) < abs(slip_angle_max_y):
        forces[1] = load * max_force_y_at_1N / slip_angle_max_y * slip_angle
    else:
        forces[1] = load * post_saturation_force_y * np.sign(slip_angle)

"""           
[0] X_world m
[1] Y_world m
[2] Yaw_World rad
[3] X_dot_Car m/s
[4] Y_dot_Car m/s
[5] Yaw_Rate rad/s
[6] Pitch Moment Nm
[7] Downforce N
[8] Front Wheel Speed rad/s
[9] rear Wheel Speed rad/s
"""
def dynamics(state, action):
    yaw_world = state[2]
    x_dot_car = state[3]
    y_dot_car = state[4]
    yaw_rate = state[5]

    front_wheel_speed = state[6]
    rear_wheel_speed = state[7]

    steering_angle = action[0]
    torque_front, torque_rear = 0, 0
    if torque_mode == TorqueMode.AWD:
        torque_front = action[1] * 0.5
        torque_rear = action[1] * 0.5
    elif torque_mode == TorqueMode.FWD:
        torque_front = action[1]
        torque_rear = 0
    elif torque_mode == TorqueMode.RWD:
        torque_front = 0
        torque_rear = action[1]

    torque_front *= gear_ratio
    torque_rear *= gear_ratio

    y_dot_front_tire = y_dot_car + yaw_rate * cg_to_front
    front_tire_vel_cross = cross2d(x_dot_car, y_dot_front_tire, np.cos(steering_angle), np.sin(steering_angle))
    front_tire_speed = np.sqrt(x_dot_car * x_dot_car + y_dot_front_tire * y_dot_front_tire)

    front_slip_angle = 0 if front_tire_speed == 0 else np.arcsin(np.clip(front_tire_vel_cross / front_tire_speed, -1.0, 1.0))

    y_dot_rear_tire = y_dot_car - yaw_rate * cg_to_rear
    rear_tire_speed = np.sqrt(x_dot_car * x_dot_car + y_dot_rear_tire * y_dot_rear_tire)
    rear_slip_angle = 0 if rear_tire_speed == 0 else np.arcsin(np.clip(y_dot_rear_tire / rear_tire_speed, -1.0, 1.0))

    front_tire_long_component = x_dot_car * np.cos(steering_angle) + y_dot_front_tire * np.sin(steering_angle)
    front_slip_ratio = calculate_slip_ratio(front_wheel_speed, front_tire_long_component)
    rear_slip_ratio = calculate_slip_ratio(rear_wheel_speed, x_dot_car)

    front_load = (car_mass * gravity) * cg_to_rear / body_length
    rear_load = (car_mass * gravity) * cg_to_front / body_length

    front_forces_tire = np.zeros(2)
    rear_forces_tire = np.zeros(2)

    tire_model(front_slip_ratio, front_slip_angle, front_load, front_forces_tire)
    tire_model(rear_slip_ratio, rear_slip_angle, rear_load, rear_forces_tire)

    front_wheel_speed_next = front_wheel_speed + (torque_front - whl_radius * front_forces_tire[0] - np.sign(front_wheel_speed) * rolling_resistance_tire_torque) / wheel_rotational_inertia * timestep
    rear_wheel_speed_next = rear_wheel_speed + (torque_rear - whl_radius * rear_forces_tire[0] - np.sign(rear_wheel_speed) * rolling_resistance_tire_torque) / wheel_rotational_inertia * timestep
    front_slip_ratio_next = calculate_slip_ratio(front_wheel_speed_next, front_tire_long_component)
    rear_slip_ratio_next = calculate_slip_ratio(rear_wheel_speed_next, x_dot_car)

    tire_model(front_slip_ratio_next, front_slip_angle, front_load, front_forces_tire)
    tire_model(rear_slip_ratio_next, rear_slip_angle, rear_load, rear_forces_tire)

    front_force_x_car = front_forces_tire[0] * np.cos(steering_angle) - front_forces_tire[1] * np.sin(steering_angle)
    front_force_y_car = front_forces_tire[0] * np.sin(steering_angle) + front_forces_tire[1] * np.cos(steering_angle)

    rear_force_x_car = rear_forces_tire[0]
    rear_force_y_car = rear_forces_tire[1]

    state[0] += x_dot_car * np.cos(yaw_world) - y_dot_car * np.sin(yaw_world)
    state[1] += x_dot_car * np.sin(yaw_world) + y_dot_car * np.cos(yaw_world)
    state[2] += yaw_rate
    state[3] += (front_force_x_car + rear_force_x_car) / car_mass + y_dot_car * yaw_rate
    state[4] += (front_force_y_car + rear_force_y_car) / car_mass - x_dot_car * yaw_rate
    state[5] += (cg_to_front * front_force_y_car - cg_to_rear * rear_force_y_car) / car_rotational_inertia
    state[6] += (torque_front - whl_radius * front_forces_tire[0]) / wheel_rotational_inertia
    state[7] += (torque_rear - whl_radius * rear_forces_tire[0]) / wheel_rotational_inertia
    return state

