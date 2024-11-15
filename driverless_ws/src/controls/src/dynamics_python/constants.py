from enum import Enum

class TorqueMode(Enum):
    AWD = 0
    FWD = 1
    RWD = 2

# Slipless model constants

timestep = 0.1
cg_to_front = 0.775
cg_to_rear = 0.775
# cg_to_nose = 1.5
# whl_base = 2.0
whl_radius = 0.2286
gear_ratio = 15.0
car_mass_default = 210.0
rolling_drag_default = 100.0  # N
long_tractive_capability_default = 4.0  # m/s^2
# lat_tractive_capability = 6.0  # m/s^2
understeer_slope_default = 0.0
# brake_enable_speed = 1.0
# saturating_motor_torque = (long_tractive_capability + rolling_drag / car_mass) * car_mass * whl_radius / gear_ratio
torque_mode = TorqueMode.FWD
slipless_state_dim = 4

# Bicycle model constants
gravity = 9.81  # in m/s^2
body_length = cg_to_front + cg_to_rear
wheel_rotational_inertia = 0.439  # in kg*m^2 TAKEN ALONG AXLE AXIS
car_rotational_inertia = 105.72  # in kg*m^2. TAKEN ALONG Z AXIS THROUGH CG

# Bicycle Tire model constants
slip_ratio_saturation = 0.1
slip_ratio_max_x = 0.1
slip_angle_max_y = 0.1
max_force_y_at_1N = 1.0
max_force_x_at_1N = 0.8
post_saturation_force_x = 0.6
post_saturation_force_y = 0.8
rolling_resistance_tire_torque = 10.0

if __name__ == "__main__":
    print(understeer_slope)
    print(cg_to_front)
    print(cg_to_rear)
    print(gear_ratio)
    print(saturating_motor_torque)