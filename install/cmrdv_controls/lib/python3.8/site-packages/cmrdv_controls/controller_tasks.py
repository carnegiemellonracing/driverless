import numpy as np


class LQR(object):

    def __init__(self):
        # attributes initialized here
        self.desired_state = None
        self.current_state = None
        self.gain_matrix = None

    def get_control_action(self):
        rel_error = np.subtract(self.desired_state, self.current_state)
        print(f'REL ERROR: {rel_error}')
        return -np.matmul(self.gain_matrix, rel_error)


class Feedforward(object):
    def __init__(self):
        self.wheel_base_length = 2  # TODO: add real values
        self.rear_mass_length = 0.7
        self.tire_radius = 0.2286

        # x, y, yaw, dx, dy, dyaw (6x1)
        # inertial or relative (actually doesn't matter)
        self.curvature = 0

    def estimate_control_action(self):
        return np.array([
            [0],
            [np.rad2deg(np.arctan(self.curvature * self.wheel_base_length / 
                                 np.sqrt((1 - (self.curvature * self.rear_mass_length) ** 2))))]
        ])

        """return np.array([
            [long_vel / self.tire_radius],  # wheel speed
            [np.rad2deg(np.arctan(self.wheel_base_length * np.deg2rad(self.desired_state[5][0]) / long_vel) \
                if long_vel != 0 else 0)]  # swangle
        ])"""

class SetpointGenerator(object):
    def __init__(self):
        pass

    def get_setpoint(self, speed, car_inertial_position, carrot_relative_position,
                     car_yaw, tangent_relative_yaw, curvature):
        return np.array([
            [car_inertial_position[0] + carrot_relative_position[0]],
            [car_inertial_position[1] + carrot_relative_position[1]],
            [np.rad2deg(car_yaw + tangent_relative_yaw)],
            [speed * np.cos(car_yaw + tangent_relative_yaw)],
            [speed * np.sin(car_yaw + tangent_relative_yaw)],
            [np.rad2deg(speed * curvature)]
        ])
    

