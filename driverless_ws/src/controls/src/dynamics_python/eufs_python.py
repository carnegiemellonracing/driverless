import math
import numpy as np

class Param:
    def __init__(self, 
                 m: float, g: float, I_z: float,
                 l: float, b_F: float, b_R: float, w_front: float, l_F: float, l_R: float, axle_width: float,
                 tire_coefficient: float, B: float, C: float, D: float, E: float, radius: float,
                 c_down: float, c_drag: float,
                 acc_min: float, acc_max: float, vel_min: float, vel_max: float, delta_min: float, delta_max: float):
        # Inertia parameters
        self.m = m
        self.g = g
        self.I_z = I_z
        
        # Kinematic parameters
        self.l = l
        self.b_F = b_F
        self.b_R = b_R
        self.w_front = w_front
        self.l_F = l_F
        self.l_R = l_R
        self.axle_width = axle_width
        
        # Tire parameters
        self.tire_coefficient = tire_coefficient
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.radius = radius
        
        # Aero parameters
        self.c_down = c_down
        self.c_drag = c_drag
        
        # Input ranges
        self.acc_min = acc_min
        self.acc_max = acc_max
        self.vel_min = vel_min
        self.vel_max = vel_max
        self.delta_min = delta_min
        self.delta_max = delta_max


class EUFS_Dynamics:
    def __init__(self, m: float, g: float, I_z: float,
                 l: float, b_F: float, b_R: float, w_front: float, l_F: float, l_R: float, axle_width: float,
                 tire_coefficient: float, B: float, C: float, D: float, E: float, radius: float,
                 c_down: float, c_drag: float,
                 acc_min: float, acc_max: float, vel_min: float, vel_max: float, delta_min: float, delta_max: float):
        self.param = Param(self, m, g, I_z, l, b_F, b_R, w_front, l_F, l_R, axle_width, tire_coefficient, B, C, D, E, radius,c_down, c_drag,acc_min, acc_max, vel_min, vel_max, delta_min, delta_max)



    
    def dynamics(self, state, action) -> np.ndarray: # returns new state
        x, y, yaw, v = state[:slipless_state_dim]
        swangle = action[0]
        torque = action[1]

        slip_angle = math.arctan(self.l_R * math.tan(swangle)/ self.l)
        x_dot = v * math.cos(slip_angle + yaw)
        y_dot = v * math.sin(slip_angle + yaw)
        theta_dot = v * math.tan(swangle) * math.cos(slip_angle) / self.l


        next_x = x + x_dot
        next_y = y + y_dot
        next_yaw = yaw + theta_dot
        next_speed = 

        


        
        
        new_state = np.array([next_x, next_y, next_yaw, next_speed])
        return new_state