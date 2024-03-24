#pragma once

#include <constants.hpp>
#include <utils/cuda_utils.cuh>

namespace controls {
    namespace model {
        namespace slipless {
            /**
             * Action : R^2 [swangle (rad), torque (N x m)]
             * State : R^4 [x (m), y (m), yaw (rad), speed (m/s)]
             */

            __host__ __device__ static float angular_accel(const float speed, const float swangle) {
                const float back_whl_to_center_of_rot = (cg_to_front + cg_to_rear) / tanf(swangle);
                return copysignf(speed / sqrtf(cg_to_rear * cg_to_rear + back_whl_to_center_of_rot * back_whl_to_center_of_rot), swangle);
            }

            __host__ __device__ static void dynamics(const float state[], const float action[], float state_dot[], float timestep) {
                const float yaw = state[state_yaw_idx];
                const float speed = state[state_speed_idx];

                const float swangle = action[action_swangle_idx];
                const float torque = action[action_torque_idx] * gear_ratio;

                const float slip_angle = atanf(cg_to_rear / (cg_to_front + cg_to_rear) * tanf(swangle));
                const float speed_yaw = yaw + slip_angle;

                constexpr float saturating_tire_torque = long_tractive_capability * car_mass * whl_radius / 2;
                float torque_front, torque_rear;
                switch (torque_mode) {
                    case TorqueMode::AWD:
                        torque_front = clamp(torque * 0.5f, -saturating_tire_torque, saturating_tire_torque);
                        torque_rear = clamp(torque * 0.5f, -saturating_tire_torque, saturating_tire_torque);
                        break;

                    case TorqueMode::FWD:
                        torque_front = clamp(torque, -saturating_tire_torque, saturating_tire_torque);
                        torque_rear = 0;
                        break;

                    case TorqueMode::RWD:
                        torque_front = 0;
                        torque_rear = clamp(torque, -saturating_tire_torque, saturating_tire_torque);
                        break;
                }

                const float speed_dot =
                    (torque_front * cosf(swangle - slip_angle) + torque_rear * cosf(slip_angle)) / (whl_radius * car_mass)
                    - rolling_drag / car_mass;

                // prevent reversing, and prevent accidentally overshooting 0 speed in integration
                const float speed_dot_adj = max(speed_dot, -max(speed, 0.0f) / timestep);

                state_dot[state_x_idx] = speed * cosf(speed_yaw);
                state_dot[state_y_idx] = speed * sinf(speed_yaw);
                state_dot[state_yaw_idx] = angular_accel(speed, swangle);
                state_dot[state_speed_idx] = speed_dot_adj;
            }
        }
    }
}