/**
 * @file model.cuh
 * @brief Slipless dynamics model for MPPI and state estimation.
 *
 * For more information, see the @rst :doc:`overview </source/explainers/slipless_model>`. @endrst
 */

#pragma once
#include <constants.hpp>
#include <utils/cuda_utils.cuh>
#include <cassert>

namespace controls {
    namespace model {
        namespace slipless {


            /*
             * Action : R^2 [swangle (rad), torque (N x m)]
             * State : R^4 [x (m), y (m), yaw (rad), speed (m/s)]
             */

            /**
             * @brief Calculate the kinematic swangle (adjusted for understeering), used for determining center/radius of rotation.
             * @param[in] speed Speed of the car in m/s.
             * @param[in] swangle Steering angle in radians.
             * @return Kinematic steering angle in radians.
             */
            __host__ __device__ static float kinematic_swangle(const float speed, const float swangle) {
                return swangle / (1 + understeer_slope * speed);
            }

            /**
             * @brief Calculates angular speed. Since we assume uniform circular motion, this is the same as yaw rate.
             * @param[in] speed Speed of the car in m/s.
             * @param[in] kinematic_swangle_ Kinematic steering angle in radians.
             * @return Angular speed in rad/s.
             */
            __host__ __device__ static float angular_speed(const float speed, const float kinematic_swangle_) {
                if (kinematic_swangle_ == 0) {
                    return 0;
                }

                const float back_whl_to_center_of_rot = (cg_to_front + cg_to_rear) / tanf(kinematic_swangle_);
                // speed / radius of rotation
                return copysignf(
                    speed / sqrtf(cg_to_rear * cg_to_rear + back_whl_to_center_of_rot * back_whl_to_center_of_rot),
                    kinematic_swangle_
                );
            }

            /**
             * @brief Calculate centripedal acceleration. This is the acceleration towards the center of the turning circle.
             * @param[in] speed Speed of the car in m/s.
             * @param[in] swangle Steering angle in radians.
             * @return Centripedal acceleration in m/s^2.
             */
            __host__ __device__ static float centripedal_accel(const float speed, const float swangle) {
                if (swangle == 0) {
                    return 0;
                }

                const float kinematic_swangle_ = kinematic_swangle(speed, swangle);
                // also equal to angular_speed * speed
                const float back_whl_to_center_of_rot = (cg_to_front + cg_to_rear) / tanf(kinematic_swangle_);
                return copysignf(
                    speed * speed / sqrtf(cg_to_rear * cg_to_rear + back_whl_to_center_of_rot * back_whl_to_center_of_rot),
                    kinematic_swangle_
                );
            }

            /**
             * @brief Calculate slip angle. This is the angle between the direction the car is pointing (heading) and the direction the car is moving (trajectory).
             * It does not necessarily imply the car is slipping.
             * @param[in] kinematic_swangle Kinematic steering angle in radians.
             * @return Slip angle in radians.
             */
            __host__ __device__ static float slip_angle(const float kinematic_swangle) {
                return atanf(cg_to_rear / (cg_to_front + cg_to_rear) * tanf(kinematic_swangle));
            }

            /**
             * @brief Slipless dynamics model. This is the core function that calculates the next state given the current state and action. Can be used in-place (i.e. next_state = state).
             * @param[in] state Current state of the car.
             * @param[in] action Action to take.
             * @param[out] next_state Next state of the car.
             * @param[in] timestep Model time step in seconds.
             */
            __host__ __device__ static void dynamics(const float state[], const float action[], float next_state[], float timestep) {
                assert(false);
                const float x = state[state_x_idx];
                const float y = state[state_y_idx];
                const float yaw = state[state_yaw_idx];
                const float speed = state[state_speed_idx];

                // printf("x: %f, y: %f, yaw: %f, speed: %f\n", x, y, yaw, speed);

                const float swangle = -action[action_swangle_idx]; // ^ This is negated, because the model expects yaw and swangle to have the same direction
                // wheel torque
                const float torque = action[action_torque_idx] * gear_ratio;

                float kinematic_swangle_ = kinematic_swangle(speed, swangle);
                float slip_angle_ = slip_angle(kinematic_swangle_);

                // printf("swangle: %f, torque: %f\n", swangle, torque);
                // printf("kinematic_swangle_: %f, slip_angle_: %f\n", kinematic_swangle_, slip_angle_);

                /// factor of 0.5 to split between front and rear
                constexpr float saturating_tire_torque = saturating_motor_torque * 0.5f * gear_ratio;
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

                // printf("sat_tire_torque: %f, torque_front: %f, torque_rear: %f\n", saturating_tire_torque, torque_front, torque_rear);

                /// direction of the forces has to do with the actual swangle
                const float next_speed_raw = speed +
                    ((torque_front * cosf(swangle - slip_angle_) + torque_rear * cosf(slip_angle_)) / (whl_radius * car_mass)
                    - rolling_drag / car_mass) * timestep;
                // car can't go backwards, negative torque is regenerative braking
                const float next_speed = std::max(0.0f, next_speed_raw);

                const float speed2 = speed * speed;
                const float next_speed2 = next_speed * next_speed;
                const float dist_avg_speed = speed != next_speed ?
                    2.0f/3.0f * (next_speed2 * next_speed - speed2 * speed) / (next_speed2 - speed2) : speed;

                kinematic_swangle_ = kinematic_swangle(dist_avg_speed, swangle);

                // printf("kinematic_swangle_: %f, slip_angle_: %f\n", kinematic_swangle_, slip_angle_);

                const float angular_speed_ = angular_speed(dist_avg_speed, kinematic_swangle_);
                const float next_yaw = yaw + angular_speed_ * timestep;

                // printf("next_speed_raw: %f, next_speed: %f, next_speed2: %f, dist_avg_speed: %f, angular_speed_: %f, next_yaw: %f\n", next_speed_raw, next_speed, next_speed2, dist_avg_speed, angular_speed_, next_yaw);

                const float rear_to_center = (cg_to_rear + cg_to_front) / tanf(kinematic_swangle_);
                // either traveling straight or turning in circular motion
                const float next_x = angular_speed_ == 0 ?
                    x + dist_avg_speed * cosf(yaw) * timestep
                  : x + cg_to_rear * (cosf(next_yaw) - cosf(yaw)) + rear_to_center * (sinf(next_yaw) - sinf(yaw));
                const float next_y = angular_speed_ == 0 ?
                    y + dist_avg_speed * sinf(yaw) * timestep
                  : y + cg_to_rear * (sinf(next_yaw) - sinf(yaw)) + rear_to_center * (-cosf(next_yaw) + cosf(yaw));

                // printf("rear_to_center: %f, next_x: %f, next_y: %f\n\n", rear_to_center, next_x, next_y);

                next_state[state_x_idx] = next_x;
                next_state[state_y_idx] = next_y;
                next_state[state_yaw_idx] = next_yaw;
                next_state[state_speed_idx] = next_speed;
            }
        }
    }
}