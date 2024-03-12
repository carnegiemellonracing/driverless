#pragma once

#include <iostream>
#include <cmath>
#include <utils/cuda_utils.cuh>
#include <cassert>

//SOURSE: SEE OVERLEAF MPPI DOCUMENTATION

namespace controls {
    namespace model {
        namespace bicycle {
            //model constants
            constexpr float pi = 3.1415926535;

            //physics constants
            constexpr float GRAVITY = 9.81; //in m/s^2
            constexpr float CG_TO_FRONT = 0.775; //in meters
            constexpr float CG_TO_REAR = 0.775; //in meters
            constexpr float BODY_LENGTH = CG_TO_FRONT + CG_TO_REAR;
            constexpr float WHEEL_RADIUS = 0.2286; //in meters
            constexpr float CAR_MASS = 310; //in KG
            // constexpr float LOGITUDNAL_AERO_CONSTANT = 0; //in Ns^2/m^2 to IMPLEMENT
            // constexpr float LATERAL_AERO_CONSTANT = 0; //in Ns^2/m^2 to IMPLEMENT
            constexpr float WHEEL_ROTATIONAL_INTERIA = .439; //in kg*m^2 TAKEN ALONG AXLE AXIS
            constexpr float CAR_ROTATIONAL_INTERTIA = 105.72; //in kg*m^2. TAKEN ALONG Z AXIS THROUGH CG
            // constexpr float NOSE_CONE_CONSTANT = .0; //in dimensionless to IMPLEMENT. how much of drag becomes downforce


            //tire model constants
            constexpr float max_force_x_at_1N = 0.8f; //Maximum force x TO IMPLEMENT
            constexpr float slip_ratio_max_x = 0.1; //slip ratio that yields the max force TO IMPLEMENT
            constexpr float post_saturation_force_x = 0.6; // After tires start slipping what force we get
            constexpr float max_force_y_at_1N = 0.8f; //Maximum force Y TO IMPLEMENT
            constexpr float slip_angle_max_y = 0.1; //slip ratio that yields the max force TO IMPLEMENT
            constexpr float post_saturation_force_y = 0.6; // After tires start slipping what force we get

            constexpr float slip_ratio_saturation = 0.1; // minimum velocity magnitude for wheel slip

            template<typename T>
            __host__ __device__ int sign(T x) {
                return x > 0 ?
                    1 : x < 0 ?
                        -1 : 0;
            }

            template<typename T>
            __host__ __device__ T square(T x) {
                return x * x;
            }

            template<typename T>
            __host__ __device__ T cross2d(T a, T b, T c, T d) {
                return a * d - b * c;
            }

            //calculates slip ratio
            //Forces array implanted with X and Y force of wheel
            //see overleaf document in MMPI documentation
            __host__ __device__ static void tireModel(float slip_ratio, float slip_angle, float load,
                                      float forces[]) {

                //gets x force
                if (abs(slip_ratio) < abs(slip_ratio_max_x)) {
                    float numerator = load * slip_ratio * max_force_y_at_1N;
                    float within_sqrt = square(tanf(slip_angle)) +
                                        square((max_force_y_at_1N / max_force_x_at_1N));
                    paranoid_assert(!std::isnan(within_sqrt) && "within sqrt was nan in linear x");
                    float denominator = slip_ratio_max_x * sqrtf(within_sqrt);
                    paranoid_assert(denominator > 0 && "denominator was 0 in linear x");
                    forces[0] = numerator / denominator;
                    paranoid_assert(!std::isnan(forces[0]) && "forces[0] was nan in linear x");
                } else {
                    float numerator = load * post_saturation_force_x * max_force_y_at_1N;
                    float within_sqrt = square(tanf(slip_angle)) +
                                        square(max_force_y_at_1N / max_force_x_at_1N);
                    paranoid_assert(!std::isnan(within_sqrt) && "within sqrt was nan in saturated x");
                    float denominator = max_force_x_at_1N * sqrtf(within_sqrt);
                    paranoid_assert(denominator > 0 && "denominator was 0 in saturated x");
                    forces[0] = sign(slip_ratio) * numerator / denominator;
                    paranoid_assert(!std::isnan(forces[0]) && "forces[0] was nan in saturated x");
                }

                //computes y force
                if (abs(slip_angle) < abs(slip_angle_max_y)) {
                    forces[1] = load * max_force_y_at_1N / slip_angle_max_y * slip_angle;
                } else {
                    forces[1] = load * post_saturation_force_y * sign(slip_angle);
                }
                paranoid_assert(!std::isnan(forces[1]) && "forces[1] was nan in tire model");
            }

            //calculates slip ratio
            __host__ __device__ static float calculate_slip_ratio(float wheel_speed, float velocity) {
                velocity = abs(velocity);
                if (velocity == 0) {
                    return sign(wheel_speed) * slip_ratio_saturation;
                }
                float tangential_velo = wheel_speed * WHEEL_RADIUS;
                return clamp(
                        (tangential_velo - velocity) / velocity,
                        -slip_ratio_saturation,
                        slip_ratio_saturation);
            }


            /*state (in order):
           [0] X_world m
           [1] Y_world m
           [2] Yaw_World deg
           [3] X_dot_Car m/s
           [4] Y_dot_Car m/s
           [5] Yaw_Rate deg/s
           [6] Pitch Moment Nm
           [7] Downforce N
           [8] Front Wheel Speed rad/s
           [9] rear Wheel Speed rad/s
           */
            __host__ __device__ static void dynamics(const float state[], const float action[], float state_dot[], float timestep) {
                //unpackages state array
                const float yaw_world = state[state_yaw_idx];
                const float x_dot_car = state[state_car_xdot_idx];
                const float y_dot_car = state[state_car_ydot_idx];
                const float yaw_rate = state[state_yawdot_idx];
                // float pitch_moment = state[6];
                // float downforce = state[7];
                const float front_wheel_speed = state[state_whl_speed_f_idx];
                const float rear_wheel_speed = state[state_whl_speed_r_idx];

                //unpackages action
                const float steering_angle = action[action_swangle_idx];
                const float torque_front = action[action_torque_idx] / 2;
                const float torque_rear = action[action_torque_idx] / 2;

                //compares wheel forces
                float y_dot_front_tire = y_dot_car + yaw_rate *CG_TO_FRONT;
                float front_tire_vel_cross = cross2d(x_dot_car, y_dot_front_tire, cosf(steering_angle), sinf(steering_angle));
                float front_tire_speed = sqrtf(x_dot_car * x_dot_car + y_dot_front_tire * y_dot_front_tire);
                paranoid_assert(!isnan(front_tire_speed) && "front_tire_speed is nan");

                float front_slip_angle = front_tire_speed == 0 ?
                        0 : asinf(clamp(front_tire_vel_cross / front_tire_speed, -1.0f, 1.0f));
                paranoid_assert(!std::isnan(front_slip_angle) && "front_slip_angle is nan");

                float y_dot_rear_tire = y_dot_car - yaw_rate *CG_TO_REAR;
                float rear_tire_speed = sqrtf(x_dot_car * x_dot_car + y_dot_rear_tire * y_dot_rear_tire);
                float rear_slip_angle = rear_tire_speed == 0 ?
                        0 : asinf(clamp(-y_dot_rear_tire / rear_tire_speed, -1.0f, 1.0f));
                paranoid_assert(!std::isnan(rear_tire_speed) && "rear tire speed is nan");
                paranoid_assert(!std::isnan(rear_slip_angle) && "rear slip angle is nan");

                float front_tire_long_component = x_dot_car * cos(steering_angle) + y_dot_front_tire * sin(steering_angle);
                float front_slip_ratio = calculate_slip_ratio(front_wheel_speed, front_tire_long_component);
                float rear_slip_ratio = calculate_slip_ratio(rear_wheel_speed, x_dot_car);
                paranoid_assert(!std::isnan(front_slip_ratio) && "front slip ratio is nan");
                paranoid_assert(!std::isnan(rear_slip_ratio) && "rear slip ratio is nan");

                float front_load =
                        (CAR_MASS * GRAVITY) * CG_TO_REAR / BODY_LENGTH; // - pitch_moment / CG_TO_FRONT; also downforce
                paranoid_assert(!std::isnan(front_load) && "front load is nan");

                float rear_load =
                        (CAR_MASS * GRAVITY) * CG_TO_FRONT / BODY_LENGTH; // + pitch_moment / CG_TO_REAR; also downforce
                paranoid_assert(!std::isnan(rear_load) && "rear load is nan");

                float front_forces_tire[2];  // wrt tire
                float rear_forces_tire[2];

                // symplectic euler: to deal with discontiuities, we take the limit from the appropriate direction
                tireModel(front_slip_ratio, front_slip_angle, front_load, front_forces_tire);
                tireModel(rear_slip_ratio, rear_slip_angle, rear_load, rear_forces_tire);
                float front_wheel_speed_next = front_wheel_speed + (torque_front - WHEEL_RADIUS * front_forces_tire[0]) / WHEEL_ROTATIONAL_INTERIA * timestep;
                float rear_wheel_speed_next = rear_wheel_speed + (torque_rear - WHEEL_RADIUS * rear_forces_tire[0]) / WHEEL_ROTATIONAL_INTERIA * timestep;
                float front_slip_ratio_next = calculate_slip_ratio(front_wheel_speed_next, front_tire_long_component);
                float rear_slip_ratio_next = calculate_slip_ratio(rear_wheel_speed_next, x_dot_car);
                paranoid_assert(!std::isnan(front_slip_ratio_next) && "front slip ratio next is nan");
                paranoid_assert(!std::isnan(rear_slip_ratio_next) && "rear slip ratio next is nan");
                tireModel(front_slip_ratio_next, front_slip_angle, front_load, front_forces_tire);
                tireModel(rear_slip_ratio_next, rear_slip_angle, rear_load, rear_forces_tire);

                float front_force_x_car =
                        front_forces_tire[0] * cosf(steering_angle)
                        - front_forces_tire[1] * sinf(steering_angle);
                paranoid_assert(!std::isnan(front_force_x_car) && "front force x car is nan");

                float front_force_y_car =
                        front_forces_tire[0] * sinf(steering_angle)
                        + front_forces_tire[1] * cosf(steering_angle);
                paranoid_assert(!std::isnan(front_force_y_car) && "front force y car is nan");

                float rear_force_x_car = rear_forces_tire[0];
                paranoid_assert(!std::isnan(rear_force_x_car) && "rear force x car is nan");

                float rear_force_y_car = rear_forces_tire[1];
                paranoid_assert(!std::isnan(rear_force_y_car) && "rear force y car is nan");

                //gets drag
                // float drag_x = LOGITUDNAL_AERO_CONSTANT * square(x_dot_car);
                // float drag_y = LATERAL_AERO_CONSTANT * square(y_dot_car);

                //Updates dot array
                state_dot[state_x_idx] = x_dot_car * cosf(yaw_world) - y_dot_car * sinf(yaw_world);
                state_dot[state_y_idx] = x_dot_car * sinf(yaw_world) + y_dot_car * cosf(yaw_world);
                state_dot[state_yaw_idx] = yaw_rate;
                state_dot[state_car_xdot_idx] = (front_force_x_car + rear_force_x_car) / CAR_MASS + y_dot_car * yaw_rate;
                state_dot[state_car_ydot_idx] = (front_force_y_car + rear_force_y_car) / CAR_MASS - x_dot_car * yaw_rate;
                state_dot[state_yawdot_idx] = (CG_TO_FRONT * front_force_y_car - CG_TO_REAR * rear_force_y_car) / CAR_ROTATIONAL_INTERTIA;
                state_dot[state_my_idx] = 0; //2 * drag_x * NOSE_CONE_CONSTANT * x_dot_car * state_dot[3];
                state_dot[state_fz_idx] = 0; //might need to change
                state_dot[state_whl_speed_f_idx] = (torque_front - WHEEL_RADIUS * front_forces_tire[0]) / WHEEL_ROTATIONAL_INTERIA;
                state_dot[state_whl_speed_r_idx] = (torque_rear - WHEEL_RADIUS * rear_forces_tire[0]) / WHEEL_ROTATIONAL_INTERIA;
            }
        }
    }
}
