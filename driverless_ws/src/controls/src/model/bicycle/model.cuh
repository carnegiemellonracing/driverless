#include <iostream>
#include <cmath>

//SOURSE: SEE OVERLEAF MPPI DOCUMENTATION

namespace controls {
    namespace model {
        namespace bicycle {
            //model constants
            constexpr float pi = 3.1415926535;
            constexpr float ACTION_DIMS = 3;
            constexpr float STATE_DIMS = 10;

            //physics constants
            constexpr float GRAVITY = 9.81; //in m/s^2
            constexpr float CG_TO_FRONT = 0.775; //in meters
            constexpr float CG_TO_REAR = 0.775; //in meters
            constexpr float BODY_LENGTH = CG_TO_FRONT + CG_TO_REAR;
            constexpr float WHEEL_RADIUS = 0.2286; //in meters
            constexpr float CAR_MASS = 310; //in KG
            constexpr float LOGITUDNAL_AERO_CONSTANT = 0; //in Ns^2/m^2 to IMPLEMENT
            constexpr float LATERAL_AERO_CONSTANT = 0; //in Ns^2/m^2 to IMPLEMENT
            constexpr float WHEEL_ROTATIONAL_INTERIA = .439; //in kg*m^2 TAKEN ALONG AXLE AXIS
            constexpr float CAR_ROTATIONAL_INTERTIA = 105.72; //in kg*m^2. TAKEN ALONG Z AXIS THROUGH CG
            constexpr float NOSE_CONE_CONSTANT = .0; //in dimensionless to IMPLEMENT. how much of drag becomes downforce


            //tire model constants
            constexpr float max_force_x_at_1N = 1.3f; //Maximum force x TO IMPLEMENT
            constexpr float slip_ratio_max_x = 0.1; //slip ratio that yields the max force TO IMPLEMENT
            constexpr float post_saturation_force_x = 1; // After tires start slipping what force we get
            constexpr float max_force_y_at_1N = 1.5f; //Maximum force Y TO IMPLEMENT
            constexpr float slip_angle_max_y = 0.1; //slip ratio that yields the max force TO IMPLEMENT
            constexpr float post_saturation_force_y = 1; // After tires start slipping what force we get

            //trig functions in degrees
//            __host__ __device__ static float cosd(float degrees) {
//                return cosf(degrees * pi / 180);
//            }
//
//            //trig functions in degrees
//            __host__ __device__ static float sind(float degrees) {
//                return sinf(degrees * pi / 180);
//            }
//
//            //trig functions in degrees
//            __host__ __device__ static float tand(float degrees) {
//                return tanf(degrees * pi / 180);
//            }
//
//            //trig functions in degrees
//            __host__ __device__ static float cotd(float degrees) {
//                return 1 / tanf(degrees * pi / 180);
//            }
//
//            //trig functions in degrees
//            __host__ __device__ static float arccosd(float input) {
//                return 180*acos(input)/pi;
//            }

            template<typename T>
            __host__ __device__ int sign(T x) {
                return x > 0 ?
                    1 : x < 0 ?
                        -1 : 0;
            }


            //calculates slip ratio
            //Forces array implanted with X and Y force of wheel
            //see overleaf document in MMPI documentation
            __host__ __device__ static void tireModel(float slip_ratio, float slip_angle, float load,
                                      float forces[]) {

                //gets x force
                if (abs(slip_ratio) < abs(slip_ratio_max_x)) {
                    float numerator = load * slip_ratio * max_force_y_at_1N;
                    float within_sqrt = powf(tanf(slip_angle), 2) +
                                        powf((max_force_y_at_1N / max_force_x_at_1N), 2);
                    float denominator = slip_ratio_max_x * sqrtf(within_sqrt);
                    forces[0] = numerator / denominator;
                } else {
                    float numerator = load * post_saturation_force_x * max_force_y_at_1N;
                    float within_sqrt = powf(tanf(slip_angle), 2) +
                                        powf((max_force_y_at_1N / max_force_x_at_1N), 2);
                    float denominator = max_force_x_at_1N * sqrtf(within_sqrt);
                    forces[0] = sign(slip_ratio) * numerator / denominator;
                }

                //computes y force
                if (slip_angle < slip_angle_max_y) {
                    forces[1] = load * max_force_y_at_1N / slip_angle_max_y * slip_angle;
                } else {
                    forces[1] = load * post_saturation_force_y;
                }
            }

            //calculates slip ratio
            __host__ __device__ static float calculate_slip_ratio(float wheel_speed, float velocity) {
                if (velocity == 0)
                    return 0;

                float tangential_velo = wheel_speed * WHEEL_RADIUS;
                return (tangential_velo - velocity) / velocity;
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
            __host__ __device__ static void dynamics(const float state[], const float action[], float state_dot[]) {
                //unpackages state array
                float x_world = state[0];
                float y_world = state[1];
                float yaw_world = state[2];
                float x_dot_car = state[3];
                float y_dot_car = state[4];
                float yaw_rate = state[5];
                float pitch_moment = state[6];
                float downforce = state[7];
                float front_wheel_speed = state[8];
                float rear_wheel_speed = state[9];

                //unpackages action
                float steering_angle = action[0];
                float torque_front = action[1];
                float torque_rear = action[2];

                //compares wheel forces
                float front_slip_ratio = calculate_slip_ratio(front_wheel_speed,
                                                              x_dot_car);
                float rear_slip_ratio = calculate_slip_ratio(rear_wheel_speed,
                                                             x_dot_car);
                float y_dot_front_tire = y_dot_car + yaw_rate *CG_TO_FRONT;
                float numerator = x_dot_car * cosf(steering_angle) + (y_dot_front_tire) * sinf(steering_angle);
                float denominator = sqrtf(x_dot_car * x_dot_car + y_dot_front_tire * y_dot_front_tire);

                float front_slip_angle = denominator == 0 ?
                        0 : acosf(numerator / denominator);

                float y_dot_rear_tire = y_dot_car - yaw_rate *CG_TO_REAR;
                numerator = x_dot_car;
                denominator = sqrtf(x_dot_car * x_dot_car + y_dot_rear_tire *y_dot_rear_tire);
                float rear_slip_angle = denominator == 0 ?
                        0 : acosf(x_dot_car / denominator);

                float front_load =
                        (CAR_MASS * GRAVITY + downforce) * CG_TO_REAR / BODY_LENGTH - pitch_moment / CG_TO_FRONT;
                float rear_load =
                        (CAR_MASS * GRAVITY + downforce) * CG_TO_FRONT / BODY_LENGTH + pitch_moment / CG_TO_REAR;

                float front_forces_tire[2];  // wrt tire
                float rear_forces_tire[2];

                tireModel(front_slip_ratio, front_slip_angle, front_load, front_forces_tire);
                tireModel(rear_slip_ratio, rear_slip_angle, rear_load, rear_forces_tire);

                float front_force_x_car =
                        front_forces_tire[0] * cosf(steering_angle)
                        - front_forces_tire[1] * sinf(steering_angle);
                float front_force_y_car =
                        front_forces_tire[0] * sinf(steering_angle)
                        + front_forces_tire[1] * cosf(steering_angle);
                float rear_force_x_car = rear_forces_tire[0];
                float rear_force_y_car = rear_forces_tire[1];

                //gets drag
                float drag_x = LOGITUDNAL_AERO_CONSTANT * powf(x_dot_car, 2);
                float drag_y = LATERAL_AERO_CONSTANT * powf(y_dot_car, 2);


                //Updates dot array
                state_dot[0] =
                        x_dot_car * cosf(yaw_world) - y_dot_car * sinf(yaw_world);
                state_dot[1] =
                        x_dot_car * sinf(yaw_world) + y_dot_car * cosf(yaw_world);
                state_dot[2] = yaw_rate;
                state_dot[3] = (front_force_x_car + rear_force_x_car - drag_x) / CAR_MASS +
                               y_dot_car * yaw_rate;
                state_dot[4] = (front_force_y_car + rear_force_y_car - drag_y) / CAR_MASS -
                               x_dot_car * yaw_rate;
                state_dot[5] =
                        (CG_TO_FRONT * front_force_x_car - CG_TO_REAR * rear_force_x_car) /
                        CAR_ROTATIONAL_INTERTIA;
                state_dot[6] = 0; //might need to change
                state_dot[7] = 0;
                        //2 * drag_x * NOSE_CONE_CONSTANT * x_dot_car * state_dot[3];
                state_dot[8] =
                        (torque_front - WHEEL_RADIUS * front_forces_tire[0])
                        / WHEEL_ROTATIONAL_INTERIA;
                state_dot[9] =
                        (torque_rear - WHEEL_RADIUS * rear_forces_tire[0])
                        / WHEEL_ROTATIONAL_INTERIA;
            }
        }
    }
}
