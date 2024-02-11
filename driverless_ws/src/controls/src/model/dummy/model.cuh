namespace controls {
    namespace model {
        namespace dummy {

            __host__ __device__ static void dynamics(const float state[], const float action[], float state_dot[]) {
                //unpackages state array
                const float x_world = state[0];
                const float y_world = state[1];
                const float yaw_world = state[2];
                const float x_dot_car = state[3];
                const float y_dot_car = state[4];
                const float yaw_rate = state[5];
                const float pitch_moment = state[6];
                const float downforce = state[7];
                const float front_wheel_speed = state[8];
                const float rear_wheel_speed = state[9];

                //unpackages action
                const float steering_angle = action[0];
                const float torque_front = action[1];
                const float torque_rear = action[2];

                state_dot[0] =
                        x_dot_car * cosf(yaw_world) - y_dot_car * sinf(yaw_world);
                state_dot[1] =
                        x_dot_car * sinf(yaw_world) + y_dot_car * cosf(yaw_world);
                state_dot[2] = yaw_rate;
                state_dot[3] = cosf(steering_angle) * torque_front + torque_rear;
                state_dot[4] = sinf(steering_angle) * torque_front;
                state_dot[5] = sinf(steering_angle) * torque_front;
                state_dot[6] = 0;
                state_dot[7] = 0;
                state_dot[8] = 0;
                state_dot[9] = 0;
            }

        }
    }
}