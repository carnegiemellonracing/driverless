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

                //unpackages action
                const float angle = action[0];
                const float accel = action[1];

                state_dot[0] = x_dot_car ;
                state_dot[1] = y_dot_car;
                state_dot[2] = 0;
                state_dot[3] = cosf(angle) * accel;
                state_dot[4] = sinf(angle) * accel;
                state_dot[5] = 0;
            }

        }
    }
}