#pragma once

#include <thrust/device_ptr.h>
#include <types.hpp>

#include "cuda_constants.cuh"
#include "types.cuh"
#include "mppi.hpp"


namespace controls {
    namespace mppi {

        __device__ static void curv_state_to_world_state(float state[], SplineFrame frame) {
            const float offset = state[state_y_idx];
            const float yaw_curv = state[state_yaw_idx];

            const float2 normal {-sinf(frame.tangent_angle), cosf(frame.tangent_angle)};
            state[state_x_idx] = frame.x + offset * normal.x;
            state[state_y_idx] = frame.y + offset * normal.y;
            state[state_yaw_idx] = frame.tangent_angle + yaw_curv;
        }

        __device__ static void world_state_dot_to_curv_state_dot(float state_dot[], SplineFrame frame) {
            const float xdot = state_dot[state_x_idx];
            const float ydot = state_dot[state_y_idx];
            const float yawdot = state_dot[state_yaw_idx];

            const float s = sinf(frame.tangent_angle);
            const float c = cosf(frame.tangent_angle);

            const float prog_dot = xdot * c+ ydot * s;
            const float curv_y_dot = -xdot * s + ydot * c;
            const float curv_yaw_dot = -frame.curvature * prog_dot + yawdot;

            state_dot[state_x_idx] = prog_dot;
            state_dot[state_y_idx] = curv_y_dot;
            state_dot[state_yaw_idx] = curv_yaw_dot;
        }

        /**
         * \brief Advances curvilinear state one timestep according to action. `curv_state` and `action` may be
         *        invalid after call.
         *
         * \param[in] curv_state Curvilinear state of the vehicle
         * \param[in] action Action taken
         * \param[out] curv_state_out Returned next state
         * \param[in] timestep Timestep
         */
        __device__ static void model(const float curv_state[], const float action[], float curv_state_out[], float timestep) {
            const float progress = curv_state[state_x_idx];
            const float tex_coord = progress / spline_frame_separation;
            const SplineFrame frame {tex1D<float4>(cuda_globals::d_spline_texture_object, tex_coord)};

            float world_state[state_dims];
            memcpy(world_state, curv_state, sizeof(world_state));
            curv_state_to_world_state(world_state, frame);

            float world_state_dot[state_dims];
            ONLINE_DYNAMICS_FUNC(world_state, action, world_state_dot);

            world_state_dot_to_curv_state_dot(world_state_dot, frame);
            const auto& curv_state_dot = world_state_dot;

            for (uint8_t i = 0; i < state_dims; i++) {
                curv_state_out[i] = curv_state_dot[i] * timestep;  // TODO: Euler's method, make better
            }
        }

        __device__ static float cost(float state[]) {
            // TODO: make real cost function

            // placeholder: sum the vector of state
            float sum = 0;
            for (size_t i = 0; i < state_dims; i++) {
                sum += state[i];
            }
            return sum;
        }

        class MppiController_Impl : public MppiController {
        public:
            MppiController_Impl();

            Action generate_action() override;

            ~MppiController_Impl() override;

        private:
            /**
             * num_samples x num_timesteps x actions_dims device tensor. Used to store action brownians,
             * perturbations, and action trajectories at different points in the algorithm.
             */
            thrust::device_ptr<float> m_action_trajectories;

            /**
             * num_samples x num_timesteps array of costs to go. Used for action weighting.
             */
            thrust::device_ptr<float> m_cost_to_gos;

            /**
             * num_timesteps x action_dims array. Best-guess action trajectory to which perturbations are added.
             */
            thrust::device_ptr<float> m_last_action_trajectory;


            void generate_brownians();

            /**
             * @brief Retrieves action based on cost to go using reduction.
             * @return Action
             */
            DeviceAction reduce_actions();

            /**
             * @brief Calculates costs to go
             */
            void populate_cost();

        };




    }
}