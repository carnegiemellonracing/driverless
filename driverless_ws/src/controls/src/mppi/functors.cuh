#pragma once

#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <constants.hpp>
#include <cuda_utils.cuh>
#include <cuda_constants.cuh>

#include "types.cuh"


namespace controls {
    namespace mppi {

        __device__ static SplineFrame get_interpolated_frame(float progress) {
            const float texcoord = progress / spline_frame_separation;
            const int low_idx = floorf(texcoord);
            const int high_idx = low_idx + 1;
            const float t = texcoord - low_idx;
            const SplineFrame frame_low {tex1Dfetch<float4>(cuda_globals::d_spline_texture_object, low_idx)};
            const SplineFrame frame_high {tex1Dfetch<float4>(cuda_globals::d_spline_texture_object, high_idx)};

            return {
                (1 - t) * frame_low.x + t * frame_high.x,
                (1 - t) * frame_low.y + t * frame_high.y,
                (1 - t) * frame_low.tangent_angle + t * frame_high.tangent_angle,
                (1 - t) * frame_low.curvature + t * frame_high.curvature,
            };
        }

        __device__ static void curv_state_to_world_state(float state[], SplineFrame frame) {
            const float offset = state[state_y_idx];
            const float yaw_curv = state[state_yaw_idx];

            const float2 normal {-sinf(frame.tangent_angle), cosf(frame.tangent_angle)};
            state[state_x_idx] = frame.x + offset * normal.x;
            state[state_y_idx] = frame.y + offset * normal.y;
            state[state_yaw_idx] = frame.tangent_angle + yaw_curv;
        }

        __device__ static void world_state_dot_to_curv_state_dot(float state_dot[], SplineFrame frame, float yaw_curv) {
            const float xdot = state_dot[state_x_idx];
            const float ydot = state_dot[state_y_idx];
            const float yawdot = state_dot[state_yaw_idx];

            const float s = sinf(frame.tangent_angle);
            const float c = cosf(frame.tangent_angle);

            const float prog_dot = xdot * c + ydot * s;
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
            const float yaw_curv = curv_state[state_yaw_idx];
            const SplineFrame frame = get_interpolated_frame(progress);

            // If first thread, print info
            if (__cudaGet_threadIdx().x == 0 && __cudaGet_blockIdx().x == 0) {
                printf("action: %f %f\n", action[0], action[1]);
                printf("curv state: %f %f %f %f %f %f %f %f %f %f\n", curv_state[0], curv_state[1], curv_state[2],
                       curv_state[3], curv_state[4], curv_state[5], curv_state[6], curv_state[7], curv_state[8], curv_state[9]);
                printf("Frame: %f %f %f %f\n", frame.x, frame.y, frame.tangent_angle, frame.curvature);
            }

            // create local world state vector
            float world_state[state_dims];
            memcpy(world_state, curv_state, sizeof(world_state));
            curv_state_to_world_state(world_state, frame);
            printf("world state: %f %f %f %f %f %f %f %f %f %f\n", world_state[0], world_state[1], world_state[2],
                       world_state[3], world_state[4], world_state[5], world_state[6], world_state[7], world_state[8], world_state[9]);
            assert(!any_nan(world_state, state_dims) && "World state was nan during model");


            // Call dynamics model. Outputs dstate/dt, but may take timestep into consideration for stability
            // or accuracy purposes. Extenionsally, forward euler should be done on this.

            // We do this instead of directly calculating the next step because world state dot -> curv state dot is
            // much cheaper to calculate (given current curv state) than world state -> curv state
            float world_state_dot[state_dims];
            ONLINE_DYNAMICS_FUNC(world_state, action, world_state_dot, timestep);
            printf("world state dot: %f %f %f %f %f %f %f %f %f %f\n", world_state_dot[0], world_state_dot[1], world_state_dot[2],
                       world_state_dot[3], world_state_dot[4], world_state_dot[5], world_state_dot[6], world_state_dot[7], world_state_dot[8], world_state_dot[9]);
            assert(!any_nan(world_state_dot, state_dims) && "World state dot was nan directly after dynamics call");


            world_state_dot_to_curv_state_dot(world_state_dot, frame, yaw_curv);
            assert(!any_nan(world_state_dot, state_dims) && "Curv state dot was nan after dynamics call");

            const auto& curv_state_dot = world_state_dot;
            for (uint8_t i = 0; i < state_dims; i++) {
                curv_state_out[i] = curv_state[i] + curv_state_dot[i] * timestep;
            }

            if (__cudaGet_threadIdx().x == 0 && __cudaGet_blockIdx().x == 0) {
                printf("curv state dot: %f %f %f %f %f %f %f %f %f %f\n", curv_state_dot[0], curv_state_dot[1], curv_state_dot[2],
                       curv_state_dot[3], curv_state_dot[4], curv_state_dot[5], curv_state_dot[6], curv_state_dot[7], curv_state_dot[8], curv_state_dot[9]);
            }
        }

        __device__ static float cost(float curv_state[]) {
            const float curv_yaw = curv_state[state_yaw_idx];
            const float xdot = curv_state[state_car_xdot_idx];
            const float ydot = curv_state[state_car_ydot_idx];
            const float offset = curv_state[state_y_idx];

            const float progress_dot = xdot * cosf(curv_yaw) - ydot * sin(curv_yaw);
            const float speed_cost = zero_speed_cost * expf(-speed_cost_decay_factor * progress_dot);

            const float distance_cost = offset_1m_cost * offset * offset;

            return speed_cost + distance_cost;
        }

        // Functors for Brownian Generation
        struct TransformStdNormal {
            thrust::device_ptr<float> std_normals;

            explicit TransformStdNormal(thrust::device_ptr<float> std_normals)
                    : std_normals {std_normals} { }

            __device__ void operator() (size_t idx) const {
                const size_t action_idx = (idx / action_dims) * action_dims;
                const size_t action_dim = idx % action_dims;
                const size_t row_idx = action_dim * action_dims;

                const auto res = dot<float>(
                    &cuda_globals::perturbs_incr_std[row_idx],
                    &std_normals.get()[action_idx],
                    action_dims);

                std_normals.get()[idx] = clamp(
                    res * m_sqrt_timestep,
                    cuda_globals::action_deriv_min[action_dim] * controller_period,
                    cuda_globals::action_deriv_max[action_dim] * controller_period);
            }

            private:
                float m_sqrt_timestep = std::sqrt(controller_period);  // sqrt seconds
        };

        // Functors for cost calculation

        // Gets us the costs to go
        struct PopulateCost {
            float* brownians;
            float* sampled_action_trajectories;
#ifdef PUBLISH_STATES
            float* sampled_state_trajectories;
#endif
            float* cost_to_gos;

            const DeviceAction* action_trajectory_base;

            PopulateCost(thrust::device_ptr<float> brownians,
                         thrust::device_ptr<float> sampled_action_trajectories,
#ifdef PUBLISH_STATES
                         thrust::device_ptr<float> sampled_state_trajectories,
#endif
                         thrust::device_ptr<float> cost_to_gos,
                         const thrust::device_ptr<DeviceAction>& action_trajectory_base)
                    : brownians {brownians.get()},
                      sampled_action_trajectories {sampled_action_trajectories.get()},
#ifdef PUBLISH_STATES
                      sampled_state_trajectories {sampled_state_trajectories.get()},
#endif
                      cost_to_gos {cost_to_gos.get()},
                      action_trajectory_base {action_trajectory_base.get()} {}

            __device__ void operator() (uint32_t i) const {
                float j_curr = 0;
                float x_curr[state_dims];

                // printf("curv state: %f %f %f %f %f %f %f %f %f %f\n", curr_state[0], curr_state[1], curr_state[2],
                //        curr_state[3], curr_state[4], curr_state[5], curr_state[6], curr_state[7], curr_state[8], curr_state[9]);
                assert(!any_nan(cuda_globals::curr_state, state_dims) && "State was nan in populate cost entry");

                // printf("POPLATE COST %i: copying curr_state", i);
                // copy current state into x_curr
                memcpy(x_curr, cuda_globals::curr_state, sizeof(float) * state_dims);

                // for each timestep, calculate cost and add to get cost to go
                for (uint32_t j = 0; j < num_timesteps; j++) {
                    float* u_ij = IDX_3D(sampled_action_trajectories, dim3(num_samples, num_timesteps, action_dims), dim3(i, j, 0));

                    // VERY CURSED. We want the last action in the best guess action to be the same as the second
                    // to last one (since we have to initialize it something). Taking the min of j and m - 1 saves
                    // us a host->device copy
                    const uint32_t idx = min(j, num_timesteps - 2);
                    assert(!any_nan(action_trajectory_base[idx].data, action_dims) && "Control action base was nan");

                    for (uint32_t k = 0; k < action_dims; k++) {
                        u_ij[k] = action_trajectory_base[idx].data[k]
                                  + *IDX_3D(brownians, dim3(num_samples, num_timesteps, action_dims), dim3(i, j, k));
                        u_ij[k] = clamp(
                            u_ij[k],
                            cuda_globals::action_min[k],
                            cuda_globals::action_max[k]
                        );
                    }

                    // printf("control action: %f, %f, %f\n", u_ij[0], u_ij[1], u_ij[2]);
                    assert(!any_nan(u_ij, action_dims) && "Control was nan before model step");

                    if (__cudaGet_blockIdx().x == 0 && __cudaGet_threadIdx().x == 0) {
                        printf("j: %i\n", j);
                    }
                    model(x_curr, u_ij, x_curr, controller_period);

                    assert(!any_nan(x_curr, state_dims) && "State was nan after model step");

#ifdef PUBLISH_STATES
                    float* world_state = IDX_3D(
                        sampled_state_trajectories,
                        dim3(num_samples, num_timesteps, state_dims),
                        dim3(i, j, 0)
                    );
                    memcpy(world_state, x_curr, sizeof(float) * state_dims);
                    curv_state_to_world_state(world_state, get_interpolated_frame(x_curr[state_x_idx]));
#endif

                    const float c = cost(x_curr);
                    j_curr -= c;
                    cost_to_gos[i * num_timesteps + j] = j_curr;
                }
            }
        };

        // Functors to operate on Action

        struct AddActions {
            __device__ DeviceAction operator() (const DeviceAction& a1, const DeviceAction& a2) const {
                DeviceAction res;
                for (uint8_t i = 0; i < action_dims; i++) {
                    res.data[i] = a1.data[i] + a2.data[i];
                }
                return res;
            }
        };

        template<size_t k>
        struct DivBy {
            __device__ size_t operator() (size_t i) const {
                return i / k;
            }
        };

        template<typename T>
        struct Equal {
            __device__ bool operator() (T a, T b) {
                return a == b;
            }
        };


        // Functors for action reduction
        struct ActionAverage {
            __device__ ActionWeightTuple operator()(const ActionWeightTuple& action_weight_t0,
                                                    const ActionWeightTuple& action_weight_t1) const
            {
                const float w0 = action_weight_t0.weight;
                const float w1 = action_weight_t1.weight;
                const DeviceAction& a0 = action_weight_t0.action;
                const DeviceAction& a1 = action_weight_t1.action;

                return {
                    w0 + w1 == 0 ?
                        DeviceAction {} : (w0 * a0 + w1 * a1) / (w0 + w1),
                    w0 + w1
                };
            }
        };

        /** 
         * @brief Captures pointers to action trajectories and cost to gos
         * Produces a weight for a single action based on its cost to go
        */
        struct IndexToActionWeightTuple {
            const float* action_trajectories;
            const float* cost_to_gos;

            __device__ IndexToActionWeightTuple (const float* action_trajectories, const float* cost_to_gos)
                    : action_trajectories {action_trajectories},
                      cost_to_gos {cost_to_gos} {}

            /**
             * \param idx refers to the index for the timesteps x samples matrix 
             * (transposed from normal to make the reduction work)
             * \return a tuple of the action indexed from the trajectories 
             * and the weight of that action (strictly positive) based on its cost to go*/            
            __device__ ActionWeightTuple operator() (const uint32_t idx) const {
                ActionWeightTuple res {};
                // as we increment idx, we iterate over the samples first
                const uint32_t i = idx % num_samples; // sample
                // j should hold constant over a single reduction, it represents which timestep we are reducing over
                const uint32_t j = idx / num_samples;  // timestep
                memcpy(&res.action.data, IDX_3D(action_trajectories,
                                                action_trajectories_dims,
                                                dim3(i, j, 0)), sizeof(float) * action_dims);

                assert(cuda_globals::action_min[0] <= res.action.data[0]);
                assert(cuda_globals::action_max[0] >= res.action.data[0]);
                assert(cuda_globals::action_min[1] <= res.action.data[1]);
                assert(cuda_globals::action_max[1] >= res.action.data[1]);

                // right now cost to gos is shifted down by the value in the last timestep, so adjust for that
                const float anthony_adjustment = cost_to_gos[(i + 1) * num_timesteps - 2] - 2 * cost_to_gos[(i + 1) * num_timesteps - 1];
                const float cost_to_go = cost_to_gos[i * num_timesteps + j] + anthony_adjustment;
                res.weight = __expf(-1.0f / temperature * cost_to_go);

                return res;
            }
        };

        struct ReduceTimestep {
            DeviceAction* averaged_action;
            DeviceAction* action_trajectory_base;

            const float* action_trajectories;
            const float* cost_to_gos;

            const ActionAverage reduction_functor {};

            ReduceTimestep (DeviceAction* averaged_action, DeviceAction* action_trajectory_base, const float* action_trajectories, const float* cost_to_gos)
                    : averaged_action {averaged_action},
                      action_trajectory_base {action_trajectory_base},
                      action_trajectories {action_trajectories},
                      cost_to_gos {cost_to_gos} { }

            __device__ void operator() (const uint32_t idx) {
                // idx ranges from 0 to num_timesteps - 1
                // idx represents a timestep
                thrust::counting_iterator<uint32_t> indices {idx * num_samples};

                // applies a unary operator - IndexToActionWeightTuple before reducing over the samples with reduction_functor
                const ActionWeightTuple res = thrust::transform_reduce(
                        thrust::device,
                        indices, indices + num_samples,
                        IndexToActionWeightTuple {action_trajectories, cost_to_gos},
                        ActionWeightTuple { DeviceAction {}, 0 }, reduction_functor);

                if (res.weight == 0) {
                    printf("Action at timestep %i had 0 weight. Using previous.\n", idx);
                    averaged_action[idx] = action_trajectory_base[min(idx, num_timesteps - 2)];
                } else {
                    averaged_action[idx] = res.action;
                }
            }
        };
    }
}

