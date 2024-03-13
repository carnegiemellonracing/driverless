#pragma once

#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <constants.hpp>
#include <utils/cuda_utils.cuh>
#include <cuda_constants.cuh>
#include <cuda_globals/helpers.cuh>

#include "types.cuh"


namespace controls {
    namespace mppi {

        /**
         * Advances curvilinear state one timestep according to action.
         *
         * @param[in] world_state world state of the vehicle
         * @param[in] action Action taken
         * @param[out] world_state_out Returned next state
         * @param[in] timestep Timestep
         */
        __device__ static void model(const float world_state[], const float action[], float world_state_out[], float timestep) {
            // Call dynamics model. Outputs dstate/dt, but may take timestep into consideration for stability
            // or accuracy purposes. Extenionsally, forward euler should be done on this.

            // We do this instead of directly calculating the next step because updating directly didn't work in
            // curvilinear coordinates. TODO: refactor model to directly update next state
            float world_state_dot[state_dims];
            ONLINE_DYNAMICS_FUNC(world_state, action, world_state_dot, timestep);
            paranoid_assert(!any_nan(world_state_dot, state_dims) && "World state dot was nan directly after dynamics call");
            paranoid_assert(!any_nan(world_state, state_dims) && "World state was nan directly after dynamics call");

            for (uint8_t i = 0; i < state_dims; i++) {
                world_state_out[i] = world_state[i] + world_state_dot[i] * timestep;
            }
            paranoid_assert(!any_nan(world_state_out, state_dims) && "World state out was nan directly after dynamics call");
        }

        /**
         * Calculate cost at a particular state. Potentially divergent, so __syncthreads() after calling if needed.
         *
         * @param world_state World state of the vehicle
         * @param start_progress Progress at the start of the trajectory
         * @param time_since_traj_start Time elapsed since trjaectory start
         * @returns Cost at the given state
         */
        __device__ static float cost(float world_state[], float start_progress, float time_since_traj_start) {
            float curv_pose[3];
            bool out_out_bounds;
            cuda_globals::sample_curv_state(world_state, curv_pose, out_out_bounds);

            if (out_out_bounds) {
                return std::numeric_limits<float>::infinity();
            }

            const float progress = curv_pose[0];
            const float offset = curv_pose[1];

            const float approx_speed_along = (progress - start_progress) / time_since_traj_start;
            const float speed_deviation = target_speed - approx_speed_along;
            const float speed_cost = speed_weight * abs(speed_deviation);

            const float distance_cost = offset_1m_cost * offset * offset;

            return speed_cost + distance_cost;
        }

        // Functors for Brownian Generation
        struct TransformStdNormal {
            thrust::device_ptr<float> std_normals;

            explicit TransformStdNormal(thrust::device_ptr<float> std_normals)
                    : std_normals {std_normals} { }

            __device__ void operator() (size_t idx) const {
                const size_t action_idx = (idx / action_dims) * action_dims; // index into std_normals for beginning of action
                const size_t action_dim = idx % action_dims;
                const size_t row_idx = action_dim * action_dims;  // index into perturbs_incr_std for beginning of row

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
#ifdef DISPLAY
            float* sampled_state_trajectories;
#endif
            float* cost_to_gos;

            const DeviceAction* action_trajectory_base;

            PopulateCost(thrust::device_ptr<float> brownians,
                         thrust::device_ptr<float> sampled_action_trajectories,
#ifdef DISPLAY
                         thrust::device_ptr<float> sampled_state_trajectories,
#endif
                         thrust::device_ptr<float> cost_to_gos,
                         const thrust::device_ptr<DeviceAction>& action_trajectory_base)
                    : brownians {brownians.get()},
                      sampled_action_trajectories {sampled_action_trajectories.get()},
#ifdef DISPLAY
                      sampled_state_trajectories {sampled_state_trajectories.get()},
#endif
                      cost_to_gos {cost_to_gos.get()},
                      action_trajectory_base {action_trajectory_base.get()} {}

            __device__ void operator() (uint32_t i) const {
                float j_curr = 0;
                float x_curr[state_dims];

                paranoid_assert(!any_nan(cuda_globals::curr_state, state_dims) && "State was nan in populate cost entry");

                // copy current state into x_curr
                memcpy(x_curr, cuda_globals::curr_state, sizeof(float) * state_dims);

                float init_curv_pose[3];
                bool out_of_bounds;
                cuda_globals::sample_curv_state(x_curr, init_curv_pose, out_of_bounds);
                paranoid_assert(!out_of_bounds && "Initial state was out of bounds");
                __syncthreads();

                // for each timestep, calculate cost and add to get cost to go
                for (uint32_t j = 0; j < num_timesteps; j++) {
                    float* u_ij = IDX_3D(sampled_action_trajectories, dim3(num_samples, num_timesteps, action_dims), dim3(i, j, 0));

                    // VERY CURSED. We want the last action in the best guess action to be the same as the second
                    // to last one (since we have to initialize it something). Taking the min of j and m - 1 saves
                    // us a host->device copy
                    const uint32_t idx = min(j, num_timesteps - 2);
                    paranoid_assert(!any_nan(action_trajectory_base[idx].data, action_dims) && "Control action base was nan");

                    for (uint32_t k = 0; k < action_dims; k++) {
                        u_ij[k] = action_trajectory_base[idx].data[k]
                                  + *IDX_3D(brownians, dim3(num_samples, num_timesteps, action_dims), dim3(i, j, k));
                        u_ij[k] = clamp(
                            u_ij[k],
                            cuda_globals::action_min[k],
                            cuda_globals::action_max[k]
                        );
                    }

                    paranoid_assert(!any_nan(u_ij, action_dims) && "Control was nan before model step");
                    model(x_curr, u_ij, x_curr, controller_period);
                    paranoid_assert(!any_nan(x_curr, state_dims) && "State was nan after model step");
                    __syncthreads();

#ifdef DISPLAY
                    float* world_state = IDX_3D(
                        sampled_state_trajectories,
                        dim3(num_samples, num_timesteps, state_dims),
                        dim3(i, j, 0)
                    );
                    memcpy(world_state, x_curr, sizeof(float) * state_dims);
#endif

                    const float c = cost(x_curr, init_curv_pose[0], controller_period * (j + 1));
                    __syncthreads();

                    j_curr -= c;
                    cost_to_gos[i * num_timesteps + j] = j_curr;
                    paranoid_assert(!isnan(c) && "cost-to-go was nan");
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
         * Captures pointers to action trajectories and cost to gos
         * Produces a weight for a single action based on its cost to go
        */
        struct IndexToActionWeightTuple {
            const float* action_trajectories;
            const float* cost_to_gos;

            __device__ IndexToActionWeightTuple (const float* action_trajectories, const float* cost_to_gos)
                    : action_trajectories {action_trajectories},
                      cost_to_gos {cost_to_gos} {}

            /**
             * @param idx refers to the index for the timesteps x samples matrix
             * (transposed from normal to make the reduction work)
             * @returns a tuple of the action indexed from the trajectories
             * and the weight of that action (strictly positive) based on its cost to go
             */
            __device__ ActionWeightTuple operator() (const uint32_t idx) const {
                ActionWeightTuple res {};
                // as we increment idx, we iterate over the samples first
                const uint32_t i = idx % num_samples; // sample
                // j should hold constant over a single reduction, it represents which timestep we are reducing over
                const uint32_t j = idx / num_samples;  // timestep
                memcpy(&res.action.data, IDX_3D(action_trajectories,
                                                action_trajectories_dims,
                                                dim3(i, j, 0)), sizeof(float) * action_dims);

                paranoid_assert(cuda_globals::action_min[0] <= res.action.data[0]);
                paranoid_assert(cuda_globals::action_max[0] >= res.action.data[0]);
                paranoid_assert(cuda_globals::action_min[1] <= res.action.data[1]);
                paranoid_assert(cuda_globals::action_max[1] >= res.action.data[1]);

                // right now cost to gos is shifted down by the value in the last timestep, so adjust for that
                const float final_step = cost_to_gos[(i + 1) * num_timesteps - 1];
                const float penult_step = cost_to_gos[(i + 1) * num_timesteps - 2];

                float cost_to_go;
                if (isinf(final_step)) {
                    cost_to_go = std::numeric_limits<float>::infinity();
                } else {
                    const float anthony_adjustment = penult_step - 2 * final_step;
                    cost_to_go = cost_to_gos[i * num_timesteps + j] + anthony_adjustment;
                }
                paranoid_assert(!isnan(cost_to_go) && "cost-to-go was nan in tuple generation");
                __syncthreads();

                res.weight = expf(-1.0f / temperature * cost_to_go);
                paranoid_assert(!isnan(res.weight) && "weight was nan");

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
                paranoid_assert(!any_nan(averaged_action[idx].data, action_dims) && "Averaged action was nan");
            }
        };
    }
}

