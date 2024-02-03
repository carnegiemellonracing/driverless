#pragma once

#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <types.hpp>
#include <constants.hpp>

#include "types.cuh"


namespace controls {
    namespace mppi {
        // Functors for Brownian Generation
        struct TransformStdNormal {
            thrust::device_ptr<float> std_normals;

            explicit TransformStdNormal(thrust::device_ptr<float> std_normals)
                    : std_normals {std_normals} { }

            __host__ __device__ void operator() (size_t idx) const {
                const size_t action_idx = (idx / action_dims) * action_dims;
                const size_t row_idx = idx % action_dims * action_dims;

                const float res = dot<float>(&perturbs_incr_std[row_idx], &std_normals.get()[action_idx], action_dims);
                std_normals[idx] = res * sqrt_timestep;
            }
        };

        // Functors for cost calculation

        // Gets us the costs to go
        struct PopulateCost {
            float* brownians;
            float* sampled_action_trajectories;
            float* cost_to_gos;

            const float* action_trajectory_base;
            const float* curr_state;

            PopulateCost(thrust::device_ptr<float>& brownians,
                         thrust::device_ptr<float>& sampled_action_trajectories,
                         thrust::device_ptr<float>& cost_to_gos,
                         const thrust::device_ptr<float>& action_trajectory_base,
                         const thrust::device_ptr<float>& curr_state)
                    : brownians {brownians.get()},
                      sampled_action_trajectories {sampled_action_trajectories.data().get()},
                      cost_to_gos {cost_to_gos.data().get()},
                      action_trajectory_base {action_trajectory_base.data().get()},
                      curr_state {curr_state.data().get()} {}

            __device__ void operator() (uint32_t i) const {
                float j_curr = 0;
                float x_curr[state_dims];

                // copy current state into x_curr
                memcpy(x_curr, curr_state, sizeof(float) * state_dims);

                // for each timestep, calculate cost and add to get cost to go
                for (uint32_t j = 0; j < num_timesteps; j++) {
                    float* u_ij = IDX_3D(sampled_action_trajectories, dim3(num_samples, num_timesteps, action_dims), dim3(i, j, 0));

                    for (uint32_t k = 0; k < action_dims; k++) {
                        const uint32_t idx = j * action_dims + k;
                        u_ij[k] = action_trajectory_base[idx]
                                  + *IDX_3D(brownians, dim3(num_samples, num_timesteps, action_dims), dim3(i, j, k));
                    }

                    model(x_curr, u_ij, x_curr, controller_period_ms); // Euler's method, TODO make better;

                    j_curr -= cost(x_curr);
                    cost_to_gos[i * num_timesteps + j] = j_curr;
                }
            }
        };


        // Functors for action reduction
        struct ActionAverage {
            __device__ ActionWeightTuple operator()(const ActionWeightTuple& action_weight_t0,
                                                    const ActionWeightTuple& action_weight_t1) const
            {
                const float w0 = action_weight_t0.weight;
                const float w1 = action_weight_t1.weight;
                const Action a0 = action_weight_t0.action;
                const Action a1 = action_weight_t1.action;

                return {(w0 * a0 + w1 * a1) / (w0 + w1), w0 + w1};
            }
        };

        struct IndexToActionWeightTuple {
            const float* action_trajectories;
            const float* cost_to_gos;

            __device__ IndexToActionWeightTuple (const float* action_trajectories, const float* cost_to_gos)
                    : action_trajectories {action_trajectories},
                      cost_to_gos {cost_to_gos} {}

            __device__ ActionWeightTuple operator() (const uint32_t idx) const {
                ActionWeightTuple res {};

                const uint32_t i = idx % num_samples;
                const uint32_t j = idx / num_samples;
                memcpy(&res.action.data, IDX_3D(action_trajectories,
                                                action_trajectories_dims,
                                                dim3(i, j, 0)), sizeof(float) * action_dims);

                const float cost_to_go = cost_to_gos[idx];
                res.weight = __expf(-1.0f / temperature * cost_to_go);

                return res;
            }
        };

        struct ReduceTimestep {
            Action* averaged_actions;

            const float* action_trajectories;
            const float* cost_to_gos;

            const ActionAverage reduction_functor {};

            ReduceTimestep (Action* averaged_actions, const float* action_trajectories, const float* cost_to_gos)
                    : averaged_actions {averaged_actions},
                      action_trajectories {action_trajectories},
                      cost_to_gos {cost_to_gos} { }

            __device__ void operator() (const uint32_t idx) {
                thrust::counting_iterator<uint32_t> indices {idx * num_samples};

                averaged_actions[idx] = thrust::transform_reduce(
                        thrust::device,
                        indices, indices + num_samples,
                        IndexToActionWeightTuple {action_trajectories, cost_to_gos},
                        ActionWeightTuple { Action {}, 0 }, reduction_functor).action;
            }
        };
    }
}

