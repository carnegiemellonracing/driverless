#pragma once

#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <constants.hpp>

#include "cuda_constants.cuh"
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

                const auto res = dot<float>(&cuda_globals::perturbs_incr_std[row_idx], &std_normals.get()[action_idx],
                                            action_dims);

                std_normals.get()[idx] = res * m_sqrt_timestep;
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

            const float* action_trajectory_base;
            const float* curr_state;

            PopulateCost(thrust::device_ptr<float> brownians,
                         thrust::device_ptr<float> sampled_action_trajectories,
#ifdef PUBLISH_STATES
                         thrust::device_ptr<float> sampled_state_trajectories,
#endif
                         thrust::device_ptr<float> cost_to_gos,
                         const thrust::device_ptr<float>& action_trajectory_base,
                         const thrust::device_ptr<float>& curr_state)
                    : brownians {brownians.get()},
                      sampled_action_trajectories {sampled_action_trajectories.get()},
#ifdef PUBLISH_STATES
                      sampled_state_trajectories {sampled_state_trajectories.get()},
#endif
                      cost_to_gos {cost_to_gos.get()},
                      action_trajectory_base {action_trajectory_base.get()},
                      curr_state {curr_state.get()} {}

            __device__ void operator() (uint32_t i) const {
                float j_curr = 0;
                float x_curr[state_dims];

                // printf("POPLATE COST %i: copying curr_state", i);
                // copy current state into x_curr
                memcpy(x_curr, curr_state, sizeof(float) * state_dims);

                // for each timestep, calculate cost and add to get cost to go
                for (uint32_t j = 0; j < num_timesteps; j++) {
                    float* u_ij = IDX_3D(sampled_action_trajectories, dim3(num_samples, num_timesteps, action_dims), dim3(i, j, 0));

                    for (uint32_t k = 0; k < action_dims; k++) {

                        // VERY CURSED. We want the last action in the best guess action to be the same as the second
                        // to last one (since we have to initialize it something). Taking the min of j and m - 1 saves
                        // us a host->device copy
                        const uint32_t idx = min(j, num_timesteps - 1) * action_dims + k;

                        u_ij[k] = action_trajectory_base[idx]
                                  + *IDX_3D(brownians, dim3(num_samples, num_timesteps, action_dims), dim3(i, j, k));
                    }

                    model(x_curr, u_ij, x_curr, controller_period);
#ifdef PUBLISH_STATES
                    memcpy(
                        IDX_3D(
                            sampled_state_trajectories,
                            dim3(num_samples, num_timesteps, state_dims),
                            dim3(i, j, 0)
                        ),
                        x_curr,
                        sizeof(float) * state_dims
                    );
#endif

                    const float c = cost(x_curr);
                    j_curr -= c;
                    // printf("sample: %i\ntime: %f state: { x: %f, y: %f, yaw: %f, xdot: %f, ydot: %f, yawdot: %f }\ncost: %f\n\n",
                    //     i, j * controller_period, x_curr[0],  x_curr[1], x_curr[2], x_curr[3], x_curr[4], x_curr[5], c);
                    cost_to_gos[i * num_timesteps + j] = j_curr;
                }
            }
        };

        // Functors to operate on Action

        struct AddActions {
            __host__ __device__ DeviceAction operator() (const DeviceAction& a1, const DeviceAction& a2) const {
                return a1 + a2;
            }
        };

        template<size_t k>
        struct DivBy {
            __host__ __device__ size_t operator() (size_t i) const {
                return i / k;
            }
        };

        template<typename T>
        struct Equal {
            __host__ __device__ bool operator() (T a, T b) {
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

                // right now cost to gos is shifted down by the value in the last timestep, so adjust for that
                const float cost_to_go = cost_to_gos[i * num_timesteps + j] - cost_to_gos[(i + 1) * num_timesteps - 1];
                res.weight = __expf(-1.0f / temperature * cost_to_go);

                return res;
            }
        };

        struct ReduceTimestep {
            DeviceAction* averaged_action;

            const float* action_trajectories;
            const float* cost_to_gos;

            const ActionAverage reduction_functor {};

            ReduceTimestep (DeviceAction* averaged_action, const float* action_trajectories, const float* cost_to_gos)
                    : averaged_action {averaged_action},
                      action_trajectories {action_trajectories},
                      cost_to_gos {cost_to_gos} { }

            __device__ void operator() (const uint32_t idx) {
                thrust::counting_iterator<uint32_t> indices {idx * num_samples};

                averaged_action[idx] = thrust::transform_reduce(
                        thrust::device,
                        indices, indices + num_samples,
                        IndexToActionWeightTuple {action_trajectories, cost_to_gos},
                        ActionWeightTuple { DeviceAction {}, 0 }, reduction_functor).action;
            }
        };
    }
}

