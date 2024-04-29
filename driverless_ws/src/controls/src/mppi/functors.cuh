#pragma once

#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <constants.hpp>
#include <utils/cuda_utils.cuh>
#include <cuda_constants.cuh>
#include <cuda_globals/helpers.cuh>
#include <math_constants.h>
#include <model/slipless/model.cuh>

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
            // used to be not a trivial function (lol). Probably optimized out anyway.
            ONLINE_DYNAMICS_FUNC(world_state, action, world_state_out, timestep);
            paranoid_assert(!any_nan(world_state_out, state_dims) && "World state out was nan directly after dynamics call");
        }

        /**
         * Calculate cost at a particular state.
         *
         * @param world_state World state of the vehicle
         * @param action Action considered
         * @param last_taken_action Last action actually taken by the vehicle
         * @param start_progress Progress at the start of the trajectory
         * @param time_since_traj_start Time elapsed since trjaectory start
         * @param first whether this is the first action in the trajectory
         * @returns Cost at the given state
         */
        __device__ static float cost(
            const float world_state[], const float action[], const float last_taken_action[],
            float start_progress, float time_since_traj_start, bool first) {

            float nose_curv_pose[3];
            float cent_curv_pose[3];
            bool nose_out_of_bounds;
            bool cent_out_of_bounds;

            float forward_x = cosf(world_state[state_yaw_idx]) * cg_to_nose;
            float forward_y = sinf(world_state[state_yaw_idx]) * cg_to_nose;
            float nose_pose[3] = {world_state[state_x_idx] + forward_x, world_state[state_y_idx] + forward_y, world_state[state_yaw_idx]};
            cuda_globals::sample_curv_state(nose_pose, nose_curv_pose, nose_out_of_bounds);
            cuda_globals::sample_curv_state(world_state, cent_curv_pose, cent_out_of_bounds);

            const float centripedal_accel = model::slipless::centripedal_accel(world_state[state_speed_idx], action[action_swangle_idx]);
            const float abs_centripedal_accel = fabsf(centripedal_accel);

            if (nose_out_of_bounds || cent_out_of_bounds
             || abs_centripedal_accel > lat_tractive_capability) {
                return out_of_bounds_cost;
            }

            const float progress = cent_curv_pose[0];

            const float approx_speed_along = (progress - start_progress) / time_since_traj_start;
            const float speed_deviation = approx_speed_along - target_speed;
            const float speed_cost = speed_off_1mps_cost * fmaxf(-speed_deviation, 0.0f);

            const float distance_cost = offset_1m_cost * fmax(
                fabsf(nose_curv_pose[state_y_idx]), fabsf(cent_curv_pose[state_y_idx])
            );

            // const float deriv_cost = first ?
            //     fabsf(action[action_torque_idx] - last_taken_action[action_torque_idx]) / controller_period / 10 * torque_10Nps_cost
            //   + fabsf(action[action_swangle_idx] - last_taken_action[action_swangle_idx]) / controller_period * swangle_1radps_cost
            //   : 0;

            return speed_cost + distance_cost;// + fabsf(action[action_torque_idx]) * 0.05f;// + deriv_cost;
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


        /**@brief
         *
         * @Author:Ayush Garg and Anthony Yip
         *
         * @params perturbation n x m x q disturbance matrix
         * logProabilityDensities n x m matrix to store result
         *
         * globals used:
         * magic_matrix q x q covariance matrix (inversed then transposed)
         * TRANSPOSED FOR EFFCIENECY AND LOCALITY
         * magic_constant ugly thing with square roots and determinant of covariance
         * TODO: Implement magic_matrix, magic_constant
         */
        struct LogProbabilityDensity {
            thrust::device_ptr<float> perturbation;
            thrust::device_ptr<float> log_probability_densities;


            explicit LogProbabilityDensity(thrust::device_ptr<float> perturbation,
            thrust::device_ptr<float> log_probability_densities)
                    : perturbation {perturbation},
                      log_probability_densities {log_probability_densities} { }

            __device__ void operator() (size_t idx) const {
                float perturb[action_dims];
                memcpy(&perturb, &perturbation.get()[idx * action_dims], sizeof(float) * action_dims);

                float res = 0;
                for(uint8_t i= 0; i < action_dims; i++){

                    //Dot product of pertubations and ith column
                    const float intermediate =
                        dot<float>( perturb, &cuda_globals::perturbs_incr_var_inv[i*action_dims], action_dims);

                    //part i of dot product of final dot product
                    res += intermediate * perturbation.get()[idx * action_dims +i] * controller_period;
                }

                //actually set the log probability distribution
                log_probability_densities.get()[idx] = -.5*res;
            }

        };

        // Functors for cost calculation

        /** @brief Computes cost using the model and cost function, triangularizes both D and J
         *
         * @params brownians[in] brownian perturbation with mean 0
         * sampled_action_trajectories[in] same as brownians?
         * sampled_state_trajectories[in] for publishing states
         * cost_to_gos[out] stores the negative cost so far
         * log_prob_densities[in/out] stores the negative log_prob_densities so far
         * action_trajectory_base[in]curr
         * i[in] index over timesteps
         *
         */ //TODO: why do we need both brownians and sampled_action_trajectories?
        struct PopulateCost {
            float* brownians;
            float* sampled_action_trajectories;
#ifdef DISPLAY
            float* sampled_state_trajectories;
#endif
            float* cost_to_gos;
            float* log_prob_densities;
            DeviceAction last_taken_action;

            const DeviceAction* action_trajectory_base;

            PopulateCost(thrust::device_ptr<float> brownians,
                         thrust::device_ptr<float> sampled_action_trajectories,
#ifdef DISPLAY
                         thrust::device_ptr<float> sampled_state_trajectories,
#endif
                         thrust::device_ptr<float> cost_to_gos,
                         thrust::device_ptr<float> log_prob_densities,
                         const thrust::device_ptr<DeviceAction>& action_trajectory_base,
                         DeviceAction last_taken_action)
                    : brownians {brownians.get()},
                      sampled_action_trajectories {sampled_action_trajectories.get()},
#ifdef DISPLAY
                      sampled_state_trajectories {sampled_state_trajectories.get()},
#endif
                      cost_to_gos {cost_to_gos.get()},
                      log_prob_densities {log_prob_densities.get()},
                      action_trajectory_base {action_trajectory_base.get()},
                      last_taken_action {last_taken_action} {}

                      // i iterates over num_samples
            __device__ void operator() (uint32_t i) const {
                //Varibles used in ToGo Calculation
                float j_curr = 0; // accumulator of negative cost so far
                float d_curr = 0; // accumulator of negative log probability so far
                float x_curr[state_dims]; // current world state

                paranoid_assert(!any_nan(cuda_globals::curr_state, state_dims) && "State was nan in populate cost entry");

                // copy current state into x_curr
                memcpy(x_curr, cuda_globals::curr_state, sizeof(float) * state_dims);

                float init_curv_pose[3];
                bool out_of_bounds;
                cuda_globals::sample_curv_state(x_curr, init_curv_pose, out_of_bounds);
                // paranoid_assert(!out_of_bounds && "Initial state was out of bounds");

                // for each timestep, calculate cost and add to get cost to go
                // iterate through time because state depends on previous state (can't parallelize)
                for (uint32_t j = 0; j < num_timesteps; j++) {
                    float* u_ij = IDX_3D(sampled_action_trajectories, dim3(num_samples, num_timesteps, action_dims), dim3(i, j, 0));

                    // VERY CURSED. We want the last action in the best guess action to be the same as the second
                    // to last one (since we have to initialize it something). Taking the min of j and m - 2 saves
                    // us a host->device copy
                    const uint32_t idx = min(j, num_timesteps - 2);
                    paranoid_assert(!any_nan(action_trajectory_base[idx].data, action_dims) && "Control action base was nan");

                    // perturb initial control action guess with the brownian, clamp to reasonable bounds
                    for (uint32_t k = 0; k < action_dims; k++) {
                        const float recentered_brownian = action_trajectory_base[idx].data[k]
                                  + *IDX_3D(brownians, dim3(num_samples, num_timesteps, action_dims), dim3(i, j, k));
                        const float clamped_brownian = clamp(
                            recentered_brownian,
                            cuda_globals::action_min[k],
                            cuda_globals::action_max[k]
                        );
                        const float deadzoned = k == action_torque_idx && x_curr[state_speed_idx] < brake_enable_speed ?
                            max(clamped_brownian, 0.0f) : clamped_brownian;

                        u_ij[k] = deadzoned;
                    }

                    assert(!any_nan(u_ij, action_dims) && "Control was nan before model step");
                    model(x_curr, u_ij, x_curr, controller_period);
                    // printf("j: %i, x: %f, y: %f, yaw: %f, speed: %f\n", j, x_curr[state_x_idx], x_curr[state_yaw_idx], x_curr[state_yaw_idx], x_curr[state_speed_idx]);
                    paranoid_assert(!any_nan(x_curr, state_dims) && "State was nan after model step");

#ifdef DISPLAY
                    float* world_state = IDX_3D(
                        sampled_state_trajectories,
                        dim3(num_samples, num_timesteps, state_dims),
                        dim3(i, j, 0)
                    );
                    memcpy(world_state, x_curr, sizeof(float) * state_dims);
#endif
                    // Converts D and J Matrices to To-Go
                    // ALSO VERY CURSED FOR 2 REASONS:
                    // REASON 1: We have decided to not lower-triangularize the cost-to-gos, but also the log prob
                    // densities at the same time
                    // REASON 2: To save a loop over timesteps, we have essentially calculated the negative cost
                    // so far and stored it in cost_to_gos. During weighting, we will add back the total cost of the
                    // entire trajectory (which is |cost_to_gos| at final timestep). Likewise with log_prob_densities
                    const float c = cost(x_curr, u_ij, last_taken_action.data, init_curv_pose[0], controller_period * (j + 1), j == 0);
                    const float d = log_prob_densities[i*num_timesteps + j];
                    j_curr -= c;
                    d_curr -= d;
                    cost_to_gos[i * num_timesteps + j] = j_curr;
                    log_prob_densities[i * num_timesteps + j] = d_curr;
                    
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
            __host__ __device__ size_t operator() (size_t i) const {
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

        __device__ static ActionWeightTuple operator+ (
            const ActionWeightTuple& action_weight_t0,
            const ActionWeightTuple& action_weight_t1)
        {
            // log(1 + exp(log w2 - log w1)) + log w1

            DeviceAction a_min, a_max;
            float lw_min, lw_max;
            if (action_weight_t1.log_weight >= action_weight_t0.log_weight) {
                a_min = action_weight_t0.action;
                a_max = action_weight_t1.action;
                lw_min = action_weight_t0.log_weight;
                lw_max = action_weight_t1.log_weight;
            } else {
                a_min = action_weight_t1.action;
                a_max = action_weight_t0.action;
                lw_min = action_weight_t1.log_weight;
                lw_max = action_weight_t0.log_weight;
            }

            paranoid_assert(!isnan(lw_min) && !isnan(lw_max) && "log weight was nan in action reduce start");
            paranoid_assert(!any_nan(a_min.data, action_dims) && !any_nan(a_max.data, action_dims) && "Action was nan in action reduce start");

            paranoid_assert(lw_min <= lw_max && "Log weights were not sorted");

            const bool lw_min_inf = isinf(lw_min);
            const bool lw_max_inf = isinf(lw_max);

            paranoid_assert(!(lw_min_inf && lw_max_inf && lw_min > 0 && lw_max > 0) && "Both log weights were inf and positive");

            ActionWeightTuple res;
            if (lw_max_inf && lw_max < 0) {
                res = {DeviceAction {}, -CUDART_INF_F};
            } else {
                const float lw_min_prime = lw_min - lw_max;
                const float w_min_prime = expf(lw_min_prime);
                res = {
                    (a_max + w_min_prime * a_min) / (1 + w_min_prime),
                    lw_max + log1pf(w_min_prime)
                };
            }

            paranoid_assert(!isnan(res.log_weight) && "log weight was nan in action reduce end");
            paranoid_assert(!any_nan(res.action.data, action_dims) && "Action was nan in action reduce end");

            return res;
        }

        /** 
         * Captures pointers to action trajectories and cost to gos
         * Produces a weight for a single action based on its cost to go
        */
        struct IndexToActionWeightTuple {
            const float* action_trajectories;
            const float* cost_to_gos;
            const float* log_prob_densities;
            ActionWeightTuple* action_weight_tuples;

            IndexToActionWeightTuple (ActionWeightTuple* action_weight_tuples, const float* action_trajectories, const float* cost_to_gos, const float* log_prob_densities)
                    : action_weight_tuples {action_weight_tuples},
                      action_trajectories {action_trajectories},
                      cost_to_gos {cost_to_gos},
                      log_prob_densities {log_prob_densities} {}

            /**
             * @param idx refers to the index for the timesteps x samples matrix
             * (transposed from normal to make the reduction work)
             * @returns a tuple of the action indexed from the trajectories
             * and the weight of that action (strictly positive) based on its cost to go
             */
            __device__ void operator() (const uint32_t idx) const {
                // as we increment idx, we iterate over the samples first
                const uint32_t i = idx % num_samples; // sample
                // j should hold constant over a single reduction, it represents which timestep we are reducing over
                const uint32_t j = idx / num_samples;  // timestep

                ActionWeightTuple res {};

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
                    cost_to_go = CUDART_INF_F;
                } else {
                    const float anthony_adjustment = penult_step - 2 * final_step;
                    cost_to_go = cost_to_gos[i * num_timesteps + j] + anthony_adjustment;
                }
                paranoid_assert(!isnan(cost_to_go) && "cost-to-go was nan in tuple generation");

                const float ayush_adjustment = log_prob_densities[(i + 1) * num_timesteps - 2] - 2 * log_prob_densities[(i + 1) * num_timesteps - 1];
                const float log_prob_density = log_prob_densities[i * num_timesteps + j] + ayush_adjustment;
                paranoid_assert(!isnan(log_prob_density) && "log prob density was nan in tuple generation");
                // printf("j: %i Log prob density: %f\n", j, log_prob_density);

                res.log_weight = -1.0f / temperature * cost_to_go - log_prob_density;
                paranoid_assert(!isnan(res.log_weight) && "log weight was nan");

                action_weight_tuples[idx] = res;
            }
        };

        struct ActionWeightTupleToAction {
            __device__ DeviceAction operator() (const ActionWeightTuple& awt) const {
                return awt.action;
            }
        };

        // struct ReduceTimestep {
        //     DeviceAction* averaged_action;
        //     DeviceAction* action_trajectory_base;
        //
        //     const float* action_trajectories;
        //     const float* cost_to_gos;
        //     const float* log_probability_densities;
        //
        //     const ActionAverage reduction_functor {};
        //
        //     ReduceTimestep (DeviceAction* averaged_action, DeviceAction* action_trajectory_base, const float* action_trajectories, const float* cost_to_gos, const float* log_probability_densities)
        //             : averaged_action {averaged_action},
        //               action_trajectory_base {action_trajectory_base},
        //               action_trajectories {action_trajectories},
        //               cost_to_gos {cost_to_gos},
        //               log_probability_densities {log_probability_densities}
        //               { }
        //     __device__ void operator() (const uint32_t idx) {
        //         // idx ranges from 0 to num_timesteps - 1
        //         // idx represents a timestep
        //         thrust::counting_iterator<uint32_t> indices {idx * num_samples};
        //
        //         // applies a unary operator - IndexToActionWeightTuple before reducing over the samples with reduction_functor
        //         const ActionWeightTuple res = thrust::transform_reduce(
        //                 thrust::device,
        //                 indices, indices + num_samples,
        //                 IndexToActionWeightTuple {action_trajectories, cost_to_gos, log_probability_densities},
        //                 ActionWeightTuple { DeviceAction {}, -CUDART_INF_F }, reduction_functor);
        //
        //         if (res.log_weight == 0) {
        //             printf("Action at timestep %i had 0 weight. Using previous.\n", idx);
        //             averaged_action[idx] = action_trajectory_base[min(idx, num_timesteps - 2)];
        //         } else {
        //             averaged_action[idx] = res.action;
        //         }
        //         paranoid_assert(!any_nan(averaged_action[idx].data, action_dims) && "Averaged action was nan");
        //     }
        // };
    }
}

