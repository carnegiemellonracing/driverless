/**
 * @file functors.cuh
 * @brief Functors for MPPI control algorithm
 *
 * Functors are classes whose instances can be called like functions (by overloading operator()).
 * They can be passed parameters both initially during construction and when called.
 * MPPI uses functors because Thrust requires "functions" that only take in one parameter (e.g. an index).
 * We pass in other parameters that the functions need (e.g. pointers to memory structures) during construction.
 */

#pragma once

#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <constants.hpp>
#include <utils/cuda_utils.cuh>
#include <cuda_constants.cuh>
#include <cuda_globals/helpers.cuh>
#include <math_constants.h>
#include <utils/cuda_macros.cuh>

#include "types.cuh"
#include <utils/general_utils.hpp>
#include <curand_kernel.h>


namespace controls {
    namespace mppi
    {
        //TODO: wrap this in a namespace, then "using functors" in mppi.cu

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

        __device__ static void corner_creation(const float center_state[], float forward, float right, float corner_state[]) {
            float change_x = cosf(center_state[state_yaw_idx]) * forward + sinf(center_state[state_yaw_idx]) * right;
            float change_y = sinf(center_state[state_yaw_idx]) * forward - cosf(center_state[state_yaw_idx]) * right;
            corner_state[0] = center_state[state_x_idx] + change_x;
            corner_state[1] = center_state[state_y_idx] + change_y;
            corner_state[2] = center_state[state_yaw_idx];
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
            const float world_state[state_dims], const float action[], const float last_taken_action[],
            float start_progress, float time_since_traj_start, bool first, bool follow_midline_only) {

            float cent_curv_pose[3];
            bool cent_out_of_bounds;

            cuda_globals::sample_curv_state(world_state, cent_curv_pose, cent_out_of_bounds);

            const float centripedal_accel = CENTRIPEDAL_ACCEL_FUNC(world_state[state_speed_idx], action[action_swangle_idx]);
            const float abs_centripedal_accel = fabsf(centripedal_accel);

            if (abs_centripedal_accel > lat_tractive_capability) {
                return out_of_bounds_cost;
            }

            bool corner_out_of_bounds;
            float dummy_progress[3];
            float corner_state[3];
            for (int forward_mult = -1; forward_mult <= 1; forward_mult++) {
                for (int right_mult = -1; right_mult <= 1; right_mult += 2) {
                    corner_creation(world_state, cg_to_nose * forward_mult, cg_to_side * right_mult, corner_state);
                    cuda_globals::sample_curv_state(corner_state, dummy_progress, corner_out_of_bounds);
                    if (corner_out_of_bounds) {
                        return out_of_bounds_cost;
                    }
                }
            }

            const float progress = cent_curv_pose[0];

            const float approx_speed_along = (progress - start_progress) / time_since_traj_start;
            // ^ This is gotten from the state projection strategy
            const float actual_speed_along = world_state[3];
            const float speed_above_threshold_cost = (actual_speed_along > maximum_speed_ms) ? above_speed_threshold_cost : 0.0f;

            (void)actual_speed_along;
            // if (fabsf(approx_speed_along - actual_speed_along) > 1.0f) {
            //     printf("Approx speed along: %f, actual speed along: %f\n", approx_speed_along, actual_speed_along);
            // }
            const float speed_deviation = approx_speed_along - target_speed;
            const float speed_cost = speed_off_1mps_cost * fmaxf(-speed_deviation, 0.0f);
            // const float speed_cost = speed_off_1mps_cost * (-speed_deviation);
            (void)speed_cost;

            const float distance_cost = offset_1m_cost * cent_curv_pose[state_y_idx];
            const float progress_cost = progress_cost_multiplier * (-progress);

            const float deriv_cost = first ?
                0 :
                fabsf(action[action_torque_idx] - last_taken_action[action_torque_idx]) / controller_period * torque_1Nps_cost
              + fabsf(action[action_swangle_idx] - last_taken_action[action_swangle_idx]) / controller_period * swangle_1radps_cost
              ;

            float total_cost;
            if (follow_midline_only) {
                total_cost = speed_cost + distance_cost;
            } else {
                total_cost = speed_cost + speed_above_threshold_cost;
            }
 
            //TODO: delete?

            // return speed_cost;
            return total_cost;
            // + fabsf(action[action_torque_idx]) * 0.05f;// + deriv_cost;
        }

        // Functors for Brownian Generation
        // TODO: add a reference to how this conversion works
        /**
         * @brief Modifies a disturbance tensor from standard normal to brownian in-place, keeping within reasonable bounds.
         *
         * @param[in] std_normals Standard normal disturbance tensor generated by cuRAND, size is @c num_action_trajectories
         * @param[in] idx index into @c std_normals, used by thrust::for_each
         *
         * Uses @ref cuda_globals::perturbs_incr_std for variance information.
         */
        struct TransformStdNormal {
            thrust::device_ptr<float> std_normals;

            // Constructor is made explicit so no ugly implicit type conversions happen (single argument constructor)
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

                std_normals.get()[idx] = res * m_sqrt_timestep;
            }

            private:
                float m_sqrt_timestep = std::sqrt(controller_period);  // sqrt seconds //TODO: const
        };


        /**@brief Computes and stores log probability density from sampled perturbations for the sake of @rst :ref:`importance_sampling` @endrst.
         * Uses the covariance matrix in @ref cuda_globals::perturbs_incr_var_inv.
         *
         * @param[in] perturbation n x m x q perturbation matrix
         * @param[out] log_probability_densities n x m matrix to store result
         * @param[in] idx index from 0 to n x m, used for @c thrust::for_each
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

        // Computes the operation v^TSv where S is perturbs_incr_var_inv
        static __device__ float dot_with_action_matrix(const float* perturb) {
            float res = 0;
            for(uint8_t i= 0; i < action_dims; i++){

                //Dot product of pertubations and ith column
                const float intermediate =
                    dot<float>(perturb, &cuda_globals::perturbs_incr_var_inv[i*action_dims], action_dims);

                //part i of dot product of final dot product
                res += intermediate * perturb[i];
            }
            return res;
        }

        // Functors for cost calculation

        /** @brief Computes cost using the model and cost function, triangularizes both D and J
         *
         * @param[in] brownians brownian perturbation with mean 0
         * @param[out] sampled_action_trajectories same as brownians but doesn't have to be
         * @param[in] sampled_state_trajectories for publishing states
         * @param[out] cost_to_gos stores the negative cost so far
         * @param[in/out] log_prob_densities stores the negative log_prob_densities so far
         * @param[in] action_trajectory_base current best guess of optimal action trajectory, from previous iteration
         * of mppi.
         * @param[in] i index over timesteps
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
            bool follow_midline_only;

            const DeviceAction* action_trajectory_base;

            PopulateCost(thrust::device_ptr<float> brownians,
                         thrust::device_ptr<float> sampled_action_trajectories,
#ifdef DISPLAY
                         thrust::device_ptr<float> sampled_state_trajectories,
#endif
                         thrust::device_ptr<float> cost_to_gos,
                         thrust::device_ptr<float> log_prob_densities,
                         const thrust::device_ptr<DeviceAction>& action_trajectory_base,
                         DeviceAction last_taken_action,
                         bool follow_midline_only)
                    : brownians {brownians.get()},
                      sampled_action_trajectories {sampled_action_trajectories.get()},
#ifdef DISPLAY
                      sampled_state_trajectories {sampled_state_trajectories.get()},
#endif
                      cost_to_gos {cost_to_gos.get()},
                      log_prob_densities {log_prob_densities.get()},
                      action_trajectory_base {action_trajectory_base.get()},
                      last_taken_action {last_taken_action},
                      follow_midline_only {follow_midline_only}
                      {}

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
                // Note that it is perfectly possible for initial position to be out_of_bounds (though undesirable)
                // Since out_of_bounds_cost is not infinite, we still prioritize heading back into the bounds

                curandState curand_state;
                curand_init(i, 0, 0, &curand_state);


                // for each timestep, calculate cost and add to get cost to go
                // iterate through time because state depends on previous state (can't parallelize)
                for (uint32_t j = 0; j < num_timesteps; j++) {
                    //TODO: what is u_ij?
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
                        const float lower_bound = std::max(cuda_globals::action_min[k], last_taken_action.data[k] - cuda_globals::action_deriv_max[k] * controller_period * (j + 1));
                        const float upper_bound = std::min(cuda_globals::action_max[k], last_taken_action.data[k] + cuda_globals::action_deriv_max[k] * controller_period * (j + 1));
                        const float clamped_brownian = clamp_uniform(
                            recentered_brownian,
                            lower_bound,
                            upper_bound,
                            &curand_state
                        );
                        // TODO: document what deadzoned means
                        const float deadzoned = k == action_torque_idx && x_curr[state_speed_idx] < brake_enable_speed ?
                            max(clamped_brownian, 0.0f) : clamped_brownian;

                        u_ij[k] = deadzoned;
                    }

                    // // Importance sampling fix start (Comment this block out to restore it back to original behavior)
                    // float difference_from_mean[action_dims];
                    // for (uint32_t k = 0; k < action_dims; k++) {
                    //     difference_from_mean[k] = u_ij[k] - action_trajectory_base[idx].data[k];
                    // }
                    // float dot_result = dot_with_action_matrix(difference_from_mean);
                    // dot_result = dot_result / (-2.f * j * controller_period);
                    // log_prob_densities[i * num_timesteps + j] = dot_result;
                    // // Importance sampling fix end



                    paranoid_assert(!any_nan(u_ij, action_dims) && "Control was nan before model step");
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
                    // COST CALCULATION DONE HERE
                    const float c = cost(x_curr, u_ij, last_taken_action.data, init_curv_pose[0], controller_period * (j + 1), j == 0, follow_midline_only);
                    // Converts D and J Matrices to To-Go
                    // ALSO VERY CURSED FOR 2 REASONS:
                    // REASON 1: We have decided to not lower-triangularize the cost-to-gos, but also the log prob
                    // densities at the same time
                    // REASON 2: To save a loop over timesteps, we have essentially calculated the negative cost
                    // so far and stored it in cost_to_gos. During weighting, we will add back the total cost of the
                    // entire trajectory (which is |cost_to_gos| at final timestep). Likewise with log_prob_densities
                    const float d = log_prob_densities[i*num_timesteps + j];
                    j_curr -= c;
                    d_curr -= d;
                    cost_to_gos[i * num_timesteps + j] = j_curr;
                    log_prob_densities[i * num_timesteps + j] = d_curr;
                    
                    paranoid_assert(!isnan(c) && "cost-to-go was nan");
                }
            }
        };

        /// Note: Thrust needs a functor (similar to std::hash), so we can't use a standalone function
        /// Adds two device actions
        struct AddActions {
            __device__ DeviceAction operator() (const DeviceAction& a1, const DeviceAction& a2) const {
                DeviceAction res;
                for (uint8_t i = 0; i < action_dims; i++) {
                    res.data[i] = a1.data[i] + a2.data[i];
                }
                return res;
            }
        };

        /// Unary function to divide by the template parameter
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

        /**
         * @brief Adds two ActionWeightTuples whilst checking/clamping to reasonable bounds
         */
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
         * Produces a weight for a single action based on its cost to go.
         * Bundles the action and weight into an ActionWeightTuple, then stores it in the output array
         * in a different position so as to "transpose" the matrix.
         *
         * @param[out] action_weight_tuples output (transposed) matrix of action-weight tuples
         * @param[in] action_trajectories perturbed action trajectories
         * @param[in] cost_to_gos cost to go for each action
         * @param[in] log_prob_densities log probability densities for each action
         * @param[in] idx index into action_weight_tuples
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

        /// Extracts the action inside an ActionWeightTuple
        //TODO: can't this also be a function
        struct ActionWeightTupleToAction {
            __device__ DeviceAction operator() (const ActionWeightTuple& awt) const {
                return awt.action;
            }
        };

        struct AbsoluteError
        {
            __device__ Action operator()(const DeviceAction &x, const DeviceAction &y) const
            {
                Action result;
                for (size_t i = 0; i < action_dims; i++)
                {
                    float percentage_diff = (x.data[i] - y.data[i]) / (cuda_globals::action_max[i] * 2);
                    if (percentage_diff > 0) {
                        result[i] = percentage_diff;
                    }
                    else {
                        result[i] = -percentage_diff;
                    }
                }
                return result;
            }
        };

        struct SquaredError
        {
            __device__ Action operator()(const DeviceAction &x, const DeviceAction &y) const
            {
                Action result;
                for (size_t i = 0; i < action_dims; i++) {
                    float percentage_diff = (x.data[i] - y.data[i]) / (cuda_globals::action_max[i] * 2);
                    result[i] = percentage_diff * percentage_diff;
                }
                return result;
            }
        };

        struct GeometricSquaredError
        {
            __device__ Action operator()(const DeviceAction &x, const DeviceAction &y) const
            {
                Action result;
                float a_1 = 1.0/3.0;
                float r = 2.0/3.0;
                for (size_t i = 0; i < action_dims; i++) {
                    float percentage_diff = (x.data[i] - y.data[i]) / (cuda_globals::action_max[i] * 2);
                    result[i] = a_1 * percentage_diff * percentage_diff;
                    a_1 *= r;
                }
                return result;
            }
        };
    }
}

