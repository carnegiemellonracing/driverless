#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <mutex>
#include <cuda_globals/cuda_globals.cuh>
#include <cmath>
#include <utils/cuda_utils.cuh>

#include "mppi.cuh"
#include "functors.cuh"


namespace controls {
    namespace mppi {

        static Action device_action_to_action (const DeviceAction& device_action) {
            Action returned_action;
            std::copy(
                std::begin(device_action.data), std::end(device_action.data),
                returned_action.begin()
            );
            return returned_action;
        }

        std::shared_ptr<MppiController> MppiController::create(std::mutex& mutex, LoggerFunc logger) {
            return std::make_shared<MppiController_Impl>(mutex, logger);
        }

        MppiController_Impl::MppiController_Impl(std::mutex& mutex, LoggerFunc logger)
            : m_action_trajectories(num_action_trajectories),
              m_cost_to_gos(num_samples * num_timesteps),
              m_log_prob_densities(num_samples * num_timesteps),
              m_action_weight_tuples(num_samples * num_timesteps),
              m_rng(),
              m_last_action {},
#ifdef DISPLAY
              m_state_trajectories(num_samples * num_timesteps * state_dims),
              m_last_curr_state {},
#endif
              m_last_action_trajectory(num_timesteps - 1),
              m_logger {logger},
              m_mutex (mutex) {  // -1 because last element will always be
                                                                             // inferred from second to last
            for (uint32_t i = 0; i < num_timesteps - 1; i++) {
                DeviceAction to_send;
                for (int j = 0; j < action_dims; j++) {
                    to_send.data[j] = init_action_trajectory[i * action_dims + j];
                }
                m_last_action_trajectory[i] = to_send;
            }

            CURAND_CALL(curandCreateGenerator(&m_rng, rng_type));
            CURAND_CALL(curandSetPseudoRandomGeneratorSeed(m_rng, seed));
        }

        MppiController_Impl::~MppiController_Impl() {
            CURAND_CALL(curandDestroyGenerator(m_rng));
        }

        void MppiController_Impl::hardcode_last_action_trajectory(std::vector<Action> actions) {
            if (m_last_action_trajectory.size() != actions.size()) {
                throw std::runtime_error("Last action trajectory size does not match actions size");
            }
            for (size_t i = 0; i < m_last_action_trajectory.size(); i++) {
                Action action = actions[i];
                m_last_action_trajectory[i] = DeviceAction {action[0], action[1]};
            }
        };


        Action MppiController_Impl::generate_action() {
            std::lock_guard<std::mutex> guard {m_mutex};

            // call kernels
            m_logger("generating brownians");
            generate_brownians();

            m_logger("generating Log Probability Densities");
            generate_log_probability_density();

            m_logger("populating cost");
            populate_cost();

            m_logger("generating action weight tuples");
            generate_action_weight_tuples();

            m_logger("reducing actions");
            thrust::device_vector<DeviceAction> averaged_trajectory = reduce_actions();

            // not actually on device, just still in a device action struct
            DeviceAction host_action = m_last_action * action_momentum + (1 - action_momentum) * averaged_trajectory[0];
            m_last_action = host_action;
            Action result_action = device_action_to_action(host_action);

#ifdef DATA
            // we want to compare the first num_timesteps - 1 elements of averaged_trajectory and m_last_action_trajectory
            thrust::device_vector<Action> diff(num_timesteps - 1);
            thrust::transform(averaged_trajectory.begin(), averaged_trajectory.end() - 1, m_last_action_trajectory.begin(), diff.begin(), AbsoluteError {});

            thrust::host_vector<Action> host_diff = diff;                       

            m_percentage_diff_trajectory.assign(host_diff.begin(), host_diff.end());
            float sum_swangle = 0;
            float sum_throttle = 0;
            float max_swangle = 0;
            float max_throttle = 0;
            for (int i = 0; i < m_percentage_diff_trajectory.size(); i++) {
                sum_swangle += m_percentage_diff_trajectory[i][0];
                sum_throttle += m_percentage_diff_trajectory[i][1];
                if (m_percentage_diff_trajectory[i][0] > max_swangle) {
                    max_swangle = m_percentage_diff_trajectory[i][0];
                }
                if (m_percentage_diff_trajectory[i][1] > max_throttle) {
                    max_throttle = m_percentage_diff_trajectory[i][1];
                }
            }
            m_diff_statistics.mean_swangle = sum_swangle / m_percentage_diff_trajectory.size();
            m_diff_statistics.mean_throttle = sum_throttle / m_percentage_diff_trajectory.size();
            m_diff_statistics.max_swangle = max_swangle;
            m_diff_statistics.max_throttle = max_throttle;

            m_averaged_trajectory.clear();
            m_averaged_trajectory.reserve(averaged_trajectory.size());
            for (const DeviceAction& device_action : averaged_trajectory) {
                m_averaged_trajectory.push_back(Action {device_action.data[0], device_action.data[1]});
            }
            m_last_action_trajectory_logging.clear();
            m_last_action_trajectory_logging.reserve(m_last_action_trajectory.size());
            for (const DeviceAction &device_action : m_last_action_trajectory)
            {
                m_last_action_trajectory_logging.push_back(Action{device_action.data[0], device_action.data[1]});
            }

#endif

            // Save the averaged trajectory into a member variable for use by the next iteration of MPPI
            thrust::copy(
                averaged_trajectory.begin() + 1,
                averaged_trajectory.end(),
                m_last_action_trajectory.begin());


#ifdef DISPLAY
            CUDA_CALL(cudaMemcpyFromSymbol(
                m_last_curr_state.data(),
                cuda_globals::curr_state,
                state_dims * sizeof(float)
            ));
#endif

            return result_action;
        }

        Action MppiController_Impl::get_next_averaged_action() {
            std::lock_guard<std::mutex> guard {m_mutex};
            DeviceAction action_to_take = m_last_action * action_momentum + (1 - action_momentum) * m_last_action_trajectory[0];
            m_last_action = action_to_take;
            /**
             * From the CPP reference:
             * If arg3 is in [arg1, arg2), then the behavior is undefined. Thankfully this is not the case for us.
             */
            thrust::copy(
                m_last_action_trajectory.begin() + 1,
                m_last_action_trajectory.end(),
                m_last_action_trajectory.begin());

            return device_action_to_action(action_to_take);
        }

        void MppiController_Impl::set_logger(LoggerFunc logger) {
            std::lock_guard<std::mutex> guard {m_mutex};

            m_logger = logger;
        }

#ifdef DISPLAY
        std::vector<float> MppiController_Impl::last_state_trajectories(uint32_t num) {
            std::lock_guard<std::mutex> guard {m_mutex};
            const uint32_t num_floats = num * state_dims * num_timesteps;

            std::vector<float> result (num_floats);
            thrust::copy(m_state_trajectories.begin(), m_state_trajectories.begin() + num_floats, result.begin());

            return result;
        }

        std::vector<glm::fvec2> MppiController_Impl::last_reduced_state_trajectory() {
            std::lock_guard<std::mutex> guard {m_mutex};

            std::vector<glm::fvec2> result (m_last_action_trajectory.size() + 1);

            DeviceAction action = m_last_action;

            State state = m_last_curr_state;
            result[0] = {state[state_x_idx], state[state_y_idx]};

            size_t i = 0;
            while (i < m_last_action_trajectory.size() + 1) {
                if (i >= 1) {
                    action = m_last_action_trajectory[i - 1];
                }

                ONLINE_DYNAMICS_FUNC(state.data(), action.data, state.data(), controller_period);

                result[i] = {state[state_x_idx], state[state_y_idx]};

                i++;
            }

            return result;
        }
#endif

        // Private member functions of the controller
        // wrapper to do a prefix scan over the timesteps (middle) dimension of the n x m x q tensor
        ///@note only works on n x m x q I think
        void prefix_scan(thrust::device_ptr<float> normal) {
            auto actions = thrust::device_pointer_cast((DeviceAction*)normal.get());
            auto keys = thrust::make_transform_iterator(thrust::make_counting_iterator(0), DivBy<num_timesteps> {});

            thrust::inclusive_scan_by_key(keys, keys + num_samples * num_timesteps,
                                          actions, actions,
                                          Equal<size_t> {},
                                          AddActions {});
        }

        void MppiController_Impl::generate_brownians() {
            // generate normals, put it in device memory pointed to by m_action_trajectories
            // .data().get() returns the raw pointer from a device vector
            CURAND_CALL(curandGenerateNormal(m_rng, m_action_trajectories.data().get(), num_action_trajectories, 0, 1));

            // make the normals brownian
            thrust::counting_iterator<size_t> indices {0};
            thrust::for_each(indices, indices + num_action_trajectories, TransformStdNormal {m_action_trajectories.data()});
            /// TODO: why the prefix scan at the end?
            prefix_scan(m_action_trajectories.data());
        }


        void MppiController_Impl::generate_log_probability_density() {
            
            // Calculates Log probability density
            thrust::counting_iterator<size_t> indices {0};
            thrust::for_each(
                indices, indices + num_samples * num_timesteps,
                LogProbabilityDensity {m_action_trajectories.data(), m_log_prob_densities.data()}
            );
        }



        void MppiController_Impl::populate_cost() {
            thrust::counting_iterator<uint32_t> indices {0};

            PopulateCost populate_cost {
                m_action_trajectories.data(),
                m_action_trajectories.data(),
#ifdef DISPLAY
                m_state_trajectories.data(),
#endif
                m_cost_to_gos.data(),
                m_log_prob_densities.data(), 
                m_last_action_trajectory.data(),
                m_last_action,
                m_follow_midline_only
            };
            // populates cost for each sample trajectory
            thrust::for_each(indices, indices + num_samples, populate_cost);
        }

        thrust::device_vector<DeviceAction> MppiController_Impl::reduce_actions() {
            // averaged_actions is where the weighted averages are stored
            // initialize it to 0
            thrust::device_vector<DeviceAction> averaged_actions (num_timesteps);
            thrust::device_vector<ActionWeightTuple> averaged_awts(num_timesteps);
            thrust::device_vector<uint32_t> keys_out (num_timesteps);
            thrust::counting_iterator<uint32_t> indices {0};
            auto keys = thrust::make_transform_iterator(indices, DivBy<num_samples> {});

            thrust::reduce_by_key(
                    keys, keys + num_samples * num_timesteps, m_action_weight_tuples.begin(),
                    keys_out.begin(), averaged_awts.begin()
            );

            thrust::transform(
                    averaged_awts.begin(), averaged_awts.end(), averaged_actions.begin(),
                    ActionWeightTupleToAction {}
            );

            return averaged_actions;
        }
        void MppiController_Impl::generate_action_weight_tuples() {
            thrust::counting_iterator<uint32_t> indices {0};

            IndexToActionWeightTuple transform_to_tuple {
                m_action_weight_tuples.data().get(),
                m_action_trajectories.data().get(),
                m_cost_to_gos.data().get(),
                m_log_prob_densities.data().get()
            };

            thrust::for_each(indices, indices + num_timesteps * num_samples, transform_to_tuple);
        }
    }
}