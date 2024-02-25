#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <mutex>
#include <cuda_globals/cuda_globals.cuh>
#include <cmath>

#include "cuda_utils.cuh"
#include "mppi.cuh"
#include "functors.cuh"


namespace controls {
    namespace mppi {

        std::shared_ptr<MppiController> MppiController::create() {
            return std::make_shared<MppiController_Impl>();
        }

        MppiController_Impl::MppiController_Impl()
            : m_action_trajectories(num_action_trajectories),
              m_cost_to_gos(num_samples * num_timesteps),
              m_rng(),
#ifdef PUBLISH_STATES
              m_state_trajectories(num_samples * num_timesteps * state_dims),
#endif
              m_last_action_trajectory(num_timesteps - 1) {  // -1 because last element will always be
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

        Action MppiController_Impl::generate_action() {
            assert(cuda_globals::spline_texture_created);

#ifdef PUBLISH_STATES
            {
                std::lock_guard<std::mutex> guard {m_state_trajectories_mutex};
                memcpy(m_last_curr_state.data(), cuda_globals::curr_world_state_host, sizeof(cuda_globals::curr_world_state_host));
            }
#endif

            // call kernels
            std::cout << "generating brownians..." << std::endl;
            generate_brownians();
            cudaDeviceSynchronize();

            // print_tensor(m_action_trajectories, action_trajectories_dims);
            // std::cout << std::endl;


            std::cout << "populating cost..." << std::endl;
            {
#ifdef PUBLISH_STATES
                std::lock_guard<std::mutex> state_trajectories_guard {m_state_trajectories_mutex};
#endif
                populate_cost();
            }

            // std::cout << "Action Trajectories:" << std::endl;
            // print_tensor(m_action_trajectories, action_trajectories_dims);

            // std::cout << "Costs to go: " << std::endl;
            // print_tensor(m_cost_to_gos, dim3(num_samples, 1, num_timesteps));
            // std::cout << std::endl;

            std::cout << "\nreducing actions..." << std::endl;
            // not actually on device, just still in a device action struct
            thrust::device_vector<DeviceAction> averaged_trajectory = reduce_actions();

            DeviceAction host_action = averaged_trajectory[0];

            Action result_action;
            std::copy(
                std::begin(host_action.data), std::end(host_action.data),
                result_action.begin()
            );


            {
#ifdef PUBLISH_STATES
                std::lock_guard<std::mutex> guard {m_state_trajectories_mutex};
#endif

                thrust::copy(
                    averaged_trajectory.begin() + 1,
                    averaged_trajectory.end(),
                    m_last_action_trajectory.begin()
                );

#ifdef PUBLISH_STATES
                m_last_action = host_action;
#endif

                return result_action;
            }
        }

#ifdef PUBLISH_STATES
        std::vector<float> MppiController_Impl::last_state_trajectories() {
            std::lock_guard<std::mutex> guard {m_state_trajectories_mutex};

            std::vector<float> result (m_state_trajectories.size());
            thrust::copy(m_state_trajectories.begin(), m_state_trajectories.end(), result.begin());

            return result;
        }

        std::vector<glm::fvec2> MppiController_Impl::last_reduced_state_trajectory() {
            std::lock_guard<std::mutex> guard {m_last_action_trajectory_mutex};

            std::vector<glm::fvec2> result (m_last_action_trajectory.size() + 1);

            DeviceAction action = m_last_action;

            State state = m_last_curr_state;
            result[0] = {state[state_x_idx], state[state_y_idx]};

            int64_t i = -1;
            while (i < m_last_action_trajectory.size()) {
                if (i >= 0) {
                    action = m_last_action_trajectory[i];
                }

                // State state_dot;
                // ONLINE_DYNAMICS_FUNC(state.data(), action.data, state_dot.data(), controller_period);
                // for (uint8_t j = 0; j < state_dims; j++) {
                //     state[j] += state_dot[j] * controller_period;
                // }

                result[i + 1] = {i, i};

                i++;
            }

            return result;
        }
#endif

        // Private member functions of the controller
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
            prefix_scan(m_action_trajectories.data());
        }


        thrust::device_vector<DeviceAction> MppiController_Impl::reduce_actions() {
            // averaged_actions is where the weighted averages are stored
            // initialize it to 0 
            thrust::device_vector<DeviceAction> averaged_actions (num_timesteps);
            thrust::counting_iterator<uint32_t> indices {0};

            // for_each applies the ReduceTimestep functor to every idx in the range [0, num_timesteps)
            thrust::for_each(indices, indices + num_timesteps, ReduceTimestep {
                averaged_actions.data().get(),
                m_last_action_trajectory.data().get(),
                m_action_trajectories.data().get(),
                m_cost_to_gos.data().get()
            });

            return averaged_actions;
        }

        void MppiController_Impl::populate_cost() {
            thrust::counting_iterator<uint32_t> indices {0};

            PopulateCost populate_cost {
                m_action_trajectories.data(),
                m_action_trajectories.data(),
#ifdef PUBLISH_STATES
                m_state_trajectories.data(),
#endif
                m_cost_to_gos.data(), m_last_action_trajectory.data()};

            thrust::for_each(indices, indices + num_samples, populate_cost);
        }
    }
}