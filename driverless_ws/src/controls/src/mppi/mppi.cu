#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>

#include <cuda_globals/cuda_globals.cuh>
#include <cmath>

#include "cuda_utils.cuh"
#include "mppi.cuh"
#include "functors.cuh"


namespace controls {
    namespace mppi {

        std::unique_ptr<MppiController> MppiController::create() {
            return std::make_unique<MppiController_Impl>();
        }

        MppiController_Impl::MppiController_Impl() {
            m_action_trajectories = thrust::device_malloc<float>(num_action_trajectories);
            m_cost_to_gos = thrust::device_malloc<float>(num_timesteps * num_samples);
            m_last_action_trajectory = thrust::device_malloc<float>(num_timesteps * action_dims);

            thrust::copy(
                init_action_trajectory, init_action_trajectory + num_timesteps * action_dims,
                m_last_action_trajectory
            );
        }

        MppiController_Impl::~MppiController_Impl() {
            thrust::device_free(m_action_trajectories);
            thrust::device_free(m_cost_to_gos);
            thrust::device_free(m_last_action_trajectory);
        }

        Action MppiController_Impl::generate_action() {
            assert(cuda_globals::spline_texture_created);

            // call kernels
            std::cout << "generating brownians..." << std::endl;
            generate_brownians();

            print_tensor(m_action_trajectories, action_trajectories_dims);
            std::cout << std::endl;

            std::cout << "populating cost..." << std::endl;
            populate_cost();

            std::cout << "Action Trajectories:" << std::endl;
            print_tensor(m_action_trajectories, action_trajectories_dims);

            std::cout << "Costs to go: " << std::endl;
            print_tensor(m_cost_to_gos, dim3(num_samples, 1, num_timesteps));
            std::cout << std::endl;

            std::cout << "\nreducing actions..." << std::endl;
            // not actually on device, just still in a device action struct
            DeviceAction dev_action = reduce_actions();

            Action action;
            std::copy(std::begin(dev_action.data), std::end(dev_action.data), action.begin());
            return action;
        }

        // Private member functions of the controller
        void prefix_scan(thrust::device_ptr<float> normal) {
            auto actions = thrust::device_pointer_cast((DeviceAction*)normal.get());
            auto keys = thrust::make_transform_iterator(thrust::make_counting_iterator(0), DivBy<num_timesteps> {});

            thrust::inclusive_scan_by_key(keys, keys + num_samples * num_timesteps,
                                          actions, actions,
                                          Equal<size_t> {}, AddActions {});
        }

        void MppiController_Impl::generate_brownians() {

            // create the random generator
            curandGenerator_t rng;
            CURAND_CALL(curandCreateGenerator(&rng, rng_type));
            CURAND_CALL(curandSetPseudoRandomGeneratorSeed(rng, seed));

            // generate normals, put it in device memory pointed to by m_action_trajectories
            CURAND_CALL(curandGenerateNormal(rng, m_action_trajectories.get(), num_action_trajectories, 0, 1));

            // make the normals brownian
            thrust::counting_iterator<size_t> indices {0};
            thrust::for_each(indices, indices + num_action_trajectories, TransformStdNormal {m_action_trajectories});
            prefix_scan(m_action_trajectories);

            // clean up memory
            CURAND_CALL(curandDestroyGenerator(rng));  // TODO: not do this every frame :(
        }


        DeviceAction MppiController_Impl::reduce_actions() {
            thrust::device_vector<DeviceAction> averaged_actions (num_timesteps);
            thrust::counting_iterator<uint32_t> indices {0};

            thrust::for_each(indices, indices + num_timesteps, ReduceTimestep {
                averaged_actions.data().get(),
                m_action_trajectories.get(),
                m_cost_to_gos.get()
            });

            thrust::host_vector<DeviceAction> averaged_actions_host = averaged_actions;
            DeviceAction res;

            // copy averaged action into result for returning
            for (int i = 0; i < action_dims; i++) {
                res.data[i] = averaged_actions_host.data()[0].data[i];
            }
            return res;
        }

        void MppiController_Impl::populate_cost() {
            thrust::counting_iterator<uint32_t> indices {0};

            float* curr_state_ptr;
            cudaGetSymbolAddress(
                reinterpret_cast<void**>(&curr_state_ptr),
                cuda_globals::curr_state
            );

            PopulateCost populate_cost {m_action_trajectories, m_action_trajectories,
                                        m_cost_to_gos, m_last_action_trajectory,
                                        thrust::device_pointer_cast(curr_state_ptr)};

            thrust::for_each(indices, indices + num_samples, populate_cost);
        }
    }
}