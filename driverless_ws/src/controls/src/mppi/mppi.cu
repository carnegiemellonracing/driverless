#include <cuda_utils.cuh>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "mppi.cuh"


namespace controls {
    namespace mppi {

        std::unique_ptr<Controller> make_mppi_controller() {
            return std::make_unique<MppiController>();
        }

        MppiController::MppiController() {
            m_action_trajectories =
                thrust::device_malloc(num_timesteps * num_samples * action_dims * sizeof(float));

            m_cost_to_gos =
                thrust::device_malloc(num_timesteps * num_samples * sizeof(float));

            m_last_action_trajectory =
                thrust::device_malloc(num_timesteps * action_dims * sizeof(float));
        }

        MppiController::~MppiController() {
            thrust::device_free(m_action_trajectories);
            thrust::device_free(m_cost_to_gos);
            thrust::device_free(m_last_action_trajectory);
        }
    }
}