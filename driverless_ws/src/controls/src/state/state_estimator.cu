#include <cuda_utils.cuh>

#include "state_estimator.cuh"


namespace controls {
    namespace state {

        void StateEstimator_Impl::on_spline(const SplineMsg& spline_msg) {
            CUDA_CALL(cudaMemcpyToArray());
        }

        StateEstimator_Impl::StateEstimator_Impl() {

        }

    }
}