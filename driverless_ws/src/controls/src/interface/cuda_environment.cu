#include <exception>

#include "cuda_environment.cuh"


namespace controls {
    namespace interface {
        void CudaEnvironment::update_spline(const controls::SplineMsg &msg) {
            throw std::runtime_error("cuda spline update not implemented");
        }

        void CudaEnvironment::update_gps(const controls::SplineMsg &msg) {
            throw std::runtime_error("cuda spline update not implemented");
        }
    }
}