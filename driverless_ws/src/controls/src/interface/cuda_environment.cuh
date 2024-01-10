#include <types.hpp>
#include <cuda_types.cuh>
#include <cuda_constants.cuh>
#include <cuda_types.cuh>
#include <planning/src/planning_codebase/raceline/raceline.hpp>

#include "interface.hpp"


namespace controls {
    namespace cuda {
        cfloat* d_spline_curvatures;  // allocated on first CudaEnvironment construction
    }

    namespace interface {
        // should be able to be constructed/destructed multiple times without an issue
        // but there can only be one at once
        class CudaEnvironment : public Environment {
        public:
            void update_spline(const SplineMsg& msg) override;
            void update_slam(const SlamMsg& msg) override;
            void update_gps(const GpsMsg& msg) override;

            State get_curv_state() const override;
            State get_world_state() const override;
            double get_curvature(double progress_from_current) const override;
        };
    }
}
