#include "cpu_environment.hpp"

namespace  controls {
    namespace interface {
        void CpuEnvironment::update_spline(const SplineMsg& msg) {
        }

        void CpuEnvironment::update_slam(const SlamMsg& msg) {
        }

        void CpuEnvironment::update_gps(const GpsMsg& msg) {
        }

        State CpuEnvironment::get_curv_state() const {
        }

        State CpuEnvironment::get_world_state() const {
        }

        double CpuEnvironment::get_curvature(double progress_from_current) const {
        }
    }
}
