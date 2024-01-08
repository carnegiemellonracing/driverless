#pragma once

#include <planning/src/planning_codebase/raceline/raceline.hpp>
#include <interfaces/msg/spline_list.hpp>

#include "interface.hpp"


namespace controls {
    namespace interface {
        class CpuEnvironment : public Environment {
        public:
            CpuEnvironment() = default;

            void update_spline(const SplineMsg& msg) override;
            void update_slam(const SlamMsg& msg) override;
            void update_gps(const GpsMsg& msg) override;

            State get_curv_state() const override;
            State get_world_state() const override;
            double get_curvature(double progress_from_current) const override;
        };
    }
}