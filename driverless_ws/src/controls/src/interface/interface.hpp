#pragma once

#include <mutex>
#include <condition_variable>
#include <types.hpp>
#include <planning/src/planning_codebase/raceline/raceline.hpp>

namespace controls {
    namespace interface {
        class Environment {
        public:
            virtual ~Environment() =0;

            virtual void update_spline(const SplineMsg& msg) =0;
            virtual void update_slam(const SlamMsg& msg) =0;
            virtual void update_gps(const GpsMsg& msg) =0;

            virtual State get_curv_state() const =0;
            virtual State get_world_state() const =0;
            virtual double get_curvature(double progress_from_current) const =0;

            bool get_valid() const;


            std::mutex mutex;

        protected:
            bool m_valid {false};
        };

        void deserialize_spline(const SplineMsg& msg, Spline& out);
    }
}


// included so other classes only need to include one header for all of them
// at bottom to avoid loop

#include "cpu_environment.hpp"

#ifndef CONTROLS_NO_CUDA
#include "cuda_environment.cuh"
#endif
