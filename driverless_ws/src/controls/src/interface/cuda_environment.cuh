#include <types.hpp>

#include "interface.hpp"


namespace controls {
    namespace interface {
        class CudaEnvironment : public Environment {
        public:
            void update_spline(const SplineMsg& msg) override;
            void update_slam(const SlamMsg& msg) override;
            void update_gps(const GpsMsg& msg) override;

            State get_curv_state() const override;
            State get_world_state() const override;
            double get_curvature(double progress_from_current) const override;

            std::mutex* get_mutex() override;
            std::condition_variable* get_notifier() override;
            std::atomic<bool>* get_dirty_flag() override;

            void save() override;
        };
    }
}
