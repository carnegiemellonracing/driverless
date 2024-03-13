#pragma once

#include <types.hpp>
#include <glm/glm.hpp>


namespace controls {
    namespace state {

        class StateEstimator {
        public:
            static std::shared_ptr<StateEstimator> create(std::mutex& mutex);

            virtual void on_spline(const SplineMsg& spline_msg) =0;
            virtual void on_state(const StateMsg& state_msg) =0;

            virtual void sync_to_device() =0;

#ifdef DISPLAY
            struct OffsetImage {
                std::vector<float> pixels;
                uint pix_width;
                uint pix_height;
                glm::fvec2 center;
                float world_width;
            };

            virtual std::vector<glm::fvec2> get_spline_frames() =0;
            virtual void get_offset_pixels(OffsetImage& offset_image) =0;
#endif

            virtual ~StateEstimator() = default;
        };

    }
}
