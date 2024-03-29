#pragma once

#include <types.hpp>
#include <glm/glm.hpp>


namespace controls {
    namespace state {

        class StateEstimator {
        public:
            static std::shared_ptr<StateEstimator> create(std::mutex& mutex, LoggerFunc logger = no_log);

            virtual void on_spline(const SplineMsg& spline_msg) =0;
            virtual void on_world_twist(const TwistMsg& twist_msg) =0;
            virtual void on_world_quat(const QuatMsg& quat_msg) =0;
            virtual void on_world_pose(const PoseMsg& pose_msg) =0;

            virtual void sync_to_device(float swangle) =0;

            virtual bool is_ready() =0;

            virtual State get_state() =0;

            virtual void set_logger(LoggerFunc logger) =0;

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
