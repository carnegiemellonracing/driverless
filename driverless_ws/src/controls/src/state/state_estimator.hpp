#pragma once

#include <types.hpp>
#include <glm/glm.hpp>
#include <builtin_interfaces/msg/time.hpp>


namespace controls {
    namespace state {

        class StateEstimator {
        public:
            // TODO: ask why this is not a constructor (compared to Impl)
            static std::shared_ptr<StateEstimator> create(std::mutex& mutex, LoggerFunc logger = no_log);

            virtual void on_spline(const SplineMsg& spline_msg) =0;
            virtual void on_twist(const TwistMsg& twist_msg, const rclcpp::Time &time) =0;
            virtual void on_pose(const PoseMsg& pose_msg) =0;

            virtual void sync_to_device(const rclcpp::Time &time) =0;
            virtual bool is_ready() =0;
            virtual State get_projected_state() =0;

            virtual void set_logger(LoggerFunc logger) =0;
            virtual rclcpp::Time get_orig_spline_data_stamp() =0;
            virtual void record_control_action(const Action &action, const rclcpp::Time &ros_time) =0;

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
