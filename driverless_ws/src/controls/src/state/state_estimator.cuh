#pragma once

#include <cuda_globals/cuda_globals.cuh>
#include <thrust/device_vector.h>
#include <utils/gl_utils.hpp>

#include "state_estimator.hpp"


namespace controls {
    namespace state {

        class StateProjector {
        public:
            void record_action(Action action, rclcpp::Time time);
            void record_speed(float speed, rclcpp::Time time);
            void record_pose(float x, float y, float yaw, rclcpp::Time time);

            State project(const rclcpp::Time& time, LoggerFunc logger) const;
            bool is_ready() const;

        private:
            struct Record {
                enum class Type {
                    Action,
                    Speed,
                    Pose
                };

                union {
                    Action action;
                    float speed;

                    struct {
                        float x;
                        float y;
                        float yaw;
                    } pose;
                };

                rclcpp::Time time;
                Type type;
            };

            void print_history() const;

            Record m_init_action { .action {}, .time = rclcpp::Time(0UL, default_clock_type), .type = Record::Type::Action};
            Record m_init_speed { .speed = 0, .time = rclcpp::Time(0UL, default_clock_type), .type = Record::Type::Speed};
            std::optional<Record> m_pose_record = std::nullopt;

            struct CompareRecordTimes {
                bool operator() (const Record& a, const Record& b) const {
                    return a.time < b.time;
                }
            };
            std::multiset<Record, CompareRecordTimes> m_history_since_pose {};
        };

        class StateEstimator_Impl : public StateEstimator {
        public:
            StateEstimator_Impl(std::mutex& mutex, LoggerFunc logger);

            void on_spline(const SplineMsg& spline_msg) override;
            void on_twist(const TwistMsg& twist_msg, const rclcpp::Time &time) override;
            void on_pose(const PoseMsg& pose_msg) override;

            void sync_to_device(const rclcpp::Time &time) override;
            bool is_ready() override;
            State get_projected_state() override;
            void set_logger(LoggerFunc logger) override;

            rclcpp::Time get_orig_spline_data_stamp() override;
            void record_control_action(const Action &action, const rclcpp::Time &time) override;

#ifdef DISPLAY
            std::vector<glm::fvec2> get_spline_frames() override;
            void get_offset_pixels(OffsetImage& offset_image) override;
#endif

            ~StateEstimator_Impl() override;

        private:
            constexpr static GLint shader_scale_loc = 0;
            constexpr static GLint shader_center_loc = 1;

            void gen_tex_info(glm::fvec2 car_pos);
            void render_curv_frame_lookup();
            void map_curv_frame_lookup();
            void unmap_curv_frame_lookup();
            void sync_world_state();
            void sync_tex_info();
            void gen_curv_frame_lookup_framebuffer();
            void gen_gl_path();
            void fill_path_buffers(glm::fvec2 car_pos);

            std::vector<glm::fvec2> m_spline_frames;

            StateProjector m_state_projector;
            State m_synced_projected_state;

            rclcpp::Time m_orig_spline_data_stamp;

            cudaGraphicsResource_t m_curv_frame_lookup_rsc;
            cuda_globals::CurvFrameLookupTexInfo m_curv_frame_lookup_tex_info;
            GLuint m_curv_frame_lookup_fbo;
            GLuint m_curv_frame_lookup_rbo;

            utils::GLObj m_gl_path;
            GLuint m_gl_path_shader;
            SDL_Window* m_gl_window;
            SDL_GLContext m_gl_context;

            std::mutex m_gl_context_mutex;
            bool m_curv_frame_lookup_mapped = false;

            LoggerFunc m_logger;
            std::mutex& m_mutex;
        };

    }
}
