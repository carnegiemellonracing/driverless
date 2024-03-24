#pragma once

#include <cuda_globals/cuda_globals.cuh>
#include <thrust/device_vector.h>
#include <utils/gl_utils.hpp>

#include "state_estimator.hpp"


namespace controls {
    namespace state {

        class StateEstimator_Impl : public StateEstimator {
        public:
            StateEstimator_Impl(std::mutex& mutex);

            void on_spline(const SplineMsg& spline_msg) override;
            void on_world_twist(const TwistMsg& twist_msg) override;
            void on_world_quat(const QuatMsg& quat_msg) override;
            void on_world_pose(const PoseMsg& pose_msg) override;

            void sync_to_device(float swangle) override;

            bool is_ready() override;

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

            State m_world_state = {};

            cudaGraphicsResource_t m_curv_frame_lookup_rsc;
            cuda_globals::CurvFrameLookupTexInfo m_curv_frame_lookup_tex_info;
            GLuint m_curv_frame_lookup_fbo;
            GLuint m_curv_frame_lookup_rbo;

            utils::GLObj m_gl_path;
            GLuint m_gl_path_shader;
            SDL_Window* m_gl_window;
            SDL_GLContext m_gl_context;

            std::mutex& m_mutex;
            std::mutex m_gl_context_mutex;
            bool m_curv_frame_lookup_mapped = false;

            bool m_spline_ready = false;
            bool m_world_twist_ready = false;
            bool m_world_yaw_ready = false;

            float m_gps_heading;
        };

    }
}
