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
            void on_state(const StateMsg& state_msg) override;

            void sync_to_device() override;

            std::vector<glm::fvec2> get_spline_frames() override;

            ~StateEstimator_Impl() override;

        private:
            constexpr static GLint shader_scale_loc = 0;
            constexpr static GLint shader_center_loc = 1;

            void gen_tex_info();
            void render_curv_frame_lookup();
            void sync_world_state();
            void sync_tex_info();
            void gen_curv_frame_lookup_framebuffer();
            void gen_gl_path();
            void fill_path_buffers();

            std::vector<glm::fvec2> m_spline_frames;

            State m_world_state = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            cudaTextureObject_t m_curv_frame_lookup_texture;
            cuda_globals::CurvFrameLookupTexInfo m_curv_frame_lookup_tex_info;
            GLuint m_curv_frame_lookup_fbo;
            GLuint m_curv_frame_lookup_rbo;

            utils::GLObj m_gl_path;
            GLuint m_gl_path_shader;
            SDL_Window* m_gl_window;
            SDL_GLContext m_gl_context;

            std::mutex& m_mutex;
        };

    }
}
