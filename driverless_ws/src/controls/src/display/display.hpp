#pragma once

#include <glad/glad.h>
#include <mppi/mppi.hpp>
#include <state/state_estimator.hpp>
#include <SDL2/SDL.h>
#include <vector>
#include <glm/glm.hpp>
#include <utils/gl_utils.hpp>

namespace controls {
    namespace display {
        class Display {
        public:
            static constexpr int width = 1024;
            static constexpr int height = 1024;
            static constexpr float framerate = 60;
            static constexpr float strafe_speed = 1.5;
            static constexpr float scale_speed = 1;

            static constexpr GLint traj_shader_cam_pos_loc = 0;
            static constexpr GLint traj_shader_cam_scale_loc = 1;
            static constexpr GLint traj_shader_color_loc = 2;

            static constexpr GLint img_shader_cam_pos_loc = 0;
            static constexpr GLint img_shader_cam_scale_loc = 1;
            static constexpr GLint img_shader_img_center_loc = 2;
            static constexpr GLint img_shader_img_width_loc = 3;
            static constexpr GLint img_shader_img_tex_loc = 4;

            static constexpr size_t num_samples_to_draw = std::min(1024U, num_samples);

            Display(
                std::shared_ptr<mppi::MppiController> controller,
                std::shared_ptr<state::StateEstimator> state_estimator
            );

            void run();

        private:
            class DrawableLine {
            public:
                DrawableLine(glm::fvec4 color, float thickness, GLuint program);

                void draw();
                void draw_points();

                std::vector<float> vertex_buf;

            private:
                std::vector<float> fill_triangle_points();

                glm::fvec4 color;
                float thickness;
                GLuint program;
                GLint color_loc;
                GLuint VBO;
                GLuint VAO;
            };

            void init_gl(SDL_Window* window);
            void init_img();
            void init_trajectories();
            void init_spline();
            void init_best_guess();
            void init_raceline();

            void fill_trajectories();
            void draw_trajectories();

            void draw_spline();
            void draw_cones();
            void draw_best_guess();

            void draw_offset_image();
            void draw_triangles();

            void draw_car();

            void draw_raceline();
            void update_raceline();

            void update_loop(SDL_Window* window);

            glm::fvec2 m_cam_pos {0.0f, 0.0f};
            float m_cam_scale = 10.0f;

            GLuint m_trajectory_shader_program;
            GLuint m_img_shader_program;

            std::vector<DrawableLine> m_trajectories;
            std::unique_ptr<DrawableLine> m_spline = nullptr;
			std::unique_ptr<DrawableLine> m_left_cone_trajectory = nullptr;
			std::unique_ptr<DrawableLine> m_right_cone_trajectory = nullptr;
            std::unique_ptr<DrawableLine> m_left_cone_trajectory_visible = nullptr;
            std::unique_ptr<DrawableLine> m_right_cone_trajectory_visible = nullptr;
            std::unique_ptr<DrawableLine> m_best_guess = nullptr;


            std::vector<glm::fvec2> m_raceline_points;
            std::unique_ptr<DrawableLine> m_raceline_line = nullptr;


            utils::GLObj m_offset_img_obj;
            GLuint m_offset_img_tex;

            std::shared_ptr<mppi::MppiController> m_controller;
            std::shared_ptr<state::StateEstimator> m_state_estimator;

            std::vector<glm::fvec2> m_spline_frames;
            std::vector<glm::fvec2> m_all_left_cone_points;
            std::vector<glm::fvec2> m_all_right_cone_points;
            std::vector<glm::fvec2> m_left_cone_points;
            std::vector<glm::fvec2> m_right_cone_points;
            
            std::vector<glm::fvec2> m_last_reduced_state_trajectory;
            std::vector<float> m_last_state_trajectories;
            state::OffsetImage m_offset_image;
        };
    }
}
