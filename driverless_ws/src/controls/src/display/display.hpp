#pragma once

#include <glad/glad.h>
#include <mppi/mppi.hpp>
#include <state/state_estimator.hpp>
#include <SDL2/SDL.h>
#include <vector>
#include <glm/glm.hpp>

namespace controls {
    namespace display {
        class Display {
        public:
            static constexpr int width = 680;
            static constexpr int height = 680;
            static constexpr float framerate = 60;
            static constexpr float strafe_speed = 1.5;
            static constexpr float scale_speed = 1;


            Display(
                std::shared_ptr<mppi::MppiController> controller,
                std::shared_ptr<state::StateEstimator> state_estimator
            );

            void run();

        private:
            class Trajectory {
            public:
                Trajectory(glm::fvec4 color, GLuint program);

                void draw();

                std::vector<float> vertex_buf;

            private:
                glm::fvec4 color;
                GLuint program;
                GLint color_loc;
                GLuint VBO;
                GLuint VAO;
            };

            SDL_Window* init_sdl2();
            void init_gl(SDL_Window* window);
            void init_trajectories();
            void init_spline();

            void fill_trajectories();
            void draw_trajectories();

            void draw_spline();

            void update_loop(SDL_Window* window);

            glm::fvec2 m_cam_pos {0.0f, 0.0f};
            float m_cam_scale = 1.0f;

            GLuint m_shader_program;
            GLint m_cam_pos_loc;
            GLint m_cam_scale_loc;

            std::vector<Trajectory> m_trajectories;
            std::unique_ptr<Trajectory> m_spline = nullptr;

            std::shared_ptr<mppi::MppiController> m_controller;
            std::shared_ptr<state::StateEstimator> m_state_estimator;
        };
    }
}
