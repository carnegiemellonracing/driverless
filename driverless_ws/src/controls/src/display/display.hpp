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
            static constexpr int width = 1080;
            static constexpr int height = 1080;
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
                Trajectory(glm::fvec4 color, float thickness, GLuint program);

                void draw();

                std::vector<float> vertex_buf;

            private:
                glm::fvec4 color;
                float thickness;
                GLuint program;
                GLint color_loc;
                GLuint VBO;
                GLuint VAO;
            };

            void init_gl(SDL_Window* window);
            void init_trajectories();
            void init_spline();
            void init_best_guess();

            void fill_trajectories();
            void draw_trajectories();

            void draw_spline();
            void draw_best_guess();

            void update_loop(SDL_Window* window);

            glm::fvec2 m_cam_pos {0.0f, 0.0f};
            float m_cam_scale = 1.0f;

            GLuint m_shader_program;
            GLint m_cam_pos_loc;
            GLint m_cam_scale_loc;

            std::vector<Trajectory> m_trajectories;
            std::unique_ptr<Trajectory> m_spline = nullptr;
            std::unique_ptr<Trajectory> m_best_guess = nullptr;

            std::shared_ptr<mppi::MppiController> m_controller;
            std::shared_ptr<state::StateEstimator> m_state_estimator;
        };
    }
}
