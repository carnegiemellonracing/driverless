#include "display.hpp"

#include <chrono>
#include <cuda_constants.cuh>
#include <iostream>
#include <SDL2/SDL.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <gsl/gsl_odeiv2.h>
#include <mppi/types.cuh>
#include <utils/gl_utils.hpp>

using namespace std::chrono_literals;


namespace controls {
    namespace display {

        Display::Trajectory::Trajectory(glm::fvec4 color, float thickness, GLuint program)
            : color(color), program(program), thickness(thickness) {

            glGenVertexArrays(1, &VAO);
            glGenBuffers(1, &VBO);

            glBindVertexArray(VAO);

            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 2, (void*)0);
            glEnableVertexAttribArray(0);

            color_loc = glGetUniformLocation(program, "col");
        }

        void Display::Trajectory::draw() {
            glUniform4f(color_loc, color.x, color.y, color.z, color.w);
            glLineWidth(thickness);

            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertex_buf.size(), vertex_buf.data(), GL_DYNAMIC_DRAW);

            glBindVertexArray(VAO);

            assert(vertex_buf.size() % 2 == 0);
            glDrawArrays(GL_LINE_STRIP, 0, vertex_buf.size() / 2);
        }

        Display::Display(
            std::shared_ptr<mppi::MppiController> controller,
            std::shared_ptr<state::StateEstimator> state_estimator)
                : m_controller {std::move(controller)},
                  m_state_estimator {std::move(state_estimator)} {
        }

        void Display::init_gl(SDL_Window* window) {
            const char* vertexShaderSource = R"(
                #version 330 core
                layout (location = 0) in vec2 aPos;
                uniform vec2 camPos;
                uniform float camScale;
                void main()
                {
                   gl_Position = vec4((aPos - camPos) / camScale, 0.0f, 1.0f);
                }
            )";


            const char* fragmentShaderSource = R"(
                #version 330 core
                out vec4 FragColor;
                uniform vec4 col;
                void main()
                {
                   FragColor = col;
                }
            )";

            m_shader_program = utils::compile_shader(vertexShaderSource, fragmentShaderSource);

            m_cam_pos_loc = glGetUniformLocation(m_shader_program, "camPos");
            m_cam_scale_loc = glGetUniformLocation(m_shader_program, "camScale");

            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glLineWidth(1.0f);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_CULL_FACE);
        }

        void Display::init_trajectories() {
            for (uint32_t i = 0; i < num_samples; i++) {
                m_trajectories.emplace_back(glm::fvec4 {1.0f, 0.0f, 0.0f, 0.0f}, 1, m_shader_program);
            }
        }

        void Display::init_spline() {
            m_spline = std::make_unique<Trajectory>(glm::fvec4 {1.0f, 1.0f, 1.0f, 1.0f}, 2, m_shader_program);
        }

        void Display::init_best_guess() {
            m_best_guess = std::make_unique<Trajectory>(glm::fvec4 {0.0f, 1.0f, 0.0f, 1.0f}, 5, m_shader_program);
        }

        void Display::fill_trajectories() {
            using namespace glm;

            std::vector<float> states = m_controller->last_state_trajectories();

            for (uint32_t i = 0; i < num_samples; i++) {
                if (m_trajectories[i].vertex_buf.size() < num_timesteps) {
                    m_trajectories[i].vertex_buf = std::vector<float> (num_timesteps * 2);
                }

                for (uint32_t j = 0; j < num_timesteps; j++) {
                    uint32_t state_idx = i * state_dims * num_timesteps + j * state_dims;
                    m_trajectories[i].vertex_buf[2 * j] = states[state_idx];
                    m_trajectories[i].vertex_buf[2 * j + 1] = states[state_idx + 1];
                }
            }
        }

        void Display::draw_trajectories() {
            glUseProgram(m_shader_program);
            for (uint32_t i = 0; i < num_samples; i++) {
                Trajectory& t = m_trajectories[i];
                t.draw();
            }
        }

        void Display::draw_spline() {
            auto frames = m_state_estimator->get_spline_frames();

            assert(m_spline != nullptr);
            m_spline->vertex_buf = std::vector<float>(frames.size() * 2);
            for (size_t i = 0; i < frames.size(); i++) {
                m_spline->vertex_buf[2 * i] = frames[i].x;
                m_spline->vertex_buf[2 * i + 1] = frames[i].y;
            }

            m_spline->draw();
        }

        void Display::draw_best_guess() {
            auto frames = m_controller->last_reduced_state_trajectory();

            assert(m_best_guess != nullptr);
            m_best_guess->vertex_buf = std::vector<float>(frames.size() * 2);
            for (size_t i = 0; i < frames.size(); i++) {
                m_best_guess->vertex_buf[2 * i] = frames[i].x;
                m_best_guess->vertex_buf[2 * i + 1] = frames[i].y;
            }

            m_best_guess->draw();
        }

        void Display::run() {
            SDL_Window* window = utils::create_sdl2_gl_window("MPPI Display", width, height);
            init_gl(window);
            init_trajectories();
            init_spline();
            init_best_guess();

            update_loop(window);
        }

        void Display::update_loop(SDL_Window* window) {
            bool should_close = false;
            auto last_time = std::chrono::system_clock::now();
            float delta_time = 1 / framerate;
            while (!should_close) {
                SDL_Event e;
                while (SDL_PollEvent(&e) > 0) {
                    switch (e.type) {
                        case SDL_QUIT:
                            should_close = true;
                            break;

                        default:
                            break;
                    }
                }

                const uint8_t* keyboard_state = SDL_GetKeyboardState(nullptr);

                if (keyboard_state[SDL_SCANCODE_LEFT]) {
                    m_cam_pos += m_cam_scale * strafe_speed * delta_time * glm::fvec2(-1,0);
                }
                if (keyboard_state[SDL_SCANCODE_RIGHT]) {
                    m_cam_pos += m_cam_scale * strafe_speed * delta_time * glm::fvec2(1,0);
                }
                if (keyboard_state[SDL_SCANCODE_UP]) {
                    m_cam_pos += m_cam_scale * strafe_speed * delta_time * glm::fvec2(0,1);
                }
                if (keyboard_state[SDL_SCANCODE_DOWN]) {
                    m_cam_pos += m_cam_scale * strafe_speed * delta_time * glm::fvec2(0,-1);
                }

                if (keyboard_state[SDL_SCANCODE_S]) {
                    m_cam_scale *= pow(1 + scale_speed, delta_time);
                }
                if (keyboard_state[SDL_SCANCODE_W]) {
                    m_cam_scale *= pow(1 / (1 + scale_speed), delta_time);
                }

                glUseProgram(m_shader_program);
                glUniform2f(m_cam_pos_loc, m_cam_pos.x, m_cam_pos.y);
                glUniform1f(m_cam_scale_loc, m_cam_scale);

                glClear(GL_COLOR_BUFFER_BIT);

                fill_trajectories();
                draw_trajectories();

                draw_spline();
                draw_best_guess();

                SDL_GL_SwapWindow(window);

                auto now = std::chrono::system_clock::now();
                uint32_t to_go_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
                SDL_Delay(std::max(0u, to_go_ms));

                now = std::chrono::system_clock::now();
                delta_time = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count() / 1000.0f;
                last_time = now;
            }

            SDL_QuitSubSystem(SDL_INIT_VIDEO);
        }

    }
}
